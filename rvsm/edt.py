"""Exact Euclidean distance transform and the few other ndimage operators the distance fields need, in torch.

`rvsm.targets.block_fields` builds its targets out of scipy.ndimage calls on the CPU. `block_fields_torch`
builds the same targets from these, on whatever device the tensors live on (a CUDA device in the producer,
the CPU in the tests):

    edt(surf)            distance_transform_edt(~surf, return_indices=True): distance from every voxel to
                         the nearest True voxel of `surf`, and WHICH voxel that is
    edt2(surf)           the same with the SQUARED distance (exact integers, as float32) and no sqrt
    max_filter3(x)       maximum_filter(x, size=3, mode="nearest")
    gaussian3(x, s)      gaussian_filter(x, s, mode="nearest") (truncate 4, per-axis float32 rounding)
    binary_dilation(m)   binary_dilation(m): one step of the 6-neighbour cross, border 0
    label(m)             label(m, ones((3,3,3)))[0] up to a relabelling (only EQUALITY of labels is used)
    label_forest(m)      its union-find forest on CUDA (for a caller that compares a few voxels' roots)
    gaussian3_box(...)   gaussian3 of several fields, exact only inside a device-side box per volume
    pool2(v)             ladder.pool2 (2x mean pool of uint8 = floor(sum / 8)), byte for byte

THE EDT. The separable exact construction (Saito & Toriwaki; Maurer et al., which is what scipy runs):
the squared distance is min over i of g(i)^2 + (y - i)^2 along one axis at a time -- axis 0, then 1,
then 2, scipy's order -- starting from 0 on `surf` and +inf off it. Each pass is a brute-force min over
the whole line (224 candidates for a production block): on CUDA a small Triton kernel (one program per
line, the line in registers, no N^2 intermediate), elsewhere -- and on CUDA without Triton -- a chunked
`torch.min` over a (lines, N, N) tensor. The argmin of every pass is carried along (gathered through the
earlier passes' indices), so the result also names the nearest voxel. All squared distances are
integers below 2^24 and exact in float32, so the DISTANCES are bit-identical to scipy's (the sqrt of the
same integer, correctly rounded).

TIES. Where several surface voxels are equally near, each pass here takes the FIRST (lowest-index)
minimum along its line. scipy's Voronoi pass keeps tied sites in its envelope and its query stops at the
first of equal candidates, i.e. the same rule, and the passes run in the same axis order, so the nearest
INDICES agree with scipy's as well: exactly, on every random and structured mask the tests try. That
is an observed equivalence, not a proof about scipy's internals; if a scipy version ever broke a tie
differently, what would change is only what is computed FROM the index (the side of a voxel exactly
equidistant from faces on both sides, and the same-sheet checks there), never a distance.

THE GAUSSIAN is the one other place bits can differ from scipy: both accumulate each 1-D pass in float64
and round to float32 between passes, but the order of the sum is not guaranteed to be scipy's, so a
smoothed value can differ in its last float32 bit (it has matched on every test block so far).

DETERMINISM. Every operator here is a fixed sequence of elementwise ops, gathers, max-pools and
first-index reductions, plus atomics in `label` (the torch scatter-max, the union-find's max links)
whose result does not depend on the order they land in: no cudnn, no autotuning. Same inputs on the same device -> the same bits.
"""
import math

import torch
import torch.nn.functional as F

INF = float("inf")
_TRITON = None          # the compiled kernel, False once it is known to be unusable


def _triton_kernels():
    """(pass kernel, gaussian kernel), or False where Triton is not installed."""
    global _TRITON
    if _TRITON is None:
        try:
            import triton
            import triton.language as tl

            @triton.jit
            def _pass(G, V, I, A, B, OA, OB, n, inner, CAP2, NC: tl.constexpr, BY: tl.constexpr,
                      BK: tl.constexpr):
                # one program = outputs y0 .. y0+BY-1 of the lines k0 .. k0+BK-1 of one outer index.
                # Candidate i can only win at y if (y - i)^2 <= g(y)^2 (i = y itself scores g(y)^2),
                # so the loop covers the tile widened by R = ceil(sqrt(max g over the tile)): every
                # candidate outside is STRICTLY worse, and the first minimum is the full loop's.
                o = tl.program_id(0).to(tl.int64)
                k = tl.program_id(1).to(tl.int64) * BK + tl.arange(0, BK)
                y0 = tl.program_id(2) * BY
                y = y0 + tl.arange(0, BY)
                kok = k < inner
                ok = (y < n)[:, None] & kok[None, :]
                base = o * n * inner + k                                   # [BK]
                off = base[None, :] + y.to(tl.int64)[:, None] * inner
                gt = tl.load(G + off, mask=ok, other=0.0)
                # a CAP (`edt2(cap=)`): only results <= CAP2 are wanted, and a candidate further than
                # sqrt(CAP2) from y scores more than that, so the window never needs to be wider
                r2 = tl.minimum(tl.max(tl.max(gt, axis=1), axis=0), CAP2)
                r = tl.minimum(tl.ceil(tl.sqrt(r2)), n * 1.0).to(tl.int32)
                lo = tl.maximum(y0 - r, 0)
                hi = tl.minimum(y0 + BY + r, n)
                yf = y.to(tl.float32)
                best = tl.full([BY, BK], float("inf"), tl.float32)
                bi = tl.zeros([BY, BK], tl.int32)
                for i in range(lo, hi):
                    g = tl.load(G + base + i * inner, mask=kok, other=float("inf"))
                    d = yf - i
                    c = g[None, :] + (d * d)[:, None]
                    m = c < best                     # strict: the FIRST minimum along the line wins
                    best = tl.where(m, c, best)
                    bi = tl.where(m, i, bi)
                best = tl.where(best > CAP2, float("inf"), best)        # beyond the cap: unknown
                tl.store(V + off, best, mask=ok)
                tl.store(I + off, bi, mask=ok)
                if NC >= 1:
                    src = base[None, :] + bi.to(tl.int64) * inner
                    tl.store(OA + off, tl.load(A + src, mask=ok), mask=ok)
                    if NC >= 2:
                        tl.store(OB + off, tl.load(B + src, mask=ok), mask=ok)

            @triton.jit
            def _first(M, V, I, n, inner, BK: tl.constexpr):
                # the first pass from the binary mask, O(n) per line: the nearest True voxel on either
                # side (a forward then a backward scan), the lower index on a tie, as `_pass` would give
                o = tl.program_id(0).to(tl.int64)
                k = tl.program_id(1).to(tl.int64) * BK + tl.arange(0, BK)
                kok = k < inner
                base = o * n * inner + k
                last = tl.full([BK], -1, tl.int32)
                for i in range(0, n):
                    m = tl.load(M + base + i * inner, mask=kok, other=0)
                    last = tl.where(m != 0, i, last)
                    tl.store(I + base + i * inner, last, mask=kok)
                nxt = tl.full([BK], -1, tl.int32)
                for j in range(0, n):
                    i = n - 1 - j
                    m = tl.load(M + base + i * inner, mask=kok, other=0)
                    nxt = tl.where(m != 0, i, nxt)
                    left = tl.load(I + base + i * inner, mask=kok, other=-1)
                    dl = tl.where(left >= 0, i - left, 1 << 30)
                    dr = tl.where(nxt >= 0, nxt - i, 1 << 30)
                    use_l = dl <= dr
                    d = tl.where(use_l, dl, dr)
                    at = tl.where(use_l, left, nxt)
                    df = d.to(tl.float32)
                    tl.store(V + base + i * inner, tl.where(d < (1 << 30), df * df, float("inf")), mask=kok)
                    tl.store(I + base + i * inner, tl.where(at >= 0, at, 0), mask=kok)

            @triton.jit
            def _gauss(X, Y, W, total, n, inner, R: tl.constexpr, BLOCK: tl.constexpr):
                offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                ok = offs < total
                k = offs % inner
                i = (offs // inner) % n
                base = (offs // (inner * n)) * n * inner + k
                acc = tl.load(W + R) * tl.load(X + base + i * inner, mask=ok, other=0.0).to(tl.float64)
                for j in tl.static_range(1, R + 1):
                    lo = tl.maximum(i - j, 0)
                    hi = tl.minimum(i + j, n - 1)
                    a = tl.load(X + base + lo * inner, mask=ok, other=0.0).to(tl.float64)
                    b = tl.load(X + base + hi * inner, mask=ok, other=0.0).to(tl.float64)
                    acc = acc + tl.load(W + R + j) * (a + b)
                tl.store(Y + offs, acc.to(tl.float32), mask=ok)

            @triton.jit
            def _gauss_box(X, Y, W, BOX, DIN, DOUT, B, V, Z, YN, XN, MARG, EY, EX, AX: tl.constexpr,
                           R: tl.constexpr, BLOCK: tl.constexpr):
                # `_gauss` along axis AX of volume b of field f (program_id(1) = f * B + b), computed
                # only inside that volume's box (BOX[f, b] = zmin, -zmax, ymin, -ymax, xmin, -xmax),
                # widened by MARG on every axis and by EY / EX more on y / x, clamped to the volume.
                # The programs walk the box (the ones past its end return at once); per voxel the SAME
                # loads and the same float64 expression as `_gauss`, so the same bits.
                fb = tl.program_id(1)
                f = fb // B
                b = fb - f * B
                bx = BOX + fb * 6
                z0 = tl.maximum(tl.load(bx) - MARG, 0)
                z1 = tl.minimum(-tl.load(bx + 1) + MARG, Z - 1)
                y0 = tl.maximum(tl.load(bx + 2) - MARG - EY, 0)
                y1 = tl.minimum(-tl.load(bx + 3) + MARG + EY, YN - 1)
                x0 = tl.maximum(tl.load(bx + 4) - MARG - EX, 0)
                x1 = tl.minimum(-tl.load(bx + 5) + MARG + EX, XN - 1)
                nz = tl.maximum(z1 - z0 + 1, 0)
                ny = tl.maximum(y1 - y0 + 1, 0)
                nx = tl.maximum(x1 - x0 + 1, 0)
                start = tl.program_id(0) * BLOCK
                if start < nz * ny * nx:
                    t = start + tl.arange(0, BLOCK)
                    ok = t < nz * ny * nx
                    z = z0 + t // (ny * nx)
                    y = y0 + (t // nx) % ny
                    x = x0 + t % nx
                    offs = (z * YN + y) * XN + x
                    if AX == 0:
                        i = z
                        n = Z
                        inner = YN * XN
                    elif AX == 1:
                        i = y
                        n = YN
                        inner = XN
                    else:
                        i = x
                        n = XN
                        inner = 1
                    base = offs - i * inner
                    src = X + f.to(tl.int64) * DIN + b.to(tl.int64) * V
                    acc = tl.load(W + R) * tl.load(src + base + i * inner, mask=ok, other=0.0).to(tl.float64)
                    for j in tl.static_range(1, R + 1):
                        lo = tl.maximum(i - j, 0)
                        hi = tl.minimum(i + j, n - 1)
                        a = tl.load(src + base + lo * inner, mask=ok, other=0.0).to(tl.float64)
                        c = tl.load(src + base + hi * inner, mask=ok, other=0.0).to(tl.float64)
                        acc = acc + tl.load(W + R + j) * (a + c)
                    tl.store(Y + f.to(tl.int64) * DOUT + b.to(tl.int64) * V + offs, acc.to(tl.float32),
                             mask=ok)

            from triton.language.extra import libdevice

            @triton.jit
            def _walk(PA, PB, BOFF, ACT, NB, FT, FTS, NMAX, BR, BV, HIT, M, Y, X, BLOCK: tl.constexpr):
                # one lane = one segment p_a -> p_b; sample i of n at rint(a + f_i * (b - a)), with
                # f_i = float32(i / n) from the table and the multiply and add rounded separately
                offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                ok = offs < M
                act = (tl.load(ACT + offs, mask=ok, other=0) != 0) & ok
                n = tl.load(NB + offs, mask=ok, other=1)
                az = tl.load(PA + offs, mask=ok, other=0).to(tl.float32)
                ay = tl.load(PA + M + offs, mask=ok, other=0).to(tl.float32)
                ax = tl.load(PA + 2 * M + offs, mask=ok, other=0).to(tl.float32)
                sz = tl.load(PB + offs, mask=ok, other=0).to(tl.float32) - az
                sy = tl.load(PB + M + offs, mask=ok, other=0).to(tl.float32) - ay
                sx = tl.load(PB + 2 * M + offs, mask=ok, other=0).to(tl.float32) - ax
                boff = tl.load(BOFF + offs, mask=ok, other=0)
                nmax = tl.load(NMAX)
                left = offs < 0
                inv = offs < 0
                hit = offs < 0
                for i in range(0, nmax + 1):
                    a_ = act & (i <= n)
                    f = tl.load(FT + n.to(tl.int64) * FTS + i, mask=a_, other=0.0)
                    qz = libdevice.rint(libdevice.add_rn(az, libdevice.mul_rn(sz, f))).to(tl.int64)
                    qy = libdevice.rint(libdevice.add_rn(ay, libdevice.mul_rn(sy, f))).to(tl.int64)
                    qx = libdevice.rint(libdevice.add_rn(ax, libdevice.mul_rn(sx, f))).to(tl.int64)
                    qi = boff + (qz * Y + qy) * X + qx
                    rr = tl.load(BR + qi, mask=a_, other=0) != 0
                    vv = tl.load(BV + qi, mask=a_, other=0) != 0
                    hit = hit | (a_ & ((left & rr & ~vv) | (inv & ~vv)))
                    left = left | (a_ & ~rr)
                    inv = inv | (a_ & vv)
                tl.store(HIT + offs, hit.to(tl.int8), mask=ok)

            @triton.jit
            def _uf_find(L, a, m):
                # the root of a's tree: parents only ever grow (L[a] >= a, L[a] == a at a root), so
                # a stale parent is still an ancestor and the walk ends
                p = tl.load(L + a, mask=m, other=0, cache_modifier=".cg")
                go = m & (p != a)
                while tl.max(go.to(tl.int32), axis=0) > 0:
                    # path halving: a's parent becomes its grandparent (still an ancestor, larger)
                    gp = tl.load(L + p, mask=go, other=0, cache_modifier=".cg")
                    tl.atomic_max(L + a, gp, mask=go & (gp != p))
                    a = tl.where(go, gp, a)
                    p = tl.load(L + a, mask=go, other=0, cache_modifier=".cg")
                    go = go & (p != a)
                return a

            @triton.jit
            def _uf_union(L, a, b, m):
                # link the smaller root under the larger (atomic max); if it was no longer a root, retry
                # from the parent the atomic returned (Playne & Hawick's union, max instead of min)
                go = m
                while tl.max(go.to(tl.int32), axis=0) > 0:
                    a = _uf_find(L, a, go)
                    b = _uf_find(L, b, go)
                    up = go & (a > b)
                    dn = go & (b > a)
                    lo = tl.where(up, b, a)
                    hi = tl.where(up, a, b)
                    old = tl.atomic_max(L + lo, hi, mask=up | dn)
                    fin = old == lo
                    b = tl.where(up & ~fin, old, b)
                    a = tl.where(dn & ~fin, old, a)
                    go = (up | dn) & ~fin

            @triton.jit
            def _uf_nb(M, i, z, y, x, m, Y, X, dz: tl.constexpr, dy: tl.constexpr, dx: tl.constexpr):
                inb = m & (z + dz >= 0) & (y + dy >= 0) & (y + dy < Y) & (x + dx >= 0) & (x + dx < X)
                return inb & (tl.load(M + i + ((dz * Y + dy) * X + dx), mask=inb, other=0) != 0)

            @triton.jit
            def _uf_merge(M, L, N, Z, Y, X, BLOCK: tl.constexpr):
                # every mask voxel is joined to its 13 earlier (lower flat index) 26-neighbours in the
                # mask and its own volume -- except a neighbour 26-adjacent to one handled before it: that
                # pair is joined by the later one's own merge, so it is connected to this voxel anyway
                # (inside a solid band a voxel makes ONE union, with the voxel below it)
                offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                ok = offs < N
                m = (tl.load(M + offs, mask=ok, other=0) != 0) & ok
                V = Z * Y * X
                r = offs % V
                z = r // (Y * X)
                y = (r // X) % Y
                x = r % X
                i = offs.to(tl.int32)
                n0 = _uf_nb(M, i, z, y, x, m, Y, X, -1, 0, 0)
                n1 = _uf_nb(M, i, z, y, x, m, Y, X, 0, -1, 0)
                n2 = _uf_nb(M, i, z, y, x, m, Y, X, 0, 0, -1)
                n3 = _uf_nb(M, i, z, y, x, m, Y, X, -1, -1, 0)
                n4 = _uf_nb(M, i, z, y, x, m, Y, X, -1, 1, 0)
                n5 = _uf_nb(M, i, z, y, x, m, Y, X, -1, 0, -1)
                n6 = _uf_nb(M, i, z, y, x, m, Y, X, -1, 0, 1)
                n7 = _uf_nb(M, i, z, y, x, m, Y, X, 0, -1, -1)
                n8 = _uf_nb(M, i, z, y, x, m, Y, X, 0, -1, 1)
                n9 = _uf_nb(M, i, z, y, x, m, Y, X, -1, -1, -1)
                n10 = _uf_nb(M, i, z, y, x, m, Y, X, -1, -1, 1)
                n11 = _uf_nb(M, i, z, y, x, m, Y, X, -1, 1, -1)
                n12 = _uf_nb(M, i, z, y, x, m, Y, X, -1, 1, 1)
                _uf_union(L, i, i + ((-1 * Y + 0) * X + 0), n0)
                _uf_union(L, i, i + ((0 * Y + -1) * X + 0), n1 & ~(n0))
                _uf_union(L, i, i + ((0 * Y + 0) * X + -1), n2 & ~(n0 | n1))
                _uf_union(L, i, i + ((-1 * Y + -1) * X + 0), n3 & ~(n0 | n1 | n2))
                _uf_union(L, i, i + ((-1 * Y + 1) * X + 0), n4 & ~(n0 | n2))
                _uf_union(L, i, i + ((-1 * Y + 0) * X + -1), n5 & ~(n0 | n1 | n2 | n3 | n4))
                _uf_union(L, i, i + ((-1 * Y + 0) * X + 1), n6 & ~(n0 | n1 | n3 | n4))
                _uf_union(L, i, i + ((0 * Y + -1) * X + -1), n7 & ~(n0 | n1 | n2 | n3 | n5))
                _uf_union(L, i, i + ((0 * Y + -1) * X + 1), n8 & ~(n0 | n1 | n3 | n6))
                _uf_union(L, i, i + ((-1 * Y + -1) * X + -1), n9 & ~(n0 | n1 | n2 | n3 | n5 | n7))
                _uf_union(L, i, i + ((-1 * Y + -1) * X + 1), n10 & ~(n0 | n1 | n3 | n6 | n8))
                _uf_union(L, i, i + ((-1 * Y + 1) * X + -1), n11 & ~(n0 | n2 | n4 | n5))
                _uf_union(L, i, i + ((-1 * Y + 1) * X + 1), n12 & ~(n0 | n4 | n6))

            @triton.jit
            def _uf_final(M, L, OUT, N, V, BLOCK: tl.constexpr):
                # 1 + the root's index within its volume on the mask, 0 off it
                offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                ok = offs < N
                m = (tl.load(M + offs, mask=ok, other=0) != 0) & ok
                i = offs.to(tl.int32)
                root = _uf_find(L, i, m)
                base = ((offs // V) * V).to(tl.int32)
                tl.store(OUT + offs, tl.where(m, root - base + 1, 0), mask=ok)

            @triton.jit
            def _pass_face(G, U, D, IZ, IY, OZ, OY, OX, DY, DX, SYB, SYZ, SYY, SYX, SXB, SXZ, SXY, SXX,
                           R, EV, REACH, n, ZD, YD, CAP2, CODE: tl.constexpr, BY: tl.constexpr):
                # `_pass` along the last axis (inner = 1) of a (B, ZD, YD, n) batch, carrying the z / y
                # nearest coordinates, with `face_distance`'s arithmetic as its epilogue: u = sqrt(v),
                # the side of the displacement from the nearest voxel along the radial direction
                # (dy, dx) -- (y - iy) * dy + (x - ix) * dx, each product and the sum rounded on its own
                # -- d = -u on the inner side, u elsewhere; the indices go straight into their (3, ...)
                # planes OZ / OY / OX; CODE > 0: the reason codes R get CODE where still 0, inside EV
                # and further than REACH (`_stage_a`'s no_recto / no_verso rule).
                o = tl.program_id(0).to(tl.int64)
                y0 = tl.program_id(1) * BY
                y = y0 + tl.arange(0, BY)
                ok = y < n
                base = o * n
                off = base + y.to(tl.int64)
                gt = tl.load(G + off, mask=ok, other=0.0)
                r2 = tl.minimum(tl.max(gt, axis=0), CAP2)
                r = tl.minimum(tl.ceil(tl.sqrt(r2)), n * 1.0).to(tl.int32)
                lo = tl.maximum(y0 - r, 0)
                hi = tl.minimum(y0 + BY + r, n)
                yf = y.to(tl.float32)
                best = tl.full([BY], float("inf"), tl.float32)
                bi = tl.zeros([BY], tl.int32)
                for i in range(lo, hi):
                    g = tl.load(G + base + i)
                    d = yf - i
                    c = g + d * d
                    m = c < best                     # strict: the FIRST minimum along the line wins
                    best = tl.where(m, c, best)
                    bi = tl.where(m, i, bi)
                best = tl.where(best > CAP2, float("inf"), best)        # beyond the cap: unknown
                src = base + bi.to(tl.int64)
                iz = tl.load(IZ + src, mask=ok)
                iy = tl.load(IY + src, mask=ok)
                tl.store(OZ + off, iz, mask=ok)
                tl.store(OY + off, iy, mask=ok)
                tl.store(OX + off, bi, mask=ok)
                yy = o % YD
                zb = o // YD
                zz = zb % ZD
                bb = zb // ZD
                u = libdevice.sqrt_rn(best)
                ry = tl.load(DY + bb * SYB + zz * SYZ + yy * SYY + y * SYX, mask=ok, other=0.0)
                rx = tl.load(DX + bb * SXB + zz * SXZ + yy * SXY + y * SXX, mask=ok, other=0.0)
                gy = (yy - iy.to(tl.int64)).to(tl.float32)
                gx = (y - bi).to(tl.float32)
                side = libdevice.add_rn(libdevice.mul_rn(gy, ry), libdevice.mul_rn(gx, rx))
                tl.store(U + off, u, mask=ok)
                tl.store(D + off, tl.where(side < 0, -u, u), mask=ok)
                if CODE > 0:
                    rc = tl.load(R + off, mask=ok, other=1)
                    e = tl.load(EV + off, mask=ok, other=0) != 0
                    tl.store(R + off, tl.where((rc == 0) & e & (u > REACH), CODE, rc).to(tl.uint8), mask=ok)

            _TRITON = (_pass, _gauss, _first, _walk, _uf_merge, _uf_final, _gauss_box, _uf_find, _pass_face)
        except Exception:  # noqa: BLE001  -- no triton: the torch path
            _TRITON = False
    return _TRITON


def _triton_failed(e):
    global _TRITON
    _TRITON = False
    import warnings
    warnings.warn(f"rvsm.edt: Triton kernels unusable ({e!r}); using the torch implementations")


def _minplus_torch(g, chunk_bytes=256 << 20, cap2=INF):
    """(v, i) with v[r, y] = min_i g[r, i] + (y - i)^2 and i its first argmin; g (R, N) float32. A
    value above `cap2` is returned as +inf (as the capped Triton pass does)."""
    R, N = g.shape
    ar = torch.arange(N, device=g.device, dtype=torch.float32)
    d2 = (ar[:, None] - ar[None, :]) ** 2                    # [y, i]
    v = torch.empty_like(g)
    ix = torch.empty((R, N), dtype=torch.int32, device=g.device)
    c = max(1, int(chunk_bytes // (4 * N * N)))
    for s in range(0, R, c):
        vv, ii = torch.min(g[s:s + c, None, :] + d2[None], dim=2)
        if cap2 != INF:
            vv = torch.where(vv > cap2, INF, vv)
        v[s:s + c] = vv
        ix[s:s + c] = ii.to(torch.int32)
    return v, ix


def _axis_pass_torch(g, axis, carry, cap2=INF):
    gp = g.movedim(axis, -1)
    shp = gp.shape
    v, i = _minplus_torch(gp.contiguous().view(-1, shp[-1]), cap2=cap2)
    v = v.view(shp).movedim(-1, axis).contiguous()
    i = i.view(shp).movedim(-1, axis).contiguous()
    il = i.long()
    return v, i, [torch.gather(c, axis, il) for c in carry]


def _axis_pass(g, axis, carry=(), torch_only=False, cap2=INF):
    """One separable pass along `axis` of the contiguous 3-D float32 `g`: (v, i, carried) with
    v = min over the line position p of g[..p..] + (y - p)^2, i its first argmin (int32) and every
    int32 tensor of `carry` gathered at that argmin. A v above `cap2` is +inf, its i and carried
    values unspecified (`edt2(cap=)`)."""
    k = _triton_kernels() if (g.is_cuda and not torch_only) else False
    if k:
        shp = g.shape
        n = int(shp[axis])
        inner = 1
        for s in shp[axis + 1:]:
            inner *= int(s)
        outer = g.numel() // (n * inner)
        v = torch.empty_like(g)
        i = torch.empty(shp, dtype=torch.int32, device=g.device)
        out = [torch.empty_like(c) for c in carry]
        a = carry[0] if len(carry) > 0 else i
        b = carry[1] if len(carry) > 1 else i
        oa = out[0] if len(out) > 0 else i
        ob = out[1] if len(out) > 1 else i
        BK = 1 if inner == 1 else min(16, 1 << (inner - 1).bit_length())
        BY = 32 if BK > 1 else 64
        try:
            k[0][(outer, -(-inner // BK), -(-n // BY))](g, v, i, a, b, oa, ob, n, inner, float(cap2),
                                                         NC=len(carry), BY=BY, BK=BK,
                                                         num_warps=4 if BK > 1 else 2)
            return v, i, out
        except Exception as e:  # noqa: BLE001  -- a host where Triton cannot compile: the torch path
            _triton_failed(e)
    return _axis_pass_torch(g, axis, list(carry), cap2)


def _first_pass(surf, axis, torch_only=False):
    """`_axis_pass` along `axis` of the float version of the mask (0 on it, +inf off it)."""
    k = _triton_kernels() if (surf.is_cuda and not torch_only) else False
    if k:
        n = int(surf.shape[axis])
        inner = 1
        for q in surf.shape[axis + 1:]:
            inner *= int(q)
        outer = surf.numel() // max(n * inner, 1)
        v = torch.empty(surf.shape, dtype=torch.float32, device=surf.device)
        i = torch.empty(surf.shape, dtype=torch.int32, device=surf.device)
        BK = min(128, 1 << max(inner - 1, 0).bit_length())
        try:
            k[2][(outer, -(-inner // BK))](surf.contiguous().view(torch.uint8), v, i, n, inner, BK=BK)
            return v, i
        except Exception as e:  # noqa: BLE001
            _triton_failed(e)
    g = torch.where(surf, 0.0, INF).to(torch.float32).contiguous()
    v, i, _ = _axis_pass(g, axis, (), torch_only=True)
    return v, i


def edt2(surf, indices=True, torch_only=False, index_dtype=torch.int64, cap=None):
    """(squared distance float32, nearest index (3, *surf.shape)) to the True voxels of the bool tensor
    `surf`, a (Z,Y,X) volume or a (B,Z,Y,X) batch of them (each transformed on its own; the index is
    within its own volume). A volume with no True voxel at all is +inf everywhere (the index 0).
    `indices=False` returns (squared distance, None) and carries no indices through the passes.
    `torch_only` skips the Triton kernels (the tests compare the two); `index_dtype=torch.int32` keeps
    the indices at half the memory.

    `cap`: only distances up to `cap` are wanted. A voxel whose squared distance is <= cap^2 gets
    EXACTLY the uncapped result -- value and nearest index -- and any other voxel +inf (index
    unspecified): each pass then scans at most ceil(cap) candidates on either side instead of up to the
    whole line. Exact because a pass's winner for such a voxel lies within `cap` of it and scores
    <= cap^2, so its own input value is exact (by induction over the passes), and every candidate
    further away scores more than cap^2 -- it can neither win nor tie, so the first minimum is the same."""
    assert surf.dim() in (3, 4) and surf.dtype == torch.bool
    assert 3 * max(surf.shape[-3:]) ** 2 < 2 ** 24, "squared distances must stay exact in float32"
    a0 = surf.dim() - 3
    # scipy's order: axis 0, then 1, then 2; each pass carries the earlier passes' nearest coordinates
    v, iz = _first_pass(surf, a0, torch_only)
    cap2 = INF if cap is None else float(cap) * float(cap)
    v, iy, c = _axis_pass(v, a0 + 1, (iz,) if indices else (), torch_only, cap2)
    del iz
    v, ix, c = _axis_pass(v, a0 + 2, (c[0], iy) if indices else (), torch_only, cap2)
    if not indices:
        return v, None
    return v, torch.stack((c[0], c[1], ix)).to(index_dtype)


def edt(surf):
    """(distance float32, nearest index int64 (3,Z,Y,X)): `scipy.ndimage.distance_transform_edt(~surf,
    return_indices=True)` with the distances exact and ties resolved as in the module docstring."""
    d2, idx = edt2(surf)
    return torch.sqrt(d2), idx


def face_edt(surf, dy, dx, cap=None, reason=None, ev=None, reach=None, code=0, torch_only=False):
    """`targets.face_distance_torch`'s transform of a (B,Z,Y,X) bool batch: (d, u, ix int32 (3,B,Z,Y,X))
    with u the distance to the nearest True voxel (`edt2(cap=)` then sqrt), ix that voxel, and d = -u
    where the displacement from it, dotted with the radial direction (dy, dx) (float32, broadcastable
    to the batch), is negative, u elsewhere. `code` > 0: the uint8 `reason` codes that are still 0
    inside the bool `ev` and whose u > `reach` get `code`, in place.

    On CUDA the last pass does all of that in its epilogue (`_pass_face`: the same rounding, one
    rounded product per term and a rounded sum, no contraction) and writes the indices into their
    planes directly; elsewhere, and on CUDA without Triton, the torch steps. The same bits (tested)."""
    assert surf.dim() == 4 and surf.dtype == torch.bool
    assert 3 * max(surf.shape[-3:]) ** 2 < 2 ** 24, "squared distances must stay exact in float32"
    B, Z, Y, X = (int(v) for v in surf.shape)
    k = _triton_kernels() if (surf.is_cuda and not torch_only) else False
    if k:
        cap2 = INF if cap is None else float(cap) * float(cap)
        v, iz = _first_pass(surf, 1)
        v, iy, c = _axis_pass(v, 2, (iz,), cap2=cap2)
        del iz
        k = _triton_kernels()                 # a failing pass above falls back and disables them
    if k:
        dev = surf.device
        ix = torch.empty((3, B, Z, Y, X), dtype=torch.int32, device=dev)
        u = torch.empty(surf.shape, dtype=torch.float32, device=dev)
        d = torch.empty_like(u)
        dye = dy.to(torch.float32).expand(B, Z, Y, X)
        dxe = dx.to(torch.float32).expand(B, Z, Y, X)
        f32 = float(torch.tensor(float(reach), dtype=torch.float32)) if code else 0.0
        r_ = reason if code else u
        e_ = ev.view(torch.uint8) if code else u
        BY = 64
        try:
            k[8][(B * Z * Y, -(-X // BY))](v, u, d, c[0], iy, ix[0], ix[1], ix[2], dye, dxe, *dye.stride(),
                                           *dxe.stride(), r_, e_, f32, X, Z, Y, cap2, CODE=int(code), BY=BY,
                                           num_warps=2)
            return d, u, ix
        except Exception as e:  # noqa: BLE001
            _triton_failed(e)
    d2, ix = edt2(surf, index_dtype=torch.int32, cap=cap, torch_only=True)
    u = torch.sqrt(d2)
    del d2
    gy = (torch.arange(Y, device=surf.device)[None, None, :, None] - ix[1]).to(torch.float32)
    gx = (torch.arange(X, device=surf.device)[None, None, None, :] - ix[2]).to(torch.float32)
    side = gy * dy
    side = side + gx * dx
    del gy, gx
    d = torch.where(side < 0, -u, u)
    if code:
        reason.masked_fill_((reason == 0) & ev & (u > reach), int(code))
    return d, u, ix


def max_filter3(x):
    """`maximum_filter(x, size=3, mode="nearest")` of a (Z,Y,X) float tensor, or of each volume of a
    (B,Z,Y,X) batch (replicating the border cannot raise a maximum, so padding with -inf is the same)."""
    sh = x.shape
    return F.max_pool3d(x.reshape(-1, 1, *sh[-3:]), 3, stride=1, padding=1).view(sh)


def _shift(x, a, s, fill):
    """x shifted by s (+-1) along axis a: out[i] = x[i - s], `fill` where that is outside."""
    out = torch.full_like(x, fill)
    n = x.shape[a]
    if s > 0:
        out.narrow(a, s, n - s).copy_(x.narrow(a, 0, n - s))
    else:
        out.narrow(a, 0, n + s).copy_(x.narrow(a, -s, n + s))
    return out


def binary_dilation(m):
    """`scipy.ndimage.binary_dilation(m)`: one step of the 6-neighbour cross, border value 0 (over the
    last three axes: a batch is dilated volume by volume)."""
    out = m.clone()
    for a in (-3, -2, -1):
        for s in (1, -1):
            out |= _shift(m, a, s, False)
    return out


def gaussian_weights(sigma, truncate=4.0):
    """scipy's `_gaussian_kernel1d(sigma, 0, radius)` with `radius = int(truncate * sigma + 0.5)`."""
    r = int(truncate * float(sigma) + 0.5)
    w = [math.exp(-0.5 / (float(sigma) ** 2) * (i * i)) for i in range(-r, r + 1)]
    s = sum(w)
    return [v / s for v in w], r


def gaussian3(x, sigma, truncate=4.0):
    """`scipy.ndimage.gaussian_filter(x, sigma, mode="nearest")` of a 3-D float32 tensor: one 1-D
    correlation per axis in the order 0, 1, 2, each accumulated in float64 with scipy's symmetric
    pairing (w0 x_i + sum_j w_j (x_{i-j} + x_{i+j})) and rounded to float32 in between, as scipy's
    float32 output array does. The last bits can still differ from scipy (the order of the sum). A
    (B,Z,Y,X) batch is filtered volume by volume (over its last three axes)."""
    w, r = gaussian_weights(sigma, truncate)
    y = x.to(torch.float32).contiguous()
    k = _triton_kernels() if y.is_cuda else False
    a0 = y.dim() - 3
    if k:
        try:
            wt = torch.tensor(w, dtype=torch.float64).to(y.device, non_blocking=True)
            for a in range(a0, a0 + 3):
                n = int(y.shape[a])
                inner = 1
                for s in y.shape[a + 1:]:
                    inner *= int(s)
                out = torch.empty_like(y)
                BLOCK = 1024
                k[1][(-(-y.numel() // BLOCK),)](y, out, wt, y.numel(), n, inner, R=r, BLOCK=BLOCK)
                y = out
            return y
        except Exception as e:  # noqa: BLE001
            _triton_failed(e)
            y = x.to(torch.float32).contiguous()
    for a in range(a0, a0 + 3):
        n = y.shape[a]
        idx = torch.clamp(torch.arange(-r, n + r, device=y.device), 0, n - 1)
        p = torch.index_select(y, a, idx).to(torch.float64)
        acc = w[r] * p.narrow(a, r, n)
        for j in range(1, r + 1):
            acc = acc + w[r + j] * (p.narrow(a, r - j, n) + p.narrow(a, r + j, n))
        del p
        y = acc.to(torch.float32)
        del acc
    return y


_WEIGHTS = {}          # (sigma, truncate, device) -> the float64 weights on that device


def _weights_on(sigma, truncate, dev):
    key = (float(sigma), float(truncate), str(dev))
    if key not in _WEIGHTS:
        w, _ = gaussian_weights(sigma, truncate)
        _WEIGHTS[key] = torch.tensor(w, dtype=torch.float64).to(dev)
    return _WEIGHTS[key]


BOX_EMPTY = 1 << 30   # an entry of an empty `gaussian3_box` box: min = BOX_EMPTY, -max = BOX_EMPTY


def gaussian3_box(xs, box, sigma, truncate=4.0, margin=1):
    """`gaussian3` of each (B,Z,Y,X) float32 CUDA tensor of `xs` (all one shape), wanted only near some
    points: `box` (len(xs), B, 6) int32 on the device holds, per field and volume, the points' extent
    (zmin, -zmax, ymin, -ymax, xmin, -xmax; `BOX_EMPTY` everywhere for no point), and the result is
    exact -- the same bits as `gaussian3` -- on that extent widened by `margin` voxels per axis (clamped
    to the volume); elsewhere it is unspecified. One (len(xs), B, Z, Y, X) tensor, or None where the
    Triton kernels are unusable (the caller then filters the whole volumes).

    The separable passes run in `gaussian3`'s order (axis 0, 1, 2), each over the part of the volume the
    NEXT passes read: the last (x) pass over the widened box, the y pass over it widened by the radius
    along x, the z pass over it widened along y and x as well. Each output voxel is the same loads and
    the same float64 expression as `gaussian3`'s kernel, and the clamping ("nearest") is to the whole
    volume, so every voxel the final box needs sees exactly the values the whole-volume filter
    would. The box lives on the device (no synchronisation): the grid covers the whole volume and the
    voxels outside the box do nothing. One launch per pass for all of `xs`."""
    k = _triton_kernels()
    if not k:
        return None
    x0 = xs[0]
    for x in xs:
        assert x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and x.shape == x0.shape
    assert x0.dim() == 4
    B, Z, Y, X = (int(v) for v in x0.shape)
    V = Z * Y * X
    nf = len(xs)
    _, r = gaussian_weights(sigma, truncate)
    wt = _weights_on(sigma, truncate, x0.device)
    din = 0 if nf == 1 else (xs[1].data_ptr() - x0.data_ptr()) // 4
    for i, x in enumerate(xs):
        assert x.data_ptr() - x0.data_ptr() == 4 * i * din
    out = [torch.empty((nf, B, Z, Y, X), dtype=torch.float32, device=x0.device) for _ in range(2)]
    BLOCK = 1024
    grid = (-(-V // BLOCK), nf * B)
    try:
        src, sd = x0, din
        for ax, (ey, ex), dst in ((0, (r, r), out[0]), (1, (0, r), out[1]), (2, (0, 0), out[0])):
            k[6][grid](src, dst, wt, box, sd, B * V, B, V, Z, Y, X, int(margin), ey, ex, AX=ax, R=r,
                       BLOCK=BLOCK)
            src, sd = dst, B * V
        return out[0]
    except Exception as e:  # noqa: BLE001
        _triton_failed(e)
        return None


def walk_crossings(pa, pb, boff, act, nb, nt, br, bv, torch_only=False):
    """`targets._pair_checks`' ordered walk for M segments at once: for each segment with `act`, the
    samples i = 0..n (n = `nb`, per segment) at rint(pa + float32(i / n) * (pb - pa)) of the bands `br`
    / `bv` ((B,Z,Y,X) bool; `pa` / `pb` (3, M) int coordinates within the segment's own volume, whose
    flat offset is `boff`); True where the walk re-enters a recto band after leaving it or leaves a
    verso band after entering it. `nt` bounds every n (the float32(i / n) table is (nt + 1)^2).

    The numpy walk samples every segment of a block with the block's n, one sample per Python
    iteration; this is the same arithmetic per segment (float32 multiply, float32 add, round half to
    even), in one kernel launch on CUDA."""
    dev = pa.device
    M = int(pa.shape[1])
    Y, X = int(br.shape[-2]), int(br.shape[-1])
    nt = int(nt)
    ni = torch.arange(nt + 1, device=dev, dtype=torch.float64)
    ftab = (ni[None, :] / torch.clamp(ni, min=1)[:, None]).to(torch.float32).contiguous()   # [n, i]
    nb = torch.clamp(nb.to(torch.int32), 1, nt)
    brf, bvf = br.reshape(-1), bv.reshape(-1)
    k = _triton_kernels() if (pa.is_cuda and not torch_only) else False
    if k and M:
        try:
            hit = torch.empty(M, dtype=torch.int8, device=dev)
            nmax = torch.amax(torch.where(act, nb, 0)).reshape(1)
            BLOCK = 256
            k[3][(-(-M // BLOCK),)](pa.to(torch.int32).contiguous(), pb.to(torch.int32).contiguous(),
                                    boff.to(torch.int64).contiguous(), act.to(torch.uint8).contiguous(),
                                    nb.contiguous(), ftab, nt + 1, nmax, brf.view(torch.uint8),
                                    bvf.view(torch.uint8), hit, M, Y, X, BLOCK=BLOCK)
            return hit.bool()
        except Exception as e:  # noqa: BLE001
            _triton_failed(e)
    a, b = pa.to(torch.float32), pb.to(torch.float32)
    seg = b - a
    left = torch.zeros(M, dtype=torch.bool, device=dev)
    inv = torch.zeros_like(left)
    hit = torch.zeros_like(left)
    nmax = int(torch.amax(torch.where(act, nb, 0)).item()) if M else 0
    nl = nb.long()
    for i in range(nmax + 1):
        a_ = act & (i <= nb)
        f = ftab[nl, i]
        q = torch.round(a + seg * f).long()
        qi = boff + (q[0] * Y + q[1]) * X + q[2]
        qi = torch.where(a_, qi, 0)
        rr, vv = brf[qi], bvf[qi]
        hit |= a_ & ((left & rr & ~vv) | (inv & ~vv))
        left |= a_ & ~rr
        inv |= a_ & vv
    return hit


def _label_triton(k, m, sh, vol, N):
    """`label` in two kernels and no synchronisation: a concurrent union-find over the mask voxels
    (`_uf_merge`: each voxel joins its earlier 26-neighbours, a smaller root always linked under a
    larger one with an atomic max; finds halve their paths), then every voxel's root (`_uf_final`).
    Parents only grow and always name a voxel of the same component, and after the merge every
    26-adjacent pair of mask voxels is in one tree, so each tree is one component with its LARGEST
    voxel as the root, whatever order the atomics land in: the labels are the torch iteration's fixed
    point -- each component's largest 1 + index -- bit for bit (tested). One warp per program: a
    program's loops run until its slowest lane's find is done."""
    mu, par = _forest(k, m, sh, N)
    out = torch.empty(N, dtype=torch.int32, device=m.device)
    k[5][(-(-N // 32),)](mu, par, out, N, vol, BLOCK=32, num_warps=1)
    return out.view(sh)


def _forest(k, m, sh, N):
    mu = m.contiguous().view(-1).view(torch.uint8)
    par = torch.arange(N, device=m.device, dtype=torch.int32)
    Z, Y, X = (int(v) for v in sh[-3:])
    k[4][(-(-N // 32),)](mu, par, N, Z, Y, X, BLOCK=32, num_warps=1)
    return mu, par


def label_forest(m):
    """The union-find forest of `label` on CUDA, before the labels are read out: an int32 parent array
    over the flat voxels of `m` (a CUDA bool volume or batch, fewer than 2^31 voxels) in which two voxels
    OF THE MASK are in one 26-connected component (of one volume) iff following parents from each ends
    at the same root (`_uf_find`, the kernel set's last entry, which a caller's kernel can use). For a
    caller that compares the labels of a few voxels only; None where the Triton kernels are unusable."""
    k = _triton_kernels() if m.is_cuda else False
    if not k or m.numel() >= 2 ** 31:
        return None
    try:
        return _forest(k, m, m.shape, m.numel())[1]
    except Exception as e:  # noqa: BLE001
        _triton_failed(e)
        return None


LABEL_CHECK = 8      # the torch iteration: iterations between convergence checks (each is a device sync)


def label(m, max_iter=100000):
    """Connected components of the bool tensor `m`, a (Z,Y,X) volume or a (B,Z,Y,X) batch of them,
    with 26-connectivity: an int32 tensor, 0 off `m`, equal on a voxel pair of one volume iff they are
    in one component. The labels are NOT scipy's numbering (only equality within a volume is
    meaningful): each component ends as 1 + the largest flat index (within its volume) it contains, a
    deterministic function of `m`.

    On CUDA a union-find in two kernel launches (`_label_triton`). Elsewhere an iteration with the same
    fixed point: every voxel starts as its own 1 + flat index; a label always names a voxel of the same
    component and only ever grows. Per iteration: each voxel's 3^3 neighbourhood maximum (max_pool3d, per volume) is
    hooked onto the voxel its label names (a scatter-max, order independent), taken by the voxel itself,
    and the labels are pointer-jumped (a voxel takes the label of the voxel its label names). It stops
    when every voxel already holds its neighbourhood maximum; that is tested every `LABEL_CHECK`
    iterations (a converged iteration changes nothing), so the device is synchronised only then."""
    sh = m.shape
    vol = 1
    for q in sh[-3:]:
        vol *= int(q)
    nb_ = m.numel() // max(vol, 1)
    N = m.numel()
    k = _triton_kernels() if m.is_cuda else False
    if k and N < 2 ** 31:
        try:
            return _label_triton(k, m, sh, vol, N)
        except Exception as e:  # noqa: BLE001
            _triton_failed(e)
    dt = torch.float32 if vol < 2 ** 24 else torch.float64
    dev = m.device
    mf = m.reshape(-1)
    loc = torch.arange(1, vol + 1, device=dev, dtype=dt).repeat(nb_)
    ext = torch.zeros(N + 1, device=dev, dtype=dt)          # the last slot takes the off-mask scatters
    lab = ext[:N]
    lab.copy_(torch.where(mf, loc, torch.zeros((), device=dev, dtype=dt)))
    del loc
    bofs = (torch.arange(nb_, device=dev, dtype=torch.int64) * vol).repeat_interleave(vol)
    dummy = torch.full((), N, device=dev, dtype=torch.int64)
    zero = torch.zeros((), device=dev, dtype=dt)
    for it in range(int(max_iter)):
        nb = torch.where(mf, F.max_pool3d(lab.view(-1, 1, *sh[-3:]), 3, stride=1, padding=1).view(-1), zero)
        if it % LABEL_CHECK == 0 and torch.equal(nb, lab):
            break
        ext.scatter_reduce_(0, torch.where(mf, bofs + lab.long() - 1, dummy), nb, "amax")
        lab.copy_(torch.where(mf, torch.maximum(lab, nb), zero))
        del nb
        for _ in range(4):
            j = torch.where(mf, bofs + lab.long() - 1, dummy)
            lab.copy_(torch.where(mf, torch.maximum(lab, ext[j]), zero))
    return lab.view(sh).to(torch.int32)


def pool2(v):
    """`rvsm.ladder.pool2` of a uint8 tensor: zero-padded to an even shape, floor(sum of 8 / 8)."""
    s = list(v.shape)
    pad = []
    for a in (2, 1, 0):
        pad += [0, s[a] % 2]
    if any(pad):
        v = F.pad(v, pad)
    Z, Y, X = (int(q) // 2 for q in v.shape)
    r = v.reshape(Z, 2, Y, 2, X, 2)
    t = r[:, 0, :, 0, :, 0].to(torch.int16)
    for a, b, c in ((0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)):
        t += r[:, a, :, b, :, c]
    return (t >> 3).to(torch.uint8)
