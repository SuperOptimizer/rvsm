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
first-index reductions, plus one scatter-MAX in `label` whose result does not depend on the order the
atomics land in: no cudnn, no autotuning. Same inputs on the same device -> the same bits.
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
            def _pass(G, V, I, A, B, OA, OB, n, inner, NC: tl.constexpr, BY: tl.constexpr,
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
                r2 = tl.max(tl.max(gt, axis=1), axis=0)
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

            _TRITON = (_pass, _gauss, _first)
        except Exception:  # noqa: BLE001  -- no triton: the torch path
            _TRITON = False
    return _TRITON


def _triton_failed(e):
    global _TRITON
    _TRITON = False
    import warnings
    warnings.warn(f"rvsm.edt: Triton kernels unusable ({e!r}); using the torch implementations")


def _minplus_torch(g, chunk_bytes=256 << 20):
    """(v, i) with v[r, y] = min_i g[r, i] + (y - i)^2 and i its first argmin; g (R, N) float32."""
    R, N = g.shape
    ar = torch.arange(N, device=g.device, dtype=torch.float32)
    d2 = (ar[:, None] - ar[None, :]) ** 2                    # [y, i]
    v = torch.empty_like(g)
    ix = torch.empty((R, N), dtype=torch.int32, device=g.device)
    c = max(1, int(chunk_bytes // (4 * N * N)))
    for s in range(0, R, c):
        vv, ii = torch.min(g[s:s + c, None, :] + d2[None], dim=2)
        v[s:s + c] = vv
        ix[s:s + c] = ii.to(torch.int32)
    return v, ix


def _axis_pass_torch(g, axis, carry):
    gp = g.movedim(axis, -1)
    shp = gp.shape
    v, i = _minplus_torch(gp.contiguous().view(-1, shp[-1]))
    v = v.view(shp).movedim(-1, axis).contiguous()
    i = i.view(shp).movedim(-1, axis).contiguous()
    il = i.long()
    return v, i, [torch.gather(c, axis, il) for c in carry]


def _axis_pass(g, axis, carry=(), torch_only=False):
    """One separable pass along `axis` of the contiguous 3-D float32 `g`: (v, i, carried) with
    v = min over the line position p of g[..p..] + (y - p)^2, i its first argmin (int32) and every
    int32 tensor of `carry` gathered at that argmin."""
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
            k[0][(outer, -(-inner // BK), -(-n // BY))](g, v, i, a, b, oa, ob, n, inner, NC=len(carry),
                                                         BY=BY, BK=BK, num_warps=4 if BK > 1 else 2)
            return v, i, out
        except Exception as e:  # noqa: BLE001  -- a host where Triton cannot compile: the torch path
            _triton_failed(e)
    return _axis_pass_torch(g, axis, list(carry))


def _first_pass(surf, torch_only=False):
    """`_axis_pass` along axis 0 of the float version of the mask (0 on it, +inf off it)."""
    k = _triton_kernels() if (surf.is_cuda and not torch_only) else False
    if k:
        Z = int(surf.shape[0])
        inner = surf.numel() // max(Z, 1)
        v = torch.empty(surf.shape, dtype=torch.float32, device=surf.device)
        i = torch.empty(surf.shape, dtype=torch.int32, device=surf.device)
        BK = min(128, 1 << max(inner - 1, 0).bit_length())
        try:
            k[2][(1, -(-inner // BK))](surf.contiguous().view(torch.uint8), v, i, Z, inner, BK=BK)
            return v, i
        except Exception as e:  # noqa: BLE001
            _triton_failed(e)
    g = torch.where(surf, 0.0, INF).to(torch.float32).contiguous()
    v, i, _ = _axis_pass(g, 0, (), torch_only=True)
    return v, i


def edt2(surf, indices=True, torch_only=False, index_dtype=torch.int64):
    """(squared distance float32 (Z,Y,X), nearest index int64 (3,Z,Y,X)) to the True voxels of the 3-D
    bool tensor `surf`. With no True voxel at all the distance is +inf everywhere (the index 0).
    `indices=False` returns (squared distance, None) and carries no indices through the passes.
    `torch_only` skips the Triton kernels (the tests compare the two); `index_dtype=torch.int32` keeps
    the indices at half the memory."""
    assert surf.dim() == 3 and surf.dtype == torch.bool
    assert 3 * max(surf.shape) ** 2 < 2 ** 24, "squared distances must stay exact in float32"
    # scipy's order: axis 0, then 1, then 2; each pass carries the earlier passes' nearest coordinates
    v, iz = _first_pass(surf, torch_only)
    v, iy, c = _axis_pass(v, 1, (iz,) if indices else (), torch_only)
    del iz
    v, ix, c = _axis_pass(v, 2, (c[0], iy) if indices else (), torch_only)
    if not indices:
        return v, None
    return v, torch.stack((c[0], c[1], ix)).to(index_dtype)


def edt(surf):
    """(distance float32, nearest index int64 (3,Z,Y,X)): `scipy.ndimage.distance_transform_edt(~surf,
    return_indices=True)` with the distances exact and ties resolved as in the module docstring."""
    d2, idx = edt2(surf)
    return torch.sqrt(d2), idx


def max_filter3(x):
    """`maximum_filter(x, size=3, mode="nearest")` of a 3-D float tensor (replicating the border cannot
    raise a maximum, so padding with -inf is the same)."""
    return F.max_pool3d(x[None, None], 3, stride=1, padding=1)[0, 0]


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
    """`scipy.ndimage.binary_dilation(m)`: one step of the 6-neighbour cross, border value 0."""
    out = m.clone()
    for a in range(3):
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
    float32 output array does. The last bits can still differ from scipy (the order of the sum)."""
    w, r = gaussian_weights(sigma, truncate)
    y = x.to(torch.float32).contiguous()
    k = _triton_kernels() if y.is_cuda else False
    if k:
        try:
            wt = torch.tensor(w, dtype=torch.float64, device=y.device)
            for a in range(3):
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
    for a in range(3):
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


def label(m, max_iter=100000):
    """Connected components of the 3-D bool tensor `m` with 26-connectivity: an int64 tensor, 0 off
    `m`, equal on a voxel pair iff they are in one component. The labels are NOT scipy's numbering (only
    equality is meaningful), but they are a deterministic function of `m`.

    Every voxel starts as its own 1 + flat index; a label always names a voxel of the same component and
    only ever grows. Per iteration: each voxel's 3^3 neighbourhood maximum (max_pool3d) is hooked onto the
    voxel its label names (a scatter-max, order independent), taken by the voxel itself, and the labels
    are pointer-jumped (a voxel takes the label of the voxel its label names). It stops when every voxel
    already holds its neighbourhood maximum, i.e. the labels are constant on each component."""
    n = m.numel()
    dt = torch.float32 if n < 2 ** 24 else torch.float64
    sel = torch.nonzero(m.reshape(-1)).squeeze(1)
    lab = torch.zeros(n, device=m.device, dtype=dt)
    lab[sel] = (sel + 1).to(dt)
    for _ in range(int(max_iter)):
        nb = F.max_pool3d(lab.view(1, 1, *m.shape), 3, stride=1, padding=1).view(-1)[sel]
        cur = lab[sel]
        if torch.equal(nb, cur):
            break
        lab.scatter_reduce_(0, cur.long() - 1, nb, "amax")
        lab[sel] = torch.maximum(lab[sel], nb)
        for _ in range(4):
            c = lab[sel]
            lab[sel] = torch.maximum(c, lab[c.long() - 1])
    return lab.view(m.shape).long()


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
