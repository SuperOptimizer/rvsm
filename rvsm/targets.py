"""The distance fields: signed midline distance and sheet thickness, derived from a region's own stores.

Once a region has a `recto` store and, from the verso pass, a `verso` one, the two bands determine a
geometry the probability channels do not carry: where the MIDDLE of the sheet is, which side of it a
voxel is on, and how thick the sheet is there. `region_fields` computes that geometry and writes it as
two more stores, which the sampler then reads as the `midline` and `thickness` target channels.

THE ENCODING (the tracer contract, shared with `rvsm.losses`):

    signed   code = 128 + round(d / 0.25),  clamped to 1..255  ->  d = (code - 128) * 0.25 voxels
    unsigned code =       round(t / 0.25),  clamped to 1..255  ->  t =  code        * 0.25 voxels
    code 0   = NO DATA (the loader gives the voxel weight 0)

i.e. offset 128 and 0.25-voxel units, cap +-31.75 voxels: the +-32 cap of the contract with the one
endpoint given up so that 0 can be the no-data marker. The stores are written with `q=0` (LOSSLESS):
volcomp's q=8 rounds, and a stored 0 that reads back as a 6 would silently turn "no data" into a
-30.5-voxel distance. The cap is a STORAGE limit only: it is applied by the encoders, after all of the
geometry below has been computed on raw distances.

THE SIGN AND THE ORIENTATION. Positive is the radially OUTWARD side, pinned to `rvsm.axis.radial` (the
unit vector away from the umbilicus, z component 0): the same axis interpolation builds the model's
radial input channel, the sign of the target here and the sign of the exported normal, so they cannot
disagree about where the axis is. The RECTO face of a sheet is its OUTWARD face (larger radius), the
VERSO face its inward one: `losses.pair_bands` puts the recto band at m = +t/2 and the verso band at
m = -t/2, `export.SIGN_CONVENTION` points the normal from verso to recto (radially outward), and
`infer`'s verso trick (`sign=-1` negates only the radial input channels, so a recto-trained student
marks the other face) is the same statement from the network's side. Along the radial direction, a
sheet with its recto face at a and its verso face at a - t gives, for a voxel at radial coordinate x,

    d_r = x - a        d_v = x - (a - t)        so   m = (d_r + d_v) / 2   and   t = d_v - d_r > 0

outside the sheet on either side and inside it alike. Both signed distances INCREASE outward, so the
face normals n = grad d of the two faces of one sheet point the SAME way (outward): n_r . n_v ~ +1.

THE TARGET DEFINITION (`TARGET_DEF = "paired-v3"`; review findings T01-T04 and pass-3 P3-06..08; see
docs/recipe.md "Distance-field targets"). Per block, from the recto and verso probability stores at the
store's rung, over core + `halo`, with REACH, TMAX in rung-2 voxels divided by 2^(rung-2) and TMIN = 3
voxels at every rung (rung 2: 24 / 3 / 24, rung 3: 12 / 3 / 12, rung 4: 6 / 3 / 6):

1. FACES. The recto and verso faces are the medial surfaces of the thresholded bands; `d_r`, `d_v` are
   RAW (unclipped) signed Euclidean distances to them. Clamping to +-31.75 happens only at encoding.
2. REACH AND COVERAGE. `d_r` is valid only if the nearest recto face voxel is within REACH (`REACH <
   halo` is asserted), likewise `d_v`. A block with no recto face within reach is code 0 (T01); no verso
   store or an empty verso band is code 0 for both fields, never `midline = d_r` (T04). The voxel must
   also be further than max(|d_r|, |d_v|) + COVER_MARGIN from the edge of the REGION's stores: outside
   them nothing is observed (zero-filled air is not background), so a nearer face could hide there.
3. THICKNESS. t = d_v - d_r must lie in [TMIN, TMAX]. TMIN = 3 is the decoder's own floor
   (`losses.soft_thickness` = 3 + softplus at every rung, and the constructed bands have half-width 1.5,
   so a thinner pair is unrepresentable). A thinner or negative pair is REJECTED, never clamped. The
   coarse rungs therefore deliberately lose thin-sheet support: a 10-voxel rung-2 sheet is 2.5 voxels at
   rung 4 and has no field target there.
4. SAME-SHEET PAIRING. With p_r the voxel's nearest recto face point and p_v its nearest verso point:
   (a) RECIPROCAL: p_v's nearest recto point lies in the same connected recto BAND component as p_r
       (26-connected) or within sqrt(3) of it, and likewise p_r's nearest verso point against p_v;
   (b) NORMALS: n_r = grad d_r at p_r and n_v = grad d_v at p_v (central differences of the distances
       smoothed with a sigma-1.5 Gaussian; magnitude at least NORMAL_MIN, else degenerate) satisfy
       n . radial >= 0.5 for both and n_r . n_v >= 0.95 (both point outward, see THE SIGN);
   (c) NO INTERVENING FACE: walking the segment p_r -> p_v at <= 0.5-voxel spacing against BOTH bands,
       the walk leaves the recto band once and enters the verso band once: re-entering a recto band,
       or leaving a verso band after entering one, is another observed face in between. The bands, not
       the medial surfaces, are walked, because a digital medial surface of a curved sheet has gaps a
       segment can slip through; there is no distance threshold.
5. STENCIL AND GRADIENT. midline = (d_r + d_v) / 2 and thickness = d_v - d_r are kept only where the
   voxel and all six face neighbours pass 1-4 (the full central-difference stencil the Eikonal term
   uses) and the unquantised midline gradient norm lies in [0.8, 1.2]. Everything else is code 0, and
   so is everything within `axis_r_um` microns of the umbilicus axis.

Every field store records `target_def`, the per-rung `reach_vox` / `tmin_vox` / `tmax_vox`, `thr`, the
`recto_digest` / `verso_digest` of the source stores it was built from, a `support` histogram of why
core voxels were rejected (`SUPPORT`) and the same histogram per block (`support_blocks`). `_current`
is the single "is this field store up to date" predicate: `region_fields` skips on it and the producer's
scheduler (`run._next_job`) asks `fields_current`, so a store from an older definition, other
parameters or other source stores is regenerated, through `stores.write`'s tmp-dir-then-rename.

UNITS ARE VOXELS OF THAT RUNG, so a distance field is NEVER pooled: a 2x mean pool of a distance field
is not the distance field of the pooled mask (it is a distance in the FINE rung's voxels, halved by
nothing). Rungs 3 and 4 are therefore RECOMPUTED, from the 2x / 4x mean pool of the probability stores.
Above rung 4 there is no distance target at all: higher up the "band" is a pooled FRACTION and its 0.5
level set is not a surface.

WHERE THE STORES GO. One store per rung, under the round's own directory:

    <root>/stores/round_<r>/midline/region_<z>_<y>_<x>.zarr        rung 2
    <root>/stores/round_<r>/midline_r3/region_<z>_<y>_<x>.zarr     rung 3
    <root>/stores/round_<r>/midline_r4/region_<z>_<y>_<x>.zarr     rung 4

and likewise `thickness`, `thickness_r3`, `thickness_r4`. Each store records its own `rung` and
`voxel_um` in its attrs, and `region_fields` never pools one into another -- the suffix is there so that
a reader cannot accidentally take a rung-3 field for a rung-2 one by opening the wrong directory.

COST AND DETERMINISM. Two scipy EDTs per block per rung plus the pairing checks over the candidate
voxels, CPU only; `jobs > 1` spreads the blocks over worker processes. Every block is computed from the
stores alone, with a `halo` of context, and the parent assembles the cores, so the bytes written do not
depend on how the blocks were handed out: `jobs=4` is byte-identical to `jobs=1`.
"""
import hashlib
import json
import os

import numpy as np

from rvsm import axis as AX, ladder, stores

UNIT = 0.25          # voxels per code step
OFF = 128            # the code of distance 0
CAP = 31.75          # +-CAP voxels is the representable range (codes 1..255); applied at ENCODING only
TMIN = 3.0           # voxels AT EVERY RUNG: the decoder's thickness floor (3 + softplus); thinner = code 0
TMAX = 24.0          # RUNG-2 voxels (57.6 um): a thicker raw pair is not one sheet; / 2^(rung-2) above
REACH = 24.0         # RUNG-2 voxels (57.6 um): a face further than this is not the voxel's face; / 2^(rung-2)
COVER_MARGIN = 2.0   # voxels: the nearest-face ball must stay this far inside the region's stores
RECIPROCAL = 3 ** 0.5    # voxels: a reciprocal nearest point off the face's component may be this close
NORMAL_SIGMA = 1.5   # voxels: the Gaussian the signed distances are smoothed with before a face normal
NORMAL_MIN = 0.4     # |grad d| at a face point below this is a degenerate (tangent / crossing) face
NORMAL_RADIAL = 0.5  # n . radial >= this for both faces
NORMAL_AGREE = 0.95  # n_r . n_v >= this (both signed-distance normals point outward)
GRAD = (0.8, 1.2)    # the unquantised midline gradient norm a kept voxel must have
TARGET_DEF = "paired-v3"   # recorded in every field store; a done store under another one is regenerated
AXIS_R_UM = 400.0    # microns around the umbilicus axis that are dropped
MAX_RUNG = 4         # no distance target above this rung
BLOCK = 128          # the core a single EDT pair covers: exactly one store chunk
HALO = 48            # context voxels on every side; must exceed REACH (asserted)
KINDS = ("midline", "thickness")
SUPPORT = ("voxels", "no_recto", "no_verso", "coverage", "thickness", "reciprocal", "normal",
           "crossing", "stencil", "gradient", "valid")
_REASON = {k: i for i, k in enumerate(SUPPORT[1:-1], start=1)}     # reason code per rejection


def channel(kind, rung):
    """The store channel name of one field at one rung: `midline` at rung 2, `midline_r3` above it."""
    return str(kind) if int(rung) == 2 else f"{kind}_r{int(rung)}"


def rung_params(rung, reach=REACH, tmin=TMIN, tmax=TMAX):
    """(reach, tmin, tmax) in RUNG-`rung` voxels: `reach` and `tmax` are given in rung-2 voxels and scale
    with the rung (the same microns at every rung); `tmin` is the decoder's floor, the same number of
    voxels at every rung."""
    f = float(2 ** (int(rung) - 2))
    return float(reach) / f, float(tmin), float(tmax) / f


def thickness_bounds(rung, tmin=TMIN, tmax=TMAX):
    """(tmin, tmax) in RUNG-`rung` voxels (see `rung_params`)."""
    return rung_params(rung, REACH, tmin, tmax)[1:]


# ------------------------------------------------------------------------------------ the encoding

def encode_signed(d, valid, cap=CAP):
    """float voxels -> uint8, with `valid` False becoming code 0 (no data). The ONLY place the cap is
    applied to a midline."""
    c = np.rint(np.clip(d, -cap, cap) / UNIT) + OFF
    return np.where(valid, np.clip(c, 1, 255), 0).astype(np.uint8)


def decode_signed(u):
    """uint8 -> float voxels (code 0 decodes to -32; the CALLER must use the weight, not the value)."""
    return (np.asarray(u, np.float32) - OFF) * UNIT


def encode_unsigned(t, valid, cap=255 * UNIT):
    c = np.rint(np.clip(t, UNIT, cap) / UNIT)
    return np.where(valid, np.clip(c, 1, 255), 0).astype(np.uint8)


def decode_unsigned(u):
    return np.asarray(u, np.float32) * UNIT


# ------------------------------------------------------------------------------- geometry per block

def axis_offsets(ax, lo, shape):
    """(dy, dx, r) of every voxel of the block from the scroll axis, in this rung's voxels.

    `ax` is `axis.axis_at(axis, k)` -- the umbilicus polyline resampled to rung k -- and `lo` the block
    corner in the same voxels. This is exactly the expression `axis.radial` / `axis.radius` build the
    radial unit vector and the radius plane from, so the radial channel and the axis exclusion can never
    disagree about where the axis is."""
    Z, Y, X = (int(v) for v in shape)
    z = np.arange(Z, dtype=np.float64) + int(lo[0])
    cy, cx = np.interp(z, ax[0], ax[1]), np.interp(z, ax[0], ax[2])
    dy = (np.arange(Y, dtype=np.float64) + int(lo[1]))[None, :, None] - cy[:, None, None]
    dx = (np.arange(X, dtype=np.float64) + int(lo[2]))[None, None, :] - cx[:, None, None]
    r = np.sqrt(dy * dy + dx * dx)
    return dy.astype(np.float32), dx.astype(np.float32), r.astype(np.float32)


def medial(band):
    """A one-voxel-thick medial surface of a binary band: the voxels of `band` that are a local maximum
    of the distance to background. The same construction as `losses.skeleton`, on the CPU with an exact
    Euclidean transform instead of a capped Chebyshev one."""
    from scipy import ndimage as ndi
    if not band.any():
        return np.zeros(band.shape, bool)
    d = ndi.distance_transform_edt(band)
    return band & (d >= ndi.maximum_filter(d, size=3, mode="nearest") - 1e-6)


def face_distance(surf, dy, dx):
    """(d, u, ix) to the one-voxel surface `surf`, or None when `surf` is empty.

    `u` is the RAW (unclipped) Euclidean distance to the nearest surface voxel, `ix` (3,Z,Y,X) WHICH
    voxel that is (`distance_transform_edt(..., return_indices=True)`), and `d = +-u` the signed
    distance, POSITIVE on the radially outward side: the displacement from the nearest surface voxel to
    this one, dotted with the (unnormalised) radial direction (dy, dx, z-component 0), is the side. The
    radial vector is the one the model gets as an input channel, so "positive" means the same thing in
    the target, in the stem's radial channels and in the exported normal (`dot(n, radial) > 0`).

    Where the displacement is exactly perpendicular to the radial direction the side is arbitrary
    (resolved to +); such a face has a degenerate normal and is rejected by the pairing checks."""
    from scipy import ndimage as ndi
    if not surf.any():
        return None
    u, ix = ndi.distance_transform_edt(~surf, return_indices=True)
    gy = (np.arange(surf.shape[1], dtype=np.int32)[None, :, None] - ix[1]).astype(np.float32)
    gx = (np.arange(surf.shape[2], dtype=np.int32)[None, None, :] - ix[2]).astype(np.float32)
    u = u.astype(np.float32)
    d = np.where(gy * dy + gx * dx < 0, -u, u).astype(np.float32)
    if max(surf.shape) < 2 ** 15:
        ix = ix.astype(np.int16)        # both faces' indices are held at once by `block_fields`
    return d, u, ix


def _grad_at(d, p):
    """Central-difference gradient (3, n) of the field `d` at the integer points `p` (3, n); one-sided at
    the array border."""
    g = np.empty(p.shape, np.float32)
    for a in range(3):
        hi, lo = p.copy(), p.copy()
        hi[a] = np.minimum(p[a] + 1, d.shape[a] - 1)
        lo[a] = np.maximum(p[a] - 1, 0)
        g[a] = (d[tuple(hi)] - d[tuple(lo)]) / np.maximum(hi[a] - lo[a], 1)
    return g


def _pair_checks(sel, shape, br, bv, dr, dv, ixr, ixv, dy, dx):
    """Rule 4 for the flat voxel indices `sel`: a reason code per voxel (0 = a same-sheet pair). `br` /
    `bv` are the thresholded BANDS (solid, unlike a digital medial surface, which fragments on a curved
    sheet and would let a segment slip through its gaps)."""
    from scipy import ndimage as ndi
    cube = np.ones((3, 3, 3), bool)
    lr, lv = ndi.label(br, cube)[0].reshape(-1), ndi.label(bv, cube)[0].reshape(-1)
    pr = ixr.reshape(3, -1)[:, sel].astype(np.int64)
    pv = ixv.reshape(3, -1)[:, sel].astype(np.int64)
    fr = np.ravel_multi_index(tuple(pr), shape)
    fv = np.ravel_multi_index(tuple(pv), shape)
    # (a) reciprocal nearest points: same band component, or a 26-neighbour
    back_r = ixr.reshape(3, -1)[:, fv].astype(np.int64)
    back_v = ixv.reshape(3, -1)[:, fr].astype(np.int64)
    fbr = np.ravel_multi_index(tuple(back_r), shape)
    fbv = np.ravel_multi_index(tuple(back_v), shape)
    ok_rec = ((lr[fbr] == lr[fr]) | (np.sqrt(((back_r - pr) ** 2).sum(0)) <= RECIPROCAL + 1e-6)) & \
             ((lv[fbv] == lv[fv]) | (np.sqrt(((back_v - pv) ** 2).sum(0)) <= RECIPROCAL + 1e-6))
    # (b) face normals from the SMOOTHED signed distances (a digital face is jagged at one voxel):
    # stable, radial, and the two faces of one sheet facing the same (outward) way
    z, y, x = np.unravel_index(sel, shape)
    ry = np.broadcast_to(dy, shape)[z, y, x].astype(np.float32)
    rx = np.broadcast_to(dx, shape)[z, y, x].astype(np.float32)
    rn = np.maximum(np.sqrt(ry * ry + rx * rx), 1e-6)
    ry, rx = ry / rn, rx / rn
    nr = _grad_at(ndi.gaussian_filter(dr, NORMAL_SIGMA, mode="nearest"), pr)
    nv = _grad_at(ndi.gaussian_filter(dv, NORMAL_SIGMA, mode="nearest"), pv)
    mr, mv = np.sqrt((nr * nr).sum(0)), np.sqrt((nv * nv).sum(0))
    nr, nv = nr / np.maximum(mr, 1e-6), nv / np.maximum(mv, 1e-6)
    ok_n = (mr >= NORMAL_MIN) & (mv >= NORMAL_MIN) & \
           (nr[1] * ry + nr[2] * rx >= NORMAL_RADIAL) & (nv[1] * ry + nv[2] * rx >= NORMAL_RADIAL) & \
           ((nr * nv).sum(0) >= NORMAL_AGREE)
    code = np.where(~ok_rec, _REASON["reciprocal"], np.where(~ok_n, _REASON["normal"], 0)).astype(np.uint8)
    # (c) the ordered walk p_r -> p_v at <= 0.5-voxel spacing: leave the own recto band once, enter the
    # verso band once and stay in it. Re-entering a recto band, or leaving a verso band after entering
    # one, is another observed face on the segment.
    w = np.flatnonzero(code == 0)
    if w.size:
        a, b = pr[:, w].astype(np.float32), pv[:, w].astype(np.float32)
        seg = b - a
        n = max(1, int(np.ceil(2.0 * float(np.sqrt((seg * seg).sum(0)).max()))))
        left_r = np.zeros(w.size, bool)
        in_v = np.zeros(w.size, bool)
        hit = np.zeros(w.size, bool)
        for i in range(n + 1):
            qi = tuple(np.rint(a + (i / n) * seg).astype(np.int64))
            rr, vv = br[qi], bv[qi]
            hit |= left_r & rr & ~vv
            left_r |= ~rr
            hit |= in_v & ~vv
            in_v |= vv
        code[w[hit]] = _REASON["crossing"]
    return code


def block_fields(recto, verso, dy, dx, thr=0.5, reach=REACH, tmin=TMIN, tmax=TMAX, core=None,
                 cover=None):
    """(midline, thickness, valid, support) of one block; `reach` / `tmin` / `tmax` in THIS block's voxels.

    `recto` / `verso` are uint8 probability blocks (verso may be None). `core` marks the voxels whose
    result is wanted (default: all); the rules (module docstring, 1-5) are evaluated on `core` plus its
    six-neighbour ring, which the stencil rule needs. `cover`, when given, is each voxel's distance to
    the edge of the observed stores (rule 2). `midline` / `thickness` are raw float voxels, meaningful
    only where `valid`; `support` counts, over `core`, the first rule each rejected voxel failed."""
    from scipy import ndimage as ndi
    shape = recto.shape
    core = np.ones(shape, bool) if core is None else np.asarray(core, bool)
    ev = ndi.binary_dilation(core)
    reason = np.zeros(shape, np.uint8)

    def fail(bad, key):
        reason[(reason == 0) & ev & bad] = _REASON[key]

    def result(m, t):
        valid = (reason == 0) & core
        cnt = np.bincount(reason[core], minlength=len(SUPPORT) - 1)
        sup = {"voxels": int(core.sum()), "valid": int(valid.sum())}
        sup.update({k: int(cnt[i]) for k, i in _REASON.items()})
        z = np.float32(0)
        return np.where(valid, m, z).astype(np.float32), np.where(valid, t, z).astype(np.float32), valid, sup

    zero = np.zeros(shape, np.float32)
    lvl = int(round(thr * 255))
    br = recto >= lvl
    fr = face_distance(medial(br), dy, dx)
    if fr is None:
        fail(np.ones(shape, bool), "no_recto")
        return result(zero, zero)
    dr, ur, ixr = fr
    fail(ur > reach, "no_recto")
    bv = None if verso is None else verso >= lvl
    fv = None if bv is None else face_distance(medial(bv), dy, dx)
    if fv is None:
        fail(np.ones(shape, bool), "no_verso")
        return result(zero, zero)
    dv, uv, ixv = fv
    fail(uv > reach, "no_verso")
    if cover is not None:
        fail(cover <= np.maximum(ur, uv) + COVER_MARGIN, "coverage")
    t = dv - dr
    fail((t < tmin) | (t > tmax), "thickness")
    sel = np.flatnonzero((reason == 0) & ev)
    if sel.size:
        reason.reshape(-1)[sel] = _pair_checks(sel, shape, br, bv, dr, dv, ixr, ixv, dy, dx)
    pair = (reason == 0) & ev
    m = np.where(pair, 0.5 * (dr + dv), 0.0).astype(np.float32)
    full = pair.copy()
    for a in range(3):
        for s in (1, -1):
            nb = np.zeros(shape, bool)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            src[a], dst[a] = (slice(1, None), slice(None, -1)) if s == 1 else (slice(None, -1), slice(1, None))
            nb[tuple(dst)] = pair[tuple(src)]
            full &= nb
    fail(pair & ~full, "stencil")
    g2 = np.zeros(shape, np.float32)
    for a in range(3):
        g = np.zeros(shape, np.float32)
        hi, lo, mid = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        hi[a], lo[a], mid[a] = slice(2, None), slice(None, -2), slice(1, -1)
        g[tuple(mid)] = 0.5 * (m[tuple(hi)] - m[tuple(lo)])
        g2 += g * g
    gn = np.sqrt(g2)
    fail(full & ((gn < GRAD[0]) | (gn > GRAD[1])), "gradient")
    return result(m, t)


def cover_distance(lo, shape, extent):
    """Each voxel's distance to the nearest voxel OUTSIDE the stores (0 outside them), for a box at
    store-local corner `lo` of `shape`, the stores spanning [0, extent) per axis."""
    out = None
    for a in range(3):
        c = np.arange(int(shape[a]), dtype=np.float32) + float(lo[a])
        d = np.where((c >= 0) & (c < extent[a]), np.minimum(c + 1, float(extent[a]) - c), 0.0)
        d = d.reshape([-1 if i == a else 1 for i in range(3)])
        out = d if out is None else np.minimum(out, d)
    return np.broadcast_to(out, tuple(int(v) for v in shape)).astype(np.float32)


# ------------------------------------------------------------------------- reading a rung out of a store

def read_pooled(arr, rung, lo, shape):
    """A (Z,Y,X) uint8 block of a rung-2 store, at `rung` (2, 3 or 4), in that rung's voxels.

    The window is read from the store at the FINE resolution and pooled here. Because the fine window is
    aligned to the pooling factor (the store's origin is the region corner and `lo` is in rung-`rung`
    voxels), pooling the window equals pooling the whole store and then slicing it -- so a block's
    contents do not depend on the block grid. Anything outside the store reads as 0 (air), and zero
    filling commutes with the pool for the same alignment reason."""
    d = int(rung) - 2
    assert d >= 0, f"read_pooled: rung {rung} is below the store's rung 2"
    f = 1 << d
    lo_f = np.asarray(lo, np.int64) * f
    n_f = np.asarray(shape, np.int64) * f
    S = np.asarray(arr.shape[-3:], np.int64)
    out = np.zeros(tuple(int(v) for v in n_f), np.uint8)
    a, b = np.maximum(lo_f, 0), np.minimum(lo_f + n_f, S)
    if (b > a).all():
        blk = np.asarray(arr[tuple(slice(int(p), int(q)) for p, q in zip(a, b))], np.uint8)
        st = a - lo_f
        out[st[0]:st[0] + blk.shape[0], st[1]:st[1] + blk.shape[1], st[2]:st[2] + blk.shape[2]] = blk
    for _ in range(d):
        out = ladder.pool2(out)
    return out


# ------------------------------------------------------------------------------ the per-block worker

_CTX = {}


def _init(recto_path, verso_path, ax, thr, cap, tmin, tmax, reach, axis_r_um, halo):
    """Per-process state of a `jobs > 1` worker. Nothing useful is inherited across the fork (an open
    zarr array does not survive it), so every worker opens the stores itself, once."""
    _CTX.clear()
    _CTX.update(recto_path=recto_path, verso_path=verso_path, ax=np.asarray(ax, np.float64), thr=thr,
                cap=cap, tmin=tmin, tmax=tmax, reach=reach, axis_r_um=axis_r_um, halo=halo, arr={},
                axk={})


def _arr(key):
    if key not in _CTX["arr"]:
        p = _CTX[f"{key}_path"]
        _CTX["arr"][key] = None if not p else stores.open_store(p)
    return _CTX["arr"][key]


def _axis_at(k):
    if k not in _CTX["axk"]:
        _CTX["axk"][k] = AX.axis_at(_CTX["ax"], k)
    return _CTX["axk"][k]


def _block(task):
    """One core block: (rung, lo, core shape) -> (rung, lo, midline u8, thickness u8, support).

    `lo` is the core corner in GLOBAL rung-`rung` voxels; the store is read with a `halo` of context on
    every side so that a distance measured inside the core sees every face within reach of it."""
    k, lo, n = task
    halo = _CTX["halo"]
    rec_a, ver_a = _arr("recto"), _arr("verso")
    o = np.asarray(rec_a.attrs["origin_zyx"], np.int64) >> (int(k) - 2)   # the region corner at this rung
    loc = np.asarray(lo, np.int64) - o                                   # store-local core corner
    rlo = loc - halo
    rsh = tuple(int(v) + 2 * halo for v in n)
    sl = tuple(slice(halo, halo + int(v)) for v in n)
    dy, dx, r = axis_offsets(_axis_at(k), np.asarray(lo, np.int64) - halo, rsh)
    core = np.zeros(rsh, bool)
    core[sl] = r[sl] >= (_CTX["axis_r_um"] / ladder.rung_um(k))   # the crushed core carries no sign
    ext = np.asarray(rec_a.shape[-3:], np.int64) >> (int(k) - 2)
    rec = read_pooled(rec_a, k, rlo, rsh)
    ver = None if ver_a is None else read_pooled(ver_a, k, rlo, rsh)
    reach, tmin, tmax = rung_params(k, _CTX["reach"], _CTX["tmin"], _CTX["tmax"])
    m, t, ok, sup = block_fields(rec, ver, dy, dx, thr=_CTX["thr"], reach=reach, tmin=tmin, tmax=tmax,
                                 core=core, cover=cover_distance(rlo, rsh, ext))
    return k, tuple(int(v) for v in lo), encode_signed(m[sl], ok[sl], _CTX["cap"]), \
        encode_unsigned(t[sl], ok[sl]), sup


def _block_in(arg):
    """`_block` for a PERSISTENT pool (`field_pool`): the task carries its region's init, and a worker
    re-initialises only when the region changes."""
    init, task = arg
    key = (init[0], init[1]) + tuple(init[3:])       # the store paths and the scalars (not the axis array)
    if _CTX.get("key") != key:
        _init(*init)
        _CTX["key"] = key
    return _block(task)


def die_with_parent():
    """Ask the kernel to SIGKILL this process when its parent dies (Linux `PR_SET_PDEATHSIG`; a no-op
    elsewhere). A fields-pool worker's parent is the pool's forkserver, which exits when the producer
    does -- so a producer the supervisor kills (or that crashes) no longer leaves its pool computing
    EDTs beside the producer that replaces it."""
    try:
        import ctypes
        import signal
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, int(signal.SIGKILL), 0, 0, 0)      # 1 = PR_SET_PDEATHSIG
        if os.getppid() == 1:                              # the parent died before the prctl
            os._exit(0)
    except Exception:  # noqa: BLE001
        pass


def _nice():
    die_with_parent()
    try:
        os.nice(10)       # the fields are background work: the trainer's loader keeps the cores it needs
    except OSError:
        pass


def field_pool(jobs):
    """A process pool for `region_fields(pool=...)` that lives as long as the producer.

    `forkserver`, not `fork`: the producer that owns it has CUDA and several threads, and a forked child
    of a threaded process can inherit a lock some other thread held at the fork. The workers run at
    nice 10 and die with their parent (`die_with_parent`)."""
    import concurrent.futures as cf
    import multiprocessing as mp
    return cf.ProcessPoolExecutor(max_workers=int(jobs), mp_context=mp.get_context("forkserver"),
                                  initializer=_nice)


def _blocks(shape, block):
    for z in range(0, int(shape[0]), block):
        for y in range(0, int(shape[1]), block):
            for x in range(0, int(shape[2]), block):
                yield (z, y, x), (min(block, int(shape[0]) - z), min(block, int(shape[1]) - y),
                                  min(block, int(shape[2]) - x))


def _pad128(n):
    return int(-(-int(n) // stores.CHUNK) * stores.CHUNK)


# ------------------------------------------------------------------------------ generation identity

def source_digest(path):
    """The identity of a finished source store ("" when it is not done): sha256 over its `zarr.json`
    (shape, codec and attrs, which carry the producer, checkpoint and step) and the relative path and
    size of every data file. Not a content hash -- reading every shard on each scheduler poll is not
    affordable -- but any rewrite by a different producer or generation changes it."""
    if not stores.is_done(path):
        return ""
    h = hashlib.sha256()
    with open(os.path.join(path, "zarr.json"), "rb") as f:
        h.update(f.read())
    for d, dirs, files in sorted(os.walk(path)):
        dirs.sort()
        for fn in sorted(files):
            if fn == "zarr.json" and d == path:
                continue
            p = os.path.join(d, fn)
            h.update(f"{os.path.relpath(p, path)}:{os.path.getsize(p)};".encode())
    return h.hexdigest()[:16]


def _want(root, lo, round_, rung, reach, tmin, tmax, thr):
    rk, tn, tx = rung_params(rung, reach, tmin, tmax)
    return {"target_def": TARGET_DEF, "reach_vox": rk, "tmin_vox": tn, "tmax_vox": tx, "thr": float(thr),
            "recto_digest": source_digest(stores.store_path(root, "recto", lo, round_)),
            "verso_digest": source_digest(stores.store_path(root, "verso", lo, round_))}


def _current(path, want):
    """THE predicate for "this field store is up to date", shared by `region_fields` (skip) and the
    producer's scheduler (`fields_current`): done, and either written by `region_fields` under exactly
    `want` (definition, per-rung parameters, threshold and source-store digests) or written by something
    else entirely (a student pass's own rung-2 field head, which records no `field`)."""
    if not stores.is_done(path):
        return False
    with open(os.path.join(path, "zarr.json")) as f:
        a = json.load(f).get("attributes", {})
    if "field" not in a:
        return True
    return all(a.get(key) == v for key, v in want.items())


def fields_current(root, lo, round_=0, rungs=(2, 3, 4), reach=REACH, tmin=TMIN, tmax=TMAX, thr=0.5):
    """True iff every field store of the region at `rungs` is `_current`: exactly the test
    `region_fields` skips a rung on. The scheduler must ask this, not `stores.is_done`, or a store from an
    older definition or older source stores is never rebuilt."""
    return all(_current(stores.store_path(root, channel(kind, k), lo, round_),
                        _want(root, lo, round_, k, reach, tmin, tmax, thr))
               for k in rungs for kind in KINDS)


def region_fields(root, lo, ax, round_=0, jobs=1, rungs=(2, 3, 4), axis_r_um=AXIS_R_UM,
                  thr=0.5, cap=CAP, tmin=TMIN, tmax=TMAX, reach=REACH, block=BLOCK, halo=HALO,
                  force=False, pool=None):
    """Build the `midline` and `thickness` stores of one region, at every rung in `rungs`.

    `root` is the run directory, `lo` the region corner in rung-2 voxels, `ax` the umbilicus control
    points in rung-2 voxels (`axis.load`), `round_` the self-distillation round whose stores to read and
    write. The region's `recto` store must be done; `verso` may be missing, in which case both fields
    are no-data (code 0) everywhere: there is no recto-only midline (module docstring, rule 2).

    `reach` / `tmax` are in RUNG-2 voxels and `tmin` in voxels of every rung (`rung_params`); `reach`
    must be below `halo`.

    Returns a report dict, with a per-rung `support` histogram of why core voxels were rejected. A rung
    whose two stores are `_current` is skipped (`force` recomputes), which is what makes this resumable
    at region granularity. `pool` is a `field_pool` the caller keeps alive across regions (a producer);
    without one, `jobs > 1` forks a pool for this call.

    A pooled rung whose shape is not a multiple of 128 is padded up to one, because a store's shape must
    be; the padding is code 0, i.e. no data. For the production region (1024 at rung 2) rungs 3 and 4 are
    512 and 256 and nothing is padded."""
    assert 0 < float(reach) < int(halo), f"reach {reach} must be positive and below the halo {halo}"
    assert 0 <= float(tmin) <= float(tmax), (tmin, tmax)
    rp = stores.store_path(root, "recto", lo, round_)
    vp = stores.store_path(root, "verso", lo, round_)
    rec = stores.open_store(rp)                       # raises unless the recto pass has finished
    if not stores.is_done(vp):
        vp = ""
    S2 = np.asarray(rec.shape[-3:], np.int64)
    origin2 = np.asarray(rec.attrs["origin_zyx"], np.int64)
    rep = {"region": [int(v) for v in lo], "round": int(round_), "verso": bool(vp), "rungs": {}}
    todo = []
    for k in sorted(int(q) for q in rungs):
        assert 2 <= k <= MAX_RUNG, f"no distance target at rung {k} (2..{MAX_RUNG} only)"
        paths = {kind: stores.store_path(root, channel(kind, k), lo, round_) for kind in KINDS}
        want = _want(root, lo, round_, k, reach, tmin, tmax, thr)
        if not force and all(_current(p, want) for p in paths.values()):
            rep["rungs"][k] = {"skipped": "done"}
            continue
        Sk = S2 >> (k - 2)
        todo.append((k, Sk, paths, want))
    if not todo:
        return rep
    tasks = [(k, tuple(int(q) + int(o) for q, o in zip(b, origin2 >> (k - 2))), n)
             for k, Sk, _, _ in todo for b, n in _blocks(Sk, block)]
    init = (rp, vp, np.asarray(ax, np.float64), thr, cap, float(tmin), float(tmax), float(reach),
            float(axis_r_um), int(halo))
    out = {k: {kind: np.zeros(tuple(_pad128(v) for v in Sk), np.uint8) for kind in KINDS}
           for k, Sk, _, _ in todo}
    sup = {k: dict.fromkeys(SUPPORT, 0) for k, _, _, _ in todo}
    per_block = {k: [] for k, _, _, _ in todo}

    def take(res):
        k, blo, m, t, s = res
        _store_block(out[k], k, blo, origin2, m, t)
        for key in SUPPORT:
            sup[k][key] += int(s[key])
        per_block[k].append([int(v) for v in blo] + [int(s[key]) for key in SUPPORT])

    if pool is not None and len(tasks) >= 2:
        for res in pool.map(_block_in, [(init, t) for t in tasks], chunksize=1):
            take(res)
    elif int(jobs) <= 1 or len(tasks) < 2:
        _init(*init)
        for t in tasks:
            take(_block(t))
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=int(jobs), mp_context=mp.get_context("fork"),
                                    initializer=_init, initargs=init) as ex:
            for res in ex.map(_block, tasks, chunksize=1):
                take(res)
    for k, Sk, paths, want in todo:
        for kind in KINDS:
            stores.write(paths[kind], out[k][kind], tuple(int(v) for v in (origin2 >> (k - 2))),
                         rung=k, channels=(channel(kind, k),), q=0,
                         attrs={"field": kind, "unit_vox": UNIT, "offset": OFF if kind == "midline" else 0,
                                "cap_vox": float(cap), "axis_r_um": float(axis_r_um),
                                "shape_true": [int(v) for v in Sk], "source_round": int(round_),
                                "verso": bool(vp), "support": sup[k],
                                "support_blocks": {"columns": ["z", "y", "x", *SUPPORT],
                                                   "rows": per_block[k]}, **want})
        rep["rungs"][k] = {"shape": [int(v) for v in Sk], "blocks": sum(1 for t in tasks if t[0] == k),
                           "support": sup[k], "written": [paths[kind] for kind in KINDS]}
    return rep


def _store_block(dst, k, blo, origin2, m, t):
    """Place one worker's core block into the rung-k output arrays (parent side, so the assembly order
    cannot change the bytes)."""
    loc = np.asarray(blo, np.int64) - (origin2 >> (int(k) - 2))
    sl = tuple(slice(int(a), int(a) + int(s)) for a, s in zip(loc, m.shape))
    dst["midline"][sl] = m
    dst["thickness"][sl] = t
