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

COST AND DETERMINISM. Per block and rung: four EDTs (the medial surface of each band, then the distance
to each face with its nearest-voxel indices), two 26-connected labellings, two Gaussians and the pairing
checks over the candidate voxels. Two implementations of the same rules:

  * numpy / scipy (`block_fields`), CPU: ~7-9 s per 224^3 block on one core; `jobs > 1` spreads the
    blocks over worker processes (`field_pool` in the producer). ~1000 s per 1024^3 region at rungs
    2-4 on the 8-core production host.
  * torch (`block_fields_torch`, `region_fields(device=...)`), the producer's GPU: `rvsm.edt`'s exact
    EDT (Triton kernels on CUDA) and ndimage replacements, written op for op in the numpy version's
    float32 order, `FIELD_BATCH` = 3 blocks per device batch. ~50 ms of device time per block on an
    RTX 5080 laptop GPU, ~0.83 GB of VRAM per block in the batch (2.5 GB at peak); the whole region (the
    stores decoded one chunk row at a time, pooled on the device, 584 blocks, then the six q=0 writes)
    ~53 s, of which ~13 s is the volcomp encode of the last stores. Host memory peaks at ~3.4 GB over
    the process (the two 1 GB rung-2 output arrays the one-write-per-shard rule needs, a few decoded
    chunk rows, the pooled rungs), no child process. On a GPU behind a proxy what counts is launches and
    synchronisations, so a batch goes through in ~540 launches (~180 per block, against ~1190 for one
    block before batching and fused labelling / walk kernels) and ~4 synchronisations (the pair-check
    compaction, one convergence check per labelling, the one read-back).

The two agree: EDT distances are exact integers under a sqrt in both, the nearest-voxel TIES are broken
the same way (each pass takes the first minimum along its line, in scipy's axis order -- see
`rvsm.edt`, TIES; observed identical on every test mask, not proven for every scipy version), and the
Gaussian can differ from scipy's only in the last float32 bit (summation order). On every synthetic
fixture and region in the tests the stores are byte-identical; the tests hold them to 1e-3 and 0.1% of
the valid mask. A difference would show only at a voxel exactly equidistant from two face voxels or at a
normal test on its threshold.

Either way every block is computed from the stores alone, with a `halo` of context, and the cores are
assembled in block order through `_store_block`, so the bytes written do not depend on how the blocks
were handed out: `jobs=4` is byte-identical to `jobs=1`, and the device path is deterministic on one
device (no atomics whose order matters, no cudnn, no autotuning).
"""
import hashlib
import json
import math
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


# ------------------------------------------------------------------------ the same block, in torch
#
# `block_fields_torch` is `block_fields` op for op, with scipy.ndimage replaced by `rvsm.edt` so that it
# runs on the producer's GPU. The float32 arithmetic is written in the same order as the numpy version
# (no fused expressions), so where the EDT names the same nearest voxel the results are the same bits;
# what can differ is listed in `rvsm.edt` (TIES) and at `gaussian3`.
#
# It works on a BATCH of equally shaped blocks at once, (B, Z, Y, X), so that a GPU behind a proxy that
# charges per kernel launch or per synchronisation pays each once per batch, not once per block: every
# operator acts per volume, the pair checks carry each voxel's block offset, and nothing is read back
# to the host until the batch's encoded cores and support counts. The numpy version's early returns (no
# recto face, no verso face) become masks: an empty face set has +inf distance everywhere, which fails
# exactly the voxels the early return failed, with the same reason, and leaves no voxel for the pair
# checks.

def _tt(a, dev, dt=None):
    import torch
    t = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(a))
    t = t.to(dev)
    return t if dt is None else t.to(dt)


def medial_torch(band, cap=None):
    """`medial` of a bool tensor (a volume or a batch). The local-maximum test is made on the SQUARED
    distances, which are exact integers: for integers a < b, sqrt(b) - sqrt(a) > 1e-6 at any size a block
    can have, so `d2 >= max(d2)` is exactly scipy's `d >= max(d) - 1e-6`. An empty band has no medial
    voxel (its complement is everywhere at distance 0).

    `cap` (a band is a few voxels thick): the transform is capped (`edt.edt2(cap=)`), which is exact
    wherever the depth is <= cap; when some band voxel is deeper (it comes back +inf) the batch is
    transformed again uncapped, so the result is always the uncapped one."""
    import torch
    from rvsm import edt as E
    d2, _ = E.edt2(~band, indices=False, cap=cap)
    if cap is not None and bool(torch.any(band & torch.isinf(d2))):
        d2, _ = E.edt2(~band, indices=False)
    return band & (d2 >= E.max_filter3(d2))


def face_distance_torch(surf, dy, dx, cap=None):
    """`face_distance` of a bool (B,Z,Y,X) batch: (d, u, ix int32 (3,B,Z,Y,X)). A block with no surface
    voxel has u = +inf (and d = +-inf) everywhere instead of `None`. With `cap` (`edt_cap`), a voxel
    further than `cap` from the surface has u = +inf and an unspecified index; every other voxel's
    d, u and ix are exactly the uncapped ones."""
    import torch
    from rvsm import edt as E
    d2, ix = E.edt2(surf, index_dtype=torch.int32, cap=cap)
    u = torch.sqrt(d2)
    del d2
    Y, X = surf.shape[-2:]
    gy = (torch.arange(Y, device=surf.device)[None, None, :, None] - ix[1]).to(torch.float32)
    gx = (torch.arange(X, device=surf.device)[None, None, None, :] - ix[2]).to(torch.float32)
    side = gy * dy
    side = side + gx * dx
    del gy, gx
    d = torch.where(side < 0, -u, u)
    return d, u, ix


def _grad_at_torch(d, b, p):
    """`_grad_at` of the (B,Z,Y,X) field `d` at the points `p` (3, n) int64 of the blocks `b` (n,)."""
    import torch
    g = torch.empty(p.shape, dtype=torch.float32, device=d.device)
    flat = d.reshape(-1)
    base = _ravel(p, d.shape) + b * (int(d.shape[1]) * int(d.shape[2]) * int(d.shape[3]))
    step = (int(d.shape[2]) * int(d.shape[3]), int(d.shape[3]), 1)
    for a in range(3):
        hi = torch.clamp(p[a] + 1, max=d.shape[a + 1] - 1)
        lo = torch.clamp(p[a] - 1, min=0)
        g[a] = (flat[base + (hi - p[a]) * step[a]] - flat[base + (lo - p[a]) * step[a]]) / \
            torch.clamp(hi - lo, min=1)
    return g


def _ravel(p, shape):
    """Flat int64 index within one (Z,Y,X) volume of the coordinates `p` (3, n)."""
    return (p[0].long() * int(shape[-2]) + p[1]) * int(shape[-1]) + p[2]


def _pair_checks_torch(sel, shape, br, bv, dr, dv, ixr, ixv, dy, dx, walk_n):
    """`_pair_checks` for the flat int64 indices `sel` into the (B,Z,Y,X) batch `shape`: a reason code
    per voxel. `walk_n` bounds the walk's sample count (4 * reach + 2 covers every same-sheet pair)."""
    import torch
    from rvsm import edt as E
    B, Z, Y, X = (int(v) for v in shape)
    V = Z * Y * X
    lr, lv = E.label(br).reshape(-1), E.label(bv).reshape(-1)
    b = sel // V
    loc = sel - b * V
    boff = b * V
    ir, iv = ixr.reshape(3, -1), ixv.reshape(3, -1)
    pr, pv = ir[:, sel], iv[:, sel]                      # int32 coordinates: half the memory
    fr, fv = boff + _ravel(pr, shape), boff + _ravel(pv, shape)
    back_r, back_v = ir[:, fv], iv[:, fr]
    fbr, fbv = boff + _ravel(back_r, shape), boff + _ravel(back_v, shape)

    def near(q, p):
        q = (q - p).to(torch.float64)
        return torch.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) <= RECIPROCAL + 1e-6
    ok_rec = ((lr[fbr] == lr[fr]) | near(back_r, pr)) & ((lv[fbv] == lv[fv]) | near(back_v, pv))
    del lr, lv, back_r, back_v, fbr, fbv, fr, fv
    z, y, x = loc // (Y * X), (loc // X) % Y, loc % X
    del loc
    ry = dy.expand(B, Z, Y, X)[b, z, y, x].to(torch.float32)
    rx = dx.expand(B, Z, Y, X)[b, z, y, x].to(torch.float32)
    del z, y, x
    rn = torch.clamp(torch.sqrt(ry * ry + rx * rx), min=1e-6)
    ry, rx = ry / rn, rx / rn
    nr = _grad_at_torch(E.gaussian3(dr, NORMAL_SIGMA), b, pr)
    nv = _grad_at_torch(E.gaussian3(dv, NORMAL_SIGMA), b, pv)

    def norm(n):
        return torch.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    mr, mv = norm(nr), norm(nv)
    nr, nv = nr / torch.clamp(mr, min=1e-6), nv / torch.clamp(mv, min=1e-6)
    agree = nr[0] * nv[0] + nr[1] * nv[1] + nr[2] * nv[2]
    ok_n = (mr >= NORMAL_MIN) & (mv >= NORMAL_MIN) & \
           (nr[1] * ry + nr[2] * rx >= NORMAL_RADIAL) & (nv[1] * ry + nv[2] * rx >= NORMAL_RADIAL) & \
           (agree >= NORMAL_AGREE)
    code = torch.where(~ok_rec, _REASON["reciprocal"], torch.where(~ok_n, _REASON["normal"], 0)).to(torch.uint8)
    del nr, nv, mr, mv, ry, rx, ok_n, ok_rec, agree
    # (c) the walk, over the voxels that passed (a) and (b), with each BLOCK's own sample count:
    # n = max(1, ceil(2 * max |p_v - p_r|)) over that block's walked voxels, as numpy sets it
    act = code == 0
    seg = (pv - pr).to(torch.float32)
    ln = torch.sqrt(seg[0] * seg[0] + seg[1] * seg[1] + seg[2] * seg[2])
    del seg
    lmax = torch.zeros(B, dtype=torch.float32, device=sel.device)
    lmax.scatter_reduce_(0, b, torch.where(act, ln, 0.0), "amax")
    nblk = torch.clamp(torch.ceil(2.0 * lmax.to(torch.float64)), min=1).to(torch.int32)
    hit = E.walk_crossings(pr, pv, boff, act, nblk[b], walk_n, br, bv)
    return code.masked_fill(act & hit, _REASON["crossing"])


_PAIR_K = None       # the pair-check kernels, False once they are known to be unusable
_FTAB = {}           # (nt, device) -> the walk's float32(i / n) table


def _pair_kernels():
    """The Triton kernels of `_pair_checks_triton` (reciprocal, normals, walk), or False."""
    global _PAIR_K
    if _PAIR_K is None:
        try:
            import triton
            import triton.language as tl
            from triton.language.extra import libdevice as ld
            from rvsm import edt as E

            @triton.jit
            def _rav(boff, p0, p1, p2, Y, X):
                return boff + (p0.to(tl.int64) * Y + p1) * X + p2

            ek = E._triton_kernels()
            if not ek:
                raise RuntimeError("no rvsm.edt kernels")
            _uf_find = ek[7]

            @triton.jit
            def _rec(SEL, IXR, IXV, PAR, PR, PV, OKR, BOX, M, B, BV, V, Y, X, BIG,
                     BLOCK: tl.constexpr):
                # rule 4a per selected voxel, and each block's extent of the face points whose
                # normals rule 4b will read (BOX[0, b] recto, BOX[1, b] verso: min, -max per axis)
                offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                ok = offs < M
                s = tl.load(SEL + offs, mask=ok, other=0)
                b = s // V
                boff = b * V
                r0 = tl.load(IXR + s, mask=ok, other=0)
                r1 = tl.load(IXR + BV + s, mask=ok, other=0)
                r2 = tl.load(IXR + 2 * BV + s, mask=ok, other=0)
                v0 = tl.load(IXV + s, mask=ok, other=0)
                v1 = tl.load(IXV + BV + s, mask=ok, other=0)
                v2 = tl.load(IXV + 2 * BV + s, mask=ok, other=0)
                fr = _rav(boff, r0, r1, r2, Y, X)
                fv = _rav(boff, v0, v1, v2, Y, X)
                q0 = tl.load(IXR + fv, mask=ok, other=0)          # p_v's nearest recto point
                q1 = tl.load(IXR + BV + fv, mask=ok, other=0)
                q2 = tl.load(IXR + 2 * BV + fv, mask=ok, other=0)
                w0 = tl.load(IXV + fr, mask=ok, other=0)          # p_r's nearest verso point
                w1 = tl.load(IXV + BV + fr, mask=ok, other=0)
                w2 = tl.load(IXV + 2 * BV + fr, mask=ok, other=0)
                # one band component <=> one root in the band forests (recto volumes first, then verso);
                # every point here is a face voxel, i.e. in its band
                er = _uf_find(PAR, _rav(boff, q0, q1, q2, Y, X).to(tl.int32), ok) == \
                    _uf_find(PAR, fr.to(tl.int32), ok)
                ev = _uf_find(PAR, (BV + _rav(boff, w0, w1, w2, Y, X)).to(tl.int32), ok) == \
                    _uf_find(PAR, (BV + fv).to(tl.int32), ok)
                # within sqrt(3) (+1e-6) of each other <=> squared integer distance <= 3
                dr2 = (q0 - r0) * (q0 - r0) + (q1 - r1) * (q1 - r1) + (q2 - r2) * (q2 - r2)
                dv2 = (w0 - v0) * (w0 - v0) + (w1 - v1) * (w1 - v1) + (w2 - v2) * (w2 - v2)
                okr = ok & (er | (dr2 <= 3)) & (ev | (dv2 <= 3))
                tl.store(PR + offs, r0, mask=ok)
                tl.store(PR + M + offs, r1, mask=ok)
                tl.store(PR + 2 * M + offs, r2, mask=ok)
                tl.store(PV + offs, v0, mask=ok)
                tl.store(PV + M + offs, v1, mask=ok)
                tl.store(PV + 2 * M + offs, v2, mask=ok)
                tl.store(OKR + offs, okr.to(tl.int8), mask=ok)
                # `sel` is sorted, so a program's voxels are the first block's but at a block boundary:
                # one reduced atomic per value for that block, per-voxel atomics for the others
                b0 = tl.min(tl.where(ok, b, 1 << 40))
                m0 = okr & (b == b0)
                mo = okr & (b != b0)
                rb = BOX + b0 * 6
                vb = BOX + (B + b0) * 6
                tl.atomic_min(rb, tl.min(tl.where(m0, r0, BIG)))
                tl.atomic_min(rb + 1, tl.min(tl.where(m0, -r0, BIG)))
                tl.atomic_min(rb + 2, tl.min(tl.where(m0, r1, BIG)))
                tl.atomic_min(rb + 3, tl.min(tl.where(m0, -r1, BIG)))
                tl.atomic_min(rb + 4, tl.min(tl.where(m0, r2, BIG)))
                tl.atomic_min(rb + 5, tl.min(tl.where(m0, -r2, BIG)))
                tl.atomic_min(vb, tl.min(tl.where(m0, v0, BIG)))
                tl.atomic_min(vb + 1, tl.min(tl.where(m0, -v0, BIG)))
                tl.atomic_min(vb + 2, tl.min(tl.where(m0, v1, BIG)))
                tl.atomic_min(vb + 3, tl.min(tl.where(m0, -v1, BIG)))
                tl.atomic_min(vb + 4, tl.min(tl.where(m0, v2, BIG)))
                tl.atomic_min(vb + 5, tl.min(tl.where(m0, -v2, BIG)))
                rb = BOX + b * 6
                vb = BOX + (B + b) * 6
                tl.atomic_min(rb, r0, mask=mo)
                tl.atomic_min(rb + 1, -r0, mask=mo)
                tl.atomic_min(rb + 2, r1, mask=mo)
                tl.atomic_min(rb + 3, -r1, mask=mo)
                tl.atomic_min(rb + 4, r2, mask=mo)
                tl.atomic_min(rb + 5, -r2, mask=mo)
                tl.atomic_min(vb, v0, mask=mo)
                tl.atomic_min(vb + 1, -v0, mask=mo)
                tl.atomic_min(vb + 2, v1, mask=mo)
                tl.atomic_min(vb + 3, -v1, mask=mo)
                tl.atomic_min(vb + 4, v2, mask=mo)
                tl.atomic_min(vb + 5, -v2, mask=mo)

            @triton.jit
            def _grad(Gb, p0, p1, p2, Z, Y, X, m):
                # `_grad_at`: central differences, one-sided at the border, (f(hi) - f(lo)) / (hi - lo)
                hi = tl.minimum(p0 + 1, Z - 1)
                lo = tl.maximum(p0 - 1, 0)
                g0 = ld.div_rn(ld.sub_rn(tl.load(Gb + _rav(0, hi, p1, p2, Y, X), mask=m, other=0.0),
                                         tl.load(Gb + _rav(0, lo, p1, p2, Y, X), mask=m, other=0.0)),
                               tl.maximum(hi - lo, 1).to(tl.float32))
                hi = tl.minimum(p1 + 1, Y - 1)
                lo = tl.maximum(p1 - 1, 0)
                g1 = ld.div_rn(ld.sub_rn(tl.load(Gb + _rav(0, p0, hi, p2, Y, X), mask=m, other=0.0),
                                         tl.load(Gb + _rav(0, p0, lo, p2, Y, X), mask=m, other=0.0)),
                               tl.maximum(hi - lo, 1).to(tl.float32))
                hi = tl.minimum(p2 + 1, X - 1)
                lo = tl.maximum(p2 - 1, 0)
                g2 = ld.div_rn(ld.sub_rn(tl.load(Gb + _rav(0, p0, p1, hi, Y, X), mask=m, other=0.0),
                                         tl.load(Gb + _rav(0, p0, p1, lo, Y, X), mask=m, other=0.0)),
                               tl.maximum(hi - lo, 1).to(tl.float32))
                return g0, g1, g2

            @triton.jit
            def _norm3(a, b, c):
                return ld.sqrt_rn(ld.add_rn(ld.add_rn(ld.mul_rn(a, a), ld.mul_rn(b, b)), ld.mul_rn(c, c)))

            @triton.jit
            def _normal(SEL, PR, PV, OKR, G, DY, DX, SDB, SDZ, SDY, SDX, SXB, SXZ, SXY, SXX, REASON, ACT,
                        LMAX, M, BV, V, Z, Y, X, NMIN, NRAD, NAGR, EPS, C_REC: tl.constexpr,
                        C_NORM: tl.constexpr, BLOCK: tl.constexpr):
                # rule 4b per selected voxel (`_pair_checks_torch`'s float32 steps, each rounded), the
                # reason code for 4a / 4b written into REASON, and each block's longest walked segment
                offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                ok = offs < M
                s = tl.load(SEL + offs, mask=ok, other=0)
                b = s // V
                loc = s - b * V
                z = loc // (Y * X)
                y = (loc // X) % Y
                x = loc % X
                okr = ok & (tl.load(OKR + offs, mask=ok, other=0) != 0)
                ry = tl.load(DY + b * SDB + z * SDZ + y * SDY + x * SDX, mask=ok, other=0.0)
                rx = tl.load(DX + b * SXB + z * SXZ + y * SXY + x * SXX, mask=ok, other=0.0)
                rn = tl.maximum(ld.sqrt_rn(ld.add_rn(ld.mul_rn(ry, ry), ld.mul_rn(rx, rx))), EPS)
                ry = ld.div_rn(ry, rn)
                rx = ld.div_rn(rx, rn)
                r0 = tl.load(PR + offs, mask=ok, other=0)
                r1 = tl.load(PR + M + offs, mask=ok, other=0)
                r2 = tl.load(PR + 2 * M + offs, mask=ok, other=0)
                v0 = tl.load(PV + offs, mask=ok, other=0)
                v1 = tl.load(PV + M + offs, mask=ok, other=0)
                v2 = tl.load(PV + 2 * M + offs, mask=ok, other=0)
                a0, a1, a2 = _grad(G + b * V, r0, r1, r2, Z, Y, X, okr)
                c0, c1, c2 = _grad(G + BV + b * V, v0, v1, v2, Z, Y, X, okr)
                mr = _norm3(a0, a1, a2)
                mv = _norm3(c0, c1, c2)
                dr = tl.maximum(mr, EPS)
                dv = tl.maximum(mv, EPS)
                a0 = ld.div_rn(a0, dr)
                a1 = ld.div_rn(a1, dr)
                a2 = ld.div_rn(a2, dr)
                c0 = ld.div_rn(c0, dv)
                c1 = ld.div_rn(c1, dv)
                c2 = ld.div_rn(c2, dv)
                agree = ld.add_rn(ld.add_rn(ld.mul_rn(a0, c0), ld.mul_rn(a1, c1)), ld.mul_rn(a2, c2))
                okn = (mr >= NMIN) & (mv >= NMIN) & \
                    (ld.add_rn(ld.mul_rn(a1, ry), ld.mul_rn(a2, rx)) >= NRAD) & \
                    (ld.add_rn(ld.mul_rn(c1, ry), ld.mul_rn(c2, rx)) >= NRAD) & (agree >= NAGR)
                code = tl.where(okr, tl.where(okn, 0, C_NORM), C_REC)
                tl.store(REASON + s, code.to(tl.uint8), mask=ok & (code != 0))
                act = ok & (code == 0)
                tl.store(ACT + offs, act.to(tl.int8), mask=ok)
                ln = _norm3((v0 - r0).to(tl.float32), (v1 - r1).to(tl.float32), (v2 - r2).to(tl.float32))
                b0 = tl.min(tl.where(ok, b, 1 << 40))
                tl.atomic_max(LMAX + b0, tl.max(tl.where(act & (b == b0), ln, 0.0)))
                tl.atomic_max(LMAX + b, ln, mask=act & (b != b0))

            @triton.jit
            def _walk(SEL, PR, PV, ACT, LMAX, FT, FTS, NT, BR, BVB, REASON, M, V, Y, X,
                      C_CROSS: tl.constexpr, BLOCK: tl.constexpr):
                # rule 4c: `edt.walk_crossings`' walk, each segment with its block's sample count
                # n = max(1, ceil(2 * longest segment)), the crossings written into REASON
                offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                ok = offs < M
                act = (tl.load(ACT + offs, mask=ok, other=0) != 0) & ok
                s = tl.load(SEL + offs, mask=ok, other=0)
                b = s // V
                boff = b * V
                lm = tl.load(LMAX + b, mask=act, other=0.0)
                n = tl.maximum(tl.ceil(lm * 2.0), 1.0).to(tl.int32)
                n = tl.minimum(tl.maximum(n, 1), NT)
                az = tl.load(PR + offs, mask=ok, other=0).to(tl.float32)
                ay = tl.load(PR + M + offs, mask=ok, other=0).to(tl.float32)
                ax = tl.load(PR + 2 * M + offs, mask=ok, other=0).to(tl.float32)
                sz = tl.load(PV + offs, mask=ok, other=0).to(tl.float32) - az
                sy = tl.load(PV + M + offs, mask=ok, other=0).to(tl.float32) - ay
                sx = tl.load(PV + 2 * M + offs, mask=ok, other=0).to(tl.float32) - ax
                nmax = tl.max(tl.where(act, n, 0))
                left = offs < 0
                inv = offs < 0
                hit = offs < 0
                for i in range(0, nmax + 1):
                    a_ = act & (i <= n)
                    f = tl.load(FT + n.to(tl.int64) * FTS + i, mask=a_, other=0.0)
                    qz = ld.rint(ld.add_rn(az, ld.mul_rn(sz, f))).to(tl.int64)
                    qy = ld.rint(ld.add_rn(ay, ld.mul_rn(sy, f))).to(tl.int64)
                    qx = ld.rint(ld.add_rn(ax, ld.mul_rn(sx, f))).to(tl.int64)
                    qi = boff + (qz * Y + qy) * X + qx
                    rr = tl.load(BR + qi, mask=a_, other=0) != 0
                    vv = tl.load(BVB + qi, mask=a_, other=0) != 0
                    hit = hit | (a_ & ((left & rr & ~vv) | (inv & ~vv)))
                    left = left | (a_ & ~rr)
                    inv = inv | (a_ & vv)
                tl.store(REASON + s, tl.full([BLOCK], C_CROSS, tl.uint8), mask=act & hit)

            _PAIR_K = (_rec, _normal, _walk)
        except Exception:  # noqa: BLE001
            _PAIR_K = False
    return _PAIR_K


def _ftab(nt, dev):
    """The walk's float32(i / n) table [n, i], as `edt.walk_crossings` builds it, cached per device."""
    import torch
    key = (int(nt), str(dev))
    if key not in _FTAB:
        ni = torch.arange(int(nt) + 1, device=dev, dtype=torch.float64)
        _FTAB[key] = (ni[None, :] / torch.clamp(ni, min=1)[:, None]).to(torch.float32).contiguous()
    return _FTAB[key]


# the pair checks as three fused kernels on CUDA (`_pair_checks_triton`); RVSM_FIELDS_FUSED=0 runs the
# op-by-op torch version there too
PAIR_FUSED = os.environ.get("RVSM_FIELDS_FUSED", "1") not in ("0", "false", "no")


def _pair_checks_triton(sel, shape, br, bv, dr, dv, ixr, ixv, dy, dx, walk_n, reason):
    """`_pair_checks_torch` on CUDA in a few launches, the codes written straight into `reason` (whose
    `sel` entries are 0). True when done, False when the kernels are unusable (nothing written).

    (a) one kernel per voxel: the nearest points, their reciprocal points, the same-component test (the
        roots of the four face points in both bands' union-find forests, `edt.label_forest`: only the
        points compared are resolved, never a whole labelling), the within-sqrt(3) test (integers only),
        and each block's extent of the face points that passed (atomic min / max);
    (b) the Gaussian-smoothed distances only over those extents plus the one-voxel gradient stencil
        (`edt.gaussian3_box`: the same bits as the whole-volume filter there), then one kernel per voxel
        for the gradients, norms, radial and agreement tests in `_pair_checks_torch`'s float32 order,
        every product, sum, quotient and root rounded on its own (no contraction), and each block's
        longest walked segment (atomic max; a maximum does not depend on the order);
    (c) the walk, one kernel, each voxel with its block's sample count.
    The reason codes are the torch version's bit for bit (tested)."""
    import torch
    from rvsm import edt as E
    k = _pair_kernels()
    if not k or not PAIR_FUSED:
        return False
    B, Z, Y, X = (int(v) for v in shape)
    V = Z * Y * X
    BV = B * V
    dev = sel.device
    M = int(sel.numel())
    if 2 * BV >= 2 ** 31:
        return False
    # both bands' union-find forests in one batch; the reciprocal kernel finds the roots it compares
    par = E.label_forest(torch.stack((br, bv)).view(2 * B, Z, Y, X))
    if par is None:
        return False
    pr = torch.empty((3, M), dtype=torch.int32, device=dev)
    pv = torch.empty((3, M), dtype=torch.int32, device=dev)
    okr = torch.empty(M, dtype=torch.int8, device=dev)
    box = torch.full((2, B, 6), E.BOX_EMPTY, dtype=torch.int32, device=dev)
    BLOCK = 256
    grid = (-(-M // BLOCK),)
    ixr, ixv = ixr.contiguous(), ixv.contiguous()
    try:
        k[0][(-(-M // 32),)](sel, ixr, ixv, par, pr, pv, okr, box, M, B, BV, V, Y, X, E.BOX_EMPTY, BLOCK=32,
                             num_warps=1)     # a warp per program: its finds diverge
    except Exception as e:  # noqa: BLE001
        _pair_failed(e)
        return False
    del par
    dr, dv = dr.contiguous(), dv.contiguous()
    g = E.gaussian3_box([dr, dv], box, NORMAL_SIGMA)
    if g is None:
        return False
    dye, dxe = dy.expand(B, Z, Y, X), dx.expand(B, Z, Y, X)
    act = torch.empty(M, dtype=torch.int8, device=dev)
    lmax = torch.zeros(B, dtype=torch.float32, device=dev)
    f32 = lambda v: float(np.float32(v))           # noqa: E731
    try:
        k[1][grid](sel, pr, pv, okr, g, dye, dxe, *dye.stride(), *dxe.stride(), reason, act, lmax, M, BV,
                   V, Z, Y, X, f32(NORMAL_MIN), f32(NORMAL_RADIAL), f32(NORMAL_AGREE), f32(1e-6),
                   C_REC=_REASON["reciprocal"], C_NORM=_REASON["normal"], BLOCK=BLOCK)
        del g, okr
        ft = _ftab(walk_n, dev)
        k[2][grid](sel, pr, pv, act, lmax, ft, int(walk_n) + 1, int(walk_n), br.view(torch.uint8),
                   bv.view(torch.uint8), reason, M, V, Y, X, C_CROSS=_REASON["crossing"], BLOCK=BLOCK)
    except Exception as e:  # noqa: BLE001  -- the caller's torch version rewrites every `sel` code
        _pair_failed(e)
        return False
    return True


def _pair_failed(e):
    global _PAIR_K
    _PAIR_K = False
    import warnings
    warnings.warn(f"rvsm.targets: pair-check kernels unusable ({e!r}); using the torch version")


NCOUNT = len(SUPPORT)      # the per-block support row: voxels, the nine reasons, valid

# the device transforms are capped (`edt.edt2(cap=)`) at what the rules can read; RVSM_FIELDS_CAP=0
# computes them over whole lines
EDT_CAP = os.environ.get("RVSM_FIELDS_CAP", "1") not in ("0", "false", "no")
MEDIAL_CAP = 16.0    # the medial transform's cap: deeper bands fall back to the uncapped transform


def edt_cap(reach):
    """How far the face distances of a block must be exact, for rung-scaled `reach`:

    - the no_recto / no_verso / coverage / thickness rules read u, d at voxels with u <= reach;
    - the reciprocal rule reads the nearest recto face of the verso face point p_v (and the other way
      round): |p_v - p_r| <= u_v + u_r <= 2 reach, so that face is within 2 reach of p_v;
    - the normals read the Gaussian-smoothed d (sigma NORMAL_SIGMA, truncate 4: a cube of half-width
      ceil(4 sigma) per axis) one voxel either side of a face point, i.e. voxels within
      (ceil(4 sigma) + 1) * sqrt(3) of it, whose |d| is at most that distance.
    Beyond these, a voxel's value only has to fail no_recto / no_verso, which +inf does. One voxel of
    slack on top."""
    g = (math.ceil(4.0 * NORMAL_SIGMA) + 1) * math.sqrt(3.0)
    return float(math.ceil(max(2.0 * float(reach), g)) + 1)


def _block_fields_t(recto, verso, dy, dx, thr, reach, tmin, tmax, core, cover, dev):
    """The tensor core of `block_fields_torch` for a batch: `recto` / `verso` (B,Z,Y,X) uint8 (verso
    may be None), `dy` / `dx` / `core` / `cover` broadcastable to it. Returns (midline, thickness,
    valid) tensors and a (B, len(SUPPORT)) int64 tensor of support counts in `SUPPORT` order, all on
    the device."""
    import torch
    from rvsm import edt as E
    rec = _tt(recto, dev)
    shape = tuple(int(v) for v in rec.shape)
    B = shape[0]
    dy, dx = _tt(dy, dev, torch.float32), _tt(dx, dev, torch.float32)
    core = torch.ones(shape, dtype=torch.bool, device=dev) if core is None else \
        _tt(core, dev, torch.bool).expand(shape)
    ev = E.binary_dilation(core)
    reason = torch.zeros(shape, dtype=torch.uint8, device=dev)

    def fail(bad, key):
        reason.masked_fill_((reason == 0) & ev & bad, _REASON[key])

    lvl = int(round(thr * 255))
    br = rec >= lvl
    del rec
    cap = edt_cap(reach) if EDT_CAP else None
    mcap = MEDIAL_CAP if EDT_CAP else None
    dr, ur, ixr = face_distance_torch(medial_torch(br, mcap), dy, dx, cap)
    fail(ur > reach, "no_recto")                 # also every voxel of a block without a recto face
    bv = torch.zeros_like(br) if verso is None else _tt(verso, dev) >= lvl
    dv, uv, ixv = face_distance_torch(medial_torch(bv, mcap), dy, dx, cap)
    fail(uv > reach, "no_verso")                 # ... and without a verso face
    if cover is not None:
        fail(_tt(cover, dev, torch.float32) <= torch.maximum(ur, uv) + COVER_MARGIN, "coverage")
    del ur, uv
    t = dv - dr
    fail((t < tmin) | (t > tmax), "thickness")
    sel = torch.nonzero(((reason == 0) & ev).reshape(-1)).squeeze(1)       # one sync per batch
    if sel.numel():
        wn = int(np.ceil(4.0 * float(reach))) + 2
        if not (dev.type == "cuda" and
                _pair_checks_triton(sel, shape, br, bv, dr, dv, ixr, ixv, dy, dx, wn, reason.view(-1))):
            reason.view(-1)[sel] = _pair_checks_torch(sel, shape, br, bv, dr, dv, ixr, ixv, dy, dx, wn)
    del ixr, ixv, sel, br, bv
    pair = (reason == 0) & ev
    zero = torch.zeros((), dtype=torch.float32, device=dev)
    m = torch.where(pair, 0.5 * (dr + dv), zero)
    del dr, dv
    full = pair.clone()
    for a in (1, 2, 3):
        for s in (1, -1):
            full &= E._shift(pair, a, s, False)
    fail(pair & ~full, "stencil")
    g2 = torch.zeros(shape, dtype=torch.float32, device=dev)
    for a in (1, 2, 3):
        g = torch.zeros(shape, dtype=torch.float32, device=dev)
        n = shape[a]
        g.narrow(a, 1, n - 2).copy_(0.5 * (m.narrow(a, 2, n - 2) - m.narrow(a, 0, n - 2)))
        g2 += g * g
    gn = torch.sqrt(g2)
    del g, g2
    fail(full & ((gn < GRAD[0]) | (gn > GRAD[1])), "gradient")
    del gn, full, pair
    valid = (reason == 0) & core
    bidx = torch.arange(B, device=dev, dtype=torch.int64).view(B, 1, 1, 1)
    nr = NCOUNT - 1                                # reason codes 0..9
    cnt = torch.zeros(B * nr + 1, dtype=torch.int64, device=dev)      # a scatter, not a (syncing) bincount
    cnt.scatter_add_(0, torch.where(core, bidx * nr + reason.long(), B * nr).reshape(-1),
                     torch.ones((), dtype=torch.int64, device=dev).expand(core.numel()))
    cnt = cnt[:-1].view(B, nr)
    counts = torch.cat((core.reshape(B, -1).sum(1, keepdim=True), cnt[:, 1:],
                        valid.reshape(B, -1).sum(1, keepdim=True)), 1)
    return torch.where(valid, m, zero), torch.where(valid, t, zero), valid, counts


def _support(row):
    """A `SUPPORT` dict from one row of `_block_fields_t`'s counts."""
    return {k: int(v) for k, v in zip(SUPPORT, row)}


def block_fields_torch(recto, verso, dy, dx, thr=0.5, reach=REACH, tmin=TMIN, tmax=TMAX, core=None,
                       cover=None, device=None):
    """`block_fields` on a torch device (default: CUDA when available, else the CPU): the same inputs
    (numpy arrays or tensors, one block), the same rules, the same `(midline f32, thickness f32, valid
    bool, support)` as numpy arrays. Nearest-voxel ties could make a handful of voxels differ
    (`rvsm.edt`)."""
    import torch
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    def one(a, dt=None):
        return None if a is None else _tt(a, dev, dt)[None]
    with torch.no_grad():
        m, t, ok, cnt = _block_fields_t(one(recto), one(verso), one(dy, torch.float32),
                                        one(dx, torch.float32), thr, reach, tmin, tmax, one(core),
                                        one(cover), dev)
        return m[0].cpu().numpy(), t[0].cpu().numpy(), ok[0].cpu().numpy(), _support(cnt[0].tolist())


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


def _proc_start(pid):
    """The start time of `pid` (clock ticks since boot), or None when it is gone or a zombie."""
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            st = f.read()
        rest = st[st.rindex(")") + 2:].split()
        return None if rest[0] in ("Z", "X") else int(rest[19])
    except (OSError, ValueError, IndexError):
        return None


def watch_owner(pid, start, every=1.0):
    """A daemon thread that ends this process as soon as the process `pid` (started at `start`) is gone:
    the fields-pool worker's producer-death watch (review P3-09). The worker's own parent is the pool's
    forkserver, which can outlive the producer, so PDEATHSIG alone does not follow the producer."""
    import threading
    import time

    def run():
        while True:
            if _proc_start(pid) != start:
                os._exit(0)
            time.sleep(every)
    threading.Thread(target=run, name="rvsm-owner-watch", daemon=True).start()


def _nice(owner=None, owner_start=None):
    die_with_parent()
    if owner is not None:
        watch_owner(owner, owner_start)
    try:
        os.nice(10)       # the fields are background work: the trainer's loader keeps the cores it needs
    except OSError:
        pass


def field_pool(jobs, owner=None):
    """A process pool for `region_fields(pool=...)` that lives as long as the producer.

    `forkserver`, not `fork`: the producer that owns it has CUDA and several threads, and a forked child
    of a threaded process can inherit a lock some other thread held at the fork. The workers run at
    nice 10 and die with their parent (`die_with_parent`) -- and, given `owner` (the producer's pid),
    with the OWNER too (`watch_owner`), whatever becomes of the forkserver in between."""
    import concurrent.futures as cf
    import multiprocessing as mp
    args = (int(owner), _proc_start(owner)) if owner is not None else (None, None)
    return cf.ProcessPoolExecutor(max_workers=int(jobs), mp_context=mp.get_context("forkserver"),
                                  initializer=_nice, initargs=args)


# ------------------------------------------------------------------ the whole region, on one device

POOL_SLAB = 64       # fine z-planes per upload when a store is pooled on the device


def _pool_device(v, dev):
    """`ladder.pool2` of a whole host uint8 store (even shape), on `dev` in z-slabs, back on the host."""
    from rvsm import edt as E
    import torch
    out = np.empty(tuple(int(s) // 2 for s in v.shape), np.uint8)
    for z in range(0, v.shape[0], POOL_SLAB):
        s = torch.from_numpy(np.ascontiguousarray(v[z:z + POOL_SLAB])).to(dev)
        out[z // 2:(z + POOL_SLAB) // 2] = E.pool2(s).cpu().numpy()
    return out


def _window(v, lo, shape):
    """The box at store-local corner `lo` of `shape` out of the whole (pooled) store `v`, air outside
    it: `read_pooled` of that store, without the per-block chunk decodes."""
    out = np.zeros(tuple(int(s) for s in shape), np.uint8)
    lo = np.asarray(lo, np.int64)
    S = np.asarray(v.shape, np.int64)
    a, b = np.maximum(lo, 0), np.minimum(lo + np.asarray(shape, np.int64), S)
    if (b > a).all():
        st, en = a - lo, b - lo
        out[st[0]:en[0], st[1]:en[1], st[2]:en[2]] = v[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    return out


def _host_inputs(axk, los, rsh, rlos, ext, halo):
    """The host side of a batch's axis offsets and coverage, in `axis_offsets` / `cover_distance`'s
    own float64 / float32 steps: (yy, xx, cy, cx) float64 and the three per-axis coverage vectors
    float32, each (B, n)."""
    Z, Y, X = (int(v) for v in rsh)
    g0 = np.asarray(los, np.int64) - int(halo)                    # (B, 3) block corners with halo
    z = np.arange(Z, dtype=np.float64)[None, :] + g0[:, :1]
    cy = np.stack([np.interp(zz, axk[0], axk[1]) for zz in z])
    cx = np.stack([np.interp(zz, axk[0], axk[2]) for zz in z])
    yy = np.arange(Y, dtype=np.float64)[None, :] + g0[:, 1:2]
    xx = np.arange(X, dtype=np.float64)[None, :] + g0[:, 2:3]
    cov = []
    for a in range(3):
        rows = []
        for rlo in rlos:
            c = np.arange(int(rsh[a]), dtype=np.float32) + float(rlo[a])
            rows.append(np.where((c >= 0) & (c < ext[a]), np.minimum(c + 1, float(ext[a]) - c), 0.0)
                        .astype(np.float32))
        cov.append(np.stack(rows))
    return yy, xx, cy, cx, cov


def _to_dev(a, dev):
    """A host array on `dev` without a synchronising copy: through pinned memory, non-blocking."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(a))
    if dev.type == "cuda":
        return t.pin_memory().to(dev, non_blocking=True)
    return t


def _block_inputs_torch(host, rsh, axis_r_vox, halo, dev):
    """(dy, dx, core, cover) of a batch on `dev` from `_host_inputs`: dy (B,Z,Y,1) and dx (B,Z,1,X)
    float32 (computed in float64, as `axis_offsets`), the core mask with the axis exclusion, and
    `cover_distance` (B,Z,Y,X)."""
    import torch
    yy, xx, cy, cx = (_to_dev(a, dev) for a in host[:4])
    cov = [_to_dev(c, dev) for c in host[4]]
    B = int(yy.shape[0])
    Z, Y, X = (int(v) for v in rsh)
    dy = yy[:, None, :, None] - cy[:, :, None, None]
    dx = xx[:, None, None, :] - cx[:, :, None, None]
    h = int(halo)
    n = [v - 2 * h for v in (Z, Y, X)]
    dyc, dxc = dy[:, h:h + n[0], h:h + n[1], :], dx[:, h:h + n[0], :, h:h + n[2]]
    r = torch.sqrt(dyc * dyc + dxc * dxc).to(torch.float32)
    core = torch.zeros((B, Z, Y, X), dtype=torch.bool, device=dev)
    core[:, h:h + n[0], h:h + n[1], h:h + n[2]] = r >= axis_r_vox
    del r, dyc, dxc
    cover = torch.minimum(torch.minimum(cov[0][:, :, None, None], cov[1][:, None, :, None]),
                          cov[2][:, None, None, :])
    return dy.to(torch.float32), dx.to(torch.float32), core, cover


def _encode_torch(m, t, ok, cap):
    """`encode_signed(m, ok, cap)` and `encode_unsigned(t, ok)` on the device, the same float32 steps
    (clip, divide, round half to even, offset, clip to 1..255): one (2, ...) uint8 tensor."""
    import torch
    z = torch.zeros((), dtype=torch.uint8, device=m.device)
    cm = torch.round(torch.clamp(m, -float(cap), float(cap)) / UNIT) + OFF
    mu = torch.where(ok, torch.clamp(cm, 1, 255).to(torch.uint8), z)
    ct = torch.round(torch.clamp(t, UNIT, 255 * UNIT) / UNIT)
    tu = torch.where(ok, torch.clamp(ct, 1, 255).to(torch.uint8), z)
    return torch.stack((mu, tu))


FIELD_BATCH = 3      # blocks per device batch: ~0.83 GB of VRAM each at 224^3, 2.5 GB at 3
# a device block whose window has no recto (or no verso) voxel at the threshold is answered without its
# transforms (`_faceless`): the same bytes and support counts. RVSM_FIELDS_SKIP=0 computes every block
SKIP_FACELESS = os.environ.get("RVSM_FIELDS_SKIP", "1") not in ("0", "false", "no")


def _batches(tasks, size):
    """Consecutive runs of at most `size` tasks with the same rung and core shape, in task order."""
    out, cur = [], []
    for t in tasks:
        if cur and (len(cur) >= size or t[0] != cur[0][0] or tuple(t[2]) != tuple(cur[0][2])):
            out.append(cur)
            cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out


class _SlabStore:
    """A region store decoded a 128-plane z-slab (one row of chunks) at a time, for `_fields_torch`.

    Rung 2 is served from the slabs a window touches, which are decoded on first use and dropped once
    every later window starts below them (`drop_below`: the blocks run z-major), so at most the three or
    four chunk rows a halo-48 window spans are in memory instead of the whole 1 GB store. Rungs 3 and
    4 are the whole store's 2x / 4x pool (128 / 16 MB), built slab by slab as the slabs are decoded
    (`pool2` of a slab of an even number of planes is that slab's share of the whole pool)."""

    def __init__(self, arr, ks, dev):
        self.arr, self.dev = arr, dev
        self.S = tuple(int(v) for v in arr.shape[-3:])
        self.slab = stores.CHUNK
        self.rows = {}                              # chunk row -> decoded (128, Y, X) uint8
        self.pooled = set()                         # chunk rows already folded into rung 3
        self.need3 = max(ks) >= 3
        self.r3 = np.zeros(tuple(v // 2 for v in self.S), np.uint8) if self.need3 else None
        self.r4 = None
        self.lock = __import__("threading").Lock()

    def _row(self, c):
        if c not in self.rows:
            z0 = c * self.slab
            v = np.asarray(self.arr[z0:min(z0 + self.slab, self.S[0])], np.uint8)
            self.rows[c] = v
            if self.need3 and c not in self.pooled:
                self.r3[z0 // 2:(z0 + v.shape[0]) // 2] = _pool_device(v, self.dev)
                self.pooled.add(c)
        return self.rows[c]

    def drop_below(self, c):
        for q in [q for q in self.rows if q < c]:
            del self.rows[q]

    def shape(self, k):
        return tuple(v >> (int(k) - 2) for v in self.S)

    def window(self, k, lo, shape):
        """`_window` of the rung-k store (the whole store's pool above rung 2), air outside it."""
        with self.lock:
            if int(k) == 2:
                out = np.zeros(tuple(int(v) for v in shape), np.uint8)
                lo = np.asarray(lo, np.int64)
                a = np.maximum(lo, 0)
                b = np.minimum(lo + np.asarray(shape, np.int64), np.asarray(self.S, np.int64))
                if (b > a).all():
                    for c in range(int(a[0]) // self.slab, (int(b[0]) - 1) // self.slab + 1):
                        v = self._row(c)
                        z0, z1 = max(int(a[0]), c * self.slab), min(int(b[0]), c * self.slab + v.shape[0])
                        out[z0 - lo[0]:z1 - lo[0], a[1] - lo[1]:b[1] - lo[1], a[2] - lo[2]:b[2] - lo[2]] = \
                            v[z0 - c * self.slab:z1 - c * self.slab, a[1]:b[1], a[2]:b[2]]
                return out
            for c in range(-(-self.S[0] // self.slab)):     # every slab pooled, then rung 3 / 4
                if c not in self.pooled:
                    self._row(c)
                    del self.rows[c]
            self.rows.clear()
            if int(k) == 3:
                return _window(self.r3, lo, shape)
            if self.r4 is None:
                self.r4 = _pool_device(self.r3, self.dev)
            return _window(self.r4, lo, shape)


def _fields_torch(init, tasks, device, take, rung_done=None, batch=None, gpu_lock=None):
    """`region_fields`' blocks on a torch `device`, in task order, each result handed to `take`.

    The region's recto / verso stores are decoded one chunk row at a time (`_SlabStore`: a block read
    with a 48 halo straight from zarr decodes up to 27 of the 128^3 chunks for one 224^3 window; this
    decodes each chunk once and holds at most a few rows), pooled to rungs 3 / 4 on the device, and
    every block's window is cut out of them with air outside -- `read_pooled`'s bytes. The blocks go
    through the device `batch` (`FIELD_BATCH`) at a time: the windows and the per-block axis / coverage
    vectors are prepared on a helper thread and copied without synchronising, everything `_block`
    computes on the host (axis offsets, core, coverage, the fields, the encoding) is computed on the
    device, and each batch is read back once (its encoded cores and support rows). `rung_done(k)` is
    called after the last block of rung k has been handed to `take`. Host memory: a few chunk rows and
    the pooled rungs of each store, the batch's windows and its pinned staging, released at the end."""
    import contextlib

    import torch
    rp, vp, ax, thr, cap, tmin, tmax, reach, axis_r_um, halo = init
    dev = torch.device(device)
    batch = int(batch or FIELD_BATCH)
    rec_a = stores.open_store(rp)
    ver_a = stores.open_store(vp) if vp else None
    origin2 = np.asarray(rec_a.attrs["origin_zyx"], np.int64)
    ks = sorted({int(t[0]) for t in tasks})
    src = {"recto": _SlabStore(rec_a, ks, dev), "verso": None if ver_a is None else _SlabStore(ver_a, ks, dev)}
    groups = _batches(tasks, batch)
    lvl = int(round(thr * 255))
    stats = {"blocks": 0, "no_recto": 0, "no_verso": 0}      # blocks answered without their transforms
    axk = {k: AX.axis_at(np.asarray(ax, np.float64), k) for k in ks}
    stream = torch.cuda.Stream(dev) if dev.type == "cuda" else None
    ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()

    def cut(gi):
        group = groups[gi]
        k, n = int(group[0][0]), group[0][2]
        rsh = tuple(int(v) + 2 * int(halo) for v in n)
        rlos = [np.asarray(lo, np.int64) - (origin2 >> (k - 2)) - int(halo) for _, lo, _ in group]
        rec = np.stack([src["recto"].window(k, r, rsh) for r in rlos])
        ver = None if src["verso"] is None else np.stack([src["verso"].window(k, r, rsh) for r in rlos])
        if k == 2:                                  # rung-2 rows no later window reads are dropped
            later = [int(t[1][0]) - int(origin2[0]) - int(halo) for g in groups[gi + 1:] for t in g
                     if int(t[0]) == 2]
            low = (max(min(later), 0) // stores.CHUNK) if later else 1 << 30
            for sv in src.values():
                if sv is not None:
                    sv.drop_below(low)
        ext = list(src["recto"].shape(k))
        # which windows have a face at all (on this helper thread, off the device's critical path)
        hr = (rec >= lvl).reshape(len(group), -1).any(1)
        hv = np.zeros(len(group), bool) if ver is None else (ver >= lvl).reshape(len(group), -1).any(1)
        return rsh, rec, ver, _host_inputs(axk[k], [lo for _, lo, _ in group], rsh, rlos, ext, halo), \
            (hr, hv)

    import concurrent.futures as cf
    if gpu_lock is None:
        gpu_lock = contextlib.nullcontext()
    # a lock that can make way (`run.GpuGate.fields_hold`) is offered the card back between batches:
    # nothing of the fields is left on the device there, so it only waits for its own stream first
    yield_point = getattr(gpu_lock, "yield_point", None)

    def flush():
        if stream is not None:
            stream.synchronize()
            torch.cuda.empty_cache()
    with gpu_lock, torch.no_grad(), ctx, cf.ThreadPoolExecutor(1, thread_name_prefix="rvsm-fcut") as cutter:
        nxt = cutter.submit(cut, 0) if groups else None
        for gi, group in enumerate(groups):
            if gi and yield_point is not None:
                yield_point(flush)
            rsh, rec, ver, host, (hr, hv) = nxt.result()
            nxt = cutter.submit(cut, gi + 1) if gi + 1 < len(groups) else None
            k, n = int(group[0][0]), group[0][2]
            dy, dx, core, cover = _block_inputs_torch(host, rsh, float(axis_r_um) / ladder.rung_um(k),
                                                      halo, dev)
            rk, tn, tx = rung_params(k, reach, tmin, tmax)
            sl = (slice(None),) + tuple(slice(int(halo), int(halo) + int(v)) for v in n)
            stats["blocks"] += len(group)
            full = hr & hv
            if full.all() or not SKIP_FACELESS:
                m, t, ok, cnt = _block_fields_t(_to_dev(rec, dev), None if ver is None else _to_dev(ver, dev),
                                                dy, dx, thr, rk, tn, tx, core, cover, dev)
                enc = _encode_torch(m[sl], t[sl], ok[sl], cap)
                del m, t, ok
            else:
                enc, cnt = _faceless(rec, ver, hr, hv, dy, dx, core, cover, thr, rk, tn, tx, cap, sl, n,
                                     dev)
                stats["no_recto"] += int((~hr).sum())
                stats["no_verso"] += int((hr & ~hv).sum())
            del dy, dx, core, cover
            if stream is not None:                 # one read-back per batch
                he = torch.empty(enc.shape, dtype=torch.uint8, pin_memory=True)
                hc = torch.empty(cnt.shape, dtype=torch.int64, pin_memory=True)
                he.copy_(enc, non_blocking=True)
                hc.copy_(cnt, non_blocking=True)
                stream.synchronize()
                enc_h, cnt_h = he.numpy(), hc.numpy()
            else:
                enc_h, cnt_h = enc.numpy(), cnt.numpy()
            del enc, cnt
            for j, (_, lo, _) in enumerate(group):
                take((k, tuple(int(v) for v in lo), enc_h[0, j].copy(), enc_h[1, j].copy(),
                      _support(cnt_h[j].tolist())))
            if rung_done is not None and (gi + 1 == len(groups) or int(groups[gi + 1][0][0]) != k):
                rung_done(k)
        del src
        stats["skipped"] = stats["no_recto"] + stats["no_verso"]
        if stream is not None:              # still under the lock: the next pass finds the card clean
            stream.synchronize()
            torch.cuda.empty_cache()        # hand the fields' blocks back to the process's other users
            try:
                torch._C._host_emptyCache()  # and the pinned staging buffers back to the host
            except Exception:  # noqa: BLE001
                pass
    return stats


def _faceless(rec, ver, hr, hv, dy, dx, core, cover, thr, rk, tn, tx, cap, sl, n, dev):
    """A device batch in which some windows have no face: (enc (2,B,*n) uint8, counts (B, NCOUNT)) as
    `_block_fields_t` + `_encode_torch` would give them, computing only what those blocks' answer
    depends on.

    `hr` / `hv`: does each window have a recto / verso voxel at the threshold. Without a recto voxel
    there is no recto face, `ur` is +inf everywhere, and every core voxel fails "no_recto" first:
    nothing is valid, the codes are 0, the counts are (voxels, no_recto = voxels, 0, ...). With a recto
    face but no verso voxel, `uv` is +inf: a core voxel fails "no_recto" where `ur > reach` and
    "no_verso" everywhere else, so only the recto face distance is computed (the numpy path's own early
    return is the same rule, `block_fields`). The blocks with both run through `_block_fields_t` as a
    smaller batch -- a batch's blocks are computed independently, so its size never changes a byte."""
    import torch
    B = len(hr)
    enc = torch.zeros((2, B) + tuple(int(v) for v in n), dtype=torch.uint8, device=dev)
    cnt = torch.zeros((B, NCOUNT), dtype=torch.int64, device=dev)
    nvox = core.reshape(B, -1).sum(1)
    full = np.flatnonzero(hr & hv)
    if full.size:
        ti = torch.as_tensor(full, device=dev)
        m, t, ok, c = _block_fields_t(_to_dev(rec[full], dev), _to_dev(ver[full], dev), dy[ti], dx[ti],
                                      thr, rk, tn, tx, core[ti], cover[ti], dev)
        enc[:, ti] = _encode_torch(m[sl], t[sl], ok[sl], cap)
        cnt[ti] = c
        del m, t, ok, c
    ro = np.flatnonzero(hr & ~hv)
    if ro.size:
        ti = torch.as_tensor(ro, device=dev)
        _, ur, _ = face_distance_torch(medial_torch(_to_dev(rec[ro], dev) >= int(round(thr * 255)),
                                                    MEDIAL_CAP if EDT_CAP else None),
                                       dy[ti], dx[ti], edt_cap(rk) if EDT_CAP else None)
        nr = (core[ti] & (ur > rk)).reshape(len(ro), -1).sum(1)
        del ur
        cnt[ti, 0] = nvox[ti]
        cnt[ti, 1] = nr
        cnt[ti, 2] = nvox[ti] - nr
    em = np.flatnonzero(~hr)
    if em.size:
        ti = torch.as_tensor(em, device=dev)
        cnt[ti, 0] = nvox[ti]
        cnt[ti, 1] = nvox[ti]
    return enc, cnt


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


def source_verso(root, lo, round_=0):
    """(generation, path) of the verso the WRITER builds fields from: the newest FINISHED generation
    (a regeneration's new verso), which readers see only once its fields are committed with it."""
    g = max(stores.store_gen(root, "verso", lo, round_), 0)
    return g, stores.gen_path(stores.store_path(root, "verso", lo, round_), g)


def source_recto(root, lo, round_=0):
    """(generation, path) of the recto the WRITER builds fields from: the newest FINISHED generation (a
    teacher regeneration's new recto, `run.recto_needs_regen`), which readers see only once the fields
    built from it are committed with it."""
    g = max(stores.store_gen(root, "recto", lo, round_), 0)
    return g, stores.gen_path(stores.store_path(root, "recto", lo, round_), g)


def field_gen(root, lo, round_=0):
    """The generation the fields of the region's newest sources are written at: max(verso gen, recto
    gen). Both regenerations draw from one per-region counter (`stores.next_gen`), so this is a fresh
    directory whenever either source is new."""
    return max(source_verso(root, lo, round_)[0], source_recto(root, lo, round_)[0])


def _want(root, lo, round_, rung, reach, tmin, tmax, thr):
    rk, tn, tx = rung_params(rung, reach, tmin, tmax)
    return {"target_def": TARGET_DEF, "reach_vox": rk, "tmin_vox": tn, "tmax_vox": tx, "thr": float(thr),
            "recto_digest": source_digest(source_recto(root, lo, round_)[1]),
            "verso_digest": source_digest(source_verso(root, lo, round_)[1])}


def field_path(root, kind, k, lo, round_=0):
    """Where a field store of the region lives: at the GENERATION of its sources (`field_gen`), so a
    verso regenerated once (round 0, `run.verso_needs_regen`) or a recto from a new teacher set
    (`run.recto_needs_regen`) gets its fields in a new directory beside the old ones, never over them."""
    return stores.gen_path(stores.store_path(root, channel(kind, k), lo, round_), field_gen(root, lo, round_))


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
    return all(_current(field_path(root, kind, k, lo, round_),
                        _want(root, lo, round_, k, reach, tmin, tmax, thr))
               for k in rungs for kind in KINDS)


def region_fields(root, lo, ax, round_=0, jobs=1, rungs=(2, 3, 4), axis_r_um=AXIS_R_UM,
                  thr=0.5, cap=CAP, tmin=TMIN, tmax=TMAX, reach=REACH, block=BLOCK, halo=HALO,
                  force=False, pool=None, device=None, batch=None, gpu_lock=None):
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
    without one, `jobs > 1` forks a pool for this call. `device` (a torch device or its name) computes
    every block with `block_fields_torch` on that device, in this process and in block order, instead
    (`_fields_torch`), `batch` blocks at a time (default `FIELD_BATCH`), holding `gpu_lock` (a lock
    shared with whatever else uses the card in this process) for the device work -- and, when it has a
    `yield_point(flush)` (`run.GpuGate.fields_hold`), calling it between batches, where it may hand
    the card to someone else and take it back; `jobs` and `pool` are then unused.

    A pooled rung whose shape is not a multiple of 128 is padded up to one, because a store's shape must
    be; the padding is code 0, i.e. no data. For the production region (1024 at rung 2) rungs 3 and 4 are
    512 and 256 and nothing is padded."""
    assert 0 < float(reach) < int(halo), f"reach {reach} must be positive and below the halo {halo}"
    assert 0 <= float(tmin) <= float(tmax), (tmin, tmax)
    rp = source_recto(root, lo, round_)[1]            # the writer's sources: the newest finished recto
    vgen, vp = source_verso(root, lo, round_)         # ... and the newest finished verso
    rec = stores.open_store(rp)                       # raises unless the recto pass has finished
    if not stores.is_done(vp):
        vp = ""
    S2 = np.asarray(rec.shape[-3:], np.int64)
    origin2 = np.asarray(rec.attrs["origin_zyx"], np.int64)
    rep = {"region": [int(v) for v in lo], "round": int(round_), "verso": bool(vp), "rungs": {}}
    todo = []
    for k in sorted(int(q) for q in rungs):
        assert 2 <= k <= MAX_RUNG, f"no distance target at rung {k} (2..{MAX_RUNG} only)"
        paths = {kind: field_path(root, kind, k, lo, round_) for kind in KINDS}
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

    def write_rung(k, Sk, paths, want):
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

    if device is not None:
        # a rung's two stores are written (volcomp-encoded, on one thread, in rung order) as soon as its
        # last block is in, while the device computes the next rung's blocks: the same writes, earlier
        import concurrent.futures as cf
        byk = {k: (k, Sk, paths, want) for k, Sk, paths, want in todo}
        with cf.ThreadPoolExecutor(1, thread_name_prefix="rvsm-fwrite") as wr:
            futs = []
            rep["skipped_blocks"] = _fields_torch(
                init, tasks, device, take, rung_done=lambda k: futs.append(wr.submit(write_rung, *byk[k])),
                batch=batch, gpu_lock=gpu_lock)
            for f in futs:
                f.result()
        return rep
    elif pool is not None and len(tasks) >= 2:
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
        write_rung(k, Sk, paths, want)
    return rep


def _store_block(dst, k, blo, origin2, m, t):
    """Place one worker's core block into the rung-k output arrays (parent side, so the assembly order
    cannot change the bytes)."""
    loc = np.asarray(blo, np.int64) - (origin2 >> (int(k) - 2))
    sl = tuple(slice(int(a), int(a) + int(s)) for a, s in zip(loc, m.shape))
    dst["midline"][sl] = m
    dst["thickness"][sl] = t
