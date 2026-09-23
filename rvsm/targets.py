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
m = -t/2, and `infer`'s verso trick (`sign=-1` negates only the radial input channels, so a
recto-trained student marks the other face) is the same statement from the network's side. Along the
radial direction, a sheet with its recto face at a and its verso face at a - t gives, for a voxel at
radial coordinate x,

    d_r = x - a        d_v = x - (a - t)        so   m = (d_r + d_v) / 2   and   t = d_v - d_r > 0

outside the sheet on either side and inside it alike. A negative d_v - d_r therefore means the two
nearest faces are NOT the two faces of one sheet.

THE TARGET DEFINITION (`TARGET_DEF = "paired-v2"`; the fix of review findings T01-T04, see
docs/recipe.md "Distance-field targets"). Per block, from the recto and verso probability stores at the
store's rung, over core + `halo`:

1. The recto and verso FACES are the medial surfaces of the thresholded bands; `d_r`, `d_v` are RAW
   (unclipped) signed Euclidean distances to them. Clamping to +-31.75 happens only at encoding (T02:
   clipping first made both distances saturate to one sign far from a sheet, and the thickness collapsed
   to the old TMIN floor over most of a block).
2. REACH. A voxel's d_r is valid only if its nearest recto face voxel is within `reach` (default 24
   voxels of the store's rung; `reach < halo` is asserted, so the nearest face found in the halo'd box
   IS the nearest face anywhere); likewise d_v. A block with no recto face anywhere in core + halo is
   code 0 everywhere (T01: it used to be written as a valid zero distance). There is NO `midline = d_r`
   fallback: without a verso store, or with an empty verso band, midline and thickness are code 0
   (T04). A midline store always means the midline of a PAIRED sheet.
3. PAIRING (T03). The two faces must belong to the same sheet. The pair is valid iff the raw thickness
   t = d_v - d_r lies in [tmin, tmax] (defaults 2 and 24 RUNG-2 voxels, divided by 2^(rung-2) at
   rungs 3 and 4), AND the straight segment from the voxel's nearest recto point to its nearest verso
   point crosses no OTHER recto face. The segment is sampled at <= 1-voxel steps; a sample whose own
   signed recto distance is > `CROSS` (+1.5 voxels) lies OUTWARD of some recto face although the walk
   left the voxel's recto face inward, i.e. a second sheet's recto face sits between the two faces (a
   missing verso segment pairing one sheet's recto with the next sheet's verso). Negative or
   implausible thickness is REJECTED, never clamped up to tmin.
4. Over the valid paired voxels only: midline = (d_r + d_v) / 2, thickness = d_v - d_r, encoded as
   above. Everything else is code 0 (no data), and so is everything within `axis_r_um` microns of the
   umbilicus axis, where the core is crushed and "which side is recto" is close to a coin flip.

Each store's attrs record `target_def`, `reach_vox`, `tmin_vox`, `tmax_vox` and a `support` count of
why core voxels were rejected. A done field store written under another definition or other parameters,
or written before the region's verso store existed, is RECOMPUTED rather than reused.

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

COST AND DETERMINISM. Two scipy EDTs per block per rung plus the pairing walk over the candidate core
voxels, CPU only; `jobs > 1` spreads the blocks over worker processes. Every block is computed from the
stores alone, with a `halo` of context, and the parent assembles the cores, so the bytes written do not
depend on how the blocks were handed out: `jobs=4` is byte-identical to `jobs=1`.
"""
import os

import numpy as np

from rvsm import axis as AX, ladder, stores

UNIT = 0.25          # voxels per code step
OFF = 128            # the code of distance 0
CAP = 31.75          # +-CAP voxels is the representable range (codes 1..255); applied at ENCODING only
TMIN = 2.0           # RUNG-2 voxels: a thinner raw pair is rejected (never clamped); / 2^(rung-2) above
TMAX = 24.0          # RUNG-2 voxels: a thicker raw pair is not one sheet; / 2^(rung-2) above
REACH = 24.0         # voxels OF THE STORE'S RUNG: a face further than this from a voxel is not its face
CROSS = 1.5          # voxels: a pairing-walk sample this far OUTWARD of a recto face has crossed one
TARGET_DEF = "paired-v2"   # recorded in every field store; a done store under another one is recomputed
AXIS_R_UM = 400.0    # microns around the umbilicus axis that are dropped
MAX_RUNG = 4         # no distance target above this rung
BLOCK = 128          # the core a single EDT pair covers: exactly one store chunk
HALO = 48            # context voxels on every side; must exceed REACH (asserted)
KINDS = ("midline", "thickness")
SUPPORT = ("voxels", "no_recto", "no_verso", "thickness", "crossing", "valid")


def channel(kind, rung):
    """The store channel name of one field at one rung: `midline` at rung 2, `midline_r3` above it."""
    return str(kind) if int(rung) == 2 else f"{kind}_r{int(rung)}"


def thickness_bounds(rung, tmin=TMIN, tmax=TMAX):
    """(tmin, tmax) in RUNG-`rung` voxels, from the rung-2 values."""
    f = float(2 ** (int(rung) - 2))
    return float(tmin) / f, float(tmax) / f


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

    Where the displacement is exactly perpendicular to the radial direction the side is arbitrary; that
    is a measure-zero set on a sheet roughly perpendicular to the radius, and it is resolved to +."""
    from scipy import ndimage as ndi
    if not surf.any():
        return None
    u, ix = ndi.distance_transform_edt(~surf, return_indices=True)
    gy = (np.arange(surf.shape[1], dtype=np.int32)[None, :, None] - ix[1]).astype(np.float32)
    gx = (np.arange(surf.shape[2], dtype=np.int32)[None, None, :] - ix[2]).astype(np.float32)
    u = u.astype(np.float32)
    d = np.where(gy * dy + gx * dx < 0, -u, u).astype(np.float32)
    if max(surf.shape) < 2 ** 15:
        ix = ix.astype(np.int16)        # both blocks' indices are held at once by `block_fields`
    return d, u, ix


def _crossed(dr, ixr, ixv, sel):
    """For the flat voxel indices `sel`: does the straight segment from each one's nearest recto point to
    its nearest verso point pass OUTWARD of a recto face (signed recto distance > CROSS) anywhere
    strictly between its ends? Sampled at <= 1-voxel steps; every sample lies inside the box, because
    both ends are voxels of it."""
    out = np.zeros(sel.size, bool)
    if not sel.size:
        return out
    Y, X = dr.shape[1], dr.shape[2]
    pr = ixr.reshape(3, -1)[:, sel].astype(np.float32)
    seg = ixv.reshape(3, -1)[:, sel].astype(np.float32) - pr
    n = int(np.ceil(float(np.sqrt((seg * seg).sum(0)).max()))) + 1
    drf = dr.reshape(-1)
    for i in range(1, n):
        q = np.rint(pr + (i / n) * seg).astype(np.int64)
        out |= drf[(q[0] * Y + q[1]) * X + q[2]] > CROSS
    return out


def block_fields(recto, verso, dy, dx, thr=0.5, reach=REACH, tmin=TMIN, tmax=TMAX, mask=None):
    """(midline, thickness, valid, support) of one block, `tmin` / `tmax` / `reach` in THIS block's voxels.

    `recto` / `verso` are uint8 probability blocks (verso may be None); `mask` restricts the voxels that
    can be valid (the caller passes the core minus the axis exclusion; default: all). The faces are the
    medial surfaces of the bands at `thr`; the rules are the module docstring's 1-4. `midline` and
    `thickness` are raw float voxels and meaningful only where `valid`; `support` counts, over `mask`,
    the first rule each rejected voxel failed (`SUPPORT`)."""
    shape = recto.shape
    mask = np.ones(shape, bool) if mask is None else np.asarray(mask, bool)
    sup = dict.fromkeys(SUPPORT, 0)
    sup["voxels"] = int(mask.sum())
    zero = np.zeros(shape, np.float32)
    lvl = int(round(thr * 255))
    fr = face_distance(medial(recto >= lvl), dy, dx)
    if fr is None:
        sup["no_recto"] = sup["voxels"]
        return zero, zero, np.zeros(shape, bool), sup
    dr, ur, ixr = fr
    rok = mask & (ur <= reach)
    sup["no_recto"] = sup["voxels"] - int(rok.sum())
    fv = None if verso is None else face_distance(medial(verso >= lvl), dy, dx)
    if fv is None:
        sup["no_verso"] = int(rok.sum())
        return zero, zero, np.zeros(shape, bool), sup
    dv, uv, ixv = fv
    ok = rok & (uv <= reach)
    sup["no_verso"] = int(rok.sum()) - int(ok.sum())
    t = dv - dr
    cand = ok & (t >= tmin) & (t <= tmax)
    sup["thickness"] = int(ok.sum()) - int(cand.sum())
    sel = np.flatnonzero(cand)
    bad = _crossed(dr, ixr, ixv, sel)
    valid = cand.copy()
    valid.reshape(-1)[sel[bad]] = False
    sup["crossing"] = int(bad.sum())
    sup["valid"] = int(valid.sum())
    m = np.where(valid, 0.5 * (dr + dv), 0.0).astype(np.float32)
    return m, np.where(valid, t, 0.0).astype(np.float32), valid, sup


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
    every side so that a distance measured inside the core sees every face within `reach` of it."""
    k, lo, n = task
    halo = _CTX["halo"]
    rec_a, ver_a = _arr("recto"), _arr("verso")
    o = np.asarray(rec_a.attrs["origin_zyx"], np.int64) >> (int(k) - 2)   # the region corner at this rung
    loc = np.asarray(lo, np.int64) - o                                   # store-local core corner
    rlo = loc - halo
    rsh = tuple(int(v) + 2 * halo for v in n)
    sl = tuple(slice(halo, halo + int(v)) for v in n)
    dy, dx, r = axis_offsets(_axis_at(k), np.asarray(lo, np.int64) - halo, rsh)
    mask = np.zeros(rsh, bool)
    mask[sl] = r[sl] >= (_CTX["axis_r_um"] / ladder.rung_um(k))   # the crushed core carries no sign
    rec = read_pooled(rec_a, k, rlo, rsh)
    ver = None if ver_a is None else read_pooled(ver_a, k, rlo, rsh)
    tmin, tmax = thickness_bounds(k, _CTX["tmin"], _CTX["tmax"])
    m, t, ok, sup = block_fields(rec, ver, dy, dx, thr=_CTX["thr"], reach=_CTX["reach"], tmin=tmin,
                                 tmax=tmax, mask=mask)
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


def _nice():
    try:
        os.nice(10)       # the fields are background work: the trainer's loader keeps the cores it needs
    except OSError:
        pass


def field_pool(jobs):
    """A process pool for `region_fields(pool=...)` that lives as long as the producer.

    `forkserver`, not `fork`: the producer that owns it has CUDA and several threads, and a forked child
    of a threaded process can inherit a lock some other thread held at the fork. The workers run at
    nice 10."""
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


def _current(path, want):
    """A done field store that `region_fields` may reuse: written by `region_fields` under the same
    definition and parameters (`want`), or written by something else entirely (a student pass's own
    rung-2 field head, which records no `field`). Anything else -- an older definition, other
    parameters, or a store built before the region's verso existed -- is recomputed."""
    if not stores.is_done(path):
        return False
    a = stores.open_store(path).attrs
    if "field" not in a:
        return True
    return all(a.get(key) == v for key, v in want.items())


def _want(root, lo, round_, rung, tmin, tmax, reach):
    tk = thickness_bounds(rung, tmin, tmax)
    return {"target_def": TARGET_DEF, "reach_vox": float(reach), "tmin_vox": tk[0], "tmax_vox": tk[1],
            "verso": stores.is_done(stores.store_path(root, "verso", lo, round_))}


def fields_current(root, lo, round_=0, rungs=(2, 3, 4), tmin=TMIN, tmax=TMAX, reach=REACH):
    """True iff every field store of the region at `rungs` is done AND current (`_current`): exactly the
    test `region_fields` skips a rung on. A scheduler that asks "are this region's fields built?" should
    ask this rather than `stores.is_done`, or a store from an older target definition is never rebuilt."""
    return all(_current(stores.store_path(root, channel(kind, k), lo, round_),
                        _want(root, lo, round_, k, tmin, tmax, reach))
               for k in rungs for kind in KINDS)


def region_fields(root, lo, ax, round_=0, jobs=1, rungs=(2, 3, 4), axis_r_um=AXIS_R_UM,
                  thr=0.5, cap=CAP, tmin=TMIN, tmax=TMAX, reach=REACH, block=BLOCK, halo=HALO,
                  force=False, pool=None):
    """Build the `midline` and `thickness` stores of one region, at every rung in `rungs`.

    `root` is the run directory, `lo` the region corner in rung-2 voxels, `ax` the umbilicus control
    points in rung-2 voxels (`axis.load`), `round_` the self-distillation round whose stores to read and
    write. The region's `recto` store must be done; `verso` may be missing, in which case both fields
    are no-data (code 0) everywhere: there is no recto-only midline (module docstring, rule 2).

    `tmin` / `tmax` are in RUNG-2 voxels (scaled per rung by `thickness_bounds`); `reach` is in voxels of
    each store's own rung and must be below `halo`.

    Returns a report dict, with a per-rung `support` count of why core voxels were rejected. A rung whose
    two stores are already `done` under the current definition is skipped (`force` recomputes), which is
    what makes this resumable at region granularity. `pool` is a `field_pool` the caller keeps alive
    across regions (a producer); without one, `jobs > 1` forks a pool for this call.

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
        want = _want(root, lo, round_, k, tmin, tmax, reach)
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

    def take(res):
        k, blo, m, t, s = res
        _store_block(out[k], k, blo, origin2, m, t)
        for key in SUPPORT:
            sup[k][key] += int(s[key])

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
                                "support": sup[k], **want})
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
