"""The distance fields: signed midline distance and sheet thickness, derived from a region's own stores.

Once a region has a `recto` store (and, from the verso pass, a `verso` one), the two bands determine a
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
-30.5-voxel distance.

THE SIGN. Positive is the RECTO, radially OUTWARD side, pinned to `rvsm.axis.radial`: the same axis
interpolation builds the model's radial input channel, the sign of the target here and the sign of the
exported normal, so they cannot disagree about where the axis is.

UNITS ARE VOXELS OF THAT RUNG, so a distance field is NEVER pooled: a 2x mean pool of a distance field
is not the distance field of the pooled mask (it is a distance in the FINE rung's voxels, halved by
nothing). Rungs 3 and 4 are therefore RECOMPUTED, from the 2x / 4x mean pool of the probability stores.
Above rung 4 there is no distance target at all: higher up the "band" is a pooled FRACTION and its 0.5
level set is not a surface.

WHERE THERE IS NO DATA (code 0): outside the recto band's reach; where there is no verso band at all
(the midline then falls back to the recto face and the thickness is not measurable); and within
`axis_r_um` microns of the umbilicus axis, where the core is crushed and "which side is recto" is close
to a coin flip.

WHERE THE STORES GO. One store per rung, under the round's own directory:

    <root>/stores/round_<r>/midline/region_<z>_<y>_<x>.zarr        rung 2
    <root>/stores/round_<r>/midline_r3/region_<z>_<y>_<x>.zarr     rung 3
    <root>/stores/round_<r>/midline_r4/region_<z>_<y>_<x>.zarr     rung 4

and likewise `thickness`, `thickness_r3`, `thickness_r4`. Each store records its own `rung` and
`voxel_um` in its attrs, and `region_fields` never pools one into another -- the suffix is there so that
a reader cannot accidentally take a rung-3 field for a rung-2 one by opening the wrong directory.

COST AND DETERMINISM. Two scipy EDTs per block per rung, CPU only; `jobs > 1` spreads the blocks over
worker processes. Every block is computed from the stores alone, with a `halo` of context, and the
parent assembles the cores, so the bytes written do not depend on how the blocks were handed out:
`jobs=4` is byte-identical to `jobs=1`.
"""
import numpy as np

from rvsm import axis as AX, ladder, stores

UNIT = 0.25          # voxels per code step
OFF = 128            # the code of distance 0
CAP = 31.75          # +-CAP voxels is the representable range (codes 1..255)
TMIN = 3.0           # voxels: the floor on a stored sheet thickness (see `losses.pair_bands`)
AXIS_R_UM = 400.0    # microns around the umbilicus axis that are dropped
MAX_RUNG = 4         # no distance target above this rung
BLOCK = 128          # the core a single EDT pair covers: exactly one store chunk
HALO = 48            # context voxels on every side: more than the +-31.75 the encoding can represent
KINDS = ("midline", "thickness")


def channel(kind, rung):
    """The store channel name of one field at one rung: `midline` at rung 2, `midline_r3` above it."""
    return str(kind) if int(rung) == 2 else f"{kind}_r{int(rung)}"


# ------------------------------------------------------------------------------------ the encoding

def encode_signed(d, valid, cap=CAP):
    """float voxels -> uint8, with `valid` False becoming code 0 (no data)."""
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


def signed_to(surf, dy, dx, cap=CAP):
    """Signed distance (voxels) to the one-voxel surface `surf`, POSITIVE on the radially outward side.

    `distance_transform_edt(..., return_indices=True)` gives, per voxel, both the distance to the nearest
    surface voxel and WHICH voxel that is; the displacement from that voxel to this one, dotted with the
    (unnormalised) radial direction (dy, dx, z-component 0), is the side. The radial vector is the same
    one the model gets as an input channel, so "positive" means the same thing in the target, in the
    stem's radial channels and in the exported normal (`dot(n, radial) > 0`).

    Where the displacement is exactly perpendicular to the radial direction the side is arbitrary; that
    is a measure-zero set on a sheet roughly perpendicular to the radius, and it is resolved to +."""
    from scipy import ndimage as ndi
    if not surf.any():
        return np.zeros(surf.shape, np.float32), np.zeros(surf.shape, bool)
    u, ix = ndi.distance_transform_edt(~surf, return_indices=True)
    gy = (np.arange(surf.shape[1], dtype=np.int32)[None, :, None] - ix[1]).astype(np.float32)
    gx = (np.arange(surf.shape[2], dtype=np.int32)[None, None, :] - ix[2]).astype(np.float32)
    s = gy * dy + gx * dx
    d = np.where(s < 0, -1.0, 1.0).astype(np.float32) * u.astype(np.float32)
    return np.clip(d, -cap, cap), np.ones(surf.shape, bool)


def block_fields(recto, verso, dy, dx, thr=0.5, cap=CAP, tmin=TMIN):
    """(sdist, midline, thickness, valid_rv) of one block.

    `recto` / `verso` are uint8 probability blocks (verso may be None). The recto FACE is the medial
    surface of the recto band and `d_r` is the signed distance to it; likewise `d_v` for the verso band.
    Along the radial direction a sheet sits between the two faces, so for a voxel at radial coordinate x,
    a recto face at a and a verso face at a - t,

        d_r = x - a           d_v = x - (a - t) = d_r + t

    and therefore, exactly,

        midline  m = (d_r + d_v) / 2        thickness  t = d_v - d_r

    Without a verso band there is no second face: the midline falls back to the recto face (m = d_r) and
    the thickness is NOT measurable, so its `valid_rv` is False and the store records code 0 there."""
    br = recto >= int(round(thr * 255))
    dr, ok = signed_to(medial(br), dy, dx, cap)
    if verso is None:
        return dr, dr, np.zeros_like(dr), np.zeros(dr.shape, bool)
    bv = verso >= int(round(thr * 255))
    if not bv.any():
        return dr, dr, np.zeros_like(dr), np.zeros(dr.shape, bool)
    dv, _ = signed_to(medial(bv), dy, dx, cap)
    m = 0.5 * (dr + dv)
    t = np.maximum(dv - dr, tmin)
    return dr, m, t, ok


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


def _init(recto_path, verso_path, ax, thr, cap, tmin, axis_r_um, halo):
    """Per-process state of a `jobs > 1` worker. Nothing useful is inherited across the fork (an open
    zarr array does not survive it), so every worker opens the stores itself, once."""
    _CTX.clear()
    _CTX.update(recto_path=recto_path, verso_path=verso_path, ax=np.asarray(ax, np.float64), thr=thr,
                cap=cap, tmin=tmin, axis_r_um=axis_r_um, halo=halo, arr={}, axk={})


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
    """One core block: (rung, lo, core shape) -> (rung, lo, midline u8, thickness u8).

    `lo` is the core corner in GLOBAL rung-`rung` voxels; the store is read with a `halo` of context on
    every side so that a distance measured inside the core sees the band that continues outside it."""
    k, lo, n = task
    halo = _CTX["halo"]
    rec_a, ver_a = _arr("recto"), _arr("verso")
    o = np.asarray(rec_a.attrs["origin_zyx"], np.int64) >> (int(k) - 2)   # the region corner at this rung
    loc = np.asarray(lo, np.int64) - o                                   # store-local core corner
    rlo = loc - halo
    rsh = tuple(int(v) + 2 * halo for v in n)
    rec = read_pooled(rec_a, k, rlo, rsh)
    ver = None if ver_a is None else read_pooled(ver_a, k, rlo, rsh)
    dy, dx, r = axis_offsets(_axis_at(k), np.asarray(lo, np.int64) - halo, rsh)
    _, m, t, okt = block_fields(rec, ver, dy, dx, thr=_CTX["thr"], cap=_CTX["cap"], tmin=_CTX["tmin"])
    ok = r >= (_CTX["axis_r_um"] / ladder.rung_um(k))     # the crushed core carries no usable sign
    sl = tuple(slice(halo, halo + int(v)) for v in n)
    return k, tuple(int(v) for v in lo), encode_signed(m, ok, _CTX["cap"])[sl], \
        encode_unsigned(t, (ok & okt))[sl]


def _blocks(shape, block):
    for z in range(0, int(shape[0]), block):
        for y in range(0, int(shape[1]), block):
            for x in range(0, int(shape[2]), block):
                yield (z, y, x), (min(block, int(shape[0]) - z), min(block, int(shape[1]) - y),
                                  min(block, int(shape[2]) - x))


def _pad128(n):
    return int(-(-int(n) // stores.CHUNK) * stores.CHUNK)


def region_fields(root, lo, ax, round_=0, jobs=1, rungs=(2, 3, 4), axis_r_um=AXIS_R_UM,
                  thr=0.5, cap=CAP, tmin=TMIN, block=BLOCK, halo=HALO, force=False):
    """Build the `midline` and `thickness` stores of one region, at every rung in `rungs`.

    `root` is the run directory, `lo` the region corner in rung-2 voxels, `ax` the umbilicus control
    points in rung-2 voxels (`axis.load`), `round_` the self-distillation round whose stores to read and
    write. The region's `recto` store must be done; `verso` may be missing, in which case the midline
    falls back to the recto face and the thickness is no-data everywhere (see `block_fields`).

    Returns a report dict. A rung whose two stores are already `done` is skipped (`force` recomputes),
    which is what makes this resumable at region granularity.

    A pooled rung whose shape is not a multiple of 128 is padded up to one, because a store's shape must
    be; the padding is code 0, i.e. no data. For the production region (1024 at rung 2) rungs 3 and 4 are
    512 and 256 and nothing is padded."""
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
        if not force and all(stores.is_done(p) for p in paths.values()):
            rep["rungs"][k] = {"skipped": "done"}
            continue
        Sk = S2 >> (k - 2)
        todo.append((k, Sk, paths))
    if not todo:
        return rep
    tasks = [(k, tuple(int(q) + int(o) for q, o in zip(b, origin2 >> (k - 2))), n)
             for k, Sk, _ in todo for b, n in _blocks(Sk, block)]
    init = (rp, vp, np.asarray(ax, np.float64), thr, cap, tmin, float(axis_r_um), int(halo))
    out = {k: {kind: np.zeros(tuple(_pad128(v) for v in Sk), np.uint8) for kind in KINDS}
           for k, Sk, _ in todo}
    if int(jobs) <= 1 or len(tasks) < 2:
        _init(*init)
        results = (_block(t) for t in tasks)
        for k, blo, m, t in results:
            _store_block(out[k], k, blo, origin2, m, t)
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=int(jobs), mp_context=mp.get_context("fork"),
                                    initializer=_init, initargs=init) as ex:
            for k, blo, m, t in ex.map(_block, tasks, chunksize=1):
                _store_block(out[k], k, blo, origin2, m, t)
    for k, Sk, paths in todo:
        for kind in KINDS:
            stores.write(paths[kind], out[k][kind], tuple(int(v) for v in (origin2 >> (k - 2))),
                         rung=k, channels=(channel(kind, k),), q=0,
                         attrs={"field": kind, "unit_vox": UNIT, "offset": OFF if kind == "midline" else 0,
                                "cap_vox": float(cap), "axis_r_um": float(axis_r_um),
                                "shape_true": [int(v) for v in Sk], "verso": bool(vp),
                                "source_round": int(round_)})
        rep["rungs"][k] = {"shape": [int(v) for v in Sk], "blocks": sum(1 for t in tasks if t[0] == k),
                           "written": [paths[kind] for kind in KINDS]}
    return rep


def _store_block(dst, k, blo, origin2, m, t):
    """Place one worker's core block into the rung-k output arrays (parent side, so the assembly order
    cannot change the bytes)."""
    loc = np.asarray(blo, np.int64) - (origin2 >> (int(k) - 2))
    sl = tuple(slice(int(a), int(a) + int(s)) for a, s in zip(loc, m.shape))
    dst["midline"][sl] = m
    dst["thickness"][sl] = t
