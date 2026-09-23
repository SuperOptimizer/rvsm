"""Regions: which 1024^3 boxes are worth producing, in what order, and where their targets live.

usrm2 decided all of this from a published MASK pyramid -- a tile was worth visiting if the mask's block
maximum over it was non-zero. rvsm has no mask, so occupancy comes from the CT itself and it is a block
MEAN rather than a block maximum:

    tiles_fraction = mean(ct_level > 0) over each tile's footprint at the occupancy rung

A mean is the right statistic without a mask. The maximum of `ct > 0` is 1 for any tile with a single
non-air voxel, which on a CT (where the mask's zeros were the only air) keeps essentially every tile in
the bounding box. The mean says how much of the tile is scroll, which is both the keep/drop test
(>= 0.05 at rung 2, >= 0.01 at rungs >= 5 -- coarse tiles are mostly air by construction) and the draw
weight (x sqrt(fraction): a tile half papyrus is drawn ~2.2x as often as one at 10 %, not 5x, because
the windows inside a dense tile are correlated).

The rest of the module is the target side. One store per region per channel at rung 2 (`stores.py`), and
above it:

    rungs 3     the store read as its own 2x pool (`stores.read_store`)
    rungs 4-6   `pooled()`: the rung-2 store pooled 2^(k-2) ON THE FLY and cached, rather than written
                out as a per-region mini-pyramid -- one array per store stays the contract, a pooled
                block costs microseconds and nothing can go stale
    rungs 7-11  `coarse.zarr/<k>`: one whole-scroll array per rung, fed one pooled block per finished
                region, with a `coverage/<k>` bitmap on the same grid. A coarse window straddles many
                regions, most of which are not produced yet, so the loader weight there is the coverage
                fraction and an unfed voxel simply carries weight 0.

`Catalog` is the region state machine, derived from the disk: a store is done iff its `zarr.json` says
so. A 30 s TTL keeps the trainer from stat-ing thousands of directories per step while still picking up
what the producer finished half a minute ago; a HIT is sticky, because a done store never un-finishes.
"""
from __future__ import annotations

import os
import time

import numpy as np

from rvsm import ladder, stores

OCC_VOX = 64 << 20     # the occupancy check reads the finest CT level with at most this many voxels
COARSE_RUNGS = (7, 8, 9, 10, 11)
COARSE_CHUNK = 128     # was 32: a 256^3 coarse window touched 512 chunks per array (~0.8 s a draw)
COARSE_MIN = 5         # rungs >= this use the coarse occupancy threshold
TTL = 30.0             # seconds a MISSING store stays missing in a catalog (the producer writes as we train)


# --------------------------------------------------------------------------- occupancy

def occupancy_rung(pyr, cap=OCC_VOX):
    """The finest rung of a pyramid whose whole level is at most `cap` voxels (the coarsest if none is)."""
    for k in sorted(pyr):
        if int(np.prod(pyr[k].shape[-3:])) <= cap:
            return k
    return max(pyr)


def occupancy(pyr, k=None, cap=OCC_VOX):
    """(rung, bool array of the whole level): where the CT is not air. A few MB per scroll."""
    k = occupancy_rung(pyr, cap) if k is None else int(k)
    a = ladder.full_level(pyr, k)
    if a is None:
        a = ladder._read(pyr[k], (slice(None),) * 3)
    return k, np.asarray(a, np.uint8) > 0


def shard_grid(pyr, k):
    """The shard (write-chunk) size, in rung-k voxels, of the level a pyramid is read from at rung k:
    what a region is snapped to, so that one region is one shard footprint."""
    src = max(r for r in pyr if r <= k)
    g = np.array(getattr(pyr[src], "shards", None) or pyr[src].chunks, np.int64)[-3:]
    return np.maximum(g >> (k - src), 1)


def region_tiles(pyr, k, region=1024, patch=256):
    """The shard-aligned tiles covering the volume at rung k: (per-axis origins, tile size). The tile is
    `region` rounded DOWN to a multiple of the shard grid (never below one shard, nor above the volume),
    so every tile origin is a shard boundary and one visit touches one shard footprint per level."""
    p, bs = ladder.shape3(patch), ladder.rung_shape(pyr, k)
    g = shard_grid(pyr, k)
    R = np.maximum((ladder.shape3(region) // g) * g, g)
    R = np.minimum(R, -(-np.maximum(bs, p) // g) * g)  # a volume smaller than a region: one aligned tile
    ax = [np.arange(0, int(bs[d]), int(R[d]), dtype=np.int64) for d in range(3)]
    return ax, R


def tiles_fraction(occ, ko, k, ax):
    """The block MEAN of the occupancy array `occ` (given at rung `ko`) over each tile's footprint at
    rung k: an (n_z, n_y, n_x) array of fractions in 0..1, one per tile of the `ax` grid.

    This replaces usrm2's block MAXIMUM against a mask. When a tile is smaller than one coarse voxel the
    blocks merge and neighbouring tiles share a fraction, which keeps a tile its neighbour occupies --
    the check only ever errs towards keeping, exactly as the maximum did."""
    o = np.asarray(occ, np.float64)
    for d in range(3):
        st = (ax[d] >> (ko - k)) if ko >= k else (ax[d] << (k - ko))
        st = np.clip(st, 0, max(o.shape[d] - 1, 0))
        u, inv = np.unique(st, return_inverse=True)
        s = np.add.reduceat(o, u, axis=d)
        ln = np.diff(np.append(u, o.shape[d])).astype(np.float64)
        o = np.take(s / ln.reshape([-1 if i == d else 1 for i in range(3)]), inv, axis=d)
    return o


def rung_probs(pyr, patch=256, rungs=None, boost=None):
    """{rung: probability} of the rung mix: p_k ~ sqrt(n_k), n_k = voxels at rung k / voxels per patch
    (floored at 1), times `boost`. n_k falls 8x per rung, so each rung is 2*sqrt(2) times rarer than the
    one below until the floor flattens the top -- which is what keeps the coarse rungs, where the whole
    scroll is a handful of patches, from vanishing out of the mix."""
    per = float(np.prod(ladder.shape3(patch)))
    allowed = None if rungs in (None, True) else {int(r) for r in rungs}
    b = {int(q): float(v) for q, v in dict(boost or {}).items()}
    out = {}
    for k in range(min(pyr), ladder.NRUNGS):
        if allowed is not None and k not in allowed:
            continue
        n = max(float(np.prod(ladder.rung_shape(pyr, k))) / per, 1.0)
        out[k] = (n ** 0.5) * b.get(k, 1.0)
    tot = sum(out.values()) or 1.0
    return {k: v / tot for k, v in out.items()}


def _scaled(o, s, k):
    """A rung-2 (origin, size) expressed in rung-k voxels."""
    d = int(k) - 2
    o, s = np.asarray(o, np.int64), np.asarray(s, np.int64)
    return (o >> d, np.maximum(s >> d, 1)) if d >= 0 else (o << -d, s << -d)


def region_list(pyr, rungs=None, patch=256, region=1024, boost=None, exclude=(), cap=OCC_VOX,
                occ_min_fine=0.05, occ_min_coarse=0.01, log=None):
    """Every (rung, shard-aligned region) worth visiting, with its CT occupancy fraction and its draw
    weight. `exclude` is the held-out regions as (origin, size) at rung 2: a region entirely inside one
    is dropped at EVERY rung. The weights sum to 1; the list is in enumeration order (`walk_order` gives
    the visit order)."""
    ko, occ = occupancy(pyr, cap=cap)
    out = []
    for k, pk in rung_probs(pyr, patch, rungs, boost).items():
        ax, R = region_tiles(pyr, k, region, patch)
        fr = tiles_fraction(occ, ko, k, ax)
        thr = float(occ_min_coarse if k >= COARSE_MIN else occ_min_fine)
        ex = [_scaled(o, s, k) for o, s in exclude]
        rows = []
        for a, b, c in np.argwhere(fr >= thr):
            lo = np.array([ax[0][a], ax[1][b], ax[2][c]], np.int64)
            if any(np.all(lo >= eo) and np.all(lo + R <= eo + es) for eo, es in ex):
                continue
            rows.append((lo, float(fr[a, b, c])))
        if not rows:
            continue
        sw = sum(f ** 0.5 for _, f in rows) or 1.0     # the draw weight goes as sqrt(fraction)
        out += [{"k": int(k), "lo": [int(v) for v in lo], "size": [int(v) for v in R],
                 "f": f, "w": float(pk) * (f ** 0.5) / sw} for lo, f in rows]
        if log:
            log(f"  rung {k:>2}: {len(rows):>7} regions of "
                f"{int(np.prod([len(q) for q in ax])):>7} (tile {tuple(int(v) for v in R)})")
    tot = sum(q["w"] for q in out) or 1.0
    for q in out:
        q["w"] /= tot
    return out


def region_visits(regions, cap=64):
    """`region_list` expanded into VISITS, so the rung mix holds throughout the walk and not only in the
    expectation of a prefix.

    A weighted shuffle front-loads the heavy items, and a rung with few regions but a large weight (rung
    9 of a scroll is ONE region) is therefore used up in the first percent of the walk and never seen
    again. Giving that region v = round(w * R) visits, each of weight w / v, makes almost every entry
    weigh 1 / R: the order becomes near-uniform and the coarse rungs are spread over the whole epoch. A
    visit draws its own windows, so v > 1 is a denser sampling of a region, not the same windows twice.
    The fine rungs keep their one visit each."""
    out, R = [], len(regions)
    for r in regions:
        v = int(min(max(round(r["w"] * R), 1), max(int(cap), 1)))
        for j in range(v):
            out.append(dict(r, w=r["w"] / v, v=j))
    return out


def walk_order(w, seed=0):
    """A weighted shuffle WITHOUT replacement (Efraimidis-Spirakis): key = Exp(1) / w, ascending. The
    first item is i with probability w_i / sum(w), and every item appears exactly once."""
    rng = np.random.default_rng(int(seed))
    w = np.asarray([q["w"] if isinstance(q, dict) else q for q in w], np.float64)
    return np.argsort(rng.exponential(size=len(w)) / np.maximum(w, 1e-300), kind="stable")


def held_out(regions, n=8, seed=0, ax=None, f_min=0.5):
    """`n` rung-2 regions held out of the walk for good, stratified over z-thirds and radius-thirds.

    The held-out set is the only thing every round is scored on, so it must not all come from one end of
    the scroll or one depth: a metric measured on eight core regions says nothing about the outer wraps,
    where the sheets separate. Candidates are the rung-2 regions with occupancy >= `f_min` (a region
    half air cannot carry a recall number); they are bucketed by the z third of the volume and the third
    of the radial range their CENTRE falls in, and taken round-robin over the buckets so the set spreads
    even when one bucket is huge. `ax` is the axis in rung-2 voxels (`axis.load`); without one the
    radius stratification is skipped."""
    cand = [r for r in regions if int(r["k"]) == 2 and float(r.get("f", 1.0)) >= float(f_min)]
    if not cand:
        cand = [r for r in regions if int(r["k"]) == 2]
    if not cand:
        return []
    c = np.array([np.array(r["lo"], np.float64) + np.array(r["size"], np.float64) / 2 for r in cand])
    zt = _third(c[:, 0])
    if ax is None:
        rt = np.zeros(len(cand), np.int64)
    else:
        ax = np.asarray(ax, np.float64)
        cy, cx = np.interp(c[:, 0], ax[0], ax[1]), np.interp(c[:, 0], ax[0], ax[2])
        rt = _third(np.hypot(c[:, 1] - cy, c[:, 2] - cx))
    rng = np.random.default_rng(int(seed))
    buckets = {}
    for i, (a, b) in enumerate(zip(zt, rt)):
        buckets.setdefault((int(a), int(b)), []).append(i)
    keys = sorted(buckets)
    for k in keys:
        rng.shuffle(buckets[k])
    out, j = [], 0
    while len(out) < int(n) and any(buckets[k] for k in keys):
        k = keys[j % len(keys)]
        if buckets[k]:
            out.append(cand[buckets[k].pop()])
        j += 1
    return out


def _third(v):
    """Which third of the observed range each value falls in (0, 1 or 2)."""
    v = np.asarray(v, np.float64)
    lo, hi = float(v.min()), float(v.max())
    return np.clip(((v - lo) / max(hi - lo, 1e-9) * 3).astype(np.int64), 0, 2)


def exclude_boxes(regions):
    """The held-out regions as the (origin, size) pairs at rung 2 that `region_list` excludes."""
    return [(np.array(r["lo"], np.int64), np.array(r["size"], np.int64)) for r in regions]


# --------------------------------------------------------------------------- the catalog

class Catalog:
    """Which region stores are finished, derived from the disk and cached for `ttl` seconds.

    A store is done iff `stores.is_done` says so, and a done store never un-finishes -- so a HIT is
    cached for the life of the catalog and only a MISS expires. That is the whole region state machine:
    there is no database, and `rvsm ledger --rebuild` is a directory scan."""

    def __init__(self, root, round_=0, ttl=TTL):
        self.root, self.round, self.ttl = str(root), int(round_), float(ttl)
        self._c = {}

    def path(self, channel, lo):
        return stores.store_path(self.root, channel, lo, self.round)

    def done(self, channel, lo):
        p = self.path(channel, lo)
        hit = self._c.get(p)
        if hit is not None and (hit[0] or time.time() - hit[1] < self.ttl):
            return bool(hit[0])
        ok = stores.is_done(p)
        self._c[p] = (ok, time.time())
        return ok

    def ready(self, lo, need=("recto",)):
        """Is this region produced for every channel a sample needs?"""
        return all(self.done(c, lo) for c in need)

    def open(self, channel, lo):
        """The opened store, or None when it is not finished (the miss is cached, so this is cheap)."""
        if not self.done(channel, lo):
            return None
        p = self.path(channel, lo)
        a = self._c.get(("arr", p))
        if a is None:
            try:
                a = stores.open_store(p)
            except Exception:  # noqa: BLE001  (finished between the check and the open, or a bad build)
                return None
            self._c[("arr", p)] = a
        return a

    def list_done(self, channel):
        """[(z, y, x), ...] of every finished region of a channel, in name order."""
        d = os.path.join(self.root, "stores", f"round_{self.round}", str(channel))
        if not os.path.isdir(d):
            return []
        out = []
        for n in sorted(os.listdir(d)):
            if not (n.startswith("region_") and n.endswith(".zarr")):
                continue
            try:
                z, y, x = (int(v) for v in n[len("region_"):-len(".zarr")].split("_"))
            except ValueError:
                continue
            if stores.is_done(os.path.join(d, n)):
                out.append((z, y, x))
        return sorted(out)


# --------------------------------------------------------------------------- pooling a region

_POOL = {}     # {(path, k): ndarray} -- the pooled views of a region store, built once per loader
POOL_BYTES = 256 << 20   # ... and at most this many bytes of them per process (LRU)
POOL_SLAB = 64           # z-planes of OUTPUT per slab when a whole store is pooled


def pool_store(a, d, slab=POOL_SLAB):
    """The whole region store `a` mean-pooled 2^d (`ladder.pool2` d times), read in z-slabs.

    Byte-identical to reading the store whole and pooling it (a slab is a multiple of 2^d planes, so
    only the last one can meet `pool2`'s end padding, exactly as the whole volume would), but the
    transient is one slab -- 64 << d planes, 128 MB for a rung-3 pool of a 1024^3 store -- instead of the
    1 GB store plus its pooling copies. Six loader workers each holding that 3-4 GB transient at once
    was a ~20 GB host-RAM spike on the 64 GB production host."""
    from rvsm import ladder
    d = int(d)
    S = tuple(int(v) for v in a.shape[-3:])
    step = int(slab) << d
    parts = []
    for z in range(0, S[0], step):
        v = np.asarray(a[z:min(z + step, S[0])], np.uint8)
        for _ in range(d):
            v = ladder.pool2(v)
        parts.append(v)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def pooled(root, channel, lo, k, round_=0, cache=_POOL, limit=64, max_bytes=None):
    """The rung-2 region store of `channel` at `lo`, mean-pooled 2^(k-2) to rung k (3 <= k <= 6).

    A per-region mini-pyramid WRITTEN into the store group would make a store more than one array and
    give a finished store a way to be half-updated; pooling on the fly keeps the one-array contract and
    costs a few hundred microseconds per region per rung, once, because the result is cached (at most
    `limit` entries and `max_bytes` -- `POOL_BYTES` -- bytes, least recently used out first). Returns
    None when the store is not finished."""
    k = int(k)
    p = stores.store_path(root, channel, lo, round_)
    key = (p, k)
    if key in cache:
        v = cache.pop(key)
        cache[key] = v                   # most recently used last
        return v
    if not stores.is_done(p):
        return None
    v = pool_store(stores.open_store(p), k - 2)
    cap = int(POOL_BYTES if max_bytes is None else max_bytes)
    while cache and (len(cache) >= int(limit) or sum(x.nbytes for x in cache.values()) + v.nbytes > cap):
        cache.pop(next(iter(cache)))
    cache[key] = v
    return v


def pooled_window(root, channel, lo, k, win, shape, round_=0):
    """(cube, inside) of a pooled region read at rung k: `win`/`shape` in rung-k voxels, `lo` the
    region's rung-2 origin. The same contract as `stores.read_store`, so the sampler treats rungs 2-3
    and 4-6 identically."""
    v = pooled(root, channel, lo, k, round_)
    shape = tuple(int(s) for s in ladder.shape3(shape))
    out, ins = np.zeros(shape, np.uint8), np.zeros(shape, bool)
    if v is None:
        return out, ins
    o = np.asarray(lo, np.int64) >> (int(k) - 2)
    w = np.asarray(win, np.int64) - o
    S = np.array(v.shape, np.int64)
    a, b = np.maximum(w, 0), np.minimum(w + np.array(shape, np.int64), S)
    if (b > a).all():
        st = a - w
        blk = v[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
        sl = tuple(slice(int(s), int(s) + int(n)) for s, n in zip(st, blk.shape))
        out[sl], ins[sl] = blk, True
    return out, ins


def clear_pool():
    _POOL.clear()


# --------------------------------------------------------------------------- coarse.zarr

def coarse_root(root, channel, round_=0):
    return os.path.join(str(root), "stores", f"round_{int(round_)}", str(channel), "coarse.zarr")


def _coarse_array(root, channel, k, round_=0, shape2=None, coverage=False, create=True):
    """`coarse.zarr/<k>` (or `coarse.zarr/coverage/<k>`), created lazily at the whole-scroll shape.

    Plain zarr, uint8, 32^3 chunks: these arrays are a few MB per rung for a whole scroll, they are
    written one small block at a time by the producer and read at arbitrary corners by the loader, so
    neither volcomp (whose 128^3 block is larger than a whole coarse level) nor a shard makes sense."""
    import zarr
    d = coarse_root(root, channel, round_)
    p = os.path.join(d, "coverage", str(int(k))) if coverage else os.path.join(d, str(int(k)))
    if os.path.exists(os.path.join(p, "zarr.json")):
        return zarr.open_array(p, mode="r+")
    if not create or shape2 is None:
        return None
    sh = tuple(int(v) for v in np.maximum(-(-np.asarray(shape2, np.int64) >> (int(k) - 2)), 1))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return zarr.create_array(p, shape=sh, chunks=(COARSE_CHUNK,) * 3, dtype="uint8", fill_value=0,
                             overwrite=False)


def pool_chain(u8, ks=COARSE_RUNGS):
    """{k: the rung-2 uint8 block mean-pooled 2^(k-2)} for every k in `ks`, each rung pooled from the one
    below it ONCE (a fresh chain per rung pooled the 1 GB block five times: ~50 s a region).

    `u8` may be a numpy array or a uint8 TENSOR on any device; a tensor is pooled there (avg_pool3d of
    the zero-padded block, then floored, which is `ladder.pool2` bit for bit: the mean of eight integers
    is exact in float32 and the cast truncates) and only the small results come back to the host."""
    want = sorted(int(k) for k in ks)
    if not want:
        return {}
    out = {}
    if hasattr(u8, "is_cuda") or type(u8).__module__.startswith("torch"):
        import torch
        import torch.nn.functional as F
        v = u8
        for k in range(3, max(want) + 1):
            if min(v.shape) < 2:
                break
            pad = [q for d in (2, 1, 0) for q in (0, int(v.shape[d]) % 2)]
            x = F.pad(v.float()[None, None], pad)
            v = torch.floor(F.avg_pool3d(x, 2))[0, 0].to(torch.uint8)
            if k in want:
                out[k] = v.cpu().numpy()
        return out
    v = np.ascontiguousarray(u8, np.uint8)
    for k in range(3, max(want) + 1):
        if min(v.shape) < 2:
            break
        v = ladder.pool2(v)
        if k in want:
            out[k] = v
    return out


def feed_coarse(root, channel, lo, u8, round_=0, shape2=None, ks=COARSE_RUNGS, pooled=None):
    """Fold a finished rung-2 region block into the whole-scroll coarse arrays, rungs 7..11.

    `u8` is the region's rung-2 uint8 block and `lo` its rung-2 origin. For each rung k the block is mean
    pooled 2^(k-2) and written at `lo >> (k-2)`, and the same footprint of `coverage/<k>` is set to 255.
    A rung whose pooling factor is larger than the region itself is SKIPPED: the block would be under a
    single coarse voxel and two different regions would collide in it. Returns the rungs written.

    `pooled` is `pool_chain(u8, ks)` when the caller already has it (a producer pools on the GPU); `u8`
    is then only consulted for its shape."""
    lo = np.asarray(lo, np.int64)
    shp = tuple(int(q) for q in u8.shape)
    ok = [int(k) for k in ks if (1 << (int(k) - 2)) <= min(shp) and not (lo % (1 << (int(k) - 2))).any()]
    pooled = pool_chain(u8, ok) if pooled is None else pooled
    out = []
    for k in ok:
        v = pooled[k]
        a = _coarse_array(root, channel, k, round_, shape2=shape2)
        c = _coarse_array(root, channel, k, round_, shape2=shape2, coverage=True)
        if a is None or c is None:
            continue
        o = lo >> (int(k) - 2)
        S = np.array(a.shape, np.int64)
        b = np.minimum(o + np.array(v.shape, np.int64), S)
        if (b <= o).any():
            continue
        sl = tuple(slice(int(x), int(y)) for x, y in zip(o, b))
        blk = v[:int(b[0] - o[0]), :int(b[1] - o[1]), :int(b[2] - o[2])]
        a[sl] = blk
        c[sl] = np.uint8(255)
        out.append(int(k))
    return out


def read_coarse(root, channel, k, lo, shape, round_=0):
    """(cube, coverage) of the coarse array at rung k over the window `lo`/`shape` (rung-k voxels).

    `coverage` is a float in 0..1 per voxel: 1 where a produced region has been folded in, 0 where
    nothing has been produced yet. The loader multiplies the sample weight by it, so a coarse window
    that straddles thirty regions of which two are done trains on those two and ignores the rest."""
    shape = tuple(int(s) for s in ladder.shape3(shape))
    out, cov = np.zeros(shape, np.uint8), np.zeros(shape, np.float32)
    a = _coarse_array(root, channel, k, round_, create=False)
    c = _coarse_array(root, channel, k, round_, coverage=True, create=False)
    if a is None or c is None:
        return out, cov
    lo = np.asarray(lo, np.int64)
    S = np.array(a.shape, np.int64)
    aa, bb = np.maximum(lo, 0), np.minimum(lo + np.array(shape, np.int64), S)
    if (bb <= aa).any():
        return out, cov
    sl = tuple(slice(int(x), int(y)) for x, y in zip(aa, bb))
    st = aa - lo
    blk, cvb = np.asarray(a[sl], np.uint8), np.asarray(c[sl], np.uint8)
    dst = tuple(slice(int(s), int(s) + int(n)) for s, n in zip(st, blk.shape))
    out[dst] = blk
    cov[dst] = cvb.astype(np.float32) / 255.0
    return out, cov
