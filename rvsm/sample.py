"""The sampler: the region walk in, compact uint8 training items out.

One sample is (rung k, region, window). The walk (`regions.walk_order`) fixes which region is visited
when; a visit draws `windows_per_region` windows inside that one region, so every read of the visit
lands in the shards the producer just fetched for it. What the worker yields is deliberately NOT a model
input: it is the uint8 cubes and the uint8 target, and `prep.prepare` does every float operation on the
GPU (`rung_item`'s docstring says what each key is; `config.RUNG_ITEM_KEYS` is the contract and this
module emits exactly those keys, always, so a collate never has to branch).

Where a target comes from is the one thing that changes with the rung:

    rung 2      the region store, read directly              (`stores.read_store`)
    rung 3      the same store as its 2x mean pool            (`stores.read_store`, k = rung + 1)
    rungs 4-6   the same store pooled 2^(k-2) on the fly      (`regions.pooled`)
    rungs 3-4   a DISTANCE channel: its own rung-k store      (`targets.channel(kind, k)`, never pooled)
    rungs 7-11  the whole-scroll coarse array x its coverage  (`regions.read_coarse`)

A rung-2 window is drawn inside its region, so it reads exactly one store (and a rung-2 window that
straddles two, which `_draw` never makes, gets weight 0). At rungs 3-6 one window covers several
region stores -- a rung-4 window is a whole region's footprint, a rung-5 one eight -- so a "one store or
nothing" rule gave those rungs weight 0 almost always (paris4, step 400). There the target is STITCHED
per voxel: every voxel is read from the region store it lies in, with that store's own `inside`, so a
voxel whose store is not finished (or is a held-out region's) carries weight 0 and nothing is invented.
The window is drawn around the visit's HOME region -- the one region `run.region_route` produced for
that visit -- so it always overlaps a finished store and reads the shards just fetched for it.

The REJECTION RULES are one rule. usrm2 had four -- air, minimum foreground, density power, all-masked
-- and three of them needed a label to evaluate, which is exactly what rvsm does not have when it starts.
What is left is the CT-air rejection: a window that is more than 90 % air is kept with probability
`air_keep`. Everything else is expressed as WEIGHT rather than as rejection, which is strictly more
information: the sample still trains wherever it has a target.

The weight of a voxel is `inside x (CT > 0) x rw x coverage`, times two masks that are about physics
rather than data: a DISTANCE channel (midline, thickness) exists only at rungs 2-4 and its code 0 means
"no data", and the verso / distance / confidence channels carry weight 0 within `AXIS_R_UM` of the
umbilicus, where the scroll's own centre makes recto and verso meaningless.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from rvsm import axis as AX, ladder, regions as RG, scanmeta as SM, stores
from rvsm.config import RUNG_ITEM_KEYS

NORM = None            # None = per-patch z-score; (mean, std) = a fixed scan-level normalisation
AXIS_R_UM = 400.0      # verso / distance / confidence carry no weight this close to the scroll axis
DIST_CHANNELS = ("midline", "thickness")
NEAR_AXIS_ZERO = ("verso", "midline", "thickness", "conf")
DIST_MAX_RUNG = 4      # a distance is never pooled: rungs 2..4 only
RW = "rw"              # the per-voxel agreement weight store written beside round 0's recto


def zscore(x):
    x = np.asarray(x, np.float32)
    if NORM is not None:
        return (x - NORM[0]) / NORM[1]
    return (x - x.mean()) / (x.std() + 1e-3)


# --------------------------------------------------------------------------- the 48 cube symmetries

SYM_PERMS = tuple(itertools.permutations(range(3)))  # the 6 axis permutations


def sym_decode(sym):
    """A cube symmetry index 0..47 -> (axis permutation, flips): sym = 8 * permutation index + flips."""
    return np.array(SYM_PERMS[int(sym) // 8]), np.array([bool(int(sym) >> d & 1) for d in range(3)])


def draw_sym(rng, shape):
    """The cube symmetry of one patch, as an index 0..47. Only permutations that keep the patch shape
    are drawn (a 384x512x512 patch may swap y and x, not z). Index 0 is the identity."""
    sh = np.array(shape)
    perms = [q for q in SYM_PERMS if (sh[list(q)] == sh).all()]
    perm, flip = perms[rng.integers(len(perms))], rng.random(3) < 0.5
    return SYM_PERMS.index(perm) * 8 + int(flip[0]) + 2 * int(flip[1]) + 4 * int(flip[2])


def sym_apply(sym, x, tg):
    """Cube symmetry `sym` applied to the (C,Z,Y,X) input and the (T,Z,Y,X) target; the radial vector
    channels (the LAST 3 of x) are permuted and negated to match. `prep.sym_apply_t` is the same map on
    the GPU and the tests check the two agree for all 48 symmetries."""
    perm, flip = sym_decode(sym)
    sl = tuple(slice(None, None, -1 if f else 1) for f in flip)
    tg = np.ascontiguousarray(np.transpose(tg, (0,) + tuple(perm + 1))[(slice(None),) + sl])
    x = np.transpose(x, (0,) + tuple(perm + 1))[(slice(None),) + sl]
    ni = x.shape[0] - 3
    x = np.concatenate([x[:ni],
                        x[ni + perm] * np.where(flip, -1, 1).astype(np.float32)[:, None, None, None]])
    return np.ascontiguousarray(x), tg


# --------------------------------------------------------------------------- one compact sample

def compact_rows(tg, w):
    """(tch, tg, w) with only the target rows that carry anything: `tch` (T,) uint8 is 1 for a row whose
    weight OR target is non-zero anywhere, and `tg` / `w` keep just those rows, in channel order. Every
    row present: `tg` / `w` come back as they were. A row that is dropped is all zeros in BOTH, so
    `prep.expand_tw` (zeros elsewhere) rebuilds the full pair bit for bit."""
    T = int(tg.shape[0])
    tch = (tg.reshape(T, -1).any(1) | w.reshape(T, -1).any(1)).astype(np.uint8)
    if tch.all():
        return tch, tg, w
    keep = np.flatnonzero(tch)
    return tch, np.ascontiguousarray(tg[keep]), np.ascontiguousarray(w[keep])


def rung_item(ct, tg, w, k, lo, ax, sym=0, norm=None, cm=None, cx=None, lo1=None, rmax=0.0, meta=None,
              compact=False):
    """The compact sample the loader yields: everything uint8, so a 256^3 sample is ~200 MB instead of
    the ~1 GB of float32 a built model input would be. It carries EXACTLY `config.RUNG_ITEM_KEYS`:

    ct   (1 + nctx, Z, Y, X) uint8: the CT cube and the context cubes as read, not z-scored
    tgt  (T, Z, Y, X) uint8: the target fields (recto, verso, midline, thickness), CT-air zeroed
    w    (T, Z, Y, X) uint8: the per-voxel weight, 255 = 1.0
    tch  (T,) uint8: which of the T target rows `tgt` / `w` carry, in order. All ones unless `compact`:
         then only the rows with any weight or target (`compact_rows`) are sent -- round 0 has no verso
         and no distance stores, so a 256^3 item drops 3 of 4 rows (~96 MB of ~320 MB of pageable
         host-to-device copy). `prep.prepare` scatters them back into the full (T, ...) pair, zeros
         elsewhere, before anything reads them; `prep.full_tw` does the same for a CPU consumer.
    lo   (3,) int64: the corner, in rung-k voxels
    cyx  (2, Z) float64: the scroll axis (y, x) at each z of the cube -- what `radial_t` interpolates
    sym  (): the cube symmetry index drawn by the worker (0 = identity), applied on the GPU
    rung (), norm (2,): the rung k and the (mean, std) of the z-score (std 0 = per-patch)
    cm   (Z/2, Y/2, X/2) uint8: the rung-(k+1) target block over the patch footprint -- the `mask` source
         of the cascade channel, upsampled 2x on the GPU. EMPTY (0, 0, 0) in a training draw whose
         cascade mode never reads it (self, off, label-free): the key stays, so the stem width and the
         collate do not change, but the block is neither read by the worker nor copied to the card
    cx   (1, Z, Y, X) uint8: the TENTH context cube (rung k + ctx[-1] + 1), the one extra cube the
         rung-(k+1) input needs that the rung-k input does not. EMPTY (1, 0, 0, 0) in a training draw
         whose cascade mode never reads it (mask, off, label-free). In self / mix it is always sent: the
         `cascade_drop` draw happens on the device, after the copy
    lo1  (3,) int64 / cyx1 (2, Z): the corner and axis of that rung-(k+1) cube, so `prep` can rebuild
         the coarse input's radial vector
    rmax (): the radius plane's denominator, in rung-k voxels
    meta (5,): this scan's normalised conditioning values (`scanmeta.scan_planes`)

    A key with nothing behind it is ZERO rather than absent: an optional key would make every consumer
    branch, and `prep` is already the one place that knows what each channel means."""
    ct = np.ascontiguousarray(ct)
    p = np.array(ct.shape[-3:], np.int64)
    lo = np.asarray(lo, np.int64)
    a = AX.axis_at(ax, k)
    z = np.arange(int(p[0])) + lo[0]
    cyx = np.stack([np.interp(z, a[0], a[1]), np.interp(z, a[0], a[2])])
    nm = (0.0, 0.0) if (norm or NORM) is None else tuple(float(v) for v in (norm or NORM))
    hp = np.maximum(p // 2, 1)
    cm = np.zeros(tuple(hp), np.uint8) if cm is None else np.asarray(cm, np.uint8)
    cx = np.zeros((1,) + tuple(p), np.uint8) if cx is None else np.asarray(cx, np.uint8)
    cx = cx if cx.ndim == 4 else cx[None]
    lo1 = ((lo + p // 2) // 2 - p // 2) if lo1 is None else np.asarray(lo1, np.int64)
    a1 = AX.axis_at(ax, int(k) + 1)
    z1 = np.arange(int(p[0])) + lo1[0]
    cyx1 = np.stack([np.interp(z1, a1[0], a1[1]), np.interp(z1, a1[0], a1[2])])
    meta = np.zeros(len(SM.META_RANGE), np.float32) if meta is None else np.asarray(meta, np.float32)
    tg, w = np.asarray(tg), np.asarray(w)
    if compact:
        tch, tg, w = compact_rows(tg, w)
    else:
        tch = np.ones(int(tg.shape[0]), np.uint8)
    return {"ct": torch.from_numpy(ct),
            "tgt": torch.from_numpy(np.ascontiguousarray(tg)),
            "w": torch.from_numpy(np.ascontiguousarray(w)),
            "tch": torch.from_numpy(tch),
            "lo": torch.from_numpy(np.ascontiguousarray(lo)),
            "cyx": torch.from_numpy(np.ascontiguousarray(cyx)),
            "sym": torch.tensor(int(sym)), "rung": torch.tensor(int(k)),
            "norm": torch.tensor(nm, dtype=torch.float32),
            "cm": torch.from_numpy(np.ascontiguousarray(cm)),
            "cx": torch.from_numpy(np.ascontiguousarray(cx)),
            "lo1": torch.from_numpy(np.ascontiguousarray(lo1)),
            "cyx1": torch.from_numpy(np.ascontiguousarray(cyx1)),
            "rmax": torch.tensor(float(rmax), dtype=torch.float32),
            "meta": torch.as_tensor(meta)}


# --------------------------------------------------------------------------- the dataset

def target_channels(cfg):
    """The target fields of a sample, in head order: the probability channels then the distances."""
    return list(cfg.channels) + list(DIST_CHANNELS)


def _clip_read(arr, o, n):
    """(cube, inside) of `arr[o:o+n]` with whatever part lies outside `arr` zero and not inside."""
    o, n = np.asarray(o, np.int64), np.asarray(n, np.int64)
    out, ins = np.zeros(tuple(n), np.uint8), np.zeros(tuple(n), np.float32)
    S = np.array(arr.shape, np.int64)
    a, b = np.maximum(o, 0), np.minimum(o + n, S)
    if (b > a).all():
        st = a - o
        blk = arr[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
        sl = tuple(slice(int(st[j]), int(st[j]) + blk.shape[j]) for j in range(3))
        out[sl], ins[sl] = blk, 1.0
    return out, ins


class Patches(torch.utils.data.IterableDataset):
    """An endless stream of compact samples, drawn along the region walk.

    `label_free=True` is the pretraining mode: no store is opened, every target and weight is zero, and
    the only rule is the CT-air rejection -- which makes the whole mirrored scroll training data before
    a single teacher pass has run."""

    def __init__(self, cfg, root=None, ct=None, ax=None, round_=0, region_records=None, heldout=(),
                 label_free=False, seed=0, sym=True, need=None, meta=None, windows=None):
        super().__init__()
        self.cfg = cfg
        self.root = str(root or cfg.out)
        self.ct = str(ct or cfg.ct)
        self.round = int(round_)
        self.label_free = bool(label_free)
        self.seed, self.sym = int(seed), bool(sym)
        self.patch = ladder.shape3(cfg.patch)
        self.ctx = tuple(int(q) for q in cfg.ctx)
        self.channels = target_channels(cfg)
        self.need = tuple(need if need is not None else (self.channels[0],))
        self.windows = int(cfg.windows_per_region if windows is None else windows)
        self.heldout = list(heldout)
        self._ax_in, self._meta_in, self._records = ax, meta, region_records
        self.pyr = self.cat = self.ax = self.order = None
        # send only the target rows with support (`rung_item(compact=True)`); `loader` turns it on for
        # batch 1 -- items with different row sets would not collate
        self.compact = False
        # training draws leave empty the cascade extra the run's cascade mode never reads (`uses`)
        self.lean = True

    # ---- opening (in the worker, never in the parent) --------------------

    def _open(self):
        self.pyr = ladder.rungs(self.ct)
        self.ax = (np.asarray(self._ax_in, np.float64) if self._ax_in is not None
                   else AX.load(self.cfg.umbilicus, ct=self.ct))
        self.cat = RG.Catalog(self.root, self.round)
        if self._records is None:
            self._records = RG.region_list(self.pyr, rungs=self.cfg.rungs, patch=self.cfg.patch,
                                           region=self.cfg.region, boost=self.cfg.rung_boost,
                                           exclude=RG.exclude_boxes(self.heldout),
                                           occ_min_fine=self.cfg.occ_min_fine,
                                           occ_min_coarse=self.cfg.occ_min_coarse)
        self.visits = RG.region_visits(self._records, self.cfg.visits_max)
        self.order = RG.walk_order(self.visits, self.seed)
        kn = min(self.pyr)
        self.rmax_um = AX.rmax_vox(AX.axis_at(self.ax, kn), ladder.rung_shape(self.pyr, kn)) \
            * ladder.rung_um(kn)
        self.meta5 = (np.asarray(self._meta_in, np.float32) if self._meta_in is not None
                      else SM.scan_planes(SM.fetch(self.ct)))
        self.shape2 = ladder.rung_shape(self.pyr, 2)
        # held-out region stores never supply a training target, even as a stitched neighbour
        self._held = {tuple(int(v) for v in h["lo"]) for h in self.heldout
                      if int(h.get("k", 2)) == 2}
        # the rung-2 regions worth a visit of their own (occupancy >= occ_min_fine, not held out)
        self._fine2 = {tuple(int(v) for v in r["lo"]) for r in self._records if int(r["k"]) == 2}

    # ---- where a target comes from --------------------------------------

    def _region_of(self, k, lo, shape=None):
        """The rung-2 origin of the ONE region store covering this window, or None when it straddles
        two (or lies outside the volume). Rungs 2-6 only: above that the coarse array is the source."""
        d = int(k) - 2
        R = int(self.cfg.region)
        shape = self.patch if shape is None else ladder.shape3(shape)
        lo2 = np.asarray(lo, np.int64) << d
        hi2 = ((np.asarray(lo, np.int64) + shape) << d) - 1
        if (lo2 < 0).any():
            return None
        a, b = lo2 // R, hi2 // R
        return None if not np.array_equal(a, b) else a * R

    def _source(self, chan, k, lo, shape):
        """(cube, weight_fraction) of one channel over a window, whatever rung it is at. The weight
        fraction is `inside` at rungs 2-6 (a store either covers a voxel or does not) and the COVERAGE
        fraction at rungs 7-11 (a coarse voxel is as trustworthy as the share of it that was fed)."""
        shape = ladder.shape3(shape)
        k = int(k)
        if k >= RG.COARSE_RUNGS[0]:
            v, cov = RG.read_coarse(self.root, chan, k, lo, shape, self.round)
            held = getattr(self, "_held", ())
            if held:
                # the TRAINING sampler: a coarse voxel any part of which lies over a held-out region's
                # footprint carries no weight -- the coarse array folds EVERY produced region in, the
                # held-out references included, and training on them leaks the evaluation set. The
                # validation view (no held-out list) keeps them.
                cov = cov.copy()
                zero_footprints(cov, k, lo, held, int(self.cfg.region))
            return v, cov
        if chan in DIST_CHANNELS and 3 <= k <= DIST_MAX_RUNG:
            return self._dist_stitched(chan, k, lo, shape)
        if k >= 3:
            return self._stitched(chan, k, lo, shape)
        r = self._region_of(k, lo, shape)
        if r is None:
            return np.zeros(tuple(shape), np.uint8), np.zeros(tuple(shape), np.float32)
        a = self.cat.open(chan, r)
        if a is None:
            return np.zeros(tuple(shape), np.uint8), np.zeros(tuple(shape), np.float32)
        v, ins = stores.read_store(a, k, lo, shape)
        return v, ins.astype(np.float32)

    def _regions_over(self, k, lo, shape):
        """The rung-2 origins of every region whose footprint meets the rung-k window [lo, lo+shape)."""
        d, R = int(k) - 2, int(self.cfg.region)
        lo2 = np.maximum(np.asarray(lo, np.int64) << d, 0)
        hi2 = (np.asarray(lo, np.int64) + ladder.shape3(shape)) << d
        if (hi2 <= lo2).any():
            return []
        a, b = lo2 // R, (hi2 - 1) // R
        return [np.array([z, y, x], np.int64) * R for z in range(int(a[0]), int(b[0]) + 1)
                for y in range(int(a[1]), int(b[1]) + 1) for x in range(int(a[2]), int(b[2]) + 1)]

    def _stitched(self, chan, k, lo, shape):
        """Rungs 3-6: (cube, inside) with every voxel read from the region store it lies in. A region
        whose store is not finished, or that is held out, leaves its voxels at inside = 0."""
        k, d, R = int(k), int(k) - 2, int(self.cfg.region)
        shape = ladder.shape3(shape)
        lo = np.asarray(lo, np.int64)
        out = np.zeros(tuple(shape), np.uint8)
        ins = np.zeros(tuple(shape), np.float32)
        home = getattr(self, "_home_r", None)
        for r in self._regions_over(k, lo, shape):
            if tuple(int(v) for v in r) in getattr(self, "_held", ()):
                continue
            f0 = r >> d
            plo = np.maximum(lo, f0)
            phi = np.minimum(lo + shape, (r + R) >> d)
            if (phi <= plo).any():
                continue
            pn = phi - plo
            if k == 3 and home is not None and tuple(int(v) for v in r) == home:
                got = self._pool3(chan, r)     # the visit's own region: pooled once per visit
                if got is None:
                    continue
                v, i_ = _clip_read(got, plo - f0, pn)
            elif k == 3:
                a = self.cat.open(chan, r)
                if a is None:
                    continue
                v, i_ = stores.read_store(a, 3, plo, pn)
            else:
                v, i_ = RG.pooled_window(self.root, chan, r, k, plo, pn, self.round)
            o = plo - lo
            sl = tuple(slice(int(o[j]), int(o[j] + pn[j])) for j in range(3))
            out[sl] = v
            ins[sl] = i_
        return out, ins

    def _dist_stitched(self, kind, k, lo, shape):
        """Rungs 3-4 of a DISTANCE channel: (cube, inside) with every voxel read from its region's OWN
        rung-k field store (`targets.channel(kind, k)`: `midline_r3`, `thickness_r4`, ...), at that
        store's own rung and origin.

        A distance is in the voxels of the rung it was computed at, and its code is an offset encoding
        (code 0 = no data): the 2x mean pool `_stitched` takes of a probability store would read the
        RUNG-2 field, halve nothing, and average a valid code with a no-data 0 into a new "valid" code.
        So nothing here is ever pooled: a store at another rung, a missing store, a held-out region
        and the padding past the store's `shape_true` all leave their voxels at inside = 0."""
        from rvsm import targets as TG
        k, d, R = int(k), int(k) - 2, int(self.cfg.region)
        shape = ladder.shape3(shape)
        lo = np.asarray(lo, np.int64)
        out = np.zeros(tuple(shape), np.uint8)
        ins = np.zeros(tuple(shape), np.float32)
        chan = TG.channel(kind, k)
        for r in self._regions_over(k, lo, shape):
            if tuple(int(v) for v in r) in getattr(self, "_held", ()):
                continue
            a = self.cat.open(chan, r)
            if a is None or int(a.attrs.get("rung", -1)) != k:
                continue
            o = np.asarray(a.attrs["origin_zyx"], np.int64)
            st = a.attrs.get("shape_true")
            top = o + (np.asarray(st, np.int64) if st else np.asarray(a.shape[-3:], np.int64))
            plo = np.maximum(lo, r >> d)
            phi = np.minimum(np.minimum(lo + shape, (r + R) >> d), top)
            if (phi <= plo).any():
                continue
            pn = phi - plo
            v, i_ = stores.read_store(a, k, plo, pn)
            q = plo - lo
            sl = tuple(slice(int(q[j]), int(q[j] + pn[j])) for j in range(3))
            out[sl] = v
            ins[sl] = i_
        return out, ins

    def _pool3(self, chan, r):
        """The whole region store of `chan` read at rung 3 (its 2x mean pool), kept for the visit.

        A rung-3 window reads 512^3 voxels of the rung-2 store to pool them, so every window of a
        visit re-decoded ~8x its own size; the whole region pooled once is 128 MB a channel. Only the
        current region's pools are kept (a new region drops them), so a worker holds at most one region's
        rung-3 pools, one visit's context super-cubes (`_ctx_cube`) and `regions.POOL_BYTES` of rung 4-6
        pools at any time."""
        # the store's PATH is part of the key: a committed regeneration is a different store
        key = (str(chan), tuple(int(v) for v in r), self.cat.path(chan, r))
        if getattr(self, "_p3key", None) != key[1]:
            self._p3key, self._p3 = key[1], {}
        if key not in self._p3:
            a = self.cat.open(chan, r)
            if a is None:
                return None
            # in z-slabs: reading the 1 GB store whole to pool it was a 3-4 GB transient per worker
            self._p3[key] = RG.pool_store(a, 1)
        return self._p3[key]

    def _near_axis(self, k, lo, shape):
        """Voxels within `AXIS_R_UM` of the umbilicus: the core, where recto and verso are the same
        sheet seen twice and a distance to a midline means nothing."""
        shape = tuple(int(v) for v in ladder.shape3(shape))
        a = AX.axis_at(self.ax, k)
        z = np.arange(shape[0]) + int(lo[0])
        cy, cx = np.interp(z, a[0], a[1]), np.interp(z, a[0], a[2])
        r2 = (AXIS_R_UM / ladder.rung_um(k)) ** 2
        ys, xs = np.arange(shape[1]) + int(lo[1]), np.arange(shape[2]) + int(lo[2])
        # the nearest point of each z-slice's (y, x) box to the axis: when even that is outside the
        # radius at every z -- nearly every window of a scroll -- no voxel is near, and the 16 M-voxel
        # float64 test below is skipped (it was a large share of a draw on the trainer's workers)
        ny = np.clip(cy, ys[0], ys[-1]) - cy
        nx = np.clip(cx, xs[0], xs[-1]) - cx
        if bool(((ny * ny + nx * nx) >= r2).all()):
            return np.zeros(shape, bool)
        dy = ys[None, :, None] - cy[:, None, None]
        dx = xs[None, None, :] - cx[:, None, None]
        return (dy * dy + dx * dx) < r2

    def _rung_target(self, k, lo, ct):
        """(target, weight) at rung k as uint8 (255 = 1.0), one row per channel of `target_channels`.

        The weight is `rint(255 * fraction * air * [code != 0] * ~near * rw)`. It is computed in
        uint8 and bool, not float64: a store's fraction is 0/1 at rungs 2-6 (so `255 * fraction * rw`
        IS the rw code) and the coarse coverage is quantised once. Same bytes as the float formula
        (test), a fraction of the time -- the float64 temporaries were ~800 ms of a rung-2 draw."""
        p = tuple(int(v) for v in self.patch)
        tg = np.zeros((len(self.channels),) + p, np.uint8)
        w = np.zeros_like(tg)
        if self.label_free:
            return tg, w
        air = ct > 0
        near = None
        rw8 = None
        if int(k) <= 6:
            # the agreement weight where an rw store covers the voxel, full weight where none does
            rv, rf = self._source(RW, k, lo, p)
            if (rf > 0).any():
                rw8 = np.where(rf > 0, rv, np.uint8(255))
        for c, chan in enumerate(self.channels):
            dist = chan in DIST_CHANNELS
            if dist and int(k) > DIST_MAX_RUNG:
                continue                      # a distance is never pooled: no target here, so weight 0
            v, frac = self._source(chan, k, lo, p)
            np.copyto(tg[c], v, where=air)    # masked CT: air carries no surface
            m = air.copy()
            if dist:
                m &= v != 0                   # code 0 IS the store's no-data marker
            if chan in NEAR_AXIS_ZERO:
                near = self._near_axis(k, lo, p) if near is None else near
                m &= ~near
            if int(k) <= 6:                   # a fraction of 0 or 1: the weight is 255, or the rw code
                m &= frac > 0
                if rw8 is not None and not dist:
                    np.copyto(w[c], rw8, where=m)
                else:
                    w[c][m] = 255
            else:                             # the coarse coverage fraction, quantised once
                q = np.rint(frac * np.float32(255.0))
                np.copyto(w[c], np.clip(q, 0, 255).astype(np.uint8), where=m)
        return tg, w

    def _target_block(self, chan, k, lo, shape):
        """Just the field of one channel over a window: what the cascade's `cm` extra is."""
        if self.label_free or int(k) >= ladder.NRUNGS:
            return np.zeros(tuple(int(v) for v in ladder.shape3(shape)), np.uint8)
        return self._source(chan, k, lo, shape)[0]

    # ---- the extras ------------------------------------------------------

    def _cascade_extras(self, k, lo, shape, rec=None, lean=False):
        """`cm` (the rung-(k+1) target block over the patch footprint, half the patch on every axis) and,
        for the self/mix cascade modes, the tenth context cube `cx` and the corner `lo1` of the coarse
        cube. The top rung has no rung above it, so its cascade block is zero -- which is also what
        `cascade_drop` teaches the model to expect. `lean=True` (a training draw) leaves EMPTY the one
        of the two this run's cascade mode never reads (`uses`); the validation grid always has both."""
        p, lo = ladder.shape3(shape), np.asarray(lo, np.int64)
        hp = np.maximum(p // 2, 1)
        use_cm, use_cx = self.uses() if lean else (True, True)
        cm = self._target_block(self.channels[0], int(k) + 1, lo // 2, hp) if use_cm else \
            np.zeros((0, 0, 0), np.uint8)
        d = (int(self.ctx[-1]) + 1) if self.ctx else 1
        c0 = lo + p // 2
        if not use_cx:
            cx = np.zeros((1, 0, 0, 0), np.uint8)
        elif rec is not None:
            cx = self._ctx_cube(rec, k, d, lo, p)
        else:
            cx = ladder.read_rung(self.pyr, int(k) + d, c0 // (1 << d) - p // 2, p, dtype=np.uint8)
        return {"cm": cm, "cx": cx, "lo1": c0 // 2 - p // 2}

    def uses(self):
        """(cm, cx): whether this run's training cascade reads the coarse target block (mask, mix) and
        the tenth context cube (self, mix). A label-free draw feeds no cascade at all. `lean` False
        sends both whatever the mode."""
        if not self.lean:
            return True, True
        if self.label_free:
            return False, False
        mode = str(getattr(self.cfg, "cascade", "self") or "off")
        return mode in ("mask", "mix"), mode in ("self", "mix")

    SUPER_MAX = 128 << 20    # a visit's context super-cube is cached when it is at most this many voxels

    def _ctx_cube(self, rec, k, d, lo, p):
        """The rung-(k+d) context cube of the window at `lo` -- `ladder.context`'s cube, bit for bit --
        sliced from ONE super-cube per visit covering every window centre the visit can draw.

        A visit draws `windows_per_region` windows in one tile, and above d = 1 their context cubes
        overlap almost entirely (at d >= 3 the whole tile is smaller than one cube): reading nine cubes
        per window re-decoded the same coarse chunks ~128 times and was 2.4 s of a 4.1 s draw on
        tnr-0. A super-cube larger than `SUPER_MAX` voxels (d = 1: 640^3 for a 1024 tile) is not
        cached; that cube is read per window as before."""
        key = (int(rec["k"]), tuple(int(v) for v in rec["lo"]), rec.get("v"))
        if getattr(self, "_vkey", None) != key:
            self._vkey, self._vcubes = key, {}
        p3 = ladder.shape3(p)
        c0 = np.asarray(lo, np.int64) + p3 // 2
        if d not in self._vcubes:
            rlo, rsz = np.array(rec["lo"], np.int64), np.array(rec["size"], np.int64)
            hi = np.maximum(rlo + rsz - p3, rlo)
            cmin, cmax = np.minimum(rlo, hi) + p3 // 2, hi + p3 // 2
            a = cmin // (1 << int(d)) - p3 // 2
            b = cmax // (1 << int(d)) - p3 // 2 + p3
            self._vcubes[d] = None if int(np.prod(b - a)) > self.SUPER_MAX else \
                (a, ladder.read_rung(self.pyr, int(k) + int(d), a, b - a, dtype=np.uint8))
        ent = self._vcubes[d]
        lo_d = c0 // (1 << int(d)) - p3 // 2
        if ent is None:
            return ladder.read_rung(self.pyr, int(k) + int(d), lo_d, p3, dtype=np.uint8)
        a, cube = ent
        o = lo_d - a
        return cube[o[0]:o[0] + p3[0], o[1]:o[1] + p3[1], o[2]:o[2] + p3[2]]

    def _plane_extras(self, k):
        """The per-sample numbers the radius and scan planes need: r_max at this rung, and the five
        normalised scan values (the same for every sample of a run, read once)."""
        return {"rmax": float(self.rmax_um) / ladder.rung_um(int(k)), "meta": self.meta5}

    # ---- drawing ---------------------------------------------------------

    def _home(self, rec):
        """The rung-2 origin of the region a visit's producer job makes: the region holding the record's
        corner (`run.region_route`'s rule)."""
        d, R = max(int(rec["k"]) - 2, 0), int(self.cfg.region)
        return tuple(int(v) // R * R for v in (np.array(rec["lo"], np.int64) << d))

    def _draw_rec(self, rec):
        """The box a visit's windows are drawn from, as a record (`lo`, `size`; `_ctx_cube` keys and
        sizes its super-cube on it). Rung 2 and the coarse rungs: the record itself. Rungs 3-6: around
        the HOME region's rung-k footprint -- wholly inside it when the footprint is larger than a
        window (rung 3), otherwise with the window centre inside it (rungs 4-6), so every window
        overlaps the store the producer made for this visit."""
        k = int(rec["k"])
        if not 3 <= k <= 6 or self.label_free:
            return rec
        d, p = k - 2, ladder.shape3(self.patch).astype(np.int64)
        f0 = np.array(self._home(rec), np.int64) >> d
        fs = np.full(3, max(int(self.cfg.region) >> d, 1), np.int64)
        lo_a = np.where(fs > p, f0, f0 - p // 2)
        lo_b = np.where(fs > p, f0 + fs - p, f0 + fs - 1 - p // 2)
        top = np.maximum(np.asarray(ladder.rung_shape(self.pyr, k), np.int64) - p, 0)
        lo_a, lo_b = np.clip(lo_a, 0, top), np.clip(lo_b, 0, top)
        lo_b = np.maximum(lo_a, lo_b)
        return dict(rec, lo=[int(v) for v in lo_a], size=[int(v) for v in lo_b - lo_a + p])

    def air_budget(self):
        """How many CT-air windows one visit may keep: `air_keep` of its windows (at least one).

        `air_keep` keeps that share of the air windows DRAWN, and a visit draws until it has
        `windows` of them -- so a home region that is nearly all air filled its whole visit with
        weight-0 air windows (paris4 steps 780-1140: 128 in a row from one rung-3 visit). Capped per
        visit, air stays about `air_keep` of the items and a mostly-air visit ends early instead."""
        return max(int(round(float(self.cfg.air_keep) * self.windows)), 1)

    def _draw(self, rng, rec, air_ok=True):
        """One window inside one region, or None when the CT-air rule rejects it (always, once the
        visit's air budget is spent: `air_ok=False`). `self._last_air` says whether a kept window
        was an air one."""
        k = int(rec["k"])
        if 3 <= k <= 6:
            self._home_r = self._home(rec)
            rec = self._draw_rec(rec)
        p = self.patch
        rlo, rsz = np.array(rec["lo"], np.int64), np.array(rec["size"], np.int64)
        hi = np.maximum(rlo + rsz - p, rlo)
        lo = rng.integers(np.minimum(rlo, hi), hi + 1)
        ct = ladder.read_rung(self.pyr, k, lo, p, dtype=np.uint8)
        air = bool((ct == 0).mean() > 0.9)
        if air and (not air_ok or rng.random() > self.cfg.air_keep):
            return None
        self._last_air = air
        tg, w = self._rung_target(k, lo, ct)
        sym = int(draw_sym(rng, tuple(p))) if self.sym else 0
        cx = [self._ctx_cube(rec, k, d, lo, ct.shape) for d in self.ctx] if self.ctx else ()
        return rung_item(np.stack([ct] + list(cx)), tg, w, k, lo, self.ax, sym, compact=self.compact,
                         **self._plane_extras(k), **self._cascade_extras(k, lo, ct.shape, rec=rec, lean=True))

    def _dead(self, rec):
        """A rung 3-6 visit whose HOME region is held out, or is not a rung-2 region of the walk (its
        occupancy is under `occ_min_fine`: a coarse tile passes the occupancy test on the MEAN of all
        its regions, but its windows are drawn on the home region alone). Every window it could draw
        is weight 0 -- a held-out store is never a target, and an air home yields 128 `air_keep`
        windows (paris4: 8 % of the items of the first 2000 steps) -- and a rung-5/6 record is
        visited up to `visits_max` times. The walk skips it for good; `run.region_route` does not
        produce its home either."""
        k = int(rec["k"])
        if not 3 <= k <= 6 or self.label_free:
            return False
        h = self._home(rec)
        fine = getattr(self, "_fine2", None)
        return h in getattr(self, "_held", ()) or bool(fine) and h not in fine

    def _visitable(self, rec):
        """Is this visit worth making? A fine region with no finished store would yield only zero
        weights, so the walk steps over it (and picks it up on a later pass, once the producer has been
        there). Coarse rungs are always visitable: coverage decides per voxel."""
        if self.label_free or int(rec["k"]) >= RG.COARSE_RUNGS[0]:
            return True
        r = self._region_of(int(rec["k"]), np.array(rec["lo"], np.int64), rec["size"])
        if r is None:
            r = (np.array(rec["lo"], np.int64) << (int(rec["k"]) - 2)) // self.cfg.region \
                * self.cfg.region
        return self.cat.ready(r, self.need)

    def __iter__(self):
        if self.pyr is None:
            self._open()
        info = torch.utils.data.get_worker_info()
        w, W = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng(self.seed + 1000 * w)
        mine = [int(i) for i in self.order[w::W]] or [int(i) for i in self.order]
        while True:
            served = 0
            for i in mine:
                rec = self.visits[i]
                if self._dead(rec) or not self._visitable(rec):
                    continue
                left, fails, air = self.windows, 0, self.air_budget()
                while left > 0 and fails < 8 * max(self.windows, 1):
                    got = self._draw(rng, rec, air_ok=air > 0)
                    if got is None:
                        fails += 1
                        continue
                    air -= int(self._last_air)
                    left, fails, served = left - 1, 0, served + 1
                    yield got
            if not served:   # nothing is produced yet: do not spin the CPU on an empty walk
                import time
                time.sleep(1.0)


# --------------------------------------------------------------------------- validation and loading

def zero_footprints(cov, k, lo, held, region):
    """Zero `cov` (a rung-k window at `lo`) over the rung-k footprint of every held-out rung-2 region
    origin in `held` (each `region` voxels on a side at rung 2), rounded OUTWARD: a coarse voxel that
    touches a footprint at all is zeroed."""
    d = int(k) - 2
    lo = np.asarray(lo, np.int64)
    shp = np.array(cov.shape, np.int64)
    for h in held:
        h2 = np.asarray(h, np.int64)
        a = (h2 >> d) - lo                                   # floor
        b = (-((-(h2 + int(region))) >> d)) - lo             # ceil
        a, b = np.maximum(a, 0), np.minimum(b, shp)
        if (b > a).all():
            cov[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = 0
    return cov


def _grid_corners(cfg, heldout, p, rungs=None, limit=8):
    """[(k, lo)] of the validation grid, in build order: non-overlapping tiles of each held-out region
    at each rung, evenly subsampled to `limit` per (region, rung)."""
    return [(k, lo) for k, lo, _ in _grid_corners_by_region(cfg, heldout, p, rungs, limit)]


def _grid_corners_by_region(cfg, heldout, p, rungs=None, limit=8):
    """`_grid_corners` with the index into `heldout` of the region each corner tiles."""
    out = []
    for ri, r in enumerate(heldout):
        lo2, sz2 = np.array(r["lo"], np.int64), np.array(r["size"], np.int64)
        for k in (cfg.rungs if rungs is None else rungs):
            d = int(k) - 2
            org, sz = (lo2 >> d, np.maximum(sz2 >> d, 1)) if d >= 0 else (lo2 << -d, sz2 << -d)
            cs = [(z, y, x) for z in range(0, max(int(sz[0]) - int(p[0]), 0) + 1, int(p[0]))
                  for y in range(0, max(int(sz[1]) - int(p[1]), 0) + 1, int(p[1]))
                  for x in range(0, max(int(sz[2]) - int(p[2]), 0) + 1, int(p[2]))]
            if limit and len(cs) > limit:
                cs = [cs[i] for i in np.linspace(0, len(cs) - 1, limit).astype(int)]
            out += [(int(k), org + np.array(c, np.int64), ri) for c in cs]
    return out


GRID_SCHEMA = "grid-v2"   # bump when what a grid item holds changes


def grid_heads(ds, heldout):
    """The probability heads the held-out regions carry targets for: ("recto",), plus "verso" once
    EVERY held-out region has a finished verso store."""
    heads = ["recto"]
    if heldout and all(ds.cat.done("verso", tuple(int(v) for v in h["lo"])) for h in heldout):
        heads.append("verso")
    return tuple(heads)


def grid_sources(ds, heldout):
    """The identity of every label store the held-out grid reads, AS A READER SEES IT (committed
    generations): a regenerated verso / its fields, or any rewritten store, changes the grid key
    (pass-4 P4-04: changed labels returned the cached targets)."""
    from rvsm import targets as TG
    chans = sorted({str(c) for c in ds.channels} | {"rw"} |
                   {TG.channel(kind, k) for kind in TG.KINDS for k in (2, 3, 4)})
    out = []
    for h in heldout:
        lo = tuple(int(v) for v in h["lo"])
        out.append([list(lo)] + [TG.source_digest(ds.cat.path(c, lo)) for c in chans])
    return out


def grid_global(cfg, round_, heads):
    """The part of a grid item's identity that every item shares: the schema and target-definition
    versions, the store round, the heads with targets and the config fields an item is built from
    (`grid_key` without the corners and the per-region sources)."""
    from rvsm import targets as TG
    return {"round": int(round_), "heads": list(heads), "schema": GRID_SCHEMA, "target_def": TG.TARGET_DEF,
            "keys": [q for q in RUNG_ITEM_KEYS if q != "tch"], "patch": int(cfg.patch),
            "ctx": [int(v) for v in cfg.ctx], "channels": [str(c) for c in cfg.channels],
            "planes": str(cfg.planes), "rungs": [int(v) for v in cfg.rungs]}


def _item_name(glob, source, k, lo):
    """A grid item's file name: a digest of everything the item is built from (the global part, its
    region's source entry, its rung and corner). An item whose inputs changed gets a NEW name, so a
    rebuild never overwrites a file a live `DiskGrid` may be reading."""
    import hashlib
    import json
    h = hashlib.sha256(json.dumps({"g": glob, "src": source, "k": int(k), "lo": [int(v) for v in lo]},
                                  sort_keys=True).encode()).hexdigest()[:20]
    return f"item_r{int(k)}_{h}.pt"


def grid_key(cfg, corners, round_, heads, sources=None):
    """The identity of a validation grid: the corners, the store round, which heads have targets, the
    schema and target-definition versions and the config fields an item is built from -- NOT the whole
    config fingerprint, which moves whenever an unrelated field joins its exclusions and rebuilt the
    grid (12 min) for nothing."""
    import hashlib
    import json
    from rvsm import targets as TG
    return hashlib.sha256(json.dumps({
        "corners": [[k, [int(v) for v in lo]] for k, lo in corners], "round": int(round_),
        "heads": list(heads), "schema": GRID_SCHEMA, "target_def": TG.TARGET_DEF,
        # `tch` is how the rows travel, not what they hold: a grid item is always full, so it stays out of
        # the key and a grid built before it existed is reused (`prep.expand_tw` reads its absence as full)
        "keys": [q for q in RUNG_ITEM_KEYS if q != "tch"], "patch": int(cfg.patch), "ctx": [int(v) for v in cfg.ctx],
        "channels": [str(c) for c in cfg.channels], "planes": str(cfg.planes),
        "rungs": [int(v) for v in cfg.rungs], "sources": sources or []},
        sort_keys=True).encode()).hexdigest()[:16]


def grid_dir(spill, heads):
    """Where the grid for `heads` lives: the recto-only grid in `spill` itself, a grid with more heads
    in its own sibling generation (`<spill>_recto+verso`), so the recto reference grid stays exactly
    as it was when the verso targets appear."""
    heads = tuple(heads)
    return spill if heads == ("recto",) else f"{str(spill).rstrip('/')}_{'+'.join(heads)}"


def val_grid(cfg, heldout, root=None, ct=None, ax=None, round_=0, rungs=None, limit=8, spill=None,
             threads=1, **kw):
    """A FIXED set of windows over the held-out regions: the same corners at every evaluation, so two
    checkpoints are compared on identical data. Non-overlapping tiles of each held-out region at each
    rung the region can supply, evenly subsampled to `limit` per (region, rung).

    A grid item is ~310 MB (the CT and nine context cubes at 256^3 are 160 MB of it), and eight held-out
    regions make ~190 items: ~60 GB, which is the whole host on a 64 GB machine. With `spill=<dir>` the
    grid is written there compressed as it is built and returned as a `DiskGrid`, which holds ONE item
    in memory at a time. The directory is reused ITEM BY ITEM: `grid.json` keeps a record per item (its
    file, rung, corner, region, that region's `grid_sources` entry and the global part, `grid_global`),
    and only the items of a region whose source changed -- or that are missing -- are rebuilt, under new
    file names (`_item_name`), before the manifest is replaced atomically and the files no manifest
    references are deleted. So a restart rebuilds nothing, and one held-out region's committed fields
    bundle rebuilds that region's items only. The item order is always the corner order, and an item
    is byte-for-byte what a from-scratch build writes for the same sources. The returned `DiskGrid`
    carries `rebuilt` / `reused` item counts.

    Once every held-out region has a verso store, the grid with the verso targets is a NEW generation in
    its own directory (`grid_dir`); the recto grid is untouched. `threads` builds items concurrently
    (the reads are volcomp decodes, which run without the GIL); the item order, and so the grid, does
    not depend on it."""
    ds = Patches(cfg, root=root, ct=ct, ax=ax, round_=round_, region_records=list(heldout),
                 sym=False, **kw)
    ds._open()
    p = ds.patch
    corners = _grid_corners(cfg, heldout, p, rungs=rungs, limit=limit)
    assert corners, "the held-out regions are smaller than one patch at every rung"

    def build(kl):
        k, lo = kl
        ct_ = ladder.read_rung(ds.pyr, int(k), lo, p, dtype=np.uint8)
        tg, w = ds._rung_target(int(k), lo, ct_)
        cx = ladder.context(ds.ct, lo, ct_.shape, ds.ctx, rung=int(k)) if ds.ctx else ()
        return rung_item(np.stack([ct_] + list(cx)), tg, w, int(k), lo, ds.ax, 0,
                         **ds._plane_extras(int(k)), **ds._cascade_extras(int(k), lo, ct_.shape))

    def items():
        yield from _bounded_map(build, corners, threads)

    if spill is None:
        return list(items())
    import json
    import os
    heads = grid_heads(ds, heldout)
    sources = grid_sources(ds, heldout)
    key = grid_key(cfg, corners, round_, heads, sources)
    glob = grid_global(cfg, round_, heads)
    spill = grid_dir(spill, heads)
    os.makedirs(spill, exist_ok=True)
    man = os.path.join(spill, "grid.json")
    want = []                                   # one record per corner, in corner (build) order
    for k, lo, ri in _grid_corners_by_region(cfg, heldout, p, rungs=rungs, limit=limit):
        src = sources[ri]
        want.append({"file": _item_name(glob, src, k, lo), "rung": int(k), "lo": [int(v) for v in lo],
                     "region": [int(v) for v in heldout[ri]["lo"]], "source": src, "global": glob})
    old = {}
    try:
        with open(man) as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = {}
    # an item is reused when the manifest lists it with the same global part and region source (its
    # name is a digest of both, plus rung and corner) and its file is there; an old-format manifest
    # (key + items only) has no records, so everything is rebuilt, under the new names
    have = {}
    for r in old.get("records") or []:
        try:
            if os.path.exists(os.path.join(spill, r["file"])):
                have[r["file"]] = r
        except (TypeError, KeyError):
            continue
    names = [w["file"] for w in want]
    todo = [i for i, w in enumerate(want)
            if not (w["file"] in have and have[w["file"]].get("global") == w["global"]
                    and have[w["file"]].get("source") == w["source"])]
    if not todo and old.get("items") == names and old.get("key") == key:
        _drop_orphans(spill, names)             # a crash between the manifest and the cleanup
        g = DiskGrid(spill, names)
        g.rebuilt, g.reused = 0, len(names)
        return g
    built = set()
    todo_c = [(want[i]["rung"], np.asarray(want[i]["lo"], np.int64)) for i in todo]
    for i, it in zip(todo, _bounded_map(build, todo_c, threads)):
        name = want[i]["file"]
        if name not in built:                   # two held-out regions sharing a tile: one file
            _save_item(os.path.join(spill, name), it)
            built.add(name)
        del it
    with open(man + ".tmp", "w") as f:
        json.dump({"key": key, "items": names, "n": len(names), "records": want}, f)
    os.replace(man + ".tmp", man)
    _drop_orphans(spill, names)
    g = DiskGrid(spill, names)
    g.rebuilt, g.reused = len(built), len(names) - len(todo)
    return g


def _bounded_map(fn, xs, threads):
    """`fn` over `xs` in order, on `threads` threads, with at most `threads` results built ahead of the
    consumer. `ThreadPoolExecutor.map` submits EVERYTHING at once and holds each finished result until
    it is taken, so with a consumer slower than the builders (compressing and writing a ~310 MB item)
    the whole grid could pile up in memory; here at most 2 * `threads` items exist at a time (building
    or waiting) plus the one being written."""
    n = max(int(threads), 1)
    if n <= 1 or len(xs) < 2:
        for x in xs:
            yield fn(x)
        return
    import collections
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(n) as ex:
        q = collections.deque()
        it = iter(xs)
        for x in it:
            q.append(ex.submit(fn, x))
            if len(q) >= 2 * n:
                break
        end = object()
        while q:
            r = q.popleft().result()
            nxt = next(it, end)
            if nxt is not end:
                q.append(ex.submit(fn, nxt))
            yield r
            del r


def _drop_orphans(spill, names):
    """Delete the item files in `spill` the manifest (`names`) no longer references."""
    import os
    keep = set(names)
    for fn in os.listdir(spill):
        if fn.startswith("item_") and (fn.endswith(".pt") or fn.endswith(".pt.tmp")) and fn not in keep:
            try:
                os.remove(os.path.join(spill, fn))
            except OSError:
                pass


_PACK_MIN = 1 << 20      # arrays at least this big are compressed on disk


def _save_item(path, item):
    """One grid item to disk: every large array Blosc-zstd compressed (the targets and weights are
    mostly zeros; the CT compresses less), everything else as it is."""
    import numcodecs
    codec = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    rec = {}
    for k, v in item.items():
        a = v.numpy() if torch.is_tensor(v) else v
        if isinstance(a, np.ndarray) and a.nbytes >= _PACK_MIN:
            rec[k] = ("blosc", codec.encode(np.ascontiguousarray(a)), a.shape, str(a.dtype),
                      torch.is_tensor(v))
        else:
            rec[k] = ("raw", v)
    torch.save(rec, path + ".tmp")
    import os
    os.replace(path + ".tmp", path)


def _load_item(path):
    import numcodecs
    codec = numcodecs.Blosc()
    rec = torch.load(path, weights_only=False)
    out = {}
    for k, v in rec.items():
        if v[0] == "blosc":
            _, buf, shape, dt, is_t = v
            a = np.frombuffer(codec.decode(buf), dtype=np.dtype(dt)).reshape(shape).copy()
            out[k] = torch.from_numpy(a) if is_t else a
        else:
            out[k] = v[1]
    return out


class DiskGrid:
    """The validation grid on disk (`val_grid(spill=...)`), used like the list it replaces: `len`,
    indexing, slicing and iteration. Iteration decodes the next `ahead` items on background threads
    while the caller scores the current one (`prefetched`): an item is a ~310 MB compressed file, and
    its read + decode, not the forward, set the pace of an evaluation. At most `ahead + 1` items are
    held by the iterator at a time. A slice or `subset` is a DiskGrid too (lazy, prefetched)."""

    AHEAD = 2           # items decoded ahead of the one being scored

    def __init__(self, root, names, ahead=None):
        import os
        self.paths = [os.path.join(root, n) for n in names]
        self.rebuilt, self.reused = len(self.paths), 0      # `val_grid` sets what it actually did
        self.ahead = self.AHEAD if ahead is None else int(ahead)

    def __len__(self):
        return len(self.paths)

    def __bool__(self):
        return bool(self.paths)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.subset(range(len(self.paths))[i])
        return _load_item(self.paths[i])

    def subset(self, idx):
        """The items at these indices, in this order, as a DiskGrid (nothing is read until iterated)."""
        g = DiskGrid("", [], ahead=self.ahead)
        g.paths = [self.paths[int(i)] for i in idx]
        g.rebuilt, g.reused = 0, len(g.paths)
        return g

    def rungs(self):
        """Each item's rung, from its file name (`_item_name`: item_r<k>_<digest>.pt); None for a name
        that does not carry one."""
        import os
        import re
        out = []
        for p in self.paths:
            m = re.match(r"item_r(\d+)_", os.path.basename(p))
            out.append(int(m.group(1)) if m else None)
        return out

    def of_rungs(self, rungs):
        """The items of these rungs (a subset, never read to find out), or self when a name has none."""
        ks = self.rungs()
        if any(k is None for k in ks):
            return self
        want = {int(k) for k in rungs}
        return self.subset([i for i, k in enumerate(ks) if k in want])

    def __iter__(self):
        # a snapshot of the paths: the driver's `refresh_grid` may replace them in place between two
        # evaluations, never in the middle of one
        return prefetched(list(self.paths), _load_item, self.ahead)


def prefetched(xs, fn, ahead=2):
    """`fn(x)` for every x, in order, with up to `ahead` of the next results computed on `ahead`
    background threads while the caller holds the current one. A consumer that stops early cancels the
    loads that have not started (the ones running finish and are dropped). `ahead` 0: in the caller."""
    ahead = max(int(ahead), 0)
    if not ahead or len(xs) < 2:
        for x in xs:
            yield fn(x)
        return
    import collections
    import concurrent.futures as cf
    ex = cf.ThreadPoolExecutor(ahead, thread_name_prefix="rvsm-grid")
    q = collections.deque()
    try:
        nxt = 0
        while nxt < len(xs) and len(q) < ahead:
            q.append(ex.submit(fn, xs[nxt]))
            nxt += 1
        while q:
            cur = q.popleft().result()
            if nxt < len(xs):
                q.append(ex.submit(fn, xs[nxt]))
                nxt += 1
            yield cur
            del cur
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def loader(patches, workers=0, batch=1, pin_memory=True):
    """Workers start as fresh processes (forkserver), never forks: the parent has usually opened zarr
    already, and its asyncio loop thread does not survive a fork -- which breaks streamed reads in a way
    that only shows up minutes later.

    At batch 1 a `Patches` sends compact items (only the target rows with support, `rung_item`'s `tch`);
    at a larger batch every item carries all T rows, because two items with different row sets do not
    collate."""
    if isinstance(patches, Patches):
        patches.compact = int(batch) == 1
    return torch.utils.data.DataLoader(patches, batch_size=batch, num_workers=int(workers),
                                       pin_memory=bool(pin_memory), persistent_workers=workers > 0,
                                       prefetch_factor=2 if workers else None,
                                       multiprocessing_context="forkserver" if workers else None)
