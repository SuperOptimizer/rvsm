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
    rungs 7-11  the whole-scroll coarse array x its coverage  (`regions.read_coarse`)

and a window that straddles two regions at rungs 2-6 is simply skipped for that channel (weight 0),
because a store is a per-region object and stitching two of them would invent a seam.

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

def rung_item(ct, tg, w, k, lo, ax, sym=0, norm=None, cm=None, cx=None, lo1=None, rmax=0.0, meta=None):
    """The compact sample the loader yields: everything uint8, so a 256^3 sample is ~200 MB instead of
    the ~1 GB of float32 a built model input would be. It carries EXACTLY `config.RUNG_ITEM_KEYS`:

    ct   (1 + nctx, Z, Y, X) uint8: the CT cube and the context cubes as read, not z-scored
    tgt  (T, Z, Y, X) uint8: the target fields (recto, verso, midline, thickness), CT-air zeroed
    w    (T, Z, Y, X) uint8: the per-voxel weight, 255 = 1.0
    lo   (3,) int64: the corner, in rung-k voxels
    cyx  (2, Z) float64: the scroll axis (y, x) at each z of the cube -- what `radial_t` interpolates
    sym  (): the cube symmetry index drawn by the worker (0 = identity), applied on the GPU
    rung (), norm (2,): the rung k and the (mean, std) of the z-score (std 0 = per-patch)
    cm   (Z/2, Y/2, X/2) uint8: the rung-(k+1) target block over the patch footprint -- the `mask` source
         of the cascade channel, upsampled 2x on the GPU
    cx   (1, Z, Y, X) uint8: the TENTH context cube (rung k + ctx[-1] + 1), the one extra cube the
         rung-(k+1) input needs that the rung-k input does not
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
    return {"ct": torch.from_numpy(ct),
            "tgt": torch.from_numpy(np.ascontiguousarray(tg)),
            "w": torch.from_numpy(np.ascontiguousarray(w)),
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
            return v, cov
        r = self._region_of(k, lo, shape)
        if r is None:
            return np.zeros(tuple(shape), np.uint8), np.zeros(tuple(shape), np.float32)
        if k <= 3:
            a = self.cat.open(chan, r)
            if a is None:
                return np.zeros(tuple(shape), np.uint8), np.zeros(tuple(shape), np.float32)
            v, ins = stores.read_store(a, k, lo, shape)
        else:
            v, ins = RG.pooled_window(self.root, chan, r, k, lo, shape, self.round)
        return v, ins.astype(np.float32)

    def _near_axis(self, k, lo, shape):
        """Voxels within `AXIS_R_UM` of the umbilicus: the core, where recto and verso are the same
        sheet seen twice and a distance to a midline means nothing."""
        shape = tuple(int(v) for v in ladder.shape3(shape))
        a = AX.axis_at(self.ax, k)
        z = np.arange(shape[0]) + int(lo[0])
        cy, cx = np.interp(z, a[0], a[1]), np.interp(z, a[0], a[2])
        dy = (np.arange(shape[1]) + int(lo[1]))[None, :, None] - cy[:, None, None]
        dx = (np.arange(shape[2]) + int(lo[2]))[None, None, :] - cx[:, None, None]
        return (dy * dy + dx * dx) < (AXIS_R_UM / ladder.rung_um(k)) ** 2

    def _rung_target(self, k, lo, ct):
        """(target, weight) at rung k as uint8 (255 = 1.0), one row per channel of `target_channels`."""
        p = tuple(int(v) for v in self.patch)
        tg = np.zeros((len(self.channels),) + p, np.uint8)
        w = np.zeros_like(tg)
        if self.label_free:
            return tg, w
        air = ct > 0
        near = None
        rwv = None
        r = self._region_of(k, lo)
        if r is not None and int(k) <= 6 and self.cat.done(RW, r):
            rwv = self._source(RW, k, lo, p)[0].astype(np.float32) / 255.0
        for c, chan in enumerate(self.channels):
            dist = chan in DIST_CHANNELS
            if dist and int(k) > DIST_MAX_RUNG:
                continue                      # a distance is never pooled: no target here, so weight 0
            v, frac = self._source(chan, k, lo, p)
            np.copyto(tg[c], v, where=air)    # masked CT: air carries no surface
            ok = frac * air
            if dist:
                ok = ok * (v != 0)            # code 0 IS the store's no-data marker
            if chan in NEAR_AXIS_ZERO:
                near = self._near_axis(k, lo, p) if near is None else near
                ok = ok * ~near
            if rwv is not None and not dist:
                ok = ok * rwv
            w[c] = np.clip(np.rint(255.0 * ok), 0, 255).astype(np.uint8)
        return tg, w

    def _target_block(self, chan, k, lo, shape):
        """Just the field of one channel over a window: what the cascade's `cm` extra is."""
        if self.label_free or int(k) >= ladder.NRUNGS:
            return np.zeros(tuple(int(v) for v in ladder.shape3(shape)), np.uint8)
        return self._source(chan, k, lo, shape)[0]

    # ---- the extras ------------------------------------------------------

    def _cascade_extras(self, k, lo, shape):
        """`cm` (the rung-(k+1) target block over the patch footprint, half the patch on every axis) and,
        for the self/mix cascade modes, the tenth context cube `cx` and the corner `lo1` of the coarse
        cube. The top rung has no rung above it, so its cascade block is zero -- which is also what
        `cascade_drop` teaches the model to expect."""
        p, lo = ladder.shape3(shape), np.asarray(lo, np.int64)
        hp = np.maximum(p // 2, 1)
        cm = self._target_block(self.channels[0], int(k) + 1, lo // 2, hp)
        d = (int(self.ctx[-1]) + 1) if self.ctx else 1
        c0 = lo + p // 2
        cx = ladder.read_rung(self.pyr, int(k) + d, c0 // (1 << d) - p // 2, p, dtype=np.uint8)
        return {"cm": cm, "cx": cx, "lo1": c0 // 2 - p // 2}

    def _plane_extras(self, k):
        """The per-sample numbers the radius and scan planes need: r_max at this rung, and the five
        normalised scan values (the same for every sample of a run, read once)."""
        return {"rmax": float(self.rmax_um) / ladder.rung_um(int(k)), "meta": self.meta5}

    # ---- drawing ---------------------------------------------------------

    def _draw(self, rng, rec):
        """One window inside one region, or None when the CT-air rule rejects it."""
        p, k = self.patch, int(rec["k"])
        rlo, rsz = np.array(rec["lo"], np.int64), np.array(rec["size"], np.int64)
        hi = np.maximum(rlo + rsz - p, rlo)
        lo = rng.integers(np.minimum(rlo, hi), hi + 1)
        ct = ladder.read_rung(self.pyr, k, lo, p, dtype=np.uint8)
        if (ct == 0).mean() > 0.9 and rng.random() > self.cfg.air_keep:
            return None
        tg, w = self._rung_target(k, lo, ct)
        sym = int(draw_sym(rng, tuple(p))) if self.sym else 0
        cx = ladder.context(self.ct, lo, ct.shape, self.ctx, rung=k) if self.ctx else ()
        return rung_item(np.stack([ct] + list(cx)), tg, w, k, lo, self.ax, sym,
                         **self._plane_extras(k), **self._cascade_extras(k, lo, ct.shape))

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
                if not self._visitable(rec):
                    continue
                left, fails = self.windows, 0
                while left > 0 and fails < 8 * max(self.windows, 1):
                    got = self._draw(rng, rec)
                    if got is None:
                        fails += 1
                        continue
                    left, fails, served = left - 1, 0, served + 1
                    yield got
            if not served:   # nothing is produced yet: do not spin the CPU on an empty walk
                import time
                time.sleep(1.0)


# --------------------------------------------------------------------------- validation and loading

def val_grid(cfg, heldout, root=None, ct=None, ax=None, round_=0, rungs=None, limit=8, **kw):
    """A FIXED set of windows over the held-out regions: the same corners at every evaluation, so two
    checkpoints are compared on identical data. Non-overlapping tiles of each held-out region at each
    rung the region can supply, evenly subsampled to `limit` per (region, rung)."""
    ds = Patches(cfg, root=root, ct=ct, ax=ax, round_=round_, region_records=list(heldout),
                 sym=False, **kw)
    ds._open()
    p = ds.patch
    out = []
    for r in heldout:
        lo2, sz2 = np.array(r["lo"], np.int64), np.array(r["size"], np.int64)
        for k in (cfg.rungs if rungs is None else rungs):
            d = int(k) - 2
            org, sz = (lo2 >> d, np.maximum(sz2 >> d, 1)) if d >= 0 else (lo2 << -d, sz2 << -d)
            cs = [(z, y, x) for z in range(0, max(int(sz[0]) - int(p[0]), 0) + 1, int(p[0]))
                  for y in range(0, max(int(sz[1]) - int(p[1]), 0) + 1, int(p[1]))
                  for x in range(0, max(int(sz[2]) - int(p[2]), 0) + 1, int(p[2]))]
            if limit and len(cs) > limit:
                cs = [cs[i] for i in np.linspace(0, len(cs) - 1, limit).astype(int)]
            for c in cs:
                lo = org + np.array(c, np.int64)
                ct_ = ladder.read_rung(ds.pyr, int(k), lo, p, dtype=np.uint8)
                tg, w = ds._rung_target(int(k), lo, ct_)
                cx = ladder.context(ds.ct, lo, ct_.shape, ds.ctx, rung=int(k)) if ds.ctx else ()
                out.append(rung_item(np.stack([ct_] + list(cx)), tg, w, int(k), lo, ds.ax, 0,
                                     **ds._plane_extras(int(k)),
                                     **ds._cascade_extras(int(k), lo, ct_.shape)))
    assert out, "the held-out regions are smaller than one patch at every rung"
    return out


def loader(patches, workers=0, batch=1):
    """Workers start as fresh processes (forkserver), never forks: the parent has usually opened zarr
    already, and its asyncio loop thread does not survive a fork -- which breaks streamed reads in a way
    that only shows up minutes later."""
    return torch.utils.data.DataLoader(patches, batch_size=batch, num_workers=int(workers),
                                       pin_memory=True, persistent_workers=workers > 0,
                                       prefetch_factor=2 if workers else None,
                                       multiprocessing_context="forkserver" if workers else None)
