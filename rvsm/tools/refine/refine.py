"""Prediction-guided surface refinement: move each point of a published tifxyz surface along its normal to the
nearest probability peak of a recto store, with a confidence-weighted smoothing of the displacement field over
the (H,W) grid so the sheet moves coherently. Points outside the store, in masked CT, or without a peak stay put.

All surfaces crossing the store's box are refined JOINTLY, lasagna-style: along each point's normal ray the
neighbouring sheets (nearest below and above) and the probability peaks are matched one-to-one in order
(`assign()`), so neighbouring sheets can neither merge onto one band nor cross, and a sheet that is a whole gap
off still finds its own band.

The VERSO term (with a verso store of the same box): n points from VERSO to RECTO (radially outward, the
export.SIGN_CONVENTION), so a sheet's verso face lies one thickness INWARD of its recto face. A candidate recto
peak is rejected when a verso peak lies strictly between the vertex and that candidate on the ray -- reaching it
would mean stepping across the back face of a sheet, i.e. jumping onto the next wrap -- and a candidate earns a
bonus when a verso peak sits about one thickness inward of it (the sheet's own back face). The thickness is read
from a thickness store when one is given, else estimated from the recto-verso peak spacing in the box.

Render3d drag anchors (`--anchors`) are hard constraints: a Gaussian-kernel displacement field over the grid,
tapered to zero away from the anchors (the usrm ui.anchor_displacement design), is applied before the snap, and
the snap is then faded out around each anchor so it cannot pull an anchored cell back.

Frames: published tifxyz live in their own volume's voxels (7.91 um legacy or 2.4 um fine); every store here is
in the 2.4 um fine frame. The refiner works in the fine frame and writes the refined surface back in the frame it
was read in (so it stays a drop-in tifxyz for vc3d).

WARNING -- CIRCULARITY. Evaluate with a DIFFERENT store than the one refined against (--eval-store). Refining a
surface onto the peaks of store A and then measuring recall / offset against store A measures only that the
optimiser did its job: the "gain" is circular. The numbers printed without --eval-store are labelled so.

    python -m rvsm.tools.refine.refine --recto R.zarr [--verso V.zarr] --paths DIR --umbilicus U.json --out DIR
    (see --help; `rvsm refine ...` is the same entry)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

from rvsm import evalsurf as E

LEGACY_VOLUME = "20230205180739"   # PHerc Paris 4, 7.91 um: the frame the volpkg segments were traced in
FINE_VOLUME = "20260411134726"     # PHerc Paris 4, 2.4 um: the frame every rvsm store is in
THICK_UNIT = 0.25                  # the thickness store's code step (targets.encode_unsigned), voxels
TMIN, TMAX = 2.0, 24.0             # a recto-verso spacing outside this is not one sheet (rung-2 voxels)


# =============================================================================== sampling the band

def sample(V, q):
    """V (Z,Y,X) sampled trilinearly at q (N,3) zyx voxel coords, 0 outside; a uint8 V is read as V/255.
    (map_coordinates, not evalsurf.trilerp: the same numbers, without materialising a float copy of a
    multi-GB uint8 store.)"""
    from scipy.ndimage import map_coordinates
    q = np.asarray(q, np.float32)
    if not len(q):
        return np.zeros(0, np.float32)
    out = map_coordinates(V, q.T, order=1, mode="constant", cval=0.0, prefilter=False, output=np.float32)
    return out / 255.0 if V.dtype == np.uint8 else out


def profile(V, q, n, far):
    """Probability sampled along the normal: (2*far+1, N)."""
    ts = np.arange(-far, far + 1, dtype=np.float32)
    return sample(V, (q[None] + ts[:, None, None] * n[None]).reshape(-1, 3)).reshape(len(ts), len(q))


def peaks(S, far, thr):
    """Sub-voxel peak offset per point (parabola through the argmax) and its confidence (peak height, 0 below thr)."""
    k = np.clip(S.argmax(0), 1, 2 * far - 1)
    y0, y1, y2 = (np.take_along_axis(S, k[None] + j, 0)[0] for j in (-1, 0, 1))
    den = np.minimum(y0 - 2 * y1 + y2, -1e-6)
    off = (k - far) + np.clip(0.5 * (y0 - y2) / den, -1, 1)
    conf = np.where(y1 >= thr, y1, 0.0).astype(np.float32)
    return off.astype(np.float32), conf


def local_maxima(S, far, thr, P=6):
    """Top-P local maxima of each profile: (positions (P,N) in voxels, strengths (P,N)); padding has strength -1."""
    ts = np.arange(-far, far + 1, dtype=np.float32)
    mid = S[1:-1]
    # strict on the left: a two-sample plateau (a band centred between two samples) is ONE peak, not two --
    # usrm2's `>=` on both sides returned it twice, and two sheets could then share one band in assign()
    ismax = (mid > S[:-2]) & (mid >= S[2:]) & (mid >= thr)
    st = np.where(ismax, mid, -1.0)
    idx = np.argsort(-st, axis=0)[:P]  # strongest first
    stren = np.take_along_axis(st, idx, 0)
    y0, y1, y2 = (np.take_along_axis(S, idx + 1 + j, 0) for j in (-1, 0, 1))
    den = np.minimum(y0 - 2 * y1 + y2, -1e-6)
    pos = ts[idx + 1] + np.clip(0.5 * (y0 - y2) / den, -1, 1)
    return pos.astype(np.float32), stren.astype(np.float32)


def assign(pos, stren, below, above, alpha=0.08):
    """Order-preserving one-to-one matching of the sheets on a ray (nearest other sheet below at offset `below`
    (<0, NaN if none), this sheet at 0, nearest above at `above` (>0, NaN if none)) to the peaks (pos, stren)
    (P,N). Cost = alpha * |offset - peak| - strength, summed over the matched sheets. Returns this sheet's
    displacement (NaN where no peak is available)."""
    from itertools import combinations
    P, N = pos.shape
    order = np.argsort(np.where(stren > 0, pos, np.inf), axis=0)  # peaks by position, padding last
    pos, stren = np.take_along_axis(pos, order, 0), np.take_along_axis(stren, order, 0)
    out = np.full(N, np.nan, np.float32)
    best = np.full(N, np.inf, np.float32)
    zero = np.zeros(N, np.float32)
    for K, sheets, me in ((3, (below, zero, above), 1), (2, (below, zero), 1), (2, (zero, above), 0), (1, (zero,), 0)):
        use = np.ones(N, bool)
        for s in sheets:
            use &= np.isfinite(s)
        use &= ~np.isfinite(best)  # only points not yet matched with more sheets
        if not use.any():
            continue
        offs = [np.nan_to_num(s) for s in sheets]
        for combo in combinations(range(P), K):
            valid = use & (stren[list(combo)] > 0).all(0)
            if not valid.any():
                continue
            cost = sum(alpha * np.abs(offs[k] - pos[c]) - stren[c] for k, c in enumerate(combo))
            better = valid & (cost < best)
            best[better], out[better] = cost[better], pos[combo[me]][better]
        best[use & np.isfinite(best)] = np.minimum(best[use & np.isfinite(best)], 1e6)  # matched at this K: done
    return out


def ray_neighbours(q, n, others, R, lateral=3.0, k=48):
    """Offsets along the normal of the nearest other-sheet points below (<0) and above (>0) each point, NaN if
    none. Vectorised over the k nearest other-sheet points (usrm2 looped a ball query per point, which is
    minutes per iteration at the millions of points a whole slab has)."""
    from scipy.spatial import cKDTree
    below, above = np.full(len(q), np.nan, np.float32), np.full(len(q), np.nan, np.float32)
    if not len(others) or not len(q):
        return below, above
    tree = cKDTree(others)
    kk = min(int(k), len(others))
    for i in range(0, len(q), 200_000):
        sl = slice(i, i + 200_000)
        dist, idx = tree.query(q[sl], k=kk, distance_upper_bound=R + lateral)
        dist, idx = dist.reshape(len(dist), -1), idx.reshape(len(idx), -1)
        ok = np.isfinite(dist)
        d = others[np.minimum(idx, len(others) - 1)] - q[sl][:, None]
        t = (d * n[sl][:, None]).sum(-1)
        lat = np.linalg.norm(d - t[..., None] * n[sl][:, None], axis=-1)
        ok &= (lat <= lateral) & (np.abs(t) <= R)
        lo = np.where(ok & (t < -0.5), t, -np.inf).max(1)
        hi = np.where(ok & (t > 0.5), t, np.inf).min(1)
        below[sl] = np.where(np.isfinite(lo), lo, np.nan)
        above[sl] = np.where(np.isfinite(hi), hi, np.nan)
    return below, above


# ===================================================================================== the verso term

def verso_adjust(pos, stren, vpos, vstren, T, margin=0.5, beta=0.5, block=np.inf, tol=None):
    """Recto candidates (pos, stren) (P,N) re-weighted by the verso peaks (vpos, vstren) (Pv,N) on the same ray.

    - BLOCKED: a verso peak lies strictly between the vertex (0) and the candidate, on the candidate's side and
      more than `margin` from both ends. Reaching that candidate means crossing a sheet's back face, i.e.
      jumping a wrap. `block=inf` rejects it (strength -1), a finite `block` subtracts that much.
    - PAIRED: a verso peak one thickness T inward of the candidate (at pos - T; n points verso->recto) adds
      `beta * verso strength * exp(-((gap - T) / tol)^2 / 2)`, the sheet's own back face confirming it.
    T is a scalar or (N,) voxels. Returns (strength', blocked (P,N) bool)."""
    valid = stren > 0
    p = pos[:, None, :]
    v = vpos[None]
    vs = (vstren > 0)[None]
    between = vs & (np.sign(v) == np.sign(p)) & (np.abs(v) > margin) & (np.abs(v) < np.abs(p) - margin)
    blocked = between.any(1) & valid
    T = np.broadcast_to(np.asarray(T, np.float32), pos.shape[1:])
    tol = np.maximum(1.5, 0.35 * T) if tol is None else np.float32(tol)
    gap = p - v                                                    # > 0: the verso is inward of the candidate
    pair = np.where(vs & (gap > 0), np.exp(-0.5 * ((gap - T[None, None]) / tol) ** 2) * vstren[None], 0.0).max(1)
    st = np.where(valid, stren + beta * pair, stren)
    if np.isinf(block):
        st = np.where(blocked, -1.0, st)
    else:
        st = np.where(blocked, np.maximum(st - block, 1e-3), st)
    return st.astype(np.float32), blocked


def thickness_from_peaks(pos, stren, vpos, vstren, tmin=TMIN, tmax=TMAX):
    """The median sheet thickness of a set of rays: for the strongest recto peak of each ray, the gap to the
    nearest verso peak inward of it (within tmin..tmax). NaN when no ray has a pair."""
    if not pos.shape[1]:
        return float("nan")
    j = stren.argmax(0)
    p = np.take_along_axis(pos, j[None], 0)[0]
    ok = np.take_along_axis(stren, j[None], 0)[0] > 0
    gap = np.where(vstren > 0, p[None] - vpos, np.inf)
    gap = np.where((gap >= tmin) & (gap <= tmax), gap, np.inf).min(0)
    g = gap[ok & np.isfinite(gap)]
    return float(np.median(g)) if len(g) else float("nan")


# ============================================================================================ grids

def crop_to_box(g, origin, size, margin=64, pad=8):
    """The grid rows/cols whose points touch the box (+margin voxels), plus `pad` cells; (crop, (r0, c0))."""
    o, s = np.asarray(origin, np.float32), np.asarray(size, np.float32)
    k = np.isfinite(g).all(-1) & ((g >= o - margin) & (g < o + s + margin)).all(-1)
    if not k.any():
        return g[:0, :0], (0, 0)
    rows, cols = np.where(k.any(1))[0], np.where(k.any(0))[0]
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, g.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, g.shape[1])
    return g[r0:r1, c0:c1], (int(r0), int(c0))


def _tif_channels(d):
    """The z, y, x tifs of a tifxyz as (H,W) arrays: memory maps when uncompressed (vc3d writes them so),
    else read whole."""
    import tifffile
    out = []
    for c in "zyx":
        try:
            out.append(tifffile.memmap(f"{d}/{c}.tif", mode="r"))
        except Exception:  # noqa: BLE001 - compressed / tiled: no memmap
            out.append(np.asarray(tifffile.imread(f"{d}/{c}.tif"), np.float32))
    return out


def read_surface_box(d, frame, origin, size, margin=64, pad=8, rows=256):
    """crop_to_box(frame.to_fine(E.read_surface(d)), origin, size, margin, pad) without holding the whole
    grid: the tifs are memory-mapped and scanned `rows` rows at a time (fine-frame surfaces by their z
    channel first), and only the crop is converted. Returns (crop (h,w,3) fine zyx, (r0, c0), (H, W))
    or None when no point touches the box. A published 2.4 um grid is ~460 MB and a slab needs ~25 of
    its ~4000 rows."""
    o, s = np.asarray(origin, np.float32), np.asarray(size, np.float32)
    lo_m, hi_m = o - margin, o + s + margin
    ch = _tif_channels(d)
    H, Wd = ch[0].shape
    rany, cany = np.zeros(H, bool), np.zeros(Wd, bool)
    ident = frame is IDENTITY

    def grid(r0, r1, c0=0, c1=None):
        g = np.stack([np.asarray(c[r0:r1, c0:c1], np.float32) for c in ch], -1)
        g = np.where((g > 0).all(-1)[..., None], g, np.nan)
        return g if ident else frame.to_fine(g)
    for r in range(0, H, rows):
        r1 = min(r + rows, H)
        if ident:  # the z channel alone rules out rows
            z = np.asarray(ch[0][r:r1], np.float32)
            zr = ((z >= lo_m[0]) & (z < hi_m[0])).any(1)
            if not zr.any():
                continue
            ri = np.where(zr)[0]
            r, r1 = r + int(ri[0]), r + int(ri[-1]) + 1
        g = grid(r, r1)
        k = np.isfinite(g).all(-1) & ((g >= lo_m) & (g < hi_m)).all(-1)
        rany[r:r1] |= k.any(1)
        cany |= k.any(0)
    if not rany.any():
        del ch
        return None
    rr, cc = np.where(rany)[0], np.where(cany)[0]
    r0, r1 = max(int(rr.min()) - pad, 0), min(int(rr.max()) + pad + 1, H)
    c0, c1 = max(int(cc.min()) - pad, 0), min(int(cc.max()) + pad + 1, Wd)
    crop = grid(r0, r1, c0, c1)
    del ch
    return crop, (r0, c0), (H, Wd)


def column_pieces(g, lo, hi, pad=8):
    """The column runs of the grid g (H,W,3 zyx) with cells inside [lo, hi), each with its own row range and
    `pad` cells around it: [(r0, r1, c0, c1)]. Runs closer than 2*pad+1 columns merge, so pieces never
    share a cell. A sheet crossing a slab at a slant is a thin diagonal band of its grid; its pieces are
    small where its bounding rectangle is not."""
    lo, hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
    k = np.isfinite(g).all(-1) & ((g >= lo) & (g < hi)).all(-1)
    cols = np.nonzero(k.any(0))[0]
    if not len(cols):
        return []
    H, W = k.shape
    brk = np.nonzero(np.diff(cols) > 2 * pad + 1)[0]
    out = []
    for a, b in zip(np.r_[0, brk + 1], np.r_[brk, len(cols) - 1]):
        ca, cb = int(cols[a]), int(cols[b])
        rows = np.nonzero(k[:, ca:cb + 1].any(1))[0]
        out.append((max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad + 1, H),
                    max(ca - pad, 0), min(cb + pad + 1, W)))
    return out

def upsample(g, f, order=1):
    """A denser grid: (H,W,3) -> ((H-1)f+1, (W-1)f+1, 3) by spline interpolation of each coordinate (holes filled
    for the interpolation, then re-masked: a new cell is a hole if any of the 4 old cells around it was one).
    Node (r, c) of the input is node (r*f, c*f) of the output. Published tifxyz grids are 1/20 voxel pitch
    (meta scale 0.05) of their own volume; see `auto_up`."""
    from scipy.ndimage import zoom
    if f == 1:
        return g.copy()
    ok = np.isfinite(g).all(-1)
    fg = filled(g, sigma=2.0)
    H, W = g.shape[:2]
    zf = ((H - 1) * f + 1) / H, ((W - 1) * f + 1) / W
    out = np.stack([zoom(fg[..., i], zf, order=order, mode="nearest") for i in range(3)], -1)
    okz = zoom(ok.astype(np.float32), zf, order=1, mode="nearest") > 0.999
    out[~okz] = np.nan
    return out.astype(np.float32)


def pitch(g):
    """The median distance between grid neighbours, in voxels (NaN for an empty grid)."""
    d = [np.linalg.norm(np.diff(g, axis=a), axis=-1) for a in (0, 1)]
    d = np.concatenate([x[np.isfinite(x)] for x in d])
    return float(np.median(d)) if len(d) else float("nan")


def auto_up(g, target=4.0, cap=32):
    """The upsampling factor that brings the grid pitch to ~`target` voxels."""
    p = pitch(g)
    return 1 if not np.isfinite(p) else int(np.clip(round(p / target), 1, cap))


def smooth(field, w, sigma):
    """Confidence-weighted Gaussian smoothing over the grid (normalized convolution); NaN/zero-weight cells get
    the neighbourhood's value."""
    from scipy.ndimage import gaussian_filter
    num = gaussian_filter(np.nan_to_num(field * w), sigma)
    den = gaussian_filter(w, sigma)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def filled(g, sigma=1.0):
    """The grid with only its holes (NaN) filled from their neighbourhood; valid cells are untouched."""
    ok = np.isfinite(g).all(-1)
    w = ok.astype(np.float32)
    f = np.stack([smooth(np.where(ok, g[..., i], 0), w, sigma) for i in range(3)], -1)
    return np.where(ok[..., None], g, f)


def normals(g, ax):
    """Hole-tolerant unit normals oriented outward from the axis (VERSO -> RECTO): computed on the filled grid."""
    n = E._normals_raw(filled(g), ax)
    return np.where(np.isfinite(g).all(-1)[..., None], n, np.nan)


def box_taper(g, origin, size, taper):
    """(H,W) weight in 0..1: 1 deeper than `taper` voxels inside the box, falling linearly to 0 at its faces --
    so a surface refined in a slab joins the untouched part outside without a step."""
    o, s = np.asarray(origin, np.float32), np.asarray(size, np.float32)
    d = np.minimum(g - o, o + s - 1 - g).min(-1)
    if taper <= 0:
        return np.where(np.isfinite(d) & (d >= 0), 1.0, 0.0).astype(np.float32)
    return np.clip(np.nan_to_num(d, nan=-1.0) / float(taper), 0.0, 1.0).astype(np.float32)


# ============================================================================================ refine

def refine_many(grids, V, origin, ax, far=12, sigma=2.0, iters=3, thr=0.5, ct=None, chunk=400_000,
                W=None, thick=None, T=None, verso_thr=None, verso_beta=0.5, verso_block=np.inf, verso_margin=0.5,
                holds=None, taper=0.0, box=None, sigma_final=None, local_normal=True, log=None):
    """Joint refinement of several (H,W,3) zyx grids (NaN = hole) against V, a (Z,Y,X) probability (float in
    [0,1] or uint8) at `origin`. Each iteration: normals; per point the peaks along the normal ray and the
    neighbouring sheets on it are matched in order (assign()); the matched peak's offset is smoothed over the
    grid (confidence-weighted) and the sheet moves.

    W: the verso probability over the same box (enables `verso_adjust`); thick: a thickness volume in voxels
    over the box (0 = no data); T: a fixed thickness (else estimated from the peaks at iteration 0).
    holds: per grid (H,W) weights in 0..1, 1 = anchored (never moved by the snap). taper: see `box_taper`.
    sigma: grid cells, one value or one per grid.
    sigma_final: the smoothing (grid cells, one value or one per grid) of the LAST iteration only, so the
    final pass follows the band locally while the earlier ones stay robust (default: sigma).
    local_normal: recompute the normals from the current (refined) grid every iteration, so the search
    direction follows the refined shape; False keeps the published grid's normals throughout.
    box: (origin, size) of the region being refined when V covers only part of it (a tile of a slab): points
    must be inside both, and the taper fades at `box`'s faces, not V's.
    Returns (refined grids, per-iteration stats)."""
    o = np.asarray(origin, np.float32)
    grids = [g.copy() for g in grids]
    size = np.array(V.shape, np.float32)
    lo_b, hi_b = o, o + size - 1
    bo, bs = (o, size) if box is None else (np.asarray(box[0], np.float32), np.asarray(box[1], np.float32))
    lo_b, hi_b = np.maximum(lo_b, bo), np.minimum(hi_b, bo + bs - 1)
    insides = [np.isfinite(g).all(-1) & ((g >= lo_b) & (g <= hi_b)).all(-1) for g in grids]
    fades = [box_taper(g, bo, bs, taper) * (1.0 - (np.clip(h, 0, 1) if h is not None else 0.0))
             for g, h in zip(grids, holds or [None] * len(grids))]
    vthr = thr if verso_thr is None else verso_thr
    sigmas = list(np.broadcast_to(np.asarray(sigma, np.float64), (len(grids),)))
    sigmas_f = sigmas if sigma_final is None else list(np.broadcast_to(np.asarray(sigma_final, np.float64), (len(grids),)))
    n_fixed = None
    tunit = THICK_UNIT if thick is not None and thick.dtype == np.uint8 else 1.0
    Tg = None if T is None else float(T)
    stats = []
    for it in range(iters):
        r = max(3, int(round(far / (1.5 ** it))))
        st = {"iter": it, "far": r, "points": 0, "with_peak": 0.0, "mean_abs_move": 0.0, "max_move": 0.0,
              "capped": 0.0}
        if W is not None:
            st["verso_blocked"] = 0.0
        if local_normal or n_fixed is None:
            ns = [normals(g, ax) for g in grids]
            if not local_normal:
                n_fixed = ns
        else:   # the published normals, masked to the cells that are still points
            ns = [np.where(np.isfinite(g).all(-1)[..., None], n, np.nan) for g, n in zip(grids, n_fixed)]
        sig_it = sigmas_f if it == iters - 1 else sigmas
        oks = [ins & np.isfinite(n).all(-1) for ins, n in zip(insides, ns)]
        pts = [g[ok] for g, ok in zip(grids, oks)]
        if W is not None and Tg is None:   # one thickness for the box, from the recto-verso spacing
            q = np.concatenate([p for p in pts if len(p)] or [np.zeros((0, 3), np.float32)])
            nq = np.concatenate([n[ok] for n, ok in zip(ns, oks) if ok.any()] or [np.zeros((0, 3), np.float32)])
            sub = np.linspace(0, len(q) - 1, min(len(q), 200_000)).astype(np.int64) if len(q) else np.zeros(0, np.int64)
            rp, rs = local_maxima(profile(V, q[sub] - o, nq[sub], far), far, thr)
            vp, vs = local_maxima(profile(W, q[sub] - o, nq[sub], far), far, vthr)
            Tg = thickness_from_peaks(rp, rs, vp, vs)
            if not np.isfinite(Tg):
                Tg = 8.0
        if W is not None:
            st["thickness"] = round(float(Tg), 3)
        moves, tot = [], 0
        bbs = [(p.min(0), p.max(0)) if len(p) else None for p in pts]
        for j, (g, n, ok, q) in enumerate(zip(grids, ns, oks, pts)):
            if not ok.any():
                moves.append(np.zeros(g.shape[:2], np.float32))
                continue
            qi = q - o
            # other sheets' points that can be within reach (ray_neighbours' bound is r + 3): a tree over the
            # whole box per sheet is the slab's cost otherwise
            pad = r + 4.0
            bl, bh = bbs[j][0] - pad, bbs[j][1] + pad
            others = [p[((p >= bl) & (p <= bh)).all(-1)] for k, p in enumerate(pts)
                      if k != j and bbs[k] is not None and (bbs[k][0] <= bh).all() and (bbs[k][1] >= bl).all()]
            others = np.concatenate([p for p in others if len(p)] or [np.zeros((0, 3), np.float32)])
            nn = n[ok]
            below, above = ray_neighbours(q, nn, others, r)
            off, conf = np.zeros(len(q), np.float32), np.zeros(len(q), np.float32)
            for i in range(0, len(q), chunk):
                sl = slice(i, i + chunk)
                pos, stren = local_maxima(profile(V, qi[sl], nn[sl], r), r, thr)
                if W is not None:
                    vpos, vstren = local_maxima(profile(W, qi[sl], nn[sl], r), r, vthr)
                    Ti = Tg
                    if thick is not None:
                        idx = np.clip(np.rint(qi[sl]).astype(np.int64), 0, np.array(thick.shape) - 1)
                        tv = np.asarray(thick[idx[:, 0], idx[:, 1], idx[:, 2]], np.float32) * tunit
                        Ti = np.where(tv > 0, tv, Tg).astype(np.float32)
                    stren, blocked = verso_adjust(pos, stren, vpos, vstren, Ti, margin=verso_margin,
                                                  beta=verso_beta, block=verso_block)
                    st["verso_blocked"] += float(blocked.any(0).sum())
                d = assign(pos, stren, below[sl], above[sl])
                hit = np.isfinite(d)
                off[sl] = np.where(hit, d, 0.0)
                conf[sl] = np.where(hit, np.clip(stren.max(0), 0, None), 0.0)
            if ct is not None:  # masked CT: no evidence
                idx = np.clip(np.rint(qi).astype(int), 0, np.array(V.shape) - 1)
                conf[ct[idx[:, 0], idx[:, 1], idx[:, 2]] == 0] = 0
            # never cross a neighbouring sheet: stay at least 1 voxel on this side of it
            hi = np.where(np.isfinite(above), above - 1.0, r).astype(np.float32)
            lo = np.where(np.isfinite(below), below + 1.0, -r).astype(np.float32)
            field, w = np.zeros(g.shape[:2], np.float32), np.zeros(g.shape[:2], np.float32)
            field[ok], w[ok] = np.clip(off, lo, hi), conf
            dd = smooth(field, w, sig_it[j])
            lof, hif = np.full(g.shape[:2], -float(r), np.float32), np.full(g.shape[:2], float(r), np.float32)
            lof[ok], hif[ok] = lo, hi
            dd = np.clip(dd, lof, hif) * ok * fades[j]
            moves.append(dd)
            st["points"] += int(ok.sum())
            tot += len(q)
            st["with_peak"] += float((conf > 0).sum())
            st["capped"] += float((np.isfinite(below) | np.isfinite(above)).sum())
            st["mean_abs_move"] += float(np.abs(dd[ok]).sum())
            st["max_move"] = max(st["max_move"], float(np.abs(dd).max()))
        for j, (g, n, dd) in enumerate(zip(grids, ns, moves)):  # move all sheets after all were measured
            grids[j] = g + dd[..., None] * np.where(np.isfinite(n), n, 0)
        for k in ("with_peak", "mean_abs_move", "capped", "verso_blocked"):
            if k in st:
                st[k] = st[k] / max(tot, 1)
        stats.append(st)
        if log:
            log(json.dumps(st))
    return grids, stats


def refine(g, V, origin, ax, **kw):
    gs, stats = refine_many([g], V, origin, ax, **kw)
    return gs[0], stats


# =========================================================================================== anchors

def read_anchors(path):
    """The live anchor set of a render3d anchors.jsonl: replay add/del in order. Returns [add record, ...]."""
    live = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue            # a torn last line of an append-only log
            if r.get("op") == "add" and r.get("id"):
                live[r["id"]] = r
            elif r.get("op") == "del":
                live.pop(r.get("id"), None)
    return list(live.values())


def anchor_matches(a, names):
    """Does anchor `a` name this surface? `names`: every name the surface goes by (segment id, tifxyz dir)."""
    s = str(a.get("surface", "")).rstrip("/")
    s = os.path.basename(s)
    s2 = s[:-len(".tifxyz")] if s.endswith(".tifxyz") else s
    return any(s in (nm, nm + ".tifxyz") or s2 == nm for nm in names)


def anchor_field(g, anchors, sigma=48.0, clip=64.0, rc_map=None, max_dist=None):
    """A (H,W,3) displacement field over the grid g (fine zyx) from anchors [{grid_rc, from_zyx, to_zyx}]
    (zyx already in g's frame), and its (H,W) reach `top` in 0..1.

    Gaussian-kernel interpolation in GRID space (usrm ui.anchor_field):
        w_i(r,c) = exp(-|cell - cell_i|^2 / (2 s^2)),  s = sigma voxels / grid pitch
        field    = (sum_i w_i d_i / sum_i w_i) * max_i w_i
    The Nadaraya-Watson average reproduces each anchor's displacement exactly at its cell and blends
    conflicting anchors between them; the max_i w_i factor tapers the field to ZERO away from any anchor
    (no information there). |d| is clipped to `clip` voxels. An anchor's cell is `rc_map(grid_rc)` (the
    published row/col mapped into this grid), else the grid point nearest its `from_zyx` -- skipped when that
    is farther than `max_dist` voxels (a piece of a surface must not pull in an anchor from elsewhere)."""
    H, Wd = g.shape[:2]
    field = np.zeros((H, Wd, 3), np.float64)
    den = np.zeros((H, Wd), np.float64)
    top = np.zeros((H, Wd), np.float64)
    if not anchors or not H or not Wd:
        return field.astype(np.float32), top.astype(np.float32)
    p = pitch(g)
    s = max(float(sigma) / (p if np.isfinite(p) and p > 0 else 1.0), 0.5)
    rr, cc = np.mgrid[:H, :Wd].astype(np.float64)
    ok = np.isfinite(g).all(-1)
    gv, iv = g[ok], np.argwhere(ok)
    for a in anchors:
        f, t = np.asarray(a["from_zyx"], np.float64), np.asarray(a["to_zyx"], np.float64)
        d = t - f
        nrm = np.linalg.norm(d)
        if nrm > clip:
            d *= clip / nrm
        rc = None
        if a.get("grid_rc") is not None and rc_map is not None:
            rc = rc_map(a["grid_rc"])
        if rc is None:
            if not len(gv):
                continue
            d2 = ((gv - f) ** 2).sum(-1)
            i = int(np.argmin(d2))
            if max_dist is not None and d2[i] > float(max_dist) ** 2:
                continue
            rc = iv[i].astype(np.float64)
        r, c = float(rc[0]), float(rc[1])
        if not (-3 * s <= r < H + 3 * s and -3 * s <= c < Wd + 3 * s):
            continue
        w = np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2.0 * s * s))
        field += w[..., None] * d
        den += w
        np.maximum(top, w, out=top)
    m = den > 1e-12
    field[m] = field[m] / den[m][:, None] * top[m][:, None]
    field[~m] = 0
    return field.astype(np.float32), top.astype(np.float32)


# ============================================================================================ frames

class Frame:
    """A surface's own voxel frame <-> the fine (2.4 um) store frame, on (N,3) zyx arrays."""

    def __init__(self, name, to_fine, to_src):
        self.name, self._to_fine, self._to_src = name, to_fine, to_src

    def to_fine(self, zyx):
        return self._apply(self._to_fine, zyx)

    def to_src(self, zyx):
        return self._apply(self._to_src, zyx)

    @staticmethod
    def _apply(fn, zyx):
        zyx = np.asarray(zyx, np.float64)
        sh = zyx.shape
        flat = zyx.reshape(-1, 3)
        ok = np.isfinite(flat).all(-1)
        out = np.full(flat.shape, np.nan, np.float64)
        if ok.any():
            out[ok] = fn(flat[ok])
        return out.reshape(sh).astype(np.float32)


IDENTITY = Frame("fine", lambda p: p, lambda p: p)


def affine_frame(M_xyz, name="legacy"):
    """A frame from a 3x4 xyz affine mapping the SOURCE (surface) frame to the fine frame."""
    M = np.asarray(M_xyz, np.float64).reshape(3, 4)
    A, b = M[:, :3], M[:, 3]
    Ai = np.linalg.inv(A)
    fwd = lambda p: (p[:, ::-1] @ A.T + b)[:, ::-1]         # noqa: E731  zyx -> xyz -> fine xyz -> zyx
    inv = lambda p: ((p[:, ::-1] - b) @ Ai.T)[:, ::-1]      # noqa: E731
    return Frame(name, fwd, inv)


def scale_offset_frame(scale, offset_zyx, name="legacy"):
    """fine = src * scale + offset (per axis zyx; `scale` a scalar or 3 values)."""
    s = np.broadcast_to(np.asarray(scale, np.float64), (3,))
    o = np.asarray(offset_zyx, np.float64).reshape(3)
    return Frame(name, lambda p: p * s + o, lambda p: (p - o) / s)


def transform_json_frame(path, direction="auto"):
    """A frame from a vc3d-style transform.json (`transformation_matrix`, xyz, 3x4 or 4x4). Direction:
    'fine_to_legacy' (the published PHercParis4 transform: moving = the 2.4 um volume, fixed = the legacy
    volume, M @ moving = fixed), 'legacy_to_fine', or 'auto' (fine_to_legacy unless the landmarks say
    otherwise)."""
    t = json.load(open(path))
    M = np.asarray(t["transformation_matrix"], np.float64)[:3, :4]
    if direction == "auto":
        direction = "fine_to_legacy"
        if "fixed_landmarks" in t and "moving_landmarks" in t:
            f, m = np.asarray(t["fixed_landmarks"], np.float64), np.asarray(t["moving_landmarks"], np.float64)
            e1 = np.abs(m @ M[:, :3].T + M[:, 3] - f).mean()      # M: moving -> fixed
            e2 = np.abs(f @ M[:, :3].T + M[:, 3] - m).mean()      # M: fixed -> moving
            fixed_is_legacy = LEGACY_VOLUME in str(t.get("fixed_volume", LEGACY_VOLUME))
            maps_moving_to_fixed = e1 <= e2
            direction = "fine_to_legacy" if maps_moving_to_fixed == fixed_is_legacy else "legacy_to_fine"
    if direction == "legacy_to_fine":
        return affine_frame(M, "legacy")
    A = np.vstack([M, [0, 0, 0, 1]])
    return affine_frame(np.linalg.inv(A)[:3], "legacy")


def function_frame(to_fine_xyz, to_src_xyz=None, name="legacy"):
    """A frame from `legacy_to_fine(xyz (N,3)) -> xyz` (and optionally its inverse). Without the inverse it is
    solved numerically: an affine fitted to the forward map, then a few Newton steps with that affine's
    Jacobian -- exact for an affine map, and good to 1e-4 voxel for a smooth one."""
    fwd = lambda p: np.asarray(to_fine_xyz(p[:, ::-1]), np.float64)[:, ::-1]   # noqa: E731
    if to_src_xyz is not None:
        inv = lambda p: np.asarray(to_src_xyz(p[:, ::-1]), np.float64)[:, ::-1]  # noqa: E731
        return Frame(name, fwd, inv)
    probe = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], np.float64) * 10000.0
    img = fwd(probe)
    X = np.hstack([probe, np.ones((len(probe), 1))])
    Aa = np.linalg.lstsq(X, img, rcond=None)[0]           # img = probe @ A + b
    A, b = Aa[:3], Aa[3]
    Ai = np.linalg.inv(A)

    def inv(p):
        x = (p - b) @ Ai
        for _ in range(4):
            x = x - (fwd(x) - p) @ Ai
        return x
    return Frame(name, fwd, inv)


def module_frame(transform=None):
    """The legacy frame from `rvsm.tools.refine.frame`: its `Frame(transform)` (to_fine / to_legacy on xyz;
    with no path it fetches the fine volume's published transform.json), or module-level
    `legacy_to_fine` / `fine_to_legacy` functions. None when the module is not there."""
    try:
        from rvsm.tools.refine import frame as F
    except Exception:  # noqa: BLE001
        return None
    if hasattr(F, "Frame"):
        f = F.Frame(transform) if transform else F.Frame()
        return function_frame(f.to_fine, f.to_legacy, "legacy")
    fwd = getattr(F, "legacy_to_fine", None)
    if fwd is None:
        return None
    return function_frame(fwd, getattr(F, "fine_to_legacy", None), "legacy")


def frame_kind(d, default="fine"):
    """'fine' / 'legacy' / 'coarse' for a tifxyz directory, from its name (the S3 re-publication names each
    frame '<id>-on-<volume>-<um>um.tifxyz') or meta.json's 'volume', else `default`."""
    name = os.path.basename(str(d).rstrip("/"))
    if "2.4um" in name or FINE_VOLUME in name:
        return "fine"
    if "7.91um" in name or LEGACY_VOLUME in name:
        return "legacy"
    if "um" in name.split("-")[-1] and name.split("-")[-1][0].isdigit():
        return "coarse"
    try:
        vol = str(json.load(open(os.path.join(d, "meta.json"))).get("volume", ""))
    except Exception:  # noqa: BLE001
        vol = ""
    return "fine" if FINE_VOLUME in vol else "legacy" if LEGACY_VOLUME in vol else default


# ======================================================================================= surfaces io

def find_surfaces(paths, prefer=("2.4um", "7.91um")):
    """[(name, segid, tifxyz dir)] under `paths`: one tifxyz per segment directory, the first variant in
    `prefer` order (a '<segid>/<id>-on-<vol>-<um>um.tifxyz' layout) or the directory itself when it is a
    tifxyz (x.tif + meta.json)."""
    out = []
    for seg in sorted(glob.glob(os.path.join(paths, "*"))):
        if not os.path.isdir(seg):
            continue
        sid = os.path.basename(seg)
        if os.path.exists(os.path.join(seg, "x.tif")):
            out.append((sid, sid, seg))
            continue
        vs = [d for d in sorted(glob.glob(os.path.join(seg, "*"))) if os.path.exists(os.path.join(d, "x.tif"))]
        pick = next((d for p in prefer for d in vs if p in os.path.basename(d)), None)
        if pick is None and vs and not prefer:
            pick = vs[0]
        if pick:
            nm = os.path.basename(pick)
            out.append((nm[:-len(".tifxyz")] if nm.endswith(".tifxyz") else nm, sid, pick))
    return out


def write_tifxyz(src_dir, out_dir, g, note, up=1):
    """A tifxyz directory like `src_dir` with the points g (meta.json copied, bbox recomputed, scale x up)."""
    import tifffile
    os.makedirs(out_dir, exist_ok=True)
    for i, c in enumerate("zyx"):
        # -1 = invalid, as vc3d writes it (evalsurf.read_surface reads any value <= 0 as a hole)
        tifffile.imwrite(f"{out_dir}/{c}.tif", np.where(np.isfinite(g).all(-1), g[..., i], -1.0).astype(np.float32))
    meta = json.load(open(f"{src_dir}/meta.json"))
    if up != 1 and "scale" in meta:
        meta["scale"] = [float(v) * up for v in meta["scale"]]
    v = g[np.isfinite(g).all(-1)]
    meta["bbox"] = [v.min(0)[::-1].tolist(), v.max(0)[::-1].tolist()] if len(v) else meta.get("bbox")
    meta["refined"] = note
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1)
    for f in os.listdir(src_dir):  # any other per-surface files ride along untouched
        if f not in ("z.tif", "y.tif", "x.tif", "meta.json") and os.path.isfile(f"{src_dir}/{f}"):
            shutil.copy(f"{src_dir}/{f}", f"{out_dir}/{f}")
    return out_dir


def copy_tifxyz(src_dir, out_dir, note):
    """A byte-exact copy of a tifxyz directory, with `note` added to its meta.json as "refined"."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if os.path.isfile(os.path.join(src_dir, f)) and f != "meta.json":
            shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))
    meta = json.load(open(os.path.join(src_dir, "meta.json")))
    meta["refined"] = note
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return out_dir

def write_back(g_src, crop_rc, up, old_nodes, new_fine, frame, tol=1e-3, inplace=False):
    """The full published grid (its own frame) with the nodes the refiner moved replaced: node (r, c) of the
    published grid is node ((r - r0) * up, (c - c0) * up) of the dense crop `new_fine`; `old_nodes` is the
    published crop in the fine frame. Nodes that moved by more than `tol` voxels get
    src + (to_src(new) - to_src(old)), so a node the refiner did not touch is bit-identical."""
    r0, c0 = crop_rc
    o, n = old_nodes, new_fine[::up, ::up]
    out = g_src if inplace else g_src.copy()
    h, w = o.shape[:2]
    blk = out[r0:r0 + h, c0:c0 + w]
    moved = np.isfinite(o).all(-1) & np.isfinite(n).all(-1) & (np.abs(n - o) > tol).any(-1)
    moved &= np.isfinite(blk).all(-1)
    if moved.any():
        blk[moved] = blk[moved] + (frame.to_src(n[moved]) - frame.to_src(o[moved]))
    out[r0:r0 + h, c0:c0 + w] = blk
    return out


# ============================================================================================ stores

def store_box(path):
    """(origin zyx, shape zyx, rung) of a store: origin_zyx/rung from its attrs (0 / 2 when absent)."""
    from rvsm import ladder
    a = ladder.open_zarr(path)
    at = dict(a.attrs)
    return (np.asarray(at.get("origin_zyx", (0, 0, 0)), np.int64), np.asarray(a.shape[-3:], np.int64),
            int(at.get("rung", 2)))


def read_box(path, lo, shape, near=None, reach=48, block=128, jobs=8):
    """The uint8 (Z,Y,X) of a store over the fine-frame box lo..lo+shape (0 outside the store).

    near: (N,3) fine zyx points -- then only the `block`^3 blocks (box-aligned) within `reach` voxels of a
    point are read and the rest stays 0 (untouched np.zeros pages are never backed). That does NOT bound
    a real slab: ~50 sheets cross an 8192^2 slab and visit nearly every block, so `run` reads one tile
    at a time instead (docs/refine.md, "Memory")."""
    from concurrent.futures import ThreadPoolExecutor

    from rvsm import ladder
    a = ladder.open_zarr(path)
    o, S, _ = store_box(path)
    lo, shape = np.asarray(lo, np.int64), np.asarray(shape, np.int64)
    out = np.zeros(tuple(shape), np.uint8)
    if near is None:
        blocks = [(np.zeros(3, np.int64), shape)]
    else:
        q = np.asarray(near, np.float64)
        q = q[np.isfinite(q).all(-1)] - lo
        nb = -(-shape // block)
        idx = set()
        for off in np.array(np.meshgrid(*[(-reach, 0, reach)] * 3, indexing="ij")).reshape(3, -1).T:
            b = np.floor((q + off) / block).astype(np.int64)
            b = b[((b >= 0) & (b < nb)).all(-1)]
            idx.update(map(tuple, np.unique(b, axis=0).tolist()))
        blocks = [(np.array(b) * block, np.minimum((np.array(b) + 1) * block, shape) - np.array(b) * block)
                  for b in sorted(idx)]

    def one(bl):
        b0, n = bl
        aa, bb = np.maximum(lo + b0 - o, 0), np.minimum(lo + b0 + n - o, S)
        if (bb > aa).all():
            sl = tuple(slice(int(x), int(y)) for x, y in zip(aa, bb))
            blk = np.asarray(a[(0,) + sl] if a.ndim == 4 else a[sl], np.uint8)
            st = aa + o - lo
            out[st[0]:st[0] + blk.shape[0], st[1]:st[1] + blk.shape[1], st[2]:st[2] + blk.shape[2]] = blk
    with ThreadPoolExecutor(max(1, int(jobs))) as ex:
        list(ex.map(one, blocks))
    return out


def read_ct(ct, lo, shape, down=1):
    """uint8 CT over a fine-frame box; None when there is no CT. down = 2^d reads the rung-(2+d) level (the
    box in fine voxels, the result in that level's voxels) -- an overview picture of an 8192^2 slab read at
    rung 2 would decode every 128^3 chunk of the slab for one slice."""
    if not ct:
        return None
    from rvsm import ladder
    pyr = ladder.rungs(ct)
    k0 = min(pyr)
    d = max(0, int(np.floor(np.log2(max(int(down), 1)))))
    lo, shape = np.asarray(lo, np.int64), np.asarray(shape, np.int64)
    return ladder.read_rung(pyr, k0 + d, lo >> d, np.maximum(-(-shape >> d), 1), dtype=np.uint8)


# ========================================================================================= metrics

def surface_metrics(Vu, origin, g, ax, thr=0.5, far=40, win=16, r=4, um=2.4, chunk=200_000, mask=None):
    """evalsurf's per-surface numbers -- recall@r, offset bias/spread, merge_frac (E.metrics), continuity
    (E.continuity_one) and the expected run length (E.erl) -- for the grid g against a uint8 store over
    `origin`, computed without evalsurf's float copy of the whole store. mask: (H,W) cells to count (a tile's
    core); the rest of the grid only gives the normals."""
    o = np.asarray(origin, np.float32)
    n = normals(g, ax)
    k = np.isfinite(g).all(-1) & ((g >= o) & (g <= o + np.array(Vu.shape) - 1)).all(-1) & np.isfinite(n).all(-1)
    if mask is not None:
        k &= mask
    if not k.any():
        return {"n_points": 0}
    q, nn = g[k] - o, n[k]
    S = np.concatenate([profile(Vu, q[i:i + chunk], nn[i:i + chunk], far) for i in range(0, len(q), chunk)], 1)
    c = far
    m = {f"recall@{rr}": float((S[c - rr:c + rr + 1].max(0) >= thr).mean()) for rr in (2, 4, 8)}
    w = S[c - win:c + win + 1]
    kk = np.clip(w.argmax(0), 1, 2 * win - 1)
    y0, y1, y2 = (np.take_along_axis(w, kk[None] + j, 0)[0] for j in (-1, 0, 1))
    den = np.minimum(y0 - 2 * y1 + y2, -1e-6)
    off = (kk - win) + np.clip(0.5 * (y0 - y2) / den, -1, 1)
    hit_off = w.max(0) >= thr
    ov = off[hit_off]
    b = S >= thr
    runs = b[0].astype(np.int32) + (b[1:] & ~b[:-1]).sum(0)
    m.update({"n_points": int(k.sum()), "offset_frac": float(hit_off.mean()),
              "offset_mean": float(ov.mean()) if len(ov) else float("nan"),
              "offset_std": float(ov.std()) if len(ov) else float("nan"),
              "offset_le3": float((np.abs(ov) <= 3).mean()) if len(ov) else float("nan"),
              "merge_frac": float((runs > 1).mean())})
    hit = np.zeros(g.shape[:2], bool)
    hit[k] = b[c - r:c + r + 1].max(0)
    merged = np.zeros(g.shape[:2], bool)
    merged[k] = runs > 1
    inner = k.copy()
    inner[:1], inner[-1:], inner[:, :1], inner[:, -1:] = False, False, False, False
    nb = np.ones(g.shape[:2], bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nb[1:-1, 1:-1] &= hit[1 + dy:hit.shape[0] - 1 + dy, 1 + dx:hit.shape[1] - 1 + dx]
    cc = inner & hit
    m["continuity"] = float(nb[cc].sum()) / max(int(cc.sum()), 1)
    gf = np.where(np.isfinite(g), g, 0.0)
    good = k & hit & ~merged
    rl, tl = [], 0.0
    for a in (0, 1):
        ra, la, _ = E._walk_axis(good, k, gf, a, um)
        rl.append(ra)
        tl += la
    rl = np.concatenate(rl)
    m["erl_um"] = float((rl ** 2).sum() / tl) if tl > 0 else 0.0
    return m


EVAL_KEYS = ("recall@2", "recall@4", "offset_mean", "offset_std", "offset_le3", "merge_frac", "continuity", "erl_um")


# ========================================================================================== pictures

def compare_png(path, ct, V, origin, before, after, crop=320, scale=3, slab=2.5, dot=2, zi=None, shape=None):
    """Before | after panels for one surface: a CT z-slice (the one with most surface points unless `zi`), the
    probability band as a red tint, published points (left, green) and refined points (right, magenta),
    cropped and upscaled. `ct`: None (black), an array indexed [z] in box voxels, or a callable
    ct(zi, y0, y1, x0, x1) -> the (y, x) crop (box voxels). V float or uint8, or a callable
    V(zi, y0, y1, x0, x1) -> the probability crop in 0..1 (then `shape` is the box shape). before/after are
    grids or (N,3) point sets."""
    from PIL import Image
    o = np.asarray(origin, np.float32)
    shp = np.array(V.shape if shape is None else shape)
    pts = [g[np.isfinite(g).all(-1) & ((g >= o) & (g < o + shp)).all(-1)] - o for g in (before, after)]
    if not len(pts[0]):
        return None
    if zi is None:
        zi = int(np.bincount(np.clip(np.rint(pts[0][:, 0]).astype(int), 0, shp[0] - 1)).argmax())
    yx = [np.rint(p[np.abs(p[:, 0] - zi) <= slab, 1:]).astype(int) for p in pts]
    cy, cx = (yx[0].mean(0) if len(yx[0]) else shp[1:] // 2).astype(int)
    y0, x0 = int(max(cy - crop // 2, 0)), int(max(cx - crop // 2, 0))
    y1, x1 = int(min(y0 + crop, shp[1])), int(min(x0 + crop, shp[2]))
    if callable(V):
        vzc = np.asarray(V(zi, y0, y1, x0, x1), np.float32)
    else:
        vzc = np.asarray(V[zi, y0:y1, x0:x1], np.float32) / (255.0 if V.dtype == np.uint8 else 1.0)
    if ct is None:
        cz = np.zeros((y1 - y0, x1 - x0), np.float32)
    elif callable(ct):
        cz = np.asarray(ct(zi, y0, y1, x0, x1), np.float32)
    else:
        cz = np.asarray(ct[zi], np.float32)[y0:y1, x0:x1]
    base = np.repeat(cz[..., None], 3, -1) * 0.85
    base[..., 0] = np.clip(base[..., 0] + 140 * vzc, 0, 255)
    panel = np.repeat(np.repeat(base, scale, 0), scale, 1)
    panels = []
    for q, col in ((yx[0], (0, 255, 0)), (yx[1], (255, 0, 255))):
        img = panel.copy()
        q = (q - (y0, x0)) * scale + scale // 2
        q = q[(q >= 0).all(1) & (q[:, 0] < img.shape[0]) & (q[:, 1] < img.shape[1])]
        for dy in range(-dot, dot + 1):
            for dx in range(-dot, dot + 1):
                img[np.clip(q[:, 0] + dy, 0, img.shape[0] - 1), np.clip(q[:, 1] + dx, 0, img.shape[1] - 1)] = col
        panels.append(img)
    sep = np.full((panel.shape[0], 6, 3), 255, np.float32)
    Image.fromarray(np.concatenate([panels[0], sep, panels[1]], 1).astype(np.uint8)).save(path)
    return path


def plane_segments(g, z):
    """The polyline where the grid surface g (H,W,3 zyx) crosses the plane Z=z: (M, 2, 2) segments of (y, x)
    points, by marching squares over the grid cells (a cell with a hole corner is skipped)."""
    if g.shape[0] < 2 or g.shape[1] < 2:
        return np.zeros((0, 2, 2), np.float32)
    f = g[..., 0] - z
    P = g[..., 1:]
    c = [(slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)),
         (slice(1, None), slice(1, None)), (slice(1, None), slice(None, -1))]
    fs = [f[s] for s in c]
    ps = [P[s] for s in c]
    cross, pts = [], []
    for e in range(4):
        fa, fb, pa, pb = fs[e], fs[(e + 1) % 4], ps[e], ps[(e + 1) % 4]
        x = np.isfinite(fa) & np.isfinite(fb) & ((fa < 0) != (fb < 0))
        t = np.where(x, fa / np.where(x, fa - fb, 1.0), 0.0)
        cross.append(x)
        pts.append(pa + t[..., None] * (pb - pa))
    cross, pts = np.stack(cross), np.stack(pts)           # (4,h,w), (4,h,w,2)
    cnt = cross.sum(0)
    order = np.argsort(~cross, axis=0, kind="stable")      # crossing edges first, in edge order
    sp = np.take_along_axis(pts, order[..., None], 0)
    segs = [np.stack([sp[0][cnt >= 2], sp[1][cnt >= 2]], 1), np.stack([sp[2][cnt == 4], sp[3][cnt == 4]], 1)]
    return np.concatenate(segs).astype(np.float32)


def _draw_segments(img, segs, y0, x0, f, color, width=1):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for s in segs:
        (ya, xa), (yb, xb) = (s - (y0, x0)) * f
        d.line([(float(xa), float(ya)), (float(xb), float(yb))], fill=color, width=width)


def slab_view(ext, max_px=2000, min_px=900):
    """(stride, upscale) of a slab picture whose larger side spans `ext` voxels."""
    stride = max(1, -(-int(ext) // max_px))
    return stride, (max(1, min_px // max(int(ext), 1)) if stride == 1 else 1)


def slab_pngs(out_dir, zs, plane_fn, ct_fn, extent, segs_before, segs_after, max_px=2000, min_px=900):
    """Before | after cross-sections of the slab at each absolute z in `zs` over extent = ((y0, x0), (y1, x1))
    (absolute voxels): CT grayscale (`ct_fn(z, y0, y1, x0, x1, stride)` -> uint8 at that stride, or None),
    recto tinted red and verso blue (`plane_fn(z, stride)` -> (recto, verso) 0..1 over the extent at that
    stride), the unrefined surfaces' polylines magenta (left) and the refined ones green (right, over a dim
    magenta trace of the originals). segs_before/after: {z: [(M,2,2) (y, x) segments]}."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = np.asarray(extent[0], np.int64), np.asarray(extent[1], np.int64)
    outs = []
    if (hi <= lo).any():
        return outs
    stride, up = slab_view(int((hi - lo).max()), max_px, min_px)
    for z in zs:
        sb, sa = segs_before.get(z, []), segs_after.get(z, [])
        if not sum(len(x) for x in sb + sa):
            continue
        vz, wz = plane_fn(z, stride)
        ctz = ct_fn(int(z), int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), stride) if ct_fn else None
        base = np.zeros(vz.shape + (3,), np.float32)
        if ctz is not None:
            ctz = np.asarray(ctz, np.float32)
            iy = np.minimum(np.arange(vz.shape[0]) * ctz.shape[0] // max(vz.shape[0], 1), ctz.shape[0] - 1)
            ix = np.minimum(np.arange(vz.shape[1]) * ctz.shape[1] // max(vz.shape[1], 1), ctz.shape[1] - 1)
            base += ctz[iy][:, ix][..., None] * 0.8
        base[..., 0] += 150 * vz
        base[..., 2] += 150 * wz
        base = np.clip(base, 0, 255).astype(np.uint8)
        if up > 1:
            base = np.repeat(np.repeat(base, up, 0), up, 1)
        f = up / stride
        left, right = Image.fromarray(base), Image.fromarray(base.copy())
        for s in sb:
            _draw_segments(left, s, lo[0], lo[1], f, (255, 0, 255), 2 if up > 1 else 1)
            _draw_segments(right, s, lo[0], lo[1], f, (110, 0, 110), 1)
        for s in sa:
            _draw_segments(right, s, lo[0], lo[1], f, (0, 255, 0), 2 if up > 1 else 1)
        sep = Image.new("RGB", (6, base.shape[0]), (255, 255, 255))
        img = Image.new("RGB", (2 * base.shape[1] + 6, base.shape[0]))
        img.paste(left, (0, 0))
        img.paste(sep, (base.shape[1], 0))
        img.paste(right, (base.shape[1] + 6, 0))
        p = os.path.join(out_dir, f"slab_z{int(z)}.png")
        img.save(p)
        outs.append(p)
    return outs


def pick_crops(cd, n, win=128, cell=64):
    """Up to n crop sites for one surface from its per-level data cd = {z: (segs_b, segs_a, [yx], [|move|])}:
    one per z level (spread over the slab), alternately at the densest `cell`-voxel cell and at the cell
    with the largest mean |move| (among cells with >= 25% of the densest's points), each at least `win`
    voxels in yx from the sites already picked. Returns [(z, cy, cx, local mean |move|, why)]."""
    out = []
    levels = [z for z in sorted(cd) if cd[z][2]]
    for k, z in enumerate(levels[:n]):
        P, M = np.concatenate(cd[z][2]), np.concatenate(cd[z][3])
        key = np.floor(P / cell).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        mm = np.bincount(inv, weights=M) / cnt
        cyx = np.stack([np.bincount(inv, weights=P[:, i]) / cnt for i in (0, 1)], -1)
        by_move = k % 2 == 1
        score = np.where(cnt >= 0.25 * cnt.max(), mm, -1.0) if by_move else cnt.astype(np.float64)
        order = np.argsort(-score, kind="stable")
        sep = np.array([min([max(abs(cyx[i, 0] - q[1]), abs(cyx[i, 1] - q[2])) for q in out] or [np.inf])
                        for i in order])
        ok = np.nonzero(sep >= win)[0]
        i = order[ok[0]] if len(ok) else order[int(np.argmax(sep))]   # a small surface: the farthest site
        near = (np.abs(P - cyx[i]) <= win / 2).all(-1)
        out.append((z, float(cyx[i, 0]), float(cyx[i, 1]), float(M[near].mean()),
                    "largest moves" if by_move else "densest"))
    return out


def crop_png(vz, wz, ctz, sb, sa, others, y0, x0, scale, label):
    """One before | after crop (a PIL image): CT gray, recto red, verso blue, this surface's polylines
    (magenta before; green after over a dim magenta trace), the other surfaces' dim, and a label."""
    from PIL import Image, ImageDraw
    n = vz.shape[0]
    base = np.zeros((n, n, 3), np.float32)
    if ctz is not None:
        c = np.asarray(ctz, np.float32)[:n, :n]
        base[:c.shape[0], :c.shape[1]] += c[..., None] * 0.8
    base[..., 0] += 150 * vz
    if wz is not None:
        base[..., 2] += 150 * wz
    base = np.repeat(np.repeat(np.clip(base, 0, 255).astype(np.uint8), scale, 0), scale, 1)
    left, right = Image.fromarray(base), Image.fromarray(base.copy())
    lo, hi = np.array([y0, x0]) - 2, np.array([y0, x0]) + n + 2

    def inwin(ss):
        ss = [q for q in ss if len(q)]
        if not ss:
            return np.zeros((0, 2, 2), np.float32)
        q = np.concatenate(ss)
        m = q.mean(1)
        return q[((m >= lo) & (m < hi)).all(-1)]
    for ob, oa in others:
        _draw_segments(left, inwin(ob), y0, x0, scale, (120, 120, 120), 1)
        _draw_segments(right, inwin(oa), y0, x0, scale, (0, 110, 0), 1)
    b, a = inwin(sb), inwin(sa)
    _draw_segments(left, b, y0, x0, scale, (255, 0, 255), 2)
    _draw_segments(right, b, y0, x0, scale, (110, 0, 110), 1)
    _draw_segments(right, a, y0, x0, scale, (0, 255, 0), 2)
    W = base.shape[1]
    img = Image.new("RGB", (2 * W + 6, base.shape[0] + 16), (255, 255, 255))
    img.paste(left, (0, 16))
    img.paste(right, (W + 6, 16))
    ImageDraw.Draw(img).text((3, 2), label + "   before | after", fill=(0, 0, 0))
    return img

def hist_png(path, moves, far, bins=None, cell=(260, 120)):
    """Small multiples of the signed normal displacement (voxels) per surface: {name: (N,) array}."""
    from PIL import Image, ImageDraw
    names = sorted(moves)
    if not names:
        return None
    bins = np.linspace(-far, far, 2 * int(far) + 1) if bins is None else bins
    cols = min(4, len(names))
    rows = -(-len(names) // cols)
    cw, ch = cell
    img = Image.new("RGB", (cols * cw, rows * ch), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, nm in enumerate(names):
        x0, y0 = (i % cols) * cw, (i // cols) * ch
        v = np.asarray(moves[nm], np.float32)
        v = v[np.isfinite(v)]
        h, _ = np.histogram(np.clip(v, bins[0], bins[-1]), bins)
        hm = max(int(h.max()), 1) if len(h) else 1
        bw = (cw - 20) / max(len(h), 1)
        for j, c in enumerate(h):
            bh = (ch - 40) * c / hm
            d.rectangle([x0 + 10 + j * bw, y0 + ch - 20 - bh, x0 + 10 + (j + 1) * bw - 1, y0 + ch - 20],
                        fill=(60, 120, 200))
        zx = x0 + 10 + (cw - 20) * (0 - bins[0]) / (bins[-1] - bins[0])
        d.line([(zx, y0 + 18), (zx, y0 + ch - 20)], fill=(200, 0, 0))
        mean = float(np.abs(v).mean()) if len(v) else 0.0
        d.text((x0 + 6, y0 + 3), f"{nm[:28]}  |d|={mean:.2f} n={len(v)}", fill=(0, 0, 0))
        d.text((x0 + 6, y0 + ch - 16), f"{bins[0]:.0f}", fill=(0, 0, 0))
        d.text((x0 + cw - 26, y0 + ch - 16), f"{bins[-1]:+.0f}", fill=(0, 0, 0))
    img.save(path)
    return path


# ============================================================================================== run

def load_frame(args):
    """The legacy -> fine frame the CLI asks for: --transform / --legacy-scale beat the data agent's frame.py,
    which beats nothing (None: a legacy surface is then an error)."""
    if args.transform:
        if str(args.transform).endswith(".json"):
            return transform_json_frame(args.transform, args.transform_direction)
        M = np.asarray([float(v) for v in str(args.transform).replace(";", ",").split(",") if v.strip()])
        if M.size != 12:
            raise SystemExit("--transform: a .json file or 12 comma-separated numbers (3x4 xyz legacy->fine)")
        return affine_frame(M.reshape(3, 4)) if args.transform_direction != "fine_to_legacy" else \
            affine_frame(np.linalg.inv(np.vstack([M.reshape(3, 4), [0, 0, 0, 1]]))[:3])
    if args.legacy_scale is not None:
        return scale_offset_frame(args.legacy_scale, args.legacy_offset or (0, 0, 0))
    return module_frame()


def run(args):
    from rvsm import axis as AX
    log = lambda s: print(s, flush=True)   # noqa: E731
    o_s, S_s, _ = store_box(args.recto)
    lo, shape = o_s.copy(), S_s.copy()
    if args.z0 is not None:
        z1 = min(int(args.z0) + int(args.dz), int(o_s[0] + S_s[0]))
        lo[0] = max(int(args.z0), int(o_s[0]))
        shape[0] = z1 - lo[0]
        if shape[0] <= 0:
            raise SystemExit(f"--z0 {args.z0} --dz {args.dz} misses the store's z range {o_s[0]}..{o_s[0] + S_s[0]}")
    if args.box:
        lo, shape = np.asarray(args.box[:3], np.int64), np.asarray(args.box[3:], np.int64)
    o, s = lo.astype(np.float32), shape.astype(np.float32)
    log(json.dumps({"box": [*lo.tolist(), *shape.tolist()], "recto": args.recto, "verso": args.verso}))
    from rvsm import ladder
    attrs = dict(ladder.open_zarr(args.recto).attrs)
    umb = args.umbilicus or attrs.get("umbilicus")
    if not umb:
        raise SystemExit("--umbilicus is required (the store names none)")
    ax = AX.load(umb)
    ct = None if args.no_ct else (args.ct or attrs.get("volume") or None)
    legacy = None

    # ---- surfaces: only the published rows/cols that touch the box are ever held (read_surface_box)
    todo = []   # (name, segid, dir)
    for d in args.surface or []:
        nm = os.path.basename(d.rstrip("/"))
        todo.append((nm[:-7] if nm.endswith(".tifxyz") else nm, nm, d))
    if args.paths:
        todo += find_surfaces(args.paths, tuple(p for p in args.prefer.split(",") if p))
    margin = float(args.far) + 16
    surf = []
    for name, sid, d in todo:
        kind = frame_kind(d, args.src_frame)
        if kind == "coarse":
            continue
        if kind == "legacy":
            if legacy is None:
                legacy = load_frame(args)
                if legacy is None:
                    raise SystemExit(f"{d} is in the legacy frame: pass --transform transform.json (or "
                                     "--legacy-scale/--legacy-offset), or provide rvsm/tools/refine/frame.py")
            fr = legacy
        else:
            fr = IDENTITY
        try:   # cheap bbox test first (meta bbox corners through the frame)
            b = np.asarray(json.load(open(os.path.join(d, "meta.json")))["bbox"], np.float64)[:, ::-1]
            corners = np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            cf = fr.to_fine(corners)
            m = float(args.far) + 64
            if (cf.max(0) < o - m).any() or (cf.min(0) > o + s + m).any():
                continue
        except Exception:  # noqa: BLE001  - no bbox: read it
            pass
        got = read_surface_box(d, fr, o, s, margin=margin)
        if got is None:
            continue
        crop, rc, full_shape = got
        k = np.isfinite(crop).all(-1) & ((crop >= o) & (crop < o + s)).all(-1)
        up = auto_up(crop, args.pitch) if args.up == 0 else int(args.up)
        if int(k.sum()) * up * up < args.min_pts:
            continue
        surf.append({"name": name, "segid": sid, "dir": d, "frame": fr, "rc": rc, "up": up, "crop": crop,
                     "shape": full_shape, "n_published_inside": int(k.sum()),
                     "own_r": [], "own_c": [], "own_v": []})
    if not surf:
        raise SystemExit("no surface crosses the box")
    log(json.dumps({"surfaces": [(x["name"], x["frame"].name, x["up"], x["n_published_inside"]) for x in surf]}))

    # ---- anchors (converted to the fine frame once; applied per piece)
    anchors = {}
    if args.anchors:
        live = read_anchors(args.anchors)
        afr = IDENTITY if args.anchor_frame == "fine" else (legacy or load_frame(args))
        if afr is None:
            raise SystemExit("--anchor-frame legacy needs a legacy frame (--transform)")
        for i, x in enumerate(surf):
            mine = [a for a in live if anchor_matches(a, (x["name"], x["segid"], os.path.basename(x["dir"].rstrip("/"))))]
            if mine:
                anchors[i] = [dict(a, from_zyx=afr.to_fine(np.array([a["from_zyx"]]))[0].tolist(),
                                   to_zyx=afr.to_fine(np.array([a["to_zyx"]]))[0].tolist()) for a in mine]
                log(json.dumps({"anchors": x["name"], "n": len(mine)}))

    # ---- tiles: every published node is refined in the tile whose core holds it, with a halo around the core
    # wide enough for everything that couples nodes (neighbour sheets, the smoothing and anchor kernels, the
    # snap's reach). Only one tile's stores and dense grids are in memory at a time.
    sigma_vox = float(args.sigma_vox) if args.sigma_vox is not None else 2.0 * float(args.pitch)
    sigma_fin = float(args.sigma_final) if args.sigma_final is not None else sigma_vox / 2.0
    log(json.dumps({"sigma_vox": sigma_vox, "sigma_final_vox": sigma_fin, "pitch": args.pitch, "far": args.far,
                    "iters": args.iters, "local_normal": bool(args.local_normal)}))
    halo = max(int(args.halo), 3 * int(args.far) + 8, 44, int(np.ceil(3 * max(sigma_vox, sigma_fin))))
    tile = int(args.tile) if args.tile and args.tile > 0 else int(max(shape[1], shape[2]))
    rpad = int(args.far) + 4                     # the snap samples +-far around a halo point
    circ = not args.eval_store or os.path.abspath(args.eval_store) == os.path.abspath(args.recto)
    if circ:
        log("WARNING: metrics below are measured on the SAME store the surfaces were refined against -- the "
            "gain is circular. Pass --eval-store with a different store (another teacher / the student) for "
            "a real before/after.")
    zs = np.linspace(lo[0] + 0.5 + shape[0] / (2 * args.slices), lo[0] + shape[0] - 0.5 - shape[0] / (2 * args.slices),
                     args.slices) if args.slices > 1 else [lo[0] + shape[0] / 2]
    zs = [float(int(z)) + 0.5 for z in zs]
    ext_lo, ext_hi = lo[1:].copy(), lo[1:] + shape[1:]
    pstride, _ = slab_view(int((ext_hi - ext_lo).max()))
    cshape = tuple(int(v) for v in -(-(ext_hi - ext_lo) // pstride))
    canv = {z: [np.zeros(cshape, np.uint8), np.zeros(cshape, np.uint8)] for z in zs}
    segs_b, segs_a = {z: [] for z in zs}, {z: [] for z in zs}
    moves = {x["name"]: [] for x in surf}
    ncrop = 0 if args.no_surface_png else max(int(args.crops), 0)
    cz_lo, cz_hi = lo[0] + args.taper + 4, lo[0] + shape[0] - args.taper - 4
    czs = [float(int(z)) + 0.5 for z in (np.linspace(cz_lo, cz_hi, ncrop) if ncrop > 1 and cz_hi > cz_lo
                                         else [lo[0] + shape[0] / 2] * min(ncrop, 1))]
    czs = sorted(set(czs))
    # per surface and crop level: (before segments, after segments, band points yx, band |move|)
    cdat = {x["name"]: {z: ([], [], [], []) for z in czs} for x in surf}
    met = {x["name"]: {"before": [], "after": []} for x in surf}
    stats = []
    cores = [(y0, min(y0 + tile, int(lo[1] + shape[1])), x0, min(x0 + tile, int(lo[2] + shape[2])))
             for y0 in range(int(lo[1]), int(lo[1] + shape[1]), tile)
             for x0 in range(int(lo[2]), int(lo[2] + shape[2]), tile)]
    log(json.dumps({"tiles": len(cores), "tile": tile, "halo": halo}))
    box_lo, box_hi = lo.astype(np.int64), (lo + shape).astype(np.int64)
    for ti, (cy0, cy1, cx0, cx1) in enumerate(cores):
        tlo = np.array([box_lo[0], max(cy0 - halo, box_lo[1]), max(cx0 - halo, box_lo[2])], np.int64)
        thi = np.array([box_hi[0], min(cy1 + halo, box_hi[1]), min(cx1 + halo, box_hi[2])], np.int64)
        # pieces: per surface, the column runs of its published crop that reach into this tile (each with its
        # own row range, so a sheet that crosses the slab at a slant never becomes one huge rectangle)
        pieces = []
        for si, x in enumerate(surf):
            for (r0, r1, c0, c1) in column_pieces(x["crop"], tlo - margin, thi + margin):
                pc = x["crop"][r0:r1, c0:c1]
                dense = upsample(pc, x["up"])
                if int((np.isfinite(dense).all(-1) & ((dense >= tlo) & (dense < thi)).all(-1)).sum()) == 0:
                    continue
                pieces.append({"si": si, "r0": r0, "c0": c0, "pub": dense})
        if not pieces:
            continue
        rlo = np.array([box_lo[0], max(int(tlo[1]) - rpad, box_lo[1]), max(int(tlo[2]) - rpad, box_lo[2])], np.int64)
        rhi = np.array([box_hi[0], min(int(thi[1]) + rpad, box_hi[1]), min(int(thi[2]) + rpad, box_hi[2])], np.int64)
        rs = rhi - rlo
        V = read_box(args.recto, rlo, rs)
        W = read_box(args.verso, rlo, rs) if args.verso else None
        thick = read_box(args.thickness_store, rlo, rs) if args.thickness_store else None
        ct_mask = read_ct(ct, rlo, rs) if (ct and args.ct_mask) else None
        holds, g0 = [], []
        for pz in pieces:
            x = surf[pz["si"]]
            g = pz["pub"]
            h = None
            if pz["si"] in anchors:
                up, pr0, pc0 = x["up"], x["rc"][0] + pz["r0"], x["rc"][1] + pz["c0"]
                rc_map = lambda rc, pr0=pr0, pc0=pc0, up=up: ((float(rc[0]) - pr0) * up, (float(rc[1]) - pc0) * up)  # noqa: E731
                pp = pitch(g)
                fld, top = anchor_field(g, anchors[pz["si"]], sigma=args.anchor_sigma, clip=args.anchor_clip,
                                        rc_map=rc_map, max_dist=2 * (pp if np.isfinite(pp) else 4.0) + 2)
                if top.max() > 0:
                    g, h = g + fld, top
            g0.append(g)
            holds.append(h)
        pit = [pitch(g) for g in g0]
        sig = [sigma_vox / max(p, 1e-3) if np.isfinite(p) else 2.0 for p in pit]
        sig_f = [sigma_fin / max(p, 1e-3) if np.isfinite(p) else 1.0 for p in pit]
        g1, st = refine_many(g0, V, rlo.astype(np.float32), ax, far=args.far, sigma=sig,
                             iters=args.iters, thr=args.thr, ct=ct_mask, W=W, thick=thick, T=args.thickness,
                             verso_thr=args.verso_thr, verso_beta=args.verso_beta,
                             verso_block=np.inf if args.verso_block is None else args.verso_block,
                             verso_margin=args.verso_margin, holds=holds, taper=args.taper,
                             box=(o, s), sigma_final=sig_f, local_normal=bool(args.local_normal), log=None)
        stats.append({"tile": [cy0, cy1, cx0, cx1], "pieces": len(pieces), "iters": st})
        log(json.dumps({"tile": ti + 1, "of": len(cores), "core_yx": [cy0, cy1, cx0, cx1], "pieces": len(pieces),
                        **{k: v for k, v in st[-1].items() if k != "iter"}}))
        Ve = V if circ else read_box(args.eval_store, rlo, rs)
        for pz, b in zip(pieces, g1):
            x, a, up = surf[pz["si"]], pz["pub"], surf[pz["si"]]["up"]
            ina = np.isfinite(a).all(-1) & ((a >= o) & (a < o + s)).all(-1)
            core = ina & (a[..., 1] >= cy0) & (a[..., 1] < cy1) & (a[..., 2] >= cx0) & (a[..., 2] < cx1)
            if not core.any():
                continue
            # published nodes this tile owns -> the surface's result
            cn = core[::up, ::up]
            rr, cc = np.nonzero(cn)
            x["own_r"].append((rr + pz["r0"]).astype(np.int32))
            x["own_c"].append((cc + pz["c0"]).astype(np.int32))
            x["own_v"].append(b[::up, ::up][cn].astype(np.float32))
            n0 = normals(a, ax)
            dvg = ((b - a) * np.nan_to_num(n0)).sum(-1)
            moves[x["name"]].append(dvg[core].astype(np.float32))
            for z in czs:
                cd = cdat[x["name"]][z]
                for g, dst in ((a, cd[0]), (b, cd[1])):
                    sg = plane_segments(g, z)
                    if len(sg):
                        mid = sg.mean(1)
                        keep = (mid[:, 0] >= cy0) & (mid[:, 0] < cy1) & (mid[:, 1] >= cx0) & (mid[:, 1] < cx1)
                        if keep.any():
                            dst.append(sg[keep])
                bk = core & (np.abs(a[..., 0] - z) <= 2.5)
                if bk.any():
                    cd[2].append(a[bk][:, 1:].astype(np.float32))
                    cd[3].append(np.abs(dvg[bk]).astype(np.float32))
            for z in zs:
                for g, dst in ((a, segs_b), (b, segs_a)):
                    sg = plane_segments(g, z)
                    if len(sg):
                        mid = sg.mean(1)
                        keep = (mid[:, 0] >= cy0) & (mid[:, 0] < cy1) & (mid[:, 1] >= cx0) & (mid[:, 1] < cx1)
                        if keep.any():
                            dst[z].append(sg[keep])
            for nm, g in (("before", a), ("after", b)):
                mm = surface_metrics(Ve, rlo, g, ax, thr=args.thr, mask=core)
                if mm.get("n_points", 0):
                    met[x["name"]][nm].append(mm)
        # the slab pictures' recto/verso planes, filled from this tile's core
        iy = np.arange(cshape[0]) * pstride + ext_lo[0]
        ix = np.arange(cshape[1]) * pstride + ext_lo[1]
        sy, sx = np.nonzero((iy >= cy0) & (iy < cy1))[0], np.nonzero((ix >= cx0) & (ix < cx1))[0]
        if len(sy) and len(sx):
            for z in zs:
                zi = int(z - rlo[0])
                for arr, c in ((V, 0), (W, 1)):
                    if arr is not None:
                        canv[z][c][np.ix_(sy, sx)] = arr[zi][np.ix_(iy[sy] - rlo[1], ix[sx] - rlo[2])]
        del V, W, thick, Ve, ct_mask, g0, g1, pieces

    # ---- write: one full published grid in memory at a time
    os.makedirs(args.out, exist_ok=True)
    for x in surf:
        dv = np.concatenate(x_m) if (x_m := moves[x["name"]]) else np.zeros(0, np.float32)
        moves[x["name"]] = dv
        new = x["crop"].copy()
        if x["own_r"]:
            new[np.concatenate(x["own_r"]), np.concatenate(x["own_c"])] = np.concatenate(x["own_v"])
        x["own_r"] = x["own_c"] = x["own_v"] = None
        note = {"recto": str(args.recto), "verso": str(args.verso or ""), "box": [*lo.tolist(), *shape.tolist()],
                "far": args.far, "sigma_vox": sigma_vox, "sigma_final_vox": sigma_fin, "iters": args.iters, "thr": args.thr, "up": x["up"],
                "joint_with": len(surf), "frame": x["frame"].name, "anchors": bool(anchors.get(surf.index(x))),
                "tile": tile, "halo": halo}
        src = E.read_surface(x["dir"])
        full = write_back(src, x["rc"], 1, x["crop"], new, x["frame"], inplace=True)
        del src, new
        copy_tifxyz(x["dir"], os.path.join(args.out, x["name"] + ".before"), {"unrefined_input": True})
        write_tifxyz(x["dir"], os.path.join(args.out, x["name"]), full, note)
        del full
        log(json.dumps({"wrote": os.path.join(args.out, x["name"]), "mean_abs_move": float(np.abs(dv).mean()) if len(dv) else 0.0,
                        "p95_abs_move": float(np.percentile(np.abs(dv), 95)) if len(dv) else 0.0}))

    # ---- pictures
    png = args.png_dir or os.path.join(args.out, "png")
    os.makedirs(png, exist_ok=True)
    hist_png(os.path.join(png, "displacement_hist.png"), moves, args.far)
    ct_cache = {}

    def ct_fn(z, y0, y1, x0, x1, stride=1):
        if not ct:
            return None
        key = (z, y0, y1, x0, x1, stride)
        if key not in ct_cache:
            try:
                ct_cache[key] = read_ct(ct, (z, y0, x0), (1, y1 - y0, x1 - x0), down=stride)[0]
            except Exception as e:  # noqa: BLE001 - a picture without CT beats no picture
                log(f"CT read failed ({e!r}); drawing without CT")
                ct_cache[key] = None
        return ct_cache[key]

    def plane_fn(z, stride):
        assert stride == pstride
        return canv[z][0].astype(np.float32) / 255.0, canv[z][1].astype(np.float32) / 255.0
    for p in slab_pngs(png, zs, plane_fn, ct_fn, (ext_lo, ext_hi), segs_b, segs_a):
        log(p)
    del canv, segs_b, segs_a
    if ncrop:
        def read_plane(path, z, y0, x0, n):
            return read_box(path, (int(z), int(y0), int(x0)), (1, n, n))[0].astype(np.float32) / 255.0 if path else None
        n = args.crop_px // args.crop_scale
        for x in surf:
            picks = pick_crops(cdat[x["name"]], ncrop, win=n)
            imgs = []
            for k, (z, cy, cx, mv, why) in enumerate(picks):
                y0, x0 = int(cy) - n // 2, int(cx) - n // 2
                others = [(cdat[nm][z][0], cdat[nm][z][1]) for nm in cdat if nm != x["name"]]
                im = crop_png(read_plane(args.recto, z, y0, x0, n), read_plane(args.verso, z, y0, x0, n),
                              ct_fn(int(z), y0, y0 + n, x0, x0 + n) if ct else None,
                              cdat[x["name"]][z][0], cdat[x["name"]][z][1], others, y0, x0, args.crop_scale,
                              f"{x['name'][:28]}  zyx {int(z)} {int(cy)} {int(cx)}  mean|move| {mv:.1f} vox ({why})")
                im.save(os.path.join(png, f"{x['name']}_crop{k}.png"))
                imgs.append(im)
            if imgs and args.crop_montage:
                from PIL import Image
                wd, ht = max(i.size[0] for i in imgs), sum(i.size[1] for i in imgs) + 4 * (len(imgs) - 1)
                mt = Image.new("RGB", (wd, ht), (255, 255, 255))
                yy = 0
                for i in imgs:
                    mt.paste(i, (0, yy))
                    yy += i.size[1] + 4
                mt.save(os.path.join(png, f"{x['name']}_crops.png"))
            if imgs:
                log(json.dumps({"crops": x["name"], "n": len(imgs)}))
    del cdat

    # ---- metrics: per surface, the tiles' numbers pooled by their point counts
    tot = {"before": {}, "after": {}}
    move_rep = {}
    for x in surf:
        dv = np.abs(moves[x["name"]])
        mr = {"points": int(len(dv)),
              "frac_moved_gt2": round(float((dv > 2).mean()), 4) if len(dv) else 0.0,
              "mean_abs_move": round(float(dv.mean()), 3) if len(dv) else 0.0,
              "median_abs_move": round(float(np.median(dv)), 3) if len(dv) else 0.0}
        if len(dv) and mr["median_abs_move"] < 0.5:
            mr["warning"] = "over-smoothed? median |move| < 0.5 voxel"
        elif len(dv) and mr["median_abs_move"] > args.far / 2:
            mr["warning"] = f"runaway? median |move| > far/2 = {args.far / 2:g} voxels"
        if "warning" in mr:
            log(f"WARNING: {x['name']}: {mr['warning']} (median {mr['median_abs_move']}, mean {mr['mean_abs_move']})")
        move_rep[x["name"]] = mr
        rec = {"surface": x["name"], "eval_store": str(args.eval_store or args.recto), "circular": bool(circ),
               "tiles": len(met[x["name"]]["before"]), "moves": mr}
        for nm in ("before", "after"):
            ms = met[x["name"]][nm]
            npt = sum(m["n_points"] for m in ms)
            rec[nm] = {}
            for q in EVAL_KEYS:
                vals = [(m[q], m["n_points"]) for m in ms if q in m and np.isfinite(m[q])]
                if vals:
                    v = sum(a * w for a, w in vals) / max(sum(w for _, w in vals), 1)
                    rec[nm][q] = round(v, 4)
                    tot[nm].setdefault(q, []).append((v, npt))
            rec["points"] = npt
        log(json.dumps(rec))
    pooled = {nm: {q: round(sum(v * w for v, w in vals) / max(sum(w for _, w in vals), 1), 4) for q, vals in t.items()}
              for nm, t in tot.items()}
    log(json.dumps({"pooled": pooled, "circular": bool(circ)}))
    with open(os.path.join(args.out, "refine_report.json"), "w") as f:
        json.dump({"box": [*lo.tolist(), *shape.tolist()], "tile": tile, "halo": halo, "moves": move_rep,
                   "sigma_vox": sigma_vox, "sigma_final_vox": sigma_fin, "stats": stats,
                   "pooled": pooled, "circular": bool(circ), "surfaces": [x["name"] for x in surf]}, f, indent=1)
    return 0

def parser():
    ap = argparse.ArgumentParser(prog="rvsm refine", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recto", required=True, help="recto probability store (zarr v3 volcomp, fine frame)")
    ap.add_argument("--verso", help="verso probability store over the same box (enables the verso term)")
    ap.add_argument("--thickness-store", help="thickness store (code * 0.25 voxels, 0 = no data)")
    ap.add_argument("--thickness", type=float, help="a fixed sheet thickness in voxels (else from the peaks)")
    ap.add_argument("--paths", help="directory of published surfaces (<segid>/<frame>.tifxyz or <name>/x.tif)")
    ap.add_argument("--surface", action="append", help="one tifxyz directory (repeatable)")
    ap.add_argument("--prefer", default="2.4um,7.91um", help="tifxyz variant order under --paths")
    ap.add_argument("--src-frame", default="fine", choices=("fine", "legacy"),
                    help="the frame of a tifxyz whose name/meta does not say")
    ap.add_argument("--transform", help="transform.json, or 12 numbers: 3x4 xyz affine")
    ap.add_argument("--transform-direction", default="auto", choices=("auto", "fine_to_legacy", "legacy_to_fine"))
    ap.add_argument("--legacy-scale", type=float, nargs="+", help="fine = legacy * scale + offset (zyx)")
    ap.add_argument("--legacy-offset", type=float, nargs=3)
    ap.add_argument("--umbilicus", help="umbilicus json/txt in fine (rung-2) voxels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--z0", type=int, help="fine-frame slab start (cross-section mode)")
    ap.add_argument("--dz", type=int, default=128)
    ap.add_argument("--box", type=int, nargs=6, help="Z Y X DZ DY DX instead of the store / slab box")
    ap.add_argument("--far", type=int, default=12)
    ap.add_argument("--sigma-vox", type=float, default=None,
                    help="displacement smoothing, voxels along the sheet (default 2 x --pitch: follows locally; "
                         "usrm2 used 40 = 2 cells of its 20-voxel grid)")
    ap.add_argument("--sigma-final", type=float, default=None,
                    help="smoothing of the LAST iteration only, voxels (default sigma-vox / 2)")
    ap.add_argument("--local-normal", action=argparse.BooleanOptionalAction, default=True,
                    help="recompute the normals from the refined grid every iteration (default on)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--verso-thr", type=float)
    ap.add_argument("--verso-beta", type=float, default=0.5)
    ap.add_argument("--verso-block", type=float, help="penalty instead of rejection for a verso-first candidate")
    ap.add_argument("--verso-margin", type=float, default=0.5)
    ap.add_argument("--up", type=int, default=0, help="grid upsampling (0 = auto to --pitch voxels)")
    ap.add_argument("--pitch", type=float, default=4.0)
    ap.add_argument("--taper", type=float, default=8.0, help="voxels over which moves fade out at the box faces")
    ap.add_argument("--min-pts", type=int, default=50)
    ap.add_argument("--tile", type=int, default=1024,
                    help="yx core of a tile, voxels (0 = the whole box at once); bounds the memory: one tile's "
                         "stores and grids at a time")
    ap.add_argument("--halo", type=int, default=160,
                    help="voxels of context around a tile core (>= 3x the smoothing and anchor sigmas)")
    ap.add_argument("--anchors", help="render3d anchors.jsonl")
    ap.add_argument("--anchor-frame", default="fine", choices=("fine", "legacy"))
    ap.add_argument("--anchor-sigma", type=float, default=48.0, help="voxels")
    ap.add_argument("--anchor-clip", type=float, default=64.0, help="voxels")
    ap.add_argument("--ct", help="CT pyramid (fine frame) for the pictures (default: the store's 'volume' attr)")
    ap.add_argument("--no-ct", action="store_true", help="draw the pictures without CT")
    ap.add_argument("--ct-mask", action="store_true", help="also zero the evidence where the CT is masked (reads the box)")
    ap.add_argument("--eval-store", help="a DIFFERENT store to measure before/after on")
    ap.add_argument("--slices", type=int, default=4)
    ap.add_argument("--png-dir")
    ap.add_argument("--no-surface-png", action="store_true", help="no per-surface crops")
    ap.add_argument("--crops", type=int, default=6,
                    help="per surface, comparison crops at different z / along-surface positions (densest and "
                         "largest-move places, alternately): png/<surface>_crop<k>.png")
    ap.add_argument("--crop-px", type=int, default=384, help="crop panel size, pixels")
    ap.add_argument("--crop-scale", type=int, default=3, help="pixels per voxel in a crop")
    ap.add_argument("--crop-montage", action="store_true", help="also png/<surface>_crops.png: all crops stacked")
    return ap


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
