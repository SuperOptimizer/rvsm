"""Evaluation: what a predicted recto band is worth, measured three ways.

1. AGAINST HUMAN MESHES (`surface_rows` / `pool` / `bootstrap`), when a `tifxyz` directory is given.
   The published meshes lie ON the recto face, so the probability band should peak at offset 0 along the
   surface normal, and every number here is measured at the published surface points: recall within r
   voxels, the sub-voxel offset bias and its spread, a merge count (how often the ray crosses the
   threshold more than once), along-sheet continuity, and the Expected Run Length in MICROMETRES. The
   rows carry the sufficient statistics `pool()` needs, so the pooled numbers are exact, and
   `bootstrap()` resamples SURFACES (points inside one surface are far too correlated to resample
   individually) for the confidence intervals every gate in the run compares against.

2. AGAINST TOPOLOGY (`betti` / `betti_error`). Betti numbers of the CUBICAL COMPLEX in which every
   foreground voxel is a closed unit cube -- 26-connected foreground, 6-connected background. `b1` is
   derived from the exact Euler characteristic rather than counted, which gives the right NUMBER but
   says nothing about WHERE a loop is (one spurious handle can cancel one missing loop). Measured on the
   box interior, inside a band around the reference sheet, with the one-voxel mesh staircase dilated
   first so the two sides are on the same footing.

3. AGAINST ANOTHER STORE (`compare_stores`), which is what a held-out rvsm region has: no meshes, but a
   reference store from the round that produced it. Dice and overlap, the Betti-0/1 error on the
   interior, and an ERL-like run length along the reference's own SKELETON -- the same skeleton the
   training loss uses (`losses.skeleton`), so a break that costs run length here is a break the loss was
   also asked about.

`evaluate(pred_fn, box, tifxyz=None)` ties them together for one box.

`fit_curve` is the plateau fit the round gate reads: is this metric still moving, or is it done?

There is no noise ceiling on published masks and no CLI ceremony here: rvsm never looks up a published
store, and the evaluator is a function the trainer calls.
"""
import glob
import json
import os

import numpy as np

CHUNK = 64  # z-slices per slab of the Euler pass


# ============================================================================== published surfaces

def read_surface(d):
    """(H,W,3) zyx level-0 points of one tifxyz directory, invalid ones NaN."""
    import tifffile
    g = np.stack([np.asarray(tifffile.imread(f"{d}/{c}.tif"), np.float32) for c in "zyx"], -1)
    return np.where((g > 0).all(-1)[..., None], g, np.nan)


def _smooth(field, w, sigma):
    """Confidence-weighted Gaussian smoothing over the grid (normalised convolution); zero-weight cells
    get the neighbourhood's value."""
    from scipy.ndimage import gaussian_filter
    num = gaussian_filter(np.nan_to_num(field * w), sigma)
    den = gaussian_filter(w, sigma)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def filled(g, sigma=1.0):
    """The grid with only its holes (NaN) filled from their neighbourhood; valid cells are untouched."""
    ok = np.isfinite(g).all(-1)
    w = ok.astype(np.float32)
    f = np.stack([_smooth(np.where(ok, g[..., i], 0), w, sigma) for i in range(3)], -1)
    return np.where(ok[..., None], g, f)


def _normals_raw(g, ax):
    """(H,W,3) unit normals n = normalize(cross(t_v, t_u)), oriented so dot(n, radial) >= 0."""
    n = np.cross(np.gradient(g, axis=0), np.gradient(g, axis=1))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    cy, cx = np.interp(g[..., 0], ax[0], ax[1]), np.interp(g[..., 0], ax[0], ax[2])
    r = np.stack([np.zeros_like(cy), g[..., 1] - cy, g[..., 2] - cx], -1)
    return n * np.where((n * r).sum(-1, keepdims=True) < 0, -1.0, 1.0)


def normals(g, ax):
    """Hole-tolerant unit normals, oriented outward from the axis: computed on the hole-filled grid, so
    a NaN in the middle of a surface does not poison the gradient of its neighbours, and then masked back
    to the points that were real."""
    n = _normals_raw(filled(g), ax)
    return np.where(np.isfinite(g).all(-1)[..., None], n, np.nan)


def surfaces_in(tifxyz, origin, size, min_pts=200):
    """tifxyz directories whose points fall inside the box (bbox test, then a point count)."""
    o, s, out = np.asarray(origin, np.float32), np.asarray(size, np.float32), []
    for m in sorted(glob.glob(f"{tifxyz}/*/*/meta.json")) or sorted(glob.glob(f"{tifxyz}/*/meta.json")):
        b = np.asarray(json.load(open(m))["bbox"], np.float32)[:, ::-1]   # [[x,y,z]min,max] -> zyx
        if (b[1] < o).any() or (b[0] > o + s).any():
            continue
        d = os.path.dirname(m)
        g = read_surface(d)
        if (np.isfinite(g).all(-1) & ((g >= o) & (g < o + s)).all(-1)).sum() >= min_pts:
            out.append(d)
    return out


def surface_list(origin, size, tifxyz, ax, min_pts=200):
    """[(name, g (H,W,3), inside (H,W) bool, n (H,W,3))] for every published surface crossing the box.

    The same selection rule as `sites()`, so the per-surface breakdown covers exactly the surfaces the
    pooled numbers are computed from."""
    o, s, out = np.asarray(origin, np.float32), np.asarray(size, np.float32), []
    for d in surfaces_in(tifxyz, o, s, min_pts):
        g = read_surface(d)
        n = normals(g, ax)
        k = np.isfinite(g).all(-1) & ((g >= o) & (g < o + s)).all(-1) & np.isfinite(n).all(-1)
        if k.sum() >= min_pts:
            out.append((os.path.basename(d), g, k, n))
    return out


def sites(origin, size, tifxyz, ax, min_pts=200):
    """All published surface points (+normals) inside the box: (pts (N,3), nrm (N,3), per-surface counts)."""
    pts, nrm, counts = [], [], {}
    for name, g, k, n in surface_list(origin, size, tifxyz, ax, min_pts):
        pts.append(g[k])
        nrm.append(n[k])
        counts[name] = int(k.sum())
    return (np.concatenate(pts) if pts else np.zeros((0, 3), np.float32),
            np.concatenate(nrm) if nrm else np.zeros((0, 3), np.float32), counts)


# ================================================================================ sampling the band

def trilerp(V, q):
    """V (Z,Y,X) float32 sampled at q (N,3) float, 0 outside."""
    f = np.floor(q).astype(np.int64)
    d = (q - f).astype(np.float32)
    sh = np.array(V.shape)
    out = np.zeros(len(q), np.float32)
    for c in range(8):
        e = np.array([(c >> 2) & 1, (c >> 1) & 1, c & 1])
        i = f + e
        w = np.prod(np.where(e, d, 1 - d), axis=1)
        ok = ((i >= 0) & (i < sh)).all(1)
        j = np.clip(i, 0, sh - 1)
        out += w * np.where(ok, V[j[:, 0], j[:, 1], j[:, 2]], 0)
    return out


def profile(V, q, n, far):
    """Probability sampled along the normal: (2*far+1, N)."""
    ts = np.arange(-far, far + 1, dtype=np.float32)
    return np.stack([trilerp(V, q + t * n) for t in ts])


def metrics(p_u8, origin, pts, nrm, thr=0.5, far=40, win=16, precision=True):
    """recall@r / offset bias / precision proxy / merge count at the surface points.

    `precision=False` drops `precision6` / `pos_frac` only (their KD-tree is over the whole box and is
    not a per-surface quantity)."""
    V, q = np.asarray(p_u8, np.float32) / 255.0, (pts - np.asarray(origin, np.float32))
    ts = np.arange(-far, far + 1, dtype=np.float32)
    S = np.stack([trilerp(V, q + t * nrm) for t in ts])   # (2*far+1, N)
    c = far
    m = {f"recall@{r}": float((S[c - r:c + r + 1].max(0) >= thr).mean()) for r in (2, 4, 8)}
    w = S[c - win:c + win + 1]
    k = np.clip(w.argmax(0), 1, 2 * win - 1)
    y0, y1, y2 = (np.take_along_axis(w, k[None] + j, 0)[0] for j in (-1, 0, 1))
    den = np.minimum(y0 - 2 * y1 + y2, -1e-6)   # the parabola is only a peak where it is concave
    off = (k - win) + np.clip(0.5 * (y0 - y2) / den, -1, 1)
    hit = w.max(0) >= thr                       # an argmax is only meaningful where there is a band
    o = off[hit]
    b = S >= thr
    runs = b[0].astype(np.int32) + (b[1:] & ~b[:-1]).sum(0)
    m.update({"n_points": int(len(pts)), "offset_frac": float(hit.mean()),
              "offset_mean": float(o.mean()) if len(o) else float("nan"),
              "offset_std": float(o.std()) if len(o) else float("nan"),
              "offset_le3": float((np.abs(o) <= 3).mean()) if len(o) else float("nan"),
              "merge_runs": float(runs.mean()), "merge_frac": float((runs > 1).mean())})
    if precision:
        from scipy.spatial import cKDTree   # EDT of the rasterised points, exactly (points are float)
        pos = np.argwhere(V >= thr).astype(np.float32)
        d = cKDTree(q).query(pos, distance_upper_bound=6.0)[0] if len(pos) and len(q) else np.array([np.inf])
        m.update({"precision6": float((d <= 6.0).mean()), "pos_frac": float((V >= thr).mean())})
    return m


def _abs_offsets(V, q, nrm, far=40, win=16, thr=0.5):
    """|sub-voxel peak offset| at the points where a band is found -- what the HD95 / P99 are taken from."""
    ts = np.arange(-far, far + 1, dtype=np.float32)
    S = np.stack([trilerp(V, q + t * nrm) for t in ts])
    w = S[far - win:far + win + 1]
    k = np.clip(w.argmax(0), 1, 2 * win - 1)
    y0, y1, y2 = (np.take_along_axis(w, k[None] + j, 0)[0] for j in (-1, 0, 1))
    den = np.minimum(y0 - 2 * y1 + y2, -1e-6)
    off = (k - win) + np.clip(0.5 * (y0 - y2) / den, -1, 1)
    return np.abs(off[w.max(0) >= thr]).astype(np.float32)


def continuity_one(p_u8, origin, g, k, n, thr=0.5, r=4):
    """Along-sheet continuity for ONE surface: a grid cell is HIT when the probability along its normal
    reaches `thr` within +-r voxels; continuity is the fraction of hit cells whose 8 grid neighbours are
    ALL hit, so a fragmented band scores low even at high recall. Also the mean run length of hits along
    grid rows, and the counts `pool()` weights them by."""
    V, o = np.asarray(p_u8, np.float32) / 255.0, np.asarray(origin, np.float32)
    S = profile(V, g[k] - o, n[k], r)
    hit = np.zeros(g.shape[:2], bool)
    hit[k] = S.max(0) >= thr
    inner = k.copy()
    inner[:1], inner[-1:], inner[:, :1], inner[:, -1:] = False, False, False, False
    nb = np.ones(g.shape[:2], bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nb[1:-1, 1:-1] &= hit[1 + dy:hit.shape[0] - 1 + dy, 1 + dx:hit.shape[1] - 1 + dx]
    c = inner & hit
    runs = []
    for row in hit & k:
        if row.any():
            e = np.diff(np.concatenate([[0], row.astype(int), [0]]))
            runs += (np.where(e == -1)[0] - np.where(e == 1)[0]).tolist()
    return {"continuity": float(nb[c].sum()) / max(int(c.sum()), 1),
            "hit_frac": int(c.sum()) / max(int(k.sum()), 1),
            "mean_run": float(np.mean(runs)) if runs else 0.0,
            "_nhit": int(c.sum()), "_ncont": int(k.sum()), "_nruns": len(runs)}


def continuity(p_u8, origin, size, tifxyz, ax, thr=0.5, r=4, min_pts=200, surfaces=None):
    """`continuity_one` over every published surface crossing the box, point-weighted."""
    rows = [continuity_one(p_u8, origin, g, k, n, thr=thr, r=r)
            for _, g, k, n in (surfaces if surfaces is not None
                               else surface_list(origin, size, tifxyz, ax, min_pts))]
    nhit = sum(r_["_nhit"] for r_ in rows)
    tot = sum(r_["_ncont"] for r_ in rows)
    nruns = sum(r_["_nruns"] for r_ in rows)
    return {"continuity": sum(r_["continuity"] * r_["_nhit"] for r_ in rows) / max(nhit, 1),
            "hit_frac": nhit / max(tot, 1),
            "mean_run": sum(r_["mean_run"] * r_["_nruns"] for r_ in rows) / max(nruns, 1),
            "n_points": tot}


# ==================================================================================== expected run length

def _runs_1d(ok, ln):
    """Total length of every maximal run of True edges: ok (M,) bool, ln (M,) edge lengths -> (R,)."""
    e = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    a, b = np.where(e == 1)[0], np.where(e == -1)[0]
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, ln, 0.0))])
    return c[b] - c[a]


def _walk_axis(good, valid, g, axis, um):
    """Runs and path length along one grid axis: (run lengths um, total path length um, per-vertex length
    share um (H,W))."""
    sl0 = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
    sl1 = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
    ev = valid[sl0] & valid[sl1]                                   # an edge exists between two in-box points
    ln = np.linalg.norm(g[sl1] - g[sl0], axis=-1).astype(np.float64) * um
    ln = np.where(ev & np.isfinite(ln), ln, 0.0)
    ok = ev & good[sl0] & good[sl1]
    share = np.zeros(valid.shape, np.float64)                      # half of each incident edge
    share[sl0] += 0.5 * ln
    share[sl1] += 0.5 * ln
    lines = ln.T if axis == 0 else ln                              # walk along the axis -> put it last
    oks = ok.T if axis == 0 else ok
    sep = np.zeros((lines.shape[0], 1))
    r = _runs_1d(np.concatenate([oks, sep.astype(bool)], 1).ravel(),
                 np.concatenate([lines, sep], 1).ravel())
    return r, float(ln.sum()), share


def erl(p_u8, origin, g, k, n, thr=0.5, r=4, far=40, um=2.4):
    """Expected run length (Januszewski et al. 2018) along one published surface, in MICROMETRES.

    The tifxyz UV grid is the walk graph: an edge between two neighbouring in-box grid points is
    traversable when BOTH its endpoints are good, where a vertex is good when the probability reaches
    `thr` within +-r voxels along its normal (no break) AND the ray crosses `thr` exactly once over
    +-far (no merge -- the same test `metrics()` uses). ERL = sum(L_i^2)/sum(L_total): the expected
    length of the run containing a point drawn uniformly by length, so a 2 mm break costs far more than
    a 2 voxel one. `erl_break_um` / `erl_merge_um` repeat the walk with only one of the two stopping
    conditions, and `lost_break_frac` / `lost_merge_frac` split the surface length between them by
    giving every vertex half of each incident edge."""
    V, o = np.asarray(p_u8, np.float32) / 255.0, np.asarray(origin, np.float32)
    S = profile(V, g[k] - o, n[k], far)                            # (2*far+1, M)
    c = far
    b = S >= thr
    hit = np.zeros(g.shape[:2], bool)
    hit[k] = b[c - r:c + r + 1].max(0)
    merged = np.zeros(g.shape[:2], bool)
    merged[k] = (b[0].astype(np.int32) + (b[1:] & ~b[:-1]).sum(0)) > 1
    gf = np.where(np.isfinite(g), g, 0.0)
    out, tot = {}, 0.0
    for name, good in (("", k & hit & ~merged), ("_break", k & hit), ("_merge", k & ~merged)):
        rl, tl = [], 0.0
        for a in (0, 1):
            ra, la, _ = _walk_axis(good, k, gf, a, um)
            rl.append(ra)
            tl += la
        rl = np.concatenate(rl) if rl else np.zeros(0)
        out["erl" + name + "_um"] = float((rl ** 2).sum() / tl) if tl > 0 else 0.0
        out["_runsq" + name] = float((rl ** 2).sum())
        tot = tl
    share = sum(_walk_axis(k, k, gf, a, um)[2] for a in (0, 1))
    out["path_um"] = tot
    out["break_um"] = float(share[k & ~hit].sum())
    out["merge_um"] = float(share[k & hit & merged].sum())
    out["lost_break_frac"] = out["break_um"] / tot if tot > 0 else 0.0
    out["lost_merge_frac"] = out["merge_um"] / tot if tot > 0 else 0.0
    return out


# ============================================================================== per surface, pooled, CI

def surface_rows(p_u8, origin, size, tifxyz, ax, ct=None, thr=0.5, r=4, far=40, um=2.4, min_pts=200,
                 surfaces=None):
    """Per-surface metrics: the whole suite restricted to one surface, plus ERL. One row per surface,
    with the sufficient statistics (`_*` keys) `pool()` needs to recombine them exactly."""
    o = np.asarray(origin, np.float32)
    rows = []
    for name, g, k, n in (surfaces if surfaces is not None
                          else surface_list(origin, size, tifxyz, ax, min_pts)):
        pts, nrm = g[k], n[k]
        if ct is not None:   # points inside masked CT cannot be predicted
            keep = np.asarray(ct)[tuple(np.clip(np.rint(pts - o).astype(int), 0,
                                                np.asarray(size) - 1).T)] > 0
            pts, nrm = pts[keep], nrm[keep]
        m = metrics(p_u8, origin, pts, nrm, thr=thr, far=far, precision=False) if len(pts) else {}
        cy = continuity_one(p_u8, origin, g, k, n, thr=thr, r=r)
        e = erl(p_u8, origin, g, k, n, thr=thr, r=r, far=far, um=um)
        V = np.asarray(p_u8, np.float32) / 255.0
        off = _abs_offsets(V, pts - o, nrm, far=far, win=16, thr=thr) if len(pts) else np.zeros(0, np.float32)
        rows.append({"surface": name,
                     **{kk: vv for kk, vv in m.items() if kk not in ("precision6", "pos_frac")},
                     **cy, **e,
                     "offset_hd95": float(np.percentile(off, 95)) if len(off) else float("nan"),
                     "offset_p99": float(np.percentile(off, 99)) if len(off) else float("nan"),
                     "_n": int(len(pts)), "_noff": int(round(len(pts) * m.get("offset_frac", 0.0))),
                     "_absoff": off})
    return rows


POOL_W = {"recall@2": "_n", "recall@4": "_n", "recall@8": "_n", "offset_frac": "_n", "merge_runs": "_n",
          "merge_frac": "_n", "offset_le3": "_noff", "continuity": "_nhit", "hit_frac": "_ncont",
          "mean_run": "_nruns"}


def pool(rows):
    """Recombine per-surface rows into the pooled numbers, exactly: point-weighted means, pooled
    variance, length-weighted ERL, quantiles over the concatenated offsets."""
    if not rows:
        return {}
    out = {"n_surfaces": len(rows), "n_points": int(sum(r["_n"] for r in rows))}
    for key, wk in POOL_W.items():
        v = np.array([r.get(key, np.nan) for r in rows], float)
        w = np.array([r.get(wk, 0) for r in rows], float)
        m = np.isfinite(v) & (w > 0)
        out[key] = float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")
    v = np.array([r.get("offset_mean", np.nan) for r in rows], float)
    sd = np.array([r.get("offset_std", np.nan) for r in rows], float)
    w = np.array([r["_noff"] for r in rows], float)
    m = np.isfinite(v) & np.isfinite(sd) & (w > 0)
    if m.any():
        mu = float((v[m] * w[m]).sum() / w[m].sum())
        out["offset_mean"] = mu
        out["offset_std"] = float(np.sqrt(max(((sd[m] ** 2 + v[m] ** 2) * w[m]).sum() / w[m].sum() - mu ** 2, 0.0)))
    else:
        out["offset_mean"] = out["offset_std"] = float("nan")
    a = np.concatenate([r["_absoff"] for r in rows]) if any(len(r["_absoff"]) for r in rows) else np.zeros(0)
    out["offset_hd95"] = float(np.percentile(a, 95)) if len(a) else float("nan")
    out["offset_p99"] = float(np.percentile(a, 99)) if len(a) else float("nan")
    tot = sum(r["path_um"] for r in rows)
    for suf in ("", "_break", "_merge"):
        out["erl" + suf + "_um"] = float(sum(r["_runsq" + suf] for r in rows) / tot) if tot > 0 else 0.0
    out["path_um"] = float(tot)
    out["lost_break_frac"] = float(sum(r["break_um"] for r in rows) / tot) if tot > 0 else 0.0
    out["lost_merge_frac"] = float(sum(r["merge_um"] for r in rows) / tot) if tot > 0 else 0.0
    return out


def bootstrap(rows, n=200, seed=0, lo=2.5, hi=97.5):
    """Resample SURFACES with replacement (points inside one surface are far too correlated to resample
    individually) and repool; returns {metric: [lo, hi]} percentile intervals."""
    if len(rows) < 2:
        return {}
    rng = np.random.default_rng(seed)
    draws = [pool([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(int(n))]
    keys = [k for k in draws[0] if isinstance(draws[0][k], float)]
    return {k: [float(np.nanpercentile([d[k] for d in draws], lo)),
                float(np.nanpercentile([d[k] for d in draws], hi))] for k in keys}


# ======================================================================================= topology

def _or_count(mp, spec, chunk=CHUNK):
    """Number of sliding-window positions of `mp` (bool, already zero-padded by 1 on BOTH sides of every
    axis) that contain a True, where spec[a] is 2 for a window of 2 over the whole padded axis and 1 for
    a window of 1 over `mp[1:-1]` of that axis. Chunked along z; z-chunks overlap by spec[0]-1."""
    n, z = 0, mp.shape[0]
    lo, hi = (0, z - 1) if spec[0] == 2 else (1, z - 1)
    for a in range(lo, hi, chunk):
        b = mp[a:min(a + chunk + spec[0] - 1, z)]
        b = (b[:, :-1] | b[:, 1:]) if spec[1] == 2 else b[:, 1:-1]
        b = (b[:, :, :-1] | b[:, :, 1:]) if spec[2] == 2 else b[:, :, 1:-1]
        b = (b[:-1] | b[1:]) if spec[0] == 2 else b
        n += int(b.sum())
    return n


def euler(m, chunk=CHUNK):
    """Euler characteristic chi = V - E + F - C of the closed-unit-cube complex of the bool volume `m`."""
    mp = np.zeros(tuple(s + 2 for s in m.shape), bool)
    mp[1:-1, 1:-1, 1:-1] = m
    V = _or_count(mp, (2, 2, 2), chunk)
    E = sum(_or_count(mp, tuple(1 if i == a else 2 for i in range(3)), chunk) for a in range(3))
    F = sum(_or_count(mp, tuple(2 if i == a else 1 for i in range(3)), chunk) for a in range(3))
    C = _or_count(mp, (1, 1, 1), chunk)
    return V - E + F - C


def betti(m, chunk=CHUNK):
    """(b0, b1, b2, chi) of a bool volume.

    b0 = 26-connected components of the foreground; b2 = 6-connected components of the padded complement
    minus the unbounded one; chi exact; b1 = b0 + b2 - chi, which is the right NUMBER for a cubical
    subcomplex of R^3 (torsion-free over Z, by Alexander duality) but says nothing about WHERE the loops
    are -- one spurious handle can cancel one missing loop. That is the documented Betti-NUMBER error;
    the spatially matched version needs a persistent-homology dependency we do not carry."""
    from scipy import ndimage as ndi
    m = np.ascontiguousarray(m, bool)
    if not m.any():
        return 0, 0, 0, 0
    b0 = int(ndi.label(m, structure=np.ones((3, 3, 3), np.uint8))[1])
    bg = np.ones(tuple(s + 2 for s in m.shape), bool)
    bg[1:-1, 1:-1, 1:-1] = ~m
    b2 = int(ndi.label(bg, structure=ndi.generate_binary_structure(3, 1))[1]) - 1
    chi = euler(m, chunk)
    return b0, b0 + b2 - chi, b2, int(chi)


def band_of(ref, radius):
    """Voxels within `radius` of the reference sheet (EDT of ~ref): the region topology is measured in."""
    from scipy import ndimage as ndi
    if radius <= 0 or not ref.any():
        return np.ones_like(ref)
    return ndi.distance_transform_edt(~ref) <= float(radius)


def betti_error(pred, ref, margin=8, band=6, dilate=2.0, chunk=CHUNK):
    """Betti-0/1 error of `pred` against `ref` (both bool, same box), measured on the box INTERIOR.

    `margin` voxels are cropped off every face first, so a sheet the box merely cuts through does not
    read as a component or a loop the model invented. `band` restricts both volumes to voxels within
    that many voxels of the reference sheet: a reference covers only SOME of the sheets crossing the box,
    so an unrestricted count would charge the model for every correctly predicted sheet that has none. It
    must stay below half the sheet pitch or two sheets' bands fuse.

    `dilate` thickens the reference. A mesh rasterises to a ONE-voxel staircase, and a 26-connected
    staircase traps a background voxel in every corner -- thousands of cavities and loops that are pure
    rasterisation artefacts -- while a predicted band at thr 0.5 is 3-5 voxels thick and traps none of
    them. The two must be put on the same footing before they are counted."""
    assert pred.shape == ref.shape, (pred.shape, ref.shape)
    from scipy import ndimage as ndi
    s = tuple(slice(margin, -margin if margin else None) for _ in range(3))
    p, r = np.ascontiguousarray(pred[s], bool), np.ascontiguousarray(ref[s], bool)
    d = ndi.distance_transform_edt(~r) if r.any() and (band > 0 or dilate > 0) else None
    b = np.ones_like(r) if d is None or band <= 0 else (d <= float(band))
    if d is not None and dilate > 0:
        r = d <= float(dilate)
    p, r = p & b, r & b
    p0, p1, p2, pc = betti(p, chunk)
    r0, r1, r2, rc = betti(r, chunk)
    return {"betti0": p0, "betti1": p1, "betti2": p2, "euler": pc,
            "betti0_ref": r0, "betti1_ref": r1, "betti2_ref": r2, "euler_ref": rc,
            "betti0_err": abs(p0 - r0), "betti1_err": abs(p1 - r1),
            "betti0_err_norm": abs(p0 - r0) / max(r0, 1), "betti1_err_norm": abs(p1 - r1) / max(r1, 1),
            "betti_margin": int(margin), "betti_band": float(band), "betti_dilate": float(dilate),
            "betti_interior_vox": int(p.size)}


def rasterize(grids, shape, origin, step=0.7, pad=64.0):
    """Bool volume of the published surfaces: every mesh quad whose four corners are finite is sampled on
    a regular (u x u) lattice dense enough that consecutive samples are `step` voxels apart, and the
    samples are rounded into the box. The tifxyz grid is many voxels coarse, so the quads MUST be filled
    in or the reference would be a cloud of disconnected specks with a meaningless b0.

    A published surface spans the whole scroll and the box is one window of it, so quads with no corner
    within `pad` voxels of the box are dropped BEFORE sampling."""
    out = np.zeros(shape, bool)
    o = np.asarray(origin, np.float32)
    lo, hi = o - pad, o + np.asarray(shape, np.float32) + pad
    for g in grids:
        v = np.isfinite(g).all(-1)
        near = v & ((g >= lo) & (g < hi)).all(-1)
        q = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
        q &= near[:-1, :-1] | near[1:, :-1] | near[:-1, 1:] | near[1:, 1:]
        if not q.any():
            continue
        C = np.stack([g[:-1, :-1][q], g[1:, :-1][q], g[:-1, 1:][q], g[1:, 1:][q]]).astype(np.float32)
        sh = np.asarray(shape)
        # The sampling density is PER QUAD, bucketed to powers of two: one hole in the grid can leave a
        # single quad hundreds of voxels wide, and a global density taken from it would cost 10^4 times
        # more samples on every ordinary 20-voxel quad.
        d = np.maximum(np.abs(C[1] - C[0]).max(-1), np.abs(C[2] - C[0]).max(-1))
        ub = np.clip(1 << np.ceil(np.log2(np.maximum(np.ceil(d / step) + 1, 2))).astype(int), 2, 512)
        for u in np.unique(ub):
            Cu = C[:, ub == u]
            u = int(u)
            t = np.linspace(0, 1, u, dtype=np.float32)
            wa, wb = t[:, None, None, None], t[None, :, None, None]
            blk = max(1, 4_000_000 // (u * u))
            for j in range(0, Cu.shape[1], blk):
                c = Cu[:, j:j + blk]
                p = ((1 - wa) * (1 - wb) * c[0] + wa * (1 - wb) * c[1]
                     + (1 - wa) * wb * c[2] + wa * wb * c[3])
                i = np.rint(p.reshape(-1, 3) - o).astype(np.int64)
                i = i[((i >= 0) & (i < sh)).all(1)]
                out[i[:, 0], i[:, 1], i[:, 2]] = True
    return out


def mesh_reference(origin, size, surfaces):
    """The mesh-derived binary reference sheet for the box (quads filled in; see `rasterize`)."""
    return rasterize([g for _, g, _, _ in surfaces], tuple(int(x) for x in size), origin)


# ============================================================ store vs store, for a held-out region

def _skel_runs(skel, good):
    """(run lengths in voxels, total skeleton voxels) of the connected pieces of `skel & good`.

    26-connected components of the skeleton restricted to where the prediction agrees: the ERL analogue
    for a volume with no mesh to walk along. A break in the prediction cuts a component in two exactly
    where the walk along a mesh would have stopped."""
    from scipy import ndimage as ndi
    tot = int(skel.sum())
    g = skel & good
    if not g.any():
        return np.zeros(0, np.float64), tot
    lab, n = ndi.label(g, structure=np.ones((3, 3, 3), np.uint8))
    return np.bincount(lab.ravel())[1:].astype(np.float64), tot


COMPARE_BLOCK = 256   # the edge of one `compare_stores` block, in voxels


def _blocks(lo, hi, n):
    """[(start, stop)] tiling [lo, hi) in steps of `n` (the last one short)."""
    return [(a, min(a + int(n), int(hi))) for a in range(int(lo), int(hi), int(n))]


def compare_stores(a_u8, b_u8, thr=0.5, margin=8, skel_iters=4, betti_band=6, betti_dilate=0.0,
                   block=COMPARE_BLOCK, device=None):
    """Compare two probability volumes of the same box -- a held-out region's reference store and a
    prediction of it -- without any mesh.

    `a_u8` is the REFERENCE. Either argument may be a numpy array or anything sliceable like one (an
    open zarr store is read block by block, never whole). On the interior (`margin` voxels cropped off
    every face):

        dice        2|A and B| / (|A| + |B|)
        overlap     |A and B| / |A|, i.e. how much of the reference the prediction covers
        precision   |A and B| / |B|
        betti*_err  `betti_error` of B against A, in a band around A (A is already a thick band, so the
                    default `dilate` is 0: unlike a mesh it needs no thickening)
        erl_vox     an ERL along A's own SKELETON (`losses.skeleton`, the same construction the training
                    loss uses): the skeleton is cut where B does not cover it, and sum(L^2)/sum(L) over
                    the surviving 26-connected pieces is the expected length of the piece containing a
                    skeleton voxel drawn uniformly. `erl_frac` is that as a fraction of the whole
                    skeleton, so 1.0 means "no break anywhere".

    STREAMED IN BLOCKS. The interior is tiled into `block`^3 cores; each core is read with a halo wide
    enough that the band (an EDT within `betti_band`), the dilation and the skeleton (`skel_iters + 1`
    pooling ops) are exact on the core, so dice / overlap / precision / skel_recall are the
    whole-volume numbers. Only bool/uint8 blocks of (block + 2 halo)^3 are ever alive, which is what
    lets the verso gate run beside production on a 64 GB host (the whole-region version held a dozen
    1 GB+ temporaries -- float64 EDTs, int32 labels, float32 skeleton stacks -- and OOMed tnr-0 at
    paris4 step 2000). Two numbers become PER-BLOCK sums, because a connected component is counted
    inside each block: the Betti numbers (and their errors, summed as |error| per block, so a block's
    extra piece cannot cancel another block's missing one) and the ERL runs, which a block face cuts.
    A volume no larger than one block is one block, and then every number is the old whole-volume one.
    `device` runs the skeleton's pooling there (a CUDA card is ~50x the CPU)."""
    import torch
    from scipy import ndimage as ndi
    from rvsm import losses as L
    S = tuple(int(v) for v in a_u8.shape[-3:])
    assert S == tuple(int(v) for v in b_u8.shape[-3:]), (a_u8.shape, b_u8.shape)
    t8 = int(round(thr * 255))
    m = int(margin)
    lo, hi = [m] * 3, [s - m for s in S]
    halo = int(max(np.ceil(max(float(betti_band), float(betti_dilate))) + 1, int(skel_iters) + 2))
    dev = torch.device(device) if device is not None else torch.device("cpu")
    na = nb = ni = 0
    bsum = {"betti0": 0, "betti1": 0, "betti2": 0, "euler": 0, "betti0_ref": 0, "betti1_ref": 0,
            "betti2_ref": 0, "euler_ref": 0, "betti0_err": 0, "betti1_err": 0}
    interior, nblk = 0, 0
    tot = hit = 0
    l2 = 0.0

    def rd(v, sl):
        if v.ndim > 3:
            return np.asarray(v[(0,) * (v.ndim - 3) + sl], np.uint8)
        return np.asarray(v[sl], np.uint8)

    for z0, z1 in _blocks(lo[0], hi[0], block):
        for y0, y1 in _blocks(lo[1], hi[1], block):
            for x0, x1 in _blocks(lo[2], hi[2], block):
                c0, c1 = (z0, y0, x0), (z1, y1, x1)
                h0 = tuple(max(c - halo, l) for c, l in zip(c0, lo))
                h1 = tuple(min(c + halo, h) for c, h in zip(c1, hi))
                hs = tuple(slice(a, b) for a, b in zip(h0, h1))
                cs = tuple(slice(a - h, b - h) for a, b, h in zip(c0, c1, h0))
                A = rd(a_u8, hs) >= t8
                B = rd(b_u8, hs) >= t8
                ab, bb = A[cs], B[cs]
                interior += int(ab.size)
                nblk += 1
                ka, kb = int(ab.sum()), int(bb.sum())
                na, nb = na + ka, nb + kb
                if ka and kb:
                    ni += int((ab & bb).sum())
                if not A.any():
                    continue        # no reference within reach: an empty band, no skeleton
                # ---- topology, in the band around the reference, counted on the core
                if betti_band > 0 or betti_dilate > 0:
                    d = ndi.distance_transform_edt(~A)
                    band = (d <= float(betti_band)) if betti_band > 0 else np.ones_like(A)
                    r = (d <= float(betti_dilate)) if betti_dilate > 0 else A
                    del d
                else:
                    band, r = np.ones_like(A), A
                p_, r_ = (B & band)[cs], (r & band)[cs]
                del band, r
                p0, p1, p2, pc = betti(p_)
                r0, r1, r2, rc = betti(r_)
                del p_, r_
                for k, v in (("betti0", p0), ("betti1", p1), ("betti2", p2), ("euler", pc),
                             ("betti0_ref", r0), ("betti1_ref", r1), ("betti2_ref", r2),
                             ("euler_ref", rc), ("betti0_err", abs(p0 - r0)),
                             ("betti1_err", abs(p1 - r1))):
                    bsum[k] += int(v)
                # ---- the reference's skeleton, cut where the prediction does not cover it
                with torch.no_grad():
                    t = torch.from_numpy(A).to(dev, torch.float32)[None, None]
                    sk = (L.skeleton(t, iters=int(skel_iters), thr=0.5)[0, 0] > 0.5).cpu().numpy()[cs]
                    del t
                rl, n = _skel_runs(sk, bb)
                tot += n
                hit += int(rl.sum())
                l2 += float((rl ** 2).sum())
                del A, B, sk
    out = {"dice": (2.0 * ni / (na + nb)) if (na + nb) else 1.0,
           "overlap": (ni / na) if na else float("nan"),
           "precision": (ni / nb) if nb else float("nan"),
           "n_ref": na, "n_pred": nb, "margin": m}
    out.update(bsum)
    out.update({"betti0_err_norm": bsum["betti0_err"] / max(bsum["betti0_ref"], 1),
                "betti1_err_norm": bsum["betti1_err"] / max(bsum["betti1_ref"], 1),
                "betti_margin": 0, "betti_band": float(betti_band),
                "betti_dilate": float(betti_dilate), "betti_interior_vox": int(interior),
                "betti_blocks": int(nblk), "compare_block": int(block)})
    out["skel_vox"] = tot
    out["erl_vox"] = float(l2 / tot) if tot else 0.0
    out["erl_frac"] = float(out["erl_vox"] / tot) if tot else 0.0
    out["skel_recall"] = float(hit / tot) if tot else float("nan")
    return out


# ======================================================================================== one box

def evaluate(pred_fn, box, tifxyz=None, ax=None, ct=None, ref_u8=None, thr=0.5, um=2.4, boot=200,
             seed=0, min_pts=200, margin=8, betti=True):
    """Every metric available for one box, from whatever references there are.

    `pred_fn(origin, size)` returns the predicted uint8 probability volume for the box; `box` is
    `(origin, size)` in the voxels of the rung being evaluated. With `tifxyz` (and `ax`, the umbilicus)
    the mesh suite runs: per-surface rows, the pooled numbers, the bootstrap CI and the Betti error
    against the rasterised meshes. With `ref_u8` (a held-out region's reference store) `compare_stores`
    runs as well. With neither, the result holds only the box description -- which is the honest answer.
    """
    (o, s) = box
    o, s = tuple(int(v) for v in o), tuple(int(v) for v in s)
    p = np.asarray(pred_fn(o, s), np.uint8)
    out = {"box": [*o, *s], "voxel_um": float(um), "thr": float(thr)}
    if tifxyz:
        assert ax is not None, "the mesh suite needs the umbilicus (ax) to orient the surface normals"
        surfaces = surface_list(o, s, tifxyz, ax, min_pts)
        out["surfaces"] = {n: int(k.sum()) for n, _, k, _ in surfaces}
        rows = surface_rows(p, o, s, tifxyz, ax, ct=ct, thr=thr, um=um, min_pts=min_pts,
                            surfaces=surfaces)
        out["metrics"] = pool(rows)
        out["ci"] = bootstrap(rows, n=boot, seed=seed)
        if betti and surfaces:
            out["topo"] = betti_error(p >= int(round(thr * 255)), mesh_reference(o, s, surfaces),
                                      margin=margin)
    if ref_u8 is not None:
        out["vs_store"] = compare_stores(ref_u8, p, thr=thr, margin=margin)
    return out


# ================================================================================ the plateau fit

def fit_curve(steps, vals, bounded=True, gain95=0.95, smooth=1, tail=1.0):
    """Fit a saturating curve to (step, metric) and say whether the run is done.

    Two forms are tried and the lower-RMSE one wins (Hestness et al. 2017 for the power law; the
    literature's own caveat is that a power law does not saturate below 1, so for a [0,1] metric a
    logistic in log-step is usually the honest fit -- but it needs a FLOOR, since a val dice starts near
    0.3, not 0, and a floorless logistic just pins its asymptote to the upper bound):

        power      y = c - a * step^-alpha
        logistic4  y = y0 + (c - y0) / (1 + exp(-k * (log10 step - m)))

    `smooth` is the width of a centred running median applied first (raw per-checkpoint numbers are not
    monotone and an unsmoothed fit is unstable); `tail` keeps only the last fraction of the points, which
    is what the scaling-law literature fits when only the plateau matters.

    Returns the fitted asymptote `c`, `step95` (where 95% of the gain still outstanding at the LAST
    measured step has been collected) and `slope_per_10k` (dy/dstep * 1e4 at the last step). Do not
    trust a saturation call from fewer than ~10-15 points, and never without a bootstrap CI to compare
    the slope against. `asymptote_at_bound` means the fit ran into the ceiling of the parameter range and
    the asymptote is not to be believed."""
    s = np.asarray(steps, float)
    y = np.asarray(vals, float)
    k = np.isfinite(s) & np.isfinite(y) & (s > 0)
    s, y = s[k], y[k]
    o = np.argsort(s)
    s, y = s[o], y[o]
    if int(smooth) > 1 and len(y) > int(smooth):
        w, z = int(smooth) | 1, y.copy()      # FULL windows only: a truncated window at the end drags
        h = w // 2                            # a rising curve down and fakes a plateau
        for i in range(h, len(y) - h):
            z[i] = np.median(y[i - h:i + h + 1])
        y = z
    if 0 < tail < 1:
        s, y = s[int(len(s) * (1 - tail)):], y[int(len(y) * (1 - tail)):]
    if len(s) < 4:
        return {"model": None, "n": int(len(s)), "error": "need at least 4 points"}
    hi = 1.0 if bounded else float(y.max() * 4 + 1)
    ymax, ymin = float(y.max()), float(y.min())
    from scipy.optimize import curve_fit

    def power(x, c, a, al):
        return c - a * x ** (-al)

    def logis(x, c, kk, m, y0):
        return y0 + (c - y0) / (1.0 + np.exp(-kk * (np.log10(x) - m)))

    fits = []
    for f, p0, bnd in (
            (power, [min(ymax * 1.05, hi), max(ymax - ymin, 1e-3) * s[0] ** 0.5, 0.5],
             ([ymax, 0.0, 1e-3], [hi, np.inf, 5.0])),
            (logis, [min(ymax * 1.05, hi), 2.0, float(np.log10(s.mean())), ymin],
             ([ymax, 0.05, -10.0, ymin - 1.0], [hi, 10.0, 12.0, ymax]))):
            # k <= 10: a steeper logistic in log-step is a STEP function, which fits any finished run
            # with zero residual slope and would declare every run saturated
        try:
            p, _ = curve_fit(f, s, y, p0=p0, bounds=bnd, maxfev=60000)
            fits.append((float(np.sqrt(np.mean((f(s, *p) - y) ** 2))), f.__name__, p))
        except Exception:  # noqa: BLE001
            pass
    if not fits:
        return {"model": None, "n": int(len(s)), "error": "no fit converged"}
    rmse, name, p = min(fits, key=lambda t: t[0])
    c, last = float(p[0]), float(s[-1])
    ylast = float(power(last, *p) if name == "power" else logis(last, *p))
    rem = c - ylast
    if name == "power":
        _, a, al = p
        slope = float(a * al * last ** (-al - 1) * 1e4)
        s95 = float(np.exp(min(np.log(a / max((1 - gain95) * rem, 1e-12)) / al, 700.0))) if rem > 1e-9 else last
    else:
        _, kk, m, y0 = p
        e = np.exp(-kk * (np.log10(last) - m))
        slope = float((c - y0) * kk * e / (1 + e) ** 2 / (last * np.log(10)) * 1e4)
        t = ylast + gain95 * rem                   # the value 95% of the way to the asymptote
        z = (c - y0) / max(t - y0, 1e-12) - 1
        s95 = float(10 ** min(m - np.log(max(z, 1e-12)) / kk, 300.0)) if rem > 1e-9 and z > 0 else last
    at_bound = [bool(abs(float(v) - b) < 1e-6 * max(abs(b), 1.0)) for v, b in
                zip(p, ([hi, np.inf, 5.0] if name == "power" else [hi, 10.0, 12.0, ymax]))]
    return {"model": name, "n": int(len(s)), "rmse": rmse, "params": [float(x) for x in p],
            "asymptote": c, "asymptote_at_bound": bool(c >= hi - 1e-6), "params_at_bound": at_bound,
            "last_step": last, "last_value": ylast, "remaining": float(rem),
            "smooth": int(smooth), "tail": float(tail),
            "step95": s95, "steps_to_95": float(max(s95 - last, 0.0)), "slope_per_10k": slope}
