"""Flattened renders of tifxyz surfaces (the refiner's outputs): the CT and the recto store sampled on the
surface's own (u, v) grid, a few voxels either side along the normal.

Per surface, inside --box (default: the surface's bbox cut to the store's box):

    layers.tif          the CT stack (layers, H, W) uint8, one page per normal offset d in --layers
    store_layers.tif    the same for the recto store (with --store)
    ct_d0.png           the d = 0 layer, ct_min / ct_max / ct_mean.png the projections over the stack
    ct_contact.png      every layer, labelled
    store_d0.png, store_mean.png
                        the recto probability on the surface (bright = on the predicted ridge)
    ridge_offset.png    signed distance along the normal to the nearest recto local max >= --thr within
                        +-far voxels (blue < 0 inward, red > 0 outward, gray = no ridge in reach);
    ridge_hist.png      and its histogram; stats.json the numbers

--compare A B renders two surfaces on ONE grid window (e.g. <run>/<name>.before and <run>/<name>, the
published and the refined surface of a refine run; they share the published grid layout) and writes
compare_<name>.png: A | B side by side for the d = 0 CT layer, the CT stack mean, the store d = 0 layer and
the ridge offset, on shared colour scales, plus compare_<name>_hist.png.

A slab crosses a sheet in thin bands of its grid (one per wrap); each band (a column run of in-box nodes,
refine.column_pieces-style) is one piece, upsampled to ~1 voxel pitch (--up 0) and laid out as rows of a
fixed width, so a 128 x 1024 x 6144 strip is a stack of short wide rows, not one 128 x 30000 line.

Memory: the CT and the store are read per uv tile of a piece (--tile nodes square, its sample points'
bbox), never the whole box; the pieces' dense grids are made one at a time; the sampled stacks are uint8;
--up 0 lowers the upsampling until the pieces' bboxes total --max-nodes (a 6144-wide strip lands at ~2
voxel pitch); the TIFF pages and contact-sheet tiles are laid out one layer at a time.

    python -m rvsm.tools.refine.render --surface DIR [--surface DIR ...] | --compare A B
        [--ct URL|PATH] [--store recto.zarr] [--layers=-12..12:2] [--out DIR] [--box Z Y X DZ DY DX]
        [--down 1] [--up 0] [--far 12] [--thr 0.5] [--umbilicus U.json] [--transform T.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from rvsm.tools.refine import refine as R

CT_CACHE = "/home/forrest/refine/ct_cache/ct"
HOLE = (40, 40, 60)       # no node here (hole / outside the box / padding)
NORIDGE = (150, 150, 150)  # a node with no ridge within reach


# ================================================================================================ sources

class ArraySource:
    """A (Z,Y,X) numpy volume at fine-frame origin `origin` (tests; 0 outside)."""

    def __init__(self, V, origin=(0, 0, 0)):
        self.V, self.o = V, np.asarray(origin, np.int64)

    def read(self, lo, shape):
        lo, shape = np.asarray(lo, np.int64), np.asarray(shape, np.int64)
        out = np.zeros(tuple(shape), self.V.dtype)
        a, b = np.maximum(lo - self.o, 0), np.minimum(lo + shape - self.o, self.V.shape)
        if (b > a).all():
            st = a + self.o - lo
            out[st[0]:st[0] + b[0] - a[0], st[1]:st[1] + b[1] - a[1], st[2]:st[2] + b[2] - a[2]] = \
                self.V[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
        return out


class StoreSource:
    def __init__(self, path):
        self.path = path

    def read(self, lo, shape):
        return R.read_box(self.path, lo, shape)


class CTSource:
    """A CT pyramid read at rung 2 + log2(down); read() returns the level's voxels, `down` tells the sampler."""

    def __init__(self, ct, down=1):
        self.ct, self.down = ct, max(1, int(down))

    def read(self, lo, shape):
        return R.read_ct(self.ct, lo, shape, down=self.down)


def parse_layers(s, step=None):
    """'-12..12' / '-12..12:2' (step 2) / '-4,0,4' -> offsets in voxels (floats)."""
    s = str(s).strip()
    if ".." in s:
        a, rest = s.split("..", 1)
        b, st = (rest.split(":", 1) + [None])[:2]
        st = float(st if st is not None else (step or 2))
        return np.arange(float(a), float(b) + st / 2, st, dtype=np.float32)
    return np.asarray([float(v) for v in s.replace(";", ",").split(",") if v.strip()], np.float32)


# ================================================================================================ pieces

def mask_pieces(k, pad=2):
    """The column runs of a (H,W) bool mask, each with its row range and `pad` cells: [(r0, r1, c0, c1)]
    (refine.column_pieces on a mask)."""
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


def inbox(g, o, s):
    return np.isfinite(g).all(-1) & ((g >= o) & (g < o + s)).all(-1)


def prepare(grids, ax, o, s, up, pad=2, min_nodes=16):
    """grids: [(H,W,3) fine zyx] on ONE window. -> yields {"g": [dense grids], "n": [normals], "rc": piece}
    per piece (lazily: the dense grids of a strip's pieces are GBs together): the pieces of the union in-box mask, each upsampled by `up`, its normals taken on the dense grid,
    nodes outside the box masked, trimmed to the rows/cols any grid still has, and transposed so that
    H <= W (the long axis runs across the page)."""
    k = np.zeros(grids[0].shape[:2], bool)
    for g in grids:
        k |= inbox(g, o, s)
    for r0, r1, c0, c1 in mask_pieces(k, pad):
        gs, ns = [], []
        for g in grids:
            sub = g[r0:r1, c0:c1]
            d = R.upsample(sub, up) if up > 1 and min(sub.shape[:2]) > 1 else sub.copy()
            n = R.normals(d, ax) if min(d.shape[:2]) > 1 else np.full_like(d, np.nan)
            m = inbox(d, o, s)
            d[~m], n[~m] = np.nan, np.nan
            gs.append(d.astype(np.float32))
            ns.append(n.astype(np.float32))
        v = np.zeros(gs[0].shape[:2], bool)
        for d in gs:
            v |= np.isfinite(d).all(-1)
        if v.sum() < min_nodes:
            continue
        rr, cc = np.nonzero(v.any(1))[0], np.nonzero(v.any(0))[0]
        sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
        gs, ns = [d[sl] for d in gs], [n[sl] for n in ns]
        if gs[0].shape[0] > gs[0].shape[1]:
            gs, ns = [d.transpose(1, 0, 2) for d in gs], [n.transpose(1, 0, 2) for n in ns]
        yield {"g": gs, "n": ns, "rc": (int(r0), int(r1), int(c0), int(c1))}


def auto_up_budget(grids, o, s, max_nodes=12e6, pad=2):
    """The upsampling to ~1 voxel pitch, lowered until the pieces' dense bounding boxes (what the stacks and the
    pictures hold) total <= max_nodes: a slanted band's bbox is mostly empty, and a 6144-wide strip's ~10 bands
    at 1 voxel pitch are ~40M nodes."""
    up = R.auto_up(grids[0], 1.0, cap=32)
    k = np.zeros(grids[0].shape[:2], bool)
    for g in grids:
        k |= inbox(g, o, s)
    area = sum((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in mask_pieces(k, 0))
    if area:
        up = min(up, max(1, int(np.sqrt(max_nodes / area))))
    return up


# ============================================================================================== sampling

def _sample_src(src, q, down=1):
    """src sampled trilinearly at (N,3) fine zyx points; read over their bbox only; 0..1 floats."""
    if not len(q):
        return np.zeros(0, np.float32)
    qd = q / float(down)
    lo = np.floor(qd.min(0)).astype(np.int64) - 1
    hi = np.ceil(qd.max(0)).astype(np.int64) + 2
    V = src.read(lo * down, (hi - lo) * down) if down > 1 else src.read(lo, hi - lo)
    if V is None:
        return np.zeros(len(q), np.float32)
    return R.sample(V, qd - lo)


def render_piece(g, n, ct=None, store=None, layers=None, far=12, thr=0.5, tile=256, down=1):
    """Sample one dense grid g (H,W,3) with normals n: {"ct": (L,H,W) 0..1 or None, "st": (L,H,W) or None,
    "off": (H,W) signed ridge offset (NaN: no ridge / no node), "ok": (H,W)}. The sources are read per
    tile x tile block of nodes (the bbox of that block's sample points)."""
    layers = np.asarray(parse_layers("-12..12:2") if layers is None else layers, np.float32)
    H, W = g.shape[:2]
    ok = np.isfinite(g).all(-1) & np.isfinite(n).all(-1)
    L = len(layers)
    CT = np.zeros((L, H, W), np.uint8) if ct is not None else None
    ST = np.zeros((L, H, W), np.uint8) if store is not None else None
    q8 = lambda v: np.clip(v * 255 + 0.5, 0, 255).astype(np.uint8)   # noqa: E731  the sources are uint8 anyway
    OFF = np.full((H, W), np.nan, np.float32)
    ts = np.arange(-far, far + 1, dtype=np.float32)
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            m = ok[r:r + tile, c:c + tile]
            if not m.any():
                continue
            p = g[r:r + tile, c:c + tile][m]
            nn = n[r:r + tile, c:c + tile][m]
            ii, jj = np.nonzero(m)
            ii, jj = ii + r, jj + c
            qL = (p[None] + layers[:, None, None] * nn[None]).reshape(-1, 3)
            if CT is not None:
                CT[:, ii, jj] = q8(_sample_src(ct, qL, getattr(ct, "down", down))).reshape(L, -1)
            if ST is not None:
                qP = (p[None] + ts[:, None, None] * nn[None]).reshape(-1, 3)
                S = _sample_src(store, np.concatenate([qL, qP]))
                ST[:, ii, jj] = q8(S[:len(qL)]).reshape(L, -1)
                prof = S[len(qL):].reshape(len(ts), -1)
                pos, stren = R.local_maxima(prof, far, thr)
                d = np.where(stren >= thr, pos, np.inf)
                best = np.take_along_axis(d, np.abs(d).argmin(0)[None], 0)[0]
                OFF[ii, jj] = np.where(np.isfinite(best), best, np.nan)
    return {"ct": Stack(CT, ok) if CT is not None else None, "st": Stack(ST, ok) if ST is not None else None,
            "off": OFF, "ok": ok}


class Stack:
    """A (L,H,W) uint8 layer stack held compactly; indexing a layer (or np.asarray of the whole stack) gives
    0..1 floats with NaN off the surface -- made on demand, one piece at a time."""

    def __init__(self, u8, ok):
        self.u8, self.ok = u8, ok

    def __len__(self):
        return len(self.u8)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            return np.where(self.ok, self.u8[i].astype(np.float32) / 255.0, np.nan)
        return np.asarray(self)[i]

    def __array__(self, dtype=None, copy=None):
        a = np.where(self.ok[None], self.u8.astype(np.float32) / 255.0, np.nan)
        return a if dtype is None else a.astype(dtype)


# ================================================================================================ layout

def plan_layout(shapes, gap=6, aspect=1.6, wmin=256, wmax=2048):
    """Row width for a list of (H, W) pieces: long thin pieces are cut into segments of width Lw and stacked,
    Lw chosen so the whole picture is roughly `aspect` wide:tall."""
    if not shapes:
        return wmin
    area = sum((h + gap) * w for h, w in shapes)
    lw = int(np.sqrt(aspect * area))
    return int(np.clip(max(lw, max(h for h, _ in shapes)), wmin, wmax))


def layout(imgs, lw, gap=6, label_h=12, labels=None):
    """RGB pieces (H,W,3) uint8 -> one RGB canvas: each piece cut into lw-wide segments stacked top to bottom,
    a thin label line (piece label, column range) above each segment."""
    from PIL import Image, ImageDraw
    segs = []
    for i, im in enumerate(imgs):
        W = im.shape[1]
        sw = -(-W // max(1, -(-W // lw)))    # equal segments, none a sliver
        for c0 in range(0, W, sw):
            segs.append((i, c0, min(c0 + sw, W)))
    width = max((c1 - c0 for _, c0, c1 in segs), default=1)
    height = sum(imgs[i].shape[0] + gap + label_h for i, _, _ in segs) or 1
    can = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(can)
    y = 0
    for i, c0, c1 in segs:
        lab = (labels[i] if labels else f"piece {i}") + f"  u {c0}..{c1}"
        d.text((2, y), lab, fill=(90, 90, 90))
        y += label_h
        can.paste(Image.fromarray(np.ascontiguousarray(imgs[i][:, c0:c1])), (0, y))
        y += imgs[i].shape[0] + gap
    return np.asarray(can)


def gray_rgb(a, lo, hi):
    """(H,W) float -> RGB uint8 on [lo, hi]; NaN -> HOLE."""
    v = np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    out = np.repeat((v * 255 + 0.5).astype(np.uint8)[..., None], 3, -1)
    out[~np.isfinite(a)] = HOLE
    return out


def diverging_rgb(off, ok, far):
    """Signed offset -> blue (-far) / white (0) / red (+far); ok & NaN -> NORIDGE; ~ok -> HOLE."""
    t = np.clip(np.nan_to_num(off) / float(far), -1, 1)
    r = np.where(t < 0, 1 + t, 1.0)
    b = np.where(t > 0, 1 - t, 1.0)
    gch = 1 - np.abs(t)
    out = (np.stack([r, gch, b], -1) * 255 + 0.5).astype(np.uint8)
    out[ok & ~np.isfinite(off)] = NORIDGE
    out[~ok] = HOLE
    return out


def colorbar(width, far, h=14):
    x = np.linspace(-far, far, width, dtype=np.float32)
    return diverging_rgb(np.repeat(x[None], h, 0), np.ones((h, width), bool), far)


def save_png(path, rgb, title=None, foot=None):
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.ascontiguousarray(rgb))
    th, fh = (16 if title else 0), (16 if foot else 0)
    if th or fh:
        can = Image.new("RGB", (im.size[0], im.size[1] + th + fh), (255, 255, 255))
        can.paste(im, (0, th))
        d = ImageDraw.Draw(can)
        if title:
            d.text((3, 2), title, fill=(0, 0, 0))
        if foot:
            d.text((3, th + im.size[1] + 2), foot, fill=(0, 0, 0))
        im = can
    im.save(path)
    return path


def stack_v(rgbs, gap=8, titles=None):
    from PIL import Image, ImageDraw
    th = 16 if titles else 0
    W = max(r.shape[1] for r in rgbs)
    H = sum(r.shape[0] + th + gap for r in rgbs)
    can = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(can)
    y = 0
    for i, r in enumerate(rgbs):
        if titles:
            d.text((3, y + 2), titles[i], fill=(0, 0, 0))
            y += th
        can.paste(Image.fromarray(np.ascontiguousarray(r)), (0, y))
        y += r.shape[0] + gap
    return np.asarray(can)


def stack_h(rgbs, gap=12):
    H = max(r.shape[0] for r in rgbs)
    parts = []
    for i, r in enumerate(rgbs):
        pad = np.full((H, r.shape[1], 3), 255, np.uint8)
        pad[:r.shape[0]] = r
        parts.append(pad)
        if i < len(rgbs) - 1:
            parts.append(np.full((H, gap, 3), 255, np.uint8))
    return np.concatenate(parts, 1)


def shrink(rgb, max_w):
    if rgb.shape[1] <= max_w:
        return rgb
    from PIL import Image
    f = max_w / rgb.shape[1]
    return np.asarray(Image.fromarray(rgb).resize((max_w, max(1, int(rgb.shape[0] * f))), Image.BILINEAR))


def hist_rgb(series, far, size=(520, 200)):
    """Overlaid step histograms of signed offsets: [(label, values, rgb)]."""
    from PIL import Image, ImageDraw
    W, H = size
    img = Image.new("RGB", (W, H + 16 * (len(series) + 1)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    bins = np.linspace(-far, far, 4 * int(far) + 1)
    hs = [np.histogram(np.asarray(v)[np.isfinite(v)], bins)[0] for _, v, _ in series]
    top = max([int(h.max()) for h in hs if len(h)] + [1])
    bw = (W - 20) / (len(bins) - 1)
    zx = 10 + (W - 20) / 2
    d.line([(zx, 8), (zx, H - 5)], fill=(200, 200, 200))
    for (lab, v, col), h in zip(series, hs):
        pts = []
        for i, c in enumerate(h):
            yv = H - 5 - (H - 15) * c / top
            pts += [(10 + i * bw, yv), (10 + (i + 1) * bw, yv)]
        if pts:
            d.line(pts, fill=col, width=2)
    d.text((10, H), f"ridge offset {-far:g}..{far:g} vox (signed, along the normal)", fill=(0, 0, 0))
    for i, (lab, v, col) in enumerate(series):
        d.text((10, H + 16 * (i + 1)), lab, fill=col)
    return np.asarray(img)


# ================================================================================================ stats

def off_stats(off, ok):
    v = off[ok]
    have = np.isfinite(v)
    a = np.abs(v[have])
    q = lambda p: round(float(np.percentile(a, p)), 3) if len(a) else None   # noqa: E731
    return {"nodes": int(ok.sum()), "frac_ridge": round(float(have.mean()), 4) if len(v) else None,
            "abs_median": q(50), "abs_mean": round(float(a.mean()), 3) if len(a) else None, "abs_p90": q(90),
            "frac_le1": round(float((a <= 1).sum() / max(len(v), 1)), 4),
            "frac_le2": round(float((a <= 2).sum() / max(len(v), 1)), 4),
            "signed_median": round(float(np.median(v[have])), 3) if have.any() else None}


def field_stats(res, li0):
    """Pooled numbers of a surface's pieces: ridge offsets, the store on the surface, CT texture."""
    ok = np.concatenate([r["ok"].ravel() for r in res])
    off = np.concatenate([r["off"].ravel() for r in res])
    s = off_stats(off, ok)
    if res[0]["st"] is not None:
        st0 = np.concatenate([r["st"][li0][r["ok"]] for r in res])
        s["store_d0_mean"] = round(float(np.nanmean(st0)), 4) if len(st0) else None
        s["store_d0_ge_thr"] = round(float(np.nanmean(st0 >= 0.5)), 4) if len(st0) else None
    if res[0]["ct"] is not None:
        # texture: mean |gradient| of the d=0 CT layer over the grid, relative to its std (scale-free sharpness)
        gs, sd = [], []
        for r in res:
            a = r["ct"][li0]
            for ax_ in (0, 1):
                d = np.abs(np.diff(a, axis=ax_))
                gs.append(d[np.isfinite(d)])
            sd.append(a[np.isfinite(a)])
        gv, sv = np.concatenate(gs), np.concatenate(sd)
        s["ct_d0_mean"] = round(float(sv.mean() * 255), 2) if len(sv) else None
        s["ct_d0_grad_over_std"] = round(float(gv.mean() / max(sv.std(), 1e-9)), 4) if len(gv) else None
    return s


def fmt_stats(s):
    if s.get("abs_median") is None:
        return f"n={s['nodes']} no ridge"
    t = (f"|off| med {s['abs_median']:.2f} mean {s['abs_mean']:.2f} p90 {s['abs_p90']:.2f}, <=2 {100 * s['frac_le2']:.0f}%, "
         f"no ridge {100 * (1 - s['frac_ridge']):.0f}%")
    if s.get("store_d0_mean") is not None:
        t += f", store@d0 {s['store_d0_mean']:.2f}"
    return t


# ================================================================================================ writers

def ct_range(res_lists):
    rs = [r for res in res_lists for r in res if r["ct"] is not None]
    tot = sum(int(r["ok"].sum()) * len(r["ct"]) for r in rs)
    k = max(1, tot // 4_000_000)
    v = [r["ct"].u8[:, r["ok"]].ravel()[::k] for r in rs]
    v = np.concatenate(v).astype(np.float32) / 255.0 if v else np.zeros(0)
    if not len(v):
        return 0.0, 1.0
    lo, hi = np.percentile(v, (0.5, 99.5))
    return float(lo), float(max(hi, lo + 1e-3))


def write_stack(path, page, layers):
    """A (layers, H, W) uint8 multi-page TIFF written one page at a time (page(i) -> (H,W) uint8)."""
    import itertools

    import tifffile
    first = page(0)
    pages = itertools.chain([first], (page(i) for i in range(1, len(layers))))
    tifffile.imwrite(path, data=pages, shape=(len(layers), *first.shape), dtype=np.uint8,
                     metadata={"axes": "ZYX", "normal_offsets_vox": [float(x) for x in layers]})
    return path


def write_surface(out, name, res, layers, far, lw, labels, rng=None, log=print):
    """All the per-surface outputs of a list of rendered pieces."""
    os.makedirs(out, exist_ok=True)
    li0 = int(np.argmin(np.abs(layers)))
    rng = rng or ct_range([res])
    paths = []
    lay = lambda f: layout([f(r) for r in res], lw, labels=labels)   # noqa: E731
    if res[0]["ct"] is not None:
        # the multi-page stack: every piece's layer laid out on the same canvas, one page per layer
        write_stack(os.path.join(out, "layers.tif"), lambda i: lay(lambda r: gray_rgb(r["ct"][i], 0.0, 1.0))[..., 0],
                    layers)
        paths.append(os.path.join(out, "layers.tif"))
        with np.errstate(all="ignore"):
            for tag, fn in (("d0", lambda a: a[li0]), ("min", lambda a: np.nanmin(a, 0)),
                            ("max", lambda a: np.nanmax(a, 0)), ("mean", lambda a: np.nanmean(a, 0))):
                p = os.path.join(out, f"ct_{tag}.png")
                save_png(p, lay(lambda r: gray_rgb(fn(r["ct"]), *rng)), f"{name}  CT {tag} (normal offsets "
                         f"{layers[0]:+g}..{layers[-1]:+g})" if tag != "d0" else f"{name}  CT d=0")
                paths.append(p)
        cols = int(np.ceil(np.sqrt(len(layers))))
        tiles = [shrink(lay(lambda r, i=i: gray_rgb(r["ct"][i], *rng)), 6000 // cols) for i in range(len(layers))]
        rows_ = [stack_h(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
        titles = [" | ".join(f"d={layers[j]:+g}" for j in range(k, min(k + cols, len(tiles))))
                  for k in range(0, len(tiles), cols)]
        p = os.path.join(out, "ct_contact.png")
        save_png(p, shrink(stack_v(rows_, titles=titles), 6000), f"{name}  CT layers, left to right")
        paths.append(p)
    if res[0]["st"] is not None:
        write_stack(os.path.join(out, "store_layers.tif"),
                    lambda i: lay(lambda r: gray_rgb(r["st"][i], 0.0, 1.0))[..., 0], layers)
        with np.errstate(all="ignore"):
            for tag, fn in (("d0", lambda a: a[li0]), ("mean", lambda a: np.nanmean(a, 0))):
                p = os.path.join(out, f"store_{tag}.png")
                save_png(p, lay(lambda r: gray_rgb(fn(r["st"]), 0.0, 1.0)), f"{name}  recto store {tag}")
                paths.append(p)
        st = field_stats(res, li0)
        p = os.path.join(out, "ridge_offset.png")
        cb = colorbar(min(lw, 512), far)
        save_png(p, stack_v([lay(lambda r: diverging_rgb(r["off"], r["ok"], far)), cb]),
                 f"{name}  ridge offset (blue inward, red outward, gray none within {far:g})",
                 f"colour bar {-far:g}..{far:g} vox;  {fmt_stats(st)}")
        paths.append(p)
        off = np.concatenate([r["off"][r["ok"]] for r in res])
        p = os.path.join(out, "ridge_hist.png")
        save_png(p, hist_rgb([(name, off, (40, 90, 200))], far))
        paths.append(p)
    else:
        st = field_stats(res, li0)
    json.dump({"name": name, "layers": [float(x) for x in layers], "far": far, "pieces": len(res),
               "stats": st}, open(os.path.join(out, "stats.json"), "w"), indent=1)
    for p in paths:
        log(p)
    return st


def write_compare(path, title, names, res_a, res_b, layers, far, lw, labels, max_px=40e6):
    li0 = int(np.argmin(np.abs(layers)))
    rng = ct_range([res_a, res_b])
    sa, sb = field_stats(res_a, li0), field_stats(res_b, li0)
    rows, titles = [], []
    with np.errstate(all="ignore"):
        fields = []
        if res_a[0]["ct"] is not None:
            fields += [("CT d=0", lambda r: gray_rgb(r["ct"][li0], *rng)),
                       ("CT stack mean", lambda r: gray_rgb(np.nanmean(r["ct"], 0), *rng))]
        if res_a[0]["st"] is not None:
            fields += [("recto store d=0", lambda r: gray_rgb(r["st"][li0], 0.0, 1.0)),
                       (f"ridge offset (blue in / red out / gray none within {far:g})",
                        lambda r: diverging_rgb(r["off"], r["ok"], far))]
        for lab, fn in fields:
            rows.append(stack_h([layout([fn(r) for r in res_a], lw, labels=labels),
                                 layout([fn(r) for r in res_b], lw, labels=labels)], gap=16))
            titles.append(f"{lab}:  left {names[0]}  |  right {names[1]}")
    if res_a[0]["st"] is not None:
        rows.append(colorbar(min(lw, 512), far))
        titles.append(f"ridge offset colour bar {-far:g}..{far:g} vox")
    img = stack_v(rows, gap=10, titles=titles)
    foot = f"left {names[0]}: {fmt_stats(sa)}\nright {names[1]}: {fmt_stats(sb)}"
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (img.shape[1], img.shape[0] + 20 + 34), (255, 255, 255))
    im.paste(Image.fromarray(img), (0, 20))
    d = ImageDraw.Draw(im)
    d.text((3, 3), title, fill=(0, 0, 0))
    d.text((3, 20 + img.shape[0] + 3), foot, fill=(0, 0, 0))
    if im.size[0] * im.size[1] > max_px:   # a whole strip at ~1 voxel pitch is ~100 Mpx: viewers balk
        f = float(np.sqrt(max_px / (im.size[0] * im.size[1])))
        im = im.resize((int(im.size[0] * f), int(im.size[1] * f)), Image.LANCZOS)
    im.save(path)
    hp = path[:-4] + "_hist.png"
    if res_a[0]["st"] is not None:
        oa = np.concatenate([r["off"][r["ok"]] for r in res_a])
        ob = np.concatenate([r["off"][r["ok"]] for r in res_b])
        save_png(hp, hist_rgb([(f"{names[0]}: {fmt_stats(sa)}", oa, (150, 150, 150)),
                               (f"{names[1]}: {fmt_stats(sb)}", ob, (0, 150, 0))], far, size=(760, 220)), title)
    return path, sa, sb


# ================================================================================================ cli

def window(d, frame, o, s, rc=None, pad=2):
    """(grid in the fine frame, (r0, r1, c0, c1)) of a tifxyz over the rows/cols touching the box, or over
    the given window rc."""
    if rc is None:
        got = R.read_surface_box(d, frame, o, s, margin=0, pad=pad)
        if got is None:
            return None, None
        g, (r0, c0), _ = got
        return g, (r0, r0 + g.shape[0], c0, c0 + g.shape[1])
    r0, r1, c0, c1 = rc
    ch = R._tif_channels(d)
    g = np.stack([np.asarray(c[r0:r1, c0:c1], np.float32) for c in ch], -1)
    del ch
    g = np.where((g > 0).all(-1)[..., None], g, np.nan)
    return (g if frame is R.IDENTITY else frame.to_fine(g)), rc


def surface_frame(d, args):
    kind = R.frame_kind(d, "fine")
    if kind == "fine":
        return R.IDENTITY
    if kind == "legacy" and args.transform:
        return R.transform_json_frame(args.transform, args.transform_direction)
    fr = R.module_frame() if kind == "legacy" else None
    if fr is None:
        raise SystemExit(f"{d} is not in the fine frame: pass --transform transform.json")
    return fr


def surface_bbox(d, frame):
    b = np.asarray(json.load(open(os.path.join(d, "meta.json")))["bbox"], np.float64)[:, ::-1]
    c = np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    cf = frame.to_fine(c)
    return np.floor(cf.min(0)), np.ceil(cf.max(0)) + 1


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", action="append", default=[], help="tifxyz dir (repeatable)")
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"),
                    help="two tifxyz dirs on one grid layout (e.g. <run>/<name>.before <run>/<name>); repeatable")
    ap.add_argument("--ct", help="CT pyramid url/path (default: the local cache of the store's volume)")
    ap.add_argument("--no-ct", action="store_true")
    ap.add_argument("--store", help="recto probability store (zarr v3 volcomp)")
    ap.add_argument("--layers", default="-12..12", help="normal offsets: LO..HI[:STEP] or a comma list (use --layers=-12..12)")
    ap.add_argument("--step", type=float, default=2.0, help="step for LO..HI without :STEP")
    ap.add_argument("--out", help="default: <first surface's parent>/render")
    ap.add_argument("--box", nargs=6, type=int, metavar=("Z", "Y", "X", "DZ", "DY", "DX"))
    ap.add_argument("--down", type=int, default=1, help="CT read at rung 2+log2(down)")
    ap.add_argument("--up", type=int, default=0, help="grid upsampling factor; 0 = to ~1 voxel pitch")
    ap.add_argument("--max-nodes", type=float, default=12e6,
                    help="--up 0: lower the upsampling until the pieces' dense bboxes total this many nodes")
    ap.add_argument("--far", type=float, default=12.0, help="ridge search half-length, voxels")
    ap.add_argument("--thr", type=float, default=0.5, help="a ridge is a recto local max >= thr")
    ap.add_argument("--tile", type=int, default=256, help="nodes per side of a sampling tile")
    ap.add_argument("--umbilicus", help="default: the store's attr")
    ap.add_argument("--transform", help="transform.json for legacy-frame surfaces")
    ap.add_argument("--transform-direction", default="auto", choices=("auto", "fine_to_legacy", "legacy_to_fine"))
    return ap


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean/nanmin over hole columns
    a = parser().parse_args(argv)
    log = lambda s: print(s, flush=True)   # noqa: E731
    if not a.surface and not a.compare:
        raise SystemExit("give --surface and/or --compare")
    layers = parse_layers(a.layers, a.step)
    far = int(round(a.far))
    from rvsm import axis as AX
    from rvsm import ladder
    attrs = dict(ladder.open_zarr(a.store).attrs) if a.store else {}
    umb = a.umbilicus or attrs.get("umbilicus")
    if not umb:
        raise SystemExit("--umbilicus is required (no --store naming one)")
    ax = AX.load(umb)
    ct = None
    if not a.no_ct:
        ct = a.ct or attrs.get("volume")
        loc = os.path.join(CT_CACHE, os.path.basename(str(ct).rstrip("/"))) if ct else None
        if not a.ct and loc and os.path.isdir(loc):
            ct = loc
    ct_src = CTSource(ct, a.down) if ct else None
    st_src = StoreSource(a.store) if a.store else None
    first = (a.compare[0][1] if a.compare else a.surface[0]).rstrip("/")
    out = a.out or os.path.join(os.path.dirname(first), "render")
    os.makedirs(out, exist_ok=True)
    log(json.dumps({"ct": ct, "store": a.store, "layers": layers.tolist(), "out": out, "box": a.box}))

    def box_for(d, fr):
        blo, bhi = surface_bbox(d, fr)
        if a.box:
            lo, sh = np.asarray(a.box[:3], np.float64), np.asarray(a.box[3:], np.float64)
        else:
            lo, sh = blo, bhi - blo
            if a.store:
                so, ss, _ = R.store_box(a.store)
                l2, h2 = np.maximum(lo, so), np.minimum(lo + sh, so + ss)
                lo, sh = l2, np.maximum(h2 - l2, 0)
        return lo.astype(np.float32), sh.astype(np.float32)

    def run_set(dirs):
        """Render the dirs on one shared window; [(name, [piece results])], labels, lw."""
        frs = [surface_frame(d, a) for d in dirs]
        o, s = box_for(dirs[0], frs[0])
        g0, rc = window(dirs[0], frs[0], o, s)
        if g0 is None:
            log(f"{dirs[0]}: no node in the box")
            return None
        grids = [g0]
        for d, fr in zip(dirs[1:], frs[1:]):
            gb, rcb = window(d, fr, o, s)
            if gb is None:
                grids.append(np.full_like(g0, np.nan))
                continue
            rc2 = (min(rc[0], rcb[0]), max(rc[1], rcb[1]), min(rc[2], rcb[2]), max(rc[3], rcb[3]))
            if rc2 != rc:   # widen to the union window and re-read everything on it
                rc = rc2
                grids = [window(dd, ff, o, s, rc)[0] for dd, ff in zip(dirs[:len(grids)], frs)]
            grids.append(window(d, fr, o, s, rc)[0])
        up = int(a.up) if a.up > 0 else auto_up_budget(grids, o, s, a.max_nodes)
        log(json.dumps({"surfaces": [os.path.basename(d.rstrip("/")) for d in dirs], "box": [*o.tolist(), *s.tolist()],
                        "window": rc, "up": up}))
        res, labels = [[] for _ in dirs], []
        for pi, p in enumerate(prepare(grids, ax, o, s, up)):
            for k in range(len(dirs)):
                res[k].append(render_piece(p["g"][k], p["n"][k], ct_src, st_src, layers, far, a.thr, a.tile, a.down))
            labels.append(f"rows {p['rc'][0]}..{p['rc'][1]} cols {p['rc'][2]}..{p['rc'][3]} (published grid), x{up}")
            log(f"piece {pi + 1} {p['g'][0].shape[:2]} sampled")
            del p
        del grids
        lw = plan_layout([r["ok"].shape for r in res[0]])
        return res, labels, lw

    summary = {}
    for d in a.surface:
        got = run_set([d])
        if got is None:
            continue
        res, labels, lw = got
        nm = os.path.basename(d.rstrip("/"))
        summary[nm] = write_surface(os.path.join(out, nm), nm, res[0], layers, far, lw, labels, log=log)
    for A, B in a.compare:
        got = run_set([A, B])
        if got is None:
            continue
        res, labels, lw = got
        na, nb = os.path.basename(A.rstrip("/")), os.path.basename(B.rstrip("/"))
        rng = ct_range(res)
        for nm, rr in ((na, res[0]), (nb, res[1])):
            summary[nm] = write_surface(os.path.join(out, nm), nm, rr, layers, far, lw, labels, rng=rng, log=log)
        run_name = os.path.basename(os.path.dirname(os.path.abspath(B.rstrip("/"))))
        base = nb.split("-on-")[0]
        p, sa, sb = write_compare(os.path.join(out, f"compare_{base}.png"),
                                  f"{run_name}: {nb}  (left {na}, right {nb})", (na, nb), res[0], res[1],
                                  layers, far, lw, labels)
        log(p)
        log(json.dumps({"compare": p, "a": {na: sa}, "b": {nb: sb}}))
    json.dump(summary, open(os.path.join(out, "render_stats.json"), "w"), indent=1)
    log(json.dumps({"done": len(summary)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
