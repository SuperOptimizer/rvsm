"""More before | after crops of a finished refine run, at fresh places.

`refine` draws six crops per surface (densest / largest-move cells, one per z level). This re-reads a run's
output -- <run>/<name>/ (refined) and <run>/<name>.before/ (published), both fine-frame tifxyz -- and draws N
new crops per surface with refine.py's own renderer (plane_segments + crop_png), stacked into one montage
per surface and mode:

    random   N random z levels in the box, each at a random point of the published polyline there
    worst    alternately the refined node farthest from a recto ridge (nearest local max >= thr along its
             normal) and the node whose before -> after move is largest, each >= one crop from the others
    dense    refine's own pick_crops (densest / largest-move cells) at N jittered z levels

    python -m rvsm.tools.refine.crops --run DIR --out DIR [--per-surface 6 --crop 200 --scale 4
        --seed 0 --mode random,worst]

The recto/verso stores default to the ones named in the refined meta.json; the CT to the local cache of the
store's volume (--ct, or --no-ct). Everything is read inside the run's box only: the recto box (uint8) is the
one big array held (worst mode).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from rvsm.tools.refine import refine as R

CT_CACHE = "/home/forrest/refine/ct_cache/ct"


def load_pair(run, name, o, s, margin=32):
    """(before, after) fine zyx grids over the published rows/cols that touch the box (same window)."""
    got = R.read_surface_box(os.path.join(run, name + ".before"), R.IDENTITY, o, s, margin=margin)
    if got is None:
        return None
    before, (r0, c0), _ = got
    h, w = before.shape[:2]
    ch = R._tif_channels(os.path.join(run, name))
    after = np.stack([np.asarray(c[r0:r0 + h, c0:c0 + w], np.float32) for c in ch], -1)
    after = np.where((after > 0).all(-1)[..., None], after, np.nan)
    del ch
    return before, after


def segs_near(g, z, y0, x0, n, up=1, pad=48):
    """plane_segments of g at z, from only the cells near the plane and the crop window, upsampled by `up`
    there (refine draws its polylines on the upsampled grid) -- as a 1-list."""
    k = (np.isfinite(g).all(-1) & (np.abs(g[..., 0] - z) <= pad)
         & (g[..., 1] >= y0 - pad) & (g[..., 1] < y0 + n + pad) & (g[..., 2] >= x0 - pad) & (g[..., 2] < x0 + n + pad))
    if not k.any():
        return [np.zeros((0, 2, 2), np.float32)]
    rr, cc = np.nonzero(k.any(1))[0], np.nonzero(k.any(0))[0]
    sub = g[max(rr.min() - 1, 0):rr.max() + 2, max(cc.min() - 1, 0):cc.max() + 2]
    return [R.plane_segments(R.upsample(sub, up) if up > 1 and min(sub.shape[:2]) > 1 else sub, z)]


def ridge_dist(V, o, g, n, far=12, thr=0.5, chunk=200_000):
    """|offset| along the normal from each point to the nearest recto local max >= thr (far+1 if none)."""
    q = g - o
    out = []
    for i in range(0, len(q), chunk):
        S = R.profile(V, q[i:i + chunk], n[i:i + chunk], far)
        pos, st = R.local_maxima(S, far, thr)
        d = np.where(st >= thr, np.abs(pos), np.inf).min(0)
        out.append(np.where(np.isfinite(d), d, far + 1.0).astype(np.float32))
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def local_move(b_pts, a_pts, mv, z, cy, cx, n, dz=10.0):
    """(mean |move|, mean signed move) of the nodes whose published OR refined position is within the crop
    window and dz of the plane."""
    near = np.zeros(len(mv), bool)
    for p in (b_pts, a_pts):
        near |= (np.abs(p[:, 0] - z) <= dz) & (np.abs(p[:, 1] - cy) <= n / 2) & (np.abs(p[:, 2] - cx) <= n / 2)
    return (float(np.abs(mv[near]).mean()), float(mv[near].mean())) if near.any() else (float("nan"), float("nan"))


def caption(im, text):
    """im with a second label line above it (crop_png's own line holds only ~70 characters)."""
    from PIL import Image, ImageDraw
    out = Image.new("RGB", (im.size[0], im.size[1] + 14), (255, 255, 255))
    out.paste(im, (0, 14))
    ImageDraw.Draw(out).text((3, 1), text, fill=(0, 0, 0))
    return out


def montage(imgs, path):
    from PIL import Image
    wd, ht = max(i.size[0] for i in imgs), sum(i.size[1] for i in imgs) + 4 * (len(imgs) - 1)
    mt = Image.new("RGB", (wd, ht), (255, 255, 255))
    yy = 0
    for i in imgs:
        mt.paste(i, (0, yy))
        yy += i.size[1] + 4
    mt.save(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a refine --out directory (refine_report.json, <name>/, <name>.before/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-surface", type=int, default=6)
    ap.add_argument("--crop", type=int, default=200, help="panel size, pixels")
    ap.add_argument("--scale", type=int, default=4, help="pixels per voxel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="random,worst", help="comma list of random|worst|dense")
    ap.add_argument("--recto", help="default: the refined meta.json's")
    ap.add_argument("--verso", help="default: the refined meta.json's")
    ap.add_argument("--umbilicus", help="default: the recto store's attr")
    ap.add_argument("--ct", help="CT pyramid (default: the local cache of the recto store's volume)")
    ap.add_argument("--no-ct", action="store_true")
    ap.add_argument("--far", type=int, default=12, help="worst mode: ridge search half-length, voxels")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--edge", type=float, default=8.0, help="keep crop centres this far inside the box in z")
    ap.add_argument("--surfaces", help="comma list of names (default: the report's)")
    a = ap.parse_args(argv)
    log = lambda s: print(s, flush=True)   # noqa: E731
    modes = [m for m in a.mode.split(",") if m]
    bad = set(modes) - {"random", "worst", "dense"}
    if bad:
        raise SystemExit(f"unknown mode(s) {sorted(bad)}")
    rep = json.load(open(os.path.join(a.run, "refine_report.json")))
    lo, shape = np.asarray(rep["box"][:3], np.int64), np.asarray(rep["box"][3:], np.int64)
    o, s = lo.astype(np.float32), shape.astype(np.float32)
    names = a.surfaces.split(",") if a.surfaces else rep["surfaces"]
    meta = json.load(open(os.path.join(a.run, names[0], "meta.json"))).get("refined", {})
    recto, verso = a.recto or meta["recto"], a.verso or meta.get("verso") or None
    from rvsm import axis as AX
    from rvsm import ladder
    attrs = dict(ladder.open_zarr(recto).attrs)
    ax = AX.load(a.umbilicus or attrs["umbilicus"])
    ct = None
    if not a.no_ct:
        ct = a.ct or attrs.get("volume")
        loc = os.path.join(CT_CACHE, os.path.basename(str(ct).rstrip("/"))) if ct else None
        if not a.ct and loc and os.path.isdir(loc):
            ct = loc
    n = a.crop // a.scale
    os.makedirs(a.out, exist_ok=True)
    log(json.dumps({"box": rep["box"], "surfaces": len(names), "modes": modes, "n": a.per_surface,
                    "crop_vox": n, "recto": recto, "verso": verso, "ct": ct}))

    # ---- every surface's before/after grids at the published pitch (upsampled only where a crop draws them:
    # 14 dense grids are ~1 GB each) and its per-node normal move
    S = {}
    for nm in names:
        got = load_pair(a.run, nm, o, s)
        if got is None:
            log(f"{nm}: no point in the box")
            continue
        b, af = got
        up = R.auto_up(b, 4.0)
        nb = R.normals(b, ax)
        mv = ((af - b) * np.nan_to_num(nb)).sum(-1)
        S[nm] = {"b": b, "a": af, "mv": mv.astype(np.float32), "up": up}
        del nb
        log(json.dumps({"surface": nm, "grid": list(b.shape[:2]), "up": up}))
    V = None
    if "worst" in modes:
        log("reading the recto box")
        V = R.read_box(recto, lo, shape)

    def read_plane(path, z, y0, x0):
        return R.read_box(path, (int(z), int(y0), int(x0)), (1, n, n))[0].astype(np.float32) / 255.0 if path else None

    def ct_crop(z, y0, x0):
        if not ct:
            return None
        try:
            return R.read_ct(ct, (int(z), int(y0), int(x0)), (1, n, n))[0]
        except Exception as e:  # noqa: BLE001 - a picture without CT beats no picture
            log(f"CT read failed ({e!r})")
            return None

    def render(nm, z, cy, cx, label):
        y0 = int(np.clip(int(cy) - n // 2, lo[1], lo[1] + shape[1] - n))
        x0 = int(np.clip(int(cx) - n // 2, lo[2], lo[2] + shape[2] - n))
        me = S[nm]
        others = [(segs_near(S[k]["b"], z, y0, x0, n, S[k]["up"]), segs_near(S[k]["a"], z, y0, x0, n, S[k]["up"]))
                  for k in S if k != nm]
        return R.crop_png(read_plane(recto, z, y0, x0), read_plane(verso, z, y0, x0), ct_crop(z, y0, x0),
                          segs_near(me["b"], z, y0, x0, n, me["up"]), segs_near(me["a"], z, y0, x0, n, me["up"]), others,
                          y0, x0, a.scale, label)

    zlo, zhi = float(lo[0] + a.edge), float(lo[0] + shape[0] - a.edge)
    ylo, yhi = float(lo[1] + n / 2), float(lo[1] + shape[1] - n / 2)
    xlo, xhi = float(lo[2] + n / 2), float(lo[2] + shape[2] - n / 2)
    outs = []
    for si, nm in enumerate(S):
        me = S[nm]
        ok = np.isfinite(me["b"]).all(-1) & np.isfinite(me["a"]).all(-1) & np.isfinite(me["mv"])
        bp, ap_, mvp = me["b"][ok], me["a"][ok], me["mv"][ok]
        inb = ((bp[:, 0] >= zlo) & (bp[:, 0] < zhi) & (bp[:, 1] >= ylo) & (bp[:, 1] < yhi)
               & (bp[:, 2] >= xlo) & (bp[:, 2] < xhi))
        short = nm.split("-on-")[0]
        for mode in modes:
            rng = np.random.default_rng([a.seed, si, ["random", "worst", "dense"].index(mode)])
            picks = []   # (z, cy, cx, extra label)
            if mode == "random":
                tries = 0
                while len(picks) < a.per_surface and tries < 50 * a.per_surface:
                    tries += 1
                    z = float(int(rng.uniform(zlo, zhi))) + 0.5
                    sg = R.plane_segments(me["b"], z)
                    if not len(sg):
                        continue
                    m = sg.mean(1)
                    m = m[(m[:, 0] >= ylo) & (m[:, 0] < yhi) & (m[:, 1] >= xlo) & (m[:, 1] < xhi)]
                    if not len(m):
                        continue
                    cy, cx = m[rng.integers(len(m))]
                    picks.append((z, float(cy), float(cx), "random"))
            elif mode == "worst":
                q = ap_[inb]
                rd = ridge_dist(V, o, q, np.nan_to_num(R.normals(me["a"], ax)[ok][inb]), far=a.far, thr=a.thr)
                mag = np.abs(mvp[inb])
                if len(q):
                    tie = rng.random(len(q))   # many nodes share rd = far+1 (no ridge in reach): spread them
                    orders = [np.lexsort((tie, -rd)), np.lexsort((tie, -mag))]
                    ptr = [0, 0]
                    while len(picks) < a.per_surface:
                        which = len(picks) % 2
                        od = orders[which]
                        found = None
                        while ptr[which] < len(od):
                            i = od[ptr[which]]
                            ptr[which] += 1
                            if all(abs(q[i, 1] - p[1]) >= n or abs(q[i, 2] - p[2]) >= n or abs(q[i, 0] - p[0]) >= 16
                                   for p in picks):
                                found = i
                                break
                        if found is None:
                            if ptr[0] >= len(orders[0]) and ptr[1] >= len(orders[1]):
                                break
                            continue
                        i = found
                        if which == 0:
                            why = (f"ridge {rd[i]:.1f} vox off" if rd[i] <= a.far
                                   else f"no ridge within {a.far} vox")
                            c = q[i]
                        else:   # centred between the published and refined node, so both show when they fit
                            why = f"max move {mvp[inb][i]:+.1f}"
                            c = 0.5 * (q[i] + bp[inb][i])
                        picks.append((float(int(c[0])) + 0.5, float(c[1]), float(c[2]), why))
            else:  # dense: refine's pick_crops at jittered, evenly spread z levels
                zs = np.linspace(zlo, zhi, a.per_surface + 1)[:-1] + rng.uniform(0, (zhi - zlo) / a.per_surface,
                                                                                   a.per_surface)
                cd = {}
                for z in sorted(float(int(v)) + 0.5 for v in zs):
                    band = np.abs(bp[:, 0] - z) <= 2.5
                    cd[z] = ([], [], [bp[band][:, 1:]] if band.any() else [], [np.abs(mvp[band])] if band.any() else [])
                picks = [(z, cy, cx, why) for z, cy, cx, _, why in R.pick_crops(cd, a.per_surface, win=n)]
            imgs = []
            for z, cy, cx, why in picks:
                mabs, msgn = local_move(bp, ap_, mvp, z, cy, cx, n)
                im = render(nm, z, cy, cx, f"{short} z{int(z)} yx {int(cy)},{int(cx)}")
                imgs.append(caption(im, f"{mode}: move {msgn:+.1f} vox (|{mabs:.1f}|), {why}"))
            if imgs:
                p = os.path.join(a.out, f"{nm}_{mode}.png")
                outs.append(montage(imgs, p))
                log(json.dumps({"png": p, "n": len(imgs), "picks": [[round(v, 1) for v in pk[:3]] + [pk[3]] for pk in picks]}))
    log(json.dumps({"done": len(outs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
