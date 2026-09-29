"""Re-score finished refine runs against ANOTHER store (a fair, non-circular before/after).

    python -m rvsm.tools.refine.evalrun --store TEACHER.zarr --box Z Y X DZ DY DX --umbilicus U.json RUN [RUN ...]

For every `<name>.before` / `<name>` pair a run wrote, both grids are scored with `refine.surface_metrics` (recall@2/4,
offset, merge_frac, continuity, ERL) against `--store`, over the points inside the box AND the store's own box (16
voxels in, so the profiles stay inside it) -- the rule `refine --eval-store` uses. Fine-pitch outputs
(`--write-pitch fine`) are scored as written; published-pitch outputs are cropped to the box and upsampled by the
run's `up`, the before and the after alike. Per surface and pooled by point count, one JSON line per run."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from rvsm.tools.refine import refine as R


def load(d, lo, shape, margin=64):
    from rvsm import evalsurf as E
    g = E.read_surface(d)
    meta = json.load(open(os.path.join(d, "meta.json"))).get("refined", {})
    return g, meta


def main(argv=None):
    from rvsm import axis as AX
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--store", required=True)
    ap.add_argument("--box", type=int, nargs=6, required=True)
    ap.add_argument("--umbilicus", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--inset", type=float, default=16.0)
    a = ap.parse_args(argv)
    ax = AX.load(a.umbilicus)
    lo, shape = np.asarray(a.box[:3], np.int64), np.asarray(a.box[3:], np.int64)
    so, ss, _ = R.store_box(a.store)
    e_lo = np.maximum(so + a.inset, lo).astype(np.float32)
    e_hi = np.minimum(so + ss - a.inset, lo + shape).astype(np.float32)
    if (e_hi <= e_lo).any():
        raise SystemExit("the store does not cover the box")
    pad = 48
    rlo = np.maximum(lo - [0, pad, pad], so)
    rhi = np.minimum(lo + shape + [0, pad, pad], so + ss)
    V = R.read_box(a.store, rlo, rhi - rlo)
    for run in a.runs:
        rep, tot = {}, {"before": {}, "after": {}}
        for bd in sorted(glob.glob(os.path.join(run, "*.before"))):
            nm = os.path.basename(bd)[:-len(".before")]
            ad = os.path.join(run, nm)
            if not os.path.isdir(ad):
                continue
            b, _ = load(bd, lo, shape)
            g, meta = load(ad, lo, shape)
            if b.shape != g.shape:
                continue
            if meta.get("write_pitch") != "fine":   # published pitch: the box crop, upsampled like the run did
                k = np.isfinite(b).all(-1) & ((b >= lo - 64) & (b < lo + shape + 64)).all(-1)
                if not k.any():
                    continue
                rr, cc = np.nonzero(k.any(1))[0], np.nonzero(k.any(0))[0]
                sl = (slice(max(rr.min() - 2, 0), rr.max() + 3), slice(max(cc.min() - 2, 0), cc.max() + 3))
                up = int(meta.get("up", 1) or 1)
                b, g = R.upsample(b[sl], up), R.upsample(g[sl], up)
            m = ((b >= e_lo) & (b < e_hi)).all(-1) & ((g >= e_lo) & (g < e_hi)).all(-1)
            if not m.any():
                continue
            rec = {}
            for key, grid in (("before", b), ("after", g)):
                mm = R.surface_metrics(V, rlo, grid, ax, thr=a.thr, mask=m)
                if not mm.get("n_points"):
                    continue
                rec[key] = {q: round(float(mm[q]), 4) for q in R.EVAL_KEYS if q in mm and np.isfinite(mm[q])}
                rec[key]["n_points"] = int(mm["n_points"])
                for q, v in rec[key].items():
                    if q != "n_points":
                        tot[key].setdefault(q, []).append((v, mm["n_points"]))
            rep[nm] = rec
        pooled = {k: {q: round(sum(v * w for v, w in vals) / max(sum(w for _, w in vals), 1), 4) for q, vals in t.items()}
                  for k, t in tot.items()}
        print(json.dumps({"run": run, "store": a.store, "eval_box": [*e_lo.tolist(), *(e_hi - e_lo).tolist()],
                          "pooled": pooled, "surfaces": rep}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
