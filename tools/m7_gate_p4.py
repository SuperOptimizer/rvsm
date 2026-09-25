#!/usr/bin/env python3
"""Stride gate on the Paris 4 val box (m7 at level 2 = 9.6 um of the 2.4 um volume).

A 448^3 level-2 box centred on the usrm2 val box [34432,15104,18432] + (256,1024,1024) (L0) is predicted
with each stride; every stride is scored against stride 96 on the inner box (margin 96) and against the
usrm2 store p4val256_m7_tta0 (L0, 4x mean-pooled onto the level-2 grid) over the val box, dice at p>=0.5
and corr over CT>0."""
import json, os, sys, time
import numpy as np, torch, zarr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volcomp_zarr  # noqa: F401
import m7_engine_lab as LAB
import m7_wholevol as M

CT = "/vesuvius/usrm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2"
REF = "/vesuvius/usrm2/teacher/p4val256_m7_tta0.zarr"
lo0, sz0 = np.array([34432, 15104, 18432]), np.array([256, 1024, 1024])
c2 = (lo0 + sz0 // 2) // 4
H = 224
b0 = c2 - H
ct = zarr.open_array(CT, mode="r")[b0[0]:b0[0] + 2 * H, b0[1]:b0[1] + 2 * H, b0[2]:b0[2] + 2 * H]
fg = ct > 0
ref_store = zarr.open_array(REF, mode="r")[:].astype(np.float32) / 255.0
rs = ref_store.reshape(64, 4, 256, 4, 256, 4).mean(axis=(1, 3, 5))
vo = lo0 // 4 - b0                     # val box inside our box (level 2)
vs = (slice(vo[0], vo[0] + 64), slice(vo[1], vo[1] + 256), slice(vo[2], vo[2] + 256))
dev = torch.device("cuda:0")
plan = sys.argv[1]
strides = [int(s) for s in sys.argv[2].split(",")]
eng = LAB.Eng(plan, dev)
inner = (slice(96, -96),) * 3
base = None
for st in strides:
    t = time.time()
    p, nwin = LAB.box_pred(ct, eng, 192, st, dev)
    p = np.where(fg, p, 0)
    if base is None:
        base = p
    def sc(a, b, m):
        a1, b1 = a >= 0.5, b >= 0.5
        return round(float(2 * (a1 & b1).sum() / max(1, a1.sum() + b1.sum())), 4), round(float(np.corrcoef(a[m], b[m])[0, 1]), 4)
    d96, c96 = sc(p[inner], base[inner], fg[inner])
    dr, cr = sc(p[vs], np.where(fg[vs], rs, 0), fg[vs])
    print(json.dumps({"stride": st, "step": round(st / 192, 3), "windows": nwin, "s": round(time.time() - t, 1),
                      "dice_vs_s96": d96, "corr_vs_s96": c96, "dice_vs_p4val256_m7": dr, "corr_vs_p4val256_m7": cr}), flush=True)
