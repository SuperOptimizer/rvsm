"""render.py on a synthetic sinusoidal sheet: the flattened d=0 layer is bright, the ridge offset ~0, and a
copy shifted along the normal reads its shift back."""
import json
import os

import numpy as np

from rvsm.tools.refine import render as RD


def _scene(Z=40, Y=72, X=64):
    f = lambda y: 32 + 5 * np.sin(2 * np.pi * y / 40)          # noqa: E731  sheet: x = f(y)
    fp = lambda y: 5 * 2 * np.pi / 40 * np.cos(2 * np.pi * y / 40)   # noqa: E731
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    dist = (x - f(y)) / np.sqrt(1 + fp(y) ** 2)
    sheet = np.exp(-dist ** 2 / (2 * 1.5 ** 2))
    ct = (30 + 200 * sheet).astype(np.uint8)
    st = (255 * sheet).astype(np.uint8)
    rr, cc = np.meshgrid(np.arange(4, Z - 4, 2.0), np.arange(4, Y - 4, 2.0), indexing="ij")
    g = np.stack([rr, cc, f(cc)], -1).astype(np.float32)          # (H,W,3) zyx, pitch 2 voxels
    ax = np.array([[0.0, 100.0], [36.0, 36.0], [-1000.0, -1000.0]])   # axis far on -x: normals point +x
    return ct, st, g, ax


def test_render_synthetic(tmp_path):
    ct, st, g, ax = _scene()
    o, s = np.zeros(3, np.float32), np.asarray(ct.shape, np.float32)
    shifted = g.copy()
    shifted[..., 2] += 3.0                                        # 3 voxels outward of the sheet
    pcs = list(RD.prepare([g, shifted], ax, o, s, up=2))
    assert len(pcs) == 1
    layers = RD.parse_layers("-12..12:2")
    assert len(layers) == 13 and layers[6] == 0
    ctS, stS = RD.ArraySource(ct), RD.ArraySource(st)
    ra = RD.render_piece(pcs[0]["g"][0], pcs[0]["n"][0], ctS, stS, layers, far=12, thr=0.5)
    rb = RD.render_piece(pcs[0]["g"][1], pcs[0]["n"][1], ctS, stS, layers, far=12, thr=0.5)
    ok = ra["ok"]
    assert ok.mean() > 0.8
    prof = np.nanmean(ra["ct"][:, ok], 1)
    assert prof.argmax() == 6 and prof[6] > 0.7 and prof[0] < 0.25 and prof[-1] < 0.25   # CT is read as 0..1
    assert np.nanmean(ra["st"][6][ok]) > 0.9
    off = ra["off"][ok]
    assert np.isfinite(off).mean() > 0.95 and np.nanmedian(np.abs(off)) < 0.3
    offb = rb["off"][rb["ok"]]
    assert abs(np.nanmedian(offb) + 3.0) < 0.4                     # the ridge is 3 voxels inward of B

    lw = RD.plan_layout([ok.shape])
    sa = RD.write_surface(str(tmp_path / "a"), "a", [ra], layers, 12, lw, ["p0"])
    assert sa["abs_median"] < 0.3
    for f in ("layers.tif", "ct_d0.png", "ct_mean.png", "ct_min.png", "ct_max.png", "ct_contact.png",
              "store_d0.png", "store_mean.png", "ridge_offset.png", "ridge_hist.png", "stats.json"):
        assert os.path.exists(tmp_path / "a" / f), f
    import tifffile
    stack = tifffile.imread(tmp_path / "a" / "layers.tif")
    assert stack.shape[0] == 13
    p, _, sb = RD.write_compare(str(tmp_path / "cmp.png"), "t", ("a", "b"), [ra], [rb], layers, 12, lw, ["p0"])
    assert os.path.exists(p) and os.path.exists(str(tmp_path / "cmp_hist.png"))
    assert abs(sb["signed_median"] + 3.0) < 0.4
    assert json.load(open(tmp_path / "a" / "stats.json"))["pieces"] == 1


def test_layout_wraps_long_strips():
    im = np.zeros((20, 3000, 3), np.uint8)
    lw = RD.plan_layout([im.shape[:2]])
    assert lw < 3000
    can = RD.layout([im], lw)
    assert can.shape[1] <= lw and can.shape[0] > 2 * 20
    k = np.zeros((10, 100), bool)
    k[2:5, 5:20] = True
    k[3:8, 60:70] = True
    assert len(RD.mask_pieces(k, pad=2)) == 2
