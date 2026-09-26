"""The THINNED-BAND rung-2 target (`thin_band`, docs/recipe.md §6 "Thinned band target").

The rung-2 recto target of an m7-only run is m7's 9.6 um output upsampled 4x: a soft band 4-8 voxels
wide. With `thin_band = 1` the trainer replaces it (on the device, per step) by a `thin_band_width`-voxel
sheet about the band's medial surface, weight 0 on the band's flanks, background unchanged.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from rvsm import config as CFG, losses as L, train as TR
from tests.test_route import _routed_item
from tests.test_train import _factory


def _plane(n=32, lo=12, hi=20, p=0.9):
    t = torch.zeros(1, 1, n, n, n)
    t[..., lo:hi] = p                                   # a band 8 voxels wide about x = 15.5
    return t


def test_a_plane_band_thins_to_a_sheet_on_its_medial_plane():
    t = _plane()
    tt, kp = L.thin_band(t, width=4.0, soft=1.0)
    prof, keep = tt[0, 0, 16, 16].tolist(), kp[0, 0, 16, 16].tolist()
    assert prof[15] == 1.0 and prof[16] == 1.0          # the medial plane (two voxels: even width)
    assert prof[14] == 1.0 and prof[17] == 1.0          # within width/2 - soft of it
    for x in (12, 13, 18, 19):                          # >= 2 voxels from M: off the sheet ...
        assert prof[x] == 0.0 and keep[x] == 0.0        # ... and not punished either way
    for x in list(range(0, 12)) + list(range(20, 32)):  # outside the band: background, full weight
        assert prof[x] == 0.0 and keep[x] == 1.0
    assert torch.equal(tt[0, 0], tt[0, 0, :1, :1].expand_as(tt[0, 0]))   # the same on every line
    # an odd width puts the fade on the grid: 9 wide about x = 16, width 5, soft 1 -> 1 at |x-16| <= 1,
    # 0.5 at |x-16| = 2, 0 at 3+
    tt9, kp9 = L.thin_band(_plane(lo=12, hi=21), width=5.0, soft=1.0)
    p9 = tt9[0, 0, 5, 5].tolist()
    assert p9[15:18] == [1.0, 1.0, 1.0] and p9[14] == 0.5 and p9[18] == 0.5
    assert p9[13] == 0.0 and p9[19] == 0.0 and kp9[0, 0, 5, 5, 13] == 0.0 and kp9[0, 0, 5, 5, 14] == 1.0


def test_a_curved_band_thins_about_its_middle_surface():
    n, r0 = 48, 16.0
    z, y, x = np.indices((n, n, n)) - (n - 1) / 2.0
    r = np.sqrt(z * z + y * y + x * x)
    t = torch.from_numpy(((np.abs(r - r0) < 4.0) * 0.8).astype(np.float32)).view(1, 1, n, n, n)
    tt, kp = L.thin_band(t, width=4.0, soft=1.0)
    tt, kp = tt[0, 0].numpy(), kp[0, 0].numpy()
    mid = np.abs(r - r0) < 0.5
    assert (tt[mid] == 1.0).mean() > 0.97               # the sheet sits on the band's middle surface
    assert (tt[np.abs(r - r0) >= 3.0] == 0).all()      # ~0 three voxels out
    flank = (np.abs(r - r0) >= 3.0) & (np.abs(r - r0) < 4.0)
    assert (kp[flank] == 0).all() and (kp[np.abs(r - r0) >= 4.0] == 1).all()


def test_an_empty_band_and_unknown_weight():
    z = torch.zeros(2, 1, 16, 16, 16)
    tt, kp = L.thin_band(z)
    assert (tt == 0).all() and (kp == 1).all()
    t = _plane()
    tg = torch.cat([t, torch.zeros_like(t), torch.full_like(t, 0.5), torch.full_like(t, 0.5)], 1)
    wv = torch.ones_like(tg)
    wv[:, :, :8] = 0                                    # unknown (e.g. outside the stores)
    t2, w2 = L.thin_apply(tg, wv, [0])
    assert (w2[:, :, :8] == 0).all()                    # unknown stays unknown
    assert torch.equal(t2[:, 1:], tg[:, 1:]) and torch.equal(w2[:, 1:], wv[:, 1:])   # other rows untouched
    t3, w3 = L.thin_apply(tg, wv, [])                   # no rung-2 sample: bit-identical
    assert t3 is tg and w3 is wv


def test_the_routing_gap_gets_the_direct_thinned_target():
    n = 32
    tb = _plane(n)                                      # the m7 band the fine teacher missed
    wb = torch.full_like(tb, 0.5)                       # routed, c_A = 0 everywhere
    wb[:, :, :, :, :4] = 1.0                            # ... trusted in a corner far from the band
    wb[:, :, :4] = 0.0                                  # ... and an unrouted slab (z < 4)
    rec = torch.zeros_like(tb)                          # the fine teacher sees nothing
    rec[:, :, :4] = tb[:, :, :4]                        # the unrouted slab's recto is still m7's band
    tg = torch.cat([rec, torch.zeros_like(rec), torch.full_like(rec, 0.5), torch.full_like(rec, 0.5)], 1)
    wv = torch.ones_like(tg)
    m = L.route_masks(tb, wb, dilate=2)
    t2, w2, wc = L.route_apply(tg, wv, tb, m)
    gap = m["gap"].bool()
    assert (w2[:, :1][gap] == 0).all()                  # routing alone: no direct term in the gap
    t3, w3 = L.thin_apply(t2, w2, [0], width=4.0, soft=1.0, orig=(tg[:, :1], wv[:, :1]), tb=tb, m=m)
    prof_t, prof_w = t3[0, 0, 16, 16].tolist(), w3[0, 0, 16, 16].tolist()
    assert prof_t[15] == 1.0 and prof_w[15] == 1.0      # the gap's sheet: a DIRECT target now
    assert prof_w[12] == 0.0 and prof_t[12] == 0.0      # the band's flank: still no direct term
    assert prof_w[10] == 1.0 and prof_t[10] == 0.0      # the dilation ring (outside B): background
    assert prof_w[5] == 1.0                             # outside: route_apply's, unchanged
    assert torch.equal(t3[:, :1][m["outside"].bool()], t2[:, :1][m["outside"].bool()])
    assert torch.equal(t3[:, :1][m["ca"].bool()], t2[:, :1][m["ca"].bool()])
    assert torch.equal(w3[:, :1][m["ca"].bool()], w2[:, :1][m["ca"].bool()])   # c_A = 1 unchanged
    ur = t3[0, 0, 1, 16].tolist()                       # the unrouted slab: the standalone thinning
    assert ur[15] == 1.0 and ur[12] == 0.0 and w3[0, 0, 1, 16, 12] == 0.0
    assert torch.equal(t3[:, 1:], t2[:, 1:])


def test_the_switch_is_resume_switchable():
    assert CFG.Config().thin_band == 0
    for k in ("thin_band", "thin_band_width", "thin_band_soft"):
        assert k in CFG.FINGERPRINT_EXCLUDE and k in CFG.LOSS_SWITCH_FIELDS
    assert CFG.Config(thin_band=1).fingerprint() == CFG.Config().fingerprint()


def test_off_never_touches_the_targets(small_cfg, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("thin_band ran with the switch off")
    monkeypatch.setattr(L, "thin_band", boom)
    monkeypatch.setattr(L, "thin_apply", boom)
    TR.train(replace(small_cfg, steps=2, eval_every=1000), patches_factory=_factory(small_cfg), device="cpu")


def test_twenty_steps_with_the_thinned_band(small_cfg, monkeypatch):
    cfg = replace(small_cfg, thin_band=1, steps=20, eval_every=1000)
    seen = []
    real = L.thin_apply

    def spy(tg, wv, sel, **kw):
        seen.append(list(sel))
        return real(tg, wv, sel, **kw)
    monkeypatch.setattr(L, "thin_apply", spy)
    TR.train(cfg, patches_factory=_factory(cfg), device="cpu")
    assert seen and all(s == [0] for s in seen)         # rung-2 samples only (rung 3 untouched)
    rows = [json.loads(q) for q in (Path(cfg.out) / "logs" / "train.jsonl").read_text().splitlines()]
    tz = [r["thin_zero"] for r in rows if "thin_zero" in r]
    assert tz and all(np.isfinite(v) and 0 < v < 1 for v in tz)
    assert all(r["thin_gain"] == 0.0 and "thin_gap_w" not in r for r in rows if "thin_zero" in r)
    assert all(np.isfinite(r["loss"]) for r in rows if "loss" in r)


def test_the_routed_trainer_takes_the_thinned_gap(small_cfg, monkeypatch):
    cfg = replace(small_cfg, rungs=(2,), teacher_route={"2": "recto"}, loss_band=1.0, thin_band=1,
                  steps=20, eval_every=1000, aug="none")
    seen = {}
    real = L.thin_apply

    def spy(tg, wv, sel, **kw):
        got = real(tg, wv, sel, **kw)
        gap = kw["m"]["gap"].bool()
        seen["gap_w"] = float(got[1][:, :1][gap].sum())
        seen["gap_w0"] = float(wv[:, :1][gap].sum())
        return got
    monkeypatch.setattr(L, "thin_apply", spy)

    def gen():
        for i in range(10 ** 6):
            yield _routed_item(cfg, seed=i)
    TR.train(cfg, patches_factory=gen, device="cpu")
    assert seen["gap_w0"] == 0 and seen["gap_w"] > 0   # the gap now carries a direct term
    # ... and the log says so: every voxel is routed here, so nothing with weight was zeroed
    # (thin_zero 0.0, as on paris4), while the gap's direct target shows as thin_gain / thin_gap_w
    rows = [json.loads(q) for q in (Path(cfg.out) / "logs" / "train.jsonl").read_text().splitlines()]
    tr = [r for r in rows if "thin_gain" in r]
    assert tr and all(r["thin_zero"] == 0.0 and r["thin_gain"] > 0 and 0 < r["thin_gap_w"] <= 1
                      for r in tr)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_the_gpu_matches_the_cpu():
    n, r0 = 48, 16.0
    z, y, x = np.indices((n, n, n)) - (n - 1) / 2.0
    r = np.sqrt(z * z + y * y + x * x)
    t = torch.from_numpy(((np.abs(r - r0) < 4.0) * 0.8).astype(np.float32)).view(1, 1, n, n, n)
    a, ka = L.thin_band(t)
    b, kb = L.thin_band(t.cuda())
    assert torch.equal(a, b.cpu()) and torch.equal(ka, kb.cpu())
