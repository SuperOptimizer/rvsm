"""Per-rung teacher ROUTING with gap-fill (`config.route_spec`, docs/recipe.md §7).

`teacher_route = {"2": "recto", "3": "m7", "4": "m7"}`: rung-2 recto targets from the 2.4 um recto
teacher where it is confident (c_A), the m7 probability beside them as the `band` store, rungs >= 3 from
m7 as before. The producer regenerates every round-0 recto into a routed generation (reusing a committed
m7 store as the band), the sampler adds one ROUTING row at rung 2, and the trainer turns it into the
gap-fill masks. The default (`{}`) changes nothing.
"""
import dataclasses
import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from rvsm import config as CFG, infer, losses as L, regions as RG, run as RUN, sample, stores, train as TR

ROUTE = {"2": "recto", "3": "m7", "4": "m7"}


def _fake_store(path, **attrs):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"attributes": {"done": True, **attrs}}, f)


# ------------------------------------------------------------------------------ config

def test_the_default_routes_nothing_and_keeps_the_fingerprint(small_cfg):
    base = small_cfg.fingerprint()
    assert CFG.Config().teacher_route == {} and CFG.route_spec(small_cfg) is None
    for kw in ({"teacher_route": ROUTE}, {"loss_band": 0.5}, {"band_dilate": 3}, {"band_eps": 0.1}):
        assert replace(small_cfg, **kw).fingerprint() == base, kw
    assert sample.target_channels(small_cfg) == ["recto", "verso", "midline", "thickness"]
    assert "route" not in sample.grid_global(small_cfg, 0, ("recto",))
    assert RUN.teacher_set(small_cfg) == RUN.teacher_names(small_cfg)
    d = CFG.Config()
    assert (d.loss_band, d.band_dilate, d.band_eps) == (0.0, 2, 0.05)
    assert {"loss_band", "band_dilate"} <= set(CFG.LOSS_SWITCH_FIELDS)


def test_the_route_spec():
    r = CFG.route_spec(CFG.Config(teacher_route=ROUTE))
    assert (r.fine, r.base, r.fine_rungs) == ("recto", "m7", (2,)) and "cA-v1" in r.sig
    assert CFG.route_spec(CFG.Config(teacher_route={"2": "recto"})).base == "m7"
    assert CFG.route_spec(CFG.Config(teacher_route={"2": "m7", "3": "m7"})) is None   # nothing to route
    for bad in ({"3": "recto", "4": "m7"}, {"2": "recto", "3": "m7", "4": "x"}, {"z": "m7"}):
        with pytest.raises(ValueError):
            CFG.route_spec(CFG.Config(teacher_route=bad))
    # TOML hands the keys as strings; a config.json round trip keeps the route
    c = CFG.load(None, {"teacher_route": ROUTE})
    assert CFG.route_spec(c) == r
    c = CFG.Config(teacher_route=ROUTE)
    assert sample.target_channels(c)[-1] == "band"
    assert sample.grid_global(c, 0, ("recto",))["route"] == r.sig
    assert RUN.teacher_set(c) == ["recto", "m7", RUN.ROUTE_TOKEN + r.sig]


# ------------------------------------------------------------------------------ the coverage rule

def test_the_coverage_rule_trusts_confident_voxels_near_the_fine_teachers_faces():
    P = np.full((96, 16, 16), 20, np.uint8)            # confident background everywhere ...
    P[40:42] = 255                                      # ... one face ...
    P[43] = 120                                         # ... and an undecided voxel beside it
    P[95] = 160                                         # a confident voxel (itself a face) far away
    c = infer.route_coverage_u8(P, pool=1).numpy()
    assert c[40, 0, 0] == 255 and c[41, 0, 0] == 255   # the face
    assert c[43, 0, 0] == 0                             # 0.2 < p < 0.6: undecided
    assert c[44, 0, 0] == 255 and c[41 + 24, 0, 0] == 255 and c[40 - 24, 0, 0] == 255   # within REACH
    assert c[41 + 25, 0, 0] == 0 and c[40 - 25, 0, 0] == 0   # background beyond REACH: not trusted
    assert c[95, 0, 0] == 255                           # p >= 0.6: trusted anywhere
    # the pooled production path is exact to within pool - 1 voxels
    c4 = infer.route_coverage_u8(P).numpy()
    assert c4.shape == P.shape and (c4[44:66, 0, 0] == 255).all() and c4[43, 0, 0] == 0 and c4[0:12, 0, 0].max() == 0
    assert (c4[40 - 24 + 3:40, 0, 0] == 255).all()


# ------------------------------------------------------------------------------ the trainer's masks

def _t(a):
    return torch.as_tensor(np.asarray(a, np.float32)).view(1, 1, -1, 1, 1)


def test_the_routing_masks_and_the_gap_fill_target():
    #            0     1     2     3     4     5     6     7     8     9
    tb = _t([0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])      # the m7 band at 3-4
    wb = _t([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5])      # c_A at 0; unrouted at 8
    m = L.route_masks(tb, wb, dilate=1)
    f = {k: v.flatten().tolist() for k, v in m.items()}
    assert f["routed"] == [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    assert f["ca"] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert f["band"] == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]                 # 3-4 dilated by 1
    assert f["gap"] == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert f["outside"] == [0, 1, 0, 0, 0, 0, 1, 1, 0, 1]
    rec = _t([0.8, 0.3, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.7, 0.1])       # the fine teacher
    tg = torch.cat([rec, torch.zeros_like(rec), torch.full_like(rec, 0.5)], 1)
    wv = torch.ones_like(tg)
    t2, w2, wc = L.route_apply(tg, wv, tb, m)
    t0, w0 = t2[:, 0].flatten().tolist(), w2[:, 0].flatten().tolist()
    # c_A = 1 and unrouted: the fine teacher; c_A = 0: the base teacher
    assert t0[0] == pytest.approx(0.8) and t0[8] == pytest.approx(0.7)
    assert t0[1] == 0.0 and t0[6] == 0.0 and t0[3] == pytest.approx(0.9)
    assert w0 == [1, 1, 0, 0, 0, 0, 1, 1, 1, 1]                        # no direct BCE / dice in a gap
    assert torch.equal(t2[:, 1:], tg[:, 1:]) and torch.equal(w2[:, 1:], wv[:, 1:])   # other rows untouched
    assert torch.equal(wc, wv)                  # the continuity terms keep the gap's weight
    # the band penalty: relu(p - eps)^2 over `outside` only
    p = _t([0.9, 0.05, 0.9, 0.9, 0.9, 0.9, 0.35, 0.0, 0.9, 0.55])
    got = float(L.band_penalty(p, m["outside"], eps=0.05))
    assert got == pytest.approx((0.0 + 0.3 ** 2 + 0.0 + 0.5 ** 2) / 4, rel=1e-5)
    assert float(L.band_penalty(p, torch.zeros_like(p))) == 0.0


def test_the_skeleton_of_the_gap_is_the_base_band(small_cfg):
    """In a gap zone the skeleton-recall target is the base band's skeleton: route_apply puts the base
    teacher's probability in the recto row there, and the continuity weight keeps the gap."""
    n = 16
    tb = torch.zeros(1, 1, n, n, n)
    tb[:, :, 7:10] = 1.0                                 # an m7 sheet the fine teacher missed
    wb = torch.full_like(tb, 0.5)                        # c_A = 0 everywhere: all gap / outside
    tg = torch.zeros(1, 4, n, n, n)
    wv = torch.ones_like(tg)
    m = L.route_masks(tb, wb, dilate=2)
    t2, w2, wc = L.route_apply(tg, wv, tb, m)
    lay = replace(small_cfg, channels=("recto", "verso")).layout()
    logit = torch.full((1, lay.cout, n, n, n), 4.0)      # predicts the sheet everywhere: full recall
    got = L.aux_losses(logit, t2, w2, lay, w_skel=1.0, w_cont=wc)
    assert float(got["skel"]) < 0.05
    got0 = L.aux_losses(logit, t2, w2, lay, w_skel=1.0)  # without w_cont the gap has no weight
    assert float(got0["skel"]) == 1.0          # (no live skeleton weight: no recall credit)


# ------------------------------------------------------------------------------ the state machine

def test_a_routed_config_regenerates_an_m7_store_and_reuses_it_as_the_band(tmp_path):
    out, lo = str(tmp_path / "r"), (0, 0, 0)
    ts = RUN.teacher_set(CFG.Config(teacher_route=ROUTE))
    _fake_store(stores.store_path(out, "recto", lo, 0), teachers=["m7"])
    _fake_store(stores.store_path(out, "rw", lo, 0), teachers=["m7"])
    cat = RG.Catalog(out, 0, ttl=0.0)
    assert RUN._next_job(cat, lo, 0, False, out, teachers=["m7"]) is None
    assert RUN._next_job(cat, lo, 0, False, out, teachers=ts) == "reteach"
    assert RUN.recto_stale(out, lo, ts)
    assert RUN.band_reuse(out, lo, ts) == stores.store_path(out, "recto", lo, 0)   # no m7 rerun
    assert RUN.band_reuse(out, lo, ["m7"]) is None
    g = stores.next_gen(out, lo, 0)
    attrs = {"teachers": ["recto", "m7"], "route": ts[2][len(RUN.ROUTE_TOKEN):]}
    _fake_store(stores.gen_path(stores.store_path(out, "recto", lo, 0), g), **attrs)
    assert RUN._next_job(cat, lo, 0, False, out, teachers=ts) == "reteach"   # the band is not there yet
    _fake_store(stores.gen_path(stores.store_path(out, "band", lo, 0), g), **attrs)
    assert RUN._next_job(cat, lo, 0, False, out, teachers=ts) == "reteach"   # ... nor its rw
    assert RUN.commit_sources(out, lo, 0) is None
    _fake_store(stores.gen_path(stores.store_path(out, "rw", lo, 0), g), **attrs)
    assert RUN._next_job(cat, lo, 0, False, out, teachers=ts) is None
    assert RUN.commit_sources(out, lo, 0) == {"gen": 0, "verso": 0, "recto": g}
    c2 = RG.Catalog(out, 0, ttl=0.0)
    assert c2.path("band", lo) == stores.gen_path(stores.store_path(out, "band", lo, 0), g)
    assert not RUN.recto_stale(out, lo, ts)
    assert RUN.band_reuse(out, lo, ts) == c2.path("band", lo)                 # a routed band is reused too
    # switching the route OFF again regenerates back to the plain set
    assert RUN._next_job(c2, lo, 0, False, out, teachers=["m7"]) == "reteach"
    assert RUN.recto_stale(out, lo, ["recto", "m7"])       # a routed store is not the old fusion either


def test_a_resume_that_turns_the_route_on_logs_a_teacher_switch(tmp_path, small_cfg):
    out = str(tmp_path / "sw")
    os.makedirs(out)
    RUN.write_state(out, step=60000)
    ck = {"recto": "/w/recto.pth", "m7": "/w/m7.pth"}
    old = replace(small_cfg, teacher_ckpts={"m7": "/w/m7.pth"}).to_json()
    old["config"].pop("teacher_route")                   # a config.json from before the field
    new = replace(small_cfg, teacher_ckpts=ck, teacher_route=ROUTE, loss_band=0.1)
    got = {r["kind"]: r for r in RUN.log_switches(out, old, new)}
    t = got["teacher_switch"]
    assert t["regenerate"] and t["old_route"] == {} and t["new_route"] == ROUTE
    assert t["old_set"] == ["m7"] and t["new_set"] == RUN.teacher_set(new)
    assert RUN.log_switches(out, new.to_json(), new) == []
    # the route alone moving (same checkpoints) is a switch too
    got = RUN.log_switches(out, replace(new, teacher_route={}).to_json(), new)
    assert got[0]["kind"] == "teacher_switch" and got[0]["regenerate"]


def test_the_bank_needs_both_routed_teachers(small_cfg):
    with pytest.raises(SystemExit):
        RUN.TeacherBank(replace(small_cfg, teacher_ckpts={"m7": "/w/m7.pth"}, teacher_route=ROUTE),
                        small_cfg.out)
    bank = RUN.TeacherBank(replace(small_cfg, teacher_ckpts={"recto": "a", "m7": "b"}, teacher_route=ROUTE),
                           small_cfg.out)
    assert bank.route.fine == "recto" and [r[0] for r in bank.items] == ["recto", "m7"]


# ------------------------------------------------------------------------------ the sampler

def _routed_gen(root, lo, fine, band, cov, sig):
    g = stores.next_gen(root, lo, 0)
    attrs = {"teachers": ["fine", "base"], "route": sig, "gen": g}
    for ch, blk in (("recto", fine), ("band", band), ("rw", cov)):
        stores.write(stores.gen_path(stores.store_path(root, ch, lo, 0), g), blk, lo, rung=2,
                     channels=(ch,), q=8, attrs=attrs)
    stores.commit_bundle(root, lo, 0, 0, verso=0, recto=g)
    RG.clear_pool()
    return g


def test_the_sampler_routes_rung_2_and_keeps_the_base_teacher_above(synth_run):
    root = synth_run.root
    cfg = replace(synth_run.cfg, teacher_route={"2": "fine", "3": "base"})
    lo = synth_run.lo[0]
    old = np.asarray(stores.open_store(stores.store_path(root, "recto", lo, 0))[:])
    fine = np.where(old > 0, np.uint8(200), np.uint8(0))
    fine[:, :, :64] = 0                                   # the fine teacher misses half the sheet
    cov = np.zeros_like(old)
    cov[:, :, 64:] = 255                                  # ... and is trusted only on the other half
    _routed_gen(root, lo, fine, old, cov, CFG.route_spec(cfg).sig)

    def patches(c):
        ds = sample.Patches(c, root=root, ct=c.ct, ax=synth_run.ax, region_records=synth_run.regions)
        ds._open()
        return ds
    from rvsm import ladder
    ds = patches(cfg)
    assert ds.channels[-1] == "band"
    p = ds.patch
    w0 = (0, 0, 32)
    wr = np.asarray(lo) + np.asarray(w0)
    ct = ladder.read_rung(ds.pyr, 2, wr, p, dtype=np.uint8)
    tg, w = ds._rung_target(2, wr, ct)
    air = ct > 0
    assert tg.shape[0] == 5
    sl = (slice(0, 32), slice(0, 32), slice(32, 64))
    # row 0: the FINE teacher at full weight (the rw is not a recto weight any more)
    assert np.array_equal(tg[0][air], fine[sl][air]) and (w[0][air] == 255).all()
    # the routing row: the base teacher, weight 255 where c_A, 128 where not
    assert np.array_equal(tg[4][air], old[sl][air])
    wx = np.asarray(w[4])
    assert (wx[air] == 128).all()                          # x 32..63: c_A = 0 here
    tg2, w2 = ds._rung_target(2, np.asarray(lo) + np.array([0, 0, 64]), ladder.read_rung(
        ds.pyr, 2, np.asarray(lo) + np.array([0, 0, 64]), p, dtype=np.uint8))
    a2 = w2[0] > 0
    assert (w2[4][a2] == 255).all()
    # rung 3: the BASE teacher (the band pooled), never the fine one
    lo3 = np.asarray(lo) // 2
    ct3 = ladder.read_rung(ds.pyr, 3, lo3, p, dtype=np.uint8)
    t3, w3 = ds._rung_target(3, lo3, ct3)
    ref = patches(synth_run.cfg)                           # unrouted: reads the committed recto (fine)
    v_band, _ = stores.read_store(stores.open_store(ds.cat.path("band", lo)), 3, lo3, p)
    assert np.array_equal(t3[0][ct3 > 0], v_band[ct3 > 0])
    assert (t3[4] == 0).all() and (w3[4] == 0).all()     # no routing row above rung 2
    # the unrouted view of the same stores: 4 rows, the committed recto and its rw as before
    tgu, wu = ref._rung_target(2, wr, ct)
    assert tgu.shape[0] == 4 and np.array_equal(tgu[0][air], fine[sl][air])
    assert (wu[0][air] == 0).all()                         # rw = c_A = 0 on this half
    # another region, not regenerated yet: the old meaning, and no routing row
    lo_b = synth_run.lo[1]
    ct_b = ladder.read_rung(ds.pyr, 2, np.asarray(lo_b), p, dtype=np.uint8)
    tb_, wb_ = ds._rung_target(2, np.asarray(lo_b), ct_b)
    tr_, wr_ = ref._rung_target(2, np.asarray(lo_b), ct_b)
    assert (wb_[4] == 0).all() and np.array_equal(tb_[:4], tr_) and np.array_equal(wb_[:4], wr_)
    # the grid of a routed config rebuilds (its sources and items carry the band)
    held = [r for r in synth_run.regions if r["k"] == 2 and tuple(r["lo"]) == tuple(lo)]
    src = sample.grid_sources(ds, held)
    assert len(src[0]) == len(sample.grid_sources(ref, held)[0]) + 1


# ------------------------------------------------------------------------------ the trainer

def _routed_item(cfg, seed=0):
    lay = cfg.layout()
    p = int(cfg.patch)
    rng = np.random.default_rng(seed)
    x = np.arange(p)
    band = np.zeros((p,) * 3, np.uint8)
    band[:, :, 10:13] = 230                                # the m7 sheet
    band[:, :, 22:25] = 230                                # ... and a second one the fine teacher misses
    fine = np.zeros_like(band)
    fine[:, :, 11:12] = 240                                # the fine teacher sees the first sheet only
    ct = np.stack([np.maximum(band, fine)] * (1 + lay.nctx)).astype(np.uint8)
    ct = np.clip(ct.astype(np.int16) + 30 + rng.integers(-20, 20, ct.shape), 1, 255).astype(np.uint8)
    wb = np.full_like(band, 128)
    wb[:, :, x < 18] = 255                                 # trusted around the first sheet only
    z = np.zeros_like(band)
    tg = np.stack([fine, z, z, z, band])
    w = np.stack([np.full_like(band, 255), z, z, z, wb])
    ax = np.array([[0.0, 4096.0], [p / 2, p / 2], [-1000.0, -1000.0]])
    return sample.rung_item(ct, tg, w, 2, (0, 0, 0), ax, sym=0, cm=(fine[::2, ::2, ::2]).copy(),
                            cx=rng.integers(0, 255, (1, p, p, p), dtype=np.uint8), lo1=(0, 0, 0),
                            rmax=float(4 * p), meta=np.linspace(0.2, 0.8, 5, dtype=np.float32))


def test_the_trainer_takes_the_gap_fill_path(small_cfg, monkeypatch):
    cfg = replace(small_cfg, rungs=(2,), teacher_route={"2": "recto"}, loss_band=1.0, steps=2,
                  eval_every=1000, aug="none")
    seen = {"band": []}
    real = L.route_apply
    real_pen = L.band_penalty

    def pen(p, outside, w=None, eps=0.05):
        v = real_pen(p, outside, w, eps)
        seen["band"].append((float(v.detach()), bool(v.requires_grad), eps))
        return v
    monkeypatch.setattr(L, "band_penalty", pen)

    def spy(tg, wv, tb, m):
        seen["gap"] = float(m["gap"].sum())
        seen["out"] = float(m["outside"].sum())
        seen["ca"] = float(m["ca"].sum())
        got = real(tg, wv, tb, m)
        seen["rows"] = int(got[0].shape[1])
        return got
    monkeypatch.setattr(L, "route_apply", spy)

    def gen():
        for i in range(10 ** 6):
            yield _routed_item(cfg, seed=i)
    TR.train(cfg, patches_factory=gen, device="cpu")
    assert seen["gap"] > 0 and seen["out"] > 0 and seen["ca"] > 0
    assert seen["rows"] == cfg.layout().cout_t               # the routing row left the batch
    # the band penalty ran on every microbatch, with a gradient, at the configured margin
    assert len(seen["band"]) >= 2 and all(g and e == cfg.band_eps and np.isfinite(v) and v > 0
                                         for v, g, e in seen["band"])


def test_an_unrouted_batch_never_touches_the_routing(small_cfg, monkeypatch):
    called = []
    monkeypatch.setattr(L, "route_masks", lambda *a, **k: called.append(1))
    from tests.test_train import _factory
    TR.train(replace(small_cfg, steps=2, eval_every=1000), patches_factory=_factory(small_cfg), device="cpu")
    assert called == []


# ------------------------------------------------------------------------------ the producer, end to end

def test_the_producer_routes_a_running_round_0(region_cfg, fake_teacher, has_volcomp):
    """A round 0 made by the base teacher alone (the m7-only mode), then a resume with the route on:
    every produced region is regenerated as a ROUTED generation (recto from the fine teacher, the band
    reused from the committed base store -- no base rerun --, rw = c_A), committed, generation 0 kept,
    the held-out reference first; the loader then reads the routing row and the trainer trains on it."""
    if not has_volcomp:
        pytest.skip("a region store is a volcomp array; no libvolcomp on this host")
    from rvsm import teachers
    from tests.test_reteach import _produce_until
    teachers.register("fake2", dataclasses.replace(fake_teacher.spec, name="fake2"))
    try:
        cfg1 = replace(region_cfg, teacher_ckpts={"fake2": fake_teacher.ckpt}, heldout=1,
                       lookahead_extra=1, reserve_gb=0.001, compile=False)
        ctx = RUN.setup(cfg1, cfg1.out)
        out = cfg1.out

        def done_recto():
            return RG.Catalog(out, 0, ttl=0.0).list_done("recto")
        assert _produce_until(cfg1, lambda: len(done_recto()) >= 2)
        first = done_recto()
        cfg2 = replace(cfg1, teacher_ckpts={"fake": fake_teacher.ckpt, "fake2": fake_teacher.ckpt},
                       teacher_route={"2": "fake", "3": "fake2"}, loss_band=0.5)
        assert cfg2.fingerprint() == cfg1.fingerprint()
        RUN.setup(cfg2, out)
        sw = [r for r in RUN.tail_jsonl(os.path.join(out, "logs", "sched.jsonl"), 50)
              if r.get("kind") == "teacher_switch"]
        assert sw and sw[-1]["regenerate"] and sw[-1]["new_route"] == {"2": "fake", "3": "fake2"}
        ts = RUN.teacher_set(cfg2)
        assert _produce_until(cfg2, lambda: not any(RUN.recto_stale(out, lo, ts) for lo in first))
    finally:
        teachers.TEACHERS.pop("fake2", None)
    cat = RG.Catalog(out, 0, ttl=0.0)
    for lo in first:
        g0 = stores.store_path(out, "recto", lo, 0)
        assert stores.is_done(g0) and stores.read_attrs(g0)["teachers"] == ["fake2"]    # kept
        p, b, w = cat.path("recto", lo), cat.path("band", lo), cat.path("rw", lo)
        assert p != g0 and stores.is_done(b) and stores.is_done(w)
        a = stores.read_attrs(p)
        assert a["route"] == CFG.route_spec(cfg2).sig and a["teachers"] == ["fake", "fake2"]
        assert a["band_source"] == "store:" + g0                                # m7 not rerun
        assert stores.read_attrs(w)["encoding"] == "coverage_u8"
        old = np.asarray(stores.open_store(g0)[:])
        band = np.asarray(stores.open_store(b)[:])
        d = np.abs(band.astype(int) - old.astype(int))                        # the base store, re-coded q8
        assert d.max() <= 16 and d.mean() < 1.0
        cov = np.asarray(stores.open_store(w)[:])
        assert set(np.unique(cov)) <= {0, 255} or np.abs(cov.astype(int) - 255).min() <= 8
    log = RUN.tail_jsonl(os.path.join(out, "logs", "produce.jsonl"), 5000)
    re = [tuple(r["region"]) for r in log if r.get("kind") == "reteach"]
    assert set(first) <= set(re)
    held = tuple(int(v) for v in ctx["heldout"][0]["lo"])
    if held in first:
        assert re[0] == held, "the held-out reference is regenerated first"
    # the loader of the routed config reads the routing row, and the trainer runs the gap-fill on it
    recs = [r for r in ctx["records"] if r["k"] == 2 and tuple(r["lo"]) in set(first)]
    items = sample.val_grid(cfg2, recs, root=out, ct=cfg2.ct, ax=ctx["ax"], rungs=(2,), limit=2)
    assert items and all(int(it["tgt"].shape[0]) == 5 for it in items)
    assert any(float(it["w"][4].float().max()) > 0 for it in items)
    cfg3 = replace(cfg2, rungs=(2,), steps=4, eval_every=1000, out=os.path.join(out, "tr"))
    TR.train(cfg3, patches_factory=lambda: iter(items * 4), device="cpu")


# ------------------------------------------------------------------------------ the reteach GPU share

def _stale_backlog(out, n):
    los = [(0, 0, 128 * i) for i in range(n)]
    for lo in los:
        _fake_store(stores.store_path(out, "recto", lo, 0), teachers=["m7"])
        _fake_store(stores.store_path(out, "rw", lo, 0), teachers=["m7"])
    return los


def _run_reteach(out, lo, ts):
    """What a routed reteach unit leaves on disk: the next generation's recto, band, rw, committed."""
    g = stores.next_gen(out, lo, 0)
    attrs = {"teachers": ["recto", "m7"], "route": ts[2][len(RUN.ROUTE_TOKEN):]}
    for ch in ("recto", "band", "rw"):
        _fake_store(stores.gen_path(stores.store_path(out, ch, lo, 0), g), **attrs)
    RUN.commit_sources(out, lo, 0)


def _simulate(out, los, share, hours=3.0, verso_s=65.0, reteach_s=15.0, window=None):
    """The producer's scheduling of the recto backlog, pass by pass, with the REAL functions
    (`_recto_rescan`, `_recto_backlog`, `_share_backlog`, `ReteachMeter`) and a window that always
    holds one verso pass (never idle): each pass runs its units in order and advances the clock by
    their GPU seconds. Returns (regions regenerated, reteach seconds, all seconds)."""
    import threading
    ts = RUN.teacher_set(CFG.Config(teacher_route=ROUTE))
    rr, meter, busy, keys, lock = {}, RUN.ReteachMeter(), set(), set(), threading.Lock()
    now, t_end, done, rs, tot, k = 1000.0, 1000.0 + hours * 3600, 0, 0.0, 0.0, 0
    route = list(los)
    while now < t_end:
        cat = RG.Catalog(out, 0, ttl=0.0)
        RUN._recto_rescan(out, rr, route, [], ts, now=now, meter=meter, share=share)
        units = list(window(k)) if window else [((9, 9, 9 + k), "verso")]
        k += 1

        def step(u, now=now, cat=cat):
            return RUN._recto_backlog(out, rr, route, [], ts, cat, False, (2, 3, 4), busy, lock, {},
                                      lambda lo: True, u, keys, lambda lo: None, now=now)
        if units:
            RUN._share_backlog(out, rr, meter, share, units, [], step, now=now)
        else:
            step(units)
            if not units:
                now += 5.0                               # an idle pass with nothing left: time moves on
        for lo, job in units:
            s = reteach_s if job == "reteach" else verso_s
            if job == "reteach":
                _run_reteach(out, lo, ts)
                done += 1
                rs += s
            tot += s
            meter.add(job, s, now=now)
            now += s
    return done, rs, tot


def test_the_backlog_drains_at_its_share_while_verso_keeps_the_window_busy(tmp_path):
    out = str(tmp_path / "sh")
    los = _stale_backlog(out, 400)
    done, rs, tot = _simulate(out, los, 0.25)
    assert 0.20 <= rs / tot <= 0.27, rs / tot
    assert done >= 150                                   # ~0.25 of 3 h at 15 s a reteach: ~180
    lines = RUN.tail_jsonl(os.path.join(out, "logs", "produce.jsonl"), 100000)
    adm = [r for r in lines if r.get("kind") == "recto_backlog_admit"]
    assert len(adm) == done and all(r["share"] < 0.25 and r["share_target"] == 0.25 for r in adm)
    # the periodic line: every RECTO_TODO_S of the (busy) run, with the route token and the share
    reg = [r for r in lines if r.get("kind") == "recto_regen"]
    assert len(reg) >= int(3 * 3600 / (RUN.RECTO_TODO_S + 100))   # rescans ride the passes (<= 100 s each)
    assert any(t.startswith(RUN.ROUTE_TOKEN) for t in reg[0]["teachers"])
    assert reg[-1]["remaining"] < reg[0]["remaining"] and 0.15 < reg[-1]["share"] <= 0.3


def test_share_0_is_the_old_idle_only_rule(tmp_path):
    out = str(tmp_path / "s0")
    los = _stale_backlog(out, 20)
    done, rs, _ = _simulate(out, los, 0.0, hours=1.0)
    assert done == 0 and rs == 0.0                       # a busy window: nothing, as before
    reg = [r for r in RUN.tail_jsonl(os.path.join(out, "logs", "produce.jsonl"), 1000)
           if r.get("kind") == "recto_regen"]
    assert reg and reg[-1]["remaining"] == 20           # ... but the backlog is visible
    done, _, _ = _simulate(out, los, 0.0, hours=0.2, window=lambda k: [])
    assert done > 0                                      # an idle window still works through it
    assert CFG.Config().reteach_share == 0.25
    assert replace(CFG.Config(), reteach_share=0.0).fingerprint() == CFG.Config().fingerprint()


def test_blocking_and_leased_units_keep_their_priority(tmp_path):
    import threading
    out = str(tmp_path / "pr")
    los = _stale_backlog(out, 5)
    ts = RUN.teacher_set(CFG.Config(teacher_route=ROUTE))
    rr, meter = {}, RUN.ReteachMeter()
    cat = RG.Catalog(out, 0, ttl=0.0)

    def step(u):
        return RUN._recto_backlog(out, rr, los, [], ts, cat, False, (2, 3, 4), set(), threading.Lock(),
                                  {}, lambda lo: True, u, set(), lambda lo: None)
    for units, leased in (([((9, 9, 9), "teacher")], []), ([((9, 9, 9), "verso")], [(9, 9, 9)]),
                          ([((9, 9, 9), "self")], [])):
        u = list(units)
        assert RUN._share_backlog(out, rr, meter, 0.25, u, leased, step) == [] and u == units
    u = [((9, 9, 9), "verso")]
    got = RUN._share_backlog(out, rr, meter, 0.25, u, [], step)
    assert got and u[0] == got[0] and u[-1] == ((9, 9, 9), "verso") and got[0][1] == "reteach"
    # over the share: nothing more
    meter.add("reteach", 100.0)
    meter.add("verso", 100.0)
    u = [((9, 9, 9), "verso")]
    assert RUN._share_backlog(out, rr, meter, 0.25, u, [], step) == []
