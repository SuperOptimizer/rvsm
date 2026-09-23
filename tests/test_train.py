"""The trainer: the schedule, the warm start, the deep-supervision pooling and the whole step loop.

Everything here runs on the CPU at size `1m`, on SYNTHETIC samples built by hand -- a bright slab with
targets that agree with it -- so the loop is exercised with every channel present and every loss term
on, without a store, a teacher or a GPU.
"""
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from rvsm import losses as L, model as M, prep, sample, train as TR
from rvsm.config import Config



@pytest.fixture(autouse=True)
def cpu_threads():
    """Four threads, not the machine's default.

    The `1m` net at these patch sizes is far too small to fill a many-core host: measured here, one
    forward+backward at 32^3 costs 0.77 s on four threads and 7.6 s on twenty-four, because the
    per-operator thread synchronisation dwarfs the arithmetic. Every real run is on a GPU, so this is
    purely about not spending ten minutes on a twenty-step CPU test.
    """
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 4))
    yield
    torch.set_num_threads(n)


# --------------------------------------------------------------------- the schedule

def test_lr_lambda_wsd_is_flat_until_stable_until_and_zero_at_the_end():
    f = TR.lr_lambda(1000, warmup=10, sched="wsd", stable_until=900, cooldown=100)
    assert f(0) == pytest.approx(1 / 10)          # linear warmup
    assert f(9) == pytest.approx(1.0)
    for s in (10, 100, 500, 899):                 # the plateau is FLAT, to the last float op
        assert f(s) == pytest.approx(1.0)
    assert f(950) == pytest.approx(0.5)           # half way through the cosine cooldown
    assert f(1000) == pytest.approx(0.0, abs=1e-12)
    # the defaults put the cooldown at the last 10 % of the budget
    g = TR.lr_lambda(1000, warmup=10, sched="wsd")
    assert TR.wsd_stable_until(1000) == (900, 100)
    assert g(899) == pytest.approx(1.0) and g(1000) == pytest.approx(0.0, abs=1e-12)
    # cosine is kept as an option and is NOT flat
    c = TR.lr_lambda(1000, warmup=10, sched="cosine")
    assert c(500) == pytest.approx(0.5) and c(1000) == pytest.approx(0.0, abs=1e-12)


def test_ema_auto_is_one_minus_k_over_steps_clamped():
    assert TR.ema_auto(1000, 50) == pytest.approx(0.95)
    assert TR.ema_auto(20000, 50) == pytest.approx(0.9975)
    assert TR.ema_auto(10, 50) == 0.9            # clamped low
    assert TR.ema_auto(10 ** 8, 50) == 0.9999    # clamped high


# --------------------------------------------------------------------- the warm start

def _net(cfg):
    L_ = cfg.layout()
    return M.build(cfg.size, cin=L_.cin, cout=L_.cout, verbose=False)


def _x(cfg, p=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, cfg.layout().cin, p, p, p)


def test_warm_start_reproduces_the_source_and_reports_the_new_tensors(tmp_path):
    cfg = Config(size="1m")
    src = _net(cfg).eval()
    ck = tmp_path / "src.pt"
    torch.save({"ema": src.state_dict(), "layout": cfg.layout().to_json()}, ck)
    st = torch.load(ck, map_location="cpu", weights_only=False)

    # same layout: every tensor is copied by name, nothing is new, the outputs are identical
    dst = _net(cfg).eval()
    sd, newp = TR.warm_start(st["ema"], dst, cfg.layout(), src_layout=st["layout"])
    dst.load_state_dict(sd, strict=False)
    assert newp == set()
    x = _x(cfg)
    with torch.no_grad():
        assert torch.allclose(src(x), dst(x), atol=1e-5)

    # a GROWN head (one more affinity offset): the shared rows still reproduce, the head is `newp`
    big = replace(cfg, aff_offsets=(8, 16, 32, 64))
    lb = big.layout()
    assert lb.cout == cfg.layout().cout + 3
    dst2 = _net(big).eval()
    sd2, newp2 = TR.warm_start(st["ema"], dst2, lb, src_layout=st["layout"])
    dst2.load_state_dict(sd2, strict=False)
    assert "head.weight" in newp2 and "head.bias" in newp2
    assert "enc.0.0.weight" not in newp2          # the stem did not change
    assert TR.new_param_names(newp2, dst2) == newp2
    with torch.no_grad():
        a, b = src(x), dst2(x)
    assert torch.allclose(a, b[:, :a.shape[1]], atol=1e-5)
    assert float(b[:, a.shape[1]:].abs().max()) == 0.0     # the new rows are zero-initialised


def test_warm_start_from_a_usrm2_checkpoint_maps_by_name():
    """usrm2's stem is [CT, ctx_1..9, scale, rz, ry, rx] (14) and its head [recto, verso] (cout 2)."""
    cfg = Config(size="1m")
    lay = cfg.layout()
    assert (lay.cin, lay.cout) == (21, 14)
    src = M.build("1m", cin=14, cout=2, verbose=False)
    sd = src.state_dict()
    dst = _net(cfg)
    got, newp = TR.warm_start(sd, dst, lay)          # no `layout` recorded: inferred from the shapes
    assert TR.usrm2_stem_names(14) == ["CT"] + [f"ctx_{i}" for i in range(1, 10)] + \
        ["scale", "rz", "ry", "rx"]
    assert TR.usrm2_stem_names(15)[10] == "cascade"

    w, s = got["enc.0.0.weight"], sd["enc.0.0.weight"]
    assert w.shape[1] == 21
    assert torch.equal(w[:, :10], s[:, :10])                      # CT + the nine context cubes
    assert torch.equal(w[:, lay.i_scale:], s[:, 10:])             # scale + the radial unit vector
    assert float(w[:, lay.i_cas:lay.i_scale].abs().max()) == 0.0  # cascade, radius and the 5 scan planes

    h, hs = got["head.weight"], sd["head.weight"]
    assert torch.equal(h[:2], hs)                                 # recto, verso -> rows 0 and 1
    assert float(h[2:].abs().max()) == 0.0
    assert torch.equal(got["head.bias"][:2], sd["head.bias"])
    assert {"enc.0.0.weight", "head.weight", "head.bias"} <= newp
    # a tensor that did not change shape was copied verbatim and is NOT new
    assert "enc.1.0.weight" not in newp and torch.equal(got["enc.1.0.weight"], sd["enc.1.0.weight"])


# --------------------------------------------------------------------- deep supervision

def test_deep_losses_scores_every_level_against_the_pooled_target():
    torch.manual_seed(0)
    lay = Config(channels=("recto", "verso"), aff_offsets=()).layout()
    t = (torch.rand(1, lay.cout_t, 8, 8, 8) > 0.5).float()
    w = torch.ones_like(t)
    lg = [torch.randn(1, lay.cout, 8, 8, 8), torch.randn(1, lay.cout, 4, 4, 4)]
    bce, dice = L.deep_losses(lg, t, w, layout=lay)
    n = lay.nprob
    b0, d0 = L.losses_tw(lg[0][:, :n], t[:, :n], w[:, :n])
    b1, d1 = L.losses_tw(lg[1][:, :n], torch.nn.functional.avg_pool3d(t[:, :n], 2),
                         torch.nn.functional.avg_pool3d(w[:, :n], 2))
    assert float(bce) == pytest.approx(float(b0 + 0.5 * b1), rel=1e-5)
    assert float(dice) == pytest.approx(float(d0 + 0.5 * d1), rel=1e-5)


# --------------------------------------------------------------------- synthetic samples

def _slab(p, shift=0, seed=0):
    """A bright tilted slab, the same construction the CT fixture uses, at an arbitrary patch size."""
    z, y, x = np.indices((p, p, p))
    d = np.abs(y - (p / 2 + 0.25 * (x - p / 2)) - shift)
    return (d < max(p / 8, 2)).astype(np.uint8) * np.uint8(255)


def _item(cfg, k=2, seed=0, verso_w=255, dist_w=255):
    """One compact sample with EVERY channel present: CT + context cubes, recto/verso bands, a midline
    code and a thickness code, and a per-channel weight the test can zero."""
    lay = cfg.layout()
    p = int(cfg.patch)
    rng = np.random.default_rng(seed)
    recto = _slab(p, 0)
    verso = _slab(p, 3)
    ct = np.stack([np.maximum(recto, verso)] * (1 + lay.nctx)).astype(np.uint8)
    ct = np.clip(ct.astype(np.int16) + rng.integers(-20, 20, ct.shape), 0, 255).astype(np.uint8)
    code = np.where(np.maximum(recto, verso) > 0, np.uint8(140), np.uint8(0))
    tg = np.stack([recto, verso, code, code])
    w = np.stack([np.full((p,) * 3, 255, np.uint8),
                  np.full((p,) * 3, verso_w, np.uint8),
                  np.full((p,) * 3, dist_w, np.uint8),
                  np.full((p,) * 3, dist_w, np.uint8)])
    ax = np.array([[0.0, 4096.0], [p / 2, p / 2], [-1000.0, -1000.0]])
    return sample.rung_item(ct, tg, w, k, (0, 0, 0), ax, sym=0,
                            cm=(recto[::2, ::2, ::2]).copy(),
                            cx=rng.integers(0, 255, (1, p, p, p), dtype=np.uint8),
                            lo1=(0, 0, 0), rmax=float(4 * p),
                            meta=np.linspace(0.2, 0.8, 5, dtype=np.float32))


def _factory(cfg, n=10 ** 6, **kw):
    def gen():
        for i in range(n):
            yield _item(cfg, k=cfg.rungs[(i + 1) % len(cfg.rungs)], seed=i, **kw)
    return gen


@pytest.fixture
def full_cfg(tmp_path, ct_origin, umbilicus):
    """The FULL 21-channel stem and 14-row head -- the real recipe -- at size 1m and patch 64."""
    return Config(ct=ct_origin.path, umbilicus=umbilicus[0], out=str(tmp_path / "run"),
                  size="1m", patch=40, batch=1, rungs=(2, 3), aff_offsets=(8, 16, 32),
                  ect_block=16, steps=20, eval_every=10, workers=0, compile=False, heldout=1)


@pytest.fixture
def tiny_cfg(tmp_path, ct_origin, umbilicus):
    """A cheaper variant for the tests that only need the loop to turn over."""
    return Config(ct=ct_origin.path, umbilicus=umbilicus[0], out=str(tmp_path / "tiny"),
                  size="1m", patch=32, batch=1, ctx=(1, 2, 3), rungs=(2, 3), aff_offsets=(4, 8),
                  ect_block=8, steps=4, eval_every=4, workers=0, compile=False, heldout=1)


# --------------------------------------------------------------------- end to end

def test_twenty_steps_of_the_full_recipe(full_cfg):
    cfg = full_cfg
    lay = cfg.layout()
    assert (lay.cin, lay.cout, lay.cout_t) == (21, 14, 4)
    val = [_item(cfg, k=2, seed=900), _item(cfg, k=3, seed=901)]
    ck = TR.train(cfg, patches_factory=_factory(cfg), val_items=val, device="cpu")

    out = __import__("pathlib").Path(cfg.out)
    assert (out / "ckpt.pt").exists()
    st = torch.load(ck, map_location="cpu", weights_only=False)
    assert st["step"] == 20
    assert st["cfg"]["fingerprint"] == cfg.fingerprint()
    assert st["layout"]["heads"][:2] == ["recto", "verso"] and len(st["layout"]["heads"]) == 14
    assert st["layout"]["stem"][0] == "CT" and len(st["layout"]["stem"]) == 21
    assert st["temps"], "the calibration ran after the evaluation and left temperatures"

    rows = [json.loads(q) for q in (out / "logs" / "train.jsonl").read_text().splitlines()]
    steps = [r for r in rows if "loss" in r]
    assert steps, "no step was logged"
    for r in steps:                       # every term of the full recipe is there and finite
        for k in ("loss", "bce", "dice", "sdist", "eikonal", "thick", "pair_bce", "pair_dice",
                  "ect", "excl", "selfcons", "skel", "affinity"):
            assert k in r, f"{k} missing from {sorted(r)}"
            assert math.isfinite(r[k]), f"{k} is {r[k]}"
        assert r["rung"] and r["train_wait_s"] >= 0 and r["vox_s"] > 0

    ev = [json.loads(q) for q in (out / "logs" / "eval.jsonl").read_text().splitlines()]
    assert len(ev) >= 2
    last = ev[-1]
    for k in ("bce", "dice", "mae", "overlap", "dice_recto", "dice_verso", "dice_r2", "dice_r3",
              "mae_midline", "mae_thickness", "dice_raw", "dice_best", "thr_best", "dice_soft"):
        assert k in last and math.isfinite(last[k]), f"{k}: {last.get(k)}"
    if cfg.cascade in ("self", "mix"):       # the mask-cascade bracket beside the self-cascade numbers
        for k in ("dice_mask", "bce_mask", "dice_mask_r2"):
            assert k in last and math.isfinite(last[k]), f"{k}: {last.get(k)}"
    if cfg.calibrate:
        assert "2" in last["temps"], last.get("temps")         # rung 2 is always calibrated now
    assert sorted(p.name for p in (out / "eval").glob("val_*.png"))
    from PIL import Image
    im = Image.open(sorted((out / "eval").glob("val_*.png"))[-1])
    assert im.size == (3 * cfg.patch, 2 * cfg.patch)      # three tiles wide, one row per val patch


def test_resume_continues_and_refuses_a_different_config(tiny_cfg):
    cfg = tiny_cfg
    val = [_item(cfg, k=2, seed=7)]
    TR.train(cfg, patches_factory=_factory(cfg), val_items=val, device="cpu")
    st = torch.load(__import__("pathlib").Path(cfg.out) / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    assert st["step"] == 4

    more = replace(cfg, steps=8)                 # `steps` is excluded from the fingerprint
    assert more.fingerprint() == cfg.fingerprint()
    ck = TR.train(more, resume=True, patches_factory=_factory(more), val_items=val, device="cpu")
    assert torch.load(ck, map_location="cpu", weights_only=False)["step"] == 8

    other = replace(cfg, lr=1e-3)                # a recipe field: the run is not the same run
    assert other.fingerprint() != cfg.fingerprint()
    with pytest.raises(AssertionError, match="fingerprint"):
        TR.train(other, resume=True, patches_factory=_factory(other), val_items=val, device="cpu")


class _Oracle(torch.nn.Module):
    """A 'net' whose probability rows are the target itself (+-12 logits) and whose distance rows are
    an arbitrary regression value -- what a real distance head emits: voxels, not a logit."""

    def __init__(self, tgt, cout, nprob):
        super().__init__()
        self.tgt, self.cout, self.nprob = tgt, cout, nprob

    def forward(self, x):
        y = torch.full((x.shape[0], self.cout) + tuple(x.shape[2:]), -3.0)
        y[:, :self.nprob] = (self.tgt[:, :self.nprob] >= 0.5).float() * 24 - 12
        return y


def test_evaluate_scores_only_the_probability_heads(full_cfg):
    """bce / dice / mae are PROBABILITY metrics: a distance channel's weight must not pull a distance
    regression through a sigmoid into them. A perfect probability prediction scores dice 1 and bce ~0
    whatever the distance heads say (the distances have their own mae_midline / mae_thickness)."""
    lay = full_cfg.layout()
    it = _item(full_cfg, dist_w=255)
    _, tg, _ = prep.prepare(prep.batch1(it), torch.device("cpu"))
    out = TR.evaluate(_Oracle(tg, lay.cout, lay.nprob), [it], torch.device("cpu"), lay)
    assert out["dice_recto"] > 0.99 and out["dice_verso"] > 0.99
    assert out["dice"] > 0.99, out
    assert out["bce"] < 1e-3 and out["mae"] < 1e-3, out
    assert "mae_midline" in out and "mae_thickness" in out


def test_evaluate_skips_windows_with_no_weight(full_cfg):
    """A held-out window with no store behind it (weight 0 everywhere) has nothing to score: it must
    not enter bce / dice as a 0, and a rung made only of such windows has no dice_r{k} to drag the
    headline mean down."""
    lay = full_cfg.layout()
    good = _item(full_cfg, k=2, dist_w=0)
    empty = _item(full_cfg, k=3, dist_w=0)
    empty["w"] = torch.zeros_like(empty["w"])
    _, tg, _ = prep.prepare(prep.batch1(good), torch.device("cpu"))
    out = TR.evaluate(_Oracle(tg, lay.cout, lay.nprob), [good, empty], torch.device("cpu"), lay)
    assert out["n_scored"] == 1
    assert "dice_r3" not in out and out["dice_r2"] > 0.99
    assert out["dice"] > 0.99 and out["bce"] < 1e-3, out


class _Fixed(torch.nn.Module):
    """A 'net' that says +12 (sheet) everywhere: right on the band, confidently wrong off it."""

    def __init__(self, cout):
        super().__init__()
        self.cout = cout

    def forward(self, x):
        return torch.full((x.shape[0], self.cout) + tuple(x.shape[2:]), 12.0)


def test_evaluate_is_voxel_weighted_and_the_headline_is_the_fine_rungs(full_cfg):
    """A coarse window with a sliver of weight counts that sliver, not a whole window: pooled over
    voxels, and the headline dice / bce pool rungs 2-4 only while dice_coarse / bce_coarse carry 5-11.
    Per-window averaging let a handful of rung 7-11 corner voxels drive the paris4 headline bce to ~1."""
    lay = full_cfg.layout()
    fine = _item(full_cfg, k=2, dist_w=0, verso_w=0)
    fine2 = _item(full_cfg, k=3, dist_w=0, verso_w=0, seed=1)
    coarse = _item(full_cfg, k=9, dist_w=0, verso_w=0, seed=2)
    wc = torch.zeros_like(coarse["w"])
    wc[0, :2, :2, :2] = 255                               # eight weighted voxels in a corner ...
    coarse["w"] = wc
    coarse["tgt"] = torch.zeros_like(coarse["tgt"])       # ... all background, all called sheet
    net = _Fixed(lay.cout)
    alone = TR.evaluate(net, [fine, fine2], torch.device("cpu"), lay)
    both = TR.evaluate(net, [fine, coarse, fine2], torch.device("cpu"), lay)
    assert both["n_scored"] == 3 and both["fine_rungs"] == [2, 3]
    for k in ("dice", "bce", "mae", "dice_r2", "dice_r3", "dice_recto"):
        assert both[k] == pytest.approx(alone[k]), k       # the coarse sliver cannot move the headline
    assert both["bce_r9"] == pytest.approx(12.0, abs=1e-3) and both["dice_r9"] == 0.0
    assert both["bce_coarse"] == pytest.approx(both["bce_r9"]) and both["dice_coarse"] == 0.0
    assert "bce_coarse" not in alone

    # and inside the fine rungs the pooling is by voxel: the headline is the two rungs' pooled sums,
    # which is the weight-weighted mean of their bce (not the mean of the two per-rung numbers)
    w2, w3 = float(fine["w"][0].sum()), float(fine2["w"][0].sum())
    want = (alone["bce_r2"] * w2 + alone["bce_r3"] * w3) / (w2 + w3)
    assert alone["bce"] == pytest.approx(want, rel=1e-5)

    only_coarse = TR.evaluate(net, [coarse], torch.device("cpu"), lay)   # no fine rung: fall back
    assert only_coarse["fine_rungs"] == [9] and only_coarse["bce"] == pytest.approx(12.0, abs=1e-3)


class _Shrunk(torch.nn.Module):
    """A net that RANKS the band perfectly but never crosses 0.5: logit -1 on the band, -4 off it."""

    def __init__(self, tgt, cout, nprob):
        super().__init__()
        self.tgt, self.cout, self.nprob = tgt, cout, nprob

    def forward(self, x):
        y = torch.full((x.shape[0], self.cout) + tuple(x.shape[2:]), -4.0)
        y[:, :self.nprob] = (self.tgt[:, :self.nprob] >= 0.5).float() * 3 - 4
        return y


def test_evaluate_reports_a_threshold_free_dice(full_cfg):
    """An under-confident net that ranks the band perfectly scores dice 0 at the 0.5 threshold (and
    no temperature changes that: sigmoid(l/T) >= 0.5 iff l >= 0), but dice_best finds the threshold
    that separates it and dice_soft sees the ranking."""
    lay = full_cfg.layout()
    it = _item(full_cfg, dist_w=0, verso_w=0)
    _, tg, _ = prep.prepare(prep.batch1(it), torch.device("cpu"))
    out = TR.evaluate(_Shrunk(tg, lay.cout, lay.nprob), [it], torch.device("cpu"), lay)
    assert out["dice"] < 0.01 and out["dice_raw"] == out["dice"]
    assert out["dice_best"] > 0.99 and 0.018 < out["thr_best"] <= 0.27     # between off and on
    assert out["dice_best_r2"] == pytest.approx(out["dice_best"])
    assert 0.0 < out["dice_soft"] < out["dice_best"]
    only3 = TR.evaluate(_Shrunk(tg, lay.cout, lay.nprob), [it], torch.device("cpu"), lay, rungs=(3,))
    assert only3["n_scored"] == 0                                    # the rung filter


def test_a_run_with_no_verso_and_no_distance_stores_still_trains(tiny_cfg):
    """Round 0 before the verso gate: those channels arrive with weight 0, and nothing special-cases
    them -- the losses stay finite and the head rows they own get no gradient."""
    cfg = replace(tiny_cfg, out=str(__import__("pathlib").Path(tiny_cfg.out).parent / "noverso"))
    val = [_item(cfg, k=2, seed=7, verso_w=0, dist_w=0)]
    ck = TR.train(cfg, patches_factory=_factory(cfg, verso_w=0, dist_w=0), val_items=val,
                  device="cpu")
    rows = [json.loads(q) for q in
            (__import__("pathlib").Path(cfg.out) / "logs" / "train.jsonl").read_text().splitlines()]
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"{k} is {v}"
    ev = [json.loads(q) for q in
          (__import__("pathlib").Path(cfg.out) / "logs" / "eval.jsonl").read_text().splitlines()]
    assert ev and math.isfinite(ev[-1]["dice"])
    assert "dice_verso" not in ev[-1] and "mae_midline" not in ev[-1]   # nothing weighs them

    # ... and the verso head row carries no gradient at all
    lay = cfg.layout()
    net = M.build(cfg.size, cin=lay.cin, cout=lay.cout, verbose=False)
    net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
    dev = torch.device("cpu")
    x, tg, wt = prep.prepare(prep.batch1(_item(cfg, k=2, seed=3, verso_w=0, dist_w=0)), dev)
    y = net(x).float()
    bce, dice = L.deep_losses(y, tg, wt, layout=lay)
    d = y[:, lay.i_mid:lay.i_mid + 1]
    th = L.soft_thickness(y[:, lay.i_thick:lay.i_thick + 1])
    a, b = L.pair_logits(d, th, half=cfg.pair_band, tau=cfg.pair_tau)
    pb, pd = L.losses_tw(torch.cat([a, b], 1), tg[:, :2], wt[:, :2])
    ax = L.aux_losses(y, tg, wt, lay, w_excl=cfg.loss_excl, w_skel=cfg.loss_skel,
                      w_affinity=cfg.loss_affinity, skel_iters=cfg.skel_iters)
    loss = bce + dice + pb + pd + ax.get("aux", torch.zeros(()))
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    g = net.head.weight.grad
    assert float(g[1].abs().max()) == 0.0, "the verso row must get no gradient without a verso store"
    assert float(g[0].abs().max()) > 0.0   # while the recto row does


def test_the_distance_heads_drop_the_non_isometric_spatial_augs():
    """usrm2 section 29.2: a rotation / scaling / shear / elastic field / sheet compression resamples
    the grid the distance stores were measured on, so the resampled target is a wrong NUMBER. With
    `loss_sdist` on -- the fixed default -- they come out of the recipe; the cube symmetries and every
    intensity and physics op stay."""
    cfg = Config()
    assert cfg.loss_sdist > 0 and cfg.layout().cout_t > cfg.layout().nprob
    full = TR.aug_for(replace(cfg, loss_sdist=0.0))
    got = TR.aug_for(cfg)
    for k in TR.NON_ISOMETRIC:
        assert k in full, f"{k} is not in the {cfg.aug} preset to begin with"
        assert k not in got, f"{k} survived a run with a distance head"
    # everything else is untouched: the intensity chain, the physics ops and the composition shuffle
    assert set(full) - set(got) == set(TR.NON_ISOMETRIC)
    for k in ("gamma", "noise", "paganin", "pool", "tone", "thick", "volcomp", "shuffle", "sym"):
        assert k in got


def test_a_run_without_distance_heads_keeps_every_spatial_aug():
    cfg = Config(channels=("recto", "verso"))
    got = TR.aug_for(replace(cfg, loss_sdist=0.0))
    assert all(k in got for k in TR.NON_ISOMETRIC)


# --------------------------------------------------------------------- the subcommand

def test_rvsm_train_runs_on_real_region_stores(synth_run, tmp_path):
    """`rvsm train cfg.toml` end to end: the config file, the region walk, the held-out grid, the
    checkpoint, the evaluation log and the PNG -- on the stores `rvsm produce` writes."""
    from rvsm import cli

    cfg = replace(synth_run.cfg, steps=20, eval_every=10, patch=32, batch=1, heldout=1,
                  windows_per_region=4, compile=False, workers=0)
    toml = tmp_path / "cfg.toml"
    toml.write_text(
        f'ct = "{cfg.ct}"\numbilicus = "{cfg.umbilicus}"\nout = "{cfg.out}"\nsize = "1m"\n'
        f'patch = 32\nbatch = 1\nregion = {cfg.region}\nctx = [1, 2, 3]\nrungs = [2, 3]\n'
        f'aff_offsets = [4, 8]\nect_block = 8\nsteps = 20\neval_every = 10\nworkers = 0\n'
        f'heldout = 1\nwindows_per_region = 4\ncompile = false\n')
    assert cli.main(["train", str(toml), "--device", "cpu"]) == 0

    out = __import__("pathlib").Path(cfg.out)
    st = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    assert st["step"] == 20 and st["layout"]["cout"] == 11        # 2 + 2 + 1 + 2 offsets x 3
    ev = [json.loads(q) for q in (out / "logs" / "eval.jsonl").read_text().splitlines()]
    assert ev and math.isfinite(ev[-1]["dice"])
    assert list((out / "eval").glob("val_*.png"))


def test_the_self_p_schedule_has_two_breakpoints():
    """0.1 at 0 -> 0.7 at 20k -> 0.9 at 30k, held; the mask source keeps the rest."""
    from rvsm.config import Config
    c = Config()
    want = {0: 0.1, 10000: 0.4, 20000: 0.7, 25000: 0.8, 30000: 0.9, 40000: 0.9}
    for st, v in want.items():
        assert TR.self_p_at(c, st, 60000) == pytest.approx(v), st
    old = replace(c, self_p_mid_step=0)                  # the old whole-run schedule
    assert old.fingerprint() == c.fingerprint() == replace(c, self_p_end=0.5).fingerprint()
    assert TR.self_p_at(old, 30000, 60000) == pytest.approx(0.4)


def test_val_png_draws_the_verso_only_once_verso_is_on(full_cfg, tmp_path):
    """Round 0 before the gate: the untrained verso head (saturated near 1) must not paint the
    prediction tile blue over the recto. With verso off the tiles carry no blue; on, they do."""
    from PIL import Image
    lay = full_cfg.layout()
    it = _item(full_cfg, dist_w=0)
    _, tg, _ = prep.prepare(prep.batch1(it), torch.device("cpu"))

    class Verso1(torch.nn.Module):          # recto = the target, verso saturated everywhere
        def forward(self, x):
            y = torch.full((x.shape[0], lay.cout) + tuple(x.shape[2:]), -12.0)
            y[:, 0] = (tg[:, 0] >= 0.5).float() * 24 - 12
            y[:, 1] = 12.0
            return y

    ev = tmp_path / "run" / "eval"
    ev.mkdir(parents=True)
    (tmp_path / "run" / "state.json").write_text('{"verso_on": false}')
    TR.val_png(ev / "off.png", Verso1(), [it], torch.device("cpu"), lay)
    TR.val_png(ev / "on.png", Verso1(), [it], torch.device("cpu"), lay, verso=True)
    off = np.asarray(Image.open(ev / "off.png")).astype(int)
    on = np.asarray(Image.open(ev / "on.png")).astype(int)
    w = off.shape[1] // 3
    pred_off, pred_on = off[:, 2 * w:], on[:, 2 * w:]
    blue = lambda a: ((a[..., 2] - a[..., 0]) > 60).mean()   # noqa: E731
    assert blue(pred_off) == 0.0 and (pred_off[..., 0] - pred_off[..., 2] > 60).any()   # recto red
    assert blue(pred_on) > 0.5                                  # the saturated verso, when asked for
