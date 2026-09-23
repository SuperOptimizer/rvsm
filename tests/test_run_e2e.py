"""The driver end to end: `rvsm run` on the fixture CT, two rounds, on the CPU.

This is the one test that runs the whole project as a user would: a config file, `rvsm run`, and then
nothing but the run directory to look at. The teacher is the tiny `fake` one, the region is 128^3 (the
smallest a volcomp store can be), the student is the `1m` preset, and the producer runs as a thread in
the supervisor (`mode = cpu`: there is no card to share, so there is nothing for a second process to
buy). Everything else -- the walk both sides derive, the store state machine, the verso gate, the round
gate, the rounds driver, STOP, resume -- is the real code.

The gates are forced by the config rather than by the metrics: `verso_after_steps` and `round_steps`
are the plan's unconditional fallbacks, and at twenty steps on a synthetic slab no honest metric gate
would ever fire. The metric halves of both gates are tested directly, on rows, below.

Budget: under two minutes on four CPU threads.
"""
import json
import os
import threading
import time
from dataclasses import asdict, fields, replace

import numpy as np
import pytest
import torch

from rvsm import cli, run as RUN, stores
from rvsm.config import Config


@pytest.fixture(autouse=True)
def cpu_threads():
    """Four threads, for the reason `tests/test_train.py` gives: the `1m` net at these patch sizes is
    far too small to fill a many-core host, and the thread synchronisation dwarfs the arithmetic."""
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 4))
    yield
    torch.set_num_threads(n)


def write_toml(cfg, path):
    """`Config` -> a flat TOML file, so the test drives `rvsm run cfg.toml` and not a Python object."""
    def lit(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(lit(q) for q in v) + "]"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{k} = {lit(q)}" for k, q in v.items()) + "}"
        return json.dumps(str(v))
    lines = [f"{f.name} = {lit(getattr(cfg, f.name))}" for f in fields(cfg)]
    with open(str(path), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture
def run_cfg(region_cfg, fake_teacher, tmp_path, has_volcomp):
    """The end-to-end config: the `fake` teacher, both gates on their step fallbacks, two rounds."""
    if not has_volcomp:
        pytest.skip("a region store is a volcomp array; no libvolcomp on this host")
    return replace(region_cfg,
                   out=str(tmp_path / "e2e"), teacher_ckpts={"fake": fake_teacher.ckpt},
                   mode="cpu", gpus=(), rounds=2, steps=20, eval_every=5,
                   verso_after_steps=5, round_steps=10, heldout=1, workers=0,
                   min_regions_before_train=2, lookahead_extra=2, reserve_gb=0.001,
                   infer_window=64, infer_halo=8, cascade_depth=1, calibrate=True,
                   aff_offsets=(4, 8, 16))


# ===================================================================================== the whole run

def test_rvsm_run_two_rounds_end_to_end(run_cfg, tmp_path, capsys):
    cfg = run_cfg
    out = cfg.out
    t0 = time.time()
    assert cli.main(["run", write_toml(cfg, tmp_path / "cfg.toml")]) == 0
    took = time.time() - t0

    # ---- what the supervisor froze
    frozen = json.load(open(os.path.join(out, "config.json")))
    assert frozen["fingerprint"] == cfg.fingerprint()
    # cout is the plan's 14 rows ([recto, verso | midline, thickness | logvar | 3 x 3 affinities]);
    # cin is 15 and not 21 only because the fixture pyramid stops at rung 5, so the fixture config
    # carries three context cubes instead of nine -- the ONE contract number a 256^3 CT cannot have.
    assert frozen["layout"]["cout"] == 14 and frozen["layout"]["cin"] == 15
    assert os.path.exists(os.path.join(out, "umbilicus.json"))
    assert os.path.exists(os.path.join(out, "metadata.json"))
    held = json.load(open(os.path.join(out, "eval", "heldout.json")))["regions"]
    assert len(held) == 1

    st = RUN.read_state(out)
    assert st["verso_on"] is True, "the verso gate never fired"
    assert st["round"] == 1, f"the round gate never fired: {st}"
    assert st["step"] >= 20

    # ---- round 0: the teacher stores, and the verso the gate opened
    cnt = RUN.store_counts(out)
    assert cnt[0]["recto"] >= 2 and cnt[0]["rw"] == cnt[0]["recto"]
    assert cnt[0].get("verso", 0) >= 1, f"no verso store was ever written: {cnt}"
    a = stores.open_store(stores.store_path(out, "recto", held[0]["lo"], 0))
    assert a.attrs["done"] and a.attrs["producer"].startswith("teacher:")
    v = stores.open_store(stores.store_path(out, "verso", _first_done(out, "verso", 0), 0))
    assert v.attrs["radial_sign"] == -1, "the verso store is the FLIPPED-sign pass"

    # ---- round 1: every produced region carries the whole multi-head contract
    assert 1 in cnt, f"round 1 produced nothing: {cnt}"
    from rvsm import regions as RG
    for lo in RG.Catalog(out, 1, ttl=0.0).list_done("recto"):
        for ch in ("recto", "verso", "midline", "thickness", "conf"):
            assert stores.is_done(stores.store_path(out, ch, lo, 1)), (ch, lo)
        assert stores.open_store(stores.store_path(out, "recto", lo, 1)).attrs["producer"] == "student"

    # ---- the checkpoint, the eval, the temperatures
    from rvsm import infer
    _st, ck_cfg, layout, sd, temps, step = infer.load_student_ckpt(
        os.path.join(out, "ckpt", "student.pt"))
    assert layout.cout == 14 and layout.cin == 15
    assert ck_cfg.size == cfg.size and step >= 20
    # The calibration runs after every evaluation and its verdict is in the checkpoint. On THIS fixture
    # the verdict is an empty dict, and correctly so: the tiny `fake` teacher writes a soft probability
    # everywhere, so every rung's target is a pooled fraction and `calib.run` declines to fit a
    # temperature to a fraction (`calib.BINARY_FRAC`). What is asserted is that it ran and that what it
    # produced is usable, not that a synthetic slab has a meaningful temperature.
    assert "temps" in _st, "the calibration never ran"
    assert all(np.isfinite(v) and v > 0 for v in temps.values())
    assert os.path.exists(os.path.join(out, "ckpt", "teacher_round_1.pt"))

    ev = RUN.tail_jsonl(os.path.join(out, "logs", "eval.jsonl"), 100)
    assert ev and all(np.isfinite(r["dice"]) and np.isfinite(r["bce"]) for r in ev)
    pr = [r for r in RUN.tail_jsonl(os.path.join(out, "logs", "produce.jsonl"), 500)
          if r.get("kind") in ("teacher", "verso", "self", "fields")]
    assert {r["kind"] for r in pr} >= {"teacher", "verso", "self"}, {r["kind"] for r in pr}
    assert all(isinstance(r["s"], float) for r in pr)
    assert RUN.tail_jsonl(os.path.join(out, "logs", "sched.jsonl"), 500)

    # ---- `rvsm status` reads the directory and prints it
    capsys.readouterr()
    assert cli.main(["status", "--out", out]) == 0
    printed = capsys.readouterr().out
    assert "rvsm status" in printed and "round 1" in printed and "stores" in printed

    # ---- a restart asserts the fingerprint and keeps the state
    RUN.setup(cfg, out)                      # the same config: fine
    assert RUN.read_state(out)["round"] == 1
    with pytest.raises(AssertionError, match="fingerprint"):
        RUN.setup(replace(cfg, patch=64), out)
    assert took < 180, f"the end-to-end run took {took:.0f}s"


def _first_done(out, channel, round_):
    from rvsm import regions as RG
    got = RG.Catalog(out, round_, ttl=0.0).list_done(channel)
    assert got, f"no finished {channel} store in round {round_}"
    return got[0]


def test_rvsm_stop_ends_the_run_within_one_unit(region_cfg, fake_teacher, tmp_path, has_volcomp,
                                              monkeypatch):
    """`rvsm stop` touches STOP; the trainer checkpoints at its next evaluation and exits, and the
    producer stops after the region it is on. Nothing is killed."""
    if not has_volcomp:
        pytest.skip("a region store is a volcomp array; no libvolcomp on this host")
    cfg = replace(region_cfg, out=str(tmp_path / "stopme"),
                  teacher_ckpts={"fake": fake_teacher.ckpt}, mode="cpu", gpus=(), rounds=1,
                  steps=10000, eval_every=1, verso_after_steps=0, round_steps=10 ** 9,
                  heldout=1, workers=0, min_regions_before_train=1, reserve_gb=0.001,
                  infer_window=64, infer_halo=8, cascade_depth=1)
    box = {}
    # what a crashed run leaves behind: an hours-old producer heartbeat and a RAM-guard pause marker.
    # Neither may outlive the restart (the stale heartbeat read as a silent producer and restarted the
    # one just spawned; a stale pause would hold the producer forever)
    os.makedirs(os.path.join(cfg.out, "workers"), exist_ok=True)
    RUN._write_json(os.path.join(cfg.out, "workers", "produce.json"),
                    {"pid": 1, "phase": "round0", "last_ts": time.time() - 7000})
    open(os.path.join(cfg.out, RUN.PAUSE_FILE), "w").close()
    real = RUN.produce_loop

    def slow_start(*a, **k):             # a spawned producer takes seconds to stamp its first heartbeat
        time.sleep(1.5)
        return real(*a, **k)

    monkeypatch.setattr(RUN, "produce_loop", slow_start)

    def go():
        box["ck"] = RUN.run(cfg, out=cfg.out)

    th = threading.Thread(target=go, daemon=True)
    th.start()
    for _ in range(600):                       # wait for the trainer to be training
        if int(RUN.read_state(cfg.out).get("step", 0)) >= 1:
            break
        time.sleep(0.1)
    else:
        RUN.request_stop(cfg.out)
        th.join(60)
        pytest.fail("the trainer never reached its first evaluation")
    assert cli.main(["stop", "--out", cfg.out]) == 0
    th.join(90)
    assert not th.is_alive(), "the run did not stop"
    assert os.path.exists(box["ck"]) and box["ck"].endswith("student.pt")
    assert RUN.read_state(cfg.out)["round"] == 0
    sched = RUN.tail_jsonl(os.path.join(cfg.out, "logs", "sched.jsonl"), 10 ** 4)
    assert not [r for r in sched if r.get("kind") == "restart"], "a stale heartbeat restarted the producer"
    assert not RUN.producer_paused(cfg.out)


# =============================================================== the pieces, without a run around them

def test_choose_mode_and_the_vram_budget_table():
    """`auto` places the roles from the cards found, and a table that does not fit refuses to start."""
    big = RUN.choose_mode("auto", [(0, 80.0)])
    assert big == {"mode": "resident", "train_gpu": 0, "produce_gpu": 0, "phases": False,
                   "total_gb": 80.0}
    two = RUN.choose_mode("auto", [(0, 32.0), (1, 32.0)])
    assert two["mode"] == "timeshare" and two["train_gpu"] == 0 and two["produce_gpu"] == 1
    assert two["phases"] is False                     # one role per card: no switching
    one = RUN.choose_mode("auto", [(0, 32.0)])
    assert one["phases"] is True and one["train_gpu"] == one["produce_gpu"] == 0
    assert RUN.choose_mode("auto", [])["mode"] == "cpu"

    b = RUN.budget(big)
    assert b["train"]["gb"] + b["produce"]["gb"] <= 80.0 - RUN.HEADROOM_GB
    assert 0 < b["train"]["fraction"] < 1 and 0 < b["produce"]["fraction"] < 1
    # what an "80 GB" A100 actually reports: the table must fit it (it used to refuse by 0.03 GB)
    a100 = RUN.budget({"mode": "resident", "phases": False, "total_gb": 79.25})
    assert a100["train"]["gb"] + a100["produce"]["gb"] <= 79.25 - RUN.HEADROOM_GB + 1e-9
    assert a100["train"]["gb"] > 45.0 and a100["produce"]["gb"] > 29.0
    # the same table on a card it cannot fit on: refuse, and print the table
    with pytest.raises(SystemExit) as e:
        RUN.budget({"mode": "resident", "phases": False, "total_gb": 40.0},
                   table={"train": 46.0, "produce": 30.0})
    assert "train" in str(e.value) and "40.0 GB" in str(e.value) and "sum" in str(e.value)
    # phases: each role may use the whole card, because they are never resident together
    ph = RUN.budget(one)
    assert ph["train"]["gb"] == ph["produce"]["gb"] == pytest.approx(32.0 - RUN.HEADROOM_GB)


def test_the_producer_job_order_is_the_state_machine(tmp_path, has_volcomp):
    """Which pass a region lacks is read off the disk, in the order the state machine allows."""
    if not has_volcomp:
        pytest.skip("no libvolcomp on this host")
    from rvsm import regions as RG, targets as TG
    out, lo = str(tmp_path / "jobs"), (0, 0, 0)
    blk = np.zeros((128, 128, 128), np.uint8)
    cat = RG.Catalog(out, 0, ttl=0.0)
    assert RUN._next_job(cat, lo, 0, False, out) == "teacher"
    stores.write(stores.store_path(out, "recto", lo, 0), blk, lo, rung=2, channels=("recto",), q=8)
    cat = RG.Catalog(out, 0, ttl=0.0)
    assert RUN._next_job(cat, lo, 0, False, out) is None        # the gate has not fired: nothing to do
    assert RUN._next_job(cat, lo, 0, True, out) == "verso"
    stores.write(stores.store_path(out, "verso", lo, 0), blk, lo, rung=2, channels=("verso",), q=8)
    cat = RG.Catalog(out, 0, ttl=0.0)
    assert RUN._next_job(cat, lo, 0, True, out) == "fields"
    for k in (2, 3, 4):
        stores.write(stores.store_path(out, TG.channel("midline", k), lo, 0),
                     blk[:128 >> (k - 2), :128 >> (k - 2), :128 >> (k - 2)] if k == 2 else
                     np.zeros((128,) * 3, np.uint8), lo, rung=k,
                     channels=(TG.channel("midline", k),), q=0)
    cat = RG.Catalog(out, 0, ttl=0.0)
    assert RUN._next_job(cat, lo, 0, True, out) is None
    # round 1 is one multi-head pass, then the pooled fields
    assert RUN._next_job(RG.Catalog(out, 1, ttl=0.0), lo, 1, True, out) == "self"


def test_the_walk_is_the_same_on_both_sides_and_holds_the_heldout_first(small_cfg):
    """Both halves derive the walk from the same inputs, so `region_route` needs no message -- and the
    held-out regions come first, because their round-0 stores are the reference."""
    recs = [{"k": 2, "lo": [z * 128, 0, 0], "size": [128, 128, 128], "f": 1.0, "w": 1.0 / 6}
            for z in range(6)]
    held = [recs[3]]
    cfg = replace(small_cfg, region=128)
    v1, o1 = RUN.walk(cfg, recs, held)
    v2, o2 = RUN.walk(cfg, recs, held)
    assert o1 == o2 and len(v1) == len(v2)
    assert all(tuple(v["lo"]) != tuple(held[0]["lo"]) for v in v1), "a held-out region entered the walk"
    route, pos = RUN.region_route(cfg, v1, o1, held)
    assert route[0] == tuple(held[0]["lo"])
    assert len(route) == len(set(route)) == 6
    assert sorted(pos.values()) == sorted(set(pos.values()))
    # the window follows the cursor: the held-out region is always in it, the passed regions are not
    w = RUN._window(route, pos, cursor=0, L=1, held=held)
    assert w[0] == route[0] and len(w) < len(route)


def test_the_lookahead_is_re_estimated_from_the_logs(tmp_path, small_cfg):
    """L = ceil(T_produce / T_train) * K_active + extra, from `logs/produce.jsonl` and state.json."""
    out = str(tmp_path / "L")
    cfg = replace(small_cfg, lookahead_extra=4)
    k = 5
    assert RUN.lookahead(cfg, out, k) == k + 4                  # cold: the floor
    for s in (40.0, 41.0, 39.0):
        RUN.jlog(out, "produce", {"kind": "teacher", "s": s}, echo=False)
    RUN._write_json(os.path.join(out, "state.json"), {"region_s": 10.0})
    assert RUN.lookahead(cfg, out, k) == 4 * k + 4              # ceil(40 / 10) = 4
    RUN._write_json(os.path.join(out, "state.json"), {"region_s": 1e-3})
    assert RUN.lookahead(cfg, out, k) == RUN.LOOKAHEAD_MAX      # clamped


def test_the_verso_gate_needs_the_dice_and_the_betti_baseline(small_cfg):
    cfg = replace(small_cfg, verso_after_steps=1000, verso_gate_dice=0.6)
    good = [{"dice": 0.8, "betti0_err": 1.0, "base_betti0_err": 1.0},
            {"dice": 0.75, "betti0_err": 1.2, "base_betti0_err": 1.0}]
    bad = [{"dice": 0.3, "betti0_err": 1.0, "base_betti0_err": 1.0},
           {"dice": 0.35, "betti0_err": 1.0, "base_betti0_err": 1.0}]
    topo = [{"dice": 0.9, "betti0_err": 40.0, "base_betti0_err": 1.0},
            {"dice": 0.9, "betti0_err": 41.0, "base_betti0_err": 1.0}]
    assert RUN.verso_gate(cfg, "", 10, lambda: good)[0] is True
    assert RUN.verso_gate(cfg, "", 10, lambda: bad)[0] is False
    assert RUN.verso_gate(cfg, "", 10, lambda: topo)[0] is False, \
        "a topological blow-up must veto the gate"
    assert RUN.verso_gate(cfg, "", 10, lambda: [])[0] is False
    # the eval's fine-rung dice screens the held-out pass: far below the gate it is never paid for
    called = []
    ok, why = RUN.verso_gate(cfg, "", 10, lambda: called.append(1) or good, screen=0.1)
    assert not ok and why["why"] == "eval dice below the gate" and not called
    assert RUN.verso_gate(cfg, "", 10, lambda: called.append(1) or good, screen=0.55)[0] is True
    assert called == [1]                       # near the gate: the held-out pass decides
    called = []
    ok, why = RUN.verso_gate(cfg, "", 1000, lambda: called.append(1) or [])   # the fallback
    assert ok and why["why"] == "verso_after_steps"
    assert not called, "the fallback must not pay for a student pass it does not need"


def test_the_round_gate_wants_a_plateau_and_then_the_quality(tmp_path, small_cfg):
    out = str(tmp_path / "rounds")
    cfg = replace(small_cfg, round_steps=10 ** 9)
    import math
    rising = str(tmp_path / "rising")
    for s in range(1, 9):                                       # still climbing: not a plateau
        RUN.jlog(rising, "eval", {"step": s * 100, "dice": 0.85 * (1 - math.exp(-s * 100 / 3000.0))},
                 echo=False)
    assert RUN.plateau(rising, "dice")[0] is False
    assert RUN.round_gate(cfg, rising, 800, 0)[0] is False
    for s in range(1, 20):                                      # ... and a saturating one that is
        RUN.jlog(out, "eval", {"step": s * 100, "dice": 0.85 * (1 - math.exp(-s * 100 / 200.0))},
                 echo=False)
    flat, why = RUN.plateau(out, "dice")
    assert flat and why["remaining"] < RUN.ROUND_GAIN, why
    rows = [{"precision": 0.8, "betti0_err": 1.0}, {"precision": 0.82, "betti0_err": 1.2}]
    # round 0 has no previous round to be worse than: the plateau is the whole condition, and what it
    # measured becomes the reference the next round is judged against
    fire, why = RUN.round_gate(cfg, out, 1900, 0, lambda: rows)
    assert fire is True and why["rows"]["precision"] == pytest.approx(0.81)
    assert RUN.round_gate(cfg, out, 1900, 1, lambda: rows, why["rows"])[0] is True
    worse = [{"precision": 0.8, "betti0_err": 90.0}, {"precision": 0.8, "betti0_err": 91.0}]
    assert RUN.round_gate(cfg, out, 1900, 1, lambda: worse, why["rows"])[0] is False, \
        "a worse topology than round 0's discards the round"
    merged = [{"precision": 0.2, "betti0_err": 1.0}, {"precision": 0.2, "betti0_err": 1.0}]
    assert RUN.round_gate(cfg, out, 1900, 1, lambda: merged, why["rows"])[0] is False, \
        "more merging than round 0 discards the round"
    # the rows are only ever asked for once the plateau has opened the question
    seen = []
    assert RUN.round_gate(replace(cfg, round_steps=10 ** 9), rising, 800, 0,
                          lambda: seen.append(1) or rows)[0] is False
    assert not seen


def test_state_and_markers_are_atomic_and_readable_by_anyone(tmp_path):
    out = str(tmp_path / "bus")
    os.makedirs(out)
    RUN.write_state(out, round=0, step=3, cursor=7)
    RUN.write_state(out, step=9)
    st = RUN.read_state(out)
    assert (st["round"], st["step"], st["cursor"]) == (0, 9, 7)
    assert not RUN.stop_requested(out)
    RUN.request_stop(out)
    assert RUN.stop_requested(out)
    assert RUN.read_phase(out, "train") == "train"
    RUN.write_phase(out, "produce")
    assert RUN.read_phase(out) == "produce"


def test_old_rounds_are_deleted_only_once_superseded(tmp_path, has_volcomp):
    """At most two rounds live on disk, and a region's old store goes only when the new one is done."""
    if not has_volcomp:
        pytest.skip("no libvolcomp on this host")
    out = str(tmp_path / "gc")
    blk = np.zeros((128, 128, 128), np.uint8)
    for r in (0, 1, 2):
        for lo in ((0, 0, 0), (128, 0, 0)):
            if r == 2 and lo == (128, 0, 0):
                continue                                        # this region has NOT been regenerated
            stores.write(stores.store_path(out, "recto", lo, r), blk, lo, rung=2,
                         channels=("recto",), q=8)
    RUN._clean_old_rounds(out, 2, keep=2)
    assert not stores.is_done(stores.store_path(out, "recto", (0, 0, 0), 0))
    assert stores.is_done(stores.store_path(out, "recto", (128, 0, 0), 0)), \
        "an unsuperseded region must keep its old store"
    assert stores.is_done(stores.store_path(out, "recto", (0, 0, 0), 1))


def test_status_and_stop_refuse_a_directory_that_is_not_there(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["status", "--out", str(tmp_path / "nope")])
    with pytest.raises(SystemExit):
        cli.main(["stop", "--out", str(tmp_path / "nope")])
    assert cli.main(["run", "--help"]) == 0


def test_a_config_of_defaults_is_still_a_config(tmp_path):
    """`write_toml` round-trips every field, so the e2e test really does drive the CLI's own loader."""
    from rvsm import config as CFG
    cfg = Config(ct="x", out=str(tmp_path / "o"), teacher_ckpts={"fake": "p.pth"})
    got = CFG.load(write_toml(cfg, tmp_path / "c.toml"))
    assert got.fingerprint() == cfg.fingerprint()
    assert asdict(got) == asdict(cfg)


def test_setup_on_a_url_finds_the_regions(small_cfg, ct_origin, tmp_path):
    """A fresh run on a CT URL: the occupancy must be read from shards that are THERE. setup used to
    fetch only the metadata, read the occupancy level as air and freeze a run with zero regions and no
    held-out set."""
    from dataclasses import replace
    cfg = replace(small_cfg, ct=ct_origin.url, out=str(tmp_path / "url_run"))
    ctx = RUN.setup(cfg, cfg.out)
    ref = RUN.setup(replace(small_cfg, out=str(tmp_path / "path_run")), str(tmp_path / "path_run"))
    assert len(ctx["records"]) == len(ref["records"]) > 0
    assert len(ctx["heldout"]) == len(ref["heldout"]) > 0


def test_the_walk_sampler_class_pickles_by_name():
    """The loader's forkserver workers unpickle the dataset by its qualified name; a class built inside
    a function cannot be found that way (`rvsm run` with workers > 0 died on it on the first GPU run)."""
    import pickle
    cls = RUN.walk_patches()
    assert pickle.loads(pickle.dumps(cls)) is cls
    assert cls.__module__ == "rvsm.walk"


def test_the_trainer_walks_the_producers_walk(small_cfg, tmp_path):
    """Both halves must step along the SAME walk: the producer's route comes from `run.walk`, and the
    trainer's sampler builds its visits and order from the records it is handed. It used to be handed
    setup's full list (held-out regions included), so the two orders differed."""
    from dataclasses import replace
    cfg = replace(small_cfg, out=str(tmp_path / "w"), heldout=1)
    ctx = RUN.setup(cfg, cfg.out)
    assert ctx["heldout"]
    ds = RUN.walk_patches()(cfg, cfg.out, root=cfg.out, ct=ctx["ct"], ax=ctx["ax"], round_=0,
                            heldout=ctx["heldout"], meta=ctx["meta5"],
                            region_records=RUN.walk_records(ctx["records"], ctx["heldout"]))
    ds._open()
    assert [int(i) for i in ds.order] == [int(i) for i in ctx["order"]]
    assert [(v["k"], v["lo"], v.get("v")) for v in ds.visits] == \
        [(v["k"], v["lo"], v.get("v")) for v in ctx["visits"]]


def test_the_ram_guard_pauses_the_producer_and_lets_it_go(tmp_path):
    """Above RAM_PAUSE_FRAC of MemTotal (host in use, or the run's own tree RSS) the supervisor drops
    the PAUSE marker the producer honours, and says so loudly; below RAM_RESUME_FRAC it lifts it."""
    out = str(tmp_path / "g")
    os.makedirs(os.path.join(out, "logs"))
    G = 2 ** 30
    said = []
    rec = RUN.ram_guard(out, mem=(64 * G, 40 * G), rss=(10 * G, 9), log=said.append)
    assert not RUN.producer_paused(out) and "action" not in rec and rec["rss_gb"] == 10.0
    rec = RUN.ram_guard(out, mem=(64 * G, 5 * G), rss=(20 * G, 9), log=said.append)   # host 92 %
    assert RUN.producer_paused(out) and rec["action"] == "pause" and "PRODUCER PAUSED" in said[-1]
    rec = RUN.ram_guard(out, mem=(64 * G, 12 * G), rss=(20 * G, 9), log=said.append)  # 81 %: hold
    assert RUN.producer_paused(out) and "action" not in rec
    rec = RUN.ram_guard(out, mem=(64 * G, 30 * G), rss=(56 * G, 9), log=said.append)  # tree 87 %
    assert RUN.producer_paused(out)
    rec = RUN.ram_guard(out, mem=(64 * G, 40 * G), rss=(20 * G, 9), log=said.append)
    assert not RUN.producer_paused(out) and rec["action"] == "resume"
    kinds = [r.get("action") for r in RUN.tail_jsonl(os.path.join(out, "logs", "sched.jsonl"))]
    assert kinds == ["pause_producer", "resume_producer"]


def test_eval_dice_reads_the_evaluation_at_that_step(tmp_path):
    out = str(tmp_path / "e")
    for st, d in ((2000, 0.1), (4000, 0.2)):
        RUN.jlog(out, "eval", {"step": st, "dice": d}, echo=False)
    assert RUN.eval_dice(out, 4000) == 0.2 and RUN.eval_dice(out, 2000) == 0.1
    assert RUN.eval_dice(out, 6000) is None


def test_tree_rss_counts_this_process_and_its_children():
    import subprocess
    import sys
    me, n1 = RUN.tree_rss()
    assert me > 0 and n1 >= 1
    p = subprocess.Popen([sys.executable, "-c", "import time; x = bytearray(64 << 20); time.sleep(30)"])
    try:
        import time
        for _ in range(100):
            both, n2 = RUN.tree_rss()
            if n2 > n1 and both > me + (48 << 20):
                break
            time.sleep(0.1)
        assert n2 > n1 and both > me + (48 << 20)
    finally:
        p.kill()
