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
                   verso_after_steps=5, verso_min_dice=0.0, round_min_steps_after_verso=0, verso_min_regions=1,
                   verso_regen_gain=10.0, round_steps=5, heldout=1, workers=0,
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
    # round 0's stats are persisted as the reference every later round is judged against, so a
    # restart keeps the veto; the round's own start is recorded for the per-round step count
    assert st.get("round_ref") and all(k in st["round_ref"] for k in ("precision", "betti0_err"))
    assert st.get("verso_on") and int(st.get("round_step", 0)) >= cfg.round_steps
    assert st.get("verso_on_step") is not None
    # every round-0 verso store records the checkpoint step that made it (and its generation)
    from rvsm import regions as _RG, stores as _ST
    vs = _RG.Catalog(out, 0).list_done("verso")
    assert vs and all("step" in _ST.read_attrs(_ST.store_path(out, "verso", v, 0)) and
                      "gen" in _ST.read_attrs(_ST.store_path(out, "verso", v, 0)) for v in vs)
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
    # a behaviour test, not a benchmark: ~190 s on an idle 8-core box, 450+ s on a loaded laptop
    assert took < 1200, f"the end-to-end run took {took:.0f}s"


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
                  steps=10000, eval_every=1, verso_after_steps=0, verso_min_dice=0.0, round_steps=10 ** 9,
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
    for _ in range(3000):                      # wait for the trainer to be training (slow under load)
        if int(RUN.read_state(cfg.out).get("step", 0)) >= 1:
            break
        time.sleep(0.1)
    else:
        RUN.request_stop(cfg.out)
        th.join(60)
        pytest.fail("the trainer never reached its first evaluation")
    assert cli.main(["stop", "--out", cfg.out]) == 0
    th.join(300)
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
        for kind in TG.KINDS:              # `targets.fields_current`: every field store at every rung
            stores.write(stores.store_path(out, TG.channel(kind, k), lo, 0),
                         blk[:128 >> (k - 2), :128 >> (k - 2), :128 >> (k - 2)] if k == 2 else
                         np.zeros((128,) * 3, np.uint8), lo, rung=k,
                         channels=(TG.channel(kind, k),), q=0)
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


def test_the_window_reaches_past_the_fastest_worker(tmp_path):
    """Workers drift apart by whole visits. The window must cover the FASTEST worker's next visit, not
    only `L` past the slowest one -- otherwise that worker waits on a store nobody produces, the
    in-order DataLoader waits on it, the cursor never moves: the paris4 deadlock after the resume."""
    out = str(tmp_path / "c")
    for w, p in enumerate((7, 5, 5, 5, 5, 6)):            # the paris4 cursor files at the deadlock
        RUN._write_json(os.path.join(RUN.cursor_dir(out), f"w{w}.json"),
                        {"pos": p, "stride": 6, "worker": w})
    assert RUN.read_cursor(out) == 30 and RUN.read_cursor_head(out) == 42
    route = [("h",)] + [(n,) for n in range(100)]
    pos = {(n,): n for n in range(100)}
    old = RUN._window(route, pos, 30, 9, [None])
    new = RUN._window(route, pos, 30, 9, [None], head=RUN.read_cursor_head(out))
    assert (42,) not in old, "the old window stopped short of worker 0's next visit"
    assert (42,) in new and (51,) in new and (52,) not in new and (29,) not in new


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
    ok, why = RUN.verso_gate(cfg, "", 1000, lambda: called.append(1) or [], screen=0.2,
                             r2=[0.32, 0.35])                        # the fallback
    assert ok and why["why"] == "verso_after_steps" and why["dice_r2"] == 0.35
    assert why["dice_r2_prev"] == 0.32                               # both values are logged
    # ... which needs the RUNG-2 dice at verso_min_dice on TWO consecutive evaluations
    ok, why = RUN.verso_gate(cfg, "", 1000, lambda: called.append(1) or [], screen=0.2, r2=[0.1, 0.35])
    assert not ok and "two consecutive" in why["why"] and why["verso_min_dice"] == 0.3
    assert not RUN.verso_gate(cfg, "", 1000, lambda: [], r2=[0.4])[0], "one evaluation is not two"
    assert not RUN.verso_gate(cfg, "", 1000, lambda: [], screen=None)[0]
    assert not called, "the fallback must not pay for a student pass it does not need"


def test_the_round_gate_fails_closed(tmp_path, small_cfg):
    """Every doubt is a NO: round 0 needs verso on; a round needs round_steps of ITS OWN steps; the
    global budget must leave a round to train; no rows, non-finite rows, or (round >= 1) no reference
    never promote; and the rows are only paid for once the cheap conditions pass."""
    out = str(tmp_path / "rounds")
    cfg = replace(small_cfg, round_steps=1000, steps=60000)
    rows = [{"precision": 0.8, "betti0_err": 1.0}, {"precision": 0.82, "betti0_err": 1.2}]
    seen = []

    def rf():
        seen.append(1)
        return rows
    no, why = RUN.round_gate(cfg, out, 5000, 0, rf, verso_on=False)
    assert not no and "verso_on" in why["why"]
    no, why = RUN.round_gate(cfg, out, 5500, 1, rf, rows[0], round_start=5000)   # (a) 500 in round
    assert not no and why["why"] == "round_step below round_steps" and why["round_step"] == 500
    no, why = RUN.round_gate(cfg, out, 59500, 0, rf)                              # (d) no budget left
    assert not no and "budget" in why["why"]
    assert not seen, "the rows are paid for only once the cheap conditions pass"
    vv = dict(verso_on_step=-5000, verso_regions=500)
    assert RUN.round_gate(cfg, out, 5000, 0, lambda: [], **vv)[1]["why"].startswith("no complete")
    bad = [{"precision": float("nan"), "betti0_err": 1.0}]
    assert RUN.round_gate(cfg, out, 5000, 0, lambda: bad, **vv)[1]["why"].startswith("no complete")
    v = dict(verso_on_step=1000, verso_regions=500)                              # verso long enough on
    assert RUN.round_gate(cfg, out, 5000, 0, rf, verso_on_step=4000, verso_regions=500)[1]["why"] \
        .startswith("verso has not been on"), "P3-03: 1000 steps of verso < 8000"
    assert RUN.round_gate(cfg, out, 5000, 0, rf, verso_on_step=-5000, verso_regions=50)[1]["why"] \
        .startswith("fewer than verso_min_regions")
    v = dict(verso_on_step=-5000, verso_regions=500)
    partial = [{"precision": 0.8, "betti0_err": 1.0}, {"precision": 0.9}]   # a row missing a metric
    fire, why = RUN.round_gate(cfg, out, 5000, 0, lambda: partial, **v)
    assert fire and why["n"] == 1, "a row missing a metric is dropped whole"
    fire, why = RUN.round_gate(cfg, out, 5000, 0, rf, **v)                      # round 0: rows = ref
    assert fire and why["rows"]["precision"] == pytest.approx(0.81)
    ref = why["rows"]
    no, why = RUN.round_gate(cfg, out, 6500, 1, rf, None, round_start=5000)      # (c) no reference
    assert not no and "reference" in why["why"]
    assert RUN.round_gate(cfg, out, 6500, 1, rf, ref, round_start=5000)[0] is True
    worse = [{"precision": 0.8, "betti0_err": 90.0}, {"precision": 0.8, "betti0_err": 91.0}]
    assert RUN.round_gate(cfg, out, 6500, 1, lambda: worse, ref, round_start=5000)[0] is False
    merged = [{"precision": 0.2, "betti0_err": 1.0}, {"precision": 0.2, "betti0_err": 1.0}]
    assert RUN.round_gate(cfg, out, 6500, 1, lambda: merged, ref, round_start=5000)[0] is False


def test_the_plateau_reads_only_this_rounds_evaluations(tmp_path):
    import math
    out = str(tmp_path / "pl")
    for s in range(1, 20):                                      # round 0: a saturated curve
        RUN.jlog(out, "eval", {"step": s * 100, "dice": 0.85 * (1 - math.exp(-s * 100 / 200.0))},
                 echo=False)
    for s in range(1, 4):                                       # round 1: three points so far
        RUN.jlog(out, "eval", {"step": 2000 + s * 100, "dice": 0.3 * s}, echo=False)
    flat, why = RUN.plateau(out, "dice", since=2000)            # ... this round's does not
    assert not flat and why["why"] == "too few eval points" and why["n"] == 3


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


def _stub_walk(out, n=40, start=None, L=3, dead=(), wait_until=None):
    """A WalkPatches over `n` synthetic visits whose windows are just the visit ids (one per visit)."""
    from rvsm.walk import WalkPatches
    ds = WalkPatches.__new__(WalkPatches)
    ds.out, ds.L, ds.wait_s, ds.start = str(out), L, 0.0, start
    ds.windows, ds.seed, ds.round, ds.root = 1, 0, 0, str(out)
    ds.pyr = object()
    ds.visits = [{"k": 2, "lo": [i * 128, 0, 0], "size": [128] * 3, "id": i} for i in range(n)]
    ds.order = list(range(n))[::-1]
    ds.cfg = type("C", (), {"region": 128})()

    class Cat:
        def __init__(self, *a):
            pass

        def done(self, ch, lo):
            return True                      # no verso revisits in this test
    ds.cat = Cat()
    ds._dead = lambda rec: rec["id"] in dead
    ds._visitable = lambda rec: wait_until is None or rec["id"] not in wait_until
    ds.air_budget = lambda: 1
    ds._last_air = 0
    ds._draw = lambda rng, rec, air_ok=True: rec["id"]
    return ds


def test_a_resumed_walk_continues_where_the_checkpoint_left_it(tmp_path):
    """The standing rule: a restart never repeats training data. The walk position the workers
    publish is snapshotted into state.json at every checkpoint; a resumed sampler starts there (and
    skips the visits it had already made past it) instead of at 0, and the producer's window follows."""
    import itertools
    out = tmp_path / "wk"
    os.makedirs(out / "logs")
    first = list(itertools.islice(iter(_stub_walk(out, dead={37})), 12))
    assert first == [39, 38, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27]     # order reversed, 37 dead
    snap = RUN.walk_snapshot(str(out), 0)
    assert snap["stride"] == 1 and snap["workers"]["0"]["pos"] == 13
    RUN.write_state(str(out), round=0, walk=snap)
    assert RUN.resume_walk(str(out), 0) == snap
    # the resumed walk starts at the saved position: nothing from the first run comes again
    again = list(itertools.islice(iter(_stub_walk(out, dead={37}, start=RUN.resume_walk(str(out), 0))), 27))
    assert again[0] == 26 and not set(again) & set(first)
    assert sorted(first + again) == [i for i in range(40) if i != 37]    # the same visits, once each
    # the producer's window follows the restored walk: a resumed worker publishes its saved position
    # before its first window, and the window over the walk starts there, not at 0
    it = iter(_stub_walk(out, dead={37}, start=snap))
    assert next(it) == 26                                    # the visit at the saved 13 starts
    assert RUN.read_cursor(str(out)) == 14 and RUN.read_cursor_head(str(out)) == 14
    route = [(i,) for i in range(40)]
    win = RUN._window(route, {(i,): i for i in range(40)}, RUN.read_cursor(str(out)), 3, [],
                      head=RUN.read_cursor_head(str(out)))
    assert win == [(i,) for i in range(14, 18)]
    # a new round walks from 0; a different worker count ignores the snapshot
    assert RUN.resume_walk(str(out), 1) is None
    assert list(itertools.islice(iter(_stub_walk(out, start={**snap, "stride": 6})), 1)) == [39]


def test_a_resumed_walk_skips_what_it_visited_past_a_waiting_region(tmp_path):
    """A visit that waited for its store leaves the saved `pos` behind it; the ones visited past it
    are in `done` and are not visited again after the restart."""
    import itertools
    out = tmp_path / "wk2"
    os.makedirs(out / "logs")
    it = iter(_stub_walk(out, L=4, wait_until={39}))
    got = [next(it) for _ in range(2)]                  # 39 waits for its store: 38, 37 go first
    assert got == [38, 37]
    snap = RUN.walk_snapshot(str(out), 0)
    assert snap["workers"]["0"]["pos"] == 0 and 38 not in snap["workers"]["0"]["done"]
    assert snap["workers"]["0"]["done"] == [1, 2]
    after = list(itertools.islice(iter(_stub_walk(out, L=4, start=snap)), 6))
    assert after[0] == 39 and 38 not in after and 37 not in after


def test_round_r_student_passes_use_the_frozen_round_teacher(tmp_path, monkeypatch):
    """Round 0's verso passes track the live student.pt; round r >= 1 uses the frozen
    ckpt/teacher_round_<r>.pt from state.json, never the student being trained, and the slot knows
    the sha256 of what it loaded (the producer writes it into the store attrs)."""
    import hashlib
    from rvsm import infer
    out = tmp_path / "slot"
    (out / "ckpt").mkdir(parents=True)
    (out / "ckpt" / "student.pt").write_bytes(b"live")
    tp = out / "ckpt" / "teacher_round_1.pt"
    loads = []
    got_bytes = []

    def fake(p, device=None, compile=True, data=None):
        loads.append(p)
        got_bytes.append(data)
        return p
    monkeypatch.setattr(infer, "student_fn", fake)
    slot = RUN.StudentSlot(str(out), compile=False)
    assert slot.get(0) == str(out / "ckpt" / "student.pt")
    assert slot.sha == hashlib.sha256(b"live").hexdigest()
    assert got_bytes[-1] == b"live", "the weights are loaded from the very bytes that were hashed"
    assert slot.get(1, str(tp)) is None, "round 1 must not fall back to the live student"
    tp.write_bytes(b"frozen")
    assert slot.get(1, str(tp)) == str(tp) and slot.sha == hashlib.sha256(b"frozen").hexdigest()
    (out / "ckpt" / "student.pt").write_bytes(b"live, newer")          # training moves on ...
    assert slot.get(1, str(tp)) == str(tp) and loads.count(str(tp)) == 1  # ... round 1 does not


def test_the_scan_metadata_is_frozen_on_the_first_setup(tmp_path):
    """metadata.json / meta5.json are written once and a resume never refetches or overwrites them;
    on a fresh run an unreadable metadata.json is an error, never the defaults (review O13)."""
    import pathlib
    ex = pathlib.Path(__file__).resolve().parent.parent / "docs" / "example_metadata.json"
    vol = tmp_path / "vol.zarr"
    vol.mkdir()
    (vol / "metadata.json").write_text(ex.read_text())
    out = tmp_path / "run"
    out.mkdir()
    meta, m5 = RUN.frozen_meta(str(out), str(vol))
    assert not meta.get("missing") and len(m5) == 5
    frozen = (out / "meta5.json").read_text()
    (vol / "metadata.json").unlink()                       # the source is gone (or changed) ...
    meta2, m52 = RUN.frozen_meta(str(out), str(vol))       # ... the resume keeps the frozen copy
    assert m52 == m5 and (out / "meta5.json").read_text() == frozen
    # a DEFINITE absence (no file / HTTP 404) freezes the defaults with absent: true and zero planes
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    said = []
    meta3, m53 = RUN.frozen_meta(str(fresh), str(vol), log=said.append)
    assert meta3["absent"] is True and m53 == [0.0] * 5 and (fresh / "meta5.json").exists()
    assert said and "ABSENT" in said[0]
    # a TRANSPORT failure is an error on a fresh run -- never frozen defaults
    from rvsm import scanmeta as SM
    orig = SM.probe
    try:
        SM.probe = lambda p, timeout=10.0: (None, "error")
        err = tmp_path / "err"
        err.mkdir()
        with pytest.raises(SystemExit, match="transport"):
            RUN.frozen_meta(str(err), str(vol))
        assert not (err / "meta5.json").exists()
        # ... and a resume never calls the network at all: paris4's frozen defaults stay as they are
        again, m5again = RUN.frozen_meta(str(fresh), str(vol), log=said.append)
        assert m5again == [0.0] * 5
    finally:
        SM.probe = orig


def test_the_metadata_probe_tells_a_404_from_a_transport_failure(tmp_path):
    """HTTP 404 -> absent; a refused connection -> error (review O13 follow-up)."""
    import http.server
    import threading
    from rvsm import scanmeta as SM

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        assert SM.probe(f"http://127.0.0.1:{srv.server_port}/v.zarr")[1] == "absent"
    finally:
        srv.shutdown()
    assert SM.probe(f"http://127.0.0.1:{srv.server_port}/v.zarr", timeout=2)[1] == "error"


# --------------------------------------------------------------------------- T19: the round transition

def test_a_stale_old_round_publish_after_the_reset_is_rejected(tmp_path):
    """Review T19: a pending old-round fetch (or a persistent loader worker) can publish its cursor
    AFTER the round transition reset the directory. The transition bumps the round first, a stale
    walk's `_publish` then refuses to write, and every cursor reader rejects a record stamped with an
    old round even when one does land (the check and the write are not atomic)."""
    out = tmp_path / "t19"
    os.makedirs(out / "logs")
    RUN.write_state(str(out), round=0)
    ds = _stub_walk(out)                               # round 0
    assert ds._publish(0, 1, 7, 1.0) is True
    assert RUN.read_cursor(str(out)) == 7
    published = []

    def quiesce():                                     # the old round's last fetch lands mid-transition
        published.append(ds._publish(0, 1, 9, 1.0))
    RUN.round_transition(str(out), 1, quiesce, round_step=5)
    assert published == [False], "a stale writer published after the round was bumped"
    assert RUN.read_state(str(out))["round"] == 1 and RUN.read_state(str(out))["cursor"] == 0
    assert not os.path.exists(RUN.cursor_dir(str(out)))
    # a late old-round publish after the reset: refused, and the directory is not recreated
    assert ds._publish(0, 1, 11, 1.0) is False
    assert not os.path.exists(RUN.cursor_dir(str(out)))
    # one that raced past the check anyway is ignored by every reader; the new round's own counts
    RUN._write_json(os.path.join(RUN.cursor_dir(str(out)), "w0.json"),
                    {"pos": 11, "stride": 1, "worker": 0, "round": 0, "region_s": 3.0})
    assert RUN.read_cursor(str(out)) == 0 and RUN.read_cursor_head(str(out)) == 0
    assert RUN.walk_snapshot(str(out), 1) is None and RUN.region_seconds(str(out)) == 0.0
    new = _stub_walk(out)
    new.round = 1
    assert new._publish(1, 2, 4, 2.0) is True
    assert RUN.read_cursor(str(out)) == 8 and RUN.walk_snapshot(str(out), 1)["workers"] == {
        "1": {"pos": 4, "done": [], "pass": 0}}


def test_an_old_round_walk_stops_in_its_wait_loop_and_between_windows(tmp_path):
    """The walk's wait loop checks the round as well as STOP (it waited forever for old-round stores
    nobody will produce), and a visit in progress stops at its next window."""
    import itertools
    out = tmp_path / "t19w"
    os.makedirs(out / "logs")
    RUN.write_state(str(out), round=0)
    waiting = _stub_walk(out, wait_until=set(range(40)))   # nothing is ever ready
    got = []
    th = threading.Thread(target=lambda: got.extend(iter(waiting)), daemon=True)
    th.start()
    time.sleep(0.2)
    assert th.is_alive(), "the walk should be waiting"
    RUN.write_state(str(out), round=1)
    th.join(5.0)
    assert not th.is_alive() and got == []
    # mid-visit: the round moves on after the first window of a three-window visit
    RUN.write_state(str(out), round=0)
    ds = _stub_walk(out)
    ds.windows = 3
    n = {"i": 0}

    def draw(rng, rec, air_ok=True):
        n["i"] += 1
        if n["i"] == 2:
            RUN.write_state(str(out), round=1)
        return rec["id"]
    ds._draw = draw
    assert list(itertools.islice(iter(ds), 10)) == [39, 39]


class _Endless(torch.utils.data.IterableDataset):
    def __iter__(self):
        i = 0
        while True:
            i += 1
            yield torch.tensor([i])


def test_close_quiesces_the_loader_and_its_persistent_workers():
    """`DevicePrefetch.close` is the shutdown protocol the round transition calls: no further batch,
    and the loader's persistent workers are gone before it returns (they can publish nothing after)."""
    import itertools

    from rvsm import train as TR
    dl = torch.utils.data.DataLoader(_Endless(), batch_size=None, num_workers=2,
                                     persistent_workers=True, prefetch_factor=2,
                                     multiprocessing_context="forkserver")
    src = TR.DevicePrefetch(dl, torch.device("cpu"))
    it = iter(src)
    assert int(next(it)[0]) >= 1
    workers = list(dl._iterator._workers)
    assert workers and all(w.is_alive() for w in workers)
    t0 = time.time()
    assert src.close(timeout=10.0) is True
    assert time.time() - t0 < 30.0
    assert dl._iterator is None
    for w in workers:
        w.join(5.0)
    assert not any(w.is_alive() for w in workers)
    assert list(itertools.islice(it, 3)) == []          # nothing after close
    assert src.close() is True                           # idempotent


def test_close_gives_up_on_a_fetch_that_never_returns():
    """A fetch in flight that never returns (an old-round walk blocked forever) does not block the
    transition past its timeout: the daemon thread is abandoned, never joined."""
    import threading as th
    from rvsm import train as TR
    gate = th.Event()

    def stuck():
        yield {"x": torch.zeros(1)}
        gate.wait()                                     # never set: the next batch never comes
        yield {"x": torch.zeros(1)}
    src = TR.DevicePrefetch(stuck(), torch.device("cpu"))
    src._it = stuck()
    next(src._it)
    src._fut = TR._spawn(lambda: next(src._it))
    t0 = time.time()
    assert src.close(timeout=0.5) is False
    assert time.time() - t0 < 5.0
    gate.set()


def test_the_walk_leases_its_homes_and_waits_for_an_evicted_one(tmp_path, monkeypatch):
    """D04, the trainer's side: every publish leases the visit being read and the pending ones, and on
    a streamed mirror a visit whose home lost its shards does not start -- the worker leases it while
    it waits, and starts it once the producer has fetched it again."""
    from rvsm import stream
    out = tmp_path / "lease"
    os.makedirs(out / "logs")
    RUN.write_state(str(out), round=0)
    shard = lambda home: str(tmp_path / ("shard_%d" % home[0]))          # noqa: E731
    monkeypatch.setattr(stream, "region_paths", lambda pyr, home, ctx, region: [shard(home)])
    ds = _stub_walk(out, n=4)                          # order 3, 2, 1, 0; homes (128 * id, 0, 0)
    ds.stream_mirror, ds.ctx = True, ()
    for i in (0, 1, 2):
        open(shard((128 * i, 0, 0)), "w").close()       # visit 3's home is not on disk
    halt = threading.Event()

    def keeper():                                       # the producer's ack: the homes on disk
        while not halt.is_set():
            for r in RUN.cursor_records(str(out)):
                ready = [lo for lo in r.get("lease") or () if os.path.exists(shard(lo))]
                RUN.write_lease_ack(str(out), r["worker"], r.get("lease_id"), ready)
            halt.wait(0.01)
    threading.Thread(target=keeper, daemon=True).start()
    got, it = [], iter(ds)
    got += [next(it) for _ in range(3)]
    assert got == [2, 1, 0], "the non-resident visit 3 must not start"
    rec = RUN._read_json(os.path.join(RUN.cursor_dir(str(out)), "w0.json"))
    assert [384, 0, 0] in rec["lease"] and rec["round"] == 0
    th = threading.Thread(target=lambda: got.append(next(it)), daemon=True)
    th.start()
    t0 = time.time()
    while time.time() - t0 < 5 and RUN.cursor_leases(str(out)) != [(384, 0, 0)]:
        time.sleep(0.01)
    assert RUN.cursor_leases(str(out)) == [(384, 0, 0)], "the waiting worker leases what it waits on"
    assert th.is_alive() and got == [2, 1, 0]
    open(shard((384, 0, 0)), "w").close()               # the producer fetched it again
    th.join(5.0)
    assert got == [2, 1, 0, 3]
    assert (384, 0, 0) in RUN.cursor_leases(str(out))   # held while its windows are drawn
    halt.set()


# --------------------------------------------------------------------------- O10: producer supervision

class _FakeProc:
    """A producer process stand-in: `alive` until killed (or forever, `unkillable`)."""

    def __init__(self, alive=True, exitcode=None, unkillable=False, pid=None):
        self.alive, self.exitcode, self.unkillable = alive, exitcode, unkillable
        self.pid = pid
        self.calls = []

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.calls.append("terminate")
        if not self.unkillable:
            self.alive, self.exitcode = False, -15

    def kill(self):
        self.calls.append("kill")
        if not self.unkillable:
            self.alive, self.exitcode = False, -9

    def join(self, t=None):
        self.calls.append("join")


def _watch(tmp_path, first, clock):
    out = str(tmp_path / "sup")
    os.makedirs(os.path.join(out, "workers"), exist_ok=True)
    spawned = []

    def respawn():
        p = _FakeProc()
        spawned.append(p)
        return p
    procs = {"produce": first}
    w = RUN.ProducerWatch(out, procs, respawn, log=lambda m: None, clock=lambda: clock[0], join_s=0.0)
    return out, procs, spawned, w


def test_a_dead_producer_is_restarted_at_once_then_with_backoff(tmp_path):
    """O10: liveness is the process, not the heartbeat's age -- an immediate crash used to stall the run
    for SILENT_MAX_S. The first restart is immediate, a producer that keeps dying is restarted with a
    doubling, bounded delay, and a long healthy life resets that."""
    clock = [1000.0]
    out, procs, spawned, w = _watch(tmp_path, _FakeProc(alive=False, exitcode=1), clock)
    RUN._write_json(os.path.join(out, "workers", "produce.json"), {"last_ts": clock[0]})
    assert w.check() == "restart" and len(spawned) == 1       # at once, the stamp is fresh
    assert RUN._read_json(os.path.join(out, "workers", "produce.json"))["phase"] == "spawning"
    spawned[0].alive, spawned[0].exitcode = False, 1           # dies again straight away
    assert w.check() == "backoff" and len(spawned) == 1
    clock[0] += RUN.RESTART_MIN_S + 0.1
    assert w.check() == "restart" and len(spawned) == 2
    spawned[1].alive = False
    clock[0] += RUN.RESTART_MIN_S + 0.1                        # the delay has doubled
    assert w.check() == "backoff"
    clock[0] += RUN.RESTART_MIN_S
    assert w.check() == "restart" and len(spawned) == 3
    for _ in range(20):                                        # bounded
        procs["produce"].alive = False
        clock[0] += RUN.RESTART_MAX_S + 0.1
        assert w.check() == "restart"
    assert w.fails >= RUN.RESTART_FATAL
    assert any(r.get("kind") == "producer_fatal" for r in RUN.tail_jsonl(os.path.join(out, "logs",
                                                                                       "sched.jsonl")))
    # a producer that lives RESTART_RESET_S has earned a fresh start
    RUN._write_json(os.path.join(out, "workers", "produce.json"), {"last_ts": clock[0]})
    clock[0] += RUN.RESTART_RESET_S + 1
    RUN._write_json(os.path.join(out, "workers", "produce.json"), {"last_ts": clock[0]})
    assert w.check() is None and w.fails == 0
    procs["produce"].alive = False
    assert w.check() == "restart"                              # immediate again


def test_a_silent_producer_is_replaced_only_once_it_has_exited(tmp_path):
    """A silent producer is terminated (then killed); a new one is spawned only once the old one has
    EXITED -- one that will not die is reported, never duplicated -- and a producer THREAD (cpu mode),
    which cannot be stopped, is never restarted while it lives. A healthy one is left alone."""
    clock = [5000.0]
    hung = _FakeProc(unkillable=True)
    out, procs, spawned, w = _watch(tmp_path, hung, clock)
    hb = os.path.join(out, "workers", "produce.json")
    RUN._write_json(hb, {"last_ts": clock[0] - 10})
    assert w.check() is None and not hung.calls                # healthy
    RUN._write_json(hb, {"last_ts": clock[0] - RUN.SILENT_MAX_S - 1})
    assert w.check() == "stuck" and not spawned
    assert "terminate" in hung.calls and "kill" in hung.calls
    assert procs["produce"] is hung
    hung.unkillable = False                                    # it finally responds to a signal
    assert w.check() == "restart" and len(spawned) == 1 and procs["produce"] is spawned[0]
    assert hung.calls.count("terminate") == 2

    class _Thread:                                             # no terminate(): cannot be stopped
        def is_alive(self):
            return True
    procs["produce"] = _Thread()
    RUN._write_json(hb, {"last_ts": clock[0] - RUN.SILENT_MAX_S - 1})
    assert w.check() == "silent_thread" and len(spawned) == 1


def _sleep_forever():
    time.sleep(3600)


def _producer_with_a_pool():
    import multiprocessing as mp
    kid = mp.get_context("fork").Process(target=_sleep_forever, daemon=False)
    kid.start()
    time.sleep(3600)


def test_a_silent_producer_is_killed_with_its_process_tree(tmp_path):
    """A real process tree: the silent producer and its child (the fields pool) are both gone before
    the replacement is spawned."""
    import multiprocessing as mp
    pr = mp.get_context("fork").Process(target=_producer_with_a_pool)
    pr.start()
    try:
        t0 = time.time()
        while time.time() - t0 < 10 and not RUN.proc_tree(pr.pid):
            time.sleep(0.05)
        tree = RUN.proc_tree(pr.pid)
        assert len(tree) == 1
        clock = [time.time()]
        out, procs, spawned, w = _watch(tmp_path, pr, clock)
        w.join_s = 10.0
        RUN._write_json(os.path.join(out, "workers", "produce.json"),
                        {"last_ts": clock[0] - RUN.SILENT_MAX_S - 1})
        assert w.check() == "restart" and len(spawned) == 1
        assert not pr.is_alive()
        kid = tree[0][0]
        t0 = time.time()
        while time.time() - t0 < 10 and _running(kid):
            time.sleep(0.05)
        assert not _running(kid), "the producer's child outlived it"
    finally:
        if pr.is_alive():
            pr.kill()


def _running(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            st = f.read()
    except OSError:
        return False
    return st[st.rindex(")") + 2] != "Z"


def _orphan_parent(q):
    import multiprocessing as mp
    from rvsm import targets as TG

    def child():
        TG.die_with_parent()
        time.sleep(3600)
    kid = mp.get_context("fork").Process(target=child)
    kid.start()
    q.put(kid.pid)
    time.sleep(0.5)
    os._exit(0)                                                # dies without reaping or killing it


def test_a_fields_pool_worker_dies_with_its_parent():
    """`targets.die_with_parent` (the fields pool's initializer): when the parent goes, so does the
    worker, whatever killed the parent."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_orphan_parent, args=(q,))
    p.start()
    kid = q.get(timeout=10)
    p.join(10)
    t0 = time.time()
    while time.time() - t0 < 10 and _running(kid):
        time.sleep(0.05)
    assert not _running(kid)


def test_the_heartbeat_ticker_stamps_during_a_long_unit():
    """The producer's own stamping thread: the loop stamps between units only, and a unit longer than
    SILENT_MAX_S (a first TensorRT compile) got a healthy producer restarted."""
    stop = threading.Event()
    n = []
    t = threading.Thread(target=RUN.hb_ticker, args=(lambda: n.append(1), stop, 0.02), daemon=True)
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(2.0)
    assert not t.is_alive() and len(n) >= 5
    assert RUN.HB_TICK_S < RUN.SILENT_MAX_S / 10


def test_a_visit_draws_only_after_its_lease_is_acknowledged(tmp_path):
    """P3-02, the worker's side: after publishing, a visit waits for the producer's ack of THAT lease
    (an ack of an older lease id does not count), ends on a round change, and after the timeout
    proceeds loudly rather than stopping training."""
    from rvsm import walk as WK
    out = tmp_path / "ack"
    os.makedirs(out / "logs")
    RUN.write_state(str(out), round=0)
    ds = _stub_walk(out)
    ds.stream_mirror = True
    assert ds._publish(0, 1, 0, 0.0, lease=[(0, 0, 0)])
    old = ds._lease_id
    assert ds._publish(0, 1, 0, 0.0, lease=[(0, 0, 0)])
    assert ds._lease_id != old
    RUN.write_lease_ack(str(out), 0, old, [[0, 0, 0]])                   # stale: an older lease
    got = []
    th = threading.Thread(target=lambda: got.append(ds._await_ack(0, (0, 0, 0), timeout=10)),
                          daemon=True)
    th.start()
    time.sleep(0.3)
    assert th.is_alive(), "an ack of another lease id must not release the visit"
    RUN.write_lease_ack(str(out), 0, ds._lease_id, [[128, 0, 0]])        # this lease, other home
    time.sleep(0.2)
    assert th.is_alive()
    RUN.write_lease_ack(str(out), 0, ds._lease_id, [[0, 0, 0], [128, 0, 0]])
    th.join(5)
    assert got == [True]
    # timeout: proceed, and say so
    assert ds._await_ack(0, (256, 0, 0), timeout=0.1) is True
    assert any(r.get("kind") == "lease_ack_timeout"
               for r in RUN.tail_jsonl(os.path.join(str(out), "logs", "train.jsonl")))
    # a round change ends the wait
    RUN.write_state(str(out), round=1)
    assert ds._await_ack(0, (256, 0, 0), timeout=10) is False
    assert WK.LEASE_ACK_S == 60.0


# --------------------------------------------------------------------------- P3-09: the fields pool dies too

def _pool_producer(conn, group):
    """A stand-in producer, SPAWNED like the real one: it builds the real `targets.field_pool`
    (forkserver context), proves a worker initialised and is alive, reports (worker, forkserver) and
    dies without shutting the pool down."""
    from rvsm import run as R, targets as TG
    if group:
        R.own_process_group()
    pool = TG.field_pool(1, owner=os.getpid())
    worker = pool.submit(os.getpid).result(timeout=60)
    import subprocess
    # something else the producer started, with no death watch of its own: only its group ties it
    other = subprocess.Popen(["sleep", "120"]).pid if group else None
    conn.send((worker, _ppid(worker), os.getpgid(0), other))
    time.sleep(0.5)
    os._exit(0)


def _ppid(pid):
    with open(f"/proc/{pid}/stat") as f:
        st = f.read()
    return int(st[st.rindex(")") + 2:].split()[1])


def _spawn_pool_producer(group):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    pr = ctx.Process(target=_pool_producer, args=(b, group))
    pr.start()
    assert a.poll(120), "the pool worker must acknowledge a completed initialisation"
    return (pr,) + tuple(a.recv())


def _gone(pids, within=15.0):
    t0 = time.time()
    while time.time() - t0 < within and any(_running(p) for p in pids):
        time.sleep(0.1)
    return not any(_running(p) for p in pids)


def test_the_real_field_pool_dies_with_its_producer():
    """P3-09 with the production context: the worker's parent is the pool's FORKSERVER, so PDEATHSIG
    followed the forkserver and both outlived a producer that exited (the review's probe). The owner
    watch in the worker's initializer ends it when the producer is gone; the forkserver then exits."""
    pr, worker, forkserver, _, _ = _spawn_pool_producer(group=False)
    try:
        assert forkserver != pr.pid and _running(worker) and _running(forkserver)
        pr.join(30)
        assert not pr.is_alive()
        assert _gone([worker, forkserver]), "the fields pool outlived its producer"
    finally:
        for p in (worker, forkserver):
            if _running(p):
                os.kill(p, 9)


def test_the_supervisor_kills_a_dead_producers_group_before_respawning(tmp_path):
    """P3-09, the supervisor's half: a producer that died on its own is replaced only after its whole
    process group -- forkserver and fields pool included -- has been killed (the already-dead branch
    cleaned nothing). The heartbeat records the group; the watch kills it, then respawns."""
    pr, worker, forkserver, pgid, other = _spawn_pool_producer(group=True)
    try:
        assert pgid == pr.pid and all(os.getpgid(p) == pgid for p in (worker, forkserver, other))
        pr.join(30)
        time.sleep(0.5)
        assert not pr.is_alive() and _running(other), "the orphan should outlive the producer"
        clock = [time.time()]
        out, procs, spawned, w = _watch(tmp_path, pr, clock)
        RUN._write_json(os.path.join(out, "workers", "produce.json"),
                        {"pid": pr.pid, "pgid": pgid, "last_ts": clock[0]})
        assert w.check() == "restart" and len(spawned) == 1
        kinds = [r.get("kind") for r in RUN.tail_jsonl(os.path.join(out, "logs", "sched.jsonl"))]
        assert "producer_group_killed" in kinds
        assert kinds.index("producer_group_killed") < kinds.index("restart"), kinds
        assert _gone([worker, forkserver, other], within=5.0)
    finally:
        for p in (worker, forkserver, other):
            if _running(p):
                os.kill(p, 9)


# --------------------------------------------------------------------------- P3-10: a stuck unit

def test_a_live_heartbeat_with_stale_progress_is_a_stall(tmp_path):
    """P3-10: the ticker keeps `last_ts` fresh even when the unit never progresses. A fresh heartbeat
    with `progress_ts` older than UNIT_STALL_S is reported as a STALL (once, with the unit's identity),
    a deliberate pause is exempt, and at twice the limit the producer is restarted like a silent one.
    The review's probe: fresh ticker, 24 h frozen unit -> the supervisor now acts."""
    clock = [1_000_000.0]
    out, procs, spawned, w = _watch(tmp_path, _FakeProc(), clock)
    hb = os.path.join(out, "workers", "produce.json")
    sched = os.path.join(out, "logs", "sched.jsonl")
    unit = {"phase": "round0", "job": "teacher", "region": [0, 1024, 0]}
    RUN._write_json(hb, {**unit, "last_ts": clock[0], "progress_ts": clock[0] - 60})
    assert w.check() is None
    RUN._write_json(hb, {**unit, "last_ts": clock[0], "progress_ts": clock[0] - RUN.UNIT_STALL_S - 60})
    assert w.check() == "stall" and w.check() == "stall" and not spawned
    stalls = [r for r in RUN.tail_jsonl(sched) if r.get("kind") == "STALL"]
    assert len(stalls) == 1 and stalls[0]["unit"]["job"] == "teacher" and \
        stalls[0]["unit"]["region"] == [0, 1024, 0]
    # a deliberate pause is not a stall
    RUN._write_json(hb, {"phase": "paused_ram", "last_ts": clock[0], "progress_ts": clock[0] - 86400})
    assert w.check() is None
    open(os.path.join(out, RUN.PAUSE_FILE), "w").close()
    RUN._write_json(hb, {**unit, "last_ts": clock[0], "progress_ts": clock[0] - 86400})
    assert w.check() is None
    os.remove(os.path.join(out, RUN.PAUSE_FILE))
    # 24 h: alert and restart
    first = procs["produce"]
    assert w.check() == "restart" and len(spawned) == 1
    assert "terminate" in first.calls
    rs = [r for r in RUN.tail_jsonl(sched) if r.get("kind") == "restart"]
    assert rs and rs[-1]["reason"] == "stalled"
    assert len([r for r in RUN.tail_jsonl(sched) if r.get("kind") == "STALL"]) == 2

    # the probe verbatim: an alive producer that cannot be stopped (no terminate) is not "healthy"
    class Alive:
        def is_alive(self):
            return True
    now = 100000.0
    out2 = str(tmp_path / "probe")
    RUN._write_json(os.path.join(out2, "workers", "produce.json"),
                    {"last_ts": now, "progress_ts": now - 86400, "phase": "round0", "job": "teacher"})
    watch = RUN.ProducerWatch(out2, {"produce": Alive()}, lambda: None, clock=lambda: now,
                              log=lambda m: None)
    assert watch.check() is not None


def _fake_store(path, **attrs):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"attributes": {"done": True, **attrs}}, f)


def test_the_verso_is_regenerated_once_as_a_new_generation(tmp_path, small_cfg):
    """The first evaluation with rung-2 dice >= verso_min_dice + verso_regen_gain marks the verso
    stores made by older checkpoints; the producer rewrites each as generation 1 BESIDE the old one
    (never in place), its fields follow at generation 1, every reader moves to it, and the trigger
    never fires twice (pass-3 item 11)."""
    from rvsm import regions as RG, stores, targets as TG
    out = str(tmp_path / "rg")
    cfg = replace(small_cfg, verso_min_dice=0.3, verso_regen_gain=0.15)
    lo = (0, 1024, 2048)
    _fake_store(stores.store_path(out, "recto", lo, 0), step=0)
    _fake_store(stores.store_path(out, "verso", lo, 0), step=10000)      # the producing ckpt step
    fields = [stores.store_path(out, TG.channel(kind, k), lo, 0) for kind in TG.KINDS for k in (2, 3, 4)]
    for f in fields:
        _fake_store(f)
    RUN.write_state(out, round=0, verso_on=True, verso_on_step=10000)
    cat = RG.Catalog(out, 0, ttl=0.0)
    assert RUN._next_job(cat, lo, 0, True, out, rungs=(2, 3, 4)) is None
    RUN.jlog(out, "eval", {"step": 20000, "dice_r2": 0.40}, echo=False)
    assert not RUN.maybe_regen_verso(cfg, out, 20000)                    # 0.40 < 0.45
    RUN.jlog(out, "eval", {"step": 22000, "dice_r2": 0.47}, echo=False)
    assert RUN.maybe_regen_verso(cfg, out, 22000)
    regen = RUN.read_state(out)["verso_regen"]
    assert regen == {"step": 22000, "dice_r2": 0.47}
    RUN.jlog(out, "eval", {"step": 24000, "dice_r2": 0.6}, echo=False)
    assert not RUN.maybe_regen_verso(cfg, out, 24000), "the regeneration fires once"
    assert RUN._next_job(cat, lo, 0, True, out, rungs=(2, 3, 4), regen=regen) == "verso"
    g1 = stores.gen_path(stores.store_path(out, "verso", lo, 0), 1)
    assert g1.endswith(".g1.zarr") and not os.path.exists(g1)
    _fake_store(g1, step=22000, gen=1)                                    # the producer's new store
    assert stores.is_done(stores.store_path(out, "verso", lo, 0)), "generation 0 is left as it was"
    assert stores.store_gen(out, "verso", lo, 0) == 1
    assert RG.Catalog(out, 0).path("verso", lo) == g1                   # every reader moves to it
    assert TG.field_path(out, "midline", 4, lo, 0).endswith(".g1.zarr")   # and its fields follow
    assert RUN._next_job(cat, lo, 0, True, out, rungs=(2, 3, 4), regen=regen) == "fields"
    for f in fields:
        _fake_store(stores.gen_path(f, 1))                               # the fields at generation 1
    assert RUN._next_job(cat, lo, 0, True, out, rungs=(2, 3, 4), regen=regen) is None
    assert stores.read_attrs(g1)["step"] == 22000
