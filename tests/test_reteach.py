"""The m7-only mode and the recto REGENERATION (2026-09-25, paris4 from ~step 56000).

A run whose `teacher_ckpts` names one teacher writes that teacher's probability as `recto` and rw = 1;
every round-0 recto another teacher set made (a store without a `teachers` attr is the old recto + m7
fusion) is redone as the NEXT generation beside the old one, and readers move to it only once it --
and, where the region has a verso, the fields rebuilt from it -- are committed.
"""
import json
import os
from dataclasses import replace

import numpy as np
import pytest

from rvsm import config as CFG, regions as RG, run as RUN, sample, stores, targets as TG


def _fake_store(path, **attrs):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"attributes": {"done": True, **attrs}}, f)


def _fields(out, lo, gen):
    for kind in TG.KINDS:
        for k in (2, 3, 4):
            _fake_store(stores.gen_path(stores.store_path(out, TG.channel(kind, k), lo, 0), gen))


# ------------------------------------------------------------------------------ the teacher set

def test_the_teacher_set_is_the_keys_of_teacher_ckpts(small_cfg):
    assert RUN.teacher_names(small_cfg) == ["recto", "m7"]
    assert RUN.teacher_names(replace(small_cfg, teacher_ckpts={"m7": "/w/m7.pth"})) == ["m7"]
    bank = RUN.TeacherBank(replace(small_cfg, teacher_ckpts={"m7": "/w/m7.pth"}), small_cfg.out)
    assert [r[0] for r in bank.items] == ["m7"]            # no recto weights are ever loaded
    assert bank.footprint() == 0


def test_a_single_teacher_pass_writes_its_probability_and_rw_ones(region_cfg, fake_teacher):
    cfg = replace(region_cfg, teacher_ckpts={"fake": fake_teacher.ckpt})
    bank = RUN.TeacherBank(cfg, cfg.out, device="cpu")
    P, W, attrs = bank.probs_u8(cfg.ct, (0, 0, 0), (128, 128, 128))
    P, W = P.cpu().numpy(), W.cpu().numpy()
    assert P.dtype == np.uint8 and P.shape == (128, 128, 128) and P.any()
    assert (W == 255).all()
    assert attrs["teachers"] == ["fake"] and attrs["producer"] == "teacher:fake" and attrs["rw"] == "ones"
    assert list(attrs["ckpt"]) == ["fake"]


# ------------------------------------------------------------------------------ the state machine

def test_a_fused_recto_is_regenerated_as_the_next_generation(tmp_path):
    out, lo = str(tmp_path / "rg"), (0, 1024, 2048)
    _fake_store(stores.store_path(out, "recto", lo, 0), producer="teacher:recto,m7")   # no `teachers`
    _fake_store(stores.store_path(out, "rw", lo, 0))
    _fake_store(stores.store_path(out, "verso", lo, 0), step=10000)
    _fields(out, lo, 0)
    cat = RG.Catalog(out, 0, ttl=0.0)
    # the old set, or no set, asks for nothing; a new set asks for a reteach (not blocking)
    assert RUN._next_job(cat, lo, 0, True, out) is None
    assert RUN._next_job(cat, lo, 0, True, out, teachers=["recto", "m7"]) is None
    assert RUN._next_job(cat, lo, 0, True, out, teachers=["m7", "recto"]) is None
    assert RUN._next_job(cat, lo, 0, True, out, teachers=["m7"]) == "reteach"
    assert RUN.recto_stale(out, lo, ["m7"]) and not RUN.recto_stale(out, lo, ["recto", "m7"])
    assert stores.next_gen(out, lo, 0) == 1

    r1 = stores.gen_path(stores.store_path(out, "recto", lo, 0), 1)
    w1 = stores.gen_path(stores.store_path(out, "rw", lo, 0), 1)
    _fake_store(r1, teachers=["m7"], gen=1, regeneration=True)
    # the recto alone (the unit was cut before its rw): still to do
    assert RUN._next_job(cat, lo, 0, True, out, teachers=["m7"]) == "reteach"
    _fake_store(w1, teachers=["m7"], gen=1)
    # the old pair stays in place and readable; the fields are rebuilt at generation 1 first
    assert stores.is_done(stores.store_path(out, "recto", lo, 0))
    assert TG.field_path(out, "midline", 2, lo, 0).endswith(".g1.zarr")
    assert RUN._next_job(cat, lo, 0, True, out, teachers=["m7"]) == "fields"
    assert RUN.commit_sources(out, lo, 0) is None, "no commit before the fields are current"
    assert cat.path("recto", lo) == stores.store_path(out, "recto", lo, 0)
    assert cat.path("rw", lo) == stores.store_path(out, "rw", lo, 0)
    assert RUN.recto_stale(out, lo, ["m7"])
    _fields(out, lo, 1)
    assert RUN._next_job(cat, lo, 0, True, out, teachers=["m7"]) is None
    assert RUN.commit_sources(out, lo, 0) == {"gen": 1, "verso": 0, "recto": 1}
    assert RUN.commit_sources(out, lo, 0) is None                       # idempotent
    c2 = RG.Catalog(out, 0, ttl=0.0)
    assert c2.path("recto", lo) == r1 and c2.path("rw", lo) == w1       # readers move to the new pair
    assert c2.path("verso", lo) == stores.store_path(out, "verso", lo, 0)   # ... the verso stays
    assert c2.path("midline", lo).endswith(".g1.zarr")                   # ... with fields built from both
    assert not RUN.recto_stale(out, lo, ["m7"])
    assert stores.is_done(stores.store_path(out, "recto", lo, 0)), "generation 0 is never touched"

    # a later verso regeneration takes the NEXT free generation of the region, never the recto's
    regen = {"step": 20000, "ckpt": "x"}
    assert RUN._next_job(c2, lo, 0, True, out, regen=regen, teachers=["m7"]) == "verso"
    assert stores.next_gen(out, lo, 0) == 2
    v2 = stores.gen_path(stores.store_path(out, "verso", lo, 0), 2)
    _fake_store(v2, step=20000, gen=2)
    assert stores.store_gen(out, "verso", lo, 0) == 2                   # a gap at g1 is fine
    assert TG.field_path(out, "thickness", 3, lo, 0).endswith(".g2.zarr")
    assert RUN._next_job(c2, lo, 0, True, out, regen=regen, teachers=["m7"]) == "fields"
    _fields(out, lo, 2)
    assert RUN.commit_sources(out, lo, 0) == {"gen": 2, "verso": 2, "recto": 1}
    c3 = RG.Catalog(out, 0, ttl=0.0)
    assert c3.path("verso", lo) == v2 and c3.path("recto", lo) == r1
    # the verso backlog reads the committed VERSO generation, not the fields'
    os.makedirs(os.path.dirname(RUN.backlog_path(out)), exist_ok=True)
    RUN.write_json_atomic(RUN.backlog_path(out), {"step": 20000, "regions": [list(lo)]})
    assert RUN.regen_remaining(out) == []


def test_a_region_without_a_verso_commits_its_new_recto_at_once(tmp_path):
    out, lo = str(tmp_path / "nv"), (0, 0, 0)
    _fake_store(stores.store_path(out, "recto", lo, 0))
    _fake_store(stores.store_path(out, "rw", lo, 0))
    cat = RG.Catalog(out, 0, ttl=0.0)
    assert RUN._next_job(cat, lo, 0, False, out, teachers=["m7"]) == "reteach"
    _fake_store(stores.gen_path(stores.store_path(out, "recto", lo, 0), 1), teachers=["m7"])
    _fake_store(stores.gen_path(stores.store_path(out, "rw", lo, 0), 1), teachers=["m7"])
    assert RUN._next_job(cat, lo, 0, False, out, teachers=["m7"]) is None
    assert RUN.commit_sources(out, lo, 0) == {"gen": 0, "verso": 0, "recto": 1}
    assert RG.Catalog(out, 0, ttl=0.0).path("recto", lo).endswith(".g1.zarr")
    # its first verso comes later, at generation 0: the fields go to generation 1 (the recto's) and
    # are committed with it
    _fake_store(stores.store_path(out, "verso", lo, 0), step=1)
    cat = RG.Catalog(out, 0, ttl=0.0)
    assert RUN._next_job(cat, lo, 0, True, out, teachers=["m7"]) == "fields"
    _fields(out, lo, 1)
    assert RUN.commit_sources(out, lo, 0) == {"gen": 1, "verso": 0, "recto": 1}


def test_the_recto_backlog_goes_held_out_first_then_walk_order(tmp_path):
    out = str(tmp_path / "bl")
    los = [(0, 0, 0), (0, 0, 128), (0, 128, 0), (128, 0, 0)]
    for lo in los:
        _fake_store(stores.store_path(out, "recto", lo, 0))
        _fake_store(stores.store_path(out, "rw", lo, 0))
    _fake_store(stores.store_path(out, "recto", (128, 128, 128), 0), teachers=["m7"])   # already m7
    route = [los[2], los[0], los[3], los[1]]
    held = [{"lo": list(los[1])}]
    assert RUN.recto_regen_todo(out, route, held, ["m7"]) == [los[1], los[2], los[0], los[3]]
    # the old set sees only the m7 store as foreign (an un-attributed store is the old fusion)
    assert RUN.recto_regen_todo(out, route, held, ["recto", "m7"]) == [(128, 128, 128)]
    os.remove(os.path.join(stores.store_path(out, "recto", (128, 128, 128), 0), "zarr.json"))

    import threading
    rr, busy, gpu, keys, fsub = {}, {los[1]}, [], set(), []
    cat = RG.Catalog(out, 0, ttl=0.0)
    n = RUN._recto_backlog(out, rr, route, held, ["m7"], cat, True, (2, 3, 4), busy, threading.Lock(),
                           {}, lambda lo: True, gpu, keys, fsub.append, now=1000.0)
    assert n == 4 and gpu == [(los[2], "reteach")] and keys == {los[2]}   # the busy one is skipped
    lines = [r for r in RUN.tail_jsonl(os.path.join(out, "logs", "produce.jsonl"), 20)
             if r.get("kind") == "recto_regen"]
    assert lines and lines[-1]["remaining"] == 4 and lines[-1]["heldout_remaining"] == 1
    for lo in los:                                   # every region regenerated and committed
        for ch in ("recto", "rw"):
            _fake_store(stores.gen_path(stores.store_path(out, ch, lo, 0), 1), teachers=["m7"])
    gpu.clear()
    busy.clear()
    n = RUN._recto_backlog(out, rr, route, held, ["m7"], cat, True, (2, 3, 4), busy, threading.Lock(),
                           {}, lambda lo: True, gpu, keys, fsub.append, now=1001.0)
    assert n == 0 and gpu == [] and all(not RUN.recto_stale(out, lo, ["m7"]) for lo in los)
    RUN._recto_backlog(out, rr, route, held, ["m7"], cat, True, (2, 3, 4), busy, threading.Lock(),
                       {}, lambda lo: True, gpu, keys, fsub.append, now=1000.0 + RUN.RECTO_TODO_S)
    kinds = [r.get("kind") for r in RUN.tail_jsonl(os.path.join(out, "logs", "produce.jsonl"), 20)]
    assert kinds.count("recto_regen_done") == 1


def test_a_reteach_is_not_blocking_and_held_out_ones_go_first():
    a, b, c, h = (0, 0, 0), (0, 0, 1), (0, 0, 2), (9, 9, 9)
    units = [(a, "reteach"), (b, "verso"), (c, "teacher"), (h, "reteach")]
    assert RUN._gpu_order(units, held=[h]) == [(c, "teacher"), (h, "reteach"), (a, "reteach"), (b, "verso")]


# ------------------------------------------------------------------------------ readers

def test_the_loader_and_the_eval_grid_follow_the_committed_recto(synth_run, tmp_path):
    root = synth_run.root
    held = [r for r in synth_run.regions if r["k"] == 2][:1]
    lo = tuple(int(v) for v in held[0]["lo"])
    kw = dict(root=root, ct=synth_run.cfg.ct, ax=synth_run.ax, rungs=(2, 3), limit=2)
    d = tmp_path / "grid"
    g0 = sample.val_grid(synth_run.cfg, held, spill=str(d), **kw)
    ds = sample.Patches(synth_run.cfg, root=root, ct=synth_run.cfg.ct, ax=synth_run.ax,
                        region_records=synth_run.regions)
    ds._open()
    src0 = sample.grid_sources(ds, held)
    old = np.asarray(stores.open_store(stores.store_path(root, "recto", lo, 0))[:])
    new = np.where(old > 0, np.uint8(128), np.uint8(0))
    for ch, blk in (("recto", new), ("rw", np.full(new.shape, 200, np.uint8))):
        stores.write(stores.gen_path(stores.store_path(root, ch, lo, 0), 1), blk, lo, rung=2,
                     channels=(ch,), q=8, attrs={"teachers": ["m7"], "gen": 1})
    # written but not committed: every reader, and the grid, still sees generation 0
    assert sample.grid_sources(ds, held) == src0
    v, _ = ds._source("recto", 2, lo, (32, 32, 32))
    assert np.array_equal(v, old[:32, :32, :32])
    stores.commit_bundle(root, lo, 0, 0, verso=0, recto=1)
    RG.clear_pool()
    ds2 = sample.Patches(synth_run.cfg, root=root, ct=synth_run.cfg.ct, ax=synth_run.ax,
                         region_records=synth_run.regions)
    ds2._open()
    assert sample.grid_sources(ds2, held) != src0
    v, f = ds2._source("recto", 2, lo, (32, 32, 32))
    assert np.array_equal(v, new[:32, :32, :32]) and (f > 0).all()
    r, _ = ds2._source("rw", 2, lo, (32, 32, 32))
    assert (r == 200).all()
    g1 = sample.val_grid(synth_run.cfg, held, spill=str(d), **kw)
    assert g1.rebuilt == len(g1) and g1.reused == 0               # the region's items were rebuilt
    assert len(g0) == len(g1)


def test_a_reteach_refeeds_the_coarse_rungs_once_per_generation(tmp_path, has_volcomp):
    if not has_volcomp:
        pytest.skip("no libvolcomp on this host")
    out, lo = str(tmp_path / "cf"), (0, 0, 0)
    shape2 = (512, 512, 512)
    a = np.full((128, 128, 128), 40, np.uint8)
    assert RUN.feed_coarse_once(out, lo, 0, a, shape2)
    assert RUN.feed_coarse_once(out, lo, 0, a, shape2) == []            # once per generation
    b = np.full((128, 128, 128), 200, np.uint8)
    assert RUN.feed_coarse_once(out, lo, 0, b, shape2, gen=1)
    assert RUN.feed_coarse_once(out, lo, 0, b, shape2, gen=1) == []
    v, cov = RG.read_coarse(out, "recto", 7, (0, 0, 0), (4, 4, 4))
    assert v[0, 0, 0] == 200 and cov[0, 0, 0] == 1.0


# ------------------------------------------------------------------------------ config / resume

def test_the_teacher_set_and_the_continuity_weights_are_not_in_the_fingerprint(small_cfg):
    base = small_cfg.fingerprint()
    assert replace(small_cfg, teacher_ckpts={"m7": "/w/m7.pth"}).fingerprint() == base
    assert replace(small_cfg, teacher_ckpts={"recto": "a", "m7": "b"}).fingerprint() == base
    for k in ("loss_skel", "loss_affinity", "loss_ect", "loss_selfcons"):
        assert replace(small_cfg, **{k: 0.5}).fingerprint() == base, k
    assert replace(small_cfg, loss_excl=0.5).fingerprint() != base      # not every loss is free
    # the defaults did not move
    assert (CFG.Config().loss_skel, CFG.Config().loss_affinity, CFG.Config().loss_ect,
            CFG.Config().loss_selfcons) == (0.05, 0.1, 0.05, 0.1)


def test_a_resume_logs_the_teacher_and_the_loss_switch(tmp_path, small_cfg):
    out = str(tmp_path / "sw")
    os.makedirs(out)
    RUN.write_state(out, step=56000)
    old = replace(small_cfg, teacher_ckpts={"recto": "/w/recto.pth", "m7": "/w/m7.pth"}).to_json()
    new = replace(small_cfg, teacher_ckpts={"m7": "/w/m7.pth"}, loss_skel=0.1)
    got = RUN.log_switches(out, old, new)
    kinds = {r["kind"]: r for r in got}
    t = kinds["teacher_switch"]
    assert t["old"] == ["recto", "m7"] and t["new"] == ["m7"] and t["regenerate"] and t["step"] == 56000
    assert t["new_ckpts"] == {"m7": "/w/m7.pth"}
    assert kinds["loss_switch"]["weights"] == {"loss_skel": {"old": 0.05, "new": 0.1}}
    sched = RUN.tail_jsonl(os.path.join(out, "logs", "sched.jsonl"), 10)
    assert {r["kind"] for r in sched} == {"teacher_switch", "loss_switch"}
    assert RUN.log_switches(out, new.to_json(), new) == []              # nothing moved: nothing logged
    # an old config without `teacher_ckpts` keys was the default recto + m7 set
    o2 = replace(small_cfg).to_json()
    got = RUN.log_switches(out, o2, replace(small_cfg, teacher_ckpts={"m7": "p"}))
    assert got[0]["old"] == ["recto", "m7"] and got[0]["new"] == ["m7"]


# ------------------------------------------------------------------------------ the producer, for real

def _produce_until(cfg, cond, limit=600.0):
    import threading
    import time
    stop = threading.Event()
    box = {}

    def go():
        try:
            RUN.produce_loop(cfg, cfg.out, device="cpu", stop=stop, max_s=limit)
        except BaseException as e:  # noqa: BLE001
            box["err"] = e
    th = threading.Thread(target=go, daemon=True)
    th.start()
    t0 = time.time()
    while time.time() - t0 < limit and th.is_alive() and not cond():
        time.sleep(0.5)
    ok = cond()
    stop.set()
    th.join(120)
    assert "err" not in box, box.get("err")
    return ok


def test_the_producer_switches_a_running_round_0_to_one_teacher(region_cfg, fake_teacher, has_volcomp):
    """The mid-run switch end to end on the producer: a round 0 made by one teacher set, a resume with
    another (the `teacher_switch` line), and every produced recto redone as generation 1 by the new set
    alone and committed -- the held-out reference first -- while generation 0 stays on disk."""
    if not has_volcomp:
        pytest.skip("a region store is a volcomp array; no libvolcomp on this host")
    import dataclasses

    from rvsm import teachers
    cfg1 = replace(region_cfg, teacher_ckpts={"fake": fake_teacher.ckpt}, heldout=1,
                   lookahead_extra=1, reserve_gb=0.001, compile=False)
    ctx = RUN.setup(cfg1, cfg1.out)
    out = cfg1.out

    def done_recto():
        return RG.Catalog(out, 0, ttl=0.0).list_done("recto")
    assert _produce_until(cfg1, lambda: len(done_recto()) >= 2)
    first = done_recto()
    assert all(stores.read_attrs(stores.store_path(out, "recto", lo, 0))["teachers"] == ["fake"]
               for lo in first)

    teachers.register("fake2", dataclasses.replace(fake_teacher.spec, name="fake2"))
    try:
        cfg2 = replace(cfg1, teacher_ckpts={"fake2": fake_teacher.ckpt})
        RUN.setup(cfg2, out)                                  # the resume: same fingerprint
        sw = [r for r in RUN.tail_jsonl(os.path.join(out, "logs", "sched.jsonl"), 50)
              if r.get("kind") == "teacher_switch"]
        assert sw and sw[-1]["old"] == ["fake"] and sw[-1]["new"] == ["fake2"]
        assert _produce_until(cfg2, lambda: not any(RUN.recto_stale(out, lo, ["fake2"]) for lo in first))
    finally:
        teachers.TEACHERS.pop("fake2", None)
    held = tuple(int(v) for v in ctx["heldout"][0]["lo"])
    cat = RG.Catalog(out, 0, ttl=0.0)
    for lo in first:
        g0 = stores.store_path(out, "recto", lo, 0)
        assert stores.is_done(g0) and stores.read_attrs(g0)["teachers"] == ["fake"]   # kept
        p = cat.path("recto", lo)
        assert p != g0 and stores.read_attrs(p)["teachers"] == ["fake2"]
        assert stores.read_attrs(p)["regeneration"] and stores.read_attrs(p)["replaces_teachers"] == ["fake"]
        assert stores.read_attrs(cat.path("rw", lo))["teachers"] == ["fake2"]
    log = RUN.tail_jsonl(os.path.join(out, "logs", "produce.jsonl"), 5000)
    re = [tuple(r["region"]) for r in log if r.get("kind") == "reteach"]
    assert set(first) <= set(re)
    if held in first:
        assert re[0] == held, "the held-out reference is regenerated first"
