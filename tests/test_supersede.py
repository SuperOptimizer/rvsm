"""Round >= 1 on a full disk (paris4 2026-09-28): band-limited student field stores, a round-r walk that
reads each region's newest round instead of waiting, and the supersession GC of the older round."""
import json
import os
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rvsm import config as CFG, ladder, regions as RG, run, sample, store_gc as SG, stores

OLD = time.time() - 7200


# ------------------------------------------------------------------ 1. band-limited field planes

def _sheet_planes(Z=40, Y=24, X=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    rec = torch.full((Z, Y, X), 0.1)
    rec[18:20] = 0.9                                   # a thin recto sheet at z 18-19
    rec[0:2, :, :4] = 0.0                              # a little CT air (rec == 0: never valid)
    ver = torch.full((Z, Y, X), 0.05)
    ver[16:18, :, 10:] = 0.8                           # the verso face beside it, on half the x range
    return {"recto": rec.half(), "verso": ver.half(),
            "midline": ((torch.rand((Z, Y, X), generator=g) - 0.5) * 80).half(),
            "thickness": (torch.rand((Z, Y, X), generator=g) * 30 + 1).half(),
            "conf": (torch.rand((Z, Y, X), generator=g) * 0.9 + 0.1).half()}


def test_field_band_is_the_dilated_sheet_and_torch_matches_numpy():
    pl = _sheet_planes()
    rec, ver = pl["recto"].float().numpy(), pl["verso"].float().numpy()
    for r in (1, 3, 6):
        b = run.field_band(rec, ver, r)
        bt = run.field_band_t(pl["recto"], pl["verso"], r).numpy()
        assert (b == bt).all(), r
        z = np.nonzero(b.any(axis=(1, 2)))[0]
        assert z.min() == 16 - r and z.max() == 19 + r      # Chebyshev radius r around z 16..19
        # the verso sheet only covers x >= 10: at z 16 - r the band stops at x = 10 - r
        assert not b[16 - r, :, :10 - r].any() and b[16 - r, :, 10 - r:].all()
    assert run.field_band(rec, ver, 0).all() and run.field_band(rec, ver, None).all()   # 0 = dense
    assert run.field_band_t(pl["recto"], pl["verso"], 0).all()


def test_student_rows_code_0_outside_the_band_numpy_and_chunked_torch_agree():
    pl = _sheet_planes()
    lay = SimpleNamespace(channels=("recto", "verso"))
    npl = {k: v.float().numpy() for k, v in pl.items()}
    ref = {ch: blk for ch, blk, _, _ in run.student_rows(npl, lay, "all", band=6)}
    dense = {ch: blk for ch, blk, _, _ in run.student_rows(npl, lay, "all")}
    sup = run.field_band(npl["recto"], npl["verso"], 6)
    valid = sup & (npl["recto"] > 0)
    for ch in ("midline", "thickness"):
        assert not ref[ch][~valid].any(), ch                  # code 0 = no data outside the band
        assert (ref[ch][valid] == dense[ch][valid]).all(), ch  # the same codes inside it
        assert (ref[ch][valid] > 0).all(), ch
    assert not ref["conf"][~sup].any() and (ref["conf"][sup] == dense["conf"][sup]).all()
    for ch in ("recto", "verso"):
        assert (ref[ch] == dense[ch]).all()                   # the probabilities are untouched
    for chunk in (5, 7, 64):                                  # halo'd z-chunks == the whole plane
        rows = run.student_rows_t(pl, lay, "all", chunk=chunk, band=6)
        assert [r[0] for r in rows] == ["recto", "verso", "midline", "thickness", "conf"]
        for ch, blk, q, _ in rows:
            assert (blk == ref[ch]).all(), (chunk, ch)
    # band None / 0: the old dense rows exactly
    for ch, blk, _, _ in run.student_rows_t(pl, lay, "all", chunk=7):
        assert (blk == dense[ch]).all(), ch


def _du(p):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)


def test_a_banded_q0_field_store_is_at_least_10x_smaller(tmp_path, has_volcomp):
    """One synthetic 256^3 region: a dense-noise midline / thickness plane (the worst case of a student
    head) with one thin sheet. Dense q0 store vs the banded one (`cfg.self_band_vox` = 6)."""
    if not has_volcomp:
        pytest.skip("no libvolcomp")
    n = 256
    rng = np.random.default_rng(0)
    rec = np.full((n, n, n), 0.1, np.float32)
    rec[128:130] = 0.95                                # a thin sheet: 2-voxel recto face ...
    ver = np.full((n, n, n), 0.1, np.float32)
    ver[126:128] = 0.9                                 # ... and verso face; the band is z 120..135
    pl = {"recto": rec, "verso": ver,
          "midline": rng.uniform(-30, 30, (n, n, n)).astype(np.float32),
          "thickness": rng.uniform(1, 60, (n, n, n)).astype(np.float32),
          "conf": rng.uniform(0.05, 1, (n, n, n)).astype(np.float32)}
    lay = SimpleNamespace(channels=("recto", "verso"))
    sizes = {}
    for tag, band in (("dense", None), ("band", CFG.Config().self_band_vox)):
        for ch, blk, q, enc in run.student_rows(pl, lay, "all", band=band):
            if ch in ("recto", "verso"):
                continue
            p = stores.write(str(tmp_path / tag / f"{ch}.zarr"), blk, (0, 0, 0), rung=2, channels=(ch,), q=q)
            sizes[(tag, ch)] = _du(p)
            back = np.asarray(stores.open_store(p)[:])
            assert (back == blk).all(), (tag, ch)             # q0 is lossless: code 0 stays 0
    for ch in ("midline", "thickness", "conf"):
        d, b = sizes[("dense", ch)], sizes[("band", ch)]
        print(f"{ch}: dense {d / 2**20:.2f} MiB, banded {b / 2**20:.2f} MiB, {d / b:.1f}x")
        assert d >= 10 * b, (ch, d, b)


def test_housekeeping_fields_are_not_in_the_fingerprint():
    base = CFG.Config().fingerprint()
    for kw in ({"reserve_gb": 10.0}, {"cache_gb": 8.0}, {"self_band_vox": 0}, {"self_band_vox": 12}):
        assert CFG.Config(**kw).fingerprint() == base, kw
    assert {"reserve_gb", "cache_gb", "self_band_vox"} <= set(CFG.FINGERPRINT_EXCLUDE)
    # a stored paris4 config predating `self_band_vox` still matches
    stored = json.loads(json.dumps(CFG.Config().to_json()))
    stored["config"].pop("self_band_vox")
    stored["config"]["reserve_gb"] = 50.0
    assert CFG.stored_fingerprint(stored) == replace(CFG.Config(), reserve_gb=20.0).fingerprint()


# ------------------------------------------------------------------ 2. a round-r walk never waits

def _round1_recto(root, lo, value=77):
    """A round-1 recto store over region `lo` (the fixture region shape): `value` where the round-0
    recto is set, so a reader can tell the rounds apart."""
    a = stores.open_store(stores.store_path(root, "recto", lo, 0))
    blk = np.where(np.asarray(a[:]) > 0, np.uint8(value), np.uint8(0))
    return stores.write(stores.store_path(root, "recto", lo, 1), blk, lo, rung=2, channels=("recto",), q=0)


def test_round_1_reads_round_0_stores_until_its_own_land(synth_run, monkeypatch):
    monkeypatch.setattr(sample, "AXIS_R_UM", 0.0)
    ds = sample.Patches(synth_run.cfg, root=synth_run.root, ct=synth_run.cfg.ct, ax=synth_run.ax,
                        region_records=synth_run.regions, round_=1)
    ds._open()
    assert isinstance(ds.cat, RG.RoundCatalog)
    from tests.test_sample_prep import _dense
    lo = _dense(ds, synth_run.lo)
    r = tuple(int(v) for v in ds._region_of(2, lo))
    # no round-1 store anywhere: every region is visitable, read from round 0
    assert all(ds._visitable(rec) for rec in synth_run.two)
    assert ds.cat.region_round(r) == 0
    ct = ladder.read_rung(ds.pyr, 2, lo, ds.patch, dtype=np.uint8)
    tg, w = ds._rung_target(2, lo, ct)
    ver, mid = ds.channels.index("verso"), ds.channels.index("midline")
    assert w[0].max() > 0 and w[ver].max() > 0 and w[mid].max() > 0
    assert tg[0][w[0] > 0].min() > 200                       # round 0's recto (q8: ~255)
    # the round-1 recto lands: after the catalog's TTL the region is read from round 1, WHOLE -- its
    # verso / midline (not written yet) carry no weight rather than round 0's
    _round1_recto(synth_run.root, r)
    assert ds.cat.region_round(r) == 0                         # a lower resolution is cached for TTL
    ds.cat = type(ds.cat)(ds.root, ds.round, ttl=0.0)
    assert ds.cat.region_round(r) == 1
    tg1, w1 = ds._rung_target(2, lo, ct)
    assert set(np.unique(tg1[0][w1[0] > 0])) == {77}           # round 1's recto preferred
    assert not w1[ver].any() and not w1[mid].any()
    # a region without a round-1 store keeps reading round 0
    other = next(x for x in synth_run.lo if tuple(x) != r)
    assert ds.cat.region_round(other) == 0 and ds.cat.done("verso", other)


def test_a_round_1_walk_with_no_round_1_store_draws_without_waiting(synth_run):
    from rvsm import walk
    out = synth_run.root
    ds = walk.WalkPatches(synth_run.cfg, out, root=out, ct=synth_run.cfg.ct, ax=synth_run.ax,
                          region_records=synth_run.regions, round_=1, lookahead_n=2, wait_s=0.01)
    it = iter(ds)
    t0 = time.time()
    items = [next(it) for _ in range(3)]
    assert len(items) == 3 and time.time() - t0 < 120
    log = os.path.join(out, "logs", "train.jsonl")
    waits = [ln for ln in (open(log).read().splitlines() if os.path.exists(log) else [])
             if json.loads(ln).get("kind") == "wait"]
    assert not waits, waits


# ------------------------------------------------------------------ 3. supersession GC

def _store(out, ch, lo, r, g=0, nbytes=4096, old=True):
    p = stores.gen_path(stores.store_path(str(out), ch, lo, r), g)
    os.makedirs(os.path.join(p, "c", "0", "0"), exist_ok=True)
    with open(os.path.join(p, "zarr.json"), "w") as f:
        json.dump({"attributes": {"done": True}}, f)
    with open(os.path.join(p, "c", "0", "0", "0"), "wb") as f:
        f.write(b"\1" * nbytes)
    if old:
        for d, dirs, files in os.walk(p):
            for n in files + dirs:
                os.utime(os.path.join(d, n), (OLD, OLD))
        os.utime(p, (OLD, OLD))
    return p


A, B, H = (0, 0, 0), (0, 0, 1024), (0, 1024, 0)


def _rounds_fixture(out):
    """Round 0: A, B and H (held out) with recto (+ g1), rw, band, verso, midline, midline_r3. Round 1:
    A complete (recto, verso, midline, thickness, conf, midline_r3); H complete; B none yet."""
    for lo in (A, B, H):
        for ch in ("recto", "rw", "band", "verso", "midline", "midline_r3"):
            _store(out, ch, lo, 0)
        _store(out, "recto", lo, 0, g=1)
        stores.commit_bundle(str(out), lo, 0, 0, verso=0, recto=1)
    for lo in (A, H):
        for ch in ("recto", "verso", "midline", "thickness", "conf", "midline_r3"):
            _store(out, ch, lo, 1)
    os.makedirs(os.path.join(out, "eval"), exist_ok=True)
    with open(os.path.join(out, "eval", "heldout.json"), "w") as f:
        json.dump({"regions": [{"lo": list(H), "size": [1024] * 3, "k": 2}]}, f)


def test_superseded_by_round_lists_the_covered_channels_only(tmp_path):
    _rounds_fixture(tmp_path)
    got = sorted((c["channel"], c["gen"]) for c in SG.superseded_by_round(tmp_path, A, 1))
    # every round-0 channel round 1 has, all generations, plus rw / band beside the recto
    assert got == [("band", 0), ("midline", 0), ("midline_r3", 0), ("recto", 0), ("recto", 1),
                   ("rw", 0), ("verso", 0)]
    assert SG.superseded_by_round(tmp_path, B, 1) == []       # no round-1 recto: readers still on 0
    assert SG.superseded_by_round(tmp_path, A, 0) == []


def test_supersede_due_deletes_after_the_grace_logs_and_pins_held_out(tmp_path):
    _rounds_fixture(tmp_path)
    out = str(tmp_path)
    q = {A: (time.time() + 1000, 1), H: (time.time() - 1, 1)}
    assert run.supersede_due(out, q, held=[H]) == []           # H is held out: never; A not due
    assert A in q and H not in q
    assert stores.is_done(stores.store_path(out, "recto", H, 0))
    q[A] = (time.time() - 1, 1)
    got = run.supersede_due(out, q, held=[H])
    assert len(got) == 1 and got[0]["kind"] == "supersede_gc" and got[0]["region"] == list(A)
    assert got[0]["freed_gb"] >= 0 and got[0]["stores"] == 7 and not q
    for ch in ("recto", "rw", "band", "verso", "midline", "midline_r3"):
        assert not os.path.exists(stores.store_path(out, ch, A, 0)), ch
        assert os.path.exists(stores.store_path(out, ch, B, 0)), ch   # B has no round 1 yet
    assert not os.path.exists(stores.gen_path(stores.store_path(out, "recto", A, 0), 1))
    assert stores.is_done(stores.store_path(out, "recto", A, 1))
    lines = [json.loads(ln) for ln in open(os.path.join(out, "logs", "produce.jsonl"))]
    assert [ln["region"] for ln in lines if ln.get("kind") == "supersede_gc"] == [list(A)]
    # a reader of round 1 still resolves A (round 1) and B (round 0)
    cat = RG.RoundCatalog(out, 1, ttl=0.0)
    assert cat.region_round(A) == 1 and cat.region_round(B) == 0 and cat.done("recto", B)
    assert not [n for _d, ds_, fs in os.walk(out) for n in fs + ds_ if SG.GC_TAG in n]


def test_store_gc_superseded_by_round_cli(tmp_path, capsys):
    _rounds_fixture(tmp_path)
    out = str(tmp_path)
    res = SG.scan_superseded(out, 1, min_age_s=60)
    assert sorted({tuple(c["region"]) for c in res["candidates"]}) == [A]
    assert {s["why"] for s in res["skipped"]} == {"held out"}
    # a fresh round-1 recto is "superseded recently"
    os.utime(os.path.join(stores.store_path(out, "recto", A, 1), "zarr.json"))
    assert {s["why"] for s in SG.scan_superseded(out, 1, min_age_s=60)["skipped"]} == \
        {"held out", "superseded recently"}
    os.utime(os.path.join(stores.store_path(out, "recto", A, 1), "zarr.json"), (OLD, OLD))
    assert SG.main(["--out", out, "--superseded-by-round", "1"]) == 0      # dry run
    assert os.path.exists(stores.store_path(out, "rw", A, 0))
    assert "dry run" in capsys.readouterr().out
    assert SG.main(["--out", out, "--superseded-by-round", "1", "--delete",
                    "--log", os.path.join(out, "gc.json")]) == 0
    assert not os.path.exists(stores.store_path(out, "rw", A, 0))
    assert os.path.exists(stores.store_path(out, "rw", H, 0)) and os.path.exists(stores.store_path(out, "rw", B, 0))
    rec = json.load(open(os.path.join(out, "gc.json")))
    assert len(rec["removed"]) == 7 and rec["bytes"] > 0
