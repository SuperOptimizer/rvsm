"""`rvsm store-gc`: which store generations are superseded, and that deleting them is safe."""
import json
import os
import time

from rvsm import regions as RG, store_gc as SG, stores

OLD = time.time() - 7200


def _store(out, ch, lo, g, done=True, old=True, nbytes=1000):
    p = stores.gen_path(stores.store_path(str(out), ch, lo, 0), g)
    os.makedirs(os.path.join(p, "c", "0", "0"), exist_ok=True)
    with open(os.path.join(p, "zarr.json"), "w") as f:
        json.dump({"attributes": {"done": bool(done)}}, f)
    with open(os.path.join(p, "c", "0", "0", "0"), "wb") as f:
        f.write(b"\0" * nbytes)
    if old:
        for d, dirs, files in os.walk(p):
            for n in files + dirs:
                os.utime(os.path.join(d, n), (OLD, OLD))
        os.utime(p, (OLD, OLD))
    return p


def _fixture(out, bundle_old=True):
    """Region A: recto + rw + band at generations 0, 1, 2 (recto committed at 1, 2 in progress),
    verso 0, 1 (committed 1), midline + midline_r3 at 0, 1, 2 (fields committed at 2); `conf` has one
    generation. Region B: no bundle file (everything is generation 0, committed)."""
    A, B = (0, 1024, 2048), (1024, 0, 0)
    for ch in ("recto", "rw", "band"):
        for g in (0, 1, 2):
            _store(out, ch, A, g)
    for g in (0, 1):
        _store(out, "verso", A, g)
    for ch in ("midline", "midline_r3"):
        for g in (0, 1, 2):
            _store(out, ch, A, g)
    _store(out, "conf", A, 0)
    for ch in ("recto", "rw", "verso"):
        _store(out, ch, B, 0)
    bf = stores.commit_bundle(str(out), A, 0, 2, verso=1, recto=1)
    if bundle_old:
        os.utime(bf, (OLD, OLD))
    return A, B


def _got(res):
    return sorted((c["channel"], c["gen"]) for c in res["candidates"])


def test_superseded_generations_only(tmp_path):
    A, B = _fixture(tmp_path)
    res = SG.scan(tmp_path)
    # without --gen0: only the g >= 1 generations older than the committed one
    assert _got(res) == [("midline", 1), ("midline_r3", 1)]
    gen0 = sorted((s["channel"], s["gen"]) for s in res["skipped"] if s["why"] == "gen0")
    assert gen0 == [("band", 0), ("midline", 0), ("midline_r3", 0), ("recto", 0), ("rw", 0), ("verso", 0)]
    res0 = SG.scan(tmp_path, gen0=True)
    assert _got(res0) == sorted(gen0 + [("midline", 1), ("midline_r3", 1)])
    # never: a committed store, an in-progress one, a single-generation channel, a bundle-less region
    paths = {c["path"] for c in res0["candidates"]}
    for ch in ("recto", "rw", "band", "verso", "midline", "midline_r3", "conf"):
        assert stores.current_path(str(tmp_path), ch, A, 0) not in paths
    assert stores.gen_path(stores.store_path(str(tmp_path), "recto", A, 0), 2) not in paths
    assert not any(tuple(c["region"]) == B for c in res0["candidates"])
    per = SG.summary(res0)
    assert per["round_0/recto"]["n"] == 1 and per["round_0/recto"]["bytes"] > 1000


def test_guards_not_done_recent_and_fresh_commit(tmp_path):
    A, _B = _fixture(tmp_path)
    p1 = stores.gen_path(stores.store_path(str(tmp_path), "midline", A, 0), 1)
    with open(os.path.join(p1, "zarr.json"), "w") as f:
        json.dump({"attributes": {"done": False}}, f)
    os.utime(os.path.join(p1, "zarr.json"), (OLD, OLD))
    p3 = stores.gen_path(stores.store_path(str(tmp_path), "midline_r3", A, 0), 1)
    os.utime(os.path.join(p3, "c", "0", "0", "0"), None)          # written just now
    res = SG.scan(tmp_path)
    assert res["candidates"] == []
    why = {(s["channel"], s["gen"]): s["why"] for s in res["skipped"]}
    assert why[("midline", 1)] == "not done" and why[("midline_r3", 1)] == "modified recently"
    # a bundle committed within the window protects the whole region: a read may be in flight
    t2 = tmp_path / "b"
    _fixture(t2, bundle_old=False)
    res2 = SG.scan(t2, gen0=True)
    assert res2["candidates"] == [] and {s["why"] for s in res2["skipped"]} == {"bundle committed recently"}


def test_delete_renames_then_removes_logs_and_keeps_readers_working(tmp_path):
    A, B = _fixture(tmp_path)
    before = {ch: RG.Catalog(str(tmp_path), 0, ttl=0.0).list_done(ch)
              for ch in ("recto", "rw", "verso", "midline", "band")}
    res = SG.scan(tmp_path, gen0=True)
    # a leftover of an interrupted run is swept too
    left = stores.store_path(str(tmp_path), "recto", A, 0) + ".gc-1"
    os.makedirs(left)
    res = SG.scan(tmp_path, gen0=True)
    assert res["leftover"] == [left]
    logp = str(tmp_path / "logs" / "gc.json")
    rec = SG.delete(res, logp, log=lambda m: None)
    assert len(rec["removed"]) == len(res["candidates"]) == 8 and not rec["failed"]
    assert not any(os.path.exists(c["path"]) for c in res["candidates"]) and not os.path.exists(left)
    assert not any(".gc-" in n for d, _ds, fs in os.walk(tmp_path / "stores") for n in _ds + fs)
    assert json.load(open(logp))["bytes"] == sum(c["bytes"] for c in res["candidates"])
    # every committed store is still there and every reader still sees every region
    for ch in ("recto", "rw", "verso", "midline", "midline_r3"):
        assert stores.is_done(stores.current_path(str(tmp_path), ch, A, 0))
    after = {ch: RG.Catalog(str(tmp_path), 0, ttl=0.0).list_done(ch) for ch in before}
    assert after["recto"] == before["recto"] == sorted([A, B])
    assert after["verso"] == before["verso"] and after["midline"] == before["midline"]
    # the in-progress generation and the generation counter are untouched
    assert stores.gens(stores.store_path(str(tmp_path), "recto", A, 0)) == [1, 2]
    assert stores.next_gen(str(tmp_path), A, 0) == 3
    # and a second pass finds nothing
    assert SG.scan(tmp_path, gen0=True)["candidates"] == []


def test_list_done_parses_generation_names(tmp_path):
    lo = (0, 0, 1024)
    assert stores.parse_region_name("region_0_0_1024.zarr") == (lo, 0)
    assert stores.parse_region_name("region_0_0_1024.g3.zarr") == (lo, 3)
    for n in ("region_0_0_1024.zarr.tmp", "region_0_0_1024.g1.zarr.gc-5", "region_0_0_1024.json",
              "region_0_0.zarr", "region_0_0_1024.gx.zarr", "coarse.zarr"):
        assert stores.parse_region_name(n) is None
    # a region whose generation 0 is gone is listed through its committed generation
    _store(tmp_path, "recto", lo, 1)
    cat = RG.Catalog(str(tmp_path), 0, ttl=0.0)
    assert cat.list_done("recto") == []                 # not committed: bundle says 0, which is absent
    stores.commit_bundle(str(tmp_path), lo, 0, 0, verso=0, recto=1)
    assert cat.list_done("recto") == [lo]


def test_cli_dry_run_deletes_nothing(tmp_path, capsys):
    from rvsm import cli
    _fixture(tmp_path)
    n0 = sum(len(fs) for _d, _ds, fs in os.walk(tmp_path))
    assert cli.main(["store-gc", "--out", str(tmp_path)]) == 0
    assert sum(len(fs) for _d, _ds, fs in os.walk(tmp_path)) == n0
    out = capsys.readouterr().out
    assert "round_0/midline" in out and "dry run" in out
