"""The CT shard cache: what a read needs, what a 404 leaves behind, and what eviction may take."""
import os

import numpy as np
import pytest

from rvsm import ladder, stream


@pytest.fixture
def windowed(monkeypatch):
    """Turn the whole-level cache OFF, so a read is a WINDOW of shards and not the whole array.

    The fixture pyramid is 256^3 -- far below `ladder.CACHE_VOX` -- so without this every level would
    be read whole and `rung_need` would trivially name every shard there is."""
    monkeypatch.setattr(ladder, "CACHE_VOX", 0)
    ladder.clear_caches()


def test_rung_need_covers_exactly_what_read_rung_reads(ct_origin, tmp_path, windowed):
    pyr = ladder.rungs(ct_origin.path)
    lo, p = (64, 96, 64), 32
    arr, keys, whole = stream.rung_need(pyr, 2, lo, p)
    assert not whole and keys
    # minimal: every shard named intersects the voxel range read_rung will actually touch
    _, a, b, _ = stream.rung_range(pyr, 2, lo, p)
    g = stream.chunk_grid(arr)
    for ix in keys:
        s = np.array(ix, np.int64) * g
        assert (s < b).all() and (s + g > a).all(), f"shard {ix} is outside [{a}, {b})"
    # complete: a mirror holding ONLY those shards reads the identical cube (a missing shard would
    # read as the fill value, so any gap shows up as a difference wherever the slab is)
    cache = stream.ShardCache(ct_origin.url, str(tmp_path / "c"), budget_gb=1)
    mpyr = cache.levels()
    marr, mkeys, _ = stream.rung_need(mpyr, 2, lo, p)
    assert mkeys == keys
    cache._fetch(cache._paths(marr, mkeys), key="w")
    want = ladder.read_rung(pyr, 2, lo, p, dtype=np.uint8)
    got = ladder.read_rung(mpyr, 2, lo, p, dtype=np.uint8)
    assert want.max() > 0 and np.array_equal(got, want)
    cache.close()


def test_ctx_need_covers_the_context_cubes(ct_origin, tmp_path, windowed):
    pyr = ladder.rungs(ct_origin.path)
    lo, p, ctx = (64, 96, 64), 32, (1, 2, 3)
    need = stream.ctx_need(pyr, 2, lo, p, ctx)
    assert len(need) == len(ctx)
    cache = stream.ShardCache(ct_origin.url, str(tmp_path / "c"), budget_gb=1)
    mpyr = cache.levels()
    for arr, keys, _ in stream.ctx_need(mpyr, 2, lo, p, ctx):
        cache._fetch(cache._paths(arr, keys), key="w")
    want = ladder.context(ct_origin.path, lo, (p,) * 3, ctx, rung=2)
    got = ladder.context(cache.base, lo, (p,) * 3, ctx, rung=2)
    assert all(np.array_equal(a, b) for a, b in zip(got, want))
    cache.close()


def test_404_leaves_an_absent_marker(ct_origin, tmp_path):
    """An absent key on the origin IS the array's fill value (air). Without the marker every later
    window would ask for it again."""
    cache = stream.ShardCache(ct_origin.url, str(tmp_path / "c"), budget_gb=1)
    cache.levels()                      # the level probe itself 404s on the rungs that do not exist
    n0, h0 = cache._f.absent, cache._f.have
    p = os.path.join(cache.base, "0", "c", "99", "0", "0")
    cache._fetch([p], key="w")
    assert os.path.exists(p + ".absent") and not os.path.exists(p)
    assert cache._f.absent == n0 + 1
    cache._fetch([p], key="w")          # the marker is a hit, not a second request
    assert cache._f.absent == n0 + 1 and cache._f.have == h0 + 1
    cache.close()


def test_release_and_evict_keep_pinned_levels_and_live_regions(ct_origin, tmp_path, windowed):
    cache = stream.ShardCache(ct_origin.url, str(tmp_path / "c"), budget_gb=1)
    pinned_rungs = cache.pin_small_levels(max_vox=4_000_000)   # rungs 3..5, not the 256^3 level 0
    assert pinned_rungs and 2 not in pinned_rungs
    pin_files = [p for p in cache.pin if os.path.exists(p)]
    assert pin_files

    k1 = cache.fetch_region((0, 64, 0), ctx=(1, 2, 3), patch=32, region=64)
    k2 = cache.fetch_region((64, 128, 64), ctx=(1, 2, 3), patch=32, region=64)
    p1 = {p for p in cache.regions[k1] if os.path.exists(p)}
    p2 = {p for p in cache.regions[k2] if os.path.exists(p)}
    assert p1 and p2 and not (p1 & p2)    # the coarse rungs are pinned, so these are rung-2 shards only

    cache.budget = 1                      # anything evictable must go
    cache.release(k1)
    assert not any(os.path.exists(p) for p in p1), "a released region's shards must be evictable"
    assert all(os.path.exists(p) for p in p2), "a live region's shards must never be evicted"
    assert all(os.path.exists(p) for p in pin_files), "a pinned level must never be evicted"
    cache.close()


def test_a_local_volume_is_symlinked_not_copied(ct_origin, tmp_path):
    cache = stream.ShardCache(ct_origin.path, str(tmp_path / "c"), budget_gb=1)
    assert os.path.islink(cache.base)
    assert os.path.realpath(cache.base) == os.path.realpath(ct_origin.path)
    assert sorted(cache.levels()) == sorted(ladder.rungs(ct_origin.path))
    cache.fetch_region((0, 64, 0), ctx=(1, 2, 3), patch=32, region=64)
    assert cache.stats()["gb"] == 0 and cache.stats()["fetched"] == 0
    cache.release(cache.region_key((0, 64, 0)))
    assert os.path.realpath(cache.base) == os.path.realpath(ct_origin.path)  # nothing was deleted


def test_a_seed_mirror_is_linked_and_only_its_gaps_are_fetched(ct_origin, tmp_path, windowed):
    """`ct_seed`: a partial local mirror of the URL. What it has is linked in (never fetched, and
    eviction removes only the cache's link); a gap in an INCOMPLETE level goes to the origin; a gap in a
    level whose mirror.json says complete is absent without asking anyone."""
    import json
    import shutil
    seed = tmp_path / "seed"
    shutil.copytree(ct_origin.path, seed)
    lv = sorted((d for d in os.listdir(seed) if d.isdigit()), key=int)
    fine, coarse = lv[0], lv[-1]
    (seed / coarse / "mirror.json").write_text(json.dumps({"complete": True}))
    shards = [os.path.join(r, f) for r, _, fs in os.walk(seed / fine / "c") for f in fs]
    assert len(shards) > 1
    gone = shards[0]
    os.remove(gone)                                  # level `fine` is partial: this one must be fetched
    cache = stream.ShardCache(ct_origin.url, str(tmp_path / "c"), budget_gb=1, seed=str(seed))
    pyr = cache.levels()
    assert sorted(pyr) == sorted(ladder.rungs(ct_origin.path))
    rel = [os.path.relpath(p, seed) for p in shards]
    cache._fetch([os.path.join(cache.base, r) for r in rel], key="w")
    st = cache.stats()
    assert st["fetched"] == 1 and st["seeded"] >= len(shards) - 1
    assert os.path.samefile(os.path.join(cache.base, rel[1]), shards[1])   # a link, not a copy
    miss = os.path.join(cache.base, coarse, "c", "99", "0", "0")
    n = cache._f.requests
    cache._fetch([miss], key="w")
    assert os.path.exists(miss + ".absent") and cache._f.requests == n   # complete level: no request
    a = ladder.read_rung(ladder.rungs(ct_origin.path), 2, (0, 0, 0), 64, dtype=np.uint8)
    b = ladder.read_rung(ladder.rungs(cache.base), 2, (0, 0, 0), 64, dtype=np.uint8)
    assert np.array_equal(a, b)
    os.remove(os.path.join(cache.base, rel[1]))      # what eviction does
    assert os.path.exists(shards[1])
    cache.close()


def test_a_failed_shard_aborts_the_unit_and_books_nothing(ct_origin, tmp_path):
    """One `fail` (retries exhausted) raises FetchFailed before anything is charged or held, counts
    the failed unit and logs it; an absent shard (no file) is never charged either (review D01)."""
    logs = []
    cache = stream.ShardCache(ct_origin.url, str(tmp_path / "c"), budget_gb=1, log=logs.append)
    cache.levels()
    good = os.path.join(cache.base, "0", "c", "0", "0", "0")
    bad = os.path.join(cache.base, "0", "c", "0", "0", "1")
    absent = os.path.join(cache.base, "0", "c", "99", "0", "0")

    class Fake:
        async def get(self, p):
            if p == bad:
                return "fail", 0
            if p == absent:
                return "absent", 0
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(b"x" * 10)
            return "new", 10

    async def fake_fetcher():
        return Fake()
    cache._fetcher = fake_fetcher
    before = (cache.cache_bytes, dict(cache.hold), dict(cache.regions))
    with pytest.raises(stream.FetchFailed) as e:
        cache._fetch([good, bad, absent], key="r1")
    assert e.value.paths == [bad] and cache.failed_units == 1
    assert (cache.cache_bytes, dict(cache.hold), dict(cache.regions)) == before
    assert any("FAILED unit r1" in m for m in logs)
    cache._fetch([good, absent], key="r2")                  # the retry: the absent one is not held
    assert good in cache.regions["r2"] and absent not in cache.regions["r2"]
    assert absent not in cache.size
    cache.close()


# --------------------------------------------------------------------------- residency (D04 / D10)

def _rolling(base):
    """What a filesystem walk says the rolling part of a mirror is: every shard file (not metadata,
    not an absent marker), with its size."""
    out = {}
    for dp, _dn, fn in os.walk(base):
        for n in fn:
            if not (n.endswith(".json") or n.endswith(".absent")):
                out[os.path.join(dp, n)] = os.path.getsize(os.path.join(dp, n))
    return out


A, B = (0, 64, 0), (64, 128, 64)
KW = {"ctx": (1, 2, 3), "patch": 32, "region": 64}


def test_a_restart_inventories_old_shards_and_evicts_them_first(ct_origin, tmp_path, windowed):
    """D10: after a restart the books were empty, so every shard an earlier process had fetched was
    never charged and never an eviction candidate (paris4: 122 GB on disk against a 64 GB budget).
    The constructor now charges what is on disk with its mtime as the last reference: the budget is
    honoured at once, and the oldest untouched shards go first."""
    root = str(tmp_path / "c")
    c1 = stream.ShardCache(ct_origin.url, root, budget_gb=1)
    c1.pin_small_levels(max_vox=4_000_000)
    ka, kb = c1.fetch_region(A, **KW), c1.fetch_region(B, **KW)
    pa = {p for p in c1.regions[ka] if os.path.exists(p)}
    pb = {p for p in c1.regions[kb] if os.path.exists(p)}
    pins = {p for p in c1.pin if os.path.exists(p)}
    assert pa and pb and not (pa & pb) and pins
    c1.close()
    old = os.path.getmtime(next(iter(pb))) - 3600
    for p in pa:                                 # A was last touched an hour before B
        os.utime(p, (old, old))
    open(next(iter(pb)) + ".part", "wb").write(b"half")   # a killed process's half shard

    c2 = stream.ShardCache(ct_origin.url, root, budget_gb=1)
    assert not os.path.exists(next(iter(pb)) + ".part"), "a .part is garbage and is removed"
    c2.pin_small_levels(max_vox=4_000_000)
    disk = _rolling(c2.base)
    rolling = {p: s for p, s in disk.items() if p not in c2.pin}
    assert set(rolling) == pa | pb
    st = c2.stats()
    assert c2.cache_bytes == sum(rolling.values()) and st["inventoried"] == len(disk)
    assert abs(st["pinned_gb"] * (1 << 30) - sum(disk[p] for p in pins)) < 1
    # a budget that holds B but not A: nothing is held after a restart, the old A goes, B stays
    bb = sum(rolling[p] for p in pb)
    c2.budget = int(bb / 0.9) + 1
    assert c2.cache_bytes > c2.budget
    assert c2.evict() == len(pa)
    assert not any(os.path.exists(p) for p in pa) and all(os.path.exists(p) for p in pb)
    assert all(os.path.exists(p) for p in pins) and c2.cache_bytes <= c2.budget
    c2.close()


def test_seed_linked_shards_are_not_charged_and_never_evicted(ct_origin, tmp_path, windowed):
    """A shard hard-linked from the seed costs no disk: it is booked as `linked`, not against the
    budget, and eviction never removes it -- a revisit of a seeded level can never find it gone."""
    import shutil
    seed = tmp_path / "seed"
    shutil.copytree(ct_origin.path, seed)
    root = str(tmp_path / "c")
    c = stream.ShardCache(ct_origin.url, root, budget_gb=1, seed=str(seed))
    k = c.fetch_region(A, **KW)
    files = [p for p in c.regions[k] if os.path.exists(p)]
    assert files and all(os.stat(p).st_nlink > 1 for p in files)
    assert c.cache_bytes == 0 and set(files) <= c.linked
    c.release(k)
    c.budget = 0
    c.cache_bytes = 1                           # force an eviction pass
    c.evict()
    assert all(os.path.exists(p) for p in files)
    c.close()
    c2 = stream.ShardCache(ct_origin.url, root, budget_gb=1, seed=str(seed))   # and after a restart
    assert set(files) <= c2.linked and c2.cache_bytes == 0
    c2.close()


def test_a_leased_region_survives_eviction_under_a_tiny_budget(ct_origin, tmp_path, windowed):
    """D04: the producer releases a region once the cursor has passed it, but the cursor is published
    when a visit STARTS -- the region the worker is reading is already behind it. Its lease keeps its
    shards through any eviction until the worker lets go."""
    c = stream.ShardCache(ct_origin.url, str(tmp_path / "c"), budget_gb=1)
    c.pin_small_levels(max_vox=4_000_000)
    ka = c.fetch_region(A, **KW)
    pa = {p for p in c.regions[ka] if os.path.exists(p)}
    c.release(ka)                               # the producer is done with A ...
    c.lease([ka])                               # ... a worker is still reading it
    c.budget = 1
    kb = c.fetch_region(B, **KW)                # evicts everything it may
    pb = {p for p in c.regions[kb] if os.path.exists(p)}
    assert pa and all(os.path.exists(p) for p in pa), "a leased region's shards were evicted"
    assert not c.missing(ka)
    c.lease([])                                 # the worker moved on
    c.evict()
    assert not any(os.path.exists(p) for p in pa) and all(os.path.exists(p) for p in pb)
    assert c.missing(ka)
    c.close()


def test_a_revisit_whose_home_was_evicted_is_fetched_again(ct_origin, tmp_path, windowed,
                                                          monkeypatch):
    """A rung 3-6 visit's home region was produced (and released) when the walk passed it; by the
    visit the budget has evicted its shards. The worker's lease names it, the producer sees a leased
    region it does not hold and fetches it again (`produce_loop.follow_leases` -> `need`), and the
    worker's residency check passes only then. `_release_passed` never releases a leased region."""
    from rvsm import run as RUN
    out = str(tmp_path / "run")
    c = stream.ShardCache(ct_origin.url, out, budget_gb=1)
    c.pin_small_levels(max_vox=4_000_000)
    pyr = c.levels()
    ka = c.fetch_region(A, **KW)
    c.release(ka)
    c.budget = 1
    kb = c.fetch_region(B, **KW)
    home = stream.region_paths(pyr, A, KW["ctx"], KW["region"])
    assert not stream.resident(home), "the fixture should have evicted A"
    # the worker waiting on its revisit publishes the lease
    RUN.write_state(out, round=0)
    RUN._write_json(os.path.join(RUN.cursor_dir(out), "w0.json"),
                    {"pos": 5, "stride": 1, "worker": 0, "round": 0, "lease": [list(A), list(B)]})
    leased = RUN.cursor_leases(out)
    assert leased == sorted([A, B])
    keys = {B: kb}                              # the producer holds B, not A
    c.lease(c.region_key(lo) for lo in leased)
    todo = [lo for lo in leased if lo not in keys]
    assert todo == [A]
    n0 = c.stats()["fetched"]
    keys[A] = c.fetch_region(np.array(A, np.int64), **KW)
    assert c.stats()["fetched"] > n0 and stream.resident(home)
    # both are behind the cursor and finished: only the unleased one may be released
    monkeypatch.setattr(RUN, "_next_job", lambda *a, **k: None)
    RUN._release_passed(c, keys, {A: 0, B: 1}, 5, None, 0, True, out, rungs=(2,), leased=[A])
    assert A in keys and B not in keys
    c.close()


def test_a_published_lease_protects_before_the_producer_has_looked(tmp_path):
    """P3-02, the review's probe: a worker sees its bytes, publishes its lease and draws at once, while
    the producer is deep in a GPU unit with an old lease snapshot (here: none). Every eviction now
    re-reads the leases from the run directory itself, so the file is still there for the first draw;
    once the worker's next publish drops the lease, the shard is evictable again."""
    from rvsm import run as RUN
    from rvsm.walk import WalkPatches
    tmp = str(tmp_path)
    cache = stream.ShardCache("https://example.invalid/volume", tmp, budget_gb=1)
    path = os.path.join(cache.base, "0", "c", "0", "0", "0")
    os.makedirs(os.path.dirname(path))
    open(path, "wb").write(b"CT-data")
    key = cache.region_key((0, 0, 0))
    cache._paths_of[key] = [path]
    cache.charge(path)
    cache.hold[path] = {key}
    cache.regions[key] = {path}
    cache.release(key)
    RUN.write_state(tmp, round=0)
    ds = WalkPatches.__new__(WalkPatches)
    ds.out, ds.round, ds.stream_mirror = tmp, 0, True
    ds._rpaths = {(0, 0, 0): [path]}
    ds._region_lo = lambda rec: (0, 0, 0)
    assert ds._resident({})
    assert ds._publish(0, 1, 1, 0, lease=[(0, 0, 0)])
    assert RUN.cursor_leases(tmp) == [(0, 0, 0)] and not cache.leased   # the owner has not looked
    cache.budget = 0
    cache.evict()
    assert os.path.exists(path), "a published lease must protect at the very next eviction decision"
    assert ds._publish(0, 1, 2, 0, lease=[])                           # the visit is over
    cache.evict()
    assert not os.path.exists(path)
    cache.close()


def test_the_lease_keeper_refetches_and_acknowledges_a_lease(ct_origin, tmp_path, windowed):
    """One pass of the producer's lease keeper: a NEW lease (worker, lease_id) whose home was evicted is
    fetched again and acknowledged with that home ready; the same lease is not re-resolved; a new
    lease id is."""
    import threading

    from rvsm import run as RUN
    out = str(tmp_path / "run")
    c = stream.ShardCache(ct_origin.url, out, budget_gb=1)
    c.pin_small_levels(max_vox=4_000_000)
    ka = c.fetch_region(A, **KW)
    c.release(ka)
    c.budget = 1
    kb = c.fetch_region(B, **KW)
    assert c.missing(ka)
    RUN.write_state(out, round=0)
    RUN._write_json(os.path.join(RUN.cursor_dir(out), "w0.json"),
                    {"pos": 5, "stride": 1, "worker": 0, "round": 0, "lease": [list(A)],
                     "lease_id": "p.1"})
    keys, st, lk = {B: kb}, {}, threading.Lock()
    fk = {"ctx": KW["ctx"], "patch": KW["patch"], "region": KW["region"]}
    assert RUN.acknowledge_leases(out, c, keys, lk, st, fk) == [(0, "p.1")]
    ack = RUN.read_lease_ack(out, 0)
    assert ack["lease_id"] == "p.1" and ack["ready"] == [list(A)]
    assert A in keys and not c.missing(ka)
    assert stream.resident(stream.region_paths(c.levels(), A, KW["ctx"], KW["region"]))
    c.evict()                                        # 1-byte budget: A is leased, so it stays
    assert not c.missing(ka)
    assert RUN.acknowledge_leases(out, c, keys, lk, st, fk) == []          # already acknowledged
    RUN._write_json(os.path.join(RUN.cursor_dir(out), "w0.json"),
                    {"pos": 6, "stride": 1, "worker": 0, "round": 0, "lease": [list(B)],
                     "lease_id": "p.2"})
    assert RUN.acknowledge_leases(out, c, keys, lk, st, fk) == [(0, "p.2")]
    assert RUN.read_lease_ack(out, 0)["ready"] == [list(B)]
    c.close()


def test_a_lease_whose_fetch_fails_is_acknowledged_unready_and_retried(tmp_path, monkeypatch):
    """`fetch_region` raises `FetchFailed` when a shard does not arrive. The lease keeper acknowledges
    that lease with the home NOT ready (the worker keeps waiting, then times out loudly) instead of
    dying, and retries it only after LEASE_RETRY_S."""
    import threading

    from rvsm import run as RUN
    out = str(tmp_path / "run")
    c = stream.ShardCache("https://example.invalid/volume", out, budget_gb=1)
    calls = []

    def boom(lo, **kw):
        calls.append(tuple(int(v) for v in lo))
        raise stream.FetchFailed("k", ["p"])
    monkeypatch.setattr(c, "fetch_region", boom)
    monkeypatch.setattr(c, "missing", lambda key: True)
    RUN.write_state(out, round=0)
    RUN._write_json(os.path.join(RUN.cursor_dir(out), "w0.json"),
                    {"pos": 1, "stride": 1, "worker": 0, "round": 0, "lease": [list(A)],
                     "lease_id": "p.1"})
    st, lk = {}, threading.Lock()
    assert RUN.acknowledge_leases(out, c, {}, lk, st, {}) == [(0, "p.1")]
    assert RUN.read_lease_ack(out, 0)["ready"] == [] and calls == [A]
    assert RUN.acknowledge_leases(out, c, {}, lk, st, {}) == []          # not before the retry delay
    st[0] = (st[0][0], st[0][1], st[0][2] - RUN.LEASE_RETRY_S - 1)
    assert RUN.acknowledge_leases(out, c, {}, lk, st, {}) == [(0, "p.1")] and calls == [A, A]
