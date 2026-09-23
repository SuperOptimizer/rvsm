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
