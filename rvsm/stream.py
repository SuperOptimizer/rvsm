"""The CT shard cache: a rolling local mirror of the one volume a run reads, per region.

rvsm starts from a CT zarr URL and nothing else, and it produces one 1024^3 region at a time. Between
those two facts sits this module: given a region corner, it fetches exactly the shards that region's
reads will touch -- the rung-2 window plus the nine context cubes, which is what `sample.Patches` and
the region runner ask the ladder for -- into the plain MIRROR LAYOUT

    <root>/ct/<vol>.zarr/<level>/{zarr.json, c/z/y/x[, .absent]}

so that `ladder.rungs(cache.base)` opens it as an ordinary sharded zarr and every other module reads the
cache without knowing it is one. A local CT path is not copied at all: `<root>/ct/<vol>.zarr` becomes a
SYMLINK to it and nothing is ever fetched or evicted.

There is no planner process here. usrm2's stream.py carried a whole second process -- a walk, a
`queue.jsonl`, a resume cursor, a meta file and the replay protocol on the loader side -- so a trainer on
one host could consume a plan made on another. rvsm's producer and trainer share a filesystem, so the
cache is a plain object the producer calls: `fetch_region` before a pass, `release` after it.

What survives verbatim from usrm2 is the part that was load-bearing:

- ONE keep-alive `aiohttp` session with a bounded pool of 32 connections (on the 5090, 32 gave 21 MiB/s
  from dl.ash2txt.org with no errors against 8 MiB/s at 16; 64 churning jobs got 3), because the origin rewards reuse and
  punishes bursts;
- `.part` + `os.replace`, so a killed process never leaves a half shard that decodes as garbage;
- a per-path lock, so two regions wanting the same shard cost one GET;
- 404 -> a zero-length `<path>.absent` marker: an absent key IS the array's fill value (air), and
  without the marker every later window re-asks the origin for it;
- retries with linear backoff, and a loud line on the final failure.

Residency. The coarse levels of a scroll are a handful of shards and every sample's context cubes read
them, so `pin_small_levels` pulls whole levels at or below `ladder.CACHE_VOX` voxels ONCE and never
evicts them; the fine levels are the rolling part. Eviction is LRU by LAST REFERENCE and never touches a
shard a region still holds: `release(key)` is what makes that region's shards evictable, and the budget
is only enforced down to 90 % so a run does not spend its life at the boundary.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import numpy as np

from rvsm import ladder


# ------------------------------------------------------------------ which chunks a read touches

def chunk_grid(arr):
    """The WRITE-chunk (shard) size of an array: the unit that is fetched, cached and evicted."""
    return np.array(getattr(arr, "shards", None) or arr.chunks, np.int64)[-3:]


def chunk_key(arr, ix):
    """The key of chunk (z, y, x), relative to the array directory."""
    z, y, x = (int(v) for v in ix)
    v2 = int(getattr(getattr(arr, "metadata", None), "zarr_format", 3)) == 2
    return f"{z}.{y}.{x}" if v2 else f"c/{z}/{y}/{x}"


def keys_in(arr, a, b):
    """The chunk indices of `arr` covering the voxel range [a, b)."""
    g, S = chunk_grid(arr), np.array(arr.shape[-3:], np.int64)
    a, b = np.maximum(np.asarray(a, np.int64), 0), np.minimum(np.asarray(b, np.int64), S)
    if (b <= a).any():
        return []
    lo, hi = a // g, -(-b // g)
    return [(z, y, x) for z in range(lo[0], hi[0]) for y in range(lo[1], hi[1])
            for x in range(lo[2], hi[2])]


def all_keys(arr):
    n = -(-np.array(arr.shape[-3:], np.int64) // chunk_grid(arr))
    return [(z, y, x) for z in range(n[0]) for y in range(n[1]) for x in range(n[2])]


def rung_range(pyr, k, lo, p):
    """(array, a, b, whole) that `ladder.read_rung(pyr, k, lo, p)` will read: the voxel range [a, b) of
    the SOURCE level, or whole=True when `ladder.full_level` keeps that level decoded and reads all of
    it. It mirrors `read_rung` / `read_block` / `full_level`: the source is the highest rung at or below
    k, and when either that level or the rung-k view of it is small enough to be kept whole, the WHOLE
    level is read (once) instead of a window of it."""
    src = max(r for r in pyr if r <= k)
    arr = pyr[src]
    S = np.array(arr.shape[-3:], np.int64)
    if (int(np.prod(ladder.rung_shape(pyr, k))) <= ladder.CACHE_VOX
            or int(np.prod(S)) <= ladder.CACHE_VOX):
        return arr, np.zeros(3, np.int64), S, True
    p, lo = ladder.shape3(p), np.asarray(lo, np.int64)
    e = 1 << (k - src)
    jlo, jhi = np.maximum(-lo, 0), np.minimum(-(-S // e) - lo, p)
    if (jhi <= jlo).any():          # the cube does not touch the array at all
        return arr, np.zeros(3, np.int64), np.zeros(3, np.int64), False
    return arr, (lo + jlo) * e, np.minimum((lo + jhi) * e, S), False


def rung_need(pyr, k, lo, p):
    """(array, chunk indices, whole) of the same read: the outer (shard) keys it touches."""
    arr, a, b, whole = rung_range(pyr, k, lo, p)
    return arr, (all_keys(arr) if whole else keys_in(arr, a, b)), whole


def ctx_need(pyr, k, lo, p, ctx):
    """`ladder.context`'s reads: one (array, keys, whole) per context offset."""
    c0 = np.asarray(lo, np.int64) + ladder.shape3(p) // 2
    out = []
    for d in ctx:
        lo_d = c0 // (1 << int(d)) - ladder.shape3(p) // 2
        out.append(rung_need(pyr, k + int(d), lo_d, p))
    return out


# ------------------------------------------------------------------ fetching

class Fetcher:
    """Concurrent GETs of mirror objects from their origin. 404 = absent = a zero-length marker file.

    `urlof` maps a local mirror path to its origin URL; the cache owns that mapping because it owns the
    mirror root."""

    def __init__(self, session, urlof, jobs=16, retries=4, log=print, seedof=None):
        self.session, self.urlof, self.log, self.seedof = session, urlof, log, seedof
        self.sem, self.retries = asyncio.Semaphore(int(jobs)), int(retries)
        self.lock = {}   # one download per shard: concurrent regions wanting the same object wait for it
        self.bytes = self.fetched = self.absent = self.have = self.failed = self.requests = 0
        self.seeded = 0

    async def get(self, path):
        """(status, bytes) with status in have / new / absent / fail; `path` is the LOCAL mirror path.
        `have` counts buffer hits -- a shard some earlier region pulled -- and `fetched` the misses,
        which is what a hit rate is made of."""
        if os.path.exists(path):
            self.have += 1
            return "have", 0
        if os.path.exists(path + ".absent"):
            self.have += 1
            return "absent", 0
        lk = self.lock.setdefault(path, asyncio.Lock())
        async with lk:
            try:
                return await self._get(path)
            finally:
                if not lk.locked():
                    self.lock.pop(path, None)

    async def _get(self, path):
        if os.path.exists(path):        # another region pulled this shard while we waited for the lock
            self.have += 1
            return "have", 0
        if os.path.exists(path + ".absent"):
            self.have += 1
            return "absent", 0
        got = self._from_seed(path) if self.seedof is not None else None
        if got is not None:
            return got
        url = self.urlof(path)
        async with self.sem:
            for attempt in range(self.retries):
                try:
                    async with self.session.get(url) as r:
                        if r.status == 404:
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            open(path + ".absent", "wb").close()
                            self.absent += 1
                            return "absent", 0
                        r.raise_for_status()
                        buf = await r.read()
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path + ".part", "wb") as f:
                        f.write(buf)
                    os.replace(path + ".part", path)   # never a half shard under a live reader
                    self.bytes += len(buf)
                    self.fetched += 1
                    self.requests += 1
                    return "new", len(buf)
                except Exception as e:  # noqa: BLE001
                    if attempt == self.retries - 1:
                        self.log(f"rvsm cache: FAILED {url}: {e!r}")
                        self.failed += 1
                        return "fail", 0
                    await asyncio.sleep(2 * (attempt + 1))


    def _from_seed(self, path):
        """Serve `path` from the local seed mirror when it can answer: a file it has is HARD-LINKED in
        (copied across filesystems), so eviction later removes the cache's link and never the seed's
        file; a file it lacks in a level its `mirror.json` marks complete is absent on the origin too.
        None = the seed cannot tell, ask the origin."""
        src, complete = self.seedof(path)
        if src is None:
            return None
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                os.link(src, path + ".part")
            except OSError:
                import shutil
                shutil.copyfile(src, path + ".part")
            os.replace(path + ".part", path)
            self.seeded += 1
            return "new", 0
        if os.path.isfile(src + ".absent") or (complete and os.path.basename(src) != "zarr.json"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path + ".absent", "wb").close()
            self.absent += 1
            return "absent", 0
        return None


async def fetch_group_meta(f, base):
    """Fetch a pyramid group's `zarr.json` and every level's `zarr.json` into the mirror, so that
    `ladder.rungs(base)` opens the pyramid locally. Level names come from the group's OME multiscales
    when it has them, and the integer names are probed as well (a 404 probe leaves nothing behind --
    an `.absent` marker on a level's metadata would make that level permanently invisible)."""
    st, _ = await f.get(f"{base}/zarr.json")
    names = []
    if st in ("have", "new"):
        j = json.load(open(f"{base}/zarr.json"))
        at = j.get("attributes", j) if isinstance(j, dict) else {}
        ms = (at.get("ome") or at).get("multiscales") if isinstance(at, dict) else None
        names = [str(d["path"]) for d in ms[0]["datasets"]] if ms else []
    else:
        os.makedirs(base, exist_ok=True)
    cand = list(dict.fromkeys(names + [str(i) for i in range(ladder.NRUNGS)]))
    res = await asyncio.gather(*[f.get(f"{base}/{n}/zarr.json") for n in cand])
    keep = []
    for n, (st, _) in zip(cand, res):
        if st in ("have", "new"):
            keep.append(n)
            continue
        for q in (f"{base}/{n}/zarr.json.absent", f"{base}/{n}"):
            try:
                os.remove(q) if os.path.isfile(q) else os.rmdir(q)
            except OSError:
                pass
    if not keep:
        raise RuntimeError(f"{base}: no pyramid levels on the origin")
    return keep


# ------------------------------------------------------------------ the cache

class ShardCache:
    """The run's CT, mirrored one shard at a time under `<root>/ct/<vol>.zarr`.

    `ShardCache(url_or_path, root, budget_gb)`; `levels()` opens the mirror as a pyramid,
    `pin_small_levels()` pulls the coarse levels whole, `fetch_region(lo2)` pulls what one region needs
    and `release(key)` gives those shards back to the LRU. Every public method is synchronous: the
    asyncio loop and the keep-alive session are this object's private business."""

    def __init__(self, src, root, budget_gb=64.0, jobs=32, retries=4, log=print, seed=None):
        self.src = ladder.pyramid_base(str(src))
        self.seed = os.path.abspath(str(seed)) if seed else None
        self._complete = {}
        self.root, self.jobs, self.retries, self.log = str(root), int(jobs), int(retries), log
        self.remote = ladder.is_url(self.src)
        self.vol = os.path.basename(self.src.rstrip("/"))
        self.base = os.path.join(self.root, "ct", self.vol)
        self.budget = int(float(budget_gb) * (1 << 30))
        self.size, self.ref, self.pin, self.hold, self.regions = {}, {}, set(), {}, {}
        self.cache_bytes, self.clock, self.evicted = 0, 0, 0
        self._loop = self._sess = self._f = None
        self._meta = False
        os.makedirs(os.path.dirname(self.base), exist_ok=True)
        if not self.remote:                 # a local volume is NOT copied: the mirror is a symlink to it
            tgt = os.path.abspath(self.src)
            if os.path.islink(self.base):
                if os.path.realpath(self.base) != tgt:
                    os.remove(self.base)
                    os.symlink(tgt, self.base)
            elif not os.path.exists(self.base):
                os.symlink(tgt, self.base)
            self._meta = True

    # ---- the asyncio side, kept private ---------------------------------

    def _run(self, coro):
        """Run one coroutine on THIS cache's loop, so the keep-alive session survives between calls."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    async def _fetcher(self):
        if self._f is None:
            import aiohttp
            self._sess = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=self.jobs, force_close=False),
                timeout=aiohttp.ClientTimeout(total=600, sock_connect=30))
            self._f = Fetcher(self._sess, self.url_of, jobs=self.jobs, retries=self.retries,
                              log=self.log, seedof=self.seed_of if self.seed else None)
        return self._f

    def url_of(self, path):
        """A local mirror path -> its origin URL (the inverse of the mirror layout)."""
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(self.base)).replace(os.sep, "/")
        return f"{self.src.rstrip('/')}/{rel}"

    def seed_of(self, path):
        """A mirror path -> (the same object in the seed mirror, whether its level is complete there).
        A level is complete when the seed's `<level>/mirror.json` says `{"complete": true}`; only then
        does a missing file mean absent rather than not-yet-mirrored."""
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(self.base))
        if rel.startswith(".."):
            return None, False
        lvl = rel.split(os.sep, 1)[0]
        if lvl not in self._complete:
            try:
                with open(os.path.join(self.seed, lvl, "mirror.json")) as f:
                    self._complete[lvl] = bool(json.load(f).get("complete"))
            except (OSError, ValueError, AttributeError):
                self._complete[lvl] = False
        return os.path.join(self.seed, rel), self._complete[lvl] and os.sep in rel

    def close(self):
        if self._sess is not None:
            self._run(self._sess.close())
            self._sess = self._f = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    # ---- the pyramid ----------------------------------------------------

    def meta(self):
        """Make sure the mirror has the group's and the levels' `zarr.json`; returns the level names."""
        if self._meta:
            return sorted((d for d in os.listdir(self.base) if d.isdigit()), key=int)

        async def go():
            f = await self._fetcher()
            return await fetch_group_meta(f, self.base)
        names = self._run(go())
        self._meta = True
        return names

    def levels(self):
        """{rung: zarr array} of the MIRROR. Once `meta()` has run this is a plain sharded zarr group,
        so `ladder.rungs(cache.base)` anywhere else gives the same arrays."""
        self.meta()
        return ladder.rungs(self.base)

    # ---- residency ------------------------------------------------------

    def _paths(self, arr, keys):
        d = ladder.array_dir(arr)
        return [f"{d}/{chunk_key(arr, ix)}" for ix in keys]

    def charge(self, path, sz=None):
        """Book a buffered shard against the budget, and stamp it with the current clock (LRU by LAST
        reference: a shard a later region touches again moves back to the end of the queue)."""
        if sz is None:
            sz = os.path.getsize(path) if os.path.exists(path) else 0
        self.cache_bytes += sz - self.size.get(path, 0)
        self.size[path] = sz
        self.clock += 1
        self.ref[path] = self.clock
        return sz

    def pin_small_levels(self, max_vox=ladder.CACHE_VOX):
        """Fetch WHOLE every level of at most `max_vox` voxels and pin it: the coarse rungs are a few
        shards each and every sample's context cubes read them, so they are pulled once and never
        evicted. Returns the pinned rungs."""
        pyr, out = self.levels(), []
        todo = []
        for k in sorted(pyr):
            if int(np.prod(pyr[k].shape[-3:])) > int(max_vox):
                continue
            todo += self._paths(pyr[k], all_keys(pyr[k]))
            out.append(k)
        self._fetch(todo, pin=True)
        return out

    def _fetch(self, paths, pin=False, key=None):
        """Fetch `paths` (local mirror paths) concurrently, book them, and bind them to `key`."""
        paths = list(dict.fromkeys(paths))
        if self.remote and paths:
            async def go():
                f = await self._fetcher()
                return await asyncio.gather(*[f.get(p) for p in paths])
            self._run(go())
        for p in paths:
            if pin:
                self.pin.add(p)
                self.size.setdefault(p, os.path.getsize(p) if os.path.exists(p) else 0)
                continue
            if p in self.pin or not self.remote:
                continue
            self.charge(p)
            self.hold.setdefault(p, set())
            if key is not None:
                self.hold[p].add(key)
                self.regions.setdefault(key, set()).add(p)
        return paths

    def region_key(self, lo2):
        return "%d_%d_%d" % tuple(int(v) for v in lo2)

    def fetch_region(self, lo2, ctx=tuple(range(1, ladder.NCTX + 1)), patch=256, region=1024, rung=2):
        """Pull everything a 1024^3 region at rung-2 corner `lo2` needs: the rung-2 shards the region's
        own reads touch, plus the `ctx_need` shards of the nine context rungs of a window CENTRED in it
        (the context cubes of any window inside the region live in the same coarse shards, because one
        coarse shard covers the whole region's footprint many times over). Returns the region key that
        `release` takes. A local volume fetches nothing."""
        pyr = self.levels()
        key = self.region_key(lo2)
        lo2 = np.asarray(lo2, np.int64)
        R = ladder.shape3(region)
        need = [rung_need(pyr, int(rung), lo2, R)]
        need += ctx_need(pyr, int(rung), lo2, R, ctx)
        paths = []
        for arr, keys, _whole in need:
            paths += self._paths(arr, keys)
        t0 = time.time()
        f = self._f
        b0, n0, s0 = (f.bytes, f.fetched, f.seeded) if f is not None else (0, 0, 0)
        self._fetch(paths, key=key)
        self.evict()
        if self.remote:
            f, dt = self._f, max(time.time() - t0, 1e-6)
            mb = ((f.bytes - b0) if f is not None else 0) / (1 << 20)
            got = (f.fetched - n0) if f is not None else 0
            seeded = (f.seeded - s0) if f is not None else 0
            self.log(f"rvsm cache: region {key} {len(paths)} shards in {dt:.1f}s: {got} fetched "
                     f"({mb:.0f} MiB, {mb / dt:.1f} MiB/s), {seeded} from the seed "
                     f"({self.cache_bytes / (1 << 30):.1f} GB buffered)")
        return key

    def release(self, key):
        """The region is done with: its shards become evictable (the ones no other live region holds)."""
        for p in self.regions.pop(str(key), ()):
            h = self.hold.get(p)
            if h is not None:
                h.discard(str(key))
        self.evict()

    def held(self, path):
        return bool(self.hold.get(path))

    def evict(self):
        """Delete released shards, oldest reference first, down to 90 % of the budget. A pinned level and
        a shard a live region still holds are never candidates, so a budget smaller than the working set
        simply does not shrink -- which is the honest failure, not a cache that deletes what is in use."""
        if not self.remote or self.cache_bytes <= self.budget:
            return 0
        target, n = 0.9 * self.budget, 0
        for p, _i in sorted(self.ref.items(), key=lambda q: q[1]):
            if self.cache_bytes <= target:
                break
            if p in self.pin or self.held(p):
                continue
            try:
                os.remove(p)
            except OSError:
                pass
            self.cache_bytes -= self.size.pop(p, 0)
            self.ref.pop(p, None)
            self.hold.pop(p, None)
            self.evicted += 1
            n += 1
        return n

    def stats(self):
        f = self._f
        return {"gb": self.cache_bytes / (1 << 30), "shards": len(self.size), "pinned": len(self.pin),
                "regions": len(self.regions), "evicted": self.evicted,
                "fetched": getattr(f, "fetched", 0), "seeded": getattr(f, "seeded", 0), "have": getattr(f, "have", 0),
                "absent": getattr(f, "absent", 0), "failed": getattr(f, "failed", 0)}
