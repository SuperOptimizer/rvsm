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

- ONE keep-alive `aiohttp` session per calling thread (the producer's reader and its lease keeper
  download concurrently, outside their cache lock: `fetch_region_outside`) with a bounded pool of 32
  connections (on the 5090, 32 gave 21 MiB/s
  from dl.ash2txt.org with no errors against 8 MiB/s at 16; 64 churning jobs got 3), because the origin rewards reuse and
  punishes bursts;
- `.part` + `os.replace`, so a killed process never leaves a half shard that decodes as garbage;
- a per-path lock, so two regions wanting the same shard cost one GET (within a thread; across two
  threads a shard may be fetched twice, each under its own `.part` name, and the second `os.replace`
  is harmless);
- 404 -> a zero-length `<path>.absent` marker: an absent key IS the array's fill value (air), and
  without the marker every later window re-asks the origin for it;
- retries with linear backoff, and a loud line on the final failure.

Residency. The coarse levels of a scroll are a handful of shards and every sample's context cubes read
them, so `pin_small_levels` pulls whole levels at or below `ladder.CACHE_VOX` voxels ONCE and never
evicts them; the fine levels are the rolling part. Eviction is LRU by LAST REFERENCE and never touches a
shard a region still holds: `release(key)` is what makes that region's shards evictable, and the budget
is only enforced down to 90 % so a run does not spend its life at the boundary.

Across restarts (review D10). The constructor rebuilds the inventory from the files on disk -- size, and
the mtime as the last reference (every reference touches it) -- so a shard an earlier process fetched is
charged and is an eviction candidate like any other; before this, untouched old shards were invisible
and the disk grew past the budget with every restart (paris4: 122 GB against 64). The accounting is in
three parts, reported separately by `stats()` (and `disk()`, on every region line and backpressure line):

    rolling   fetched shards: charged against the budget, evicted LRU
    pinned    the small coarse levels: never evicted, not charged
    linked    shards HARD-LINKED from the local seed mirror: never evicted, not charged

NOTE for anyone reading `du`: the budget governs `rolling` only. `du <root>/ct` counts every linked
shard at full size, but those bytes are the seed's and are on disk exactly once (`du -sh seed ct`
together shows it); what the run's CT really costs is the seed plus `rolling`. To shrink it, shrink
the seed (with the producer stopped: a shard it links as the seed vanishes fails its read).
A linked shard costs no disk (its inode is the seed's), so evicting it would free nothing and only turn
a later read of it into another link -- or, worse, into a miss for a visit already under way. Keeping
every linked shard is what makes a revisit of a seeded level always safe; a shard the seed could not
link (another filesystem: `_from_seed` copies it) has one link and is rolling.

Leases (review D04). `release` only says the PRODUCER is done with a region; the trainer's workers may
still be reading it (a visit that has just started, or a rung 3-6 revisit of a home the walk passed long
ago). `lease(keys)` is the trainer's side of residency: a leased region's shards are never evicted, even
released ones, and `missing(key)` says whether one has lost shards (the producer then fetches it again).
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
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


def region_paths(pyr, lo2, ctx=None, region=1024, rung=2):
    """The local mirror paths of every shard a region's reads touch (what `ShardCache.fetch_region`
    fetches): the rung-`rung` shards of the region itself plus the `ctx_need` shards of the context
    rungs of a window centred in it. Pure: the trainer's workers use it to check residency."""
    ctx = tuple(range(1, ladder.NCTX + 1)) if ctx is None else tuple(ctx)
    lo2 = np.asarray(lo2, np.int64)
    R = ladder.shape3(region)
    need = [rung_need(pyr, int(rung), lo2, R)] + ctx_need(pyr, int(rung), lo2, R, ctx)
    paths = []
    for arr, keys, _whole in need:
        d = ladder.array_dir(arr)
        paths += [f"{d}/{chunk_key(arr, ix)}" for ix in keys]
    return list(dict.fromkeys(paths))


def present(path):
    """Is a mirror object here: the shard itself, or its absent (= air) marker?"""
    return os.path.exists(path) or os.path.exists(path + ".absent")


def resident(paths):
    """Are all of `paths` on disk (a shard or its absent marker)?"""
    return all(present(p) for p in paths)


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
                    tmp = _part(path)
                    with open(tmp, "wb") as f:
                        f.write(buf)
                    os.replace(tmp, path)   # never a half shard under a live reader
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
            tmp = _part(path)
            try:
                os.link(src, tmp)
            except OSError:
                import shutil
                shutil.copyfile(src, tmp)
            os.replace(tmp, path)
            self.seeded += 1
            return "new", 0
        if os.path.isfile(src + ".absent") or (complete and os.path.basename(src) != "zarr.json"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path + ".absent", "wb").close()
            self.absent += 1
            return "absent", 0
        return None


def _part(path):
    """The temporary name a download is written under before its `os.replace`: private to this process
    and thread (the producer's reader and lease keeper download concurrently, each on its own loop, and
    may want the same shard), and ending in `.part` so `ShardCache.inventory` removes a killed one."""
    return f"{path}.{os.getpid()}.{threading.get_ident()}.part"


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

class FetchFailed(RuntimeError):
    """A region unit's shards did not all arrive: the unit is aborted and retried later."""

    def __init__(self, key, paths):
        super().__init__(f"{len(paths)} shard(s) failed for {key}: {paths[:3]}")
        self.key, self.paths = key, list(paths)


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
        self.linked, self.leased, self._paths_of = set(), set(), {}
        # where the trainer's leases are read from, right before EVERY eviction decision (review P3-02):
        # the run directory's cursor files by default, so no snapshot the producer took earlier can be
        # the one an eviction trusts
        self.lease_source = self._run_leases if self.remote else None
        self.region_args = None             # (ctx, region, rung) of fetch_region: a leased key's paths
        self.inventoried = 0
        # the asyncio loop, session and fetcher are PER THREAD (`_tl`): `download` runs without the
        # caller's lock (the producer's lease keeper and reader), and one loop cannot run twice at once
        self._tl, self._clients, self._clients_lock = threading.local(), [], threading.Lock()
        self.pending = {}                   # region key -> downloads in flight (`fetch_region_outside`)
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
        else:
            self.inventory()

    def inventory(self):
        """Rebuild the residency books from the mirror's files: every shard on disk is charged at its
        size, with its mtime as its last reference (oldest first in the LRU, and older than anything
        this process touches), except a hard-linked seed shard, which is `linked`. A `.part` left by a
        killed process is garbage and is removed. Returns the number of shards found."""
        found = []
        for dp, _dn, fn in os.walk(self.base) if os.path.isdir(self.base) else ():
            for n in fn:
                p = os.path.join(dp, n)
                if n.endswith(".part"):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                    continue
                if n.endswith(".absent") or n.endswith(".json"):
                    continue
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                found.append((st.st_mtime, p, st))
        found.sort(key=lambda q: q[0])
        for i, (_t, p, st) in enumerate(found):
            if self._is_linked(st):
                self.linked.add(p)
                self.size[p] = int(st.st_size)
                continue
            self.cache_bytes += int(st.st_size) - self.size.get(p, 0)
            self.size[p] = int(st.st_size)
            self.ref[p] = i - len(found)       # negative: older than every reference of this process
        self.inventoried = len(found)
        return len(found)

    def _is_linked(self, st):
        return self.seed is not None and st.st_nlink > 1

    # ---- the asyncio side, kept private ---------------------------------

    # this thread's loop / session / fetcher
    _loop = property(lambda self: getattr(self._tl, "loop", None),
                     lambda self, v: setattr(self._tl, "loop", v))
    _sess = property(lambda self: getattr(self._tl, "sess", None),
                     lambda self, v: setattr(self._tl, "sess", v))
    _f = property(lambda self: getattr(self._tl, "f", None), lambda self, v: setattr(self._tl, "f", v))

    def _client(self):
        """This thread's entry in `_clients` (what `close` shuts down and `stats` sums)."""
        c = getattr(self._tl, "client", None)
        if c is None:
            c = self._tl.client = {"loop": None, "sess": None, "f": None}
            with self._clients_lock:
                self._clients.append(c)
        return c

    def _run(self, coro):
        """Run one coroutine on THIS THREAD's loop of this cache, so the keep-alive session survives
        between calls."""
        if self._loop is None:
            self._loop = self._client()["loop"] = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    async def _fetcher(self):
        if self._f is None:
            import aiohttp
            self._sess = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=self.jobs, force_close=False),
                timeout=aiohttp.ClientTimeout(total=600, sock_connect=30))
            self._f = Fetcher(self._sess, self.url_of, jobs=self.jobs, retries=self.retries,
                              log=self.log, seedof=self.seed_of if self.seed else None)
            c = self._client()
            c["sess"], c["f"] = self._sess, self._f
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
        with self._clients_lock:
            clients, self._clients = list(self._clients), []
        for c in clients:                   # every thread's session and loop
            try:
                if c["sess"] is not None:
                    c["loop"].run_until_complete(c["sess"].close())
                if c["loop"] is not None:
                    c["loop"].close()
            except Exception:  # noqa: BLE001  -- a loop still running in its thread is left to it
                pass
            c["loop"] = c["sess"] = c["f"] = None
        self._tl = threading.local()

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
        reference: a shard a later region touches again moves back to the end of the queue). The file's
        mtime is touched too, so the next process's `inventory` sees the same order. A hard-linked seed
        shard is `linked` instead: not charged, never evicted (see the module docstring)."""
        try:
            st = os.stat(path)
        except OSError:
            st = None
        if sz is None:
            sz = int(st.st_size) if st is not None else 0
        if st is not None and self._is_linked(st):
            if path in self.ref:
                self.cache_bytes -= self.size.get(path, 0)
                self.ref.pop(path, None)
            self.linked.add(path)
            self.size[path] = sz
            return sz
        self.linked.discard(path)
        if path in self.ref or path not in self.size:
            self.cache_bytes += sz - self.size.get(path, 0)
        else:
            self.cache_bytes += sz
        self.size[path] = sz
        self.clock += 1
        self.ref[path] = self.clock
        if st is not None:
            try:
                os.utime(path, None)
            except OSError:
                pass
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
        for p in todo:                  # an inventoried pinned shard is no longer rolling
            if p in self.ref:
                self.cache_bytes -= self.size.get(p, 0)
                self.ref.pop(p, None)
            self.linked.discard(p)
        self._fetch(todo, pin=True)
        return out

    def _fetch(self, paths, pin=False, key=None):
        """Fetch `paths` (local mirror paths) concurrently, book them, and bind them to `key`.

        Every status is inspected: one `fail` (retries exhausted) raises `FetchFailed` BEFORE anything
        is booked, so the unit that asked is aborted -- no store is written from a CT with holes in it,
        and the region is retried on a later pass -- and `failed_units` counts it. A path that does not
        exist on disk (an `absent` shard, the origin's 404) is never charged or held."""
        paths = list(dict.fromkeys(paths))
        self.download(paths, key=key)
        return self.book(paths, pin=pin, key=key)

    def download(self, paths, key=None):
        """The network half of `_fetch`: get `paths` onto disk (this thread's loop and session), book
        nothing, raise `FetchFailed` on a `fail`. Touches none of the books, so it may run while
        another thread holds the caller's cache lock (`fetch_region_outside`)."""
        paths = list(dict.fromkeys(paths))
        if self.remote and paths:
            async def go():
                f = await self._fetcher()
                return await asyncio.gather(*[f.get(p) for p in paths])
            got = self._run(go()) or []
            bad = [p for p, r in zip(paths, got) if (r[0] if isinstance(r, tuple) else r) == "fail"]
            if bad:
                with self._clients_lock:
                    self.failed_units = getattr(self, "failed_units", 0) + 1
                self.log(f"rvsm cache: FAILED unit {key or '(pin)'}: {len(bad)} of {len(paths)} shards "
                         f"did not arrive (failed units so far: {self.failed_units}); nothing booked")
                raise FetchFailed(key, bad)
        return paths

    def book(self, paths, pin=False, key=None):
        """The books half of `_fetch`: charge / hold / pin every one of `paths` that is on disk."""
        paths = list(dict.fromkeys(paths))
        for p in paths:
            if not os.path.exists(p):
                continue
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

    def fetch_region(self, lo2, ctx=tuple(range(1, ladder.NCTX + 1)), patch=256, region=1024, rung=2,
                     evict=True):
        """Pull everything a 1024^3 region at rung-2 corner `lo2` needs: the rung-2 shards the region's
        own reads touch, plus the `ctx_need` shards of the nine context rungs of a window CENTRED in it
        (the context cubes of any window inside the region live in the same coarse shards, because one
        coarse shard covers the whole region's footprint many times over). Returns the region key that
        `release` takes. A local volume fetches nothing. `evict=False` leaves the budget to a later
        `evict()` (a producer re-holding several leased regions at startup must hold them all first)."""
        key, paths = self.plan_region(lo2, ctx=ctx, patch=patch, region=region, rung=rung)
        t0, c0 = time.time(), self._counts()
        self._fetch(paths, key=key)
        if evict:
            self.evict()
        self._log_region(key, paths, t0, c0)
        return key

    def plan_region(self, lo2, ctx=tuple(range(1, ladder.NCTX + 1)), patch=256, region=1024, rung=2):
        """(key, paths) of `fetch_region`, recorded (`paths_of`) but not fetched."""
        pyr = self.levels()
        key = self.region_key(lo2)
        paths = region_paths(pyr, lo2, ctx, region, rung)
        self._paths_of[key] = paths
        self.region_args = (tuple(ctx), region, rung)
        return key, paths

    def fetch_region_outside(self, lo2, lock, evict=True, on_booked=None, **kw):
        """`fetch_region` for a caller that serialises its use of the cache with `lock` (the producer's
        `clock`), holding that lock only for the books: the region's paths are planned and it is marked
        PENDING under the lock, its shards are downloaded WITHOUT it (minutes behind a slow origin, while
        the other threads keep using the cache), then booked, pending cleared and (`evict`) the budget
        applied under it again. A pending region's shards are never evicted (`evict`), so the ones that
        were already on disk are still there when it is booked. `on_booked(key)` runs under the lock
        right after the booking (the caller's own record of what it holds, updated in the same step).
        Returns the key; `FetchFailed` as `fetch_region`, with nothing booked."""
        with lock:
            key, paths = self.plan_region(lo2, **kw)
            self.pending[key] = self.pending.get(key, 0) + 1
        booked = False
        try:
            t0, c0 = time.time(), self._counts()
            self.download(paths, key=key)
            with lock:
                self.book(paths, key=key)
                self._unpend(key)
                booked = True
                if on_booked is not None:
                    on_booked(key)
                if evict:
                    self.evict()
        finally:
            if not booked:
                with lock:
                    self._unpend(key)
        self._log_region(key, paths, t0, c0)
        return key

    def _unpend(self, key):
        n = self.pending.get(key, 0) - 1
        if n > 0:
            self.pending[key] = n
        else:
            self.pending.pop(key, None)

    def _counts(self):
        f = self._f
        return (f.bytes, f.fetched, f.seeded) if f is not None else (0, 0, 0)

    def _log_region(self, key, paths, t0, c0):
        if self.remote:
            (b1, n1, s1), dt = self._counts(), max(time.time() - t0, 1e-6)
            mb, got, seeded = (b1 - c0[0]) / (1 << 20), n1 - c0[1], s1 - c0[2]
            d = self.disk()
            self.log(f"rvsm cache: region {key} {len(paths)} shards in {dt:.1f}s: {got} fetched "
                     f"({mb:.0f} MiB, {mb / dt:.1f} MiB/s), {seeded} from the seed "
                     f"({d['rolling_gb']:.1f} GB buffered of a {d['budget_gb']:.0f} GB budget, "
                     f"{d['linked_gb']:.1f} GB linked from the seed at no extra disk)")

    def disk(self):
        """What the mirror costs on disk, in GiB: `rolling` (charged, evicted down to the budget) and
        `linked` (hard links into the seed: the SAME inodes as the seed's files, so `du` of the mirror
        counts them but the filesystem does not a second time, and evicting them would free nothing
        while the seed exists -- paris4: `du ct/` 197 GB = 20 rolling + 174 linked, against a 184 GB
        seed that holds the linked bytes once)."""
        g = 1 << 30
        linked = tuple(self.linked)     # one C-level copy: `_log_region` runs outside the caller's lock
        return {"rolling_gb": round(self.cache_bytes / g, 2), "budget_gb": round(self.budget / g, 2),
                "linked_gb": round(sum(self.size.get(p, 0) for p in linked) / g, 2),
                "linked": len(linked)}

    def release(self, key):
        """The region is done with: its shards become evictable (the ones no other live region holds)."""
        for p in self.regions.pop(str(key), ()):
            h = self.hold.get(p)
            if h is not None:
                h.discard(str(key))
        self.evict()

    def held(self, path):
        return bool(self.hold.get(path))

    def lease(self, keys):
        """The regions the trainer's workers are reading or about to read (their region keys). A leased
        region's shards are never evicted, whether or not the producer still holds it. Replaces the
        previous set."""
        self.leased = {str(k) for k in keys}

    def missing(self, key):
        """Has the region `key` lost any of its shards (evicted since it was fetched, or never fetched
        by this process)? A local volume never misses anything."""
        if not self.remote:
            return False
        paths = self.paths_of(key)
        return paths is None or not resident(paths)

    def _run_leases(self):
        """The region keys the run directory's cursor files lease right now (`run.cursor_leases`)."""
        from rvsm import run
        return [self.region_key(lo) for lo in run.cursor_leases(self.root)]

    def paths_of(self, key):
        """The shard paths of region `key`: as fetched, or -- a region leased but never fetched by this
        process (a restart) -- computed from the key with the last `fetch_region` arguments."""
        key = str(key)
        got = self._paths_of.get(key)
        if got is None and self.region_args is not None:
            try:
                lo = tuple(int(v) for v in key.split("_"))
            except ValueError:
                return None
            ctx, region, rung = self.region_args
            got = self._paths_of[key] = region_paths(self.levels(), lo, ctx, region, rung)
        return got

    def _leased_paths(self):
        """The paths no eviction may take: the leases set with `lease()` (the producer's last look) AND
        the ones on disk this instant (`lease_source`) -- a stale snapshot can only protect more."""
        keys = set(self.leased)
        if self.lease_source is not None:
            try:
                self._live = {str(k) for k in self.lease_source()}
            except Exception as e:  # noqa: BLE001  -- keep the last live set rather than none
                self.log(f"rvsm cache: lease read failed, keeping the last set: {e!r}")
            keys |= getattr(self, "_live", set())
        out = set()
        for k in keys:
            out.update(self.paths_of(k) or ())
        return out

    def evict(self):
        """Delete released shards, oldest reference first, down to 90 % of the budget. A pinned level,
        a linked seed shard, a shard a live region still holds and a shard of a region the trainer has
        leased are never candidates, so a budget smaller than the working set simply does not shrink --
        which is the honest failure, not a cache that deletes what is in use."""
        if not self.remote or self.cache_bytes <= self.budget:
            return 0
        target, n = 0.9 * self.budget, 0
        keep = self._leased_paths()
        for k in list(self.pending):        # a region being downloaded outside the lock
            keep.update(self.paths_of(k) or ())
        for p, _i in sorted(self.ref.items(), key=lambda q: q[1]):
            if self.cache_bytes <= target:
                break
            if p in self.pin or p in self.linked or self.held(p) or p in keep:
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
        g = 1 << 30
        with self._clients_lock:
            fs = [c["f"] for c in self._clients if c["f"] is not None]
        f = type("F", (), {k: sum(getattr(x, k, 0) for x in fs)
                           for k in ("fetched", "seeded", "have", "absent", "failed")})
        return {"gb": self.cache_bytes / g, "shards": len(self.size), "pinned": len(self.pin),
                "rolling_gb": self.cache_bytes / g,
                "pinned_gb": sum(self.size.get(p, 0) for p in self.pin) / g,
                "linked_gb": sum(self.size.get(p, 0) for p in self.linked) / g,
                "linked": len(self.linked), "leased": len(self.leased),
                "inventoried": self.inventoried,
                "regions": len(self.regions), "evicted": self.evicted,
                "fetched": getattr(f, "fetched", 0), "seeded": getattr(f, "seeded", 0), "have": getattr(f, "have", 0),
                "absent": getattr(f, "absent", 0), "failed": getattr(f, "failed", 0)}
