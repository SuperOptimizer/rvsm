"""Region stores: the only thing the trainer ever reads as a target, and what every producer writes.

A store is a zarr v3 array of uint8, 128^3 inner chunks, ONE shard (= one data file) per <= 1024^3 box, inner
codec chain exactly [volcomp] (`compressors=None`: zstd after volcomp measured 0.2% for a decode step on every
read). `q` is volcomp's quantisation step: 8 for probabilities (lossy: a stored 0 may read back non-zero,
harmless for a probability), 0 (lossless) for any field whose code 0 means "no data" or whose codes are a
distance in fixed units. A lossy q compounds under partial-chunk writes, so a shard region is filled in ONE
write. A finished store is never rewritten in place: a new round is a new directory. `done` is set last.
"""
import hashlib
import os
import shutil

import numpy as np

from rvsm import ladder

CHUNK = 128
SHARD = 1024
_BUILD = None


def volcomp_build():
    """sha256 of the libvolcomp.so in use (the same bytes decode slightly differently across builds)."""
    global _BUILD
    if _BUILD is None:
        lib = os.environ.get("VOLCOMP_LIB", "")
        _BUILD = hashlib.sha256(open(lib, "rb").read()).hexdigest()[:16] if lib and os.path.exists(lib) else "unknown"
    return _BUILD


def shard_shape(shape, chunk=CHUNK, cap=SHARD):
    """One shard per `cap`^3 box, a multiple of the chunk and covering the array (a store smaller than `cap`
    on an axis is a single shard)."""
    return tuple(min(cap, -(-int(s) // chunk) * chunk) for s in shape)


def region_name(lo):
    return "region_%d_%d_%d.zarr" % tuple(int(v) for v in lo)


def parse_region_name(name):
    """`region_<z>_<y>_<x>.zarr` -> ((z, y, x), 0), `region_<z>_<y>_<x>.g<N>.zarr` -> ((z, y, x), N),
    anything else (a `.tmp`, a `.gc-<ts>` being removed, a bundle file) -> None."""
    n = str(name)
    if not (n.startswith("region_") and n.endswith(".zarr")):
        return None
    stem, g = n[len("region_"):-len(".zarr")], 0
    if "." in stem:
        stem, _, tail = stem.partition(".")
        if not (tail.startswith("g") and tail[1:].isdigit() and int(tail[1:]) > 0):
            return None
        g = int(tail[1:])
    try:
        z, y, x = (int(v) for v in stem.split("_"))
    except ValueError:
        return None
    return (z, y, x), g


def store_path(root, channel, lo, round_=0):
    """<root>/stores/round_<r>/<channel>/region_<z>_<y>_<x>.zarr"""
    return os.path.join(root, "stores", f"round_{int(round_)}", str(channel), region_name(lo))


def gen_path(path, gen=0):
    """A store's GENERATION `gen` path: generation 0 is the store itself, generation g > 0 sits beside
    it as `region_<z>_<y>_<x>.g<g>.zarr`. A regenerated store (round 0's verso, rewritten once from a
    better student) is a NEW directory; a finished store is never rewritten in place."""
    g = int(gen)
    if g <= 0:
        return path
    assert path.endswith(".zarr"), path
    return path[:-len(".zarr")] + f".g{g}.zarr"


def store_gen(root, channel, lo, round_=0):
    """The highest FINISHED generation of a region store, or -1 when none is."""
    base = store_path(root, channel, lo, round_)
    best = 0 if is_done(base) else -1
    for g in gens(base):
        if is_done(gen_path(base, g)):
            best = max(best, g)
    return best


def gens(base):
    """The generations g >= 1 that have a directory beside the store `base`, ascending. Generations
    may have gaps: a region's verso and recto regenerations share one counter (`next_gen`). Probed up
    to GEN_MAX (a stat each: the channel directories hold thousands of stores, a listdir per call was
    the slower way)."""
    return [g for g in range(1, GEN_MAX + 1) if os.path.isdir(gen_path(base, g))]


GEN_MAX = 8     # a region is regenerated a handful of times at most (the verso once, the recto per switch)


BUNDLED = ("verso", "midline", "thickness")   # the channels a verso regeneration replaces TOGETHER
TEACHER_BUNDLED = ("recto", "rw", "band")     # ... and what a teacher regeneration replaces together: the
                                              # recto + rw pair, and a ROUTED generation's base-teacher band
                                              # (`config.route_spec`; absent otherwise)


def is_bundled(channel):
    c = str(channel)
    return any(c == b or c.startswith(b + "_r") for b in BUNDLED)


def _bundle_file(root, lo, round_):
    return os.path.join(str(root), "stores", f"round_{int(round_)}", "bundle", region_name(lo)[:-5] + ".json")


def bundle_state(root, lo, round_=0):
    """The region's COMMITTED generations, what every reader uses: {"gen": the fields' generation,
    "verso": the verso's, "recto": the recto + rw pair's}. A bundle file written before the recto
    regeneration existed is {"gen": g}, meaning verso = fields = g and recto = 0; no file is all 0."""
    try:
        import json
        with open(_bundle_file(root, lo, round_)) as f:
            d = dict(json.load(f))
    except (OSError, ValueError, TypeError):
        d = {}
    g = int(d.get("gen", 0))
    return {"gen": g, "verso": int(d.get("verso", g)), "recto": int(d.get("recto", 0))}


def bundle_gen(root, lo, round_=0):
    """The COMMITTED generation of a region's field stores (midline*, thickness*): 0 until a
    regenerated bundle is complete and committed (`commit_bundle`)."""
    return bundle_state(root, lo, round_)["gen"]


def commit_bundle(root, lo, round_, gen, verso=None, recto=None, **info):
    """Make generation `gen` of the region's fields -- with its verso at generation `verso` (default
    `gen`) and its recto + rw at `recto` (default: the pair committed now) -- the ones readers use.
    Written ONLY once every store it names is finished, so a reader never mixes a new verso or recto
    with fields built from the old one (pass-4 P4-04). Atomic (tmp + rename)."""
    import json
    cur = bundle_state(root, lo, round_)
    rec = {"gen": int(gen), "verso": int(gen if verso is None else verso),
           "recto": int(cur["recto"] if recto is None else recto), **info}
    p = _bundle_file(root, lo, round_)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(p + ".tmp", p)
    return p


def next_gen(root, lo, round_=0):
    """A generation number no store of the region uses yet: one past the highest generation DIRECTORY
    (finished or not) of its recto, rw and verso. Both regenerations (the verso's, the recto's) write
    there, so the fields rebuilt for them -- at max(verso gen, recto gen), `targets.field_path` -- land
    in a fresh directory and never over committed ones."""
    top = 0
    for ch in TEACHER_BUNDLED + ("verso",):
        top = max([top] + gens(store_path(root, ch, lo, round_)))
    assert top < GEN_MAX, f"region {tuple(lo)}: generation {top + 1} is past GEN_MAX"
    return top + 1


def current_path(root, channel, lo, round_=0):
    """The path a READER uses: the committed generation (`bundle_state`) of the verso, of a field
    channel (midline*, thickness*) and of the teacher pair (recto, rw); any other channel has one
    generation."""
    base = store_path(root, channel, lo, round_)
    c = str(channel)
    if c in TEACHER_BUNDLED:
        return gen_path(base, bundle_state(root, lo, round_)["recto"])
    if c == "verso" or c.startswith("verso_r"):
        return gen_path(base, bundle_state(root, lo, round_)["verso"])
    return gen_path(base, bundle_gen(root, lo, round_)) if is_bundled(channel) else base


def read_attrs(path):
    """A store's attributes from its zarr.json, without opening the array ({} when unreadable)."""
    try:
        import json
        with open(os.path.join(path, "zarr.json")) as f:
            return dict(json.load(f).get("attributes", {}))
    except Exception:  # noqa: BLE001
        return {}


ENCODE_THREADS = max(int(os.environ.get("RVSM_ENCODE_THREADS", "4")), 0)
_ENC = {}


def _codec(q):
    """The volcomp serializer a store is written with: `VolcompCodec(q)`, whose chunk encodes run on a
    pool of `ENCODE_THREADS` threads (RVSM_ENCODE_THREADS; 0 = the codec as it is).

    The codec's `_encode_single` is a coroutine that makes a blocking ctypes call, so under zarr's one
    event loop every chunk of a store was encoded one after another (a 1024^3 q=0 field store: ~9 s on
    the laptop, and two stores on two Python threads took exactly as long as in turn). Here the SAME
    coroutine runs to completion on a pool thread (ctypes releases the GIL), so zarr's concurrent map
    encodes chunks in parallel (laptop, 1024^3: a q=0 field store 8.2 -> 3.0 s, a q=8 probability
    store 3.8 -> 2.75 s with 4 threads; 8 were no faster). The chunk bytes are the codec's own and
    the sharding codec lays them out by chunk index, not completion order: byte-identical stores, the
    same zarr.json (the subclass serialises as "volcomp")."""
    from volcomp_zarr import VolcompCodec
    if ENCODE_THREADS <= 0:
        return VolcompCodec(q=int(q))
    cls = _ENC.get("cls")
    if cls is None:
        import asyncio
        import concurrent.futures as cf
        pool = _ENC["pool"] = cf.ThreadPoolExecutor(ENCODE_THREADS, thread_name_prefix="rvsm-enc")
        base = VolcompCodec._encode_single

        class PooledVolcompCodec(VolcompCodec):
            async def _encode_single(self, chunk_array, chunk_spec):
                return await asyncio.get_running_loop().run_in_executor(
                    pool, lambda: asyncio.run(base(self, chunk_array, chunk_spec)))
        cls = _ENC["cls"] = PooledVolcompCodec
    return cls(q=int(q))


def out_array(path, shape, origin, rung=2, channels=("recto",), q=8, volume="", umbilicus="", attrs=None):
    """Create the store (overwriting); returns the zarr array. Shape must be multiples of 128."""
    import zarr
    ladder.require_volcomp()
    shape = tuple(int(s) for s in shape)
    assert all(s % CHUNK == 0 for s in shape), f"store shape must be multiples of {CHUNK}: {shape}"
    z = zarr.create_array(path, shape=shape, chunks=(CHUNK,) * 3, shards=shard_shape(shape), dtype="uint8",
                          fill_value=0, overwrite=True, serializer=_codec(q), compressors=None)
    z.attrs.update({"channels": [str(c) for c in channels], "voxel_um": ladder.rung_um(rung), "rung": int(rung),
                    "volcomp_q": int(q), "origin_zyx": [int(v) for v in origin], "scale": 1.0,
                    "volume": str(volume), "umbilicus": str(umbilicus), "volcomp_build": volcomp_build(),
                    "done": False, **(attrs or {})})
    return z


def u8(prob):
    return np.clip(np.rint(np.asarray(prob, np.float32) * 255), 0, 255).astype(np.uint8)


def write(path, u8_block, origin, rung=2, channels=("recto",), q=8, attrs=None, **kw):
    """Write a finished uint8 block as a store atomically: <path>.tmp -> rename, `done` last."""
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    a = out_array(tmp, u8_block.shape, origin, rung=rung, channels=channels, q=q, attrs=attrs, **kw)
    a[:] = np.ascontiguousarray(u8_block, dtype=np.uint8)
    a.attrs["done"] = True
    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp, path)
    return path


def is_done(path):
    try:
        import json
        with open(os.path.join(path, "zarr.json")) as f:
            return bool(json.load(f).get("attributes", {}).get("done"))
    except Exception:
        return False


def open_store(path):
    a = ladder.open_zarr(path)
    if not a.attrs.get("done"):
        raise FileNotFoundError(f"store not done: {path}")
    return a


def read_store(a, k, lo, shape):
    """(cube, inside) of a region store read at rung k: k = the store's rung, or k = rung+1 as its 2x mean
    pool. `lo`/`shape` in rung-k voxels; `inside` marks the voxels the store covers."""
    k0 = int(a.attrs["rung"])
    d = int(k) - k0
    assert 0 <= d <= 1, f"read_store: rung {k} from a rung-{k0} store"
    o, S = np.array(a.attrs["origin_zyx"], np.int64), np.array(a.shape, np.int64)
    p3 = np.array(shape, np.int64)
    lo2, n2 = (np.asarray(lo, np.int64) << d) - o, p3 << d
    out = np.zeros(tuple(n2), np.uint8)
    aa, bb = np.maximum(lo2, 0), np.minimum(lo2 + n2, S)
    inside = np.zeros(tuple(n2), bool)
    if (bb > aa).all():
        sl = tuple(slice(int(x), int(y)) for x, y in zip(aa, bb))
        blk = np.asarray(a[sl], np.uint8)
        st = aa - lo2
        out[st[0]:st[0] + blk.shape[0], st[1]:st[1] + blk.shape[1], st[2]:st[2] + blk.shape[2]] = blk
        inside[st[0]:st[0] + blk.shape[0], st[1]:st[1] + blk.shape[1], st[2]:st[2] + blk.shape[2]] = True
    for _ in range(d):
        out = ladder.pool2(out)
        inside = ladder.pool2(inside.astype(np.uint8) * 255) > 127
    return out, inside
