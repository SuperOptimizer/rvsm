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
    g = 1
    while os.path.isdir(gen_path(base, g)):
        if is_done(gen_path(base, g)):
            best = g
        g += 1
    return best


def current_path(root, channel, lo, round_=0):
    """The path a READER uses: the newest finished generation (generation 0's path when none is)."""
    return gen_path(store_path(root, channel, lo, round_), max(store_gen(root, channel, lo, round_), 0))


def read_attrs(path):
    """A store's attributes from its zarr.json, without opening the array ({} when unreadable)."""
    try:
        import json
        with open(os.path.join(path, "zarr.json")) as f:
            return dict(json.load(f).get("attributes", {}))
    except Exception:  # noqa: BLE001
        return {}


def out_array(path, shape, origin, rung=2, channels=("recto",), q=8, volume="", umbilicus="", attrs=None):
    """Create the store (overwriting); returns the zarr array. Shape must be multiples of 128."""
    import zarr
    ladder.require_volcomp()
    from volcomp_zarr import VolcompCodec
    shape = tuple(int(s) for s in shape)
    assert all(s % CHUNK == 0 for s in shape), f"store shape must be multiples of {CHUNK}: {shape}"
    z = zarr.create_array(path, shape=shape, chunks=(CHUNK,) * 3, shards=shard_shape(shape), dtype="uint8",
                          fill_value=0, overwrite=True, serializer=VolcompCodec(q=int(q)), compressors=None)
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
