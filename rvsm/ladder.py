"""The rung ladder: one CT pyramid, read at any rung, with the coarse rungs cached whole.

Rung k has voxel size `0.6 * 2^k` micrometres, so a 2.4 um scan's level 0 is rung 2 and rung 11 is
1228.8 um. A CT pyramid here is always the plain mirror layout -- a zarr group whose children are named
by INTEGER level index, `0` the native grid -- so level `l` of a `<native> um` volume is rung
`um_rung(native) + l`. There are no micron-named levels and no mask pyramids: rvsm reads the CT and
nothing else.

Rungs ABOVE the top of the pyramid are not an error: they are 2x mean-pooled from the highest level that
exists, which keeps the physical extent of a cube the same while the scroll shrinks inside it. That is
what lets a sample at rung 2 carry nine context cubes up to rung 11 on a pyramid that stops at rung 8.

volcomp is REQUIRED. usrm2 swallowed an import failure and then silently dropped every level the codec
could not open, which looks exactly like a short pyramid; here a missing or unloadable `libvolcomp.so`
raises before anything is read.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

RUNG0_UM = 0.6      # the voxel size of rung 0, micrometres
NRUNGS = 12         # rungs 0 .. 11
NCTX = 9            # context cubes of a sample: rungs k+1 .. k+9

CACHE_VOX = 48 << 20      # a level of at most this many voxels is decoded once and kept whole
CACHE_BUDGET = 192 << 20  # ... up to this many cached bytes per process

LEVEL_CACHE: dict = {}    # {f"{level dir}#{rung}": ndarray} -- whole decoded levels
PYR_CACHE: dict = {}      # {group base: {rung: zarr array}}

_INT = re.compile(r"[0-9]+$")


def rung_um(k):
    """Voxel size of rung k, in micrometres."""
    return RUNG0_UM * 2.0 ** int(k)


def um_rung(um):
    """The rung a voxel size belongs to (nearest in log2): 2.4 -> 2, 9.6 -> 4, 1228.8 -> 11."""
    return int(round(math.log2(float(um) / RUNG0_UM)))


def native_um(base):
    """The native voxel size of a pyramid, from its volume name ('...-2.400um-...'); 2.4 um by default."""
    name = os.path.basename(str(base).rstrip("/"))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)um", name)
    return float(m.group(1)) if m else 2.4


def require_volcomp():
    """Import volcomp_zarr (which registers the "volcomp" codec) or raise with the reason.

    Every CT volume rvsm reads is volcomp-encoded, so a build without the shared library cannot read the
    data at all -- it must say so instead of returning a pyramid with holes in it."""
    lib = os.environ.get("VOLCOMP_LIB")
    if lib and not os.path.exists(lib):
        raise RuntimeError(f"VOLCOMP_LIB={lib!r} does not exist; point it at a built libvolcomp.so "
                           f"(cmake --preset release && cmake --build --preset release --target volcomp_shim)")
    try:
        import volcomp_zarr  # noqa: F401  (registers the "volcomp" codec with zarr)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "volcomp_zarr could not be imported, so the CT pyramid cannot be decoded: "
            f"{e!r}. Install volcomp-zarr and build libvolcomp.so, then set VOLCOMP_LIB=/path/to/"
            "libvolcomp.so if it does not sit beside the package.") from e
    return True


def is_url(p):
    return "://" in str(p)


def open_zarr(path):
    """A zarr array or group, local or over https, with the volcomp codec registered.

    Unlike usrm2's, this never maps between mirrors and origins: the path given is the path used."""
    import zarr
    require_volcomp()
    path = str(path)
    if is_url(path):  # streamed: more chunk fetches in flight, and a stalled request fails, not hangs
        import aiohttp
        zarr.config.set({"async.concurrency": 64})
        return zarr.open(path, mode="r",
                         storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=300)}})
    return zarr.open(path, mode="r")


def array_dir(arr):
    """The directory (or URL) a zarr array was opened from -- the key the level cache uses."""
    p = str(getattr(arr.store, "root", "") or arr.store_path)
    p = p[len("file://"):] if p.startswith("file://") else p
    sub = getattr(arr, "path", "") or ""
    return os.path.join(p, sub) if sub and not p.endswith(sub) else p


def pyramid_base(path):
    """'<name>.zarr/0' -> the group '<name>.zarr'; anything else is already a group."""
    p = str(path).rstrip("/")
    head, _, last = p.rpartition("/")
    return head if head and _INT.fullmatch(last) else p


def _read_json(path):
    """`<base>/zarr.json` as a dict, local or over https, or None."""
    try:
        if is_url(path):
            import urllib.request
            with urllib.request.urlopen(path, timeout=30) as f:  # noqa: S310 (a bucket URL by construction)
                return json.loads(f.read().decode("utf-8"))
        return json.load(open(path)) if os.path.exists(path) else None
    except Exception:  # noqa: BLE001
        return None


def _level_names(base):
    """The integer level names of a pyramid group, from its OME multiscales when it has them and by
    probing `0, 1, 2, ...` when it does not (the only way to list a directory over plain HTTP)."""
    j = _read_json(f"{base}/zarr.json") or {}
    at = j.get("attributes", j) if isinstance(j, dict) else {}
    ms = ((at.get("ome") or at).get("multiscales") or [None])[0] if isinstance(at, dict) else None
    names = [str(d["path"]) for d in (ms or {}).get("datasets", [])] if ms else []
    names = [n for n in names if _INT.fullmatch(n)]
    if names:
        return sorted(names, key=int)
    if not is_url(base) and os.path.isdir(base):
        return sorted((d for d in os.listdir(base) if _INT.fullmatch(d) and os.path.isdir(f"{base}/{d}")),
                      key=int)
    out = []
    for l in range(NRUNGS):  # noqa: E741
        if _read_json(f"{base}/{l}/zarr.json") is None:
            if out:
                break
            continue
        out.append(str(l))
    return out


def rungs(base):
    """{rung: zarr array} of a CT pyramid, given its group (or one of its levels), URL or local path.

    Level `l` is rung `um_rung(native_um(base)) + l`. Only the levels that exist are returned; rungs
    above the top are produced on demand by `read_rung` / `full_level`."""
    base = pyramid_base(str(base))
    if base in PYR_CACHE:
        return PYR_CACHE[base]
    require_volcomp()
    k0 = um_rung(native_um(base))
    names = _level_names(base)
    if not names:
        raise RuntimeError(f"{base}: no integer pyramid levels found (a CT pyramid is a zarr group whose "
                           f"children are named 0, 1, 2, ...)")
    out = {}
    for n in names:
        out[k0 + int(n)] = open_zarr(f"{base}/{n}")
    PYR_CACHE[base] = out
    return out


def base_rung(path):
    """The rung of the level a path names ('.../x.zarr/0'), or the finest rung of the group."""
    p = str(path).rstrip("/")
    base = pyramid_base(p)
    last = p.rpartition("/")[2]
    if base != p and _INT.fullmatch(last):
        return um_rung(native_um(base)) + int(last)
    return min(rungs(base))


def rung_shape(pyr, k):
    """The (Z,Y,X) shape a pyramid has at rung k (pooled from the highest rung below when k is above it)."""
    src = max(r for r in pyr if r <= k)
    return -(-np.array(pyr[src].shape[-3:], np.int64) // (1 << (k - src)))


def shape3(p):
    """A scalar or a triple -> an int64 (3,) shape."""
    a = np.asarray(p, np.int64)
    return np.repeat(a, 3) if a.ndim == 0 else a


def pool2(v):
    """2x mean pooling of a uint8 volume, zero-padded to an even shape (as `read_rung` pools)."""
    s = np.array(v.shape, np.int64)
    n = -(-s // 2)
    if (s % 2).any():
        v = np.pad(v, [(0, int(q)) for q in n * 2 - s])
    return v.reshape(n[0], 2, n[1], 2, n[2], 2).astype(np.float32).mean((1, 3, 5)).astype(np.uint8)


def _cache_bytes():
    return sum(v.nbytes for v in LEVEL_CACHE.values())


def _read(arr, sl):
    """`arr[sl]` as uint8, tolerating a (1,Z,Y,X) store."""
    v = arr[(0,) + tuple(sl)] if arr.ndim == 4 else arr[tuple(sl)]
    return np.asarray(v, np.uint8)


def full_level(pyr, k):
    """The WHOLE level at rung k as a cached uint8 array, or None when it is too big to keep.

    Rungs above the top of the pyramid are pooled from the cached level below, so a sample's nine context
    cubes do not re-read (and re-decode) the top of the pyramid once per rung. Pooling in 2x steps
    truncates to uint8 at each step, so a cached coarse rung may differ from a direct pool by at most one
    grey level."""
    src = max((r for r in pyr if r <= k), default=None)
    if src is None:
        return None
    key = f"{array_dir(pyr[src])}#{k}"
    if key in LEVEL_CACHE:
        return LEVEL_CACHE[key]
    n = int(np.prod(rung_shape(pyr, k)))
    if n > CACHE_VOX or _cache_bytes() + n > CACHE_BUDGET:
        return None
    if k == src:
        v = np.ascontiguousarray(_read(pyr[src], (slice(None),) * 3))
    else:
        below = full_level(pyr, k - 1)
        if below is None:  # the level below is too big to keep: pool it in z slabs, keep only the result
            a, e = pyr[src], 1 << (k - 1 - src)
            S = np.array(a.shape[-3:], np.int64)
            below = np.zeros(tuple(-(-S // e)), np.uint8)
            step = max(e, (1 << 24) // max(int(S[1] * S[2]), 1) // e * e)
            for z in range(0, int(S[0]), step):
                blk = _read(a, (slice(z, z + step), slice(None), slice(None)))
                if e > 1:
                    m = -(-np.array(blk.shape, np.int64) // e)
                    pad = m * e - np.array(blk.shape)
                    if pad.any():
                        blk = np.pad(blk, [(0, int(q)) for q in pad])
                    blk = blk.reshape(m[0], e, m[1], e, m[2], e).astype(np.float32).mean((1, 3, 5)).astype(np.uint8)
                below[z // e:z // e + blk.shape[0]] = blk
        v = pool2(below)
    LEVEL_CACHE[key] = v
    return v


def read_block(pyr, k, lo, hi):
    """pyr[k][lo:hi] as uint8 (from the cached whole level when there is one)."""
    a = full_level(pyr, k)
    if a is None:
        return _read(pyr[k], (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])),
                              slice(int(lo[2]), int(hi[2]))))
    return a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def read_rung(pyr, k, lo, p, dtype=np.float32):
    """The (Z,Y,X) cube of a pyramid at rung k, corner `lo`, size `p` (both in rung-k voxels).

    A rung above the top of the pyramid is made by 2^d mean pooling the highest rung that exists -- the
    physical extent is the same, so the scroll simply shrinks inside the cube. Outside the array = 0
    (air). Only the part of the source that overlaps the array is read: a rung far above the top would
    otherwise address (p * 2^d)^3 voxels."""
    src = max((r for r in pyr if r <= k), default=None)
    if src is None:
        raise ValueError(f"pyramid has no rung at or below {k} (has {sorted(pyr)})")
    p, lo = shape3(p), np.asarray(lo, np.int64)
    cube = np.zeros(tuple(p), dtype)
    full = full_level(pyr, k)
    if full is not None:
        S = np.array(full.shape, np.int64)
        a, b = np.maximum(lo, 0), np.minimum(lo + p, S)
        if (b > a).all():
            s = a - lo
            blk = full[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
            cube[s[0]:s[0] + blk.shape[0], s[1]:s[1] + blk.shape[1], s[2]:s[2] + blk.shape[2]] = blk
        return cube
    e = 1 << (k - src)
    S = np.array(pyr[src].shape[-3:], np.int64)
    jlo = np.maximum(-lo, 0)                       # the rung-k voxels of the cube that touch the array
    jhi = np.minimum(-(-S // e) - lo, p)
    if (jhi <= jlo).any():
        return cube
    a, b = (lo + jlo) * e, np.minimum((lo + jhi) * e, S)
    blk = read_block(pyr, src, a, b)
    if e > 1:
        n = jhi - jlo
        pad = n * e - (b - a)
        if pad.any():                              # past the end of the array: pooled against air
            blk = np.pad(blk, [(0, int(q)) for q in pad])
        blk = blk.reshape(n[0], e, n[1], e, n[2], e).astype(np.float32).mean((1, 3, 5))
    cube[jlo[0]:jhi[0], jlo[1]:jhi[1], jlo[2]:jhi[2]] = blk.astype(dtype, copy=False)
    return cube


def context(volume, origin, shape, ctx, rung=None, dtype=np.uint8):
    """Coarse cubes of the SAME size centred on the same point as the patch at `origin`/`shape`:
    one (Z,Y,X) cube per OFFSET in `ctx` (1 = one rung coarser, 2 = two rungs, ...). `origin`/`shape`
    are voxels of rung `rung` (default: the rung `volume` names). Outside the pyramid = 0 (air)."""
    pyr, out = rungs(volume), []
    k0 = base_rung(volume) if rung is None else int(rung)
    c0 = np.asarray(origin, np.int64) + shape3(shape) // 2   # centre, rung-k0 voxels
    for d in ctx:
        lo = c0 // (1 << int(d)) - shape3(shape) // 2        # the same centre, in rung-(k0+d) voxels
        out.append(read_rung(pyr, k0 + int(d), lo, shape, dtype=dtype))
    return out


def clear_caches():
    """Drop the decoded-level and pyramid caches (tests, and a producer changing volume)."""
    LEVEL_CACHE.clear()
    PYR_CACHE.clear()
