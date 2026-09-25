#!/usr/bin/env python3
"""m7 over WHOLE CT volumes: one seamless probability store per volume, pooled, uploaded.

The m7 surface teacher (surface_m7_nnunet, Kaggle-winning nnU-Net ResEncL, 192^3 windows, CT norm clip
0..212) is run at LEVEL 0 of a ~8-9 um masked CT volume streamed from dl.ash2txt.org, and written as a
multiscale zarr v3 store: uint8 probability (p * 255), sharded, inner codec chain exactly [volcomp q=8].

Seamless by construction. The volume has ONE global window grid (starts 0, stride, ..., last = n - w on
every axis) and ONE global separable Gaussian weight normalisation: a window at start s along an axis is
weighted by g(z - s) / sum_s' g(z - s'), the sum over the WHOLE axis grid. The blended value of a voxel
therefore does not depend on which work unit, slab or shard computed it, and the output needs no weight
sum at all (the normalised weights of the windows covering a voxel sum to one). Windows whose CT is all
air are skipped: a voxel with CT > 0 is only ever covered by non-air windows, so the skip changes nothing
where the scroll is, and the output is masked to CT > 0 anyway.

Work layout. A volume is cut into UNITS of up to `--rows` shard rows (1024 z each). A worker (one per GPU)
claims units in queue order (smallest volume first), runs every window row whose z-extent touches the
unit, and finalises only the unit's own z range, so neighbouring units overlap by at most two window
rows of compute and never in output. Inside a unit, a window row (z start s) is processed y-row by
y-row: an fp32 GPU accumulator of (w, w, X) rolls along y, and a y-slice is added to the fp16 host
accumulator of (w, Y, X) once it is final for this row; the host accumulator rolls along z. Finished z
goes to a uint8 staging buffer of one shard row, and when a shard row is complete every shard of it is
encoded and written ONCE, together with its 2x mean-pooled levels 1..3 (a level-k shard of side 1024 >> k
is exactly the pool of one level-0 shard). Levels 4..7 are pooled from level 3 by the finisher.

Resume. `state/<vol>/row_<r>.done` is written after every shard of shard row r (all levels 0..3) is on
disk; a restarted unit skips its done rows (and recomputes only the window rows it needs for the rest).

Finisher (CPU process): levels 4..7, group + array metadata, upload with ONE sftp session per batch into
`<store>.part` then a rename, an HTTP size check of every file, then the local store is deleted and a
marker written under ~/m7_wholevol/uploaded/.

Subcommands: worker, finisher, status, seamcheck, bench.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE_URL = "https://dl.ash2txt.org/community-uploads/forrest/volcomp"
SFTP_ROOT = "/volcomp"
WORK = os.environ.get("M7W_WORK", "/vesuvius/m7_wholevol")
HOME = os.path.expanduser(os.environ.get("M7W_HOME", "~/m7_wholevol"))
CKPT = "/vesuvius/tsm/models/surface_m7_nnunet.pth"
PLAN_GLOB = "/vesuvius/tsm/models/trt/m7_p192_b1_fp16_*.plan"

W = 192                       # the m7 window
CH = 128                      # inner chunk
SH = 1024                     # level-0 shard side
NLEV = 8                      # levels 0..7
MEAN, STD, CLO, CHI = 87.54424285888672, 47.74376678466797, 0.0, 212.0   # m7 CTNormalization
Q = float(os.environ.get("M7W_Q", "8"))   # 8 for production; 0 (lossless) only for tests
RUN_TS = os.environ.get("M7W_RUN_TS", "20260925170000")   # the run id in the store name


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- volumes
def load_vols(path=None):
    path = path or os.path.join(HOME, "volumes.json")
    out = []
    for v in json.load(open(path)):
        sample, name = v["sample"], v["name"]
        k = int(v.get("level", 0))            # the CT pyramid level m7 runs at (0 for the ~8-9 um scans)
        shape0 = [int(s) for s in v["shape"]]
        shape = [int(s) for s in v.get("shape_level", shape0)]
        native = float(name.split("-")[1].replace("um", ""))
        stem = f"{sample}_{name.split('-')[0]}" + (f"_L{k}" if k else "")
        out.append(dict(sample=sample, name=name, level=k, shape=shape, shape0=shape0, native_pitch=native,
                        pitch=native * (1 << k), stem=stem, tier=v.get("tier"),
                        url=f"{BASE_URL}/{sample}/volumes/{name}.zarr"))
    return out


def store_name(v):
    return f"{v['name'].split('-')[0]}-surface-{RUN_TS}-surface-m7-L{v.get('level', 0)}-prob.zarr"


def remote_dir(v):
    return f"{SFTP_ROOT}/{v['sample']}/representations/predictions/surfaces"


def lvl_path(pitch, k):
    return ("%.3f" % (pitch * (1 << k))).rstrip("0").rstrip(".")


def lvl_shape(shape, k):
    s = list(shape)
    for _ in range(k):
        s = [-(-x // 2) for x in s]
    return s


def lvl_shard(k):
    return max(CH, SH >> k)


def units_of(v, rows_per_unit):
    nr = -(-v["shape"][0] // SH)
    return [(r, min(r + rows_per_unit, nr)) for r in range(0, nr, rows_per_unit)]


# --------------------------------------------------------------------------- volcomp (ctypes, no copies)
def _vc():
    import volcomp_zarr._lib as L
    return L


class Codec:
    def __init__(self):
        L = _vc()
        self.L, self._L = L, L._L
        self.bound = int(L.ENCODE_BOUND)
        self.tls = threading.local()

    def decode_into(self, src: np.ndarray, dst: np.ndarray, smooth: float = 0.0):
        """plain decode, or (smooth > 0) volcomp's decode-only deblocking at that strength, gated face
        filter with the zero guard (masked-air zeros kept) -- what VolcompCodec does under
        set_read_smoothing(smooth); a no-op below q4 by volcomp's own rule."""
        assert dst.flags.c_contiguous and dst.nbytes == CH ** 3
        if smooth > 0:
            flags = int(self.L.DEBLOCK_ZERO_GUARD) | int(self.L.SMOOTH_GATED)
            st = self._L.volcomp_shim_decode_smooth(src.ctypes.data, src.nbytes, dst.ctypes.data, dst.nbytes,
                                                    ctypes.c_float(smooth), flags)
        else:
            st = self._L.volcomp_shim_decode(src.ctypes.data, src.nbytes, dst.ctypes.data, dst.nbytes)
        if st != 0:
            raise RuntimeError(f"volcomp decode status {st}")

    def encode(self, chunk: np.ndarray, q=None) -> bytes:
        q = Q if q is None else q
        assert chunk.flags.c_contiguous and chunk.nbytes == CH ** 3 and chunk.dtype == np.uint8
        buf = getattr(self.tls, "buf", None)
        if buf is None:
            buf = self.tls.buf = ctypes.create_string_buffer(self.bound)
        got = ctypes.c_size_t()
        st = self._L.volcomp_shim_encode(chunk.ctypes.data, ctypes.c_float(q), buf, self.bound, ctypes.byref(got))
        if st != 0:
            raise RuntimeError(f"volcomp encode status {st}")
        return buf.raw[:got.value]


def _vc_version():
    try:
        L = _vc()
        return L._L.volcomp_shim_version().decode()
    except Exception:  # noqa: BLE001
        return "unknown"


def volcomp_build():
    lib = os.environ.get("VOLCOMP_LIB", "")
    return hashlib.sha256(open(lib, "rb").read()).hexdigest()[:16] if lib and os.path.exists(lib) else "unknown"


# --------------------------------------------------------------------------- shard files
EMPTY = np.uint64(2 ** 64 - 1)


def shard_index(buf: np.ndarray, n: int):
    raw = buf[len(buf) - (n ** 3 * 16 + 4): len(buf) - 4]
    return np.frombuffer(raw.tobytes(), "<u8").reshape(n, n, n, 2)


def write_shard(path, block: np.ndarray, side: int, codec: Codec, pool: ThreadPoolExecutor | None):
    """Encode `block` (<= side^3, the in-array part of one shard) as ONE sharded file, written once
    (.part + rename). All-zero inner chunks are not stored (zarr's fill value); an all-zero shard writes
    no file. Returns bytes written."""
    import google_crc32c
    n = side // CH
    nz, ny, nx = (-(-s // CH) for s in block.shape)
    jobs = []
    for a in range(nz):
        for b in range(ny):
            for c in range(nx):
                jobs.append((a, b, c))

    def enc(abc):
        a, b, c = abc
        sub = block[a * CH:(a + 1) * CH, b * CH:(b + 1) * CH, c * CH:(c + 1) * CH]
        if not sub.any():
            return None
        if sub.shape != (CH, CH, CH):
            full = np.zeros((CH, CH, CH), np.uint8)
            full[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
        else:
            full = np.ascontiguousarray(sub)
        return codec.encode(full)

    res = list(pool.map(enc, jobs)) if pool is not None else [enc(j) for j in jobs]
    if all(r is None for r in res):
        return 0
    idx = np.full((n, n, n, 2), EMPTY, np.uint64)
    parts, off = [], 0
    for (a, b, c), r in zip(jobs, res):
        if r is None:
            continue
        idx[a, b, c] = (off, len(r))
        parts.append(r)
        off += len(r)
    ib = idx.astype("<u8").tobytes()
    crc = google_crc32c.value(ib).to_bytes(4, "little")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as f:
        for p in parts:
            f.write(p)
        f.write(ib)
        f.write(crc)
    os.replace(path + ".part", path)
    return off + len(ib) + 4


def read_block(arr_dir, shape, side, lo, hi, codec: Codec, pool=None, out=None, smooth=0.0):
    """Decode the voxel box [lo, hi) of a local sharded volcomp array (missing shard/chunk = 0)."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    out = np.zeros(tuple(hi - lo), np.uint8) if out is None else out
    n = side // CH
    c0, c1 = lo // CH, -(-hi // CH)
    jobs = [(a, b, c) for a in range(c0[0], c1[0]) for b in range(c0[1], c1[1]) for c in range(c0[2], c1[2])]
    cache = {}
    lock = threading.Lock()

    def shard(key):
        with lock:
            if key in cache:
                return cache[key]
        p = os.path.join(arr_dir, "c", *map(str, key))
        if os.path.exists(p):
            buf = np.fromfile(p, np.uint8)
            ent = (buf, shard_index(buf, n))
        else:
            ent = None
        with lock:
            cache[key] = ent
        return ent

    def dec(abc):
        a, b, c = abc
        sk = (a // n, b // n, c // n)
        ent = shard(sk)
        if ent is None:
            return
        buf, idx = ent
        o, nb = idx[a % n, b % n, c % n]
        if o == EMPTY:
            return
        tmp = np.empty((CH, CH, CH), np.uint8)
        codec.decode_into(buf[int(o):int(o) + int(nb)], tmp, smooth)
        g0 = np.array([a, b, c]) * CH
        s0 = np.maximum(lo, g0)
        s1 = np.minimum(hi, g0 + CH)
        out[s0[0] - lo[0]:s1[0] - lo[0], s0[1] - lo[1]:s1[1] - lo[1], s0[2] - lo[2]:s1[2] - lo[2]] = \
            tmp[s0[0] - g0[0]:s1[0] - g0[0], s0[1] - g0[1]:s1[1] - g0[1], s0[2] - g0[2]:s1[2] - g0[2]]

    if pool is not None:
        list(pool.map(dec, jobs))
    else:
        for j in jobs:
            dec(j)
    return out


def pool2(a: np.ndarray) -> np.ndarray:
    """2x mean pool of uint8 (odd edges: the mean of the in-array voxels), rounded."""
    z, y, x = a.shape
    Z, Y, X = -(-z // 2), -(-y // 2), -(-x // 2)
    if (z, y, x) != (2 * Z, 2 * Y, 2 * X):
        b = np.zeros((2 * Z, 2 * Y, 2 * X), np.uint8)
        b[:z, :y, :x] = a
    else:
        b = a
    out = np.empty((Z, Y, X), np.uint8)
    cz = np.full(Z, 2.0); cz[-1] = 2 - (2 * Z - z)
    cy = np.full(Y, 2.0); cy[-1] = 2 - (2 * Y - y)
    cx = np.full(X, 2.0); cx[-1] = 2 - (2 * X - x)
    for z0 in range(0, Z, 32):      # bounded temporaries
        z1 = min(z0 + 32, Z)
        s = b[2 * z0:2 * z1].reshape(z1 - z0, 2, Y, 2, X, 2).sum(axis=(1, 3, 5), dtype=np.uint16)
        cnt = cz[z0:z1, None, None] * cy[None, :, None] * cx[None, None, :]
        out[z0:z1] = np.floor(s / cnt + 0.5).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- CT source (rolling shard cache)
class CTSource:
    """Level-0 CT shards of one volume, downloaded per shard row into WORK/ct/<stem>/c/z/y/x (404 -> an
    `.absent` marker), decoded per 128-z slab, evicted once the window rows have passed them."""

    def __init__(self, v, codec: Codec, pool: ThreadPoolExecutor, conns=16, smooth=0.0):
        self.v, self.codec, self.pool = v, codec, pool
        self.smooth = float(smooth)
        self.shape = v["shape"]
        self.dir = os.path.join(WORK, "ct", v["stem"])
        self.url = v["url"] + f"/{v.get('level', 0)}"
        self.tls = threading.local()
        self.conns = conns
        self.slabs = {}          # slab index -> np.uint8 (128, Y, X)
        self.projs = {}          # slab index -> (Y, X) bool: any CT in the slab
        self.lock = threading.Lock()
        self.fetched = set()
        self.flock = threading.Lock()
        self.inflight = {}       # slab index -> Event while it is being decoded
        self.bytes = 0
        self.dl_s = 0.0
        self.dec_s = 0.0
        self.pref = None

    def _sess(self):
        s = getattr(self.tls, "s", None)
        if s is None:
            import requests
            s = requests.Session()
            a = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=4)
            s.mount("https://", a)
            self.tls.s = s
        return s

    def _get(self, key):
        p = os.path.join(self.dir, "c", *map(str, key))
        if os.path.exists(p) or os.path.exists(p + ".absent"):
            return 0
        os.makedirs(os.path.dirname(p), exist_ok=True)
        url = self.url + "/c/" + "/".join(map(str, key))
        for attempt in range(8):
            try:
                r = self._sess().get(url, timeout=120)
                if r.status_code == 404:
                    open(p + ".absent", "w").close()
                    return 0
                r.raise_for_status()
                with open(p + ".part", "wb") as f:
                    f.write(r.content)
                os.replace(p + ".part", p)
                return len(r.content)
            except Exception as e:  # noqa: BLE001
                log(f"[ct] {url}: {e.__class__.__name__}: {e} (attempt {attempt + 1})")
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"CT shard {url} failed 8 times")

    def fetch_row(self, r):
        with self.flock:
            if r in self.fetched or r * SH >= self.shape[0]:
                return
            t = time.time()
            ny, nx = -(-self.shape[1] // SH), -(-self.shape[2] // SH)
            keys = [(r, y, x) for y in range(ny) for x in range(nx)]
            with ThreadPoolExecutor(self.conns) as ex:
                self.bytes += sum(ex.map(self._get, keys))
            self.fetched.add(r)
            self.dl_s += time.time() - t

    def evict_rows_below(self, r):
        if os.environ.get("M7W_NO_EVICT"):
            return
        with self.flock:
            rows = sorted(self.fetched)
        for rr in rows:
            if rr < r:
                shutil.rmtree(os.path.join(self.dir, "c", str(rr)), ignore_errors=True)
                self.fetched.discard(rr)

    def _decode_slab(self, i):
        self.fetch_row(i * CH // SH)
        Z = self.shape[0]
        z0, z1 = i * CH, min((i + 1) * CH, Z)
        t = time.time()
        a = read_block(os.path.join(self.dir), self.shape, SH, (z0, 0, 0), (z1, self.shape[1], self.shape[2]),
                       self.codec, self.pool, smooth=self.smooth)
        self.dec_s += time.time() - t
        with self.lock:
            self.projs[i] = a.max(axis=0) > 0
        return a

    def slab(self, i):
        while True:
            with self.lock:
                if i in self.slabs:
                    return self.slabs[i]
                ev = self.inflight.get(i)
                if ev is None:
                    ev = self.inflight[i] = threading.Event()
                    mine = True
                else:
                    mine = False
            if not mine:
                ev.wait()
                continue
            try:
                a = self._decode_slab(i)
                with self.lock:
                    self.slabs[i] = a
            finally:
                with self.lock:
                    self.inflight.pop(i, None)
                ev.set()
            return a

    def prefetch(self, i):
        """Decode slab i on a background thread (one at a time)."""
        if i * CH >= self.shape[0]:
            return
        with self.lock:
            if i in self.slabs:
                return
        if self.pref is not None and self.pref.is_alive():
            return
        self.pref = threading.Thread(target=self.slab, args=(i,), daemon=True)
        self.pref.start()

    def proj(self, z0, z1):
        """(Y, X) bool: any CT in the slabs covering [z0, z1) (a superset of the rows' exact footprint)."""
        out = None
        for i in range(z0 // CH, -(-z1 // CH)):
            self.slab(i)
            p = self.projs[i]
            out = p.copy() if out is None else (out | p)
        return out

    def keep_from(self, z):
        with self.lock:
            for i in list(self.slabs):
                if (i + 1) * CH <= z:
                    del self.slabs[i]
                    self.projs.pop(i, None)
        self.evict_rows_below(z // SH)

    def block(self, z0, z1, y0, y1, x0, x1):
        """CT [z0,z1) x [y0,y1) x [x0,x1) from the slab cache (z range may span slabs)."""
        parts = []
        for i in range(z0 // CH, -(-z1 // CH)):
            s = self.slab(i)
            a, b = max(z0, i * CH) - i * CH, min(z1, (i + 1) * CH) - i * CH
            parts.append(s[a:b, y0:y1, x0:x1])
        return parts[0] if len(parts) == 1 else np.concatenate(parts, 0)


# --------------------------------------------------------------------------- window grid + weights
def starts(n, w, stride):
    """Evenly spaced window starts from 0 to n - w with steps <= stride (nnU-Net's own rule:
    ceil((n - w) / stride) + 1 windows, step (n - w) / (count - 1), rounded)."""
    if n <= w:
        return [0]
    k = -(-(n - w) // stride) + 1
    step = (n - w) / (k - 1)
    return [int(round(step * i)) for i in range(k)]


def gauss1(w):
    x = np.arange(w, dtype=np.float64)
    return np.exp(-0.5 * ((x - (w - 1) / 2) / (w / 6)) ** 2)


def axis_weights(n, w, stride):
    """{start: normalised 1-D weight (float32, length min(w, n))} over the whole axis grid."""
    ss = starts(n, w, stride)
    g = gauss1(w)[:min(w, n)]
    tot = np.zeros(max(n, w))
    for s in ss:
        tot[s:s + w] += g[:min(w, n)]
    return ss, {s: (g / tot[s:s + w][:len(g)]).astype(np.float32) for s in ss}


# --------------------------------------------------------------------------- the model
class M7:
    def __init__(self, dev, plan=None):
        import tensorrt as trt
        import torch
        self.torch = torch
        self.dev = dev
        plan = plan or sorted(glob.glob(PLAN_GLOB))[-1]
        self.plan = plan
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.eng = self.rt.deserialize_cuda_engine(open(plan, "rb").read())
        self.ctx = self.eng.create_execution_context()
        self.stream = torch.cuda.Stream(dev)
        self.w = int(self.eng.get_tensor_shape("x")[-1])
        w = self.w
        self.x = torch.empty((1, 1, w, w, w), dtype=torch.float32, device=dev)
        self.y = torch.empty((1, 2, w, w, w), dtype=torch.float32, device=dev)
        self.ctx.set_tensor_address("x", self.x.data_ptr())
        self.ctx.set_tensor_address("y", self.y.data_ptr())

    def __call__(self, ct_u8):
        """(W,W,W) uint8 CUDA tensor -> (W,W,W) float32 surface probability (on self.stream)."""
        torch = self.torch
        self.x[0, 0].copy_(ct_u8)
        self.x.clamp_(CLO, CHI).sub_(MEAN).div_(STD)
        assert self.ctx.execute_async_v3(self.stream.cuda_stream)
        return torch.sigmoid(self.y[0, 1] - self.y[0, 0])


# --------------------------------------------------------------------------- the worker
def state_dir(v):
    return os.path.join(WORK, "state", v["stem"])


def row_done(v, r):
    return os.path.exists(os.path.join(state_dir(v), f"row_{r}.done"))


def store_dir(v):
    return os.path.join(WORK, "out", store_name(v))


SHARD_POOL = None


def write_row_shards(v, r, stage, codec, pool, stats):
    """Every shard of shard row r, levels 0..3, each written once (shards in parallel)."""
    global SHARD_POOL
    if SHARD_POOL is None:
        SHARD_POOL = ThreadPoolExecutor(6)
    Z, Y, X = v["shape"]
    nz = min(SH, Z - r * SH)
    ny, nx = -(-Y // SH), -(-X // SH)
    root = store_dir(v)
    t = time.time()

    def one(yx):
        yi, xi = yx
        blk = stage[:nz, yi * SH:min((yi + 1) * SH, Y), xi * SH:min((xi + 1) * SH, X)]
        if not blk.any():
            return 0
        nb = 0
        for k in range(4):
            blk = pool2(blk) if k else np.ascontiguousarray(blk)
            p = os.path.join(root, lvl_path(v["pitch"], k), "c", str(r), str(yi), str(xi))
            nb += write_shard(p, blk, lvl_shard(k), codec, pool)
        return nb
    nbytes = sum(SHARD_POOL.map(one, [(yi, xi) for yi in range(ny) for xi in range(nx)]))
    os.makedirs(state_dir(v), exist_ok=True)
    with open(os.path.join(state_dir(v), f"row_{r}.done"), "w") as f:
        json.dump({"bytes": nbytes, "s": round(time.time() - t, 1), "t": time.time()}, f)
    stats["enc_s"] += time.time() - t
    return nbytes


def run_unit(v, r0, r1, m7, codec, pool, stride, wstat, ct_smooth=0.0):
    """One unit = shard rows [r0, r1) of volume v (only the not-done tail of them).

    Per window row s (z extent [s, s+W)), y-row by y-row: windows add p * w into G (W, W, X) fp32 on the
    GPU; once y-slice [t, t+d) is final for this row, the z-carry C (the part of the previous window rows
    that overlaps this one, fp16) is added, z in [s, s_next) is quantised to uint8 and masked by the CT
    on the GPU and copied to the host staging buffer, and z in [s_next, s+W) becomes the new carry."""
    import torch
    dev = m7.dev
    Z, Y, X = v["shape"]
    todo_rows = [r for r in range(r0, r1) if not row_done(v, r)]
    if not todo_rows:
        return
    ra = todo_rows[0]
    Z0, Z1 = ra * SH, min(r1 * SH, Z)
    zs, wz = axis_weights(Z, W, stride)
    ys, wy = axis_weights(Y, W, stride)
    xs, wx = axis_weights(X, W, stride)
    rows = [s for s in zs if s < Z1 and s + W > Z0]
    nxt = {s: (zs[k + 1] if k + 1 < len(zs) else s + W) for k, s in enumerate(zs)}
    cap = max(s + W - nxt[s] for s in zs if nxt[s] < s + W) if len(zs) > 1 else 1
    wyt = {t: torch.from_numpy(wy[t]).to(dev) for t in ys}
    wxt = {u: torch.from_numpy(wx[u]).to(dev) for u in xs}
    ct = CTSource(v, codec, pool, smooth=ct_smooth)
    G = torch.zeros((W, W, X), dtype=torch.float32, device=dev)
    c_bytes = cap * Y * X * 2
    cdev = dev if c_bytes <= int(os.environ.get("M7W_CARRY_GPU_MAX", str(8 << 30))) else torch.device("cpu")
    # the z-carry, one contiguous block per y-row (so a slice [:k] is contiguous and its host <-> GPU
    # copies are truly asynchronous when the carry lives in pinned host memory)
    dys = [(ys[j + 1] if j + 1 < len(ys) else ys[j] + W) - ys[j] for j in range(len(ys))]
    pin = cdev.type == "cpu"
    Cj = [torch.zeros((cap, dys[j], X), dtype=torch.float16, device=cdev, pin_memory=pin) for j in range(len(ys))]
    # pinned staging for the CT strips (in) and the finished uint8 slices (out), double-buffered
    pin_in = [torch.empty((W, W, X), dtype=torch.uint8, pin_memory=True) for _ in (0, 1)]
    ev_in = [None, None]
    pin_out = [torch.empty((W * W * X,), dtype=torch.uint8, pin_memory=True) for _ in (0, 1)]
    pending = []                               # (event, pinned view, za, zb, t, d) not yet in the stage
    nout = [0]
    clen = 0                                   # C holds z in [s, s + clen)
    c_live = np.zeros(len(ys), bool)           # C[:, y-slice j] may be non-zero (else it is exactly 0)
    gid = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    stage_path = [os.path.join(WORK, f"stage_{gid}_{b}.u8") for b in (0, 1)]
    stages = [None, None]
    stage_row = [None, None]

    def stage_for(r):
        """the staging buffer of shard row r: a FRESH sparse file per row (holes read as 0, so air is never
        written and the disk holds only the scroll's footprint)."""
        b = r % 2
        if stage_row[b] != r:
            stages[b] = None
            if os.path.exists(stage_path[b]):
                os.remove(stage_path[b])
            stages[b] = np.memmap(stage_path[b], np.uint8, "w+", shape=(SH, Y, X))
            stage_row[b] = r
        return stages[b]
    writer = [None]
    stats = dict(win=0, skip=0, gpu_s=0.0, host_s=0.0, fin_s=0.0, enc_s=0.0, strip_wait_s=0.0, wait_writer_s=0.0,
                 t0=time.time(), vox=0)

    def flush_row(r, buf):
        nb = write_row_shards(v, r, buf, codec, pool, stats)
        log(f"[row] {v['stem']} shard row {r} written ({nb / 2**20:.0f} MiB, levels 0-3)")

    def put_stage(z0, z1, t, d, arr):
        """host uint8 (z1-z0, d, X) for z in [z0, z1) (inside the unit), y in [t, t+d)."""
        nz_x = np.flatnonzero(arr.any(axis=(0, 1)))
        if nz_x.size == 0:
            for r in range(z0 // SH, -(-z1 // SH)):
                stage_for(r)                       # the row's buffer exists (and is fresh) even if all air
            return
        xa, xb = int(nz_x[0]), int(nz_x[-1]) + 1
        z = z0
        while z < z1:
            r = z // SH
            e = min(z1, (r + 1) * SH)
            stage_for(r)[z - r * SH:e - r * SH, t:t + d, xa:xb] = arr[z - z0:e - z0, :, xa:xb]
            z = e

    def row_done_check(zb):
        """shard rows whose last z is < zb are complete: hand them to the writer thread."""
        for r in range(ra, -(-Z1 // SH)):
            end = min((r + 1) * SH, Z)
            if end <= zb and r not in flushed:
                tw = time.time()
                if writer[0] is not None:
                    writer[0].join()
                stats["wait_writer_s"] += time.time() - tw
                writer[0] = threading.Thread(target=flush_row, args=(r, stage_for(r)), daemon=True)
                writer[0].start()
                flushed.add(r)
    flushed = set()
    strip_ex = ThreadPoolExecutor(1)

    def drain(keep):
        """write finished slices to the stage until at most `keep` are outstanding (oldest first)."""
        while len(pending) > keep:
            ev, po, a_, b_, t_, d_ = pending.pop(0)
            tw = time.time()
            ev.synchronize()
            stats["drain_s"] = stats.get("drain_s", 0.0) + time.time() - tw
            put_stage(a_, b_, t_, d_, po.numpy())
    wstat.update(vol=v["stem"], unit=[r0, r1], rows_total=len(rows), row_i=0, started=time.time())
    log(f"[unit] {v['stem']} rows {r0}..{r1 - 1} (from {ra}): z {Z0}..{Z1}, {len(rows)} window rows, "
        f"{len(ys)}x{len(xs)} windows per row, carry {cap} slices on {cdev.type} ({c_bytes / 2**30:.1f} GiB)")
    ctx = torch.cuda.stream(m7.stream)
    ctx.__enter__()
    for i, s in enumerate(rows):
        tr = time.time()
        s_next = nxt[s]
        dz = min(s_next, s + W) - s
        ct.prefetch((s_next + W - 1) // CH)
        proj = ct.proj(s, s + W)                       # (Y, X) bool: any CT in the slabs covering the row
        ii = np.zeros((Y + 1, X + 1), np.int32)
        ii[1:, 1:] = proj.cumsum(0).cumsum(1)
        wzt = torch.from_numpy(wz[s]).to(dev)
        za, zb = max(s, Z0), min(s + dz, Z1)            # the unit's final z of this row
        G.zero_()
        g_end = -1
        nwin = nskip = 0

        def row_windows(t):
            return [u for u in xs if ii[t + W, u + W] - ii[t, u + W] - ii[t + W, u] + ii[t, u] > 0]

        def get_strip(k, t):
            """CT strip of y-row t into pinned buffer k % 2, and its per-column any prefix sum (CPU)."""
            b = k % 2
            if ev_in[b] is not None:
                ev_in[b].synchronize()          # that buffer's previous upload has finished
            a = pin_in[b].numpy()
            np.copyto(a, ct.block(s, s + W, t, t + W, 0, X))
            anyx = a.any(axis=(0, 1))
            cs = np.concatenate([[0], np.cumsum(anyx, dtype=np.int64)])
            return b, cs
        live = [t for t in ys if row_windows(t)]
        fut = strip_ex.submit(get_strip, 0, live[0]) if live else None
        for j, t in enumerate(ys):
            t_next = ys[j + 1] if j + 1 < len(ys) else t + W
            d = t_next - t
            cand = row_windows(t)
            th = time.time()
            sg = None
            if cand:
                tw = time.time()
                b, cs = fut.result()
                stats["strip_wait_s"] += time.time() - tw
                k = live.index(t)
                sg = pin_in[b].to(dev, non_blocking=True)
                ev_in[b] = torch.cuda.Event()
                ev_in[b].record(m7.stream)
                fut = strip_ex.submit(get_strip, k + 1, live[k + 1]) if k + 1 < len(live) else None
                for u in xs:
                    if int(cs[u + W]) - int(cs[u]) == 0:
                        nskip += 1
                        continue
                    p = m7(sg[:, :, u:u + W])
                    wgt = wzt[:, None, None] * wyt[t][None, :, None] * wxt[u][None, None, :]
                    G[:, :, u:u + W].addcmul_(p, wgt)
                    nwin += 1
                    g_end = t + W
            else:
                nskip += len(xs)
            stats["gpu_s"] += time.time() - th
            th = time.time()
            # y-slice [t, t+d) is final for this window row
            if t < g_end or (clen and c_live[j]):
                blk = G[:, :d, :]
                if clen and c_live[j]:
                    blk[:clen] += Cj[j][:clen].to(dev, torch.float32, non_blocking=True)
                if zb > za:
                    if sg is not None:
                        q = (blk[za - s:zb - s] * 255.0).round_().clamp_(0, 255).to(torch.uint8)
                        q.masked_fill_(sg[za - s:zb - s, :d, :] == 0, 0)
                        drain(1)                   # the pinned buffer about to be reused is written out
                        po = pin_out[nout[0] % 2][:q.numel()].view(q.shape)
                        nout[0] += 1
                        po.copy_(q, non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(m7.stream)
                        pending.append((ev, po, za, zb, t, d))
                    else:
                        put_stage(za, zb, t, d, np.zeros((zb - za, d, X), np.uint8))
                if dz < W:
                    Cj[j][:W - dz].copy_(blk[dz:].to(torch.float16), non_blocking=True)
                    c_live[j] = True
            else:
                if zb > za:
                    put_stage(za, zb, t, d, np.zeros((zb - za, d, X), np.uint8))
                if dz < W and c_live[j]:
                    m7.stream.synchronize()        # no copy into this block is in flight
                    Cj[j][:W - dz].zero_()
                c_live[j] = False
            if j + 1 < len(ys):
                G[:, :W - d] = G[:, d:].clone()
                G[:, W - d:] = 0
            stats["host_s"] += time.time() - th
        clen = W - dz
        drain(0)
        m7.stream.synchronize()
        tf = time.time()
        if zb > za:
            row_done_check(zb)
        ct.keep_from(s_next)
        stats["fin_s"] += time.time() - tf
        stats["win"] += nwin
        stats["skip"] += nskip
        stats["vox"] += max(0, zb - za) * Y * X
        el = time.time() - stats["t0"]
        wstat.update(row_i=i + 1, win=stats["win"], skip=stats["skip"], vox=stats["vox"], elapsed=el,
                     gpu_s=stats["gpu_s"], host_s=stats["host_s"], fin_s=stats["fin_s"], enc_s=stats["enc_s"],
                     strip_wait_s=stats["strip_wait_s"], wait_writer_s=stats["wait_writer_s"],
                     ct_dl_s=ct.dl_s, ct_dec_s=ct.dec_s, ct_bytes=ct.bytes, t=time.time())
        log(f"[zrow] {v['stem']} {i + 1}/{len(rows)} z={s}: {nwin} win {nskip} air in {time.time() - tr:.0f}s "
            f"({nwin / max(time.time() - tr, 1e-9):.2f} win/s); unit {stats['vox'] / max(el, 1e-9) / 1e6:.1f} "
            f"Mvox/s out, enqueue {stats['gpu_s']:.0f}s fin {stats['host_s']:.0f}s gpu_wait {stats.get('drain_s', 0):.0f}s strip_wait "
            f"{stats['strip_wait_s']:.0f}s writer_wait {stats['wait_writer_s']:.0f}s (enc {stats['enc_s']:.0f}) "
            f"ct dl {ct.dl_s:.0f}s dec {ct.dec_s:.0f}s")
    if writer[0] is not None:
        writer[0].join()
    ctx.__exit__(None, None, None)
    strip_ex.shutdown()
    stages[0] = stages[1] = None
    for pth in stage_path:
        if os.path.exists(pth):
            os.remove(pth)
    ct.keep_from(10 ** 9)


def volume_params(v, stride, plan):
    """The (window, stride, plan) of a volume, pinned by the first unit that starts it: every unit of
    one volume must use the same grid and weights, whatever the workers are configured with later."""
    p = os.path.join(state_dir(v), "params.json")
    if os.path.exists(p):
        return json.load(open(p))
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    w = int(rt.deserialize_cuda_engine(open(plan, "rb").read()).get_tensor_shape("x")[-1])
    rec = {"window": w, "stride": int(stride), "plan": plan, "t": time.time(),
           "ct_smooth": float(os.environ.get("M7W_CT_SMOOTH", "0") or 0)}
    os.makedirs(state_dir(v), exist_ok=True)
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    try:
        os.link(tmp, p)                 # first writer wins
    except FileExistsError:
        pass
    os.remove(tmp)
    return json.load(open(p))


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def claim_next(vols, rows_per_unit, me):
    qd = os.path.join(WORK, "queue")
    os.makedirs(qd, exist_ok=True)
    for v in vols:
        if os.path.exists(os.path.join(HOME, "uploaded", v["stem"] + ".json")):
            continue
        for (r0, r1) in units_of(v, rows_per_unit):
            if all(row_done(v, r) for r in range(r0, r1)):
                continue
            c = os.path.join(qd, f"{v['stem']}_{r0}.claim")
            rec = json.dumps({"worker": me, "pid": os.getpid(), "t": time.time()}).encode()
            try:
                fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    old = json.load(open(c))
                except (OSError, ValueError):
                    continue
                if _alive(old.get("pid")):
                    continue
                tmp = c + f".{os.getpid()}"
                open(tmp, "wb").write(rec)
                os.replace(tmp, c)          # a dead worker's claim: take it over
                log(f"[claim] took over {os.path.basename(c)} from dead {old.get('worker')} pid {old.get('pid')}")
                return v, r0, r1, c
            os.write(fd, rec)
            os.close(fd)
            return v, r0, r1, c
    return None


def cmd_worker(a):
    import torch
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    me = f"gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"
    vols = load_vols()
    plan0 = os.environ.get("M7W_PLAN") or sorted(glob.glob(PLAN_GLOB))[-1]
    m7 = M7(dev, plan0)
    codec = Codec()
    pool = ThreadPoolExecutor(a.threads)
    wstat = {"worker": me, "pid": os.getpid()}
    stat_path = os.path.join(WORK, "state", f"worker_{me}.json")
    os.makedirs(os.path.dirname(stat_path), exist_ok=True)
    stop = threading.Event()

    def dump():
        while not stop.is_set():
            try:
                with open(stat_path + ".tmp", "w") as f:
                    json.dump(wstat, f)
                os.replace(stat_path + ".tmp", stat_path)
            except OSError:
                pass
            stop.wait(15)
    threading.Thread(target=dump, daemon=True).start()
    log(f"[worker] {me} default plan {os.path.basename(m7.plan)} stride {a.stride} rows/unit {a.rows} "
        f"(a started volume keeps its own pinned params)")
    while not os.path.exists(os.path.join(HOME, "STOP")):
        c = claim_next(vols, a.rows, me)
        if c is None:
            log("[worker] queue empty")
            wstat.update(vol=None, idle=True)
            break
        v, r0, r1, cpath = c
        prm = volume_params(v, a.stride, plan0)
        if prm["plan"] != m7.plan:
            del m7
            m7 = M7(dev, prm["plan"])
        global W
        W = int(prm["window"])
        assert W == m7.w, (W, m7.w)
        run_unit(v, r0, r1, m7, codec, pool, int(prm["stride"]), wstat, float(prm.get("ct_smooth", 0.0)))
        os.remove(cpath)
    stop.set()


# --------------------------------------------------------------------------- finisher: levels 4..7, metadata, upload
def ckpt_sha():
    p = os.path.join(HOME, "ckpt.sha256")
    if os.path.exists(p):
        return open(p).read().strip()
    h = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()
    open(p, "w").write(h)
    return h


def level_meta(shape, side):
    return {"zarr_format": 3, "node_type": "array", "shape": list(shape), "data_type": "uint8",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [side] * 3}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0,
            "codecs": [{"name": "sharding_indexed", "configuration": {
                "chunk_shape": [CH] * 3,
                "codecs": [{"name": "volcomp", "configuration": {"q": Q}}],
                "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
                "index_location": "end"}}],
            "attributes": {"volcomp": {"q": Q, "encoding": "q"}},
            "dimension_names": ["z", "y", "x"]}


def write_meta(v, stride, plan, ct_smooth=0.0):
    root = store_dir(v)
    levels = []
    for k in range(NLEV):
        sh = lvl_shape(v["shape"], k)
        side = lvl_shard(k)
        p = os.path.join(root, lvl_path(v["pitch"], k))
        os.makedirs(p, exist_ok=True)
        m = level_meta(sh, side)
        m["attributes"]["volcomp"]["voxel_size_um"] = round(v["pitch"] * (1 << k), 6)
        json.dump(m, open(os.path.join(p, "zarr.json"), "w"), indent=2)
        levels.append({"path": lvl_path(v["pitch"], k), "voxel_size_um": round(v["pitch"] * (1 << k), 6),
                       "shape": sh, "encoding": "q", "q": Q, "shard": side})
    name = store_name(v)
    g = {"zarr_format": 3, "node_type": "group", "attributes": {
        "ome": {"version": "0.5", "multiscales": [{
            "version": "0.5", "name": name,
            "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
            "datasets": [{"path": L["path"], "coordinateTransformations": [
                {"type": "scale", "scale": [L["voxel_size_um"]] * 3}]} for L in levels],
            "type": "mean",
            "metadata": {"codec": "volcomp", "description": "2x2x2 mean pooling of the level below (odd edges: "
                         "mean of the in-array voxels); level 0 is the source CT grid, unresampled"}}]},
        "volcomp": {
            "content": "surface probability (uint8 = round(p * 255)), m7 surface teacher, whole volume in one "
                       "seamless sliding-window pass; 0 where the masked CT is 0",
            "model": "surface_m7_nnunet (scrollprize, Kaggle 1st place nnU-Net ResEncL, fold_0 checkpoint_best)",
            "model_url": "https://huggingface.co/scrollprize/surface_m7_nnunet",
            "checkpoint_sha256": ckpt_sha(),
            "inference": {"backend": "TensorRT fp16", "engine": os.path.basename(plan), "window": W,
                          "stride": stride, "overlap": W - stride, "halo": (W - stride) // 2,
                          "blend": "separable Gaussian sigma=w/6, normalised over the whole-volume window grid",
                          "normalisation": {"clip": [CLO, CHI], "mean": MEAN, "std": STD},
                          "output": "softmax foreground channel (surface)", "air_windows": "skipped (all-zero CT)",
                          "ct_decode_smooth": ct_smooth,
                          "ct_decode": ("volcomp_decode_smooth strength %g, gated, zero guard" % ct_smooth) if ct_smooth
                          else "plain volcomp decode (no deblocking)",
                          "tta": 1},
            "source_volume": v["url"] + "/", "source_level": v.get("level", 0),
            "source_shape": v.get("shape0", v["shape"]), "source_level_shape": v["shape"],
            "native_voxel_size_um": v.get("native_pitch", v["pitch"]), "inference_voxel_size_um": v["pitch"],
            "rung_voxel_size_um": v["pitch"], "resampled": False, "resample_scale": 1.0,
            "shape": v["shape"], "levels": levels,
            "encoding": {"name": "volcomp q", "q": Q, "codec_chain": ["volcomp"], "inner_chunk": CH,
                         "volcomp_build_sha256_16": volcomp_build(), "volcomp_version": _vc_version(),
                         "volume_compressor": "github.com/SuperOptimizer/volume-compressor main f31b0e2"},
            "produced_by": "github.com/SuperOptimizer/rvsm tools/m7_wholevol.py",
            "commit": os.environ.get("M7W_COMMIT", "unknown"),
            "host": "forlindesk2 (2x RTX 5060 Ti)",
            "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}}}
    json.dump(g, open(os.path.join(root, "zarr.json"), "w"), indent=2)


def build_upper_levels(v, codec, pool):
    """Levels 4..7 from level 3 (small: level 3 is 1/512 of level 0)."""
    root = store_dir(v)
    sh3 = lvl_shape(v["shape"], 3)
    a = read_block(os.path.join(root, lvl_path(v["pitch"], 3)), sh3, lvl_shard(3), (0, 0, 0), sh3, codec, pool)
    for k in range(4, NLEV):
        a = pool2(a)
        side = lvl_shard(k)
        sh = a.shape
        for zi in range(-(-sh[0] // side)):
            for yi in range(-(-sh[1] // side)):
                for xi in range(-(-sh[2] // side)):
                    blk = np.ascontiguousarray(a[zi * side:(zi + 1) * side, yi * side:(yi + 1) * side,
                                                 xi * side:(xi + 1) * side])
                    p = os.path.join(root, lvl_path(v["pitch"], k), "c", str(zi), str(yi), str(xi))
                    write_shard(p, blk, side, codec, pool)


def sftp_env():
    netrc = os.path.expanduser("~/.volcomp-netrc")
    toks = open(netrc).read().split()
    login = toks[toks.index("login") + 1]
    pw = toks[toks.index("password") + 1]
    ap = os.path.join(HOME, ".askpass.sh")
    if not os.path.exists(ap):
        with open(ap, "w") as f:
            f.write(f"#!/bin/bash\necho '{pw}'\n")
        os.chmod(ap, 0o700)
    env = dict(os.environ, SSH_ASKPASS=ap, SSH_ASKPASS_REQUIRE="force", DISPLAY="none")
    return login, env


def sftp_batch(lines, tag):
    login, env = sftp_env()
    bf = os.path.join(HOME, f".sftp_batch_{tag}")
    of = os.path.join(HOME, f".sftp_out_{tag}")
    with open(bf, "w") as f:
        f.write("\n".join(lines + ["quit"]) + "\n")
    t = time.time()
    r = subprocess.run(["setsid", "-w", "sftp", "-q", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=no",
                        "-o", "Compression=no", "-R", "64", "-P", "9238", "-b", bf,
                        f"{login}@dl.ash2txt.org"], env=env, stdout=open(of, "w"), stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t


def upload_store(v, batch_files=400):
    root = store_dir(v)
    name = store_name(v)
    rd = remote_dir(v)
    part = f"{rd}/{name}.part"
    files, dirs = [], set()
    for dp, _dn, fn in os.walk(root):
        rel = os.path.relpath(dp, root)
        dirs.add(rel)
        for f in fn:
            if f.endswith(".part"):
                continue
            files.append(os.path.join(rel, f) if rel != "." else f)
    total = sum(os.path.getsize(os.path.join(root, f)) for f in files)
    # the data first, the metadata (zarr.json) last
    files.sort(key=lambda f: (os.path.basename(f) == "zarr.json", f))
    pre = [f"-mkdir {SFTP_ROOT}/{v['sample']}", f"-mkdir {SFTP_ROOT}/{v['sample']}/representations",
           f"-mkdir {SFTP_ROOT}/{v['sample']}/representations/predictions", f"-mkdir {rd}", f"-mkdir {part}"]
    pre += [f"-mkdir {part}/{d}" for d in sorted(dirs, key=lambda d: (d.count("/"), d)) if d != "."]
    t0 = time.time()
    for b in range(0, len(files), batch_files):
        chunk = files[b:b + batch_files]
        lines = (pre if b == 0 else []) + [f"put {os.path.join(root, f)} {part}/{f}" for f in chunk]
        for attempt in range(5):
            rc, s = sftp_batch(lines, v["stem"])
            if rc == 0:
                break
            log(f"[upload] {v['stem']} batch {b // batch_files} rc={rc} (attempt {attempt + 1}), retrying in 60 s")
            time.sleep(60)
        else:
            raise RuntimeError(f"upload of {v['stem']} failed")
    rc, _ = sftp_batch([f"rename {part} {rd}/{name}"], v["stem"])
    el = time.time() - t0
    log(f"[upload] {v['stem']}: {len(files)} files, {total / 2**20:.0f} MiB in {el:.0f}s "
        f"({total / 2**20 / max(el, 1e-9):.1f} MiB/s), rename rc={rc}")
    return files, total, el


def verify_remote(v, files):
    import requests
    s = requests.Session()
    root = store_dir(v)
    base = f"{BASE_URL}/{v['sample']}/representations/predictions/surfaces/{store_name(v)}"
    bad = []
    for f in files:
        want = os.path.getsize(os.path.join(root, f))
        for attempt in range(3):
            try:
                r = s.head(f"{base}/{f}", timeout=60)
                got = int(r.headers.get("content-length", -1)) if r.status_code == 200 else -1
                break
            except Exception:  # noqa: BLE001
                got = -2
                time.sleep(3)
        if got != want:
            bad.append((f, want, got))
    return bad


def volume_done(v, rows_per_unit):
    nr = -(-v["shape"][0] // SH)
    return all(row_done(v, r) for r in range(nr))


def cmd_finisher(a):
    codec = Codec()
    pool = ThreadPoolExecutor(a.threads)
    vols = load_vols()
    os.makedirs(os.path.join(HOME, "uploaded"), exist_ok=True)
    plan = sorted(glob.glob(PLAN_GLOB))[-1]
    while True:
        busy = False
        for v in vols:
            mark = os.path.join(HOME, "uploaded", v["stem"] + ".json")
            if os.path.exists(mark):
                continue
            if not volume_done(v, a.rows):
                busy = True
                continue
            t = time.time()
            if not os.path.exists(os.path.join(state_dir(v), "levels.done")):
                build_upper_levels(v, codec, pool)
                prm = json.load(open(os.path.join(state_dir(v), "params.json")))
                global W
                W = int(prm["window"])
                write_meta(v, int(prm["stride"]), prm["plan"], float(prm.get("ct_smooth", 0.0)))
                open(os.path.join(state_dir(v), "levels.done"), "w").close()
                log(f"[finish] {v['stem']}: levels 4-7 + metadata in {time.time() - t:.0f}s")
            files, total, el = upload_store(v)
            bad = verify_remote(v, files)
            if bad:
                log(f"[verify] {v['stem']}: {len(bad)} files differ on the server, e.g. {bad[:3]}; retry next pass")
                busy = True
                continue
            rec = {"store": store_name(v), "remote": f"{remote_dir(v)}/{store_name(v)}",
                   "url": f"{BASE_URL}/{v['sample']}/representations/predictions/surfaces/{store_name(v)}",
                   "files": len(files), "bytes": total, "upload_s": round(el, 1), "t": time.time(),
                   "date": dt.datetime.now(dt.timezone.utc).isoformat()}
            json.dump(rec, open(mark, "w"), indent=1)
            log(f"[verify] {v['stem']}: all {len(files)} files match on the server -> deleting the local store")
            if not a.keep:
                shutil.rmtree(store_dir(v), ignore_errors=True)
        if not busy:
            log("[finish] nothing left to do")
            break
        if os.path.exists(os.path.join(HOME, "STOP")):
            break
        time.sleep(60)


# --------------------------------------------------------------------------- status
def cmd_status(a):
    """status.json: per-worker progress and per-volume projected completion. The projection counts
    non-air WINDOWS (window_estimates.json: {stem: {stride: n}}, from the CT's level 4), not voxels: the
    air fraction differs 5x between volumes. Rate = the live windows/s of the workers (both GPUs)."""
    vols = load_vols()
    now = time.time()
    out = {"t": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"), "workers": {}, "volumes": []}
    wps = []
    for p in glob.glob(os.path.join(WORK, "state", "worker_*.json")):
        try:
            w = json.load(open(p))
        except (OSError, ValueError):
            continue
        out["workers"][w.get("worker")] = {k: (round(x, 1) if isinstance(x, float) else x) for k, x in w.items()}
        if w.get("gpu_s") and w.get("win") and now - w.get("t", 0) < 3600:
            wps.append(w["win"] / max(w.get("elapsed", 1), 1))
    try:
        est = json.load(open(os.path.join(HOME, "window_estimates.json")))
    except (OSError, ValueError):
        est = {}
    rate = sum(wps) if wps else None            # windows/s, all workers
    acc = 0.0
    for v in vols:
        nr = -(-v["shape"][0] // SH)
        done = sum(row_done(v, i) for i in range(nr))
        up = os.path.exists(os.path.join(HOME, "uploaded", v["stem"] + ".json"))
        vox = v["shape"][0] * v["shape"][1] * v["shape"][2]
        try:
            stride = json.load(open(os.path.join(state_dir(v), "params.json")))["stride"]
        except (OSError, ValueError, KeyError):
            stride = int(os.environ.get("M7W_STRIDE", "96"))
        nwin = est.get(v["stem"], {}).get(str(stride))
        rec = {"vol": v["stem"], "name": v["name"], "Gvox": round(vox / 1e9, 1), "rows_done": f"{done}/{nr}",
               "stride": stride, "est_windows": nwin, "uploaded": up, "store": store_name(v)}
        if up:
            rec["eta_utc"] = "uploaded"
        elif rate and nwin:
            acc += nwin * (1 - done / nr)
            rec["eta_utc"] = dt.datetime.fromtimestamp(now + acc / rate + 1800, dt.timezone.utc).strftime(
                "%Y-%m-%d %H:%MZ")
        out["volumes"].append(rec)
    if rate:
        out["windows_per_s_total"] = round(rate, 2)
    js = json.dumps(out, indent=1)
    if a.write:
        with open(os.path.join(HOME, "status.json.tmp"), "w") as f:
            f.write(js)
        os.replace(os.path.join(HOME, "status.json.tmp"), os.path.join(HOME, "status.json"))
    print(js)


# --------------------------------------------------------------------------- seam check / bench
def ref_blend(v, lo, hi, m7, ct, stride):
    """The blended probability over [lo, hi) computed directly (no units, no shards, no rolling): every
    global-grid window intersecting the box, weighted by the same normalised weights, in fp32."""
    import torch
    dev = m7.dev
    Z, Y, X = v["shape"]
    zs, wz = axis_weights(Z, W, stride)
    ys, wy = axis_weights(Y, W, stride)
    xs, wx = axis_weights(X, W, stride)
    sel = [[s for s in ss if s < h and s + W > l] for ss, l, h in zip((zs, ys, xs), lo, hi)]
    e0 = [min(s) for s in sel]
    e1 = [max(s) + W for s in sel]
    acc = torch.zeros(tuple(b - a for a, b in zip(e0, e1)), dtype=torch.float32, device=dev)
    cb = ct.block(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])
    cg = torch.from_numpy(np.ascontiguousarray(cb)).to(dev)
    for s in sel[0]:
        for t in sel[1]:
            for u in sel[2]:
                win = cg[s - e0[0]:s - e0[0] + W, t - e0[1]:t - e0[1] + W, u - e0[2]:u - e0[2] + W]
                if not bool(win.any()):
                    continue
                with torch.cuda.stream(m7.stream):
                    p = m7(win)
                    wgt = (torch.from_numpy(wz[s]).to(dev)[:, None, None] * torch.from_numpy(wy[t]).to(dev)[None, :, None]
                           * torch.from_numpy(wx[u]).to(dev)[None, None, :])
                    acc[s - e0[0]:s - e0[0] + W, t - e0[1]:t - e0[1] + W, u - e0[2]:u - e0[2] + W] += p * wgt
                m7.stream.synchronize()
    sl = tuple(slice(a - b, c - b) for a, b, c in zip(lo, e0, hi))
    q = (acc[sl] * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    cbox = cb[sl]
    q[cbox == 0] = 0
    return q, cbox


def cmd_seamcheck(a):
    import torch
    vols = {v["stem"]: v for v in load_vols()}
    v = vols[a.vol]
    dev = torch.device("cuda:0")
    m7 = M7(dev)
    codec = Codec()
    pool = ThreadPoolExecutor(16)
    ct = CTSource(v, codec, pool)
    c = np.array([int(x) for x in a.center.split(",")])
    h = a.half
    lo, hi = c - h, c + h
    root = a.store or store_dir(v)
    st = read_block(os.path.join(root, lvl_path(v["pitch"], 0)), v["shape"], SH, lo, hi, codec, pool)
    ref, cbox = ref_blend(v, lo, hi, m7, ct, a.stride)
    d = st.astype(np.int16) - ref.astype(np.int16)
    fg = cbox > 0
    rec = {"box_lo": lo.tolist(), "box_hi": hi.tolist(), "store_vs_direct_blend": {
        "max_abs": int(np.abs(d).max()), "mean_abs": float(np.abs(d[fg]).mean()) if fg.any() else 0.0,
        "corr": float(np.corrcoef(st[fg].astype(np.float32), ref[fg].astype(np.float32))[0, 1]) if fg.any() else None,
        "dice_0.5": float(2 * ((st >= 128) & (ref >= 128)).sum() / max(1, (st >= 128).sum() + (ref >= 128).sum()))}}
    # discontinuity across the shard planes through the centre vs the neighbouring plane pairs
    for ax in range(3):
        def step(arr, i):
            sa = [slice(None)] * 3
            sb = [slice(None)] * 3
            sa[ax], sb[ax] = i, i - 1
            return float(np.abs(arr[tuple(sa)].astype(np.int16) - arr[tuple(sb)].astype(np.int16)).mean())
        steps = [step(st, i) for i in range(h - 6, h + 7)]
        rec[f"axis{ax}_plane_steps"] = {"at_shard_boundary": round(steps[6], 3),
                                        "neighbours_mean": round(float(np.mean(steps[:6] + steps[7:])), 3)}
    # a single direct window centred on the corner, unblended
    o = c - W // 2
    win = torch.from_numpy(np.ascontiguousarray(ct.block(o[0], o[0] + W, o[1], o[1] + W, o[2], o[2] + W))).to(dev)
    with torch.cuda.stream(m7.stream):
        p1 = (m7(win) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    s1 = read_block(os.path.join(root, lvl_path(v["pitch"], 0)), v["shape"], SH, o + 24, o + W - 24, codec, pool)
    p1 = p1[24:-24, 24:-24, 24:-24]
    m = s1 > 0
    rec["single_window_centre_vs_store"] = {
        "corr": float(np.corrcoef(p1[m].astype(np.float32), s1[m].astype(np.float32))[0, 1]) if m.any() else None,
        "dice_0.5": float(2 * ((p1 >= 128) & (s1 >= 128)).sum() / max(1, (p1 >= 128).sum() + (s1 >= 128).sum()))}
    print(json.dumps(rec, indent=1))


def cmd_bench(a):
    """Throughput of the real pipeline on a few window rows of a volume (no store written)."""
    import torch
    vols = {v["stem"]: v for v in load_vols()}
    v = vols[a.vol]
    dev = torch.device("cuda:0")
    m7 = M7(dev)
    codec = Codec()
    pool = ThreadPoolExecutor(16)
    ct = CTSource(v, codec, pool)
    Z, Y, X = v["shape"]
    s = a.z
    t = time.time()
    strip_all = ct.block(s, s + W, 0, Y, 0, X)
    log(f"CT row read {time.time() - t:.1f}s (dl {ct.dl_s:.1f} dec {ct.dec_s:.1f}), nonzero frac {float((strip_all > 0).mean()):.3f}")
    ys = starts(Y, W, a.stride)
    xs = starts(X, W, a.stride)
    n = sk = 0
    torch.cuda.synchronize()
    t = time.time()
    for tt in ys[len(ys) // 2 - a.yrows // 2:len(ys) // 2 + a.yrows - a.yrows // 2]:
        sg = torch.from_numpy(np.ascontiguousarray(strip_all[:, tt:tt + W])).to(dev)
        for u in xs:
            w = sg[:, :, u:u + W]
            if not bool(w.any()):
                sk += 1
                continue
            with torch.cuda.stream(m7.stream):
                m7(w)
            n += 1
        m7.stream.synchronize()
    el = time.time() - t
    log(f"{n} windows ({sk} air) in {el:.1f}s = {n / el:.2f} win/s, {n * (a.stride ** 3) / el / 1e6:.1f} Mvox/s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("worker", "finisher", "status", "seamcheck", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--stride", type=int, default=int(os.environ.get("M7W_STRIDE", "144")))
        p.add_argument("--rows", type=int, default=int(os.environ.get("M7W_ROWS", "4")))
        p.add_argument("--threads", type=int, default=16)
        if name == "finisher":
            p.add_argument("--keep", action="store_true")
        if name == "status":
            p.add_argument("--write", action="store_true")
        if name in ("seamcheck", "bench"):
            p.add_argument("--vol", required=True)
        if name == "seamcheck":
            p.add_argument("--center", required=True)
            p.add_argument("--half", type=int, default=96)
            p.add_argument("--store", default=None)
        if name == "bench":
            p.add_argument("--z", type=int, default=2048)
            p.add_argument("--yrows", type=int, default=4)
    a = ap.parse_args()
    {"worker": cmd_worker, "finisher": cmd_finisher, "status": cmd_status, "seamcheck": cmd_seamcheck,
     "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
