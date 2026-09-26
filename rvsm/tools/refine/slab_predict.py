"""The student's recto / verso (+ midline, thickness) over a z-slab of the fine Paris 4 volume -- TILED
and RAM-BOUNDED.

WHY THIS SHAPE. The first version of this tool held slab-sized outputs on the host and OOM-killed the
23 GB laptop VM five times. Here nothing slab-sized ever exists in host memory:

  * the slab box (default lo = [55552, 13696, 13312], shape = [128, 8192, 8192], fine 2.4 um zyx) is cut
    into `tile`^2 tiles (128 x 1024 x 1024 -> 64 tiles);
  * each tile is `sub`^2 `infer.student_region`-equivalent passes (`StudentInputs` + `run_region`:
    window 256, halo 32, cascade 3, fine-CT margin 64; sub 512 -> 4 passes a tile), whose fp16
    accumulators live on the GPU and are sub-box sized. One 1024^2 pass peaked at 14 GB of the 16 GB
    laptop card, spilled, and ran at 16 s a window (519 s a tile); 512 sub-boxes: ~0.6 s a window;
  * the planes are quantised to uint8 ON THE GPU and only the uint8 tile (4 x 128 MiB) crosses to the
    host, and each plane is written at once as exactly ONE shard of its store (shard = tile);
  * a tile whose four shards already exist is skipped (resumable -- the shard files ARE the ledger);
  * a tile that is all air at the coarse CT level is skipped without touching the network (its shards
    stay absent = fill 0 = no data).

A watchdog thread samples VmRSS (/proc/self/status) every 0.2 s; above --rss-limit-gb it dumps every
thread's stack (faulthandler) into the log and hard-exits, so a runaway shows WHERE it ran away instead
of taking the VM down. Per tile the log gets windows, seconds, Mvox/s, RSS now and RSS peak.

Stores (zarr v3, 128^3 chunks, shards (128, 1024, 1024), codec exactly [volcomp]) in <out>/<plane>.zarr,
shape = the whole slab, attrs origin_zyx = the slab corner in fine voxels:
    recto, verso   q8 probability (sigmoid(logit / T(rung 2)))
    midline        q0 signed_u8_off128_q0.25 (export.enc_t), 0 = no data
    thickness      q0 unsigned_u8_q0.25 (targets.encode_unsigned), 0 = no data

ALWAYS run it detached under a memory cap, e.g.

    VOLCOMP_LIB=... TORCHINDUCTOR_COMPILE_THREADS=1 setsid nohup systemd-run --user --scope -q \\
        -p MemoryMax=8G -p MemorySwapMax=0 .venv/bin/python -m rvsm.tools.refine.slab_predict \\
        --max-tiles 1 >> /home/forrest/refine/pred/slab.out 2>&1 &
"""
from __future__ import annotations

import argparse
import ctypes
import faulthandler
import json
import os
import sys
import threading
import time

import numpy as np

CT_URL = ("https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/volumes/"
          "20260411134726-2.400um-0.2m-78keV-masked.zarr")
PLANES = ("recto", "verso", "midline", "thickness")
ENC = {"recto": (8, "prob_u8"), "verso": (8, "prob_u8"), "midline": (0, "signed_u8_off128_q0.25"),
       "thickness": (0, "unsigned_u8_q0.25")}
SLAB_LO = (55552, 13696, 13312)
SLAB_SHAPE = (128, 8192, 8192)


def rss_gb(pid="self"):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    return 0.0


class Guard:
    """RSS watchdog: `peak` since the last `reset()`, `phase` for the log, hard exit above `limit_gb`."""

    def __init__(self, limit_gb, say, period=0.2):
        self.limit, self.say, self.period = float(limit_gb), say, float(period)
        self.peak, self.phase = 0.0, "start"
        threading.Thread(target=self._run, daemon=True).start()

    def reset(self):
        self.peak = rss_gb()

    def _run(self):
        while True:
            r = rss_gb()
            self.peak = max(self.peak, r)
            if r > self.limit:
                self.say(f"[guard] RSS {r:.2f} GB > {self.limit} GB in phase {self.phase!r}: stacks follow, exiting")
                try:
                    faulthandler.dump_traceback(file=self.say.log, all_threads=True)
                    self.say.log.flush()
                except Exception:
                    pass
                os._exit(3)
            time.sleep(self.period)


def trim():
    """Give freed heap back to the OS (glibc keeps it otherwise, and RSS only ratchets up)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def tiles_of(lo, n, tile):
    """[(z, y, x, dz, dy, dx)] fine-voxel boxes, row-major over (y, x)."""
    return [(int(lo[0]), int(lo[1] + y), int(lo[2] + x), int(n[0]), int(min(tile, n[1] - y)),
             int(min(tile, n[2] - x))) for y in range(0, int(n[1]), tile) for x in range(0, int(n[2]), tile)]


def shard_file(store, lo, t, shard):
    """The on-disk shard of tile `t` in `store` (default chunk-key encoding: c/<i>/<j>/<k>)."""
    ix = [(t[k] - lo[k]) // shard[k] for k in range(3)]
    return os.path.join(store, "c", *(str(int(i)) for i in ix))


def tile_paths(pyr, t, margin, ctx):
    """Every CT shard a tile's pass reads: the padded box plus a 256 border (cascade halos), and the
    context rungs of `stream.region_paths` over that box."""
    from rvsm import stream
    lo = np.array(t[:3]) - margin - 256
    R = np.array(t[3:]) + 2 * (margin + 256)
    return stream.region_paths(pyr, lo, ctx, tuple(int(v) for v in R), 2)


def tile_is_air(pyr, t, rung=6):
    """True when the tile's box at a coarse rung (16x: 8 x 64 x 64 voxels) is all zero CT."""
    from rvsm import ladder
    e = 1 << (rung - 2)
    lo = [int(t[k]) // e for k in range(3)]
    hi = [-(-(int(t[k]) + int(t[3 + k])) // e) for k in range(3)]
    c = ladder.read_rung(pyr, rung, lo, [hi[k] - lo[k] for k in range(3)], dtype=np.uint8)
    return not bool(c.any())


class Counter:
    """Wraps the student's forward to count windows (every rung, cascade passes included)."""

    def __init__(self, st):
        self.n, self.f = 0, st.net
        st.net = self

    def __call__(self, x):
        self.n += int(x.shape[0])
        return self.f(x)


def run(a):
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "slab.log"), "a")

    def say(*s):
        msg = time.strftime("%H:%M:%S ") + " ".join(str(q) for q in s)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()
    say.log = logf

    guard = Guard(a.rss_limit_gb, say)
    say(f"[slab] pid {os.getpid()} args {json.dumps(vars(a))} rss {rss_gb():.2f} GB")

    import torch
    from rvsm import axis as AX, export as EX, infer, ladder, stores, stream, targets as TG

    guard.phase = "ct-open"
    cache = stream.ShardCache(a.ct, a.cache, budget_gb=a.cache_gb, jobs=a.jobs, log=say)
    pyr = cache.levels()
    ax = AX.load(a.umbilicus, ct=cache.base)
    lo = np.array(a.lo, np.int64)
    n = np.array(a.shape, np.int64)
    assert (n % 128 == 0).all(), n
    shard = stores.shard_shape(tuple(int(v) for v in n))
    assert n[0] == shard[0] and a.tile == shard[1] == shard[2], \
        f"tile {a.tile} x slab depth {n[0]} must be exactly one shard {shard}"
    say(f"[slab] box lo {lo.tolist()} shape {n.tolist()} = {np.prod(n) / 1e9:.2f} Gvox, shard {shard}, "
        f"rss {rss_gb():.2f} GB")

    guard.phase = "ckpt"
    st = infer.student_fn(a.ckpt, compile=not a.no_compile, mode=a.compile_mode,
                          gn_bf16=(True if a.gn_bf16 else None))
    cnt = Counter(st)
    w, h, d = st.cfg.infer_window, st.cfg.infer_halo, st.cfg.cascade_depth
    say(f"[slab] ckpt {a.ckpt} step {st.step} window {w} halo {h} cascade {d} margin {a.margin} "
        f"temps {st.temps} gn_bf16 {st.gn_bf16} compile {st.compiled} rss {rss_gb():.2f} GB")
    meta = np.asarray(json.load(open(a.meta5)) if a.meta5 else [0.0] * 5, np.float32)

    arrs = {}
    for ch in PLANES:
        p = os.path.join(a.out, ch + ".zarr")
        if os.path.exists(os.path.join(p, "zarr.json")):
            import zarr
            ladder.require_volcomp()
            z = zarr.open_array(p, mode="r+")
            assert tuple(z.shape) == tuple(int(v) for v in n) and \
                list(z.attrs.get("origin_zyx", [])) == lo.tolist(), f"{p}: another slab ({z.shape})"
            arrs[ch] = z
            continue
        q, enc = ENC[ch]
        arrs[ch] = stores.out_array(p, tuple(int(v) for v in n), tuple(int(v) for v in lo), rung=2,
                                    channels=(ch,), q=q, volume=a.ct, umbilicus=a.umbilicus,
                                    attrs={"encoding": enc, "no_data": 0, "axis_order": "ZYX",
                                           "producer": "student", "ckpt": a.ckpt, "step": int(st.step),
                                           "window": w, "halo": h, "cascade_depth": d,
                                           "margin": a.margin, "radial_sign": 1, "tile": a.tile,
                                           "sign_convention": EX.SIGN_CONVENTION})
    tl = tiles_of(lo, n, a.tile)
    if a.tiles:
        want = {int(i) for i in a.tiles.split(",")}
        tl_run = [(i, t) for i, t in enumerate(tl) if i in want]
    else:
        tl_run = list(enumerate(tl))
    names = list(PLANES)
    fn = st.plane_fn(names, 2)
    bounded = [q in ("recto", "verso") for q in names]
    rows, n_new, t_all = [], 0, time.time()
    for i, t in tl_run:
        key = "%d_%d_%d" % t[:3]
        sfs = {ch: shard_file(os.path.join(a.out, ch + ".zarr"), lo, t, shard) for ch in PLANES}
        if all(os.path.exists(f) for f in sfs.values()):
            continue
        guard.reset()
        t0 = time.time()
        guard.phase = f"tile {i} air-check"
        if tile_is_air(pyr, t):
            say(f"[slab] tile {i}/{len(tl)} {key} all air at rung 6: skipped")
            continue
        guard.phase = f"tile {i} fetch"
        paths = tile_paths(pyr, t, a.margin, st.cfg.ctx)
        miss = [p for p in paths if not stream.present(p)]
        cache.download(miss, key=key)
        t_fetch = time.time() - t0
        n0 = cnt.n
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1 = time.time()
        # the tile in `sub`^2 sub-boxes, each its own region pass (the accumulators, the padded CT and
        # the cascade are sub-box sized on the card), assembled as uint8 on the host (4 x 128 MiB)
        host = {ch: np.zeros(tuple(t[3:]), np.uint8) for ch in PLANES}
        sb = int(a.sub)
        t_in, rss_in, fine_w = 0.0, 0.0, 0
        for sy in range(0, t[4], sb):
            for sx in range(0, t[5], sb):
                bl = (t[0], t[1] + sy, t[2] + sx)
                bs = (t[3], min(sb, t[4] - sy), min(sb, t[5] - sx))
                if tile_is_air(pyr, bl + bs):
                    continue
                t3 = time.time()
                guard.phase = f"tile {i} sub {sy},{sx} inputs+cascade"
                inp = infer.StudentInputs(cache.base, ax, bl, bs, st.layout, meta=meta, rung=2,
                                          ctx=st.cfg.ctx, window=w, halo=h, sign=1.0, cascade_depth=d,
                                          head0=st.head0, device=st.dev, pyr=pyr, batch=a.batch,
                                          margin=a.margin)
                torch.cuda.synchronize()
                t_in += time.time() - t3
                rss_in = max(rss_in, rss_gb())
                guard.phase = f"tile {i} sub {sy},{sx} run_region"
                stats = {}
                out = infer.run_region(fn, inp, bs, w, h, batch=a.batch, bounded=bounded,
                                       planes=len(names), offs=inp.offs, out_dtype=torch.float16,
                                       stats=stats)
                fine_w += int(stats.get("windows") or 0)
                del inp
                # quantise on the card; only uint8 crosses the bus
                valid = out[0] > 0
                enc = {
                    "recto": torch.clamp(torch.round(out[0].float() * 255), 0, 255).to(torch.uint8),
                    "verso": torch.clamp(torch.round(out[1].float() * 255), 0, 255).to(torch.uint8),
                    "midline": EX.enc_t(out[2], valid, -EX.TRACER_CAP, EX.TRACER_CAP, EX.TRACER_UNIT,
                                        EX.TRACER_OFF),
                    "thickness": EX.enc_t(out[3], valid, TG.UNIT, 255 * TG.UNIT, TG.UNIT),
                }
                del out, valid
                for ch in PLANES:
                    host[ch][:, sy:sy + bs[1], sx:sx + bs[2]] = enc.pop(ch).cpu().numpy()
                del enc
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t_gpu = time.time() - t1
        stats = {"windows": fine_w}
        guard.phase = f"tile {i} write"
        sl = tuple(slice(int(t[k] - lo[k]), int(t[k] - lo[k] + t[3 + k])) for k in range(3))
        means = {}
        t2 = time.time()
        for ch in ("thickness", "midline", "verso", "recto"):      # recto last
            b = host.pop(ch)
            means[ch] = round(float(b.mean()), 3)
            arrs[ch][sl] = b
            del b
        t_w = time.time() - t2
        trim()
        t_tot = time.time() - t0
        nv = int(np.prod(t[3:]))
        n_new += 1
        row = {"i": i, "tile": key, "fetched": len(miss), "fetch_s": round(t_fetch, 1),
               "inputs_s": round(t_in, 1), "gpu_s": round(t_gpu, 1), "write_s": round(t_w, 1),
               "total_s": round(t_tot, 1), "fine_windows": stats.get("windows"), "all_windows": cnt.n - n0,
               "mvox_per_s": round(nv / 1e6 / t_tot, 3), "mvox_per_s_gpu": round(nv / 1e6 / t_gpu, 3),
               "means": means, "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "rss_after_inputs_gb": round(rss_in, 2), "rss_gb": round(rss_gb(), 2),
               "rss_peak_gb": round(guard.peak, 2)}
        rows.append(row)
        left = sum(1 for _, u in tl_run if not all(os.path.exists(shard_file(
            os.path.join(a.out, c + ".zarr"), lo, u, shard)) for c in PLANES))
        avg = sum(r["total_s"] for r in rows) / len(rows)
        say(f"[slab] tile {i}/{len(tl)} {json.dumps(row)} | left {left} eta {left * avg / 60:.1f} min")
        if a.max_tiles and n_new >= a.max_tiles:
            break
    guard.phase = "done"
    left = [t for t in tl if not all(os.path.exists(shard_file(os.path.join(a.out, c + ".zarr"), lo, t, shard))
                                     for c in PLANES)]
    say(f"[slab] finished {n_new} new tiles in {time.time() - t_all:.0f} s; {len(left)} tiles without "
        f"shards (air or not yet run); peak rss {guard.peak:.2f} GB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/home/forrest/refine/models/student_100000.pt")
    ap.add_argument("--ct", default=CT_URL)
    ap.add_argument("--umbilicus", default="/home/forrest/refine/umbilicus/"
                    "20260411134726-umbilicus-20260524235033.json")
    ap.add_argument("--meta5", default="/home/forrest/refine/umbilicus/run_meta5.json",
                    help="the run's frozen scan planes (json list of 5); empty = zeros")
    ap.add_argument("--lo", type=int, nargs=3, default=list(SLAB_LO), help="slab corner, fine zyx")
    ap.add_argument("--shape", type=int, nargs=3, default=list(SLAB_SHAPE), help="slab shape, fine zyx")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--tiles", default="", help="comma list of tile indices to run (default all)")
    ap.add_argument("--margin", type=int, default=64)
    ap.add_argument("--sub", type=int, default=512,
                    help="region-pass box inside a tile (y/x); 1024 = one pass per tile. On the 16 GB "
                         "laptop card a 1024 pass peaked at 14 GB VRAM and spilled: 16 s a window")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default="/home/forrest/refine/pred")
    ap.add_argument("--cache", default="/home/forrest/refine/ct_cache")
    ap.add_argument("--cache-gb", type=float, default=200.0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--compile-mode", default="default",
                    help="torch.compile mode (max-autotune spawns benchmark workers: RAM on a small host)")
    ap.add_argument("--gn-bf16", action="store_true")
    ap.add_argument("--max-tiles", type=int, default=0, help="stop after this many new tiles (a probe)")
    ap.add_argument("--rss-limit-gb", type=float, default=6.0, help="dump stacks and hard-exit above this")
    a = ap.parse_args(argv)
    run(a)


if __name__ == "__main__":
    sys.exit(main())
