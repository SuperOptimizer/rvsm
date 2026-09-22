"""The driver: one supervisor, one producer, the lookahead, the two gates and the rounds.

`rvsm run` is the whole project in one process tree. The supervisor freezes what the run is (the
resolved config, the umbilicus, the scan metadata, the held-out set), picks the GPU mode and the VRAM
budget, spawns ONE producer process and then becomes the trainer. The two halves never talk: they talk
through `<out>/` (plan §1, user decision 4).

    <out>/config.json     the resolved config + its fingerprint; a resume asserts against it
    <out>/umbilicus.json  the axis in rung-2 voxels, derived once
    <out>/metadata.json   the scan metadata, flattened, frozen
    <out>/state.json      the ONLY mutable metadata: {round, step, cursor, verso_on}, trainer-written
    <out>/workers.json    pid / phase / last_ts per role, written by the supervisor's heartbeat
    <out>/STOP            touch it (or `rvsm stop`) and both sides finish the unit in flight and exit
    <out>/PHASE           one-card timeshare: "train" or "produce", whichever role owns the card now

The producer and the trainer walk the SAME deterministic walk -- the same `region_list`, the same
`region_visits`, the same `walk_order` seed -- so "the next L regions" means the same thing on both
sides without either telling the other anything. The trainer publishes how far along the walk it is
(`cursor`); the producer keeps the window ahead of it ready and releases a region's CT shards once the
cursor is past it.

What each side does with a region is decided by what is ON DISK, not by a message:

    round 0   no `recto`          -> the teacher pass (recto, fused with m7 by agreement -> recto + rw)
              `verso_on` and no `verso`
                                  -> the student at sign -1, verso head only (the flipped-sign trick)
              recto + verso, no `midline`
                                  -> `targets.region_fields`: midline + thickness at rungs 2-4
    round r>=1
              no `recto` in round r
                                  -> ONE multi-head student pass at sign +1: recto, verso, midline,
                                     thickness, conf
    always    the finished recto is folded into the coarse rungs 7-11 (`regions.feed_coarse`)

and the held-out regions are produced FIRST, because their round-0 stores are the fixed reference every
gate and every evaluation is scored against.

The two gates live on the trainer side, in a hook `train.train` calls at every evaluation:

    the VERSO gate (round 0 only)   the student's own recto over the held-out regions, compared to the
                                    reference stores with `evalsurf.compare_stores`, pooled over regions
                                    with a bootstrap CI: dice >= `verso_gate_dice` and a betti0 error no
                                    worse than the reference-against-itself baseline by more than the CI
                                    -- or `verso_after_steps`, unconditionally. Then `verso_on` goes
                                    into state.json and the producer starts the flipped-sign passes.
    the ROUND gate                  `evalsurf.fit_curve` on `logs/eval.jsonl` says less than 2 % of the
                                    gain is left (or `round_steps` elapsed), AND the held-out comparison
                                    is not worse than the round-0 reference beyond the CI. Then the
                                    calibrated EMA is snapshotted as `ckpt/teacher_round_<r+1>.pt`, the
                                    round in state.json is bumped, and the producer regenerates the
                                    stores. A round that fails the quality half is DISCARDED: the round
                                    is not bumped and the training simply continues (WSD makes that
                                    free).

Two deviations from the plan's letter, both recorded here because they are visible in the code:

- the sampler reads ONE round (`Patches(round_=r)`), so "prefer the newest round per region" is
  satisfied by the trainer waiting for a region's round-r store rather than by falling back to round
  r-1 in the same batch. The wait is the same wait the cold start already has, and it is what forces
  the producer to finish round r+1 before the trainer can train on it.
- the verso gate scores at most `GATE_REGIONS` held-out regions per attempt. A full student pass over
  eight 1024^3 regions at every eval would cost more than the training it is gating.
- the plan words the verso gate as "recall@4 and continuity vs the fused teacher reference within the
  bootstrap CI". Both of those are MESH metrics (`evalsurf.surface_rows` walks tifxyz surfaces), and a
  run has meshes only when `tifxyz` is set -- by user decision 3 that is optional and eval-only, while
  the reference the gate must use is a STORE. The store-only analogue is `compare_stores`: dice for the
  recall side and the betti0 error against the reference-vs-itself baseline for the continuity side,
  with the same bootstrap over regions. `rvsm eval --tifxyz` still runs the mesh suite.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time

import numpy as np

from rvsm import config as CFG

STOP_FILE = "STOP"
PHASE_FILE = "PHASE"
HEARTBEAT_S = 30.0          # how often the supervisor stamps workers.json / logs/sched.jsonl
SILENT_MAX_S = 600.0        # a producer that has not stamped its heartbeat for this long is restarted
WAIT_S = 5.0                # the trainer's sleep when nothing in the lookahead window is ready
IDLE_S = 2.0                # the producer's sleep when there is nothing to produce
ROUND_GAIN = 0.02           # "< 2 % remaining gain" is the plateau half of the round gate
GATE_REGIONS = 2            # held-out regions scored per gate attempt (see the module docstring)
LOOKAHEAD_MAX = 64
REEST_S = 600.0             # the lookahead L is re-estimated from the logs this often

# The plan's budget table (§1 "GPU modes"), in GB on one 80 GB card: trainer 46-57 (30m6, 256^3, batch
# 2, ckpt-act 1-2), producer ~30 (a teacher and the student, never concurrently). The low end of the
# trainer's range is the one that fits beside the producer under the 4 GB headroom, so it is the one the
# table starts from; a bigger card scales both roles by the same factor.
BUDGET_GB = {"train": 46.0, "produce": 30.0}
REFERENCE_GB = 80.0
HEADROOM_GB = 4.0
RESIDENT_MIN_GB = 70.0      # one card at least this big runs every role resident


# --------------------------------------------------------------------------- #
# the filesystem bus
# --------------------------------------------------------------------------- #
def _read_json(path, default=None):
    try:
        with open(str(path)) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001  -- a half-written file is not a reason to stop a run
        return default


def _write_json(path, obj):
    """Atomically: write beside, rename over. Every reader is another process."""
    path = str(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda v: list(v) if isinstance(v, tuple) else str(v))
    os.replace(tmp, path)
    return path


def jlog(out, name, rec, echo=True):
    """One json line in `<out>/logs/<name>.jsonl`, with a wall-clock stamp."""
    rec = {"t": round(time.time(), 3), **rec}
    p = os.path.join(str(out), "logs", f"{name}.jsonl")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    if echo:
        print(f"[{name}] {rec}", flush=True)
    return rec


def tail_jsonl(path, n=200):
    """The last `n` parsed lines of a jsonl file (missing file -> [])."""
    try:
        with open(str(path)) as f:
            lines = f.readlines()[-int(n):]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def read_state(out, default=None):
    return _read_json(os.path.join(str(out), "state.json"), default if default is not None else {})


def write_state(out, **upd):
    """Merge `upd` into state.json and rewrite it atomically. ONLY the trainer calls this."""
    st = read_state(out)
    st.update(upd)
    st["ts"] = round(time.time(), 3)
    _write_json(os.path.join(str(out), "state.json"), st)
    return st


def stop_requested(out):
    return os.path.exists(os.path.join(str(out), STOP_FILE))


def request_stop(out):
    p = os.path.join(str(out), STOP_FILE)
    os.makedirs(str(out), exist_ok=True)
    with open(p, "w") as f:
        f.write(json.dumps({"t": time.time()}))
    return p


def read_phase(out, default="train"):
    try:
        with open(os.path.join(str(out), PHASE_FILE)) as f:
            v = f.read().strip()
        return v or default
    except OSError:
        return default


def write_phase(out, value):
    p = os.path.join(str(out), PHASE_FILE)
    os.makedirs(str(out), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(value))
    os.replace(tmp, p)
    return value


def free_gb(path):
    try:
        return shutil.disk_usage(str(path)).free / (1 << 30)
    except OSError:
        return float("inf")


# --------------------------------------------------------------------------- #
# the GPUs and the VRAM budget
# --------------------------------------------------------------------------- #
def cards(gpus=(0,)):
    """[(ordinal, total GB)] for the CUDA devices this run may use; [] when there is no CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        n = torch.cuda.device_count()
        return [(int(g), torch.cuda.get_device_properties(int(g)).total_memory / (1 << 30))
                for g in gpus if int(g) < n]
    except Exception:  # noqa: BLE001  -- no torch CUDA build, no driver: the CPU path is always there
        return []


def choose_mode(mode, found):
    """The role placement: `{mode, train_gpu, produce_gpu, phases, total_gb}`.

    `auto` (plan §1): one card of at least 70 GB -> **resident**, both roles on it; two cards -> one
    role per card, no switching; one small card -> **timeshare** phases through `<out>/PHASE`. No card
    at all -> `cpu`, and the producer runs as a thread in the supervisor (nothing to share)."""
    mode = str(mode or "auto")
    if not found:
        return {"mode": "cpu", "train_gpu": None, "produce_gpu": None, "phases": False,
                "total_gb": 0.0}
    big = max(g for _, g in found)
    if mode == "auto":
        mode = "resident" if (len(found) == 1 and big >= RESIDENT_MIN_GB) else "timeshare"
    if mode == "resident":
        g = max(found, key=lambda q: q[1])[0]
        return {"mode": "resident", "train_gpu": g, "produce_gpu": g, "phases": False,
                "total_gb": float(dict(found)[g])}
    if len(found) >= 2:   # one role per card: no switching, no PHASE file
        a, b = found[0][0], found[1][0]
        return {"mode": "timeshare", "train_gpu": a, "produce_gpu": b, "phases": False,
                "total_gb": float(min(found[0][1], found[1][1]))}
    g, tot = found[0]
    return {"mode": "timeshare", "train_gpu": g, "produce_gpu": g, "phases": True,
            "total_gb": float(tot)}


def budget(place, table=None, headroom=HEADROOM_GB):
    """`{role: {gb, fraction}}` for a placement, scaled from the plan's 80 GB table.

    Two roles resident on one card must fit under `total - headroom`; when they do not, the run does
    not start -- it prints the table and refuses (plan §1). Roles on separate cards are budgeted
    against their own card, and a one-card timeshare gives each role the whole card minus the headroom,
    because the phases mean they are never resident together."""
    table = dict(table or BUDGET_GB)
    total = float(place["total_gb"])
    if place["mode"] == "cpu" or total <= 0:
        return {r: {"gb": 0.0, "fraction": 0.0} for r in table}
    scale = total / REFERENCE_GB
    if place["mode"] == "resident":
        want = {r: v * scale for r, v in table.items()}
        if sum(want.values()) > total - headroom:
            raise SystemExit(budget_table(want, total, headroom))
        return {r: {"gb": v, "fraction": min(v / total, 1.0)} for r, v in want.items()}
    if place["phases"]:   # alternating on one card: each role may use the whole card in its phase
        v = max(total - headroom, 1.0)
        return {r: {"gb": v, "fraction": min(v / total, 1.0)} for r in table}
    want = {r: min(v * scale, total - headroom) for r, v in table.items()}
    return {r: {"gb": v, "fraction": min(v / total, 1.0)} for r, v in want.items()}


def budget_table(want, total, headroom=HEADROOM_GB):
    """The refusal message: the table that does not fit, printed so the operator can see why."""
    rows = "\n".join(f"  {r:<10} {v:6.1f} GB" for r, v in sorted(want.items()))
    return (f"rvsm run: the VRAM budget does not fit on this card.\n{rows}\n"
            f"  {'sum':<10} {sum(want.values()):6.1f} GB\n"
            f"  {'card':<10} {total:6.1f} GB  (usable {total - headroom:.1f}, "
            f"{headroom:.0f} GB headroom)\n"
            f"Run with --mode timeshare (the roles alternate on the card), give the producer a second "
            f"card with --gpus 0,1, or use a smaller --size.")


def set_memory_fraction(frac, device=0):
    """`torch.cuda.set_per_process_memory_fraction`, quietly skipped without CUDA."""
    try:
        import torch
        if torch.cuda.is_available() and 0 < float(frac) <= 1.0:
            torch.cuda.set_per_process_memory_fraction(float(frac), int(device))
            return True
    except Exception as e:  # noqa: BLE001
        print(f"[run] memory fraction: {e!r}", flush=True)
    return False


# --------------------------------------------------------------------------- #
# what both sides derive the same way
# --------------------------------------------------------------------------- #
def heldout_path(out):
    return os.path.join(str(out), "eval", "heldout.json")


def heldout_set(out, cfg, recs, ax, write=True):
    """The held-out regions, `<out>/eval/heldout.json` being authoritative once it exists.

    The reference set must not move between two evaluations of one run, so the file wins over a fresh
    `regions.held_out` -- and the producer reads the same file, which is how both sides agree on which
    regions never enter the walk."""
    from rvsm import regions as RG
    d = _read_json(heldout_path(out))
    if d:
        rows = d.get("regions") if isinstance(d, dict) else d
        return [{"lo": [int(v) for v in r["lo"]], "size": [int(v) for v in r["size"]],
                 "k": int(r.get("k", 2)), "f": float(r.get("f", 1.0)),
                 "w": float(r.get("w", 1.0))} for r in rows]
    held = [{"lo": [int(v) for v in r["lo"]], "size": [int(v) for v in r["size"]], "k": int(r["k"]),
             "f": float(r.get("f", 1.0)), "w": float(r.get("w", 1.0))}
            for r in RG.held_out(recs, n=int(cfg.heldout), ax=ax)]
    if write and held:
        _write_json(heldout_path(out), {"regions": held, "seed": 0, "n": len(held), "ct": str(cfg.ct)})
    return held


def walk(cfg, recs, heldout=(), seed=0):
    """(visits, order): the deterministic walk both sides step along, from the same inputs."""
    from rvsm import regions as RG
    recs = [r for r in recs if not _inside_any(r, heldout)]
    visits = RG.region_visits(recs, cfg.visits_max)
    return visits, [int(i) for i in RG.walk_order(visits, seed)]


def _inside_any(rec, heldout):
    """Is this record entirely inside a held-out rung-2 box? (`region_list`'s own exclusion rule.)"""
    k = int(rec["k"])
    lo = np.array(rec["lo"], np.int64) << max(k - 2, 0)
    sz = np.array(rec["size"], np.int64) << max(k - 2, 0)
    for h in heldout:
        ho, hs = np.array(h["lo"], np.int64), np.array(h["size"], np.int64)
        if np.all(lo >= ho) and np.all(lo + sz <= ho + hs):
            return True
    return False


def region_size(pyr, lo, region):
    """The store shape of the region at `lo`: the region edge, clipped to the volume and rounded up to
    whole 128^3 chunks -- a region at the far corner writes a smaller store, not a store of air."""
    from rvsm import ladder, stores
    shape = np.asarray(ladder.rung_shape(pyr, 2), np.int64)
    avail = np.maximum(shape - np.asarray(lo, np.int64), 0)
    n = np.minimum(np.full(3, int(region), np.int64),
                   -(-avail // stores.CHUNK) * stores.CHUNK)
    return tuple(int(v) for v in n) if (n > 0).all() else None


def region_route(cfg, visits, order, heldout=()):
    """The rung-2 region origins in walk order, deduplicated, with the held-out regions FIRST.

    This is the producer's job list: the held-out regions are the round-0 reference every gate is
    scored against, so they are produced before anything the trainer will ever see, and after them the
    regions come in exactly the order the trainer will ask for them."""
    out, seen = [], set()
    for h in heldout:
        lo = tuple(int(v) for v in h["lo"])
        if lo not in seen:
            seen.add(lo)
            out.append(lo)
    pos = {}
    for n, i in enumerate(order):
        rec = visits[i]
        k = int(rec["k"])
        lo2 = np.array(rec["lo"], np.int64) << max(k - 2, 0)
        lo = tuple(int(v) // int(cfg.region) * int(cfg.region) for v in lo2)
        if lo in seen:
            continue
        seen.add(lo)
        pos[lo] = n
        out.append(lo)
    return out, pos


# --------------------------------------------------------------------------- #
# the producer
# --------------------------------------------------------------------------- #
class TeacherBank:
    """The round-0 teachers, loaded ONCE per producer process (and their TRT engines with them).

    `teacher_region` would otherwise re-read a 400 MB checkpoint and rebuild the engine for every
    region; a region is minutes of work, but the load is seconds of it and the engine build is
    minutes."""

    def __init__(self, cfg, out, device=None, backend="torch"):
        from rvsm import teachers as T
        self.cfg, self.out, self.device, self.backend = cfg, str(out), device, str(backend)
        names = [n for n in (cfg.teacher_ckpts or {})] or ["recto", "m7"]
        self.items = []
        for n in names:
            if n not in T.TEACHERS:
                raise SystemExit(f"rvsm run: unknown teacher {n!r} (have {sorted(T.TEACHERS)})")
            spec = T.TEACHERS[n]
            ckpt = (cfg.teacher_ckpts or {}).get(n)
            if not ckpt:
                if not spec.url:
                    raise SystemExit(f"rvsm run: teacher {n!r} has no published weights; give its path "
                                     f"in the config's `teacher_ckpts`")
                ckpt = T.fetch_weights(n)
            self.items.append([n, spec, str(ckpt), None])

    def _net(self, row):
        from rvsm import teachers as T
        if row[3] is None:
            net, spec = T.load_teacher(row[0], row[2], device=self.device or "cpu")
            row[1], row[3] = spec, net
        return row[3]

    def probs(self, ct, lo, size):
        """(fused probability, agreement weight, attrs) for one region, over every loaded teacher."""
        from rvsm import infer
        ps, names = [], []
        for row in self.items:
            net = self._net(row)
            ps.append(infer.teacher_region(ct, lo, size, row[1], row[2], device=self.device,
                                           backend=self.backend, net=net,
                                           engine_dir=os.path.join(self.out, "ckpt", "trt")))
            names.append(row[0])
        if len(ps) >= 2:
            p, rw = infer.fuse_agreement(ps[0], ps[1])
        else:
            p, rw = ps[0], np.ones_like(ps[0])
        return p, rw, {"producer": "teacher:" + ",".join(names), "radial_sign": 1,
                       "ckpt": {r[0]: r[2] for r in self.items}, "backend": self.backend}


class StudentSlot:
    """The latest `ckpt/student.pt`, reloaded when its step changes and not otherwise."""

    def __init__(self, out, device=None, compile=True):
        self.path = os.path.join(str(out), "ckpt", "student.pt")
        self.device, self.compile = device, bool(compile)
        self.st, self.mtime = None, None

    def get(self):
        from rvsm import infer
        if not os.path.exists(self.path):
            return None
        m = os.path.getmtime(self.path)
        if self.st is None or m != self.mtime:
            try:
                self.st = infer.student_fn(self.path, device=self.device, compile=self.compile)
                self.mtime = m
            except Exception as e:  # noqa: BLE001  -- a checkpoint caught mid-rename comes back next loop
                print(f"[produce] student reload: {e!r}", flush=True)
                return self.st
        return self.st


def student_rows(planes, layout, heads):
    """[(channel, uint8 block, q, encoding)] for a student pass.

    q8 for the probabilities (a probability is read as a weight, so a lossy codec is harmless) and q0
    for every FIELD, whose code 0 is the no-data marker and whose other codes are a distance in
    0.25-voxel steps -- neither of which a codec may round. The same contract `rvsm produce --student`
    writes; the encoders are `export` / `targets`' own, so there is one definition of it."""
    from rvsm import export as EX, stores, targets as TG
    first = str(layout.channels[0])
    if heads == "verso":
        return [("verso", stores.u8(planes[first]), 8, "prob_u8")]
    rec = planes["recto"]
    valid = rec > 0
    return [("recto", stores.u8(rec), 8, "prob_u8"),
            ("verso", stores.u8(planes["verso"]), 8, "prob_u8"),
            ("midline", EX.enc_signed(planes["midline"], valid), 0, "signed_u8_off128_q0.25"),
            ("thickness", TG.encode_unsigned(planes["thickness"], valid), 0, "unsigned_u8_q0.25"),
            ("conf", stores.u8(planes["conf"]), 0, "conf_u8")]


def write_rows(out, lo, rows, cfg, round_, attrs):
    from rvsm import infer, stores
    for ch, block, q, enc in rows:
        p = stores.store_path(out, ch, lo, round_)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        stores.write(p, block, tuple(int(v) for v in lo), rung=infer.RUNG, channels=(ch,), q=q,
                     volume=str(cfg.ct), umbilicus=os.path.join(str(out), "umbilicus.json"),
                     attrs={"encoding": enc, "no_data": 0, "axis_order": "ZYX", "round": int(round_),
                            **attrs})
    return [r[0] for r in rows]


def _fed_marker(out, lo, round_):
    from rvsm import regions as RG
    return os.path.join(RG.coarse_root(out, "recto", round_), "fed",
                        "region_%d_%d_%d" % tuple(int(v) for v in lo))


def feed_coarse_once(out, lo, round_, block, shape2):
    """Fold a finished recto block into the coarse rungs, once per region and round (a marker file, so
    a restart does not redo it and two producers could not double-count it)."""
    from rvsm import regions as RG
    m = _fed_marker(out, lo, round_)
    if os.path.exists(m):
        return []
    ks = RG.feed_coarse(out, "recto", lo, block, round_=round_, shape2=shape2)
    os.makedirs(os.path.dirname(m), exist_ok=True)
    with open(m, "w") as f:
        f.write(json.dumps({"rungs": ks, "t": time.time()}))
    return ks


def lookahead(cfg, out, k_active):
    """L = ceil(T_produce / T_train) * K_active + `lookahead_extra`, from the logs (plan §1).

    T_produce is the median seconds a region has taken lately (`logs/produce.jsonl`), T_train the
    seconds the trainer spends on one region of the walk (it publishes that in state.json). Without
    either, L is its floor -- which is what a cold start wants anyway."""
    extra = int(cfg.lookahead_extra)
    ps = [float(r["s"]) for r in tail_jsonl(os.path.join(str(out), "logs", "produce.jsonl"), 40)
          if isinstance(r.get("s"), (int, float))][-10:]
    tt = float(read_state(out).get("region_s") or 0.0)
    if not ps or tt <= 0:
        return int(k_active + extra)
    ratio = math.ceil(float(np.median(ps)) / max(tt, 1e-6))
    return int(min(max(ratio * int(k_active) + extra, k_active + extra), LOOKAHEAD_MAX))


def produce_loop(cfg, out, role_gpu=None, device=None, mem_frac=None, stop=None, max_s=None,
                 backend="torch"):
    """The producer: keep the lookahead window of the walk produced, for whatever round is current.

    Runs until `<out>/STOP` appears (or `stop`, a threading.Event, is set, or `max_s` elapses),
    finishing the unit in flight. Every region is one json line in `logs/produce.jsonl`."""
    from rvsm import axis as AX, ladder, regions as RG, stores, stream, targets as TG
    out = str(out)
    if role_gpu is not None and device is None:
        device = f"cuda:{int(role_gpu)}"
    if mem_frac:
        set_memory_fraction(mem_frac, 0)

    t_start = time.time()
    hb = os.path.join(out, "workers", "produce.json")
    _write_json(hb, {"pid": os.getpid(), "phase": "start", "last_ts": time.time()})

    cache = stream.ShardCache(cfg.ct, out, budget_gb=cfg.cache_gb,
                              log=lambda m: jlog(out, "produce", {"kind": "cache", "msg": str(m)},
                                                 echo=False))
    pinned = cache.pin_small_levels()
    ct_local = cache.base
    pyr = cache.levels()
    shape2 = ladder.rung_shape(pyr, 2)
    ax = AX.load(os.path.join(out, "umbilicus.json"), ct=ct_local)
    meta5 = _read_json(os.path.join(out, "meta5.json")) or None
    recs = RG.region_list(pyr, rungs=cfg.rungs, patch=cfg.patch, region=cfg.region,
                          boost=cfg.rung_boost, occ_min_fine=cfg.occ_min_fine,
                          occ_min_coarse=cfg.occ_min_coarse)
    held = heldout_set(out, cfg, recs, ax, write=False)
    visits, order = walk(cfg, recs, held)
    route, pos = region_route(cfg, visits, order, held)
    k_active = max(len([k for k in cfg.rungs if int(k) < RG.COARSE_RUNGS[0]]), 1)
    frungs = field_rungs(cfg)
    jobs = max(min(int(os.cpu_count() or 1) // 2, 4), 1)
    jlog(out, "produce", {"kind": "start", "pid": os.getpid(), "device": str(device),
                          "regions": len(route), "heldout": len(held), "pinned": pinned,
                          "backend": str(backend), "field_rungs": list(frungs), "jobs": jobs})

    bank, slot = None, StudentSlot(out, device=device, compile=cfg.compile)
    keys = {}
    L = k_active + int(cfg.lookahead_extra)
    t_reest = 0.0

    def stopping():
        return stop_requested(out) or (stop is not None and stop.is_set()) or \
            (max_s is not None and time.time() - t_start > float(max_s))

    try:
        while not stopping():
            st = read_state(out)
            round_ = int(st.get("round", 0))
            verso_on = bool(st.get("verso_on", False))
            cursor = int(st.get("cursor", 0))
            if time.time() - t_reest > REEST_S:
                L, t_reest = lookahead(cfg, out, k_active), time.time()
            _write_json(hb, {"pid": os.getpid(), "phase": f"round{round_}", "last_ts": time.time(),
                             "L": L, "cursor": cursor})

            # one-card timeshare: no PHASE file means nobody is taking turns
            if os.path.exists(os.path.join(out, PHASE_FILE)) and read_phase(out) != "produce":
                time.sleep(IDLE_S)
                continue
            if free_gb(out) < float(cfg.reserve_gb):
                jlog(out, "produce", {"kind": "backpressure", "free_gb": round(free_gb(out), 1)})
                time.sleep(IDLE_S * 5)
                continue

            cat = RG.Catalog(out, round_, ttl=1.0)
            window = _window(route, pos, cursor, L, held)
            did = False
            for lo in window:
                if stopping():
                    break
                size = region_size(pyr, lo, cfg.region)
                if size is None:
                    continue
                job = _next_job(cat, lo, round_, verso_on, out, rungs=frungs)
                if job is None:
                    continue
                did = True
                t0 = time.time()
                if lo not in keys:
                    keys[lo] = cache.fetch_region(np.array(lo, np.int64), ctx=cfg.ctx,
                                                  patch=cfg.patch, region=cfg.region)
                if job == "teacher":
                    if bank is None:
                        bank = TeacherBank(cfg, out, device=device, backend=backend)
                    p, rw, attrs = bank.probs(ct_local, lo, size)
                    write_rows(out, lo, [("recto", stores.u8(p), 8, "prob_u8"),
                                         ("rw", stores.u8(rw), 8, "prob_u8")], cfg, round_, attrs)
                    feed_coarse_once(out, lo, round_, stores.u8(p), shape2)
                elif job in ("verso", "self"):
                    stu = slot.get()
                    if stu is None:
                        did = False
                        break
                    heads = "verso" if job == "verso" else "all"
                    sign = -1.0 if job == "verso" else 1.0
                    want = [str(stu.layout.channels[0])] if heads == "verso" else "all"
                    planes = _student_planes(stu, ct_local, ax, lo, size, sign, want, meta5, pyr)
                    attrs = {"producer": "student", "ckpt": stu.ckpt, "step": int(stu.step),
                             "radial_sign": int(sign), "window": int(stu.cfg.infer_window),
                             "halo": int(stu.cfg.infer_halo),
                             "cascade_depth": int(stu.cfg.cascade_depth),
                             "temps": {str(k): float(v) for k, v in stu.temps.items()}}
                    rows = student_rows(planes, stu.layout, heads)
                    write_rows(out, lo, rows, cfg, round_, attrs)
                    if heads == "all":
                        feed_coarse_once(out, lo, round_, rows[0][1], shape2)
                elif job == "fields":
                    TG.region_fields(out, lo, ax, round_=round_, rungs=frungs, jobs=jobs)
                jlog(out, "produce", {"kind": job, "region": list(lo), "round": round_,
                                      "s": round(time.time() - t0, 2), "L": L,
                                      "cursor": cursor, **_vram()})
                _release_passed(cache, keys, pos, cursor, cat, round_, verso_on, out, frungs)
            if not did:
                _release_passed(cache, keys, pos, cursor, cat, round_, verso_on, out, frungs)
                time.sleep(IDLE_S)
                if read_phase(out, "") == "produce":
                    write_phase(out, "train")     # the window is drained: give the card back
    finally:
        try:
            cache.close()
        except Exception:  # noqa: BLE001
            pass
        _write_json(hb, {"pid": os.getpid(), "phase": "exit", "last_ts": time.time()})
        jlog(out, "produce", {"kind": "exit", "pid": os.getpid()})
    return 0


def _vram():
    try:
        import torch
        if torch.cuda.is_available():
            return {"vram_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _student_planes(stu, ct, ax, lo, size, sign, want, meta5, pyr):
    from rvsm import infer
    return infer.student_region(stu, ct, ax, lo, size, sign=sign, heads=want, meta=meta5, pyr=pyr)


def _window(route, pos, cursor, L, held):
    """The producer's working set: every held-out region that is not finished, then the regions whose
    walk position is within `L` visits of the trainer's cursor."""
    out = [lo for lo in route[:len(held)]]
    for lo in route[len(held):]:
        n = pos.get(lo)
        if n is None or n < cursor:
            continue
        if n <= cursor + int(L):
            out.append(lo)
    return out


def field_rungs(cfg):
    """The distance rungs this run needs: 2..4, and only the ones it trains on. A run whose ladder
    stops at rung 3 has no use for a rung-4 EDT, and an EDT is the most expensive thing the producer
    does that is not a network."""
    from rvsm import targets as TG
    ks = tuple(k for k in (2, 3, 4) if int(k) <= TG.MAX_RUNG and int(k) in tuple(cfg.rungs))
    return ks or (2,)


def _next_job(cat, lo, round_, verso_on, out, rungs=(2, 3, 4)):
    """Which pass this region lacks, in the order the state machine allows -- or None when it is done.

    Round 0: the teacher pass, then (once the gate has fired) the flipped-sign verso, then the distance
    fields. Round r >= 1: one multi-head student pass, then the fields at the pooled rungs."""
    from rvsm import stores, targets as TG
    if round_ == 0:
        if not cat.done("recto", lo):
            return "teacher"
        if verso_on and not cat.done("verso", lo):
            return "verso"
        if cat.done("verso", lo) and not stores.is_done(
                stores.store_path(out, TG.channel("midline", max(rungs)), lo, round_)):
            return "fields"
        return None
    if not cat.done("recto", lo):
        return "self"
    if not stores.is_done(stores.store_path(out, TG.channel("midline", max(rungs)), lo, round_)):
        return "fields"
    return None


def _release_passed(cache, keys, pos, cursor, cat, round_, verso_on, out, rungs=(2, 3, 4)):
    """Give back the CT of every region whose stores are finished and whose walk position the trainer's
    cursor has passed. A region still ahead of the cursor keeps its shards: the trainer is about to
    read them."""
    for lo in list(keys):
        n = pos.get(lo)
        if n is None or n >= int(cursor):
            continue
        if _next_job(cat, lo, round_, verso_on, out, rungs=rungs) is None:
            cache.release(keys.pop(lo))


def _produce_entry(cfg_json, out, gpu, frac, backend):
    """The spawned producer's entry point. Sets `CUDA_VISIBLE_DEVICES` BEFORE torch is imported, which
    is why this module imports torch nowhere at the top level."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    cfg = CFG.Config(**{k: CFG._coerce(k, v) for k, v in cfg_json.items() if k in CFG._TYPES})
    try:
        return produce_loop(cfg, out, role_gpu=None, device=("cuda:0" if gpu is not None else None),
                            mem_frac=frac, backend=backend)
    except Exception as e:  # noqa: BLE001  -- the supervisor restarts a producer that dies
        jlog(out, "produce", {"kind": "crash", "err": repr(e)})
        raise


# --------------------------------------------------------------------------- #
# the gates
# --------------------------------------------------------------------------- #
def heldout_rows(cfg, out, ckpt, held, ax, meta5=None, round_=0, device=None, n=GATE_REGIONS,
                 ct=None):
    """`compare_stores` of the student's own recto against the round-0 reference, per held-out region.

    Each row also carries the reference against ITSELF (`base_*`), which is the only honest baseline
    for a topological error: a betti0 error of 3 means nothing until you know what the reference scores
    against a perfect copy of itself."""
    from rvsm import evalsurf as EV, infer, stores
    rows = []
    stu = None
    for h in list(held)[:int(n)]:
        lo = tuple(int(v) for v in h["lo"])
        p = stores.store_path(out, "recto", lo, 0)
        if not stores.is_done(p):
            continue
        ref = np.asarray(stores.open_store(p)[:], np.uint8)
        if stu is None:
            stu = infer.student_fn(ckpt, device=device, compile=False)
        planes = infer.student_region(stu, ct or cfg.ct, ax, lo, ref.shape, sign=1.0,
                                      heads=[str(stu.layout.channels[0])], meta=meta5)
        pred = stores.u8(planes[str(stu.layout.channels[0])])
        r = EV.compare_stores(ref, pred)
        b = EV.compare_stores(ref, ref)
        rows.append({"region": list(lo), **r, **{f"base_{k}": v for k, v in b.items()
                                                 if k.startswith("betti") or k == "euler"}})
    return rows


def _boot(vals, n=200, seed=0, lo=2.5, hi=97.5):
    """(mean, lo, hi) over REGIONS: eight regions are the independent unit, their voxels are not."""
    v = np.asarray([x for x in vals if np.isfinite(x)], np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(int(seed))
    m = rng.integers(0, v.size, size=(int(n), v.size))
    b = v[m].mean(1)
    return float(v.mean()), float(np.percentile(b, lo)), float(np.percentile(b, hi))


def verso_gate(cfg, out, step, rows_fn=None):
    """Has round 0's recto earned the flipped-sign verso passes?

    Pass when the pooled dice over the held-out regions is at least `verso_gate_dice` AND the betti0
    error is no worse than the reference-against-itself baseline by more than the bootstrap CI's width
    -- or unconditionally at `verso_after_steps`, which is the fallback the plan gives the gate so a
    run can never stall on it."""
    if int(step) >= int(cfg.verso_after_steps):
        return True, {"why": "verso_after_steps", "step": int(step)}
    rows = list((rows_fn() if rows_fn is not None else None) or [])
    if not rows:
        return False, {"why": "no held-out reference yet"}
    d, dlo, dhi = _boot([r["dice"] for r in rows])
    b, blo, bhi = _boot([r.get("betti0_err", float("nan")) for r in rows])
    base, _, _ = _boot([r.get("base_betti0_err", 0.0) for r in rows])
    ci = max(bhi - blo, 0.0)
    ok = bool(d >= float(getattr(cfg, "verso_gate_dice", 0.6))) and \
        (not np.isfinite(b) or b <= base + ci)
    return ok, {"why": "compare_stores", "dice": d, "dice_ci": [dlo, dhi], "betti0_err": b,
                "betti0_base": base, "betti0_ci": [blo, bhi], "n": len(rows)}


def plateau(out, key="dice", min_points=6, gain=ROUND_GAIN):
    """Is the eval curve flat? `fit_curve` on `logs/eval.jsonl`: the remaining gain to the fitted
    asymptote, as a fraction of it. Too few points is NOT a plateau."""
    from rvsm import evalsurf as EV
    recs = [(int(r["step"]), float(r[key])) for r in
            tail_jsonl(os.path.join(str(out), "logs", "eval.jsonl"), 500)
            if isinstance(r.get(key), (int, float)) and np.isfinite(r.get(key))]
    if len(recs) < int(min_points):
        return False, {"why": "too few eval points", "n": len(recs)}
    s, y = [q[0] for q in recs], [q[1] for q in recs]
    fit = EV.fit_curve(s, y)
    c = fit.get("asymptote")
    if fit.get("model") is None or c is None or not np.isfinite(c) or c <= 0:
        return False, {"why": "no fit", **{k: v for k, v in fit.items() if k in ("model", "error")}}
    if fit.get("asymptote_at_bound"):
        # the fit ran into the top of the parameter range: the asymptote is not to be believed, and
        # "no remaining gain" is exactly the answer such a fit gives for free
        return False, {"why": "asymptote at the bound", "n": len(recs)}
    rem = max(float(fit.get("remaining", 0.0)), 0.0) / max(float(c), 1e-9)
    return bool(rem < float(gain)), {"why": "fit_curve", "asymptote": float(c),
                                     "last": float(fit.get("last_value", y[-1])),
                                     "remaining": rem, "model": fit.get("model"), "n": len(recs)}


def round_gate(cfg, out, step, round_, rows_fn=None, ref=None):
    """Should round `round_` end and round `round_ + 1` open?

    The plateau (or `round_steps`) opens the question -- and only then is the held-out comparison run,
    because it costs a student pass per region. It answers it: the merge side (`precision`, how much of
    what the student calls sheet the reference agrees with) and `betti0_err` must not be worse than the
    ROUND-0 reference beyond the bootstrap CI. A round that fails is DISCARDED: the round does not
    advance, the training continues, and WSD makes that extension free (plan §1).

    `ref` is what the previous gate measured. Round 0 has no previous round to be worse than, so its
    only condition is the plateau -- "not worse than the round-0 reference" starts to mean something at
    round 1. `rows_fn()` returns the held-out rows and is called at most once.

    Returns (fire, why) and the rows it measured in `why["rows"]`, so the caller can keep them as the
    next round's reference."""
    flat, why = plateau(out, "dice")
    if not (flat or int(step) >= int(cfg.round_steps)):
        return False, {"why": "not a plateau", **why}
    rows = list((rows_fn() if rows_fn is not None else None) or [])
    if not rows:
        return True, {"why": "plateau, no held-out reference to veto it", **why}
    prec, plo, phi = _boot([r.get("precision", float("nan")) for r in rows])
    b, blo, bhi = _boot([r.get("betti0_err", float("nan")) for r in rows])
    got = {"precision": prec, "betti0_err": b}
    ok = True
    if ref:
        ok = (not np.isfinite(b) or b <= float(ref.get("betti0_err", b)) + max(bhi - blo, 0.0)) and \
             (not np.isfinite(prec) or prec >= float(ref.get("precision", prec)) - max(phi - plo, 0.0))
    return bool(ok), {"why": "plateau + quality", "precision": prec, "betti0_err": b,
                      "ref": ref, "ok": bool(ok), "rows": got, "n": len(rows), **why}


# --------------------------------------------------------------------------- #
# the trainer side: the walk cursor and the lookahead
# --------------------------------------------------------------------------- #
def cursor_dir(out):
    return os.path.join(str(out), "logs", "cursor")


def read_cursor(out):
    """The walk position of the SLOWEST sampler worker: the producer must stay ahead of that one."""
    d = cursor_dir(out)
    vals = []
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        r = _read_json(os.path.join(d, n))
        if r and isinstance(r.get("pos"), int):
            vals.append(int(r["pos"]) * max(int(r.get("stride", 1)), 1))
    return min(vals) if vals else 0


def region_seconds(out):
    """Seconds per consumed walk entry, averaged over the workers (T_train of the lookahead rule)."""
    d = cursor_dir(out)
    vals = []
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        r = _read_json(os.path.join(d, n)) or {}
        if isinstance(r.get("region_s"), (int, float)) and r["region_s"] > 0:
            vals.append(float(r["region_s"]))
    return float(np.mean(vals)) if vals else 0.0


# The trainer's sampler is a `sample.Patches` subclass built on first use, so this module imports
# neither torch nor sample at the top level (the spawned producer must set CUDA_VISIBLE_DEVICES before
# torch is imported at all).
_WALK_PATCHES = None


def walk_patches():
    """The trainer's sampler: `sample.Patches`' walk with the lookahead rule on top.

    `sample.Patches` steps along the walk and skips a region whose stores are not finished. That is
    almost the rule the plan asks for, but not quite: the trainer must take the FIRST READY region
    inside a window of `L` visits (so it never runs ahead of the producer into an unproduced tail), a
    region whose verso lands after its visit earns ONE revisit with fresh windows, and the position
    reached has to be published, so the producer knows what to prepare and what to release.

    It is a thin subclass rather than a rewrite: the drawing, the targets, the context cubes and the
    planes are all `Patches`', and only `__iter__` differs."""
    global _WALK_PATCHES
    if _WALK_PATCHES is not None:
        return _WALK_PATCHES
    from rvsm import sample

    class _WalkPatches(sample.Patches):
        def __init__(self, cfg, out, *, lookahead_n=8, wait_s=WAIT_S, **kw):
            super().__init__(cfg, **kw)
            self.out = str(out)
            self.L = int(lookahead_n)
            self.wait_s = float(wait_s)

        def _publish(self, w, W, pos, region_s):
            _write_json(os.path.join(cursor_dir(self.out), f"w{int(w)}.json"),
                        {"pos": int(pos), "stride": int(W), "worker": int(w),
                         "region_s": float(region_s), "t": time.time()})

        def _region_lo(self, rec):
            k = int(rec["k"])
            lo2 = np.array(rec["lo"], np.int64) << max(k - 2, 0)
            return tuple(int(v) // int(self.cfg.region) * int(self.cfg.region) for v in lo2)

        def __iter__(self):
            import torch
            if self.pyr is None:
                self._open()
            info = torch.utils.data.get_worker_info()
            w, W = (info.id, info.num_workers) if info else (0, 1)
            rng = np.random.default_rng(self.seed + 1000 * w)
            mine = [int(i) for i in self.order[w::W]] or [int(i) for i in self.order]
            pos, pend, revisit, seen_no_verso = 0, [], [], {}
            t_last, region_s = time.time(), 0.0
            while True:
                while len(pend) < max(self.L, 1) and pos < len(mine):
                    pend.append(mine[pos])
                    pos += 1
                if not pend:                       # the walk is exhausted: start it again
                    pos = 0
                    continue
                pick = next((j for j, i in enumerate(pend) if self._visitable(self.visits[i])), None)
                if pick is None:
                    if stop_requested(self.out):
                        return
                    jlog(self.out, "train", {"kind": "wait", "worker": int(w),
                                             "train_wait_s": self.wait_s, "pending": len(pend)},
                         echo=False)
                    time.sleep(self.wait_s)
                    self.cat = type(self.cat)(self.root, self.round)   # drop the cached MISSes
                    continue
                i = pend.pop(pick)
                rec = self.visits[i]
                lo = self._region_lo(rec)
                had_verso = self.cat.done("verso", lo)
                if not had_verso and i not in seen_no_verso:
                    seen_no_verso[i] = True
                    revisit.append(i)
                left, fails = self.windows, 0
                while left > 0 and fails < 8 * max(self.windows, 1):
                    got = self._draw(rng, rec)
                    if got is None:
                        fails += 1
                        continue
                    left, fails = left - 1, 0
                    yield got
                now = time.time()
                region_s = 0.5 * region_s + 0.5 * (now - t_last) if region_s else now - t_last
                t_last = now
                self._publish(w, W, pos - len(pend), region_s)
                # a region whose verso landed after its visit earns exactly one revisit
                for j in list(revisit):
                    if self.cat.done("verso", self._region_lo(self.visits[j])):
                        revisit.remove(j)
                        pend.insert(0, j)

    _WALK_PATCHES = _WalkPatches
    return _WALK_PATCHES


# --------------------------------------------------------------------------- #
# the supervisor
# --------------------------------------------------------------------------- #
def setup(cfg, out=None):
    """Freeze what the run is, once: config.json (asserted on a resume), umbilicus.json, metadata.json,
    the CT mirror and the held-out set. Returns the context both halves need."""
    from rvsm import axis as AX, ladder, regions as RG, scanmeta as SM, stream
    out = str(out or cfg.out)
    for d in ("logs", "ckpt", "stores", "eval", "workers"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    cp = os.path.join(out, "config.json")
    old = _read_json(cp)
    if old:
        got = old.get("fingerprint")
        assert got == cfg.fingerprint(), (
            f"resume: {cp} was written by a config whose fingerprint is {got}, this run's is "
            f"{cfg.fingerprint()}. Everything but {CFG.FINGERPRINT_EXCLUDE} must match.")
    _write_json(cp, cfg.to_json())

    AX.ensure(out, cfg.umbilicus, ct=cfg.ct)
    meta = SM.fetch(cfg.ct)
    _write_json(os.path.join(out, "metadata.json"), meta)
    meta5 = [float(v) for v in SM.scan_planes(meta)]
    _write_json(os.path.join(out, "meta5.json"), meta5)

    mirror = stream.ShardCache(cfg.ct, out, budget_gb=cfg.cache_gb)
    try:
        mirror.meta()
        ct_local = mirror.base
    finally:
        mirror.close()

    ax = AX.load(os.path.join(out, "umbilicus.json"), ct=ct_local)
    pyr = ladder.rungs(ct_local)
    recs = RG.region_list(pyr, rungs=cfg.rungs, patch=cfg.patch, region=cfg.region,
                          boost=cfg.rung_boost, occ_min_fine=cfg.occ_min_fine,
                          occ_min_coarse=cfg.occ_min_coarse)
    held = heldout_set(out, cfg, recs, ax)
    visits, order = walk(cfg, recs, held)
    route, _pos = region_route(cfg, visits, order, held)
    st = read_state(out)
    write_state(out, round=int(st.get("round", 0)), step=int(st.get("step", 0)),
                cursor=int(st.get("cursor", 0)), verso_on=bool(st.get("verso_on", False)),
                fingerprint=cfg.fingerprint(), ct=str(cfg.ct), dir=out)
    return {"out": out, "ct": ct_local, "ax": ax, "pyr": pyr, "meta5": meta5, "records": recs,
            "heldout": held, "visits": visits, "order": order, "route": route}


def heartbeat(out, place, stop_ev, procs, respawn):
    """The supervisor's own thread: stamp `workers.json` and `logs/sched.jsonl`, and restart a producer
    that has gone silent for longer than `SILENT_MAX_S` (plan §1)."""
    while not stop_ev.is_set():
        w = {"mode": place["mode"], "phases": bool(place["phases"]), "t": time.time(),
             "train": {"pid": os.getpid(), "phase": "train", "last_ts": time.time(),
                       **{k: read_state(out).get(k) for k in ("step", "round", "cursor", "verso_on")}}}
        p = _read_json(os.path.join(out, "workers", "produce.json")) or {}
        w["produce"] = p
        _write_json(os.path.join(out, "workers.json"), w)
        jlog(out, "sched", {"kind": "heartbeat", **{k: w["train"].get(k) for k in
                                                    ("step", "round", "cursor")},
                            "produce_phase": p.get("phase"),
                            "produce_age_s": round(time.time() - float(p.get("last_ts") or 0), 1)},
             echo=False)
        age = time.time() - float(p.get("last_ts") or time.time())
        pr = procs.get("produce")
        if pr is not None and age > SILENT_MAX_S and not stop_requested(out):
            jlog(out, "sched", {"kind": "restart", "reason": "producer silent", "age_s": round(age)})
            try:
                pr.terminate()
                pr.join(30)
            except Exception:  # noqa: BLE001
                pass
            procs["produce"] = respawn()
        stop_ev.wait(HEARTBEAT_S)


def run(cfg, out=None, init=None, device=None, backend="torch", producer=True):
    """The whole run: freeze, place the roles, spawn the producer, be the trainer, drive the rounds.

    Returns the checkpoint path. `producer=False` runs the trainer alone (the `rvsm train` case with a
    driver's bookkeeping); `--mode cpu` runs the producer as a thread in this process, which is what
    the end-to-end test and any CPU smoke want -- there is no card to share.
    """
    import threading

    from rvsm import calib as CAL, infer, sample, train as TR
    ctx = setup(cfg, out)
    out = ctx["out"]
    found = cards(cfg.gpus)
    place = choose_mode(cfg.mode, found)
    bud = budget(place)
    jlog(out, "sched", {"kind": "start", "mode": place["mode"], "phases": place["phases"],
                        "gpus": [g for g, _ in found], "budget_gb": {k: v["gb"] for k, v in bud.items()},
                        "regions": len(ctx["records"]), "heldout": len(ctx["heldout"])})
    if place["phases"]:
        write_phase(out, "train")
    elif os.path.exists(os.path.join(out, PHASE_FILE)):
        os.remove(os.path.join(out, PHASE_FILE))
    if os.path.exists(os.path.join(out, STOP_FILE)):
        os.remove(os.path.join(out, STOP_FILE))

    dev = device or (f"cuda:{place['train_gpu']}" if place["train_gpu"] is not None else "cpu")
    if place["train_gpu"] is not None:
        set_memory_fraction(bud["train"]["fraction"], place["train_gpu"])

    # ---- the producer: a spawned process, or a thread when there is no card to share
    procs, stop_ev = {}, threading.Event()
    cfg_json = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}

    def spawn_producer():
        if not producer:
            return None
        if place["mode"] == "cpu":
            t = threading.Thread(target=produce_loop, args=(cfg, out),
                                 kwargs={"device": "cpu", "stop": stop_ev, "backend": backend},
                                 daemon=True)
            t.start()
            return t
        import multiprocessing as mp
        p = mp.get_context("spawn").Process(
            target=_produce_entry, args=(cfg_json, out, place["produce_gpu"],
                                         bud["produce"]["fraction"], backend), daemon=False)
        p.start()
        return p

    procs["produce"] = spawn_producer()
    hb = threading.Thread(target=heartbeat, args=(out, place, stop_ev, procs, spawn_producer),
                          daemon=True)
    hb.start()

    # ---- the cold start: the producer finishes `min_regions_before_train` regions first
    from rvsm import regions as RG
    cat = RG.Catalog(out, 0, ttl=1.0)
    want = min(int(cfg.min_regions_before_train), max(len(ctx["route"]), 1))
    t0 = time.time()
    while producer and len(cat.list_done("recto")) < want and not stop_requested(out):
        if time.time() - t0 > 3600:
            break
        time.sleep(1.0)
        cat = RG.Catalog(out, 0, ttl=1.0)
    jlog(out, "sched", {"kind": "cold_start", "recto_done": len(cat.list_done("recto")),
                        "wanted": want, "s": round(time.time() - t0, 1)})

    # ---- the trainer, round by round. The validation grid is ALWAYS round 0's stores: the round-0
    # reference is the fixed reference every round is compared against (plan §3).
    ckpt = os.path.join(out, "ckpt", "student.pt")
    val = sample.val_grid(cfg, ctx["heldout"], root=out, ct=ctx["ct"], ax=ctx["ax"], round_=0,
                          meta=ctx["meta5"])
    state = {"round": int(read_state(out).get("round", 0)), "stop": False, "ref": None}
    k_active = max(len([k for k in cfg.rungs if int(k) < RG.COARSE_RUNGS[0]]), 1)
    resume = os.path.exists(ckpt)

    def hook(info):
        """Every `eval_every` steps: publish the state, honour STOP and PHASE, run the two gates."""
        step = int(info["step"])
        write_state(out, step=step, round=state["round"], cursor=read_cursor(out),
                    region_s=region_seconds(out),
                    verso_on=bool(read_state(out).get("verso_on", False)))
        if stop_requested(out):
            jlog(out, "sched", {"kind": "stop", "step": step})
            state["stop"] = True
            return True
        if place["phases"]:
            _phase_pause(cfg, out, info, dev)
        st = read_state(out)
        box = {}

        def rows_fn():
            """The held-out comparison, computed at most once per evaluation and only when a gate has
            got far enough to want it: it is a student pass per region."""
            if "rows" not in box:
                box["rows"] = heldout_rows(cfg, out, ckpt, ctx["heldout"], ctx["ax"],
                                           meta5=ctx["meta5"], device=dev, ct=ctx["ct"]) \
                    if os.path.exists(ckpt) else []
            return box["rows"]

        if state["round"] == 0 and not st.get("verso_on"):
            ok, why = verso_gate(cfg, out, step, rows_fn)
            jlog(out, "sched", {"kind": "verso_gate", "step": step, "pass": bool(ok), **why})
            if ok:
                write_state(out, verso_on=True, verso_gate_step=step)
        if state["round"] + 1 < int(cfg.rounds):
            ok, why = round_gate(cfg, out, step, state["round"], rows_fn, state["ref"])
            if why.get("why") != "not a plateau":
                jlog(out, "sched", {"kind": "round_gate", "step": step, "round": state["round"],
                                    "pass": bool(ok), **why})
            if ok:
                nxt = state["round"] + 1
                tp = os.path.join(out, "ckpt", f"teacher_round_{nxt}.pt")
                infer.save_student(tp, {k: v.cpu() for k, v in info["ema"].items()}, cfg,
                                   temps=CAL.temps_of(info.get("temps") or {}), step=step,
                                   round=nxt)
                state["ref"] = why.get("rows") or state["ref"]
                state["round"] = nxt
                write_state(out, round=nxt, teacher=tp, round_step=step)
                jlog(out, "sched", {"kind": "round", "round": nxt, "teacher": tp, "step": step})
                return True
        return False

    def patches_factory():
        ds = walk_patches()(cfg, out, root=out, ct=ctx["ct"], ax=ctx["ax"], round_=state["round"],
                            heldout=ctx["heldout"], meta=ctx["meta5"],
                            region_records=ctx["records"],
                            lookahead_n=lookahead(cfg, out, k_active))
        return sample.loader(ds, workers=cfg.workers, batch=cfg.batch)

    ck = ckpt
    try:
        for _ in range(max(int(cfg.rounds), 1)):
            ck = TR.train(cfg, out=out, init=init, resume=resume, patches_factory=patches_factory,
                          device=dev, val_items=val, hook=hook, ckpt=ckpt)
            init, resume = None, True
            if state["stop"]:
                break
            if int(read_state(out).get("step", 0)) >= int(cfg.steps):
                break
            _clean_old_rounds(out, state["round"])
    finally:
        stop_ev.set()
        request_stop(out)
        pr = procs.get("produce")
        if pr is not None:
            try:
                pr.join(120)
            except Exception:  # noqa: BLE001
                pass
            if hasattr(pr, "terminate") and getattr(pr, "is_alive", lambda: False)():
                pr.terminate()
        jlog(out, "sched", {"kind": "exit", "round": state["round"],
                            "step": int(read_state(out).get("step", 0)), "ckpt": ck})
    return ck


def _phase_pause(cfg, out, info, dev):
    """One-card timeshare: hand the card to the producer and wait for it back.

    The net and the optimizer go to the CPU and the cache is emptied, so the producer's process can
    allocate; both processes stay alive across the swap (the compile cache survives, and the ~1 min
    recompile on the way back is under 5 % of a 20-minute phase)."""
    import torch
    if read_phase(out) != "train":
        return
    t0 = float(info.get("phase_t0") or 0.0)
    started = _PHASE_T0.setdefault(out, time.time())
    if time.time() - max(started, t0) < float(cfg.train_min) * 60.0:
        return
    net, opt = info.get("net"), info.get("opt")
    jlog(out, "sched", {"kind": "phase", "to": "produce", "step": int(info["step"])})
    if net is not None:
        net.to("cpu")
    if opt is not None:
        for s in opt.state.values():
            for k, v in list(s.items()):
                if torch.is_tensor(v):
                    s[k] = v.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_phase(out, "produce")
    t = time.time()
    while read_phase(out) == "produce" and not stop_requested(out):
        if time.time() - t > float(cfg.produce_max_min) * 60.0:
            write_phase(out, "train")
            break
        time.sleep(1.0)
    if net is not None:
        net.to(dev)
    if opt is not None:
        for s in opt.state.values():
            for k, v in list(s.items()):
                if torch.is_tensor(v):
                    s[k] = v.to(dev)
    _PHASE_T0[out] = time.time()
    jlog(out, "sched", {"kind": "phase", "to": "train", "step": int(info["step"]),
                        "produced_s": round(time.time() - t, 1)})


_PHASE_T0 = {}


def _clean_old_rounds(out, round_, keep=2):
    """At most `keep` rounds live on disk: a region's round-r stores go once round r+1 has superseded
    them for that region (plan §1). Nothing is deleted while it is the newest thing there is."""
    from rvsm import regions as RG
    old = int(round_) - int(keep)
    if old < 0:
        return []
    new = RG.Catalog(out, int(round_), ttl=1.0)
    gone = []
    for ch in ("recto", "rw", "verso", "midline", "thickness", "conf"):
        d = os.path.join(out, "stores", f"round_{old}", ch)
        if not os.path.isdir(d):
            continue
        for lo in RG.Catalog(out, old, ttl=1.0).list_done(ch):
            if not new.done("recto", lo):
                continue
            p = os.path.join(d, "region_%d_%d_%d.zarr" % lo)
            shutil.rmtree(p, ignore_errors=True)
            gone.append(p)
    if gone:
        jlog(out, "sched", {"kind": "gc", "round": old, "removed": len(gone)})
    return gone


# --------------------------------------------------------------------------- #
# `rvsm status` / `rvsm stop`
# --------------------------------------------------------------------------- #
def store_counts(out, rounds=4):
    """{round: {channel: n done}} from the directory itself -- the same scan `ledger --rebuild` is."""
    from rvsm import stores
    got = {}
    for r in range(int(rounds)):
        d = os.path.join(str(out), "stores", f"round_{r}")
        if not os.path.isdir(d):
            continue
        per = {}
        for ch in sorted(os.listdir(d)):
            p = os.path.join(d, ch)
            if not os.path.isdir(p):
                continue
            n = sum(1 for n_ in os.listdir(p)
                    if n_.endswith(".zarr") and stores.is_done(os.path.join(p, n_)))
            if n:
                per[ch] = n
        if per:
            got[r] = per
    return got


def rates(out):
    """(produce s/region, train steps/s) from the tails of the two logs."""
    pr = [float(r["s"]) for r in tail_jsonl(os.path.join(str(out), "logs", "produce.jsonl"), 60)
          if isinstance(r.get("s"), (int, float))][-10:]
    tr = [(float(r["t"]), int(r["step"])) for r in
          tail_jsonl(os.path.join(str(out), "logs", "train.jsonl"), 60)
          if isinstance(r.get("step"), int) and isinstance(r.get("t"), (int, float))][-10:]
    sps = 0.0
    if len(tr) >= 2 and tr[-1][0] > tr[0][0]:
        sps = (tr[-1][1] - tr[0][1]) / (tr[-1][0] - tr[0][0])
    return (float(np.median(pr)) if pr else 0.0), sps


def status(out, log=print):
    """What the run is doing, from the directory alone: no process is asked anything."""
    out = str(out)
    st = read_state(out)
    w = _read_json(os.path.join(out, "workers.json")) or {}
    ev = tail_jsonl(os.path.join(out, "logs", "eval.jsonl"), 1)
    ps, sps = rates(out)
    log(f"rvsm status {out}")
    log(f"  state     round {st.get('round')}  step {st.get('step')}  cursor {st.get('cursor')}  "
        f"verso_on {st.get('verso_on')}  fingerprint {st.get('fingerprint')}")
    log(f"  mode      {w.get('mode')}  phases {w.get('phases')}  phase file "
        f"{read_phase(out, '-')}  STOP {stop_requested(out)}")
    for role in ("train", "produce"):
        r = w.get(role) or {}
        age = time.time() - float(r.get("last_ts") or 0) if r.get("last_ts") else float("nan")
        log(f"  {role:<9} pid {r.get('pid')}  phase {r.get('phase')}  last seen {age:.0f}s ago")
    for rnd, per in sorted(store_counts(out).items()):
        log(f"  stores    round {rnd}: " + "  ".join(f"{k} {v}" for k, v in sorted(per.items())))
    log(f"  rate      produce {ps:.1f} s/region   train {sps:.2f} steps/s")
    if ev:
        log("  eval      " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in ev[-1].items() if k != "t"}))
    return {"state": st, "workers": w, "stores": store_counts(out), "produce_s": ps, "steps_s": sps,
            "eval": ev[-1] if ev else None}


def stop(out, log=print):
    """Ask a live run to finish the unit in flight and exit."""
    p = request_stop(out)
    log(f"rvsm stop: touched {p}; the trainer checkpoints and exits at its next evaluation, the "
        f"producer after the region it is on.")
    return p
