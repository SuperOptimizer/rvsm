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
VERSO_HOLD_FILE = "VERSO_HOLD"   # while it exists the verso gate never turns verso on (a manual hold)
PHASE_FILE = "PHASE"
PAUSE_FILE = "PAUSE_RAM"    # while it exists the producer starts no new unit (the host-RAM guard's)
RAM_PAUSE_FRAC = 0.85       # host memory in use (or the run's process-tree RSS) above this share of
RAM_RESUME_FRAC = 0.75      # MemTotal pauses the producer; below this share it resumes
RAM_CHECK_S = 5.0           # the guard looks this often (the paris4 OOM went 90 % -> 99.5 % in 4 s)
HEARTBEAT_S = 30.0          # how often the supervisor stamps workers.json / logs/sched.jsonl
SILENT_MAX_S = 1800.0       # a producer that has not stamped its heartbeat for this long is restarted
LEASE_POLL_S = 0.25         # how often the producer's lease keeper looks for new leases
LEASE_RETRY_S = 10.0        # a leased region whose (re-)fetch left it incomplete is retried this often
UNIT_STALL_S = 2700.0       # a live producer whose loop has not progressed this long is a STALL (alert);
                            # twice this long and it is restarted like a silent one
HB_TICK_S = 60.0            # the producer's own stamping thread: alive while a long unit runs
RESTART_MIN_S = 10.0        # a dead producer is restarted AT ONCE; a second death waits this long, doubling
RESTART_MAX_S = 600.0       # ... up to this
RESTART_RESET_S = 1800.0    # a producer that has lived this long has earned a fresh backoff
RESTART_FATAL = 6           # this many restarts without a healthy run in between is shouted as FATAL
                            # (it stamps once per unit; one teacher region with its first engine
                            # builds is ~10 min on an A100, so the margin is 3x that)
RECYCLE_MIN_S = 300.0       # a producer lives this long before `producer_recycle_rss_gb` may end it: a
                            # baseline already over the cap must not become a respawn loop
WAIT_S = 5.0                # the trainer's sleep when nothing in the lookahead window is ready
IDLE_S = 2.0                # the producer's sleep when there is nothing to produce
ROUND_GAIN = 0.02           # "< 2 % remaining gain" is the plateau half of the round gate
GATE_REGIONS = 2            # held-out regions scored per gate attempt (see the module docstring)
GATE_SCREEN = 0.1           # the verso gate's held-out pass is skipped while the eval's fine-rung dice
                            # is more than this below `verso_gate_dice` (see `verso_gate`)
LOOKAHEAD_MAX = 64
REEST_S = 600.0             # the lookahead L is re-estimated from the logs this often
VRAM_REPORT_S = 1800.0      # the producer's `vram_report` line: at start, after each job kind's first
                            # pass, and this often

# The plan's budget table (§1 "GPU modes"), in GB on one 80 GB card: trainer 46-57 (30m6, 256^3, batch
# 2, ckpt-act 1-2), producer ~30 (a teacher and the student, never concurrently). The low end of the
# trainer's range is the one that fits beside the producer under the 4 GB headroom, so it is the one the
# table starts from; a bigger card scales both roles by the same factor.
BUDGET_GB = {"train": 46.0, "produce": 30.0}
REFERENCE_GB = 80.0
HEADROOM_GB = 4.0
RESIDENT_MIN_GB = 70.0      # one card at least this big runs every role resident
RESIDENT_SLACK = 0.05       # resident roles may get up to 5 % less than the table before the run refuses


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


def verso_held(out):
    return os.path.exists(os.path.join(str(out), VERSO_HOLD_FILE))


def verso_hold(out, on=True, why="manual"):
    """Place (`on`) or lift the manual verso hold. The gate still computes and logs its streak every
    evaluation, so the log shows when it WOULD have fired; `verso_on` stays false while held."""
    p = os.path.join(str(out), VERSO_HOLD_FILE)
    if on:
        with open(p, "w") as f:
            f.write(json.dumps({"t": time.time(), "why": str(why)}))
    elif os.path.exists(p):
        os.remove(p)
    return verso_held(out)


def apply_verso_hold(out, step, ok, why):
    """The gate's decision after the manual hold: while VERSO_HOLD exists it is logged as what it WOULD
    have been (`verso_hold`, `would_pass`, the streak values) and False is returned."""
    if not verso_held(out):
        return bool(ok)
    jlog(out, "sched", {"kind": "verso_hold", "step": int(step), "would_pass": bool(ok),
                        "gate_why": (why or {}).get("why"),
                        **{k: v for k, v in (why or {}).items() if k != "why"}})
    return False


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
        # The usable card is split between the roles in the table's proportions. The table sums to
        # exactly 80 - 4, and an "80 GB" card reports 79.3 GiB, so scaling it by total / 80 and then
        # demanding the 4 GB headroom on top refused the very card it was written for. A role may get
        # at most RESIDENT_SLACK less than its table entry; below that the run refuses.
        fit = (total - headroom) / max(sum(table.values()), 1e-9)
        if fit < 1.0 - RESIDENT_SLACK:
            raise SystemExit(budget_table(table, total, headroom))
        want = {r: v * fit for r, v in table.items()}
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


def walk_records(recs, heldout=()):
    """The region records the walk is built from: every record not inside a held-out region. The
    trainer's sampler must be handed THESE, not `setup`'s full list -- it builds its own visits and
    order from what it is given, and a list that still held the held-out regions gave the trainer a
    different walk from the producer's (the trainer then waited on regions the producer had no reason
    to make)."""
    return [r for r in recs if not _inside_any(r, heldout)]


def walk(cfg, recs, heldout=(), seed=0):
    """(visits, order): the deterministic walk both sides step along, from the same inputs."""
    from rvsm import regions as RG
    recs = walk_records(recs, heldout)
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
    # a rung 3-6 visit whose home is not a rung-2 region of the walk is dead (`Patches._dead`): the
    # trainer never draws it, so its home -- nearly all air -- is not worth a teacher pass
    fine2 = {tuple(int(v) for v in r["lo"]) for r in visits if int(r["k"]) == 2}
    for n, i in enumerate(order):
        rec = visits[i]
        k = int(rec["k"])
        lo2 = np.array(rec["lo"], np.int64) << max(k - 2, 0)
        lo = tuple(int(v) // int(cfg.region) * int(cfg.region) for v in lo2)
        if lo in seen:
            continue
        if 3 <= k <= 6 and fine2 and lo not in fine2:
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
        self.fast = {}
        self.on_device = True               # False while the weights are parked in host memory (`offload`)
        self._host = {}                     # teacher -> pinned host copies of its parameters and buffers
        names = teacher_names(cfg)          # ONLY the configured teachers are built (m7 alone: no recto)
        # the per-rung ROUTE (`config.route_spec`): both of its teachers must be in the bank, and the
        # bank then writes recto (fine) + band (base) + rw (c_A) instead of a fusion (`probs_routed`)
        self.route = CFG.route_spec(cfg)
        if self.route is not None:
            miss = [n for n in (self.route.fine, self.route.base) if n not in names]
            if miss:
                raise SystemExit(f"rvsm run: teacher_route names {miss} but the teacher set is {names}; "
                                 f"give both in `teacher_ckpts`")
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
        from rvsm import infer, teachers as T
        if row[3] is None:
            net, spec = T.load_teacher(row[0], row[2], device=self.device or "cpu")
            row[1], row[3] = spec, net
            self.fast[row[0]] = (infer.fast_teacher(net, self.device or "cpu", compile=self.cfg.compile)
                                 if self.cfg.teacher_bf16 else None)
        return row[3]

    def load(self):
        """Load every teacher now (instead of lazily in the first pass); returns the seconds it took."""
        t0 = time.time()
        self.onload()
        for row in self.items:
            self._net(row)
        return time.time() - t0

    def movable(self):
        """Can the weights be parked in host memory? The torch backend on CUDA only (a TensorRT engine
        holds its own copy, and a CPU bank has nowhere to move from)."""
        return self.backend == "torch" and str(self.device or "").startswith("cuda") and \
            bool(TEACHER_OFFLOAD)

    @staticmethod
    def _tensors(net):
        return list(net.parameters()) + list(net.buffers())

    def offload(self):
        """Park the loaded teachers' weights in PINNED host memory and give their VRAM back; returns the
        seconds it took (None: nothing to do). The modules, and the graphs `torch.compile` built for
        them, stay: each parameter keeps its identity and only its `.data` moves, and back on the card
        the compiled forward's guards (device, dtype, shape, stride) pass again -- no recompile
        (measured, laptop 5080: recto + m7, 0.91 GB: 1.4 s out the first time, 0.07 s back, 0 new
        graphs). The teachers are frozen, so the host copy made on the first offload is kept and every
        later offload is only a pointer swap."""
        if not self.on_device or not self.movable():
            return None
        import torch
        t0 = time.time()
        for row in self.items:
            if row[3] is None:
                continue
            ts = self._tensors(row[3])
            hs = self._host.get(row[0])
            if hs is None:
                hs = [torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in ts]
                for t, h in zip(ts, hs):
                    h.copy_(t.data, non_blocking=True)
                torch.cuda.synchronize()
                self._host[row[0]] = hs
            for t, h in zip(ts, hs):
                t.data = h
        self.on_device = False
        torch.cuda.empty_cache()
        return time.time() - t0

    def onload(self):
        """The weights back on the card (`offload`'s inverse); returns the seconds (None: already there)."""
        if self.on_device:
            return None
        import torch
        t0 = time.time()
        dev = torch.device(self.device)
        for row in self.items:
            if row[3] is None:
                continue
            for t, h in zip(self._tensors(row[3]), self._host[row[0]]):
                t.data = h.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        self.on_device = True
        return time.time() - t0

    def footprint(self):
        """Bytes of the loaded teachers' parameters and buffers ON THE CARD (0 before the first pass loads
        them, and while they are parked in host memory)."""
        return sum(module_bytes(row[3]) for row in self.items) if getattr(self, "on_device", True) else 0

    def read(self, ct, lo, size, pyr=None, skip=()):
        """Every teacher's CT for a region, read ahead of its pass (`infer.teacher_read`): the producer's
        reader thread calls this for the NEXT region while the GPU runs the current one. A teacher named
        in `skip` (a routed reteach that reuses the base teacher's committed store) reads nothing (None)."""
        from rvsm import infer, ladder
        pyr = ladder.rungs(ct) if pyr is None else pyr
        m = int(getattr(self.cfg, "infer_margin", 0))
        return [None if row[0] in skip else infer.teacher_read(ct, lo, size, row[1], pyr=pyr, margin=m)
                for row in self.items]

    def _one_u8(self, i, ct, lo, size, roi=None):
        """Teacher `i` of the bank over one region, as a uint8 probability tensor on the device (its
        float plane is quantised at once and freed: the routed pass never holds two)."""
        from rvsm import infer
        row = self.items[i]
        net = self._net(row)
        p = infer.teacher_region(ct, lo, size, row[1], row[2], device=self.device, backend=self.backend,
                                 net=net, as_tensor=True, fast=self.fast.get(row[0]), roi=roi,
                                 margin=int(getattr(self.cfg, "infer_margin", 0)),
                                 engine_dir=os.path.join(self.out, "ckpt", "trt"))
        u = infer.u8_t(p)
        del p
        return u

    def probs_routed(self, ct, lo, size, rois=None, band=None, band_from=None):
        """(u8 fine-teacher probability, u8 c_A coverage, u8 base-teacher band, attrs) for one region under
        the ROUTE (`config.route_spec`), every block a uint8 tensor on the device.

        recto = the fine (2.4 um recto) teacher's probability, as it is; band = the base (m7) teacher's
        probability at rung 2 -- `band` (a host uint8 block, the base teacher's committed store read by
        the producer: no base rerun) when given, else a base-teacher pass; rw = c_A
        (`infer.route_coverage_u8`: 255 where the fine teacher is trusted). The passes run one after
        the other on the one card, each freed to uint8 before the next."""
        import torch

        from rvsm import infer
        self.onload()
        r = self.route
        names = [row[0] for row in self.items]
        fi, bi = names.index(r.fine), names.index(r.base)
        P = self._one_u8(fi, ct, lo, size, roi=(rois[fi] if rois is not None else None))
        if band is not None:
            B = torch.as_tensor(np.ascontiguousarray(band, np.uint8)).to(P.device)
            if tuple(B.shape) != tuple(P.shape):     # a store is padded to 128s: the region's own box
                Bf = torch.zeros_like(P)
                s = tuple(slice(0, min(int(a), int(b))) for a, b in zip(B.shape, P.shape))
                Bf[s] = B[s]
                B = Bf
            src = "store:" + str(band_from or "")
        else:
            B = self._one_u8(bi, ct, lo, size, roi=(rois[bi] if rois is not None else None))
            src = "teacher"
        W = infer.route_coverage_u8(P)
        return P, W, B, {"producer": f"teacher:{r.fine}+band:{r.base}", "teachers": [r.fine, r.base],
                         "route": r.sig, "radial_sign": 1, "rw": "coverage", "band_source": src,
                         "coverage": {"hi": infer.COV_HI, "lo": infer.COV_LO, "face": infer.COV_FACE,
                                      "reach": infer.COV_REACH, "pool": infer.COV_POOL},
                         "ckpt": {x[0]: x[2] for x in self.items}, "backend": self.backend,
                         "margin": int(getattr(self.cfg, "infer_margin", 0))}

    def probs_u8(self, ct, lo, size, rois=None):
        """(u8 fused probability, u8 agreement weight, attrs) for one region, over every loaded teacher,
        as uint8 TENSORS on the device: the teachers' float planes never leave the card (the host-side
        fuse of two 1024^3 float32 volumes was ~40 s a region)."""
        from rvsm import infer
        self.onload()                       # never a pass on parked weights
        ps, names = [], []
        for i, row in enumerate(self.items):
            net = self._net(row)
            ps.append(infer.teacher_region(ct, lo, size, row[1], row[2], device=self.device,
                                           backend=self.backend, net=net, as_tensor=True,
                                           fast=self.fast.get(row[0]),
                                           roi=(rois[i] if rois is not None else None),
                                           margin=int(getattr(self.cfg, "infer_margin", 0)),
                                           engine_dir=os.path.join(self.out, "ckpt", "trt")))
            names.append(row[0])
        if len(ps) >= 2:
            P, W = infer.fuse_agreement_u8(ps[0], ps[1])
        else:
            # ONE teacher: its probability as it is, and rw = 1 everywhere. The loader's weight is
            # already `inside x (CT > 0) x rw`, so air (m7's coarse air mask zeroed P there, the fine
            # CT masks it again) needs nothing from rw; the agreement down-weighting has no second
            # opinion to disagree with, and a self-confidence weight would change the loss (the bce of
            # an undecided voxel would lose its pull towards 0.5) rather than keep it
            P = infer.u8_t(ps[0])
            W = torch_full_like_u8(P, 255)
        del ps
        return P, W, {"producer": "teacher:" + ",".join(names), "teachers": list(names), "radial_sign": 1,
                      "rw": "agreement" if len(names) >= 2 else "ones",
                      "ckpt": {r[0]: r[2] for r in self.items}, "backend": self.backend,
                      "margin": int(getattr(self.cfg, "infer_margin", 0))}

    def probs(self, ct, lo, size):
        """(fused probability, agreement weight, attrs) as host float32 arrays (the u8 path, decoded)."""
        P, W, attrs = self.probs_u8(ct, lo, size)
        return (P.cpu().numpy().astype(np.float32) / 255.0, W.cpu().numpy().astype(np.float32) / 255.0,
                attrs)


def torch_full_like_u8(t, v):
    import torch
    return torch.full_like(t, int(v), dtype=torch.uint8)


STUDENT_COMPILE_MODE = "default"   # the producer's student: torch.compile without autotuning
# park the teacher bank's weights in host memory while no teacher pass is left in the window
# (`TeacherBank.offload`); RVSM_TEACHER_OFFLOAD=0 keeps them on the card
TEACHER_OFFLOAD = os.environ.get("RVSM_TEACHER_OFFLOAD", "1") not in ("0", "false", "no")


class StudentSlot:
    """The student the producer runs, reloaded when its file changes and not otherwise.

    Round 0 (the verso passes) tracks the LIVE `ckpt/student.pt`. Round r >= 1 (the self passes that
    write round r's targets) uses the FROZEN round teacher `state["teacher"]`
    (`ckpt/teacher_round_<r>.pt`, the EMA snapshotted when round r opened): a round's targets must come
    from one fixed network, not from the student that is being trained on them."""

    def __init__(self, out, device=None, compile=True, mode=None):
        self.out = str(out)
        self.path = os.path.join(self.out, "ckpt", "student.pt")
        self.device, self.compile = device, bool(compile)
        # the producer's compile mode: "default" (no autotuning) unless RVSM_STUDENT_COMPILE_MODE says
        # otherwise. max-autotune benchmarks every candidate kernel on the card, and behind Thunder's
        # GPU proxy (an RPC per benchmark) the first student pass sat in that for 25+ minutes
        self.mode = str(mode or os.environ.get("RVSM_STUDENT_COMPILE_MODE", STUDENT_COMPILE_MODE))
        self.st, self.mtime, self.loaded, self.sha = None, None, None, None

    def source(self, round_=0, teacher=None):
        """The checkpoint this round's student passes must use (None: not there yet)."""
        if int(round_) >= 1:
            return teacher if teacher and os.path.exists(teacher) else None
        return self.path if os.path.exists(self.path) else None

    def get(self, round_=0, teacher=None, path=None):
        """The student for this round -- or, with `path`, exactly that checkpoint file (the FROZEN
        checkpoint a verso regeneration runs from)."""
        from rvsm import infer
        p = (path if path and os.path.exists(path) else None) if path is not None else \
            self.source(round_, teacher)
        if p is None:
            return None
        m = os.path.getmtime(p)
        if self.st is None or p != self.loaded or m != self.mtime:
            try:
                import hashlib
                with open(p, "rb") as f:          # read ONCE: the digest is of the bytes loaded
                    buf = f.read()
                # a new checkpoint of the same network goes INTO the loaded (compiled) module: one
                # compile per slot per process, not one per publish (`infer.Student.reload`)
                reload = getattr(self.st, "reload", None)
                if reload is not None and reload(p, data=buf):
                    st = self.st
                else:
                    st = infer.student_fn(p, device=self.device, compile=self.compile, data=buf,
                                          **({"mode": self.mode} if self.compile else {}))
                self.st, self.mtime, self.loaded = st, m, p
                self.sha = hashlib.sha256(buf).hexdigest()
                del buf
            except Exception as e:  # noqa: BLE001  -- a checkpoint caught mid-rename comes back next loop
                print(f"[produce] student reload: {e!r}", flush=True)
                return self.st if self.loaded == p else None
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


def student_rows_t(planes, layout, heads):
    """`student_rows` for DEVICE float planes (`student_region(..., as_tensor=True)`): the same codes,
    computed on the card, returned as host uint8 arrays -- one byte per voxel crosses the bus."""
    from rvsm import export as EX, infer, targets as TG
    first = str(layout.channels[0])
    if heads == "verso":
        return [("verso", infer.u8_t(planes[first]).cpu().numpy(), 8, "prob_u8")]
    rec = planes["recto"]
    valid = rec > 0
    return [("recto", infer.u8_t(rec).cpu().numpy(), 8, "prob_u8"),
            ("verso", infer.u8_t(planes["verso"]).cpu().numpy(), 8, "prob_u8"),
            ("midline", EX.enc_t(planes["midline"], valid, -EX.TRACER_CAP, EX.TRACER_CAP,
                                 EX.TRACER_UNIT, EX.TRACER_OFF).cpu().numpy(), 0,
             "signed_u8_off128_q0.25"),
            ("thickness", EX.enc_t(planes["thickness"], valid, TG.UNIT, 255 * TG.UNIT,
                                   TG.UNIT).cpu().numpy(), 0, "unsigned_u8_q0.25"),
            ("conf", infer.u8_t(planes["conf"]).cpu().numpy(), 0, "conf_u8")]


def write_rows(out, lo, rows, cfg, round_, attrs, gen=0):
    """Write a unit's stores; `gen` > 0 writes a NEW generation beside the finished one (a regenerated
    verso), never over it."""
    from rvsm import infer, stores
    for ch, block, q, enc in rows:
        p = stores.gen_path(stores.store_path(out, ch, lo, round_), gen)
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


def feed_coarse_once(out, lo, round_, block, shape2, pooled=None, gen=0):
    """Fold a finished recto block into the coarse rungs, once per region, round and GENERATION (a
    marker file, so a restart does not redo it and two producers could not double-count it). A
    regenerated recto (`reteach`, generation g > 0) overwrites the region's footprint of the coarse
    arrays once: the coarse rungs have no generations, they follow the newest teacher pass at once
    (their footprint is the same, only the values move)."""
    from rvsm import regions as RG
    m = _fed_marker(out, lo, round_)
    if os.path.exists(m) and int((_read_json(m) or {}).get("gen", 0)) >= int(gen):
        return []
    ks = RG.feed_coarse(out, "recto", lo, block, round_=round_, shape2=shape2, pooled=pooled)
    os.makedirs(os.path.dirname(m), exist_ok=True)
    with open(m + ".tmp", "w") as f:
        f.write(json.dumps({"rungs": ks, "t": time.time(), "gen": int(gen)}))
    os.replace(m + ".tmp", m)
    return ks


def commit_sources(out, lo, round_, rungs=(2, 3, 4)):
    """Commit the region's newest finished sources for every reader, in two independent parts:

    - the RECTO: the newest finished recto generation, as soon as its rw beside it is finished too
      (the pair is written recto then rw, each atomically). It does NOT wait for the fields: they are
      the pair / distance losses' input only, and fields built from the previous recto are the same
      sheet a voxel or two off -- so a reteach switches the loader, the eval grid and the gate
      reference at once (paris4 2026-09-25: gating it on the fields queue meant > 1 day of old targets).
    - the FIELDS with their VERSO: the fields' generation (`targets.field_gen`) and the newest verso,
      only once every field store built from exactly the newest verso and recto is finished
      (`targets.fields_current`). A verso still never moves without its fields (pass-4 P4-04).

    Every generation named is a finished store; a half-written one is never committed. Logs a
    `recto_commit` and / or `fields_commit` line. Returns the committed bundle, or None when nothing
    moved. Idempotent: the producer calls it after every reteach and fields job, and for any region
    of the regeneration backlog with nothing left to run (a commit a restart cut off)."""
    from rvsm import stores, targets as TG
    cur = stores.bundle_state(out, lo, round_)
    want = dict(cur)
    gr = TG.source_recto(out, lo, round_)[0]
    if gr >= 0 and (int(round_) != 0 or gr == 0 or
                    stores.is_done(stores.gen_path(stores.store_path(out, "rw", lo, round_), gr))):
        want["recto"] = gr
    if stores.store_gen(out, "verso", lo, round_) >= 0 and TG.fields_current(out, lo, round_, rungs):
        want["gen"] = TG.field_gen(out, lo, round_)
        want["verso"] = TG.source_verso(out, lo, round_)[0]
    if want == cur:
        return None
    stores.commit_bundle(out, lo, round_, want["gen"], verso=want["verso"], recto=want["recto"],
                         t=time.time())
    if want["recto"] != cur["recto"]:
        jlog(out, "produce", {"kind": "recto_commit", "region": [int(v) for v in lo], "round": int(round_),
                              "gen": want["recto"], "was": cur["recto"]}, echo=False)
    if (want["gen"], want["verso"]) != (cur["gen"], cur["verso"]):
        jlog(out, "produce", {"kind": "fields_commit", "region": [int(v) for v in lo], "round": int(round_),
                              "gen": want["gen"], "verso": want["verso"], "was": cur["gen"]}, echo=False)
    return want


def lookahead(cfg, out, k_active):
    """L = ceil(T_produce / T_train) * K_active + `lookahead_extra`, from the logs (plan §1).

    T_produce is the median seconds a region has taken lately (`logs/produce.jsonl`), T_train the
    seconds the trainer spends on one region of the walk (it publishes that in state.json). Without
    either, L is its floor -- which is what a cold start wants anyway."""
    extra = int(cfg.lookahead_extra)
    # T_produce is the GPU thread's time per unit (its read wait + its passes): the writer and the
    # fields pool overlap it, so their wall time says nothing about how fast the window advances.
    ps = [float(r.get("gpu_s", r["s"])) + float(r.get("read_wait_s", 0.0))
          for r in tail_jsonl(os.path.join(str(out), "logs", "produce.jsonl"), 40)
          if r.get("kind") in ("teacher", "verso", "self") and isinstance(r.get("s"), (int, float))][-10:]
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
    # glibc's malloc, before the big allocations: the per-region host-RSS creep is arenas
    # (docs/recipe.md §7 "producer RSS creep")
    malloc_set = set_mallopt(thresholds=bool(getattr(cfg, "producer_mallopt", True)),
                             arena_max=int(getattr(cfg, "producer_arena_max", 0) or 0))
    ncomp = limit_compile_threads()
    if role_gpu is not None and device is None:
        device = f"cuda:{int(role_gpu)}"
    if mem_frac:
        set_memory_fraction(mem_frac, 0)

    import threading
    t_start = time.time()
    hb = os.path.join(out, "workers", "produce.json")
    hb_rec, hb_lock, hb_stop = {}, threading.Lock(), threading.Event()

    def stamp(rec):
        """The loop's heartbeat: what it is doing, `last_ts` (alive) and `progress_ts` (the loop itself
        got here). One lock for this and the ticker: both write the same file through the same tmp."""
        with hb_lock:
            now = time.time()
            hb_rec.clear()
            hb_rec.update(rec, pid=os.getpid(), pgid=os.getpgid(0), last_ts=now, progress_ts=now)
            _write_json(hb, dict(hb_rec))

    def tick():
        with hb_lock:
            if hb_rec:
                hb_rec["last_ts"] = time.time()
                _write_json(hb, dict(hb_rec))

    stamp({"phase": "start"})
    threading.Thread(target=hb_ticker, args=(tick, hb_stop), name="rvsm-hb", daemon=True).start()
    import torch  # noqa: F401  -- loaded now, so that the USR1 handler below is the last one set
    register_stack_dumps()
    stacks = stack_dump_file(out, "produce")    # the stall watch writes here (and to stderr)
    unit_now = {}                               # the GPU unit in flight: job, region, t0
    unit_now["locks"] = []                      # the traced locks whose holders a stall dump names
    threading.Thread(target=unit_watchdog, args=(out, unit_now, stacks, hb_stop),
                     name="rvsm-unit-watch", daemon=True).start()

    cache = stream.ShardCache(cfg.ct, out, budget_gb=cfg.cache_gb, seed=cfg.ct_seed or None,
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
    teachers = teacher_set(cfg)         # a round-0 recto from another set (or route) is regenerated (`reteach`)
    held_los = [tuple(int(v) for v in h["lo"]) for h in held]
    rr = {"todo": None, "t": 0.0, "n": None, "t_log": 0.0}   # the recto regeneration's work list
    meter = ReteachMeter()              # the reteach share of the recent GPU time (`cfg.reteach_share`)
    rshare = float(getattr(cfg, "reteach_share", 0.0) or 0.0)
    # the distance fields: on the producer's own GPU when it has one (`targets.block_fields_torch`,
    # ~2.5 GB of extra VRAM at peak, inside the memory fraction), else a CPU pool of
    # every core at low priority
    fdev = str(device) if device is not None and str(device).startswith("cuda") else None
    jobs = 1 if fdev is not None else max(int(os.cpu_count() or 1), 1)
    vram_cap = _vram_cap(device, mem_frac)      # bytes this process may reserve on its card (None: CPU)
    fbatch = int(getattr(cfg, "fields_batch", 0) or 0) or \
        (3 if vram_cap is not None and vram_cap >= 30 * (1 << 30) else 1)
    # the GPU fields never run beside a network pass: a pass holds this for its whole forward, the
    # fields hold it for a region's device work, and both empty the allocator's cache after
    gpu_lock = TracedLock("gpu", log=lambda rec: jlog(out, "produce", rec, echo=False))
    # ... and by priority: the fields start only when no pass is pending (verso-only: after
    # FIELDS_DEFER_S) and give the card back between batches to a blocking pass (`GpuGate`)
    gate = GpuGate(gpu_lock, log=lambda rec: jlog(out, "produce", rec, echo=False))
    jlog(out, "produce", {"kind": "start", "pid": os.getpid(), "device": str(device),
                          "regions": len(route), "heldout": len(held), "pinned": pinned,
                          "backend": str(backend), "field_rungs": list(frungs), "jobs": jobs,
                          "fields_device": fdev or "cpu", "compile_threads": ncomp,
                          "fields_batch": fbatch, "mallopt": malloc_set, "rss_gb": round(own_rss_gb(), 2),
                          "vram_cap_gb": None if vram_cap is None else round(vram_cap / (1 << 30), 2)})

    bank, slot = None, StudentSlot(out, device=device, compile=cfg.compile)
    peaks = {}                                  # job kind -> the last pass's peak allocated bytes
    vram_report(out, bank, slot, peaks, vram_cap, "start")
    t_vrep = time.time()
    # the frozen regeneration student is the SAME slot: its checkpoint is loaded into the one compiled
    # module in place (`infer.Student.reload`) and the live one back after it. A second compiled
    # student beside the live one and the teacher bank left no room for a pass's transient memory
    # inside the producer's VRAM fraction (the 16:48 stall)
    rslot = slot
    keys = {}
    backlog_keys = set()                        # regions fetched for the regeneration backlog

    def release_backlog():
        """A backlog region sits outside the walk (no position), so `_release_passed` never gives its
        CT back: it is released here once it needs nothing more."""
        st_ = read_state(out)
        rg = st_.get("verso_regen")
        c0 = RG.Catalog(out, 0, ttl=1.0)
        for lo in list(backlog_keys):
            with lock:
                if lo in busy:
                    continue
            # the backlog's own passes only: a verso the window has not asked for yet is not one
            if _next_job(c0, lo, 0, bool(st_.get("verso_on")) and c0.done("verso", lo), out,
                         rungs=frungs, regen=rg, teachers=teachers) in (None, "fields"):   # the fields read no CT
                backlog_keys.discard(lo)
                if lo in keys:
                    cache.release(keys.pop(lo))
    # a restart: the inventory charged every shard an earlier process left, and nothing is held yet --
    # the regions the trainer's workers are reading are re-held (and re-fetched if they lost shards)
    # BEFORE the first eviction, which then honours the budget at once
    held0 = cursor_leases(out)
    for lo in held0:
        try:
            keys[lo] = cache.fetch_region(np.array(lo, np.int64), ctx=cfg.ctx, patch=cfg.patch,
                                          region=cfg.region, evict=False)
        except stream.FetchFailed:              # the lease keeper retries it
            pass
    cache.lease(cache.region_key(lo) for lo in held0)
    n_ev = cache.evict()
    jlog(out, "produce", {"kind": "cache_start", "leased": len(held0), "evicted": n_ev,
                          **{k: (round(v, 2) if isinstance(v, float) else v)
                             for k, v in cache.stats().items()}}, echo=False)
    L = k_active + int(cfg.lookahead_extra)
    t_reest = 0.0

    # the RECYCLE (`producer_recycle_fields` / `producer_recycle_rss_gb`): a clean exit like STOP's --
    # the unit in flight finishes, the writer and the fields drain, the cache closes -- then a `recycle`
    # line and exit code 0 with `recycle` in produce.json, which the supervisor respawns at once and
    # never counts as a failure (`ProducerWatch`). The host RSS the process leaked goes with it
    recycle = {"why": None, "fields": 0}
    recycle_n = int(getattr(cfg, "producer_recycle_fields", 0) or 0)
    # the RSS cap is this process's own: a producer THREAD (cpu mode) shares the supervisor's RSS, and
    # re-starting it would free nothing
    recycle_gb = float(getattr(cfg, "producer_recycle_rss_gb", 0.0) or 0.0) \
        if threading.current_thread() is threading.main_thread() else 0.0

    def stopping():
        return recycle["why"] is not None or stop_requested(out) or \
            (stop is not None and stop.is_set()) or \
            (max_s is not None and time.time() - t_start > float(max_s))

    # THE OVERLAP. The GPU thread (this one) only ever runs network passes. Around it:
    #   reader   fetches the NEXT unit's shards and decodes its CT while the current unit infers
    #   writer   volcomp-encodes and writes the stores, folds the coarse rungs, logs the unit
    #   fields   the distance fields (`targets.region_fields`): on this GPU (`fdev`, a side stream) or,
    #            without one, a process pool of `jobs`
    # A region with a unit in the writer or the fields queue is BUSY: its stores are not on disk yet,
    # so the disk-derived state machine would hand out the same unit again.
    import concurrent.futures as cf
    reader = cf.ThreadPoolExecutor(1, thread_name_prefix="rvsm-read")
    writer = cf.ThreadPoolExecutor(1, thread_name_prefix="rvsm-write")
    fielder = cf.ThreadPoolExecutor(1, thread_name_prefix="rvsm-fields")
    wslots = threading.BoundedSemaphore(2)     # at most two finished units waiting for the writer
    busy, pend, lock = set(), [], threading.Lock()
    fpool = TG.field_pool(jobs, owner=os.getpid()) if fdev is None and jobs > 1 else None
    pre = {}                                    # (lo, job) -> future of the reader's inputs

    # the shard cache is not thread-safe: one caller at a time. Traced: a wait over a minute logs who
    # holds it (and since when), and the stall watch prints the holder with every stack dump
    clock = TracedLock("clock", log=lambda rec: jlog(out, "produce", rec, echo=False))
    unit_now["locks"] += [clock, gpu_lock]
    readers = {"ex": reader}                    # replaced when a read times out (the stuck thread is left)
    starved_t = {}                              # worker -> when its last `starved` line was logged
    skip_until = {}                             # region -> time before which no GPU unit is started on it

    def need(lo, job):
        """What the reader prepares for a unit: its shards always, and a teacher unit's CT as well."""
        with clock:
            have = lo in keys
        if not have:                            # the download itself runs WITHOUT the cache lock
            cache.fetch_region_outside(np.array(lo, np.int64), clock, ctx=cfg.ctx, patch=cfg.patch,
                                       region=cfg.region, on_booked=lambda k: keys.__setitem__(lo, k))
        if job in TEACHER_JOBS and bank is not None:
            if bank.route is None:
                return bank.read(ct_local, lo, region_size(pyr, lo, cfg.region), pyr=pyr)
            # a ROUTED pass: a reteach reuses the base teacher's committed probability (no m7 rerun);
            # the store is decoded here, on the reader thread, beside the GPU's current unit
            bp = band_reuse(out, lo, teachers) if job == "reteach" else None
            band = np.asarray(stores.open_store(bp)[:], np.uint8) if bp else None
            return {"rois": bank.read(ct_local, lo, region_size(pyr, lo, cfg.region), pyr=pyr,
                                      skip=(bank.route.base,) if bp else ()),
                    "band": band, "band_from": bp}
        return None

    def prefetch(lo, job):
        if (lo, job) not in pre:
            pre[(lo, job)] = readers["ex"].submit(need, lo, job)

    lease_state = {}                            # worker -> (lease_id, ready?, last attempt)

    def keep_leases():
        """The producer's side of a lease (`acknowledge_leases`), on its own thread: a long GPU unit
        must never delay a worker's acknowledgement."""
        while not hb_stop.is_set() and not stopping():
            try:
                acknowledge_leases(out, cache, keys, clock, lease_state,
                                   {"ctx": cfg.ctx, "patch": cfg.patch, "region": cfg.region})
            except Exception as e:  # noqa: BLE001  -- a bad cursor file must not end the keeper
                jlog(out, "produce", {"kind": "lease_keeper_error", "err": repr(e)}, echo=False)
            hb_stop.wait(LEASE_POLL_S)

    if cache.remote:
        threading.Thread(target=keep_leases, name="rvsm-lease", daemon=True).start()

    def settle():
        """Raise a background failure here, on the producer's own thread, and forget finished work."""
        with lock:
            done = [f for f in pend if f.done()]
            for f in done:
                pend.remove(f)
        for f in done:
            f.result()

    def finish(kind, lo, round_, t0, rows, attrs, pooled=None, extra=None):
        chained = False
        try:
            write_rows(out, lo, rows, cfg, round_, attrs, gen=int(attrs.get("gen", 0)))
            if pooled is not None or (kind == "self" and rows):
                feed_coarse_once(out, lo, round_, rows[0][1], shape2, pooled=pooled,
                                 gen=int(attrs.get("gen", 0)))
            jlog(out, "produce", {"kind": kind, "region": list(lo), "round": round_,
                                  "s": round(time.time() - t0, 2), **(extra or {})})
            # a reteach's recto + rw are committed for every reader NOW (`commit_sources`); the fields
            # rebuilt from them follow on their own. A window / held-out region's rebuild is chained
            # like a verso's; a backlog region's is fed by `_recto_backlog`, a few at a time, so the
            # backlog never queues hundreds of ~90 s fields jobs ahead of the window's own
            if kind == "reteach":
                commit_sources(out, lo, round_, frungs)
            reteach_fields = kind == "reteach" and lo not in backlog_keys and \
                stores.store_gen(out, "verso", lo, round_) >= 0
            # the unit that makes a region's fields possible hands it straight to the fields pool
            # (still busy): waiting for the GPU loop's next pass over the window left the fields
            # of every verso region undone until a whole window of ~1 min verso passes had run
            if (kind in ("verso", "self") or reteach_fields) and not stopping():
                with lock:
                    pend.append(fielder.submit(fields, lo, round_, time.time(), cursor_now()))
                chained = True
        finally:
            if not chained:
                with lock:
                    busy.discard(lo)
            wslots.release()

    def cursor_now():
        return max(read_cursor(out), 0)

    def fields(lo, round_, t0, cursor):
        try:
            rep = TG.region_fields(out, lo, ax, round_=round_, rungs=frungs, jobs=jobs, pool=fpool,
                             device=fdev, batch=fbatch,
                             gpu_lock=gate.fields_hold(t0, tag={"region": list(lo), "round": round_})
                             if fdev else None)
            # a new verso or recto AND all of the fields built from it are finished: only now do
            # readers move to them (`commit_sources`)
            commit_sources(out, lo, round_, frungs)
            g = TG.field_gen(out, lo, round_)
            # the region's host garbage back to the OS (a no-op without glibc); before / after logged
            rss0 = own_rss_gb()
            trimmed = malloc_trim()
            rss1 = own_rss_gb() if trimmed else rss0
            jlog(out, "produce", {"kind": "fields", "region": list(lo), "round": round_,
                                  "s": round(time.time() - t0, 2), "cursor": cursor, "gen": g,
                                  "device": fdev or "cpu",
                                  "skipped": (rep or {}).get("skipped_blocks"),
                                  "graphs": (rep or {}).get("graphs"),
                                  "rss_gb": round(rss0, 3), "rss_trim_gb": round(rss1, 3)})
            with lock:
                recycle["fields"] += 1
                if recycle_n and recycle["fields"] >= recycle_n and recycle["why"] is None:
                    recycle["why"] = f"fields {recycle['fields']} >= producer_recycle_fields {recycle_n}"
        finally:
            with lock:
                busy.discard(lo)

    try:
        while not stopping():
            settle()
            if recycle_gb and recycle["why"] is None and time.time() - t_start > RECYCLE_MIN_S:
                r_ = own_rss_gb()                           # between units: the RSS cap
                if r_ > recycle_gb:
                    recycle["why"] = f"rss {r_:.2f} GB > producer_recycle_rss_gb {recycle_gb:g}"
                    break
            if time.time() - t_vrep >= VRAM_REPORT_S:
                vram_report(out, bank, slot, peaks, vram_cap, "periodic")
                t_vrep = time.time()
            st = read_state(out)
            round_ = int(st.get("round", 0))
            verso_on = bool(st.get("verso_on", False))
            # the LIVE cursor (the sampler workers publish it after every visit), not state.json's
            # copy: that one is only rewritten at an evaluation, every `eval_every` steps -- dozens of
            # visits -- and a window that lags that far behind the trainer starves it
            cursor = max(read_cursor(out, round_), 0)
            head = max(read_cursor_head(out, round_), cursor)
            if time.time() - t_reest > REEST_S:
                L, t_reest = lookahead(cfg, out, k_active), time.time()
            recs = cursor_records(out, round_)
            leased = cursor_leases(out, round_)
            # the regions the workers have LEASED -- their declared need, in lease order -- come first,
            # wherever they lie; the window past the head is the speculative part (`_working_set`)
            lorder = lease_order(recs)
            lset = set(lorder)
            stamp({"phase": f"round{round_}", "last_ts": time.time(),
                             "L": L, "cursor": cursor, "head": head})

            # one-card timeshare: no PHASE file means nobody is taking turns
            if os.path.exists(os.path.join(out, PHASE_FILE)) and read_phase(out) != "produce":
                time.sleep(IDLE_S)
                continue
            if producer_paused(out):        # the supervisor's host-RAM guard: no new unit
                stamp({"phase": "paused_ram", "last_ts": time.time(),
                                 "L": L, "cursor": cursor})
                time.sleep(IDLE_S)
                continue
            if free_gb(out) < float(cfg.reserve_gb):
                with clock:
                    ct_disk = cache.disk()
                jlog(out, "produce", {"kind": "backpressure", "free_gb": round(free_gb(out), 1),
                                      "reserve_gb": float(cfg.reserve_gb), "ct": ct_disk})
                time.sleep(IDLE_S * 5)
                continue

            cat = RG.Catalog(out, round_, ttl=1.0)
            units = []
            for lo in _working_set(lorder, _window(route, pos, cursor, L, held, head=head)):
                with lock:
                    if lo in busy:
                        continue
                if region_size(pyr, lo, cfg.region) is None:
                    continue
                job = _next_job(cat, lo, round_, verso_on, out, rungs=frungs,
                                regen=st.get("verso_regen"), teachers=teachers)
                if job is not None and job != "fields" and skip_until.get(lo, 0.0) > time.time():
                    continue                    # its read timed out within the hour: not again yet
                if job is not None:
                    units.append((lo, job))
            gpu_units = _gpu_order([u for u in units if u[1] != "fields"], leased=lorder, held=held_los)
            if cache.remote and cache.cache_bytes > cache.budget:
                # over the CT budget: no speculative region is fetched; the leased ones (and the regions
                # already held) still are
                with clock:
                    have = set(keys)
                gpu_units = [u for u in gpu_units if u[0] in lset or u[0] in have]
            regen = st.get("verso_regen")
            if round_ == 0 and regen and not st.get("verso_regen_done") and not gpu_units:
                # the regeneration BACKLOG, worked through when the lookahead window has nothing for
                # the GPU (never ahead of the trainer's own regions): one region per pass
                rem = regen_remaining(out)
                if rem == []:
                    write_state(out, verso_regen_done=True)
                    jlog(out, "produce", {"kind": "verso_regen_done", "regions": regen.get("backlog")})
                for lo in (rem or []):
                    with lock:
                        if lo in busy:
                            continue
                    job = _next_job(cat, lo, round_, verso_on, out, rungs=frungs, regen=regen,
                                    teachers=teachers)
                    if job == "fields":
                        with lock:
                            busy.add(lo)
                            pend.append(fielder.submit(fields, lo, round_, time.time(), cursor))
                        backlog_keys.add(lo)
                    elif job is not None:
                        gpu_units.append((lo, job))
                        backlog_keys.add(lo)
                        break
            if round_ == 0:
                # the RECTO regeneration (a new teacher set or route, `teacher_set`): every produced
                # region whose committed recto another set made; held-out regions first, then walk order.
                # Its work list is rescanned and logged on every pass (`recto_regen`), and it runs when
                # the window has nothing for the GPU -- or, with `reteach_share` > 0, beside the verso
                # passes while its share of the recent GPU time is under the target (`_share_backlog`)
                _recto_rescan(out, rr, route, held, teachers, meter=meter, share=rshare)

                def _rstep(units_):
                    return _recto_backlog(out, rr, route, held, teachers, cat, verso_on, frungs, busy,
                                          lock, skip_until,
                                          lambda lo: region_size(pyr, lo, cfg.region) is not None,
                                          units_, backlog_keys,
                                          lambda lo: pend.append(fielder.submit(fields, lo, round_,
                                                                                time.time(), cursor)))
                if not gpu_units:
                    _rstep(gpu_units)
                else:
                    _share_backlog(out, rr, meter, rshare, gpu_units, lorder, _rstep)
            gate.set_pending(j for _, j in gpu_units)   # the fields make way for these
            _log_starved(out, recs, starved_t, {u[0] for u in units} | {u[0] for u in gpu_units},
                         lambda lo: _next_job(cat, lo, round_, verso_on, out, rungs=frungs,
                                              regen=st.get("verso_regen"), teachers=teachers),
                         lambda lo: {"busy": lo in busy, "skipped": skip_until.get(lo, 0.0) > time.time(),
                                     "no_size": region_size(pyr, lo, cfg.region) is None})
            did = False
            for lo, job in units:               # the CPU units go straight to their own pool
                if job == "fields" and not stopping():
                    with lock:
                        busy.add(lo)
                    with lock:
                        pend.append(fielder.submit(fields, lo, round_, time.time(), cursor))
                    did = True
            if gpu_units:
                prefetch(*gpu_units[0])
            for i, (lo, job) in enumerate(gpu_units):
                if stopping():
                    break
                size = region_size(pyr, lo, cfg.region)
                if job in TEACHER_JOBS and bank is None:
                    bank = TeacherBank(cfg, out, device=device, backend=backend)
                    pre.pop((lo, job), None)        # read before the bank existed: no CT in it
                    gate.pass_acquire(job)
                    try:
                        jlog(out, "produce", {"kind": "bank_build", "s": round(bank.load(), 2),
                                              "gb": round(bank.footprint() / (1 << 30), 3)})
                    finally:
                        gate.release()
                _rg = st.get("verso_regen") or {}
                _frozen = _rg.get("ckpt") if (job == "verso" and round_ == 0 and cat.done("verso", lo)) else None
                if job in ("verso", "self") and (rslot.get(path=_frozen) if _frozen else
                                                 slot.get(round_, st.get("teacher"))) is None:
                    break
                t0 = time.time()
                stamp({"phase": f"round{round_}", "job": job,
                                 "region": list(lo), "last_ts": t0, "L": L, "cursor": cursor})
                prefetch(lo, job)
                unit_now.update(job=job, region=list(lo), t0=t0, dumped=0, phase="read")
                _stall_timer(stacks)                # a C-level timer too: it needs no GIL to dump
                jlog(out, "produce", {"kind": "unit_start", "job": job, "region": list(lo),
                                      "round": round_, "phase": "read"}, echo=False)
                import concurrent.futures as _cf
                try:
                    got = pre.pop((lo, job)).result(timeout=READ_TIMEOUT_S)
                except _cf.TimeoutError:
                    # the reader never came back (a shard fetch or the cache lock that hung): give up on
                    # this unit, put the region aside for an hour, and read on a FRESH thread -- the
                    # stuck one is abandoned, and every read queued behind it is resubmitted
                    _unit_done(unit_now)
                    skip_until[lo] = time.time() + READ_SKIP_S
                    jlog(out, "produce", {"kind": "reader_timeout", "job": job, "region": list(lo),
                                          "s": round(time.time() - t0, 1),
                                          "clock_holder": clock.holder()})
                    readers["ex"].shutdown(wait=False, cancel_futures=True)
                    readers["ex"] = cf.ThreadPoolExecutor(1, thread_name_prefix="rvsm-read")
                    pre.clear()
                    break
                except stream.FetchFailed as e:
                    # a shard did not arrive: nothing was booked and no store is written; the region
                    # stays unfinished, so a later pass over the window retries it
                    _unit_done(unit_now)
                    jlog(out, "produce", {"kind": "fetch_failed", "region": list(lo), "job": job,
                                          "shards": len(e.paths),
                                          "failed_units": int(getattr(cache, "failed_units", 0))})
                    gate.set_pending(j for _, j in gpu_units[i + 1:])
                    continue
                t_in = time.time() - t0
                if i + 1 < len(gpu_units):
                    prefetch(*gpu_units[i + 1])    # the next unit's read overlaps this unit's pass
                wslots.acquire()                    # backpressure: never more than two units unwritten
                with lock:
                    busy.add(lo)
                gate.pass_acquire(job)              # no GPU fields beside the pass (and none cached)
                try:
                    if bank is not None:
                        _bank_park(out, bank, job, gpu_units[i:])
                    _vram_check(out, job, lo, vram_cap)
                    _cuda_peak(reset=vram_cap is not None)   # this pass's own peak, below
                    t1 = time.time()
                    # a pass that will COMPILE first (a student slot's first forward in this process)
                    # can take many minutes; say so before it starts, so a stall is visible
                    cst = None if job in TEACHER_JOBS else (rslot if _frozen else slot).st
                    pending = bool(cst is not None and getattr(cst, "compiled", False)
                                   and not getattr(cst, "warm", True))
                    g0 = _compiled_graphs()
                    unit_now.update(phase="gpu")
                    jlog(out, "produce", {"kind": "unit_gpu", "job": job, "region": list(lo),
                                          "round": round_, "size": [int(v) for v in size],
                                          "compile_pending": pending,
                                          "step": None if cst is None else int(getattr(cst, "step", 0))},
                         echo=False)
                    if job in TEACHER_JOBS:
                        B = None
                        if bank.route is not None:
                            g_ = got if isinstance(got, dict) else {"rois": got}
                            P, W, B, attrs = bank.probs_routed(ct_local, lo, size, rois=g_.get("rois"),
                                                               band=g_.get("band"),
                                                               band_from=g_.get("band_from"))
                            reused = g_.get("band") is not None
                            del g_
                            got = None
                        else:
                            P, W, attrs = bank.probs_u8(ct_local, lo, size, rois=got)
                        if job == "reteach":
                            # a new teacher set's recto + rw: the NEXT generation, beside the old pair
                            # (which readers keep using until `commit_sources` moves them)
                            attrs = {**attrs, "regeneration": True, "gen": stores.next_gen(out, lo, 0),
                                     "replaces_teachers": store_teachers(
                                         stores.current_path(out, "recto", lo, 0))}
                        if B is not None:
                            # ROUTED: the coarse rungs follow the BASE teacher (rungs >= 3 are its
                            # targets); a band reused from its committed store fed them already
                            pooled = None if reused else RG.pool_chain(B, RG.COARSE_RUNGS)
                            # rw LAST: its `done` finishes the generation (`recto_needs_regen`)
                            rows = [("recto", P.cpu().numpy(), 8, "prob_u8"),
                                    ("band", B.cpu().numpy(), 8, "prob_u8"),
                                    ("rw", W.cpu().numpy(), 8, "coverage_u8")]
                        else:
                            pooled = RG.pool_chain(P, RG.COARSE_RUNGS)
                            rows = [("recto", P.cpu().numpy(), 8, "prob_u8"),
                                    ("rw", W.cpu().numpy(), 8, "prob_u8")]
                        del P, W, B
                    else:
                        rg = st.get("verso_regen") or {}
                        regen_unit = job == "verso" and round_ == 0 and cat.done("verso", lo) \
                            and bool(rg.get("ckpt"))
                        # a REGENERATION runs from the frozen qualifying checkpoint, never the live one
                        use = rslot if regen_unit else slot
                        stu = use.get(path=rg["ckpt"]) if regen_unit else use.get(round_, st.get("teacher"))
                        heads = "verso" if job == "verso" else "all"
                        sign = -1.0 if job == "verso" else 1.0
                        want = [str(stu.layout.channels[0])] if heads == "verso" else "all"
                        planes = _student_planes(stu, ct_local, ax, lo, size, sign, want, meta5, pyr,
                                                 margin=int(cfg.infer_margin))
                        attrs = {"producer": "student", "ckpt": stu.ckpt, "step": int(stu.step),
                                 "ckpt_sha256": use.sha, "frozen_teacher": bool(round_ >= 1),
                                 "regeneration": bool(regen_unit),
                                 # a round-0 verso that already has a finished generation is the ONE
                                 # regeneration: it goes to the next generation, beside the old one
                                 # (the region's one counter, shared with the recto regeneration)
                                 "gen": (stores.next_gen(out, lo, 0)
                                         if job == "verso" and cat.done("verso", lo) else 0),
                                 "radial_sign": int(sign), "window": int(stu.cfg.infer_window),
                                 "halo": int(stu.cfg.infer_halo), "margin": int(cfg.infer_margin),
                                 "cascade_depth": int(stu.cfg.cascade_depth),
                                 "temps": {str(k): float(v) for k, v in stu.temps.items()}}
                        rows = student_rows_t(planes, stu.layout, heads)
                        del planes
                        pooled = None
                    t_gpu = time.time() - t1
                    meter.add(job, t_gpu)
                    first = job not in peaks
                    peaks[job] = _cuda_peak()
                    gate.set_pending(j for _, j in gpu_units[i + 1:])
                    gate.release()
                    if first:                       # the first pass of its kind: what it really costs
                        vram_report(out, bank, slot, peaks, vram_cap, "first_" + job)
                    _unit_done(unit_now)
                    g1 = _compiled_graphs()
                    if g1 > g0:                     # this pass compiled (or recompiled) something
                        jlog(out, "produce", {"kind": "compile", "job": job, "region": list(lo),
                                              "graphs": g1 - g0, "total_graphs": g1,
                                              "pass_s": round(t_gpu, 1)})
                    extra = {"L": L, "cursor": cursor, "read_wait_s": round(t_in, 2),
                             "gpu_s": round(t_gpu, 2), **_vram()}
                    with lock:
                        pend.append(writer.submit(finish, job, lo, round_, t0, rows, attrs,
                                                  pooled, extra))
                except BaseException:
                    if gate.holds():
                        gate.release()
                    _unit_done(unit_now)
                    with lock:
                        busy.discard(lo)
                    wslots.release()
                    raise
                did = True
                with clock:
                    _release_passed(cache, keys, pos, cursor, cat, round_, verso_on, out, frungs,
                                    leased=leased)
                    release_backlog()
            gate.set_pending(())                    # recomputed on the next pass over the window
            if bank is not None and round_ >= 1:
                # the round-0 teachers have no further use: drop them (VRAM and pinned host copies)
                gb = bank.footprint()
                bank = None
                _empty_cuda()
                jlog(out, "produce", {"kind": "bank_release", "round": round_,
                                      "gb": round(gb / (1 << 30), 3),
                                      "allocated_gb": round(_cuda_alloc() / (1 << 30), 3)})
            for k in [k for k in pre if k not in gpu_units]:
                pre.pop(k)                          # a read for a unit this pass no longer wants
            if not did:
                with clock:
                    _release_passed(cache, keys, pos, cursor, cat, round_, verso_on, out, frungs,
                                    leased=leased)
                    release_backlog()
                time.sleep(IDLE_S)
                if read_phase(out, "") == "produce" and not busy:
                    write_phase(out, "train")     # the window is drained: give the card back
    finally:
        gate.set_pending(())                        # nothing left to make way for: the fields finish
        readers["ex"].shutdown(wait=False, cancel_futures=True)
        for ex in (writer, fielder):
            ex.shutdown(wait=True)
        if fpool is not None:
            fpool.shutdown(wait=True)
        try:
            cache.close()
        except Exception:  # noqa: BLE001
            pass
        hb_stop.set()
        if recycle["why"] is not None:
            rec = {"kind": "recycle", "reason": recycle["why"], "pid": os.getpid(),
                   "fields": recycle["fields"], "rss_gb": round(own_rss_gb(), 2),
                   "uptime_s": round(time.time() - t_start, 1)}
            jlog(out, "sched", rec, echo=False)
            jlog(out, "produce", rec)
            stamp({"phase": "exit", "recycle": True, "reason": recycle["why"]})
        else:
            stamp({"phase": "exit"})
        jlog(out, "produce", {"kind": "exit", "pid": os.getpid()})
    settle()
    return 0


def _vram():
    try:
        import torch
        if torch.cuda.is_available():
            return {"vram_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _bank_park(out, bank, job, rest):
    """Before a unit's pass (under the gpu lock): the teacher bank's weights go to host memory when no
    teacher pass is left in this pass over the window (`rest`: this unit and the ones after it), and
    back to the card before a teacher pass -- so a verso / self pass has the VRAM the bank held.
    Logs `bank_offload` / `bank_onload` with the seconds each took."""
    if job in TEACHER_JOBS:
        s = bank.onload()
        kind = "bank_onload"
    elif not any(j in TEACHER_JOBS for _, j in rest):
        s = bank.offload()
        kind = "bank_offload"
    else:
        return None
    if s is not None:
        jlog(out, "produce", {"kind": kind, "s": round(s, 3), "for": job,
                              "allocated_gb": round(_cuda_alloc() / (1 << 30), 3)}, echo=False)
    return s


def _cuda_alloc():
    try:
        import torch
        return int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


def _empty_cuda():
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def module_bytes(mod):
    """Bytes of a torch module's parameters and buffers (0 for None)."""
    if mod is None:
        return 0
    try:
        return int(sum(t.numel() * t.element_size() for t in list(mod.parameters()) + list(mod.buffers())))
    except Exception:  # noqa: BLE001
        return 0


def _cuda_peak(reset=False):
    """torch.cuda.max_memory_allocated (None without CUDA); `reset` starts a new peak (per pass)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001
        return None


def vram_report(out, bank, slot, peaks, cap, why):
    """One `vram_report` line: what the producer's process has allocated / reserved on its card, what
    of it the teacher bank and the student are (their parameter and buffer bytes), and the last pass's
    peak of each job kind -- the numbers the trainer / producer budget split is tuned from. Nothing
    without CUDA (`cap` None)."""
    if cap is None:
        return None
    try:
        import torch
        g = float(1 << 30)
        stu = getattr(getattr(slot, "st", None), "raw", None)
        rec = {"kind": "vram_report", "why": str(why),
               "allocated_gb": round(torch.cuda.memory_allocated() / g, 3),
               "reserved_gb": round(torch.cuda.memory_reserved() / g, 3),
               "cap_gb": round(cap / g, 3),
               "bank_gb": round((bank.footprint() if bank is not None else 0) / g, 3),
               "bank_loaded": bank is not None and any(r[3] is not None for r in bank.items),
               "student_gb": round(module_bytes(stu) / g, 3),
               "peak_gb": {str(k): round(v / g, 3) for k, v in peaks.items() if v is not None}}
    except Exception as e:  # noqa: BLE001  -- a report must never take the producer down
        rec = {"kind": "vram_report", "why": str(why), "err": repr(e)}
    jlog(out, "produce", rec, echo=False)
    return rec


def _vram_cap(device, mem_frac):
    """The bytes this process may reserve on its CUDA card: `mem_frac` of it (the producer's fraction),
    or all of it; None without CUDA."""
    if device is None or not str(device).startswith("cuda"):
        return None
    try:
        import torch
        tot = torch.cuda.get_device_properties(torch.device(device)).total_memory
        return int(tot * float(mem_frac)) if mem_frac else int(tot)
    except Exception:  # noqa: BLE001
        return None


VRAM_HEADROOM = 1 << 30      # a pass starting with less than this left under the cap is logged


def _vram_check(out, job, lo, cap):
    """Before a network pass: give the allocator's cached blocks back (the fields' among them) and log
    a `vram_pressure` line when what is still reserved leaves less than `VRAM_HEADROOM` under `cap` --
    the state in which the caching allocator thrashes (free / retry at the fraction cap, no progress)
    instead of failing."""
    if cap is None:
        return None
    try:
        import torch
        torch.cuda.empty_cache()
        res, alloc = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
    except Exception:  # noqa: BLE001
        return None
    if cap - res < VRAM_HEADROOM:
        jlog(out, "produce", {"kind": "vram_pressure", "job": job, "region": list(lo),
                              "reserved_gb": round(res / (1 << 30), 2),
                              "allocated_gb": round(alloc / (1 << 30), 2),
                              "cap_gb": round(cap / (1 << 30), 2)})
    return res


def _compiled_graphs():
    """How many graphs torch.compile has compiled in this process (0 before torch is imported)."""
    import sys
    utils = sys.modules.get("torch._dynamo.utils")
    if utils is None:
        return 0
    try:
        return int(utils.counters["stats"]["unique_graphs"])
    except Exception:  # noqa: BLE001
        return 0


def _student_planes(stu, ct, ax, lo, size, sign, want, meta5, pyr, margin=None):
    from rvsm import infer
    return infer.student_region(stu, ct, ax, lo, size, sign=sign, heads=want, meta=meta5, pyr=pyr,
                                as_tensor=True, margin=margin)


def _gpu_order(units, leased=None, held=None):
    """The GPU units of one pass over the window, blocking passes first. A region without its recto
    (round 0: `teacher`; round r >= 1: `self`) is one a sampler worker cannot use at all, and the
    in-order DataLoader then holds every worker until it lands; a region lacking only its `verso` is
    already trainable (the verso earns a revisit later). Before this the window ran in walk order,
    so once verso came on a window of ~20 one-minute verso passes ran ahead of the teacher pass the
    fastest worker was waiting on and the trainer sat idle for 14 minutes (paris4, step 46440).
    Stable: walk order is kept within each class.

    `leased` (`lease_order`): the blocking passes of LEASED regions go before every other, in lease
    order -- a lease is a worker's declared need (paris4, 20:13-20:46: a worker waited 33 minutes on a
    region outside the window while 16 verso passes ran).

    A `reteach` (a new teacher pass over a region whose recto came from another teacher set) is NOT
    blocking: the region trains on its old recto meanwhile. It runs after the blocking passes -- a
    held-out region's first (`held`: the evaluation reference), the others beside the verso passes."""
    rank = {lo: i for i, lo in enumerate(leased or ())}
    held = {tuple(int(v) for v in h) for h in (held or ())}

    def key(u):
        if u[1] == "reteach":
            return (2, 0) if tuple(u[0]) in held else (2, 1)
        if u[1] == "verso":
            return (2, 1)
        return (0, rank[u[0]]) if u[0] in rank else (1, 0)
    return sorted(units, key=key)


def lease_order(recs):
    """The workers' leased regions (`cursor_records`), each once, in the order they are needed: every
    worker's first lease (the home it is reading or waiting on), then every worker's second, and so on."""
    ls = []
    for r in sorted(recs, key=lambda r: int(r.get("worker", 0) or 0)):
        one = []
        for lo in r.get("lease") or ():
            try:
                one.append(tuple(int(v) for v in lo))
            except (TypeError, ValueError):
                continue
        ls.append(one)
    out, seen = [], set()
    for k in range(max((len(x) for x in ls), default=0)):
        for x in ls:
            if k < len(x) and x[k] not in seen:
                seen.add(x[k])
                out.append(x[k])
    return out


def _working_set(lorder, window):
    """The producer's regions for one pass: every LEASED region first (lease order), wherever it lies
    in the walk -- a worker leases the homes of its next visits along its own stride, far past
    `head + L`, and a region whose first walk position is behind the cursor is not in `_window` at all
    -- then the window's regions not already named."""
    seen = set(lorder)
    return list(lorder) + [lo for lo in window if lo not in seen]


STARVED_S = 60.0         # a worker waiting this long for its pending homes gets a `starved` line


def _log_starved(out, recs, last, in_set, job_of, why_of, now=None):
    """One `starved` line per worker per STARVED_S while it waits (its cursor record's `wait_since`):
    the leased regions it waits on that still lack a pass, each with that pass, whether it is in this
    producer pass's set, and the reasons it might not be (busy, skipped after a read timeout, no
    size). Returns the lines logged."""
    now = time.time() if now is None else now
    got = []
    for r in recs:
        ws = r.get("wait_since")
        w = r.get("worker")
        if not isinstance(ws, (int, float)) or now - ws < STARVED_S or now - last.get(w, 0.0) < STARVED_S:
            continue
        regs = []
        for lo in r.get("lease") or ():
            try:
                lo = tuple(int(v) for v in lo)
                job = job_of(lo)
            except Exception:  # noqa: BLE001
                continue
            if job in ("teacher", "self"):
                regs.append({"region": list(lo), "job": job, "in_set": lo in in_set, **why_of(lo)})
        rec = {"kind": "starved", "worker": w, "waited_s": round(now - ws, 1), "waiting_on": regs}
        jlog(out, "produce", rec)
        last[w] = now
        got.append(rec)
    return got


def _window(route, pos, cursor, L, held, head=None):
    """The producer's working set: every held-out region that is not finished, then the regions whose
    walk position lies between the trainer's cursor (the SLOWEST worker) and `L` visits past the
    FASTEST worker's `head` (`read_cursor_head`; the cursor when omitted).

    Both ends matter. Each worker walks its own stride of the walk and the DataLoader takes their
    batches in turn, so the workers drift apart by whole visits (a mostly-air visit ends early) --
    and a window of `L` past the slowest one can end before the fastest one's next visit. That
    worker then waits for a store nobody will produce, the in-order DataLoader waits for that worker,
    the slow workers never advance the cursor, and the run deadlocks (paris4 after the resume at step
    2000: cursor 30, window to 39, worker 0 waiting on 42)."""
    top = max(int(cursor), int(head if head is not None else cursor)) + int(L)
    out = [lo for lo in route[:len(held)]]
    for lo in route[len(held):]:
        n = pos.get(lo)
        if n is None or n < cursor:
            continue
        if n <= top:
            out.append(lo)
    return out


def field_rungs(cfg):
    """The distance rungs this run needs: 2..4, and only the ones it trains on. A run whose ladder
    stops at rung 3 has no use for a rung-4 EDT, and an EDT is the most expensive thing the producer
    does that is not a network."""
    from rvsm import targets as TG
    ks = tuple(k for k in (2, 3, 4) if int(k) <= TG.MAX_RUNG and int(k) in tuple(cfg.rungs))
    return ks or (2,)


def verso_needs_regen(out, lo, regen):
    """Is this region's round-0 verso a generation-0 store made by a checkpoint older than the
    regeneration trigger (`state["verso_regen"]`)? Only generation 0 is ever regenerated (once)."""
    from rvsm import stores
    if not regen or stores.store_gen(out, "verso", lo, 0) != 0:
        return False
    made = stores.read_attrs(stores.store_path(out, "verso", lo, 0)).get("step")
    return made is None or int(made) < int(regen.get("step", 0))


DEFAULT_TEACHERS = ("recto", "m7")   # the teacher set of a config without `teacher_ckpts`, and of a
                                     # recto store written before stores recorded theirs (paris4's fused
                                     # recto + m7 stores up to the 2026-09-25 switch)


def teacher_names(cfg):
    """The teachers this run's round-0 recto targets come from, in fusion order: the KEYS of
    `teacher_ckpts` (`{"m7": path}` alone is the m7-only mode), or DEFAULT_TEACHERS when it is empty."""
    return [str(n) for n in (getattr(cfg, "teacher_ckpts", None) or {})] or list(DEFAULT_TEACHERS)


def store_teachers(path):
    """The teacher set a finished recto store records (`teachers` attr), or DEFAULT_TEACHERS for a store
    written before the attr existed."""
    from rvsm import stores
    t = stores.read_attrs(path).get("teachers")
    return [str(n) for n in t] if isinstance(t, (list, tuple)) and t else list(DEFAULT_TEACHERS)


ROUTE_TOKEN = "route:"    # the identity token of a ROUTED recto (`teacher_set`, `store_ident`)


def teacher_set(cfg):
    """The IDENTITY a round-0 recto of this config must have: the teacher set (`teacher_names`) -- or,
    under a per-rung route (`config.route_spec`), its fine and base teachers plus a `route:<sig>` token.
    A store is regenerated (`recto_needs_regen`) when its own identity (`store_ident`) differs, so turning
    the route on (or changing it, or its coverage rule) regenerates exactly like the m7 switch did, and
    turning it off regenerates back to the plain set."""
    r = CFG.route_spec(cfg)
    if r is None:
        return teacher_names(cfg)
    return [r.fine, r.base, ROUTE_TOKEN + r.sig]


def store_ident(path):
    """A finished recto store's identity, comparable with `teacher_set`: its `teachers`, plus the
    `route:<sig>` token when it is a routed store (attr `route`)."""
    from rvsm import stores
    r = stores.read_attrs(path).get("route")
    return store_teachers(path) + ([ROUTE_TOKEN + str(r)] if r else [])


def routed_set(teachers):
    """Is this identity (`teacher_set`) a routed one?"""
    return any(str(t).startswith(ROUTE_TOKEN) for t in (teachers or ()))


def band_reuse(out, lo, teachers):
    """For a routed reteach: the path of a finished store holding the BASE teacher's rung-2 probability
    already, so the base (m7) teacher is not run again -- the region's committed recto when the base
    teacher alone made it (the m7-only stores of paris4), or the committed generation's `band` when it is
    a routed store with the same base. None: the base teacher must run."""
    from rvsm import stores
    if not routed_set(teachers):
        return None
    base = str(teachers[1])
    p = stores.current_path(out, "recto", lo, 0)
    if stores.is_done(p) and store_ident(p) == [base]:
        return p
    b = stores.current_path(out, "band", lo, 0)
    if stores.is_done(b) and store_teachers(b)[1:2] == [base]:
        return b
    return None


def recto_needs_regen(out, lo, teachers):
    """WRITER side: is the region's newest finished round-0 recto (with its rw beside it) made by a
    teacher set other than `teachers`? Then a new teacher pass is due, written as the next generation
    (`stores.next_gen`) beside the old one -- never in place. None/empty `teachers`: never."""
    from rvsm import stores
    if not teachers:
        return False
    g = stores.store_gen(out, "recto", lo, 0)
    if g < 0:
        return False                                 # no recto at all: that is the first teacher pass
    p = stores.gen_path(stores.store_path(out, "recto", lo, 0), g)
    if sorted(store_ident(p)) != sorted(str(t) for t in teachers):
        return True
    # the rw of the same generation is written after the recto (and a routed band): a unit cut between
    # them is redone
    if routed_set(teachers) and not stores.is_done(stores.gen_path(stores.store_path(out, "band", lo, 0), g)):
        return True
    return not stores.is_done(stores.gen_path(stores.store_path(out, "rw", lo, 0), g))


def recto_stale(out, lo, teachers):
    """READER side: does the region's COMMITTED round-0 recto come from another teacher set? True
    until the regenerated recto + rw are committed (at once after the reteach, `commit_sources`)."""
    from rvsm import stores
    if not teachers:
        return False
    p = stores.current_path(out, "recto", lo, 0)
    return stores.is_done(p) and sorted(store_ident(p)) != sorted(str(t) for t in teachers)


def fields_behind(out, lo):
    """Is the region's committed recto newer than its committed fields (a reteach whose fields
    rebuild has not landed yet)? Fields built from a recto sit at a generation >= that recto's
    (`targets.field_gen`), so committed recto > committed fields means they came from an older one.
    Only a region with a verso has fields."""
    from rvsm import stores
    b = stores.bundle_state(out, lo, 0)
    return b["recto"] > b["gen"] and stores.store_gen(out, "verso", lo, 0) >= 0


def recto_regen_todo(out, route, held, teachers):
    """Every produced round-0 region the recto regeneration still owes something: a stale committed
    recto (`recto_stale`) or fields not yet rebuilt from the new one (`fields_behind`), in priority
    order: the held-out regions first (the evaluation reference), then walk order, then any other
    produced region (none, normally)."""
    from rvsm import regions as RG
    if not teachers:
        return []
    done = [tuple(int(v) for v in lo) for lo in RG.Catalog(out, 0, ttl=0.0).list_done("recto")]
    rank = {}
    for i, h in enumerate(held or ()):
        rank.setdefault(tuple(int(v) for v in h["lo"]), (0, i))
    for i, lo in enumerate(route or ()):
        rank.setdefault(tuple(int(v) for v in lo), (1, i))
    todo = [lo for lo in done if recto_stale(out, lo, teachers) or fields_behind(out, lo)]
    return sorted(todo, key=lambda lo: rank.get(lo, (2, 0)))


RECTO_TODO_S = 300.0     # the recto regeneration's work list is rescanned (and logged) this often
RECTO_FIELDS_INFLIGHT = 1   # backlog fields rebuilds queued at once (the window's own fields go first)


def _recto_rescan(out, rr, route, held, teachers, now=None, meter=None, share=None):
    """Rescan the recto regeneration's work list (`recto_regen_todo`) when it is RECTO_TODO_S old, and
    log it: `recto_regen` (stale rectos and fields rebuilds remaining, the identity `teachers` -- a
    routed one carries its `route:<sig>` token -- and, from the producer, the backlog's GPU share)
    while work remains, `recto_regen_done` once when it reaches zero. The producer calls it on EVERY
    pass over the window, idle or not, so the backlog is visible even while it gets no GPU time.
    Returns True when it rescanned."""
    now = time.time() if now is None else now
    if not teachers:
        return False
    if rr.get("todo") is not None and now - float(rr.get("t", 0.0)) < RECTO_TODO_S:
        return False
    rr["todo"] = recto_regen_todo(out, route, held, teachers)
    rr["t"] = now
    n, last = len(rr["todo"]), rr.get("n")
    if n:
        held_set = {tuple(int(v) for v in h["lo"]) for h in (held or ())}
        st = [lo for lo in rr["todo"] if recto_stale(out, lo, teachers)]
        rec = {"kind": "recto_regen", "remaining": len(st),
               "fields_remaining": sum(1 for lo in rr["todo"] if fields_behind(out, lo)),
               "teachers": list(teachers),
               "heldout_remaining": sum(1 for lo in st if lo in held_set)}
        if meter is not None:
            rec.update({"share": round(meter.share(now), 4), "share_target": share,
                        "reteach_s": round(meter.seconds(now, reteach=True), 1),
                        "gpu_s": round(meter.seconds(now), 1)})
        jlog(out, "produce", rec)
    elif last:
        jlog(out, "produce", {"kind": "recto_regen_done", "teachers": list(teachers)})
    rr["n"] = n
    return True


SHARE_WINDOW_S = 1200.0   # the reteach share is measured over this much of the producer's recent GPU time
RETEACH_EST_S = 20.0      # a backlog reteach's GPU seconds before any has been measured
SHARE_MAX_PER_PASS = 4    # backlog reteaches admitted at most per pass over the window


class ReteachMeter:
    """The producer's GPU seconds per pass over the last SHARE_WINDOW_S, and the share of them that
    `reteach` passes took (`cfg.reteach_share` is the target)."""

    def __init__(self, window=SHARE_WINDOW_S):
        self.window = float(window)
        self.rows = []                   # (t, gpu seconds, is a reteach)

    def add(self, job, gpu_s, now=None):
        now = time.time() if now is None else now
        self.rows.append((float(now), max(float(gpu_s), 0.0), str(job) == "reteach"))
        self._trim(now)

    def _trim(self, now):
        lo = float(now) - self.window
        while self.rows and self.rows[0][0] < lo:
            self.rows.pop(0)

    def seconds(self, now=None, reteach=None):
        now = time.time() if now is None else now
        self._trim(now)
        return sum(s for _, s, r in self.rows if reteach is None or r == bool(reteach))

    def share(self, now=None, extra=0.0):
        """reteach seconds / all seconds, with `extra` seconds of reteach added to both (a projection
        of the reteaches just admitted). 0 before any pass has been measured."""
        tot = self.seconds(now) + float(extra)
        return (self.seconds(now, reteach=True) + float(extra)) / tot if tot > 0 else 0.0

    def est(self, now=None):
        """A reteach pass's expected GPU seconds: the mean of the recent ones, else RETEACH_EST_S."""
        s = [q for _, q, r in self.rows if r]
        return float(np.mean(s)) if s else RETEACH_EST_S


def _share_backlog(out, rr, meter, share, gpu_units, leased, step, now=None):
    """The recto regeneration's GPU-time SHARE: while the window's GPU units hold no blocking pass (a
    first-visit `teacher`, a round-r `self`, a LEASED region's anything: they keep their priority) and
    the reteach share of the last SHARE_WINDOW_S of GPU time is under `share`, admit backlog reteaches
    -- `step(units)` is `_recto_backlog` appending at most one unit -- at the FRONT of the window's
    non-blocking units (the verso passes, which a worker never waits on), up to SHARE_MAX_PER_PASS,
    projecting each admitted pass at `meter.est()` seconds. Logs `recto_backlog_admit` with the share
    per admitted unit. `share <= 0`: nothing (the backlog then runs only when the window is idle).
    Returns the admitted units."""
    now = time.time() if now is None else now
    if float(share or 0.0) <= 0 or not gpu_units:
        return []
    lset = {tuple(int(v) for v in lo) for lo in (leased or ())}
    if any(j in BLOCKING_JOBS or tuple(int(v) for v in lo) in lset for lo, j in gpu_units):
        return []
    got, est = [], meter.est(now)
    while len(got) < SHARE_MAX_PER_PASS:
        cur = meter.share(now, extra=est * len(got))
        if cur >= float(share):
            break
        new = []
        step(new)
        if not new:
            break
        for u in new:
            jlog(out, "produce", {"kind": "recto_backlog_admit", "region": [int(v) for v in u[0]],
                                  "job": u[1], "share": round(cur, 4), "share_target": float(share),
                                  "reteach_s": round(meter.seconds(now, reteach=True), 1),
                                  "gpu_s": round(meter.seconds(now), 1)}, echo=False)
        got += new
    gpu_units[:0] = got
    return got


def _recto_backlog(out, rr, route, held, teachers, cat, verso_on, rungs, busy, lock, skip_until, sized,
                   gpu_units, backlog_keys, submit_fields, now=None):
    """One idle step of the recto regeneration (the producer calls it when the window has no GPU unit).

    `rr` is the producer's state: the work list (`recto_regen_todo`, rescanned every RECTO_TODO_S),
    the backlog fields in flight and the last count logged. Walks the list in priority order: a region
    that owes nothing more is dropped, a region with nothing left to run is committed
    (`commit_sources`: a restart cut its commit off), a `fields` job goes to the fields pool
    (`submit_fields`, the region marked busy) while fewer than RECTO_FIELDS_INFLIGHT backlog fields are
    in flight, and the first GPU job (the `reteach`, or a verso regeneration's own pass) is appended to
    `gpu_units` -- one per call, like the verso backlog. Logs `recto_regen` (stale rectos remaining,
    fields rebuilds remaining) at every rescan while work remains and `recto_regen_done` once when both
    reach zero. Returns the regions still owed something."""
    now = time.time() if now is None else now
    if not teachers:
        return 0
    _recto_rescan(out, rr, route, held, teachers, now=now)
    with lock:
        infl = {lo for lo in rr.get("fields", ()) if lo in busy}
    rr["fields"] = infl
    keep = []
    picked = False
    for lo in rr["todo"]:
        if picked:
            keep.append(lo)
            continue
        with lock:
            if lo in busy:
                keep.append(lo)
                continue
        stale = recto_stale(out, lo, teachers)
        if not stale and not fields_behind(out, lo):
            continue
        keep.append(lo)
        if not sized(lo) or skip_until.get(lo, 0.0) > now:
            continue
        # the backlog's own passes only: no first verso for a region the window has not reached
        job = _next_job(cat, lo, 0, bool(verso_on) and cat.done("verso", lo), out, rungs=rungs,
                        regen=read_state(out).get("verso_regen"), teachers=teachers)
        if job is None:
            commit_sources(out, lo, 0, rungs)
            if not recto_stale(out, lo, teachers) and not fields_behind(out, lo):
                keep.pop()
        elif job == "fields":
            if len(infl) >= RECTO_FIELDS_INFLIGHT:
                continue
            with lock:
                busy.add(lo)
                submit_fields(lo)
            infl.add(lo)
            backlog_keys.add(lo)
        elif job != "teacher":
            gpu_units.append((lo, job))
            backlog_keys.add(lo)
            picked = True
    rr["todo"] = keep
    return len(keep)


def _next_job(cat, lo, round_, verso_on, out, rungs=(2, 3, 4), regen=None, teachers=None):
    """Which pass this region lacks, in the order the state machine allows -- or None when it is done.

    Round 0: the teacher pass, then -- when the region's recto comes from another teacher set than the
    configured one (`teachers`, `recto_needs_regen`) -- the `reteach` pass that writes the next
    generation of recto + rw, then (once the gate has fired) the flipped-sign verso, then the distance
    fields -- at the sources' GENERATION: a verso regenerated once (`regen`, see `verso_needs_regen`) or
    a regenerated recto gets its own fields. Round r >= 1: one multi-head student pass, then the fields
    at the pooled rungs."""
    from rvsm import targets as TG
    if round_ == 0:
        if not cat.done("recto", lo):
            return "teacher"
        if teachers and recto_needs_regen(out, lo, teachers):
            return "reteach"
        if verso_on and not cat.done("verso", lo):
            return "verso"
        if verso_on and verso_needs_regen(out, lo, regen):
            return "verso"
        if cat.done("verso", lo) and not TG.fields_current(out, lo, round_, rungs):   # the writer's own test
            return "fields"
        return None
    if not cat.done("recto", lo):
        return "self"
    if not TG.fields_current(out, lo, round_, rungs):
        return "fields"
    return None


def _release_passed(cache, keys, pos, cursor, cat, round_, verso_on, out, rungs=(2, 3, 4), leased=()):
    """Give back the CT of every region whose stores are finished and whose walk position the trainer's
    cursor has passed. A region still ahead of the cursor keeps its shards: the trainer is about to
    read them -- and so does a region a worker has LEASED (the cursor is published when a visit starts,
    so it is already past the visit being read; and a rung 3-6 visit's home can sit far behind it)."""
    leased = set(leased)
    for lo in list(keys):
        n = pos.get(lo)
        if n is None or n >= int(cursor) or lo in leased:
            continue
        if _next_job(cat, lo, round_, verso_on, out, rungs=rungs,
                     regen=read_state(out).get("verso_regen")) is None:
            cache.release(keys.pop(lo))


UNIT_STALL_S = 300.0    # a unit in flight this long gets every thread's stack dumped, and again every
                        # UNIT_STALL_S after that
READ_TIMEOUT_S = 1200.0  # a unit's read (its shards, a teacher's CT) that has not come back is abandoned
READ_SKIP_S = 3600.0     # ... and its region gets no GPU unit for this long
LOCK_WAIT_LOG_S = 60.0   # a traced lock waited on this long logs its holder
_STACK_FILES = {}        # kept open for the process's life


class TracedLock:
    """A `threading.Lock` that knows who holds it: the holder's thread name and since when. `with` it
    as with a lock; a wait longer than `LOCK_WAIT_LOG_S` logs a `lock_wait` line naming the holder (and
    keeps waiting). The producer's shard-cache lock is one: a stall behind it names its cause."""

    def __init__(self, name, log=None):
        import threading
        self.name, self.log = str(name), log
        self._lk = threading.Lock()
        self._who, self._since = None, None

    def holder(self):
        who, since = self._who, self._since
        return None if who is None else {"thread": who, "s": round(time.time() - since, 1)}

    def acquire(self):
        import threading
        t0 = time.time()
        while not self._lk.acquire(timeout=LOCK_WAIT_LOG_S):
            if self.log is not None:
                try:
                    self.log({"kind": "lock_wait", "lock": self.name,
                              "thread": threading.current_thread().name,
                              "waited_s": round(time.time() - t0, 1), "holder": self.holder()})
                except Exception:  # noqa: BLE001
                    pass
        self._who, self._since = threading.current_thread().name, time.time()
        return True

    def release(self):
        self._who = self._since = None
        self._lk.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


BLOCKING_JOBS = ("teacher", "self")   # the passes a sampler worker is blocked on (`_gpu_order`)
TEACHER_JOBS = ("teacher", "reteach")  # the passes the teacher bank runs (a reteach never blocks)
FIELDS_DEFER_S = 600.0   # GPU fields that have waited this long may start while only verso passes are pending


class GpuGate:
    """Who gets the producer's card: its network passes, or its GPU distance fields.

    `lock` is the one `gpu_lock` (a `TracedLock`): nothing else ever runs beside a pass. A pass takes
    it outright (`pass_acquire`, `release`). The fields take it only through `fields_hold(since)` and
    by priority rules, because a fields region is up to ~75 s of device work and a pass that a
    sampler worker is waiting on (a BLOCKING pass, `BLOCKING_JOBS`: teacher in round 0, self in
    round >= 1) must never queue behind it:

    - the fields START only when no GPU unit is pending in the window (`set_pending`, kept by the pass
      loop), or when the pending units are all verso passes and the fields job was queued at least
      `FIELDS_DEFER_S` ago (verso passes are never blocking, and they must not starve the fields);
    - between batches (`_FieldsHold.yield_point`) the fields give the lock back as soon as a blocking
      pass is pending, and start again under the same rule; each yield logs a `fields_yield` line.

    The rules decide only WHEN the fields' batches run, never what they compute: the stores are the
    same bytes."""

    def __init__(self, lock, log=None, defer_s=None, now=None):
        import threading
        self.lock, self.log = lock, log
        self.defer_s = FIELDS_DEFER_S if defer_s is None else float(defer_s)
        self.now = now or time.time
        self.cv = threading.Condition()
        self.pending = ()                   # the GPU jobs still to run in this pass over the window
        self.waiting = 0                    # blocking passes inside `pass_acquire`

    def set_pending(self, jobs):
        with self.cv:
            self.pending = tuple(str(j) for j in jobs)
            self.cv.notify_all()

    def blocking(self):
        """Is a blocking pass pending (in the window, or waiting for the lock)?"""
        return self.waiting > 0 or any(j in BLOCKING_JOBS for j in self.pending)

    def fields_may_start(self, since):
        if self.blocking():
            return False
        return not self.pending or self.now() - float(since) >= self.defer_s

    def pass_acquire(self, job):
        blk = str(job) in BLOCKING_JOBS
        if blk:
            with self.cv:
                self.waiting += 1
        try:
            self.lock.acquire()
        finally:
            if blk:
                with self.cv:
                    self.waiting -= 1
                    self.cv.notify_all()
        return True

    def release(self):
        self.lock.release()
        with self.cv:
            self.cv.notify_all()

    def holds(self):
        """Does the calling thread hold the lock?"""
        import threading
        h = self.lock.holder()
        return h is not None and h["thread"] == threading.current_thread().name

    def _fields_acquire(self, since):
        """Wait until the fields may start, then take the lock -- and, when a pass became pending
        while this thread waited for the lock, give it straight back and wait again."""
        while True:
            with self.cv:
                while not self.fields_may_start(since):
                    self.cv.wait(timeout=5.0)       # the age rule is a clock, not an event
            self.lock.acquire()
            if self.fields_may_start(since):
                return
            self.release()

    def fields_hold(self, since, tag=None):
        """The `gpu_lock` a fields job hands to `targets.region_fields`: a context manager that takes the
        card by the rules above, with the `yield_point` `_fields_torch` calls between batches.
        `since` is when the job was queued."""
        return _FieldsHold(self, since, tag)


class _FieldsHold:
    def __init__(self, gate, since, tag=None):
        self.gate, self.since, self.tag = gate, float(since), dict(tag or {})
        self.yields = 0

    def __enter__(self):
        self.gate._fields_acquire(self.since)
        return self

    def __exit__(self, *exc):
        self.gate.release()
        return False

    def yield_point(self, flush=None):
        """Between two batches: when a blocking pass is pending, finish the device work in flight
        (`flush`), release the card, and take it back once the fields may start again. True when it
        yielded."""
        g = self.gate
        if not g.blocking():
            return False
        if flush is not None:
            flush()
        why = [j for j in g.pending if j in BLOCKING_JOBS] or ["waiting"]
        t0 = time.time()
        g.release()
        g._fields_acquire(self.since)
        self.yields += 1
        if g.log is not None:
            try:
                g.log({"kind": "fields_yield", **self.tag, "for": why, "n": self.yields,
                       "waited_s": round(time.time() - t0, 2)})
            except Exception:  # noqa: BLE001
                pass
        return True


def _unit_done(unit):
    """The unit in flight is over: forget it (the watch's lock list stays) and stop its stall timer."""
    for k in ("job", "region", "t0", "phase", "dumped"):
        unit.pop(k, None)
    try:
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
    except Exception:  # noqa: BLE001
        pass


def _stall_timer(f, every=None):
    """`faulthandler.dump_traceback_later`: every `every` (`UNIT_STALL_S`) seconds until `_unit_done`,
    all thread stacks into `f` from faulthandler's own C thread -- which, unlike the Python watch,
    dumps even while some thread holds the GIL in native code."""
    try:
        import faulthandler
        faulthandler.dump_traceback_later(float(every or UNIT_STALL_S), repeat=True, file=f)
    except Exception:  # noqa: BLE001
        pass


def register_stack_dumps():
    """`kill -USR1 <pid>` dumps every thread's Python stack to this process's stderr (for the producer:
    run.log). Called first thing in the spawned producer and again once torch and its libraries are
    loaded, so nothing installed in between can have replaced it."""
    try:
        import faulthandler
        import signal
        import sys
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
        return True
    except (AttributeError, ValueError, RuntimeError, OSError):
        return False


def stack_dump_file(out, name):
    """`<out>/logs/<name>_stacks.txt`, opened once for the process's life: where the stall watch writes
    its dumps (and stderr gets them too)."""
    p = os.path.join(str(out), "logs", f"{name}_stacks.txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    f = _STACK_FILES.get(p)
    if f is None:
        f = _STACK_FILES[p] = open(p, "a", buffering=1)
    return f


def unit_watchdog(out, unit, f, stop, every=UNIT_STALL_S, poll=10.0):
    """The producer's stall watch: while a unit (its read or its GPU pass) has been in flight longer
    than `every` seconds, dump all thread stacks into `f` and to stderr and log a `unit_stall` line
    naming the holders of the traced locks in `unit["locks"]` -- once per `every`, so the next stall
    says where it is without anyone attaching to the process."""
    import faulthandler
    import sys
    while not stop.wait(poll):
        try:
            u = dict(unit)
            if "t0" not in u:
                continue
            age = time.time() - float(u["t0"])
            if age >= every * (int(u.get("dumped", 0)) + 1):
                holders = {lk.name: lk.holder() for lk in u.get("locks") or ()}
                head = (f"\n==== unit_stall {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} "
                        f"job {u.get('job')} region {u.get('region')} phase {u.get('phase')} "
                        f"in flight {age:.0f} s, locks {holders} ====\n")
                for dst in (f, sys.stderr):
                    try:
                        dst.write(head)
                        dst.flush()
                        faulthandler.dump_traceback(file=dst, all_threads=True)
                        dst.flush()
                    except Exception:  # noqa: BLE001
                        pass
                unit["dumped"] = int(u.get("dumped", 0)) + 1
                jlog(out, "produce", {"kind": "unit_stall", "job": u.get("job"), "region": u.get("region"),
                                      "phase": u.get("phase"), "s": round(age, 1), "locks": holders,
                                      "stacks": f.name}, echo=False)
        except Exception:  # noqa: BLE001  -- the watch must never take the producer down
            pass


def limit_compile_threads(n=1):
    """Keep torch inductor's compile workers in this process: `TORCHINDUCTOR_COMPILE_THREADS` (unless
    the environment already sets it; `RVSM_COMPILE_THREADS` overrides `n`), also applied to an
    already-imported inductor config.

    Inductor's default is one compile SUBPROCESS per core. On the 8-core production host a producer
    compiling the student with max-autotune started eight of them beside the trainer, and host RAM
    went from 20 to 64 GB in two minutes (the 17:20 reboot). In-process compiling is slower once per
    process and costs no extra processes."""
    import sys
    n = int(os.environ.get("RVSM_COMPILE_THREADS", n))
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(max(n, 1)))
    cfgmod = sys.modules.get("torch._inductor.config")
    if cfgmod is not None:
        cfgmod.compile_threads = int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
    return int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])


def _produce_entry(cfg_json, out, gpu, frac, backend):
    """The spawned producer's entry point. Sets `CUDA_VISIBLE_DEVICES` BEFORE torch is imported, which
    is why this module imports torch nowhere at the top level."""
    own_process_group()         # the producer, its forkserver and its fields pool: one killable group
    register_stack_dumps()      # `kill -USR1` dumps the producer's threads into run.log
    limit_compile_threads()
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

    Each row also carries the reference against ITSELF (`base_*`), the baseline a topological error is
    read against. With `compare_stores`' default `betti_dilate = 0` the reference compared with itself
    is the same bool volume twice, so that baseline is exactly zero error at the reference's own Betti
    numbers: it is written down from the reference's side of the one comparison instead of paying for a
    second one.

    HOST MEMORY. This runs in the trainer, beside six loader workers and the producer, on a 64 GB host
    (paris4 step 2000 OOMed it here). So: the student pass stays on the card (`as_tensor`, cascade
    included) and only its uint8 recto crosses to the host (1 GB for 1024^3); the reference store is
    never read whole -- `compare_stores` streams it in haloed blocks; and each region's arrays are
    dropped before the next one starts. Peak: ~1 GB plus one block's working set (~1 GB)."""
    import torch
    from rvsm import evalsurf as EV, infer, stores
    rows = []
    stu = None
    for h in list(held)[:int(n)]:
        lo = tuple(int(v) for v in h["lo"])
        p = stores.current_path(out, "recto", lo, 0)   # the COMMITTED reference (a regenerated recto)
        if not stores.is_done(p):
            continue
        ref = stores.open_store(p)               # lazy: read block by block by `compare_stores`
        shape = tuple(int(v) for v in ref.shape[-3:])
        if stu is None:
            stu = infer.student_fn(ckpt, device=device, compile=False)
        head = str(stu.layout.channels[0])
        planes = infer.student_region(stu, ct or cfg.ct, ax, lo, shape, sign=1.0, heads=[head],
                                      meta=meta5, as_tensor=True, margin=int(cfg.infer_margin))
        pred = infer.u8_t(planes[head]).cpu().numpy()
        del planes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        r = EV.compare_stores(ref, pred, device=(device if str(device or "cpu") != "cpu" else None))
        del pred, ref
        base = {k: r[k] for k in r if k.startswith("betti") or k == "euler"}
        base.update({"betti0": r["betti0_ref"], "betti1": r["betti1_ref"], "betti2": r["betti2_ref"],
                     "euler": r["euler_ref"], "betti0_err": 0, "betti1_err": 0,
                     "betti0_err_norm": 0.0, "betti1_err_norm": 0.0})
        rows.append({"region": list(lo), **r, **{f"base_{k}": v for k, v in base.items()}})
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


GATE_METRIC = "dice_recto_r2"   # the recto head alone at rung 2, against the immutable recto grid


def eval_streak(out, step, key=GATE_METRIC):
    """((previous, current) values of `key`, None) for the evaluation AT `step` and the one at the
    immediately preceding distinct step -- or (None, why). Fails closed (pass-4 P4-03): no row at
    the current step, more than one row at either step, a missing / non-finite value, or the two rows
    from different eval schemas all break the streak."""
    from rvsm import train as TR
    rows = [r for r in tail_jsonl(os.path.join(str(out), "logs", "eval.jsonl"), 200)
            if isinstance(r.get("step"), (int, float)) and int(r["step"]) <= int(step)]
    cur = [r for r in rows if int(r["step"]) == int(step)]
    if len(cur) != 1:
        return None, ("no evaluation row at this step" if not cur else "duplicate rows at this step")
    before = sorted({int(r["step"]) for r in rows if int(r["step"]) < int(step)})
    if not before:
        return None, "no earlier evaluation"
    prev = [r for r in rows if int(r["step"]) == before[-1]]
    if len(prev) != 1:
        return None, "duplicate rows at the previous step"
    a, b = prev[0], cur[0]
    if a.get("eval_schema") != TR.EVAL_SCHEMA or b.get("eval_schema") != TR.EVAL_SCHEMA:
        return None, "eval schema differs from this code's"
    va, vb = a.get(key), b.get(key)
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in (va, vb)):
        return None, f"{key} missing or non-finite"
    return (float(va), float(vb)), None


def eval_dice(out, step):
    """The headline (fine-rung, voxel-weighted) `dice` of the evaluation at `step` from
    `logs/eval.jsonl`, or None when there is none."""
    for r in reversed(tail_jsonl(os.path.join(str(out), "logs", "eval.jsonl"), 20)):
        if int(r.get("step", -1)) == int(step) and isinstance(r.get("dice"), (int, float)):
            return float(r["dice"])
    return None


def verso_gate(cfg, out, step, rows_fn=None, screen=None, r2=None):
    """Has round 0's recto earned the flipped-sign verso passes?

    Pass when the pooled dice over the held-out regions is at least `verso_gate_dice` AND the betti0
    error is no worse than the reference-against-itself baseline by more than the bootstrap CI's width
    -- or at `verso_after_steps`, the fallback the plan gives the gate so a run cannot stall on a
    held-out comparison, PROVIDED the evaluation's fine-rung dice (`screen`) has reached
    `verso_min_dice`: a verso pass from a student that cannot yet find the recto writes garbage verso
    labels (paris4: dice 0.15 -> 0.07 over steps 4000-8000, the fallback due at 10000). Without an
    evaluation at that step the fallback waits too.

    `screen` is the evaluation's own fine-rung dice at this step (`eval_dice`): the validation grid is
    tiles of the SAME held-out regions scored against the SAME round-0 stores, so while it is more than
    `GATE_SCREEN` below `verso_gate_dice` the held-out pass cannot pass and is not paid for. That pass
    is two student region passes plus two streamed `compare_stores` in the trainer, ~13 min on tnr-0
    (paris4 step 2000: dice 0.10 against a 0.6 gate)."""
    floor = float(getattr(cfg, "verso_min_dice", 0.0))
    if int(step) >= int(cfg.verso_after_steps):
        # the fallback reads the RUNG-2 self-cascade dice (`r2`: the last two evaluations, oldest
        # first) -- verso is written at rung 2 -- and needs it at verso_min_dice on TWO consecutive
        # evaluations, so one lucky evaluation cannot start the verso passes
        vals = [float(v) for v in (r2 if r2 is not None else ([] if screen is None else [screen]))]
        ok = len(vals) >= 2 and all(np.isfinite(v) and v >= floor for v in vals[-2:])
        rec = {"step": int(step), f"{GATE_METRIC}_prev": vals[-2] if len(vals) >= 2 else None,
               GATE_METRIC: vals[-1] if vals else None, "verso_min_dice": floor, "eval_dice": screen}
        if not ok:
            return False, {"why": "verso_after_steps reached, but the rung-2 recto dice is not at "
                                  "verso_min_dice on two consecutive evaluations: waiting", **rec}
        return True, {"why": "verso_after_steps", **rec}
    need = float(getattr(cfg, "verso_gate_dice", 0.6))
    if screen is not None and np.isfinite(screen) and float(screen) < need - GATE_SCREEN:
        return False, {"why": "eval dice below the gate", "eval_dice": float(screen),
                       "screen": need - GATE_SCREEN}
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


def plateau(out, key="dice", min_points=6, gain=ROUND_GAIN, since=None):
    """Is the eval curve flat? `fit_curve` on `logs/eval.jsonl`: the remaining gain to the fitted
    asymptote, as a fraction of it. Too few points is NOT a plateau. `since`: only the evaluations
    after that step -- this round's, not the previous rounds' curve."""
    from rvsm import evalsurf as EV
    recs = [(int(r["step"]), float(r[key])) for r in
            tail_jsonl(os.path.join(str(out), "logs", "eval.jsonl"), 500)
            if isinstance(r.get(key), (int, float)) and np.isfinite(r.get(key))
            and (since is None or int(r["step"]) > int(since))]
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


def maybe_regen_verso(cfg, out, step):
    """ONE regeneration of round 0's verso: the first time (verso on, round 0) the rung-2 dice clears
    `verso_min_dice + verso_regen_gain`, `state["verso_regen"]` records the step, and the producer
    rewrites every verso store made by an earlier checkpoint as a NEW generation (never in place;
    `verso_needs_regen`). Returns True when it fires; never fires twice."""
    st = read_state(out)
    if not bool(getattr(cfg, "verso_regen", True)):
        return False
    if int(st.get("round", 0)) != 0 or not st.get("verso_on") or st.get("verso_regen"):
        return False
    pair_, _why = eval_streak(out, step)          # the EXACT current record, never an older one
    need = float(cfg.verso_min_dice) + float(getattr(cfg, "verso_regen_gain", 0.15))
    if not pair_ or pair_[1] < need:
        return False
    # FREEZE the qualifying checkpoint (a hard link: the live student.pt is replaced by rename, the
    # link keeps these bytes) and its hash: the whole regeneration runs from this one network
    import hashlib
    src = os.path.join(str(out), "ckpt", "student.pt")
    frozen = os.path.join(str(out), "ckpt", "verso_regen.pt")
    if os.path.exists(frozen):
        os.remove(frozen)
    try:
        os.link(src, frozen)
    except OSError:
        shutil.copyfile(src, frozen)
    h = hashlib.sha256()
    with open(frozen, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    backlog = regen_backlog(out, int(step))
    write_json_atomic(backlog_path(out), {"step": int(step), "regions": [list(lo) for lo in backlog]})
    write_state(out, verso_regen={"step": int(step), GATE_METRIC: float(pair_[1]), "ckpt": frozen,
                                  "sha256": h.hexdigest(), "backlog": len(backlog)})
    jlog(out, "sched", {"kind": "verso_regen", "step": int(step), GATE_METRIC: float(pair_[1]),
                        "need": need, "ckpt_sha256": h.hexdigest(), "backlog": len(backlog)})
    return True


def backlog_path(out):
    return os.path.join(str(out), "stores", "round_0", "bundle", "regen_backlog.json")


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_json(path, obj)


def regen_backlog(out, step):
    """Every region whose finished generation-0 verso was made by a checkpoint older than `step`: the
    regeneration's whole work list, persisted, and worked through independently of the trainer's
    lookahead window (pass-4 P4-06)."""
    from rvsm import regions as RG, stores
    out_ = []
    for lo in RG.Catalog(out, 0).list_done("verso"):
        if stores.store_gen(out, "verso", lo, 0) != 0:
            continue
        made = stores.read_attrs(stores.store_path(out, "verso", lo, 0)).get("step")
        if made is None or int(made) < int(step):
            out_.append(tuple(int(v) for v in lo))
    return out_


def regen_remaining(out):
    """The backlog regions whose regenerated bundle is not committed yet ([] = the regeneration is
    complete; None = there is no regeneration)."""
    from rvsm import stores
    b = _read_json(backlog_path(out))
    if not b:
        return None
    return [tuple(lo) for lo in b.get("regions", []) if stores.bundle_state(out, tuple(lo), 0)["verso"] < 1]


def _complete_rows(rows, keys=("precision", "betti0_err")):
    """Only the rows that carry every metric, finite: a row missing one is dropped WHOLE, never
    metric by metric (that would pool different region sets per metric)."""
    out = []
    for r in rows:
        try:
            if all(np.isfinite(float(r[k])) for k in keys):
                out.append(r)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def round_gate(cfg, out, step, round_, rows_fn=None, ref=None, round_start=0, verso_on=True,
               verso_on_step=None, verso_regions=None):
    """Should round `round_` end and round `round_ + 1` open? Every doubt answers NO (fail closed).

    In order, each one a refusal with its reason logged:
      - round 0 without `verso_on`: self-distillation needs round 0's verso stores to exist; nor
        before verso has been on for `round_min_steps_after_verso` steps (state `verso_on_step`) and
        `verso_min_regions` verso stores are finished;
      - fewer than `round_steps` steps IN THIS ROUND (`step - round_start`, not the absolute step:
        the absolute one made round 1 due the moment round 0 was over);
      - less than `round_steps` of the global `steps` budget left: a promoted round must get to train;
      - no COMPLETE held-out comparison rows (a row missing a metric is dropped whole), or a
        non-finite pooled precision / betti0 error;
      - round >= 1 with no round-0-anchored reference `ref` (it is persisted in state.json, so a
        restart keeps the veto), or a merge side (`precision`) / `betti0_err` worse than `ref` beyond
        the bootstrap CI.
    Round 0 has nothing earlier to be worse than: its measured rows become `ref`. The plateau fit
    (this round's evaluations only) is reported beside the decision. A round that fails is not
    discarded wholesale: it simply keeps training, and WSD makes that extension free (plan §1).

    Returns (fire, why); `why["rows"]` holds the measured stats, the next round's reference."""
    in_round = int(step) - int(round_start or 0)
    base = {"round_step": in_round, "round_start": int(round_start or 0)}
    if int(round_) == 0 and not verso_on:
        return False, {"why": "verso_on is false: round 0's verso stores must exist first", **base}
    if in_round < int(cfg.round_steps):
        return False, {"why": "round_step below round_steps", "round_steps": int(cfg.round_steps),
                       **base}
    if int(cfg.steps) - int(step) < int(cfg.round_steps):
        return False, {"why": "less than round_steps of the global steps budget left for the next "
                              "round", "steps": int(cfg.steps), **base}
    if int(round_) == 0:
        after = None if verso_on_step is None else int(step) - int(verso_on_step)
        need = int(getattr(cfg, "round_min_steps_after_verso", 0))
        if after is None or after < need:
            return False, {"why": "verso has not been on for round_min_steps_after_verso",
                           "steps_since_verso_on": after, "need": need, **base}
        nreg = int(getattr(cfg, "verso_min_regions", 0))
        if verso_regions is None or int(verso_regions) < nreg:
            return False, {"why": "fewer than verso_min_regions finished verso stores",
                           "verso_regions": verso_regions, "need": nreg, **base}
    flat, pwhy = plateau(out, "dice", since=round_start)
    base.update({"plateau": bool(flat), "plateau_why": pwhy.get("why")})
    rows = _complete_rows(list((rows_fn() if rows_fn is not None else None) or []))
    if not rows:
        return False, {"why": "no complete held-out comparison rows (fail closed)", **base}
    prec, plo, phi = _boot([r.get("precision", float("nan")) for r in rows])
    b, blo, bhi = _boot([r.get("betti0_err", float("nan")) for r in rows])
    got = {"precision": prec, "betti0_err": b}
    if not (np.isfinite(prec) and np.isfinite(b)):
        return False, {"why": "non-finite held-out metrics (fail closed)", "rows": got, **base}
    if int(round_) >= 1:
        if not ref or not all(np.isfinite(float(ref.get(k, float("nan"))))
                              for k in ("precision", "betti0_err")):
            return False, {"why": "no round-0 reference to compare with (fail closed)", "rows": got,
                           **base}
        ok = b <= float(ref["betti0_err"]) + max(bhi - blo, 0.0) and \
            prec >= float(ref["precision"]) - max(phi - plo, 0.0)
        return bool(ok), {"why": "quality vs reference", "precision": prec, "betti0_err": b,
                          "ref": ref, "ok": bool(ok), "rows": got, "n": len(rows), **base}
    return True, {"why": "round 0: verso on, round_steps done, held-out rows measured",
                  "precision": prec, "betti0_err": b, "rows": got, "n": len(rows), **base}


# --------------------------------------------------------------------------- #
# the trainer side: the walk cursor and the lookahead
# --------------------------------------------------------------------------- #
def cursor_dir(out):
    return os.path.join(str(out), "logs", "cursor")


def round_transition(out, nxt, quiesce=None, **state):
    """Move the run to round `nxt`, in the one order that cannot let the old round's walk leak into the
    new one:

    1. state.json gets the new round first. From here every old-round walk stops by itself (its wait
       loop and its draw loop check the round), its `_publish` refuses to write, and every cursor
       reader rejects a record stamped with the old round.
    2. `quiesce()` (the trainer's `train.DevicePrefetch.close`): the background H2D fetch is waited for
       (bounded) and the loader's persistent workers are shut down -- a worker still in the middle of a
       draw is terminated, not left to publish later.
    3. Only then is the cursor directory reset: the next round walks from the start again (the old
       round's cursor would put the producer's window past the positions the new sampler asks for
       first -- a deadlock the end-to-end test hit whenever those regions fell outside the old window).
    """
    write_state(out, round=int(nxt), cursor=0, walk=None, **state)
    if quiesce is not None:
        quiesce()
    shutil.rmtree(cursor_dir(out), ignore_errors=True)
    shutil.rmtree(ack_dir(out), ignore_errors=True)


def acknowledge_leases(out, cache, keys, clock, state, fetch_kw):
    """One pass of the producer's lease keeper (review D04 / P3-02). Register every worker's lease with
    the cache; for each NEW lease (worker, lease_id), resolve it -- (re-)fetch a leased home this
    producer does not hold or that lost shards (a rung 3-6 revisit, an eviction, a restart) -- and
    ACKNOWLEDGE it with the homes that are protected and fully on disk. A worker draws nothing before
    that ack (`WalkPatches._await_ack`), and every eviction re-reads the leases itself
    (`ShardCache.lease_source`), so nothing leased at decision time is taken. Overlapping leases are a
    union: a home stays protected while ANY worker's current lease names it, and a lease is released
    by that worker's next publish. `state` is the keeper's memory: worker -> (lease_id, complete, t)."""
    import numpy as np

    from rvsm import stream
    recs = cursor_records(out)
    leased = sorted({tuple(int(v) for v in lo) for r in recs for lo in r.get("lease") or ()})
    with clock:
        cache.lease(cache.region_key(lo) for lo in leased)
    acked = []
    for r in recs:
        lid, w = r.get("lease_id"), r.get("worker")
        if lid is None or w is None:
            continue
        last = state.get(w)
        if last is not None and last[0] == lid and (last[1] or time.time() - last[2] < LEASE_RETRY_S):
            continue
        homes = [tuple(int(v) for v in lo) for lo in r.get("lease") or ()]
        with clock:
            todo = [lo for lo in homes if lo not in keys or cache.missing(cache.region_key(lo))]
        for lo in todo:
            # the lock is held only for the books: a download (minutes behind a slow origin) must not
            # hold up the reader, whose next unit waits on the same lock (`fetch_region_outside`)
            try:
                cache.fetch_region_outside(np.array(lo, np.int64), clock, evict=False,
                                           on_booked=lambda k, lo=lo: keys.__setitem__(lo, k), **fetch_kw)
            except stream.FetchFailed:              # not ready; retried after LEASE_RETRY_S
                continue
        with clock:
            ready = [list(lo) for lo in homes if not cache.missing(cache.region_key(lo))]
            cache.evict()
        write_lease_ack(out, w, lid, ready, r.get("round"))
        state[w] = (lid, len(ready) == len(homes), time.time())
        acked.append((w, lid))
        if len(ready) < len(homes):
            jlog(out, "produce", {"kind": "lease_incomplete", "worker": w, "lease_id": lid,
                                  "missing": [list(lo) for lo in homes if list(lo) not in ready]},
                 echo=False)
    return acked


def ack_dir(out):
    return os.path.join(str(out), "logs", "lease_ack")


def write_lease_ack(out, worker, lease_id, ready, round_=None):
    """The producer's acknowledgement of one worker's lease: `ready` lists the leased homes that are
    protected from eviction and fully on disk."""
    return _write_json(os.path.join(ack_dir(out), f"w{int(worker)}.json"),
                       {"worker": int(worker), "lease_id": str(lease_id), "round": round_,
                        "ready": [list(v) for v in ready], "t": time.time()})


def read_lease_ack(out, worker):
    return _read_json(os.path.join(ack_dir(out), f"w{int(worker)}.json")) or {}


_CURSOR_CACHE = {}      # path -> ((mtime_ns, size), record): leases are read before every eviction


def _cursor_read(path):
    try:
        st = os.stat(path)
    except OSError:
        _CURSOR_CACHE.pop(path, None)
        return None
    sig = (st.st_mtime_ns, st.st_size, st.st_ino)
    hit = _CURSOR_CACHE.get(path)
    if hit is not None and hit[0] == sig:
        return hit[1]
    r = _read_json(path)
    _CURSOR_CACHE[path] = (sig, r)
    return r


def cursor_records(out, round_=None):
    """The sampler workers' cursor files of the CURRENT round (`round_`, default state.json's): a record
    stamped with another round is a stale writer -- an old-round worker or prefetch that published after
    the round transition reset the directory -- and is ignored. An unstamped record (a run older than
    the stamp) is taken as it is. A file is re-parsed only when its stat changes."""
    if round_ is None:
        round_ = read_state(out).get("round")
    d = cursor_dir(out)
    out_ = []
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not n.endswith(".json"):
            continue
        r = _cursor_read(os.path.join(d, n))
        if not r or not isinstance(r.get("pos"), int):
            continue
        if round_ is not None and r.get("round") is not None and int(r["round"]) != int(round_):
            continue
        out_.append(r)
    return out_


def cursor_leases(out, round_=None):
    """The union of the workers' leases: the rung-2 region corners (tuples) whose CT they are reading or
    are about to read (`WalkPatches._publish`). A stale round's leases are not the trainer's."""
    got = set()
    for r in cursor_records(out, round_):
        for lo in r.get("lease") or ():
            try:
                got.add(tuple(int(v) for v in lo))
            except (TypeError, ValueError):
                continue
    return sorted(got)


def read_cursor(out, round_=None):
    """The walk position of the SLOWEST sampler worker: the producer must stay ahead of that one."""
    vals = [int(r["pos"]) * max(int(r.get("stride", 1)), 1) for r in cursor_records(out, round_)]
    return min(vals) if vals else 0


def walk_snapshot(out, round_=None):
    """The sampler workers' walk positions, from their cursor files: `{"stride", "round", "workers":
    {w: {"pos", "done", "pass"}}}`, or None when there are none. Written into state.json at every
    checkpoint so a resume continues the walk where the checkpoint left it (`WalkPatches(start=)`)."""
    ws, stride = {}, None
    for r in cursor_records(out, round_):
        stride = int(r.get("stride", 1))
        ws[str(int(r.get("worker", 0)))] = {"pos": int(r["pos"]), "done": list(r.get("done") or []),
                                            "pass": int(r.get("pass", 0))}
    if not ws:
        return None
    return {"stride": stride, "round": None if round_ is None else int(round_), "workers": ws}


def checkpoint_state(out, step, round_):
    """state.json for the checkpoint just written at `step`: the step, the slowest worker's cursor and
    the walk snapshot a resume starts from (`resume_walk`). Written at EVERY checkpoint boundary --
    evaluation, `ckpt_every`, the RAM guard's -- after the checkpoint itself, so the walk it records is
    at or past what that checkpoint trained on and a resume never repeats a visit."""
    return write_state(out, step=int(step), round=int(round_), cursor=read_cursor(out, round_),
                       region_s=region_seconds(out, round_), walk=walk_snapshot(out, round_),
                       verso_on=bool(read_state(out).get("verso_on", False)))


def resume_walk(out, round_):
    """The walk a resumed trainer starts from: the snapshot the last checkpoint wrote into state.json,
    else (a run older than the snapshot) the workers' own cursor files -- never position 0 when the
    run has been somewhere. None for a new round (its walk starts over) or a fresh run."""
    st = read_state(out)
    w = st.get("walk")
    if w and (w.get("round") is None or int(w["round"]) == int(round_)):
        return w
    if int(st.get("round", 0)) == int(round_):
        return walk_snapshot(out, round_)
    return None


def read_cursor_head(out, round_=None):
    """The walk position of the FASTEST sampler worker's next visit: worker w of W at its own position
    p is at `p * W + w` of the shared walk. The producer's window must reach past this one too."""
    vals = [int(r["pos"]) * max(int(r.get("stride", 1)), 1) + int(r.get("worker", 0))
            for r in cursor_records(out, round_)]
    return max(vals) if vals else 0


def region_seconds(out, round_=None):
    """Seconds per consumed walk entry, averaged over the workers (T_train of the lookahead rule)."""
    vals = [float(r["region_s"]) for r in cursor_records(out, round_)
            if isinstance(r.get("region_s"), (int, float)) and r["region_s"] > 0]
    return float(np.mean(vals)) if vals else 0.0


# The trainer's sampler is a `sample.Patches` subclass that lives in `rvsm/walk.py`, so this module
# imports neither torch nor sample at the top level (the spawned producer must set CUDA_VISIBLE_DEVICES
# before torch is imported at all). It must be a MODULE-level class: the loader's forkserver workers
# unpickle it by name, and a class defined inside a function cannot be (it was, and `rvsm run` with
# workers > 0 died on "Can't get local object 'walk_patches.<locals>._WalkPatches'").


def walk_patches():
    """The trainer's sampler: `sample.Patches`' walk with the lookahead rule on top.

    `sample.Patches` steps along the walk and skips a region whose stores are not finished. That is
    almost the rule the plan asks for, but not quite: the trainer must take the FIRST READY region
    inside a window of `L` visits (so it never runs ahead of the producer into an unproduced tail), a
    region whose verso lands after its visit earns ONE revisit with fresh windows, and the position
    reached has to be published, so the producer knows what to prepare and what to release.

    It is a thin subclass rather than a rewrite: the drawing, the targets, the context cubes and the
    planes are all `Patches`', and only `__iter__` differs."""
    from rvsm import walk
    return walk.WalkPatches


# --------------------------------------------------------------------------- #
# the supervisor
# --------------------------------------------------------------------------- #
def _meta_source(ct):
    """The path / URL `scanmeta.fetch` would read metadata.json from (a level path -> its group)."""
    p = str(ct or "").rstrip("/")
    if p and not p.endswith(".json"):
        head, _, last = p.rpartition("/")
        if head and last.isdigit():
            p = head
    return p


def frozen_meta(out, ct, log=print):
    """(metadata dict, the five conditioning values) of the run, FROZEN on the first successful setup.

    A resume loads `<out>/metadata.json` and `<out>/meta5.json` and never rewrites or refetches them
    (no network call): the scan planes are a network input, and a resume that fetched them again -- or
    fell back to the defaults because the fetch failed that day -- would silently change what the
    checkpoint was trained on.

    On a fresh run the fetch has three outcomes (`scanmeta.probe`): read -> frozen as is; DEFINITELY
    ABSENT (HTTP 404/403, no local file) -> the documented defaults, `absent: true` recorded, the five
    planes ZERO; anything else (timeout, 5xx, unparsable) -> an ERROR, never the defaults. Either way
    the startup log says which planes are zero because the metadata is absent."""
    from rvsm import scanmeta as SM
    mp, m5 = os.path.join(str(out), "metadata.json"), os.path.join(str(out), "meta5.json")
    meta, meta5 = _read_json(mp), _read_json(m5)
    if meta and meta5 is not None:
        meta5 = [float(v) for v in meta5]
    else:
        raw, status = SM.probe(_meta_source(ct))
        if status == "error":
            raise SystemExit(f"rvsm run: could not read metadata.json beside {ct} (transport error); "
                             f"refusing to freeze defaults on a fresh run -- retry when it is reachable")
        meta = SM.flatten(raw)
        meta["absent"] = status == "absent"
        meta5 = [float(v) for v in SM.scan_planes(meta)]
        _write_json(mp, meta)
        _write_json(m5, meta5)
    if meta.get("missing") or meta.get("absent"):
        names = [r[0] for r in SM.META_RANGE]
        log(f"[setup] scan metadata ABSENT for {ct}: the planes {names} are all zero (frozen in {mp})")
    return meta, meta5


def log_switches(out, old, cfg):
    """The deliberate mid-run changes a resume makes, as `sched` lines against the previous
    config.json (`old`): `teacher_switch` when the round-0 teacher set (the keys of `teacher_ckpts`)
    or a teacher's weights moved, `loss_switch` naming every loss weight of `config.LOSS_SWITCH_FIELDS`
    that moved (the trainer uses the live config's weights), `precision_switch` when `gn_bf16` moved (a
    config.json that predates the field counts as False). Returns the lines logged."""
    d = (old or {}).get("config") or {}
    step = int(read_state(out).get("step", 0) or 0)
    got = []
    oc = dict(d.get("teacher_ckpts") or {})
    nc = dict(cfg.teacher_ckpts or {})
    # the per-rung route (a config.json that predates the field had none)
    orr = {str(k): str(v) for k, v in dict(d.get("teacher_route") or {}).items()}
    nrr = {str(k): str(v) for k, v in dict(cfg.teacher_route or {}).items()}
    if ("teacher_ckpts" in d and oc != nc) or orr != nrr:
        on = [str(n) for n in oc] or list(DEFAULT_TEACHERS)
        try:
            oset = teacher_set(CFG.Config(teacher_ckpts=oc, teacher_route=orr))
        except ValueError:
            oset = on
        rec = {"kind": "teacher_switch", "step": step, "old": on, "new": teacher_names(cfg),
               "old_ckpts": oc, "new_ckpts": nc,
               "regenerate": sorted(oset) != sorted(teacher_set(cfg))}
        if orr or nrr:
            rec.update({"old_route": orr, "new_route": nrr, "old_set": oset, "new_set": teacher_set(cfg)})
        jlog(out, "sched", rec)
        got.append(rec)
    moved = {}
    for k in CFG.LOSS_SWITCH_FIELDS:
        # a config.json that predates a weight ran it at its default (`loss_skel_prec` is new and 0)
        ov = float(d[k]) if k in d else (float(getattr(CFG.Config(), k)) if d else None)
        if ov is not None and ov != float(getattr(cfg, k)):
            moved[k] = {"old": ov, "new": float(getattr(cfg, k))}
    if moved:
        rec = {"kind": "loss_switch", "step": step, "weights": moved}
        jlog(out, "sched", rec)
        got.append(rec)
    og = bool(CFG._coerce("gn_bf16", d.get("gn_bf16", False)))
    if og != bool(cfg.gn_bf16):
        rec = {"kind": "precision_switch", "step": step, "gn_bf16": {"old": og, "new": bool(cfg.gn_bf16)}}
        jlog(out, "sched", rec)
        got.append(rec)
    return got


def setup(cfg, out=None):
    """Freeze what the run is, once: config.json (asserted on a resume), umbilicus.json, metadata.json,
    the CT mirror and the held-out set. Returns the context both halves need."""
    from rvsm import axis as AX, ladder, regions as RG, scanmeta as SM, stream
    out = str(out or cfg.out)
    r = CFG.route_spec(cfg)                    # a malformed teacher_route fails here, not in the producer
    if r is not None and not {r.fine, r.base} <= set(teacher_names(cfg)):
        raise SystemExit(f"rvsm run: teacher_route {dict(cfg.teacher_route)} needs both {r.fine!r} and "
                         f"{r.base!r} in teacher_ckpts (the teacher set is {teacher_names(cfg)})")
    for d in ("logs", "ckpt", "stores", "eval", "workers"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    cp = os.path.join(out, "config.json")
    old = _read_json(cp)
    if old:
        got = CFG.stored_fingerprint(old)
        assert got == cfg.fingerprint(), (
            f"resume: {cp} was written by a config whose fingerprint is {got}, this run's is "
            f"{cfg.fingerprint()}. Everything but {CFG.FINGERPRINT_EXCLUDE} must match.")
        log_switches(out, old, cfg)
    _write_json(cp, cfg.to_json())

    AX.ensure(out, cfg.umbilicus, ct=cfg.ct)
    meta, meta5 = frozen_meta(out, cfg.ct)

    mirror = stream.ShardCache(cfg.ct, out, budget_gb=cfg.cache_gb, seed=cfg.ct_seed or None)
    try:
        # the coarse levels WHOLE, not just the metadata: the occupancy below reads one of them, and a
        # remote level whose shards are not here yet reads as air -- a fresh run on a URL found 0 regions
        mirror.pin_small_levels()
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


def host_mem():
    """(MemTotal, MemAvailable) in bytes, from /proc/meminfo; (0, 0) where there is none."""
    got = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k in ("MemTotal", "MemAvailable"):
                    got[k] = int(v.split()[0]) * 1024
    except OSError:
        return 0, 0
    return got.get("MemTotal", 0), got.get("MemAvailable", 0)


def tree_rss(pid=None):
    """(total RSS in bytes, number of processes) of `pid` (default: this process) and every descendant
    -- the supervisor/trainer, its loader workers, the producer and the producer's fields pool."""
    pid = int(pid or os.getpid())
    kids, rss = {}, {}
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    try:
        names = os.listdir("/proc")
    except OSError:
        return 0, 0
    for d in names:
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                st = f.read()
            ppid = int(st[st.rindex(")") + 2:].split()[1])
            with open(f"/proc/{d}/statm") as f:
                rss[int(d)] = int(f.read().split()[1]) * page
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    tot, n, todo, seen = 0, 0, [pid], set()
    while todo:
        q = todo.pop()
        if q in seen:
            continue
        seen.add(q)
        tot, n = tot + rss.get(q, 0), n + 1
        todo += kids.get(q, [])
    return tot, n


def producer_paused(out):
    return os.path.exists(os.path.join(str(out), PAUSE_FILE))


def ram_guard(out, pid=None, mem=None, rss=None, log=print):
    """One look at the host's memory: pause the producer above `RAM_PAUSE_FRAC` of MemTotal, resume it
    below `RAM_RESUME_FRAC`. Pressure is the larger of the host's memory in use (MemTotal -
    MemAvailable) and this run's process-tree RSS. Returns the numbers, for the heartbeat's record.

    The kernel OOM on a 64 GB host with no swap takes the whole box down (paris4, step 2000); a paused
    producer finishes its unit in flight and starts no other, which is the one lever the supervisor
    has that costs nothing but time."""
    total, avail = mem if mem is not None else host_mem()
    r, n = rss if rss is not None else tree_rss(pid)
    if not total:
        return {}
    used = total - avail
    frac = max(used, r) / total
    rec = {"rss_gb": round(r / 2 ** 30, 2), "procs": n, "host_used_gb": round(used / 2 ** 30, 2),
           "host_total_gb": round(total / 2 ** 30, 2), "mem_frac": round(frac, 3)}
    p = os.path.join(str(out), PAUSE_FILE)
    if frac >= RAM_PAUSE_FRAC and not os.path.exists(p):
        with open(p, "w") as f:
            f.write(json.dumps({"t": time.time(), **rec}))
        jlog(out, "sched", {"kind": "ram_guard", "action": "pause_producer", **rec}, echo=False)
        log(f"!!!! [ram_guard] HOST RAM {frac:.0%} of {rec['host_total_gb']} GB (run tree RSS "
            f"{rec['rss_gb']} GB in {n} procs, host used {rec['host_used_gb']} GB): PRODUCER PAUSED "
            f"until < {RAM_RESUME_FRAC:.0%} !!!!")
        rec["action"] = "pause"
    elif frac < RAM_RESUME_FRAC and os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass
        jlog(out, "sched", {"kind": "ram_guard", "action": "resume_producer", **rec}, echo=False)
        log(f"[ram_guard] host RAM back to {frac:.0%}: producer resumed")
        rec["action"] = "resume"
    return rec


def hb_ticker(tick, stop_ev, every=None):
    """The producer's stamping thread: `tick()` every `HB_TICK_S` until `stop_ev`. A unit may take far
    longer than the supervisor's silence limit (a first TensorRT compile, a 1024^3 teacher pass on a
    slow card); the loop stamps only between units, so without this a healthy producer was restarted."""
    every = HB_TICK_S if every is None else float(every)
    while not stop_ev.wait(every):
        try:
            tick()
        except Exception:  # noqa: BLE001  -- a full disk must not kill the thread that proves liveness
            pass


def proc_tree(pid):
    """[(pid, start time)] of every descendant of `pid` (not `pid` itself), from /proc. The start time
    makes a later kill safe against pid reuse."""
    kids, start = {}, {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return []
    for d in names:
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                st = f.read()
            rest = st[st.rindex(")") + 2:].split()
            kids.setdefault(int(rest[1]), []).append(int(d))
            start[int(d)] = int(rest[19])
        except (OSError, ValueError, IndexError):
            continue
    out, todo, seen = [], list(kids.get(int(pid), [])), set()
    while todo:
        q = todo.pop()
        if q in seen:
            continue
        seen.add(q)
        out.append((q, start.get(q)))
        todo += kids.get(q, [])
    return out


def own_process_group():
    """Make this process the leader of a new process group (the spawned producer's first act). Its
    forkserver and every fields-pool worker inherit the group, whoever their parent is, so the supervisor
    can remove all of them with one `killpg` -- PDEATHSIG alone follows the forkserver, not the
    producer (review P3-09). Returns the group id."""
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    return os.getpgid(0)


def kill_group(pgid, sig=None):
    """SIGKILL the process group `pgid` -- never the caller's own group, never 0/1. True if sent."""
    import signal
    sig = signal.SIGKILL if sig is None else sig
    try:
        pgid = int(pgid)
        if pgid <= 1 or pgid == os.getpgid(0):
            return False
        os.killpg(pgid, sig)
        return True
    except (OSError, TypeError, ValueError):
        return False


def kill_tree(tree, sig=None):
    """SIGKILL every (pid, start time) of `proc_tree` that is still that same process."""
    import signal
    sig = signal.SIGKILL if sig is None else sig
    n = 0
    for pid, t0 in tree:
        try:
            with open(f"/proc/{pid}/stat") as f:
                st = f.read()
            if int(st[st.rindex(")") + 2:].split()[19]) != t0:
                continue                 # the pid now belongs to someone else
            os.kill(pid, sig)
            n += 1
        except (OSError, ValueError, IndexError):
            continue
    return n


class ProducerWatch:
    """The supervisor's producer supervision, one `check()` per tick (review O10).

    LIVENESS is the process itself: `is_alive()` / `exitcode`. A producer that has died is restarted at
    the next tick, not after `SILENT_MAX_S` of silence (an immediate crash stalled a run for half an
    hour). SILENCE -- alive but no heartbeat for `SILENT_MAX_S`, which the producer's own stamping
    thread makes a frozen process rather than a long unit -- gets terminate, then kill, and its whole
    process tree (the fields pool) is killed with it. A new producer is spawned only once the old one
    has EXITED: one that survives even SIGKILL (a D-state hang) is left alone and reported every tick,
    never duplicated, and a producer THREAD (cpu mode) cannot be stopped at all, so a silent one is
    reported, not restarted. Restarts back off: the first is immediate, the next waits `RESTART_MIN_S`,
    doubling to `RESTART_MAX_S`; a producer that lives `RESTART_RESET_S` resets that, and
    `RESTART_FATAL` restarts in a row are shouted in the log."""

    def __init__(self, out, procs, respawn, log=print, clock=time.time, join_s=30.0,
                 unit_stall_s=None):
        self.out, self.procs, self.respawn, self.log = str(out), procs, respawn, log
        self.clock, self.join_s = clock, float(join_s)
        self.unit_stall_s = float(UNIT_STALL_S if unit_stall_s is None else unit_stall_s)
        self._alerted = None            # the progress stamp of the stall already reported
        self.fails, self.next_at, self.born = 0, 0.0, clock()
        self.restarts = 0

    def _hb(self):
        return _read_json(os.path.join(self.out, "workers", "produce.json")) or {}

    def _kill_group(self, pr):
        """Remove the producer's process group (forkserver, fields pool, anything it started): the
        group it recorded in its heartbeat when that heartbeat is this process's, else the one it leads
        (`own_process_group` makes the pgid its pid). A producer THREAD has no group of its own."""
        pid = getattr(pr, "pid", None)
        if pid is None:
            return False
        hb = self._hb()
        pgid = hb.get("pgid") if hb.get("pid") == pid and hb.get("pgid") else pid
        sent = kill_group(pgid)
        if sent:
            jlog(self.out, "sched", {"kind": "producer_group_killed", "pgid": int(pgid), "pid": pid},
                 echo=False)
        return sent

    def _paused(self, hb):
        """A deliberate pause is not a stall: the host-RAM guard's, or a one-card timeshare turn."""
        if producer_paused(self.out) or str(hb.get("phase", "")) == "paused_ram":
            return True
        return os.path.exists(os.path.join(self.out, PHASE_FILE)) and read_phase(self.out) != "produce"

    def _stall(self, hb, now):
        """None, "stall" (reported: no progress for `unit_stall_s`, the heartbeat itself fresh -- the
        ticker only proves the process is alive, review P3-10) or "stalled" (twice that: restart it)."""
        prog = hb.get("progress_ts")
        if prog is None or self._paused(hb):
            return None
        page = now - float(prog)
        if page <= self.unit_stall_s:
            return None
        if self._alerted != prog:
            self._alerted = prog
            unit = {k: hb.get(k) for k in ("phase", "job", "region", "round", "cursor")
                    if hb.get(k) is not None}
            jlog(self.out, "sched", {"kind": "STALL", "progress_age_s": round(page), "unit": unit,
                                     "pid": hb.get("pid"), "limit_s": self.unit_stall_s}, echo=False)
            self.log(f"!!!! [supervisor] STALL: the producer is alive but its unit {unit} has made no "
                     f"progress for {page / 60:.0f} min (restart at {2 * self.unit_stall_s / 60:.0f}) !!!!")
        return "stalled" if page > 2 * self.unit_stall_s else "stall"

    def _recycled(self, pr):
        """Did THIS producer exit as a recycle (`producer_recycle_*`)? Its last stamp says `recycle` under
        its own pid (a thread's pid is ours) and a process exited 0."""
        hb = self._hb()
        if not hb.get("recycle"):
            return False
        pid = getattr(pr, "pid", None)
        if pid is None:                         # a producer thread (cpu mode) stamps our pid
            return hb.get("pid") == os.getpid()
        return hb.get("pid") == pid and getattr(pr, "exitcode", None) == 0

    def _respawn_recycled(self, pr, now):
        """A recycle is a HEALTHY exit: respawned at once, whatever the backoff says, and the backoff
        (`fails`, `next_at`) is left exactly as it was -- never grown, never counted toward FATAL."""
        hb = self._hb()
        self.restarts += 1
        jlog(self.out, "sched", {"kind": "restart", "reason": "recycle", "recycle": True,
                                 "why": hb.get("reason"), "pid": getattr(pr, "pid", None),
                                 "restarts": self.restarts, "fails": self.fails}, echo=False)
        _write_json(os.path.join(self.out, "workers", "produce.json"),
                    {"pid": None, "phase": "spawning", "last_ts": now})
        self.procs["produce"] = self.respawn()
        # `born` is left too: a line of recycled producers is ONE healthy life for RESTART_RESET_S
        return "recycle"

    def check(self):
        """One look. Returns what happened: None (healthy / nothing to do), "stall" (reported),
        "restart", "recycle" (a clean recycle exit, respawned at once), "backoff", "stuck" (will not
        exit) or "silent_thread"."""
        pr = self.procs.get("produce")
        if pr is None or stop_requested(self.out):
            return None
        now = self.clock()
        if pr.is_alive():
            hb = self._hb()
            age = now - float(hb.get("last_ts") or now)
            stall = self._stall(hb, now) if age <= SILENT_MAX_S else None
            if age <= SILENT_MAX_S and stall != "stalled":
                if stall is None and self.fails and now - self.born > RESTART_RESET_S:
                    self.fails = 0
                return stall
            why = "producer silent" if stall is None else "producer stalled"
            if not hasattr(pr, "terminate"):
                jlog(self.out, "sched", {"kind": "producer_silent_thread", "age_s": round(age),
                                         "why": why})
                return "silent_thread"
            jlog(self.out, "sched", {"kind": "restart", "reason": why, "age_s": round(age),
                                     "pid": getattr(pr, "pid", None)})
            tree = proc_tree(pr.pid) if getattr(pr, "pid", None) else []
            for stop in ("terminate", "kill"):
                try:
                    getattr(pr, stop)()
                    pr.join(self.join_s)
                except Exception:  # noqa: BLE001
                    pass
                if not pr.is_alive():
                    break
            kill_tree(tree)
            self._kill_group(pr)
            if pr.is_alive():
                jlog(self.out, "sched", {"kind": "producer_stuck", "pid": getattr(pr, "pid", None)})
                self.log(f"!!!! [supervisor] producer pid {getattr(pr, 'pid', None)} is silent and will "
                         f"not exit: NOT starting another one !!!!")
                return "stuck"
            reason = "silent" if stall is None else "stalled"
        else:
            reason = f"exit {getattr(pr, 'exitcode', None)}"
            # it died on its own: its forkserver and fields pool may not have (P3-09)
            self._kill_group(pr)
            if self._recycled(pr):
                return self._respawn_recycled(pr, now)
        if now < self.next_at:
            return "backoff"
        self.fails += 1
        self.restarts += 1
        self.next_at = now + min(RESTART_MIN_S * 2 ** (self.fails - 1), RESTART_MAX_S)
        rec = {"kind": "restart", "reason": reason, "restarts": self.restarts, "fails": self.fails,
               "next_after_s": round(self.next_at - now, 1)}
        jlog(self.out, "sched", rec, echo=False)
        if self.fails >= RESTART_FATAL:
            jlog(self.out, "sched", {**rec, "kind": "producer_fatal"}, echo=False)
            self.log(f"!!!! [supervisor] FATAL: the producer has died {self.fails} times in a row "
                     f"({reason}); see logs/produce.jsonl -- still retrying every "
                     f"{min(RESTART_MIN_S * 2 ** (self.fails - 1), RESTART_MAX_S):.0f} s !!!!")
        _write_json(os.path.join(self.out, "workers", "produce.json"),
                    {"pid": None, "phase": "spawning", "last_ts": now})
        self.procs["produce"] = self.respawn()
        self.born = now
        return "restart"


def trainer_ram_verdict(trainer_gb, host_frac, limit_gb, host_max):
    """Why the TRAINER must stop now, or None: its own RSS past `limit_gb`, or the host's memory in use
    past `host_max` of MemTotal. The producer pause cannot help against a trainer that leaks (paris4
    step 24000: +280 MB a step, 6 -> 47.7 GB in 180 steps, the kernel killed everything)."""
    if limit_gb and float(trainer_gb) >= float(limit_gb):
        return f"trainer RSS {float(trainer_gb):.1f} GB >= ram_trainer_gb {float(limit_gb):.1f}"
    if host_max and float(host_frac) >= float(host_max):
        return f"host memory {float(host_frac):.0%} >= ram_host_exit {float(host_max):.0%}"
    return None


def own_rss_gb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except (OSError, ValueError):
        return 0.0


# glibc's mallopt parameters (malloc.h). Linux/glibc only: every call below is a no-op elsewhere
M_TRIM_THRESHOLD, M_MMAP_THRESHOLD, M_ARENA_MAX = -1, -3, -8
_LIBC = []        # [glibc CDLL or None], resolved once


def glibc():
    """The process's glibc through ctypes, or None (macOS, musl, Windows: no `malloc_trim`, and the
    mallopt parameter numbers above are glibc's own)."""
    if not _LIBC:
        lib = None
        try:
            import ctypes
            import platform
            if platform.system() == "Linux" and platform.libc_ver()[0] == "glibc":
                c = ctypes.CDLL("libc.so.6")
                if hasattr(c, "malloc_trim") and hasattr(c, "mallopt"):
                    lib = c
        except (OSError, AttributeError, ValueError):
            lib = None
        _LIBC.append(lib)
    return _LIBC[0]


def malloc_trim():
    """glibc `malloc_trim(0)`: give every arena's free top and free whole pages back to the OS. False
    where there is no glibc."""
    c = glibc()
    if c is None:
        return False
    try:
        return bool(c.malloc_trim(0))
    except Exception:  # noqa: BLE001
        return False


def set_mallopt(thresholds=True, arena_max=0, trim_mb=64, mmap_mb=16):
    """The producer's malloc settings, once at its start. `thresholds`: M_TRIM_THRESHOLD `trim_mb` and
    M_MMAP_THRESHOLD `mmap_mb` (the fields' 32.8 MB windows are then mmapped and unmapped on free, not
    carved from an arena). `arena_max` > 0: M_ARENA_MAX, unless MALLOC_ARENA_MAX is in the environment
    (glibc read that already). Returns what was set ({} where there is no glibc)."""
    c = glibc()
    if c is None:
        return {}
    got = {}
    try:
        if thresholds:
            got["trim_threshold"] = int(c.mallopt(M_TRIM_THRESHOLD, int(trim_mb) << 20))
            got["mmap_threshold"] = int(c.mallopt(M_MMAP_THRESHOLD, int(mmap_mb) << 20))
        if int(arena_max) > 0 and not os.environ.get("MALLOC_ARENA_MAX"):
            got["arena_max"] = int(c.mallopt(M_ARENA_MAX, int(arena_max)))
    except Exception as e:  # noqa: BLE001
        got["err"] = repr(e)
    return got


RAM_EXIT = {}     # {out: reason} set by the supervisor's heartbeat, read by the trainer every step


def trainer_guard(out, limit_gb, host_max, mem=None, rss_gb=None, log=print):
    """One look for the TRAINER's sake: past `trainer_ram_verdict`, record the reason in `RAM_EXIT`
    (the training loop reads it every step, checkpoints if worthwhile and exits cleanly), log a
    `ram_exit` record and a loud line. Returns the reason (None when all is well)."""
    total, avail = mem if mem is not None else host_mem()
    frac = (total - avail) / total if total else 0.0
    r = own_rss_gb() if rss_gb is None else float(rss_gb)
    why = trainer_ram_verdict(r, frac, limit_gb, host_max)
    if why and str(out) not in RAM_EXIT:
        RAM_EXIT[str(out)] = why
        jlog(out, "sched", {"kind": "ram_exit", "reason": why, "trainer_gb": round(r, 2),
                            "host_frac": round(frac, 3)}, echo=False)
        log(f"!!!! [ram_guard] {why}: the TRAINER checkpoints (if >= 200 steps since the last) and "
            f"exits cleanly before the kernel kills the host !!!!")
    return why


def heartbeat(out, place, stop_ev, procs, respawn, limits=(0.0, 0.0)):
    """The supervisor's own thread: stamp `workers.json` and `logs/sched.jsonl` (with the run's
    process-tree RSS and the host's memory), supervise the producer (`ProducerWatch`: a dead one is
    restarted at the next tick, a silent one after `SILENT_MAX_S`) and run the host-RAM guard, both
    every `RAM_CHECK_S`."""
    watch = ProducerWatch(out, procs, respawn, log=lambda m: print(m, flush=True))
    say = lambda m: print(m, flush=True)   # noqa: E731
    while not stop_ev.is_set():
        mem = ram_guard(out, log=say)
        trainer_guard(out, *limits, log=say)
        w = {"mode": place["mode"], "phases": bool(place["phases"]), "t": time.time(),
             "train": {"pid": os.getpid(), "phase": "train", "last_ts": time.time(),
                       **{k: read_state(out).get(k) for k in ("step", "round", "cursor", "verso_on")}}}
        p = _read_json(os.path.join(out, "workers", "produce.json")) or {}
        w["produce"] = p
        _write_json(os.path.join(out, "workers.json"), w)
        jlog(out, "sched", {"kind": "heartbeat", **{k: w["train"].get(k) for k in
                                                    ("step", "round", "cursor")},
                            "produce_phase": p.get("phase"),
                            "produce_age_s": round(time.time() - float(p.get("last_ts") or 0), 1),
                            "produce_progress_age_s": round(time.time() - float(p.get("progress_ts")
                                                                                or p.get("last_ts") or 0), 1),
                            **{k: v for k, v in mem.items() if k != "action"},
                            "producer_paused": producer_paused(out)},
             echo=False)
        watch.check()
        t_next = time.time() + HEARTBEAT_S
        while not stop_ev.is_set() and time.time() < t_next:
            stop_ev.wait(min(RAM_CHECK_S, max(t_next - time.time(), 0.0)))
            if not stop_ev.is_set() and time.time() < t_next:
                ram_guard(out, log=say)
                trainer_guard(out, *limits, log=say)
                watch.check()


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
    bud = budget(place, table={"train": float(cfg.vram_train_gb), "produce": float(cfg.vram_produce_gb)})
    jlog(out, "sched", {"kind": "start", "mode": place["mode"], "phases": place["phases"],
                        "gpus": [g for g, _ in found], "budget_gb": {k: v["gb"] for k, v in bud.items()},
                        "regions": len(ctx["records"]), "heldout": len(ctx["heldout"])})
    if place["phases"]:
        write_phase(out, "train")
    elif os.path.exists(os.path.join(out, PHASE_FILE)):
        os.remove(os.path.join(out, PHASE_FILE))
    for f in (STOP_FILE, PAUSE_FILE):
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))

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

    # a produce.json left by the previous process (a crash, a reboot) is hours old: the first
    # heartbeat read it as a silent producer and restarted the one it had just spawned
    _write_json(os.path.join(out, "workers", "produce.json"),
                {"pid": None, "phase": "spawning", "last_ts": time.time()})
    procs["produce"] = spawn_producer()
    RAM_EXIT.pop(str(out), None)
    hb = threading.Thread(target=heartbeat, args=(out, place, stop_ev, procs, spawn_producer),
                          kwargs={"limits": (float(cfg.ram_trainer_gb), float(cfg.ram_host_exit))},
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
    t_g = time.time()
    def build_grid():
        return sample.val_grid(cfg, ctx["heldout"], root=out, ct=ctx["ct"], ax=ctx["ax"], round_=0,
                               meta=ctx["meta5"], spill=os.path.join(out, "eval", "grid"),
                               threads=max(min(int(os.cpu_count() or 1) // 2, 4), 1))

    val = build_grid()
    jlog(out, "sched", {"kind": "val_grid", "items": len(val), "s": round(time.time() - t_g, 1),
                        "rebuilt": int(getattr(val, "rebuilt", len(val))),
                        "reused": int(getattr(val, "reused", 0))})

    def refresh_grid():
        """At an evaluation boundary: when the held-out labels a reader sees have changed (the verso
        appeared, or a regenerated bundle was committed), the items of the regions whose labels changed
        are rebuilt under new names (the rest are reused, `sample.val_grid`); the recto reference grid
        directory is never touched. The trainer's DiskGrid is updated IN PLACE, so the next evaluation
        reads the new items."""
        t_r = time.time()
        new = build_grid()
        if getattr(new, "paths", None) is not None and list(new.paths) != list(val.paths):
            val.paths[:] = list(new.paths)
            jlog(out, "sched", {"kind": "val_grid_refresh", "items": len(val),
                                "dir": os.path.dirname(val.paths[0]) if val.paths else "",
                                "s": round(time.time() - t_r, 1), "rebuilt": int(new.rebuilt),
                                "reused": int(new.reused)})
    st0 = read_state(out)
    state = {"round": int(st0.get("round", 0)), "stop": False, "ref": st0.get("round_ref")}
    k_active = max(len([k for k in cfg.rungs if int(k) < RG.COARSE_RUNGS[0]]), 1)
    resume = os.path.exists(ckpt)

    def hook(info):
        """Every `eval_every` steps: publish the state, honour STOP and PHASE, run the two gates, and
        refresh the validation grid when the held-out labels changed. At a checkpoint-only boundary
        (`kind` "ckpt", `cfg.ckpt_every`) or after the RAM guard's checkpoint ("stop"): publish the
        state and honour STOP, nothing else -- no evaluation ran, so no gate may read one."""
        step = int(info["step"])
        kind = info.get("kind", "eval")
        if kind != "eval":
            checkpoint_state(out, step, state["round"])
            if kind == "ckpt" and stop_requested(out):
                jlog(out, "sched", {"kind": "stop", "step": step})
                state["stop"] = True
                return True
            return False
        if isinstance(val, sample.DiskGrid):
            refresh_grid()
        checkpoint_state(out, step, state["round"])
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
            t_g = time.time()
            pair_, why_r2 = eval_streak(out, step)
            ok, why = verso_gate(cfg, out, step, rows_fn, screen=eval_dice(out, step),
                                 r2=list(pair_) if pair_ else [])
            if why_r2:
                why = {**why, "streak": why_r2}
            why = {**why, "gate_s": round(time.time() - t_g, 1)}
            jlog(out, "sched", {"kind": "verso_gate", "step": step, "pass": bool(ok), **why})
            ok = apply_verso_hold(out, step, ok, why)
            if ok:
                write_state(out, verso_on=True, verso_gate_step=step, verso_on_step=step)
        elif state["round"] == 0:
            maybe_regen_verso(cfg, out, step)
        if state["round"] + 1 < int(cfg.rounds):
            st2 = read_state(out)
            nver = None
            if state["round"] == 0 and st2.get("verso_on"):
                from rvsm import regions as _RG
                nver = len(_RG.Catalog(out, 0).list_done("verso"))
            ok, why = round_gate(cfg, out, step, state["round"], rows_fn, state["ref"],
                                 round_start=int(st.get("round_step", 0) or 0),
                                 verso_on=bool(st2.get("verso_on", False)),
                                 verso_on_step=st2.get("verso_on_step", st2.get("verso_gate_step")),
                                 verso_regions=nver)
            if why.get("why") != "not a plateau":
                jlog(out, "sched", {"kind": "round_gate", "step": step, "round": state["round"],
                                    "pass": bool(ok), **why})
            if ok:
                nxt = state["round"] + 1
                tp = os.path.join(out, "ckpt", f"teacher_round_{nxt}.pt")
                infer.save_student(tp, {k: v.cpu() for k, v in info["ema"].items()}, cfg,
                                   temps=CAL.temps_of(info.get("temps") or {}), step=step,
                                   round=nxt)
                if state["round"] == 0 or not state["ref"]:
                    state["ref"] = why.get("rows")          # round 0's stats anchor every later round
                state["round"] = nxt
                round_transition(out, nxt, info.get("quiesce"), teacher=tp, round_step=step,
                                 round_ref=state["ref"])
                jlog(out, "sched", {"kind": "round", "round": nxt, "teacher": tp, "step": step})
                return True
        return False

    resume_now = {"on": bool(resume)}

    def patches_factory():
        from rvsm import ladder
        ds = walk_patches()(cfg, out, root=out, ct=ctx["ct"], ax=ctx["ax"], round_=state["round"],
                            heldout=ctx["heldout"], meta=ctx["meta5"],
                            region_records=walk_records(ctx["records"], ctx["heldout"]),
                            lookahead_n=lookahead(cfg, out, k_active),
                            stream_mirror=ladder.is_url(ladder.pyramid_base(str(cfg.ct))),
                            start=(resume_walk(out, state["round"]) if resume_now["on"] else None))
        # every later loader resumes too: a new round's walk=None and empty cursor dir give None there
        resume_now["on"] = True
        return sample.loader(ds, workers=cfg.workers, batch=cfg.batch, pin_memory=cfg.pin_memory)

    ck = ckpt
    try:
        for _ in range(max(int(cfg.rounds), 1)):
            ck = TR.train(cfg, out=out, init=init, resume=resume, patches_factory=patches_factory,
                          device=dev, val_items=val, hook=hook, ckpt=ckpt, accum=cfg.accum,
                          stop_now=lambda: RAM_EXIT.get(str(out)))
            init, resume = None, True
            if RAM_EXIT.get(str(out)):
                jlog(out, "sched", {"kind": "exit_ram", "reason": RAM_EXIT[str(out)],
                                    "step": int(read_state(out).get("step", 0))})
                break
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
                tree = proc_tree(pr.pid)        # its fields pool goes with it
                pr.terminate()
                pr.join(30)
                kill_tree(tree)
            if getattr(pr, "pid", None):
                kill_group(pr.pid)              # and whatever of its group outlived it
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
    them for that region (plan §1). Nothing is deleted while it is the newest thing there is, and the
    held-out regions' round-0 stores (the reference, `eval/heldout.json`) are never deleted."""
    from rvsm import regions as RG
    old = int(round_) - int(keep)
    if old < 0:
        return []
    import glob
    new = RG.Catalog(out, int(round_), ttl=1.0)
    gone = []
    # the held-out regions' ROUND-0 stores are the fixed reference every gate and evaluation of every
    # round is scored against: pinned by the held-out manifest (eval/heldout.json), never collected
    pinned = set()
    if old == 0:
        pinned = {tuple(int(v) for v in h["lo"])
                  for h in (_read_json(heldout_path(out)) or {}).get("regions", [])}
    root = os.path.join(out, "stores", f"round_{old}")
    # EVERY channel directory of the old round -- the per-rung fields (midline_r3, thickness_r4, ...)
    # included, which a fixed list of channel names left behind -- and every generation of a store
    chans = sorted(n for n in (os.listdir(root) if os.path.isdir(root) else [])
                   if os.path.isdir(os.path.join(root, n)) and n != "coarse.zarr")
    for ch in chans:
        d = os.path.join(root, ch)
        for lo in RG.Catalog(out, old, ttl=1.0).list_done(ch):
            if not new.done("recto", lo) or tuple(lo) in pinned:
                continue
            base = os.path.join(d, "region_%d_%d_%d.zarr" % lo)
            for p in [base] + sorted(glob.glob(base[:-len(".zarr")] + ".g*.zarr")):
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
