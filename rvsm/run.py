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
WAIT_S = 5.0                # the trainer's sleep when nothing in the lookahead window is ready
IDLE_S = 2.0                # the producer's sleep when there is nothing to produce
ROUND_GAIN = 0.02           # "< 2 % remaining gain" is the plateau half of the round gate
GATE_REGIONS = 2            # held-out regions scored per gate attempt (see the module docstring)
GATE_SCREEN = 0.1           # the verso gate's held-out pass is skipped while the eval's fine-rung dice
                            # is more than this below `verso_gate_dice` (see `verso_gate`)
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
        from rvsm import infer, teachers as T
        if row[3] is None:
            net, spec = T.load_teacher(row[0], row[2], device=self.device or "cpu")
            row[1], row[3] = spec, net
            self.fast[row[0]] = (infer.fast_teacher(net, self.device or "cpu", compile=self.cfg.compile)
                                 if self.cfg.teacher_bf16 else None)
        return row[3]

    def read(self, ct, lo, size, pyr=None):
        """Every teacher's CT for a region, read ahead of its pass (`infer.teacher_read`): the producer's
        reader thread calls this for the NEXT region while the GPU runs the current one."""
        from rvsm import infer, ladder
        pyr = ladder.rungs(ct) if pyr is None else pyr
        return [infer.teacher_read(ct, lo, size, row[1], pyr=pyr) for row in self.items]

    def probs_u8(self, ct, lo, size, rois=None):
        """(u8 fused probability, u8 agreement weight, attrs) for one region, over every loaded teacher,
        as uint8 TENSORS on the device: the teachers' float planes never leave the card (the host-side
        fuse of two 1024^3 float32 volumes was ~40 s a region)."""
        from rvsm import infer
        ps, names = [], []
        for i, row in enumerate(self.items):
            net = self._net(row)
            ps.append(infer.teacher_region(ct, lo, size, row[1], row[2], device=self.device,
                                           backend=self.backend, net=net, as_tensor=True,
                                           fast=self.fast.get(row[0]),
                                           roi=(rois[i] if rois is not None else None),
                                           engine_dir=os.path.join(self.out, "ckpt", "trt")))
            names.append(row[0])
        if len(ps) >= 2:
            P, W = infer.fuse_agreement_u8(ps[0], ps[1])
        else:
            P = infer.u8_t(ps[0])
            W = torch_full_like_u8(P, 255)
        del ps
        return P, W, {"producer": "teacher:" + ",".join(names), "radial_sign": 1,
                      "ckpt": {r[0]: r[2] for r in self.items}, "backend": self.backend}

    def probs(self, ct, lo, size):
        """(fused probability, agreement weight, attrs) as host float32 arrays (the u8 path, decoded)."""
        P, W, attrs = self.probs_u8(ct, lo, size)
        return (P.cpu().numpy().astype(np.float32) / 255.0, W.cpu().numpy().astype(np.float32) / 255.0,
                attrs)


def torch_full_like_u8(t, v):
    import torch
    return torch.full_like(t, int(v), dtype=torch.uint8)


class StudentSlot:
    """The student the producer runs, reloaded when its file changes and not otherwise.

    Round 0 (the verso passes) tracks the LIVE `ckpt/student.pt`. Round r >= 1 (the self passes that
    write round r's targets) uses the FROZEN round teacher `state["teacher"]`
    (`ckpt/teacher_round_<r>.pt`, the EMA snapshotted when round r opened): a round's targets must come
    from one fixed network, not from the student that is being trained on them."""

    def __init__(self, out, device=None, compile=True):
        self.out = str(out)
        self.path = os.path.join(self.out, "ckpt", "student.pt")
        self.device, self.compile = device, bool(compile)
        self.st, self.mtime, self.loaded, self.sha = None, None, None, None

    def source(self, round_=0, teacher=None):
        """The checkpoint this round's student passes must use (None: not there yet)."""
        if int(round_) >= 1:
            return teacher if teacher and os.path.exists(teacher) else None
        return self.path if os.path.exists(self.path) else None

    def get(self, round_=0, teacher=None):
        from rvsm import infer
        p = self.source(round_, teacher)
        if p is None:
            return None
        m = os.path.getmtime(p)
        if self.st is None or p != self.loaded or m != self.mtime:
            try:
                import hashlib
                with open(p, "rb") as f:          # read ONCE: the digest is of the bytes loaded
                    buf = f.read()
                st = infer.student_fn(p, device=self.device, compile=self.compile, data=buf)
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


def feed_coarse_once(out, lo, round_, block, shape2, pooled=None):
    """Fold a finished recto block into the coarse rungs, once per region and round (a marker file, so
    a restart does not redo it and two producers could not double-count it)."""
    from rvsm import regions as RG
    m = _fed_marker(out, lo, round_)
    if os.path.exists(m):
        return []
    ks = RG.feed_coarse(out, "recto", lo, block, round_=round_, shape2=shape2, pooled=pooled)
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
    jobs = max(int(os.cpu_count() or 1), 1)   # the fields pool: every core, at low priority
    jlog(out, "produce", {"kind": "start", "pid": os.getpid(), "device": str(device),
                          "regions": len(route), "heldout": len(held), "pinned": pinned,
                          "backend": str(backend), "field_rungs": list(frungs), "jobs": jobs})

    bank, slot = None, StudentSlot(out, device=device, compile=cfg.compile)
    keys = {}
    # a restart: the inventory charged every shard an earlier process left, and nothing is held yet --
    # the regions the trainer's workers are reading are re-held (and re-fetched if they lost shards)
    # BEFORE the first eviction, which then honours the budget at once
    held0 = cursor_leases(out)
    for lo in held0:
        keys[lo] = cache.fetch_region(np.array(lo, np.int64), ctx=cfg.ctx, patch=cfg.patch,
                                      region=cfg.region, evict=False)
    cache.lease(cache.region_key(lo) for lo in held0)
    n_ev = cache.evict()
    jlog(out, "produce", {"kind": "cache_start", "leased": len(held0), "evicted": n_ev,
                          **{k: (round(v, 2) if isinstance(v, float) else v)
                             for k, v in cache.stats().items()}}, echo=False)
    L = k_active + int(cfg.lookahead_extra)
    t_reest = 0.0

    def stopping():
        return stop_requested(out) or (stop is not None and stop.is_set()) or \
            (max_s is not None and time.time() - t_start > float(max_s))

    # THE OVERLAP. The GPU thread (this one) only ever runs network passes. Around it:
    #   reader   fetches the NEXT unit's shards and decodes its CT while the current unit infers
    #   writer   volcomp-encodes and writes the stores, folds the coarse rungs, logs the unit
    #   fields   the CPU distance fields (`targets.region_fields`, a process pool of `jobs`)
    # A region with a unit in the writer or the fields queue is BUSY: its stores are not on disk yet,
    # so the disk-derived state machine would hand out the same unit again.
    import concurrent.futures as cf
    reader = cf.ThreadPoolExecutor(1, thread_name_prefix="rvsm-read")
    writer = cf.ThreadPoolExecutor(1, thread_name_prefix="rvsm-write")
    fielder = cf.ThreadPoolExecutor(1, thread_name_prefix="rvsm-fields")
    wslots = threading.BoundedSemaphore(2)     # at most two finished units waiting for the writer
    busy, pend, lock = set(), [], threading.Lock()
    fpool = TG.field_pool(jobs, owner=os.getpid()) if jobs > 1 else None
    pre = {}                                    # (lo, job) -> future of the reader's inputs

    clock = threading.Lock()                    # the shard cache is not thread-safe: one caller at a time

    def need(lo, job):
        """What the reader prepares for a unit: its shards always, and a teacher unit's CT as well."""
        with clock:
            if lo not in keys:
                keys[lo] = cache.fetch_region(np.array(lo, np.int64), ctx=cfg.ctx, patch=cfg.patch,
                                              region=cfg.region)
        if job == "teacher" and bank is not None:
            return bank.read(ct_local, lo, region_size(pyr, lo, cfg.region), pyr=pyr)
        return None

    def prefetch(lo, job):
        if (lo, job) not in pre:
            pre[(lo, job)] = reader.submit(need, lo, job)

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
            write_rows(out, lo, rows, cfg, round_, attrs)
            if pooled is not None or (kind == "self" and rows):
                feed_coarse_once(out, lo, round_, rows[0][1], shape2, pooled=pooled)
            jlog(out, "produce", {"kind": kind, "region": list(lo), "round": round_,
                                  "s": round(time.time() - t0, 2), **(extra or {})})
            # the unit that makes a region's fields possible hands it straight to the fields pool
            # (still busy): waiting for the GPU loop's next pass over the window left the fields
            # of every verso region undone until a whole window of ~1 min verso passes had run
            if kind in ("verso", "self") and not stopping():
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
            TG.region_fields(out, lo, ax, round_=round_, rungs=frungs, jobs=jobs, pool=fpool)
            jlog(out, "produce", {"kind": "fields", "region": list(lo), "round": round_,
                                  "s": round(time.time() - t0, 2), "cursor": cursor})
        finally:
            with lock:
                busy.discard(lo)

    try:
        while not stopping():
            settle()
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
            leased = cursor_leases(out, round_)
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
                jlog(out, "produce", {"kind": "backpressure", "free_gb": round(free_gb(out), 1)})
                time.sleep(IDLE_S * 5)
                continue

            cat = RG.Catalog(out, round_, ttl=1.0)
            units = []
            for lo in _window(route, pos, cursor, L, held, head=head):
                with lock:
                    if lo in busy:
                        continue
                if region_size(pyr, lo, cfg.region) is None:
                    continue
                job = _next_job(cat, lo, round_, verso_on, out, rungs=frungs)
                if job is not None:
                    units.append((lo, job))
            gpu_units = [u for u in units if u[1] != "fields"]
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
                if job == "teacher" and bank is None:
                    bank = TeacherBank(cfg, out, device=device, backend=backend)
                    pre.pop((lo, job), None)        # read before the bank existed: no CT in it
                if job in ("verso", "self") and slot.get(round_, st.get("teacher")) is None:
                    break
                t0 = time.time()
                stamp({"phase": f"round{round_}", "job": job,
                                 "region": list(lo), "last_ts": t0, "L": L, "cursor": cursor})
                prefetch(lo, job)
                try:
                    got = pre.pop((lo, job)).result()
                except stream.FetchFailed as e:
                    # a shard did not arrive: nothing was booked and no store is written; the region
                    # stays unfinished, so a later pass over the window retries it
                    jlog(out, "produce", {"kind": "fetch_failed", "region": list(lo), "job": job,
                                          "shards": len(e.paths),
                                          "failed_units": int(getattr(cache, "failed_units", 0))})
                    continue
                t_in = time.time() - t0
                if i + 1 < len(gpu_units):
                    prefetch(*gpu_units[i + 1])    # the next unit's read overlaps this unit's pass
                wslots.acquire()                    # backpressure: never more than two units unwritten
                with lock:
                    busy.add(lo)
                try:
                    t1 = time.time()
                    if job == "teacher":
                        P, W, attrs = bank.probs_u8(ct_local, lo, size, rois=got)
                        pooled = RG.pool_chain(P, RG.COARSE_RUNGS)
                        rows = [("recto", P.cpu().numpy(), 8, "prob_u8"),
                                ("rw", W.cpu().numpy(), 8, "prob_u8")]
                        del P, W
                    else:
                        stu = slot.get(round_, st.get("teacher"))
                        heads = "verso" if job == "verso" else "all"
                        sign = -1.0 if job == "verso" else 1.0
                        want = [str(stu.layout.channels[0])] if heads == "verso" else "all"
                        planes = _student_planes(stu, ct_local, ax, lo, size, sign, want, meta5, pyr)
                        attrs = {"producer": "student", "ckpt": stu.ckpt, "step": int(stu.step),
                                 "ckpt_sha256": slot.sha, "frozen_teacher": bool(round_ >= 1),
                                 "radial_sign": int(sign), "window": int(stu.cfg.infer_window),
                                 "halo": int(stu.cfg.infer_halo),
                                 "cascade_depth": int(stu.cfg.cascade_depth),
                                 "temps": {str(k): float(v) for k, v in stu.temps.items()}}
                        rows = student_rows_t(planes, stu.layout, heads)
                        del planes
                        pooled = None
                    t_gpu = time.time() - t1
                    extra = {"L": L, "cursor": cursor, "read_wait_s": round(t_in, 2),
                             "gpu_s": round(t_gpu, 2), **_vram()}
                    with lock:
                        pend.append(writer.submit(finish, job, lo, round_, t0, rows, attrs,
                                                  pooled, extra))
                except BaseException:
                    with lock:
                        busy.discard(lo)
                    wslots.release()
                    raise
                did = True
                with clock:
                    _release_passed(cache, keys, pos, cursor, cat, round_, verso_on, out, frungs,
                                    leased=leased)
            for k in [k for k in pre if k not in gpu_units]:
                pre.pop(k)                          # a read for a unit this pass no longer wants
            if not did:
                with clock:
                    _release_passed(cache, keys, pos, cursor, cat, round_, verso_on, out, frungs,
                                    leased=leased)
                time.sleep(IDLE_S)
                if read_phase(out, "") == "produce" and not busy:
                    write_phase(out, "train")     # the window is drained: give the card back
    finally:
        for ex in (reader, writer, fielder):
            ex.shutdown(wait=True)
        if fpool is not None:
            fpool.shutdown(wait=True)
        try:
            cache.close()
        except Exception:  # noqa: BLE001
            pass
        hb_stop.set()
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


def _student_planes(stu, ct, ax, lo, size, sign, want, meta5, pyr):
    from rvsm import infer
    return infer.student_region(stu, ct, ax, lo, size, sign=sign, heads=want, meta=meta5, pyr=pyr,
                                as_tensor=True)


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


def _next_job(cat, lo, round_, verso_on, out, rungs=(2, 3, 4)):
    """Which pass this region lacks, in the order the state machine allows -- or None when it is done.

    Round 0: the teacher pass, then (once the gate has fired) the flipped-sign verso, then the distance
    fields. Round r >= 1: one multi-head student pass, then the fields at the pooled rungs."""
    from rvsm import targets as TG
    if round_ == 0:
        if not cat.done("recto", lo):
            return "teacher"
        if verso_on and not cat.done("verso", lo):
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
        if _next_job(cat, lo, round_, verso_on, out, rungs=rungs) is None:
            cache.release(keys.pop(lo))


def _produce_entry(cfg_json, out, gpu, frac, backend):
    """The spawned producer's entry point. Sets `CUDA_VISIBLE_DEVICES` BEFORE torch is imported, which
    is why this module imports torch nowhere at the top level."""
    own_process_group()         # the producer, its forkserver and its fields pool: one killable group
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    from rvsm import cli
    cli._stack_dumps()          # `kill -USR1` dumps the producer's threads too
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
        p = stores.store_path(out, "recto", lo, 0)
        if not stores.is_done(p):
            continue
        ref = stores.open_store(p)               # lazy: read block by block by `compare_stores`
        shape = tuple(int(v) for v in ref.shape[-3:])
        if stu is None:
            stu = infer.student_fn(ckpt, device=device, compile=False)
        head = str(stu.layout.channels[0])
        planes = infer.student_region(stu, ct or cfg.ct, ax, lo, shape, sign=1.0, heads=[head],
                                      meta=meta5, as_tensor=True)
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


def eval_dice(out, step):
    """The headline (fine-rung, voxel-weighted) `dice` of the evaluation at `step` from
    `logs/eval.jsonl`, or None when there is none."""
    for r in reversed(tail_jsonl(os.path.join(str(out), "logs", "eval.jsonl"), 20)):
        if int(r.get("step", -1)) == int(step) and isinstance(r.get("dice"), (int, float)):
            return float(r["dice"])
    return None


def verso_gate(cfg, out, step, rows_fn=None, screen=None):
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
        if screen is None or not np.isfinite(screen) or float(screen) < floor:
            return False, {"why": "verso_after_steps reached, but the eval's fine-rung dice is below "
                                  "verso_min_dice: waiting", "eval_dice": screen, "verso_min_dice": floor,
                           "step": int(step)}
        return True, {"why": "verso_after_steps", "step": int(step), "eval_dice": float(screen)}
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


def round_gate(cfg, out, step, round_, rows_fn=None, ref=None, round_start=0, verso_on=True):
    """Should round `round_` end and round `round_ + 1` open? Every doubt answers NO (fail closed).

    In order, each one a refusal with its reason logged:
      - round 0 without `verso_on`: self-distillation needs round 0's verso stores to exist;
      - fewer than `round_steps` steps IN THIS ROUND (`step - round_start`, not the absolute step:
        the absolute one made round 1 due the moment round 0 was over);
      - less than `round_steps` of the global `steps` budget left: a promoted round must get to train;
      - no held-out comparison rows, or a non-finite precision / betti0 error;
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
    flat, pwhy = plateau(out, "dice", since=round_start)
    base.update({"plateau": bool(flat), "plateau_why": pwhy.get("why")})
    rows = list((rows_fn() if rows_fn is not None else None) or [])
    if not rows:
        return False, {"why": "no held-out comparison rows (fail closed)", **base}
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
        ready = []
        with clock:
            for lo in homes:
                k = cache.region_key(lo)
                if lo not in keys or cache.missing(k):
                    keys[lo] = cache.fetch_region(np.array(lo, np.int64), evict=False, **fetch_kw)
                if not cache.missing(k):
                    ready.append(list(lo))
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
        got = CFG.stored_fingerprint(old)
        assert got == cfg.fingerprint(), (
            f"resume: {cp} was written by a config whose fingerprint is {got}, this run's is "
            f"{cfg.fingerprint()}. Everything but {CFG.FINGERPRINT_EXCLUDE} must match.")
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

    def check(self):
        """One look. Returns what happened: None (healthy / nothing to do), "stall" (reported),
        "restart", "backoff", "stuck" (will not exit) or "silent_thread"."""
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


def heartbeat(out, place, stop_ev, procs, respawn):
    """The supervisor's own thread: stamp `workers.json` and `logs/sched.jsonl` (with the run's
    process-tree RSS and the host's memory), supervise the producer (`ProducerWatch`: a dead one is
    restarted at the next tick, a silent one after `SILENT_MAX_S`) and run the host-RAM guard, both
    every `RAM_CHECK_S`."""
    watch = ProducerWatch(out, procs, respawn, log=lambda m: print(m, flush=True))
    while not stop_ev.is_set():
        mem = ram_guard(out, log=lambda m: print(m, flush=True))
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
                ram_guard(out, log=lambda m: print(m, flush=True))
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
    t_g = time.time()
    val = sample.val_grid(cfg, ctx["heldout"], root=out, ct=ctx["ct"], ax=ctx["ax"], round_=0,
                          meta=ctx["meta5"], spill=os.path.join(out, "eval", "grid"),
                          threads=max(min(int(os.cpu_count() or 1) // 2, 4), 1))
    jlog(out, "sched", {"kind": "val_grid", "items": len(val), "s": round(time.time() - t_g, 1)})
    st0 = read_state(out)
    state = {"round": int(st0.get("round", 0)), "stop": False, "ref": st0.get("round_ref")}
    k_active = max(len([k for k in cfg.rungs if int(k) < RG.COARSE_RUNGS[0]]), 1)
    resume = os.path.exists(ckpt)

    def hook(info):
        """Every `eval_every` steps: publish the state, honour STOP and PHASE, run the two gates."""
        step = int(info["step"])
        write_state(out, step=step, round=state["round"], cursor=read_cursor(out, state["round"]),
                    region_s=region_seconds(out, state["round"]), walk=walk_snapshot(out, state["round"]),
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
            t_g = time.time()
            ok, why = verso_gate(cfg, out, step, rows_fn, screen=eval_dice(out, step))
            why = {**why, "gate_s": round(time.time() - t_g, 1)}
            jlog(out, "sched", {"kind": "verso_gate", "step": step, "pass": bool(ok), **why})
            if ok:
                write_state(out, verso_on=True, verso_gate_step=step)
        if state["round"] + 1 < int(cfg.rounds):
            ok, why = round_gate(cfg, out, step, state["round"], rows_fn, state["ref"],
                                 round_start=int(st.get("round_step", 0) or 0),
                                 verso_on=bool(read_state(out).get("verso_on", False)))
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
                          device=dev, val_items=val, hook=hook, ckpt=ckpt, accum=cfg.accum)
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
