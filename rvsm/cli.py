"""The rvsm command line. The subcommands land with the modules they drive; this is the entry point."""
from __future__ import annotations

import os
import sys

USAGE = """rvsm <command> [options]

  run        cfg.toml | --ct URL|PATH --umbilicus PATH|auto --out DIR --gpus 0[,1]
             --mode auto|resident|timeshare [--rounds N] [--steps N] [--size 30m6] [--init ckpt.pt]
  produce    --out DIR (--teacher recto,m7 | --student ckpt [--sign -1]) --region Z Y X
  train      cfg.toml
  eval       --out DIR [--ckpt P] [--round R] [--tifxyz DIR] [--json]
  export     --out DIR --ckpt P --box Z Y X DZ DY DX --dest DIR
  calibrate  --out DIR --ckpt P
  pretrain   cfg.toml [--steps N]
  ladder     cfg.toml --sizes 15m,30m6,60m
  status | stop | ledger --rebuild | umbilicus --ct URL --out umbilicus.json
  teachers   fetch [recto,m7] [--cache DIR] [--extras]
"""

COMMANDS = ("run", "produce", "train", "eval", "export", "calibrate", "pretrain", "ladder",
            "status", "stop", "ledger", "umbilicus", "teachers", "verso")


def _stack_dumps():
    """`kill -USR1 <pid>` prints every thread's Python stack to stderr (the run log). Some hosts (the
    Thunder containers) forbid ptrace, so py-spy cannot attach; this is the one way to see a stall."""
    try:
        import faulthandler
        import signal
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (AttributeError, ValueError, RuntimeError, OSError):
        pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _stack_dumps()
    if argv and argv[0] in COMMANDS:
        # A subcommand is implemented as a module-level function of the same name, appended below as the
        # commit that owns it lands; the rest still print the usage and say so.
        fn = globals().get(argv[0])
        if callable(fn):
            return int(fn(argv[1:]) or 0)
        print(USAGE, end="")
        print(f"\n{argv[0]}: not implemented yet in this build.")
        return 2
    print(USAGE, end="")
    return 0 if not argv else 2



# --------------------------------------------------------------------------- #
# `rvsm produce`: one region, no state (commit 2 = the teacher passes)
# --------------------------------------------------------------------------- #
PRODUCE_USAGE = """rvsm produce --out DIR --ct URL|PATH [--umbilicus PATH] --teacher recto[,m7]
                  --ckpt-recto P [--ckpt-m7 P] --region Z Y X [--size 1024]
                  [--backend trt|torch] [--round 0] [--device cuda] [--tta 1]
rvsm produce      --out DIR --ct URL|PATH [--umbilicus PATH] --student CKPT --region Z Y X
                  [--sign -1] [--heads verso|all] [--round R] [--fields]   (see `--student --help`)

Runs the named teacher(s) over ONE 1024^3 region and writes its round-0 stores:

  <out>/stores/round_0/recto/region_<z>_<y>_<x>.zarr   the (fused) sheet probability, q8
  <out>/stores/round_0/rw/region_<z>_<y>_<x>.zarr      the fusion weight (1 - |p_recto - p_m7|), q8

With two teachers the probability is the confidence-weighted fusion of both and `rw` is their
agreement; with one, `rw` is 1 everywhere. No walk, no state.json, no lookahead: this is the single
region a producer, a test or a hand at the terminal asks for.
"""


def _flags(argv, usage=None):
    """`--k v ...` -> {k: [v, ...]}; a flag with no value gets []."""
    out, k = {}, None
    for a in argv:
        if str(a).startswith("--"):
            k = str(a)[2:]
            out.setdefault(k, [])
        elif k is not None:
            out[k].append(a)
        else:
            raise SystemExit(f"rvsm: unexpected argument {a!r}\n\n{usage or PRODUCE_USAGE}")
    return out


def produce(argv):
    """The `produce` subcommand. Returns a process exit code."""
    import numpy as np

    from rvsm import infer, ladder, stores
    from rvsm import teachers as T

    f = _flags(argv)
    if "help" in f or "h" in f or not f:
        print(PRODUCE_USAGE, end="")
        return 0
    def one(name, default=None, cast=str):
        v = f.get(name)
        if not v:
            if default is None and name in ("out", "ct"):
                raise SystemExit(f"rvsm produce: --{name} is required\n\n{PRODUCE_USAGE}")
            return default
        return cast(v[0])

    out, ct = one("out"), one("ct")
    umb = one("umbilicus", "")
    names = [q for q in str(one("teacher", "recto")).replace(",", " ").split() if q]
    round_ = int(one("round", 0, int))
    backend = str(one("backend", "torch"))
    tta = int(one("tta", 1, int))
    device = one("device", None)
    if "region" not in f or len(f["region"]) < 3:
        raise SystemExit(f"rvsm produce: --region Z Y X is required\n\n{PRODUCE_USAGE}")
    lo = np.array([int(v) for v in f["region"][:3]], np.int64)
    size = int(one("size", 1024, int))

    # The store is always a whole number of 128^3 chunks, and never larger than what the CT can serve:
    # a region at the far corner of a volume, or a small test volume, produces a smaller store, not a
    # store full of air (and not a 1 GiB write for a 256^3 fixture).
    pyr = ladder.rungs(ct)
    shape = ladder.rung_shape(pyr, infer.RUNG)
    avail = np.maximum(np.asarray(shape, np.int64) - lo, 0)
    n = np.minimum(np.full(3, size, np.int64), -(-avail // stores.CHUNK) * stores.CHUNK)
    if (n <= 0).any():
        raise SystemExit(f"rvsm produce: region {tuple(int(v) for v in lo)} is outside the volume "
                         f"(rung-2 shape {tuple(int(v) for v in shape)})")
    size3 = tuple(int(v) for v in n)

    # A student pass is the same region, the same store layout and the same flags -- only the thing
    # being run differs -- so it is a branch of this subcommand and not a second one.
    if "student" in f:
        return _produce_student(f, one, out, ct, umb, lo, size3, round_, device)

    probs, win, hal, ckpts = {}, {}, {}, {}
    for nm in names:
        if nm not in T.TEACHERS:
            raise SystemExit(f"rvsm produce: unknown teacher {nm!r} (have {sorted(T.TEACHERS)})")
        spec = T.TEACHERS[nm]
        ckpt = one(f"ckpt-{nm}", one("ckpt", None))
        if not ckpt:
            # No explicit checkpoint: use (and, the first time, fill) the weights cache. A teacher with
            # no published URL -- the tests' `fake` -- has to be pointed at its file.
            if not spec.url:
                raise SystemExit(f"rvsm produce: --ckpt-{nm} PATH is required for teacher {nm!r} "
                                 f"(it has no published weights to download)")
            ckpt = T.fetch_weights(nm, cache_dir=one("cache", T.CACHE_DIR))
        w = int(one("window", spec.patch[0], int))
        h = int(one("halo", max(1, w // 8), int))
        print(f"[produce] {nm}: region {tuple(int(v) for v in lo)} size {size3} window {w} halo {h} "
              f"backend {backend}", flush=True)
        probs[nm] = infer.teacher_region(ct, lo, size3, spec, ckpt, device=device, backend=backend,
                                         tta=tta, window=w, halo=h,
                                         engine_dir=os.path.join(str(out), "ckpt", "trt"))
        win[nm], hal[nm], ckpts[nm] = w, h, ckpt

    if len(probs) >= 2:
        a, b = names[0], names[1]
        p, rw = infer.fuse_agreement(probs[a], probs[b])
    else:
        p = probs[names[0]]
        rw = np.ones_like(p)

    attrs = {"producer": "teacher:" + ",".join(names), "round": int(round_),
             "ckpt": {nm: str(ckpts[nm]) for nm in names},
             "window": {nm: int(win[nm]) for nm in names}, "halo": {nm: int(hal[nm]) for nm in names},
             "radial_sign": 1, "tta": int(tta), "backend": str(backend)}
    got = {}
    for ch, block, q in (("recto", stores.u8(p), 8), ("rw", stores.u8(rw), 8)):
        path = stores.store_path(out, ch, lo, round_)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stores.write(path, block, tuple(int(v) for v in lo), rung=infer.RUNG, channels=(ch,), q=q,
                     volume=str(ct), umbilicus=str(umb), attrs=attrs)
        got[ch] = path
        print(f"[produce] wrote {path} {block.shape}", flush=True)
    return 0



# --------------------------------------------------------------------------- #
# `rvsm produce --student`: the student passes (commit 5)
# --------------------------------------------------------------------------- #
STUDENT_USAGE = """rvsm produce --out DIR --ct URL|PATH [--umbilicus PATH] --student CKPT
                  --region Z Y X [--size 1024] [--sign -1] [--heads verso|all] [--round R]
                  [--device cuda] [--window 256] [--halo 32] [--cascade-depth 3] [--batch 1]
                  [--tta 1] [--no-compile] [--fields [--jobs N]]

Runs ONE student checkpoint over ONE region, in one multi-head pass, and writes its stores.

  --sign -1 --heads verso   round 0's VERSO pass: the student is recto-trained, so the verso band is
                            what it predicts with the radial input channels NEGATED. Only the
                            `verso` store is written -- the first probability plane under the flipped
                            sign, renamed, with `radial_sign: -1` in its attrs.
  --heads all               round >= 1's self-distillation pass (sign +1): `recto`, `verso` (q8) and
                            `midline`, `thickness`, `conf` (q0, because code 0 means NO DATA and a
                            codec that rounds a 1 to a 0 there invents a hole). Refused at a negative
                            sign: the field stores' sign convention is defined at +1 only.

  --fields                  after the stores are written, also build the geometric distance fields
                            (`targets.region_fields`: midline / thickness at rungs 2-4 from this
                            region's own recto + verso stores). It is OFF by default because the
                            DRIVER decides when a region's recto and verso are both final -- in round
                            0 the verso lands long after the recto, and rebuilding the fields on every
                            pass would be two scipy EDTs per block for nothing. A rung whose stores are
                            already `done` is skipped, so with `--heads all` this only fills in rungs
                            3 and 4 beside the rung-2 stores the pass just wrote.
"""


def _produce_student(f, one, out, ct, umb, lo, size3, round_, device):
    """The `--student` half of `rvsm produce`. Returns a process exit code."""
    from rvsm import axis as AX, export as EX, infer, stores, targets as TG

    ckpt = one("student")
    sign = float(one("sign", 1.0, float))
    heads = str(one("heads", "verso" if sign < 0 else "all")).strip()
    if heads not in ("verso", "all"):
        raise SystemExit(f"rvsm produce: --heads is 'verso' or 'all', not {heads!r}\n\n{STUDENT_USAGE}")
    if heads == "all" and sign < 0:
        raise SystemExit(
            "rvsm produce: --heads all at --sign -1 is refused. The field stores (midline, thickness) "
            "carry a SIGN CONVENTION -- d > 0 towards the recto face, radially outward -- which is "
            "defined at radial sign +1; writing them from a flipped-sign pass would store the "
            "opposite geometry under the same attrs. Use --sign -1 --heads verso (round 0's verso "
            f"pass) or --sign 1 --heads all (round >= 1).\n\n{STUDENT_USAGE}")

    w = one("window", None, int)
    h = one("halo", None, int)
    depth = one("cascade-depth", None, int)
    batch = int(one("batch", 1, int))
    tta = int(one("tta", 1, int))
    do_compile = "no-compile" not in f

    ax = AX.load(umb or "auto", ct=ct)
    st = infer.student_fn(ckpt, device=device, compile=do_compile)
    want = [str(st.layout.channels[0])] if heads == "verso" else "all"
    eff_w = int(w if w is not None else st.cfg.infer_window)
    eff_h = int(h if h is not None else st.cfg.infer_halo)
    eff_d = int(depth if depth is not None else st.cfg.cascade_depth)
    print(f"[produce] student {ckpt} step {st.step}: region {tuple(int(v) for v in lo)} size {size3} "
          f"sign {sign:+g} heads {heads} window {eff_w} halo {eff_h} cascade {eff_d}", flush=True)
    planes = infer.student_region(st, ct, ax, lo, size3, sign=sign, heads=want, window=w, halo=h,
                                  cascade_depth=depth, batch=batch, tta=tta)

    attrs = {"producer": "student", "ckpt": str(ckpt), "step": int(st.step), "round": int(round_),
             "radial_sign": int(sign), "window": eff_w, "halo": eff_h, "cascade_depth": eff_d,
             "tta": int(tta), "temps": {str(k): float(v) for k, v in st.temps.items()},
             "sign_convention": EX.SIGN_CONVENTION}

    # (channel, uint8 block, q, encoding). The probabilities are q8 (lossy is harmless: a probability
    # is read as a weight); every FIELD is q0, because its code 0 is the no-data marker and its other
    # codes are a distance in 0.25-voxel steps, neither of which a codec may round.
    rows = []
    if heads == "verso":
        rows.append(("verso", stores.u8(planes[str(st.layout.channels[0])]), 8, "prob_u8"))
    else:
        rec = planes["recto"]
        valid = rec > 0                       # the region pass already zeroed the CT's air
        rows.append(("recto", stores.u8(rec), 8, "prob_u8"))
        rows.append(("verso", stores.u8(planes["verso"]), 8, "prob_u8"))
        rows.append(("midline", EX.enc_signed(planes["midline"], valid), 0,
                     "signed_u8_off128_q0.25"))
        rows.append(("thickness", TG.encode_unsigned(planes["thickness"], valid), 0,
                     "unsigned_u8_q0.25"))
        rows.append(("conf", stores.u8(planes["conf"]), 0, "conf_u8"))

    got = {}
    for ch, block, q, enc in rows:
        path = stores.store_path(out, ch, lo, round_)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stores.write(path, block, tuple(int(v) for v in lo), rung=infer.RUNG, channels=(ch,), q=q,
                     volume=str(ct), umbilicus=str(umb), attrs={"encoding": enc, "no_data": 0,
                                                                "unit": "voxels_of_this_rung",
                                                                "axis_order": "ZYX", **attrs})
        got[ch] = path
        print(f"[produce] wrote {path} {block.shape} q{q} ({enc})", flush=True)

    if "fields" in f:
        rp = stores.store_path(out, "recto", lo, round_)
        if not stores.is_done(rp):
            print(f"[produce] --fields: no finished recto store at {rp}; skipping the distance fields",
                  flush=True)
        else:
            rep = TG.region_fields(out, tuple(int(v) for v in lo), ax, round_=round_,
                                   jobs=int(one("jobs", 1, int)))
            print(f"[produce] fields: {rep}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# `rvsm export`: the tracer contract over a box
# --------------------------------------------------------------------------- #
EXPORT_USAGE = """rvsm export --ckpt P --ct URL|PATH [--out DIR] [--umbilicus PATH]
                 --box Z Y X DZ DY DX --dest DIR [--marching-cubes] [--level 0.0] [--rung 2]
                 [--device cuda] [--window 256] [--halo 32] [--cascade-depth 3] [--batch 1]
                 [--tta 1] [--no-compile]

ONE multi-head student pass over the box, written as the tracer contract into --dest:

  recto, verso              q8 probabilities
  surf_sdist                the RECTO-FACE distance (midline - thickness/2), q0, 0.25-voxel steps
  nz, ny, nx, gmag          the Scharr gradient of that exported field, q0 -- derived here, never a
                            network output, so what the tracer reads is exactly the gradient of what
                            it reads
  thickness, conf           q0

d and n point from the VERSO face towards the RECTO face (radially outward). `--marching-cubes` also
meshes the zero level, one .obj per store shard, vertices in GLOBAL ZYX voxels of the rung.
`--out DIR` is only used to find `<out>/umbilicus.json` when --umbilicus is not given.
"""


def export(argv):
    """The `export` subcommand. Returns a process exit code."""
    from rvsm import axis as AX, export as EX

    f = _flags(argv, EXPORT_USAGE)
    if "help" in f or "h" in f or not f:
        print(EXPORT_USAGE, end="")
        return 0

    def one(name, default=None, cast=str):
        v = f.get(name)
        return default if not v else cast(v[0])

    ckpt, ct, dest = one("ckpt"), one("ct"), one("dest")
    out = one("out", "")
    for nm, v in (("ckpt", ckpt), ("ct", ct), ("dest", dest)):
        if not v:
            raise SystemExit(f"rvsm export: --{nm} is required\n\n{EXPORT_USAGE}")
    if "box" not in f or len(f["box"]) < 6:
        raise SystemExit(f"rvsm export: --box Z Y X DZ DY DX is required\n\n{EXPORT_USAGE}")
    box = [int(v) for v in f["box"][:6]]
    umb = one("umbilicus", "")
    if not umb and out and os.path.exists(os.path.join(out, "umbilicus.json")):
        umb = os.path.join(out, "umbilicus.json")
    ax = AX.load(umb or "auto", ct=ct)
    got = EX.export_student(ckpt, ct, ax, box[:3], box[3:], dest, device=one("device", None),
                            rung=int(one("rung", 2, int)), sign=float(one("sign", 1.0, float)),
                            window=one("window", None, int), halo=one("halo", None, int),
                            cascade_depth=one("cascade-depth", None, int),
                            batch=int(one("batch", 1, int)), tta=int(one("tta", 1, int)),
                            compile="no-compile" not in f,
                            marching_cubes="marching-cubes" in f,
                            mc_level=float(one("level", 0.0, float)), umbilicus=str(umb))
    for k in sorted(got):
        print(f"[export] {k}: {got[k]}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# `rvsm teachers`: the published weights
# --------------------------------------------------------------------------- #
TEACHERS_USAGE = """rvsm teachers fetch [recto,m7] [--cache DIR] [--extras]

Downloads the round-0 teachers' published weights (Hugging Face, scrollprize) into the local cache
(default ~/.cache/rvsm) and prints where they landed. A file already there is left alone, so this is
safe to run before every run; `rvsm produce` calls the same cache when no --ckpt-<name> is given.
`--extras` also fetches the companion files a teacher lists (m7's plans.json), which rvsm never reads:
our port infers the architecture from the state dict's own shapes and pins the normaliser in the spec.
"""


def teachers(argv):
    """The `teachers` subcommand: `fetch` today, and nothing else yet."""
    from rvsm import teachers as T
    argv = list(argv)
    sub = argv[0] if argv and not str(argv[0]).startswith("--") else ""
    if sub != "fetch":
        print(TEACHERS_USAGE, end="")
        return 0 if not argv else 2
    rest = [str(a) for a in argv[1:]]
    pos = rest[:next((i for i, a in enumerate(rest) if a.startswith("--")), len(rest))]
    f = _flags(rest[len(pos):])
    names = " ".join(pos).replace(",", " ").split() or [n for n, s in T.TEACHERS.items() if s.url]
    cache = (f.get("cache") or [T.CACHE_DIR])[0]
    for nm in names:
        if nm not in T.TEACHERS:
            raise SystemExit(f"rvsm teachers: unknown teacher {nm!r} (have {sorted(T.TEACHERS)})")
        print(T.fetch_weights(nm, cache_dir=cache, extras="extras" in f))
    return 0


# --------------------------------------------------------------------------- #
# `rvsm train`: the trainer alone, on whatever stores already exist (commit 4)
# --------------------------------------------------------------------------- #
TRAIN_USAGE = """rvsm train cfg.toml [--out DIR] [--init CKPT] [--resume] [--steps N] [--device cuda]

Trains the student on the region stores under `<out>/stores/round_<r>/`, with the held-out regions
(`--heldout` of them, stratified by z and radius) kept out of the walk and used as the validation
grid. Nothing is produced here: this is the trainer of `rvsm run` on its own, for a directory whose
stores another process (or `rvsm produce`) has already written.

  --out DIR     the run directory (default: the config's `out`)
  --init CKPT   warm start from another run's weights (copied by tensor NAME; new slots start at zero)
  --resume      continue `<out>/ckpt.pt`; the config fingerprint must match
  --steps N     override the config's step budget
"""


def train(argv):
    """The `train` subcommand. Returns a process exit code."""
    import json

    from rvsm import axis as AX, config as CFG, ladder, regions as RG, sample
    from rvsm import train as TR

    argv = list(argv)
    # the positional config file is whatever comes before the first flag
    cut = next((i for i, a in enumerate(argv) if str(a).startswith("--")), len(argv))
    pos, f = argv[:cut], _flags(argv[cut:])
    if "help" in f or "h" in f:
        print(TRAIN_USAGE, end="")
        return 0
    over = {}
    if f.get("out"):
        over["out"] = str(f["out"][0])
    if f.get("steps"):
        over["steps"] = int(f["steps"][0])
    cfg = CFG.load(str(pos[0]) if pos else None, overrides=over)
    out = str(cfg.out)
    os.makedirs(out, exist_ok=True)

    # the round is whatever the run's own state says; a bare `rvsm train` on a fresh directory is round 0
    round_ = 0
    sp = os.path.join(out, "state.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            round_ = int(json.load(fh).get("round", 0))

    pyr = ladder.rungs(cfg.ct)
    ax = AX.load(cfg.umbilicus, ct=cfg.ct)
    recs = RG.region_list(pyr, rungs=cfg.rungs, patch=cfg.patch, region=cfg.region,
                          boost=cfg.rung_boost, occ_min_fine=cfg.occ_min_fine,
                          occ_min_coarse=cfg.occ_min_coarse)
    heldout = RG.held_out(recs, n=cfg.heldout, ax=ax)
    print(f"[train] round {round_}: {len(recs)} region records, {len(heldout)} held out", flush=True)

    def patches_factory():
        ds = sample.Patches(cfg, root=out, ct=cfg.ct, ax=ax, round_=round_, heldout=heldout)
        return sample.loader(ds, workers=cfg.workers, batch=cfg.batch)

    val = sample.val_grid(cfg, heldout, root=out, ct=cfg.ct, ax=ax, round_=round_)
    ck = TR.train(cfg, out=out, init=(f["init"][0] if f.get("init") else None),
                  resume="resume" in f, patches_factory=patches_factory,
                  device=(f["device"][0] if f.get("device") else None), val_items=val)
    print(f"[train] wrote {ck}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# `rvsm eval`, `calibrate`, `ledger`, `umbilicus`, `pretrain`, `ladder` (commit 7)
# --------------------------------------------------------------------------- #
EVAL_USAGE = """rvsm eval --out DIR [--ckpt P] [--round R] [--tifxyz DIR] [--json PATH]
               [--regions N] [--device cuda] [--thr 0.5] [--boot 200] [--no-compile]

Scores the HELD-OUT regions -- the ones `<out>/eval/heldout.json` records, or, when there is no such
file, the ones `regions.held_out` derives from the CT (and then writes there, because a reference set
that moves between evaluations is not a reference set).

  --ckpt P      run THIS checkpoint over each held-out region (`infer.student_region`, recto head) and
                score what it predicts. Without it the already-produced stores of `--round R` are
                scored instead, which is free and is what the driver calls between rounds.
  --round R     which round's stores to score (default: the run's current round). Round 0's stores are
                always the REFERENCE: every number is "this prediction against the fused teacher".
  --tifxyz DIR  also run the mesh suite (recall@k, continuity, ERL, HD95, Betti) against the human
                surfaces that fall inside the held-out boxes. Optional by decision: without meshes the
                store-vs-store comparison is the whole table.
  --json PATH   write the full result (per region, pooled, CI) as json. Default `<out>/eval/eval_<tag>.json`.

Every pooled number is printed as `value [lo, hi]`, the interval being a bootstrap over REGIONS: eight
regions are the independent unit here, the millions of voxels inside one of them are not.
"""


def _run_config(out, ckpt=None):
    """The Config of a run directory: `<out>/config.json` if the driver froze one, else the checkpoint's
    own copy, else the defaults. `out` always wins, because that is the directory being scored."""
    import json

    from rvsm.config import Config, _TYPES, _coerce
    raw = {}
    p = os.path.join(str(out), "config.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        raw = dict(d.get("config") or d)
    elif ckpt and os.path.exists(str(ckpt)):
        import torch
        st = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        raw = dict(st.get("cfg") or st.get("config") or st.get("args") or {})
    cfg = Config(**{k: _coerce(k, v) for k, v in raw.items() if k in _TYPES})
    from dataclasses import replace
    return replace(cfg, out=str(out))


def _run_round(out, default=0):
    """The round `<out>/state.json` says the run is in."""
    import json
    p = os.path.join(str(out), "state.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return int(json.load(f).get("round", default))
        except Exception:  # noqa: BLE001  -- a half-written state file is not a reason to refuse to eval
            pass
    return int(default)


def heldout_regions(out, cfg, ax=None, n=None, write=True):
    """The held-out regions of a run, as `[{lo, size, k}, ...]` in rung-2 voxels.

    `<out>/eval/heldout.json` is authoritative when it exists: the set must not move between two
    evaluations of the same run, or the two numbers are not comparable. When it does not exist the same
    deterministic `regions.held_out` the trainer uses is run and (unless `write` is off) recorded there,
    so every later evaluation of this directory agrees with this one.
    """
    import json

    from rvsm import ladder, regions as RG
    p = os.path.join(str(out), "eval", "heldout.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        recs = d.get("regions") if isinstance(d, dict) else d
        return [{"lo": [int(v) for v in r["lo"]], "size": [int(v) for v in r["size"]],
                 "k": int(r.get("k", 2)), "f": float(r.get("f", 1.0)),
                 "w": float(r.get("w", 1.0))} for r in recs]
    pyr = ladder.rungs(cfg.ct)
    recs = RG.region_list(pyr, rungs=cfg.rungs, patch=cfg.patch, region=cfg.region,
                          boost=cfg.rung_boost, occ_min_fine=cfg.occ_min_fine,
                          occ_min_coarse=cfg.occ_min_coarse)
    held = RG.held_out(recs, n=int(n or cfg.heldout), ax=ax)
    held = [{"lo": [int(v) for v in r["lo"]], "size": [int(v) for v in r["size"]],
             "k": int(r["k"]), "f": float(r.get("f", 1.0)), "w": float(r.get("w", 1.0))}
            for r in held]
    if write and held:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"regions": held, "seed": 0, "n": len(held), "ct": str(cfg.ct)}, f, indent=1)
        os.replace(tmp, p)
    return held


def _pool_rows(rows, wkey=None):
    """Weight-weighted mean of every finite numeric key across per-region rows (weight `row[wkey]`,
    or 1). Regions differ enormously in how much surface they hold, so a plain mean over regions would
    let an almost-empty corner region count as much as a dense one."""
    import numpy as np
    keys = [k for r in rows for k, v in r.items() if isinstance(v, (int, float))]
    out = {}
    for k in sorted(set(keys)):
        v = np.array([float(r.get(k, np.nan)) for r in rows], float)
        w = np.array([float(r.get(wkey, 1.0)) if wkey else 1.0 for r in rows], float)
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        out[k] = float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")
    return out


def _boot_rows(rows, wkey=None, n=200, seed=0, lo=2.5, hi=97.5):
    """Percentile intervals by resampling REGIONS with replacement and repooling. Fewer than two
    regions means no interval at all -- an interval from one region would be a fabrication."""
    import numpy as np
    if len(rows) < 2 or int(n) < 2:
        return {}
    rng = np.random.default_rng(int(seed))
    draws = [_pool_rows([rows[i] for i in rng.integers(0, len(rows), len(rows))], wkey)
             for _ in range(int(n))]
    return {k: [float(np.nanpercentile([d[k] for d in draws], lo)),
                float(np.nanpercentile([d[k] for d in draws], hi))] for k in draws[0]}


def _print_table(title, pooled, ci, log=print):
    """The plan's table: one metric per line, `value [lo, hi]`."""
    import numpy as np
    if not pooled:
        return
    log(f"\n  {title}")
    for k in sorted(pooled):
        v = pooled[k]
        if not isinstance(v, float) or not np.isfinite(v):
            continue
        b = ci.get(k)
        log(f"    {k:>16}  {v:10.4f}" + (f"  [{b[0]:.4f}, {b[1]:.4f}]" if b else ""))


def _store_block(root, channel, lo, round_):
    """The whole uint8 block of one finished region store, or None when it is not there yet."""
    import numpy as np

    from rvsm import stores
    p = stores.current_path(str(root), str(channel), lo, int(round_))   # what readers use
    if not stores.is_done(p):
        return None
    a = stores.open_store(p)
    return np.asarray(a[:], np.uint8)


def eval(argv):    # noqa: A001  -- the subcommand is named `eval`; `main` dispatches by this name
    """The `eval` subcommand. Returns a process exit code."""
    import json

    import numpy as np

    from rvsm import axis as AX, evalsurf as E, ladder, stores

    f = _flags(argv, EVAL_USAGE)
    if "help" in f or "h" in f or not f:
        print(EVAL_USAGE, end="")
        return 0

    def one(name, default=None, cast=str):
        v = f.get(name)
        return default if not v else cast(v[0])

    out = one("out")
    if not out:
        raise SystemExit(f"rvsm eval: --out DIR is required\n\n{EVAL_USAGE}")
    ckpt = one("ckpt")
    cfg = _run_config(out, ckpt)
    if not cfg.ct:
        raise SystemExit(f"rvsm eval: {out} has no config.json and the checkpoint names no CT volume; "
                         "there is nothing to read the held-out regions from")
    round_ = int(one("round", _run_round(out), int))
    thr = float(one("thr", 0.5, float))
    boot = int(one("boot", 200, int))
    tifxyz = one("tifxyz", cfg.tifxyz)
    um = ladder.rung_um(2)
    ax = AX.load(cfg.umbilicus, ct=cfg.ct)
    held = heldout_regions(out, cfg, ax=ax, n=one("regions", None, int))
    if not held:
        raise SystemExit(f"rvsm eval: no held-out regions for {out}")
    chan = str(cfg.channels[0])

    st = None
    if ckpt:
        from rvsm import infer
        st = infer.student_fn(ckpt, device=one("device", None), compile="no-compile" not in f)
        print(f"[eval] {ckpt}: step {st.step}, temps {st.temps or '{}'}", flush=True)
    print(f"[eval] {len(held)} held-out regions, round {round_} vs the round-0 reference"
          + (f", meshes from {tifxyz}" if tifxyz else ""), flush=True)

    rows, mesh_rows, per_region = [], [], []
    for r in held:
        lo = np.asarray(r["lo"], np.int64)
        ref = _store_block(out, chan, lo, 0)
        if ref is None:
            print(f"[eval] region {tuple(int(v) for v in lo)}: no round-0 reference store; skipped",
                  flush=True)
            continue
        size = tuple(int(v) for v in ref.shape)
        if st is not None:
            planes = infer.student_region(st, cfg.ct, ax, lo, size, heads=[chan])
            pred = stores.u8(planes[chan])
        else:
            pred = _store_block(out, chan, lo, round_)
            if pred is None:
                print(f"[eval] region {tuple(int(v) for v in lo)}: no round-{round_} store; skipped",
                      flush=True)
                continue
        ct = ladder.read_rung(ladder.rungs(cfg.ct), 2, lo, size, dtype=np.uint8) if tifxyz else None
        rec = E.evaluate(lambda o, s, p=pred: p, (tuple(int(v) for v in lo), size), tifxyz=tifxyz or None,
                         ax=ax, ct=ct, ref_u8=ref, thr=thr, um=um, boot=boot)
        rec["region"] = [int(v) for v in lo]
        per_region.append(rec)
        vs = dict(rec.get("vs_store") or {})
        vs["_w"] = float(vs.get("n_ref", 0.0))
        rows.append(vs)
        if rec.get("metrics"):
            m = dict(rec["metrics"])
            m["_w"] = float(m.get("n_points", 0.0))
            mesh_rows.append(m)
        print(f"[eval] region {tuple(int(v) for v in lo)}: dice {vs.get('dice', float('nan')):.4f} "
              f"erl_frac {vs.get('erl_frac', float('nan')):.4f}"
              + (f" recall@4 {rec['metrics'].get('recall@4', float('nan')):.4f}"
                 if rec.get("metrics") else ""), flush=True)

    if not rows:
        raise SystemExit("rvsm eval: not one held-out region had both a reference and a prediction")
    pooled = _pool_rows(rows, "_w")
    ci = _boot_rows(rows, "_w", n=boot)
    res = {"out": str(out), "round": int(round_), "ckpt": str(ckpt or ""),
           "step": int(st.step) if st is not None else None, "thr": thr, "voxel_um": um,
           "n_regions": len(rows), "regions": per_region, "store": pooled, "store_ci": ci}
    _print_table(f"store vs the round-0 reference ({len(rows)} regions)", pooled, ci)
    if mesh_rows:
        res["mesh"] = _pool_rows(mesh_rows, "_w")
        res["mesh_ci"] = _boot_rows(mesh_rows, "_w", n=boot)
        _print_table(f"vs the human meshes ({len(mesh_rows)} regions)", res["mesh"], res["mesh_ci"])

    tag = (f"step_{st.step:06d}" if st is not None else f"round_{round_}")
    jp = one("json", os.path.join(str(out), "eval", f"eval_{tag}.json"))
    os.makedirs(os.path.dirname(str(jp)) or ".", exist_ok=True)
    with open(jp, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\n[eval] wrote {jp}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# `rvsm calibrate`
# --------------------------------------------------------------------------- #
CALIB_USAGE = """rvsm calibrate --out DIR --ckpt P [--round R] [--device cuda] [--limit 8]
                 [--all-rungs] [--dry]

Fits ONE temperature per rung on the run's own held-out grid (`sample.val_grid` over the held-out
regions, `calib.run`) and writes them into the checkpoint's `temps`. No weight moves: a calibrated
checkpoint and an uncalibrated one produce the same logits, and inference divides by the temperature of
the rung it is predicting at. A rung whose target is a pooled FRACTION rather than a binary band is
reported and left alone -- a temperature fitted against a fraction is not a calibration.
"""


def _prepared_grid(grid, dev, layout):
    """The validation grid as the `(x, t, w, rung)` batches `calib.collect` reads. The cascade channel
    is zero (`prepare`'s default: no oracle coarse target), the deterministic input."""
    from rvsm import model as M, prep
    for item in grid:
        b = item if item["rung"].ndim else prep.batch1(item)
        x, t, w = prep.prepare(b, dev, layout=layout)
        yield x.to(memory_format=M.memfmt()), t, w, int(b["rung"].reshape(-1)[0])


def calibrate(argv):
    """The `calibrate` subcommand. Returns a process exit code."""
    import torch

    from rvsm import axis as AX, calib as CAL, infer, model as M, sample

    f = _flags(argv, CALIB_USAGE)
    if "help" in f or "h" in f or not f:
        print(CALIB_USAGE, end="")
        return 0

    def one(name, default=None, cast=str):
        v = f.get(name)
        return default if not v else cast(v[0])

    out, ckpt = one("out"), one("ckpt")
    for nm, v in (("out", out), ("ckpt", ckpt)):
        if not v:
            raise SystemExit(f"rvsm calibrate: --{nm} is required\n\n{CALIB_USAGE}")
    dev = torch.device(one("device", None) or ("cuda" if torch.cuda.is_available() else "cpu"))
    st, cfg, layout, sd, temps, step = infer.load_student_ckpt(ckpt, map_location="cpu")
    from dataclasses import replace
    cfg = replace(cfg, out=str(out))
    round_ = int(one("round", _run_round(out), int))
    ax = AX.load(cfg.umbilicus, ct=cfg.ct)
    held = heldout_regions(out, cfg, ax=ax)
    if not held:
        raise SystemExit(f"rvsm calibrate: no held-out regions for {out}")
    grid = sample.val_grid(cfg, held, root=str(out), ct=cfg.ct, ax=ax, round_=round_,
                           limit=int(one("limit", 8, int)))
    net = M.build(cfg.size, cin=layout.cin, cout=layout.cout, gn_bf16=cfg.gn_bf16, verbose=False).to(dev)
    net.load_state_dict(sd)
    net.eval()
    res = CAL.run(net, _prepared_grid(grid, dev, layout), layout=layout,
                  all_rungs="all-rungs" in f)
    for row in res["rungs"]:
        print(f"[calibrate] rung {row['rung']:>2}: T {row['T']:.4f}  bce {row['bce_T1']:.6f} -> "
              f"{row['bce_T']:.6f}  binary_frac {row['binary_frac']:.3f}"
              + ("" if row.get("binary") else "   (pooled fraction: not calibrated)"), flush=True)
    if "dry" in f:
        print("[calibrate] --dry: the checkpoint was not touched", flush=True)
        return 0
    st["temps"] = {int(k): float(v) for k, v in res["temps"].items()}
    tmp = str(ckpt) + ".tmp"
    torch.save(st, tmp)
    os.replace(tmp, str(ckpt))
    print(f"[calibrate] wrote temps {st['temps']} into {ckpt} (step {step})", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# `rvsm ledger`
# --------------------------------------------------------------------------- #
LEDGER_USAGE = """rvsm ledger --out DIR [--rebuild] [--json PATH]

Counts the region stores on disk, per round and per channel: a store is DONE iff its `zarr.json` says
so, which is the only region state rvsm keeps. `--rebuild` is the same scan -- there is no index to
rebuild, which is the point of deriving the state from the directory -- and additionally lists the
regions whose recto is done but whose verso (or distance fields) are not, which is exactly the work the
producer still owes. Coarse coverage is the fraction of each `coarse.zarr/coverage/<k>` that has been
fed.
"""


def ledger(argv):
    """The `ledger` subcommand. Returns a process exit code."""
    import glob
    import json
    import re

    import numpy as np

    from rvsm import regions as RG, stores

    f = _flags(argv, LEDGER_USAGE)
    if "help" in f or "h" in f or not f:
        print(LEDGER_USAGE, end="")
        return 0
    out = (f.get("out") or [""])[0]
    if not out:
        raise SystemExit(f"rvsm ledger: --out DIR is required\n\n{LEDGER_USAGE}")
    root = os.path.join(str(out), "stores")
    rounds = sorted(glob.glob(os.path.join(root, "round_*")),
                    key=lambda p: int(re.search(r"round_(\d+)$", p).group(1)))
    res = {"out": str(out), "rounds": {}}
    for rd in rounds:
        r = int(re.search(r"round_(\d+)$", rd).group(1))
        chans = sorted(d for d in os.listdir(rd) if os.path.isdir(os.path.join(rd, d)))
        row, done_by_chan = {}, {}
        for ch in chans:
            paths = sorted(glob.glob(os.path.join(rd, ch, "region_*.zarr")))
            done = [p for p in paths if stores.is_done(p)]
            done_by_chan[ch] = {os.path.basename(p) for p in done}
            e = {"stores": len(paths), "done": len(done), "partial": len(paths) - len(done)}
            cov = []
            for k in RG.COARSE_RUNGS:
                a = RG._coarse_array(out, ch, k, r, coverage=True, create=False)
                if a is not None:
                    cov.append((int(k), float((np.asarray(a[:]) > 0).mean())))
            if cov:
                e["coarse_coverage"] = {str(k): round(v, 4) for k, v in cov}
            row[ch] = e
        res["rounds"][str(r)] = row
        print(f"round {r}:")
        for ch in chans:
            e = row[ch]
            print(f"  {ch:>10}  {e['done']:>6} done"
                  + (f", {e['partial']} unfinished" if e["partial"] else "")
                  + ("   coarse " + " ".join(f"r{k}={v:.2f}" for k, v in e["coarse_coverage"].items())
                     if e.get("coarse_coverage") else ""))
        if "rebuild" in f and done_by_chan:
            base = done_by_chan.get("recto", set())
            for ch in chans:
                if ch == "recto":
                    continue
                owed = sorted(base - done_by_chan[ch])
                if owed:
                    print(f"  owed {ch}: {len(owed)} region(s)"
                          + (f" e.g. {owed[0]}" if owed else ""))
                    row[ch]["owed"] = len(owed)
    if not rounds:
        print(f"ledger: no stores under {root}")
    jp = (f.get("json") or [None])[0]
    if jp:
        os.makedirs(os.path.dirname(str(jp)) or ".", exist_ok=True)
        with open(jp, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"[ledger] wrote {jp}")
    return 0


# --------------------------------------------------------------------------- #
# `rvsm umbilicus`
# --------------------------------------------------------------------------- #
UMB_USAGE = """rvsm umbilicus --ct URL|PATH --out PATH [--rung 9] [--step 1] [--thresh 0]

Derives the scroll axis from the CT itself -- the per-z centroid of the non-air voxels at a coarse rung
-- and writes it as the loader's umbilicus json, in RUNG-2 voxels. This is what `umbilicus = "auto"`
does inside a run; having it as a command means the axis can be derived once, inspected, corrected by
hand, and then pinned in the config.
"""


def umbilicus(argv):
    """The `umbilicus` subcommand. Returns a process exit code."""
    from rvsm import axis as AX

    f = _flags(argv, UMB_USAGE)
    if "help" in f or "h" in f or not f:
        print(UMB_USAGE, end="")
        return 0

    def one(name, default=None, cast=str):
        v = f.get(name)
        return default if not v else cast(v[0])

    ct, out = one("ct"), one("out")
    for nm, v in (("ct", ct), ("out", out)):
        if not v:
            raise SystemExit(f"rvsm umbilicus: --{nm} is required\n\n{UMB_USAGE}")
    rung = int(one("rung", 9, int))
    pts = AX.derive(ct, rung=rung, thresh=int(one("thresh", 0, int)), step=int(one("step", 1, int)))
    p = AX.write(out, pts)
    print(f"[umbilicus] {len(pts)} control points from rung {rung} of {ct} -> {p}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# `rvsm pretrain`
# --------------------------------------------------------------------------- #
PRETRAIN_USAGE = """rvsm pretrain cfg.toml [--out DIR] [--steps N] [--device cuda] [--no-mask-ctx]
                    [--block 32] [--mask-lo 0.5] [--mask-hi 0.75] [--sheet-p 0.5] [--loss l1|l2]

Label-free masked-cube pretraining of the SAME trunk `rvsm train` uses: the sampler draws windows with
no store at all (`Patches(label_free=True)`), a large fraction of the CT cube is blanked in blocks (and
the same footprint blanked in every context cube, at its own scale, so the coarse channels are not a
free answer key), and the trunk is asked to put the z-scored CT back.

The checkpoint stores the reconstruction head under `recon_head.*`, so

    rvsm pretrain cfg.toml --out pre/ && rvsm train cfg.toml --init pre/ckpt.pt

warm-starts the trunk and starts the segmentation head fresh: `train.warm_start` copies every trunk
tensor by name and reports the head as new (it gets the boosted learning rate).
"""


def pretrain(argv):
    """The `pretrain` subcommand. Returns a process exit code."""
    from rvsm import config as CFG, pretrain as PT

    argv = list(argv)
    cut = next((i for i, a in enumerate(argv) if str(a).startswith("--")), len(argv))
    pos, f = argv[:cut], _flags(argv[cut:], PRETRAIN_USAGE)
    if "help" in f or "h" in f:
        print(PRETRAIN_USAGE, end="")
        return 0
    over = {}
    if f.get("out"):
        over["out"] = str(f["out"][0])
    if f.get("steps"):
        over["steps"] = int(f["steps"][0])
    cfg = CFG.load(str(pos[0]) if pos else None, overrides=over)

    def one(name, default, cast):
        v = f.get(name)
        return default if not v else cast(v[0])

    ck = PT.pretrain(cfg, out=cfg.out, device=(f["device"][0] if f.get("device") else None),
                     block=int(one("block", PT.BLOCK, int)),
                     lo=float(one("mask-lo", PT.MASK_LO, float)),
                     hi=float(one("mask-hi", PT.MASK_HI, float)),
                     sheet_p=float(one("sheet-p", PT.SHEET_P, float)),
                     mask_ctx="no-mask-ctx" not in f, loss=str(one("loss", "l1", str)))
    print(f"[pretrain] wrote {ck}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# `rvsm ladder` / `rvsm ladder-report`
# --------------------------------------------------------------------------- #
LADDER_USAGE = """rvsm ladder cfg.toml [--sizes 15m,30m6,60m] [--steps N] [--dry] [--out-root DIR]
                  [--lr-scale same|mup] [--prefix p] [--base-out DIR]
rvsm ladder-report DIR [DIR ...] [--metric dice] [--step N] [--rungs 2,3] [--json PATH]

Experiment 12, the size ladder: three `rvsm train` runs differing ONLY in `size`, at matched steps, on
the SAME stores in the same walk order (each rung's `<out>/stores` is a symlink to the base run's), run
SEQUENTIALLY. `--dry` prints the three configs and the commands and touches nothing.

`ladder-report` then fits `1 - metric` against `log(params)` per rung and prints the slope beside the
train/val gap trend and each run's own convergence fit: saturation is a slope that flattens across >= 3
sizes AND a train/val gap that grows -- either alone means something else.
"""


def ladder(argv):
    """The `ladder` subcommand. Returns a process exit code."""
    from rvsm import config as CFG, sizeladder as SL

    argv = list(argv)
    cut = next((i for i, a in enumerate(argv) if str(a).startswith("--")), len(argv))
    pos, f = argv[:cut], _flags(argv[cut:], LADDER_USAGE)
    if "help" in f or "h" in f:
        print(LADDER_USAGE, end="")
        return 0

    def one(name, default=None, cast=str):
        v = f.get(name)
        return default if not v else cast(v[0])

    over = {}
    if f.get("steps"):
        over["steps"] = int(f["steps"][0])
    cfg = CFG.load(str(pos[0]) if pos else None, overrides=over)
    sizes = [q for q in " ".join(f.get("sizes") or []).replace(",", " ").split() if q] or list(SL.SIZES)
    SL.launch(cfg, sizes=sizes, out_root=one("out-root"), steps=one("steps", None, int),
              lr_scale=one("lr-scale", "same"), prefix=one("prefix", ""),
              base_out=one("base-out", None), dry="dry" in f)
    return 0


def ladder_report(argv):
    """The `ladder-report` subcommand. Returns a process exit code."""
    from rvsm import sizeladder as SL

    argv = list(argv)
    cut = next((i for i, a in enumerate(argv) if str(a).startswith("--")), len(argv))
    pos, f = [str(a) for a in argv[:cut]], _flags(argv[cut:], LADDER_USAGE)
    if "help" in f or "h" in f or not pos:
        print(LADDER_USAGE, end="")
        return 0 if "help" in f or "h" in f else 2

    def one(name, default=None, cast=str):
        v = f.get(name)
        return default if not v else cast(v[0])

    rungs = [int(q) for q in " ".join(f.get("rungs") or []).replace(",", " ").split()] or None
    SL.report(pos, metric=one("metric", "dice"), step=one("step", None, int), rungs=rungs,
              out=one("json", None))
    return 0


# `ladder-report` is not a Python identifier, so it is registered by hand; `main` looks its function up
# in `globals()` by the subcommand name, which a dict lookup handles perfectly well.
COMMANDS = COMMANDS + ("ladder-report",)
globals()["ladder-report"] = ladder_report



# --------------------------------------------------------------------------- #
# `rvsm run`, `rvsm status`, `rvsm stop`: the driver (commit 6)
# --------------------------------------------------------------------------- #
RUN_USAGE = """rvsm run [cfg.toml] [--ct URL|PATH] [--ct-seed LOCAL_MIRROR] [--umbilicus PATH|auto] [--out DIR] [--gpus 0[,1]]
             [--mode auto|resident|timeshare|cpu] [--rounds N] [--steps N] [--size 30m6]
             [--init ckpt.pt] [--device cuda:0] [--backend trt|torch] [--no-producer]

The whole pipeline on one machine: raw CT + umbilicus in, a self-distilled student out.

One supervisor freezes the run (`config.json` + its fingerprint, `umbilicus.json`, `metadata.json`,
the held-out set), places the roles on the cards it finds, spawns ONE producer process and becomes the
trainer. They share nothing but `<out>/`:

  --mode resident    one card >= 70 GB: trainer ~46 GB, producer ~30 GB, both resident
  --mode timeshare   two cards: one role each. One card: the roles alternate through `<out>/PHASE`
                     (`train_min` minutes of training, then the producer drains the window)
  --mode auto        pick from the cards found (the default)
  --mode cpu         no card: the producer runs as a thread in this process (tests, smoke runs)
  --no-producer      trainer only, on whatever stores are already there

The run stops at `--steps`, at `--rounds`, or whenever `<out>/STOP` appears (`rvsm stop --out DIR`):
the trainer finishes the step it is on, evaluates, checkpoints and exits; the producer finishes the
region it is on. Starting again on the same `--out` resumes: the config fingerprint is asserted
against `config.json`, the checkpoint against itself, and the walk cursor comes from `state.json`.
"""


def run(argv):
    """The `run` subcommand. Returns a process exit code."""
    from rvsm import config as CFG, run as RUN

    argv = list(argv)
    cut = next((i for i, a in enumerate(argv) if str(a).startswith("--")), len(argv))
    pos, f = argv[:cut], _flags(argv[cut:], RUN_USAGE)
    if "help" in f or "h" in f:
        print(RUN_USAGE, end="")
        return 0

    over = {}
    for k in ("ct", "umbilicus", "out", "size", "mode", "ct_seed"):
        if f.get(k) or f.get(k.replace("_", "-")):
            over[k] = str((f.get(k) or f.get(k.replace("_", "-")))[0])
    for k in ("rounds", "steps", "workers"):
        if f.get(k):
            over[k] = int(f[k][0])
    if f.get("gpus"):
        over["gpus"] = tuple(int(q) for q in " ".join(f["gpus"]).replace(",", " ").split())
    mode = str(over.get("mode", "")) or None
    if mode == "cpu":                      # `cpu` is a placement, not a config value
        over.pop("mode")
    cfg = CFG.load(str(pos[0]) if pos else None, overrides=over)
    if mode == "cpu":
        from dataclasses import replace
        cfg = replace(cfg, mode="cpu")
    if not cfg.ct:
        raise SystemExit(f"rvsm run: --ct URL|PATH (or a config that names one) is required\n\n"
                         f"{RUN_USAGE}")

    ck = RUN.run(cfg, out=cfg.out, init=(f["init"][0] if f.get("init") else None),
                 device=(f["device"][0] if f.get("device") else None),
                 backend=str(f["backend"][0] if f.get("backend") else "torch"),
                 producer="no-producer" not in f)
    print(f"[run] {ck}", flush=True)
    return 0


STATUS_USAGE = """rvsm status --out DIR [--json]

What a run directory is doing, read from the directory alone: no process is asked anything, so this
works on a run on another host's disk, on a finished run, and on a run whose supervisor has died.
Prints state.json (round, step, cursor, verso gate), the worker heartbeats, the finished stores per
channel and round, the production and training rates, and the last evaluation line.
"""


def status(argv):
    """The `status` subcommand. Returns a process exit code."""
    import json

    from rvsm import run as RUN
    f = _flags(argv, STATUS_USAGE)
    if "help" in f or "h" in f:
        print(STATUS_USAGE, end="")
        return 0
    out = str(f["out"][0]) if f.get("out") else "out"
    if not os.path.isdir(out):
        raise SystemExit(f"rvsm status: no run directory at {out}")
    if "json" in f:
        print(json.dumps(RUN.status(out, log=lambda *_a, **_k: None), indent=1, default=str))
        return 0
    RUN.status(out)
    return 0


STOP_USAGE = """rvsm stop --out DIR

Touch `<out>/STOP`. The trainer finishes the step it is on, evaluates, checkpoints and exits; the
producer finishes the region it is on. Nothing is killed and nothing is lost: `rvsm run` on the same
directory resumes from `state.json` and the checkpoint.
"""


def stop(argv):
    """The `stop` subcommand. Returns a process exit code."""
    from rvsm import run as RUN
    f = _flags(argv, STOP_USAGE)
    if "help" in f or "h" in f:
        print(STOP_USAGE, end="")
        return 0
    out = str(f["out"][0]) if f.get("out") else "out"
    if not os.path.isdir(out):
        raise SystemExit(f"rvsm stop: no run directory at {out}")
    RUN.stop(out)
    return 0

VERSO_USAGE = """rvsm verso hold|release --out DIR [--why TEXT]

`hold` places `<out>/VERSO_HOLD`: the round-0 verso gate still evaluates and logs its streak every
evaluation (sched.jsonl `verso_hold`, with what it would have decided), but `verso_on` stays false.
`release` removes it; the next evaluation decides normally. Takes effect at the next evaluation,
without a restart.
"""


def verso(argv):
    """The `verso` subcommand. Returns a process exit code."""
    from rvsm import run as RUN
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in ("hold", "release"):
        print(VERSO_USAGE, end="")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    f = _flags(argv[1:], VERSO_USAGE)
    out = str(f["out"][0]) if f.get("out") else "out"
    if not os.path.isdir(out):
        raise SystemExit(f"rvsm verso: no run directory at {out}")
    held = RUN.verso_hold(out, argv[0] == "hold", why=" ".join(f.get("why") or ["manual"]))
    print(f"rvsm verso: {'HELD' if held else 'released'} ({os.path.join(out, RUN.VERSO_HOLD_FILE)})")
    return 0


if __name__ == "__main__":       # kept LAST: `main` dispatches on the functions defined above it
    raise SystemExit(main())
