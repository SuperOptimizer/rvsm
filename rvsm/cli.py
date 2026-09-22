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
            "status", "stop", "ledger", "umbilicus", "teachers")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
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


if __name__ == "__main__":       # kept LAST: `main` dispatches on the functions defined above it
    raise SystemExit(main())


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
    pos = [a for a in argv if not str(a).startswith("--")]
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
