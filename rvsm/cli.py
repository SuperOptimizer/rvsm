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
"""

COMMANDS = ("run", "produce", "train", "eval", "export", "calibrate", "pretrain", "ladder",
            "status", "stop", "ledger", "umbilicus")


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


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
# `rvsm produce`: one region, no state (commit 2 = the teacher passes)
# --------------------------------------------------------------------------- #
PRODUCE_USAGE = """rvsm produce --out DIR --ct URL|PATH [--umbilicus PATH] --teacher recto[,m7]
                  --ckpt-recto P [--ckpt-m7 P] --region Z Y X [--size 1024]
                  [--backend trt|torch] [--round 0] [--device cuda] [--tta 1]

Runs the named teacher(s) over ONE 1024^3 region and writes its round-0 stores:

  <out>/stores/round_0/recto/region_<z>_<y>_<x>.zarr   the (fused) sheet probability, q8
  <out>/stores/round_0/rw/region_<z>_<y>_<x>.zarr      the fusion weight (1 - |p_recto - p_m7|), q8

With two teachers the probability is the confidence-weighted fusion of both and `rw` is their
agreement; with one, `rw` is 1 everywhere. No walk, no state.json, no lookahead: this is the single
region a producer, a test or a hand at the terminal asks for.
"""


def _flags(argv):
    """`--k v ...` -> {k: [v, ...]}; a flag with no value gets []."""
    out, k = {}, None
    for a in argv:
        if str(a).startswith("--"):
            k = str(a)[2:]
            out.setdefault(k, [])
        elif k is not None:
            out[k].append(a)
        else:
            raise SystemExit(f"rvsm produce: unexpected argument {a!r}\n\n{PRODUCE_USAGE}")
    return out


def produce(argv):
    """The `produce` subcommand. Returns a process exit code."""
    import numpy as np

    from rvsm import infer, ladder, stores, teachers

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

    probs, win, hal = {}, {}, {}
    for nm in names:
        if nm not in teachers.TEACHERS:
            raise SystemExit(f"rvsm produce: unknown teacher {nm!r} (have {sorted(teachers.TEACHERS)})")
        spec = teachers.TEACHERS[nm]
        ckpt = one(f"ckpt-{nm}", one("ckpt", None))
        if not ckpt:
            raise SystemExit(f"rvsm produce: --ckpt-{nm} PATH is required for teacher {nm!r}")
        w = int(one("window", spec.patch[0], int))
        h = int(one("halo", max(1, w // 8), int))
        print(f"[produce] {nm}: region {tuple(int(v) for v in lo)} size {size3} window {w} halo {h} "
              f"backend {backend}", flush=True)
        probs[nm] = infer.teacher_region(ct, lo, size3, spec, ckpt, device=device, backend=backend,
                                         tta=tta, window=w, halo=h,
                                         engine_dir=os.path.join(str(out), "ckpt", "trt"))
        win[nm], hal[nm] = w, h

    if len(probs) >= 2:
        a, b = names[0], names[1]
        p, rw = infer.fuse_agreement(probs[a], probs[b])
    else:
        p = probs[names[0]]
        rw = np.ones_like(p)

    attrs = {"producer": "teacher:" + ",".join(names), "round": int(round_),
             "ckpt": {nm: str(one(f"ckpt-{nm}", one("ckpt", ""))) for nm in names},
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
