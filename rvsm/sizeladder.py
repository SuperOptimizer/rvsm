"""The SIZE LADDER: experiment 12, ported from usrm2's `ladder.py`.

`docs/recipe.md` experiment 12: at rung 2 our unique-window supply means PARAMS, not data, are the
constraint -- so run `15m` / `30m6` / `60m` at MATCHED STEPS on the SAME stores in the SAME walk order,
then fit `1 - metric` against `log(params)` **per rung** and call saturation only when the slope
flattens across >= 3 sizes *and* the train/val gap grows. Two halves:

  * `launch` builds one `rvsm train` invocation per size, differing ONLY in `size` (and the run
    directory the checkpoint lands in). `--dry` prints them; otherwise they run SEQUENTIALLY, one
    process at a time, because two trainers on one card is not the experiment.
  * `report` reads each run's `logs/eval.jsonl` (plus any `rvsm eval --json` dumps beside it), takes
    the matched step, fits the per-rung log-log slope over the sizes and prints it next to the
    train/val gap trend and `evalsurf.fit_curve`'s per-run convergence check.

**The data has to be identical, not merely identically distributed.** A scaling-law fit compares three
numbers a couple of points apart; three different window sequences move a number by more than that. In
rvsm that is free: the walk is deterministic in (region list, `visits_max`, seed) and the region list is
derived from the CT, so three runs whose configs differ only in `size` see the same windows in the same
order -- PROVIDED they read the same stores. `launch` therefore points every rung's `<out>/stores` at
the base run's stores (a symlink) instead of copying them, and refuses to run a rung whose stores
directory would be empty.

**muP.** `docs/research` is explicit: do NOT adopt the muP reparametrisation here (transformer-centric
evidence, GroupNorm already normalises per-layer activation scale). Its one cheap recommendation --
scale a WIDENED layer's LR by 1/sqrt(width ratio) -- is about warm starts, and these rungs are fresh
inits. The default is therefore the SAME LR at every rung, which is also the experiment's own control;
`lr_scale="mup"` is available for anyone who wants that arm.

This module is `sizeladder` and not `ladder` because `rvsm/ladder.py` is the RUNG ladder (the CT
pyramid): two different ladders, two different files, one import that cannot be confused for the other.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

# The ladder, coarsest-first in parameters: `model.PRESETS` widths scaled by 1/sqrt(2) and sqrt(2),
# which is a factor-2 ladder in parameters.
SIZES = ("15m", "30m6", "60m")

# Fields a ladder rung is allowed to differ from the base config in. Anything else differing means the
# experiment is not measuring the size.
LADDER_DIFFERS = ("size", "out", "steps", "lr")


def params(size, cfg=None):
    """Parameter count of a preset at a config's own `cin` / `cout` (the numbers the report fits)."""
    from rvsm import model as M
    if cfg is None:
        return M.params(str(size))
    lay = cfg.layout()
    return M.params(str(size), cin=lay.cin, cout=lay.cout)


def lr_for(size, base_size, lr, mode="same"):
    """The LR of one rung: "same" (the default, and the experiment's control) or "mup" -- 1/sqrt of the
    width ratio, the one width-scaling rule the optimisation literature endorses for this setting."""
    from rvsm import model as M
    if str(mode) == "same":
        return float(lr)
    assert str(mode) == "mup", f"lr_scale {mode!r}: 'same' or 'mup'"
    return float(lr) * float(np.sqrt(M.PRESETS[str(base_size)][0] / M.PRESETS[str(size)][0]))


# ------------------------------------------------------------------------------- the config file


def _toml_value(v):
    """One config value as TOML. Dict keys are quoted, so `{2: 2}` comes back as `{"2": 2}` -- which
    every consumer coerces with `int(k)` (`regions.rung_probs`), so the round trip is lossless."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(q) for q in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{json.dumps(str(k))} = {_toml_value(q)}" for k, q in v.items()) + "}"
    return json.dumps(str(v))


def write_config(cfg, path):
    """Write `cfg` as a TOML `rvsm train` can load, and VERIFY it: the file is read back and its
    fingerprint compared against the config it was written from. A ladder whose rungs silently differ
    in something other than `size` measures nothing, so a mismatch is an error here and not a surprise
    twenty GPU-hours later."""
    from rvsm import config as CFG
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{k} = {_toml_value(v)}\n" for k, v in asdict(cfg).items())
    p.write_text("# written by `rvsm ladder`: one rung of the size ladder.\n" + body)
    got = CFG.load(str(p))
    assert got.fingerprint() == cfg.fingerprint(), \
        f"{p}: the written config does not read back identically ({got.fingerprint()} != {cfg.fingerprint()})"
    return str(p)


def _link_stores(src, dst):
    """Point `<dst>/stores` at `<src>/stores` (a symlink, never a copy: a round of stores is ~TB) and
    carry the run's umbilicus and metadata across so the rung resolves the same axis."""
    os.makedirs(dst, exist_ok=True)
    for name in ("stores", "ct", "umbilicus.json", "metadata.json", "eval"):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.exists(s) and not os.path.lexists(d):
            os.symlink(os.path.abspath(s), d)


# ------------------------------------------------------------------------------------ the launch


def plan(cfg, sizes=SIZES, out_root=None, steps=None, lr_scale="same", base_size="30m6", prefix=""):
    """The ladder as data: `[{size, out, cfg, params, lr}, ...]`, one entry per size.

    `cfg` is the BASE config (the one whose `out` holds the stores every rung reads). Each rung's
    config is that one with `size`, `out` and `lr` replaced and nothing else -- which `write_config`
    then proves by fingerprint.
    """
    root = Path(out_root or (Path(cfg.out) / "ladder"))
    rows = []
    for sz in [str(q) for q in sizes]:
        d = root / f"{prefix}{sz}"
        c = replace(cfg, size=sz, out=str(d), lr=lr_for(sz, base_size, cfg.lr, lr_scale),
                    **({"steps": int(steps)} if steps else {}))
        rows.append({"size": sz, "out": str(d), "cfg": c, "lr": c.lr, "params": params(sz, c),
                     "cfg_path": str(d / "config.toml")})
    return rows


def _differ(rows):
    """The config fields that are not identical across the ladder's rungs -- the experiment's own
    self-check. Must be a subset of `LADDER_DIFFERS`."""
    ds = [asdict(r["cfg"]) for r in rows]
    return sorted(k for k in ds[0] if any(json.dumps(d[k], default=str, sort_keys=True)
                                          != json.dumps(ds[0][k], default=str, sort_keys=True)
                                          for d in ds[1:]))


def launch(cfg, sizes=SIZES, out_root=None, steps=None, lr_scale="same", base_size="30m6", prefix="",
           dry=True, log=print, base_out=None):
    """Run (or, with `dry`, print) the ladder. Returns the rows `plan` built, each with its `cmd`.

    Sequential by default and by design: the rungs share one card and one store tree, and a
    side-by-side run would measure the contention, not the size.
    """
    rows = plan(cfg, sizes=sizes, out_root=out_root, steps=steps, lr_scale=lr_scale,
                base_size=base_size, prefix=prefix)
    bad = [k for k in _differ(rows) if k not in LADDER_DIFFERS]
    assert not bad, f"the ladder's rungs differ in {bad}, not only in {list(LADDER_DIFFERS)}"
    src = str(base_out or cfg.out)
    for r in rows:
        r["cmd"] = [sys.executable, "-m", "rvsm.cli", "train", r["cfg_path"]]
    log(f"# size ladder: {len(rows)} rungs, {int(rows[0]['cfg'].steps)} steps each, stores from "
        f"{os.path.join(src, 'stores')}")
    log(f"# the rungs differ in {_differ(rows)} and in nothing else")
    for r in rows:
        log(f"# {r['size']:>6}: {r['params']:,} params, lr {r['lr']:g} -> {r['out']}")
        log(" ".join(r["cmd"]))
    if dry:
        return rows
    if not os.path.isdir(os.path.join(src, "stores")):
        raise SystemExit(f"rvsm ladder: no stores under {os.path.join(src, 'stores')}; the ladder "
                         "trains three sizes on EXISTING stores (run `rvsm run` or `rvsm produce` first)")
    for r in rows:
        _link_stores(src, r["out"])
        write_config(r["cfg"], r["cfg_path"])
        log(f"[ladder] {r['size']}: {' '.join(r['cmd'])}", )
        p = subprocess.run(r["cmd"])
        r["returncode"] = int(p.returncode)
        if p.returncode != 0:
            raise SystemExit(f"rvsm ladder: the {r['size']} rung exited {p.returncode}")
    return rows


# ------------------------------------------------------------------------------------ the report


def read_evals(run_dir):
    """`[{step, metric: value, ...}]` from a run's `logs/eval.jsonl`, merged by step with any
    `rvsm eval --json` dumps beside it (`<run>/eval/*.json`, `<run>/eval/*/*.json`): the surface
    metrics (recall@4, ERL, betti0_err) are what experiment 12's decision rule really wants and they
    live in those dumps, not in the trainer's own log."""
    rows = {}
    for f in (os.path.join(run_dir, "logs", "eval.jsonl"), os.path.join(run_dir, "eval.jsonl")):
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("step") is not None:
                    rows.setdefault(int(d["step"]), {}).update(
                        {k: v for k, v in d.items() if isinstance(v, (int, float))})
    for g in sorted(glob.glob(os.path.join(run_dir, "eval", "*.json"))
                    + glob.glob(os.path.join(run_dir, "eval", "*", "*.json"))):
        try:
            with open(g) as fh:
                j = json.load(fh)
        except Exception:  # noqa: BLE001  -- a half-written dump must not sink the report
            continue
        if not isinstance(j, dict):
            continue
        st = j.get("step")
        flat = {}
        for k in ("store", "mesh", "metrics", "pooled"):
            if isinstance(j.get(k), dict):
                flat.update({a: b for a, b in j[k].items() if isinstance(b, (int, float))})
        flat.update({k: v for k, v in j.items() if isinstance(v, (int, float)) and k != "step"})
        if st is not None and flat:
            rows.setdefault(int(st), {}).update(flat)
    return [dict(v, step=k) for k, v in sorted(rows.items())]


def train_gap(run_dir, step, metric="bce", win=10):
    """`{train, val, gap}` of `metric` at a step: the median of the last `win` `logs/train.jsonl`
    records at or before the step against the eval value there. The gap is val - train, so it GROWS
    when the bigger model starts memorising -- the second half of experiment 12's decision rule."""
    tr = []
    f = os.path.join(run_dir, "logs", "train.jsonl")
    if os.path.exists(f):
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("step") is not None and int(d["step"]) <= int(step) and metric in d:
                    tr.append(float(d[metric]))
    ev = [r for r in read_evals(run_dir) if int(r["step"]) <= int(step) and metric in r]
    if not tr or not ev:
        return None
    a, b = float(np.median(tr[-int(win):])), float(ev[-1][metric])
    return {"train": a, "val": b, "gap": b - a}


def loglog_slope(par, loss):
    """Least-squares slope of log(loss) against log(params): the scaling exponent alpha in
    `loss ~ params^-alpha`. A FLAT alpha is the "params are no longer the constraint" reading; a steep
    one says the ladder is still paying."""
    p, y = np.asarray(par, float), np.asarray(loss, float)
    k = np.isfinite(p) & np.isfinite(y) & (p > 0) & (y > 0)
    p, y = np.log(p[k]), np.log(y[k])
    if len(p) < 2:
        return {"alpha": None, "n": int(len(p)), "error": "need at least 2 rungs"}
    A = np.stack([p, np.ones_like(p)], 1)
    (b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([b, c])
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"alpha": float(-b), "intercept": float(c), "n": int(len(p)),
            "r2": float(1.0 - np.sum((y - pred) ** 2) / ss) if ss > 0 else 1.0}


DECISION = (
    "  decision rule (experiment 12): saturation is BOTH a slope that flattens across >= 3 sizes AND a\n"
    "  train/val gap that grows. A flat alpha with a flat gap means the ladder is simply not the binding\n"
    "  constraint yet; a steep alpha with a growing gap means more params AND more data.")


def _size_of(run_dir):
    """The preset a run was trained at, from its checkpoint (falling back to the directory name)."""
    ck = os.path.join(run_dir, "ckpt.pt")
    if os.path.exists(ck):
        try:
            import torch
            st = torch.load(ck, map_location="cpu", weights_only=False)
            raw = st.get("cfg") or {}
            if isinstance(raw.get("config"), dict):   # `train.save` nests it under `{config, layout}`
                st.setdefault("layout", raw.get("layout"))
                raw = raw["config"]
            if isinstance(raw, dict) and raw.get("size"):
                return str(raw["size"]), raw, st.get("layout")
        except Exception:  # noqa: BLE001  -- a truncated checkpoint must not sink the report
            pass
    base = os.path.basename(os.path.normpath(run_dir))
    return (base if base in _PRESET_NAMES() else None), {}, None


def _PRESET_NAMES():  # noqa: N802  -- a lazy constant: importing the model costs torch
    from rvsm import model as M
    return set(M.PRESETS)


def _params_of(size, raw, layout):
    from rvsm import model as M
    if not size:
        return None
    cin = int((layout or {}).get("cin", 0)) or None
    cout = int((layout or {}).get("cout", 0)) or None
    if cin is None:
        from rvsm.config import Config, _coerce, _TYPES
        cfg = Config(**{k: _coerce(k, v) for k, v in (raw or {}).items() if k in _TYPES})
        lay = cfg.layout()
        cin, cout = lay.cin, lay.cout
    return M.params(str(size), cin=cin, cout=cout)


def report(runs, metric="dice", step=None, rungs=None, out=None, smooth=5, log=print):
    """The ladder report.

    Per rung of the LADDER (`dice_r2`, `dice_r3`, ...) it prints the per-size value at the MATCHED step
    (the largest step every run reached, which is the only honest comparison), the fitted log-log slope
    over the sizes and the train/val gap trend; per RUN it prints `evalsurf.fit_curve` on that run's own
    metric history, so a rung that has simply not converged cannot be read as a plateau.
    """
    from rvsm import evalsurf as E
    info = []
    for d in [str(q) for q in runs]:
        size, raw, layout = _size_of(d)
        ev = read_evals(d)
        info.append({"run": d, "size": size, "evals": ev, "params": _params_of(size, raw, layout),
                     "last_step": int(ev[-1]["step"]) if ev else 0})
    assert info, "ladder report: no runs"
    empty = [q["run"] for q in info if not q["evals"]]
    assert not empty, f"no eval rows in {empty}"
    S = int(step) if step is not None else min(q["last_step"] for q in info)
    keys = ([f"dice_r{int(k)}" for k in rungs] if rungs else
            sorted({k for q in info for r in q["evals"] for k in r
                    if isinstance(k, str) and k.startswith("dice_r")}, key=lambda s: int(s[6:])))
    keys = keys or [metric]
    res = {"matched_step": S, "metric": metric, "runs": [], "per_rung": {}}
    log(f"ladder report: {len(info)} runs, matched at step {S}")
    for q in info:
        rows = [r for r in q["evals"] if int(r["step"]) <= S]
        q["at"] = rows[-1] if rows else {}
        q["gap"] = train_gap(q["run"], S)
        q["fit"] = (E.fit_curve([r["step"] for r in q["evals"]],
                                [r.get(metric, np.nan) for r in q["evals"]], smooth=smooth)
                    if len(q["evals"]) >= 4 else {"model": None, "error": "fewer than 4 evals"})
        res["runs"].append({"run": q["run"], "size": q["size"], "params": q["params"],
                            "step": int(q["at"].get("step", 0)), "metric": q["at"].get(metric),
                            "gap": q["gap"], "converged": q["fit"]})
        g, fit = q["gap"], q["fit"]
        log(f"  {str(q['size']):>6} {(q['params'] or 0) / 1e6:8.2f} M params  "
            f"step {int(q['at'].get('step', 0)):>7}  {metric} "
            f"{float(q['at'].get(metric, float('nan'))):.4f}"
            + (f"  train/val {g['train']:.4f}/{g['val']:.4f} gap {g['gap']:+.4f}" if g else "")
            + (f"  [slope {fit['slope_per_10k']:+.4f}/10k, {fit['steps_to_95']:.0f} steps to 95 %]"
               if fit.get("model") else "  [not enough evals to fit a plateau]"))
    order = sorted(info, key=lambda q: (q["params"] or 0))
    log(f"\n  per-rung log-log fit of (1 - {metric.replace('dice', 'value')}) vs params")
    for key in keys:
        par = [q["params"] for q in order if q["at"].get(key) is not None and q["params"]]
        val = [1.0 - float(q["at"][key]) for q in order if q["at"].get(key) is not None and q["params"]]
        if len(par) < 2:
            continue
        f = loglog_slope(par, val)
        gaps = [q["gap"]["gap"] for q in order if q["gap"]]
        f["gap_trend"] = float(gaps[-1] - gaps[0]) if len(gaps) >= 2 else None
        f["values"] = {str(q["size"]): float(q["at"][key]) for q in order if q["at"].get(key) is not None}
        res["per_rung"][key] = f
        log(f"    {key:>10}: alpha {f['alpha']:+.4f}  r2 {f['r2']:.3f}  over {f['n']} sizes  "
            + "  ".join(f"{k}={v:.4f}" for k, v in f["values"].items())
            + (f"  | train/val gap trend {f['gap_trend']:+.4f}" if f["gap_trend"] is not None else ""))
    log("\n" + DECISION)
    if out:
        os.makedirs(os.path.dirname(str(out)) or ".", exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)
    return res
