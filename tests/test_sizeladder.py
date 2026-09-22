"""The SIZE ladder (experiment 12): `rvsm ladder --dry` and `rvsm ladder-report`.

(The RUNG ladder -- the CT pyramid -- is `rvsm/ladder.py` and `tests/test_ladder.py`; this file is the
three-sizes-at-matched-steps experiment.)
"""
import json
import os

import numpy as np

from rvsm import cli, config as CFG, sizeladder as SL


def _cfg_toml(tmp_path, cfg):
    return SL.write_config(cfg, str(tmp_path / "base.toml"))


def test_write_config_round_trips(small_cfg, tmp_path):
    """A rung's config file must read back as the very config it was written from -- the ladder's whole
    claim is that only `size` differs."""
    p = _cfg_toml(tmp_path, small_cfg)
    got = CFG.load(p)
    assert got.fingerprint() == small_cfg.fingerprint()
    assert {int(k): float(v) for k, v in got.rung_boost.items()} == \
           {int(k): float(v) for k, v in small_cfg.rung_boost.items()}
    assert got.ctx == small_cfg.ctx and got.rungs == small_cfg.rungs


def test_dry_run_prints_three_configs_differing_only_in_size(small_cfg, tmp_path):
    rows = SL.plan(small_cfg, out_root=str(tmp_path / "ladder"), steps=11)
    assert [r["size"] for r in rows] == list(SL.SIZES)
    assert SL._differ(rows) == ["out", "size"], SL._differ(rows)
    assert all(int(r["cfg"].steps) == 11 for r in rows)
    assert all(r["cfg"].lr == small_cfg.lr for r in rows), "the control arm is the SAME lr"
    assert rows[0]["params"] < rows[1]["params"] < rows[2]["params"]
    lines = []
    got = SL.launch(small_cfg, out_root=str(tmp_path / "ladder"), steps=11, dry=True, log=lines.append)
    assert len(got) == 3
    txt = "\n".join(lines)
    for r in got:
        assert r["out"] in txt and r["size"] in txt
        assert not os.path.exists(r["out"]), "--dry must touch nothing"
    assert txt.count("train") >= 3


def test_cli_ladder_dry(small_cfg, tmp_path, capsys):
    p = _cfg_toml(tmp_path, small_cfg)
    assert cli.main(["ladder", p, "--sizes", "15m,30m6,60m", "--steps", "7", "--dry",
                     "--out-root", str(tmp_path / "L")]) == 0
    out = capsys.readouterr().out
    assert out.count("rvsm.cli train") == 3, out
    for sz in SL.SIZES:
        assert sz in out
    assert "differ in ['out', 'size']" in out, out


def test_lr_for():
    from rvsm import model as M
    assert SL.lr_for("60m", "30m6", 3e-4) == 3e-4
    mup = SL.lr_for("60m", "30m6", 3e-4, "mup")
    assert mup < 3e-4
    assert mup == 3e-4 * float(np.sqrt(M.PRESETS["30m6"][0] / M.PRESETS["60m"][0]))


def _fake_run(d, size, dice_by_rung, steps=(100, 200, 300, 400, 500, 600)):
    """A run directory with the trainer's own logs: `logs/eval.jsonl` and `logs/train.jsonl`."""
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    with open(os.path.join(d, "logs", "eval.jsonl"), "w") as f:
        for i, s in enumerate(steps):
            frac = 1.0 - 0.5 ** (i + 1)
            rec = {"step": s, "bce": 0.5 * (1 - 0.1 * i)}
            for k, v in dice_by_rung.items():
                rec[f"dice_r{k}"] = float(v * frac)
            rec["dice"] = float(np.mean([rec[f"dice_r{k}"] for k in dice_by_rung]))
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(d, "logs", "train.jsonl"), "w") as f:
        for s in steps:
            f.write(json.dumps({"step": s, "bce": 0.2}) + "\n")
    return d


def test_report_fits_finite_slopes(tmp_path, capsys):
    """Three synthetic runs whose rung-2 dice rises with size: the fit is finite and the slope says the
    ladder is still paying."""
    runs = []
    for sz, d2, d3 in (("15m", 0.80, 0.70), ("30m6", 0.86, 0.74), ("60m", 0.90, 0.77)):
        runs.append(_fake_run(str(tmp_path / sz), sz, {2: d2, 3: d3}))
    jp = str(tmp_path / "report.json")
    res = SL.report(runs, out=jp)
    assert res["matched_step"] == 600
    assert set(res["per_rung"]) == {"dice_r2", "dice_r3"}
    for k, f in res["per_rung"].items():
        assert np.isfinite(f["alpha"]) and f["n"] == 3, (k, f)
        assert f["alpha"] > 0, "1 - dice falling with params is a POSITIVE alpha"
        assert np.isfinite(f["r2"]) and f["gap_trend"] is not None
    assert [r["size"] for r in res["runs"]] == ["15m", "30m6", "60m"]
    assert all(r["params"] and r["params"] > 0 for r in res["runs"])
    assert all(r["gap"] is not None for r in res["runs"]), "train/val gap not computed"
    assert json.load(open(jp))["per_rung"]
    out = capsys.readouterr().out
    assert "per-rung log-log fit" in out and "decision rule" in out

    # and through the CLI, with the `rvsm eval --json` dumps folded in
    os.makedirs(os.path.join(runs[0], "eval"), exist_ok=True)
    with open(os.path.join(runs[0], "eval", "eval_step_000600.json"), "w") as f:
        json.dump({"step": 600, "store": {"dice": 0.9, "erl_frac": 0.7}}, f)
    ev = SL.read_evals(runs[0])
    assert ev[-1]["erl_frac"] == 0.7
    assert cli.main(["ladder-report", *runs, "--metric", "dice"]) == 0
    assert "ladder report: 3 runs" in capsys.readouterr().out


def test_report_refuses_a_run_with_no_evals(tmp_path):
    import pytest
    a = _fake_run(str(tmp_path / "15m"), "15m", {2: 0.8})
    b = str(tmp_path / "30m6")
    os.makedirs(b, exist_ok=True)
    with pytest.raises(AssertionError, match="no eval rows"):
        SL.report([a, b])
