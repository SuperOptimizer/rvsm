"""The evaluation-side subcommands: `rvsm eval`, `calibrate`, `ledger`, `umbilicus`.

All four run on the synthetic run directory (`synth_run`: real volcomp region stores written from the
fixture CT) and on the fixture CT itself. Nothing here needs a GPU, a teacher or a trained network.
"""
import json
import os

import numpy as np
import pytest
import torch

from rvsm import cli


def _freeze_config(cfg):
    """Write `<out>/config.json`, which is what a real run directory carries and what `rvsm eval`
    reads to find the CT, the umbilicus and the region geometry."""
    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "config.json"), "w") as f:
        json.dump(cfg.to_json(), f)


def _heldout(root, regions):
    """Pin the held-out set by hand, as `<out>/eval/heldout.json`."""
    os.makedirs(os.path.join(root, "eval"), exist_ok=True)
    with open(os.path.join(root, "eval", "heldout.json"), "w") as f:
        json.dump({"regions": [{"lo": [int(v) for v in r["lo"]],
                                "size": [int(v) for v in r["size"]], "k": 2} for r in regions]}, f)


def test_eval_prints_the_table_and_writes_cis(synth_run, capsys):
    """Round-1 stores against the round-0 reference: the table prints, the json carries CIs, and a
    store compared against itself is perfect."""
    from rvsm import stores

    cfg = synth_run.cfg
    _freeze_config(cfg)
    two = [r for r in synth_run.two][:3]
    assert len(two) >= 2, "the bootstrap needs at least two held-out regions"
    _heldout(synth_run.root, two)

    # a round-1 prediction per held-out region: the reference with one y-row of the slab eaten, so the
    # numbers are neither 1.0 (which any bug would also produce) nor 0.
    for r in two:
        lo = np.array(r["lo"], np.int64)
        ref = np.asarray(stores.open_store(stores.store_path(synth_run.root, "recto", lo, 0))[:],
                         np.uint8)
        pred = ref.copy()
        pred[:, :, ::7] = 0
        stores.write(stores.store_path(synth_run.root, "recto", lo, 1), pred, lo, rung=2,
                     channels=("recto",), q=8)

    jp = os.path.join(synth_run.root, "eval", "r1.json")
    assert cli.main(["eval", "--out", synth_run.root, "--round", "1", "--json", jp,
                     "--boot", "32"]) == 0
    out = capsys.readouterr().out
    assert "store vs the round-0 reference" in out
    assert "dice" in out and "[" in out, out
    res = json.load(open(jp))
    assert res["n_regions"] == len(two)
    assert 0.5 < res["store"]["dice"] < 1.0, res["store"]["dice"]
    lo_, hi_ = res["store_ci"]["dice"]
    assert lo_ <= res["store"]["dice"] + 1e-9 and hi_ >= res["store"]["dice"] - 1e-9
    assert res["store"]["erl_frac"] <= 1.0 and np.isfinite(res["store"]["skel_recall"])
    assert len(res["regions"]) == len(two) and all("vs_store" in q for q in res["regions"])

    # round 0 against itself is the identity: dice 1
    assert cli.main(["eval", "--out", synth_run.root, "--round", "0", "--boot", "8"]) == 0
    perfect = json.load(open(os.path.join(synth_run.root, "eval", "eval_round_0.json")))
    assert perfect["store"]["dice"] == pytest.approx(1.0, abs=1e-6)


def test_eval_derives_and_records_the_heldout_set(synth_run):
    """With no `heldout.json` the same deterministic set the trainer uses is derived -- and written, so
    the next evaluation of this directory scores the same regions."""
    cfg = synth_run.cfg
    _freeze_config(cfg)
    p = os.path.join(synth_run.root, "eval", "heldout.json")
    assert not os.path.exists(p)
    held = cli.heldout_regions(synth_run.root, cfg, ax=synth_run.ax)
    assert held and os.path.exists(p)
    again = cli.heldout_regions(synth_run.root, cfg, ax=synth_run.ax)
    assert [q["lo"] for q in again] == [q["lo"] for q in held]


def test_calibrate_writes_temps_into_the_checkpoint(synth_run, capsys):
    """A 1m checkpoint gains a per-rung `temps` dict, and nothing else about it changes."""
    from rvsm import infer, model as M

    cfg = synth_run.cfg
    _freeze_config(cfg)
    layout = cfg.layout()
    net = M.build(cfg.size, cin=layout.cin, cout=layout.cout, verbose=False)
    ck = os.path.join(synth_run.root, "ckpt", "student.pt")
    infer.save_student(ck, net.state_dict(), cfg, temps={}, step=7)
    before = torch.load(ck, map_location="cpu", weights_only=False)
    assert not before["temps"]

    assert cli.main(["calibrate", "--out", synth_run.root, "--ckpt", ck, "--device", "cpu",
                     "--limit", "2"]) == 0
    after = torch.load(ck, map_location="cpu", weights_only=False)
    assert after["temps"], "no temperature was fitted"
    assert all(0.2 <= float(v) <= 5.0 for v in after["temps"].values()), after["temps"]
    assert int(after["step"]) == 7
    for k, v in before["ema"].items():
        assert torch.equal(v, after["ema"][k]), "calibration moved a weight"
    assert "rung" in capsys.readouterr().out


def test_ledger_counts_match_the_stores_on_disk(synth_run, capsys, tmp_path):
    """The ledger is a directory scan: its counts are exactly the done stores."""
    from rvsm import stores

    n = len(synth_run.lo)
    jp = str(tmp_path / "ledger.json")
    assert cli.main(["ledger", "--out", synth_run.root, "--rebuild", "--json", jp]) == 0
    out = capsys.readouterr().out
    assert "round 0:" in out
    res = json.load(open(jp))
    r0 = res["rounds"]["0"]
    for ch in ("recto", "verso", "rw", "midline", "thickness"):
        assert r0[ch]["done"] == n, (ch, r0[ch])
        assert r0[ch]["partial"] == 0

    # an unfinished store is counted as unfinished and owed
    lo = np.array(synth_run.lo[0], np.int64)
    p = stores.store_path(synth_run.root, "verso", lo, 0)
    d = json.load(open(os.path.join(p, "zarr.json")))
    d["attributes"]["done"] = False
    json.dump(d, open(os.path.join(p, "zarr.json"), "w"))
    assert cli.main(["ledger", "--out", synth_run.root, "--rebuild", "--json", jp]) == 0
    res = json.load(open(jp))
    assert res["rounds"]["0"]["verso"]["done"] == n - 1
    assert res["rounds"]["0"]["verso"]["partial"] == 1
    assert res["rounds"]["0"]["verso"]["owed"] == 1


def test_umbilicus_derives_and_writes_a_valid_json(ct_origin, tmp_path, capsys):
    """`rvsm umbilicus` writes the loader's json, in rung-2 voxels, on the fixture's own axis."""
    from rvsm import axis as AX
    p = str(tmp_path / "umbilicus.json")
    assert cli.main(["umbilicus", "--ct", ct_origin.path, "--out", p]) == 0
    assert "control points" in capsys.readouterr().out
    d = json.load(open(p))
    assert d["control_points"] and {"z", "y", "x"} <= set(d["control_points"][0])
    ax = AX.load(p, ct=ct_origin.path)
    assert ax.shape[0] == 3 and ax.shape[1] >= 2
    c = ct_origin.base / 2.0
    assert np.allclose(ax[1], c, atol=4) and np.allclose(ax[2], c, atol=4), ax[:, :3]
    assert (np.diff(ax[0]) > 0).all(), "the control points must be sorted by z"
