"""`rvsm grid-repack`: a spilled validation grid's items, pickle protocol 2 -> 4, in place."""
import json
import os
import subprocess
import sys

import numpy as np
import torch

from rvsm import cli, grid_repack as GR, sample


def _to_p2(path):
    """Rewrite an item file the way `_save_item` did before 2026-09-26 (torch's default protocol 2)."""
    torch.save(torch.load(path, weights_only=False), path)
    assert GR.protocol_of(path) == 2


def _arrays(d, names):
    return [sample._load_item(os.path.join(d, n)) for n in names]


def _equal(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        assert type(a[k]) is type(b[k]), k
        assert torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k])), k


def test_a_repacked_grid_loads_the_same_arrays_is_still_reused_and_a_second_run_is_a_noop(
        synth_run, tmp_path):
    held = [r for r in synth_run.regions if r["k"] == 2][:1]
    kw = dict(root=synth_run.root, ct=synth_run.cfg.ct, ax=synth_run.ax, rungs=(2, 3))
    spill = tmp_path / "grid"
    g = sample.val_grid(synth_run.cfg, held, spill=str(spill), **kw)
    d = os.path.dirname(g.paths[0])
    names = [os.path.basename(p) for p in g.paths]
    # one extra item big enough for Blosc (the fixture's are small): the bytes path is exercised too
    big = sample.rung_item(np.random.default_rng(0).integers(0, 255, (5, 64, 64, 64), dtype=np.uint8),
                           np.zeros((4, 64, 64, 64), np.uint8), np.zeros((4, 64, 64, 64), np.uint8), 2,
                           (0, 0, 0), np.array([[0.0, 4096.0], [32.0, 32.0], [-1000.0, -1000.0]]))
    xd = tmp_path / "extra"
    xd.mkdir()
    sample._save_item(str(xd / "item_r2_big.pt"), big)
    assert torch.load(str(xd / "item_r2_big.pt"), weights_only=False)["ct"][0] == "blosc"
    for p in list(g.paths) + [str(xd / "item_r2_big.pt")]:
        _to_p2(p)
    ref = _arrays(d, names) + _arrays(str(xd), ["item_r2_big.pt"])
    man = os.path.join(d, "grid.json")
    man_bytes, m0 = open(man, "rb").read(), os.path.getmtime(man)

    dry = GR.repack([d, str(xd)], dry_run=True, log=lambda s: None)
    assert dry["would"] == len(names) + 1 and all(GR.protocol_of(p) == 2 for p in g.paths)

    out = GR.repack([d, str(xd)], jobs=2, log=lambda s: None)
    assert out["repacked"] == len(names) + 1 and out["manifest_missing"][d] == 0
    assert out["manifest_files"][d] == len(names)
    assert all(GR.protocol_of(p) == 4 for p in g.paths)
    for a, b in zip(ref, _arrays(d, names) + _arrays(str(xd), ["item_r2_big.pt"])):
        _equal(a, b)
    assert not [n for n in os.listdir(d) if n.startswith(GR.TMP_PREFIX)]
    assert open(man, "rb").read() == man_bytes                 # the manifest is never touched ...
    g2 = sample.val_grid(synth_run.cfg, held, spill=str(spill), **kw)
    assert g2.rebuilt == 0 and list(g2.paths) == list(g.paths)   # ... and still validates: all reused
    assert os.path.getmtime(man) == m0

    again = GR.repack([d, str(xd)], jobs=2, log=lambda s: None)
    assert again["p4"] == len(names) + 1 and "repacked" not in again


def test_the_repack_waits_for_a_live_evaluation_and_ignores_a_stale_marker(synth_run, tmp_path):
    held = [r for r in synth_run.regions if r["k"] == 2][:1]
    kw = dict(root=synth_run.root, ct=synth_run.cfg.ct, ax=synth_run.ax, rungs=(2,))
    g = sample.val_grid(synth_run.cfg, held, spill=str(tmp_path / "grid"), **kw)
    d = os.path.dirname(g.paths[0])
    _to_p2(g.paths[0])
    marks = sample.mark_evaluating(g, step=4000)                # this (live) process is "evaluating"
    assert marks == [os.path.join(d, sample.EVAL_MARKER)]
    busy = GR.repack([d], wait_s=0.0, log=lambda s: None)
    assert busy["busy"] == 1 and GR.protocol_of(g.paths[0]) == 2
    sample.unmark_evaluating(marks)
    assert not os.path.exists(marks[0])
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True).stdout.strip()
    with open(marks[0], "w") as f:                              # a trainer that died mid-evaluation
        json.dump({"pid": int(dead), "host": __import__("socket").gethostname(), "step": 1}, f)
    assert cli.main(["grid-repack", d, "--wait", "0"]) == 0
    assert GR.protocol_of(g.paths[0]) == 4


def test_a_list_grid_gets_no_marker():
    assert sample.mark_evaluating([{"x": 1}], step=1) == []
