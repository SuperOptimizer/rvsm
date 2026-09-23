"""Temperature calibration: does the golden-section fit recover a temperature we planted?"""
import pytest
import torch

from rvsm import calib
from rvsm.config import Config


def _planted(T, n=20000, seed=0):
    """Logits whose CALIBRATED probability generated the labels: the true temperature is `T`, so
    `sigmoid(logit / T)` is the honest probability and `fit_temp` must find it."""
    g = torch.Generator().manual_seed(seed)
    z = torch.empty(1, 1, n, 1, 1).uniform_(-6, 6, generator=g)
    p = torch.sigmoid(z / T)
    t = (torch.rand(z.shape, generator=g) < p).float()
    return z, t, torch.ones_like(t)


@pytest.mark.parametrize("T", [0.5, 1.0, 1.7, 3.0])
def test_fit_temp_recovers_a_planted_temperature(T):
    z, t, w = _planted(T)
    fit = calib.fit_temp(z, t, w)
    assert fit == pytest.approx(T, rel=0.12)
    # and the fitted temperature really is better than no calibration at all
    assert calib.bce_at(z, t, w, fit) <= calib.bce_at(z, t, w, 1.0) + 1e-9


def test_bce_at_and_temp_lookup():
    z, t, w = _planted(2.0, n=200)
    assert calib.bce_at(z, t, w, 1.0) > 0
    assert calib.temp_for({"temps": {"3": 1.25}}, 3) == 1.25
    assert calib.temp_for({"temps": {"3": 1.25}}, 2) == 1.0        # no entry -> no calibration
    assert calib.temp_for({"temps": {"3": 1.25}}, 3, use=False) == 1.0
    assert calib.temps_of({"temps": {"2": 1.1, "3": 0.9}}) == {2: 1.1, 3: 0.9}


def test_binary_frac_separates_a_hard_band_from_a_pooled_fraction():
    hard = torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 1, 4, 1, 1)
    soft = torch.tensor([0.25, 0.5, 0.75, 0.5]).view(1, 1, 4, 1, 1)
    w = torch.ones_like(hard)
    assert calib.binary_frac(hard, w) == 0.0
    assert calib.binary_frac(soft, w) == 1.0
    half = torch.tensor([0.0, 1.0, 0.5, 0.5]).view(1, 1, 4, 1, 1)
    assert calib.binary_frac(half, w) == pytest.approx(0.5)
    assert calib.binary_frac(half, torch.tensor([1.0, 1.0, 0.0, 0.0]).view(1, 1, 4, 1, 1)) == 0.0


class _Net(torch.nn.Module):
    """A net whose logits are a fixed temperature times the target's own logit: `run` must report that
    temperature at the rung the batch came from."""

    def __init__(self, cout, T):
        super().__init__()
        self.cout, self.T = cout, T

    def forward(self, x):
        z = x[:, :1] * float(self.T)
        return torch.cat([z] * self.cout, 1)


def test_run_fits_one_temperature_per_rung_and_skips_the_pooled_rungs():
    lay = Config(channels=("recto", "verso"), aff_offsets=(8,)).layout()
    torch.manual_seed(0)
    grid = []
    for rung, T, soft in ((2, 2.0, False), (5, 1.0, True)):
        for _ in range(4):
            z = torch.empty(1, 1, 8, 8, 8).uniform_(-6, 6)
            p = torch.sigmoid(z)
            t = torch.rand_like(p) if soft else (torch.rand_like(p) < p).float()
            x = torch.cat([z, torch.zeros(1, lay.cin - 1, 8, 8, 8)], 1)
            tt = torch.cat([t] + [torch.zeros_like(t)] * (lay.cout_t - 1), 1)
            grid.append((x, tt, torch.ones_like(tt), rung))
    rep = calib.run(_Net(lay.cout, 2.0), grid, layout=lay)
    rows = {r["rung"]: r for r in rep["rungs"]}
    assert set(rows) == {2, 5}
    assert rows[2]["binary"] and rows[2]["T"] == pytest.approx(2.0, rel=0.25)
    assert 2 in rep["temps"]
    assert not rows[5]["binary"] and 5 not in rep["temps"]        # a pooled fraction is not calibrated
    assert "skipped" in rows[5]
    assert 5 in calib.run(_Net(lay.cout, 2.0), grid, layout=lay, all_rungs=True)["temps"]


def test_the_fine_rungs_are_calibrated_even_on_a_soft_teacher_target():
    """A soft teacher probability reads as 'not binary' to `binary_frac` (paris4's rung-2 target:
    0.986), which left temps = {} for a whole run. Rungs 2-4 always get their temperature."""
    lay = Config(channels=("recto", "verso"), aff_offsets=(8,)).layout()
    torch.manual_seed(1)
    grid = []
    for rung in (2, 3, 4, 6):
        z = torch.empty(1, 1, 8, 8, 8).uniform_(-6, 6)
        t = torch.sigmoid(z / 2.0)                                  # a soft probability target
        x = torch.cat([z, torch.zeros(1, lay.cin - 1, 8, 8, 8)], 1)
        tt = torch.cat([t] + [torch.zeros_like(t)] * (lay.cout_t - 1), 1)
        grid.append((x, tt, torch.ones_like(tt), rung))
    rep = calib.run(_Net(lay.cout, 2.0), grid, layout=lay)
    rows = {r["rung"]: r for r in rep["rungs"]}
    assert all(rows[k]["binary_frac"] > calib.BINARY_FRAC for k in (2, 3, 4, 6))
    assert set(rep["temps"]) == {2, 3, 4}                           # 6 is a pooled fraction


def test_collect_drops_weightless_batches():
    lay = Config().layout()
    x = torch.zeros(1, lay.cin, 4, 4, 4)
    t = torch.zeros(1, lay.cout_t, 4, 4, 4)
    per = calib.collect(_Net(lay.cout, 1.0), [(x, t, torch.zeros_like(t), 2)], layout=lay)
    assert per == {}


def test_run_on_kept_logits_is_run_on_the_grid():
    """The trainer calibrates on the logits its evaluation already computed (`keep` / `stack` /
    `run(per=...)`); that must be the same fit as `run` collecting them itself."""
    lay = Config(channels=("recto", "verso"), aff_offsets=(8,)).layout()
    torch.manual_seed(1)
    grid, kept = [], {}
    net = _Net(lay.cout, 2.0)
    for _ in range(4):
        z = torch.empty(1, 1, 8, 8, 8).uniform_(-6, 6)
        t = (torch.rand_like(z) < torch.sigmoid(z)).float()
        x = torch.cat([z, torch.zeros(1, lay.cin - 1, 8, 8, 8)], 1)
        tt = torch.cat([t] + [torch.zeros_like(t)] * (lay.cout_t - 1), 1)
        grid.append((x, tt, torch.ones_like(tt), 2))
        calib.keep(kept, net(x)[:, :lay.cout_t], tt, torch.ones_like(tt), 2)
    a = calib.run(net, grid, layout=lay)
    b = calib.run(None, None, layout=lay, per=calib.stack(kept))
    assert a["temps"] == b["temps"]
