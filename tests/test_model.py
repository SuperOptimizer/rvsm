"""The student net: every preset builds, the forward keeps the stem/head contract, up2x is interpolate."""
import torch
import torch.nn.functional as F

from rvsm import model
from rvsm.config import Config


def test_presets_build_and_count():
    L = Config().layout()
    n = {}
    for s in model.PRESETS:
        n[s] = model.params(s, L.cin, L.cout)
        assert n[s] > 0
    # the ladder is a clean factor-2 in parameters: that is what a log-log fit of loss vs log(params)
    # needs, and it is the only reason the widths are what they are
    assert 1.9 < n["30m6"] / n["15m"] < 2.1
    assert 1.9 < n["60m"] / n["30m6"] < 2.1


def test_forward_shape_matches_layout():
    L = Config().layout()
    assert (L.cin, L.cout) == (21, 14)
    net = model.build("1m", cin=L.cin, cout=L.cout, verbose=False).eval()
    x = torch.zeros(1, L.cin, 32, 32, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, L.cout, 32, 32, 32)
    assert len(L.head_names()) == L.cout and L.head_names()[:2] == ["recto", "verso"]


def test_params_allocates_nothing():
    """`params` builds on the meta device, so a 60m count costs no memory and no time."""
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    model.params("60m", 21, 14)
    assert (torch.cuda.memory_allocated() if torch.cuda.is_available() else 0) == before


def test_up2x_matches_interpolate():
    x = torch.randn(2, 3, 5, 6, 7)
    s = tuple(2 * v for v in x.shape[2:])
    a = model.up2x(x, s)
    b = F.interpolate(x, size=s, mode="trilinear", align_corners=False)
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
    # a non-2x ratio falls through to interpolate itself, unchanged
    odd = (7, 9, 11)
    assert torch.equal(model.up2x(x, odd), F.interpolate(x, size=odd, mode="trilinear",
                                                         align_corners=False))


def test_deep_heads_and_ckpt_act():
    net = model.build("1m", cin=4, cout=2, ckpt_act=2, deep=2, verbose=False)
    net.train()
    y = net(torch.zeros(1, 4, 16, 16, 16))
    assert isinstance(y, list) and len(y) == 3
    assert [tuple(q.shape[2:]) for q in y] == [(16,) * 3, (8,) * 3, (4,) * 3]
    net.eval()
    assert torch.is_tensor(net(torch.zeros(1, 4, 16, 16, 16)))
