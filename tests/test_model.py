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


def test_gn_bf16_hands_on_bf16_under_autocast_and_changes_nothing_else():
    """`gn_bf16`: the GroupNorm+SiLU output leaves in the autocast dtype; the switch off, or outside
    autocast, it is plain SiLU. The state dict is the same either way (NormAct has no parameters)."""
    x = torch.randn(2, 8, 4, 4, 4)
    a = model.NormAct()
    assert torch.equal(a(x), F.silu(x))
    a.bf16 = True
    assert torch.equal(a(x), F.silu(x))                    # no autocast: float32 stays float32
    with torch.autocast("cpu", torch.bfloat16):
        y = a(x)
    assert y.dtype == torch.bfloat16 and torch.equal(y, F.silu(x).to(torch.bfloat16))
    off = model.build("1m", cin=5, cout=3, verbose=False)
    on = model.build("1m", cin=5, cout=3, gn_bf16=True, verbose=False)
    assert list(off.state_dict()) == list(on.state_dict()) and on.gn_bf16 and not off.gn_bf16
    acts = [m for m in on.modules() if isinstance(m, model.NormAct)]
    assert acts and all(m.bf16 for m in acts)
    model.set_gn_bf16(on, False)
    assert not any(m.bf16 for m in acts) and not on.gn_bf16
    on.load_state_dict(off.state_dict())
    z = torch.randn(1, 5, 16, 16, 16)
    assert torch.equal(on(z), off(z))                      # CPU float32: identical
