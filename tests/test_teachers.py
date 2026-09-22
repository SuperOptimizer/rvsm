"""The teacher nets: architecture inferred from shapes, strict parity, normalisers, one forward.

The point of these tests is the claim the whole round-0 bootstrap rests on: rvsm can rebuild the
published networks from a checkpoint ALONE, with no config file and no `vesuvius` import, and load them
strictly. A silent key mismatch would be a teacher running with half its weights randomly initialised,
which looks like a plausible probability field and is worthless.
"""
import numpy as np
import pytest
import torch

from rvsm import teachers
from tests.teachers_fixture import PATCH, build_fake, fake_spec


def shape_only(sd):
    """A shape inventory: what a state dict looks like when nobody has the weights."""
    return {k: tuple(v.shape) for k, v in sd.items()}


def nnunet_spec():
    """A tiny stock nnU-Net ResEnc U-Net: 3 stages, transposed-conv decoder with seg layers."""
    spec = teachers.ArchSpec(in_channels=1, features_per_stage=[4, 8, 16], n_blocks_per_stage=[1, 1, 2],
                             strides=[1, 2, 2], kernel_size=3, conv_bias=True, norm="instance")
    spec.decoder = teachers.DecoderSpec(out_channels=2, n_conv_per_stage=[1, 1], residual=False)
    return spec


def test_vesuvius_arch_from_shapes_rebuilds_with_strict_parity():
    net = teachers.VesuviusUNet(fake_spec())
    sd = net.state_dict()
    spec = teachers.infer_vesuvius_arch(shape_only(sd))
    assert spec.in_channels == 1
    assert spec.features_per_stage == [4, 8]
    assert spec.strides == [1, 2]
    assert spec.task_decoders["surface"].out_channels == 2
    rebuilt = teachers.build_from_spec("vesuvius", spec)
    out = rebuilt.load_state_dict(sd, strict=True)
    assert out.missing_keys == [] and out.unexpected_keys == []
    assert set(rebuilt.state_dict()) == set(sd)


def test_nnunet_arch_from_shapes_rebuilds_with_strict_parity():
    net = teachers.NNUNetResEncUNet(nnunet_spec())
    sd = net.state_dict()
    spec = teachers.infer_nnunet_arch(shape_only(sd))
    assert spec.features_per_stage == [4, 8, 16]
    assert spec.n_blocks_per_stage == [1, 1, 2]
    assert spec.strides == [1, 2, 2]              # read off the transposed convs, not assumed
    assert spec.decoder.out_channels == 2
    rebuilt = teachers.build_from_spec("nnunet", spec)
    out = rebuilt.load_state_dict(sd, strict=True)
    assert out.missing_keys == [] and out.unexpected_keys == []
    # the decoder registers the encoder again: those duplicated keys must be present, as upstream has them
    assert any(k.startswith("decoder.encoder.") for k in sd)


def test_scse_and_group_norm_are_inferred():
    spec = teachers.ArchSpec(in_channels=1, features_per_stage=[8, 16], n_blocks_per_stage=[1, 1],
                             strides=[1, 2], norm="group", num_groups=4, squeeze_excitation="scse",
                             task_decoders={"surface": teachers.DecoderSpec(out_channels=2,
                                                                            n_conv_per_stage=[1])})
    sd = teachers.VesuviusUNet(spec).state_dict()
    got = teachers.infer_vesuvius_arch(shape_only(sd), norm_type="group")
    assert got.squeeze_excitation == "scse" and got.norm == "group"
    # instance and group norm have IDENTICAL parameter shapes: only norm_type can tell them apart
    assert teachers.infer_vesuvius_arch(shape_only(sd)).norm == "instance"
    with pytest.raises(ValueError):
        teachers.infer_vesuvius_arch(shape_only(sd), norm_type="none")


def test_strip_prefix_and_shapes_of():
    sd = {"module._orig_mod.a": torch.zeros(2, 3), "b": {"shape": [4, 5]}}
    assert sorted(teachers.strip_prefix(sd)) == ["a", "b"]   # both wrappers come off
    assert teachers.shapes_of({"a": torch.zeros(2, 3), "b": {"shape": [4, 5]}, "c": (6, 7)}) == {
        "a": (2, 3), "b": (4, 5), "c": (6, 7)}


def test_normalizer_modes():
    x = np.array([[0, 100, 200, 255]], np.uint8)
    z = teachers.Normalizer("zscore_instance")(x)
    assert abs(float(z.mean())) < 1e-5 and abs(float(z.std(unbiased=False)) - 1.0) < 1e-5
    n = teachers.TEACHERS["m7"].normalizer
    c = n(x).numpy()
    assert np.allclose(c, (np.clip(x.astype(np.float32), 0, 212) - n.mean) / n.std, atol=1e-5)
    assert float(c.max()) == pytest.approx((212 - n.mean) / n.std, abs=1e-4)  # clipped, not scaled
    assert np.allclose(teachers.Normalizer("div255")(x).numpy(), x / 255.0)
    assert np.allclose(teachers.Normalizer("none")(x).numpy(), x)
    p = teachers.Normalizer("percentile_minmax", lo_pct=0.0, hi_pct=100.0)(x).numpy()
    assert p.min() == 0.0 and p.max() == 1.0
    assert np.allclose(teachers.Normalizer("percentile_minmax")(np.zeros((4,), np.uint8)).numpy(), 0)
    with pytest.raises(ValueError):
        teachers.Normalizer("nope")


def test_apply_activation():
    y = torch.tensor([[[-1.0]], [[1.0]]])[None]              # (1, 2, 1, 1) -> softmax over channels
    s = teachers.apply_activation(y, "softmax")
    assert float(s.sum(1)) == pytest.approx(1.0)
    assert float(teachers.apply_activation(torch.zeros(1, 1), "sigmoid")) == 0.5
    assert float(teachers.apply_activation(torch.tensor([[2.0]]), "clamp01")) == 1.0
    assert float(teachers.apply_activation(torch.tensor([[2.0]]), "none")) == 2.0
    with pytest.raises(ValueError):
        teachers.apply_activation(torch.zeros(1), "softplus")


def test_registry_holds_only_the_two_round0_teachers():
    assert sorted(teachers.TEACHERS) == ["m7", "recto"]
    r, m = teachers.TEACHERS["recto"], teachers.TEACHERS["m7"]
    assert (r.kind, r.state_key, r.target, r.fg_channel, r.level) == ("vesuvius", "model", "surface", 1, 0)
    assert (m.kind, m.state_key, m.fg_channel, m.level, m.voxel_um) == ("nnunet", "network_weights", 1, 2, 9.6)
    assert m.normalizer.mode == "ct_clip" and (m.normalizer.lo, m.normalizer.hi) == (0.0, 212.0)


def test_fake_teacher_loads_and_forwards(fake_teacher):
    net, spec = teachers.load_teacher("fake", fake_teacher.ckpt, device="cpu")
    assert spec.patch == (PATCH, PATCH, PATCH) and spec.target == "surface"
    assert not any(p.requires_grad for p in net.parameters()) and not net.training
    x = torch.zeros(1, 1, PATCH, PATCH, PATCH)
    with torch.no_grad():
        y = spec.select(net(x))
    assert tuple(y.shape) == (1, 2, PATCH, PATCH, PATCH)
    p = teachers.apply_activation(y, spec.activation)
    assert torch.allclose(p.sum(1), torch.ones(1, PATCH, PATCH, PATCH), atol=1e-5)
    # ... and the weights really came off disk: the same seed rebuilds the same net
    ref = build_fake(0)
    for (k, a), b in zip(net.state_dict().items(), ref.state_dict().values()):
        assert torch.equal(a, b), k


def test_load_teacher_is_strict(fake_teacher, tmp_path):
    sd = torch.load(fake_teacher.ckpt, map_location="cpu", weights_only=False)["model"]
    sd.pop(next(k for k in sd if k.endswith("conv.weight")))
    bad = tmp_path / "bad.pth"
    torch.save({"model": sd}, bad)
    with pytest.raises((RuntimeError, ValueError)):
        teachers.load_teacher("fake", bad, device="cpu")
    with pytest.raises(FileNotFoundError):
        teachers.load_teacher("fake", tmp_path / "nope.pth", device="cpu")


def test_load_teacher_unwraps_ema_and_prefixes(fake_teacher, tmp_path):
    sd = torch.load(fake_teacher.ckpt, map_location="cpu", weights_only=False)["model"]
    p = tmp_path / "wrapped.pth"
    torch.save({"model": {"model": {"module." + k: v for k, v in sd.items()}}}, p)
    net, _ = teachers.load_teacher("fake", p, device="cpu")
    assert isinstance(net, teachers.VesuviusUNet)
