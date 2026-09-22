"""The teacher nets: architecture inferred from shapes, strict parity, normalisers, one forward.

The point of these tests is the claim the whole round-0 bootstrap rests on: rvsm can rebuild the
published networks from a checkpoint ALONE, with no config file and no `vesuvius` import, and load them
strictly. A silent key mismatch would be a teacher running with half its weights randomly initialised,
which looks like a plausible probability field and is worthless.
"""
import os

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


# --------------------------------------------------------------------------- #
# The published weights
# --------------------------------------------------------------------------- #
class _Resp:
    """The bits of an `urlopen` response `_download` uses: a context manager, `.read(n)`, `.headers`."""

    def __init__(self, blob):
        self.blob, self.i, self.headers = blob, 0, {"Content-Length": str(len(blob))}

    def read(self, n=-1):
        b = self.blob[self.i:] if n is None or n < 0 else self.blob[self.i:self.i + n]
        self.i += len(b)
        return b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def hf(monkeypatch):
    """A fake Hugging Face: `fetchme` is a teacher with a URL, and urlopen serves it from a dict."""
    import types
    import urllib.request
    served, seen = {}, []
    spec = teachers.TeacherSpec(
        name="fetchme", state_key="model", kind="vesuvius", patch=(32,) * 3,
        normalizer=teachers.Normalizer("zscore_instance"), activation="softmax", fg_channel=1,
        voxel_um=2.4, target="surface", url="https://hf.test/w.pth", file="w.pth",
        extra_urls=(("https://hf.test/plans.json", "plans.json"),))
    teachers.register("fetchme", spec)

    def urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        seen.append(url)
        if url not in served:
            raise OSError(f"404 {url}")
        return _Resp(served[url])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    try:
        yield types.SimpleNamespace(served=served, seen=seen, spec=spec)
    finally:
        teachers.TEACHERS.pop("fetchme", None)


def test_fetch_weights_downloads_once_and_renames(tmp_path, hf):
    blob = b"x" * (2 << 20)
    hf.served["https://hf.test/w.pth"] = blob
    p = teachers.fetch_weights("fetchme", cache_dir=tmp_path, log=lambda *a: None)
    assert p == str(tmp_path / "w.pth") and open(p, "rb").read() == blob
    assert not os.path.exists(p + ".part")            # the partial name never survives
    assert len(hf.seen) == 1
    # a second call is a cache hit: no request at all
    assert teachers.fetch_weights("fetchme", cache_dir=tmp_path, log=lambda *a: None) == p
    assert len(hf.seen) == 1


def test_fetch_weights_rejects_a_short_download(tmp_path, hf):
    hf.served["https://hf.test/w.pth"] = b"<html>404</html>"   # an error page, not a checkpoint
    with pytest.raises(RuntimeError, match="too small"):
        teachers.fetch_weights("fetchme", cache_dir=tmp_path, log=lambda *a: None)
    assert not list(tmp_path.iterdir())                        # and nothing is left behind


def test_fetch_weights_replaces_a_truncated_cache_entry(tmp_path, hf):
    (tmp_path / "w.pth").write_bytes(b"truncated")
    hf.served["https://hf.test/w.pth"] = b"y" * (2 << 20)
    p = teachers.fetch_weights("fetchme", cache_dir=tmp_path, log=lambda *a: None)
    assert os.path.getsize(p) == 2 << 20


def test_fetch_weights_extras(tmp_path, hf):
    hf.served["https://hf.test/w.pth"] = b"z" * (2 << 20)
    hf.served["https://hf.test/plans.json"] = b"{}" + b" " * (2 << 20)
    teachers.fetch_weights("fetchme", cache_dir=tmp_path, extras=True, log=lambda *a: None)
    assert (tmp_path / "plans.json").exists()         # fetched for a human, never read by rvsm
    assert (tmp_path / "w.pth").exists()


def test_a_teacher_without_a_url_says_so(tmp_path, fake_teacher):
    with pytest.raises(ValueError, match="no published weights"):
        teachers.fetch_weights("fake", cache_dir=tmp_path)


def test_the_two_real_teachers_carry_their_hugging_face_urls():
    for n, host in (("recto", "surface_recto_3dunet"), ("m7", "surface_m7_nnunet")):
        s = teachers.TEACHERS[n]
        assert s.url.startswith(f"https://huggingface.co/scrollprize/{host}/resolve/main/")
        assert s.file.endswith(".pth")
    assert teachers.TEACHERS["recto"].url.endswith("checkpoint_inference_ready.pth")
    assert teachers.TEACHERS["m7"].url.endswith("fold_0/checkpoint_best.pth")
    assert teachers.TEACHERS["m7"].extra_urls[0][0].endswith("plans.json")


def test_load_teacher_falls_back_to_the_cache(monkeypatch, fake_teacher):
    """`load_teacher(name)` with no path fetches the published weights (once)."""
    called = []

    def fake_fetch(name, cache_dir=teachers.CACHE_DIR, **kw):
        called.append((name, cache_dir))
        return fake_teacher.ckpt

    monkeypatch.setattr(teachers, "fetch_weights", fake_fetch)
    net, _ = teachers.load_teacher("fake", device="cpu")
    assert isinstance(net, teachers.VesuviusUNet) and called == [("fake", teachers.CACHE_DIR)]


# --------------------------------------------------------------------------- #
# The real checkpoints (~1.9 GB): opt in with RVSM_SLOW=1, or free once they are cached
# --------------------------------------------------------------------------- #
def _cached(name):
    s = teachers.TEACHERS[name]
    p = os.path.join(os.path.expanduser(teachers.CACHE_DIR), s.file)
    return os.path.exists(p) and os.path.getsize(p) >= teachers.MIN_BYTES


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RVSM_SLOW") and not (_cached("recto") and _cached("m7")),
                    reason="downloads ~1.9 GB; set RVSM_SLOW=1 (or prime the cache) to run it")
# The window each one can actually take: a stage count of n downsamples by 2^(n-1), and InstanceNorm
# refuses a bottleneck of one voxel. recto has 7 stages (64x), m7 has 6 (32x).
@pytest.mark.parametrize("name,window,stages", [("recto", 128, 7), ("m7", 64, 6)])
def test_real_teacher_loads_strictly_and_forwards(name, window, stages):
    """The whole claim, against the published bytes: fetch, infer the architecture from the state
    dict's own shapes, load STRICTLY, and forward a window to a finite probability."""
    net, spec = teachers.load_teacher(name, device="cpu")
    arch = net.spec
    assert len(arch.features_per_stage) == stages
    assert arch.features_per_stage[:5] == [32, 64, 128, 256, 320]     # the published ResEnc widths
    assert arch.strides == [1] + [2] * (stages - 1) and arch.norm == "instance"
    assert (arch.squeeze_excitation == "scse") == (name == "recto")   # villa adds scSE, nnU-Net does not
    x = torch.zeros(1, 1, window, window, window)
    with torch.no_grad():
        y = spec.select(net(x))
    assert y.shape[0] == 1 and y.shape[1] == 2 and y.shape[2:] == (window,) * 3
    p = teachers.apply_activation(y.float(), spec.activation)
    assert torch.isfinite(p).all()
    assert torch.allclose(p.sum(1), torch.ones(1, *(window,) * 3), atol=1e-4)  # both are 2-way softmax
    fg = p[:, spec.fg_channel]
    assert float(fg.min()) >= 0.0 and float(fg.max()) <= 1.0
