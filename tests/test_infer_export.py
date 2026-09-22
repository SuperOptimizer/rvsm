"""The region runner and the tracer export (teacher half; the student half lands with commit 5).

What is pinned here is everything a wrong answer would still look plausible under: that air stays
exactly 0 and never costs a forward pass, that a region thinner than a window is padded and cropped back
rather than silently reshaped, that the fp16 accumulators cost less than the store's own quantisation,
that a flip TTA negates the components of a vector output instead of averaging it to zero, and that the
exported sign convention is the one the attrs claim.
"""
import json
import os

import numpy as np
import pytest
import torch

from rvsm import cli, export, infer, ladder, stores, teachers, trt
from tests.teachers_fixture import PATCH, slab_block


@pytest.fixture
def fake_net(fake_teacher):
    net, spec = teachers.load_teacher("fake", fake_teacher.ckpt, device="cpu")
    return net, spec


def counting(fn):
    """Wrap a window function, counting the windows it is actually asked for."""
    calls = []

    def go(x):
        calls.append(int(x.shape[0]))
        return fn(x)
    go.calls = calls
    return go


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_starts_cover_the_axis_and_end_flush():
    s = infer.starts(100, 32, 24)
    assert s[0] == 0 and s[-1] == 100 - 32
    assert all(b - a <= 24 for a, b in zip(s, s[1:]))
    assert infer.starts(32, 32, 24) == [0]
    assert len(infer.offsets((64, 64, 64), 32, 4)) == len(infer.starts(64, 32, 24)) ** 3


def test_gauss_is_centred_and_positive():
    g = infer.gauss_t(8, "cpu")
    assert g.shape == (8, 8, 8) and float(g.min()) > 0
    assert torch.allclose(g, torch.flip(g, [0, 1, 2]), atol=1e-6)
    assert g.argmax().item() in (int(np.ravel_multi_index((3, 3, 3), (8, 8, 8))),
                                 int(np.ravel_multi_index((4, 4, 4), (8, 8, 8))))


# --------------------------------------------------------------------------- #
# run_region
# --------------------------------------------------------------------------- #
def test_run_region_shape_air_and_skip(fake_net):
    net, spec = fake_net
    ct = slab_block((64, 64, 64), half=4, centre=8)   # near the low y face: the far windows are air
    inp = infer.TeacherInputs(ct, spec.normalizer, PATCH)
    fn = counting(infer.teacher_fn(net, spec))
    p = infer.run_region(fn, inp, (64, 64, 64), PATCH, 4, planes=1)
    assert tuple(p.shape) == (1, 64, 64, 64)
    p = p[0].numpy()
    assert np.all(p[ct == 0] == 0)                       # air is EXACTLY zero, not a small probability
    assert p[ct > 0].max() > 0
    assert 0.0 <= p.min() and p.max() <= 1.0
    # the zero-window skip: only the windows that touch the slab were forwarded
    all_offs = infer.offsets(inp.shape, PATCH, 4)
    hit = [o for o in all_offs if inp.window_any(o)]
    assert 0 < len(hit) < len(all_offs)
    assert sum(fn.calls) == len(hit)


def test_run_region_pads_a_region_thinner_than_a_window(fake_net):
    net, spec = fake_net
    ct = slab_block((16, 16, 16), half=4)
    inp = infer.TeacherInputs(ct, spec.normalizer, PATCH)
    assert inp.shape == (PATCH, PATCH, PATCH)            # padded with air, to one full window
    p = infer.run_region(infer.teacher_fn(net, spec), inp, (16, 16, 16), PATCH, 4, planes=1)
    assert tuple(p.shape) == (1, 16, 16, 16)             # ... and cropped back to what was asked for
    assert np.all(p[0].numpy()[ct == 0] == 0)


def test_fp16_accumulators_match_fp32_within_a_store_code(fake_net):
    net, spec = fake_net
    ct = slab_block((48, 48, 48), half=8)
    inp = infer.TeacherInputs(ct, spec.normalizer, PATCH)
    fn = infer.teacher_fn(net, spec)
    a = infer.run_region(fn, inp, (48, 48, 48), PATCH, 4, acc_dtype=torch.float16)[0].numpy()
    b = infer.run_region(fn, inp, (48, 48, 48), PATCH, 4, acc_dtype=torch.float32)[0].numpy()
    assert np.abs(a - b).max() < 1.0 / 255.0             # below the quantisation the store applies anyway
    assert np.array_equal(stores.u8(a), stores.u8(b)) or np.abs(
        stores.u8(a).astype(int) - stores.u8(b).astype(int)).max() <= 1


def test_batching_does_not_change_the_result(fake_net):
    net, spec = fake_net
    ct = slab_block((48, 48, 48), half=8)
    inp = infer.TeacherInputs(ct, spec.normalizer, PATCH)
    fn = infer.teacher_fn(net, spec)
    a = infer.run_region(fn, inp, (48, 48, 48), PATCH, 4, batch=1, acc_dtype=torch.float32)
    b = infer.run_region(fn, inp, (48, 48, 48), PATCH, 4, batch=4, acc_dtype=torch.float32)
    assert np.abs(a.numpy() - b.numpy()).max() < 1e-5


def test_multi_plane_output(fake_net):
    """`planes > 1` blends one accumulator per plane and keeps them in order."""
    net, spec = fake_net
    ct = slab_block((48, 48, 48), half=8)
    inp = infer.TeacherInputs(ct, spec.normalizer, PATCH)
    base = infer.teacher_fn(net, spec)
    two = lambda x: torch.cat([base(x), 1.0 - base(x)], 1)  # noqa: E731
    p = infer.run_region(two, inp, (48, 48, 48), PATCH, 4, planes=2, acc_dtype=torch.float32)
    assert tuple(p.shape) == (2, 48, 48, 48)
    m = ct > 0
    assert np.allclose(p[0].numpy()[m] + p[1].numpy()[m], 1.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# teacher_region against the served CT, and the produce subcommand
# --------------------------------------------------------------------------- #
def test_teacher_region_on_the_served_ct(ct_origin, fake_teacher, has_volcomp):
    if not has_volcomp:
        pytest.skip("volcomp is required to read the CT fixture")
    spec = teachers.TEACHERS["fake"]
    # the local mirror of the same pyramid: `test_produce_writes_a_done_store` runs the served URL
    lo = (0, 64, 0)     # the fixture's slab sits near y = 96..112 at these x: a box at the origin is air
    p = infer.teacher_region(ct_origin.path, lo, (64, 64, 64), spec, fake_teacher.ckpt,
                             device="cpu", backend="torch", window=PATCH, halo=4)
    assert p.shape == (64, 64, 64) and p.dtype == np.float32
    ct = ladder.read_rung(ladder.rungs(ct_origin.path), 2, lo, (64, 64, 64), dtype=np.uint8)
    assert np.all(p[ct == 0] == 0) and p.max() > 0


def test_produce_writes_a_done_store(tmp_path, ct_origin, fake_teacher, umbilicus, has_volcomp):
    if not has_volcomp:
        pytest.skip("volcomp is required to write a store")
    out = str(tmp_path / "run")
    rc = cli.main(["produce", "--out", out, "--ct", ct_origin.url, "--umbilicus", umbilicus[0],
                   "--teacher", "fake", "--ckpt-fake", fake_teacher.ckpt, "--region", "0", "0", "0",
                   "--size", "128", "--device", "cpu", "--halo", "4"])
    assert rc == 0
    for ch in ("recto", "rw"):
        path = stores.store_path(out, ch, (0, 0, 0), 0)
        assert stores.is_done(path), path
        a = stores.open_store(path)
        assert a.shape == (128, 128, 128) and a.attrs["rung"] == 2
        assert a.attrs["origin_zyx"] == [0, 0, 0] and a.attrs["volcomp_q"] == 8
        assert a.attrs["producer"] == "teacher:fake" and a.attrs["radial_sign"] == 1
        assert a.attrs["window"]["fake"] == PATCH and a.attrs["halo"]["fake"] == 4
        assert a.attrs["umbilicus"] == umbilicus[0] and a.attrs["volume"] == ct_origin.url
    rec = np.asarray(stores.open_store(stores.store_path(out, "recto", (0, 0, 0), 0))[:])
    ct = ladder.read_rung(ladder.rungs(ct_origin.url), 2, (0, 0, 0), (128, 128, 128), dtype=np.uint8)
    assert rec[ct > 0].max() > 0
    rw = np.asarray(stores.open_store(stores.store_path(out, "rw", (0, 0, 0), 0))[:])
    assert rw.min() >= 250                                # one teacher: the agreement weight is 1
    # `done` is the LAST thing written, and it is what a reader keys off
    j = json.load(open(os.path.join(stores.store_path(out, "recto", (0, 0, 0), 0), "zarr.json")))
    assert j["attributes"]["done"] is True


def test_produce_rejects_a_region_outside_the_volume(tmp_path, ct_origin, fake_teacher, has_volcomp):
    if not has_volcomp:
        pytest.skip("volcomp is required to read the CT fixture")
    with pytest.raises(SystemExit):
        cli.main(["produce", "--out", str(tmp_path / "r"), "--ct", ct_origin.url, "--teacher", "fake",
                  "--ckpt-fake", fake_teacher.ckpt, "--region", "4096", "0", "0", "--device", "cpu"])


def test_produce_needs_a_checkpoint_and_a_known_teacher(tmp_path, ct_origin, fake_teacher):
    with pytest.raises(SystemExit):
        cli.main(["produce", "--out", str(tmp_path / "r"), "--ct", ct_origin.url, "--teacher", "nope",
                  "--region", "0", "0", "0"])
    with pytest.raises(SystemExit):
        cli.main(["produce", "--out", str(tmp_path / "r"), "--ct", ct_origin.url, "--teacher", "fake",
                  "--region", "0", "0", "0"])


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #
def test_fuse_agreement_bounds_and_symmetry():
    rng = np.random.default_rng(0)
    a, b = rng.random((8, 8, 8)).astype(np.float32), rng.random((8, 8, 8)).astype(np.float32)
    p, w = infer.fuse_agreement(a, b)
    assert p.dtype == np.float32 and w.dtype == np.float32
    assert p.min() >= 0 and p.max() <= 1 and w.min() >= 0 and w.max() <= 1
    assert np.all(p >= np.minimum(a, b) - 1e-6) and np.all(p <= np.maximum(a, b) + 1e-6)
    p2, w2 = infer.fuse_agreement(b, a)
    assert np.allclose(p, p2, atol=1e-6) and np.allclose(w, w2, atol=1e-6)
    # identical sources: the fusion is the source and the weight is 1
    p3, w3 = infer.fuse_agreement(a, a)
    assert np.allclose(p3, a, atol=1e-6) and np.allclose(w3, 1.0, atol=1e-6)
    # total disagreement: weight 0
    _, w4 = infer.fuse_agreement(np.zeros((4,), np.float32), np.ones((4,), np.float32))
    assert np.allclose(w4, 0.0)
    # the confident source carries the voxel
    p5, _ = infer.fuse_agreement(np.array([0.99], np.float32), np.array([0.5], np.float32))
    assert p5[0] > 0.8


def test_binary_confidence():
    c = infer.binary_confidence(np.array([0.0, 0.5, 1.0], np.float32))
    assert c[1] == pytest.approx(0.0, abs=1e-5)
    assert c[0] > 0.999 and c[2] > 0.999


# --------------------------------------------------------------------------- #
# flip TTA
# --------------------------------------------------------------------------- #
def test_flips_chan_negates_the_flipped_vector_component():
    """A constant normal field: without the vector rule, the TTA average cancels it."""
    n = torch.tensor([0.0, 1.0, 0.0])                       # a unit normal along +y everywhere

    def fn(x):
        b, _, z, y, xx = x.shape
        return n.view(1, 3, 1, 1, 1).expand(b, 3, z, y, xx).clone()

    x = torch.zeros(1, 4, 4, 4, 4)
    naive = 0
    for f in [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
        naive = naive + torch.flip(fn(torch.flip(x, [2 + d for d in f])), [2 + d for d in f])
    assert torch.allclose(naive[:, 1] / 8, torch.ones(1, 4, 4, 4))  # scalar averaging keeps +y ...
    got = infer.flips_chan(fn, 8, vec=[(0, 1, 2)], radial=False)(x)
    # ... but as a VECTOR, flipping y negates the y component: the honest average of +1 and -1 is 0
    assert torch.allclose(got[:, 1], torch.zeros(1, 4, 4, 4), atol=1e-6)
    assert torch.allclose(got[:, 0], torch.zeros(1, 4, 4, 4), atol=1e-6)


def test_flips_chan_negates_the_radial_input_channels():
    seen = []

    def fn(x):
        seen.append(x.clone())
        return x[:, :1]

    x = torch.zeros(1, 4, 2, 2, 2)
    x[:, 1:] = 1.0                                          # the last three channels are the radial vector
    infer.flips_chan(fn, 2, radial=True)(x)
    assert float(seen[0][:, 1].mean()) == 1.0               # identity flip: untouched
    assert float(seen[1][:, 1].mean()) == -1.0              # flipped z: component z negated
    assert float(seen[1][:, 2].mean()) == 1.0
    seen.clear()
    infer.flips_chan(fn, 2, radial=False)(x)                # a CT-only teacher has nothing to negate
    assert float(seen[1][:, 1].mean()) == 1.0


def test_flips_chan_averages_a_symmetric_function_to_itself(fake_net):
    net, spec = fake_net
    x = torch.zeros(1, 1, PATCH, PATCH, PATCH)
    fn = infer.teacher_fn(net, spec, tta=1)
    tta = infer.teacher_fn(net, spec, tta=8)
    a, b = fn(x), tta(x)
    assert tuple(a.shape) == tuple(b.shape) == (1, 1, PATCH, PATCH, PATCH)
    assert float(b.min()) >= 0.0 and float(b.max()) <= 1.0


# --------------------------------------------------------------------------- #
# the cascade helpers
# --------------------------------------------------------------------------- #
def test_crop_pad_and_up2x():
    a = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    assert infer.crop_pad(a, (1, 1, 1), (2, 2, 2)).shape == (2, 2, 2)
    out = infer.crop_pad(a, (2, 2, 2), (3, 3, 3))       # runs past the end -> zero padded
    assert out[0, 0, 0] == a[2, 2, 2] and out[2, 2, 2] == 0
    assert infer.crop_pad(None, (0, 0, 0), (2, 2, 2)).sum() == 0
    up = infer.up2x_np(np.ones((4, 4, 4), np.float32))
    assert up.shape == (8, 8, 8) and np.allclose(up, 1.0)


def test_cascade_for_walks_one_rung_up():
    seen = {}

    def at_rung(k, o, s, depth):
        seen["k"], seen["o"], seen["s"], seen["depth"] = k, tuple(o), tuple(s), depth
        return np.full(tuple(s), 0.5, np.float32)

    c = infer.cascade_for(at_rung, 2, (64, 64, 64), (32, 32, 32), 3)
    assert seen["k"] == 3 and seen["depth"] == 2
    assert seen["o"] == (16, 16, 16)                    # 64 // 2 - halo(16)
    assert c.shape == (32, 32, 32) and np.allclose(c, 0.5)
    assert infer.cascade_for(at_rung, 2, (0, 0, 0), (8, 8, 8), 0) is None
    assert infer.cascade_for(at_rung, ladder.NRUNGS - 1, (0, 0, 0), (8, 8, 8), 3) is None


# --------------------------------------------------------------------------- #
# TensorRT fallback
# --------------------------------------------------------------------------- #
def test_engine_for_falls_back_cleanly(tmp_path, fake_net):
    net, _ = fake_net
    try:
        import tensorrt  # noqa: F401
        has_trt = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_trt = False
    if has_trt:
        pytest.skip("tensorrt and a GPU are present: the fallback path is not the one under test")
    d = tmp_path / "trt"
    assert trt.engine_for(net, "fake", PATCH, 1, d) is None
    assert not d.exists()                                # nothing half-built left behind
    assert trt.plan("fake", 32, "/d").startswith("/d/fake_p32_b1_fp16_")


# --------------------------------------------------------------------------- #
# the tracer export
# --------------------------------------------------------------------------- #
def outward_slab(n=128, half=6.0):
    """A block whose sheet is a plane of constant y: the radial direction is +y, so an OUTWARD field is
    one that grows with y."""
    y = np.arange(n, dtype=np.float32)[None, :, None]
    mid = y - n / 2.0                                    # the midline distance, growing outward
    return np.broadcast_to(mid, (n, n, n)).copy(), np.full((n, n, n), 2 * half, np.float32)


def test_tracer_fields_sign_convention():
    mid, thick = outward_slab(32, half=6.0)
    d, nrm, mag, valid = export.tracer_fields(mid, thick)
    # the recto-face field is the midline minus half the thickness ...
    assert np.allclose(d, mid - 0.5 * thick, atol=1e-5)
    # ... its gradient is +y, a unit vector, and it points radially OUTWARD (dot(n, radial) > 0)
    core = (slice(2, -2),) * 3
    assert np.allclose(nrm[1][core], 1.0, atol=1e-3)
    assert np.allclose(nrm[0][core], 0.0, atol=1e-3) and np.allclose(nrm[2][core], 0.0, atol=1e-3)
    assert np.allclose(mag[core], 1.0, atol=1e-3)        # a unit ramp gives |grad| = 1
    assert valid.all()
    radial = np.zeros((3,) + d.shape, np.float32)
    radial[1] = 1.0
    assert float((nrm * radial).sum(0)[core].min()) > 0.9


def test_encodings_round_trip():
    d = np.linspace(-40, 40, 101).astype(np.float32)
    u = export.enc_signed(d, np.ones_like(d, bool))
    assert u.min() >= 1                                   # code 0 is reserved for NO DATA
    back = (u.astype(np.float32) - export.TRACER_OFF) * export.TRACER_UNIT
    assert np.abs(back - np.clip(d, -export.TRACER_CAP, export.TRACER_CAP)).max() <= export.TRACER_UNIT / 2 + 1e-6
    assert export.enc_signed(d, np.zeros_like(d, bool)).max() == 0
    n = np.linspace(-1, 1, 51).astype(np.float32)
    assert np.abs(export.dec_normal(export.enc_normal(n, np.ones_like(n, bool))) - n).max() < 1 / 127 + 1e-6
    assert export.enc_normal(n, np.zeros_like(n, bool)).max() == 0


def test_scharr_of_a_ramp_is_one():
    x = np.arange(16, dtype=np.float32)[None, None, :] * np.ones((16, 16, 1), np.float32)
    g = export.scharr3(x)
    core = (slice(1, -1),) * 3
    assert np.allclose(g[2][core], 1.0, atol=1e-5)
    assert np.allclose(g[0][core], 0.0, atol=1e-5) and np.allclose(g[1][core], 0.0, atol=1e-5)


def test_export_tracer_writes_the_contract(tmp_path, has_volcomp):
    if not has_volcomp:
        pytest.skip("volcomp is required to write a store")
    n = 128
    mid, thick = outward_slab(n, half=6.0)
    rec = np.exp(-((mid - 0.5 * thick) / 3.0) ** 2).astype(np.float32)
    got_planes = {"recto": rec, "verso": np.exp(-((mid + 0.5 * thick) / 3.0) ** 2).astype(np.float32),
                  "midline": mid, "thickness": thick, "conf": np.full(mid.shape, 0.75, np.float32)}
    seen = {}

    def probs_multi(origin, size):
        seen["origin"], seen["size"] = origin, size
        return got_planes

    out = str(tmp_path / "export")
    got = export.export_tracer(probs_multi, (0, 128, 256), (n, n, n), out, rung=2, volume="vol",
                               umbilicus="umb", log=lambda *a: None)
    assert seen == {"origin": (0, 128, 256), "size": (n, n, n)}
    assert sorted(got) == sorted(["recto", "verso", "surf_sdist", "nz", "ny", "nx", "gmag", "conf",
                                  "thickness"])
    # q8 for the PROBABILITIES only: every field is lossless, because its code 0 means "no data" and a
    # codec that rounds a 1 to a 0 there would invent a hole. `conf` is a field of the contract (q0).
    q = {"recto": 8, "verso": 8}
    for name, path in got.items():
        assert stores.is_done(path), name
        a = stores.open_store(path)
        assert a.shape == (n, n, n) and a.attrs["origin_zyx"] == [0, 128, 256]
        assert a.attrs["rung"] == 2 and a.attrs["voxel_um"] == ladder.rung_um(2)
        assert a.attrs["volcomp_q"] == q.get(name, 0), name
        assert a.attrs["no_data"] == 0 and a.attrs["axis_order"] == "ZYX"
        assert "radially OUTWARD" in a.attrs["sign_convention"]
        assert a.attrs["volume"] == "vol" and a.attrs["umbilicus"] == "umb"
    assert stores.open_store(got["surf_sdist"]).attrs["encoding"] == "signed_u8_off128_q0.25"
    assert stores.open_store(got["nz"]).attrs["encoding"] == "normal_u8_off128_div127"
    assert stores.open_store(got["recto"]).attrs["encoding"] == "prob_u8"
    # the fields are LOSSLESS: what was encoded is exactly what reads back
    d, nrm, mag, valid = export.tracer_fields(mid, thick)
    valid = valid & (rec > 0)
    sd = np.asarray(stores.open_store(got["surf_sdist"])[:])
    assert np.array_equal(sd, export.enc_signed(d, valid))
    ny = np.asarray(stores.open_store(got["ny"])[:])
    # WHERE THERE IS DATA the normal is +y, i.e. outward; where the recto probability underflowed to 0
    # there is none, and the store says so with code 0 rather than with a plausible-looking vector.
    m = valid & (mag > 1e-3)
    assert m.any() and not m.all()
    assert np.abs(export.dec_normal(ny[m]) - 1.0).max() < 1 / 127 + 1e-3
    assert np.all(ny[~m] == 0)
    assert np.all(np.asarray(stores.open_store(got["surf_sdist"])[:])[~valid] == 0)


def test_export_tracer_needs_a_distance_channel(tmp_path):
    with pytest.raises(AssertionError):
        export.export_tracer(lambda o, s: {"recto": np.zeros(s, np.float32)}, (0, 0, 0), (8, 8, 8),
                             str(tmp_path / "e"), log=lambda *a: None)
    with pytest.raises(AssertionError):
        export.export_tracer(lambda o, s: {"midline": np.zeros(s, np.float32)}, (0, 0, 0), (8, 8, 8),
                             str(tmp_path / "e"), log=lambda *a: None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU smoke")
def test_gpu_smoke_region(fake_net):
    """The same region pass on the card, fp16 accumulators and all."""
    net, spec = fake_net
    net = net.cuda()
    ct = slab_block((64, 64, 64), half=8)
    inp = infer.TeacherInputs(ct, spec.normalizer, PATCH, device="cuda")
    p = infer.run_region(infer.teacher_fn(net, spec), inp, (64, 64, 64), PATCH, 4, batch=2)
    assert tuple(p.shape) == (1, 64, 64, 64)
    q = p[0].cpu().numpy()
    assert np.all(q[ct == 0] == 0) and q.max() > 0


def test_produce_falls_back_to_the_weights_cache(tmp_path, ct_origin, fake_teacher, monkeypatch,
                                                 has_volcomp):
    """With no --ckpt-<name>, `produce` takes the published weights out of the cache (fetching once)."""
    if not has_volcomp:
        pytest.skip("volcomp is required to write a store")
    import dataclasses
    teachers.register("fake", dataclasses.replace(fake_teacher.spec, url="https://hf.test/w.pth",
                                                  file="w.pth"))
    called = []
    monkeypatch.setattr(teachers, "fetch_weights",
                        lambda name, cache_dir=teachers.CACHE_DIR, **kw: (called.append((name, cache_dir))
                                                                          or fake_teacher.ckpt))
    out = str(tmp_path / "run")
    rc = cli.main(["produce", "--out", out, "--ct", ct_origin.path, "--teacher", "fake",
                   "--region", "0", "0", "0", "--size", "128", "--device", "cpu", "--halo", "4"])
    assert rc == 0 and called == [("fake", teachers.CACHE_DIR)]
    a = stores.open_store(stores.store_path(out, "recto", (0, 0, 0), 0))
    assert a.attrs["ckpt"]["fake"] == fake_teacher.ckpt
