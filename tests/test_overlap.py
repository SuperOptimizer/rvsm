"""The overlap-crop consistency term (`rvsm.overlap`, `losses.overlap_loss`, the sampler's second window).

CPU only, tiny shapes. The two properties that matter are checked by construction:

  * the MAPPING: a voxel of the second window lands on the first window's voxel with the same physical
    coordinate -- with a coordinate-valued field, with a sub-crop, and through the real input builder,
    the cube symmetry and the paste;
  * the ZERO: when the student and the EMA are the same pointwise function of the CT and the two windows
    see the same values on their shared voxels, the term is ~0.
"""
import json

import numpy as np
import pytest
import torch

from rvsm import losses as L, overlap as OV, prep, sample, train as TR
from rvsm.config import Config

P = 32
AXIS = np.array([[0.0, 4096.0], [-900.0, -900.0], [-1000.0, -1000.0]])   # far outside: radial ~ constant


def _cfg(**kw):
    return Config(size="1m", patch=P, ctx=(1, 2, 3), aff_offsets=(4, 8), cascade="off", **kw)


def _hash_volume(n=96):
    """A uint8 volume whose value identifies the voxel well enough that any mis-mapping shows."""
    z, y, x = np.indices((n, n, n))
    return ((x * 7 + y * 31 + z * 131) % 251).astype(np.uint8)


def _periodic_volume(n=96, per=8):
    z, y, x = np.indices((n, n, n))
    return ((x % per) * 11 + (y % per) * 3 + (z % per) * 17 + 5).astype(np.uint8)


def _pair_item(V, lo, shift, cfg, sym=0, norm=(0.0, 1.0), k=2):
    """A first window at `lo` and its overlap window at lo + shift, cut from the volume V."""
    lay = cfg.layout()
    lo, lo2 = np.asarray(lo, np.int64), np.asarray(lo, np.int64) + np.asarray(shift, np.int64)
    cut = lambda o: V[o[0]:o[0] + P, o[1]:o[1] + P, o[2]:o[2] + P]   # noqa: E731
    ct = np.stack([cut(lo)] * (1 + lay.nctx))
    T = lay.cout_t
    tg = np.zeros((T, P, P, P), np.uint8)
    w = np.full((T, P, P, P), 255, np.uint8)
    it = sample.rung_item(ct, tg, w, k, lo, AXIS, sym=sym, norm=norm, rmax=float(4 * P),
                          meta=np.linspace(0.2, 0.8, 5, dtype=np.float32))
    it.update({"ov_ct": torch.from_numpy(np.stack([cut(lo2)] * (1 + lay.nctx)).copy()),
               "ov_lo": torch.from_numpy(lo2),
               "ov_cyx": torch.from_numpy(sample.axis_cyx(AXIS, k, lo2, P))})
    return it


def _pointwise(lay, off=0.0):
    """A fake net: every head a fixed pointwise function of the CT channel (channel 0) only."""
    def f(x):
        c = x[:, :1].float()
        y = torch.zeros((x.shape[0], lay.cout) + tuple(x.shape[2:]))
        y[:, 0:1] = 0.05 * c - 1.0 + off
        y[:, 1:2] = -0.03 * c + 0.5 + off
        y[:, lay.i_mid:lay.i_mid + 1] = 0.1 * c - 10.0 + 20 * off
        y[:, lay.i_thick:lay.i_thick + 1] = 0.02 * c + off
        return y
    return f


# --------------------------------------------------------------------- the sampler's shift

def test_overlap_shifts_are_even_in_range_and_inside_the_draw_box():
    rng = np.random.default_rng(0)
    for p in (32, 160, 256):
        lo_min, lo_max = np.zeros(3, np.int64), np.array([768, 768, 40], np.int64)
        for _ in range(300):
            lo = rng.integers(lo_min, lo_max + 1)
            s = sample.overlap_shift(rng, p, lo, lo_min, lo_max)
            if s is None:
                continue
            nz = s[s != 0]
            assert len(nz) >= 1 and (nz % 2 == 0).all()
            assert ((np.abs(nz) >= p // 4) & (np.abs(nz) <= p // 2)).all(), (p, s)
            assert (lo + s >= lo_min).all() and (lo + s <= lo_max).all()
            assert (p - np.abs(s) >= p // 2).all()           # at least half the patch shared per axis
    # no room on any axis: no second window
    z = np.zeros(3, np.int64)
    assert sample.overlap_shift(rng, 256, z, z, np.full(3, 60)) is None


def test_the_collate_drops_the_second_window_unless_every_item_has_one():
    cfg = _cfg()
    V = _hash_volume()
    a = _pair_item(V, (8, 8, 8), (0, 10, 0), cfg)
    b = {k: v for k, v in a.items() if k not in sample.OV_KEYS}
    assert "ov_ct" not in sample.collate([a, b])
    assert sample.collate([a, a])["ov_ct"].shape == (2, 1 + cfg.layout().nctx, P, P, P)


# --------------------------------------------------------------------- the mapping

@pytest.mark.parametrize("q", [0, 16])
def test_the_paste_maps_every_voxel_to_its_physical_twin(q):
    """A coordinate-valued EMA output of the second window, pasted: each first-window voxel of the box
    receives exactly its own physical coordinate, and the mask is the brute-force margin comparison."""
    rng = np.random.default_rng(1)
    lo1 = np.array([100, 200, 300])
    for _ in range(40):
        s = sample.overlap_shift(rng, P, lo1, lo1 - 64, lo1 + 64)
        o, n = OV.sub_window(P, s, q)
        box = OV.paste_box(P, s, o, n)
        assert box is not None
        a, b, ta, tb, mask = box
        # the EMA window's voxel j has physical coordinate lo1 + s + o + j
        coords2 = np.stack(np.meshgrid(*[np.arange(int(v)) for v in n], indexing="ij")) \
            + (lo1 + s + o)[:, None, None, None]
        got = coords2[:, ta[0]:tb[0], ta[1]:tb[1], ta[2]:tb[2]]
        want = np.stack(np.meshgrid(*[np.arange(a[j], b[j]) for j in range(3)], indexing="ij")) \
            + lo1[:, None, None, None]
        assert np.array_equal(got, want)
        # the mask, voxel by voxel: teacher margin (own window) > student margin (first window)
        v = np.stack(np.meshgrid(*[np.arange(a[j], b[j]) for j in range(3)], indexing="ij"))
        m1 = np.minimum(v, P - 1 - v).min(0)
        u = v - (s + o)[:, None, None, None]
        m2 = np.minimum(u, n[:, None, None, None] - 1 - u).min(0)
        assert np.array_equal(mask.numpy(), m2 > m1)
        assert mask.any()


@pytest.mark.parametrize("sym", [0, 13, 47])
@pytest.mark.parametrize("q", [0, 16])
def test_the_field_through_the_real_input_builder_is_the_first_windows_ct(sym, q):
    """End to end on a hash-valued volume: the EMA 'net' copies the CT into the midline head, the field
    is built by `teacher_batch` (the second window's own input, the sub-crop, the paste, the first
    window's symmetry), and on every masked voxel its decoded midline IS the first window's CT as
    `prep.prepare` put it in front of the student."""
    cfg = _cfg()
    lay = cfg.layout()
    V = _hash_volume()

    def copy_ct(x):
        y = torch.zeros((x.shape[0], lay.cout) + tuple(x.shape[2:]))
        y[:, lay.i_mid] = x[:, 0] / 255.0 * 60.0 - 30.0       # inside +-CAP: the code round trip is exact
        return y

    for lo, s in (((20, 30, 40), (0, 12, -16)), ((40, 40, 40), (-8, 0, 0)), ((10, 50, 20), (16, -10, 14))):
        b = prep.batch1(_pair_item(V, lo, s, cfg, sym=sym))
        x, _, _ = prep.prepare(b, torch.device("cpu"), layout=lay)
        E = OV.teacher_batch(b, copy_ct, lay, torch.device("cpu"), q=q)
        assert E is not None and E.shape == (1, OV.field_channels(lay), P, P, P)
        m = E[0, -1] > 0.5
        assert int(m.sum()) > 0
        got = L.decode_signed(E[0, lay.nprob])[m]
        want = (x[0, 0] / 255.0 * 60.0 - 30.0)[m]
        assert torch.allclose(got, want, atol=1e-4), (lo, s, float((got - want).abs().max()))


def test_the_shifted_cascade_is_the_first_windows_own_slice_at_zero_shift():
    """`Cascade.shifted` slices the kept rung-(k+1) prediction: at shift 0 it is exactly the channel the
    first window got, at a shift s it is the slice s/2 coarse voxels further on."""
    cfg = _cfg()
    lay = cfg.layout()
    V = _hash_volume()

    class Net(torch.nn.Module):
        def forward(self, x):
            y = torch.zeros((x.shape[0], lay.cout) + tuple(x.shape[2:]))
            y[:, 0] = x[:, 0]
            return y

    it = _pair_item(V, (20, 20, 20), (0, 8, -12), cfg)
    it["cx"] = torch.from_numpy(np.random.default_rng(0).integers(0, 255, (1, P, P, P), dtype=np.uint8))
    b = prep.batch1(it)
    cas = prep.Cascade("self", drop=0.0, noise=False, net=Net())
    cas.keep_coarse = True
    x, _, _ = prep.prepare(b, torch.device("cpu"), cascade=cas, layout=lay)
    assert cas.last_src == ["self"]
    assert torch.equal(cas.shifted(0, (0, 0, 0), (P, P, P)), x[:, lay.i_cas:lay.i_cas + 1])
    pc = cas.last_coarse[0]
    s = (0, 8, -12)
    a = [P // 4 + v // 2 for v in s]
    want = prep.M.up2x(pc[:, :, a[0]:a[0] + P // 2, a[1]:a[1] + P // 2, a[2]:a[2] + P // 2], (P, P, P))
    assert torch.equal(cas.shifted(0, s, (P, P, P)), want)


# --------------------------------------------------------------------- the zero, and the term

@pytest.mark.parametrize("norm,vol,shift", [((0.0, 1.0), "hash", (0, 12, -16)),
                                            ((0.0, 0.0), "periodic", (8, 0, -16))])
@pytest.mark.parametrize("q", [0, 16])
def test_student_equal_to_the_ema_on_consistent_crops_gives_zero(norm, vol, shift, q):
    """Two crops whose shared voxels carry the same input -- a global norm on any volume, or the per-patch
    z-score on a volume periodic in the shift and the patch -- and a student that IS the EMA: the term is
    zero to float precision (a teacher that differs is not)."""
    cfg = _cfg()
    lay = cfg.layout()
    V = _hash_volume() if vol == "hash" else _periodic_volume()
    b = prep.batch1(_pair_item(V, (24, 24, 24), shift, cfg, sym=21, norm=norm))
    x, tg, wt = prep.prepare(b, torch.device("cpu"), layout=lay)
    f = _pointwise(lay)
    E = OV.teacher_batch(b, f, lay, torch.device("cpu"), q=q)
    ones = torch.ones_like(wt[:, :1])
    o = L.overlap_loss(f(x), E, ones, ones, layout=lay)
    assert float(o["overlap_vox"]) > 0
    assert float(o["overlap"]) < 1e-5 and float(o["overlap_kl"]) < 1e-5 and float(o["overlap_l1"]) < 1e-4
    E2 = OV.teacher_batch(b, _pointwise(lay, off=0.3), lay, torch.device("cpu"), q=q)
    o2 = L.overlap_loss(f(x), E2, ones, ones, layout=lay)
    assert float(o2["overlap_kl"]) > 1e-3 and float(o2["overlap_l1"]) > 1.0


def test_the_term_is_one_way_and_zero_outside_the_mask():
    lay = _cfg().layout()
    torch.manual_seed(0)
    logit = torch.randn(1, lay.cout, 8, 8, 8, requires_grad=True)
    E = torch.rand(1, OV.field_channels(lay), 8, 8, 8)
    E[:, -1] = 0
    E[:, -1, :4] = 1
    o = L.overlap_loss(logit, E, None, None, layout=lay)
    o["overlap"].backward()
    g = logit.grad[0, :lay.nprob]
    assert g[:, 4:].abs().max() == 0 and g[:, :4].abs().max() > 0
    assert not E.requires_grad


# --------------------------------------------------------------------- the trainer

def _factory(cfg, lay, V):
    def gen():
        i = 0
        while True:
            s = [(0, 10, -8), (12, 0, 0), (-16, 14, 8)][i % 3]
            it = _pair_item(V, (30, 30, 30), s, cfg, sym=i % 48, k=cfg.rungs[i % len(cfg.rungs)],
                            norm=(0.0, 0.0))
            if i % 2:                        # half the items carry no second window
                it = {k: v for k, v in it.items() if k not in sample.OV_KEYS}
            it["cm"] = torch.zeros((P // 2,) * 3, dtype=torch.uint8)
            it["cx"] = torch.from_numpy(np.random.default_rng(i).integers(0, 255, (1, P, P, P), dtype=np.uint8))
            i += 1
            yield it
    return gen


@pytest.mark.parametrize("cascade,sub", [("off", 0), ("self", 16)])
def test_the_trainer_logs_the_overlap_term(tmp_path, cascade, sub):
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cfg = Config(out=str(tmp_path / "run"), size="1m", patch=P, batch=1, ctx=(1, 2, 3), rungs=(2, 3),
                 aff_offsets=(4, 8), ect_block=8, steps=20, eval_every=100, ckpt_every=0, workers=0,
                 compile=False, cascade=cascade, cascade_drop=0.0, loss_overlap=0.5, overlap_sub=sub)
    lay = cfg.layout()
    TR.train(cfg, patches_factory=_factory(cfg, lay, _hash_volume()), device="cpu", accum=1)
    rows = [json.loads(q) for q in open(tmp_path / "run" / "logs" / "train.jsonl")]
    rows = [r for r in rows if "overlap_n" in r]
    assert rows and all(r["w_overlap"] == 0.5 for r in rows)
    assert sum(r["overlap_n"] for r in rows) > 0
    got = [r for r in rows if r["overlap_n"]]
    assert all(np.isfinite(r["overlap"]) and 0 < r["overlap_vox"] < 1 for r in got)


def test_the_term_off_logs_nothing_and_the_sampler_draws_nothing(tmp_path):
    assert Config().overlap_p == 0.0 and Config().loss_overlap == 0.0
    assert TR._ov_log({"n": 3, "overlap": 1.0}, 0.0) == {}
    from rvsm.config import FINGERPRINT_EXCLUDE, LOSS_SWITCH_FIELDS
    assert {"overlap_p", "overlap_sub", "loss_overlap"} <= set(FINGERPRINT_EXCLUDE)
    assert "loss_overlap" in LOSS_SWITCH_FIELDS
    assert Config(overlap_p=0.5, loss_overlap=0.1).fingerprint() == Config().fingerprint()


def test_patches_draw_a_second_window_that_reads_the_shifted_corner(synth_run):
    """On real region stores: with overlap_p 1 every draw with room carries `OV_KEYS`, whose CT cube is
    the pyramid at ov_lo and whose context cubes are `ladder.context`'s there."""
    from dataclasses import replace

    from rvsm import ladder
    cfg = replace(synth_run.cfg, overlap_p=1.0)
    ds = sample.Patches(cfg, root=synth_run.root, ct=cfg.ct, ax=synth_run.ax,
                        region_records=synth_run.regions, seed=5)
    it = iter(ds)
    n = 0
    for _ in range(30):
        item = next(it)
        if "ov_ct" not in item:
            continue
        n += 1
        k, lo, lo2 = int(item["rung"]), item["lo"].numpy(), item["ov_lo"].numpy()
        s = lo2 - lo
        assert (s != 0).any() and (s % 2 == 0).all() and (np.abs(s) <= P // 2).all()
        assert np.array_equal(item["ov_ct"][0].numpy(), ladder.read_rung(ds.pyr, k, lo2, ds.patch, np.uint8))
        want = ladder.context(ds.ct, lo2, ds.patch, ds.ctx, rung=k)
        assert all(np.array_equal(item["ov_ct"][1 + j].numpy(), c) for j, c in enumerate(want))
        assert np.allclose(item["ov_cyx"].numpy(), sample.axis_cyx(ds.ax, k, lo2, P))
    assert n > 0
