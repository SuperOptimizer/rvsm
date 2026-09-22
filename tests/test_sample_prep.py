"""The sampler and the GPU prep: the item contract, where weight comes from, and the channel order."""
import os

import numpy as np
import pytest
import torch

from rvsm import ladder, model as M, prep, sample
from rvsm.config import RUNG_ITEM_KEYS, Config


# --------------------------------------------------------------------- the item contract

def _fake_item(nctx=9, p=8, k=2, seed=0):
    """A compact sample built by hand: enough to exercise `prepare` without touching a store."""
    rng = np.random.default_rng(seed)
    ax = np.array([[0.0, 64.0], [16.0, 16.0], [20.0, 20.0]])
    ct = rng.integers(0, 255, (1 + nctx, p, p, p), dtype=np.uint8)
    tg = rng.integers(0, 255, (4, p, p, p), dtype=np.uint8)
    w = np.full((4, p, p, p), 255, np.uint8)
    return sample.rung_item(ct, tg, w, k, (0, 4, 4), ax, sym=0,
                            cm=rng.integers(0, 255, (p // 2,) * 3, dtype=np.uint8),
                            cx=rng.integers(0, 255, (1, p, p, p), dtype=np.uint8),
                            lo1=(0, 2, 2), rmax=37.5, meta=np.linspace(0, 1, 5, dtype=np.float32))


def test_rung_item_yields_exactly_the_contract():
    it = _fake_item()
    assert tuple(sorted(it)) == tuple(sorted(RUNG_ITEM_KEYS))
    assert it["ct"].dtype == torch.uint8 and it["tgt"].dtype == torch.uint8
    assert it["cyx"].shape == (2, 8) and it["cyx1"].shape == (2, 8)
    assert int(it["rung"]) == 2 and it["meta"].shape == (5,)
    # the optional extras are ZERO, never absent: a consumer must never have to branch
    bare = sample.rung_item(np.zeros((2, 4, 4, 4), np.uint8), np.zeros((4, 4, 4, 4), np.uint8),
                            np.zeros((4, 4, 4, 4), np.uint8), 3, (0, 0, 0),
                            np.array([[0.0, 8.0], [2.0, 2.0], [2.0, 2.0]]))
    assert tuple(sorted(bare)) == tuple(sorted(RUNG_ITEM_KEYS))
    assert bare["cm"].shape == (2, 2, 2) and bare["cx"].shape == (1, 4, 4, 4)
    assert float(bare["rmax"]) == 0.0 and not bare["meta"].any()


# --------------------------------------------------------------------- prep: the channel order

def test_prepare_channel_order_matches_the_layout():
    L = Config().layout()
    assert (L.i_cas, L.i_radius, L.i_meta, L.i_scale, L.i_rad, L.cin) == (10, 11, 12, 17, 18, 21)
    b = prep.batch1(_fake_item(nctx=L.nctx, p=8, k=5))
    x, tgt, w = prep.prepare(b, torch.device("cpu"), layout=L)
    assert x.shape == (1, L.cin, 8, 8, 8) and tgt.shape == (1, 4, 8, 8, 8) and w.shape == tgt.shape
    # the image cubes are z-scored, the rest are not
    assert abs(float(x[0, 0].mean())) < 1e-4 and abs(float(x[0, 0].std()) - 1) < 0.05
    # cascade at 10: the coarse block upsampled 2x, a probability in 0..1
    cas = M.up2x(b["cm"][:, None].float() / 255.0, (8, 8, 8))
    assert torch.allclose(x[:, L.i_cas:L.i_cas + 1], cas, atol=1e-6)
    # radius at 11, the five scan planes at 12..16
    rad = prep.radius_t(b["cyx"], b["lo"], (8, 8, 8), b["rmax"].reshape(-1))
    assert torch.allclose(x[:, L.i_radius:L.i_radius + 1], rad, atol=1e-6)
    for i in range(L.n_meta):
        assert torch.allclose(x[0, L.i_meta + i], b["meta"][0, i].expand(8, 8, 8), atol=1e-6)
    # the scale plane at 17 is (k - 2) / 9, and the radial unit vector is the last three
    assert torch.allclose(x[:, L.i_scale], torch.full((1, 8, 8, 8), (5 - 2) / 9.0))
    assert torch.allclose(x[:, L.i_rad:], prep.radial_t(b["cyx"], b["lo"], (8, 8, 8)), atol=1e-6)
    assert float(x[:, L.i_rad].abs().max()) == 0.0        # the z component of the radial vector is 0
    # and what prep built is exactly what the net's stem expects
    assert prep.shapes(_fake_item(nctx=L.nctx))[0] == L.cin


def test_prepare_norad_zeroes_only_the_radial_channels():
    L = Config().layout()
    b = prep.batch1(_fake_item(nctx=L.nctx, p=8))
    x, _, _ = prep.prepare(b, torch.device("cpu"), norad=True, layout=L)
    assert not x[:, L.i_rad:].any() and x[:, L.i_scale].abs().sum() == 0  # rung 2 -> scale 0
    assert x[:, L.i_radius].any()


def test_sym_apply_t_matches_numpy_for_all_48_symmetries():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, 6, 6, 6)).astype(np.float32)
    x[-3:] /= np.linalg.norm(x[-3:], axis=0, keepdims=True) + 1e-9
    tg = rng.standard_normal((2, 6, 6, 6)).astype(np.float32)
    xt, tt = torch.from_numpy(x)[None], torch.from_numpy(tg)[None]
    for s in range(48):
        a, b = sample.sym_apply(s, x, tg)
        at, bt = prep.sym_apply_t(s, xt, tt)
        assert np.allclose(at[0].numpy(), a, atol=1e-6), f"sym {s}"
        assert np.allclose(bt[0].numpy(), b, atol=1e-6), f"sym {s}"
    assert np.array_equal(sample.sym_apply(0, x, tg)[0], x)   # 0 is the identity


def test_cascade_modes_shapes(small_cfg):
    L = small_cfg.layout()
    b = prep.batch1(_fake_item(nctx=L.nctx, p=8, k=2))
    dev = torch.device("cpu")
    x = torch.zeros((1, L.cin, 8, 8, 8))
    masked = prep.Cascade("mask", drop=0.0, noise=False).channel(b, x, b["norm"], torch.float32)
    assert masked.shape == (1, 1, 8, 8, 8) and 0.0 <= float(masked.min()) and float(masked.max()) <= 1.0
    net = M.build("1m", cin=L.cin, cout=L.cout, verbose=False)
    for mode, p in (("self", 1.0), ("mix", 0.5)):
        c = prep.Cascade(mode, self_p=p, drop=0.0, noise=True, net=net, seed=1)
        out = c.channel(b, x, b["norm"], torch.float32)
        assert out.shape == (1, 1, 8, 8, 8)
        assert c.last_self is not None and c.last_self.shape == (1,)
    # dropped altogether, and the top of the ladder has no rung above it
    assert not prep.Cascade("mask", drop=1.0).channel(b, x, b["norm"], torch.float32).any()
    top = prep.batch1(_fake_item(nctx=L.nctx, p=8, k=ladder.NRUNGS - 1))
    assert not prep.Cascade("mask", drop=0.0, noise=False).channel(
        top, x, top["norm"], torch.float32).any()
    with pytest.raises(AssertionError):
        prep.Cascade("nonsense")


def test_autocast_is_a_noop_on_cpu():
    with prep.autocast(torch.device("cpu")):
        assert torch.zeros(1).dtype == torch.float32


# --------------------------------------------------------------------- the sampler against real stores

def _patches(sr, **kw):
    return sample.Patches(sr.cfg, root=sr.root, ct=sr.cfg.ct, ax=sr.ax, region_records=sr.regions, **kw)


def _dense(ds, region_los):
    """The window inside ONE region store with the most non-air CT: the fixture's slab is a thin band,
    so a window picked blind is usually pure air and says nothing about weights."""
    best, lo_best = -1.0, None
    p = int(ds.patch[0])
    for r in region_los:
        for dz in range(0, int(ds.cfg.region), p):
            for dy in range(0, int(ds.cfg.region), p):
                for dx in range(0, int(ds.cfg.region), p):
                    lo = np.array(r, np.int64) + np.array([dz, dy, dx], np.int64)
                    if ds._region_of(2, lo) is None:
                        continue
                    m = float((ladder.read_rung(ds.pyr, 2, lo, ds.patch, dtype=np.uint8) > 0).mean())
                    if m > best:
                        best, lo_best = m, lo
    assert lo_best is not None and best > 0.1, "the fixture has no dense window"
    return lo_best


def test_patches_yields_rung_items_at_rungs_2_and_3(synth_run):
    ds = _patches(synth_run, seed=1)
    got, ks = [], set()
    it = iter(ds)
    for _ in range(40):
        item = next(it)
        got.append(item)
        ks.add(int(item["rung"]))
    assert ks <= {2, 3} and 2 in ks
    for item in got:
        assert tuple(sorted(item)) == tuple(sorted(RUNG_ITEM_KEYS))
        assert item["ct"].shape == (1 + len(synth_run.cfg.ctx), 32, 32, 32)
        assert item["tgt"].shape == (4, 32, 32, 32)
    assert any(int(q["w"][0].max()) > 0 for q in got), "no sample carried any recto weight"
    # and a batch of them goes straight through prep into the net
    L = synth_run.cfg.layout()
    b = torch.utils.data.default_collate(got[:2])
    x, tgt, w = prep.prepare(b, torch.device("cpu"), layout=L)
    assert x.shape == (2, L.cin, 32, 32, 32)
    net = M.build("1m", cin=L.cin, cout=L.cout, verbose=False).eval()
    with torch.no_grad():
        assert net(x).shape == (2, L.cout, 32, 32, 32)


def test_a_window_without_a_store_has_weight_zero(synth_run, monkeypatch):
    monkeypatch.setattr(sample, "AXIS_R_UM", 0.0)
    ds = _patches(synth_run)
    ds._open()
    lo = _dense(ds, synth_run.lo)
    ct = ladder.read_rung(ds.pyr, 2, lo, ds.patch, dtype=np.uint8)
    tg, w = ds._rung_target(2, lo, ct)
    assert w[0].max() > 0                      # inside one region store: recto is supervised
    # a window straddling two region stores is not stitched: every channel loses its weight there
    straddle = np.array(synth_run.lo[0], np.int64) + np.array([0, synth_run.cfg.region - 16, 0], np.int64)
    ct2 = ladder.read_rung(ds.pyr, 2, straddle, ds.patch, dtype=np.uint8)
    assert ds._region_of(2, straddle) is None
    assert not ds._rung_target(2, straddle, ct2)[1].any()
    # and a channel whose store simply does not exist keeps weight 0 while the others are unaffected
    import shutil

    from rvsm import stores
    shutil.rmtree(stores.store_path(synth_run.root, "verso", ds._region_of(2, lo)))
    ds.cat = type(ds.cat)(ds.root, ds.round, ttl=0.0)
    tg3, w3 = ds._rung_target(2, lo, ct)
    assert not w3[1].any() and w3[0].max() > 0


def test_code_zero_means_weight_zero_for_a_distance_channel(synth_run, monkeypatch):
    monkeypatch.setattr(sample, "AXIS_R_UM", 0.0)
    ds = _patches(synth_run)
    ds._open()
    lo = _dense(ds, synth_run.lo)
    ct = ladder.read_rung(ds.pyr, 2, lo, ds.patch, dtype=np.uint8)
    tg, w = ds._rung_target(2, lo, ct)
    mid = ds.channels.index("midline")
    assert tg[mid].max() > 0 and w[mid].max() > 0
    assert not w[mid][tg[mid] == 0].any(), "code 0 is the no-data marker and must carry weight 0"
    # a distance is never pooled: above rung 4 the channel has no target at all
    for k in (5, 7):
        lok = lo >> (k - 2)
        ctk = ladder.read_rung(ds.pyr, k, lok, ds.patch, dtype=np.uint8)
        assert not ds._rung_target(k, lok, ctk)[1][mid].any()


def test_weight_is_zero_near_the_umbilicus_for_verso(synth_run):
    ds = _patches(synth_run)
    ds._open()
    # the fixture axis runs through the volume centre; 400 um is 167 rung-2 voxels, so the dense window
    # (which straddles the centre, where the slab is) lies entirely inside the core
    lo = _dense(ds, synth_run.lo)
    ct = ladder.read_rung(ds.pyr, 2, lo, ds.patch, dtype=np.uint8)
    assert ds._near_axis(2, lo, ds.patch).all()
    tg, w = ds._rung_target(2, lo, ct)
    for c in ("verso", "midline", "thickness"):
        assert not w[ds.channels.index(c)].any(), f"{c} must carry no weight inside the core"
    assert w[ds.channels.index("recto")].max() > 0, "recto is supervised everywhere"


def test_label_free_needs_no_stores(region_cfg, umbilicus):
    """Pretraining draws windows from the CT alone: no catalog, no store, no target."""
    from rvsm import axis as AX
    ax = AX.load(region_cfg.umbilicus, ct=region_cfg.ct)
    ds = sample.Patches(region_cfg, root=str(region_cfg.out) + "_empty", ct=region_cfg.ct, ax=ax,
                        label_free=True, seed=2)
    it = iter(ds)
    got = [next(it) for _ in range(8)]
    assert all(tuple(sorted(q)) == tuple(sorted(RUNG_ITEM_KEYS)) for q in got)
    assert all(not q["w"].any() and not q["tgt"].any() for q in got)
    assert any(int(q["ct"][0].max()) > 0 for q in got)


def test_val_grid_is_fixed_and_over_the_held_out_regions(synth_run):
    from rvsm import regions as RG
    held = RG.held_out(synth_run.regions, n=2, seed=0, ax=synth_run.ax)
    g = sample.val_grid(synth_run.cfg, held, root=synth_run.root, ct=synth_run.cfg.ct,
                        ax=synth_run.ax, rungs=(2,), limit=4)
    assert g and all(tuple(sorted(q)) == tuple(sorted(RUNG_ITEM_KEYS)) for q in g)
    assert all(int(q["sym"]) == 0 for q in g)
    g2 = sample.val_grid(synth_run.cfg, held, root=synth_run.root, ct=synth_run.cfg.ct,
                         ax=synth_run.ax, rungs=(2,), limit=4)
    assert [q["lo"].tolist() for q in g] == [q["lo"].tolist() for q in g2]


def test_loader_collates_the_contract(synth_run):
    ds = _patches(synth_run, seed=5)
    dl = sample.loader(ds, workers=0, batch=2)
    b = next(iter(dl))
    assert tuple(sorted(b)) == tuple(sorted(RUNG_ITEM_KEYS))
    assert b["ct"].shape[0] == 2 and b["lo"].shape == (2, 3)


def test_a_spilled_val_grid_is_the_grid_and_is_reused(synth_run, tmp_path):
    """`val_grid(spill=...)` writes the items compressed and hands back a `DiskGrid`: the same items
    as the in-memory grid, in the same order, built the same with threads, and a second call with the
    same grid reuses the directory instead of rebuilding it."""
    held = [r for r in synth_run.regions if r["k"] == 2][:1]
    kw = dict(root=synth_run.root, ct=synth_run.cfg.ct, ax=synth_run.ax, rungs=(2, 3))
    ref = sample.val_grid(synth_run.cfg, held, **kw)
    d = tmp_path / "grid"
    g = sample.val_grid(synth_run.cfg, held, spill=str(d), threads=3, **kw)
    assert isinstance(g, sample.DiskGrid) and len(g) == len(ref) and g
    for a, b in zip(ref, g):
        assert sorted(a) == sorted(b)
        for k in a:
            va, vb = a[k], b[k]
            if torch.is_tensor(va):
                assert torch.equal(va, vb), k
            else:
                assert np.array_equal(np.asarray(va), np.asarray(vb)), k
    assert len(g[:2]) == min(2, len(ref))
    m = os.path.getmtime(d / "grid.json")
    g2 = sample.val_grid(synth_run.cfg, held, spill=str(d), **kw)
    assert len(g2) == len(g) and os.path.getmtime(d / "grid.json") == m


def test_the_cascade_self_pass_takes_a_module_without_len(small_cfg):
    """A compiled module (torch's OptimizedModule) raises on len() -- and so on bool() -- so the self
    pass must choose its forward with `is not None`, never `or` (it did, and the first compiled GPU
    run died in its first self pass)."""
    class NoBool(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.calls = 0

        def __len__(self):
            raise TypeError("no len")

        def forward(self, x):
            self.calls += 1
            return self.inner(x)

    L = small_cfg.layout()
    b = prep.batch1(_fake_item(nctx=L.nctx, p=8, k=2))
    x = torch.zeros((1, L.cin, 8, 8, 8))
    net = M.build("1m", cin=L.cin, cout=L.cout, verbose=False)
    f = NoBool(net)
    c = prep.Cascade("self", self_p=1.0, drop=0.0, noise=False, net=net, fwd=f, seed=1)
    out = c.channel(b, x, b["norm"], torch.float32)
    assert out.shape == (1, 1, 8, 8, 8) and f.calls == 1


def test_the_visit_super_cube_is_ladder_context_bit_for_bit(synth_run):
    """`Patches._ctx_cube` slices a visit's context cubes from one super-cube per rung; every window a
    visit can draw must get exactly `ladder.context`'s cubes (and the tenth cube `cx`)."""
    ds = _patches(synth_run, seed=3)
    ds._open()
    ds.SUPER_MAX = 1 << 40                      # cache every rung, including the ones prod would read
    rng = np.random.default_rng(0)
    p = ds.patch
    for rec in [v for v in ds.visits if int(v["k"]) in (2, 3)][:6]:
        k = int(rec["k"])
        rlo, rsz = np.array(rec["lo"], np.int64), np.array(rec["size"], np.int64)
        hi = np.maximum(rlo + rsz - p, rlo)
        for _ in range(4):
            lo = rng.integers(np.minimum(rlo, hi), hi + 1)
            want = ladder.context(ds.ct, lo, p, ds.ctx, rung=k)
            got = [ds._ctx_cube(rec, k, d, lo, p) for d in ds.ctx]
            assert all(np.array_equal(a, b) for a, b in zip(want, got)), (k, lo)
            ref = ds._cascade_extras(k, lo, p)["cx"]
            assert np.array_equal(ds._cascade_extras(k, lo, p, rec=rec)["cx"], ref)
    ds.SUPER_MAX = 0                             # nothing cached: the per-window read path
    ds._vkey = None
    rec = [v for v in ds.visits if int(v["k"]) == 2][0]
    lo = np.array(rec["lo"], np.int64)
    assert all(np.array_equal(a, b) for a, b in zip(
        ladder.context(ds.ct, lo, p, ds.ctx, rung=2), [ds._ctx_cube(rec, 2, d, lo, p) for d in ds.ctx]))


def _old_rung_target(ds, k, lo, ct):
    """The float64 formula `_rung_target` replaced, with the store read directly (no visit cache)."""
    from rvsm import stores
    from rvsm.sample import DIST_CHANNELS, DIST_MAX_RUNG, NEAR_AXIS_ZERO, RW

    def src(chan):
        r = ds._region_of(k, lo, ds.patch)
        if r is None:
            return np.zeros(tuple(ds.patch), np.uint8), np.zeros(tuple(ds.patch), np.float32)
        a = ds.cat.open(chan, r)
        if a is None:
            return np.zeros(tuple(ds.patch), np.uint8), np.zeros(tuple(ds.patch), np.float32)
        v, ins = stores.read_store(a, k, lo, ds.patch)
        return v, ins.astype(np.float32)

    p = tuple(int(v) for v in ds.patch)
    tg = np.zeros((len(ds.channels),) + p, np.uint8)
    w = np.zeros_like(tg)
    air = ct > 0
    r = ds._region_of(k, lo)
    rwv = src(RW)[0].astype(np.float32) / 255.0 if (r is not None and ds.cat.done(RW, r)) else None
    shape = p
    a = __import__("rvsm.axis", fromlist=["x"]).axis_at(ds.ax, k)
    z = np.arange(shape[0]) + int(lo[0])
    cy, cx = np.interp(z, a[0], a[1]), np.interp(z, a[0], a[2])
    dy = (np.arange(shape[1]) + int(lo[1]))[None, :, None] - cy[:, None, None]
    dx = (np.arange(shape[2]) + int(lo[2]))[None, None, :] - cx[:, None, None]
    from rvsm.sample import AXIS_R_UM
    near = (dy * dy + dx * dx) < (AXIS_R_UM / ladder.rung_um(k)) ** 2
    for c, chan in enumerate(ds.channels):
        dist = chan in DIST_CHANNELS
        if dist and int(k) > DIST_MAX_RUNG:
            continue
        v, frac = src(chan)
        np.copyto(tg[c], v, where=air)
        ok = frac * air
        if dist:
            ok = ok * (v != 0)
        if chan in NEAR_AXIS_ZERO:
            ok = ok * ~near
        if rwv is not None and not dist:
            ok = ok * rwv
        w[c] = np.clip(np.rint(255.0 * ok), 0, 255).astype(np.uint8)
    return tg, w


def test_the_uint8_targets_are_the_float_formula(synth_run):
    """`_rung_target` (uint8 weights, the rung-3 pool decoded once per visit, the near-axis early out)
    against the float64 formula it replaced, over windows at rungs 2 and 3 -- including the ones on
    the scroll axis, where the near-axis mask is not empty."""
    ds = _patches(synth_run, seed=5)
    ds._open()
    rng = np.random.default_rng(2)
    p = ds.patch
    n = near = 0
    for rec in [v for v in ds.visits if int(v["k"]) in (2, 3)][:8]:
        k = int(rec["k"])
        rlo, rsz = np.array(rec["lo"], np.int64), np.array(rec["size"], np.int64)
        hi = np.maximum(rlo + rsz - p, rlo)
        cands = [rng.integers(np.minimum(rlo, hi), hi + 1) for _ in range(3)]
        from rvsm import axis as AX
        ak = AX.axis_at(ds.ax, k)                      # and one window centred on the scroll axis
        zc = int(rlo[0]) + int(p[0]) // 2
        on = np.array([zc, np.interp(zc, ak[0], ak[1]), np.interp(zc, ak[0], ak[2])]) - p // 2
        cands.append(np.clip(on.astype(np.int64), np.minimum(rlo, hi), hi))
        for lo in cands:
            ct = ladder.read_rung(ds.pyr, k, lo, p, dtype=np.uint8)
            a, b = ds._rung_target(k, lo, ct), _old_rung_target(ds, k, lo, ct)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), (k, lo)
            n += int(a[1].any())
            near += int(ds._near_axis(k, lo, p).any())
    assert n > 0, "no window carried any weight: the comparison proved nothing"
    assert near > 0, "no window touched the axis: the near-axis path went untested"
