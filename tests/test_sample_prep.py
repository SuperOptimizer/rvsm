"""The sampler and the GPU prep: the item contract, where weight comes from, and the channel order."""
import json
import os

import numpy as np
import pytest
import torch

from rvsm import ladder, model as M, prep, sample, stores
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
    x, tgt, w = prep.prepare(b, torch.device("cpu"), layout=L,
                             cascade=prep.Cascade("mask", drop=0.0, noise=False))
    assert x.shape == (1, L.cin, 8, 8, 8) and tgt.shape == (1, 4, 8, 8, 8) and w.shape == tgt.shape
    # the image cubes are z-scored, the rest are not
    assert abs(float(x[0, 0].mean())) < 1e-4 and abs(float(x[0, 0].std()) - 1) < 0.05
    # cascade at 10 (the mask source, asked for by name): the coarse block upsampled 2x, in 0..1
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


# --------------------------------------------------------------------- compact target rows

def _recto_only(compact=True, seed=0, p=8, k=2, sym=0):
    """A round-0 item: recto has target and weight, verso / midline / thickness are all zero."""
    rng = np.random.default_rng(seed)
    ax = np.array([[0.0, 64.0], [16.0, 16.0], [20.0, 20.0]])
    ct = rng.integers(0, 255, (10, p, p, p), dtype=np.uint8)
    tg = np.zeros((4, p, p, p), np.uint8)
    w = np.zeros_like(tg)
    tg[0] = rng.integers(0, 255, (p, p, p), dtype=np.uint8)
    w[0] = rng.integers(0, 2, (p, p, p), dtype=np.uint8) * 255
    return sample.rung_item(ct, tg, w, k, (0, 4, 4), ax, sym=sym, compact=compact,
                            cm=rng.integers(0, 255, (p // 2,) * 3, dtype=np.uint8),
                            cx=rng.integers(0, 255, (1, p, p, p), dtype=np.uint8),
                            lo1=(0, 2, 2), rmax=37.5, meta=np.linspace(0, 1, 5, dtype=np.float32))


def _prep_all(item, mode="mask"):
    L = Config().layout()
    net = None
    if mode in ("self", "mix"):
        torch.manual_seed(0)
        net = M.build("1m", cin=L.cin, cout=L.cout, verbose=False)
    cas = prep.Cascade(mode, drop=0.0, noise=True, seed=3, net=net)
    return prep.prepare(prep.batch1(item), torch.device("cpu"), cascade=cas, layout=L)


def test_a_recto_only_item_carries_one_target_row():
    it = _recto_only()
    assert tuple(sorted(it)) == tuple(sorted(RUNG_ITEM_KEYS))
    assert it["tgt"].shape == (1, 8, 8, 8) and it["w"].shape == (1, 8, 8, 8)
    assert torch.nonzero(it["tch"]).reshape(-1).tolist() == [0] and it["tch"].shape == (4,)
    assert prep.shapes(it) == prep.shapes(_recto_only(compact=False))
    t, w = prep.full_tw(it)
    full = _recto_only(compact=False)
    assert torch.equal(t, full["tgt"]) and torch.equal(w, full["w"])


@pytest.mark.parametrize("sym,mode", [(0, "mask"), (13, "mask"), (47, "off"), (5, "self")])
def test_prepare_on_a_compact_item_is_prepare_on_the_full_one(sym, mode):
    a = _prep_all(_recto_only(compact=True, sym=sym), mode)
    b = _prep_all(_recto_only(compact=False, sym=sym), mode)
    for u, v in zip(a, b):
        assert u.shape == v.shape and u.dtype == v.dtype and torch.equal(u, v)


def test_a_full_support_item_is_unchanged_by_compact():
    a, b = _fake_item(), _fake_item()
    c = sample.rung_item(a["ct"].numpy(), a["tgt"].numpy(), a["w"].numpy(), 2, (0, 4, 4),
                         np.array([[0.0, 64.0], [16.0, 16.0], [20.0, 20.0]]), compact=True,
                         cm=a["cm"].numpy(), cx=a["cx"].numpy(), lo1=(0, 2, 2), rmax=37.5,
                         meta=np.linspace(0, 1, 5, dtype=np.float32))
    assert set(c) == set(b)
    for key in b:
        assert torch.equal(c[key], b[key]), key
    assert c["tch"].tolist() == [1, 1, 1, 1]


def test_a_row_with_target_but_no_weight_is_kept():
    """The rule is weight OR target: a dropped row must be zero in both, or the expansion is not exact."""
    tg = np.zeros((4, 4, 4, 4), np.uint8)
    w = np.zeros_like(tg)
    tg[2, 1, 1, 1] = 9
    w[0, 0, 0, 0] = 255
    tch, t, ww = sample.compact_rows(tg, w)
    assert tch.tolist() == [1, 0, 1, 0] and t.shape[0] == 2 and ww.shape[0] == 2
    empty = sample.compact_rows(np.zeros_like(tg), np.zeros_like(w))
    assert empty[0].tolist() == [0, 0, 0, 0] and empty[1].shape == (0, 4, 4, 4)


def test_expand_tw_scatters_each_sample_by_its_own_rows():
    rng = np.random.default_rng(4)
    fulls, items = [], []
    for rows in ([0, 2], [1, 3]):          # same count, different rows: these DO collate
        tg = np.zeros((4, 4, 4, 4), np.uint8)
        w = np.zeros_like(tg)
        for r in rows:
            tg[r] = rng.integers(1, 255, (4, 4, 4))
            w[r] = 255
        tch, t, ww = sample.compact_rows(tg, w)
        items.append({"tgt": torch.from_numpy(t), "w": torch.from_numpy(ww), "tch": torch.from_numpy(tch)})
        fulls.append((torch.from_numpy(tg), torch.from_numpy(w)))
    b = torch.utils.data.default_collate(items)
    t, w = prep.expand_tw(b["tgt"], b["w"], b["tch"])
    assert torch.equal(t, torch.stack([f[0] for f in fulls]))
    assert torch.equal(w, torch.stack([f[1] for f in fulls]))
    # no `tch` (a grid item written before it existed) means every row is there
    assert prep.expand_tw(t, w, None)[0] is t


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


@pytest.mark.parametrize("mode", ["self", "mask", "mix", "off"])
def test_a_training_draw_sends_only_the_cascade_extra_its_mode_reads(synth_run, mode):
    """`cm` is empty unless the cascade reads the mask source, `cx` unless it runs the self pass; what
    `prepare` builds from the lean item is the full item's, bit for bit, with the same cascade draws."""
    from dataclasses import replace
    cfg = replace(synth_run.cfg, cascade=mode)
    L = cfg.layout()
    lean = sample.Patches(cfg, root=synth_run.root, ct=cfg.ct, ax=synth_run.ax,
                          region_records=synth_run.regions, seed=7)
    full = sample.Patches(cfg, root=synth_run.root, ct=cfg.ct, ax=synth_run.ax,
                          region_records=synth_run.regions, seed=7)
    full.lean = False
    torch.manual_seed(0)
    net = M.build("1m", cin=L.cin, cout=L.cout, verbose=False) if mode in ("self", "mix") else None
    want_cm, want_cx = mode in ("mask", "mix"), mode in ("self", "mix")
    for i, (a, b) in enumerate(zip(iter(lean), iter(full))):
        if i == 6:
            break
        assert (a["cm"].numel() > 0) == want_cm and (a["cx"].numel() > 0) == want_cx
        assert b["cm"].numel() > 0 and b["cx"].numel() > 0
        outs = []
        for it in (a, b):
            cas = prep.Cascade(mode, self_p=0.5, drop=0.3, noise=True, net=net, seed=11 + i)
            outs.append(prep.prepare(prep.batch1(it), torch.device("cpu"), cascade=cas, layout=L))
        for u, v in zip(*outs):
            assert torch.equal(u, v)


def test_a_mask_cascade_on_an_item_without_its_block_fails_loudly(small_cfg):
    L = small_cfg.layout()
    it = _fake_item(nctx=L.nctx, p=8, k=2)
    it["cm"] = torch.zeros((0, 0, 0), dtype=torch.uint8)
    with pytest.raises(AssertionError, match="coarse block"):
        prep.prepare(prep.batch1(it), torch.device("cpu"), layout=L,
                     cascade=prep.Cascade("mask", drop=0.0, noise=False))


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
    assert all(q["cm"].numel() == 0 and q["cx"].numel() == 0 for q in got)   # no cascade is fed
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
    assert not ds.compact and b["tgt"].shape[1] == 4 and b["tch"].all()   # a batch of 2: full rows


def test_loader_at_batch_one_sends_compact_items_that_prepare_to_the_full_ones(synth_run):
    full = _patches(synth_run, seed=5)
    comp = _patches(synth_run, seed=5)
    dl = sample.loader(comp, workers=0, batch=1)
    assert comp.compact
    L = synth_run.cfg.layout()
    fewer = 0
    for i, (a, b) in enumerate(zip(iter(full), iter(dl))):
        if i == 12:
            break
        fewer += int(b["tgt"].shape[1] < 4)
        assert b["tch"].shape == (1, 4)
        pa = prep.prepare(prep.batch1(a), torch.device("cpu"), layout=L)
        pb = prep.prepare(b, torch.device("cpu"), layout=L)
        for u, v in zip(pa, pb):
            assert torch.equal(u, v)
    assert fewer, "no item of the fixture dropped a row: the test says nothing"


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
    dd = type(d)(sample.grid_dir(str(d), ("recto", "verso")))    # the fixture has verso stores
    m = os.path.getmtime(dd / "grid.json")
    g2 = sample.val_grid(synth_run.cfg, held, spill=str(d), **kw)
    assert len(g2) == len(g) and os.path.getmtime(dd / "grid.json") == m


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
            if ds._region_of(k, lo) is None:
                continue          # a straddling window is stitched now: see the test below
            ct = ladder.read_rung(ds.pyr, k, lo, p, dtype=np.uint8)
            a, b = ds._rung_target(k, lo, ct), _old_rung_target(ds, k, lo, ct)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), (k, lo)
            n += int(a[1].any())
            near += int(ds._near_axis(k, lo, p).any())
    assert n > 0, "no window carried any weight: the comparison proved nothing"
    assert near > 0, "no window touched the axis: the near-axis path went untested"


def test_rungs_3_to_6_stitch_region_stores_per_voxel(synth_run, monkeypatch):
    """paris4 step 400: a rung-3 window straddling two region stores got weight 0 everywhere, and a
    window at rungs 4-6 (as large as a region's footprint or larger) nearly always straddles -- those
    rungs trained on nothing. Each voxel now takes its own region store's target and `inside`: the
    stitched window equals the per-region reads, a missing store zeroes only its own voxels, and a
    held-out region never supplies a target."""
    import shutil

    from rvsm import stores
    monkeypatch.setattr(sample, "AXIS_R_UM", 0.0)
    ds = _patches(synth_run)
    ds._open()
    R, p = int(synth_run.cfg.region), ds.patch
    k = 3
    # a rung-3 window centred on the corner shared by all eight regions: it straddles every one
    c3 = np.full(3, R // 2, np.int64)
    lo = c3 - p // 2
    assert ds._region_of(k, lo) is None
    ct = ladder.read_rung(ds.pyr, k, lo, p, dtype=np.uint8)
    tg, w = ds._rung_target(k, lo, ct)
    ri = ds.channels.index("recto")
    assert w[ri].max() > 0, "a straddling rung-3 window must be supervised"
    # every octant equals that region's own store read at rung 3
    h = int(p[0]) // 2
    for r in synth_run.lo:
        o = (np.array(r, np.int64) >> 1) - lo
        o = np.clip(o, 0, h)
        a = ds.cat.open("recto", np.array(r, np.int64))
        want, ins = stores.read_store(a, k, lo + o, np.full(3, h, np.int64))
        sl = tuple(slice(int(o[j]), int(o[j]) + h) for j in range(3))
        assert ins.all()
        assert np.array_equal(np.where(ct[sl] > 0, want, 0), tg[ri][sl])
    # remove one region's recto store: only its octant loses the weight
    gone = np.array(synth_run.lo[-1], np.int64)
    shutil.rmtree(stores.store_path(synth_run.root, "recto", gone))
    ds.cat = type(ds.cat)(ds.root, ds.round, ttl=0.0)
    tg2, w2 = ds._rung_target(k, lo, ct)
    og = np.clip((gone >> 1) - lo, 0, h)
    sl = tuple(slice(int(og[j]), int(og[j]) + h) for j in range(3))
    assert not w2[ri][sl].any()
    keep = w2[ri].copy()
    keep[sl] = 0
    ref = w[ri].copy()
    ref[sl] = 0
    assert np.array_equal(keep, ref)
    # a held-out region is never a stitched neighbour
    first = np.array(synth_run.lo[0], np.int64)
    dh = _patches(synth_run, heldout=[{"lo": [int(v) for v in first], "size": [R] * 3, "k": 2}])
    dh._open()
    _, wh = dh._rung_target(k, lo, ct)
    o0 = np.clip((first >> 1) - lo, 0, h)
    sl0 = tuple(slice(int(o0[j]), int(o0[j]) + h) for j in range(3))
    assert not wh[ri][sl0].any() and wh[ri].max() > 0


def test_rung_3_to_6_windows_are_drawn_on_the_home_region(synth_run):
    """A visit at rungs 3-6 draws around the one region the producer made for it: at rung 3 (footprint
    larger than a window) wholly inside it, and every drawn window reads a finished store."""
    ds = _patches(synth_run, seed=3)
    ds._open()
    recs = [v for v in ds.visits if int(v["k"]) == 3]
    assert recs
    rng = np.random.default_rng(0)
    p = ds.patch
    for rec in recs[:4]:
        home = np.array(ds._home(rec), np.int64)
        box = ds._draw_rec(rec)
        f0, fs = home >> 1, int(synth_run.cfg.region) >> 1
        blo, bsz = np.array(box["lo"]), np.array(box["size"])
        if fs > int(p[0]):
            assert (blo >= f0).all() and (blo + bsz <= f0 + fs).all()
        for _ in range(4):
            lo = rng.integers(blo, np.maximum(blo + bsz - p, blo) + 1)
            assert ds._region_of(3, lo) is not None and tuple(ds._region_of(3, lo)) == tuple(home)


def test_a_visit_whose_home_region_is_held_out_is_dead(synth_run):
    """A rung 3-6 visit whose home region is held out can only draw weight-0 windows (the held-out
    store is never a target and nothing else is produced for it): the walk skips it, at every rung
    3-6, and never a rung-2 or coarse visit."""
    recs = [v for v in synth_run.regions if int(v["k"]) == 3]
    assert recs
    ds0 = _patches(synth_run)
    ds0._open()
    home = ds0._home(recs[0])
    ds = _patches(synth_run, heldout=[{"lo": list(home), "size": [int(synth_run.cfg.region)] * 3, "k": 2}])
    ds._open()
    assert ds._dead(recs[0]) and not ds0._dead(recs[0])
    assert not any(ds._dead(v) for v in ds.visits if int(v["k"]) == 2)
    it = iter(ds)
    for _ in range(30):
        item = next(it)
        assert int(item["rung"]) != 3 or ds._home_r != home


def test_a_visit_whose_home_region_is_not_a_walk_region_is_dead(synth_run):
    """paris4 steps 780-1140: a rung-3 tile passed the occupancy test on the mean of its eight
    regions, but its home region was air, so the visit's 128 windows were all `air_keep` windows of
    weight 0. A rung 3-6 visit whose home is not a rung-2 record of the walk is dead, and
    `run.region_route` does not produce that home."""
    from rvsm import run as RUN
    recs3 = [v for v in synth_run.regions if int(v["k"]) == 3]
    assert recs3
    ds0 = _patches(synth_run)
    ds0._open()
    home = ds0._home(recs3[0])
    assert home in ds0._fine2 and not ds0._dead(recs3[0])
    # the same walk without the home's rung-2 record: the rung-3 visit is dead
    kept = [r for r in synth_run.regions
            if not (int(r["k"]) == 2 and tuple(int(v) for v in r["lo"]) == home)]
    ds = sample.Patches(synth_run.cfg, root=synth_run.root, ct=synth_run.cfg.ct, ax=synth_run.ax,
                        region_records=kept)
    ds._open()
    assert ds._dead(recs3[0])
    assert not any(ds._dead(v) for v in ds.visits if int(v["k"]) == 2)
    # and the producer's route leaves that home out, while the full walk keeps it
    route, _ = RUN.region_route(synth_run.cfg, ds.visits, list(range(len(ds.visits))))
    assert home not in route
    route0, _ = RUN.region_route(synth_run.cfg, ds0.visits, list(range(len(ds0.visits))))
    assert home in route0


def test_a_visit_keeps_at_most_its_air_budget_of_air_windows(synth_run, monkeypatch):
    """A visit to a region that is all air used to fill all its windows with `air_keep` windows of
    weight 0. The air windows a visit keeps are capped at `air_keep` of its windows: one pass of the
    walk over an all-air visit yields exactly its budget, then the visit ends on its fail limit."""
    from dataclasses import replace
    cfg = replace(synth_run.cfg, air_keep=0.25, windows_per_region=8)
    ds = sample.Patches(cfg, root=synth_run.root, ct=cfg.ct, ax=synth_run.ax,
                        region_records=synth_run.regions, seed=0)
    ds._open()
    assert ds.air_budget() == 2
    monkeypatch.setattr(sample.ladder, "read_rung",
                        lambda pyr, k, lo, shape, dtype=np.uint8: np.zeros(tuple(ladder.shape3(shape)), dtype))
    rec = dict(ds.visits[0], k=2)
    ds.visits, ds.order = [rec], np.array([0])
    monkeypatch.setattr(ds, "_visitable", lambda r: True)
    calls = []
    real = ds._draw
    monkeypatch.setattr(ds, "_draw", lambda rng, r, air_ok=True: calls.append(air_ok) or real(rng, r, air_ok))
    it = iter(ds)
    got = [next(it) for _ in range(2)]
    assert all(not np.asarray(g["w"]).any() for g in got)
    n_first = len(calls)
    next(it)                                   # the third item is from the NEXT pass over the walk
    pass1 = calls[:len(calls)]
    assert pass1.count(False) >= 8 * 8 - 1, "the spent budget must reject every later air window"
    assert n_first < len(calls)


def test_the_default_cascade_is_self_with_drop_0_3():
    """The mask source is off by default (the student read the target out of it: paris4 step 12000),
    and a run can switch cascade source on resume."""
    from rvsm.config import Config
    c = Config()
    assert c.cascade == "self" and c.cascade_drop == pytest.approx(0.3)
    assert Config(cascade="mix", cascade_drop=0.1).fingerprint() == c.fingerprint()


def test_a_self_cascade_never_reads_the_coarse_target(small_cfg):
    """`Cascade("self")` builds its channel from the model's own coarse pass (or zeros): the coarse
    TARGET block `cm` is never touched, whatever the draw."""
    L = small_cfg.layout()
    b = dict(prep.batch1(_fake_item(nctx=L.nctx, p=8, k=2)))

    class NoCm(dict):
        def __getitem__(self, k):
            assert k != "cm", "the self cascade read the coarse target"
            return super().__getitem__(k)

        def get(self, k, d=None):
            assert k != "cm", "the self cascade read the coarse target"
            return super().get(k, d)
    nb = NoCm(b)
    x = torch.zeros((1, L.cin, 8, 8, 8))
    net = M.build("1m", cin=L.cin, cout=L.cout, verbose=False)
    for seed in range(12):
        c = prep.Cascade("self", self_p=0.0, drop=0.3, noise=True, net=net, seed=seed)
        out = c.channel(nb, x, b["norm"], torch.float32)
        assert out.shape == (1, 1, 8, 8, 8)
        assert float(c.last_self[0]) in (0.0, 1.0)     # self-scored, or dropped -- never mask
        if float(c.last_self[0]) == 0.0:
            assert float(out.abs().sum()) == 0.0


def test_training_coarse_targets_carry_no_weight_over_held_out_regions(monkeypatch):
    """Rungs 7-11 read the coarse arrays, which fold in EVERY produced region -- the held-out ones too.
    The training sampler zeroes the coverage over a held-out footprint (rounded outward); the
    validation view (no held-out list) keeps it (review D03)."""
    from rvsm import regions as RG
    monkeypatch.setattr(RG, "read_coarse", lambda root, ch, k, lo, shape, r=0: (
        np.full(tuple(shape), 200, np.uint8), np.ones(tuple(shape), np.float32)))
    ds = sample.Patches.__new__(sample.Patches)
    ds.root, ds.round = "/nowhere", 0
    ds.cfg = type("C", (), {"region": 1024})()
    k, lo, shp = 7, (0, 0, 0), (64, 64, 64)                 # rung 7: 32 rung-2 voxels a voxel
    ds._held = {(1024, 1024, 0)}                              # footprint z 32..64, y 32..64, x 0..32
    _, cov = ds._source("recto", k, lo, shp)
    assert float(cov[32:64, 32:64, 0:32].max()) == 0.0
    assert float(cov.sum()) == 64 ** 3 - 32 ** 3
    ds._held = set()                                          # the validation view keeps everything
    _, cov = ds._source("recto", k, lo, shp)
    assert float(cov.min()) == 1.0
    odd = np.ones((8, 8, 8), np.float32)                      # outward rounding of a partial voxel
    sample.zero_footprints(odd, 7, (0, 0, 0), [(24, 24, 24)], 16)   # rung-2 24..40 straddles 0 | 1
    assert odd[0, 0, 0] == 0 and odd[1, 1, 1] == 0 and odd[2, 2, 2] == 1 and float(odd.sum()) == 512 - 8


def test_rung_3_and_4_distances_read_their_own_rung_stores_never_a_pool(synth_run, monkeypatch):
    """D02: the producer writes `midline_r3` / `thickness_r4` ... at their own rungs, and a rung-3/4
    distance target must come from THAT store, read at its own rung and origin -- not the rung-2 field
    pooled (its codes are rung-2 voxels, and a pooled offset code is no code at all). Each rung's
    stores hold a different constant sentinel and the padding past `shape_true` another one; a window
    straddling every region must see exactly its own rung's constant, and code 0 stays weight 0."""
    from rvsm import stores, targets as TG
    monkeypatch.setattr(sample, "AXIS_R_UM", 0.0)
    R = int(synth_run.cfg.region)
    SENT = {2: 160, 3: 170, 4: 180}
    THICK = {3: 150, 4: 140}
    PAD = 99
    every = [np.array([z, y, x], np.int64) * R for z in (0, 1) for y in (0, 1) for x in (0, 1)]
    for r in every:                                         # all eight, not just the occupied ones
        stores.write(stores.store_path(synth_run.root, "midline", r),
                     np.full((R,) * 3, SENT[2], np.uint8), r, rung=2, channels=("midline",), q=0)
        for k in (3, 4):
            n = R >> (k - 2)
            for kind, val in (("midline", SENT[k]), ("thickness", THICK[k])):
                blk = np.full((128,) * 3, PAD, np.uint8)    # padded up to a store's 128 multiple
                blk[:n, :n, :n] = val
                if k == 3 and kind == "midline":
                    blk[:4] = 0                             # a no-data slab at the region's top
                stores.write(stores.store_path(synth_run.root, TG.channel(kind, k), r), blk,
                             r >> (k - 2), rung=k, channels=(TG.channel(kind, k),), q=0,
                             attrs={"shape_true": [n] * 3})
    ds = _patches(synth_run)
    ds._open()
    p = ds.patch
    mid, thk = ds.channels.index("midline"), ds.channels.index("thickness")
    for k in (2, 3, 4):
        # rungs 3-4: centred on the corner all eight regions share; rung 2 is never stitched
        lo = np.full(3, R >> (k - 2), np.int64) - p // 2 if k > 2 else np.full(3, 16, np.int64)
        v, ins = ds._source("midline", k, lo, p)
        assert ins.all(), f"rung {k}: every voxel lies in some region's own store"
        assert set(np.unique(v).tolist()) <= {0, SENT[k]}, (k, np.unique(v))
        assert (v == SENT[k]).any()
        if k > 2:
            tv, tins = ds._source("thickness", k, lo, p)
            assert tins.all() and set(np.unique(tv).tolist()) == {THICK[k]}
    # rung 3: the no-data slab of each region reads as code 0 and carries weight 0
    k = 3
    lo = np.full(3, R >> 1, np.int64) - p // 2
    ct = ladder.read_rung(ds.pyr, k, lo, p, dtype=np.uint8)
    v, _ = ds._source("midline", k, lo, p)
    zero = np.zeros(tuple(p), bool)
    zero[16:20] = True                                      # the lower regions' top 4 slices
    assert (v[zero] == 0).all() and (v[~zero] == SENT[3]).all()
    tg, w = ds._rung_target(k, lo, ct)
    assert (ct[~zero] > 0).any()
    assert not w[mid][zero].any()
    assert (w[mid][~zero & (ct > 0)] == 255).all()
    assert (tg[mid][~zero & (ct > 0)] == SENT[3]).all()
    assert (w[thk][ct > 0] == 255).all()
    # past a store's shape_true is its padding: a window reaching there reads inside = 0, never PAD
    last = every[-1] >> 2
    lo4 = last + (R >> 2) - p // 2                          # half past the last region's rung-4 extent
    v4, ins4 = ds._source("midline", 4, lo4, p)
    assert PAD not in np.unique(v4).tolist()
    h = int(p[0]) // 2
    assert ins4[:h, :h, :h].all() and not ins4[h:, h:, h:].any()


def test_verso_targets_make_a_new_grid_generation_and_leave_the_recto_grid(synth_run, tmp_path):
    """The grid key carries the schema, the store round and the heads with targets; when every
    held-out region gets a verso store, the verso-bearing grid is built in its own directory and the
    recto reference grid is not touched (pass-3 P3-04)."""
    import shutil
    held = [r for r in synth_run.regions if r["k"] == 2][:1]
    kw = dict(root=synth_run.root, ct=synth_run.cfg.ct, ax=synth_run.ax, rungs=(2,))
    d = tmp_path / "grid"
    lo = tuple(int(v) for v in held[0]["lo"])
    vp = stores.store_path(synth_run.root, "verso", lo, 0)
    shutil.move(vp, vp + ".aside")                                           # round 0 before verso
    try:
        g = sample.val_grid(synth_run.cfg, held, spill=str(d), **kw)
    finally:
        shutil.move(vp + ".aside", vp)                                       # ... the verso appears
    m = os.path.getmtime(d / "grid.json")
    key0 = json.load(open(d / "grid.json"))["key"]
    g2 = sample.val_grid(synth_run.cfg, held, spill=str(d), **kw)
    d2 = tmp_path / "grid_recto+verso"
    assert d2.is_dir() and json.load(open(d2 / "grid.json"))["key"] != key0
    assert os.path.getmtime(d / "grid.json") == m, "the recto reference grid must not be rebuilt"
    assert len(g2) == len(g)
    assert any(float(it["tgt"][1].float().sum()) > 0 for it in g2)          # verso targets present
    assert not any(float(it["tgt"][1].float().sum()) > 0 for it in g)       # ... and not in the old
    corners = [(2, np.zeros(3, np.int64))]
    c = synth_run.cfg
    k = sample.grid_key(c, corners, 0, ("recto",))
    assert k != sample.grid_key(c, corners, 1, ("recto",))                  # the store round
    assert k != sample.grid_key(c, corners, 0, ("recto", "verso"))          # the heads
    import unittest.mock as um
    with um.patch.object(sample, "GRID_SCHEMA", "grid-v999"):
        assert k != sample.grid_key(c, corners, 0, ("recto",))              # the schema


def test_a_committed_regeneration_changes_the_grid_and_readers_see_one_bundle(synth_run, tmp_path):
    """P4-04: the grid key includes the identity of every held-out label store a reader sees, so a
    committed verso regeneration (a new verso + its fields) rebuilds the verso/field grid; before the
    commit nothing changes."""
    import shutil
    from rvsm import regions as RG, targets as TG
    held = [r for r in synth_run.regions if r["k"] == 2][:1]
    kw = dict(root=synth_run.root, ct=synth_run.cfg.ct, ax=synth_run.ax, rungs=(2,))
    d = tmp_path / "grid"
    sample.val_grid(synth_run.cfg, held, spill=str(d), **kw)
    dd = type(d)(sample.grid_dir(str(d), ("recto", "verso")))
    k0 = json.load(open(dd / "grid.json"))["key"]
    lo = tuple(int(v) for v in held[0]["lo"])
    base = stores.store_path(synth_run.root, "verso", lo, 0)
    shutil.copytree(stores.store_path(synth_run.root, "recto", lo, 0), stores.gen_path(base, 1))
    sample.val_grid(synth_run.cfg, held, spill=str(d), **kw)
    assert json.load(open(dd / "grid.json"))["key"] == k0, "an uncommitted generation changes nothing"
    try:
        stores.commit_bundle(synth_run.root, lo, 0, 1)
        assert RG.Catalog(synth_run.root, 0).path("verso", lo) == stores.gen_path(base, 1)
        sample.val_grid(synth_run.cfg, held, spill=str(d), **kw)
        assert json.load(open(dd / "grid.json"))["key"] != k0         # the new labels: a new grid
    finally:
        os.remove(os.path.join(synth_run.root, "stores", "round_0", "bundle",
                               stores.region_name(lo)[:-5] + ".json"))
        shutil.rmtree(stores.gen_path(base, 1))
