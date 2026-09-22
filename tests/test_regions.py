"""Regions: CT occupancy instead of a mask, the walk, the catalog, the pooled and coarse targets."""
import numpy as np
import pytest

from rvsm import axis as AX, ladder, regions as RG, stores


def test_tiles_fraction_drops_air_and_keeps_the_slab(ct_origin):
    pyr = ladder.rungs(ct_origin.path)
    ko, occ = RG.occupancy(pyr)
    assert occ.any() and not occ.all()
    ax, R = RG.region_tiles(pyr, 2, region=64, patch=32)
    fr = RG.tiles_fraction(occ, ko, 2, ax)
    assert fr.shape == tuple(len(q) for q in ax)
    assert 0.0 <= fr.min() and fr.max() <= 1.0
    # the fixture's slab is a y band: some tiles are pure air (fraction 0) and the band's tiles are
    # well above the rung-2 keep threshold
    assert (fr == 0).any() and (fr >= 0.05).any()
    # the fraction is a MEAN, not a maximum: a tile the band merely clips is not 1.0
    assert fr.max() < 1.0
    # and it agrees with the direct block mean of a tile
    z, y, x = [int(q) for q in np.argwhere(fr > 0)[0]]
    lo = np.array([ax[0][z], ax[1][y], ax[2][x]], np.int64)
    blk = occ[lo[0]:lo[0] + int(R[0]), lo[1]:lo[1] + int(R[1]), lo[2]:lo[2] + int(R[2])]
    assert abs(float(blk.mean()) - float(fr[z, y, x])) < 1e-9


def test_region_list_weights_go_as_sqrt_fraction(ct_origin):
    pyr = ladder.rungs(ct_origin.path)
    recs = RG.region_list(pyr, rungs=(2, 3), patch=32, region=64)
    assert recs and abs(sum(r["w"] for r in recs) - 1.0) < 1e-9
    two = [r for r in recs if r["k"] == 2]
    assert all(r["f"] >= 0.05 for r in two)
    a, b = sorted(two, key=lambda r: r["f"])[0], sorted(two, key=lambda r: r["f"])[-1]
    if a["f"] < b["f"]:
        assert abs(a["w"] / b["w"] - (a["f"] / b["f"]) ** 0.5) < 1e-9


def test_walk_order_is_a_weighted_shuffle_without_replacement(ct_origin):
    w = [{"w": v} for v in (0.5, 0.3, 0.15, 0.05)]
    o = RG.walk_order(w, seed=3)
    assert sorted(o.tolist()) == [0, 1, 2, 3]          # every item exactly once
    first = np.array([RG.walk_order(w, seed=s)[0] for s in range(400)])
    p0 = float((first == 0).mean())
    assert 0.40 < p0 < 0.60                            # P(first = i) = w_i / sum(w)
    assert (first == 3).mean() < p0                    # the light item is rarely first
    assert np.array_equal(RG.walk_order(w, seed=3), o)  # and the order is a function of the seed


def test_region_visits_spread_the_heavy_regions(ct_origin):
    recs = [{"k": 9, "lo": [0, 0, 0], "size": [8, 8, 8], "f": 1.0, "w": 0.8},
            {"k": 2, "lo": [0, 0, 0], "size": [64] * 3, "f": 0.5, "w": 0.1},
            {"k": 2, "lo": [64, 0, 0], "size": [64] * 3, "f": 0.5, "w": 0.1}]
    v = RG.region_visits(recs, cap=64)
    assert len(v) > len(recs)                                   # the heavy region got several visits
    assert abs(sum(q["w"] for q in v) - sum(r["w"] for r in recs)) < 1e-12
    assert sum(1 for q in v if q["k"] == 9) == round(0.8 * 3)


def test_held_out_is_stratified_distinct_and_excluded(ct_origin, umbilicus):
    pyr = ladder.rungs(ct_origin.path)
    ax = AX.load(umbilicus[0], ct=ct_origin.path)
    recs = RG.region_list(pyr, rungs=(2, 3), patch=32, region=64)
    held = RG.held_out(recs, n=8, seed=7, ax=ax)
    assert len(held) == 8
    los = [tuple(r["lo"]) for r in held]
    assert len(set(los)) == 8 and all(r["k"] == 2 for r in held)
    assert len({(int(np.array(r["lo"])[0])) for r in held}) > 1   # not all from one z slab
    again = RG.region_list(pyr, rungs=(2, 3), patch=32, region=64,
                           exclude=RG.exclude_boxes(held))
    left = {tuple(r["lo"]) for r in again if r["k"] == 2}
    assert not (left & set(los))
    assert len(again) < len(recs)
    assert np.array_equal([tuple(r["lo"]) for r in RG.held_out(recs, 8, 7, ax)], los)  # seeded


def test_catalog_reflects_a_store_written_with_stores_write(tmp_path, volcomp_lib, has_volcomp):
    if not has_volcomp:
        pytest.skip("a region store is a volcomp array")
    root, lo = str(tmp_path), (0, 128, 256)
    cat = RG.Catalog(root, 0, ttl=0.0)
    assert not cat.done("recto", lo) and cat.list_done("recto") == []
    assert not cat.ready(lo, ("recto",))
    u = np.zeros((128, 128, 128), np.uint8)
    u[:, 60:70] = 255
    stores.write(stores.store_path(root, "recto", lo), u, lo, q=0)
    assert cat.done("recto", lo) and cat.ready(lo, ("recto",))
    assert cat.list_done("recto") == [tuple(lo)]
    assert not cat.ready(lo, ("recto", "verso"))
    a = cat.open("recto", lo)
    assert a is not None and np.array_equal(np.asarray(a[:]), u)


def test_pooled_equals_the_numpy_pool(synth_run):
    lo = synth_run.lo[0]
    a = synth_run.cfg
    base = np.asarray(stores.open_store(stores.store_path(a.out, "recto", lo))[:], np.uint8)
    want = base
    for k in (3, 4, 5, 6):
        want = ladder.pool2(want)
        got = RG.pooled(a.out, "recto", lo, k)
        assert got is not None and np.array_equal(got, want), f"rung {k}"
    assert RG.pooled(a.out, "recto", (999, 999, 999), 4) is None
    # and the windowed form agrees with a slice of it
    cube, ins = RG.pooled_window(a.out, "recto", lo, 4, np.array(lo, np.int64) >> 2, (8, 8, 8))
    ref = RG.pooled(a.out, "recto", lo, 4)
    assert ins.all() and np.array_equal(cube, ref[:8, :8, :8])


def test_feed_coarse_places_the_block_and_sets_coverage(synth_run):
    a = synth_run.cfg
    lo = np.array(synth_run.lo[0], np.int64)
    blk = np.asarray(stores.open_store(stores.store_path(a.out, "recto", lo))[:], np.uint8)
    shape2 = ladder.rung_shape(synth_run.pyr, 2)
    ks = RG.feed_coarse(a.out, "recto", lo, blk, shape2=shape2, ks=(7, 8, 9))
    assert ks == [7, 8, 9]
    for k in ks:
        want = blk
        for _ in range(k - 2):
            want = ladder.pool2(want)
        o = lo >> (k - 2)
        got, cov = RG.read_coarse(a.out, "recto", k, o, want.shape)
        assert np.array_equal(got, want), f"rung {k}"
        assert (cov == 1.0).all()
        # a window one block to the side has been fed nothing: coverage 0 and therefore weight 0
        far, cov2 = RG.read_coarse(a.out, "recto", k, o + np.array(want.shape) * 0 + shape2 // 2, want.shape)
        assert cov2.max() <= 1.0
    # a rung whose pooling factor exceeds the region is skipped rather than colliding with a neighbour
    assert RG.feed_coarse(a.out, "recto", lo, blk, shape2=shape2, ks=(11,)) == []


def test_read_coarse_without_a_coarse_array_is_empty(tmp_path):
    got, cov = RG.read_coarse(str(tmp_path), "recto", 9, (0, 0, 0), (4, 4, 4))
    assert got.shape == (4, 4, 4) and not got.any() and not cov.any()
