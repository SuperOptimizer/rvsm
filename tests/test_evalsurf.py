"""Evaluation: topology on synthetic volumes, store-vs-store on a held-out region, pooling and the fit."""
import numpy as np
import pytest

from rvsm import evalsurf as E


def _slab(n=32, y0=12, y1=16):
    v = np.zeros((n, n, n), bool)
    v[:, y0:y1, :] = True
    return v


def test_betti_of_a_slab_is_one_component_no_loop_no_cavity():
    b0, b1, b2, chi = E.betti(_slab())
    assert (b0, b1, b2) == (1, 0, 0)
    assert chi == 1                                   # a contractible block: chi = b0 - b1 + b2
    assert E.betti(np.zeros((8, 8, 8), bool)) == (0, 0, 0, 0)


def test_betti_counts_a_cavity_and_a_handle():
    v = np.ones((12, 12, 12), bool)
    v[5:7, 5:7, 5:7] = False                          # a sealed bubble
    assert E.betti(v)[2] == 1
    t = np.zeros((16, 16, 16), bool)                  # a solid torus-ish ring in one plane: one loop
    zz, yy, xx = np.mgrid[:16, :16, :16]
    r = np.sqrt((yy - 8.0) ** 2 + (xx - 8.0) ** 2)
    t[(r >= 3) & (r <= 5) & (zz >= 7) & (zz <= 9)] = True
    assert E.betti(t)[1] == 1


def test_a_bridge_between_two_sheets_shows_up_as_a_betti0_error():
    """Two disjoint sheets are b0 = 2; a merge bridge joins them into one, which is exactly the error
    the round gate watches (`merge_frac` sees it at the surface points, this sees it in the volume)."""
    ref = np.zeros((32, 32, 32), bool)
    ref[:, 8:11, :] = True
    ref[:, 20:23, :] = True
    assert E.betti(ref)[0] == 2
    bridged = ref.copy()
    bridged[14:18, 10:21, 14:18] = True                # one column joining the two
    assert E.betti(bridged)[0] == 1
    e = E.betti_error(bridged, ref, margin=0, band=0, dilate=0)
    assert e["betti0"] == 1 and e["betti0_ref"] == 2 and e["betti0_err"] == 1
    assert E.betti_error(ref, ref, margin=0, band=0, dilate=0)["betti0_err"] == 0


def test_betti_error_crops_the_margin_and_bands_around_the_reference():
    ref = _slab(32, 12, 16)
    pred = ref.copy()
    pred[:, 26:29, :] = True                          # a correctly predicted sheet with no reference
    far = E.betti_error(pred, ref, margin=4, band=0, dilate=0)
    near = E.betti_error(pred, ref, margin=4, band=3, dilate=0)
    assert far["betti0"] == 2 and far["betti0_err"] == 1   # unbanded: charged for the extra sheet
    assert near["betti0"] == 1 and near["betti0_err"] == 0  # banded: only the reference's neighbourhood
    assert near["betti_interior_vox"] == 24 ** 3


def _u8(mask):
    return (np.asarray(mask, bool) * 255).astype(np.uint8)


def test_compare_stores_is_perfect_against_itself():
    a = _u8(_slab(32, 12, 17))
    m = E.compare_stores(a, a, margin=4)
    assert m["dice"] == pytest.approx(1.0)
    assert m["overlap"] == pytest.approx(1.0) and m["precision"] == pytest.approx(1.0)
    assert m["betti0_err"] == 0 and m["betti1_err"] == 0
    assert m["skel_recall"] == pytest.approx(1.0)
    assert m["erl_frac"] == pytest.approx(1.0)        # one unbroken skeleton piece
    assert m["skel_vox"] > 0


def test_compare_stores_charges_a_break_in_the_skeleton():
    a = _u8(_slab(48, 20, 25))
    b = a.copy()
    b[:, :, 20:28] = 0                                 # a hole straight through the sheet
    m = E.compare_stores(a, b, margin=4)
    assert m["dice"] < 1.0 and m["overlap"] < 1.0
    assert m["precision"] == pytest.approx(1.0)        # nothing extra was predicted
    assert m["skel_recall"] < 1.0
    assert 0.0 < m["erl_frac"] < 1.0                   # the skeleton is cut in two
    full = E.compare_stores(a, a, margin=4)
    assert m["erl_vox"] < full["erl_vox"]


def test_compare_stores_refuses_a_different_box():
    with pytest.raises(AssertionError):
        E.compare_stores(_u8(_slab(16)), _u8(_slab(32)))


def _row(name, recall, n, erl_sq, path):
    return {"surface": name, "recall@2": recall, "recall@4": recall, "recall@8": recall,
            "offset_frac": 1.0, "merge_runs": 1.0, "merge_frac": 0.0, "offset_le3": 1.0,
            "offset_mean": 0.0, "offset_std": 1.0, "continuity": recall, "hit_frac": recall,
            "mean_run": 4.0, "_n": n, "_noff": n, "_nhit": n, "_ncont": n, "_nruns": n,
            "_absoff": np.zeros(n, np.float32),
            "path_um": path, "break_um": 0.0, "merge_um": 0.0,
            "_runsq": erl_sq, "_runsq_break": erl_sq, "_runsq_merge": erl_sq,
            "erl_um": erl_sq / path, "erl_break_um": erl_sq / path, "erl_merge_um": erl_sq / path,
            "lost_break_frac": 0.0, "lost_merge_frac": 0.0}


def test_pool_is_point_weighted_and_erl_is_length_weighted():
    rows = [_row("a", 0.9, 100, 4000.0, 100.0), _row("b", 0.5, 300, 900.0, 300.0)]
    p = E.pool(rows)
    assert p["n_surfaces"] == 2 and p["n_points"] == 400
    assert p["recall@4"] == pytest.approx((0.9 * 100 + 0.5 * 300) / 400)
    assert p["erl_um"] == pytest.approx(4900.0 / 400.0)
    assert p["path_um"] == pytest.approx(400.0)
    assert E.pool([]) == {}


def test_bootstrap_ci_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    rows = [_row(f"s{i}", float(v), 100 + 10 * i, 1000.0 * (i + 1), 100.0)
            for i, v in enumerate(rng.uniform(0.4, 0.95, 8))]
    p = E.pool(rows)
    ci = E.bootstrap(rows, n=200, seed=1)
    for k in ("recall@4", "continuity", "erl_um"):
        lo, hi = ci[k]
        assert lo <= p[k] <= hi, (k, lo, p[k], hi)
        assert lo < hi
    assert E.bootstrap(rows[:1]) == {}                 # one surface cannot be resampled


def test_fit_curve_recovers_a_planted_asymptote():
    steps = np.arange(1, 41) * 500.0
    c, a, al = 0.86, 3.0, 0.45
    vals = c - a * steps ** (-al)
    f = E.fit_curve(steps, vals)
    assert f["model"] in ("power", "logis")
    assert f["asymptote"] == pytest.approx(c, abs=0.02)
    assert f["last_value"] == pytest.approx(vals[-1], abs=0.01)
    assert f["slope_per_10k"] > 0 and f["steps_to_95"] > 0
    assert f["n"] == 40


def test_fit_curve_needs_enough_points_and_reports_a_flat_run():
    assert E.fit_curve([1, 2], [0.1, 0.2])["model"] is None
    steps = np.arange(1, 31) * 1000.0
    flat = np.full(30, 0.7)
    f = E.fit_curve(steps, flat, smooth=3)
    assert abs(f["slope_per_10k"]) < 1e-3 and f["remaining"] < 1e-3


def test_trilerp_and_profile_sample_the_volume():
    V = np.zeros((8, 8, 8), np.float32)
    V[4, 4, 4] = 1.0
    q = np.array([[4.0, 4.0, 4.0], [4.5, 4.0, 4.0], [100.0, 0.0, 0.0]], np.float32)
    v = E.trilerp(V, q)
    assert v[0] == pytest.approx(1.0) and v[1] == pytest.approx(0.5) and v[2] == 0.0
    n = np.tile(np.array([[1.0, 0.0, 0.0]], np.float32), (3, 1))
    S = E.profile(V, q, n, far=2)
    assert S.shape == (5, 3) and S[2, 0] == pytest.approx(1.0)


def test_evaluate_with_only_a_reference_store():
    a = _u8(_slab(32, 12, 17))
    out = E.evaluate(lambda o, s: a, ((0, 0, 0), (32, 32, 32)), ref_u8=a, margin=4)
    assert out["box"] == [0, 0, 0, 32, 32, 32]
    assert out["vs_store"]["dice"] == pytest.approx(1.0)
    assert "metrics" not in out                        # no tifxyz -> no mesh suite, and no pretending


class _Lazy:
    """A store stand-in that records every read, so a test can see how much of it was ever whole."""

    def __init__(self, v):
        self.v, self.shape, self.ndim, self.reads = v, v.shape, v.ndim, []

    def __getitem__(self, sl):
        blk = self.v[sl]
        self.reads.append(blk.size)
        return blk


def _sheets(n=96, seed=0):
    """Two wavy sheets, the second with a hole: something with components, loops and a broken skeleton."""
    zz, yy, xx = np.mgrid[:n, :n, :n]
    a = (np.abs(yy - (n * 0.3 + 4 * np.sin(xx / 9.0))) < 2.5) | (np.abs(yy - (n * 0.7 + 3 * np.cos(zz / 7.0))) < 2.5)
    b = a.copy()
    b[n // 3:n // 2, :, n // 3:n // 2] = False
    rng = np.random.default_rng(seed)
    b |= rng.random(a.shape) > 0.9995                  # specks: extra components
    return _u8(a), _u8(b)


def test_compare_stores_in_blocks_keeps_the_whole_volume_counts():
    """Dice, overlap, precision and skeleton recall do not depend on the block size (the halo makes the
    band and the skeleton exact on every core); one block IS the whole-volume comparison."""
    a, b = _sheets()
    whole = E.compare_stores(a, b, margin=4, block=10 ** 6)
    assert whole["betti_blocks"] == 1
    for blk in (32, 40):
        m = E.compare_stores(a, b, margin=4, block=blk)
        assert m["betti_blocks"] > 1
        for k in ("dice", "overlap", "precision", "skel_recall", "n_ref", "n_pred", "skel_vox",
                  "betti_interior_vox"):
            assert m[k] == pytest.approx(whole[k]), k
        assert m["betti0_err"] >= 1                    # the specks are still charged, block by block
        assert m["erl_vox"] <= whole["erl_vox"] + 1e-9  # a block face can only cut a run
    same = E.compare_stores(a, a, margin=4, block=32)
    assert same["dice"] == 1.0 and same["betti0_err"] == 0 and same["betti1_err"] == 0


def test_compare_stores_streams_the_reference_and_stays_small():
    """The gate's comparison reads the reference store block by block (never whole) and its peak
    allocation is a few blocks, not a few copies of the volume -- the whole-region float64 EDTs, int32
    labels and float32 skeleton stacks are what OOMed the 64 GB production host."""
    import tracemalloc
    n, blk = 128, 32
    a, b = _sheets(n)
    ref = _Lazy(a)
    tracemalloc.start()
    E.compare_stores(ref, b, margin=4, block=blk)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert max(ref.reads) <= (blk + 2 * 8) ** 3, "a read bigger than one haloed block"
    assert sum(ref.reads) < 3 * a.size                  # each voxel read about once (plus halos)
    # bounded by the haloed BLOCK (~45 bytes a voxel of it: the EDT's feature transform and distances,
    # the labels), not by the volume: the whole-volume version held >= 20 bytes a voxel of the VOLUME
    # (float64 EDT + int32 feature transform), 40 MB here
    halo = 8
    assert peak < 64 * (blk + 2 * halo) ** 3, f"peak {peak / 2 ** 20:.1f} MB"
    assert peak < 4 * a.size, f"peak {peak / 2 ** 20:.1f} MB for a {a.size / 2 ** 20:.1f} Mvox volume"
