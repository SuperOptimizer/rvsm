"""The distance stores: the paired-v2 target definition (raw distances, reach, same-sheet pairing, no
recto-only fallback), the encoding, the axis exclusion, and `jobs` determinism.

The geometry is analytic throughout: planes perpendicular to x with the umbilicus far away in -x, so the
radial direction is +x everywhere, a recto face at a and a verso face at a - t give d_r = x - a,
d_v = x - (a - t), midline x - (a - t/2) and thickness t exactly (see `rvsm.targets`, THE SIGN)."""
import filecmp
import os

import numpy as np
import pytest

from rvsm import ladder, stores, targets

KW = dict(block=64, halo=16, reach=12)        # small halo for speed: reach must stay below it


def _read(root, kind, lo, rung=2, round_=0):
    p = stores.store_path(root, targets.channel(kind, rung), lo, round_)
    assert stores.is_done(p), p
    return np.asarray(stores.open_store(p)[:], np.uint8)


def _bands(n, rectos, versos, half=1, zy=4):
    """(recto u8, verso u8, dy, dx) of a (zy, zy, n) box with one 2*half+1 voxel band per face and the
    radial direction +x everywhere."""
    x = np.arange(n)
    def band(cs):
        b = np.zeros(n, bool)
        for c in cs:
            b |= np.abs(x - c) <= half
        return np.broadcast_to(np.where(b, np.uint8(255), np.uint8(0)), (zy, zy, n)).copy()
    shape = (zy, zy, n)
    return band(rectos), band(versos), np.zeros(shape, np.float32), np.ones(shape, np.float32)


# ------------------------------------------------------------------------------------------ encoding

def test_encoding_round_trips_and_reserves_code_zero():
    d = np.array([0.0, 10.0, -10.0, 1000.0, -1000.0], np.float32)
    ok = np.ones(5, bool)
    u = targets.encode_signed(d, ok)
    assert list(u) == [128, 168, 88, 255, 1]                    # +-31.75 saturates at 255 / 1, never 0
    assert np.allclose(targets.decode_signed(u)[:3], [0.0, 10.0, -10.0])
    assert list(targets.encode_signed(d, ~ok)) == [0] * 5       # no data everywhere
    t = np.array([0.0, 10.0, 100.0], np.float32)
    assert list(targets.encode_unsigned(t, np.ones(3, bool))) == [1, 40, 255]
    assert list(targets.encode_unsigned(t, np.zeros(3, bool))) == [0, 0, 0]


def test_medial_of_a_slab_is_one_voxel_thick():
    band = np.zeros((16, 16, 16), bool)
    band[:, :, 6:11] = True
    m = targets.medial(band)
    assert m.sum(2).max() == 1 and set(np.unique(np.nonzero(m)[2])) == {8}


def test_thickness_bounds_scale_with_the_rung():
    assert targets.thickness_bounds(2) == (2.0, 24.0)
    assert targets.thickness_bounds(3) == (1.0, 12.0)
    assert targets.thickness_bounds(4) == (0.5, 6.0)


# ------------------------------------------------------------------------- the definition, analytic

@pytest.mark.parametrize("sep", [4, 10, 20])
def test_parallel_planes_thickness_is_exact_everywhere_valid_even_past_the_old_cap(sep):
    """T02. Raw distances: the thickness is the face separation at EVERY valid voxel, including voxels
    where a face distance exceeds the +-31.75 encoding cap (the old code clipped both distances first,
    so there it wrote a wrong thickness, or TMIN once both saturated); validity is exactly "both faces
    within reach"."""
    n, a, reach = 160, 100, 40.0
    rec, ver, dy, dx = _bands(n, [a], [a - sep])
    m, t, ok, sup = targets.block_fields(rec, ver, dy, dx, reach=reach)
    x = np.arange(n)
    want_ok = (np.abs(x - a) <= reach) & (np.abs(x - (a - sep)) <= reach)
    assert np.array_equal(ok[2, 2], want_ok)
    assert np.all(t[ok] == sep)
    assert np.allclose(m[2, 2][want_ok], (x - (a - sep / 2.0))[want_ok])
    beyond = want_ok & ((np.abs(x - a) > targets.CAP) | (np.abs(x - (a - sep)) > targets.CAP))
    assert beyond.any()                     # the old code clipped at least one face distance here
    tt = targets.decode_unsigned(targets.encode_unsigned(t, ok))[2, 2]
    assert np.all(tt[beyond] == sep)
    mm = targets.encode_signed(m, ok)[2, 2]
    far = np.abs(x - (a - sep / 2.0)) > targets.CAP
    assert (want_ok & far).any() == (sep < 12)                 # |m| stays <= 30 at sep 20, reach 40
    assert set(mm[want_ok & far].tolist()) <= {1, 255}          # the midline clamps at ENCODING only
    assert int(mm[~want_ok].max()) == 0
    assert sup["valid"] == int(ok.sum()) and sup["crossing"] == 0 and sup["thickness"] == 0


def test_two_sheets_never_pair_faces_of_different_sheets():
    """T03, the review's repro: recto 50 / 80, verso 40 / 70 (two sheets of thickness 10). The old code
    gave midline 5 / thickness 3 at x = 65 (recto 50 of one sheet against verso 70 of the other). Every
    valid voxel must carry one sheet's own pair; inter-sheet voxels whose nearest faces disagree are
    no-data."""
    n = 128
    rec, ver, dy, dx = _bands(n, [50, 80], [40, 70])
    m, t, ok, sup = targets.block_fields(rec, ver, dy, dx, reach=24.0)
    x = np.arange(n)
    row_ok, row_m, row_t = ok[2, 2], m[2, 2], t[2, 2]
    assert np.all(row_t[row_ok] == 10)
    for xi in np.nonzero(row_ok)[0]:
        assert row_m[xi] in (xi - 45.0, xi - 75.0), (xi, row_m[xi])
    assert row_ok[45] and row_m[45] == 0 and row_ok[75] and row_m[75] == 0
    assert not row_ok[65] or row_m[65] == -10.0                 # never midline 5 / thickness 3
    for xi in (58, 59, 60, 61, 62):                             # nearest recto 50, nearest verso 70
        assert not row_ok[xi], xi
    assert sup["thickness"] > 0                                 # rejected, not clamped


def test_a_segment_across_another_recto_face_is_rejected():
    """Pairing rule 3, second half: sheet 2 (recto 64) has lost its verso, so a voxel just outside it has
    recto 64 and sheet 1's verso 40 as nearest faces: t = 24 is within [tmin, tmax] and only the walk
    across sheet 1's recto (50) rejects it."""
    n = 128
    rec, ver, dy, dx = _bands(n, [50, 64], [40])
    m, t, ok, sup = targets.block_fields(rec, ver, dy, dx, reach=30.0)
    row = ok[2, 2]
    x = np.arange(n)
    assert np.array_equal(row[x <= 56], (x >= 20)[x <= 56])     # sheet 1's own support
    assert np.all(t[2, 2][x <= 56][(x >= 20)[x <= 56]] == 10)
    assert not row[58:].any()                                   # recto 64 + verso 40: never paired
    assert sup["crossing"] >= 13 * 16                           # x = 58..70 on every (z, y)


def test_empty_recto_block_is_no_data_everywhere():
    rec, ver, dy, dx = _bands(96, [], [40])
    m, t, ok, sup = targets.block_fields(rec, ver, dy, dx)
    assert not ok.any() and sup["no_recto"] == ok.size and sup["valid"] == 0


def test_missing_or_empty_verso_is_no_data_everywhere():
    """T04: no `midline = d_r` fallback."""
    rec, ver, dy, dx = _bands(96, [40], [])
    for v in (None, ver):
        m, t, ok, sup = targets.block_fields(rec, v, dy, dx)
        assert not ok.any() and sup["valid"] == 0 and sup["no_verso"] > 0


def test_a_voxel_beyond_reach_is_no_data():
    rec, ver, dy, dx = _bands(128, [80], [70])
    m, t, ok, _ = targets.block_fields(rec, ver, dy, dx, reach=8.0)
    x = np.arange(128)
    assert np.array_equal(ok[2, 2], (np.abs(x - 80) <= 8) & (np.abs(x - 70) <= 8))
    assert not ok[2, 2, 100] and ok[2, 2, 75]


def test_negative_or_too_thick_pairs_are_rejected_not_clamped():
    rec, ver, dy, dx = _bands(128, [60], [61 + 30])            # verso OUTSIDE recto: t < 0
    _, _, ok, sup = targets.block_fields(rec, ver, dy, dx, reach=40.0)
    assert not ok.any() and sup["thickness"] > 0
    rec, ver, dy, dx = _bands(128, [80], [50])                  # t = 30 > TMAX 24
    _, _, ok, _ = targets.block_fields(rec, ver, dy, dx, reach=40.0)
    assert not ok.any()


# ------------------------------------------------------------------------------- the stores, end to end

def test_signed_distance_sign_units_and_reach_on_a_slab(slab_region):
    """Radial is +x: the midline is `x - (recto_x + verso_x)/2`, POSITIVE on the outward (recto) side,
    and only where both faces are within reach."""
    r = slab_region(n=128, recto_x=80, verso_x=70)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), **KW)
    assert 2 in rep["rungs"] and rep["verso"]
    mid = _read(r.root, "midline", r.lo)
    th = _read(r.root, "thickness", r.lo)
    assert mid.shape == (128, 128, 128)
    x = np.arange(128)
    keep = (np.abs(x - 80) <= 12) & (np.abs(x - 70) <= 12)
    want = np.where(keep, np.rint((x - 75.0) / targets.UNIT) + targets.OFF, 0).astype(np.uint8)
    assert np.array_equal(mid[64, 64, :], want)
    assert np.array_equal(mid[10, 100, :], want)
    assert mid[64, 64, 75] == 128 and mid[64, 64, 80] > 128 and mid[64, 64, 70] < 128
    assert np.array_equal(th[64, 64, :], np.where(keep, 40, 0).astype(np.uint8))
    sup = rep["rungs"][2]["support"]
    ok = th > 0                     # (the region's outermost y / z layer: the band's medial surface is
    assert np.array_equal(ok[1:-1, 1:-1], np.broadcast_to(keep, (126, 126, 128)))   # cut by the air)
    assert sup["valid"] == int(ok.sum()) and sup["crossing"] == 0 and sup["thickness"] == 0
    a = stores.open_store(stores.store_path(r.root, "midline", r.lo))
    assert a.attrs["target_def"] == targets.TARGET_DEF and a.attrs["reach_vox"] == 12.0


def test_store_thickness_is_exact_past_the_old_cap(slab_region):
    """T02 through the stores, production halo: recto 100 / verso 96 with reach 40, so voxels at
    x = 64..69 (and 127) have both faces more than 31.75 away and still carry thickness 4."""
    r = slab_region(name="cap", n=128, recto_x=100, verso_x=96)
    targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=48, reach=40)
    th = _read(r.root, "thickness", r.lo)[64, 64, :]
    x = np.arange(128)
    keep = (np.abs(x - 100) <= 40) & (np.abs(x - 96) <= 40)
    assert np.array_equal(th, np.where(keep, 16, 0).astype(np.uint8))
    assert keep[64] and 100 - 64 > targets.CAP and 96 - 64 > targets.CAP


def test_empty_recto_store_is_code_zero_not_a_zero_distance(slab_region):
    """T01: the old code wrote code 128 (distance 0) over an all-air recto block."""
    r = slab_region(name="empty", n=128, recto_x=-100, verso_x=70)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2, 3), **KW)
    for rung in (2, 3):
        assert int(_read(r.root, "midline", r.lo, rung).max()) == 0
        assert int(_read(r.root, "thickness", r.lo, rung).max()) == 0
    assert rep["rungs"][2]["support"]["no_recto"] == 128 ** 3


def test_verso_missing_is_no_data_then_recomputed_when_verso_arrives(slab_region):
    """T04: without a verso store both fields are code 0 (no recto-only midline); the store records
    `verso: False`, so a later call after the verso pass recomputes instead of reusing it."""
    r = slab_region(name="noverso", n=128, recto_x=80, verso=False)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), **KW)
    assert rep["verso"] is False
    assert int(_read(r.root, "midline", r.lo).max()) == 0
    assert int(_read(r.root, "thickness", r.lo).max()) == 0
    x = np.arange(128)[None, None, :]
    v = np.broadcast_to(np.where(np.abs(x - 70) <= 1, np.uint8(255), np.uint8(0)), (128,) * 3)
    stores.write(stores.store_path(r.root, "verso", r.lo, 0), np.ascontiguousarray(v), r.lo, rung=2,
                 channels=("verso",), q=8)
    assert not targets.fields_current(r.root, r.lo, rungs=(2,), reach=12)   # the scheduler's view
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), **KW)
    assert "skipped" not in rep["rungs"][2]
    assert _read(r.root, "midline", r.lo)[64, 64, 75] == 128
    assert targets.fields_current(r.root, r.lo, rungs=(2,), reach=12)


def test_near_axis_voxels_get_weight_zero(slab_region):
    """Within `axis_r_um` microns of the umbilicus the core is crushed and "which side is recto" is a
    coin flip, so the whole field is written as code 0."""
    r = slab_region(name="onaxis", n=128, recto_x=80, verso_x=70, axis_yx=(64.0, 64.0))
    targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), axis_r_um=400.0, **KW)   # 166 voxels
    assert int(_read(r.root, "midline", r.lo).max()) == 0
    assert int(_read(r.root, "thickness", r.lo).max()) == 0

    r2 = slab_region(name="onaxis2", n=128, recto_x=80, verso_x=70, axis_yx=(64.0, 64.0))
    targets.region_fields(r2.root, r2.lo, r2.ax, rungs=(2,), axis_r_um=24.0, **KW)
    m = _read(r2.root, "midline", r2.lo)       # 10 voxels: only the core is dropped
    assert int(m[64, 64, 64]) == 0 and int(m[64, 64, 75]) == 128


def test_coarse_rungs_are_recomputed_and_never_pooled(slab_region):
    """A rung-3 field is the distance in RUNG-3 voxels, computed from the 2x pool of the bands -- half
    the rung-2 number, not a mean of rung-2 codes."""
    r = slab_region(name="rungs", n=128, recto_x=80, verso_x=64)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2, 3), **KW)
    assert set(rep["rungs"]) == {2, 3}
    a3 = stores.open_store(stores.store_path(r.root, "midline_r3", r.lo))
    assert int(a3.attrs["rung"]) == 3
    assert a3.attrs["voxel_um"] == ladder.rung_um(3)
    assert a3.attrs["shape_true"] == [64, 64, 64]
    assert a3.attrs["tmax_vox"] == 12.0
    assert a3.shape == (128, 128, 128)                        # padded up to the 128 store granularity
    m3 = np.asarray(a3[:], np.uint8)
    assert int(m3[64:, :, :].max()) == 0                      # the padding is no-data
    th3 = np.asarray(stores.open_store(stores.store_path(r.root, "thickness_r3", r.lo))[:], np.uint8)
    assert int(th3[32, 32, 36]) == int(round(8.0 / targets.UNIT))   # 16 rung-2 voxels = 8 rung-3 voxels


def test_done_is_resume_and_force_or_new_parameters_recompute(slab_region):
    r = slab_region(name="resume", n=128)
    targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), **KW)
    p = stores.store_path(r.root, "midline", r.lo)
    before = os.path.getmtime(p)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), **KW)
    assert rep["rungs"][2] == {"skipped": "done"}
    assert os.path.getmtime(p) == before
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), force=True, **KW)
    assert "skipped" not in rep["rungs"][2]
    assert targets.fields_current(r.root, r.lo, rungs=(2,), reach=12)
    assert not targets.fields_current(r.root, r.lo, rungs=(2,), reach=10)
    assert not targets.fields_current(r.root, r.lo, rungs=(2, 3), reach=12)   # rung 3 never built
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=16, reach=10)
    assert "skipped" not in rep["rungs"][2]


def test_reach_must_stay_below_the_halo(slab_region):
    r = slab_region(name="reach", n=128)
    with pytest.raises(AssertionError):
        targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=16, reach=16)


def _tree(root):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            out.append(os.path.relpath(os.path.join(d, f), root))
    return sorted(out)


def _same_stores(x, y, rungs=(2, 3)):
    for kind in ("midline", "thickness"):
        for rung in rungs:
            ch = targets.channel(kind, rung)
            pa, pb = stores.store_path(x.root, ch, x.lo), stores.store_path(y.root, ch, y.lo)
            fa, fb = _tree(pa), _tree(pb)
            assert fa == fb and fa, ch
            for f in fa:
                assert filecmp.cmp(os.path.join(pa, f), os.path.join(pb, f), shallow=False), (ch, f)


def test_jobs_are_byte_identical(slab_region):
    a = slab_region(name="j1", n=128, recto_x=80, verso_x=70)
    b = slab_region(name="j2", n=128, recto_x=80, verso_x=70)
    c = slab_region(name="j4", n=128, recto_x=80, verso_x=70)
    targets.region_fields(a.root, a.lo, a.ax, rungs=(2, 3), jobs=1, **KW)
    targets.region_fields(b.root, b.lo, b.ax, rungs=(2, 3), jobs=2, **KW)
    targets.region_fields(c.root, c.lo, c.ax, rungs=(2, 3), jobs=4, **KW)
    _same_stores(a, b)
    _same_stores(a, c)
    assert int(_read(a.root, "thickness", a.lo)[1:-1, 1:-1].max()) == 40   # parity over real support


def test_a_rung_above_four_is_refused(slab_region):
    r = slab_region(name="toohigh", n=128)
    with pytest.raises(AssertionError):
        targets.region_fields(r.root, r.lo, r.ax, rungs=(5,), **KW)


def test_read_pooled_matches_pooling_the_whole_store(slab_region):
    r = slab_region(name="pool", n=128, recto_x=80, verso_x=70)
    a = stores.open_store(stores.store_path(r.root, "recto", r.lo))
    whole = ladder.pool2(np.asarray(a[:], np.uint8))
    blk = targets.read_pooled(a, 3, (16, 16, 16), (16, 16, 16))
    assert np.array_equal(blk, whole[16:32, 16:32, 16:32])
    out = targets.read_pooled(a, 3, (-8, -8, -8), (8, 8, 8))     # wholly outside: air, not an error
    assert int(out.max()) == 0


def test_a_persistent_field_pool_is_byte_identical(slab_region):
    """The producer keeps ONE `field_pool` for its lifetime and feeds it region after region; the
    stores must be the ones `jobs=1` writes, for every region it is fed (the worker re-initialises when
    the region changes)."""
    a = slab_region(name="p1", n=128, recto_x=80, verso_x=70)
    b = slab_region(name="p2", n=128, recto_x=80, verso_x=70)
    c = slab_region(name="p3", n=128, recto_x=60, verso_x=50)
    d = slab_region(name="p4", n=128, recto_x=60, verso_x=50)
    targets.region_fields(a.root, a.lo, a.ax, rungs=(2, 3), jobs=1, **KW)
    targets.region_fields(c.root, c.lo, c.ax, rungs=(2, 3), jobs=1, **KW)
    pool = targets.field_pool(2)
    try:
        targets.region_fields(b.root, b.lo, b.ax, rungs=(2, 3), jobs=2, pool=pool, **KW)
        targets.region_fields(d.root, d.lo, d.ax, rungs=(2, 3), jobs=2, pool=pool, **KW)
    finally:
        pool.shutdown(wait=True)
    _same_stores(a, b)
    _same_stores(c, d)
