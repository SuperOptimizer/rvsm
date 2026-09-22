"""The distance stores: sign, units, clamp, the verso fallback, the axis exclusion, and `jobs` determinism."""
import filecmp
import os

import numpy as np
import pytest

from rvsm import ladder, stores, targets


def _read(root, kind, lo, rung=2, round_=0):
    p = stores.store_path(root, targets.channel(kind, rung), lo, round_)
    assert stores.is_done(p), p
    return np.asarray(stores.open_store(p)[:], np.uint8)


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


def test_signed_distance_sign_units_and_clamp_on_a_slab(slab_region):
    """The sheet is perpendicular to x and the axis lies far away in -x, so radial is +x everywhere:
    the midline must be `x - (recto_x + verso_x)/2` in voxels, POSITIVE on the outward (recto) side."""
    r = slab_region(n=128, recto_x=80, verso_x=70)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=16)
    assert 2 in rep["rungs"] and rep["verso"]
    mid = _read(r.root, "midline", r.lo)
    th = _read(r.root, "thickness", r.lo)
    assert mid.shape == (128, 128, 128)
    x = np.arange(128)
    want = np.clip(np.rint((x - 75.0) / targets.UNIT) + targets.OFF, 1, 255)
    got = mid[64, 64, :]                                        # a row through the interior of a block
    # exact only where the block's halo actually contains BOTH faces: this test runs with halo=16 for
    # speed, so the x < 64 block never sees the recto face at x = 80 (production halo is 48 > the
    # +-31.75 the encoding can represent, and every block sees everything it can encode).
    keep = (x >= 64) & (np.abs(x - 75.0) <= 14)
    assert np.array_equal(got[keep], want[keep].astype(np.uint8))
    assert got[75] == 128                                       # the midline itself: signed distance 0
    assert got[85] > 128 and got[65] < 128                      # outward positive, inward negative
    assert got[0] == 1 and got[127] == 255                      # clamped to +-31.75, never to 0
    assert th[64, 64, 75] == int(round(10.0 / targets.UNIT))     # thickness = the face separation


def test_midline_is_the_mean_of_the_two_faces(slab_region):
    r = slab_region(n=128, recto_x=90, verso_x=60)
    targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=16)
    mid = _read(r.root, "midline", r.lo)
    th = _read(r.root, "thickness", r.lo)
    assert mid[64, 64, 75] == 128                               # (90 + 60) / 2
    assert th[64, 64, 75] == int(round(30.0 / targets.UNIT))    # 90 - 60


def test_verso_missing_falls_back_to_the_recto_medial_and_thickness_is_no_data(slab_region):
    r = slab_region(name="noverso", n=128, recto_x=80, verso=False)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=16)
    assert rep["verso"] is False
    mid = _read(r.root, "midline", r.lo)
    th = _read(r.root, "thickness", r.lo)
    assert mid[64, 64, 80] == 128                               # the midline IS the recto face
    assert mid[64, 64, 90] == 128 + 40
    assert int(th.max()) == 0                                   # thickness: no data anywhere


def test_near_axis_voxels_get_weight_zero(slab_region):
    """Within `axis_r_um` microns of the umbilicus the core is crushed and "which side is recto" is a
    coin flip, so the whole field is written as code 0."""
    r = slab_region(name="onaxis", n=128, recto_x=80, verso_x=70, axis_yx=(64.0, 64.0))
    targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=16,
                          axis_r_um=400.0)     # 400 um / 2.4 um = 166 voxels: the whole region
    assert int(_read(r.root, "midline", r.lo).max()) == 0
    assert int(_read(r.root, "thickness", r.lo).max()) == 0

    r2 = slab_region(name="onaxis2", n=128, recto_x=80, verso_x=70, axis_yx=(64.0, 64.0))
    targets.region_fields(r2.root, r2.lo, r2.ax, rungs=(2,), block=64, halo=16, axis_r_um=24.0)
    m = _read(r2.root, "midline", r2.lo)       # 10 voxels: only the core is dropped
    assert int(m[64, 64, 64]) == 0 and int(m[64, 64, 120]) > 0


def test_coarse_rungs_are_recomputed_and_never_pooled(slab_region):
    """A rung-3 field is the distance in RUNG-3 voxels, computed from the 2x pool of the bands -- half
    the rung-2 number, not a mean of rung-2 codes."""
    r = slab_region(name="rungs", n=128, recto_x=80, verso_x=64)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2, 3), block=64, halo=16)
    assert set(rep["rungs"]) == {2, 3}
    a3 = stores.open_store(stores.store_path(r.root, "midline_r3", r.lo))
    assert int(a3.attrs["rung"]) == 3
    assert a3.attrs["voxel_um"] == ladder.rung_um(3)
    assert a3.attrs["shape_true"] == [64, 64, 64]
    assert a3.shape == (128, 128, 128)                        # padded up to the 128 store granularity
    m3 = np.asarray(a3[:], np.uint8)
    assert int(m3[64:, :, :].max()) == 0                      # the padding is no-data
    th3 = np.asarray(stores.open_store(stores.store_path(r.root, "thickness_r3", r.lo))[:], np.uint8)
    assert int(th3[32, 32, 36]) == int(round(8.0 / targets.UNIT))   # 16 rung-2 voxels = 8 rung-3 voxels


def test_done_is_resume_and_force_recomputes(slab_region):
    r = slab_region(name="resume", n=128)
    targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=16)
    p = stores.store_path(r.root, "midline", r.lo)
    before = os.path.getmtime(p)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=16)
    assert rep["rungs"][2] == {"skipped": "done"}
    assert os.path.getmtime(p) == before
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=16, force=True)
    assert "skipped" not in rep["rungs"][2]


def _tree(root):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            out.append(os.path.relpath(os.path.join(d, f), root))
    return sorted(out)


def test_jobs_four_is_byte_identical_to_jobs_one(slab_region):
    a = slab_region(name="j1", n=128, recto_x=80, verso_x=70)
    b = slab_region(name="j4", n=128, recto_x=80, verso_x=70)
    targets.region_fields(a.root, a.lo, a.ax, rungs=(2, 3), block=64, halo=16, jobs=1)
    targets.region_fields(b.root, b.lo, b.ax, rungs=(2, 3), block=64, halo=16, jobs=4)
    for kind in ("midline", "thickness"):
        for rung in (2, 3):
            ch = targets.channel(kind, rung)
            pa = stores.store_path(a.root, ch, a.lo)
            pb = stores.store_path(b.root, ch, b.lo)
            fa, fb = _tree(pa), _tree(pb)
            assert fa == fb and fa, ch
            for f in fa:
                assert filecmp.cmp(os.path.join(pa, f), os.path.join(pb, f), shallow=False), (ch, f)


def test_a_rung_above_four_is_refused(slab_region):
    r = slab_region(name="toohigh", n=128)
    with pytest.raises(AssertionError):
        targets.region_fields(r.root, r.lo, r.ax, rungs=(5,), block=64, halo=16)


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
    targets.region_fields(a.root, a.lo, a.ax, rungs=(2, 3), block=64, halo=16, jobs=1)
    targets.region_fields(c.root, c.lo, c.ax, rungs=(2, 3), block=64, halo=16, jobs=1)
    pool = targets.field_pool(2)
    try:
        targets.region_fields(b.root, b.lo, b.ax, rungs=(2, 3), block=64, halo=16, jobs=2, pool=pool)
        targets.region_fields(d.root, d.lo, d.ax, rungs=(2, 3), block=64, halo=16, jobs=2, pool=pool)
    finally:
        pool.shutdown(wait=True)
    for x, y in ((a, b), (c, d)):
        for kind in ("midline", "thickness"):
            for rung in (2, 3):
                ch = targets.channel(kind, rung)
                pa, pb = stores.store_path(x.root, ch, x.lo), stores.store_path(y.root, ch, y.lo)
                fa, fb = _tree(pa), _tree(pb)
                assert fa == fb and fa, ch
                for f in fa:
                    assert filecmp.cmp(os.path.join(pa, f), os.path.join(pb, f), shallow=False), (ch, f)
