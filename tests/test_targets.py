"""The distance stores: the paired-v3 target definition (raw distances, reach and coverage, the decoder's
thickness floor, same-sheet pairing, stencil/gradient validity, no recto-only fallback), the
encoding, the axis exclusion, generation identity and `jobs` determinism.

The geometry is analytic: planes perpendicular to x with the umbilicus far away in -x, so the radial
direction is +x everywhere, a recto face at a and a verso face at a - t give d_r = x - a,
d_v = x - (a - t), midline x - (a - t/2) and thickness t exactly (see `rvsm.targets`, THE SIGN); plus a
curved (annulus) sheet around an axis inside the box."""
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


def _stencil(ok1d):
    """A 1-D pair-validity row -> the voxels whose full +-1 stencil is valid (array ends excluded)."""
    out = np.zeros_like(ok1d)
    out[1:-1] = ok1d[:-2] & ok1d[1:-1] & ok1d[2:]
    return out


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


def test_rung_parameters_are_the_same_microns_and_the_decoder_floor():
    """P3-06: reach and tmax are 57.6 um at every rung; tmin is the decoder's 3-voxel floor everywhere."""
    assert targets.rung_params(2) == (24.0, 3.0, 24.0)
    assert targets.rung_params(3) == (12.0, 3.0, 12.0)
    assert targets.rung_params(4) == (6.0, 3.0, 6.0)
    assert targets.thickness_bounds(4) == (3.0, 6.0)


# ------------------------------------------------------------------------- the definition, analytic

@pytest.mark.parametrize("sep", [4, 10, 20])
def test_parallel_planes_thickness_is_exact_everywhere_valid_even_past_the_old_cap(sep):
    """T02. Raw distances: the thickness is the face separation at EVERY valid voxel, including voxels
    where a face distance exceeds the +-31.75 encoding cap (the old code clipped both distances first,
    so there it wrote a wrong thickness, or TMIN once both saturated); validity is exactly "both faces
    within reach, over the full stencil"."""
    n, a, reach = 160, 100, 40.0
    rec, ver, dy, dx = _bands(n, [a], [a - sep])
    m, t, ok, sup = targets.block_fields(rec, ver, dy, dx, reach=reach)
    x = np.arange(n)
    want_ok = _stencil((np.abs(x - a) <= reach) & (np.abs(x - (a - sep)) <= reach))
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
    assert sup["valid"] == int(ok.sum())
    assert sup["crossing"] == sup["thickness"] == sup["normal"] == sup["reciprocal"] == 0


def test_two_sheets_never_pair_faces_of_different_sheets():
    """T03, the review's repro: recto 50 / 80, verso 40 / 70 (two sheets of thickness 10). The old code
    gave midline 5 / thickness 3 at x = 65 (recto 50 of one sheet against verso 70 of the other). Every
    valid voxel must carry one sheet's own pair; inter-sheet voxels whose nearest faces disagree are
    no-data."""
    rec, ver, dy, dx = _bands(128, [50, 80], [40, 70])
    m, t, ok, sup = targets.block_fields(rec, ver, dy, dx, reach=24.0)
    row_ok, row_m, row_t = ok[2, 2], m[2, 2], t[2, 2]
    assert np.all(row_t[row_ok] == 10)
    for xi in np.nonzero(row_ok)[0]:
        assert row_m[xi] in (xi - 45.0, xi - 75.0), (xi, row_m[xi])
    assert row_ok[45] and row_m[45] == 0 and row_ok[75] and row_m[75] == 0
    assert not row_ok[65] or row_m[65] == -10.0                 # never midline 5 / thickness 3
    for xi in (58, 59, 60, 61, 62):                             # nearest recto 50, nearest verso 70
        assert not row_ok[xi], xi
    assert sup["thickness"] > 0                                 # rejected, not clamped


def test_a_recto_without_its_verso_never_borrows_the_next_sheets_verso():
    """Sheet 2 (recto 64) has lost its verso, so a voxel just outside it has recto 64 and sheet 1's
    verso 40 as nearest faces: t = 24 is within [tmin, tmax]; the reciprocal test (verso 40's nearest
    recto is sheet 1's, another band component) rejects it."""
    rec, ver, dy, dx = _bands(128, [50, 64], [40])
    m, t, ok, sup = targets.block_fields(rec, ver, dy, dx, reach=30.0)
    row = ok[2, 2]
    assert row[45] and t[2, 2, 45] == 10
    assert np.all(t[2, 2][row] == 10)
    assert not row[58:].any()
    assert sup["reciprocal"] > 0


def test_a_segment_through_another_face_of_the_same_component_is_a_crossing():
    """Two recto planes (x = 20 and 26) joined by a bridge at the y = 0 edge are ONE band component, so
    the reciprocal test passes; the ordered walk from recto 26 to verso 12 re-enters a recto band at 20
    and rejects the pair."""
    n = 48
    rec, ver, dy, dx = _bands(n, [20, 26], [12], zy=16)
    rec[:, 0:2, 20:27] = 255
    m, t, ok, sup = targets.block_fields(rec, ver, dy, dx, reach=24.0)
    assert not ok[8, 8, 27:].any()                              # recto 26 never pairs with verso 12
    assert ok[8, 8, 16] and t[8, 8, 16] == 8.0                   # sheet 1's own pair survives
    assert sup["crossing"] > 0


def test_touching_wraps_are_code_zero():
    """P3-07's first counterexample: recto planes x = 20 and 22 with only verso x = 18. The outer recto
    must never be paired through the intervening face (the old CROSS threshold missed the 1-voxel gap);
    and 20 / 18 is a 2-voxel pair, below the decoder's floor."""
    rec, ver, dy, dx = _bands(48, [20, 22], [18], half=0, zy=5)
    _, _, ok, sup = targets.block_fields(rec, ver, dy, dx)
    assert not ok.any()


def test_orthogonal_faces_are_not_a_pair():
    """P3-07's second counterexample: recto x = 20, verso y = 10, radial +x. The old code accepted it
    (target midline gradient norm 0.707); the verso face's normal is not radial."""
    shape = (5, 32, 48)
    rec = np.zeros(shape, np.uint8)
    ver = rec.copy()
    rec[:, :, 20] = 255
    ver[:, 10, :] = 255
    dy, dx = np.zeros((5, 32, 1), np.float32), np.ones(shape, np.float32)
    _, _, ok, sup = targets.block_fields(rec, ver, dy, dx)
    assert not ok.any() and sup["normal"] > 0


def test_a_curved_sheet_keeps_most_of_its_support():
    """An annulus: recto radius 50, verso radius 40, the axis in the middle of the box. The pairing
    checks must not reject a plain curved sheet; the targets are within the medial surfaces' own
    discretisation of the analytic ones."""
    n = 160
    z, y, x = np.meshgrid(np.arange(4), np.arange(n), np.arange(n), indexing="ij")
    dy, dx = (y - 80.0).astype(np.float32), (x - 80.0).astype(np.float32)
    r = np.sqrt(dy * dy + dx * dx)
    rec = np.where(np.abs(r - 50) <= 1, 255, 0).astype(np.uint8)
    ver = np.where(np.abs(r - 40) <= 1, 255, 0).astype(np.uint8)
    m, t, ok, sup = targets.block_fields(rec, ver, dy, dx)
    inner = (np.abs(r - 50) <= 24) & (np.abs(r - 40) <= 24)
    assert ok.sum() >= 0.85 * inner[1:3].sum()
    assert np.abs(t[ok] - 10).max() <= 2.1
    assert np.abs(m[ok] - (r[ok] - 45)).max() <= 1.0
    assert sup["crossing"] == 0 and sup["reciprocal"] == 0


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
    assert np.array_equal(ok[2, 2], _stencil((np.abs(x - 80) <= 8) & (np.abs(x - 70) <= 8)))
    assert not ok[2, 2, 100] and ok[2, 2, 75]


def test_thin_negative_or_too_thick_pairs_are_rejected_not_clamped():
    """P3-06 / T03: below the decoder's 3-voxel floor, negative, or above tmax: code 0, never clamped."""
    for rectos, versos in (([20], [18]), ([60], [91]), ([80], [50])):   # t = 2, t < 0, t = 30
        rec, ver, dy, dx = _bands(128, rectos, versos, half=0)
        _, _, ok, sup = targets.block_fields(rec, ver, dy, dx, reach=40.0)
        assert not ok.any() and sup["thickness"] > 0, (rectos, versos)


def test_coverage_rejects_a_ball_that_leaves_the_observed_stores():
    """Rule 2: outside the region's stores nothing is observed, so a voxel whose nearest-face ball
    reaches past them could have a nearer, unseen face."""
    rec, ver, dy, dx = _bands(64, [40], [30])
    x = np.arange(64)
    cover = np.broadcast_to((64 - x).astype(np.float32), rec.shape)     # the stores end at x = 64
    _, _, ok, sup = targets.block_fields(rec, ver, dy, dx, reach=20.0, cover=cover)
    pair = (np.abs(x - 40) <= 20) & (np.abs(x - 30) <= 20) & \
        ((64 - x) > np.maximum(np.abs(x - 40), np.abs(x - 30)) + targets.COVER_MARGIN)
    assert np.array_equal(ok[2, 2], _stencil(pair))
    assert sup["coverage"] > 0


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
    keep = _stencil((np.abs(x - 80) <= 12) & (np.abs(x - 70) <= 12))
    want = np.where(keep, np.rint((x - 75.0) / targets.UNIT) + targets.OFF, 0).astype(np.uint8)
    assert np.array_equal(mid[64, 64, :], want)
    assert np.array_equal(mid[40, 100, :], want)
    assert mid[64, 64, 75] == 128 and mid[64, 64, 80] > 128 and mid[64, 64, 70] < 128
    assert np.array_equal(th[64, 64, :], np.where(keep, 40, 0).astype(np.uint8))
    ok = th > 0
    assert np.array_equal(ok[30:-30, 30:-30], np.broadcast_to(keep, (68, 68, 128)))
    assert not ok[0].any() and not ok[:, -1].any()             # coverage: the region's edge
    sup = rep["rungs"][2]["support"]
    assert sup["valid"] == int(ok.sum()) and sup["coverage"] > 0
    a = stores.open_store(stores.store_path(r.root, "midline", r.lo))
    assert a.attrs["target_def"] == targets.TARGET_DEF and a.attrs["reach_vox"] == 12.0
    assert a.attrs["recto_digest"] and a.attrs["verso_digest"]
    blocks = a.attrs["support_blocks"]
    assert len(blocks["rows"]) == 8 and blocks["columns"][3:] == list(targets.SUPPORT)
    assert sum(row[3 + targets.SUPPORT.index("valid")] for row in blocks["rows"]) == sup["valid"]


def test_store_thickness_is_exact_past_the_old_cap(slab_region):
    """T02 through the stores, production halo: recto 100 / verso 96 with reach 40, so voxels near
    x = 64 have both faces more than 31.75 away and still carry thickness 4."""
    r = slab_region(name="cap", n=128, recto_x=100, verso_x=96)
    targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), block=64, halo=48, reach=40)
    th = _read(r.root, "thickness", r.lo)[64, 64, :]
    x = np.arange(128)
    u = np.maximum(np.abs(x - 100), np.abs(x - 96))
    keep = _stencil((u <= 40) & (np.minimum(x + 1, 128 - x) > u + targets.COVER_MARGIN))
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


def _write_band(root, name, lo, centres, shape=(128, 128, 128), half=1, round_=0):
    x = np.arange(shape[2])[None, None, :] + lo[2]
    b = np.zeros((1, 1, shape[2]), bool)
    for c in centres:
        b |= np.abs(x - c) <= half
    v = np.broadcast_to(np.where(b, np.uint8(255), np.uint8(0)), shape)
    stores.write(stores.store_path(root, name, lo, round_), np.ascontiguousarray(v), lo, rung=2,
                 channels=(name,), q=8)


def test_verso_missing_is_no_data_then_regenerated_when_verso_arrives_or_changes(slab_region):
    """T04 and P3-08: without a verso store both fields are code 0 (no recto-only midline). The fields
    record their sources' digests, so the scheduler's predicate (`fields_current`) turns False when the
    verso appears or is rewritten, and `region_fields` regenerates them."""
    r = slab_region(name="noverso", n=128, recto_x=80, verso=False)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), **KW)
    assert rep["verso"] is False
    assert int(_read(r.root, "midline", r.lo).max()) == 0
    assert int(_read(r.root, "thickness", r.lo).max()) == 0
    assert targets.fields_current(r.root, r.lo, rungs=(2,), reach=12)
    _write_band(r.root, "verso", r.lo, [70])
    assert not targets.fields_current(r.root, r.lo, rungs=(2,), reach=12)   # the scheduler's view
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), **KW)
    assert "skipped" not in rep["rungs"][2]
    assert _read(r.root, "midline", r.lo)[64, 64, 75] == 128
    assert targets.fields_current(r.root, r.lo, rungs=(2,), reach=12)
    _write_band(r.root, "verso", r.lo, [72], half=2)                        # a new verso generation
    assert not targets.fields_current(r.root, r.lo, rungs=(2,), reach=12)
    targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), **KW)
    assert _read(r.root, "midline", r.lo)[64, 64, 76] == 128


def test_a_scheduler_regenerates_a_stale_definition(slab_region, monkeypatch):
    """P3-08: `run._next_job` asks the same predicate the writer skips on, so a field store written
    under an older definition is scheduled again instead of being treated as done."""
    from types import SimpleNamespace
    from rvsm import run
    r = slab_region(name="stale", n=128, recto_x=80, verso_x=70)
    targets.region_fields(r.root, r.lo, r.ax, rungs=(2,))
    cat = SimpleNamespace(done=lambda *a: True)
    assert run._next_job(cat, r.lo, 0, True, r.root, rungs=(2,)) is None
    monkeypatch.setattr(targets, "TARGET_DEF", "paired-v4")
    assert run._next_job(cat, r.lo, 0, True, r.root, rungs=(2,)) == "fields"


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
    assert (a3.attrs["reach_vox"], a3.attrs["tmin_vox"], a3.attrs["tmax_vox"]) == (6.0, 3.0, 12.0)
    assert a3.shape == (128, 128, 128)                        # padded up to the 128 store granularity
    m3 = np.asarray(a3[:], np.uint8)
    assert int(m3[64:, :, :].max()) == 0                      # the padding is no-data
    th3 = np.asarray(stores.open_store(stores.store_path(r.root, "thickness_r3", r.lo))[:], np.uint8)
    assert int(th3[32, 32, 36]) == int(round(8.0 / targets.UNIT))   # 16 rung-2 voxels = 8 rung-3 voxels


def test_coarse_rungs_lose_thin_sheets_by_design(slab_region):
    """P3-06: a 10-voxel rung-2 sheet is 2.5 voxels at rung 4, below the decoder's 3-voxel floor, so it
    has no field target there -- it is not clamped up to 3."""
    r = slab_region(name="thin4", n=128, recto_x=80, verso_x=70)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2, 4), **KW)
    assert int(_read(r.root, "thickness", r.lo, 4).max()) == 0
    assert rep["rungs"][4]["support"]["valid"] == 0
    assert int(_read(r.root, "thickness", r.lo, 2).max()) == 40


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


def test_a_region_seam_never_invents_a_pair_the_whole_volume_rejects(tmp_path, volcomp_lib):
    """P3-08: the same continuous volume, as one 256-wide region and as two 128-wide ones. Sheet A
    (recto 116, verso 110) is in the left region, a competing wrap (verso 131, recto 137) in the right
    one. From x = 121 the whole volume sees verso 131 nearer than 110 (t < 0: no pair), but the left
    region alone cannot see it; coverage must reject the voxel rather than pair 116 with 110. Wherever
    the tiling has a value, it is the whole volume's value."""
    rectos, versos = [116, 137], [110, 131]
    zs = np.arange(0, 129, 16, dtype=np.float64)
    ax = np.stack([zs, np.full_like(zs, 64.0), np.full_like(zs, -1000.0)])
    whole, tiled = str(tmp_path / "whole"), str(tmp_path / "tiled")
    for name, cs in (("recto", rectos), ("verso", versos)):
        _write_band(whole, name, (0, 0, 0), cs, shape=(128, 128, 256))
        for lo in ((0, 0, 0), (0, 0, 128)):
            _write_band(tiled, name, lo, cs)
    targets.region_fields(whole, (0, 0, 0), ax, rungs=(2,), **KW)
    for lo in ((0, 0, 0), (0, 0, 128)):
        targets.region_fields(tiled, lo, ax, rungs=(2,), **KW)
    for kind in ("midline", "thickness"):
        w = _read(whole, kind, (0, 0, 0))
        t = np.concatenate([_read(tiled, kind, (0, 0, 0)), _read(tiled, kind, (0, 0, 128))], axis=2)
        has = t > 0
        assert has.any()
        assert np.array_equal(t[has], w[has]), kind
    assert _read(whole, "thickness", (0, 0, 0))[64, 64, 121] == 0
    assert _read(tiled, "thickness", (0, 0, 0))[64, 64, 121] == 0
    assert _read(tiled, "thickness", (0, 0, 0))[64, 64, 113] == 24       # sheet A's own pair, t = 6


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
    assert int(_read(a.root, "thickness", a.lo).max()) == 40      # the parity is over real support


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


# ------------------------------------------------------------------------------- the torch block path

TORCH_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="no CUDA device"))]


def _fixtures():
    """(name, recto, verso, dy, dx, kwargs) of the synthetic blocks the numpy tests above use, plus a
    thick-banded curved stack with two wraps and an axis that moves with z."""
    out = []
    for sep in (4, 10, 20):
        out.append((f"planes{sep}", *_bands(160, [100], [100 - sep]), dict(reach=40.0)))
    out.append(("two_sheets", *_bands(128, [50, 80], [40, 70]), dict(reach=24.0)))
    out.append(("lost_verso", *_bands(128, [50, 64], [40]), dict(reach=30.0)))
    rec, ver, dy, dx = _bands(48, [20, 26], [12], zy=16)
    rec[:, 0:2, 20:27] = 255
    out.append(("bridge", rec, ver, dy, dx, dict(reach=24.0)))
    out.append(("touching", *_bands(48, [20, 22], [18], half=0, zy=5), {}))
    shape = (5, 32, 48)
    rec = np.zeros(shape, np.uint8)
    ver = rec.copy()
    rec[:, :, 20] = 255
    ver[:, 10, :] = 255
    out.append(("orthogonal", rec, ver, np.zeros((5, 32, 1), np.float32), np.ones(shape, np.float32), {}))
    n = 160
    z, y, x = np.meshgrid(np.arange(4), np.arange(n), np.arange(n), indexing="ij")
    dy, dx = (y - 80.0).astype(np.float32), (x - 80.0).astype(np.float32)
    r = np.sqrt(dy * dy + dx * dx)
    out.append(("annulus", np.where(np.abs(r - 50) <= 1, 255, 0).astype(np.uint8),
                np.where(np.abs(r - 40) <= 1, 255, 0).astype(np.uint8), dy, dx, {}))
    n = 96
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    dy, dx = (y - (-60.0 + 0.2 * z)).astype(np.float32), (x - (30.0 + 0.1 * z)).astype(np.float32)
    ph = np.mod(np.sqrt(dy * dy + dx * dx) + 2 * np.sin(z / 9.0), 21.0)
    rec = np.clip(255 * (1 - np.abs(ph - 12.0) / 2.0), 0, 255).astype(np.uint8)
    ver = np.clip(255 * (1 - np.abs(ph - 3.0) / 2.0), 0, 255).astype(np.uint8)
    core = np.zeros(rec.shape, bool)
    core[16:-16, 16:-16, 16:-16] = True
    cov = targets.cover_distance((-16, -16, -16), rec.shape, (256, 256, 256))
    out.append(("wraps", rec, ver, dy, dx, dict(reach=12.0, core=core, cover=cov)))
    rec, ver, dy, dx = _bands(64, [40], [30])
    cover = np.broadcast_to((64 - np.arange(64)).astype(np.float32), rec.shape)
    out.append(("coverage", rec, ver, dy, dx, dict(reach=20.0, cover=cover)))
    out.append(("no_recto", *_bands(96, [], [40]), {}))
    rec, ver, dy, dx = _bands(96, [40], [])
    out.append(("no_verso", rec, ver, dy, dx, {}))
    out.append(("verso_none", rec, None, dy, dx, {}))
    return out


@pytest.mark.parametrize("dev", TORCH_DEVICES)
def test_block_fields_torch_matches_numpy(dev):
    """The torch port against the numpy reference on every synthetic block: the valid masks may differ
    on at most 0.1% of the core (nearest-voxel ties, `rvsm.edt`), midline and thickness agree to 1e-3
    where both are valid, and so do the support counts."""
    for name, rec, ver, dy, dx, kw in _fixtures():
        a = targets.block_fields(rec, ver, dy, dx, **kw)
        b = targets.block_fields_torch(rec, ver, dy, dx, device=dev, **kw)
        n = a[3]["voxels"]
        assert b[3]["voxels"] == n and b[0].dtype == b[1].dtype == np.float32 and b[2].dtype == bool
        assert (a[2] != b[2]).sum() <= 1e-3 * n, name
        both = a[2] & b[2]
        assert np.abs(a[0] - b[0])[both].max(initial=0) <= 1e-3, name
        assert np.abs(a[1] - b[1])[both].max(initial=0) <= 1e-3, name
        assert not b[0][~b[2]].any() and not b[1][~b[2]].any(), name
        for key in targets.SUPPORT:
            assert abs(a[3][key] - b[3][key]) <= 1e-3 * n, (name, key)
        if name == "wraps":
            assert a[3]["valid"] > 0.1 * n                 # the parity is over real support


def _curved_region(root, n=128, lo=(0, 0, 0)):
    """A region whose recto / verso stores are a stack of curved wraps around an axis outside it."""
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    r = np.sqrt((y + 90.0) ** 2 + (x - 50.0) ** 2) + 2 * np.sin(z / 11.0)
    ph = np.mod(r, 20.0)
    for name, c in (("recto", 11.0), ("verso", 3.0)):
        v = np.clip(255 * (1 - np.abs(ph - c) / 2.0), 0, 255).astype(np.uint8)
        stores.write(stores.store_path(root, name, lo), v, lo, rung=2, channels=(name,), q=8)
    zs = np.arange(0, n + 1, 16, dtype=np.float64)
    return np.stack([zs, np.full_like(zs, -90.0), np.full_like(zs, 50.0)])


def _decoded(root, lo, rung):
    m = _read(root, "midline", lo, rung)
    t = _read(root, "thickness", lo, rung)
    return targets.decode_signed(m), targets.decode_unsigned(t), m > 0, t > 0


@pytest.mark.parametrize("dev", TORCH_DEVICES)
def test_region_fields_on_a_device_matches_the_pool_path(tmp_path, slab_region, dev):
    """`region_fields(device=...)` (whole-store read, device pooling, torch blocks) against the numpy
    pool path, on a curved region and on the analytic slab, at rungs 2, 3 and 4."""
    lo = (0, 0, 0)
    a, b = str(tmp_path / "np"), str(tmp_path / "dev")
    ax = _curved_region(a)
    _curved_region(b)
    ra = targets.region_fields(a, lo, ax, rungs=(2, 3, 4), jobs=1, **KW)
    rb = targets.region_fields(b, lo, ax, rungs=(2, 3, 4), device=dev, **KW)
    for rung in (2, 3, 4):
        ma, ta, oka, _ = _decoded(a, lo, rung)
        mb, tb, okb, _ = _decoded(b, lo, rung)
        assert np.array_equal(oka, _read(a, "thickness", lo, rung) > 0)
        assert (oka != okb).sum() <= 1e-3 * oka.size, rung
        both = oka & okb
        assert np.abs(ma - mb)[both].max(initial=0) <= 1e-3 and np.abs(ta - tb)[both].max(initial=0) <= 1e-3
        sa, sb = ra["rungs"][rung]["support"], rb["rungs"][rung]["support"]
        for key in targets.SUPPORT:
            assert abs(sa[key] - sb[key]) <= 1e-3 * sa["voxels"], (rung, key)
        pa = stores.open_store(stores.store_path(a, targets.channel("midline", rung), lo)).attrs
        pb = stores.open_store(stores.store_path(b, targets.channel("midline", rung), lo)).attrs
        assert len(pa["support_blocks"]["rows"]) == len(pb["support_blocks"]["rows"])
        assert [r[:3] for r in pa["support_blocks"]["rows"]] == [r[:3] for r in pb["support_blocks"]["rows"]]
    assert ra["rungs"][2]["support"]["valid"] > 0.1 * 128 ** 3
    assert targets.fields_current(b, lo, rungs=(2, 3, 4), reach=12)
    # the analytic slab, where the expected codes are known exactly
    r = slab_region(name="slab", n=128, recto_x=80, verso_x=70)
    targets.region_fields(r.root, r.lo, r.ax, rungs=(2,), device=dev, **KW)
    x = np.arange(128)
    keep = _stencil((np.abs(x - 80) <= 12) & (np.abs(x - 70) <= 12))
    want = np.where(keep, np.rint((x - 75.0) / targets.UNIT) + targets.OFF, 0).astype(np.uint8)
    assert np.array_equal(_read(r.root, "midline", r.lo)[64, 64, :], want)


@pytest.mark.parametrize("dev", TORCH_DEVICES)
def test_region_fields_on_a_device_is_deterministic(slab_region, dev):
    a = slab_region(name="d1", n=128, recto_x=80, verso_x=70)
    b = slab_region(name="d2", n=128, recto_x=80, verso_x=70)
    targets.region_fields(a.root, a.lo, a.ax, rungs=(2, 3), device=dev, **KW)
    targets.region_fields(b.root, b.lo, b.ax, rungs=(2, 3), device=dev, **KW)
    _same_stores(a, b)


@pytest.mark.parametrize("dev", TORCH_DEVICES)
def test_the_device_batch_size_does_not_change_the_bytes(slab_region, tmp_path, monkeypatch, dev):
    """Blocks go through the device `FIELD_BATCH` at a time; each is computed on its own (per-volume
    operators, per-block offsets and walk sample counts), so any batch size writes the same stores --
    including a batch mixing a block with faces and blocks without."""
    lo = (0, 0, 0)
    roots = []
    for B in (1, 3, 8):
        root = str(tmp_path / f"b{B}")
        ax = _curved_region(root)
        monkeypatch.setattr(targets, "FIELD_BATCH", B)
        targets.region_fields(root, lo, ax, rungs=(2, 3), device=dev, **KW)
        roots.append(root)
    import types
    for r in roots[1:]:
        _same_stores(types.SimpleNamespace(root=roots[0], lo=lo), types.SimpleNamespace(root=r, lo=lo))
    a = slab_region(name="e1", n=128, recto_x=-100, verso_x=70)       # no recto face anywhere
    b = slab_region(name="e3", n=128, recto_x=-100, verso_x=70)
    monkeypatch.setattr(targets, "FIELD_BATCH", 1)
    targets.region_fields(a.root, a.lo, a.ax, rungs=(2,), device=dev, **KW)
    monkeypatch.setattr(targets, "FIELD_BATCH", 3)
    targets.region_fields(b.root, b.lo, b.ax, rungs=(2,), device=dev, **KW)
    _same_stores(a, b, rungs=(2,))


def test_device_fields_wait_for_the_gpu_lock(slab_region):
    """The producer's GPU fields never run beside a network pass: `region_fields(device=...)` does its
    device work under the `gpu_lock` the pass holds."""
    import threading
    import time
    from rvsm import run as RUN
    r = slab_region(name="lk", n=128, recto_x=80, verso_x=70)
    lk = RUN.TracedLock("gpu")
    lk.acquire()                                    # a pass in flight
    done = threading.Event()
    th = threading.Thread(target=lambda: (targets.region_fields(r.root, r.lo, r.ax, rungs=(2,),
                                                                device="cpu", gpu_lock=lk, **KW),
                                          done.set()))
    th.start()
    time.sleep(1.5)
    assert not done.is_set() and not targets.fields_current(r.root, r.lo, rungs=(2,), reach=12)
    lk.release()
    th.join(120)
    assert done.is_set() and targets.fields_current(r.root, r.lo, rungs=(2,), reach=12)
    assert lk.holder() is None


@pytest.mark.parametrize("dev", TORCH_DEVICES)
def test_fields_that_yield_between_batches_write_the_same_bytes(slab_region, monkeypatch, dev):
    """The producer's fields give the card back between batches to a blocking pass
    (`run.GpuGate.fields_hold`): `_fields_torch` calls the lock's `yield_point(flush)` before every batch
    but the first, and a yield in every gap -- the lock released, someone else on the card, the lock
    taken back -- leaves the stores byte-identical."""
    import threading
    from rvsm import run as RUN
    monkeypatch.setattr(targets, "FIELD_BATCH", 1)
    a = slab_region(name="y1", n=128, recto_x=80, verso_x=70)
    b = slab_region(name="y2", n=128, recto_x=80, verso_x=70)
    targets.region_fields(a.root, a.lo, a.ax, rungs=(2, 3), device=dev, **KW)

    lk = RUN.TracedLock("gpu")
    others = []

    class Always:
        """A hold that yields at every batch boundary, and checks it holds the lock whenever it works."""
        def __enter__(self):
            lk.acquire()
            return self

        def __exit__(self, *e):
            lk.release()

        def yield_point(self, flush):
            assert lk.holder()["thread"] == threading.current_thread().name
            flush()
            lk.release()
            th = threading.Thread(target=lambda: (lk.acquire(), others.append(1), lk.release()))
            th.start()
            th.join(10)
            lk.acquire()
            return True
    targets.region_fields(b.root, b.lo, b.ax, rungs=(2, 3), device=dev, gpu_lock=Always(), **KW)
    n_batches = 8 + 1                                  # 128^3 at block 64: 8 rung-2 blocks, 1 rung-3
    assert len(others) == n_batches - 1 and lk.holder() is None
    _same_stores(a, b)


@pytest.mark.parametrize("dev", TORCH_DEVICES)
@pytest.mark.parametrize("B", [1, 3, 8])
def test_faceless_blocks_are_skipped_with_the_same_bytes(slab_region, tmp_path, monkeypatch, dev, B):
    """A device block whose window has no recto voxel (all core "no_recto") or no verso voxel (the
    recto face distance only) is answered without the rest of its transforms: the stores -- codes,
    support counts, per-block support rows -- are byte-identical to computing every block, on a curved
    region, an all-air region, and slabs whose faces fall into only some blocks' windows (mixed
    batches of full / recto-only / faceless blocks)."""
    import types
    monkeypatch.setattr(targets, "FIELD_BATCH", B)
    cases = [("air", dict(recto_x=-100, verso_x=-100)),     # no face anywhere
             ("rec_only", dict(recto_x=80, verso_x=-100)),   # recto faces, never a verso
             ("split", dict(recto_x=100, verso_x=20)),       # x-low windows: verso only; x-high: recto only
             ("pair", dict(recto_x=80, verso_x=70))]         # a real pair in half the blocks
    seen = {}
    for name, kw in cases:
        roots = {}
        for skip in (False, True):
            monkeypatch.setattr(targets, "SKIP_FACELESS", skip)
            r = slab_region(name=f"{name}{int(skip)}", n=128, **kw)
            rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2, 3), device=dev, **KW)
            roots[skip] = r
            seen[(name, skip)] = rep["skipped_blocks"]
        _same_stores(roots[False], roots[True])
    assert seen[("air", True)]["no_recto"] == seen[("air", True)]["blocks"] == 9
    assert seen[("rec_only", True)]["no_verso"] > 0 and seen[("split", True)]["no_recto"] > 0
    assert seen[("split", True)]["no_verso"] > 0
    assert all(seen[(nm, False)]["skipped"] == 0 for nm, _ in cases)
    for skip in (False, True):                          # and the curved region, every block with faces
        monkeypatch.setattr(targets, "SKIP_FACELESS", skip)
        root = str(tmp_path / f"curved{int(skip)}")
        ax = _curved_region(root)
        targets.region_fields(root, (0, 0, 0), ax, rungs=(2, 3), device=dev, **KW)
    _same_stores(types.SimpleNamespace(root=str(tmp_path / "curved0"), lo=(0, 0, 0)),
                 types.SimpleNamespace(root=str(tmp_path / "curved1"), lo=(0, 0, 0)))


@pytest.mark.parametrize("dev", TORCH_DEVICES)
def test_capped_transforms_write_the_same_bytes(slab_region, tmp_path, monkeypatch, dev):
    """The device fields with their transforms capped (`edt_cap`, `MEDIAL_CAP`) against uncapped ones:
    byte-identical stores on the curved region, the paired / split / recto-only slabs, and a region
    whose recto band is thicker than twice the medial cap (the medial transform saturates and falls
    back to the uncapped one), at rungs 2 and 3."""
    import types
    from rvsm import edt as E
    calls = {"n": 0}
    real = E.edt2

    def counted(*a, **k):
        if k.get("cap") is None and not k.get("indices", True):
            calls["n"] += 1                     # an uncapped medial transform: the fallback
        return real(*a, **k)
    monkeypatch.setattr(E, "edt2", counted)
    lo = (0, 0, 0)

    def thick(root):
        n = 128
        z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        rec = np.where(np.abs(x - 60) <= 20, 255, 0).astype(np.uint8)     # 41 voxels thick: depth 21
        ver = np.where(np.abs(x - 95) <= 1, 255, 0).astype(np.uint8)
        for name, v in (("recto", rec), ("verso", ver)):
            stores.write(stores.store_path(root, name, lo), v, lo, rung=2, channels=(name,), q=8)
        zs = np.arange(0, n + 1, 16, dtype=np.float64)
        return np.stack([zs, np.full_like(zs, 64.0), np.full_like(zs, -400.0)])
    for name, make in (("curved", _curved_region), ("thick", thick)):
        roots = {}
        for cap in (False, True):
            monkeypatch.setattr(targets, "EDT_CAP", cap)
            root = str(tmp_path / f"{name}{int(cap)}")
            ax = make(root)
            n0 = calls["n"]
            targets.region_fields(root, lo, ax, rungs=(2, 3), device=dev, **KW)
            roots[cap] = types.SimpleNamespace(root=root, lo=lo)
            if name == "thick" and cap:
                assert calls["n"] > n0, "the thick band should have fallen back to the uncapped medial"
        _same_stores(roots[False], roots[True])
    for name, kw in (("pair", dict(recto_x=80, verso_x=70)), ("split", dict(recto_x=100, verso_x=20)),
                     ("reconly", dict(recto_x=80, verso_x=-100))):
        rs = {}
        for cap in (False, True):
            monkeypatch.setattr(targets, "EDT_CAP", cap)
            r = slab_region(name=f"c{name}{int(cap)}", n=128, **kw)
            targets.region_fields(r.root, r.lo, r.ax, rungs=(2, 3), device=dev, **KW)
            rs[cap] = r
        _same_stores(rs[False], rs[True])


def _fixture_bits(res):
    m, t, ok, sup = res
    return m.view(np.uint32).tobytes(), t.view(np.uint32).tobytes(), ok.tobytes(), sup


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA device")
def test_fused_pair_checks_are_the_torch_pair_checks(slab_region, tmp_path, monkeypatch):
    """The pair checks as fused CUDA kernels (`_pair_checks_triton`: the reciprocal test, the
    box-restricted Gaussian and the normals, the walk) against the op-by-op torch version: the same
    float32 bits of every fixture's fields and the same support, and byte-identical stores on the
    curved region and the paired / split slabs at rungs 2 and 3."""
    import types
    called = {"n": 0}
    real = targets._pair_checks_triton

    def counted(*a, **k):
        r = real(*a, **k)
        called["n"] += bool(r)
        return r
    monkeypatch.setattr(targets, "_pair_checks_triton", counted)
    for name, rec, ver, dy, dx, kw in _fixtures():
        out = {}
        for fused in (False, True):
            monkeypatch.setattr(targets, "PAIR_FUSED", fused)
            out[fused] = _fixture_bits(targets.block_fields_torch(rec, ver, dy, dx, device="cuda", **kw))
        assert out[False] == out[True], name
        assert out[True] == _fixture_bits(targets.block_fields(rec, ver, dy, dx, **kw)), name
    assert called["n"] > 5
    lo = (0, 0, 0)
    roots = {}
    for fused in (False, True):
        monkeypatch.setattr(targets, "PAIR_FUSED", fused)
        root = str(tmp_path / f"curved{int(fused)}")
        ax = _curved_region(root)
        targets.region_fields(root, lo, ax, rungs=(2, 3), device="cuda", **KW)
        roots[fused] = types.SimpleNamespace(root=root, lo=lo)
    _same_stores(roots[False], roots[True])
    for name, kw in (("pair", dict(recto_x=80, verso_x=70)), ("split", dict(recto_x=100, verso_x=20))):
        rs = {}
        for fused in (False, True):
            monkeypatch.setattr(targets, "PAIR_FUSED", fused)
            rs[fused] = slab_region(name=f"f{name}{int(fused)}", n=128, **kw)
            targets.region_fields(rs[fused].root, rs[fused].lo, rs[fused].ax, rungs=(2, 3), device="cuda", **KW)
        _same_stores(rs[False], rs[True])


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("B", [1, 3])
def test_fields_graphs_write_the_same_bytes(slab_region, tmp_path, monkeypatch, B):
    """The fixed-shape stages of full device batches replayed as CUDA graphs (`_FieldGraphs`) against
    the eager path: byte-identical stores on the curved region (graphs captured and replayed at rungs
    2 and 3), a region whose recto band saturates the medial cap (replayed batches recomputed eagerly,
    the uncapped fallback), and the paired / split slabs (full, faceless and partial batches mixed)."""
    import types
    monkeypatch.setattr(targets, "FIELD_BATCH", B)
    lo = (0, 0, 0)

    def thick(root):
        n = 128
        z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        rec = np.where(np.abs(x - 60) <= 20, 255, 0).astype(np.uint8)     # 41 voxels thick: depth 21
        ver = np.where(np.abs(x - 85) <= 1, 255, 0).astype(np.uint8)
        for name, v in (("recto", rec), ("verso", ver)):
            stores.write(stores.store_path(root, name, lo), v, lo, rung=2, channels=(name,), q=8)
        zs = np.arange(0, n + 1, 16, dtype=np.float64)
        return np.stack([zs, np.full_like(zs, 64.0), np.full_like(zs, -400.0)])
    reps = {}
    for name, make in (("curved", _curved_region), ("thick", thick)):
        roots = {}
        for g in (False, True):
            monkeypatch.setattr(targets, "GRAPHS", g)
            root = str(tmp_path / f"{name}{int(g)}")
            ax = make(root)
            reps[(name, g)] = targets.region_fields(root, lo, ax, rungs=(2, 3), device="cuda", **KW)
            roots[g] = types.SimpleNamespace(root=root, lo=lo)
        _same_stores(roots[False], roots[True])
    assert reps[("curved", False)]["graphs"] is None
    assert reps[("curved", True)]["graphs"]["replayed"] > 0
    if B == 1:                       # at 3 the thick region has no full batch past the first
        assert reps[("thick", True)]["graphs"]["saturated"] > 0
    for name, kw in (("pair", dict(recto_x=80, verso_x=70)), ("split", dict(recto_x=100, verso_x=20))):
        rs = {}
        for g in (False, True):
            monkeypatch.setattr(targets, "GRAPHS", g)
            rs[g] = slab_region(name=f"g{name}{int(g)}", n=128, **kw)
            targets.region_fields(rs[g].root, rs[g].lo, rs[g].ax, rungs=(2, 3), device="cuda", **KW)
        _same_stores(rs[False], rs[True])


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA device")
def test_fields_graphs_stop_capturing_past_the_cap_with_the_same_bytes(slab_region, tmp_path, monkeypatch):
    """A producer that yields the card often drops the graphs every time; past `MAX_CAPTURES` a
    region stops capturing again (eager batches, `refused`), with the same bytes."""
    import types

    class Always:
        n = 0

        def __enter__(self):
            return self

        def __exit__(self, *e):
            pass

        def yield_point(self, flush):
            Always.n += 1
            if Always.n % 3:
                return False
            flush()
            return True
    monkeypatch.setattr(targets, "FIELD_BATCH", 1)
    monkeypatch.setattr(targets, "MAX_CAPTURES", 1)
    lo = (0, 0, 0)
    roots, reps = {}, {}
    for g in (False, True):
        monkeypatch.setattr(targets, "GRAPHS", g)
        root = str(tmp_path / f"cap{int(g)}")
        ax = _curved_region(root)
        reps[g] = targets.region_fields(root, lo, ax, rungs=(2, 3), device="cuda", gpu_lock=Always(), **KW)
        roots[g] = types.SimpleNamespace(root=root, lo=lo)
    _same_stores(roots[False], roots[True])
    st = reps[True]["graphs"]
    assert st["captured"] <= 1 and st["refused"] > 0 and st["rss_gb"] is not None


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA device")
def test_fused_stage_c_is_the_torch_stage_c(monkeypatch):
    """Rule 5 and the counts as one kernel (`_stage5`) against the op-by-op torch steps, on random
    codes, bands and distances (pair voxels at the volume borders, a core that touches them, midline
    differences around the gradient bounds): the same float32 bits of the midline and thickness, the
    same valid mask and counts, and the same encoded core window."""
    import torch
    g = torch.Generator().manual_seed(3)
    dev = torch.device("cuda")
    for B, shape in ((1, (9, 13, 11)), (3, (20, 17, 24)), (2, (5, 40, 7))):
        full = (B,) + shape
        reason = torch.where(torch.rand(full, generator=g) < 0.85, 0,
                             torch.randint(1, 8, full, generator=g)).to(torch.uint8)
        ev = torch.rand(full, generator=g) < 0.95
        core = torch.rand(full, generator=g) < 0.8
        z = torch.arange(shape[0], dtype=torch.float32).view(1, -1, 1, 1)
        dr = (-3.0 + 0.9 * z + torch.randn(full, generator=g) * 0.3).to(torch.float32)
        dv = (dr + 6.0 + torch.randn(full, generator=g) * 2).to(torch.float32)
        dv[0, 0, 0, :3] = torch.tensor([40.0, -40.0, 1e-3])
        t = (dv - dr).contiguous()
        h = (2, 1, 3)
        sl = (slice(None),) + tuple(slice(a, n - a) for a, n in zip(h, shape))
        outs = {}
        for fused in (False, True):
            monkeypatch.setattr(targets, "PAIR_FUSED", fused)
            S = lambda: dict(reason=reason.clone().to(dev), ev=ev.to(dev), t=t.to(dev),  # noqa: E731
                             dr=dr.to(dev), dv=dv.to(dev))
            m, tt, ok, cnt = targets._stage_c(S(), core.to(dev), dev)
            e, c2 = targets._stage_c(S(), core.to(dev), dev, enc=(sl, 20.0))
            outs[fused] = (m.view(torch.int32).cpu(), tt.view(torch.int32).cpu(), ok.cpu(), cnt.cpu(), e.cpu(),
                           c2.cpu())
        for a, b in zip(outs[False], outs[True]):
            assert torch.equal(a, b), (B, shape)
        assert outs[True][2].any() and (outs[True][3][:, 7:9] > 0).all()     # stencil and gradient fails seen


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA device")
def test_fields_graphs_release_their_pool_on_another_key(slab_region, monkeypatch):
    """A batch of another key (a new rung or window shape) releases the previous key's graphs and pool
    at once, whether or not it can be graphed itself."""
    monkeypatch.setattr(targets, "FIELD_BATCH", 1)
    r = slab_region(name="gk", n=128, recto_x=80, verso_x=70)
    seen = []
    real = targets._FieldGraphs.drop_other

    def spy(self, key):
        had = self.g is not None
        real(self, key)
        if had and self.g is None:
            seen.append(key[0])
    monkeypatch.setattr(targets._FieldGraphs, "drop_other", spy)
    rep = targets.region_fields(r.root, r.lo, r.ax, rungs=(2, 3), device="cuda", **KW)
    assert rep["graphs"]["replayed"] > 0
    assert seen == [3]                                 # rung 2's graphs dropped by rung 3's first batch


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA device")
def test_mixed_and_short_batches_replay_the_graphs_with_the_same_bytes(slab_region, tmp_path, monkeypatch):
    """At batch > 1 a batch with some faceless windows, and a short batch at a rung's end, go through
    the graphs (the faceless windows computed in full, the short batch padded with air windows):
    byte-identical stores to the eager path with the faceless windows skipped, on slabs whose faces
    fall into only some windows and on the curved region."""
    import types
    monkeypatch.setattr(targets, "FIELD_BATCH", 3)
    lo = (0, 0, 0)
    reps = {}
    for name, kw in (("pair", dict(recto_x=80, verso_x=70)), ("edge", dict(recto_x=110, verso_x=104)),
                     ("curved", None)):
        roots = {}
        for g in (False, True):
            monkeypatch.setattr(targets, "GRAPHS", g)
            if kw is None:
                root = str(tmp_path / f"mc{int(g)}")
                ax = _curved_region(root)
                reps[(name, g)] = targets.region_fields(root, lo, ax, rungs=(2, 3), device="cuda", **KW)
                roots[g] = types.SimpleNamespace(root=root, lo=lo)
            else:
                roots[g] = slab_region(name=f"m{name}{int(g)}", n=128, **kw)
                reps[(name, g)] = targets.region_fields(roots[g].root, roots[g].lo, roots[g].ax, rungs=(2, 3),
                                                        device="cuda", **KW)
        _same_stores(roots[False], roots[True])
    st = {k: v["skipped_blocks"] for k, v in reps.items()}
    assert sum(st[(n, True)]["graphed_faceless"] for n in ("pair", "edge", "curved")) > 0
    assert sum(st[(n, True)]["graphed_padded"] for n in ("pair", "edge", "curved")) > 0
    assert all(st[(n, False)]["graphed_faceless"] == st[(n, False)]["graphed_padded"] == 0
               for n in ("pair", "edge", "curved"))
