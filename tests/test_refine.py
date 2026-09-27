"""The surface refiner (rvsm/tools/refine/refine.py): the usrm2 tests ported, plus the verso term, the anchor
field, the frame round trip and an end-to-end slab run over real volcomp stores."""
import json
import os

import numpy as np
import pytest

from rvsm.tools.refine import refine as R

AX_Y = np.array([[0.0, 100.0], [-1000.0, -1000.0], [16.0, 16.0]])   # axis far below: outward = +y


def band_volume(shape, y_center, width=2.0):
    """A soft band at y = y_center(x) inside a (Z,Y,X) volume."""
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]].astype(np.float32)
    return np.exp(-0.5 * ((y - y_center(x)) / width) ** 2).astype(np.float32)


def flat_sheet(y, z=(2, 14), x=(2, 30)):
    Z, X = np.meshgrid(np.arange(*z, dtype=np.float32), np.arange(*x, dtype=np.float32), indexing="ij")
    return np.stack([Z, np.full_like(Z, y), X], -1)


# ------------------------------------------------------------------------------------ ported from usrm2

def test_refine_moves_a_shifted_surface_onto_the_band():
    shape = (24, 64, 64)
    yc = lambda x: 30 + 4 * np.sin(x / 10.0)  # noqa: E731
    V = band_volume(shape, yc)
    xs, zs = np.arange(4, 60, dtype=np.float32), np.arange(2, 22, dtype=np.float32)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    g = np.stack([Z, np.zeros_like(Z), X], -1)
    g[..., 1] = yc(g[..., 2]) + 3.0
    g[5:8, 10:14, 1] += 4.0
    ax = np.array([[0.0, 100.0], [-1000.0, -1000.0], [32.0, 32.0]])
    g1, stats = R.refine(g, V, (0, 0, 0), ax, far=8, sigma=1.5, iters=3, thr=0.3)
    err0 = np.abs(g[..., 1] - yc(g[..., 2])).mean()
    err1 = np.abs(g1[..., 1] - yc(g1[..., 2])).mean()
    assert err0 > 2.5 and err1 < 0.8, (err0, err1)
    assert np.isfinite(g1).all() and stats[-1]["with_peak"] > 0.9


def test_refine_reads_a_uint8_store_like_a_float_one():
    shape = (16, 48, 32)
    V = band_volume(shape, lambda x: 20 + 0 * x)
    Vu = np.clip(np.rint(V * 255), 0, 255).astype(np.uint8)
    g = flat_sheet(23.0)
    a, _ = R.refine(g, V, (0, 0, 0), AX_Y, far=6, sigma=1.0, iters=2, thr=0.3)
    b, _ = R.refine(g, Vu, (0, 0, 0), AX_Y, far=6, sigma=1.0, iters=2, thr=0.3)
    assert np.abs(a - b).max() < 0.05 and abs(b[..., 1].mean() - 20) < 0.5


def test_refine_leaves_holes_and_outside_points_alone():
    shape = (16, 32, 32)
    V = band_volume(shape, lambda x: 16 + 0 * x)
    g = np.zeros((6, 6, 3), np.float32)
    g[..., 0], g[..., 2] = np.arange(6)[:, None] * 2 + 2, np.arange(6)[None] * 4 + 4
    g[..., 1] = 19.0
    g[2, 2] = np.nan
    g[0, 0] = (8, 19, 200)  # outside the store
    g1, _ = R.refine(g, V, (0, 0, 0), AX_Y, far=6, sigma=1.0, iters=2, thr=0.3)
    assert np.isnan(g1[2, 2]).all() and np.allclose(g1[0, 0], g[0, 0])
    inside = np.isfinite(g1).all(-1) & (g1[..., 2] < 100)
    assert np.abs(g1[inside][:, 1] - 16).mean() < 1.0


def test_joint_refinement_keeps_two_close_sheets_apart():
    """Two sheets 6 voxels apart, both published 4 voxels too high: alone, the upper one would jump onto the
    lower band; jointly, each stays on its own band."""
    shape = (16, 64, 32)
    V = np.maximum(band_volume(shape, lambda x: 30 + 0 * x, 1.2), band_volume(shape, lambda x: 36 + 0 * x, 1.2))
    lower, upper = flat_sheet(34.0), flat_sheet(40.0)  # true 30 and 36, both +4
    (l1, u1), st = R.refine_many([lower, upper], V, (0, 0, 0), AX_Y, far=8, sigma=1.0, iters=4, thr=0.3)
    assert abs(l1[..., 1].mean() - 30) < 1.0 and abs(u1[..., 1].mean() - 36) < 1.0, (l1[..., 1].mean(), u1[..., 1].mean())
    assert st[0]["capped"] > 0.5  # the sheets saw each other on the ray


def test_joint_assignment_keeps_two_sheets_ordered_when_both_are_nearest_one_band():
    """Both published sheets are nearest the UPPER band (36): alone each would land on it and they would
    merge; jointly the lower one takes the lower band and the order lower < upper holds at every point."""
    shape = (16, 64, 32)
    V = np.maximum(band_volume(shape, lambda x: 30 + 0 * x, 1.2), band_volume(shape, lambda x: 36 + 0 * x, 1.2))
    lower, upper = flat_sheet(34.5), flat_sheet(37.5)
    alone, _ = R.refine(lower, V, (0, 0, 0), AX_Y, far=8, sigma=1.0, iters=3, thr=0.3)
    assert abs(alone[..., 1].mean() - 36) < 1.0      # the failure the joint assignment exists for
    (l1, u1), _ = R.refine_many([lower, upper], V, (0, 0, 0), AX_Y, far=8, sigma=1.0, iters=3, thr=0.3)
    assert (l1[..., 1] < u1[..., 1] - 3).all()
    assert abs(l1[..., 1].mean() - 30) < 1.0 and abs(u1[..., 1].mean() - 36) < 1.0


def test_assign_prefers_one_peak_per_sheet_in_order():
    pos = np.array([[-4.0], [2.0], [0.0], [0.0], [0.0], [0.0]], np.float32)   # peaks at -4 and +2
    stren = np.array([[1.0], [1.0], [-1], [-1], [-1], [-1]], np.float32)
    assert abs(R.assign(pos, stren, np.array([np.nan]), np.array([np.nan]))[0] - 2.0) < 1e-4
    assert abs(R.assign(pos, stren, np.array([np.nan]), np.array([6.0]))[0] + 4.0) < 1e-4


def test_upsample_densifies_and_keeps_holes():
    Z, X = np.meshgrid(np.arange(0, 100, 20, dtype=np.float32), np.arange(0, 100, 20, dtype=np.float32), indexing="ij")
    g = np.stack([Z, 30 + 0.1 * X, X], -1)
    g[2, 2] = np.nan
    u = R.upsample(g, 5)
    assert u.shape == (21, 21, 3)
    ok = np.isfinite(u).all(-1)
    v = u[ok][:, 0] / 4
    assert np.abs(v - np.round(v)).max() < 1e-3 and np.abs(np.diff(u[0, :, 2])).mean() == pytest.approx(4.0, abs=1e-3)
    assert not ok[10, 10] and ok[0, 0] and ok[20, 20]
    assert np.allclose(u[::5, ::5][np.isfinite(g).all(-1)], g[np.isfinite(g).all(-1)], atol=1e-3)   # nodes kept
    assert R.auto_up(g, 4.0) == 5


def test_ray_neighbours_finds_the_sheets_on_the_ray():
    q = np.array([[5.0, 10.0, 5.0]], np.float32)
    n = np.array([[0.0, 1.0, 0.0]], np.float32)
    others = np.array([[5, 4, 5], [5, 16.5, 5.5], [5, 13, 9]], np.float32)  # below 6, above 6.5, a lateral miss
    b, a = R.ray_neighbours(q, n, others, R=8)
    assert b[0] == pytest.approx(-6.0) and a[0] == pytest.approx(6.5)


# ----------------------------------------------------------------------------------------- verso term

def test_verso_first_candidate_is_rejected():
    """Peaks at -11 (sheet A's recto) and +9 (sheet B's recto); B's verso is at +1, between the vertex and +9:
    reaching B's recto would cross B's back face, so only A is left."""
    pos = np.array([[-11.0], [9.0]], np.float32)
    stren = np.array([[0.9], [0.9]], np.float32)
    vpos = np.array([[1.0], [-19.0]], np.float32)
    vstren = np.array([[0.9], [0.9]], np.float32)
    st, blocked = R.verso_adjust(pos, stren, vpos, vstren, T=8.0)
    assert blocked[:, 0].tolist() == [False, True] and st[1, 0] < 0 and st[0, 0] > 0.9   # A paired (+ bonus)
    assert R.assign(pos, st, np.array([np.nan]), np.array([np.nan]))[0] == pytest.approx(-11.0)
    st2, _ = R.verso_adjust(pos, stren, vpos, vstren, T=8.0, block=0.5)        # the penalised variant
    assert 0 < st2[1, 0] < st2[0, 0]
    # a verso AT the vertex (the surface sits on the back face) does not block its own recto one T outward
    st3, b3 = R.verso_adjust(np.array([[8.0]], np.float32), np.array([[0.9]], np.float32),
                             np.array([[0.2]], np.float32), np.array([[0.9]], np.float32), T=8.0)
    assert not b3.any() and st3[0, 0] > 0.9


def test_thickness_from_peaks():
    pos = np.array([[30.0, 31.0, 29.5]], np.float32)
    stren = np.ones_like(pos)
    vpos = np.array([[22.0, 23.0, 21.0], [50.0, 50.0, 50.0]], np.float32)
    assert R.thickness_from_peaks(pos, stren, vpos, np.ones_like(vpos)) == pytest.approx(8.0)


def two_sheet_scene(T=8.0):
    """Sheets A (verso 22 / recto 30) and B (verso 42 / recto 50) along +y (outward)."""
    shape = (16, 72, 32)
    c = lambda y: (lambda x: y + 0 * x)  # noqa: E731
    V = np.maximum(band_volume(shape, c(30), 1.2), band_volume(shape, c(50), 1.2))
    W = np.maximum(band_volume(shape, c(30 - T), 1.2), band_volume(shape, c(50 - T), 1.2))
    return V, W


def test_verso_term_stops_a_jump_across_the_gap():
    """A surface of sheet A pushed to y=41, one voxel short of B's verso: on the recto alone it snaps outward
    onto B (+9, the nearer peak); with the verso the path to B crosses B's back face, so it returns to A."""
    V, W = two_sheet_scene()
    g = flat_sheet(41.0)
    alone, _ = R.refine(g, V, (0, 0, 0), AX_Y, far=12, sigma=1.0, iters=3, thr=0.3)
    assert abs(alone[..., 1].mean() - 50) < 1.0
    with_v, st = R.refine(g, V, (0, 0, 0), AX_Y, far=12, sigma=1.0, iters=3, thr=0.3, W=W)
    assert abs(with_v[..., 1].mean() - 30) < 1.0, with_v[..., 1].mean()
    assert st[0]["verso_blocked"] > 0.9 and st[0]["thickness"] == pytest.approx(8.0, abs=0.5)
    # a thickness store gives the same answer (codes of 0.25 voxel)
    thick = np.full(V.shape, 32, np.uint8)
    tv, _ = R.refine(g, V, (0, 0, 0), AX_Y, far=12, sigma=1.0, iters=3, thr=0.3, W=W, thick=thick)
    assert abs(tv[..., 1].mean() - 30) < 1.0


# -------------------------------------------------------------------------------------------- anchors

def test_anchor_log_replay_and_field(tmp_path):
    p = tmp_path / "anchors.jsonl"
    recs = [{"op": "add", "id": "a1", "surface": "segA", "grid_rc": [5, 5], "from_zyx": [5, 20, 5],
             "to_zyx": [5, 26, 5], "plane_normal_zyx": [1, 0, 0], "ts": "t"},
            {"op": "add", "id": "a2", "surface": "segA", "grid_rc": None, "from_zyx": [0, 20, 30],
             "to_zyx": [0, 10, 30], "plane_normal_zyx": [1, 0, 0], "ts": "t"},
            {"op": "del", "id": "a2"},
            {"op": "add", "id": "b1", "surface": "segB.tifxyz", "grid_rc": [0, 0], "from_zyx": [0, 0, 0],
             "to_zyx": [0, 1, 0], "plane_normal_zyx": [1, 0, 0], "ts": "t"}]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n{\"op\":\"add\",\"id\"")   # a torn last line
    live = R.read_anchors(str(p))
    assert sorted(a["id"] for a in live) == ["a1", "b1"]
    assert [a["id"] for a in live if R.anchor_matches(a, ("segB",))] == ["b1"]
    Z, X = np.meshgrid(np.arange(0, 40, 2, dtype=np.float32), np.arange(0, 80, 2, dtype=np.float32), indexing="ij")
    g = np.stack([Z, np.full_like(Z, 20.0), X], -1)             # pitch 2 voxels
    mine = [a for a in live if R.anchor_matches(a, ("segA",))]
    f, top = R.anchor_field(g, mine, sigma=6.0, rc_map=lambda rc: (rc[0], rc[1]))
    assert np.allclose(f[5, 5], (0, 6, 0), atol=1e-4) and top[5, 5] == pytest.approx(1.0)
    assert np.abs(f[15:, 25:]).max() < 1e-3 and top[15:, 25:].max() < 1e-3      # tapered to zero far away
    assert 0 < f[5, 7, 1] < 6                                                       # and in between
    # no grid_rc: the grid point nearest from_zyx
    f2, _ = R.anchor_field(g, [dict(mine[0], grid_rc=None, from_zyx=[10, 20, 14], to_zyx=[10, 26, 14])], sigma=6.0)
    assert np.allclose(f2[5, 7], (0, 6, 0), atol=1e-4)


def test_anchor_holds_its_cell_against_the_snap():
    """An anchor drags a patch onto y=26 where the band is at 20: the snap would pull it back, the hold does
    not let it at the anchor cell, and the far side of the sheet still snaps."""
    V = band_volume((16, 48, 64), lambda x: 20 + 0 * x, 1.5)
    g = flat_sheet(20.0, x=(2, 62))
    a = {"grid_rc": [6, 10], "from_zyx": g[6, 10].tolist(), "to_zyx": (g[6, 10] + (0, 6, 0)).tolist()}
    f, top = R.anchor_field(g, [a], sigma=2.0, rc_map=lambda rc: rc)
    g0 = g + f
    g1, _ = R.refine(g0, V, (0, 0, 0), AX_Y, far=8, sigma=1.0, iters=3, thr=0.3, holds=[top])
    assert g1[6, 10, 1] == pytest.approx(26.0, abs=0.05)
    assert abs(g1[6, 50, 1] - 20) < 0.5


# --------------------------------------------------------------------------------------------- frames

FINE_TO_LEGACY = np.array([[-0.2384, -0.1936, 0.0021, 11024.0],
                           [-0.1937, 0.2385, -0.0040, 2861.8],
                           [-0.0008, 0.0045, 0.3066, -9057.1]])


def test_frame_round_trip(tmp_path):
    p = tmp_path / "transform.json"
    mov = np.array([[18844.0, 22610.0, 59000.0], [15507.0, 20140.0, 46293.0], [22312.0, 18412.0, 64845.0],
                    [18687.0, 9133.0, 33095.0]])
    fix = mov @ FINE_TO_LEGACY[:, :3].T + FINE_TO_LEGACY[:, 3]
    p.write_text(json.dumps({"fixed_volume": "PHercParis4-20230205180739_masked",
                             "transformation_matrix": FINE_TO_LEGACY.tolist(),
                             "fixed_landmarks": fix.tolist(), "moving_landmarks": mov.tolist()}))
    fr = R.transform_json_frame(str(p))
    leg_zyx = fix[:, ::-1].astype(np.float32)
    fine = fr.to_fine(leg_zyx)
    assert np.abs(fine - mov[:, ::-1]).max() < 0.05                     # legacy -> fine lands on the landmarks
    assert np.abs(fr.to_src(fine) - leg_zyx).max() < 1e-2
    # a legacy_to_fine function without an inverse is inverted numerically
    A = np.linalg.inv(np.vstack([FINE_TO_LEGACY, [0, 0, 0, 1]]))[:3]
    ff = R.function_frame(lambda xyz: xyz @ A[:, :3].T + A[:, 3])
    assert np.abs(ff.to_fine(leg_zyx) - fine).max() < 0.05
    assert np.abs(ff.to_src(ff.to_fine(leg_zyx)) - leg_zyx).max() < 1e-2
    so = R.scale_offset_frame(7.91 / 2.4, (10, 20, 30))
    assert np.allclose(so.to_src(so.to_fine(leg_zyx)), leg_zyx, atol=1e-2)
    # NaN holes pass through
    assert np.isnan(fr.to_fine(np.array([[np.nan, 1, 2]]))).all()


def test_write_back_touches_only_moved_nodes():
    fr = R.scale_offset_frame(2.0, (100, 100, 100))
    src = np.random.default_rng(0).uniform(10, 50, (10, 12, 3)).astype(np.float32)
    src[0, 0] = np.nan
    fine = fr.to_fine(src)
    crop, rc = fine[2:7, 3:9], (2, 3)
    dense = R.upsample(crop, 3)
    new = dense.copy()
    new[3, 3] += (0, 4.0, 0)                                   # node (1, 1) of the crop = (3, 4) published
    out = R.write_back(src, rc, 3, crop, new, fr)
    moved = np.zeros(src.shape[:2], bool)
    moved[3, 4] = True
    assert np.array_equal(out[~moved], src[~moved], equal_nan=True)
    assert np.allclose(out[3, 4], src[3, 4] + (0, 2.0, 0), atol=1e-3)   # 4 fine voxels = 2 source voxels


def test_frame_kind_and_find_surfaces(tmp_path):
    for seg, frames in (("s1", ("s1-on-20260411134726-2.4um", "s1-on-20230205180739-7.91um", "s1-on-x-45.532um")),
                        ("s2", ("s2-on-20230205180739-7.91um",))):
        for f in frames:
            d = tmp_path / seg / (f + ".tifxyz")
            d.mkdir(parents=True)
            (d / "x.tif").write_bytes(b"")
    found = R.find_surfaces(str(tmp_path))
    assert [(n, s) for n, s, _ in found] == [("s1-on-20260411134726-2.4um", "s1"), ("s2-on-20230205180739-7.91um", "s2")]
    assert [R.frame_kind(d) for _, _, d in found] == ["fine", "legacy"]
    assert R.find_surfaces(str(tmp_path), ("7.91um",))[0][0].endswith("7.91um")
    assert R.frame_kind(str(tmp_path / "s1" / "s1-on-x-45.532um.tifxyz")) == "coarse"


def test_plane_segments_trace_a_flat_sheet():
    g = flat_sheet(20.0, z=(0, 10), x=(0, 30))
    s = R.plane_segments(g, 4.5)
    assert len(s) == 29 and np.allclose(s[..., 0], 20.0) and np.allclose(np.sort(s[..., 1].ravel())[[0, -1]], (0, 29))


# ------------------------------------------------------------------------------------ end to end (stores)

def write_tifxyz_raw(d, g, scale=0.05):
    import tifffile
    os.makedirs(d, exist_ok=True)
    for i, c in enumerate("zyx"):
        tifffile.imwrite(os.path.join(d, f"{c}.tif"), np.where(np.isfinite(g[..., i]), g[..., i], 0).astype(np.float32))
    v = g[np.isfinite(g).all(-1)]
    json.dump({"bbox": [v.min(0)[::-1].tolist(), v.max(0)[::-1].tolist()], "format": "tifxyz",
               "scale": [scale, scale], "type": "seg", "uuid": "x"}, open(os.path.join(d, "meta.json"), "w"))


def test_end_to_end_slab_in_the_legacy_frame(tmp_path, has_volcomp):
    """Two stores (recto, verso) over a 128^3 fine box; one surface published in a 'legacy' frame (fine =
    legacy * 2 + offset) 3 voxels off its band; refine a 64-deep slab. The refined tifxyz is written back in
    the legacy frame: inside the slab it sits on the band, outside it is bit-identical, and the PNGs and the
    report exist."""
    if not has_volcomp:
        pytest.skip("volcomp not available")
    from rvsm import stores
    o = np.array([128, 256, 384])
    T = 8.0
    z, y, x = np.mgrid[:128, :128, :128].astype(np.float32)
    yc = 60 + 0.1 * x                                   # recto band, box-local
    rec = np.exp(-0.5 * ((y - yc) / 1.5) ** 2)
    ver = np.exp(-0.5 * ((y - (yc - T)) / 1.5) ** 2)
    rp, vp, ep = (str(tmp_path / n) for n in ("recto.zarr", "verso.zarr", "eval.zarr"))
    stores.write(rp, stores.u8(rec), o, q=0)
    stores.write(vp, stores.u8(ver), o, q=0)
    stores.write(ep, stores.u8(np.exp(-0.5 * ((y - yc - 0.3) / 2.0) ** 2)), o, q=0)   # another "teacher"
    umb = tmp_path / "umb.json"
    umb.write_text(json.dumps({"control_points": [{"z": 0, "y": -20000, "x": 448}, {"z": 1000, "y": -20000, "x": 448}]}))
    # published surface: fine pitch 6, whole z range of the box and beyond, +3 voxels in y
    zs, xs = np.arange(100, 290, 6, dtype=np.float32), np.arange(390, 506, 6, dtype=np.float32)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    fine = np.stack([Z, o[1] + 60 + 0.1 * (X - o[2]) + 3.0, X], -1)
    scale, off = 2.0, np.array([10.0, 20.0, 30.0])
    leg = (fine - off) / scale
    sd = tmp_path / "paths" / "segA" / "segA-on-20230205180739-7.91um.tifxyz"
    write_tifxyz_raw(str(sd), leg)
    out = tmp_path / "out"
    rc = R.main(["--recto", rp, "--verso", vp, "--paths", str(tmp_path / "paths"), "--umbilicus", str(umb),
                 "--out", str(out), "--z0", str(o[0] + 32), "--dz", "64", "--legacy-scale", "2.0",
                 "--legacy-offset", "10", "20", "30", "--far", "8", "--sigma-vox", "8", "--thr", "0.3",
                 "--eval-store", ep, "--slices", "3", "--taper", "4", "--crops", "3", "--crop-montage"])
    assert rc == 0
    import tifffile
    got = np.stack([tifffile.imread(str(out / "segA-on-20230205180739-7.91um" / f"{c}.tif")) for c in "zyx"], -1)
    assert got.shape == leg.shape
    gf = got * scale + off
    err = gf[..., 1] - (o[1] + 60 + 0.1 * (gf[..., 2] - o[2]))
    inside = (gf[..., 0] >= o[0] + 40) & (gf[..., 0] < o[0] + 88)
    outside = (fine[..., 0] < o[0] + 32 - 16) | (fine[..., 0] >= o[0] + 96 + 16)
    assert np.abs(err[inside]).mean() < 0.8, np.abs(err[inside]).mean()
    assert np.array_equal(got[outside], leg[outside].astype(np.float32))
    assert (out / "segA-on-20230205180739-7.91um.before" / "x.tif").exists()
    pngs = sorted(os.listdir(out / "png"))
    assert "displacement_hist.png" in pngs and sum(p.startswith("slab_z") for p in pngs) == 3
    rep = json.load(open(out / "refine_report.json"))
    assert rep["circular"] is False
    crops = [p for p in pngs if p.startswith("segA-on-20230205180739-7.91um_crop") and p[-5].isdigit()]
    assert len(crops) == 3 and "segA-on-20230205180739-7.91um_crops.png" in pngs
    mv = rep["moves"]["segA-on-20230205180739-7.91um"]
    assert mv["frac_moved_gt2"] > 0.3 and 1.0 < mv["mean_abs_move"] < 5.0
    assert rep["pooled"]["after"]["offset_le3"] >= rep["pooled"]["before"]["offset_le3"]
    assert abs(rep["pooled"]["after"]["offset_mean"]) < abs(rep["pooled"]["before"]["offset_mean"])


# ------------------------------------------------------------------------------------ bounded memory (slab mode)

def slanted_grid(H=40, W=60, step=6.0, slope=1.5):
    """A published-like grid whose rows are NOT z: z rises along the columns too, so a z slab cuts a diagonal
    band through it (the real 2.4 um segments cross a 128-voxel slab this way)."""
    r, c = np.mgrid[:H, :W].astype(np.float32)
    z = 100 + step * r + slope * c
    x = 390 + step * c
    y = 316 + 0.1 * (x - 384)
    g = np.stack([z, y, x], -1)
    g[5:8, 10:14] = np.nan
    return g


def test_read_surface_box_equals_crop_of_the_full_read(tmp_path):
    g = slanted_grid()
    d = str(tmp_path / "s.tifxyz")
    write_tifxyz_raw(d, g)
    o, s = np.array([160, 256, 384], np.float32), np.array([64, 128, 128], np.float32)
    for fr in (R.IDENTITY, R.scale_offset_frame([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])):
        crop, rc, shp = R.read_surface_box(d, fr, o, s, margin=10, rows=7)
        ref, rc_ref = R.crop_to_box(fr.to_fine(R.E.read_surface(d)), o, s, margin=10)
        assert shp == g.shape[:2] and rc == rc_ref
        np.testing.assert_array_equal(np.isnan(crop), np.isnan(ref))
        np.testing.assert_allclose(np.nan_to_num(crop), np.nan_to_num(ref))
    assert R.read_surface_box(d, R.IDENTITY, o + 5000, s) is None


def test_column_pieces_split_a_band_and_never_share_a_cell():
    g = slanted_grid(H=40, W=200, slope=0.3)
    g[:, 90:130] = np.nan                                  # a gap: two runs
    ps = R.column_pieces(g, (160, 0, 0), (224, 1000, 2000), pad=4)
    assert len(ps) == 2
    cells = np.zeros(g.shape[:2], int)
    for r0, r1, c0, c1 in ps:
        cells[r0:r1, c0:c1] += 1
        assert (r1 - r0) < g.shape[0]                      # each piece has its own (smaller) row range
    assert cells.max() == 1
    k = np.isfinite(g).all(-1) & (g[..., 0] >= 160) & (g[..., 0] < 224)
    assert (cells[k] == 1).all()                           # every in-box cell is in a piece


def test_tiled_slab_matches_one_tile(tmp_path, has_volcomp):
    """The same slab refined as one tile and as 3x3 tiles (a 48-voxel core, the minimum halo): the published
    nodes each tile owns end up where the one-tile run puts them."""
    if not has_volcomp:
        pytest.skip("volcomp not available")
    import tifffile

    from rvsm import stores
    o = np.array([128, 256, 384])
    z, y, x = np.mgrid[:128, :128, :128].astype(np.float32)
    yc = 60 + 0.1 * x
    rp, vp = str(tmp_path / "recto.zarr"), str(tmp_path / "verso.zarr")
    stores.write(rp, stores.u8(np.exp(-0.5 * ((y - yc) / 1.5) ** 2)), o, q=0)
    stores.write(vp, stores.u8(np.exp(-0.5 * ((y - (yc - 8)) / 1.5) ** 2)), o, q=0)
    umb = tmp_path / "umb.json"
    umb.write_text(json.dumps({"control_points": [{"z": 0, "y": -20000, "x": 448}, {"z": 1000, "y": -20000, "x": 448}]}))
    g = slanted_grid(H=34, W=22, step=6.0)
    g[..., 1] = o[1] + 60 + 0.1 * (g[..., 2] - o[2]) + 3.0          # 3 voxels off the band
    write_tifxyz_raw(str(tmp_path / "paths" / "segB" / "segB-on-20260411134726-2.4um.tifxyz"), g)
    outs = {}
    for t in (0, 48):
        out = tmp_path / f"out{t}"
        assert R.main(["--recto", rp, "--verso", vp, "--paths", str(tmp_path / "paths"), "--umbilicus", str(umb),
                       "--out", str(out), "--z0", str(o[0] + 32), "--dz", "64", "--far", "8", "--sigma-vox", "8",
                       "--thr", "0.3", "--thickness", "8", "--slices", "2", "--taper", "4", "--tile", str(t),
                       "--no-surface-png"]) == 0
        outs[t] = np.stack([tifffile.imread(str(out / "segB-on-20260411134726-2.4um" / f"{c}.tif")) for c in "zyx"], -1)
        rep = json.load(open(out / "refine_report.json"))
        assert len(rep["stats"]) == (1 if t == 0 else 9)
    a, b = outs[0], outs[48]
    moved = np.abs(a - np.where(np.isfinite(g), g, -1)).max(-1) > 1e-3
    assert moved.sum() > 20
    np.testing.assert_allclose(a, b, atol=0.05)
    assert (tmp_path / "out48" / "png" / "displacement_hist.png").exists()


# ------------------------------------------------------------------------------------ local following

def bump_band(shape=(16, 48, 96), amp=3.0, width=4.0):
    """A band at y = 24 + amp * a narrow bump in x (a local feature a wide smoothing flattens)."""
    return lambda x: 24 + amp * np.exp(-0.5 * ((x - shape[2] / 2) / width) ** 2)


def test_sigma_final_follows_a_local_bump_the_wide_smoothing_flattens():
    yc = bump_band()
    V = band_volume((16, 48, 96), yc, width=1.2)
    g = flat_sheet(24.0, z=(2, 14), x=(2, 94))
    peak = np.abs(g[..., 2] - 48) <= 1
    errs = {}
    for fin in (6.0, 0.7):
        (g1,), _ = R.refine_many([g], V, (0, 0, 0), AX_Y, far=6, sigma=6.0, sigma_final=fin, iters=3, thr=0.3)
        errs[fin] = np.abs(g1[..., 1] - yc(g1[..., 2]))[peak].mean()
    assert errs[0.7] < 0.5 * errs[6.0], errs


def test_local_normal_off_keeps_the_published_normals():
    V = band_volume((16, 48, 32), lambda x: 26 + 0 * x)
    g = flat_sheet(24.0)
    (a,), _ = R.refine_many([g], V, (0, 0, 0), AX_Y, far=6, sigma=1.0, iters=2, thr=0.3, local_normal=False)
    (b,), _ = R.refine_many([g], V, (0, 0, 0), AX_Y, far=6, sigma=1.0, iters=2, thr=0.3)
    inner = (slice(2, -2), slice(2, -2))
    assert np.abs(a[inner][..., 1] - 26).mean() < 0.3 and np.abs(b[inner][..., 1] - 26).mean() < 0.3
    assert np.allclose(a[..., [0, 2]], g[..., [0, 2]])        # moved along the fixed +y normal only


def test_pick_crops_spreads_over_z_and_finds_the_big_moves():
    rng = np.random.default_rng(0)
    cd = {}
    for z in (10.5, 20.5, 30.5, 40.5):
        P = np.stack([np.full(400, 100.0), np.linspace(0, 1000, 400)], -1).astype(np.float32)
        M = np.where(P[:, 1] > 800, 6.0, 0.5).astype(np.float32) + rng.random(400).astype(np.float32) * 0.1
        cd[z] = ([], [], [P], [M])
    picks = R.pick_crops(cd, 4, win=128)
    assert [p[0] for p in picks] == [10.5, 20.5, 30.5, 40.5]
    assert picks[1][4] == "largest moves" and picks[1][2] > 800 and picks[1][3] > 5
    xs = [p[2] for p in picks]
    assert min(abs(a - b) for i, a in enumerate(xs) for b in xs[i + 1:]) >= 128


# ------------------------------------------------------------------------------------ no bunching, no folds

def wavy_case(A=3.0, lam=40.0):
    yc = lambda x: 30 + A * np.sin(2 * np.pi * x / lam)   # noqa: E731
    V = band_volume((20, 64, 128), yc, width=1.2)
    Z, X = np.meshgrid(np.arange(2, 18, 2, dtype=np.float32), np.arange(2, 126, 2, dtype=np.float32), indexing="ij")
    g = np.stack([Z, np.full_like(Z, 30.0), X], -1)
    return V, g, yc


def u_spacing_cv(g):
    e = np.linalg.norm(np.diff(g, axis=1), axis=-1)[1:-1, 2:-2]
    return float(e.std() / e.mean())


def test_wavy_band_keeps_even_spacing_and_never_folds():
    """A wavy published sheet half a wavelength out of phase with its wavy band: the old scheme (normals
    recomputed every iteration, no relaxation, no guard) slides nodes down the slopes -- spacing CV ~0.9,
    folded quads. Normal-only moves + reparametrisation + the guard: even spacing, no fold, on the band."""
    A, lam = 4.0, 32.0
    yc = lambda x: 32 + A * np.sin(2 * np.pi * x / lam)   # noqa: E731
    V = band_volume((20, 64, 128), yc, width=1.2)
    Z, X = np.meshgrid(np.arange(2, 18, 2, dtype=np.float32), np.arange(2, 126, 2, dtype=np.float32), indexing="ij")
    g = np.stack([Z, 32 + A * np.sin(2 * np.pi * (X / lam - 0.5)), X], -1).astype(np.float32)
    kw = dict(far=12, sigma=1.0, iters=6, thr=0.3)
    (old,), _ = R.refine_many([g], V, (0, 0, 0), AX_Y, local_normal=True, relax=0, guard=False, reparam_on=False, **kw)
    dg = {}
    # --reparam is off by default (on a real, jagged snap it drifted nodes by tens of voxels) but is what evens
    # out a smooth out-of-phase sheet like this one
    (new,), st = R.refine_many([g], V, (0, 0, 0), AX_Y, diag=dg, reparam_on=True, **kw)
    n0 = R.normals(g, AX_Y)
    min_sp = 0.4 * R.pitch(g)
    assert u_spacing_cv(old) > 0.5 and R.bad_nodes(old, g, n0, min_sp).sum() > 0      # the bug, reproduced
    assert u_spacing_cv(new) < 0.1, u_spacing_cv(new)
    assert R.bad_nodes(new, g, n0, min_sp).sum() == 0
    inner = (slice(1, -1), slice(2, -2))
    assert np.abs(new[inner][..., 1] - yc(new[inner][..., 2])).mean() < 0.3
    assert sum(s["folds"] for s in st) == sum(dg["folds"])


def test_reparam_restores_the_published_fractions_along_the_surface():
    g = flat_sheet(10.0, z=(0, 4), x=(0, 12))
    bunched = g.copy()
    bunched[..., 2] = 11 * (np.linspace(0, 1, 12) ** 2)[None]           # same line, nodes crowded at x=0
    out = R.reparam(bunched, g, np.ones(g.shape[:2], np.float32))
    np.testing.assert_allclose(out, g, atol=1e-5)
    held = R.reparam(bunched, g, np.zeros(g.shape[:2], np.float32))
    np.testing.assert_array_equal(held, bunched)


def test_fold_guard_pulls_back_a_crossing_node():
    g = flat_sheet(10.0, z=(0, 6), x=(0, 8)) * np.array([1, 1, 2], np.float32)   # x pitch 2
    n0 = np.zeros_like(g)
    n0[..., 1] = 1
    bad = g.copy()
    bad[3, 3, 2] += 5.0                                    # slides past two neighbours: folds
    P, nev = R.fold_guard(g, bad, g, n0, 0.8)
    assert nev > 0 and R.bad_nodes(P, g, n0, 0.8).sum() == 0
    ok = g.copy()
    ok[..., 1] += 1.0                                      # a pure normal move: nothing to guard
    P, nev = R.fold_guard(g, ok, g, n0, 0.8)
    assert nev == 0 and np.allclose(P, ok)


def test_far_schedule_and_the_half_gap_bound():
    assert R.far_schedule([40, 24, 16, 8], 6) == [40, 24, 16, 8, 8, 8] and R.far_schedule(12, 3) == [12, 12, 12]
    # a lone sheet 20 voxels off its band: a wide first pass finds it, a fixed 8 cannot
    V = band_volume((16, 64, 32), lambda x: 40 + 0 * x)
    g = flat_sheet(20.0)
    (a,), _ = R.refine_many([g], V, (0, 0, 0), AX_Y, far=[32, 16, 8], sigma=1.0, iters=3, thr=0.3)
    (b,), _ = R.refine_many([g], V, (0, 0, 0), AX_Y, far=8, sigma=1.0, iters=3, thr=0.3)
    inner = (slice(2, -2), slice(2, -2))
    assert np.abs(a[inner][..., 1] - 40).mean() < 0.5 and np.abs(b[inner][..., 1] - 20).mean() < 0.5
    # with a neighbour sheet 16 voxels above, the band 12 above is past half-way: not taken
    up = flat_sheet(36.0)
    (c, _), _ = R.refine_many([flat_sheet(24.0), up], band_volume((16, 64, 32), lambda x: 36 + 0 * x),
                              (0, 0, 0), AX_Y, far=16, sigma=1.0, iters=2, thr=0.3)
    assert np.abs(c[inner][..., 1] - 24).max() <= 8.01


def test_peak_tol_prefers_the_nearer_of_two_comparable_peaks():
    V = np.maximum(band_volume((16, 64, 32), lambda x: 36 + 0 * x),
                   0.9 * band_volume((16, 64, 32), lambda x: 27 + 0 * x))
    g = flat_sheet(24.0)
    (a,), _ = R.refine_many([g], V, (0, 0, 0), AX_Y, far=16, sigma=1.0, iters=2, thr=0.3)
    inner = (slice(2, -2), slice(2, -2))
    assert np.abs(a[inner][..., 1] - 27).mean() < 0.5


def test_mesh_opt_lowers_the_data_loss_and_keeps_the_spacing():
    V, g, yc = wavy_case()
    (snap,), _ = R.refine_many([g], V, (0, 0, 0), AX_Y, far=8, sigma=3.0, sigma_final=3.0, iters=2, thr=0.3)
    n0 = R.normals(g, AX_Y)
    mov = (R._nbr_mean(snap)[1] == 4)
    (opt,), h = R.mesh_opt([snap], (V * 255).astype(np.uint8), (0, 0, 0), [n0], [mov], steps=100, lr=0.1,
                           rest=[g], device="cpu")
    assert h["data"][1] < h["data"][0]
    assert abs(u_spacing_cv(opt) - u_spacing_cv(snap)) < 0.05
    assert R.bad_nodes(opt, g, n0, 0.4 * R.pitch(g)).sum() == 0


def test_mesh_opt_is_joint_and_keeps_two_sheets_apart():
    V = band_volume((16, 64, 32), lambda x: 30 + 0 * x)          # one band, two sheets want it
    lo_, hi_ = flat_sheet(28.0), flat_sheet(32.0)
    n0 = [R.normals(lo_, AX_Y), R.normals(hi_, AX_Y)]
    mov = [R._nbr_mean(x)[1] == 4 for x in (lo_, hi_)]
    (a, b), h = R.mesh_opt([lo_, hi_], (V * 255).astype(np.uint8), (0, 0, 0), n0, mov, steps=150, lr=0.1,
                           rest=[lo_, hi_], min_gap=4.0, w_gap=5.0, device="cpu")
    assert "gap" in h and "cross" in h
    inner = (slice(2, -2), slice(2, -2))
    gap = b[inner][..., 1] - a[inner][..., 1]
    (a0, b0), _ = R.mesh_opt([lo_, hi_], (V * 255).astype(np.uint8), (0, 0, 0), n0, mov, steps=150, lr=0.1,
                             rest=[lo_, hi_], w_gap=0.0, w_cross=0.0, device="cpu")
    gap0 = b0[inner][..., 1] - a0[inner][..., 1]
    assert gap0.mean() < 1.0                                      # alone, each sheet takes the one band
    assert gap.min() > 2.0 and gap.mean() > 3.0                   # jointly: never merged, never crossed
    st = R.pair_stats([lo_, hi_], [a, b], n0, 4.0)
    assert st["crossings"] == 0 and st["pairs"] > 0


def test_duplicate_traces_of_one_wrap_both_snap_to_it_instead_of_walling_each_other_off():
    """Published segmentations overlap: two traces of ONE wrap 3 voxels apart (the band at 30, the next wrap
    at 50). They are duplicates, not neighbours: both land on 30. As neighbours, the half-gap bound would hold
    them 3 voxels apart and one would stay off the band."""
    shape = (16, 72, 32)
    V = np.maximum(band_volume(shape, lambda x: 30 + 0 * x, 1.2), band_volume(shape, lambda x: 50 + 0 * x, 1.2))
    a, b = flat_sheet(34.0), flat_sheet(37.0)
    far_ = flat_sheet(52.0)
    dup = R.duplicate_sheets([a, b, far_], 8.0)
    assert dup[0, 1] and dup[1, 0] and not dup[0, 2] and not dup[1, 2]
    dg = {}
    (a1, b1, f1), _ = R.refine_many([a, b, far_], V, (0, 0, 0), AX_Y, far=[12, 8], iters=2, sigma=1.0,
                                    thr=0.3, dup_gap=8.0, diag=dg)
    assert abs(a1[..., 1].mean() - 30) < 1.0 and abs(b1[..., 1].mean() - 30) < 1.0, (a1[..., 1].mean(), b1[..., 1].mean())
    assert abs(f1[..., 1].mean() - 50) < 1.0
    assert dg["dup"][0, 1] and not dg["dup"][0, 2]


def test_no_cross_pulls_back_a_crossing_pair():
    lo_, hi_ = flat_sheet(30.0), flat_sheet(40.0)
    a, b = lo_.copy(), hi_.copy()
    a[3:6, 5:9, 1] = 44.0          # a patch of the lower sheet jumped through the upper one
    n = [R.normals(g, AX_Y) for g in (lo_, hi_)]
    assert R.pair_stats([lo_, hi_], [a, b], n, 4.0, stride=1)["crossings"] > 0
    (a2, b2), k = R.no_cross([lo_, hi_], [a, b], n, R=20.0)
    assert k > 0 and R.pair_stats([lo_, hi_], [a2, b2], n, 4.0, stride=1)["crossings"] == 0
    assert np.allclose(a2[0, 0], a[0, 0])     # untouched away from the crossing
