"""The scroll axis: both published formats, deriving one from the CT, and the radial channels."""
import numpy as np
import pytest

from rvsm import axis as A


def test_both_formats_parse_to_the_same_points(umbilicus):
    j, t, pts = umbilicus
    assert A.read(j) == pytest.approx(pts)
    assert A.read(t) == pytest.approx(pts)          # "x, y, z", 1-based, undone on the way in
    assert np.allclose(A.load(j), A.load(t))


def test_load_sorts_by_z_and_writes_back(tmp_path, umbilicus):
    ax = A.load(umbilicus[0])
    assert ax.shape[0] == 3
    assert (np.diff(ax[0]) > 0).all()
    p = A.write(tmp_path / "w.json", list(zip(*ax)))
    assert np.allclose(A.load(p), ax)


def test_load_rescales_a_coarse_volume(tmp_path, umbilicus):
    """A point read off level 0 of a 9.6 um volume is a rung-4 voxel: rung-2 units are 4x."""
    d = tmp_path / "20260101000000-9.600um-x.zarr"
    d.mkdir()
    ax2 = A.load(umbilicus[0])
    ax4 = A.load(umbilicus[0], ct=str(d))
    assert np.allclose(ax4, ax2 * 4)


def test_derive_recovers_the_slab_centre(ct_origin):
    from tests.conftest import BASE
    pts = A.derive(ct_origin.path, rung=5)
    ax = np.array(sorted(pts), np.float64).T
    assert ax.shape[1] >= 2
    assert np.abs(ax[1] - BASE / 2).max() < 2.0     # y centroid, in rung-2 voxels
    assert np.abs(ax[2] - BASE / 2).max() < 2.0     # x centroid


def test_auto_goes_through_load_and_ensure(tmp_path, ct_origin):
    from tests.conftest import BASE
    p = A.ensure(tmp_path / "run", "auto", ct=ct_origin.path, rung=5)
    ax = A.load(p)
    assert np.abs(ax[1] - BASE / 2).max() < 2.0
    assert A.ensure(tmp_path / "run", "auto", ct=ct_origin.path) == p  # already there: not rederived


def test_axis_at_rescales_by_rung(umbilicus):
    ax = A.load(umbilicus[0])
    assert np.allclose(A.axis_at(ax, 2), ax)
    assert np.allclose(A.axis_at(ax, 5), ax / 8)


def test_radial_is_a_unit_vector_away_from_the_axis(umbilicus):
    ax = A.load(umbilicus[0])
    r = A.radial(ax, (0, 0, 0), 64)
    assert r.shape == (3, 64, 64, 64)
    assert (r[0] == 0).all()                         # no z component
    n = np.sqrt(r[1] ** 2 + r[2] ** 2)
    assert np.abs(n - 1).max() < 1e-3
    c = 128                                          # the axis passes through (y, x) = (128, 128)
    assert r[1, 0, 0, 0] < 0 and r[2, 0, 0, 0] < 0   # a corner below the axis points away from it
    r2 = A.radial(ax, (0, c + 64, c), 4)          # far out in +y: the vector is essentially +y
    assert r2[1].min() > 0.99                        # straight out in +y


def test_radius_plane_is_normalised(umbilicus):
    ax = A.load(umbilicus[0])
    shape = (64, 256, 256)
    rmax = A.rmax_vox(ax, shape)
    r = A.radius(ax, (0, 0, 0), shape, rmax)
    assert r.shape == (1,) + shape
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert r.max() > 0.9                             # the far corner nearly reaches r_max
    assert r[0, 0, 128, 128] == pytest.approx(0.0, abs=1e-6)   # on the axis


def test_rmax_is_the_far_corner(umbilicus):
    ax = A.load(umbilicus[0])
    assert A.rmax_vox(ax, (16, 256, 256)) == pytest.approx(np.sqrt(2) * 128, rel=1e-6)


def test_scale_plane(umbilicus):
    for k, want in ((2, 0.0), (3, 1 / 9), (11, 1.0)):
        v = A.scale_plane(k, 8)
        assert v.shape == (8, 8, 8)
        assert np.allclose(v, want)
