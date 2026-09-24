"""`rvsm.edt`, the torch operators behind `targets.block_fields_torch`, against the scipy.ndimage calls
they replace: the EDT's distances and nearest indices, the 26-connected labelling (as a partition), the
Gaussian, the 3^3 maximum filter, the 6-neighbour dilation and the uint8 2x pool. On the CPU always, and
on CUDA (the Triton min-plus pass) when a GPU is present."""
import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from rvsm import edt as E, ladder

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                                reason="no CUDA device"))]


def _masks(rng):
    """Random sparse and dense masks with empty rows and planes, a single voxel, and a curved sheet."""
    out = []
    for shape, p in (((20, 31, 17), 0.002), ((24, 24, 24), 0.02), ((17, 40, 33), 0.3), ((9, 9, 9), 0.7)):
        m = rng.random(shape) < p
        m[2] = False                      # an empty z plane
        m[:, 5] = False                   # an empty y plane
        m[:, :, 3] = False                # an empty x plane
        m[4, 7] = False                   # an empty row
        m.flat[0] = True                  # never wholly empty
        out.append(m)
    one = np.zeros((13, 11, 15), bool)
    one[6, 5, 7] = True
    out.append(one)
    n = 48
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r = np.sqrt((y - 24.0) ** 2 + (x - 20.0) ** 2)
    out.append(np.broadcast_to(np.abs(r - 15) <= 0.5, (6, n, n)).copy())
    return out


@pytest.mark.parametrize("dev", DEVICES)
def test_edt_distances_and_indices_match_scipy(dev):
    rng = np.random.default_rng(0)
    for m in _masks(rng):
        u, ix = E.edt(torch.from_numpy(m).to(dev))
        u, ix = u.cpu().numpy(), ix.cpu().numpy()
        ref, rix = ndi.distance_transform_edt(~m, return_indices=True)
        assert u.dtype == np.float32 and ix.dtype == np.int64 and ix.shape == (3,) + m.shape
        assert np.abs(u - ref).max() < 1e-3
        assert m[tuple(ix)].all()                                  # every index is a True voxel ...
        at = np.sqrt(((np.indices(m.shape) - ix) ** 2).sum(0))
        assert np.abs(at - ref).max() < 1e-3                       # ... at exactly that distance
        assert np.array_equal(ix, rix)                             # and scipy's own tie order
        d2, none = E.edt2(torch.from_numpy(m).to(dev), indices=False)
        assert none is None and np.array_equal(np.sqrt(d2.cpu().numpy()), u)


def test_the_triton_kernels_equal_the_torch_implementations():
    """On CUDA the passes and the Gaussian run as Triton kernels; they must give the torch fallback's
    bits (the distances, every carried index, the smoothed values)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(3)
    for m in _masks(rng):
        t = torch.from_numpy(m).cuda()
        v, ix = E.edt2(t)
        vt, ixt = E.edt2(t, torch_only=True)
        assert torch.equal(v, vt) and torch.equal(ix, ixt)
    x = torch.from_numpy((rng.random((19, 23, 29)) * 40 - 20).astype(np.float32)).cuda()
    g = E.gaussian3(x, 1.5)
    k, E._TRITON = E._TRITON, False
    try:
        gt = E.gaussian3(x, 1.5)
    finally:
        E._TRITON = k
    assert (g - gt).abs().max().item() <= 1e-6


@pytest.mark.parametrize("dev", DEVICES)
def test_label_is_scipys_26_connected_partition(dev):
    rng = np.random.default_rng(1)
    for p in (0.05, 0.2, 0.3, 0.5):
        m = rng.random((20, 30, 25)) < p
        lab = E.label(torch.from_numpy(m).to(dev)).cpu().numpy()
        ref = ndi.label(m, np.ones((3, 3, 3), bool))[0]
        assert (lab[~m] == 0).all() and (lab[m] > 0).all()
        pairs = set(zip(lab[m].tolist(), ref[m].tolist()))
        assert len(pairs) == len(set(lab[m].tolist())) == len(set(ref[m].tolist())), p   # a bijection


@pytest.mark.parametrize("dev", DEVICES)
def test_filters_match_scipy(dev):
    rng = np.random.default_rng(2)
    x = (rng.random((19, 23, 29)) * 40 - 20).astype(np.float32)
    t = torch.from_numpy(x).to(dev)
    g = E.gaussian3(t, 1.5).cpu().numpy()
    assert g.dtype == np.float32
    assert np.abs(g - ndi.gaussian_filter(x, 1.5, mode="nearest")).max() < 1e-5
    assert np.array_equal(E.max_filter3(t).cpu().numpy(), ndi.maximum_filter(x, size=3, mode="nearest"))
    m = rng.random(x.shape) < 0.05
    assert np.array_equal(E.binary_dilation(torch.from_numpy(m).to(dev)).cpu().numpy(),
                          ndi.binary_dilation(m))
    for shape in ((16, 18, 20), (15, 17, 9)):
        v = rng.integers(0, 256, shape, dtype=np.uint8)
        assert np.array_equal(E.pool2(torch.from_numpy(v).to(dev)).cpu().numpy(), ladder.pool2(v))
