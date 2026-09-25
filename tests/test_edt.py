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
    for p in (0.1, 0.3, 0.5):
        m = torch.from_numpy(rng.random((3, 20, 30, 25)) < p).cuda()
        lab = E.label(m)
        k, E._TRITON = E._TRITON, False
        try:
            lt = E.label(m)
        finally:
            E._TRITON = k
        assert torch.equal(lab, lt)                  # both: each component's largest 1 + index
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
        m = rng.random((3, 20, 30, 25)) < p
        labs = E.label(torch.from_numpy(m).to(dev)).cpu().numpy()      # a batch: volume by volume
        for b in range(3):
            lab = labs[b]
            ref = ndi.label(m[b], np.ones((3, 3, 3), bool))[0]
            assert (lab[~m[b]] == 0).all() and (lab[m[b]] > 0).all()
            pairs = set(zip(lab[m[b]].tolist(), ref[m[b]].tolist()))
            assert len(pairs) == len(set(lab[m[b]].tolist())) == len(set(ref[m[b]].tolist())), p
        one = E.label(torch.from_numpy(m[1]).to(dev)).cpu().numpy()
        assert np.array_equal(one, labs[1])


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


@pytest.mark.parametrize("dev", DEVICES)
def test_the_walk_kernel_is_the_numpy_walk(dev):
    """`walk_crossings` against the numpy loop of `targets._pair_checks` (c): the same samples (float32
    ratio, float32 multiply and add, round half to even) and the same leave/enter rules, per segment
    with its own block's sample count."""
    rng = np.random.default_rng(4)
    shape = (2, 12, 14, 16)
    br = rng.random(shape) < 0.4
    bv = rng.random(shape) < 0.4
    M = 400
    b = rng.integers(0, 2, M)
    pa = np.stack([rng.integers(0, s, M) for s in shape[1:]])
    pb = np.stack([rng.integers(0, s, M) for s in shape[1:]])
    act = rng.random(M) < 0.8
    seg = (pb - pa).astype(np.float32)
    ln = np.sqrt((seg * seg).sum(0))
    nblk = [max(1, int(np.ceil(2.0 * float(ln[act & (b == k)].max(initial=0))))) for k in range(2)]
    want = np.zeros(M, bool)
    for k in range(2):
        w = np.flatnonzero(act & (b == k))
        n = nblk[k]
        a, sg = pa[:, w].astype(np.float32), seg[:, w]
        left, inv, hit = (np.zeros(w.size, bool) for _ in range(3))
        for i in range(n + 1):
            qi = tuple(np.rint(a + (i / n) * sg).astype(np.int64))
            rr, vv = br[k][qi], bv[k][qi]
            hit |= left & rr & ~vv
            left |= ~rr
            hit |= inv & ~vv
            inv |= vv
        want[w] = hit
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    V = int(np.prod(shape[1:]))
    got = E.walk_crossings(t(pa), t(pb), t(b.astype(np.int64) * V), t(act),
                           t(np.array(nblk, np.int32)[b]), 64, t(br), t(bv))
    assert np.array_equal(got.cpu().numpy(), want)



@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("torch_only", [False, True])
def test_a_capped_edt_is_exact_up_to_the_cap(dev, torch_only):
    """`edt2(cap=)`: every voxel within `cap` of the mask gets the uncapped squared distance AND the
    uncapped nearest index (so scipy's tie order); every other voxel +inf. Random masks (sparse ones
    leave most voxels far beyond small caps), a batch, several caps, the Triton and the torch pass."""
    if dev == "cpu" and not torch_only:
        pytest.skip("the CPU has only the torch pass")
    rng = np.random.default_rng(7)
    ms = _masks(rng) + [rng.random((3, 40, 37, 29)) < 0.001]
    for m in ms:
        t = torch.from_numpy(m).to(dev)
        v, ix = E.edt2(t, torch_only=torch_only)
        for cap in (1.0, 2.5, 6.0, 13.0, 49.0):
            vc, ixc = E.edt2(t, torch_only=torch_only, cap=cap)
            near = v <= cap * cap
            assert torch.equal(vc[near], v[near]), cap
            assert torch.equal(ixc[:, near], ix[:, near]), cap
            assert torch.isinf(vc[~near]).all(), cap
            vn, none = E.edt2(t, indices=False, torch_only=torch_only, cap=cap)
            assert none is None and torch.equal(vn, vc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_the_box_gaussian_is_the_whole_volume_filter_inside_its_box():
    """`gaussian3_box`: on each field / volume's box widened by the margin, the same bits as
    `gaussian3` of the whole volume (the nearest-mode clamping is to the volume, not the box), for
    stacked and separately allocated fields, empty boxes, boxes at the borders and thin volumes."""
    g = torch.Generator().manual_seed(0)
    for shape in ((1, 40, 37, 29), (3, 64, 64, 64), (2, 5, 70, 9)):
        ab = (torch.randn((2,) + shape, generator=g) * 10).cuda()
        whole = [E.gaussian3(ab[0], 1.5), E.gaussian3(ab[1], 1.5)]
        B = shape[0]
        box = torch.full((2, B, 6), E.BOX_EMPTY, dtype=torch.int32)
        for f in range(2):
            for b in range(B):
                if (f + b) % 3 == 2:
                    continue                       # an empty box
                for c in range(3):
                    n = shape[1 + c]
                    lo = int(torch.randint(0, n, (1,), generator=g))
                    hi = int(torch.randint(lo, n, (1,), generator=g))
                    if (f + b + c) % 4 == 0:
                        lo, hi = 0, n - 1          # touching both borders
                    box[f, b, 2 * c], box[f, b, 2 * c + 1] = lo, -hi
        out = E.gaussian3_box([ab[0], ab[1]], box.cuda(), 1.5)
        seen = 0
        for f in range(2):
            for b in range(B):
                bx = box[f, b].tolist()
                if bx[0] == E.BOX_EMPTY:
                    continue
                sl = tuple(slice(max(bx[2 * c] - 1, 0), min(-bx[2 * c + 1] + 1, shape[1 + c] - 1) + 1)
                           for c in range(3))
                assert torch.equal(out[f, b][sl].view(torch.int32), whole[f][b][sl].view(torch.int32))
                seen += 1
        assert seen
    a, b = torch.randn((2, 30, 31, 32), device="cuda"), torch.randn((2, 30, 31, 32), device="cuda")
    full = torch.tensor([[[0, -29, 0, -30, 0, -31]] * 2] * 2, dtype=torch.int32, device="cuda")
    out = E.gaussian3_box([a, b], full, 1.5)
    assert torch.equal(out[0].view(torch.int32), E.gaussian3(a, 1.5).view(torch.int32))
    assert torch.equal(out[1].view(torch.int32), E.gaussian3(b, 1.5).view(torch.int32))
