"""The losses, on synthetic tensors small enough to reason about exactly."""
import numpy as np
import pytest
import torch

from rvsm import losses as L
from rvsm.config import Config


def _layout(**kw):
    return Config(**kw).layout()


def test_exclusivity_is_zero_for_disjoint_channels():
    p = torch.zeros(1, 2, 8, 8, 8)
    p[:, 0, :4] = 0.9      # recto on one half
    p[:, 1, 4:] = 0.9      # verso on the other
    w = torch.ones_like(p)
    assert float(L.exclusivity(p, w)) == 0.0
    p[:, 1, :4] = 0.9      # now they overlap: 0.9 + 0.9 - 1 = 0.8 on half the voxels
    assert float(L.exclusivity(p, w)) == pytest.approx(0.4, abs=1e-5)
    w[:, 1] = 0.0          # ... and a sample with no verso store says nothing at all
    assert float(L.exclusivity(p, w)) == 0.0


def test_self_consistency_is_zero_when_the_pools_agree():
    torch.manual_seed(0)
    fine = torch.rand(2, 1, 8, 8, 8)
    cas = fine.clone()     # the cascade channel pools to exactly the same thing
    assert float(L.self_consistency(fine, cas)) == pytest.approx(0.0, abs=1e-6)
    assert float(L.self_consistency(fine, cas + 0.25)) == pytest.approx(0.25, abs=1e-6)
    sel = torch.tensor([1.0, 0.0])   # only the SELF samples are scored
    v = L.self_consistency(fine, cas + torch.tensor([0.4, 8.0]).view(2, 1, 1, 1, 1), sel=sel)
    assert float(v) == pytest.approx(0.4, abs=1e-6)


def test_skeleton_of_a_slab_is_one_voxel_thick_and_recall_is_perfect():
    t = torch.zeros(1, 1, 16, 16, 16)
    t[:, :, :, 6:11] = 1.0                     # a 5-voxel slab: its medial surface is the middle plane
    s = L.skeleton(t, iters=4)[0, 0].numpy()
    per_column = s[2:-2, :, 2:-2].sum(1)       # one hit per interior (z, x) column
    assert set(np.unique(per_column)) == {1.0}  # (the outer 2 columns erode away: the patch face
    assert set(np.unique(np.nonzero(s)[1])) == {8}   # is background, by design)
    assert float(L.skel_recall(t, t)) == pytest.approx(0.0, abs=1e-6)   # recall 1 -> loss 0
    assert float(L.skel_recall(torch.zeros_like(t), t)) == pytest.approx(1.0, abs=1e-6)


def test_affinity_targets_on_two_slabs_and_equivariance_under_flips():
    Z = Y = X = 24
    t = torch.zeros(1, 1, Z, Y, X)
    pitch = 8
    t[:, :, :, :, 4:6] = 1.0                   # two sheets `pitch` apart along x
    t[:, :, :, :, 4 + pitch:6 + pitch] = 1.0
    at, aw = L.affinity_targets(t, None, offsets=(pitch,), fg_only=True)
    assert at.shape[1] == 3 and aw.shape == at.shape
    ax = at[0, 2]                              # the x-axis channel of this offset
    wx = aw[0, 2]
    mid = 4 + pitch // 2                       # midway between the two sheets: both ends foreground...
    assert float(wx[:, :, mid].min()) > 0      # ... but NOT the same sheet: the gap breaks the segment
    assert float(ax[:, :, mid].max()) == 0.0
    inside = 5                                 # a pair centred inside one sheet has one end in air,
    assert float(wx[:, :, inside].max()) == 0.0    # so the question is not asked there at all

    at2, aw2 = L.affinity_targets(t.flip(-1), None, offsets=(pitch,), fg_only=True)
    assert torch.allclose(at2[:, 2], at[:, 2].flip(-1))
    assert torch.allclose(aw2[:, 2], aw[:, 2].flip(-1))


def test_affinity_names_and_counts_come_from_the_offsets_tuple():
    assert L.n_affinity((8, 16, 32)) == 9
    assert L.affinity_names((8,)) == ["aff8_z", "aff8_y", "aff8_x"]
    with pytest.raises(AssertionError):
        L.n_affinity((7,))                     # offsets are EVEN: d/2 must be a whole voxel


def test_pair_bands_never_overlap_when_the_sheet_is_thick_enough():
    half, tau = 1.5, 0.5
    m = torch.linspace(-20, 20, 801).view(1, 1, 1, 1, -1)
    for t in (2 * half, 3.0, 5.0, 12.0, 31.0):
        pr, pv = L.pair_bands(m, torch.full_like(m, float(t)), half=half, tau=tau)
        assert float((pr + pv - 1).clamp_min(0).max()) <= 1e-6, t


def test_ect_loss_is_zero_on_itself_and_has_finite_gradients():
    torch.manual_seed(0)
    p = torch.rand(1, 1, 24, 24, 24, requires_grad=True)
    hard = (p.detach() >= 0.5).float()
    assert float(L.ect_loss(hard, hard, dirs=2, res=8, margin=2, block=8, nblocks=2)) == 0.0
    v = L.ect_loss(p, hard, dirs=2, res=8, margin=2, block=8, nblocks=2)
    v.backward()
    assert torch.isfinite(v) and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0


def test_eikonal_of_a_linear_ramp_is_zero():
    Z = Y = X = 16
    x = torch.arange(X, dtype=torch.float32) - X / 2.0
    d = x.view(1, 1, 1, 1, X).expand(1, 1, Z, Y, X).contiguous()   # |grad d| = 1 exactly
    w = torch.ones(1, 1, Z, Y, X)
    assert float(L.eikonal(d, w, band=100.0)) == pytest.approx(0.0, abs=1e-6)
    assert float(L.eikonal(2 * d, w, band=100.0)) == pytest.approx(1.0, abs=1e-5)   # |grad| = 2


def test_grad3_and_normals_point_outward_from_the_field():
    X = 12
    d = (torch.arange(X, dtype=torch.float32) - 5.0).view(1, 1, 1, 1, X).expand(1, 1, 8, 8, X).contiguous()
    n = L.normals_from(d)
    assert torch.allclose(n[:, 2, 1:-1, 1:-1, 1:-1], torch.ones(1, 6, 6, X - 2), atol=1e-5)


def test_sdist_and_thickness_decode_the_store_encoding():
    code = torch.tensor([128.0, 168.0, 88.0]).view(1, 1, 1, 1, 3) / 255.0
    assert torch.allclose(L.decode_signed(code).flatten(), torch.tensor([0.0, 10.0, -10.0]), atol=1e-4)
    assert torch.allclose(L.decode_unsigned(torch.tensor([40.0]) / 255.0), torch.tensor([10.0]), atol=1e-4)
    pred = torch.zeros(1, 1, 4, 4, 4)
    tgt = torch.full((1, 1, 4, 4, 4), 128.0 / 255.0)
    w = torch.ones_like(pred)
    assert float(L.sdist_loss(pred, tgt, w)) == pytest.approx(0.0, abs=1e-6)
    assert float(L.thickness_loss(L.soft_thickness(torch.zeros(1, 1, 2, 2, 2)),
                                  torch.full((1, 1, 2, 2, 2), (L.TMIN + np.log(2)) / L.UNIT / 255.0),
                                  torch.ones(1, 1, 2, 2, 2))) == pytest.approx(0.0, abs=1e-3)


def test_dist_weight_drops_partially_resampled_voxels():
    w = torch.tensor([0.0, 0.5, 0.94, 0.96, 1.0])
    assert torch.equal(L.dist_weight(w), torch.tensor([0.0, 0.0, 0.0, 0.96, 1.0]))


def test_losses_tw_is_finite_and_carries_no_gradient_where_the_weight_is_zero():
    torch.manual_seed(0)
    logit = torch.randn(2, 2, 8, 8, 8, requires_grad=True)
    tgt = (torch.rand(2, 2, 8, 8, 8) > 0.5).float()
    w = torch.ones_like(tgt)
    w[:, 1] = 0.0                              # channel 1 has no store anywhere in this batch
    bce, dice = L.losses_tw(logit, tgt, w)
    assert torch.isfinite(bce) and torch.isfinite(dice)
    (bce + dice).backward()
    assert float(logit.grad[:, 1].abs().sum()) == 0.0     # the `live` rule: it is simply not there
    assert float(logit.grad[:, 0].abs().sum()) > 0

    z = torch.zeros(1, 1, 4, 4, 4, requires_grad=True)    # an all-zero weight gives 0, not a NaN
    b, d = L.losses_tw(z, torch.zeros(1, 1, 4, 4, 4), torch.zeros(1, 1, 4, 4, 4))
    assert float(b.detach()) == 0.0 and torch.isfinite(d)


def test_deep_losses_pool_the_target_and_the_weight():
    tgt = torch.zeros(1, 1, 8, 8, 8)
    tgt[:, :, :4] = 1.0
    w = torch.ones_like(tgt)
    big = torch.full((1, 1, 8, 8, 8), 10.0)              # a confident, correct level-0 head
    small = torch.full((1, 1, 4, 4, 4), 10.0)
    b1, d1 = L.deep_losses([big, small], tgt, w)
    b0, d0 = L.deep_losses(big, tgt, w)
    assert b1 > b0 and torch.isfinite(d1)                # the coarse head is scored too, at weight 0.5
    # the pooled target at level 1 is 1 on the first two planes and 0 on the rest: a head that says so
    perfect = torch.where(torch.arange(4).view(1, 1, 4, 1, 1) < 2, 20.0, -20.0).expand(1, 1, 4, 4, 4)
    b2, _ = L.deep_losses([torch.where(torch.arange(8).view(1, 1, 8, 1, 1) < 4, 20.0, -20.0)
                           .expand(1, 1, 8, 8, 8).contiguous(), perfect.contiguous()], tgt, w)
    assert float(b2) == pytest.approx(0.0, abs=1e-6)


def test_deep_losses_slices_to_the_probability_heads_with_a_layout():
    lay = _layout(channels=("recto", "verso"), aff_offsets=(8,))
    logit = torch.zeros(1, lay.cout, 8, 8, 8)
    tgt = torch.zeros(1, lay.cout_t, 8, 8, 8)
    w = torch.ones_like(tgt)
    bce, dice = L.deep_losses(logit, tgt, w, layout=lay)
    assert torch.isfinite(bce) and torch.isfinite(dice)


def test_aux_losses_reads_every_head_index_from_the_layout():
    lay = _layout(channels=("recto", "verso"), aff_offsets=(8,))
    torch.manual_seed(0)
    logit = torch.randn(1, lay.cout, 16, 16, 16)
    tgt = torch.zeros(1, lay.cout_t, 16, 16, 16)
    tgt[:, 0, :, :, 6:9] = 1.0
    tgt[:, 1, :, :, 1:4] = 1.0
    w = torch.ones_like(tgt)
    cas = torch.zeros(1, 1, 16, 16, 16)
    out = L.aux_losses(logit, tgt, w, lay, w_excl=0.1, w_selfcons=0.1, w_skel=0.05, w_affinity=0.1,
                       cascade=cas, cascade_self=torch.ones(1))
    assert set(out) == {"excl", "selfcons", "skel", "affinity", "aux"}
    assert all(torch.isfinite(v) for v in out.values())
    assert float(out["aux"]) == pytest.approx(0.1 * float(out["excl"]) + 0.1 * float(out["selfcons"])
                                              + 0.05 * float(out["skel"]) + 0.1 * float(out["affinity"]),
                                              rel=1e-5)
    assert L.aux_losses(logit, tgt, w, lay) == {}          # every weight 0 -> nothing computed


def test_the_weighted_dice_of_a_perfect_prediction_is_zero_under_fractional_weights():
    """(2 sum(w p t) + 1) / (sum(w p) + sum(w t) + 1): a perfect prediction scores dice loss ~0 whatever
    the weights. The old form put w^2 in the intersection, so a fractional weight charged it; the
    live-channel rule still drops a channel with no weight at all, and pair_dice goes through the same
    function (train.py scores it with `losses_tw`)."""
    torch.manual_seed(0)
    t = (torch.rand(1, 2, 16, 16, 16) > 0.7).float()
    logit = (t * 2 - 1) * 30.0                                   # p = t to float precision
    w = torch.rand(1, 2, 16, 16, 16) * 0.5 + 0.1                 # fractional everywhere
    _, d = L.losses_tw(logit, t, w)
    assert float(d) < 1e-3, float(d)
    w0 = w.clone()
    w0[:, 1] = 0                                                 # channel 1 says nothing: not live
    wrong = logit.clone()
    wrong[:, 1] = -logit[:, 1]
    _, d0 = L.losses_tw(wrong, t, w0)
    assert float(d0) < 1e-3, "a channel with no weight must not score"
    # and a wrong prediction is charged exactly the weighted formula
    p = torch.sigmoid(torch.zeros_like(logit))
    _, dz = L.losses_tw(torch.zeros_like(logit), t, w)
    want = 1 - (2 * (w * p * t).sum((0, 2, 3, 4)) + 1) / ((w * p).sum((0, 2, 3, 4)) + (w * t).sum((0, 2, 3, 4)) + 1)
    assert float(dz) == pytest.approx(float(want.mean()), rel=1e-5)


def test_ect_blocks_are_drawn_per_sample_and_skip_weightless_ones():
    """ect_loss takes `dirs` directions and `nblocks` blocks per sample drawn from a seeded generator
    (the same seed, the same draw), and only FULLY observed blocks (review T07, pass-3 P3-12)."""
    torch.manual_seed(0)
    t = (torch.rand(2, 1, 40, 40, 40) > 0.5).float()
    p = torch.rand(2, 1, 40, 40, 40)
    w = torch.zeros_like(t)
    w[0, :, 4:20, 4:20, 4:20] = 1                              # sample 0: only the first block is live
    w[0, :, 20:36, 4:20, 4:20] = 1
    w[0, :, 20, 10, 10] = 0                                    # ... this one has one unknown voxel
    a = L.ect_loss(p, t, dirs=3, res=8, margin=4, block=16, nblocks=1, w=w,
                   gen=torch.Generator().manual_seed(5))
    b = L.ect_loss(p, t, dirs=3, res=8, margin=4, block=16, nblocks=1, w=w,
                   gen=torch.Generator().manual_seed(5))
    assert float(a) == float(b) and float(a) > 0               # seeded: replayable
    only0 = L.ect_loss(p[:1], t[:1], dirs=3, res=8, margin=4, block=16, nblocks=4, w=w[:1],
                       gen=torch.Generator().manual_seed(1))
    first = L.ect_loss(p[:1], t[:1], dirs=3, res=8, margin=4, block=16, nblocks=1)
    assert float(only0) == pytest.approx(float(first))         # the one FULLY observed block, always
    none = L.ect_loss(p, t, dirs=3, res=8, margin=4, block=16, nblocks=2, w=torch.zeros_like(w))
    assert float(none) == 0.0                                   # nothing live: nothing scores
    seen = set()
    for s in range(12):                                         # the draw really moves around
        g = torch.Generator().manual_seed(s)
        seen.add(round(float(L.ect_loss(p[1:], t[1:], dirs=1, res=8, margin=4, block=16, nblocks=1,
                                        gen=g)), 8))
    assert len(seen) > 1


def test_the_ect_draw_differs_between_microbatches_of_a_step():
    from rvsm import train as TR
    torch.manual_seed(0)
    t = (torch.rand(1, 1, 72, 72, 72) > 0.5).float()
    p = torch.rand(1, 1, 72, 72, 72)
    vals = {round(float(L.ect_loss(p, t, dirs=1, res=8, margin=4, block=16, nblocks=1,
                                   gen=torch.Generator().manual_seed(TR.ect_seed(100, m)))), 8)
            for m in range(4)}
    assert len({TR.ect_seed(100, m) for m in range(4)} | {TR.ect_seed(101, 0)}) == 5
    assert len(vals) > 1


def test_no_auxiliary_loss_reads_a_zero_weight_label():
    """The held-out-label intervention: rewrite the TARGET wherever its weight is zero and every
    auxiliary loss must be unchanged -- an unknown label may not reach the skeleton recall (its
    skeleton depends on a neighbourhood) or the affinity targets (a min over the whole segment), only
    the two segment ends were checked before (pass-3 P3-14)."""
    from rvsm.config import Config
    lay = Config(channels=("recto", "verso"), aff_offsets=(8,)).layout()
    torch.manual_seed(0)
    S = 32
    tgt = torch.zeros(1, lay.cout_t, S, S, S)
    tgt[:, 0, :, 12:17, :] = 1.0                                   # a recto slab
    tgt[:, 1, :, 18:22, :] = 1.0
    wv = torch.ones_like(tgt)
    wv[:, :, :, :, 20:26] = 0                                      # an unknown stripe through both
    logit = torch.randn(1, lay.cout, S, S, S)
    kw = dict(w_excl=0.1, w_selfcons=0.1, w_skel=0.1, w_affinity=0.1, cascade=torch.rand(1, 1, S, S, S),
              cascade_self=torch.ones(1), skel_iters=4, w_skel_prec=0.1)
    base = L.aux_losses(logit, tgt, wv, lay, **kw)
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        mut = tgt.clone()
        noise = (torch.rand(tgt.shape, generator=g) > 0.5).float()
        mut = torch.where(wv > 0, mut, noise)                      # only the unknown labels change
        got = L.aux_losses(logit, mut, wv, lay, **kw)
        for k in ("skel", "skel_prec", "affinity", "excl", "selfcons"):
            assert float(got[k]) == pytest.approx(float(base[k]), abs=1e-6), (k, seed)


# ----------------------------------------------------------------------- skeleton precision (L8b)

def _slab(S=32, lo=12, hi=17, x1=None):
    t = torch.zeros(1, 1, S, S, S)
    t[:, :, :, lo:hi, :x1] = 1.0
    return t


def test_skeleton_precision_of_the_target_itself_is_perfect():
    t = _slab()
    assert float(L.skel_precision(t, t)) == pytest.approx(0.0, abs=1e-5)               # precision 1
    assert float(L.skel_precision(0.95 * t, t)) == pytest.approx(0.0, abs=1e-5)        # a soft copy too
    s = L.soft_skeleton(t, iters=3)
    assert float(s.sum()) > 0 and float((s * (1 - t)).sum()) == 0.0                   # inside the band


def test_skeleton_precision_charges_a_spur_outside_the_band():
    t = _slab()
    spur = t.clone()
    spur[:, :, 10:15, 17:28, 8:12] = 1.0          # a fin sticking 11 voxels out of the band
    base, bad = float(L.skel_precision(t, t)), float(L.skel_precision(spur, t))
    assert bad > base + 0.02, (base, bad)
    shift = torch.roll(t, 1, dims=3)               # one voxel off: inside the dilated band, not charged
    assert float(L.skel_precision(shift, t)) == pytest.approx(0.0, abs=1e-5)
    p = spur.clone().requires_grad_()              # and the gradient pushes the spur down
    L.skel_precision(p, t).backward()
    assert torch.isfinite(p.grad).all() and float(p.grad[:, :, 10:15, 20:28, 8:12].sum()) > 0


def test_skeleton_precision_does_not_charge_a_sheet_that_stops_short():
    """Stopping short is RECALL's job: a prediction that covers only half the target keeps its whole
    skeleton inside the band, so precision stays ~1 while the skeleton recall charges the gap."""
    t = _slab()
    short = _slab(x1=16)
    assert float(L.skel_precision(short, t)) == pytest.approx(0.0, abs=1e-5)
    assert float(L.skel_recall(short, t)) > 0.3


def test_skeleton_precision_gates_haze_and_unknown_labels():
    t = _slab()
    haze = torch.full_like(t, 0.2)                 # early-training haze below the gate: nothing scored
    assert float(L.skel_precision(haze, t)) == 0.0
    spur = t.clone()
    spur[:, :, 10:15, 17:28, 8:12] = 1.0
    wv = torch.ones_like(t)
    wv[:, :, :, 17:, :] = 0                        # the spur's region is unknown: it may not be charged
    assert float(L.skel_precision(spur, t, wv)) == pytest.approx(0.0, abs=1e-5)


def test_aux_losses_adds_the_skeleton_precision_only_when_weighted():
    lay = _layout(channels=("recto", "verso"), aff_offsets=(8,))
    torch.manual_seed(0)
    logit = torch.randn(1, lay.cout, 16, 16, 16)
    tgt = torch.zeros(1, lay.cout_t, 16, 16, 16)
    tgt[:, 0, :, :, 6:9] = 1.0
    w = torch.ones_like(tgt)
    out = L.aux_losses(logit, tgt, w, lay, w_skel=0.05, w_skel_prec=0.2)
    assert set(out) == {"skel", "skel_prec", "aux"}
    assert float(out["aux"]) == pytest.approx(0.05 * float(out["skel"]) + 0.2 * float(out["skel_prec"]),
                                              rel=1e-5)
    assert "skel_prec" not in L.aux_losses(logit, tgt, w, lay, w_skel=0.05)        # default: off
