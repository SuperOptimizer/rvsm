"""Every loss the student is trained with, in one module, with the head indices taken from `config.Layout`.

The training loop only orchestrates: it builds the batch, calls the net, and hands the head output plus a
`Layout` to the functions here. Nothing in this module reads a global, a flag table or a channel name --
which head is which comes from the layout it is given, so the one place the channel contract lives
(`rvsm.config.Layout`) is also the only place that has to agree with the sampler and the exporter.

Three groups:

  the supervised pair      `losses_tw` / `deep_losses`: weighted BCE + soft dice on the probability heads,
                           with the `live` rule (a channel with no weight anywhere in the batch is not
                           averaged in) and deep supervision over the decoder's multi-resolution outputs
  the auxiliary terms      `aux_losses`: exclusivity (recto/verso may not both fire), cross-rung
                           self-consistency, skeleton recall and precision, long-range affinity
  the field terms          signed midline distance (`sdist_loss`), thickness, the Eikonal regulariser,
                           normals derived from the field, construction-based pairing (`pair_bands`) and
                           the Euler-characteristic-transform topology term (`ect_loss`)

Affinity offsets are a TUPLE of even voxel counts, always (`config.Config.aff_offsets`); there is no
string parsing anywhere.

Distance encoding (shared with `rvsm.targets` and the tracer contract): a uint8 store code is
`128 + round(d / 0.25)` clamped to 1..255, code 0 = no data. The loader hands a distance channel as
`code / 255`, so `decode_signed` turns it back into voxels.
"""
import math

import torch
import torch.nn.functional as F


# ============================================================================ the supervised pair

DEEP_W = (1.0, 0.5, 0.25, 0.125)  # loss weight of the level-0 head and the coarser (deep supervision) heads


def losses_tw(logit, tgt, wv=None, ridge_w=0.0):
    """(bce, dice) of an already-prepared (target, weight) pair. The weight scales the BCE and masks the
    soft dice, so a voxel with weight 0 contributes to neither and carries no gradient.

    The BCE is ONE weighted mean over the whole tensor, so a channel that is weight 0 throughout the batch
    adds nothing to the numerator AND nothing to the denominator: it neither contributes nor rescales the
    others, and an all-zero weight tensor (the blank-patch augmentation) gives 0, not a NaN.

    The soft dice is per channel and is averaged over the channels that HAVE weight somewhere in the batch
    -- the `live` rule. An ignored channel would otherwise score a constant 0 and halve the loss of the
    channel that is actually being trained, which is exactly what lets the verso head sit beside the recto
    one: on a sample from a region with no verso store the verso channel simply is not there.

    `ridge_w` adds that much extra weight on the band's core (target >= 0.9), where the student must commit.
    """
    if ridge_w > 0 or wv is not None:
        w = wv if ridge_w == 0 else (1 + ridge_w * (tgt >= 0.9).to(tgt.dtype))
        if ridge_w > 0 and wv is not None:
            w = w * wv
        bce = (F.binary_cross_entropy_with_logits(logit, tgt, reduction="none") * w).sum() / w.sum().clamp_min(1e-6)
    else:
        bce = F.binary_cross_entropy_with_logits(logit, tgt)
    p, d = torch.sigmoid(logit), (0, 2, 3, 4)
    if wv is None:
        per = 1 - (2 * (p * tgt).sum(d) + 1) / (p.sum(d) + tgt.sum(d) + 1)   # per head
        return bce, per.mean()
    # the WEIGHTED soft dice: (2 sum(w p t) + 1) / (sum(w p) + sum(w t) + 1). Weighting p and t
    # separately (the old form) put w^2 in the intersection and w in the denominator, so a PERFECT
    # prediction under a fractional weight (the teacher agreement weight, pooled coarse weights) scored
    # a dice below 1 and was pushed to over-predict
    per = 1 - (2 * (wv * p * tgt).sum(d) + 1) / ((wv * p).sum(d) + (wv * tgt).sum(d) + 1)
    live = (wv.sum(d) > 0).to(per.dtype)          # the channels this batch says anything about
    return bce, (per * live).sum() / live.sum().clamp_min(1.0)


def deep_losses(logits, tgt, w=None, layout=None, ridge_w=0.0):
    """`losses_tw` summed over the decoder's multi-resolution outputs.

    Level k is scored against the target average-pooled 2^k times, with the per-voxel weights pooled
    alongside -- so the pooled target stays a probability and a voxel ignored at level 0 stays ignored
    above it. A single tensor (no deep supervision) is scored directly.

    With a `layout`, only the PROBABILITY heads are scored here: the distance heads have their own losses
    and the affinity heads their own BCE, and slicing by `layout.nprob` is how this function stays out of
    the business of knowing which head is which.
    """
    if layout is not None:
        n = layout.nprob
        tgt = tgt[:, :n]
        w = None if w is None else w[:, :n]
        logits = ([lg[:, :n] for lg in logits] if isinstance(logits, (list, tuple)) else logits[:, :n])
    if not isinstance(logits, (list, tuple)):
        return losses_tw(logits, tgt, w, ridge_w)
    bce = dice = 0.0
    for k, lg in enumerate(logits):
        tk = F.avg_pool3d(tgt, 2 ** k) if k else tgt
        wk = F.avg_pool3d(w, 2 ** k) if (k and w is not None) else w
        b, d = losses_tw(lg, tk, wk, ridge_w)
        bce, dice = bce + DEEP_W[k] * b, dice + DEEP_W[k] * d
    return bce, dice


# --------------------------------------------------------------------------- L3 soft exclusivity

def exclusivity(p, wv):
    """`relu(p_recto + p_verso - 1)` averaged over the voxels where BOTH channels carry weight.

    p (B, >=2, Z, Y, X) probabilities, wv (B, >=2, ...) the loader's per-voxel weights. The mask is
    `wv[:, 0] * wv[:, 1]`: only where a verso store actually covers the voxel is there anything to say
    about the pair. A sample with no verso store therefore contributes nothing (mask 0 everywhere)
    instead of pushing the untrained verso channel to 0.

    Zero for disjoint channels (p_r + p_v <= 1 everywhere) and zero-gradient where either weight is 0.
    """
    m = wv[:, 0] * wv[:, 1]
    return ((p[:, 0] + p[:, 1] - 1).clamp_min(0) * m).sum() / m.sum().clamp_min(1e-6)


# --------------------------------------------------------------------- L4 cascade self-consistency

def self_consistency(p_fine, cas, w=None, sel=None):
    """Mean |fine pooled 2x - coarse| at the COARSE resolution.

    `cas` is the CASCADE input channel this step already carries: the rung-(k+1) prediction over the same
    field of view, upsampled 2x. Pooling it back by 2 undoes the upsample up to the trilinear kernel's
    residual smoothing, which is the price of reusing the tensor instead of running a second coarse
    forward. The coarse side is detached; it comes from the EMA net under `no_grad` and has no graph.

    `sel` (B,) is 1 for the samples whose channel came from the SELF source and 0 for the rest: the
    `mask` source is the rung-(k+1) TARGET, so scoring against it would be a second, blurrier copy of the
    supervised loss rather than a consistency term, and a dropped channel is all zeros.
    """
    pf = F.avg_pool3d(p_fine[:, :1], 2)
    pc = F.avg_pool3d(cas[:, :1], 2).detach()
    ww = torch.ones_like(pf) if w is None else F.avg_pool3d(w[:, :1], 2)
    if sel is not None:
        ww = ww * sel.to(ww.dtype).view(-1, 1, 1, 1, 1)
    return ((pf - pc).abs() * ww).sum() / ww.sum().clamp_min(1e-6)


# ------------------------------------------------------------------------------ L8 skeleton recall

def erode(x, k=3):
    """One 26-connected erosion of a 0/1 field, with the OUTSIDE treated as background (explicit zero
    padding, not `max_pool3d`'s -inf, which would treat it as foreground and leave the band uneroded at
    the patch face)."""
    return -F.max_pool3d(F.pad(-x, (1,) * 6, value=0.0), k, stride=1)


def skeleton(tgt, iters=4, thr=0.5):
    """A one-voxel-thick medial surface of the binary target, on the GPU, in `iters + 1` pooling ops.

    `d = fg + sum_j erode^j(fg)` is the Chebyshev distance to background, capped at `iters` (the band is
    a few voxels thick, so a small cap is exact where it matters); the skeleton is the set of foreground
    voxels that are a local maximum of `d` in their 3x3x3 neighbourhood. For a band of thickness
    t <= 2*iters+1 that is exactly its medial surface, and it is a SURFACE, not a curve -- which is what
    the loss wants: a continuous sheet, not a centreline.
    """
    fg = (tgt >= thr).to(tgt.dtype)
    d, e = fg.clone(), fg
    for _ in range(int(iters)):
        e = erode(e)
        d = d + e
    return fg * (d >= F.max_pool3d(d, 3, stride=1, padding=1) - 1e-6).to(d.dtype)


def skel_recall(p, tgt, wv=None, iters=4, thr=0.5):
    """1 - (mean predicted probability along the target's skeleton), averaged over the channels that have
    any skeleton weight in the batch.

    The Skeleton Recall Loss (Kirchhoff/Isensee, ECCV 2024) with the hard skeleton computed here rather
    than offline: `iters + 1` pooling ops on a tensor the step already holds, cheaper than carrying a
    skeleton channel through the loader and the augmentation. Because it is a RECALL of a fixed point set
    it cannot be gamed by making the band wider, and because the point set is the TARGET's it costs no
    differentiable skeletonisation.

    It is a GAPS term only: a merge bridge is itself thin and connected and scores well here, so
    `merge_frac` must be watched while it is on.
    """
    s = skeleton(tgt, iters=iters, thr=thr)
    # a skeleton voxel is a function of the target within `iters + 1` voxels (iters 3^3 erosions and one
    # 3^3 max-pool), so it is only a LABEL where that whole neighbourhood is known (weight > 0): the
    # known mask is eroded by that radius, the patch outside counting as known (pass-3 P3-14)
    w = s if wv is None else s * wv * known_within(wv, int(iters) + 1)
    d = (0, 2, 3, 4)
    num, den = (p * w).sum(d), w.sum(d)
    live = (den > 0).to(p.dtype)
    rec = num / den.clamp_min(1e-6)
    return 1.0 - (rec * live).sum() / live.sum().clamp_min(1.0)


# ------------------------------------------------------------------- L8b skeleton precision

def dilate(x, k=3):
    """One 26-connected dilation (a 3^3 max-pool); outside the patch is background (zero padding)."""
    return F.max_pool3d(F.pad(x, (1,) * 6, value=0.0), k, stride=1)


def soft_skeleton(p, iters=3):
    """clDice's `soft_skel` (Shit et al., CVPR 2021) of a probability field, differentiable in `p`.

    Soft morphological thinning with min/max pooling: at each scale `j` the voxels an opening removes,
    `relu(e_j - dilate(erode(e_j)))` with `e_j = erode^j(p)`, join the skeleton through the soft union
    `s + relu(delta - s * delta)`. `open(e_j) = dilate(e_{j+1})` reuses the next scale's erosion, so the
    whole thing is `2 * (iters + 1)` 3^3 pooling ops (clDice's own form spends three per scale). The
    erosion is `erode`'s (outside = background). `iters` must reach about half the predicted band's
    thickness for the skeleton to be thin; the band is 3-9 voxels, so 3-4 is enough -- a thicker blob
    keeps a residual core, which still lies inside the blob and so costs no precision wherever the blob
    itself is inside the target band.
    """
    e = erode(p)
    s = F.relu(p - dilate(e))
    for _ in range(int(iters)):
        q, e = e, erode(e)
        d = F.relu(q - dilate(e))
        s = s + F.relu(d - s * d)
    return s


def skel_precision(p, tgt, wv=None, iters=3, thr=0.5, gate=0.3):
    """1 - soft-clDice PRECISION: the fraction of the PREDICTION's soft skeleton that lies inside the
    target band dilated by one voxel, averaged over the channels with any gated skeleton in the batch.

        prec = sum(S * B * m) / (sum(S * m) + eps),   S = soft_skeleton(p),   B = dilate(tgt >= thr),
        m    = wv * known_within(wv, 1) * [p >= gate]

    The complement of `skel_recall`: recall asks that the target's skeleton be covered (a GAP costs),
    precision asks that the prediction's skeleton be backed by the target (a SPUR, or a merge BRIDGE
    outside the band, costs). A prediction that STOPS SHORT keeps its skeleton inside the band and
    scores ~1 here: stopping short is recall's job, not this term's.

    The band is the target dilated by one voxel so a skeleton a voxel off the target's medial surface
    (a thin or tilted band) is not charged. `B` is a function of the target within one voxel, so a
    voxel counts only where that neighbourhood is known (`known_within(wv, 1)`, as `skel_recall` does
    for its own radius). The GATE `p >= gate` (a hard, detached mask) keeps early-training haze out: a
    diffuse low-probability field has a large, meaningless soft skeleton everywhere, and scoring it
    would make this a second, noisy background BCE. A spur above the gate is pushed down until it drops
    below it, after which the supervised BCE owns it. A channel with no gated skeleton (nothing
    confident predicted, or no weight) is not averaged in, and a batch with none scores 0.
    """
    s = soft_skeleton(p, iters=iters)
    band = dilate((tgt >= thr).to(p.dtype))
    m = (p.detach() >= gate).to(p.dtype)
    if wv is not None:
        m = m * wv * known_within(wv, 1)
    sm = s * m
    d = (0, 2, 3, 4)
    num, den = (sm * band).sum(d), sm.sum(d)
    live = (den > 1e-6).to(p.dtype)
    prec = num / (den + 1e-6)
    return ((1.0 - prec) * live).sum() / live.sum().clamp_min(1.0)


# ------------------------------------------------------------------------ O12 long-range affinity

def _offsets(offsets):
    """The offsets as a tuple of positive EVEN ints. They are a config field (`aff_offsets`), never a
    string: there is no spec to parse."""
    if offsets is None:
        return ()
    vs = tuple(int(v) for v in offsets)
    assert all(v > 0 and v % 2 == 0 for v in vs), f"affinity offsets must be positive and EVEN: {offsets}"
    return vs


def n_affinity(offsets):
    """Number of extra output channels: one per (axis, offset)."""
    return 3 * len(_offsets(offsets))


def affinity_names(offsets):
    return [f"aff{d}_{'zyx'[a]}" for d in _offsets(offsets) for a in range(3)]


def _shift(x, axis, n):
    """`x` moved by `n` voxels along `axis` (2, 3, 4), zero-filled where it comes from outside."""
    if n == 0:
        return x
    pad = [0] * 6
    i = 2 * (4 - axis)                       # (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    pad[i + (0 if n > 0 else 1)] = abs(n)
    y = F.pad(x, pad, value=0.0)
    return y.narrow(axis, 0 if n > 0 else abs(n), x.shape[axis])


def known_within(wv, r):
    """1 where every voxel within Chebyshev radius `r` has weight > 0 (a cubic erosion of the known
    mask), else 0. Outside the patch counts as known: the patch face is a convention of the crop, not
    an unknown label."""
    k = (wv > 0).to(wv.dtype)
    if r <= 0:
        return k
    return -F.max_pool3d(F.pad(-k, (int(r),) * 6, value=-1.0), 2 * int(r) + 1, stride=1)


def _pad_axis(axis, h):
    pad = [0] * 6
    i = 2 * (4 - axis)
    pad[i] = pad[i + 1] = h
    return pad


def _k_axis(axis, n):
    return tuple(n if 2 + j == axis else 1 for j in range(3))


def affinity_targets(tgt, wv=None, offsets=(8, 16, 32), thr=0.5, fg_only=True):
    """(target, weight) of the affinity channels, both (B, 3*len(offsets), Z, Y, X).

    Channel `3*i + a` of offset `d = offsets[i]` and axis `a` asks, AT THE MIDPOINT, whether the two
    voxels d/2 back and d/2 forward along that axis lie on the SAME sheet. Centring on the midpoint is
    what makes the channel set equivariant under the 48 cube symmetries: a flip maps the pair to itself
    and a permutation only permutes the three axis channels.

    Same-sheet test: the straight segment between the two voxels is entirely foreground, i.e.
        target = min over t in [-d/2, d/2] of fg(v + t e_a) = a centred 1D erosion of length d+1.
    This is connected-component labelling restricted to straight paths inside the patch, and it is
    CONSERVATIVE: a pair on one strongly curved sheet can read as "different", never the other way round,
    so the negatives -- the merge signal, "these two are the next wrap apart" -- are clean, which is the
    direction that matters. A real per-sample connected-component labelling would be seconds of CPU per
    256^3 patch and, precomputed in the loader, would not survive `aug.spatial`'s rotations; this form is
    three pooling ops on the augmented target.

    Weight: `w(v - d/2) * w(v + d/2)`, times zero unless EVERY voxel on the segment has weight > 0,
    zero where the pair leaves the patch (the shifts are zero-padded),
    and with `fg_only` (the default) zero unless BOTH voxels are foreground -- the same/different
    question is only meaningful for a pair of band voxels, and without the restriction the channel is
    >95% trivial "one end is air" zeros.

    `tgt` / `wv` are the RECTO channel (channel 0): the one channel every sample carries.
    """
    offs = _offsets(offsets)
    fg = (tgt[:, :1] >= thr).to(tgt.dtype)
    w0 = torch.ones_like(fg) if wv is None else wv[:, :1]
    kn = (w0 > 0).to(fg.dtype)
    ts, ws = [], []
    for d in offs:
        h = d // 2
        for a in (2, 3, 4):
            seg = -F.max_pool3d(F.pad(-fg, _pad_axis(a, h), value=0.0), _k_axis(a, d + 1), stride=1)
            # the target is a min over the WHOLE segment, so it is a label only where every voxel on the
            # segment is known -- not just the two ends (pass-3 P3-14)
            seg_known = -F.max_pool3d(F.pad(-kn, _pad_axis(a, h), value=0.0), _k_axis(a, d + 1),
                                      stride=1)
            lo, hi = _shift(fg, a, h), _shift(fg, a, -h)
            wl, wh = _shift(w0, a, h), _shift(w0, a, -h)
            ts.append(seg)
            ws.append(wl * wh * seg_known * (lo * hi if fg_only else 1.0))
    if not ts:
        z = tgt[:, :0]
        return z, z
    return torch.cat(ts, 1), torch.cat(ws, 1)


def affinity_loss(logit, tgt, wv=None, offsets=(8, 16, 32), thr=0.5, fg_only=True):
    """Weighted BCE of the affinity channels. `logit` is the affinity SLICE of the head's output."""
    at, aw = affinity_targets(tgt, wv, offsets, thr=thr, fg_only=fg_only)
    b = F.binary_cross_entropy_with_logits(logit, at, reduction="none")
    return (b * aw).sum() / aw.sum().clamp_min(1e-6)


# --------------------------------------------------------------------------------- the dispatcher

KEYS = ("excl", "selfcons", "skel", "skel_prec", "affinity")


def aux_losses(logit, tgt, wv, layout, w_excl=0.0, w_selfcons=0.0, w_skel=0.0, w_affinity=0.0,
               cascade=None, cascade_self=None, skel_iters=4, aff_fg_only=True, w_skel_prec=0.0,
               skel_prec_iters=3, skel_prec_gate=0.3, w_cont=None):
    """{name: unweighted loss} for every term whose weight is non-zero, plus `aux` = their weighted sum.

    `logit` is the FULL head output and `layout` says where each head lives: the probability heads are
    `[:layout.nprob]` and the affinity block is `[layout.i_aff : layout.i_aff + layout.n_aff]`. `tgt` /
    `wv` are the loader's target and weight (the `cout_t` field channels). `cascade` is the CASCADE input
    channel of this step and `cascade_self` (B,) the mask of samples that took the SELF source.
    `w_skel_prec` weighs the skeleton PRECISION term (`skel_precision`), on the same probability heads,
    the same level-0 output and so the same rungs as the skeleton recall.

    `w_cont` (default `wv`): the weight of the two CONTINUITY terms, self-consistency and skeleton
    recall. A routed rung-2 sample (`route_apply`) gives them the weight BEFORE its gap zones were
    zeroed: in a gap they are the supervision (with the skeleton of the base-teacher band there).
    """
    out, total, np_ = {}, None, layout.nprob
    wc = wv if w_cont is None else w_cont
    p = None
    if w_excl > 0 and np_ >= 2:
        p = torch.sigmoid(logit[:, :np_])
        out["excl"] = exclusivity(p, wv)
    if w_selfcons > 0 and cascade is not None:
        p = torch.sigmoid(logit[:, :np_]) if p is None else p
        out["selfcons"] = self_consistency(p, cascade, wc, cascade_self)
    if w_skel > 0:
        p = torch.sigmoid(logit[:, :np_]) if p is None else p
        out["skel"] = skel_recall(p, tgt[:, :np_], wc[:, :np_] if wc is not None else None, iters=skel_iters)
    if w_skel_prec > 0:
        p = torch.sigmoid(logit[:, :np_]) if p is None else p
        out["skel_prec"] = skel_precision(p, tgt[:, :np_], wv[:, :np_] if wv is not None else None,
                                          iters=skel_prec_iters, gate=skel_prec_gate)
    if w_affinity > 0 and layout.n_aff:
        out["affinity"] = affinity_loss(logit[:, layout.i_aff:layout.i_aff + layout.n_aff], tgt, wv,
                                        offsets=layout.aff_offsets, fg_only=aff_fg_only)
    for k, v in (("excl", w_excl), ("selfcons", w_selfcons), ("skel", w_skel), ("skel_prec", w_skel_prec),
                 ("affinity", w_affinity)):
        if k in out:
            total = v * out[k] if total is None else total + v * out[k]
    if total is not None:
        out["aux"] = total
    return out


# ============================================================== per-rung teacher routing, gap-fill

ROUTE_ON, ROUTE_CA = 0.25, 0.75   # the routing row's weight: > ON = a routed voxel, > CA = c_A = 1
                                  # (codes 128 / 255 from `sample._band_row`, thresholds robust to the
                                  # augmentation's interpolation of the weight)


def route_masks(tb, wb, dilate=2, thr=0.5):
    """The per-voxel routing of a batch from its ROUTING row (target `tb` = the base (m7) teacher's
    probability, weight `wb`; both (B, 1, Z, Y, X) in 0..1, `sample._band_row`). Float 0/1 masks:

    routed   the voxel has a routed target (a routed region's fine rung, CT > 0, inside the store)
    ca       ... and the fine teacher is trusted there (c_A = 1): BCE + dice vs the fine teacher, as today
    band     the base teacher's band (tb >= thr) dilated by `dilate` voxels (a Chebyshev cube)
    gap      routed, c_A = 0, inside `band`: NO direct BCE / dice -- the cascade self-consistency, the
             skeleton recall against the band's skeleton and the distance heads carry it
    outside  routed, c_A = 0, outside `band`: BCE + dice vs the base teacher's probability (the air
             suppression of the m7-only targets, unchanged) and the band penalty (`band_penalty`)
    """
    routed = (wb > ROUTE_ON).to(wb.dtype)
    ca = (wb > ROUTE_CA).to(wb.dtype)
    band = (tb >= thr).to(tb.dtype)
    r = int(dilate)
    if r > 0:
        band = F.max_pool3d(band, 2 * r + 1, stride=1, padding=r)
    nca = routed * (1 - ca)
    return {"routed": routed, "ca": ca, "band": band, "gap": nca * band, "outside": nca * (1 - band)}


def route_apply(tgt, wv, tb, m):
    """(tgt, wv, w_cont): the batch's recto target / weight with the routing applied, and the weight the
    continuity terms keep (`aux_losses(w_cont=...)`). The recto row's target is the fine teacher's
    probability where c_A = 1 (and on every unrouted voxel), the base teacher's elsewhere; its weight is
    zeroed in the gap zones. Every other row, and every unrouted voxel, is left exactly as it was."""
    nca = m["routed"] * (1 - m["ca"])
    t0 = tgt[:, :1] * (1 - nca) + tb * nca
    w0 = wv[:, :1] * (1 - m["gap"])
    return torch.cat([t0, tgt[:, 1:]], 1), torch.cat([w0, wv[:, 1:]], 1), wv


def band_penalty(p, outside, w=None, eps=0.05):
    """sum(relu(p - eps)^2 * m) / sum(m) with m = `outside` (x the loader weight `w`): nothing is
    predicted outside the dilated base-teacher band where the fine teacher is not confident. 0 (with no
    gradient) for a batch without such voxels."""
    m = outside if w is None else outside * w
    return ((p - float(eps)).clamp_min(0) ** 2 * m).sum() / m.sum().clamp_min(1e-6)


# ================================================================ the thinned-band target (rung 2)

THIN_MEDIAL_CAP = 16   # `targets.medial_torch`'s capped EDT (exact: it re-runs uncapped past it)


def thin_band(t, width=4.0, soft=1.0, thr=0.5):
    """(t', keep): the THINNED version of a wide soft band target `t` (B, 1, Z, Y, X), computed on the
    target's device from nothing but `t` (nothing is cached between steps).

    B = t >= thr, M = its medial surface (`targets.medial_torch`: exact Euclidean EDT, the same code
    path as the evaluation's `dice_recto_r*_thin`), d = the exact Euclidean distance to M (`edt.edt2`,
    capped at the sheet's reach), and inside B

        t' = clamp(1 - (d - width/2 + soft) / soft, 0, 1)

    i.e. 1 within width/2 - soft of M, fading linearly to 0 at width/2; t' = 0 outside B. `keep` is the
    weight multiplier: 0 inside B beyond the thinned sheet (t' == 0 there: the band's coarse flanks are
    not punished either way), 1 on the sheet and everywhere outside B (background supervision as
    before). A sample with an empty band has t' = 0 and keep = 1."""
    from rvsm import edt as E
    from rvsm.targets import medial_torch
    half, soft = float(width) / 2.0, max(float(soft), 1e-6)
    b = t[:, 0] >= float(thr)
    m = medial_torch(b, cap=THIN_MEDIAL_CAP)
    d2, _ = E.edt2(m, indices=False, cap=math.ceil(half) + 1)
    d = torch.sqrt(d2)                                   # +inf beyond the cap and for an empty M
    bf = b.to(t.dtype)
    tt = (1.0 - (d - half + soft) / soft).clamp(0.0, 1.0).nan_to_num(0.0).to(t.dtype) * bf
    keep = 1.0 - bf * (tt <= 0).to(t.dtype)
    return tt.unsqueeze(1), keep.unsqueeze(1)


def thin_apply(tgt, wv, sel, width=4.0, soft=1.0, orig=None, tb=None, m=None):
    """(tgt, wv) with the recto row's target / weight THINNED (`thin_band`) on the samples `sel` (a
    list of batch indices: the rung-2 samples). Other rows and other samples are untouched.

    Without routing (`m` None): the recto row becomes (t', w * keep), t' thinned from the recto target
    itself. With routing (the `route_masks` `m`, the routing row's probability `tb`, and `orig` = the
    recto row's (target, weight) BEFORE `route_apply`): the GAP voxels (c_A = 0 inside the dilated base
    band) get (thin(tb), w_orig * keep) as a DIRECT target where route_apply left none; UNROUTED voxels
    (a region the regeneration has not reached: its rung-2 recto is still the base teacher's) get the
    standalone thinning of their own recto; c_A = 1 and `outside` voxels keep what route_apply gave."""
    if not sel:
        return tgt, wv
    idx = torch.as_tensor(sel, device=tgt.device)
    t0, w0 = tgt[idx, :1], wv[idx, :1]
    if m is None:
        tt, kp = thin_band(t0, width, soft)
        t1, w1 = tt, w0 * kp
    else:
        to, wo = orig[0][idx], orig[1][idx]
        gap, unr = m["gap"][idx], 1.0 - m["routed"][idx]
        tt, kp = thin_band(tb[idx], width, soft)
        t1, w1 = t0 * (1 - gap) + tt * gap, w0 * (1 - gap) + wo * kp * gap
        if bool(unr.any()):
            tr, kr = thin_band(to, width, soft)
            t1, w1 = t1 * (1 - unr) + tr * unr, w1 * (1 - unr) + wo * kr * unr
    tgt, wv = tgt.clone(), wv.clone()
    tgt[idx, :1], wv[idx, :1] = t1, w1
    return tgt, wv


# ================================================ distance regression, Eikonal, normals, pairing, ECT

UNIT = 0.25    # voxels per uint8 code step of a distance store (`rvsm.targets`)
OFF = 128.0    # the code of distance 0
CAP = 31.75    # the representable range, +- this many voxels
TMIN = 3.0     # the floor on a sheet thickness, in voxels (see `pair_bands`)
WMIN = 0.95    # a distance voxel counts only at (almost) full weight: see `dist_weight`


def decode_signed(u):
    """A loader target channel (0..1, i.e. code / 255) -> voxels."""
    return (u * 255.0 - OFF) * UNIT


def decode_unsigned(u):
    return u * 255.0 * UNIT


def encode_signed(d):
    """`decode_signed`'s inverse, unquantised: voxels -> the 0..1 channel, clamped to +-CAP."""
    return (d.clamp(-CAP, CAP) / UNIT + OFF) / 255.0


def encode_unsigned(t):
    """`decode_unsigned`'s inverse, unquantised: voxels -> the 0..1 channel."""
    return (t / UNIT).clamp(0.0, 255.0) / 255.0


# ------------------------------------------------------------------------ overlap-crop consistency

OVERLAP_DIST_SCALE = 0.1   # the distance L1 (voxels) enters the term at this weight: 1 voxel ~ 0.1 nat


def overlap_loss(logit, field, w_mid=None, w_thick=None, layout=None):
    """The overlap-crop consistency term (`rvsm.overlap`): the student's heads on the first window
    against the EMA's on the same voxels seen from the second window, where the EMA sees more.

    `field` (B, nprob + 3, Z, Y, X) is `overlap.teacher_batch`'s, after the augmentation: the teacher
    probabilities, its midline / thickness in the store's 0..1 code, and the mask (gated at `WMIN`, as a
    distance weight is: a warp interpolates it at its edge, and a voxel half inside is not inside).

      probabilities   one-way binary KL(teacher || student) = BCE(student logit, teacher p) - H(teacher p),
                      so the term is 0 exactly when they agree and its gradient is the soft-target BCE's;
                      averaged over the probability heads (recto, verso)
      distances       L1 in voxels of the midline and of the thickness (after `soft_thickness`), only
                      where the first window's DISTANCE weight is > 0 (`w_mid`, `w_thick`, already
                      `dist_weight`-gated): the distance heads are unconstrained elsewhere, and the
                      teacher's values there are extrapolation. L1 rather than a KL: they are regressions,
                      and L1 is the robust choice against a teacher that is itself wrong in places.
                      Scaled by `OVERLAP_DIST_SCALE`.

    The teacher side carries no gradient (it is a tensor of values). Returns {overlap, overlap_kl,
    overlap_l1, overlap_vox}: the term, its parts, and the scored share of the voxels."""
    np_ = layout.nprob
    f = field.float()
    m = (f[:, np_ + 2:np_ + 3] >= WMIN).to(logit.dtype)
    den = m.sum().clamp_min(1.0)
    pt = f[:, :np_].clamp(1e-6, 1 - 1e-6)
    z = logit[:, :np_].float()
    ent = -(pt * pt.log() + (1 - pt) * (1 - pt).log())
    kl = ((F.binary_cross_entropy_with_logits(z, pt, reduction="none") - ent) * m).sum() / (den * np_)
    l1 = logit.new_zeros(())
    for w_, pred, tgt in ((w_mid, logit[:, layout.i_mid:layout.i_mid + 1].float(),
                           decode_signed(f[:, np_:np_ + 1])),
                          (w_thick, soft_thickness(logit[:, layout.i_thick:layout.i_thick + 1].float()),
                           decode_unsigned(f[:, np_ + 1:np_ + 2]))):
        if w_ is None:
            continue
        ww = m * (w_ > 0).to(m.dtype)
        l1 = l1 + ((pred - tgt).abs() * ww).sum() / ww.sum().clamp_min(1.0)
    tot = kl + OVERLAP_DIST_SCALE * l1
    return {"overlap": tot, "overlap_kl": kl, "overlap_l1": l1, "overlap_vox": m.mean()}


def dist_weight(w):
    """The per-voxel weight of a DISTANCE channel: the loader's weight, hard-gated at `WMIN`.

    The spatial augmentations resample the target and the weight together, so a voxel on the boundary
    between a real distance and a no-data code comes out of `aug.warp` with a fractional weight and an
    INTERPOLATED value -- and interpolating across the no-data code 0 (which decodes to -32 voxels, not
    to "nothing") gives a number that is simply wrong. For a probability channel that is a harmless soft
    edge; for a distance channel it is a bogus target, so a partially-weighted distance voxel is dropped
    instead of down-weighted. Inside a block of valid data the weight is exactly 1 and nothing is lost."""
    return w * (w >= WMIN).to(w.dtype)


def grad3(d):
    """Central differences of (B,1,Z,Y,X) along z, y, x -> (B,3,Z,Y,X), in units of "per voxel".

    The field is replicate-padded, so the outermost voxel of each face gets a ONE-SIDED difference, which
    is half the true slope of a linear field. Every caller therefore drops a one-voxel border from the
    weight (`border_mask`) rather than pretending the face is interior."""
    g = []
    for a in (2, 3, 4):
        pad = [0] * 6
        i = 2 * (4 - a)
        pad[i] = pad[i + 1] = 1
        p = F.pad(d, pad, mode="replicate")
        n = p.shape[a]
        g.append(0.5 * (p.narrow(a, 2, n - 2) - p.narrow(a, 0, n - 2)))
    return torch.cat(g, 1)


def border_mask(x, n=1):
    """1 in the interior, 0 in the outer `n` voxels of every face; shaped to broadcast over `x`."""
    m = torch.ones_like(x[:, :1])
    m[..., :n, :, :] = m[..., -n:, :, :] = 0
    m[..., :, :n, :] = m[..., :, -n:, :] = 0
    m[..., :, :, :n] = m[..., :, :, -n:] = 0
    return m


def normals_from(d, eps=1e-4):
    """Unit normals from a predicted signed distance field: `n = grad(d) / |grad(d)|`, (B,3,Z,Y,X).

    The sign follows from the target's: d grows towards the RECTO side (radially outward), so grad(d)
    points from verso to recto and `dot(n, radial) > 0`, which is the tracer contract's convention. No
    extra output channels and no extra parameters.

    This is a finite difference of the PREDICTED FIELD's values, not a normal read off the decoder's own
    autograd gradient (which for a ReLU/SiLU conv decoder is piecewise constant and noisy) -- it is the
    same quantity the export computes from the stored field, computed the same way."""
    g = grad3(d)
    return g / g.norm(dim=1, keepdim=True).clamp_min(eps)


def sdist_loss(pred, tgt, w, delta=2.0, cap=CAP, logvar=None):
    """Clamped Huber of a predicted distance (in VOXELS) against the decoded target, on weight > 0.

    `pred` (B,1,Z,Y,X) is the head's raw output -- the head predicts voxels directly, so a zero-initialised
    row (the warm start's policy for a regression channel) says "the surface is here", which is the
    encoding's own zero (code 128). `tgt` is the loader's 0..1 channel, decoded and clamped to the store's
    range, so a target the encoder saturated cannot pull the prediction past the cap.

    `logvar`, when given, is the heteroscedastic log-variance head:

        L = exp(-s) * huber(pred, tgt) + 0.5 * s

    the Gaussian-likelihood form with the residual replaced by the Huber (Kendall & Gal 2017). The model
    may raise s where it cannot localise the surface -- which is exactly the `conf` channel the tracer
    contract asks for, obtained free rather than as a separate head."""
    t = decode_signed(tgt).clamp(-cap, cap)
    h = F.huber_loss(pred, t, reduction="none", delta=float(delta))
    if logvar is not None:
        s = logvar.clamp(-8.0, 8.0)
        h = torch.exp(-s) * h + 0.5 * s
    return (h * w).sum() / w.sum().clamp_min(1e-6)


def thickness_loss(pred, tgt, w, delta=2.0):
    """Huber of a predicted thickness (voxels, already positive: see `soft_thickness`) against the store."""
    t = decode_unsigned(tgt)
    return (F.huber_loss(pred, t, reduction="none", delta=float(delta)) * w).sum() / w.sum().clamp_min(1e-6)


def eikonal(d, w, band=8.0, tgt=None):
    """`mean (|grad d| - 1)^2` over the BAND: the voxels the field says are within `band` of a surface.

    IGR (Gropp et al., ICML 2020): the Eikonal residual is what makes a regressed field an actual distance
    function between the samples that pin it down. It carries NO localisation on its own -- the Huber term
    does that -- so it is a regulariser, evaluated where the field matters: near the zero level set. The
    band is taken from the TARGET when there is one (so the term cannot be satisfied by pushing the
    predicted surface out of the patch) and from the prediction otherwise. The one-voxel border is
    dropped: `grad3` has a one-sided difference there."""
    g = grad3(d).norm(dim=1, keepdim=True)
    ref = d if tgt is None else decode_signed(tgt)
    m = w * (ref.abs() <= float(band)).to(w.dtype) * border_mask(d)
    return (((g - 1.0) ** 2) * m).sum() / m.sum().clamp_min(1e-6)


def normal_head_loss(nh, d, w, band=8.0, tgt=None):
    """Explicit normal channels pulled onto the normalised gradient of the predicted distance field
    (detached), plus a unit-norm term.

    There is no normal TARGET in any store -- a normal is a derived quantity, and the contract derives it
    from the stored distance field. So the head is a distillation of `normals_from` into three channels
    that inference can read without a finite difference. rvsm's default layout has no normal head and
    derives them; this exists for a run that wants one."""
    ref = normals_from(d).detach()
    m = w * border_mask(d)
    if tgt is not None:
        m = m * (decode_signed(tgt).abs() <= float(band)).to(m.dtype)
    e = ((nh - ref) ** 2).sum(1, keepdim=True) + (nh.norm(dim=1, keepdim=True) - 1.0) ** 2
    return (e * m).sum() / m.sum().clamp_min(1e-6)


# --------------------------------------------------------------------- construction-based pairing

def soft_thickness(raw, tmin=TMIN):
    """`t = tmin + softplus(raw)`: a predicted thickness that CANNOT go below the minimum physical sheet
    thickness, whatever the head says. A zero-initialised head row gives `tmin + ln 2`."""
    return float(tmin) + F.softplus(raw)


def band_fn(u, half=1.5, tau=0.5):
    """The soft band: `sigmoid((half - |u|) / tau)`, 0.5 exactly at |u| = half. One function, used for
    both faces, so the pair is symmetric by construction."""
    return torch.sigmoid((float(half) - u.abs()) / float(tau))


def pair_logits(m, t, half=1.5, tau=0.5):
    """The LOGITS of `pair_bands`: `(half - |m -+ t/2|) / tau`, which is what the BCE wants.

    `band_fn` is a sigmoid, so its logit is its argument -- no `log(p / (1 - p))` round trip, no
    saturation, and the BCE of the constructed pair is as well behaved as the learned one's."""
    h = 0.5 * t
    return (float(half) - (m - h).abs()) / float(tau), (float(half) - (m + h).abs()) / float(tau)


def pair_bands(m, t, half=1.5, tau=0.5):
    """(p_recto, p_verso) built from a midline distance `m` and a thickness `t`, as bands at m = +- t/2.

        p_recto = band(m - t/2)        p_verso = band(m + t/2)

    with `band(u) = sigmoid((half - |u|) / tau)`. The recto face sits at m = +t/2 because m is signed
    POSITIVE on the recto (radially outward) side, which is the stores' convention and the radial
    channel's.

    These two CANNOT overlap when `t >= 2 * half`, exactly, not approximately. For 0 <= m <= t/2 the two
    arguments are a1 = (half - t/2 + m)/tau and a2 = (half - t/2 - m)/tau, so a2 <= -a1 whenever
    t >= 2*half, and sigmoid(a1) + sigmoid(a2) <= sigmoid(a1) + sigmoid(-a1) = 1. For |m| > t/2 (>= half)
    the two arguments sum to 2(half - |m|)/tau <= 0 and the same bound applies; m < 0 is the mirror image.
    So `relu(p_r + p_v - 1)` -- the exclusivity loss -- is identically ZERO for every (m, t) with
    t >= 2*half, which is what construction-based pairing buys and what `soft_thickness(tmin=2*half)`
    guarantees. `exclusivity` stays as a backstop for the LEARNED channels."""
    a, b = pair_logits(m, t, half, tau)
    return torch.sigmoid(a), torch.sigmoid(b)


# ------------------------------------------------------------------------------ the topology term

def fib_dirs(n, device=None, dtype=torch.float32):
    """`n` roughly-uniform directions on the sphere (the Fibonacci spiral), (n,3) in ZYX order.
    Deterministic, so a resume replays the same loss to the last float."""
    i = torch.arange(n, device=device, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = (1.0 - z * z).clamp_min(0.0).sqrt()
    a = i * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([z, r * torch.cos(a), r * torch.sin(a)], 1).to(dtype)


def chi_cells(p):
    """The eight cell families of the cubical complex of a (B,1,Z,Y,X) soft occupancy field.

    A cubical complex on a voxel grid has one VERTEX per voxel, an EDGE for each pair of neighbouring
    voxels along an axis, a FACE for each 2x2 square in a coordinate plane and a CUBE for each 2x2x2
    block. For a BINARY field a cell is present iff all its vertices are, i.e. it is the PRODUCT of them;
    for a soft field that same product is the probability the cell is present under independent voxels,
    so `V - E + F - C` over the products is the EXPECTED Euler characteristic -- and it is
    differentiable, which is the whole point.

    Returns [(sign, cell tensor, spanned axes), ...]; a cell tensor is indexed by its LOWEST corner and
    `spanned` says along which axes it has a second corner (which is what fixes its filtration height).
    """
    v = p
    ez = v[:, :, :-1] * v[:, :, 1:]
    ey = v[:, :, :, :-1] * v[:, :, :, 1:]
    ex = v[..., :-1] * v[..., 1:]
    fzy = ez[:, :, :, :-1] * ez[:, :, :, 1:]
    fzx = ez[..., :-1] * ez[..., 1:]
    fyx = ey[..., :-1] * ey[..., 1:]
    cu = fzy[..., :-1] * fzy[..., 1:]
    return [(1.0, v, ()), (-1.0, ez, (0,)), (-1.0, ey, (1,)), (-1.0, ex, (2,)),
            (1.0, fzy, (0, 1)), (1.0, fzx, (0, 2)), (1.0, fyx, (1, 2)), (-1.0, cu, (0, 1, 2))]


def ect(p, dirs, res=16):
    """The Euler Characteristic Transform of a (B,1,Z,Y,X) soft field: (B, n_dirs, res).

    For a direction `xi` and a height `h`, the entry is the expected Euler characteristic of the part of
    the complex whose vertices all satisfy `<v, xi> <= h` -- the classic ECT sweep. A cell enters at the
    height of its HIGHEST corner, and because `<v, xi>` is LINEAR that height is the lowest corner's plus
    `sum over the spanned axes of max(0, xi_axis)`: no search, one add. So the whole transform is, per
    direction, eight `index_add_`s with FIXED (geometry-only, gradient-free) bin indices and the
    differentiable cell products as values, followed by a cumulative sum over the height axis.

    This is the "fast chi" family (arXiv:2507.23763): no persistence diagram, no matching, no C++
    dependency."""
    B, _, Z, Y, X = p.shape
    dev = p.device
    D = int(dirs.shape[0])
    g = [torch.arange(n, device=dev, dtype=torch.float32) for n in (Z, Y, X)]
    out = p.new_zeros((B, D, res))
    cells = chi_cells(p)
    for j in range(D):
        xi = dirs[j].to(torch.float32)
        h = (xi[0] * g[0])[:, None, None] + (xi[1] * g[1])[None, :, None] + (xi[2] * g[2])[None, None, :]
        lo = float(h.min()) + min(float(xi.clamp_min(0).sum()), 0.0)
        hi = float(h.max()) + float(xi.clamp_min(0).sum())
        sc = (res - 1) / max(hi - lo, 1e-6)
        for sg, cell, spanned in cells:
            nz, ny, nx = cell.shape[2], cell.shape[3], cell.shape[4]
            top = float(sum(max(float(xi[a]), 0.0) for a in spanned))
            b = ((h[:nz, :ny, :nx] + top - lo) * sc).round().clamp(0, res - 1).to(torch.long).reshape(-1)
            out[:, j].index_add_(1, b, (sg * cell).reshape(B, -1).to(out.dtype))
    return out.cumsum(-1)


def ect_loss(p, tgt, dirs=1, res=16, margin=8, block=32, nblocks=4, thr=0.5, w=None, gen=None,
             alt=None, support=None):
    """Mean squared difference between the ECT of the predicted probability and of the target, over
    interior sub-blocks.

    The crop is the single most important detail: a topology loss computed on a cropped patch sees every
    sheet truncated at the patch face and produces a spurious gradient at every crop edge, which is the
    known failure mode of the whole persistent-homology family. So the loss is computed only on
    sub-blocks of side `block` from the interior of the patch, at least `margin` voxels from every face.

    `dirs` is the number of ECT directions (`fib_dirs`). Per SAMPLE, `nblocks` blocks are drawn at
    random from the interior grid with `gen` (a seeded torch.Generator: a resume replays the same draw;
    without one the blocks are the first of the grid, the old fixed choice). With `w` (the target
    weight, (B, 1+, Z, Y, X)) only a FULLY observed block (weight > 0 at every voxel) is a candidate,
    and a sample with no such block does not score. The value is 0 when nothing scores.

    `alt` / `support` (pass-4 P4-05): `p` is then the band CONSTRUCTED from midline / thickness and
    `alt` the learned recto head; a chosen block reads `p` only when `support` (the paired support,
    (B, 1, Z, Y, X)) is non-zero at EVERY voxel of it, and `alt` otherwise -- per block, not per
    sample, so a corner of paired support cannot route a whole block's gradient into the fields.

    The transform is normalised by the number of vertices of a sub-block, so the value does not depend on
    `block`; it is 0 for identical inputs and finite for any input."""
    B, _, Z, Y, X = p.shape
    lo = [margin] * 3
    hi = [Z - margin, Y - margin, X - margin]
    if any(hi[i] - lo[i] < block for i in range(3)):
        return p.sum() * 0.0
    starts = []
    for i in range(3):
        n = max((hi[i] - lo[i]) // block, 1)
        starts.append([lo[i] + j * (hi[i] - lo[i] - block) // max(n - 1, 1) for j in range(n)])
    grid = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]
    nb = max(int(nblocks), 1)
    d = fib_dirs(int(dirs), device=p.device)
    t = (tgt[:, :1] >= thr).to(p.dtype)
    pbs, tbs = [], []
    for s in range(B):
        cand = grid
        if w is not None:
            # FULLY observed blocks only: a block with any zero-weight voxel has unknown target there,
            # and the Euler characteristic of a partly unknown block is not a target (pass-3 P3-12)
            ws = w[s, :1]
            cand = [g for g in grid
                    if bool((ws[:, g[0]:g[0] + block, g[1]:g[1] + block, g[2]:g[2] + block] > 0).all())]
        if not cand:
            continue
        if gen is not None:
            pick = torch.randperm(len(cand), generator=gen)[:nb].tolist()
            chosen = [cand[j] for j in pick]
        else:
            chosen = cand[:nb]
        for (z, y, x) in chosen:
            src = p
            if alt is not None:
                full = support is not None and bool(
                    (support[s:s + 1, :1, z:z + block, y:y + block, x:x + block] > 0).all())
                src = p if full else alt
            pbs.append(src[s:s + 1, :1, z:z + block, y:y + block, x:x + block])
            tbs.append(t[s:s + 1, :1, z:z + block, y:y + block, x:x + block])
    if not pbs:
        return p.sum() * 0.0
    # every chosen block of every sample is stacked into the BATCH dimension, so the whole term is two
    # `ect` calls: the transform is 8 products and 8 index_adds per direction and the kernel launches,
    # not the arithmetic, are what it costs.
    pb, tb = torch.cat(pbs), torch.cat(tbs)
    n = float(block ** 3)
    with torch.no_grad():
        b = ect(tb, d, res) / n
    return ((ect(pb, d, res) / n - b) ** 2).mean()
