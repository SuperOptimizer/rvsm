"""OVERLAP-CROP CONSISTENCY: the EMA's view of a second, shifted window, pasted into the first one's frame.

No other loss compares two crops of the same rung, yet the student's receptive field (~450 voxels) is
larger than its 256 patch: near a patch face it predicts a sheet from half its usual context, and
nothing tells it that the same voxel, seen from a window centred elsewhere, looks different. An overlap
draw (`sample.Patches._overlap`, probability `cfg.overlap_p`) carries a SECOND window of the same visit,
shifted by p/4..p/2 along one or more axes. Here the EMA net runs on it (no grad, bf16, eval mode; on the
whole window or on a `cfg.overlap_sub`^3 sub-crop of it) and its heads are written into a field on the
FIRST window's voxels, which `losses.overlap_loss` scores the student against.

Only the voxels where the EMA's window sees MORE context than the student's are kept: the distance to
the nearest face of the EMA window (in its own voxels) must exceed the distance to the nearest face of
the student's window. A one-way, stop-gradient term (as `losses.self_consistency`) trained where the
student sees more would teach it the teacher's truncation; kept where the teacher sees more, it is a
target the student can only reach by using its context -- near the face the second window extends
past. On an axis that is not shifted the two margins are equal, so those voxels are not scored.

The field is built in the PHYSICAL frame, then given the first window's cube symmetry, and rides
through the augmentation as extra target channels -- so a rotation or an elastic warp moves it exactly
as it moves the targets. The EMA sees the second window clean: no symmetry, no augmentation, its own
per-patch z-score, and a cascade channel of the same SOURCE as the first window's (zero when that was
dropped / off; the same coarse self pass, sliced at the shift, when it was `self`). A first window whose
cascade came from the `mask` source (the `mix` mode) gets no term: the coarse target block covers only
its own footprint.

Field channels: [p_0 .. p_{nprob-1}, midline, thickness, mask] -- the probabilities as they are, the two
distances in the store's 0..1 code (`losses.encode_signed` / `encode_unsigned`: the augmentation clamps
its target channels to 0..1), and the 0/1 "teacher sees more" mask.
"""
from __future__ import annotations

import numpy as np
import torch

from rvsm import ladder, losses as L, model as M, prep


def sub_window(p, shift, q=0):
    """(o, n): the corner and size, in the SECOND window's own voxels, of the part of it the EMA runs on.

    `q` 0 (or >= p): the whole window. Otherwise a q^3 sub-crop placed where it helps most: on a shifted
    axis centred on the first window's face that the second one extends past (at p - s in the second
    window's voxels for s > 0, at -s for s < 0), on an unshifted axis centred; clamped inside."""
    p = ladder.shape3(p).astype(np.int64)
    s = np.asarray(shift, np.int64)
    q = int(q or 0)
    if q <= 0 or (q >= p).all():
        return np.zeros(3, np.int64), p.copy()
    n = np.minimum(np.full(3, q, np.int64), p)
    c = np.where(s > 0, p - s, np.where(s < 0, -s, p // 2))
    return np.clip(c - n // 2, 0, p - n), n


def paste_box(p, shift, o, n, device="cpu"):
    """Where the EMA window (corner `o`, size `n` in the second window's voxels; the second window is the
    first shifted by `shift`) lands in the FIRST window, and which of those voxels it sees more of.

    Returns None (no overlap) or (a, b, ta, tb, mask): the first window's box [a, b), the same voxels'
    box [ta, tb) in the EMA output, and the (b - a) bool tensor (on `device`: built there from three 1-D
    ranges, never on the host at 256^3) of "teacher margin > student margin", a margin being the
    distance in voxels, min over the three axes, to the nearest face of the window."""
    p = ladder.shape3(p).astype(np.int64)
    t0 = np.asarray(shift, np.int64) + np.asarray(o, np.int64)       # EMA window corner, first-window voxels
    n = np.asarray(n, np.int64)
    a, b = np.maximum(t0, 0), np.minimum(t0 + n, p)
    if (b <= a).any():
        return None
    m1 = m2 = None
    for j in range(3):
        v = torch.arange(int(a[j]), int(b[j]), device=device)
        u = v - int(t0[j])
        sh = [1, 1, 1]
        sh[j] = -1
        e1 = torch.minimum(v, int(p[j]) - 1 - v).view(sh)
        e2 = torch.minimum(u, int(n[j]) - 1 - u).view(sh)
        m1 = e1 if m1 is None else torch.minimum(m1, e1)
        m2 = e2 if m2 is None else torch.minimum(m2, e2)
    return a, b, a - t0, b - t0, (m2 > m1).expand(tuple(int(q) for q in b - a))


def field_channels(layout):
    """How many channels the pasted field has: the probabilities, midline, thickness, the mask."""
    return layout.nprob + 3


def teacher_field(b, i, fwd, layout, dev, cas=None, q=0):
    """Sample i's field (1, nprob + 3, Z, Y, X) in the first window's PHYSICAL frame (no symmetry yet),
    or None when the sample has no usable second window (`module docstring`). `fwd` is the EMA forward
    (eval mode); `cas` the step's training `Cascade` (None / off: a zero cascade channel)."""
    S = tuple(int(v) for v in b["ct"].shape[2:])
    shift = (b["ov_lo"][i].to("cpu", torch.int64) - b["lo"][i].to("cpu", torch.int64)).numpy()
    casch = None
    if cas is not None and cas.on:
        src = cas.last_src[i] if i < len(cas.last_src) else "off"
        if src == "mask":
            return None
        casch = cas.shifted(i, shift, S)
    o, n = sub_window(S, shift, q)
    box = paste_box(S, shift, o, n, device=dev)
    if box is None:
        return None
    a, bb, ta, tb, mask = box
    x = prep.overlap_input(b, i, dev, cas=casch)
    if (n < np.array(S)).any():
        x = x[:, :, o[0]:o[0] + n[0], o[1]:o[1] + n[1], o[2]:o[2] + n[2]]
    with torch.no_grad(), prep.autocast(dev):
        y = fwd(x.contiguous(memory_format=M.memfmt()))
    del x
    y = y[0] if isinstance(y, (list, tuple)) else y
    y = y[:, :layout.cout_t, ta[0]:tb[0], ta[1]:tb[1], ta[2]:tb[2]].float()
    E = torch.zeros((1, field_channels(layout)) + S, device=dev)
    sl = (slice(None), slice(None), slice(a[0], bb[0]), slice(a[1], bb[1]), slice(a[2], bb[2]))
    np_ = layout.nprob
    with torch.no_grad():
        E[sl][:, :np_] = torch.sigmoid(y[:, :np_])
        E[sl][:, np_] = L.encode_signed(y[:, layout.i_mid])
        E[sl][:, np_ + 1] = L.encode_unsigned(L.soft_thickness(y[:, layout.i_thick]))
        E[sl][:, np_ + 2] = mask.to(E.dtype)
    return E


def teacher_batch(b, fwd, layout, dev, cas=None, q=0):
    """The batch's fields (B, nprob + 3, Z, Y, X), each in its own sample's cube symmetry (the frame
    `prep.prepare` left the first window in), or None when no sample has one. A sample without a usable
    second window is all zeros: its mask is 0 and the term ignores it."""
    B = int(b["ct"].shape[0])
    sym = [int(v) for v in b["sym"].reshape(-1).tolist()]
    out, any_ = [], False
    for i in range(B):
        E = teacher_field(b, i, fwd, layout, dev, cas=cas, q=q)
        if E is None:
            E = torch.zeros((1, field_channels(layout)) + tuple(b["ct"].shape[2:]), device=dev)
        else:
            any_ = True
            if sym[i]:
                E = prep.sym_field_t(sym[i], E)
        out.append(E)
    return torch.cat(out) if any_ else None
