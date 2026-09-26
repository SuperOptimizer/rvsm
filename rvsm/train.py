"""The student trainer: one process, one device, the whole v2 recipe in one step loop.

Everything the run learns happens here. A step is

    sample  ->  prep.prepare (build the 21-channel stem on the device)
            ->  aug.apply    (the scan's own augmentation ranges, converted to the sample's rung)
            ->  forward      (deep supervision when the preset has it)
            ->  Phase A      deep BCE + soft dice on the probability heads, then the auxiliary terms
                             (exclusivity, cascade self-consistency, skeleton recall, affinity)
            ->  Phase B      the signed midline distance and the thickness, with the Eikonal
                             regulariser and the heteroscedastic log-variance
            ->  Phase C      the CONSTRUCTED pair: `pair_bands(midline, thickness)` scored against the
                             recto/verso targets, so the two faces cannot overlap by construction
            ->  ECT          the topology term, at one rung, on interior sub-blocks only
            ->  backward / accumulate / clip / step / EMA

and every `eval_every` steps the EMA weights are scored on the held-out grid, a PNG is written, the
per-rung temperatures are refitted and the checkpoint is replaced atomically.

There is no DDP, no streaming queue and no flag bookkeeping: the run's identity is `Config`, and a
resume simply asserts that its fingerprint still matches. A head whose store does not exist yet -- the
verso channel in round 0, the distances before `rvsm.targets` has run -- arrives with weight 0 and
therefore with no gradient, which is why none of it is special-cased.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rvsm import aug as A
from rvsm import config as CFG
from rvsm import losses as L
from rvsm import model as M
from rvsm import overlap as OV
from rvsm import prep

# --------------------------------------------------------------------------- the schedule


def lr_lambda(steps, warmup, lr_floor=0.0, sched="wsd", stable_until=None, cooldown=0):
    """The `LambdaLR` factor.

    `wsd` (warmup-stable-decay): linear warmup, a FLAT plateau until `stable_until`, then a cosine
    cooling over `cooldown` steps. Its point is that the step budget need not be committed at run start
    -- neither the plateau end nor the cooldown length is part of the weights, so a resume may move
    them and the plateau simply runs longer. The defaults follow the literature's 10 % cooldown.

    `cosine` is kept as an option, unchanged to the last float op: warmup, then one cosine over the
    whole budget.
    """
    if str(sched) != "wsd":
        return lambda s: min((s + 1) / max(warmup, 1), 1.0) * \
            (lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * min(s / max(steps, 1), 1.0))))
    C = int(cooldown) if cooldown else max(int(round(0.1 * steps)), 1)
    S = int(stable_until) if stable_until is not None else max(int(steps) - C, 1)

    def f(s):
        wu = min((s + 1) / max(warmup, 1), 1.0)
        if s < S:
            return wu
        t = min((s - S) / max(C, 1), 1.0)
        return wu * (lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * t)))
    return f


def wsd_stable_until(steps, cooldown_frac=0.1, cooldown=None):
    """The step the WSD plateau ends at, for a run of `steps` with a `cooldown_frac` cooldown."""
    C = int(cooldown) if cooldown else max(int(round(float(cooldown_frac) * int(steps))), 1)
    return max(int(steps) - C, 1), C


EMA_K = 50   # the averaging window is steps / k, i.e. 2 % of the run at k = 50


def ema_auto(steps, k=EMA_K):
    """`1 - k / steps`, clamped to [0.9, 0.9999] so a very short or very long run stays sane."""
    return float(min(max(1.0 - float(k) / max(int(steps), 1), 0.9), 0.9999))


# --------------------------------------------------------------------------- parameter groups


def new_param_names(newp, net):
    """The subset of `newp` (what `warm_start` could not copy) that are trainable PARAMETERS.

    Parameter groups are per TENSOR, not per row, so a stem convolution that gained input planes or a
    head that gained rows lands wholly in the boosted group -- that is the standard practical form of
    the "new parameters take full LR" recipe, and it is cheap here: the head is a 1x1x1 convolution and
    the stem is one 3x3x3 convolution out of ~200 tensors.
    """
    own = {n for n, _ in net.named_parameters()}
    return {n for n in newp if n in own}


def param_groups(net, new_names, mult):
    """`([groups], split)` for AdamW: two groups when `mult != 1` and something is new, else one."""
    if float(mult) == 1.0 or not new_names:
        return [{"params": list(net.parameters())}], False
    new = [p for n, p in net.named_parameters() if n in new_names]
    old = [p for n, p in net.named_parameters() if n not in new_names]
    return [{"params": old}, {"params": new}], bool(new)


@torch.no_grad()
def ema_update(ema, model, decay=0.999):
    """`e.mul_(decay).add_(v, alpha=1-decay)` per tensor, batched with the foreach kernels: one pair of
    kernel launches for the whole state instead of two per tensor. Integer buffers are copied."""
    es, vs = [], []
    for k, v in model.state_dict().items():
        e = ema[k]
        if e.is_floating_point():
            es.append(e)
            vs.append(v.detach())
        else:
            e.copy_(v)
    if es:
        torch._foreach_mul_(es, decay)
        torch._foreach_add_(es, vs, alpha=1 - decay)


# --------------------------------------------------------------------------- the warm start

USRM2_TAIL = ("scale", "rz", "ry", "rx")


def usrm2_stem_names(n):
    """The stem channel names of a usrm2 checkpoint with `n` input planes.

    usrm2's stem is `[CT, ctx_1..ctx_9, (CASCADE), scale, rz, ry, rx]`: 14 channels without the cascade
    slot and 15 with it. The names are rvsm's own (`Layout.stem_names`), so a 14/15-channel usrm2 stem
    maps straight onto rvsm's 21-channel one by NAME and the slots rvsm added -- the radius plane and
    the five scan-metadata planes -- simply stay zero.
    """
    n = int(n)
    cas = 1 if n >= 15 else 0
    ncube = n - len(USRM2_TAIL) - cas
    assert ncube >= 1, f"a {n}-channel stem cannot be a usrm2 stem"
    names = ["CT"] + [f"ctx_{i}" for i in range(1, ncube)]
    return names + (["cascade"] if cas else []) + list(USRM2_TAIL)


def usrm2_head_names(m):
    """The head names of a usrm2 checkpoint with `m` output rows: cout 1 is recto, cout 2 recto+verso."""
    base = ["recto", "verso"]
    return [base[i] if i < len(base) else f"c{i}" for i in range(int(m))]


def _names(rec, key, fallback):
    v = (rec or {}).get(key)
    return [str(q) for q in v] if v else list(fallback)


def warm_start(src_sd, net, layout, src_layout=None):
    """Adapt another run's weights to this net: `(state_dict, newp)`.

    Every tensor whose NAME and shape the destination already has is copied verbatim. The two tensors
    that may legitimately change width are the stem convolution (more input planes) and the heads (more
    output rows), and both are matched by NAME -- `Layout.stem_names()` / `Layout.head_names()` of the
    source (recorded in its checkpoint's `layout`, or inferred as a usrm2 stem/head from the shapes)
    against this layout's. A destination slot the source has no name for is left at ZERO: a new input
    plane then contributes nothing, so the step-0 output is the source's, and a new head row starts at
    p = 0.5 (a probability) or distance 0 (a regression), which is each encoding's own zero.

    `newp` is the set of tensor names that were NOT copied -- the tensors that hold freshly initialised
    values. It is what `new_param_names` turns into the boosted parameter group.
    """
    own = net.state_dict()
    src_sd = dict(src_sd)
    out, newp = {}, set()
    dst_stem, dst_head = layout.stem_names(), layout.head_names()
    stem_key = "enc.0.0.weight"
    head_keys = {k for k in own if k == "head.weight" or k == "head.bias"
                 or (k.startswith("deep_heads.") and (k.endswith(".weight") or k.endswith(".bias")))}
    for k, dv in own.items():
        sv = src_sd.get(k)
        if sv is None:
            newp.add(k)
            continue
        if tuple(sv.shape) == tuple(dv.shape):
            out[k] = sv.clone()
            continue
        if k == stem_key and sv.ndim == dv.ndim == 5 and sv.shape[0] == dv.shape[0]:
            s_names = _names(src_layout, "stem", usrm2_stem_names(sv.shape[1]))
            w = torch.zeros_like(dv)
            idx = {n: i for i, n in enumerate(s_names)}
            hit = 0
            for j, nm in enumerate(dst_stem):
                if nm in idx and idx[nm] < sv.shape[1]:
                    w[:, j] = sv[:, idx[nm]]
                    hit += 1
            out[k] = w
            if hit < len(dst_stem):
                newp.add(k)
            continue
        if k in head_keys and sv.shape[0] != dv.shape[0]:
            s_names = _names(src_layout, "heads", usrm2_head_names(sv.shape[0]))
            w = torch.zeros_like(dv)
            idx = {n: i for i, n in enumerate(s_names)}
            hit = 0
            for j, nm in enumerate(dst_head):
                if nm in idx and idx[nm] < sv.shape[0]:
                    w[j] = sv[idx[nm]]
                    hit += 1
            out[k] = w
            if hit < len(dst_head):
                newp.add(k)
            continue
        newp.add(k)   # a real shape mismatch (another preset, another depth): keep the fresh tensor
    return out, newp


# --------------------------------------------------------------------------- evaluation


def _batch(item):
    """A grid entry -> a collated batch of one, whether it came in collated or not."""
    return item if item["rung"].ndim else prep.batch1(item)


def _rungs_of(b):
    return [int(v) for v in b["rung"].reshape(-1).tolist()]


def self_p_at(cfg, step, nsteps=None):
    """The cascade `mix` self-source probability at `step`: `self_p_lo` at 0, linear to `self_p_hi` at
    `self_p_mid_step`, linear to `self_p_end` at `self_p_end_step`, held after. The mask source takes
    the rest. `self_p_mid_step <= 0` is the old schedule, lo -> hi linearly over the whole run
    (`nsteps`). The faster anneal (0.7 at 20k instead of 60k) came after paris4's self-cascade dice
    fell 0.15 -> 0.05 over steps 4k-10k while the mask-cascade dice sat at 0.83: at self_p ~0.2 the
    student leaned on the mask source and never learned to use its own coarse prediction."""
    s, lo, hi = float(step), float(cfg.self_p_lo), float(cfg.self_p_hi)
    m = int(getattr(cfg, "self_p_mid_step", 0) or 0)
    if m <= 0:
        return lo + (hi - lo) * min(s / max(int(nsteps or cfg.steps), 1), 1.0)
    if s <= m:
        return lo + (hi - lo) * s / m
    e, end = int(cfg.self_p_end_step), float(cfg.self_p_end)
    if e <= m or s >= e:
        return end if e > m else hi
    return hi + (end - hi) * (s - m) / (e - m)


def pair_terms(lr_, lv_, tgt2, w2, paired):
    """(bce, dice) of the constructed pair (recto, verso logits built from midline / thickness) against
    the two probability targets, with the weight map multiplied by the PAIRED support (the midline
    target's weight > 0). A batch with no paired support scores 0 and sends no gradient into the
    midline / thickness heads."""
    return L.losses_tw(torch.cat([lr_, lv_], 1), tgt2, w2 * paired)


def ect_seed(step, micro=0):
    """The ECT block draw's seed: the step AND the accumulation microbatch, so the microbatches of one
    step draw different blocks, and a resume replays the same ones."""
    return 1_000_003 * int(step) + 7_919 * int(micro) + 17


def train_compile_mode():
    """The trainer's `torch.compile(net)` mode: `RVSM_TRAIN_COMPILE_MODE` when set ("max-autotune",
    "reduce-overhead", "max-autotune-no-cudagraphs", "default"), else None -- torch's default, what the
    trainer has always compiled with. An environment override, not a Config field, so the host can
    switch it for a live run without touching the resume fingerprint."""
    m = os.environ.get("RVSM_TRAIN_COMPILE_MODE", "").strip()
    return m or None


def uses_cudagraphs(mode):
    """Does this compile mode capture CUDA graphs (then every step must mark its start)?"""
    return mode in ("max-autotune", "reduce-overhead")


def train_compile_threads():
    """`RVSM_TRAIN_COMPILE_THREADS`: inductor compile workers for THIS process (the trainer) only.

    The run's environment sets TORCHINDUCTOR_COMPILE_THREADS=1 for everybody (eight inductor workers in
    the producer once took 40 GB of host RAM). This sets `torch._inductor.config.compile_threads`
    in-process instead of the environment variable: the supervisor process is also the trainer, and
    the producer it spawns inherits its environment -- an env var set here would raise the producer's
    pool too. Must run before the first compile (the worker pool is created lazily at it). Returns the
    count in force, or None when the override is not set."""
    n = os.environ.get("RVSM_TRAIN_COMPILE_THREADS", "").strip()
    if not n:
        return None
    import torch._inductor.config as ic
    ic.compile_threads = max(int(n), 1)
    return ic.compile_threads


STOP_SAVE_MIN = 200      # a stop_now() exit checkpoints only this many steps past the last checkpoint
NONFINITE_MAX = 20       # consecutive skipped steps (non-finite gradient) that abort a run


def grad_gate(net, max_norm=1.0):
    """Clip the gradients to `max_norm` and say whether they were FINITE. `clip_grad_norm_` returns
    the total norm before clipping, which is non-finite exactly when some gradient is; the caller then
    skips the optimiser step, the schedule and the EMA rather than write NaN into the weights."""
    n = torch.nn.utils.clip_grad_norm_(net.parameters(), float(max_norm))
    return bool(torch.isfinite(n))


def keep_prev(ck):
    """Before a checkpoint is replaced, keep the one it replaces as `<name>_prev<suffix>` (a hard
    link, so `ck` itself never disappears for a reader): a bad save or a poisoned step leaves the last
    good weights one file away."""
    import os
    from pathlib import Path
    ck = Path(ck)
    if not ck.exists():
        return None
    prev = ck.with_name(f"{ck.stem}_prev{ck.suffix}")
    tmp = prev.with_suffix(prev.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        os.link(ck, tmp)
    except OSError:
        import shutil
        shutil.copy2(ck, tmp)
    os.replace(tmp, prev)
    return prev


def eval_cascade_mode(mode):
    """The cascade source the evaluation feeds, following the TRAINING mode: self / mix -> self (what
    inference feeds), off -> off, mask -> mask (a mask-trained run is evaluated as it was trained).
    The pooled target otherwise enters the evaluation only through the separate `dice_mask` bracket."""
    m = str(mode or "off")
    return {"self": "self", "mix": "self", "off": "off", "mask": "mask"}[m]


EVAL_SCHEMA = "eval-v3"  # bump when an eval metric changes meaning (the gates compare like with like)
FINE_RUNGS = (2, 3, 4)   # the rungs the headline `dice` / `bce` / `mae` are pooled over
NBINS = 200              # probability bins of the threshold-free (best-threshold) dice


def _best_dice(hp, ha):
    """(best dice, its threshold) over the bin edges of weighted histograms of p: `hp` over the
    target-positive weight, `ha` over all weight. Dice at threshold j/NBINS is
    2 sum_{b>=j} hp / (sum_{b>=j} ha + sum hp + 1), the same smoothing as the thresholded dice."""
    tp = torch.flip(torch.cumsum(torch.flip(hp, [0]), 0), [0])
    pp = torch.flip(torch.cumsum(torch.flip(ha, [0]), 0), [0])
    d = 2 * tp / (pp + hp.sum() + 1.0)
    j = int(torch.argmax(d))
    return float(d[j]), j / float(len(hp))


BAND_RUNGS = (2, 3)       # the rungs the recto BAND metrics (`dice_recto_r2_thin` ...) are scored at
BAND_REACH = 2            # a thinned-reference voxel is COVERED when a student voxel is this close
TOPO_HEADS = {2: ("recto", "verso"), 4: ("recto",)}   # rung -> heads the continuity metrics score
TOPO_MARGIN, TOPO_BAND = 8, 6    # betti0: interior crop and band around the reference (`compare_stores`)
NBAND, NTOPO = 6, 5              # lengths of the two parts of a `_sheet_sums` vector


def _label26(m):
    """26-connected component labels of the bool (B, Z, Y, X) mask `m` (0 off it): `edt.label` on the
    GPU, scipy on the CPU (the torch iteration `edt.label` falls back to there is far slower)."""
    if m.is_cuda:
        from rvsm import edt as E
        return E.label(m)
    from scipy import ndimage as ndi
    st = np.ones((3, 3, 3), np.uint8)
    return torch.from_numpy(np.stack([ndi.label(v, structure=st)[0] for v in m.numpy()]).astype(np.int32))


def _ncomp(m):
    """Number of 26-connected components of the bool (B, Z, Y, X) mask `m`."""
    if not bool(m.any()):
        return 0
    lab = _label26(m)
    return int(sum(torch.unique(lab[b][m[b]]).numel() for b in range(m.shape[0])))


def _sheet_sums(p, t, w, thr=0.5, band=True, topo=False):
    """Voxel sums of the BAND and CONTINUITY metrics of one head of one window, for a reference that
    can be a WIDE soft band (m7's 9.6 um output upsampled 4x) against a student that draws thin sharp
    sheets, where plain dice caps near 0.27 whatever the student does. `p`, `t`, `w` are (B, Z, Y, X)
    probability, target and weight of the one channel; only voxels with weight > 0 are looked at.

    s = p >= thr (the student sheet, at the 0.5 threshold of `dice` -- the best threshold is only known
    after pooling), ref = t >= 0.5, thin = the medial surface of ref (`targets.medial_torch`, exact
    Euclidean), dil(.) = one 26-neighbour dilation (3^3 max-pool), cov = thin voxels within
    `BAND_REACH` (Euclidean, `edt.edt2`) of s. With `band`, the first `NBAND` entries are
      (|s & dil(thin)| + |thin & dil(s)|, |s| + |thin|)   -> dice_thin, a 1-voxel TOLERANT dice
      (|cov|, |thin|)                                     -> recall_band
      (|s & dil(ref)|, |s|)                               -> precision_band
    With `topo`, `NTOPO` more -- the store analogues of `evalsurf`'s mesh-walk continuity / ERL (which
    need tifxyz meshes the eval grid does not have) and `compare_stores`' betti0 error:
      (|cov & ~dil(thin & ~cov)|, |cov|)  -> cont: covered medial voxels whose medial neighbours are
                                             ALL covered (`continuity_one` with the medial surface as
                                             the grid)
      (sum over the 26-connected pieces of cov of size^2, |thin|)
                                          -> erl: the expected size (medial voxels) of the covered piece
                                             holding a medial voxel drawn uniformly (`compare_stores`'
                                             erl_vox, on the exact medial surface)
      |b0(s & B) - b0(ref & B)|           -> betti0: B = the interior (`TOPO_MARGIN` cropped) within
                                             `TOPO_BAND` voxels of ref, 26-connected components
    as a float64 CPU vector."""
    from rvsm import edt as E
    from rvsm.targets import medial_torch
    k = w > 0
    s = (p >= thr) & k
    ref = (t >= 0.5) & k
    thin = medial_torch(ref, cap=16) & k

    def dil(m):
        return F.max_pool3d(m.float().unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0
    cov = thin & (E.edt2(s, indices=False, cap=BAND_REACH)[0] <= BAND_REACH * BAND_REACH)
    v = []
    if band:
        v += [(s & dil(thin)).sum() + (thin & dil(s)).sum(), s.sum() + thin.sum(),
              cov.sum(), thin.sum(), (s & dil(ref)).sum(), s.sum()]
    if topo:
        v += [(cov & ~dil(thin & ~cov)).sum(), cov.sum()]
        l2 = 0.0
        if bool(cov.any()):
            lab = _label26(cov)
            for b in range(lab.shape[0]):
                _, n = torch.unique(lab[b][cov[b]], return_counts=True)
                l2 += float((n.double() ** 2).sum())
        v += [torch.tensor(l2), thin.sum()]
        m = TOPO_MARGIN
        inner = torch.zeros_like(ref)
        inner[:, m:-m or None, m:-m or None, m:-m or None] = True
        near = inner & (E.edt2(ref, indices=False, cap=TOPO_BAND)[0] <= TOPO_BAND * TOPO_BAND)
        v += [torch.tensor(float(abs(_ncomp(s & near) - _ncomp(ref & near))))]
    return torch.stack([torch.as_tensor(q).double().cpu() for q in v])


def _pool_eval(acc):
    """(bce, dice, mae, overlap) of pooled voxel sums: every weighted voxel counts once, whatever
    window or rung it came from."""
    n = max(acc["w"], 1e-6)
    return (acc["bce"] / n, 2.0 * acc["tp"] / (acc["den"] + 1.0), acc["ae"] / n,
            acc["ov"] / max(acc["vox"], 1.0))


@torch.no_grad()
def evaluate(net, grid, dev, layout, cascade=None, calib_keep=None, rungs=None, band=True,
             cost=None, capture=None):
    """bce / dice / mae over the validation grid, plus the metrics the layout makes meaningful.

    Every entry is a compact rung sample (`sample.rung_item`), whose input is built on the device by
    `prep.prepare`. The metrics are VOXEL-WEIGHTED: bce / mae are sums of the per-voxel value times the
    per-voxel weight over sums of the weight, and dice is 2 sum(h t w) / (sum(h w) + sum(t w)) over the
    same voxels -- pooled over windows, never a mean of per-window numbers. So a coarse window with
    eight weighted corner voxels counts eight voxels, not as much as a full rung-2 window (at paris4
    step 2000 exactly such rung 7-11 windows, confidently wrong on a sliver of weight, dragged the
    per-window mean bce to ~1.0 while rungs 2-4 sat at 0.44-0.54, the training value).

    Each rung is scored on its own (`dice_r{k}`, `bce_r{k}`). The HEADLINE `dice` / `bce` / `mae` pool
    the FINE rungs `FINE_RUNGS` (2-4) only -- they are what the plateau fit of the round gate and the
    verso gate's pre-screen read -- and `dice_coarse` / `bce_coarse` pool rungs 5-11. A grid with no
    fine window falls back to every rung for the headline. bce / dice / mae are PROBABILITY-head
    metrics only. A window with no probability weight at all is skipped, `n_scored` counts the windows
    that were scored, and a rung made only of such windows has no `dice_r{k}`.

    Per probability channel there is a `dice_<name>` (`dice_recto`, `dice_verso`), pooled over the
    fine rungs (the same fallback) and only over voxels whose WEIGHT says anything about that channel,
    so the verso channel is absent until a verso store covers the held-out boxes. With two probability
    channels there is also `overlap`, the mean excess `relu(p_recto + p_verso - 1)` over the fine
    rungs' voxels -- measured, never a loss term. The DISTANCE heads are a regression and are scored in
    voxels as `mae_midline` / `mae_thickness` (a mean over windows, as before).

    THRESHOLD-FREE numbers, per fine rung and pooled over them: `dice_best` (the best dice over
    `NBINS` probability thresholds, and `thr_best` the threshold that gives it) and `dice_soft`
    (2 sum(p t w) / (sum(p w) + sum(t w))). They say whether the model RANKS the band well when the
    0.5 threshold cuts in the wrong place. Note that a temperature cannot help the thresholded dice:
    sigmoid(l / T) >= 0.5 exactly when l >= 0, for every T > 0, so a calibrated `dice` is the raw one.
    `dice_raw` is written as an alias of `dice` to make that explicit in the logs.

    RECTO BAND metrics at rungs `BAND_RUNGS` (2, 3), for a reference that is a wide soft band (m7
    upsampled) while the student draws thin sheets (see `_sheet_sums`; student sheet = p >= 0.5):
    `dice_recto_r{k}_thin` (1-voxel tolerant dice against the band's medial surface),
    `recall_recto_r{k}_band` (medial-surface voxels within `BAND_REACH` voxels of a student sheet voxel)
    and `precision_recto_r{k}_band` (student sheet voxels inside the band dilated by one voxel).
    CONTINUITY metrics per head and rung of `TOPO_HEADS` (recto and verso at rung 2, recto at rung 4),
    against the same reference: `cont_<head>_r{k}` (covered medial voxels whose medial neighbours are
    all covered), `erl_<head>_r{k}` (expected covered-piece size, in medial voxels, pooled over the
    windows) and `betti0_<head>_r{k}` (mean per-window |component-count error| in a band around the
    reference) -- store analogues of `evalsurf`'s mesh metrics, which need tifxyz meshes.
    `band=False` skips all of them; a `cost` dict gets their seconds as `cost["band"]` (eval_s).

    `rungs` scores only the windows of those rungs (the mask-cascade pass scores the fine ones).

    The affinity heads are a training-only head: never scored, never drawn.

    `capture` (a `PanelCapture`, whole-grid passes only) is handed every window's forward, so the
    validation PNGs are drawn from THIS pass instead of re-reading and re-running their items.
    """
    was = net.training
    net.eval()
    per, pch, dch, scored = {}, {}, {}, 0
    np_ = layout.nprob
    chn = list(layout.channels)
    bsum, tsum, band_s = {}, {}, 0.0   # band / continuity sums per rung (`_sheet_sums`), their cost
    if rungs is not None and hasattr(grid, "of_rungs"):
        grid = grid.of_rungs(rungs)      # a DiskGrid: the other rungs' items are not even read
    for gi, item in enumerate(grid):
        b = _batch(item)
        rung = _rungs_of(b)[0]
        if rungs is not None and int(rung) not in rungs:
            continue
        x, tg, ww = prep.prepare(b, dev, cascade=cascade)
        x = x.to(memory_format=M.memfmt())
        with prep.autocast(dev):
            y = net(x)
        y = (y[0] if isinstance(y, (list, tuple)) else y).float()
        for j, nm in ((layout.i_mid, "midline"), (layout.i_thick, "thickness")):
            wc = L.dist_weight(ww[:, j:j + 1])
            if float(wc.sum()) > 0:
                dec = L.decode_unsigned if nm == "thickness" else L.decode_signed
                pv = L.soft_thickness(y[:, j:j + 1]) if nm == "thickness" else y[:, j:j + 1]
                pch.setdefault(f"mae_{nm}", []).append(
                    float((pv - dec(tg[:, j:j + 1])).abs().mul(wc).sum() / wc.sum()))
        # the PROBABILITY heads only: rows nprob..cout_t are distance regressions (voxels, not logits,
        # against a code/255 target), scored above as mae_midline / mae_thickness. Pooling them in here
        # put a sigmoid of a distance into bce/dice/mae wherever a distance store has weight.
        logit, tgt, w = y[:, :np_], tg[:, :np_], ww[:, :np_]
        if calib_keep is not None:     # the calibration's logits, off THIS forward (see calib.keep)
            from rvsm import calib as _cal
            _cal.keep(calib_keep, logit, tgt, w, rung)
        p = torch.sigmoid(logit)
        if capture is not None and rungs is None:     # the validation PNGs' slices (`PanelCapture`)
            try:
                capture.add(gi, item, x, tgt, ww, p)
            except Exception as e:  # noqa: BLE001  -- the pictures must never cost the metrics
                print("[train] val_png capture:", repr(e), flush=True)
                capture.broken, capture = repr(e), None
        if float(w.sum()) <= 0:
            # no probability weight anywhere in the window (a held-out box with no store at this rung):
            # it says nothing, so it is not scored
            continue
        scored += 1
        h, t = (p >= 0.5).float(), (tgt >= 0.5).float()
        a = per.setdefault(int(rung), {"bce": 0.0, "w": 0.0, "tp": 0.0, "den": 0.0, "ae": 0.0,
                                       "ov": 0.0, "vox": 0.0, "sp": 0.0, "sd": 0.0,
                                       "hp": torch.zeros(NBINS, dtype=torch.float64),
                                       "ha": torch.zeros(NBINS, dtype=torch.float64)})
        a["bce"] += float((F.binary_cross_entropy_with_logits(logit, tgt, reduction="none") * w).sum())
        a["w"] += float(w.sum())
        a["tp"] += float((h * t * w).sum())
        a["den"] += float((h * w).sum() + (t * w).sum())
        a["ae"] += float(((p - tgt).abs() * w).sum())
        a["sp"] += float((p * t * w).sum())
        a["sd"] += float((p * w).sum() + (t * w).sum())
        bi = (p * NBINS).long().clamp_(0, NBINS - 1).flatten()
        a["hp"] += torch.bincount(bi, weights=(t * w).flatten().double(), minlength=NBINS).cpu()
        a["ha"] += torch.bincount(bi, weights=w.flatten().double(), minlength=NBINS).cpu()
        if np_ >= 2:
            a["ov"] += float((p[:, :1] + p[:, 1:2] - 1).clamp_min(0).sum())
            a["vox"] += float(p[:, :1].numel())
        for c, nm in enumerate(chn[:np_]):     # the band / continuity metrics (`_sheet_sums`)
            do_b = band and nm == "recto" and int(rung) in BAND_RUNGS
            do_t = band and nm in TOPO_HEADS.get(int(rung), ())
            if (do_b or do_t) and float(w[:, c].sum()) > 0:
                tb = time.perf_counter()
                v = _sheet_sums(p[:, c], tgt[:, c], w[:, c], band=do_b, topo=do_t)
                if do_b:
                    bsum[int(rung)] = bsum.get(int(rung), 0) + v[:NBAND]
                if do_t:
                    q = tsum.setdefault((nm, int(rung)), [torch.zeros(NTOPO, dtype=torch.float64), 0])
                    q[0] += v[-NTOPO:]
                    q[1] += 1
                band_s += time.perf_counter() - tb
        for c in range(np_):
            wc = w[:, c]
            if float(wc.sum()) > 0:
                q = dch.setdefault(int(rung), {}).setdefault(c, [0.0, 0.0])
                q[0] += float((h[:, c] * t[:, c] * wc).sum())
                q[1] += float((h[:, c] * wc).sum() + (t[:, c] * wc).sum())
    net.train(was)

    def pooled(ks):
        acc = {"bce": 0.0, "w": 0.0, "tp": 0.0, "den": 0.0, "ae": 0.0, "ov": 0.0, "vox": 0.0,
               "sp": 0.0, "sd": 0.0, "hp": torch.zeros(NBINS, dtype=torch.float64),
               "ha": torch.zeros(NBINS, dtype=torch.float64)}
        for k in ks:
            for f in acc:
                acc[f] = acc[f] + per[k][f]
        return acc

    def free(acc):
        db, th = _best_dice(acc["hp"], acc["ha"])
        return db, th, 2.0 * acc["sp"] / (acc["sd"] + 1.0)

    fine = [k for k in sorted(per) if k in FINE_RUNGS] or sorted(per)
    coarse = [k for k in sorted(per) if k not in FINE_RUNGS]
    bce, dice, mae, ov = _pool_eval(pooled(fine)) if per else (0.0, 0.0, 0.0, 0.0)
    out = {"bce": bce, "dice": dice, "dice_raw": dice, "mae": mae, "n_scored": scored,
           "fine_rungs": [int(k) for k in fine]}
    if per:
        out["dice_best"], out["thr_best"], out["dice_soft"] = free(pooled(fine))
        for k in fine:
            out[f"dice_best_r{k}"], out[f"thr_best_r{k}"], out[f"dice_soft_r{k}"] = free(per[k])
    if coarse:
        cb, cd, cm_, _ = _pool_eval(pooled(coarse))
        out.update({"bce_coarse": cb, "dice_coarse": cd, "mae_coarse": cm_})
    for k in sorted(per):
        kb, kd, _, _ = _pool_eval(per[k])
        out[f"dice_r{k}"], out[f"bce_r{k}"] = kd, kb
    names = list(layout.channels)
    out["eval_schema"] = EVAL_SCHEMA
    for k in sorted(dch):                  # per HEAD per rung: dice_recto_r2 is the recto head alone
        for c, (num_k, den_k) in dch[k].items():
            nm = names[c] if c < len(names) else f"c{c}"
            out[f"dice_{nm}_r{k}"] = 2.0 * num_k / (den_k + 1.0)
    for c in range(np_):
        num = sum(dch.get(k, {}).get(c, [0.0, 0.0])[0] for k in fine)
        den = sum(dch.get(k, {}).get(c, [0.0, 0.0])[1] for k in fine)
        if any(c in dch.get(k, {}) for k in fine):
            out[f"dice_{names[c] if c < len(names) else f'c{c}'}"] = 2.0 * num / (den + 1.0)
    for key in sorted(k for k in pch if isinstance(k, str)):
        out[key] = float(np.mean(pch[key]))
    if np_ >= 2:
        out["overlap"] = ov
    for k in sorted(bsum):                 # the recto BAND metrics (`_sheet_sums`), rungs BAND_RUNGS
        v = bsum[k].tolist()
        out[f"dice_recto_r{k}_thin"] = v[0] / (v[1] + 1.0)
        out[f"recall_recto_r{k}_band"] = v[2] / max(v[3], 1.0)
        out[f"precision_recto_r{k}_band"] = v[4] / max(v[5], 1.0)
    for (nm, k), (v, n) in sorted(tsum.items()):   # continuity metrics (`_sheet_sums`), TOPO_HEADS
        v = v.tolist()
        out[f"cont_{nm}_r{k}"] = v[0] / max(v[1], 1.0)
        out[f"erl_{nm}_r{k}"] = v[2] / max(v[3], 1.0)
        out[f"betti0_{nm}_r{k}"] = v[4] / max(n, 1)
    if cost is not None:                   # a timing, so kept out of the (reproducible) metrics
        cost["band"] = round(band_s, 3)
    return out


def _verso_on(path):
    """`verso_on` from the run's state.json beside `<out>/eval/<png>`, or None when there is none."""
    import json
    import os
    sp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(str(path)))), "state.json")
    try:
        with open(sp) as f:
            return bool(json.load(f).get("verso_on", False))
    except (OSError, ValueError):
        return None


def _region_of_item(item, region):
    """The rung-2 region origin of a grid item's window corner."""
    k = int(torch.as_tensor(item["rung"]).reshape(-1)[0])
    lo = [int(v) for v in torch.as_tensor(item["lo"]).reshape(-1)[:3]]
    return tuple(((v << max(k - 2, 0)) // int(region)) * int(region) for v in lo)


def _radius_frac(item):
    """Distance of a grid window's centre from the scroll axis, as a fraction of r_max (both in the
    window's rung voxels: the item carries the axis over its z range and the radius denominator)."""
    cyx = torch.as_tensor(item["cyx"]).double()
    lo = [float(v) for v in torch.as_tensor(item["lo"]).reshape(-1)[:3]]
    n = int(cyx.shape[-1])
    c = cyx.reshape(2, -1)[:, n // 2]
    yx = torch.tensor([lo[1] + n / 2.0, lo[2] + n / 2.0], dtype=torch.float64)
    r = float(torch.linalg.norm(yx - c))
    rmax = float(torch.as_tensor(item.get("rmax", 0.0)).reshape(-1)[0]) if "rmax" in item else 0.0
    return r / rmax if rmax > 0 else float("nan")


def _pick_panels(order, per, n):
    """`region_panels`' choice from its per-region (foreground, index, radius fraction) lists."""
    out = []
    for r in order:
        best = sorted(per[r], key=lambda q: -q[0])[:int(n)]
        fr = [q[2] for q in best if np.isfinite(q[2])]
        out.append((r, float(np.mean(fr)) if fr else float("nan"), [q[1] for q in best]))
    return out


def _panel_stat(item, region):
    """(region origin, target foreground fraction, radius fraction) of one grid item."""
    t, w = (q[0] for q in prep.full_tw(item))
    return (_region_of_item(item, region), float(((t >= 128) & (w > 0)).float().mean()),
            _radius_frac(item))


def region_panels(grid, region=1024, n=4):
    """[(region origin, radius fraction, grid INDICES)] -- one validation panel per held-out region, in
    the grid's order (the held-out order): the `n` windows of that region with the highest target
    foreground fraction (recto >= 0.5 where weighted). One pass over the grid. Indices, not items: a
    grid item is ~300 MB in memory, and holding 32 of them for the run was ~9 GB of trainer RSS
    (paris4 step 30000). The trainer does not call this: `PanelCapture` makes the same choice inside
    `evaluate`'s own pass, so the grid is not read a second time."""
    per = {}
    order = []
    for i, it in enumerate(grid):
        r, fg, fr = _panel_stat(it, region)
        if r not in per:
            per[r] = []
            order.append(r)
        per[r].append((fg, i, fr))
    return _pick_panels(order, per, n)


def write_region_panels(out, step, net, panels, dev, layout, cascade=None, grid=None):
    """val_<step>_r<k>.png for every held-out region k, and val_<step>_regions.txt: k, the region's
    rung-2 origin (z, y, x) and its radius fraction from the umbilicus. `panels` holds grid indices
    (`region_panels`); each panel's four items are read from `grid` only while it is drawn. This is the
    SYNCHRONOUS path (it re-reads and re-runs the panel items); the trainer uses `PanelCapture`."""
    import os
    rows = {}
    for k, (_, _, idx) in enumerate(panels):
        # a DiskGrid's subset is read with the grid's prefetch, one item decoded ahead of the drawing
        items = grid.subset(idx) if hasattr(grid, "subset") else [grid[i] for i in idx]
        name = os.path.join(str(out), "eval", f"val_{int(step):06d}_r{k}.png")
        rows[k] = _panel_rows(name, net, items, dev, layout, cascade=cascade)
        del items
    _write_region_pngs(out, step, panels, lambda k, idx: rows[k])


def _write_region_pngs(out, step, panels, rows_of):
    """The panel PNGs and the regions.txt index; `rows_of(k, idx)` gives panel k's row data."""
    import os
    lines = ["k\tregion_origin_zyx\tradius_frac\tpanel"]
    for k, (r, frac, idx) in enumerate(panels):
        name = f"val_{int(step):06d}_r{k}.png"
        _render_png(os.path.join(str(out), "eval", name), rows_of(k, idx))
        lines.append(f"{k}\t{r[0]},{r[1]},{r[2]}\t{frac:.3f}\t{name}")
    with open(os.path.join(str(out), "eval", f"val_{int(step):06d}_regions.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _panel_row(x, t, ww, p, nprob, on):
    """One window's row data for `_render_png`: the middle z-slice of the CT channel, and of the target
    and prediction channels drawn (the recto, plus the verso when it is on), as CPU numpy. `x`, `t`,
    `ww`, `p` are a batch of one's input, target, weight and PROBABILITY (sigmoid of the nprob logits)."""
    nch = 1
    if nprob >= 2:
        show = on if on is not None else bool(float(ww[0, 1].sum()) > 0)
        nch = 2 if show else 1
    z = x.shape[2] // 2
    return (x[0, 0, z].cpu().numpy(), t[0, :nch, z].cpu().numpy(), p[0, :nch, z].cpu().numpy())


def _render_png(path, rows):
    """CT (gray) | target | prediction for each row (`_panel_row`), the rows stacked, saved to `path`."""
    from PIL import Image
    out = []
    for c, tt, pp in rows:
        c = (c - c.min()) / (c.max() - c.min() + 1e-6) * 255 * 0.9
        gray = np.repeat(c[..., None], 3, -1)

        def overlay(chans, gray=gray):   # channel 0 red, channel 1 blue, mixed on one tile
            a = [np.clip(v, 0, 1)[..., None] for v in chans]
            cols = [np.array([255, 40, 40]), np.array([40, 90, 255])]
            wsum = sum(a)
            al = np.maximum.reduce(a) * 0.85
            col = sum(ai * ci for ai, ci in zip(a, cols)) / np.maximum(wsum, 1e-6)
            return gray * (1 - al) + col * al
        out.append(np.concatenate([gray, overlay(list(tt)), overlay(list(pp))], 1))
    Image.fromarray(np.concatenate(out, 0).astype(np.uint8)).save(path)


def _panel_rows(path, net, grid, dev, layout, cascade=None, verso=None):
    """`_panel_row` of the first four items of `grid`, each read and run through `net` here."""
    rows = []
    was = net.training
    net.eval()
    on = _verso_on(path) if verso is None else bool(verso)
    with torch.no_grad():
        for item in grid[:4]:
            b = _batch(item)
            x, t, ww = prep.prepare(b, dev, cascade=cascade)
            with prep.autocast(dev):
                y = net(x.to(memory_format=M.memfmt()))
            y = (y[0] if isinstance(y, (list, tuple)) else y).float()
            rows.append(_panel_row(x, t, ww, torch.sigmoid(y[:, :layout.nprob]), layout.nprob, on))
    net.train(was)
    return rows


def val_png(path, net, grid, dev, layout, cascade=None, verso=None):
    """The middle z-slice of the first four validation patches, three tiles wide:

        CT (gray) | recto target (red) | recto prediction (red)

    and, once VERSO IS ON, the verso target and prediction in blue on the same two tiles (overlap
    comes out purple, the fastest way to see the exclusivity term failing). Before that -- round 0
    until the verso gate fires -- the verso head is untrained (it saturates near 1 and painted the
    whole prediction tile blue over the recto), so only the recto is drawn. `verso`: True / False, or
    None to read `verso_on` from the run's state.json (two levels above `path`), falling back to
    whether the patch carries any verso target weight. The affinity and distance heads are never drawn.

    This is the SYNCHRONOUS path (it reads and runs the items itself). The trainer draws the same bytes
    from `evaluate`'s own forwards (`PanelCapture`), rendered on a background thread (`_PngWorker`).
    """
    _render_png(path, _panel_rows(path, net, grid, dev, layout, cascade=cascade, verso=verso))


class PanelCapture:
    """Everything the validation PNGs draw, taken from `evaluate`'s OWN forwards: `val_png`'s first four
    grid items and every region panel's items (`region_panels`), as the small middle-slice numpy
    arrays of `_panel_row`. Before this, `val_png` + `write_region_panels` re-read and re-ran 36 grid
    items (~310 MB compressed each) after the evaluation, and the first evaluation of each `train()`
    call re-read the WHOLE grid to pick the panels -- `eval_s.val_png` 290 s beside `evaluate` 404 s on
    the A100 run, all of it with the trainer not stepping.

    `panels` None: the choice is made here, in the same pass, from exactly `region_panels`' numbers
    (only the current top `n` of each region hold their slices). `on`: the verso switch `val_png` would
    have read (`_verso_on`), read once at the evaluation."""

    def __init__(self, layout, on, panels=None, region=1024, n=4, first=4):
        self.nprob, self.on = int(layout.nprob), on
        self.panels, self.region, self.n, self.first = panels, int(region), int(n), int(first)
        self.need = set(range(self.first)) | {int(i) for _, _, idx in (panels or ()) for i in idx}
        self.rows, self.per, self.order = {}, {}, []
        self.broken = None           # set by `evaluate` when a capture failed: nothing is drawn

    def add(self, i, item, x, t, ww, p):
        i = int(i)
        if self.panels is None:
            r, fg, fr = _panel_stat(item, self.region)
            if r not in self.per:
                self.per[r] = []
                self.order.append(r)
            self.per[r].append((fg, i, fr))
            top = {q[1] for q in sorted(self.per[r], key=lambda q: -q[0])[:self.n]}
            for q in self.per[r]:                 # a window that has fallen out of its region's top n
                if q[1] not in top and q[1] >= self.first:
                    self.rows.pop(q[1], None)
            keep = i < self.first or i in top
        else:
            keep = i in self.need
        if keep:
            self.rows[i] = _panel_row(x, t, ww, p, self.nprob, self.on)

    def finish(self):
        """The panels (chosen here when none were given)."""
        if self.panels is None:
            self.panels = _pick_panels(self.order, self.per, self.n)
        return self.panels

    def render(self, out, step):
        """val_<step>.png, the region panels and val_<step>_regions.txt, exactly the bytes `val_png` +
        `write_region_panels` write."""
        import os
        _render_png(os.path.join(str(out), "eval", f"val_{int(step):06d}.png"),
                    [self.rows[i] for i in sorted(self.rows) if i < self.first])
        _write_region_pngs(out, step, self.finish(), lambda k, idx: [self.rows[int(i)] for i in idx])


PNG_WAIT_S = 300.0   # the longest the trainer waits for a background PNG render (next eval, exit)


class _PngWorker:
    """One evaluation's PNGs, rendered on a background thread while training resumes. ONE render at a
    time: `submit` first waits for the previous one (two never race on the files or the log), and
    `close` waits for the last one at exit, at most `PNG_WAIT_S`, and logs the wait. A render is numpy
    + PIL over a few MB of slices, so a thread is enough (no pickling, no process start)."""

    def __init__(self, log_path):
        self.log_path, self.fut, self.step, self.last_s = log_path, None, None, None

    def wait(self, timeout=PNG_WAIT_S, why="next"):
        """Wait for the render in flight; False when it is still running after `timeout`."""
        if self.fut is None:
            return True
        import concurrent.futures as cf
        tw = time.time()
        ok = True
        try:
            self.last_s = self.fut.result(timeout=timeout)
        except cf.TimeoutError:
            ok = False
        except Exception as e:  # noqa: BLE001  -- a render failure (a missing PIL) must never stop a run
            print("[train] val_png:", repr(e), flush=True)
        if why == "exit" or not ok:
            _log(self.log_path, {"kind": "val_png_wait", "at": why, "step": self.step, "done": ok,
                                 "s": round(time.time() - tw, 2)})
        self.fut = None
        return ok

    def submit(self, step, cap, out):
        """Render `cap` (a `PanelCapture`) for `step`, after the previous render has finished."""
        self.wait()

        def job():
            t = time.time()
            cap.render(out, step)
            return round(time.time() - t, 2)
        self.step = int(step)
        self.fut = _spawn(job, name="rvsm-valpng")

    def close(self, timeout=PNG_WAIT_S):
        return self.wait(timeout, why="exit")


# --------------------------------------------------------------------------- the loop


# The spatial ops that change the METRIC: a rotation, a scaling, a shear, an elastic field or a sheet
# compression resamples the patch onto a grid on which one voxel is no longer one voxel of the grid the
# distance and thickness stores were measured on. A probability rides that resampling unharmed, but a
# distance does not -- the target still says "7.25 voxels" after the augmentation has made the voxel a
# different length, so it is simply a wrong number (usrm2 docs/unified_design.md section 29.2, which
# dropped them loudly whenever a distance head was on). The 48 cube symmetries are exact isometries on
# the voxel grid and are applied by the sampler (`prep.sym_apply_t`), so the geometric variety is kept;
# every intensity and physics op is untouched.
NON_ISOMETRIC = ("rot", "scale", "shear", "elastic", "sheetcomp")


def aug_for(cfg, meta=None):
    """The augmentation configuration this recipe may actually use.

    `A.get(cfg.aug, meta)` recentred on the scan's metadata, minus `NON_ISOMETRIC` when the layout has
    DISTANCE heads and `loss_sdist` is on -- which, in rvsm's fixed recipe, it always is.
    """
    acfg = A.get(cfg.aug, meta=meta)
    layout = cfg.layout()
    if layout.cout_t > layout.nprob and float(cfg.loss_sdist) > 0:
        off = [q for q in NON_ISOMETRIC if q in acfg]
        if off:
            acfg = {k: v for k, v in acfg.items() if k not in NON_ISOMETRIC}
            print(f"[train] loss_sdist is on, so the non-isometric spatial augs {off} are dropped from "
                  f"--aug {cfg.aug}: they resample the grid the distance and thickness targets are "
                  "measured on. The 48 cube symmetries and every intensity op stay.", flush=True)
    return acfg


def _aug_cfg(cfg, out=None):
    """`aug_for` with the scan's metadata. A run directory's FROZEN `<out>/metadata.json` (written once
    by `run.frozen_meta`) is used when there is one, so the augmentation ranges cannot change on a
    resume because the source changed or was unreachable that day (pass-3 review P3-13); only a
    standalone `rvsm train` without it reads the metadata beside the CT."""
    import json
    fz = os.path.join(str(out), "metadata.json") if out is not None else None
    if fz and os.path.exists(fz):
        with open(fz) as f:
            return aug_for(cfg, json.load(f))
    meta = None
    try:
        from rvsm import scanmeta as SM
        meta = SM.fetch(cfg.ct)
    except Exception as e:   # noqa: BLE001  -- no metadata.json beside the CT: the preset's own ranges
        print(f"[train] scan metadata unavailable ({e!r}); using the preset ranges", flush=True)
    return aug_for(cfg, meta)


def _log(path, rec, echo=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if echo:
        print(os.path.basename(path), rec, flush=True)


def _prepared(grid, dev, layout, cascade=None):
    """The validation grid as the `(x, t, w, rung)` batches `calib.collect` reads."""
    for item in grid:
        b = _batch(item)
        x, t, w = prep.prepare(b, dev, cascade=cascade)
        yield x.to(memory_format=M.memfmt()), t, w, _rungs_of(b)[0]


QUIESCE_S = 60.0     # the longest `DevicePrefetch.close` waits for a fetch in flight before it stops waiting


def _run_into(fut, fn):
    if not fut.set_running_or_notify_cancel():
        return
    try:
        fut.set_result(fn())
    except BaseException as e:  # noqa: BLE001  -- re-raised by fut.result() on the caller's side
        fut.set_exception(e)


def _spawn(fn, name="rvsm-h2d-once"):
    """`fn()` on a fresh DAEMON thread, as a Future (a one-off: `DevicePrefetch` keeps ONE thread)."""
    import concurrent.futures as cf
    import threading
    fut = cf.Future()
    threading.Thread(target=_run_into, args=(fut, fn), name=name, daemon=True).start()
    return fut


class _Fetcher:
    """ONE long-lived DAEMON thread that runs submitted calls in order and returns Futures.

    One thread for the whole run, like the ThreadPoolExecutor(1) this replaces -- a thread per batch
    means per-thread CUDA / allocator state created and torn down every step -- but a daemon one: a
    `next(loader)` that never returns (an old round's walk waiting on a store nobody will produce) must
    not block the trainer's shutdown, and an executor's worker is joined at interpreter exit."""

    def __init__(self, name="rvsm-h2d"):
        import queue
        import threading
        self.q = queue.SimpleQueue()
        self.t = threading.Thread(target=self._loop, name=name, daemon=True)
        self.t.start()

    def _loop(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            _run_into(*job)
            del job                 # nothing of a finished call outlives it on this thread

    def submit(self, fn):
        import concurrent.futures as cf
        fut = cf.Future()
        self.q.put((fut, fn))
        return fut

    def stop(self):
        self.q.put(None)


def _shutdown_loader(src, it):
    """Stop a loader's worker processes NOW, persistent ones included: `_shutdown_workers` joins each
    worker briefly and terminates the ones still alive, and the DataLoader forgets the iterator, so no
    worker of this loader can publish anything afterwards. A plain generator is closed instead."""
    for obj in (it, getattr(src, "_iterator", None)):
        fn = getattr(obj, "_shutdown_workers", None)
        if fn is not None:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
    if getattr(src, "_iterator", None) is not None:
        src._iterator = None
    close = getattr(it, "close", None)
    if close is not None and not hasattr(it, "_shutdown_workers"):
        try:
            close()
        except Exception:  # noqa: BLE001  -- "generator already executing": the abandoned fetch owns it
            pass


class DevicePrefetch:
    """Batches from `src` with every tensor already on `dev`, fetched and copied AHEAD: a helper thread
    takes batch i+1 from the loader and copies it (on a side CUDA stream) while the caller runs step i;
    the default stream waits on the copy's event before it touches the batch. On a CPU device the batches
    pass straight through.

    The thread is what makes this overlap at all: from pageable memory a `non_blocking` copy is still
    synchronous for the calling thread (~110 ms for one 256^3 sample's ~300 MB of uint8 over
    Thunder's 2.7 GB/s link), and pinned memory is not an option there (the trainer hung inside CUDA
    calls whenever the loader pinned). `side=False` copies on the default stream instead (still on
    the helper thread, whose wait the RAM guard's `stop()` can abandon).

    `close()` is the SHUTDOWN PROTOCOL (review T19): it stops handing out batches, waits (at most
    `timeout` seconds) for the fetch in flight, and shuts the loader's workers down. The round
    transition calls it before it resets the cursor directory; relying on the generator's GC left a
    pending fetch and the persistent workers free to publish old-round positions into the new round.

    TIMING (host clocks only, never a CUDA sync): `take_wait_s()` is the seconds the CALLER spent
    blocked waiting for a batch since the last take (0 when the helper is ahead); `take_fetch_s()` is
    the helper's own mean seconds per batch (loader `next` + the host side of the H2D copy) since the
    last take. `ahead=True` runs the helper thread on a CPU device too (tests); on CPU it is otherwise
    off and both clocks time the caller's own `next`."""

    def __init__(self, src, dev, side=True, stop=None, ahead=None):
        self.src, self.dev, self.side, self.stop = src, dev, bool(side), stop
        self.ahead = dev.type == "cuda" if ahead is None else bool(ahead)
        self._it = self._fut = self._worker = None
        self._closed = False
        self.clean = None           # after close(): True when nothing was still running
        self._wait_s, self._fetch_s, self._fetch_n = 0.0, 0.0, 0

    def take_wait_s(self):
        """Seconds the caller blocked on the next batch since the last call."""
        w, self._wait_s = self._wait_s, 0.0
        return w

    def take_fetch_s(self):
        """The helper's mean seconds per batch (loader next + H2D copy) since the last call; None if
        no batch arrived."""
        s, n = self._fetch_s, self._fetch_n
        self._fetch_s, self._fetch_n = 0.0, 0
        return s / n if n else None

    def _fetch(self, stream):
        t = time.perf_counter()
        try:
            item = next(self._it)
        except StopIteration:
            return None
        b = _batch(item)
        if stream is None:
            return {k: (v.to(self.dev, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in b.items()}, None, time.perf_counter() - t
        with torch.cuda.stream(stream):
            d = {k: (v.to(self.dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            ev = torch.cuda.Event()
            ev.record(stream)
        return d, ev, time.perf_counter() - t

    def __iter__(self):
        self._it = iter(self.src)
        if not self.ahead:
            while not self._closed:
                t = time.perf_counter()
                try:
                    item = next(self._it)
                except StopIteration:
                    return
                dt = time.perf_counter() - t
                self._wait_s += dt
                self._fetch_s += dt
                self._fetch_n += 1
                yield item
            return
        stream = torch.cuda.Stream(self.dev) if self.side and self.dev.type == "cuda" else None
        self._worker = _Fetcher()
        fetch = self._fetch
        self._fut = self._worker.submit(lambda: fetch(stream))
        while not self._closed:
            got = self._wait()
            if got is None or self._closed:
                self._fut = None
                return
            self._fut = self._worker.submit(lambda: fetch(stream))
            cur, ev, dt = got
            got = None
            self._fetch_s += dt
            self._fetch_n += 1
            if ev is not None:
                torch.cuda.current_stream(self.dev).wait_event(ev)
                for v in cur.values():
                    if torch.is_tensor(v) and v.is_cuda:
                        v.record_stream(torch.cuda.current_stream(self.dev))
            yield cur

    def _wait(self):
        """The fetch in flight, polled every second so a `stop()` request (the RAM guard) ends the
        iteration even while a loader is stuck: the pending fetch is then abandoned, not joined."""
        import concurrent.futures as cf
        t = time.perf_counter()
        try:
            while True:
                try:
                    return self._fut.result(timeout=1.0)
                except cf.TimeoutError:
                    if self.stop is not None and self.stop():
                        return None
        finally:
            self._wait_s += time.perf_counter() - t

    def close(self, timeout=QUIESCE_S):
        """Quiesce: no further batch, the fetch in flight finished (or given up on after `timeout`
        seconds -- its daemon thread is then abandoned, never joined), the loader's workers shut down.
        Idempotent. Returns True when nothing was left running."""
        if self._closed:
            return bool(self.clean)
        self._closed = True
        clean = True
        fut, self._fut = self._fut, None
        if fut is not None:
            import concurrent.futures as cf
            try:
                fut.result(timeout=max(float(timeout), 0.0))
            except cf.TimeoutError:
                clean = False
            except Exception:  # noqa: BLE001  -- a failing last fetch is not the transition's problem
                pass
        w = getattr(self, "_worker", None)
        if w is not None:
            w.stop()                # after the fetch in flight; a stuck one keeps its daemon thread
        _shutdown_loader(self.src, self._it)
        self.clean = clean
        return clean


def _r3(x):
    return None if x is None else round(float(x), 3)


def to_device_iter(src, dev, side=True, stop=None):
    """`DevicePrefetch(src, dev, side, stop)`: the name the step loop has always used."""
    return DevicePrefetch(src, dev, side=side, stop=stop)


class _Phases:
    """Per-phase milliseconds of a training step, for `RVSM_PROFILE=1` only: every mark synchronises the
    device, which costs throughput, so it is off in production. `mark(name)` closes the phase that
    started at the previous mark; `take()` returns the per-step means since the last take."""

    def __init__(self, dev, on):
        self.on = bool(on)
        self.dev, self.acc, self.n, self.t, self.peak = dev, {}, 0, None, {}

    def start(self):
        if self.on:
            if self.dev.type == "cuda":
                torch.cuda.synchronize(self.dev)
                torch.cuda.reset_peak_memory_stats(self.dev)
            self.t = time.perf_counter()

    def mark(self, name):
        if not self.on or self.t is None:
            return
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
            pk = torch.cuda.max_memory_allocated(self.dev) / 2 ** 30
            self.peak[name] = max(self.peak.get(name, 0.0), pk)   # GB high-water within this phase
            torch.cuda.reset_peak_memory_stats(self.dev)
        now = time.perf_counter()
        self.acc[name] = self.acc.get(name, 0.0) + (now - self.t) * 1e3
        self.t = now

    def note(self, d):
        """Add externally timed pieces (ms) that sit INSIDE a phase (the cascade's own clock)."""
        for k, v in (d or {}).items():
            self.acc[k] = self.acc.get(k, 0.0) + float(v)

    def step(self):
        self.n += 1

    def take(self):
        if not self.on or not self.n:
            return {}
        out = {"ms": {k: round(v / self.n, 1) for k, v in self.acc.items()},
               "peak_gb": {k: round(v, 2) for k, v in self.peak.items()}}
        self.acc, self.n, self.peak = {}, 0, {}
        return out


def _sync_net(net, ema):
    """Copy the EMA state into `net` in place (one pair of foreach kernels, as `Cascade.sync`)."""
    pairs = [(v, ema[k]) for k, v in net.state_dict().items() if k in ema]
    if pairs:
        torch._foreach_copy_([a for a, _ in pairs], [b for _, b in pairs])


def _ov_log(acc, w):
    """The overlap term's train.jsonl fields over the microbatches since the last row: `overlap` (the
    unweighted term's mean over the microbatches that had one), its parts, the scored voxel share,
    `overlap_n` (how many had one) and the weight. Nothing when the term is off."""
    if w <= 0:
        return {}
    n = int(acc.get("n", 0))
    out = {"overlap_n": n, "w_overlap": w}
    if n:
        out.update({k: round(acc[k] / n, 6) for k in ("overlap", "overlap_kl", "overlap_l1", "overlap_vox")})
    return out


def train(cfg, out=None, init=None, resume=False, patches_factory=None, device=None, val_items=None,
          steps=None, accum=1, hook=None, ckpt=None, stop_now=None):
    """Train the student. Returns the checkpoint path.

    `patches_factory()` returns a fresh iterable of samples: either collated batches (what
    `sample.loader` yields) or bare `sample.rung_item` dicts, which are batched to one here. Nothing in
    the loop knows where they came from, so a test can hand it a generator.

    `val_items` is the held-out grid (a list of rung_items, from `sample.val_grid`); without one the
    evaluation, the PNG and the calibration are skipped and only the checkpoint is written.

    `hook(info)` is called after every evaluation+checkpoint, with `{step, net, opt, ema, temps, out,
    ckpt, kind}`; returning something truthy ENDS the loop (the checkpoint is already on disk). `kind`
    is "eval" at an evaluation boundary, "ckpt" at a checkpoint-only one (`cfg.ckpt_every`: no
    evaluation ran, so no gate may read one) and "stop" after the RAM guard's checkpoint (the loop is
    over whatever the hook returns). That is the
    one seam `rvsm/run.py` needs: the state file, the STOP marker, the timeshare phase swap and the
    verso / round gates all live there and none of them belong in the step loop. `ckpt` overrides where
    the checkpoint is written (the driver keeps it at `<out>/ckpt/student.pt`, beside the round
    teachers).
    """
    out = Path(out or cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    layout = cfg.layout()
    nsteps = int(cfg.steps if steps is None else steps)
    grid = val_items if val_items is not None else []   # a list, or a `sample.DiskGrid` (never list()ed:
                                                         # that would load a spilled grid whole)

    gnb = bool(getattr(cfg, "gn_bf16", False))
    net = M.build(cfg.size, cin=layout.cin, cout=layout.cout, ckpt_act=cfg.ckpt_act, gn_bf16=gnb,
                  verbose=False).to(dev)
    newp = set()
    if init:
        st = torch.load(init, map_location="cpu", weights_only=False)
        sd = st.get("ema") or st.get("model") or st
        src, newp = warm_start(sd, net, layout, src_layout=st.get("layout"))
        miss = net.load_state_dict(src, strict=False)
        print(f"[train] warm start from {init}: {len(src)} tensors copied, {len(newp)} new "
              f"({len(miss.missing_keys)} missing, {len(miss.unexpected_keys)} unexpected)", flush=True)

    groups, split = param_groups(net, new_param_names(newp, net), cfg.new_param_lr_mult)
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=0.01, fused=(dev.type == "cuda"))
    warm = int(cfg.rewarm) if init else int(cfg.warmup)
    S, C = wsd_stable_until(nsteps, cfg.cooldown)
    base = lr_lambda(nsteps, warm, sched=cfg.sched, stable_until=S, cooldown=C)
    if split:   # the new rows carry no memory to protect: M x LR through the plateau, then M = 1
        mult = float(cfg.new_param_lr_mult)
        end = S if str(cfg.sched) == "wsd" else nsteps
        base = [base, (lambda s, f=base, m=mult, e=end: f(s) * (m if s < e else 1.0))]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, base)
    decay = ema_auto(nsteps, cfg.ema_k) if str(cfg.ema) == "auto" else float(cfg.ema)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    step, temps = 0, {}

    ck = Path(ckpt) if ckpt else out / "ckpt.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    if resume and ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        got = CFG.stored_fingerprint(st.get("cfg") or {})
        assert got == cfg.fingerprint(), \
            f"resume: the checkpoint's config fingerprint is {got}, this run's is {cfg.fingerprint()}"
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        ema = {k: v.to(dev) for k, v in st["ema"].items()}
        step, temps = int(st["step"]), dict(st.get("temps") or {})
        for _ in range(step):
            sched.step()
        print(f"[train] resumed {ck} at step {step}", flush=True)
    step0 = step                      # the checkpoint this process started from (a resume's step)

    # CASCADE: one `Cascade` builds the TRAINING channel (stochastic: mix / dropout / noise), another
    # the validation one (deterministic: self, no noise, no dropout). The self source runs its own copy
    # of the net on the EMA weights -- never the compiled module, and never with a grad path.
    evnet = M.build(cfg.size, cin=layout.cin, cout=layout.cout, gn_bf16=gnb, verbose=False).to(dev)
    casnet = None
    if cfg.cascade in ("self", "mix"):
        casnet = M.build(cfg.size, cin=layout.cin, cout=layout.cout, gn_bf16=gnb, verbose=False).to(dev)
        casnet.eval()
    casfwd = None
    if casnet is not None and cfg.compile and dev.type == "cuda":
        casfwd = torch.compile(casnet, mode="max-autotune-no-cudagraphs", dynamic=False)
    cas = prep.Cascade(cfg.cascade, self_p=cfg.self_p_lo, drop=cfg.cascade_drop,
                       noise=cfg.cascade_noise, net=casnet, fwd=casfwd)
    # the evaluation net runs compiled too (its forward and its cascade self pass): eager, the grid's
    # evaluation was ~2 s a window on the A100, ~7 min for four held-out regions
    evfwd = evnet
    if cfg.compile and dev.type == "cuda":
        evfwd = torch.compile(evnet, mode="max-autotune-no-cudagraphs", dynamic=False)
    casval = prep.Cascade(eval_cascade_mode(cfg.cascade), self_p=1.0, drop=0.0,
                          noise=False, net=evnet, fwd=(evfwd if evfwd is not evnet else None))
    casmask = prep.Cascade("mask", self_p=0.0, drop=0.0, noise=False) \
        if cfg.cascade in ("self", "mix") else None
    # OVERLAP-CROP CONSISTENCY (`rvsm.overlap`, off unless `loss_overlap` > 0): the EMA forward of an
    # overlap draw's second window. It is the cascade's self net when there is one (already on the EMA
    # weights every step, and compiled); otherwise its own copy, synced when used.
    ov_w = float(getattr(cfg, "loss_overlap", 0.0) or 0.0)
    ovnet = ovfwd = None
    if ov_w > 0:
        if casnet is not None:
            ovnet, ovfwd = casnet, (casfwd if casfwd is not None else casnet)
        else:
            ovnet = M.build(cfg.size, cin=layout.cin, cout=layout.cout, gn_bf16=gnb, verbose=False).to(dev)
            ovnet.eval()
            ovfwd = torch.compile(ovnet, mode="max-autotune-no-cudagraphs", dynamic=False) \
                if cfg.compile and dev.type == "cuda" else ovnet
    ov_acc = {}

    acfg = _aug_cfg(cfg, out)
    cmode = train_compile_mode() if cfg.compile else None
    if cfg.compile:
        nthr = train_compile_threads()
        model = torch.compile(net, **({"mode": cmode} if cmode else {}))
        if cmode or nthr:
            print(f"[train] torch.compile mode={cmode or 'default'} compile_threads={nthr or 'env'}",
                  flush=True)
    else:
        model = net
    # CUDA graphs (max-autotune / reduce-overhead): a step's outputs live in the graph's static pool,
    # overwritten by the next replay; marking each forward as a new step tells the cudagraph trees so
    graphs = bool(cmode and uses_cudagraphs(cmode) and dev.type == "cuda")
    if graphs:
        # ... and the gradients must live OUTSIDE that pool: a .grad the compiled backward allocated
        # during capture is overwritten by the next replay, which breaks the accumulation (torch
        # raises "accessing gradient tensor output of CUDAGraphs that has been overwritten"). Stable
        # zeroed buffers from here on, zeroed in place (`set_to_none=False`). Every parameter of the
        # net receives a gradient every step, so a zero buffer changes nothing the optimiser sees.
        for p_ in net.parameters():
            if p_.requires_grad and p_.grad is None:
                p_.grad = torch.zeros_like(p_)
    aux_on = bool(cfg.loss_excl or cfg.loss_selfcons or cfg.loss_skel or cfg.loss_affinity
                  or cfg.loss_skel_prec)
    aux_dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

    bad_run, bad_total = 0, 0

    saved = {"step": None}

    def save():
        tmp = ck.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "ema": ema, "opt": opt.state_dict(), "step": step,
                    "cfg": cfg.to_json(), "layout": layout.to_json(), "temps": temps}, tmp)
        if saved["step"] != step:        # a second save at the same step must not overwrite _prev
            keep_prev(ck)
        tmp.replace(ck)
        saved["step"] = step

    ckpt_every = max(int(getattr(cfg, "ckpt_every", 0) or 0), 0)

    def checkpoint(kind):
        """`save()` at a boundary, logged as a `ckpt` line with its wall seconds."""
        tc = time.time()
        save()
        _log(str(out / "logs" / "train.jsonl"),
             {"kind": "ckpt", "step": step, "at": kind, "s": round(time.time() - tc, 2)})

    panels = {}                      # one validation panel per held-out region, chosen once per run
    pngw = _PngWorker(str(out / "logs" / "train.jsonl"))    # the PNGs render while training resumes

    def do_eval():
        nonlocal temps
        if not grid:
            return
        evnet.load_state_dict(ema)
        te = time.time()
        kept = {} if cfg.calibrate else None
        cost = {}
        cap = PanelCapture(layout, _verso_on(out / "eval" / "val.png"), panels=panels.get("p"),
                           region=int(cfg.region))
        rec = {"step": step, **evaluate(evfwd, grid, dev, layout, cascade=casval, calib_keep=kept,
                                        cost=cost, capture=cap)}
        if casmask is not None:
            # the same fine windows with the MASK cascade source (the pooled target, no noise): the
            # trajectory of what the model does with a good coarse prediction, beside the self-cascade
            # one the gates read. A leak by construction, so an upper bracket, never a gate input.
            tm = time.time()
            mk = evaluate(evfwd, grid, dev, layout, cascade=casmask, rungs=FINE_RUNGS, band=False)
            rec.update({"dice_mask": mk["dice"], "bce_mask": mk["bce"],
                        "dice_best_mask": mk.get("dice_best"),
                        **{f"dice_mask_r{k}": mk[f"dice_r{k}"] for k in FINE_RUNGS
                           if f"dice_r{k}" in mk}})
            rec["mask_s"] = round(time.time() - tm, 1)
        t_ev = time.time() - te
        # the PNGs: drawn from the evaluation's own forwards (`PanelCapture`) and rendered on a
        # background thread (`_PngWorker`). `val_png` is now only the hand-off (plus the wait for the
        # previous render, were it still running); `val_png_bg` is the PREVIOUS evaluation's render
        # seconds (this one's is not known yet)
        if cap.broken is None:
            (out / "eval").mkdir(parents=True, exist_ok=True)
            panels["p"] = cap.finish()
            pngw.submit(step, cap, out)
        del cap
        t_png = time.time() - te - t_ev
        if cfg.calibrate:        # on the evaluation's own logits: the grid is not run a second time
            from rvsm import calib
            temps = {str(k): v for k, v in
                     calib.run(evnet, None, layout=layout, per=calib.stack(kept)).get("temps", {}).items()}
            rec["temps"] = dict(temps)
        rec["eval_s"] = {"evaluate": round(t_ev, 1), "val_png": round(t_png, 1),
                         "calibrate": round(time.time() - te - t_ev - t_png, 1),
                         "band": cost.get("band", 0.0),   # inside `evaluate`: the recto band metrics
                         "val_png_bg": pngw.last_s}
        _log(str(out / "logs" / "eval.jsonl"), rec)

    _log(str(out / "logs" / "train.jsonl"),
         {"step": step, "size": cfg.size, "patch": cfg.patch, "batch": cfg.batch, "aug": cfg.aug,
          "cin": layout.cin, "cout": layout.cout, "steps": nsteps, "ema": decay,
          "fingerprint": cfg.fingerprint()})

    ph = _Phases(dev, os.environ.get("RVSM_PROFILE", "") not in ("", "0"))
    t0, s0, micro, rung_n = time.time(), step, 0, {}
    src = patches_factory() if patches_factory is not None else iter(())
    def stopping():
        return stop_now() if stop_now is not None else None

    src = to_device_iter(src, dev, side=bool(getattr(cfg, "gpu_prefetch", True)), stop=stopping)
    why_stop = None
    loss = bce = dice = None
    reg_log, aux_log = {}, {}
    nvox = 0
    for item in src:
        if step >= nsteps:
            break
        # the RAM guard is checked before every microbatch -- so also after the non-finite-gradient
        # `continue` -- and before every evaluation / hook, never only after them (pass-4 P4-02)
        why_stop = stopping()
        if why_stop:
            break
        ph.start()
        b = _batch(item)
        ph.mark("h2d_wait")
        ks = _rungs_of(b)
        for r in ks:
            rung_n[r] = rung_n.get(r, 0) + 1
        if cas.on:
            cas.sync(ema)   # the self-mode coarse pass always runs on the current EMA weights
            # the schedule means something only for "mix"; "self" is 1.0 (every non-dropped sample),
            # "mask" 0.0 -- and that is what the log says
            cas.self_p = self_p_at(cfg, step, nsteps) if cas.mode == "mix" else \
                (1.0 if cas.mode == "self" else 0.0)
        cas.clock = ph.on
        ov_now = ovnet is not None and "ov_ct" in b
        cas.keep_coarse = ov_now
        ct, tg, wt = prep.prepare(b, dev, cascade=cas, layout=layout)
        ph.mark("prepare+cascade")
        ph.note(cas.take_clock())
        ovE = None
        if ov_now:
            if ovnet is not casnet:
                _sync_net(ovnet, ema)
            ovE = OV.teacher_batch(b, ovfwd, layout, dev, cas=cas if cas.on else None,
                                   q=int(getattr(cfg, "overlap_sub", 0) or 0))
            ph.mark("overlap_ema")
        cas.last_coarse = {}
        casch = ct[:, layout.i_cas:layout.i_cas + 1].detach().clone() if cas.on else None
        # the weights ride along as extra target channels, so the geometric augs transform them
        # identically to the fields they weigh -- and so does an overlap draw's EMA field
        nt = int(tg.shape[1])        # cout_t, + the trailing ROUTING row under a teacher route
        ct, tgw = A.apply(ct, torch.cat([tg, wt] + ([ovE] if ovE is not None else []), 1), acfg,
                          nimg=layout.i_cas, rung=ks)
        tg, wt = tgw[:, :nt], tgw[:, nt:2 * nt]
        ovE = tgw[:, 2 * nt:] if ovE is not None else None
        # ---- per-rung teacher ROUTING (`config.route_spec`): the routing row leaves the batch here, and
        # at a routed rung-2 sample the recto target / weight become the gap-fill ones (`losses.route_*`)
        w_cont, route_m, tb = None, None, None
        orig0 = (tg[:, :1], wt[:, :1])
        if nt > layout.cout_t:
            tb, wb = tg[:, layout.cout_t:layout.cout_t + 1], wt[:, layout.cout_t:layout.cout_t + 1]
            tg, wt = tg[:, :layout.cout_t], wt[:, :layout.cout_t]
            route_m = L.route_masks(tb, wb, dilate=int(cfg.band_dilate))
            tg, wt, w_cont = L.route_apply(tg, wt, tb, route_m)
        # ---- the THINNED-BAND target (`thin_band`, docs/recipe.md §6): at rung 2 the recto row's wide
        # m7 band becomes a `thin_band_width`-voxel sheet about its medial surface, the band's flanks
        # weight 0; under routing only the gap (and unrouted voxels) -- computed here, on the device
        thin_sel = [i for i, k in enumerate(ks) if k == 2] if int(cfg.thin_band) else []
        if thin_sel:
            w_pre = wt[thin_sel, :1] > 0
            tg, wt = L.thin_apply(tg, wt, thin_sel, width=float(cfg.thin_band_width),
                                  soft=float(cfg.thin_band_soft), orig=orig0, tb=tb, m=route_m)
            w_post = wt[thin_sel, :1] > 0
            # thin_zero: share of the recto weight the transform ZEROED (the band's flanks where the
            # weight was > 0: the standalone / unrouted path; a routed gap was 0 already, so under full
            # routing it is 0.0). thin_gain: share of the recto weight after the transform that it
            # ADDED (the routed gap's direct target). thin_gap_w: share of the gap voxels that now
            # carry a direct term (the rest are the m7 band's flanks, left at weight 0)
            thin_log = {"thin_zero": float((w_pre & ~w_post).sum() / w_pre.sum().clamp_min(1)),
                        "thin_gain": float((~w_pre & w_post).sum() / w_post.sum().clamp_min(1))}
            if route_m is not None:
                g_ = route_m["gap"][thin_sel] > 0
                thin_log["thin_gap_w"] = float((g_ & w_post).sum() / g_.sum().clamp_min(1))
        del orig0
        ct = ct.to(memory_format=M.memfmt())
        ph.mark("aug")
        nvox += int(np.prod(ct.shape[2:])) * ct.shape[0]

        if net.ckpt_act and not ct.requires_grad:
            ct.requires_grad_()    # here, not in forward(): requires_grad_ inside it is a graph break
        if graphs:
            torch.compiler.cudagraph_mark_step_begin()
        with prep.autocast(dev):
            pred = model(ct)
            outs = [o.float() for o in pred] if isinstance(pred, (list, tuple)) else [pred.float()]
            bce, dice = L.deep_losses(outs if len(outs) > 1 else outs[0], tg, wt, layout=layout)
        loss = bce + float(cfg.loss_prob_dice) * dice
        y0 = outs[0]
        reg_log = {}
        ph.mark("forward+deep_losses")

        # ---- PHASE B: the signed midline distance, the thickness, the Eikonal regulariser
        d = y0[:, layout.i_mid:layout.i_mid + 1]
        td = tg[:, layout.i_mid:layout.i_mid + 1]
        wd = L.dist_weight(wt[:, layout.i_mid:layout.i_mid + 1])
        lv = y0[:, layout.i_log:layout.i_log + 1]
        r = {"sdist": cfg.loss_sdist * L.sdist_loss(d, td, wd, logvar=lv)}
        if cfg.loss_eikonal:
            r["eikonal"] = cfg.loss_eikonal * L.eikonal(d, wd, tgt=td)
        th = L.soft_thickness(y0[:, layout.i_thick:layout.i_thick + 1])
        r["thick"] = cfg.loss_sdist * L.thickness_loss(
            th, tg[:, layout.i_thick:layout.i_thick + 1],
            L.dist_weight(wt[:, layout.i_thick:layout.i_thick + 1]))
        # ---- PHASE C: `--pair construct`. The pair is BUILT from (midline, thickness), never crossed,
        # and scored against the same probability targets as the learned rows -- which is what makes the
        # two faces non-overlapping by construction rather than by a penalty.
        # The pair is scored only where PAIRED support exists -- the midline target has weight there
        # (a finished verso and its fields): elsewhere its gradient trained midline / thickness against
        # geometry nobody measured and competed with the recto head in round 0.
        paired = (wd > 0).to(wt.dtype)
        reg_extra = {"pair_support": float(paired.mean())}
        if thin_sel:
            reg_extra.update(thin_log)          # thin_zero / thin_gain (/ thin_gap_w): see above
        if layout.nprob >= 2:
            lr_, lv_ = L.pair_logits(d, th, half=cfg.pair_band, tau=cfg.pair_tau)
            pb_, pd_ = pair_terms(lr_, lv_, tg[:, :2], wt[:, :2], paired)
            r["pair_bce"], r["pair_dice"] = float(cfg.loss_pair) * pb_, float(cfg.loss_pair) * pd_
        if route_m is not None:
            # the gap-fill: nothing predicted outside the dilated base band where the fine teacher is
            # not confident; the routed shares say whether gap zones dominate (regression to blur)
            nr = route_m["routed"].sum().clamp_min(1.0)
            reg_extra.update({"route_vox": float(route_m["routed"].mean()),
                              "route_ca": float(route_m["ca"].sum() / nr),
                              "route_gap": float(route_m["gap"].sum() / nr),
                              "route_out": float(route_m["outside"].sum() / nr)})
            if float(cfg.loss_band) > 0:
                r["band"] = float(cfg.loss_band) * L.band_penalty(
                    torch.sigmoid(y0[:, :1]), route_m["outside"], eps=float(cfg.band_eps))
        for k_, v_ in r.items():
            loss = loss + v_
        reg_log = {k_: float(v_.detach()) for k_, v_ in r.items()}
        reg_log.update(reg_extra)

        ph.mark("sdist+eikonal+thick+pair")
        # ---- OVERLAP-CROP CONSISTENCY against the EMA's view of the second window (`rvsm.overlap`)
        if ovE is not None:
            o = L.overlap_loss(y0, ovE, wd, L.dist_weight(wt[:, layout.i_thick:layout.i_thick + 1]),
                               layout=layout)
            loss = loss + ov_w * o["overlap"]
            for k_, v_ in o.items():
                ov_acc[k_] = ov_acc.get(k_, 0.0) + float(v_.detach())
            ov_acc["n"] = ov_acc.get("n", 0) + 1
            del ovE
            ph.mark("overlap")
        # ---- the topology pilot, at ONE rung, on interior sub-blocks only
        if cfg.loss_ect:
            sel = [i for i, k in enumerate(ks) if k == int(cfg.ect_rung)]
            if sel:
                idx = torch.tensor(sel, device=y0.device)
                rec_p = torch.sigmoid(y0[:, :1])
                con_p = torch.sigmoid(L.pair_logits(d, th, cfg.pair_band, cfg.pair_tau)[0]) \
                    if layout.nprob >= 2 else rec_p
                # ect_n is the number of DIRECTIONS (it was passed as the block count); the blocks
                # are drawn per sample from a generator seeded by the step AND the microbatch index
                # (accumulation microbatches draw different blocks), so a resume replays them
                # per BLOCK: the constructed band only where the whole block has paired support, the
                # learned recto head elsewhere (pass-4 P4-05)
                e = L.ect_loss(con_p[idx], tg[idx, :1], dirs=cfg.ect_n, block=cfg.ect_block,
                               nblocks=cfg.ect_blocks, w=wt[idx, :1],
                               gen=torch.Generator().manual_seed(ect_seed(step, micro)),
                               alt=rec_p[idx], support=paired[idx])
                loss = loss + cfg.loss_ect * e
                reg_log["ect"] = float(e.detach())

        ph.mark("ect")
        # ---- PHASE A auxiliaries: every term is computed from tensors this step already holds
        aux_log = {}
        if aux_on:
            cv = (lambda q: None if q is None else q.to(aux_dt))   # noqa: E731
            ax = L.aux_losses(cv(y0), cv(tg), cv(wt), layout,
                              w_excl=cfg.loss_excl, w_selfcons=cfg.loss_selfcons,
                              w_skel=cfg.loss_skel, w_affinity=cfg.loss_affinity,
                              skel_iters=cfg.skel_iters,
                              w_skel_prec=cfg.loss_skel_prec, skel_prec_iters=cfg.skel_prec_iters,
                              skel_prec_gate=cfg.skel_prec_gate,
                              cascade=cv(casch), cascade_self=cas.last_self, w_cont=cv(w_cont))
            if "aux" in ax:
                loss = loss + ax["aux"].float()
            aux_log = {k: float(v.detach()) for k, v in ax.items()}

        ph.mark("aux(excl,selfcons,skel,skel_prec,affinity)")
        (loss / max(int(accum), 1)).backward()
        ph.mark("backward")
        micro += 1
        if micro < int(accum):
            continue
        micro = 0
        if not grad_gate(net, 1.0):
            # a non-finite gradient never reaches the optimiser, the schedule or the EMA: the step is
            # skipped (its batch is spent), counted and logged; a run of them is a diverged run
            opt.zero_grad(set_to_none=not graphs)
            bad_run, bad_total = bad_run + 1, bad_total + 1
            _log(str(out / "logs" / "train.jsonl"),
                 {"kind": "nonfinite_grad", "step": step, "consecutive": bad_run, "total": bad_total})
            if bad_run >= NONFINITE_MAX:
                save()
                raise RuntimeError(f"train: {bad_run} consecutive non-finite gradients at step {step} "
                                   f"({bad_total} in all); the last good weights are in {ck}")
            continue
        bad_run = 0
        opt.step()
        opt.zero_grad(set_to_none=not graphs)
        sched.step()
        ema_update(ema, net, decay)
        ph.mark("clip+adamw+ema")
        ph.step()
        step += 1

        if step % 20 == 0:
            dt = max(time.time() - t0, 1e-6)
            _log(str(out / "logs" / "train.jsonl"),
                 {"step": step, "loss": float(loss.detach()), "bce": float(bce.detach()),
                  "dice": float(dice.detach()), **aux_log, **reg_log,
                  "lr": sched.get_last_lr()[0], "vox_s": round(nvox / dt),
                  "vram_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else 0,
                  "rung": {str(k): rung_n[k] for k in sorted(rung_n)},
                  "self_p": round(float(cas.self_p), 4) if cas.on else None,
                  "w_prob_dice": float(cfg.loss_prob_dice), "w_pair": float(cfg.loss_pair),
                  # train_wait_s: seconds the step loop BLOCKED on the prefetch over this row (true
                  # loader starvation); fetch_s: the helper's mean seconds per batch (loader + H2D);
                  # step_s: wall seconds per optimizer step over the row
                  "train_wait_s": round(src.take_wait_s(), 3), "fetch_s": _r3(src.take_fetch_s()),
                  "step_s": round(dt / max(step - s0, 1), 3), **_ov_log(ov_acc, ov_w), **ph.take()})
            rung_n, nvox, t0, s0 = {}, 0, time.time(), step
            ov_acc = {}
        at_eval = step % max(int(cfg.eval_every), 1) == 0 or step >= nsteps
        at_ckpt = ckpt_every > 0 and step % ckpt_every == 0
        if at_eval or at_ckpt:
            why_stop = stopping()
            if why_stop:
                break
            if at_eval:
                do_eval()
            # a checkpoint-only boundary (`ckpt_every`) saves exactly what an evaluation's does and
            # hands the hook `kind="ckpt"`: the driver writes the resume state (step, walk) and honours
            # STOP, but runs no evaluation, calibration or gate
            checkpoint("eval" if at_eval else "ckpt")
            # `quiesce` is the loader's shutdown protocol: a round transition calls it before it
            # resets the round's cursor state (review T19)
            if hook is not None and hook({"step": step, "net": net, "opt": opt, "ema": ema,
                                          "temps": temps, "out": str(out), "ckpt": str(ck),
                                          "quiesce": src.close,
                                          "kind": "eval" if at_eval else "ckpt"}):
                break
            t0, s0 = time.time(), step
    why_stop = why_stop or stopping()      # the prefetch ends its iteration when stop() is set
    if why_stop:
        # `stop_now()` -> a reason: the supervisor's RAM guard asks the trainer to leave NOW (a leak was
        # about to take the host down). `step` is a complete optimiser boundary (it only moves after
        # opt.step); a partial accumulation is DISCARDED. Checkpoint only when it is worth it (>= 200
        # steps since the last one: a save is ~1.4 GB of host copies at the worst moment).
        opt.zero_grad(set_to_none=not graphs)
        last = saved["step"] if saved["step"] is not None else step0
        do_save = step - int(last) >= STOP_SAVE_MIN
        _log(str(out / "logs" / "train.jsonl"),
             {"kind": "stop_now", "step": step, "reason": str(why_stop), "checkpoint": bool(do_save),
              "last_checkpoint": int(last), "discarded_microbatches": int(micro),
              "discarded_steps": 0 if do_save else int(step - int(last))})
        src.close(timeout=5.0)
        pngw.close(timeout=10.0)        # a render in flight is a few MB of numpy: it is not held for long
        if do_save:
            checkpoint("stop")
            # the driver records the resume state of THIS checkpoint (its step and walk): without it
            # state.json kept the last evaluation's walk and a resume replayed the steps in between
            if hook is not None:
                hook({"step": step, "net": net, "opt": opt, "ema": ema, "temps": temps,
                      "out": str(out), "ckpt": str(ck), "quiesce": src.close, "kind": "stop"})
        return str(ck)
    src.close()
    pngw.close()                        # the last evaluation's PNGs (STOP, a round's end, the last step)
    save()
    return str(ck)
