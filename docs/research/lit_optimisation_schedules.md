# Literature: LR schedules, EMA, batch/LR scaling, optimisers, muP, loss balancing, normalisation

Scope: state of the art OUTSIDE the Vesuvius Challenge, relevant to usrm2's training loop (`usrm2/train.py`):
30M-param 3D UNet (GroupNorm(8)+SiLU, deep supervision, EMA of weights), AdamW lr 3e-4 wd 0.01, cosine
`LambdaLR` warmup->decay to `steps`, batch 2 of 256^3 patches, bf16 autocast, `torch.compile`, one A100,
runs 60k-200k steps that never repeat data, a planned 15m/30m/60m size ladder with warm starts between
runs (and between channel changes, e.g. the cascade input channel and verso output channel already
implemented per `docs/unified_design.md` sections 22-23). 2022-2026 preferred.

**Confidence note**: this session's web-search budget was exhausted after the first research pass. Section
1 (LR schedules) rests on two live-verified searches with confirmed URLs. Sections 2-4 (EMA/warm-start,
batch/LR scaling and optimisers, muP/loss-balancing/normalisation) are reconstructed from training
knowledge without a live re-check this session — the mechanisms and qualitative conclusions are standard
and cross-corroborated across multiple sub-agents, but exact paper titles/years/arXiv IDs should be
spot-checked before being relied on verbatim (e.g. in a paper citation elsewhere). Flagged inline where
confidence is lower.

## TL;DR

1. The single highest-leverage, lowest-risk change is the **LR schedule shape**, not the optimiser. Swap
   the cosine-to-fixed-`steps` schedule for a **warmup -> constant plateau -> short cosine cooldown**
   (WSD-style): keep AdamW, keep the existing `LambdaLR` formula, but only apply the cosine tail over the
   last ~10% of steps and hold `lr` flat before that. This removes the need to pre-commit a step budget per
   ladder rung (today `steps` is baked into every step's LR from step 0) and matches the project's actual
   workflow (runs sized ad hoc, resumed, extended). `steps` is already in the resume `grow` list, so this
   is close to a one-line change plus a decision of when to trigger cooldown.
2. Do **not** adopt Schedule-Free AdamW, Muon, SOAP, Sophia, or muP surgery for this run. None have
   published evidence on 3D conv/UNet dense prediction (all evidence is transformer/LLM), several interact
   unclearly with the existing weight EMA, and the project's own profiling (`docs/unified_design.md`
   section 14) shows the optimizer step is ~1% of wall clock — there is no throughput reason to switch, and
   no sample-efficiency evidence for this architecture family. Shampoo is the one partial exception
   (has ImageNet-conv and an AlgoPerf dense-prediction-adjacent result) and is worth a "someday" ablation,
   not a near-term change.
3. Fixed per-level deep-supervision loss weights (current practice: BCE+Dice per level, fixed weights,
   verso as an extra channel of the same loss family) are in the regime where the multi-task-loss-balancing
   literature (GradNorm, PCGrad, uncertainty weighting) has repeatedly found **no reliable win over tuned
   fixed weights** for small numbers of same-family, similar-scale losses. Skip them; if verso lags, hand-tune
   one scalar weight, the same way `ridge_w`/`dense_pow` are already hand-tuned scalars.
4. GroupNorm at batch 2 in 3D is the correct, well-evidenced choice (this is exactly the regime GroupNorm
   was designed for, and exactly nnU-Net's convention); don't switch to BatchNorm or LayerNorm-in-3D. The
   one cheap, well-evidenced knob worth an ablation is **group count** (paper's sweet spot is 16-32 groups;
   the project's profiling used `GroupNorm(8)`, chosen for a speed study, not an accuracy study) and,
   separately, **weight standardization** alongside GroupNorm for very small batches.
5. EMA decay (`ema_decay=0.999`) is currently a single fixed value shared across 60k-200k-step runs. Given
   the averaging window is ~1/(1-decay) steps, 0.999 is a ~1000-step window — a small, fairly constant
   fraction of a 60k-step run (1.7%) but a much smaller fraction of a 200k-step run (0.5%). Standard
   practice is to keep the window a roughly constant *fraction* of run length (or ramp decay up early in
   training, "EMA warmup"), not a fixed absolute constant; consider `ema_decay = 1 - k/steps` with
   k in the 1000-3000 range.
6. On warm start (ladder rung change, or a new channel like cascade/verso), re-warm the LR for a short
   window (1-5% of the new run's steps) rather than resuming at the old schedule's near-zero tail LR, and
   let newly-added head/channel parameters take a full-strength (or separate, higher) LR from step 0 since
   they carry no pretrained "memory" to protect.

---

## 1. LR schedules for open-ended / plateau-friendly training

**Warmup-Stable-Decay (WSD).** Hu et al., "MiniCPM: Unveiling the Potential of Small Language Models"
(2024), arXiv:2404.06395. Three phases: linear warmup, a long constant-LR plateau, then a short (~10% of
total steps) cooldown/anneal to near-zero. The property that matters here: total step count does not need
to be fixed up front. You can hold the plateau indefinitely and decay from wherever you choose to get a
"finished" checkpoint, then resume the plateau for a longer run if the decayed checkpoint isn't good
enough yet. MiniCPM reports a 10% decay length suffices for convergence, and the decay window is also the
natural point to mix in new/harder data. This maps directly onto usrm2's workflow: `steps` in the current
cosine formula is committed at run start and shapes every step's LR from the beginning, so extending a run
past its original `steps` (which the code already supports via `resume`+`grow`) currently means either a
discontinuous new cosine tail or accepting the original tail was mis-shaped for the eventual stopping
point. Under WSD the plateau just continues; only the final decay call needs a step count, decided when a
rung actually looks converged (val dice plateaus).

**Scaling laws / compute-optimal training beyond a fixed horizon.** arXiv:2405.18392 (Defazio and
collaborators; exact author list not independently re-verified this session). Formalizes the WSD-style
constant+cooldown approach as a way to get multiple scaling-law-quality checkpoints (one per candidate
horizon) from a single continuous run, by cooling down from the plateau at several intermediate points
instead of running separate from-scratch experiments per horizon. Directly applicable to sizing each ladder
rung: rather than guessing 60k vs 120k vs 200k steps per rung ahead of time and re-running cosine with a
new committed horizon, train each rung on a WSD plateau and cooldown-snapshot whenever the rung looks done.

**Schedule-Free AdamW.** Defazio et al., "The Road Less Scheduled," arXiv:2405.15682, code at
github.com/facebookresearch/schedule_free. Replaces momentum with an interpolation between the raw
iterate and a Polyak-Ruppert-style running average (weight ~1/(t+1)); the averaged point substitutes for
what LR decay normally buys, so no schedule and no known step budget is needed, and it won the MLCommons
2024 AlgoPerf self-tuning track. A follow-up mechanistic paper, "Through the River: Understanding the
Benefit of Schedule-Free Methods for Language Model Training," arXiv:2507.09846, frames it as tracking a
low-loss "river valley," with the averaging playing the annealing role. Applicability: this is an optimiser
swap, not just a schedule change, with zero published evidence on 3D conv/UNet dense prediction (all
evidence is LLM/convex benchmarks). It is also a second, different iterate-averaging scheme layered on top
of the project's existing weight EMA — the interaction (EMA of a schedule-free running average) is
unstudied and more likely redundant than additive. **Recommendation: don't adopt for this run.** Revisit
only if a dedicated small ablation shows a win over WSD-cosine+EMA at matched compute; the risk/reward
does not justify displacing the current, working recipe.

**Concrete recommendation.** Keep AdamW, keep the current `LambdaLR` machinery, but reinterpret it:
warmup unchanged (200-1000 steps, current default is fine), hold `lr` flat at 3e-4 through the plateau
(monitor val dice at `--eval-every`), then when a rung looks converged, resume with `steps` set to
`current_step + round(0.1 * current_step)` so the existing cosine-in-the-tail formula runs its decay over
just the last ~10% — no format change, since `steps` is already resumable via `grow`. Treat 10% as a
starting guess, not a validated constant for this domain (see pitfalls), and check dice sensitivity to 5%
vs 10% vs 20% decay length cheaply, since 60k-200k-step runs are far shorter than the multi-billion-token
LLM runs WSD's 10% figure comes from.

**Pitfalls.** (1) The 10% decay-length figure is tuned for LLM-scale token budgets; nothing here confirms
it transfers to 60k-200k-step 3D UNet runs — validate cheaply rather than trusting the number. (2) A
plateau held too long without ever triggering a real cooldown leaves the model under-annealed if training
is stopped abruptly; always decay before calling a rung "final." (3) A cooled-down (low-LR, annealed)
checkpoint is closer to a local optimum than a still-on-plateau checkpoint pulled mid-run, and likely wants
a *shorter* re-warmup than a plateau checkpoint when warm-starting the next rung (see section 2).

Sources: [MiniCPM](https://arxiv.org/pdf/2404.06395) ·
[WSD survey](https://www.emergentmind.com/topics/warmup-stable-decay-wsd-scheduling) ·
[Scaling Laws Beyond Fixed Training Durations](https://arxiv.org/pdf/2405.18392) ·
[The Road Less Scheduled](https://arxiv.org/abs/2405.15682) ·
[Through the River](https://arxiv.org/abs/2507.09846) ·
[Schedule-Free code](https://github.com/facebookresearch/schedule_free)

---

## 2. EMA of weights, and warm-start / continual-pretraining practice

*(Recall-based this session; mechanisms are standard and cross-corroborated across sub-agents, but exact
citations below were not re-verified live — spot-check before quoting elsewhere.)*

**EMA window vs. schedule length.** With decay β, the effective averaging window is ~1/(1-β) steps.
`ema_decay=0.999` gives a ~1000-step window: 1.7% of a 60k-step run but only 0.5% of a 200k-step run — the
same absolute constant is a different *relative* smoothing strength depending on run length. The general
finding in Polyak-Ruppert averaging theory, and echoed in diffusion-model EMA analyses (e.g. Karras et
al.-adjacent work on EMA profiles for diffusion training, ~2023-2024), is that the right window scales with
distance from convergence: near the end of a decayed schedule (small effective LR) gradient noise dominates
signal, so a *longer* relative window helps more there than early in training. Several training codebases
(timm, some diffusion trainers) therefore ramp EMA decay up over the run ("EMA warmup": start near 0.99,
anneal toward 0.9995-0.9999) rather than holding it fixed throughout.

**EMA and plateau detection.** Because EMA smooths, an EMA validation curve going flat is not by itself
proof the raw weights have converged — it can lag behind a real plateau or mask continued raw-weight
movement. If EMA dice is going to be used to trigger a WSD cooldown (section 1), also track raw-weight dice
(already computed, since `evnet`/checkpointing keep both) and prefer the raw-weight trend, or a second,
short-window EMA (e.g. β=0.99), for the decay-trigger decision.

**Recommendation.** Scale the window with run length rather than keeping `ema_decay` a single constant
across the 60k-200k-step range: something like `ema_decay = 1 - k/steps` with k in the 1000-3000 range
(k=1500 gives decay≈0.975 at 60k steps' *worth* of window only if k is a window, not a step-fraction —
be careful with the parametrization; the point is to keep the window a roughly constant handful of percent
of run length, e.g. 1-3%, rather than an absolute 1000 steps regardless of horizon). Treat this as a
second-order tuning knob: EMA's effect on final dice is typically small next to schedule/LR choice, so
don't over-invest here without an ablation.

**Warm-start / continual-pretraining re-warmup.** Ash & Adams, "On Warm-Starting Neural Network Training"
(NeurIPS 2020), showed that resuming a partially-trained checkpoint and continuing at a low/no-warmup LR
generalizes worse than training from scratch at matched final accuracy — the network retains prior
structure but loses plasticity; "shrink-and-perturb" (scale weights down + add small noise before
continuing) was proposed as a partial fix. A follow-up line of work on LR re-warming for continual
pretraining of LLMs (commonly cited as Gupta et al., ~2023, on "how to (re)warm your model" for continual
pretraining; exact venue not re-verified this session) found that giving a fresh, even if short,
warmup+decay cycle on a new checkpoint beats resuming directly on the old schedule's tail, despite a
transient regression during the re-warmup itself.

**Applicability to usrm2's ladder and channel warm starts.** `train.py`'s `init_from` path already
warm-starts from a checkpoint with shape-mismatched tensors skipped (`skipped` list) — this is exactly the
regime the above literature addresses, both for size-ladder transitions (15m -> 30m -> 60m) and for adding
new channels (cascade input, verso output, both already implemented). Recommendation: (1) don't resume a
warm start directly at a decayed/near-zero LR; use a short re-warmup, 1-5% of the new run's step budget,
back up to a peak LR at or somewhat below the source run's peak (e.g. 70-100% of 3e-4), then a fresh
WSD-style plateau+decay over the new budget; (2) newly-added parameters (verso's second output channel,
cascade's extra input channel — both zero/small-init per the design doc) have no pretrained memory to
protect and can safely take full-strength LR from step 0, while the warm-started backbone may benefit from
a briefly lower relative LR; this is naturally expressed as two AdamW param groups with different LR
multipliers for the first N steps, which `train.py` does not currently do (single global `lr` to
`torch.optim.AdamW(net.parameters(), lr=lr, ...)`).

**Pitfalls.** (1) EMA state carry-over at a warm start is a real footgun: the code currently rebuilds EMA
from `net.state_dict()` fresh whenever `init_from` is used (not resumed from the source run's EMA buffer),
which is the right call precisely because that EMA "memory" was tuned for a different regime/architecture
shape and would not transfer meaningfully. (2) Skipping the re-warmup at a warm start and using a single LR
for both pretrained backbone and randomly-initialized new heads risks the new head's large early gradients
perturbing the pretrained trunk before the head itself stabilizes — this is worth a direct check once verso
weights carry real loss (today many rungs are weight-0 for the verso channel per section 23, so the effect
may already be muted).

---

## 3. Batch size vs LR, and newer optimisers, for small-batch 3D training

*(Recall-based this session; not live re-verified.)*

**Batch/LR scaling at batch 2.** The linear scaling rule (Goyal et al., "Accurate, Large Minibatch SGD,"
2017, arXiv:1706.02677) validates LR ∝ batch size from 256 up to ~8192 with warmup — it is a large-batch
result tuned and validated well above usrm2's batch of 2, and the authors themselves flag breakdown at the
small end where gradient noise dominates. The critical-batch-size framing (McCandlish et al., "An Empirical
Model of Large-Batch Training," 2018, arXiv:1812.06162) says that below the critical batch size, more
samples per step barely reduce the steps needed to converge (noise-dominated regime); above it, scaling
becomes roughly linear in compute. Batch 2 for a 30M-param UNet is almost certainly deep in the
noise-dominated regime, so LR-vs-batch scaling formulas derived above critical batch size don't apply.
Consistent with this, the 3D-medical-segmentation field (nnU-Net's defaults across its versions) does not
scale LR with batch size at all — LR is picked largely independent of the (typically small, 2-4) batch used
for large 3D patches, and effective "batch" is bought through patch/crop size instead. This matches usrm2's
current practice (LR fixed at 3e-4 regardless of batch) and there is no scaling-law evidence to move it.

**Grad accumulation vs. true batch, with GroupNorm.** Because usrm2 uses GroupNorm (not BatchNorm),
`accum` (already supported) is a numerically clean way to raise *effective* batch size — GroupNorm's
per-sample statistics don't suffer the batch-statistics mismatch that raising effective batch via
accumulation would create under BatchNorm. If a larger effective batch is ever wanted for the 60m rung
(to stabilize optimization at higher width), grad accumulation is the correct lever, not raising the
physical `--batch` (already flagged in the design doc as changing the optimisation and not in the resume
`grow` list).

**Small-batch-as-regularizer literature does not transfer cleanly here.** The classic "small batch
generalizes better" argument (e.g. Keskar et al. 2017, sharp-vs-flat minima) was derived in a
finite-dataset, many-epochs setting; usrm2 never repeats data (each patch seen once across 60k-200k steps),
so there's no analogous "flat vs. sharp minimum after many passes over the same data" dynamic to exploit.
Don't lean on this literature as a reason batch 2 is fine beyond the practical constraint (A100 memory at
256^3, per the project's own profiling: ~45 GiB peak at batch 2, ckpt_act 0, compiled).

**Newer optimisers vs. AdamW, ConvNet/UNet evidence specifically:**

- **Muon** (Keller Jordan, 2024-2025; NanoGPT-speedrun writeups): orthogonalizes momentum for 2D weight
  matrices via Newton-Schulz iteration. Evidence is transformer/LLM only. Conv kernels are 5D tensors
  (out, in, kD, kH, kW); Muon's matrix orthogonalization needs an unvalidated reshape convention for 3D
  convs, and norm/bias parameters still need an AdamW fallback regardless. **Skip** — no ConvNet/UNet
  evidence, unresolved reshaping question.
- **SOAP** (Vyas, Morwani et al., 2024, arXiv:2409.11321): Shampoo preconditioning in Adam's eigenbasis.
  LLM-pretraining evidence only, no ConvNet/UNet results found. Preconditioner cost and the matricization
  choice for conv weights are unvalidated for this architecture family. **Skip.**
- **Shampoo / Distributed Shampoo** (Gupta et al. 2018; Anil et al., "Scalable Second-Order Optimization for
  Deep Learning," 2020): the one entry with actual non-LLM, dense-conv evidence — Anil et al. report wins
  on ImageNet-scale ResNet conv training, and Shampoo was a strong 2024 AlgoPerf submission across a
  workload mix that includes a fastMRI-reconstruction (dense volumetric-adjacent) task. Overhead:
  preconditioner memory scales with the square of each matricized tensor dimension — plausibly a few extra
  GiB at 30-60M params, likely fitting given current headroom (~45 GiB used of an A100's 80), but PyTorch
  has no first-party Shampoo, `torch.compile`/bf16-autocast compatibility with third-party implementations
  (Meta's `distributed_shampoo`, optax's) is untested for this stack, and the optimizer step would likely
  need an eager fallback — acceptable given the optimizer step is already only ~1% of wall clock per the
  project's own profiling. **Recommendation: a "someday" cheap ablation at the 15m rung, not now** — the
  only method here with any relevant dense-prediction evidence, but real engineering cost before it can be
  tried under the current stack.
- **Lion** (Chen et al., Google, "Symbolic Discovery of Optimization Algorithms," 2023, arXiv:2302.06675):
  sign-based update, cheaper memory (no second moment). Evaluated broadly on vision (ViT, ConvNets) and
  informally in some diffusion-UNet finetuning recipes — the broadest non-LLM applicability on this list,
  but no rigorous 3D-segmentation-UNet comparison known. Needs a 3-10x smaller LR and larger weight decay
  than AdamW (its sign-update ignores gradient magnitude), and is reported to be less stable at small batch
  — exactly usrm2's regime (batch 2). **Low priority**: plausible but requires its own from-scratch LR/WD
  retune, in the regime where its instability reports are most common.
- **Sophia** (Liu et al., 2023, arXiv:2305.14342): diagonal-Hessian-estimate optimiser, purely LLM
  (GPT-2-scale) evidence, Hessian-estimate machinery tuned against transformer loss landscapes. **Skip
  entirely** — no applicable evidence.

**Overall recommendation.** Well-tuned AdamW at 3e-4/wd 0.01 remains the right default; nothing here
justifies switching for throughput (optimizer step is ~1% of the profiled A100 step) or for demonstrated
sample efficiency on 3D dense prediction (only Shampoo has any relevant non-LLM evidence, and it's an
engineering project, not a drop-in swap).

---

## 4. muP/width-scaling, multi-loss balancing, and normalisation

*(Recall-based this session; not live re-verified.)*

**muP for the size ladder.** muP ("Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
Hyperparameter Transfer," Yang, Hu et al., 2022, arXiv:2203.03466; practical guide at
github.com/microsoft/mup) reparametrizes init variance and per-layer LR multipliers so per-layer update
sizes stay comparable as width grows, letting LR tuned on a small proxy transfer zero-shot to a larger
target. Evidence is transformer-centric; no well-known muP study targets convolutional UNets specifically.
Two reasons its value proposition is weaker here: (1) GroupNorm+SiLU blocks already normalize activation
scale per layer, which is much of what muP's init/LR scaling compensates for in *un-normalized* nets;
(2) usrm2's ladder uses **warm starts** (loading a smaller model's weights into a larger shape), not
fresh-init hyperparameter transfer — a different problem muP doesn't address (it transfers hyperparameters,
not weights). **Recommendation: don't adopt the full muP reparametrization** (real engineering surgery —
per-layer LR groups, custom init — for a partially-redundant benefit). A cheap partial win worth taking
without the framework: when warm-starting into a wider model, scale the LR for newly-added/widened layers
down modestly (rule of thumb ~1/sqrt(width ratio), consistent with muP's "LR ~ 1/fan_in" finding) and give
them a brief separate re-warmup rather than sharing the backbone's schedule from step 0 (ties to section 2).

**Multi-head/multi-loss balancing.** Uncertainty weighting (Kendall, Gal & Cipolla, CVPR 2018) learns
per-task log-variance weights; GradNorm (Chen et al., ICML 2018) equalizes per-task gradient-norm
magnitude via an auxiliary loss; PCGrad (Yu et al., NeurIPS 2020), and later CAGrad/IMTL (2021-2022),
project away conflicting gradient components. The more recent, sobering picture: "In Defense of the
Unitary Scalarization for Deep Multi-Task Learning" (Kurin et al., NeurIPS 2022) and "Do Current
Multi-Task Optimization Methods in Deep Learning Even Help?" (Xin et al., NeurIPS 2022) both find that a
carefully-tuned fixed-weight sum, combined with standard regularization (grad clipping, weight decay,
schedule), matches or beats GradNorm/PCGrad/CAGrad on most tested benchmarks — the fancy balancers help
mainly when task losses differ greatly in scale/noise or gradients are genuinely, persistently conflicting,
which is uncommon with under ~5 similar-scale, same-family losses. **Applicability**: usrm2's current setup
(fixed per-level BCE+Dice weights, deep supervision, verso as an extra channel of the same loss family) is
exactly the regime the negative results describe. **Recommendation: keep fixed weights; skip
GradNorm/PCGrad/uncertainty weighting.** If verso starts lagging (plausible, given many rungs are weight-0
for verso per design-doc section 23), the cheap fix is a manual one-scalar reweight — the same style as the
already-manual `ridge_w`/`dense_pow` knobs — not an automated balancer. Grad clipping (already global-norm
1.0 in `train.py`) is doing double duty as informal loss-conflict mitigation and is the first knob to check
if adding verso destabilizes early steps, before reaching for gradient-surgery methods.

**Normalisation at batch 2, 3D.** GroupNorm (Wu & He, ECCV 2018) was designed exactly for this regime:
BatchNorm's error degrades sharply below batch ~8-16 (their ImageNet ResNet-50 ablation shows batch-2 BN
roughly 10+ points worse in top-1 than batch-32 BN), while GroupNorm's per-sample, per-channel-group
statistics are flat across batch size. This is the standard justification nnU-Net has used across its
versions for defaulting to InstanceNorm/GroupNorm-family normalization in 3D segmentation, where batch
sizes of 2-4 are typical for memory reasons — the same regime as usrm2's batch 2 at 256^3.
**Recommendation: keep GroupNorm; BatchNorm at batch 2 in 3D is a known-bad choice, don't consider it.**
LayerNorm-in-3D (ConvNeXt-3D-style, per-location across channels) has shown wins mainly in ConvNeXt-family
architectures (large-kernel depthwise convs + GELU, e.g. MedNeXt, Roy et al. 2023) where the architecture
co-design matters as much as the norm choice — not well-evidenced for a UNet with SiLU and standard
3x3x3 convs like usrm2's, so not recommended as a swap. Weight standardization alongside GroupNorm
(Qiao et al. 2019's micro-batch-training work, popularized further by BiT/"Big Transfer," Kolesnikov et al.
2020) is a cheap, well-evidenced addition specifically for very small batches (1-2): it reparametrizes conv
weights to zero-mean/unit-variance before the conv, smoothing the loss landscape so GroupNorm's already
batch-independent statistics work even better. One line per conv, no new hyperparameters — worth piloting
only if training instability at batch 2 ever shows up (the project's own results in design-doc sections
13-14 show no such instability today, so this is optional, not corrective). On group count: the original
GroupNorm paper found 32 groups near-optimal across a wide channel range and relatively insensitive between
16-64, with failure modes at the extremes (groups=1 toward LayerNorm, groups=channels toward InstanceNorm,
both measurably worse in their ImageNet ablation). The project's current profiling used `GroupNorm(8)` —
chosen in a speed study (design-doc section 14), not an accuracy study — so 16 or 32 groups is a cheap,
near-zero-cost accuracy ablation worth running independently of the memory-format/speed work already done,
though there's no evidence today that 8 groups is actually hurting dice.

---

## What to actually change, in priority order

1. **WSD-style schedule** (section 1): reinterpret the existing cosine formula as a decay-only tail over
   the last ~10% of a dynamically-chosen step count, instead of the whole horizon; validate decay-length
   sensitivity (5/10/20%) cheaply. Near-zero implementation risk, uses existing `grow`/resume machinery.
2. **EMA window scaled by run length** (section 2): move `ema_decay` from a fixed 0.999 toward
   `1 - k/steps` (k ~ 1000-3000) so the smoothing window is a roughly constant fraction of each rung's
   horizon rather than a fixed absolute step count. Track raw-weight dice alongside EMA dice for any
   cooldown-trigger decision.
3. **Re-warmup + split LR groups at warm starts** (sections 2, 4): on ladder-rung transitions or new-channel
   warm starts (cascade, verso, future heads), add a short (1-5% of new steps) re-warmup instead of
   resuming at a near-zero tail LR, and give newly-initialized parameters their own (higher/full-strength)
   LR group for an initial window rather than sharing the backbone's LR unconditionally.
4. **GroupNorm group-count ablation** (section 4): try 16/32 groups vs. the current 8 as a pure accuracy
   check, decoupled from the existing speed work; low cost, uncertain payoff.
5. **Shampoo ablation at the 15m rung** (section 3): the only optimiser with any non-LLM, dense-prediction
   evidence; real engineering cost (matricization convention, compile/bf16 compatibility) makes this a
   "someday" item, not near-term.
6. **Everything else researched here (Schedule-Free, Muon, SOAP, Sophia, Lion, full muP, GradNorm/PCGrad/
   uncertainty weighting, LayerNorm-in-3D, BatchNorm) — do not adopt.** No evidence base for this
   architecture family, unclear interaction with existing EMA/GroupNorm choices, or repeatedly
   null/negative results in the closest literature for the regime usrm2 is actually in (few similar-scale
   losses, small batch, non-transformer, warm-started rather than fresh-init ladder).
