# Literature survey: noisy labels, self-training and distillation, outside Vesuvius Challenge (2026-09-21)

Scope: state-of-the-art *outside* the papyrus-scroll community on learning a near-perfect student from
imperfect teachers, mapped onto usrm2's actual label sources (`docs/unified_design.md` sections 3-5,
21-23; `docs/research/synthesis_future_inputs_outputs.md` sections 4-5):

- **S1 — published masks**: binary/thresholded, no confidence, from an earlier upstream model (recto
  th0.45, m7 th0.2). Large (hundreds of GB), whole-scroll, systematic-bias risk (the old model's blind
  spots), weight 1.0.
- **S2 — our own teacher probability stores**: soft, two independently-trained lineages (2.4 um recto
  model, m7), same recto head, weight 1.0.
- **S3 — human-verified segment meshes**: small, high-trust, currently not yet wired into the target
  pyramid machinery — this survey treats it as the natural "gold"/clean anchor set.
- **S4 — the model's own predictions**: consumed two ways — as a cascade *input channel* (`--cascade
  self|mix`, section 22, already implemented) and, per the unified plan, as filtered noisy-student
  *training rounds* (not yet implemented).

Four parallel searches covered (a) noisy-label learning for dense prediction, (b) self-training/pseudo-
labelling for 3D medical and EM segmentation, (c) KD from soft/ensemble targets, (d) coarse-to-fine
cascades and exposure bias, (e) equivariance/TTA-consistency as a label-free quality signal, (f) small-
clean-set reweighting of a large noisy corpus. Citations below are as returned by those searches;
arXiv/DOI ids were not independently re-verified against the live index, so treat a stale-looking id as a
paraphrase of the paper title/venue, not a guaranteed resolvable link.

---

## (a) Noise-robust losses, sample selection, label refinement

| Method | What it does | Citation | Cost | Evidence | 3D dense use |
|---|---|---|---|---|---|
| Generalized Cross Entropy (GCE) | Interpolates CCE↔MAE via power q; bounded gradient on wrong labels | Zhang & Sabuncu, NeurIPS 2018, arXiv:1805.07836 | Trivial, one hyperparameter | CIFAR-10/100, 60-80% synthetic noise | Classification only; per-voxel BCE swap is mechanical |
| Symmetric CE (SCE) | CCE + bounded Reverse-CE term | Wang et al., ICCV 2019, arXiv:1908.06112 | Trivial | CIFAR, WebVision | Classification only |
| Bootstrapping / dynamic bootstrapping | Target = convex combo of noisy label and model's own current prediction | Reed et al. 2015 (arXiv:1412.6596); Arazo et al., ICML 2019, arXiv:1904.11238 | Trivial-low | ImageNet-noisy, CIFAR, near-clean at 80% noise with mixup | Direct voxel analog: self-distillation target per voxel |
| Early-Learning Regularization (ELR) | Regularizer pulls predictions toward an EMA of the model's own early-epoch outputs, suppressing later memorization | Liu, Niles-Weed, Razavian, Fernandez-Granda, NeurIPS 2020, arXiv:2007.00151 | Low: one EMA buffer, one loss term | CIFAR 80% noise, Clothing1M, WebVision; matches/exceeds label-correction SOTA | Reused in medical-seg noisy-label papers (below); "EMA of own predictions" is exactly what `--cascade self` already computes |
| Normalized-loss framework (APL, NCE+RCE) | Any loss can be made provably noise-robust by normalization; combine an active + a passive loss | Ma et al., ICML 2020 | Low | CIFAR, WebVision | Classification only |
| Co-teaching | Two nets, each feeds its small-loss (likely-clean) minibatch subset to the *other* for the update | Han et al., NeurIPS 2018, arXiv:1804.06872 | Medium: 2x model | MNIST/CIFAR/T-ImageNet, 50-90% noise | Classification; segmentation variant exists (arXiv:2104.13766) but not clearly 3D |
| JoCoR | Joint loss over two nets: supervised CE + agreement (JS-div) term, small-loss selection on the joint loss | Wei, Feng, Chen, An, CVPR 2020, arXiv:2003.02752 | Medium: 2 nets | CIFAR, Clothing1M, 80% noise | Classification; agreement-based selection maps naturally onto "two teacher heatmaps agree/disagree" |
| Confident Learning (cleanlab) | Estimates joint noisy/true label distribution from out-of-fold predicted probabilities; prunes/ranks label errors, no noise-rate prior needed | Northcutt, Jiang, Chuang, JAIR 2021, arXiv:1911.00068 | Low-medium: needs out-of-fold probabilities | CIFAR/ImageNet cleanup, beat 7 competing methods | Classification/tabular; the math generalizes per-voxel but no published dense-3D version |
| Mean-Teacher-Assisted Confident Learning | EMA mean-teacher + confident-learning-style pruning, aimed at *training-label* noise itself (not just pseudo-labels) | Xue et al. 2022, PubMed 35604969 | Medium | Cardiac/abdominal CT/MRI, simulated annotator noise | 2D/3D medical, voxel-level, directly on-topic |
| Adaptive Label Correction (ALC) | Mean-Teacher self-ensemble; refines noisy labels using disagreement across multiple perturbed views, confidence-weighted | 2025, arXiv:2503.12218 | Medium-high: multiple augmented passes + EMA teacher | Medical segmentation benchmarks vs. CE/GCE baselines | 2D/3D medical, voxel-level |
| GSD-Net (geometric-structural dual guidance) | Geometric-distance-aware + structural cues reweight voxel supervision, trust interior over noisy boundary | 2025, arXiv:2509.02419 | Medium: extra geometric branch | **BraTS2020, 3D brain MRI with simulated label noise** | Explicit 3D voxel segmentation |
| AIO2 | Online iterative re-estimation/correction of incomplete or noisy labels during training | 2024, arXiv:2403.01641 | Medium | Remote-sensing dense (raster) segmentation | 2D dense; the iterative-correction idea generalizes directly to 3D |

**Where "student beats teacher" evidence lives in this bucket**: Noisy Student (CVPR 2020, arXiv:1911.04252,
covered fully under (c)/(b) below) and Born-Again Networks (ICML 2018, arXiv:1805.04770 — same-capacity
student trained on an unmodified teacher's soft output still exceeds it, sequential generations compound)
are the two best-evidenced "student > teacher" results in the whole survey, and neither needs a dense/3D
task to make its mechanism argument (noise injected on the student side; soft targets, not hard).

**Mapping to our sources**:

| Source | Best-fit method(s) | Expected effect | Pitfall |
|---|---|---|---|
| S1 (published masks) | GCE/SCE-style bounded loss; AIO2-style online correction; confident-learning-style pruning using disagreement with S2 | Stops the network fully memorizing stale/wrong boundary voxels from the old model's blind spots | Binary masks carry no per-voxel confidence to seed a noise model — must be estimated from S2 disagreement or the model's own early-training predictions. Robust losses assume roughly i.i.d. class-conditional noise; our errors are *structured* (systematic gaps/merges at particular geometries), which is the harder, less-covered case — validate against S3, don't assume |
| S2 (two teacher stores) | JoCoR-style agreement weighting; treat the two stores as parallel "networks" for small-loss / agreement selection | Where both teachers agree confidently, high-trust soft target; where they disagree, downweight or flag | Two teachers built on correlated pipelines/data are not independent — agreement is not ground truth. Already partly handled in `unified_design.md` §3 (`weight` per source, band averaging) without an agreement signal; this is a concrete upgrade |
| S3 (verified meshes) | The clean anchor set for confident-learning-style noise-rate estimation, ELR-style periodic recalibration, or as the sole hyperparameter-selection set | Lets us *measure* rather than assume how much S1/S2 disagree with truth | Small N — likely too small for a full per-class transition matrix; use coarse (per-source, per-region) statistics, not per-voxel |
| S4 (self-training rounds) | ELR-style EMA-of-own-predictions regularizer against drift; Noisy Student / Born-Again recipe (inject noise on the student, not the label generator) | This is the most direct lever for exceeding S1/S2 quality | Confirmation bias: an unchecked self-training loop amplifies its own systematic failure modes (the two dominant ones on the board: `merge_frac 0.44`, `continuity 0.656`, per synthesis §0/§4). Needs external checks (S3, co-teaching-style cross-filtering), not a pure self-referential loop |

---

## (b) Self-training / noisy student / pseudo-labelling for 3D medical and EM segmentation

| Method | Description | Citation | Cost | Evidence | 3D |
|---|---|---|---|---|---|
| Noisy Student | Iterative teacher→student; teacher unnoised, student heavily noised (dropout, stochastic depth, aug); student becomes next teacher | Xie et al., CVPR 2020, arXiv:1911.04252 | High: iterative retrain loop, large unlabeled pool | ImageNet 88.4% top-1 (+2.0%), large robustness gains | 2D origin, architecture-agnostic, widely ported |
| UA-MT | EMA mean-teacher + MC-dropout voxelwise entropy; consistency loss gated to low-uncertainty voxels | Yu, Wang, Li, Fu, Heng, MICCAI 2019 | Low-medium: EMA model + N MC passes | **Left-atrium MRI (MICCAI 2018 Atrial Seg Challenge)**, large Dice gains at 10-20% labels | Native 3D |
| Double-uncertainty dual mean-teacher | Two mean-teacher branches (different views/scales), each gated by its own uncertainty, cross-consistency | arXiv:2303.05126 | Medium: 2x teacher | Cardiac/brain MRI | 3D |
| Cross-Teaching 3D↔2D | 3D and 2D nets pseudo-label each other; hard-soft confidence threshold for the 3D branch | Luo et al., MICCAI 2023, arXiv:2307.16256 | Medium: two model families | Sparse-annotation abdominal CT, Dice 82.67%, beats prior SOTA semi-sup | 3D+2D hybrid |
| FPL+ | Cross-modality UDA: pseudo-labels filtered by multiple reliability signals (entropy, atlas prior) before retraining | arXiv:2404.04971 | Medium | Cross-modality abdominal/cardiac CT↔MRI | 3D |
| SRPL-SFDA | Source-free domain adaptation: SAM under multiple perturbations, pseudo-labels kept only if SAM outputs agree; unreliable voxels get entropy-min regularization instead of hard supervision | Yang et al. 2025, arXiv:2506.09403 | Medium-high: foundation model + TTA branch | Fetal brain / prostate MRI cross-domain | 3D volumes (2D-slice SAM internally) |
| nnFilterMatch | Entropy-filtered pseudo-labels bolted onto stock nnU-Net in a single pass, no iterative retrain loop | 2025, arXiv:2509.19746 | Low-medium: single pass | Multiple nnU-Net benchmarks; matches/exceeds full supervision at 5-20% labels | Native 3D |
| Reliable-pseudo-label co-training | Two nets each emit confidence-thresholded pseudo-labels for the other | arXiv:2301.04465 | Low-medium | Left atrium, pancreas CT | 3D |
| Confident Learning for noisy *segmentation* labels | Joint noise-transition estimation to identify/prune mislabeled voxels in an existing noisy label set | Xue et al., MICCAI 2020 | Medium | Cardiac MRI, synthetic + real noise | 3D-applicable |
| nnU-Net "3D U-Net Cascade" | Stage 1 low-res 3D U-Net → stage 2 full-res 3D U-Net taking the upsampled stage-1 map as an extra input channel, cropped to ROI | Isensee et al., Nature Methods 2021; KiTS21 arXiv:2307.01984 | Low: standard config, no new model class | KiTS and general nnU-Net benchmarks; consistently beats single full-res net when FOV exceeds memory | **Native 3D, structurally identical to our cascade channel** |
| FFN + iterative bootstrapping | Recurrent seed-based 3D neurite segmentation; multi-pass bootstrapping (retrain on own corrected traces) + consensus agglomeration | Januszewski et al. (Google), arXiv:1612.02120, arXiv:1905.06236 | High: distributed infra, human-in-the-loop feedback | Zebra finch EM at petabyte scale; SNEMI3D superhuman (arXiv:1706.00120) | 3D EM connectomics, closest analog to a multi-round self-training + human-verification loop at our scale |
| Sparse-annotation bootstrapping (EM) | Sparse point/skeleton annotations bootstrap dense pseudo-labels, iteratively refined | PMC11195258 | Medium | EM connectomics | 3D — directly relevant given our small verified-mesh set is also sparse relative to the corpus |
| TTA-based active learning + self-training | TTA output variance used both for active-learning sample selection and pseudo-label reliability filtering, in one loop | arXiv:2308.10727 | Medium: N augmented passes, no extra model | Medical segmentation, large gains at low label budgets | 3D-applicable |
| TTA aleatoric uncertainty | Formalizes TTA as MC sampling over an acquisition/augmentation model; principled per-voxel uncertainty for pseudo-label gating | Wang et al., arXiv:1807.07356 | Low: inference-time only | Brain tumor / fetal MRI; catches overconfident errors plain test-time-dropout misses | 3D |

**Mapping to our pipeline**: this is the bucket with the strongest direct 3D-medical/EM evidence, and it
converges hard on one recommendation independent of our own synthesis doc: **gate self-training/S2-fusion
by agreement (UA-MT/SRPL-SFDA style), not by a single confidence threshold.** Concretely:

1. Fuse S2's two stores with **entropy/agreement gating**, not naive averaging — accept a voxel as
   high-trust only where the two teachers' pooled distribution has low entropy (UA-MT's MC-dropout gate,
   SRPL-SFDA's "consistency of multiple outputs," here substituting two independent networks for TTA
   views). Cheap: inference-time only, no retraining.
2. Add a **cascade cross-consistency loss**, not just the feed-forward channel we already have — a loss
   term between the fine stage's re-pooled output and the coarse stage's prediction (cross-teaching in
   scale, à la the 3D↔2D cross-teaching paper's hard-soft threshold). This is closely related to, and a
   stronger version of, **L4 (cascade self-consistency)** already queued in Phase A of the synthesis doc
   (`synthesis_future_inputs_outputs.md` §4) — the literature suggests making it bidirectional rather than
   one-way stop-grad.
3. Run S4 self-training rounds ordered as a **curriculum**: most-agreed regions (S1 ∩ S2 ∩ high-cascade-
   agreement) first, ambiguous/boundary regions later — this directly targets `merge_frac`/`continuity`,
   which are dominated by boundary and thin-structure failure, not core-region failure.
4. **Pitfall, stressed across nearly every paper in this bucket**: confirmation bias compounds fastest
   exactly where S1 and S2 *agree but are both wrong* (correlated errors from shared upstream data/
   architecture lineage). Agreement-based filtering cannot catch this — only S3 can, so S3 must stay
   influential (oversampled, not just used once) in every self-training round, not only at initialization.

---

## (c) Knowledge distillation: soft targets, temperature, ensemble teachers

| Method | Description | Citation | Cost | Evidence | 3D |
|---|---|---|---|---|---|
| Hinton KD (baseline) | Temperature-softened targets carry inter-class "dark knowledge" hard labels discard | Hinton, Vinyals, Dean, arXiv:1503.02531, 2015 | Negligible | ImageNet/MNIST classification | Trivial to port: same softmax-per-voxel |
| Structured KD for segmentation | Pixel-wise KL + *pairwise* (local affinity) + *holistic* (adversarial) distillation terms, because segmentation is a structured-output problem, not i.i.d. per-pixel classification | Liu et al., CVPR 2019 | Moderate: extra affinity/adversarial loss, no added inference cost | PASCAL VOC 83.6 mIoU, Cityscapes, ADE20K with a compact student | Pairwise term extends to 3D voxel affinities but pair count scales badly — needs local-window sampling |
| Structural & statistical texture KD | Distills texture/structural statistics between teacher/student feature maps, not just logits | Ji et al., arXiv:2305.03944 | Moderate | Segmentation benchmarks | Resolution-agnostic, usable in 3D |
| Multi-modal → mono-modal KD | Teacher sees more modalities/scale than student; soft output recovers missing information | arXiv:2106.09564 | Moderate | Medical segmentation | Directly analogous to distilling a whole-volume/multi-context teacher into a patch-limited student |
| Adaptive/sample-wise temperature KD | Fixed global T is suboptimal; per-sample (or per-voxel) T keeps soft-label entropy calibrated to local difficulty | arXiv:2605.20357 (2026 preprint) | Low | Cityscapes-style mIoU gains over fixed-T | Per-voxel T is a natural extension — boundary/low-SNR voxels could get a different T than confident interior voxels |
| Ensemble-then-distill / multi-teacher KD | Combine several independently-trained teachers (average, instance-weighted, or fused) into one soft target, then distill into one student | Fukuda & Suzuki, Interspeech 2017; "Multi-Teacher Distillation: Ensemble-Then-Distill," NeurIPS 2024 | Low-moderate: N teacher passes (cacheable), one KL term | Multi-teacher students beat single-teacher and naive-ensemble baselines across vision/language/speech | Averaging is dimension-agnostic — **directly matches our 2-teacher (recto, m7) setup** |
| Weighted ensemble of teaching assistants | Learn per-teacher/per-sample weights instead of flat averaging (small gating module or agreement-based weight) | arXiv:2206.12005 | Low: small extra module | Beats flat-average ensemble distillation | Weighting could be voxel-local — good fit if the two teachers disagree systematically by region (as noted, recto vs. m7 bands differ in thickness, `unified_design.md` §3) |
| Multi-view theory of ensembling/KD | Formal argument: data has multiple independent "views," each independently-trained net learns a random subset; ensembling aggregates views, distillation into one model provably can exceed any individual member because it inherits the *union* of views | Allen-Zhu & Li, ICLR 2023, arXiv:2012.09816 | Theory only | Synthetic + CIFAR-style constructions | Architecture-agnostic; the formal grounding for the "uncorrelated-teacher-errors average out" intuition |
| Uncorrelated-errors / bias-variance argument | Classic result: if base learners' errors are even partially decorrelated, ensemble expected error falls roughly linearly with count | Discussed in Allen-Zhu & Li 2023; arXiv:2009.04120 | None extra | Corroborated across the multi-teacher KD papers above | Same statistics hold per-voxel; caveat below |
| "Does Knowledge Distillation Really Work?" | Empirically, the student's output distribution often does *not* actually match the teacher's — accuracy improves even when fidelity doesn't. Treat soft targets as a regularizer, not ground truth to reproduce exactly | Stanton, Izmailov et al., NeurIPS 2021 | N/A | Classification, careful ablations | Caution applies equally to us: validate empirically whether the student tracks the S2 ensemble target, especially on boundary/disagreement voxels |
| Noisy Student (again, distillation framing) | Teacher unnoised, student noised; iterated | Xie et al., CVPR 2020 | High | ImageNet | See (b) |
| Born-Again Networks | Same-architecture sequential self-distillation; final generation beats generation 0 | Furlanello et al., ICML 2018, arXiv:1805.04770 | Medium: K sequential trainings | CIFAR/ImageNet-scale, PTB LM | Architecture-agnostic; relevant if a coarser-rung student is later treated as "teacher" for a finer one, complementary to the fixed S2 ensemble |

**Mapping**: the case for ensembling S2's two teacher stores by soft-averaging (not thresholded-argmax) is
strong on three independent legs — dark knowledge (Hinton), decorrelated-error variance reduction
(bias-variance argument + multi-view theory, giving a mechanism for *exceeding* either teacher, not just
matching their mean), and direct multi-teacher-KD empirical results. Concretely:

- **Already partially implemented**: `unified_design.md` §3 says "both teacher lineages feed the same
  recto head... the student learns an average band" — this is a flat, un-weighted average today.
- **Upgrade candidate**: per-voxel or per-region agreement-weighted averaging (weighted-TA style) instead
  of flat averaging, especially since recto/m7 bands differ in thickness (32% vs 23% of a mid-scroll cube)
  — a flat average band-averages two systematically different geometries rather than trusting the sharper
  one where it is more confident.
- **Structured-KD term** (pairwise local affinity between the two teacher stores) is a plausible loss
  addition beyond per-voxel BCE, but the O(N²) pair cost needs the same local-window trick the synthesis
  doc already uses for L1 (repulsion, radius-1-2 offsets, not an O(N²) loop) — reuse that machinery rather
  than inventing a new one.
- **Caveat to respect**: "Does KD Really Work?" argues we should *measure* student/S2-ensemble fidelity on
  held-out disagreement regions rather than assume the loss form guarantees it — a natural extension of the
  synthesis doc's own discipline ("if `merge_frac` does not move... drop it").

---

## (d) Coarse-to-fine cascades, iterative refinement, exposure bias

| Method | Description | Citation | Cost | Evidence | 3D |
|---|---|---|---|---|---|
| nnU-Net 3D U-Net Cascade | Coarse-stage prediction upsampled and concatenated as an extra channel to a full-res stage-2 net | Isensee et al., Nature Methods 2021 | Low: standard config | KiTS/general nnU-Net benchmarks | Native 3D — **structurally our cascade channel**, already implemented (§22) |
| Cascaded 3D FCN (organ segmentation) | Coarse organ ROI → refine; ablation shows cascade beats single-stage on small/hard organs (pancreas Dice 68.5→82.2) | Zhou et al., Medical Image Analysis 2019, arXiv:1803.05431 | Moderate | Pancreas/liver/kidney CT | Native 3D |
| CascadePSP | Class-agnostic refinement network: takes an existing (possibly low-quality) mask + image, refines at multiple strides, trained with *synthetically degraded* masks as input to teach error correction | Cheng et al., CVPR 2020, arXiv:2005.02551 | Moderate: extra refinement net trained on corrupted priors | High-res segmentation refinement benchmarks | 2D; multi-stride refinement idea ports to a rung pyramid |
| Recurrent iterative refinement / feedback networks | RNN/ConvLSTM feeds back the previous iteration's prediction for T refinement steps | arXiv:1811.08043; arXiv:1705.07238 | Moderate-high: T passes, BPTT cost | Scene parsing | Recurrence over voxel grids is expensive; conceptually transferable with fewer, larger steps (matches our "one level of recursion" design in §22) |
| Deep recurrence / predictive-coding feedback | Output-to-input feedback within a U-Net; **explicit numerical instability without damping** (softmax projection + exponential decay needed, or it diverges) | arXiv:2507.10143, 2025 | Moderate: needs stabilization | Improves over feedforward in noisy/low-data regimes; diverges without damping on harder datasets | Direct warning for our multi-rung recursion: unconstrained self-feedback needs explicit damping, which is effectively what `--cascade-drop` and the truncated-recursion (`self`'s own cascade channel is always zero) already provide |
| Reviving Iterative Training with Mask Guidance (RITM) | Simulates the *previous iteration's predicted* (not ground-truth) mask as an input channel during training, to close the train/inference gap | Sofiiuk et al., arXiv:2102.06583 | Low-moderate | Interactive segmentation (SBD, GrabCut) | Directly matches our `--cascade mix` design — literature confirms real predicted priors beat clean-GT-only priors |
| Scheduled Sampling | Decoder input is GT with prob ε (annealed down) else the model's own previous prediction, closing the exposure-bias gap between teacher-forced training and free-running inference | Bengio et al., NeurIPS 2015 | Low: sampling schedule only | Captioning, parsing, speech | **Directly the fix our fixed `--cascade-self-p 0.5` does not yet have** — see recommendation below |
| DAgger | Iteratively roll out the *current* policy, re-label the states it actually visits, aggregate — targets the real visited-state distribution rather than a fixed schedule | Ross, Gordon, Bagnell, AISTATS 2011 | Higher: needs an oracle (here, GT) evaluated on states the current model visits | Robotics/control, stronger theoretical guarantee than scheduled sampling | Maps to periodically regenerating `mask`-mode cascade training data from the *current* checkpoint's own coarse output rather than a fixed precomputed pyramid |
| Professor Forcing | Adversarial: train so that free-running hidden-state dynamics are indistinguishable from teacher-forced dynamics | Lamb et al., NeurIPS 2016 | Moderate: discriminator + adversarial loss | Sequence generation | Portable in principle: discriminate GT-derived vs. self-predicted cascade-channel activations |
| OneSeg (medical, names the exact problem) | Cascaded slice/stage medical segmentation suffers when errors propagate at inference from reconstructed priors; adopts scheduled sampling to close the gap | arXiv:2309.13671 | Low-moderate | 3D medical segmentation | Native 3D — closest published precedent for our exact coarse-channel exposure-bias problem |
| Channel/input dropout as robustness regularizer | Randomly zero the auxiliary prior channel during training so the net degrades gracefully when it's missing | Standard in RITM-line and nnU-Net-style cascades | Negligible | Ablations show improved robustness to bad/missing priors | Already implemented (`--cascade-drop`) — literature confirms it should be *combined with*, not substituted for, scheduled sampling |

**Direct answer to the exposure-bias question asked in the brief**: yes, our cascade channel is a textbook
instance of the exposure-bias/compounding-error problem scheduled sampling and DAgger exist to fix. Section
22 already implements two of the standard mitigations (channel dropout `--cascade-drop`, and `mix` mode
which is literally RITM's "simulate the previous iteration's predicted mask" pattern generalized across
rungs) — but `--cascade-self-p` is a **fixed** 0.5, not annealed, which is exactly the gap scheduled
sampling closes. Concrete, cheap additions, in order of cost:

1. **Anneal `cascade_self_p` upward over training** (start near 0 — mostly `mask` — shift toward `self` as
   training progresses), directly following Bengio et al. 2015 / the OneSeg precedent. This is a one-line
   schedule change, no new code path.
2. **Perturb the `mask`-mode cascade channel with structured noise resembling actual coarse-rung errors**
   (boundary erosion/dilation at the merge/gap geometries, not just the existing random block dropout),
   following RITM's practice of training against realistically-corrupted priors rather than clean-GT priors
   plus generic noise. `--cascade-noise` already does erosion/dilation + block dropout (§22) — the
   literature suggests these should be *representative* of the model's actual error modes (systematic at
   thin/boundary structures), not uniform-random.
3. **DAgger-style refresh** (lower priority, higher cost): periodically rebuild the `mask`-mode cascade
   cache from the *current* checkpoint's coarse-rung predictions rather than a static precomputed pyramid,
   so the fine-rung network trains against its actual current error distribution. Given our stream-plan
   architecture already recomputes stores continuously (verso pod, teacher region stores), this is a
   plausible fit for a later noisy-student round rather than the live `u3` run.
4. **Damping warning for any future multi-step recursion**: if `cascade_depth` or an actual iterative
   (repeated) self-feedback loop is ever added beyond the current one-level truncation, arXiv:2507.10143's
   finding that unconstrained feedback diverges without explicit damping is a hard requirement, not an
   optimization — our current design avoids this by construction (the `self` mode's own cascade channel is
   always zero, i.e., truncated to one level), and that property should be preserved deliberately, not
   relaxed casually to "just recurse further."

---

## (e) Geometric/equivariance self-supervision as a label-free quality signal

| Method | Description | Citation | Cost | Evidence | 3D |
|---|---|---|---|---|---|
| Mean Teacher (MT) | EMA teacher gives consistency targets under input/model perturbation | Tarvainen & Valpola, NeurIPS 2017, arXiv:1703.01780 | Low | SSL baseline | Base for many 3D variants below |
| UA-MT | MT + MC-dropout uncertainty; consistency masked to low-uncertainty voxels | Yu et al., MICCAI 2019 | Medium: ~8 MC passes | Left-atrium MRI, large gains at 10-20% labels | Native 3D |
| Hierarchical consistency-regularized MT | Multi-scale consistency between student/teacher at several decoder depths | arXiv:2105.10369 | Medium | 3D left atrium | 3D |
| Cross-Consistency Training (CCT) | Shared encoder, multiple perturbed decoders; consistency between decoder outputs on unlabeled data at the *feature* level | Ouali et al., CVPR 2020, arXiv:2003.09005 | Medium: extra decoder heads | Cityscapes/PASCAL/CamVid | 2D origin; feature-level perturbation ports to 3D trivially |
| Unsupervised Data Augmentation (UDA) | Consistency between weak- and strong-augmentation views | Xie et al., NeurIPS 2020, arXiv:1904.12848 | Low-medium | CIFAR-10 (250 labels), IMDb (20 labels) | 2D/NLP; augmentation-strength idea generalizes to our 3D sym+intensity augs |
| Confidence-aware cross-pseudo-supervision | Two branches cross-supervise; KL/variance between them downweights noisy pseudo-labels instead of a fixed threshold | arXiv:2307.16256 and others | Medium: 2nd branch | 3D medical organ/tumor benchmarks | Several native 3D |
| TTA-based aleatoric uncertainty | Flips/rotations/small affines at inference; prediction variance across views = per-voxel uncertainty | Wang et al., Neurocomputing 2019, arXiv:1807.07356 | Medium-high: linear in view count, no retrain | Brain tumor/fetal MRI; correlates with segmentation error, improves calibration | **3D CT/MRI directly** — most transferable finding here |
| CertainTTA / budget-aware nnU-Net uncertainty | Combines TTA and ensemble/dropout uncertainty at test time; budget-aware variants select a subset of views under compute budget | ScienceDirect S1566253525003732; arXiv:2604.11798 | Configurable | Radiotherapy OAR segmentation QA, nnU-Net backbone | 3D nnU-Net — closest existing tooling |

**Mapping**: we already have 48-cube-symmetry augmentation at train time (`unified_design.md` §4) and the
cascade gives a second, free consistency signal for free (coarse vs. fine agreement). The literature's
concrete, low-cost suggestion:

- Reuse a **subset** of the 48 symmetries (the 24 proper rotations, or just the 8 flips) as **inference-time
  TTA on S1/S2-labeled regions**, no retraining — where TTA variance is low and matches S1/S2, treat as
  high-trust "silver"; where variance is high or disagrees with the cascade's coarse-stage prediction, flag
  for downweight/exclusion (UA-MT/SRPL-SFDA pattern, substituting augmentation views for the two teachers
  in (a)/(b), or combining both signals).
- **Pitfall, stressed by the literature**: TTA variance only detects *model* instability, not a systematic
  bias shared between S1/S2 and our own augmented predictions — if both are wrong the same way (plausible
  for a fiber-orientation ambiguity both the old model and ours inherit), TTA variance looks falsely low.
  Cross-check against S3, not just internal agreement — the same caution as (a)/(b)'s confirmation-bias
  warning, from an independent angle.
- **Cost pitfall**: full 8- or 24-view TTA over many-TB volumes is expensive; restrict to sparse sampling of
  candidate/boundary regions (near segment edges, low-cascade-agreement zones) rather than blanket coverage,
  and the budget-aware nnU-Net line (arXiv:2604.11798) is the closest published recipe for picking *which*
  views to spend the budget on.

---

## (f) Small clean set to weight/calibrate a large noisy set

| Method | Description | Citation | Cost | Evidence | Segmentation-native? |
|---|---|---|---|---|---|
| Learning to Reweight Examples (L2RW) | Meta-gradient: perturb per-sample weights, one SGD step, measure clean-validation loss reduction, backprop into weights | Ren, Zeng, Yang, Urtasun, ICML 2018, arXiv:1803.09050 | High: unrolled gradient step per meta-update | CIFAR imbalance + corruption, large gains | No — classification, general mechanism |
| MentorNet | Small network learns a data-driven curriculum (sample weights) to guide a student away from noisy labels | Jiang et al., ICML 2018, arXiv:1712.05055 | Medium: extra net, pretraining | Best-published WebVision (2.2M noisy) result at the time | No — classification |
| Meta-Weight-Net (MW-Net) | Small MLP maps loss→weight, meta-updated against clean data each step | Shu et al., NeurIPS 2019; extended CMW-Net arXiv:2202.05613 | High: bilevel/implicit-diff update per step | CIFAR imbalance+noise | No — classification, but CMW-Net's class-aware weighting is portable to per-source weighting |
| Gold Loss Correction (GLC) | Small trusted set estimates a noise-transition matrix from a noisy-trained model; corrects the loss (not per-sample reweighting) | Hendrycks, Mazeika, Wilson, Gimpel, NeurIPS 2018, arXiv:1802.05300 | Low-medium: one matrix estimate + corrected loss, no meta-gradient | CIFAR/ImageNet severe synthetic noise, beats forward/backward correction | No — classification, but the cheapest mechanism in this table |
| Confident Learning / cleanlab | (as in (a)) — estimate joint noisy/true distribution, prune/rank errors | Northcutt, Jiang, Chuang, JAIR 2021 | Low | CIFAR/ImageNet | No |
| Adaptive Early-Learning Correction for Segmentation | Exploits early-learning dynamics (clean regions fit first) to detect/correct noisy *per-pixel* labels during training, self-referential (no explicit clean set required) | Liu et al., CVPR 2022, arXiv:2110.03740 | Medium: correction/regularization term | PASCAL VOC / Cityscapes-style noisy-label segmentation | **Yes — 2D, dense, closest algorithmic template for voxel-level correction** |
| Walking on Two Legs | Joint label correction + per-pixel/region reweighting for noisy segmentation labels | Cheng et al., ACML 2020 | Medium-high | PASCAL VOC 2012 noisy segmentation | Yes — 2D, the "correct + reweight" framing maps to our two-teacher case |
| High-quality pseudo masks from noisy/weak annotations | Two-phase: noise-identification net (trained partly against a small clean set) revises noisy/weak masks, feeds a noise-robust segmentation net — explicitly designed for "few clean + many noisy" | 2024, PubMed 39520897 | Medium: extra identification net, one-time | Medical organ/lesion segmentation, reports pseudo-mask quality + downstream Dice gains | **Yes — closest direct template for our exact S3-vs-(S1,S2) setup** |
| ScaleBiO | First-order (Hessian-free) approximation to bilevel data-reweighting, scales to 34B-parameter LLM training | 2024/2025 | Much lower than classic MW-Net/L2RW at scale | LLM data-mixture reweighting, not segmentation | Not segmentation, but the practical fix if meta-learned reweighting is ever wanted at our scale |

**Mapping**: our setup (a few dozen-few hundred verified boxes = S3/"gold," many-TB S1+S2 = "noisy corpus")
is close to textbook GLC/L2RW/MW-Net territory, but those methods differentiate through a simulated training
step on every meta-update — at 3D-cascade scale with terabyte corpora, that is likely too expensive without
ScaleBiO's first-order trick. The cheaper, better-evidenced path for us:

- **Skip full bilevel meta-learning.** Use S3 to estimate a coarse **per-source / per-region-confidence
  statistic** (how often does S1, or S2's recto lineage, or S2's m7 lineage, agree with S3, broken down by
  e.g. rung, scroll region, cascade-agreement bucket from (e)) — this is GLC's noise-matrix idea without the
  meta-gradient, i.e., a single pass to compute counts, then a fixed (not learned) per-(source × bucket)
  loss weight. This composes directly with the existing per-source `weight` field in `unified_design.md` §3,
  which today is a flat constant (1.0, 1.0, 0.3) with no S3-measured correction.
  - **The synthesis doc's Phase A/B (`synthesis_future_inputs_outputs.md` §4) partially anticipates this**
    already — §4's "metric that must move" discipline (`merge_frac`, `continuity`, `offset<=3`) is exactly
    the machinery a GLC-style calibration pass would use S3 to validate against, just not yet used to set
    per-source weights.
- **Segmentation-native templates to borrow structure from**: Adaptive Early-Learning Correction and the
  2024 "high-quality pseudo masks" paper are both closer in spirit than the classical meta-learning line —
  both separate correct-vs-corrupted dense labels using training dynamics or a small identification network,
  then correct/reweight per-voxel or per-region, not per-sample.
- **Pitfalls**: (1) a few hundred verified boxes is too small for a fine-grained (per-class, per-voxel)
  noise matrix or an MW-Net-style MLP without high variance — stay at (source × coarse-bucket) granularity;
  (2) full bilevel meta-learning (L2RW/MW-Net/MLC) gets worse, not better, with a 3D-UNet-cascade's memory
  footprint — GLC-style one-shot correction or ScaleBiO's first-order trick are the only tractable options
  at this scale; (3) **selection bias in S3 itself** — if verified boxes cluster in easy/well-preserved
  regions (plausible, since verification is presumably easier there), the estimated noise statistics will
  not transfer to the hard regions of S1/S2 where errors actually concentrate. Stratify future verification
  effort by the same difficulty signals (e) already gives us (TTA variance, cascade disagreement) rather
  than by convenience.

---

## Cross-cutting synthesis: what this survey adds to the existing unified plan

The existing roadmap (`synthesis_future_inputs_outputs.md` §4-5) already independently arrived at two of
this survey's best-evidenced, lowest-cost ideas — **L3 soft exclusivity** and **L4 cascade
self-consistency**, both flagged Phase A, zero new labels, near-zero step cost. This survey's distinct
additions, ranked by cost:

1. **Anneal `--cascade-self-p` instead of holding it fixed at 0.5** (topic d). One-line schedule change,
   directly closes a textbook exposure-bias gap the current fixed mixture only partially addresses.
2. **Agreement/entropy-gated fusion of the two S2 teacher stores**, replacing the current flat average
   (topics a, c). Inference-time only, no retraining, and directly attacks the fact that recto/m7 bands
   differ systematically in thickness rather than just noisily.
3. **Use S3 to set coarse per-source loss weights (GLC-style, not meta-learned)**, replacing the current
   flat constant weights in `unified_design.md` §3, and to validate (not just motivate) any noise-robust
   loss or agreement-gating choice above (topic f). Single offline pass, reuses the existing surface-metric
   harness.
4. **TTA-variance as a cheap region-confidence signal**, sparsely applied to candidate/boundary regions
   identified by the cascade, feeding the same reweighting/self-training gates as (2) and (3) (topic e).
5. **Curriculum-ordered, cross-filtered self-training rounds** (agree-first, S3-anchored, co-teaching-style
   cross-filtering rather than pure self-referential relabeling) once S4 noisy-student rounds are
   implemented (topics a, b) — the single most consistently-repeated warning across all four searches is
   that self-training without an external anchor amplifies correlated teacher errors, and S3 is our only
   source that is not correlated with S1/S2's shared upstream lineage.

None of these require a new label source, a new head, or a new store; all are loss-form, scheduling, or
fusion-weight changes over sources already in the pipeline (S1-S4), which keeps them inside the same
"flag defaults to off/current behavior, bit-for-bit unchanged" discipline the codebase already uses for
`--cascade` and `--verso`.
