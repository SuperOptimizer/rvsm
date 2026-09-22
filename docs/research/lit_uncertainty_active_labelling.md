# Literature: uncertainty estimation and active learning outside Vesuvius (2026-09-21)

Scope: state of the art in general 3D/medical segmentation (not Vesuvius Challenge work) on per-voxel
uncertainty, calibration, uncertainty-guided pseudo-label filtering, active-learning region selection, and
human-in-the-loop interactive correction. Mapped to `/home/forrest/usrm2`: a 256^3-patch, multi-rung recto
(+verso) surface model, bootstrapped from published binary masks, with filtered self-training rounds and a
slab-viewer refinement loop for picking where a human fixes labels next (`docs/unified_design.md` §3-5,
21-23; `docs/research/synthesis_future_inputs_outputs.md` §4-5).

## 1. MC dropout

Method: keep dropout on at inference, run N stochastic passes, use mean/variance as prediction/uncertainty
(Gal & Ghahramani, variational-inference framing). Cost: N forward passes (10-30 typical) plus retraining
with dropout added — not post-hoc. Evidence: surveys consistently rank it below deep ensembles on
calibration and OOD detection, and it underestimates uncertainty far from the decision boundary. usrm2's
UNet has no dropout today, and N x 256^3 passes on top of the existing 3-level cascade (+14%/level, §22)
is expensive for a benefit the cheaper EMA-vs-live proxy (§3) already approximates. **Skip.**

## 2. Deep ensembles

Method: train K independent models, use ensemble mean/variance (Lakshminarayanan et al. 2017). Cost: K full
training runs + K x inference; best calibration/OOD detection of any method surveyed, the benchmark others
are judged against. Full K=3-5 retraining is not realistic against the desk's contended A100/5060 Ti budget
(§22 "desk could not be measured"). **Mapping:** usrm2 already has 3 differently-trained checkpoints on
disk (bootstrap, cascade-warm-started, verso-warm-started) — score a contested region with all three as a
free poor-man's ensemble before training anything new for this purpose.

## 3. Cheap ensembles: snapshot, multi-head, EMA/checkpoint

Snapshot ensembles (Huang et al. 2017, cyclic LR) save checkpoints at each local minimum for ~0 extra
training cost. Multi-head ensembles share a trunk and run several independent output heads, reading
disagreement in one forward pass. EMA-vs-live disagreement uses the exponential moving average weights
already kept alongside live weights as a near-free 2-member ensemble (arXiv:2403.10182). Evidence: weaker
than full deep ensembles but recovers most of the calibration benefit at a fraction of the cost. **Mapping,
concrete:**
- **EMA-vs-live** (near-zero cost): `--cascade self|mix` already runs a no-grad EMA forward pass. Emit
  `|sigmoid(live) - sigmoid(ema)|` at the current rung as a byproduct uncertainty map — cheap, and it
  correlates with regions training hasn't converged on, which is what round/region selection needs.
- **Sibling heads** (small one-time cost): add 2-3 extra 1x1x1 conv heads off the shared trunk (same pattern
  as the `w0`-parameter verso head, §23), independently initialized; their disagreement at inference is a
  genuine narrow ensemble signal for near-zero added compute.
- **Checkpoint-family disagreement** (free): score a candidate region with the bootstrap/cascade/verso
  checkpoints already on disk — do this first, before building anything new.

## 4. Evidential deep learning (EDL)

Method: replace the sigmoid output with Dirichlet/NIG evidence parameters so a single forward pass yields
both prediction and epistemic uncertainty (Sensoy et al. 2018; survey arXiv:2409.04720; medical use in DuEDL,
arXiv:2405.14444, scribble-supervised — close to usrm2's sparse-verified-mesh regime). Cost: cheapest
inference of any epistemic method, but training needs a different loss (evidential NLL + KL) with a
regularization schedule that is known to be finicky (too strong too early collapses evidence to zero).
Reported uncertainty maps are noisier than ensemble variance in head-to-head comparisons. **Mapping:** would
require changing the loss/output parameterization for the whole net, which collides with the already-tuned
BCE+dice+per-channel-ignore design of §3/§23. **Defer** — revisit only if §3/§5's cheap signals prove too
noisy in practice; it's a rewrite, not an additive flag.

## 5. TTA-disagreement as uncertainty

Method: run inference under a set of test-time transforms, invert, use the spread as uncertainty (Ayhan &
Berens 2018; BayTTA arXiv:2406.17640 shows it is complementary to, not a substitute for, weight-based
uncertainty — it catches orientation-sensitive errors instead). Cost: K forward passes, no retraining, works
on any existing checkpoint — the cheapest option here. **3D applicability is unusually good for usrm2
specifically**: the 48-cube-symmetry augmentation group (`prep.sym_apply_t`) is already implemented and
exercised every training step, so running a subset of it at inference and reporting per-voxel std is reusing
an existing capability, not building one. **Mapping:** run ~8 of the 48 symmetries through `predict.probs`,
invert, report std; combine with EMA-vs-live (§3) — TTA catches orientation-sensitive uncertainty, EMA-vs-
live catches training-recency uncertainty, and their union is a stronger score than either alone. Validate
against `merge_frac`/`continuity` before trusting it. **Pitfall:** a model trained WITH this exact symmetry
group is taught to be invariant to it, so absolute spread values will be small — use ranked/relative spread
across regions, not a fixed threshold.

## 6. Calibration of dense/per-voxel probabilities

Method: global temperature scaling (Guo et al. 2017) fits one scalar T post-hoc; spatial miscalibration
motivates local/per-voxel T (Ding et al., ICCV 2021) or joint calibration losses (arXiv:2607.01902,
arXiv:2506.03942). Mehrtash et al. (arXiv:1911.13273) is the standard medical reference: Dice-trained nets
are measurably overconfident (exactly usrm2's BCE+dice loss family), and Jungo et al. add that dataset-level
calibration can look fine while individual regions are not. Cost: global T is free (one scalar per held-out
set); local T is a small extra net. **Mapping, important:** usrm2's raw sigmoid is not a calibrated
confidence even by construction — targets above native rung are **pooled fractions**, not binary-event
probabilities (§3), and BCE+dice is the overconfidence-prone loss family the literature flags. Before
building any uncertainty-guided filtering (§7) or region-selection score (§8) off the raw sigmoid, fit one
global temperature **per rung** (not one overall — rung 2 is a hard band, rung 6+ a heavily pooled fraction)
on the existing `eval.jsonl` validation box. **Pitfall:** never calibrate against the pooled-fraction targets
above native rung — a fraction isn't a binary label; calibrate only where the target is a genuine binary band.

## 7. Uncertainty-guided pseudo-label filtering for self-training rounds

Method: filter/weight self-training pseudo-labels by uncertainty so the student doesn't learn the teacher's
confident mistakes. 2023-2025 variants: sample-level gating (DUMM, aimspress mbe.2024097), adaptive
thresholds (nnFilterMatch, arXiv:2509.19746), cross-attention ensemble mean-teacher (arXiv:2412.15380),
credible pseudo-labeling separating confident-correct from confident-wrong (UCPL). This is now the dominant
paradigm in semi-supervised medical segmentation, with sample/region-level filtering argued more stable than
pure voxel-level (DUMM). **Mapping, concrete:** usrm2 already has a per-voxel weight tensor multiplying every
loss term (source weight x box mask x CT>0 x per-channel ignore, §3) — add uncertainty as a **fifth
multiplicative factor** for self-training targets specifically: `weight *= f(uncertainty)` from whichever
cheap signal (§3/§5) validates. No new store format needed; add an uncertainty channel to the existing
region-store writer (`predict.out_array`) the way §23 added the verso channel. Sample-level fits usrm2's
existing per-patch rejection rules (`fg_min`, `dense_pow`) better than per-voxel reweighting. For **round
weighting**: score each self-training round's output against the human-verified mesh set the way
`eval.jsonl` already scores held-out boxes, and only admit rounds whose `recall@4`/`continuity`/`merge_frac`
clear a bar — curriculum gating, cheap because `evalsurf` already exists. **Pitfall:** an uncertainty
estimator derived from the same model it filters systematically under-flags that model's *confident,
systematic* failures — exactly the sheet-merge problem (`merge_frac` 0.44, the worst metric on the board).
Treat this as a precision tool for ambiguous/noisy regions, not a fix for merges; that's what the geometric
losses (L1/L3, synthesis §4 Phase A) are for.

## 8. Active-learning region selection for volumes (where humans should refine next)

Method families: uncertainty sampling (entropy/BALD); diversity/core-set (Sener & Savarese; 3D metric-
learning embeddings, arXiv:2411.15763); combined uncertainty+diversity (dominant in 2023-2025 3D medical AL,
since uncertainty alone clusters picks in one hard region); cold-start/typicality scoring for an initial
labeled set (CSCS, arXiv:2606.20765). **Key result: nnActive (arXiv:2511.19183, Nov 2025), the largest 3D
biomedical AL benchmark to date (8 query methods x 4 datasets x 3 regimes) — no AL method reliably beats a
well-designed foreground-aware random baseline; predictive entropy is the best method that comes close, but
even it doesn't clearly beat the corrected baseline.** This is the single most important finding for "which
patches to annotate": it argues against investing in BALD/core-set machinery and for validating any picker
against a coverage-aware random baseline. **Mapping, concrete:**
1. Score at **region-store granularity** (§17's boxes/regions the slab viewer already operates on), not per
   voxel — matches nnActive's lesson and the existing `done`/`origin_zyx` bookkeeping.
2. Score = cheap disagreement (§3 EMA-vs-live or §5 TTA std), aggregated per region — not a from-scratch
   BALD/core-set system, given nnActive's result.
3. Diversity = cheap and geometric (minimum physical spacing along axis/wraps using the umbilicus-relative
   coordinates `radial_t`/`data.axis_at` already compute), not a learned embedding.
4. A/B against the existing `n_k^0.5` rung-weighted random sampler (§4) restricted to foreground-passing
   regions — that comparison, not an absolute score, validates the picker.
5. Score the picker itself by `merge_frac`/`continuity` on picked regions after a refine+retrain cycle, same
   discipline the unified plan already applies to loss ablations.
**Pitfall:** confirmation bias recurs harder here — an uncertainty-only picker never surfaces a confidently
wrong merge. Combine the uncertainty term with a confidence-independent structural check (e.g. the existing
`overlap`/`merge_frac`-style geometric metrics) rather than ranking purely by disagreement.

## 9. Human-in-the-loop interactive refinement: scribbles, clicks, SAM-3D

Scribble-based: Scribble2D5 (MICCAI 2022, arXiv:2205.06779) propagates sparse scribbles via supervoxels plus
boundary regularization; 2023 follow-ups add shape priors (arXiv:2310.08084); DuEDL pairs scribbles with
evidential uncertainty. Click/point-promptable: SAM-Med3D (arXiv:2310.15161, ECCV 2024) is a genuinely
3D-native promptable model (143K masks, 22K volumes) needing 10-100x fewer clicks than slice-wise 2D SAM;
3DSAM-adapter and ProtoSAM-3D (2024-2025) confirm 3D-native prompting strictly beats 2D-per-slice, since 2D
prompting can't propagate a correction along the third axis without re-prompting every slice. Cost: these are
train-once, use-many annotation accelerators, not part of the production model; sub-second per click at
inference, substantial one-time training cost usrm2 doesn't need to pay (an existing checkpoint could be
piloted). Evidence: strong click-reduction numbers, but no evidence on transfer to CT of a carbonized,
low-contrast papyrus sheet — a real domain-gap risk. **Mapping:** the existing slab-viewer drag-anchor plan
is already point/anchor-based interactive correction, just without a learned propagation model assisting the
drag. Two concrete additions, in cost order: (1) once the planned normal head (O4, synthesis §4 Phase B)
exists, propagate a single corrected anchor along the tangent plane using it — cheapest, reuses planned work,
no new dependency; (2) pilot (not adopt) a SAM-Med3D-class model purely as a labeling-desk accelerator for
producing NEW verified-mesh training data faster, on 2-3 known-hard regions, before any real investment.
**Pitfall:** don't conflate a labeling-accelerator pilot with improving the trained recto/verso model itself
— its output still needs the same human verification step any mesh gets before entering the target pyramid.

## 10. Using uncertainty to weight losses

Two distinct patterns: (a) heteroscedastic aleatoric weighting, where the network predicts its own per-voxel
variance and the loss self-weights (Kendall & Gal 2017); (b) noisy-label-aware weighting, using uncertainty
or loss magnitude to down-weight suspected mislabeled voxels — a related but distinct problem, since high
uncertainty and label noise correlate but are not interchangeable (Frontiers 2022 frsen.2022.1100012 finds
uncertainty alone insufficient to identify noisy labels). Cost: (a) is one extra output channel, same
`w0`-parameter pattern as verso/cascade; (b) is as cheap as whatever signal feeds it. **Mapping:** lower
priority than §3/§5/§7 here because usrm2 already encodes its *known, structural* uncertainty sources
explicitly in the weight tensor (§3) rather than needing to learn them — a better fit than a learned
heteroscedastic head for a problem that's largely known-structural (published-mask softening, pooled
fractions, missing verso coverage). The one gap a learned head COULD fill: down-weighting voxels where the
two teacher lineages (recto model vs. m7) disagree most, a genuine unaddressed noise source (§3: "the
student learns an average band, no thickness harmonisation") — flag as a future-phase candidate fitting the
`losses_aux` flag-gated pattern (synthesis §4 Phase A), not needed for the current round-weighting work,
where §7's calibrated-region-gate is cheaper and better evidenced. **Pitfall:** a learned variance head
trained jointly with BCE+dice can degenerate (predict high variance everywhere to trivially minimize loss);
needs the same `merge_frac`/`continuity`-must-move ablation discipline already applied to L1-L4.

## Summary: priority for usrm2

| Method | Cost | Best usrm2 use | Priority |
|---|---|---|---|
| Symmetry-TTA disagreement (§5) | ~free (reuses `sym_apply_t`) | region-selection score, no retrain | do first |
| EMA-vs-live disagreement (§3) | ~free (cascade `self` already computes) | round + region-selection score | do first |
| Per-rung global temperature scaling (§6) | ~free (one scalar/rung) | prerequisite for trusting raw sigmoid | do first |
| Checkpoint-family disagreement (§2/§3) | free (ckpts exist) | one-off contested-region audit | as needed |
| Weight-tensor uncertainty gate (§7) | small (reuses weight pipeline) | filtered self-training rounds | next |
| Region-store uncertainty + geometric diversity (§8) | small-moderate | slab-viewer "refine next" queue | next |
| Sibling-head cheap ensemble (§3) | small (extra `w0` heads) | firmer signal if above too noisy | later |
| Normal-seeded anchor propagation (§9) | tied to Phase B O4 | speeds up drag-anchor refinement | later |
| SAM-Med3D-class pilot (§9) | pilot-sized | speeds up NEW mesh production only | pilot only |
| Heteroscedastic per-voxel weight head (§10) | small head + ablation cost | teacher-lineage disagreement down-weight | future phase |
| Full deep ensembles (§2) | K x training cost | contested-region audits only | rare |
| MC dropout (§1) | N x inference + retrain | — | skip |
| Evidential deep learning (§4) | loss/head rewrite | — | defer |

**Cross-cutting caution:** every signal derived from the model's own weights (§1-5, §10) systematically
under-flags its *confident, systematic* failures — exactly `merge_frac`, the worst metric on the board. None
of these substitute for the geometric, confidence-independent losses (L1 repulsion, L3 exclusivity) already
in the unified plan; layer uncertainty-guided filtering and region selection on top of those, and always pair
a learned-uncertainty term with at least one structural check that doesn't depend on the model's confidence.

## Sources

- Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016).
- Lakshminarayanan, Pritzel & Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (NeurIPS 2017).
- [Uncertainty Quantification in Medical Image Segmentation: A Comprehensive Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC13514988/)
- [Evaluating Uncertainty Quantification in Medical Image Segmentation: A Multi-Dataset, Multi-Algorithm Study](https://dx.doi.org/10.3390/app142110020)
- [Uncertainty quantification + segmentation, Bayesian deep learning, Communications Medicine 2024](https://www.nature.com/articles/s43856-024-00528-5)
- Huang et al., "Snapshot Ensembles: Train 1, Get M for Free" (ICLR 2017).
- [Reliable uncertainty with cheaper neural network ensembles](https://arxiv.org/html/2403.10182v1)
- Sensoy, Kaplan & Kandemir, "Evidential Deep Learning to Quantify Classification Uncertainty" (NeurIPS 2018, arXiv:1806.01768).
- [A Comprehensive Survey on Evidential Deep Learning](https://arxiv.org/pdf/2409.04720)
- [DuEDL: Dual-Branch Evidential Deep Learning for Scribble-Supervised Medical Image Segmentation](https://arxiv.org/pdf/2405.14444)
- Ayhan & Berens, "Test-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty" (MIDL 2018).
- [BayTTA: Uncertainty-aware medical image classification with optimized test-time augmentation](https://arxiv.org/pdf/2406.17640)
- Guo, Pleiss, Sun & Weinberger, "On Calibration of Modern Neural Networks" (ICML 2017).
- Ding et al., "Local Temperature Scaling for Probability Calibration" (ICCV 2021, arXiv:2008.05105).
- [Rethinking Post-Hoc Calibration in Semantic Segmentation](https://arxiv.org/pdf/2607.01902)
- [Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation](https://arxiv.org/pdf/2506.03942)
- Mehrtash et al., "Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation" (arXiv:1911.13273, IEEE TMI 2020).
- [Dual uncertainty-guided multi-model pseudo-label learning for semi-supervised medical image segmentation](https://www.aimspress.com/article/doi/10.3934/mbe.2024097?viewType=HTML)
- [nnFilterMatch: Uncertainty-Aware Pseudo-Label Filtering for Efficient Medical Segmentation](https://arxiv.org/pdf/2509.19746)
- [Uncertainty-Guided Cross Attention Ensemble Mean Teacher for Semi-supervised Medical Image Segmentation](https://arxiv.org/pdf/2412.15380)
- [nnActive: A Framework for Evaluation of Active Learning in 3D Biomedical Segmentation](https://arxiv.org/html/2511.19183)
- [Integrating Deep Metric Learning with Coreset for Active Learning in 3D Segmentation](https://arxiv.org/pdf/2411.15763)
- [Dataset-Aware Cold-Start Active Learning for Annotation-Efficient 3D Medical Image Segmentation](https://arxiv.org/html/2606.20765v1)
- Sener & Savarese, "Active Learning for Convolutional Neural Networks: A Core-Set Approach" (ICLR 2018).
- [Scribble2D5: Weakly-Supervised Volumetric Image Segmentation via Scribble Annotations](https://arxiv.org/pdf/2205.06779)
- [Volumetric Medical Image Segmentation via Scribble Annotations and Shape Priors](https://arxiv.org/abs/2310.08084)
- [SAM-Med3D: Towards General-Purpose Segmentation Models for Volumetric Medical Images](https://arxiv.org/abs/2310.15161)
- [3DSAM-adapter: Holistic adaptation of SAM from 2D to 3D for promptable tumor segmentation](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002494)
- [ProtoSAM-3D: Interactive semantic segmentation in volumetric medical imaging](https://www.sciencedirect.com/science/article/abs/pii/S0895611125000102)
- Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" (NeurIPS 2017).
- [Uncertainty- and hardness-weighted loss functions for medical image segmentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12691699/)
- [Uncertainty is not sufficient for identifying noisy labels in training data for binary segmentation of building footprints](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2022.1100012/full)
