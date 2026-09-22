# Synthesis v2: inputs, outputs, losses and training, with the external literature (2026-09-21)

Revision of `docs/research/synthesis_future_inputs_outputs.md` (built from the five in-house surveys
`villa_lasagna`, `villa_vesuvius`, `vc3d_tracer_inputs`, `tsm_ideas`, `usrm_legacy_ideas`) after reading the
twelve external surveys in this directory: `lit_topology_merge_losses`, `lit_implicit_surfaces_manifold`,
`lit_noisy_labels_self_training`, `lit_scaling_multiscale_generalisation`, `lit_ct_physics_augmentation`,
`lit_fibre_orientation`, `lit_layered_structures`, `lit_uncertainty_active_labelling`,
`lit_surface_extraction`, `lit_pretraining_foundation`, `lit_optimisation_schedules`,
`lit_evaluation_metrics`. Context: `docs/unified_design.md` §1-7 (ladder, sample, targets/weights, sampling,
sources, code changes) and §21-24 (next input channels, cascade, verso, store codecs).

Where we are (§19-20, §23): `u1_30m6_p4` val dice r2/r3/r4 = 0.740/0.740/0.682; on the val box **recall@4
0.806, continuity 0.656, merge_frac 0.44, offset<=3 0.36**. `u3_30m6_cv` (cascade mix + verso, cout=2) is
training on the A100; the desk 5060 Ti pair writes teacher/verso region stores. **Merges and gaps remain the
two dominant failures** and every item is scored against them first.

**Confidence tag on every row and recommendation:**
- **[MH] measured in-house** — a number exists in usrm2/tsm/usrm/villa on our own data.
- **[ML] measured in literature** — a published, external, quantified result on an analogous task.
- **[ML?]** — as above, but the surveying agent flagged the citation as recall-based (`[M]`) or ran out of
  web-search budget. Verify the paper before quoting a number. This affects most of
  `lit_layered_structures` (its own `[M]` marks), `lit_optimisation_schedules` §2-4, and the arXiv IDs in
  `lit_noisy_labels_self_training`.
- **[S] speculative** — a mechanism with no measurement on either side.

---

## 1. Consolidated candidate table

`M` = merges, `G` = gaps/continuity, `O` = orientation, `P` = recto/verso pairing, `T` = tracer,
`X` = cross-scan, `E` = evaluation, `R` = training recipe.

### 1a. INPUT channels

| # | channel | attacks | cost | in-house evidence | what the literature adds | conf |
|---|---|---|---|---|---|---|
| I1 | **cascade** (rung k+1 prediction, 2x up) | G, X | DONE; +6% mask / +50% self / +35% mix | +6/35/50% step cost measured, `cloud/cascade_bench.py` | nnU-Net's 3D U-Net Cascade and SegVol's zoom-out-zoom-in are the *same* mechanism, externally validated; RITM confirms a *predicted* prior beats a clean-GT prior; **new**: `--cascade-self-p` fixed at 0.5 is textbook exposure bias, scheduled sampling / OneSeg say anneal it | [MH]+[ML] |
| I2 | **normalised radius from umbilicus** | M, G, O | ~0 | none | CoordConv-family: coordinate planes are the standard fix for a translation-invariant net needing position; only add planes compatible with the augmentation group (radius is, angle is not) | [ML] |
| I3 | **scan metadata planes** (energy, pixel size, delta_beta, unsharp, distance, window, mosaic) | X | small | none | strongest external endorsement of any item here: spacing/metadata conditioning (HyperSpace, spacing-agnostic nnU-Net) is one of the two working mechanisms for cross-resolution/cross-scanner generalisation, and DG literature says condition *and* augment on the same parameters | [ML] |
| I4 | axis-tangent vector (3) | O | ~0 | tsm: fibre class overlap 0.90 -> 0.03-0.12 only after this | unchanged; only needed if an axis-relative target lands | [MH] |
| I5 | SDF to a known mesh (sparse) | M, G | high | none | tree-ring "iterative next boundary" (Gillert 2023) is the same pattern; keep it in refinement/interactive mode | [ML?] |
| I6 | CT inner/outer face classification | P | CPU-bound | usrm used it for label QA only | — | [MH] |
| I7 | lasagna cos/grad_mag/dir preprocessor output | — | very high | — | **reject, now doubly**: `lit_fibre_orientation` found *no* 2018-2026 paper that feeds a hand-built orientation/structure-tensor field into a segmentation CNN and measures a gain; every use is post-hoc analysis or pre-cleanup | [ML] |
| I8 | structure tensor / sheetness / Hessian input | O | cheap | — | **new dead item**: same finding as I7; §21's filter-redundancy rule is now externally corroborated, not just a house rule | [ML] |

### 1b. OUTPUT channels

| # | head | attacks | cost | in-house evidence | what the literature adds | conf |
|---|---|---|---|---|---|---|
| O1 | **verso probability** (cout 1->2) | P, M | DONE | in progress | cortical white/pial is the direct analogue; DeepCSR/Vox2Cortex/PialNN all needed an explicit non-crossing mechanism — two independent heads do *not* stay apart for free | [ML] |
| O2/O3 | **UDF / signed distance to the face**, clamped | G, T, P | one EDT per rung per store, offline | tsm two-face SDF dice 0.636-0.669 = best surface parameterisation it tried; usrm shipped it | **CONFIRMS**: NUDF (volume-in, distance-out, medical CT), SegRecon (several coupled distance fields from one decoder), and a whole cross-task-consistency line (distance decoder sharpens a mask decoder). NDF/UDF's "drop the sign for open surfaces" is the one apparent contradiction and resolves *for* us: the sign is the pairing signal, and a UDF's gradient is undefined at the zero set (GeoUDF), so Scharr-of-UDF normals are unstable | [MH]+[ML] |
| O3b | **sdist to the MIDLINE, plus a bounded thickness** (new) | **M, P** | same EDT, different sign origin | none | **the biggest single change in this revision.** CortexODE / TopoFit / NeurIPS-2023 Coupled Reconstruction derive both surfaces from one midline by an invertible/bounded offset, so crossing is *unrepresentable*; OCT's non-negative-thickness cumsum (He 2018/2021, MAE 2.82 vs 2.83 um for the graph method) is the cheap voxel-friendly version | [ML]+[ML?] |
| O4 | **surface normal (nz,ny,nx)**, Scharr of O3 | O, P, T | free once O3 exists | 5 of 5 in-house docs | SIREN's lesson inverts into a pitfall: a ReLU conv decoder has piecewise-constant autograd gradients, so derive the normal from the *stored* distance field, never from the decoder's own gradient | [ML] |
| O5 | **winding phase (sin,cos)** | M | high, fragile | tsm: coarse learned winding aliased 6x, `WINDING_ENABLED=False` | **partially rehabilitated, in a strictly local form**: RGT's sinusoidal 2026 variant independently reaches `(sin,cos)`; but `lit_layered_structures` is emphatic that every global-index method assumes <50 layers and fails at our density. Build **k mod M (M~8) with a CORN rank-consistent head**, weight 0 where no tracer label exists, and resolve the multiples of M by classical quality-guided unwrapping *outside* the net | [ML?]+[S] |
| O6 | thickness / half-thickness | P | free once O1+O3 | **negative**: usrm's ablation, dropping `w_thick` to 0 *increased* recto dice | reframed: thickness is not a supervised head, it is the *bounded offset parameter* of O3b. That sidesteps usrm's negative result rather than contradicting it | [MH]+[ML] |
| O7 | confidence / validity head | T, E | small | none | prefer a **heteroscedastic sdist** (probabilistic-SDF OCT, 2024): mean + log-variance on O3 gives the tracer both the SDF it wants and the confidence it wants from one head. A learned variance head can degenerate (predict high variance everywhere) — needs the metric-must-move discipline | [ML] |
| O8 | fibre direction | — | blocked | tsm accuracy 0.67, no measured effect on dice | one new unblocking route: a **self-supervised structure-tensor vt/hz label** computed offline at rung 0-2 with the eigenvalue-ratio as confidence, replacing tsm's missing external teacher. Still needs I4. Low priority | [ML] |
| O9/O10/O11 | ink / in-plane tangent / double-angle | — | — | — | unchanged; O11 stays dead (180-degree symmetric by construction) | [MH] |
| O12 | **long-range affinity channels** (new) | **M** | small: k extra output channels, targets derived from the existing binary pyramid | none | mutex-watershed/MALIS/LSD: the field's consistent answer to "stop two touching objects fusing" is a *relational* signal between voxels at an offset, not a better mask loss. Offsets tuned to the 15-35-voxel sheet pitch encode "next wrap" directly. Backed by CREMI/FIB-25-scale results, unlike L1 | [ML] |
| O13 | directional connectivity map (DconnNet) | M, G | one light head | none | cheaper approximate substitute for a topology loss; separates "connects along-sheet" from "connects across-sheet" only once rotated into the O4 normal frame. 2D-published, 3D untested | [ML]+[S] |

### 1c. LOSSES

| # | loss | attacks | est. step cost | in-house evidence | what the literature adds | conf |
|---|---|---|---|---|---|---|
| L1 | normal-gated repulsion | M | +10-15% [S] | villa ships it, **no ablation anywhere** | **downgraded relative to O12.** The EM literature's version of the same idea (repulsive long-range affinity edges) is measured at scale; villa's is not. Also note C7 (TopoInteraction, ECCV 2022 oral) is a cheaper convolutional implementation of the same "class A must not touch class B" constraint | [ML] |
| L2 | planarity | G | +5% [S] | unablated | no external support found; keep as a cheap flag, lowest priority of the Phase-A four | [S] |
| L3 | **soft exclusivity** `relu(p_r+p_v-1)` | P, M | ~0 | our own `overlap` metric, already plumbed | **upgrade path found**: OCT's differentiable-DP / soft-ordering layer makes "recto before verso along the normal, with a minimum gap" *structural* rather than penalised, at "one extra differentiable layer, no new targets" | [ML] |
| L4 | **cascade self-consistency** | G, X | ~0 in `self`/`mix` | novel here; the coarse forward is measured | cross-teaching-in-scale literature suggests making it **bidirectional** rather than one-way stop-grad; keep one side stop-grad initially (collapse risk), relax only if it helps | [ML] |
| L5 | Eikonal | G, T | +3-5% | villa ships it | IGR: Eikonal alone carves a valid SDF from oriented points, i.e. it is load-bearing, not decoration — but it gives *no* localisation, so the Huber-on-distance still has to carry that | [ML] |
| L6 | normal-vs-radial alignment | O, P | +3% | usrm used the same test as a validity flag | unchanged: gate, do not enforce | [MH] |
| L7 | topology (Betti / ECT) | M, G | high | unablated | **now decided**: pilot **fast ECT** (arXiv:2507.23763, explicit 2D+3D, stated cost advantage over PH); fall back to Efficient Betti Matching 3D (arXiv:2407.04683) only if ECT under-delivers. Pitfall the in-house version missed: patch cropping truncates real features at every crop edge, so compute topology terms on an **interior sub-block only** | [ML] |
| L7b | **homotopy warping / DMT** (new) | **M, G** | one DT pass + warping search | none | the PH-family members that attack *both* failures symmetrically with an explicit 3D claim and no PH library. DMT was built for "weak spots of connections and membranes" in 3D EM — structurally identical to two touching wraps. C4 (warping) is the better first pilot; C5 (DMT) is the better match but the heavier build | [ML] |
| L8 | **skeleton recall / clDice** (new) | **G** | near-zero (offline hard skeleton) | none | Skeleton Recall Loss (ECCV 2024) claims up to 90% of clDice's GPU cost removed; SOTA connectivity on 3D tubular benchmarks. **Caution, load-bearing**: a merge bridge is itself thin and connected, so clDice-family losses can *reward* it. Gaps-only tool; never a merge fix | [ML] |
| L9 | swap-invariant face loss | — | — | tsm needed it, we do not | unchanged, do not port | [MH] |
| L10 | per-head loss balancing | R | ~0 | tsm built it, shipped OFF, never validated | **two literatures disagree and the disagreement resolves cleanly**: Kendall uncertainty weighting has an on-point precedent (AHU-MultiNet: seg+SDF+contour, the exact head set we are heading toward), *but* "In Defense of the Unitary Scalarization" and "Do Current MTO Methods Even Help?" both find tuned fixed weights match or beat balancers for <5 same-family losses. Rule: keep fixed weights while the heads are all BCE+dice; revisit Kendall (not tsm's EMA balancer) only once a regression head is in the sum | [ML] |
| L11 | feature distillation (DINO / cached encoder) | — | large | implemented in tsm, never measured, two bugs found | **stays dead, and the replacement is named**: in-domain CNN-native masked-cube pretraining (Wald/Isensee CVPR 2025, +3 Dice on 8 datasets with a ResEnc U-Net, gains concentrated at low label counts) is what that idea should have been | [ML] |
| L12 | winding-density barrier (mesh) | M | very high | — | unchanged, out of scope | [S] |

### 1d. TRAINING RECIPE (new category, from `lit_optimisation_schedules` / `lit_scaling_...`)

| # | change | cost | what the literature says | conf |
|---|---|---|---|---|
| R1 | **WSD schedule** — warmup, flat plateau, cosine over only the last ~10% | ~0, uses the existing `LambdaLR` + `grow` machinery | MiniCPM/WSD: the step budget need not be committed at run start, which is exactly how these runs are actually sized (ad hoc, resumed, extended). Today `steps` shapes every step's LR from step 0 | [ML] |
| R2 | **EMA window as a fraction of run length** (`1 - k/steps`, k~1000-3000) rather than fixed 0.999 | ~0 | 0.999 is 1.7% of a 60k run and 0.5% of a 200k one. Also: an EMA val curve going flat is not proof of convergence — track raw-weight dice for any cooldown trigger | [ML?] |
| R3 | **Re-warmup on warm start + a separate LR group for new channels/heads** | ~0 | Ash & Adams: resuming at a decayed LR generalises worse than from scratch; continual-pretraining work says a fresh short warmup beats resuming on the old tail. New params (cascade slot, verso row) carry no memory to protect and can take full LR from step 0 | [ML?] |
| R4 | **per-rung temperature calibration** | ~0 (one scalar per rung) | Dice-trained nets are measurably overconfident (Mehrtash). Our sigmoid is not a probability *by construction* above native rung (pooled fractions). Fit T per rung, only where the target is a genuine binary band | [ML] |
| R5 | **EMA-vs-live + symmetry-TTA disagreement as the uncertainty signal** | ~free (both already computed/implemented) | cheapest members of the cheap-ensemble family; MC dropout and EDL are explicitly not worth it here. Pitfall: a model trained on the 48-symmetry group is taught invariance to it, so use ranked spread, not absolute thresholds | [ML] |
| R6 | **anneal `--cascade-self-p`; make cascade noise resemble real coarse errors** | ~0 | scheduled sampling (Bengio 2015), OneSeg (native 3D, names this exact problem), RITM. Also a hard warning: unconstrained multi-step self-feedback diverges without damping — our one-level truncation is a *property to preserve deliberately* | [ML] |
| R7 | **agreement-weighted teacher fusion; GLC-style per-source weights from the meshes** | inference-only / one offline pass | today §3 flat-averages two lineages whose bands differ systematically (32% vs 23% coverage). UA-MT/SRPL-SFDA: gate by agreement, not one threshold. GLC: use the small trusted set to set per-(source x bucket) weights, skipping bilevel meta-learning, which does not scale to 3D | [ML] |
| R8 | **size/data ladder, fitted per rung** | 3-4 runs | no Chinchilla-style law exists for 3D dense segmentation; STU-Net (14M-1.4B) still gained at 1.4B; scaling saturates task-dependently, so fit per rung, and watch the train/val gap as the saturation sign, not dice alone. BioVFM prior: at rung 2 our unique-window supply means **data is probably not the constraint; params might be** | [ML] |
| R9 | **Paganin jitter, micron-based blur sigma, shuffled artefact order** | one FFT op + a list shuffle | SinoSynth: randomise the *composition order* of the detector/scan-domain chain; physics-augmented CBCT training reached 0.74 Dice on real CBCT with zero real CBCT labels. Our `_unsharp`/`_blur` sigmas are in voxels, so the same config is a different physical blur at each rung | [ML] |
| R10 | **in-domain masked-cube pretraining** | one extra training run | CNN-native MAE, in-domain, +3 Dice avg, concentrated at low label counts and on thin structures; structure-aware masking amplifies the gain on topology metrics specifically (VAMAE). Do **not** import a CT/medical foundation checkpoint, and do not switch to ViT to make a recipe drop in | [ML] |

### 1e. EVALUATION (new category, from `lit_evaluation_metrics`)

| # | metric | cost | why | conf |
|---|---|---|---|---|
| E1 | **noise ceiling** — run the whole evalsurf suite teacher-vs-mesh, report every number as "X (ceiling Y)" | ~0, `--teacher` already exists | turns "is 0.91 saturated?" from a judgement into a comparison. The single cheapest item in this entire document | [ML] |
| E2 | **ERL** (expected run length before a break or a merge, in physical units) | small extension of `continuity`'s `mean_run` | connectomics' answer to "how far can I trace before it breaks" — the closest published analogue to our actual quantity of interest, length-weighted and symmetric over splits and merges, unlike `merge_frac` (merge-only) + `continuity` (split-only, unitless) | [ML] |
| E3 | **Betti number error**, then Betti matching error periodically | cheap / moderate | `merge_frac` and `continuity` are local hand-built proxies for b0/b1 defects; they cannot see a hole spanning more than a couple of grid cells or a handle tangent to the surface. Mask box-edge cells out or a cut sheet reads as a defect | [ML] |
| E4 | **bootstrap CIs over surfaces + a plateau fit** | pure post-processing | two adjacent checkpoints' difference is currently unfalsifiable. Resample *surfaces*, not points. For "saturated", fit a bounded (sigmoid) curve, need >=10-15 checkpoints, smooth first | [ML] |
| E5 | HD95 / P99 of the offset distribution, per-surface not pooled | ~0 | one badly-off patch hides inside a box mean; a tracer breaks on the worst excursion, not the mean | [ML] |

---

## 2. The tracer contract (updated)

The fitter (`scripts/spiral/fit_spiral.py` via `lasagna_data.py`) reads three dense fields plus geometry. The
§2 contract of the first synthesis stands, with four literature-driven changes:

1. `recto`, `verso` uint8 — unchanged.
2. `sdist` uint8, `sd_working_voxels = (v-128)*0.25`, cap ±32, v=0 = no-data, positive **outside** the sheet
   body — byte-identical to `make_surf_sdt.py`, so `lasagna_data.py` needs no change. **Changed:** train it
   internally as distance to the **midline** with a bounded half-thickness (O3b), and convert to the
   recto-face convention at export. The export is a subtraction; the training-time representation is what
   buys non-crossing. [ML]
3. `normal` uint8 x3, ZYX, `(v-128)/127`, full signed vector, `dot(n, radial) > 0`; a two-line shim emits the
   tracer's `nx`/`ny` hemisphere pair. **Changed:** derive it from the *stored* sdist by Scharr, never from
   the decoder's autograd gradient (SIREN pitfall). [ML]
4. `conf` uint8 — **promoted from optional**: get it free as the log-variance of a heteroscedastic sdist head
   rather than as a separate O7 head. [ML]
5. `phase` uint8 — **only** as `k mod M` with a CORN head and a validity mask, never a global index. [ML?]
6. **New, and it deletes a pipeline stage:** once `sdist` exports, run **Flying Edges / marching cubes per
   1024^3 shard** with halo stitching on the shard grid we already have, instead of `make_surf_sdt.py`'s
   threshold-and-EDT round trip. Optionally **tensor voting** on the normal field as a per-region gap closer
   between extraction and repair (training-free, targets continuity directly), and a **targeted min-cut**
   using the normal field as edge weight *only* at merge candidates flagged by a medial-axis fork test or the
   `overlap` metric — never a whole-volume solve. [ML]

Pitfalls unchanged and still load-bearing: ZYX everywhere; units are working voxels so `voxel_um` must be
recorded per store; 0 is reserved for no-data in every uint8 distance field; CT==0 is masked by both sides.
Two new ones: **do not use screened Poisson** anywhere in this path (documented failure mode is merging
nearby thin layers and bridging holes — our exact worst metric), and **blind topology repair cannot tell a
true tear from a true fusion**, so any repaired or completed geometry must carry a provenance flag to the
tracer. [ML]

---

## 3. Dead — do not re-try as-is

Kept from v1, all still dead, now with external corroboration where it exists:

1. **Learned coarse/global winding from a foreign checkpoint** [MH]. tsm: 42% of windows within tolerance
   against a 60% gate, ~10 conventions swept, validation harness passed on constant input. **Externally
   reinforced**: every global-ordering method surveyed (RGT, PhaseNet, OCT DP, tree rings) assumes tens of
   layers with hundreds of voxels of spacing; we have ~300 wraps at 15-35 voxels, and both the seismic and
   the phase-unwrapping literatures report the learned global field smearing exactly at density. [ML?]
2. **Collapsing recto+verso into one signed field** [MH]. tsm measured it three ways (0.636-0.669 two-face vs
   0.507 and 0.36 single-field). The UDF literature appears to argue the opposite for open surfaces and does
   not: it is right for a *lone* sheet, wrong for a *pair* where the sign is the pairing signal. [ML]
3. **Feature distillation (DINO / cached encoder)** [MH] — replaced by R10, in-domain masked-cube
   pretraining, which is the same instinct done the way the literature says works. [ML]

New dead-on-arrival items, from the external surveys:

4. **Structure-tensor / sheetness / Hessian as an input channel** — no 2018-2026 paper does this and reports
   a gain; the standard uses are post-hoc analysis or pre-CNN cleanup. [ML]
5. **Mamba / attention backbone swap** — the controlled re-benchmark (nnU-Net Revisited, MICCAI 2024) finds
   CNN nnU-Net variants beat Transformer and Mamba networks once training budget and augmentation are
   equalised; most headline wins were recipe confounds. We would also re-litigate all of §14's throughput
   work. [ML]
6. **Schedule-Free AdamW, Muon, SOAP, Sophia, Lion, full muP, GradNorm/PCGrad** — no evidence on 3D conv
   dense prediction; the optimiser step is ~1% of our wall clock; Muon's matricisation of a 5D conv kernel is
   undefined; Lion is least stable exactly at batch 2. Shampoo is the one "someday" exception. [ML]
7. **BatchNorm at batch 2 in 3D** — known-bad, ~10 points worse top-1 in GroupNorm's own ablation. Keep
   GroupNorm; group count (8 vs 16 vs 32) is the only cheap knob and it was picked in a *speed* study. [ML]
8. **MC dropout, evidential deep learning, full deep ensembles** for uncertainty — MC dropout needs a retrain
   plus N passes for a signal EMA-vs-live approximates free; EDL is a loss/head rewrite that collides with
   the tuned BCE+dice+ignore design; deep ensembles are K training runs. [ML]
9. **Screened Poisson surface reconstruction** — documented thin-layer merging. [ML]
10. **Full MALIS, inference-time mutex watershed, embedding+margin losses** — all presuppose an instance
    partition we have decided not to build; borrow the *repulsive long-range edge* concept only. [ML]
11. **Sinogram-domain noise/ring simulation, mosaic-seam synthesis, procedural papyrus CT** — we never see
    sinograms; real mosaic data is already in the mirror; procedural fibre synthesis is a subsystem, not an
    `aug.py` change. Defer the last one until rung 0/1 coverage is genuinely exhausted. [ML]
12. **Equivariant / steerable convolutions and spherical ODF heads** — 2-5x per-layer cost for sample
    efficiency on *continuous* rotations that our exact 48-symmetry augmentation already covers, solving a
    harder problem (crossing fibres) than recto-vs-verso's bimodal near-orthogonal case. [ML]

Weaker-evidence dead items carried over: the `thick` head as a supervised priority, fibre class-mode targets,
the double-angle `dir0/dir1` encoding, tsm's swap-invariant face loss, and gating a fine rung on a coarse
field.

---

## 4. Phased roadmap

### Phase A0 — measurement, before any loss change (NEW, and it moves to the front)

The literature's most uncomfortable point is that our decision rule is currently untestable: one number per
box per checkpoint, no CI, no ceiling, and two metrics (`merge_frac`, `continuity`) that are local proxies
for topology. Doing this first makes every later ablation cheap and falsifiable.

- **Noise ceiling** (E1): rerun the full suite with `--teacher` against the published meshes; report
  "X (ceiling Y)" everywhere afterwards. Also compute teacher-vs-teacher (recto lineage vs m7) as a second,
  independent ceiling — one teacher's ceiling is itself noisy.
- **ERL** (E2): walk both grid axes of each published surface in physical units, stop at the first band loss
  (split) or second crossing (merge, already computed by `merge_runs`), report length-weighted expectation.
- **Betti number error** (E3) on the thresholded band restricted to a box interior, box-edge cells masked.
- **Bootstrap CIs** (E4) by resampling surfaces with replacement, N=200-1000; **HD95/P99 per surface** (E5).

Cost: CPU only, no GPU-hours, no training. Metric: none — this *is* the metric work. [ML]

### Phase A — label-free losses on the live run

**L3** soft exclusivity, **L4** cascade self-consistency, then **L8** skeleton-recall (gaps) and **O12**
long-range affinity (merges), with **L1**/**L2** demoted behind O12.

- `usrm2/train.py`: `losses_aux(logit, wv, radial, coarse=None, ...)` returning a dict, summed into
  `bce + dice`, flags `--w-excl --w-cascade-consist --w-skelrec --w-affinity --w-repel --w-planar`, every
  default 0.0 so existing runs stay bit-for-bit — the `--cascade off` / `--verso` discipline.
- L3: `relu(p_r + p_v - 1)` masked by `wv[:,0]*wv[:,1]`. Keep an exclusivity metric on a **held-out rung
  carrying no exclusivity weight**, or `overlap` stops being evidence.
- L4: `F.avg_pool3d(sigmoid(logit), 2)` against the no-grad EMA coarse block already computed in
  `self`/`mix`. Stop-grad the coarse side (it already is). Cost ~0.
- L8: build the hard skeleton **offline**, per rung, reusing the same EDT machinery Phase B needs — this is
  the Skeleton Recall trick that removes clDice's per-step cost. Gaps only; it can reward a bridge.
- O12: k affinity output channels at offsets along ±n tuned to the 15-35-voxel pitch, targets derived
  geometrically from the existing binary pyramid (same wrap = positive, different wrap = negative).
- **R6** in the same window: anneal `--cascade-self-p` from near 0 toward ~0.7 instead of fixed 0.5, and make
  `--cascade-noise`'s erosion/dilation/dropout representative of real coarse errors (thin/boundary), not
  uniform. One-line schedule change plus a noise-shape change.

Step cost: L3, L4, R6 ~0; L8 ~0 with an offline skeleton; O12 small (k extra head channels + BCE);
L1+L2 ~+15-20% [S]. Warm start: network unchanged for everything but O12, so `u3` resumes with flags on.

**Metric and decision rule:** `merge_frac` 0.44 -> below 0.40 (O12, L3) and ERL up / `continuity` 0.656 ->
above 0.68 (L8, L4), each outside the Phase-A0 bootstrap CI. Drop any term whose loss curve visibly moves
while its target metric does not.

### Phase B — distance and normal heads from existing masks

**O3b** midline-signed distance with a bounded half-thickness (+**L5** Eikonal, **L6** radial gating),
**O4** normal by Scharr of the stored field, **O7** as the log-variance of O3b.

Target generation (`usrm2/targets.py: dist_pyramid`): per rung where the source is near-binary (native, at
most one above), threshold at 0.5, 3D EDT **in that rung's voxel units**, sign from the radial vector, clamp
±32, uint8 offset 128 unit 0.25. **Distance does not 2x-mean-pool** — every rung is computed independently.

Weights 0: above native+1; within a configured radius of the axis (tsm measured `recto_is_in` ~0.47 near the
core and *dropped* those voxels — do the same); where CT==0; outside the box.

Code: `data.Patches._rung_build` appends channels; `model.build(cout=...)` grows to 5-6; `train.losses_tw`
gains a per-channel loss *kind* (BCE+dice for probabilities, Huber on the decoded distance, `1-dot` for
normals). `train.warm_start` needs an explicit per-channel policy — `copy mod n` for probability channels,
**zero-init at the encoding's zero (128)** for regression channels; the current `mod n` rule would copy a
probability filter into a distance slot. Keep the loss weights **fixed** (L10's literature), and only
consider Kendall-style uncertainty weighting once the regression head is actually in the sum.

**Metric:** `offset<=3` 0.36 -> above 0.45, `offset_std` and HD95 down, `recall@4` not regressing.

### Phase C — the manifold pair, by construction where possible

Prerequisite: verso coverage wide enough that `dice_verso` appears for the val box at rungs 2-3, plus
Phase B.

The single biggest revision here: **the cortical-surface field has moved from penalty-based to
construction-based non-crossing, and our Phase C was squarely in the older camp.** [ML]

- **Non-crossing by construction**: predict the midline distance and a half-thickness `t/2` bounded below by
  a minimum physical thickness (rung-dependent: usrm measured 15-35 voxels at 2.4 um, so 7-17 at rung 3);
  recto and verso are `midline ± (t/2)·n`. Crossing becomes unrepresentable at zero extra parameter cost.
- **Ordering by construction**: replace L3's elementwise hinge with a 1D soft-ordering constraint sampled
  along the predicted normal (OCT differentiable-DP / non-negative-cumsum pattern): along each short ray,
  recto and verso must occur in a fixed order with a minimum gap.
- **Penalties kept, but as a backstop**: L1 with the antiparallel exemption *removed* for recto-recto and
  verso-verso pairs and *kept* for recto-verso; O12's affinities do the heavier lifting.
- **Topology**: pilot **homotopy warping** (L7b) or **fast ECT** (L7) at one mid rung, on an **interior
  sub-block** of the patch. Betti matching 3D is the fallback and a hard C++ dependency.

New `usrm2/manifold.py`, flags `--w-manifold-excl/-pair/-repel`, `--manifold-rungs 2,3`. Step cost
+10-20% [S]. **Metric:** `merge_frac` headline, plus a pair-completeness metric and the held-out exclusivity
metric — once exclusivity is a loss, `overlap` is no longer independent evidence.

### Phase D — tracer export, cross-scan conditioning, corpus fine-tune

- **Export**: `usrm2 export-tracer` writing the surf-SDT store with OME multiscales and the `nx`/`ny`
  hemisphere pair; then `fit_spiral.py` needs no code change and `make_surf_sdt.py` drops out. Add per-shard
  Flying Edges as the mesh path (§2).
- **I3 + I2 planes** before the 42-scroll fine-tune, not after; channel order
  `[CT, ctx_1..9, CASCADE, radius, meta_1..m, scale, radial(3)]`, zero-filled and explicitly slotted in
  `warm_start` exactly as the cascade channel was.
- **R9 physics augmentation in the same change**: tie `_unsharp`/`_blur` sigma to microns
  (`sigma_vox = sigma_um / voxel_um(rung)`), add `_paganin_jitter` (one FFT: the Paganin transfer-function
  ratio at a resampled delta/beta, log-uniform 0.5-2x), and shuffle the order of
  `_bias/_ring/_stripe/_cor/_haze/_unsharp/_quant` per sample. The literature is explicit that you should
  **condition on the true parameters and augment across plausible ones**, and that the two must agree in sign
  so the model never sees a physically impossible combination.
- **R7 source weighting** before the corpus grows: agreement-gated teacher fusion and GLC-style per-(source ×
  coarse bucket) weights measured against the verified meshes, replacing the flat 1.0/1.0/0.3.
- **R10 pretraining** is optional here and should be decided by its own ablation (Experiment 11).

**Metric:** cross-scan — per-scroll recall@4 / ERL on held-out scrolls, and the **variance** across them.

---

## 5. Experiments to run, in order

Cost anchors: A100 `30m6` at 256^3 batch 2 runs ~27-32 Mvox/s measured, i.e. ~1.05-1.25 s/step, so **10k
steps ≈ 3-3.5 GPU-h**; `--cascade mix` multiplies that by 1.35 (**~4.5 GPU-h / 10k steps**). The desk 5060 Ti
pair runs `5m` at 128^3 DDP at ~15.8 Mvox/s, so 10k steps there is ~1 h — but the desk cards are currently
saturated by the teacher/verso region jobs, and a 5m/128^3 result establishes the *sign* of a loss-form
change, never its final numbers. The 5090 (32 GB) cannot hold the 256^3 batch-2 production config (~45-62 GB
peak) and is best used at 128^3 or batch 1.

| # | experiment | hypothesis | flags / changes | cost | metric + decision rule | desk? | conf |
|---|---|---|---|---|---|---|---|
| 1 | **evalsurf upgrade** (Phase A0) | our current metrics cannot falsify a 2-point change, and we do not know the ceiling | add noise ceiling (`--teacher`), ERL, Betti-0/1 on the interior, bootstrap CI, HD95 per surface | 0 GPU-h, ~1 day CPU | none — gating work. Exit: every metric reported as "X (ceiling Y) ± CI" | n/a | [ML] |
| 2 | **L3 + L4 + annealed `cascade-self-p`** on the live `u3` | the three ~free label-free terms move pairing and cross-rung gaps | `--w-excl 0.1 --w-cascade-consist 0.1`, anneal self-p 0.1->0.7 | 2 arms x 10k steps on the A100 with cascade ≈ **9 GPU-h** | `merge_frac` and ERL vs the Exp-1 CI. Keep a term only if its own target metric moves outside the CI | yes, for sign | [MH]+[ML] |
| 3 | **O12 long-range affinity vs L1 repulsion** | a relational signal at the sheet pitch beats a normal-gated penalty on merges | arm A `--w-repel`, arm B affinity head (offsets ±8,16,24,32 along n), arm C both | 3 arms x 10k ≈ **14 GPU-h** | `merge_frac` and ERL-merge. Decision: ship whichever single arm beats baseline by more than the CI; drop L1 if C ≈ B | partly (128^3 limits the useful offset range) | [ML] |
| 4 | **L8 skeleton recall** | a near-free gaps-only term closes band breaks without hurting merges | offline hard skeleton per rung + `--w-skelrec` | 1 arm x 10k ≈ **4.5 GPU-h** + ~2 h CPU for skeletons | `continuity` / ERL-split up; **abort if `merge_frac` rises** — a bridge can score well under a skeleton loss | yes | [ML] |
| 5 | **O3b midline sdist + Eikonal**, vs a recto-face sdist | a distance head fixes sub-voxel localisation, and midline-parameterised is not worse than face-parameterised | `dist_pyramid`, cout 2->4, Huber + `--w-eikonal`, per-channel warm-start policy | 2 arms x 20k (a regression head needs longer) ≈ **18 GPU-h**, plus ~1 day offline EDT | `offset<=3` 0.36 -> >0.45 and HD95 down, `recall@4` not down. If midline ≈ face on localisation, pick midline for Exp 6 | no (needs the 256^3 config to be comparable) | [MH]+[ML] |
| 6 | **construction vs penalty pairing** | bounded-offset recto/verso from a midline beats an exclusivity+repulsion penalty on merges | arm A Phase-C penalties, arm B `midline ± (t/2)n` with `t` lower-bounded, arm C + soft-ordering along n | 3 arms x 20k ≈ **27 GPU-h**, needs Exp 5 | `merge_frac`, pair-completeness, held-out exclusivity. Decision: if B or C halves the merge gap to the ceiling, Phase C is rewritten around construction and L1/L3 become backstops | no | [ML] |
| 7 | **topology-loss pilot: homotopy warping vs fast ECT** | a critical-voxel/ECT term attacks merges *and* gaps where BCE+dice are blind | one mid rung only, interior sub-block only, `--w-topo` | 2 arms x 10k ≈ **9 GPU-h** + build time | Betti error and `merge_frac`. Decision: adopt only if Betti error improves *and* step cost < +20%; otherwise stop — do not escalate to the Betti-matching C++ dependency on a null result | no | [ML] |
| 8 | **teacher fusion + source weights (R7)** | agreement-gated fusion and mesh-measured per-source weights beat flat averaging of two systematically different bands | offline: per-voxel agreement gate on the two lineages; GLC-style weights per (source × rung × agreement bucket) | ~0 GPU-h for the statistics; 1 arm x 10k ≈ **4.5 GPU-h** to test | val dice per rung and recall@4 vs the flat-weight baseline. Pitfall to test explicitly: where both teachers agree *and are wrong*, only the meshes can tell | yes | [ML] |
| 9 | **schedule triple: WSD + EMA window + re-warmup (R1-R3)** | the recipe is worth more than any optimiser swap, and costs nothing | plateau LR, cosine over the last 10%, `ema_decay = 1 - k/steps`, short re-warmup + separate LR group for new params at every warm start | 3 arms x 10k ≈ **14 GPU-h**, or free if folded into an existing rung transition | val dice at matched steps, plus decay-length sensitivity 5/10/20%. Decision: adopt WSD if it matches cosine at matched steps — its value is optionality, not accuracy | yes | [ML?] |
| 10 | **physics augmentation + metadata planes (I3, R9)** | conditioning + matched augmentation is what makes one model work across 42 scrolls | `_paganin_jitter`, micron-based sigmas, shuffled artefact order, metadata planes zero-filled at warm start | 2 arms x 20k ≈ **18 GPU-h** | **held-out-scroll** recall@4/ERL and the variance across scrolls — not the aggregate mean. Decision: keep only if the cross-scroll variance falls; a mean-only gain is not the point | partly | [ML] |
| 11 | **in-domain masked-cube pretraining (R10)** | a CNN-native MAE stage on our own CT buys 1-3 points, mostly on topology metrics, mostly at rungs 0-2 | mask-and-reconstruct head on the existing encoder at rungs 0-4, structure-aware masking once a proxy mask exists, then discard the head | pretrain ~20k steps ≈ **7 GPU-h** + 2 fine-tune arms x 10k ≈ **9 GPU-h** = **~16 GPU-h** | fine-tune dice and ERL vs from-scratch at matched fine-tune steps. Decision: adopt only if the gain survives at the label counts we actually have; audit how many *distinct scans* the pretraining corpus spans first — voxel count is not scan diversity | yes (pretraining is loader-bound and 128^3-friendly) | [ML] |
| 12 | **size ladder, fitted per rung (R8)** | at rung 2 our unique-window supply means params, not data, are the constraint | 15m / 30m / 60m at matched steps, depth and width scaled together | 3 runs x 20k ≈ **35-50 GPU-h** | fit `1-dice` vs `log(params)` **per rung**; call saturation only when the slope flattens across >=3 sizes *and* the train/val gap grows | no | [ML] |

Suggested order: 1 (gating), then 2 and 4 (nearly free, on the live run), then 3, then 9 and 8 (recipe and
label quality, cheap), then 5, then 6, then 7, then 10, then 11 and 12. Experiments 1, 2, 4, 8, 9 and 11 can
be started or sanity-checked on the desk; 5, 6, 7 and 12 need the A100's 256^3 configuration to produce
numbers comparable with `u1`/`u3`.

---

## 6. Top 5 recommendations (revised)

1. **Fix the measurement before fixing the model (Phase A0 / Experiment 1).** [ML] This was not in v1 and it
   should have been first. We are tuning against `merge_frac 0.44` and `continuity 0.656` with no confidence
   interval, no noise ceiling and no global topology check, while our labels are a machine teacher whose own
   Dice ceiling is ~0.90-0.93. The literature's answer is mechanical and costs zero GPU-hours: run the
   existing suite teacher-vs-mesh for a ceiling, turn `mean_run` into a physical-units ERL, add Betti-0/1 on
   a box interior, and bootstrap over surfaces. Everything below becomes falsifiable the moment this lands.

2. **Phase A's free losses, but with the priority order changed: L3 + L4 + annealed cascade-self-p first,
   then skeleton-recall for gaps and a long-range affinity head for merges — and L1 demoted.** [ML] The
   external evidence is lopsided: villa's `NormalGatedRepulsionLoss` is unablated anywhere, while the EM
   connectomics field's answer to the identical problem (a repulsive relational signal at a fixed offset,
   mutex-watershed/MALIS/LSD lineage) is measured at CREMI/FIB-25 scale. Our sheet pitch (15-35 voxels) is
   exactly the offset range that encodes "next wrap". Annealing `--cascade-self-p` off its fixed 0.5 is a
   one-line change closing a textbook exposure-bias gap, with a native-3D precedent (OneSeg).

3. **The distance head, reparameterised: predict distance to the MIDLINE with a lower-bounded thickness, and
   derive recto/verso by a bounded offset.** [ML] v1 recommended a face-signed distance head and a Phase-C
   penalty for non-crossing. The cortical-surface literature — the closest analogous problem, two nested
   non-intersecting locally-parallel manifolds — has moved decisively from penalties to construction:
   DeepCSR needs a >30-minute-per-defect post-hoc topology fix, Vox2Cortex and PialNN are explicitly flagged
   by later papers as crossing-prone, and CortexODE / TopoFit / Coupled Reconstruction get non-intersection
   free from an invertible offset off a shared midline. OCT's non-negative-thickness cumsum is the same idea
   at voxel-head cost. This is a small change to Phase B (the EDT is the same; only the sign origin and the
   head parameterisation move) that makes a merge structurally unrepresentable rather than merely penalised,
   and it turns usrm2's negative `thick`-head result into a non-issue by making thickness a parameter rather
   than a supervised target.

4. **Scan-metadata conditioning planes AND the matching physics augmentation, together, before the 42-scroll
   fine-tune.** [ML] v1 recommended the planes; the literature says planes alone are half the fix. The
   domain-generalisation result that matters is that physics-augmented training reached 0.74 Dice on real
   CBCT with *zero* real CBCT labels, and SinoSynth's synthetic-only training beat real-but-narrow training
   for cross-site robustness. Concretely: condition on the true energy/delta_beta/pixel size, and augment
   across their plausible range in the same change — a `_paganin_jitter` FFT op, blur sigmas defined in
   microns rather than voxels (today the same config is a different physical blur at every rung), and a
   shuffled artefact composition order. Score it on cross-scroll *variance*, not the aggregate mean.

5. **Treat the training recipe as a first-class lever and the optimiser as settled.** [ML?/ML] WSD scheduling
   removes the need to commit a step budget at run start — which is how these runs are actually managed —
   using machinery `grow`/`resume` already has; EMA's window should be a fraction of run length, not a fixed
   1000 steps across 60k-200k-step runs; every warm start (rung change, cascade channel, verso head) should
   get a short re-warmup and a separate LR group for the new parameters rather than resuming on a decayed
   tail. Together these are near-zero-risk and the literature is unambiguous that they matter more than any
   optimiser swap, all of which (Schedule-Free, Muon, SOAP, Sophia, Lion, muP) lack any 3D-conv evidence and
   would displace a working recipe for nothing. Add per-rung temperature calibration in the same pass: our
   sigmoid is not a probability above native rung by construction, and every uncertainty-gated mechanism
   downstream depends on it being one.

## 7. The things not to do

1. **Do not collapse recto and verso into one signed field.** [MH] Unchanged, and the implicit-surface
   literature's apparent counter-argument (UDF for open surfaces) does not apply to a *pair* where the sign
   carries the pairing — and a UDF's gradient is undefined at the zero set, which would poison the normals.

2. **Do not build a global winding or layer-index field, learned or regressed.** [MH]+[ML?] tsm's failure is
   now backed by every external field that tried: seismic RGT, OCT DP, tree rings and phase unwrapping all
   assume tens of layers with generous spacing and all degrade at density. If a winding signal is wanted it
   is **local** (`k mod M` with a CORN head, validity-masked, tracer-labelled where labels exist) plus a
   **classical** quality-guided unwrapping pass outside the network — the sinusoidal RGT and multi-frequency
   results support the encoding, not the global regression.

3. **Do not add feature distillation or import another model's dense fields as inputs.** [MH]+[ML] Both
   remain dead; the correct version of that instinct is in-domain, CNN-native masked-cube pretraining, which
   ships with an ablation (Experiment 11) rather than an assumption — and must, because tsm's DINO line
   produced exactly zero evidence in either direction.

4. **Do not chase architecture.** [ML] No Mamba/attention backbone (controlled re-benchmarks say the wins
   were recipe confounds), no equivariant/steerable convolutions (2-5x cost for rotations our exact
   48-symmetry augmentation already covers), no hypernetwork conditioning while constant planes work, no ViT
   swap to make a pretraining recipe drop in.

5. **Do not let any model-derived uncertainty signal be trusted on merges.** [ML] Every signal derived from
   the model's own weights — TTA spread, EMA-vs-live, MC dropout, learned variance — systematically
   under-flags that model's *confident, systematic* failures, which is precisely `merge_frac`. Use them for
   round and region selection and for pseudo-label gating, and always pair them with a structural check that
   does not depend on the model's confidence. Relatedly: nnActive (the largest 3D biomedical active-learning
   benchmark) found **no query method reliably beats a foreground-aware random baseline** — so score regions
   with cheap disagreement at region-store granularity, validate against that random baseline, and do not
   build BALD/core-set machinery.

6. **Do not blindly repair topology, and never let repaired geometry reach the tracer unflagged.** [ML] A
   repair algorithm cannot distinguish a true tear from a true fusion; a confidently-wrong fill is worse than
   a visible gap in a document whose content will be read.
