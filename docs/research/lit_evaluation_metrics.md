# Literature: evaluating thin-structure / topology-critical segmentation (outside Vesuvius Challenge)

Scope: what fields with the same failure mode — a wound 2-manifold that must stay a single connected
sheet, not a blob with the right volume — use to tell "good" from "saturated," and what a Dice-vs-noisy-
labels number cannot tell us. Written against `usrm2/evalsurf.py`, which today computes: `recall@{2,4,8}`
(probability band reaches threshold within r voxels of a published point along its normal), `offset_mean/
std/le3` (sub-voxel peak location bias), `precision6` (EDT of thresholded voxels to nearest published
point), `merge_runs/merge_frac` (extra threshold crossings along the normal ray = other sheets fusing in),
and `continuity/hit_frac/mean_run` (8-neighbour hit consistency across the surface grid = fragmentation).
Dice-against-machine-labels ceilings at ~0.90-0.93 (see `docs/unified_design.md` sec. 20); this survey is
about what to add so "saturated on Dice" and "actually traceable" stop being different claims.

## 1. Surface Dice / Normalised Surface Distance (NSD)

**Definition.** Nikolov et al. (DeepMind, 2018/2021, head-and-neck OARs) define surface Dice as the
fraction of predicted and reference boundary surface within a tolerance τ of each other — a boundary
version of Dice with a slack parameter instead of exact-voxel overlap. Formalised and packaged in
**Metrics Reloaded** (Maier-Hein et al., *Nature Methods* 2024, preprint 2022,
metrics-reloaded.dkfz.de) as NSD: only boundary points within τ of the other set's boundary count as
true positives, so annotator jitter smaller than τ is free.

**What it captures / misses.** Captures boundary *localisation* under known label noise, decoupled from
region volume — exactly the axis evalsurf's `offset_*` already targets, but NSD is symmetric (precision
and recall on both boundaries) where evalsurf's `precision6`/`recall@r` are currently reported separately
and at different radii. It does **not** see topology: a surface cut into two disjoint patches that are
each locally within τ scores the same NSD as a single connected sheet.

**Cost.** O(boundary voxels), trivial in 3D once you have a surface mask; cKDTree same as `precision6`
already does. Free.

**Recommendation.** Fold `recall@r` and `precision6` into one paired NSD@τ number with τ swept (2, 4, 8
px) instead of two differently-scoped metrics — mainly a reporting/roll-up change, low priority since
the info already exists.

## 2. Hausdorff distance and its robust variants

**Definition.** HD = max over one boundary of the min distance to the other (symmetrised = Hausdorff
distance both ways). HD95 (95th percentile) and average symmetric surface distance (ASSD) are the
standard clinical-imaging variants that drop outlier points so one stray voxel doesn't dominate.

**What it captures.** Worst-case boundary error — the number a surgeon or a segment-tracer actually
cares about, since one bad excursion breaks a trace even if 99.9% of the sheet is dead-on. Dice and mean
NSD are dominated by the "easy" 99% and are blind to this.

**Cost.** Cheap (distance transform / KD-tree), but sensitive to outliers if unclipped (hence HD95).

**Recommendation.** Add HD95 of the `offset` distribution (`evalsurf.py` already has per-point offsets;
report the 95th percentile of `|offset|` where a band is found, not just mean/std) and, separately, a
max/P99 over `precision6`'s distances restricted to the *worst* surface patch, not pooled over all
patches — pooling hides a single badly-off patch inside a large box average.

## 3. Betti number error and Betti matching error

**Definition.** Betti number error (Hu et al. 2019, and used widely since) compares topological invariants
(#components b0, #loops b1) of prediction vs. ground truth — counts only, so a hole in the right place and
a hole in the wrong place score identically. **Betti matching error** (Stucki, Paetzold, Shit et al.,
ICML 2023, "Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes,"
proceedings.mlr.press/v202/stucki23a) fixes this: it uses persistent homology and induced matchings
between persistence barcodes to require topological features to match *spatially*, not just in count, and
is differentiable (usable as a loss, not just a metric). An efficient GPU implementation for 3D exists
(Stucki et al., "Efficient Betti Matching Enables Topology-Aware 3D Segmentation via Persistent
Homology," arXiv:2407.04683, 2024) — relevant since a 3D volumetric persistent-homology metric is
otherwise expensive.

**What it captures.** Exactly the wound-2-manifold failure mode: a spurious handle (two nearby sheet
patches fused into one), a spurious hole (a patch dropped from an otherwise-continuous sheet), or a
split (one sheet reported as two components) — all things a thresholded, spatially-pooled Dice can absorb
into a 1-2% score change while being catastrophic for a human or algorithm tracing the surface. evalsurf's
`merge_frac` (extra crossings along the normal — sheet fused to a neighbour in depth) and `continuity`
(8-neighbour hit consistency — in-plane fragmentation) are hand-built, cheaper, task-specific proxies for
exactly b0/b1 defects, but they are local/heuristic, not a principled global topological invariant: they
can't see a hole that spans more than a couple of grid cells, or a handle that isn't a normal-ray
double-crossing (e.g. one that runs tangent to the surface).

**Cost.** The 2024 efficient implementation makes 3D volumes tractable (was previously prohibitive);
still meaningfully more expensive than the local proxies — budget it for full-box eval, not per-step
training monitoring.

**Recommendation (high priority).** Add Betti number error (b0, b1 counts, cheap, from a Euler-
characteristic / connected-components pass on the thresholded probability volume restricted to a band
around the published surface) as a coarse global topology gate, and Betti matching error via the
efficient-Betti-matching arXiv:2407.04683 code as the periodic (not per-checkpoint) deep check — this is
the single most direct match to "does this trace as ONE manifold" that the current metric suite lacks
entirely. Ground truth for b0/b1: the published tifxyz surface itself, restricted to the same box.

## 4. clDice and skeleton recall/precision

**Definition.** Shit et al. (CVPR 2021, "clDice — A Novel Topology-Preserving Loss Function for Tubular
Structure Segmentation," arXiv:2003.07311; github.com/jocpae/clDice) skeletonise both prediction and
ground truth, then define Topology Precision = fraction of predicted skeleton inside GT mask, Topology
Sensitivity = fraction of GT skeleton inside predicted mask, and clDice = harmonic mean of the two.
Theoretically guarantees homotopy-equivalence preservation for binary 2D/3D masks under certain
conditions. Follow-ups: **Skeleton Recall Loss** (arXiv:2404.03010, 2024) drops the differentiable
skeletonisation bottleneck for large 3D volumes; **cbDice/cl-X-Dice** family adds distance/radius
weighting to fix clDice's insensitivity to caliber (diameter) errors.

**What it captures.** Designed for *tubular* structures (vessels, roads, airways) but the mechanism
generalises to any thin manifold: a metric that only rewards mask-skeleton agreement is much more
sensitive to small connectivity breaks than voxel Dice, because a single missing voxel on a 1-voxel-wide
skeleton can disconnect a whole branch, whereas the same voxel is noise-level in a bulk Dice.

**What it misses.** clDice as originally defined is for curvilinear (1D-skeleton) structures; a papyrus
sheet is a 2D manifold, so the direct analogue is a *medial-surface* recall/precision, not a
line-skeleton one — worth checking whether a 2D skeletonisation (surface thinning) is stable enough on a
sampled probability volume to be useful, or whether it is noisier than the tifxyz-grid-based `continuity`
metric already in evalsurf, which is effectively a discretised version of the same idea (grid-neighbour
consistency instead of a computed skeleton).

**Cost.** Skeletonisation of a large 3D volume was the historical bottleneck; Skeleton Recall Loss
(2024) addresses this directly and evalsurf's problem is smaller in scale (per-box) than typical use.

**Recommendation.** Lower priority than Betti matching given evalsurf already has a continuity/fragmentation
proxy; worth a cheap experiment (surface-thinning recall/precision on the thresholded band) only if
`continuity` and Betti number error disagree in a way that's hard to diagnose — a medial-surface metric
would triangulate between "grid-local defect" (continuity) and "global topology defect" (Betti).

## 5. Connectomics split/merge metrics: VOI, adapted Rand error, ERL/NERL

**Definition.** Variation of Information (VOI) decomposes into VOI-split and VOI-merge, an
information-theoretic distance between the predicted and GT partitions of voxels into objects — captures
both over- and under-segmentation with directionally separate terms. Adapted Rand error (1 − adapted Rand
index) is a pairwise-voxel-agreement analogue, used e.g. in the SNEMI3D EM-segmentation challenge, but
requires dense voxel-level GT and is impractical at scale (per the ERL paper's own critique). **Expected
Run Length (ERL)**, Januszewski et al. (Google, "High-precision automated reconstruction of neurons with
flood-filling networks," *Nature Methods* 2018; see also Google AI blog "Improving Connectomics by an
Order of Magnitude," 2018) is the physically-grounded fix: for each GT neurite skeleton, walk it and find
the expected Euclidean path length traversable before hitting a merge or split error, weighted across
skeletons by their length. **NERL** (normalised ERL, some later papers) divides by GT skeleton path length
per-neurite so long and short neurites are comparable.

**What it captures.** ERL is exactly "how far can I trace before the model breaks the sheet" — the
single closest existing-field analogue to "tracing a single wound 2-manifold." It is length-weighted
(a broken 2mm segment costs more than a broken 2-voxel segment) and merge/split are both first-class
failure types, unlike evalsurf's current `merge_frac` (merge only; nothing symmetric for split/
fragmentation except the separate `continuity` metric).

**Cost.** Needs a GT *skeleton graph* with edges (walkable path), not just point cloud + normals; for
evalsurf this means either using the tifxyz UV grid connectivity (already implicit — rows/cols of the
grid are the walk graph) or building one. Computing ERL means, per grid row/column walk, finding the
first point where the predicted band drops below threshold or where a second surface's band gets picked
up (merge) and recording the run length, then averaging weighted by GT length — this is close to what
`continuity`'s `mean_run` already half-computes (run length along grid rows) but currently isn't tied to
physical (voxel/µm) distance or symmetrised for splits vs. merges the way ERL is, and only walks rows,
not the full connectivity graph.

**Recommendation (high priority, cheapest big win).** Reframe `continuity`'s `mean_run` into an actual
ERL: walk both grid axes (rows and columns, or a proper graph walk over the tifxyz UV mesh) of each
published surface within the box, in physical distance units (voxels or µm, not grid cells, since grid
spacing isn't 1 voxel), find the first break (band lost) or merge (a second surface's band detected
along the same ray, already computed by `merge_runs`) per walk, and report expected run length before
either — this is a straightforward extension of code already in evalsurf.py and lines up eval directly
with "how much of a trace survives before manual intervention is needed," which is the actual quantity of
interest, not just a Dice number's proxy for it.

## 6. Metrics under label noise: ceilings, agreement, multi-rater bounds

**Definition/practice.** Standard result across medical imaging: inter-rater agreement (kappa, or
mean pairwise Dice between annotators) sets an empirical performance ceiling — a model scored against a
single noisy rater cannot exceed, in expectation, what that rater's own noise permits relative to a
"true" consensus. STAPLE (Warfield et al.) and majority-vote/consensus labels are the standard way to
build a less-noisy reference from several noisy ones; **Metrics Reloaded** (Maier-Hein et al. 2024)
explicitly recommends reporting multi-annotator variability alongside any single metric and warns against
treating one annotator's mask as ground truth without disclosing that ceiling. Recent segmentation-noise
papers (e.g. IMA++ dermoscopy 2025, federated noisy-label benchmark 2026) formalise "consensus
cleanliness" scoring via class-wise Fleiss' kappa plus boundary metrics (HD95) to separate *label noise*
from *model error* before reporting a headline number.

**What it captures for usrm2.** This is directly the 0.90-0.93 Dice ceiling story already documented in
`docs/unified_design.md` sec. 20 — the literature's answer to "is 0.91 good or saturated" is: compute
the *same* Dice/NSD/Betti-matching metric between two independent noisy label sources (or two thresholds
of the same teacher's probability, or two teacher checkpoints, or the machine label vs. the published
mesh) and treat that number as the ceiling, not 1.0. If the model-vs-teacher Dice is statistically
indistinguishable from teacher-vs-teacher-noise Dice, the model is saturated on that label source and
only the surface-mesh metrics (Section 20's recall@4/continuity/merge/offset) can still show real
progress, because the meshes are human-verified rather than noisy machine labels.

**Recommendation (do first, cheap).** Compute a noise ceiling directly: run evalsurf's own metric suite
with the *teacher's* probability volume in place of the model's (the `--teacher` flag already supports
this) against the published meshes, and separately compute teacher-vs-(model-trained-on-different-seed
or different-epoch) agreement, to get an empirical upper bound for every metric evalsurf reports, not
just Dice. Report every future number as "X (ceiling Y)" rather than bare X. This turns "saturated?" from
a judgment call into a number comparison.

## 7. Statistical practice: bootstrapping, per-region variance, learning-curve plateau tests

**Practice.** Metrics Reloaded and general ML-eval best practice: report metrics per-case/per-region, not
only pooled over a box, and bootstrap over regions (resample surfaces/patches with replacement, recompute
the mean, get a CI) rather than trusting a single point estimate from one validation box — a box that
happens to contain one easy, flat, well-scanned sheet patch and one hard, damaged, low-contrast patch will
have huge per-region variance that a pooled mean hides. For "is training saturated," the ML-scaling
literature (Hestness et al. 2017 "Deep Learning Scaling is Predictable, Empirically," arXiv:1712.00409;
and the general scaling-law-fitting literature, e.g. the 2024 OpenReview survey "(Mis)Fitting Scaling
Laws") fits a curve of the form error ≈ a·n^(−α) + c (power law plus an irreducible floor c) to
metric-vs-training-step or metric-vs-data-size, and calls a run saturated when the fitted c dominates —
i.e., when adding more data/steps moves the fit's power-law term by less than the run-to-run noise
(bootstrap CI) in c itself. Practical notes from that literature: smooth/monotonize noisy curves before
fitting (raw per-checkpoint eval numbers are not monotone), and prefer a sigmoid over a pure power law
when the metric is bounded in [0,1] (Dice, NSD, ERL-fraction all are) since power laws don't saturate
naturally at a ceiling <1.

**What it gives usrm2 that a single eval number doesn't.** A rigorous distinction between "this
checkpoint's Dice went from 0.905 to 0.907, meaningless noise" and "the metric has structurally stopped
improving, plateau confirmed." Right now evalsurf reports one number per box per checkpoint with no CI,
so any two adjacent checkpoints' differences are unfalsifiable.

**Recommendation.**
1. Bootstrap: evalsurf already keeps per-surface point clouds (`sites()` returns per-surface counts) —
   resample surfaces (not individual points, since points within one surface are highly correlated) with
   replacement, recompute the metric suite N=200-1000 times, report median + 90% CI per metric. This
   is a pure post-processing change on data evalsurf already computes.
2. Plateau test: log each metric (recall@4, ERL from #5, Betti matching error) per checkpoint across
   training, fit `c + a*step^-α` (or a logistic/sigmoid for bounded metrics) with the last ~30% of
   checkpoints, and call it saturated when the fitted improvement-per-1000-steps is smaller than the
   bootstrap CI width from (1) — i.e. stop training/adding data for that metric when signal < noise, not
   on a fixed step budget.

## 8. Adjacent domains worth a look but lower priority here

- **Cortical surface reconstruction** (CortexODE, arXiv:2202.08329; CorticalFlow++, MICCAI 2022): reports
  self-intersecting-face percentage and Chamfer/ASSD together, because a mesh can be geometrically close
  to GT while still being non-manifold (self-intersecting) — directly relevant if usrm2 ever meshes its
  probability band into an explicit surface rather than scoring the raw volume; CortexODE's diffeomorphic-
  flow parametrisation *guarantees* no self-intersection by construction, an idea worth remembering if a
  meshing/refinement stage is added later, but not an evalsurf metric today since evalsurf never
  constructs a mesh.
- **OCT layer segmentation**: reports Mean Absolute Distance in physical units (µm) rather than Dice,
  specifically because Dice is biased by natural thickness variation across the volume — same argument
  evalsurf already implicitly makes by using `offset_mean/std` (a physical-unit boundary-position metric)
  instead of leaning on Dice alone. No new idea to import beyond confirming the current design choice is
  aligned with the field's consensus.

## Bottom line: concrete additions to evalsurf, in priority order

1. **Noise ceiling** (Section 6): run every existing + new metric teacher-vs-mesh and report as ceiling
   alongside model-vs-mesh. No new code beyond re-invoking `metrics()`/`continuity()` on the teacher store.
2. **ERL** (Section 5): extend `continuity`'s row-run-length computation into a proper expected-run-length
   over physical distance, walking the full grid graph and using existing merge detection for the "hit a
   merge" stopping condition.
3. **Betti number error, then Betti matching error** (Section 3): add a component/loop count on the
   thresholded band vs. published surface as a cheap global-topology gate; add Betti matching error
   (arXiv:2407.04683 code) as a periodic deep check, not a per-checkpoint one.
4. **Bootstrap CIs + plateau fit** (Section 7): pure statistics on data evalsurf already produces; turns
   "is it saturated" into a testable claim instead of eyeballing two numbers.
5. Lower priority: NSD roll-up (Section 1), HD95/P99 (Section 2), medial-surface (2D clDice analogue,
   Section 4) — useful cross-checks but each overlaps substantially with a metric evalsurf already has.

## Pitfalls carried over from this literature

- Pooling any metric over a whole box hides one badly-broken patch inside many good ones (Section 2, 7) —
  always keep and report per-surface breakdowns, not just the pooled mean evalsurf currently prints.
- Betti-type metrics need a well-defined GT topology at the box boundary; a box edge that cuts a sheet
  is a "false" boundary defect no metric should charge the model for — mask box-edge cells out of any
  topology computation, the way the existing `inner` mask already does for `continuity`.
- A noise ceiling computed once, from one teacher, is itself noisy — the recommended fix is comparing at
  least two independent noisy sources (Section 6), not trusting a single teacher-vs-mesh number as *the*
  ceiling.
- Power-law plateau fits are unstable with few checkpoints and unsmoothed data (Section 7); don't trust a
  saturation call from fewer than ~10-15 checkpoints, and smooth/monotonize first.

## Key citations

- Nikolov et al., "Deep learning to achieve clinically applicable segmentation of head and neck anatomy
  for radiotherapy," 2018/2021 (surface Dice).
- Maier-Hein et al., "Metrics reloaded: pitfalls and recommendations for image analysis validation,"
  *Nature Methods*, 2024 (preprint 2022) — metrics-reloaded.dkfz.de.
- Stucki, Paetzold, Shit et al., "Topologically Faithful Image Segmentation via Induced Matching of
  Persistence Barcodes," ICML 2023, PMLR 202:32698-32727 (Betti matching error).
- Stucki et al., "Efficient Betti Matching Enables Topology-Aware 3D Segmentation via Persistent
  Homology," arXiv:2407.04683, 2024.
- Shit et al., "clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation,"
  CVPR 2021, arXiv:2003.07311; github.com/jocpae/clDice.
- Skeleton Recall Loss, arXiv:2404.03010, 2024.
- Januszewski et al., "High-precision automated reconstruction of neurons with flood-filling networks,"
  *Nature Methods*, 2018 (ERL); Google AI Blog, "Improving Connectomics by an Order of Magnitude," 2018.
- Hu et al., "Topology-Preserving Deep Image Segmentation," NeurIPS 2019 (Betti number error, persistent
  homology loss).
- Hestness et al., "Deep Learning Scaling is Predictable, Empirically," arXiv:1712.00409, 2017.
- "(Mis)Fitting Scaling Laws: A Survey of Scaling Law Fitting Techniques in Deep Learning," OpenReview,
  2024-2025.
- CortexODE, arXiv:2202.08329; CorticalFlow++, MICCAI 2022 (self-intersection metrics).
- OCT layer segmentation surveys reporting Mean Absolute Distance in µm as the boundary-localisation
  standard over Dice.
