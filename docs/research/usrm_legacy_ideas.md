# Legacy usrm ideas for unified-model input/output/loss (2026-09-21)

Survey of older usrm (src/usrm/) multi-task heads, auxiliary channels, and post-processing features that could become unified-model inputs, outputs, or losses. The unified model (unified_design.md section 1-4) currently trains ONE recto head against published and exported mask pyramids; this surveys what richer supervision was available in the predecessor.

## Summary of legacy architecture

usrm HEAD_ORDER (`src/usrm/model.py:96`): `("recto", "udf", "sdist", "thick")`. Each is a 1x1x1 output head trained with per-head loss weights; a model can use any prefix of this list. Input: `(image, valid, conf, radial_channels)` plus optional `udf`, `sdist`, `thick` fields from labels.

## Candidate ideas for usrm2

### 1. Distance-field outputs: UDF and SDIST

**What it computed**: Two scalar fields per site (src/usrm/labels.py, docs/labels.md):
- `udf` (unsigned distance field): `round(clip(d, 0, T) / T * 127)` where T = 6.0 level voxels, decoded as `d = udf.astype(f32) * T / 127`. Scalar surface distance, 0 at recto face, saturated at 6 voxels.
- `sdist` (signed distance field): `round(clip(d, -T, T) / T * 127)`, positive on papyrus side (outward normal direction), negative in air. Used for ablations and reconciling face ambiguity.

**How**: Two separate 1x1x1 convolution heads (`model.py:head_sdist`, `head_thick`). Loss terms: Huber on `T * (pred - target)`, masked by `valid` tiers and medial confidence (`losses.py:100-140`). Included soft Dice variant (`sdf_dice` at weight 0.5).

**Status/Results**: Legacy used full 4-head setup on 2.4 um (recto, udf, sdist, thick). The published recto masks are binary (0/255 at every level); UDF/SDIST came from distance transforms of those masks, or from traced surfaces via label store (`labels_medial/`). Signed head dropped in practice because it saturates everywhere (depth ambiguity dominates), kept for ablations ("sdist only" → `losses.recto_prob_from_sdist` sigmoid band).

**Mapping to unified model**:
- (a) INPUT: Likely too expensive as extra input channels (would need pre-computed DFTs per rung). Cascade channel (section 22) already provides coarse context.
- (b) OUTPUT: Could be extra heads beyond recto/verso (cout 3+). Adds `w0` parameters per head (negligible). Measured cost would be ~2 Huber terms in loss.
- (c) LOSS: Could supervise recto indirectly via UDF targets (the label pyramid already has them at every rung post-export). Would require importing distance-field label pyramids for bootstrapping rungs 0-11, similar to recto targets.

**Pitfalls**: UDF targets come from binary masks (nearest-neighbor distance transform), so upsampling to rung 0/1 loses sub-voxel precision. Thick head and medial confidence scaling (60+% of loss logic) are tied to medial-specific label sources; legacy had separate `labels_medial/` store.

---

### 2. Sheet thickness (THICK head)

**What it computed**: Per-voxel sheet **half-thickness along the normal** in 0.5-voxel units (uint8, 0 = unknown). Only in medial label stores (`labels_medial/`), not standard recto masks.

**How**: 4th head `model.head_thick` (1x1x1 conv). Target from medial labels: `thick * 0.5` voxels. Loss: Huber masked to `thick > 0` (0 means unknown, not zero-thickness), weight 0.5 (configurable `loss.w_thick`). Confidence scaling by medial_conf (0..255 → [0.25, 1.0] factor) via `losses.medial_factor`.

**Status/Results**: Synthetic ground truth (`data.py:SYNTH_HALF=2.5`) proved the head can learn something. Real medial labels (from traced surfaces + CT profiling) showed ~15-35 voxel sheets at 2.4 um. Ablation study (`losses.HEAD_WEIGHTS`) showed dropping `thick` weight to 0.0 increased recto dice slightly, suggesting a minor auxiliary signal.

**Mapping to unified model**:
- (a) INPUT: Could be a post-hoc target from field-traced surfaces (self-labelling in later rounds after recto converges). Would need medial-surface extraction at training time (expensive, `ctfaces.py` + `field.py`).
- (b) OUTPUT: Extra output head (cout 3+), same cost as UDF. Requires thick targets. Current bootstrap has no published thick source across rungs; would need to generate from recto predictions via field analysis.
- (c) LOSS: Secondary loss term to encourage locally smooth thickness estimates. No direct pixel-wise targets in published data.

**Pitfalls**: Medial label stores are a separate, smaller dataset (Paris 4 only). Self-supervision of thick from the recto prediction requires field tracing on the fly (10-100x slower than forward pass per site). Thick=0 convention is fragile (users expect "unknown" but code interprets as "don't train").

---

### 3. Inner/outer face classification (CT-derived)

**What it computed**: Per-voxel binary classification (src/usrm/ctfaces.py): is this a sheet boundary voxel, and which side of its sheet is it on (inner/recto, outer/verso, or ambiguous)?

**How**: No model head; pure post-processing on CT. Morphological closing (radius 6) → body mask. Boundary voxels classified by local march along radial (from umbilicus): if body continues outward → inner face; inward → outer face; both/neither → ambiguous. Thick bodies (>40 voxels) marked ambiguous to avoid welded-sheet artifacts.

**Status/Results**: Used as label-quality source (S2) in legacy eval pipeline: compared against published tracings (S1). Detected welding, holes, face-assignment errors. Diagnostic: thickness histogram peaks at single-sheet range (15-35 vox) confirms correct papyrus/air split.

**Mapping to unified model**:
- (a) INPUT: Could be 1-2 extra input channels (inner_face, outer_face, ambiguous uint8). Computed from CT once per site (cheap, no backprop needed). Would tell model which face polarity is physical.
- (b) OUTPUT: Not necessary (recto is the unified face, verso is backside).
- (c) LOSS: Could weight recto loss higher near detected faces (confuse_weight scaling). Requires running `ct_faces` preprocessing on every training batch (expensive unless cached).

**Pitfalls**: Sensitivity to CT threshold (fold handled per-scroll in `ctfaces.BODY_THRESHOLD_BY_SCROLL`). Morphological operations (scipy.ndimage) are CPU-bound. Thin-sheet welds are genuinely ambiguous; labeling them wrong adds noise. Body thickness is local; false positives on wrapped papyrus (same body voxels viewed radially from different angles).

---

### 4. Winding number field

**What it computed**: Continuous winding (rotation count) around the umbilicus axis in turns. Only used post-inference for surface tracing (src/usrm/tracer.py `winding_relax`), not as a model output or input.

**How**: `_theta(axis, p)` computes `atan2(y - axis_y(z), x - axis_x(z))` / 2π per point. Seed-pinned Jacobi relaxation over committed grid cells, 24 iterations. Edge winding checked against `winding_max=0.25` turns (illegal if larger). Relaxed vs. raw disagreement clamps confidence.

**Status/Results**: Essential for tracer correctness (prevents wrap-jumping), but not learned by the model. Adds ~15% CPU cost to surface tracing; runs after field inference is complete.

**Mapping to unified model**:
- (a) INPUT: Not useful (the model sees CT crops, has no window into global scroll topology).
- (b) OUTPUT: Not necessary (tracer computes it from axis alone post-inference).
- (c) LOSS: Not applicable.

**Pitfalls**: Purely geometric; offers no signal to improve surface prediction. Tracer depends on it for correctness, so dropping it would break topology.

---

### 5. Medial-surface features (sheet middle, run endpoints, confidence)

**What it computed**: For every vertex, the offset to the sheet's **midpoint** (not face), thickness, and confidence (how sharply the two faces are defined in CT) via `field.py:Field.medial_along()`. Used in tracer seed placement and topology logic.

**How**: Threshold CT at Otsu valley (clamped 50-110), partition papyrus at UDF ridges inside that run, measure confidence from CT step (air end) or UDF barrier (handover end). No model supervision; post-processing on predicted UDF field.

**Status/Results**: Medial surface sits ~half-thickness off published recto tracings (difference of convention, not error). Enables touching-sheet handover detection and thickness readout. Confidence score used in tracer to downweight unreliable vertices.

**Mapping to unified model**:
- (a) INPUT: Could pre-compute run-midpoint offsets from the recto field and feed them as 1 extra channel. Would require computing field on CPU during data loading (slow). Alternative: explicit medial-surface target at rung 2/3, distilled from field.medial_along results.
- (b) OUTPUT: Extra outputs (midpoint offset, thickness, run_conf) as in legacy thick head. Would need medial labels to supervise.
- (c) LOSS: Encourage predictions to have sharp, well-separated ridges (implicit in UDF Huber; no direct medial loss needed).

**Pitfalls**: Medial surface is not a voxel grid output; it's a per-vertex scalar on a 2D traced grid. Translating to rung voxels loses the parameterization. Confusing medial-surface targets with per-voxel thickness (the legacy `thick` head) leads to ambiguous gradients. Field computation requires UDF, so cannot supervise both simultaneously in early training.

---

### 6. Grid normals and surface orientation

**What it computed**: Oriented normals at each surface vertex (src/usrm/upsample.py `oriented_grid_normals`, src/usrm/geom.py `orient_normals`). Normals from Catmull-Rom tangents, sign resolved by majority vote per 64x64 block against radial (away from umbilicus). Flagged unreliable where `|dot(n, radial)| < 0.3`.

**How**: Pure post-processing on published surface grids. No model output; normals are derived from grid geometry.

**Status/Results**: Used for validation (sign consistency check) and downstream tracing (predicting wrapping direction). Not part of model training loop.

**Mapping to unified model**:
- (a) INPUT: Surface normals cannot be inputs (model has no access to traced grids during training). Could add implicit normal supervision via gradient consistency: `grad(udf)` at face should align with radial.
- (b) OUTPUT: Not necessary (tracer computes normals from interpolated field).
- (c) LOSS: Encourages UDF field to have well-defined gradients pointing radially. Could add `||grad(udf) - radial||` term. Low priority; recto probability already implicitly penalizes multi-modal fields.

**Pitfalls**: UDF gradients are zero at surfaces (definition of local minimum), so enforcing radial alignment only makes sense near (not on) the surface. Gradient computation in loss adds cost.

---

## Not candidates

**Spiral/fiber direction** (no legacy code): Intra-sheet writing direction never modeled. Tracer can infer it post-hoc from grid curvature; it is not a supervisable field.

**Skeleton/medial axis** (only used post-inference): Would need explicit morphological computation during training (scipy.ndimage.distance_transform_edt on every batch), expensive.

**Cascade channel** (already implemented, section 22): Rung-(k+1) prediction upsampled 2x as 15th input channel. Adds 6% cost, improves consistency across resolution ladder.

---

## Recommendations for prioritization

1. **UDF + SDIST as extra outputs** (medium priority): Already have label pyramids post-export; adds negligible model cost. Loss supervision is straightforward (Huber). Ablation value: shows whether multi-task learning on distance helps recto.

2. **Thick head** (low priority): Requires separate medial-label pipeline or self-labelling from field traces. High data prep cost. Likely small marginal gain (legacy ablation showed minimal improvement).

3. **CT inner/outer faces as inputs** (low priority): Cheap to compute; could disambiguate face-assignment errors. Requires per-scroll CT threshold tuning. Test on a small validation set first.

4. **Implicit normal supervision** (research): Add `grad(udf)` alignment term if UDF output is adopted. Measure against recto-only baseline.

5. **Winding, medial surfaces, skeleton** (not applicable): Already handle via post-processing; not actionable as model supervision.

---

**File references**: /home/forrest/usrm/src/usrm/model.py (HEAD_ORDER, head definitions), losses.py (loss terms, ablations), data.py (radial_channels, label loading), field.py (medial_along, normal computation), geom.py (axis, orientation), ctfaces.py (inner/outer classification), tracer.py (winding, field interface).
