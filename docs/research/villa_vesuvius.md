# villa/vesuvius: auxiliary targets, losses, and what usrm2 could reuse (2026-09-21)

Scope: `/home/forrest/villa/vesuvius` (the `vesuvius` training/inference package) plus a look at
`/home/forrest/villa-volcomp` (a superset checkout of the same monorepo, nothing newer for this topic) and
`/home/forrest/villa/volume-cartographer` (the tracer, vc3d). Read against
`/home/forrest/usrm2/docs/unified_design.md` sections 1-4 and 21-23 (the 12-rung ladder, 14/15-channel
input, recto+verso output, cascade channel).

No separate "villa" repo exists; `/home/forrest/villa/` is itself the monorepo (vesuvius, lasagna,
ink-detection, volume-cartographer, scripts). Nothing here trains directly on our target pyramids (uint8
0..255 fraction-of-surface at a rung) or our 256^3-cube-with-context sample shape; upstream is a nnU-Net-style
multi-task trainer over a config-driven target/loss graph, plus a separate autoregressive UV-tracing model
(`neural_tracing/`) that predicts displacements on a mesh, not per-voxel probabilities.

## 1. Every auxiliary target/channel found

All of these live under `models/training/auxiliary_tasks/*.py` (target *generation*, run once per sample from
a binary "source_target" mask already in the training data) and `models/training/loss/*.py` (the losses that
score them). A `task_type` string selects the generator; a `BaseAuxTrainer` subclass per type
(`models/training/trainers/auxiliary/*_trainer.py`) wires it into the training loop by calling
`_compute_aux_tensor` on each sample's source mask.

| name / task_type | file : function | computed from | output shape / range | loss(es) that score it |
|---|---|---|---|---|
| surface normals | `aux_surface_normals.py : compute_surface_normals_from_sdt` | binary mask -> signed distance transform (`scipy.ndimage.distance_transform_edt`, inside/outside) -> Scharr gradient of the SDT, normalized | (3,D,H,W) or (2,H,W), unit vectors, float32, **sign is outward** (gradient of outside-minus-inside) | `CosineSimilarityLoss` / `SignInvariantCosineLoss` (losses.py:167,234), or `NormalSmoothnessLoss`, `NormalGatedRepulsionLoss` (see below) |
| structure tensor | `aux_structure_tensor.py : compute_structure_tensor` | SDT (or raw binary) -> `StructureTensorComputer` (Holoborodko derivative kernels, `image_proc/geometry/structure_tensor.py`) -> Gaussian-smoothed outer-product tensor components | (6,D,H,W) upper-triangle `Jzz,Jzy,Jzx,Jyy,Jyx,Jxx` (3D) or (3,H,W) (2D); background set to `ignore_index=-100` | masked MSE / `EigenvalueLoss` (regress eigenvalues as an unordered set) |
| in-plane / tangent direction | `aux_inplane_direction.py : compute_inplane_direction` | same structure tensor, eigen-decomposed; **smallest**-eigenvalue eigenvector = the sheet's tangent (highest local coherence direction) | (3,D,H,W) unit vector, background = `ignore_index` | `SignInvariantCosineLoss` (sign-invariant: a tangent has no canonical sign) |
| distance transform | `aux_distance_transform.py : compute_distance_transform` | binary mask -> `distance_transform_edt`; modes `signed` / `inside` / `outside` | (1,...) float, voxel units, unbounded | `SignedDistanceLoss` (Smooth-L1 + optional Eikonal `‖∇d‖≈1` + Laplacian smoothness term + `surface_sigma` band weighting) |
| nearest-component vector+distance | `aux_nearest_component.py : compute_nearest_component` | SDT (optionally Gaussian-smoothed by `sdf_sigma`) -> `-∇sdf` normalized (points into the nearest correct surface) + `|sdf|` | (C_dir+1,...): 3 direction + 1 distance (3D) | direction: cosine loss on the vector channels; distance: masked regression |
| surface frame (t_u, t_v, n) | `models/training/loss/surface_frame.py` + `cross_frame_dataset.py`, trained by `SurfaceFrameTrainer` | **not** derived from a mask at train time — comes from a pre-baked "surf_norm_uv" zarr (label authored in a UV/mesh-native frame, `dataset_config.data_path` in `configuration/single_task/surface_frame.yaml`). Upstream (traced elsewhere, likely `neural_tracing/datasets/direction_helpers.py : _compute_surface_tangent_axis`, which finite-differences a `tifxyz` mesh grid `(H,W,3)` along each UV axis) — see  4.2 | (9,D,H,W): 3 vectors x 3 comps, `t_u`=tangent along one mesh-grid axis, `t_v`=the other, `n`=normal | `SurfaceFrameMultiTermLoss` (`surface_frame.py`): direction term (1-cos per vector), frame-alignment term (‖R_pred^T R_gt - I‖²), orthogonality term |
| topology (ECT) | `ect_loss.py : ECTLoss` | NOT derived from a new label — matches the **Euler Characteristic Transform** of the predicted probability volume against the target's, over `num_directions` projection directions (Fibonacci-sphere by default) and `resolution` filtration thresholds; differentiable via a sigmoid-softened indicator | scalar loss; no new voxel channel | itself a loss (`mass` variant = cumulative projected mass under a half-space sweep; `chi` variant = approximate expected Euler characteristic of a random cubical complex) |
| topology (Betti matching) | `betti_losses.py : BettiMatchingLoss`, `spherical_betti_loss.py : SphericalBettiLoss` | persistent-homology birth/death pairs of prediction vs. target (external `betti_matching` package, lazy-loaded via `_load_betti_module`); `spherical_` variant slices the volume with random planes/axis-aligned slices first, for 2D-persistence-on-slices (cheaper) | scalar loss (pushes unmatched pairs to the diagonal, matched pairs together) | itself a loss — the topology-correctness analogue of Dice: penalizes spurious loops/holes/handles (bridge merges, false splits) without needing a mesh |
| planarity | `losses.py : PlanarityLoss` | π = (λ2-λ1)/(λ0+λ1+λ2) from the *predicted* probability's own structure-tensor eigenvalues (no separate label — self-supervised regularizer on the prediction) | scalar; penalizes voxels where local structure isn't sheet-like | itself the loss |
| normal smoothness | `losses.py : NormalSmoothnessLoss` | compares a normal field (from a normals head) to its own Gaussian-blurred self, only where prob > threshold | scalar | itself the loss (no target needed — a local-coherence prior) |
| normal-gated repulsion | `losses.py : NormalGatedRepulsionLoss` | pure prediction-side: for every voxel pair within radius τ where prob>0.5 on both, penalizes voxels whose normals are near-parallel (weighted by proximity `exp(-‖Δx‖²/σ_d²)` and by angular similarity `exp(-θ²/σ_θ²)`) | scalar | itself the loss; this is the "two close-by, same-facing-normal blobs shouldn't both be foreground" prior — a **merge/self-intersection avoidance term** |
| growth direction (one-hot) | `neural_tracing/datasets/growth_direction.py` | not a mask-derived field — a constant one-hot plane over {left,right,up,down} broadcast to the patch, used by `neural_tracing` as a conditioning input telling the model which way it's extending the mesh | (4,D,H,W) constant plane | n/a (input, not target) |
| wrap-overlap band mask | `neural_tracing/datasets/detect_wrap_overlap_masks.py` | operates on **tifxyz meshes** (u,v -> x,y,z grids), not voxel volumes: for pairs of wrap segments, finds the row/column band where two spiral wraps' front/back surfaces run close and parallel (`cKDTree` nearest-neighbor + score threshold), writes `overlap_mask.tif` per wrap | 2D boolean mask in UV space | n/a (a labeling tool feeding manual/auto stitching, not a trained target) |
| surface-overlap (quad triangle) loss | `neural_tracing/loss/surf_overlap_loss.py` | 4 predicted cardinal UV points (u±, v±) define two 3D triangles; BCE+Dice against a GT mask that the model should occupy inside that quad | dense mask in a small local volume | `DC_and_BCE_loss` from `nnunet_losses.py` |
| UV-equidistance | `neural_tracing/loss/uv_equidist_loss.py` | penalizes irregular spacing among the 4 predicted cardinal-direction blob centroids (differentiable soft-threshold centroid extraction) | scalar | itself the loss — a **local-flatness / regular-parametrization** prior for the autoregressive UV tracer |
| inner/outer (2D "fiber" label fill) | `scripts/fill_inner_outer_labels.py` | 2D per-slice: alpha-shape (`alphashape` pkg) of labeled-pixel point cloud -> everything outside = "outer"; centroid + 72 angular bins -> innermost labeled radius per bin, shrunk 5% -> everything inside = "inner" | fills a 3rd label value (default 255) into the same label volume | n/a — a **label-authoring** tool (fills the umbilicus hole and the outside-the-scroll background in a 2D fiber-orientation label stack), not a network target itself |

Not found in this checkout, despite being named in the task: no "spiral"/winding **voxel** target or **network
output channel** anywhere in `vesuvius/`. "Spiral" only exists as (a) `volume-cartographer`'s C++/Python spiral
*fitting and sampling service* (`apps/diffusion/spiral*.cpp`, `scripts/spiral/*.py`) — a geometric fit of a
2D Archimedean-ish spiral to a scroll cross-section used for seeding/serving VC3D's UI, not a trained target
— and (b) the implicit winding-order encoded in a tifxyz mesh's UV grid (increasing one axis = out along the
spiral). There is no dense per-voxel "winding number" or "wrap index" channel to import.

## 2. How targets are generated (pipeline shape)

Every mask-derived aux target in `models/training/auxiliary_tasks/` follows the same recipe and the same
entry point (`base_aux_trainer.py : _inject_aux_targets`, called once per sample after the primary mask is
loaded, CPU-side, per-process, **at train time, not pre-baked**):

1. Take a *primary* binary segmentation target already in the sample (`source_target` name in the YAML).
2. `binary_mask = source[0] > 0`.
3. Compute a **signed distance transform** (`scipy.ndimage.distance_transform_edt` inside minus outside) —
   this is the shared substrate for normals, structure tensor, in-plane direction, nearest-component, and
   `SignedDistanceLoss`'s own regression target.
4. Differentiate / eigendecompose as needed (Scharr gradient for normals; Holoborodko-kernel structure
   tensor + eigh for tangent/normal-from-coherence).
5. Mask background with either `0` (normals — undefined direction zeroed) or a sentinel `ignore_index=-100`
   (structure tensor, in-plane direction) that the paired masked loss strips out.

This is **CPU-bound, per-sample, on the fly** — no caching, no pre-baked normal-grid zarr for these (unlike
`surface_frame`, which reads a pre-baked "surf_norm_uv" dataset). That matters for our pipeline: usrm2's
loader is GPU-augmentation-heavy and rung-indexed zarr pyramids; porting the villa recipe as-is (scipy EDT +
Scharr per sample) would be a CPU bottleneck at our patch rate unless precomputed into a target pyramid like
our masks already are, or done with a GPU EDT.

The **surface frame** (`t_u`, `t_v`, `n`) is the one target here actually built from a *mesh*, not a mask:
`neural_tracing/datasets/direction_helpers.py : _compute_surface_tangent_axis` finite-differences a tifxyz
`(H,W,3)` point grid along each UV axis (central difference where 3 consecutive valid points exist, one-sided
fallback at edges) to get `t_u`/`t_v`; `n` follows from their cross product (not shown in the excerpt read,
but implied by the 9-channel "3x3 frame" convention in `surface_frame.py`). This is the closest upstream
analogue to a "normal grid": it is the per-voxel-rasterized tangent/normal frame of a *specific, oriented*
mesh (one wrap, one side), authored once from segmentation meshes and rasterized into a voxel volume
(`data_path: /mnt/raid_nvme/datasets/raw/surf_norm_uv`), not generalizable across arbitrary binary masks the
way the SDT-based aux targets are.

## 3. Inference outputs and how the tracer (vc3d) consumes them

- `vesuvius/` inference (`models/run/`, not read in depth here) applies the trained heads and writes one
  probability/vector volume per target the model was configured with; nothing beyond that was found wired
  specifically to consume normals/structure-tensor/frame outputs downstream *inside this package* — those
  appear to be training-time regularizers and evaluation targets more than shipped inference products.
- `neural_tracing/inference/` (`infer_rowcol_triplet_wraps.py`, `infer_streamline.py`, `displacement_tta.py`,
  `generate_segment_cover_bboxes.py`) runs the autoregressive UV model: it predicts the next row/column of mesh
  points from CT + growth-direction conditioning, i.e. it directly *grows a tifxyz mesh* rather than
  segmenting a probability volume — a fundamentally different output contract from usrm2's dense recto/verso
  probability channels.
- `volume-cartographer` (vc3d, the C++ desktop tracer) consumes **tifxyz meshes** (`x.tif`/`y.tif`/`z.tif` +
  `meta.json`) as its native surface representation, plus the spiral-fitting service
  (`apps/diffusion/spiral*.cpp`, `apps/VC3D/Spiral*.{hpp,cpp}`) for interactive spiral-guided seeding/patch
  placement in its UI, and `python/vc/spiral_sampling.cpp` for sampling along a fitted spiral. It does **not**
  natively ingest a dense per-voxel probability volume with normal/tangent channels — the bridge from a
  probability volume (ours or villa's) to something vc3d uses is a separate meshing/patch-generation step
  (not in this slice's scope; `neural_tracing`'s streamline/wrap machinery and `lasagna`'s surface-fitting are
  the closer analogues, not read in depth here per the assigned slice).

## 4. Fit against usrm2's 12-rung, recto/verso, CT+context+radial model

| candidate | kind | label source | rungs it's usable at | what it would buy | villa evidence |
|---|---|---|---|---|---|
| **normal-gated repulsion prior** (no target needed) | LOSS (self-supervised on our own prediction) | none — computed from predicted recto/verso probability + a normals head, or approximated from prob alone via structure tensor | any rung with dense probability (2-9ish; degenerates at coarse rungs where "normal" is meaningless) | direct **merge/self-intersection avoidance**: penalizes two nearby, similarly-oriented foreground blobs, exactly the failure mode (adjacent wraps fusing) the unified-model doc flags as unsolved | `losses.py : NormalGatedRepulsionLoss` |
| **planarity prior** | LOSS (self-supervised) | none — structure tensor of our own prediction | rungs where a sheet occupies enough voxels to have local coherence (rung ~2-6) | keeps thin bands sheet-like instead of blobby; cheap regularizer, no new label needed | `losses.py : PlanarityLoss` |
| **topology loss (Betti/spherical-Betti or ECT)** | LOSS | our existing target pyramid (already binary/fractional per rung) vs. prediction | any rung; spherical-Betti is the practical one at 256^3 batch sizes (per-slice 2D persistence is far cheaper than full 3D Betti matching) | penalizes spurious handles/loops = **merge and false-split detection that Dice/BCE can't express**; complements the existing BCE+dice without a new channel | `betti_losses.py`, `spherical_betti_loss.py`, `ect_loss.py` |
| **SDT-derived distance-to-surface** (signed distance, `aux_distance_transform.py`) | OUTPUT channel, deep-supervision-compatible | our own binary/fractional mask pyramid -> EDT at the target's native rung, pooled the same way probability is (mean of |EDT| doesn't pool as cleanly as a binary fraction, needs care) | rungs with a binary-ish native mask (2-4); loses meaning where the target is already a soft fraction from upstream pooling | continuous target -> **smoother gradient signal near the boundary**, an Eikonal term is a free consistency check (`‖∇d‖≈1`), and this is exactly what usrm2's own section-9 "signed-distance ramp" export already does upstream of usrm2's loader — i.e. we could train a distance HEAD instead of (or beside) the binary/fraction head using data we already produce | `aux_distance_transform.py`, `losses.py : SignedDistanceLoss`; cross-ref `unified_design.md` section 9 |
| **surface normals** (SDT gradient, `aux_surface_normals.py`) | OUTPUT channel (extra 3-channel head) OR loss-only regularizer | derivable from our own recto (or verso) probability's SDT, no new label required, **but** sign convention (outward vs. inward) must be fixed relative to the radial vector we already carry, or trained sign-invariant (`SignInvariantCosineLoss`) | any rung with a coherent binary-ish surface (2-6ish; degenerate at coarse rungs where "surface" is a diffuse blob) | gives an explicit **recto/verso orientation signal for free** at rungs where both recto and verso are trained (the two channels' normals should point opposite ways) — could directly replace the ad hoc `verso.py --radial-sign -1` trick, and could feed the tracer's need for oriented surfaces | `aux_surface_normals.py`, and cross-ref `unified_design.md` section 23 (verso via sign flip) |
| **in-plane tangent / structure-tensor eigenvector** | OUTPUT channel or loss | structure tensor of our own SDT/probability | rungs 2-6 | could seed a coarse "wrap direction" or feed a future spiral/winding estimator, but the model's own receptive field can likely learn this internally (the unified doc's own section-21 rule: "local filters... fail that test since the first 3x3x3 layers learn them") — **low priority per usrm2's stated philosophy** | `aux_inplane_direction.py`, `aux_structure_tensor.py` |
| **radial distance from umbilicus normalization** | not villa-derived, but villa's SDT machinery is the right tool to reuse for *any* new distance-style channel we want (e.g., section 21 item 2, "normalised radius") | our own umbilicus geometry | all rungs | (already planned in unified_design.md §21 item 2, independent of villa) | n/a — noting only that the EDT/eikonal machinery in `losses.py`/`aux_distance_transform.py` is directly reusable code if we want an Eikonal consistency term on that channel |
| **surface frame (t_u,t_v,n) from meshes** | INPUT channel, sparse, refinement-only | published/traced tifxyz segments (where they exist) rasterized the way `direction_helpers.py` does (finite-difference a mesh grid) | wherever a segment mesh already covers the volume | exactly usrm2's own section-21 item 5, "existing segmentation meshes as a sparse known-surface channel... later (refinement / interactive mode only)" — villa's mesh-to-frame rasterization is the concrete recipe to reuse when that gets built, since it directly gives orientation + recto/verso side at meshed locations, not just a binary "known surface" bit | `neural_tracing/datasets/direction_helpers.py`, `cross_frame_dataset.py`, `surface_frame.py` |
| **ECT "mass" variant as a cheap global-shape term** | LOSS | prediction vs. our target pyramid, both already probability-valued | any rung, but its num_directions/resolution cost argues for a single mid rung (3-4) rather than every rung | fast (no persistent-homology dependency), differentiable proxy for "same overall shape", cheaper than spherical-Betti if full topology matching is too slow to run every step | `ect_loss.py` (mass variant needs no external `betti_matching` package, `chi` variant approximates the real Euler characteristic) |
| **inner/outer fill (`fill_inner_outer_labels.py`)** | not an output/input channel — a **label-authoring** technique | alpha-shape + angular-binned inner boundary of a 2D label slice | n/a (2D per-slice, pre-training) | irrelevant to usrm2's voxel pyramids directly, but the *idea* (fill the umbilicus hole and the scan's exterior in weight, not target) is functionally already handled by usrm2's own `weight = 1 where CT>0` rule (unified_design.md §3) — **no action needed**, we already solve the same problem a different way | `scripts/fill_inner_outer_labels.py` |
| **wrap-overlap band detection / neural_tracing UV machinery** | not portable as a dense channel | operates on tifxyz meshes, not voxel volumes | n/a | conceptually close to "spiral/winding" but is a mesh-space tool for the autoregressive tracer, not a dense per-voxel target our CNN could regress without first having a mesh — **skip unless usrm2 grows a meshing stage** | `neural_tracing/datasets/detect_wrap_overlap_masks.py` |

### Recommended priority (given usrm2's existing philosophy of "only add signal the receptive field can't compute")

1. **`NormalGatedRepulsionLoss`-style merge-avoidance term** — pure loss, no new label, directly targets
   usrm2's known unsolved failure mode (wrap merging). Cheapest win.
2. **Signed-distance head reusing the section-9 ramp export** — we already produce a continuous SDT-style
   ramp for compression reasons; training a distance head from it (with an Eikonal consistency loss) is close
   to free and gives smoother gradients than the current hard-band BCE+dice.
3. **Topology loss (spherical-Betti, or ECT-mass as a cheaper stand-in)** — highest expected value against
   merge/split errors specifically, highest engineering cost (external `betti_matching` dependency, or the
   ECT hyperparameter surface); worth a pilot at one mid rung before committing.
4. **Normals-as-output for recto/verso orientation** — attractive because it would formalize the existing ad
   hoc `--radial-sign -1` verso trick, but only pays off once both recto and verso heads have wide-enough
   coverage (verso currently rungs 2-3 only per §23) to make the orientation signal meaningful.
5. Everything else (in-plane tangent, structure tensor, mesh-rasterized surface frame) — defer; either the
   network can already learn it locally, or it needs a mesh/segment input usrm2 doesn't yet have as a first-
   class citizen.

## 5. Pitfalls

- **Axis order.** `structure_tensor.py`'s 3D layout is `(Jzz,Jzy,Jzx,Jyy,Jyx,Jxx)` — Z-first, upper-triangle,
  not the more common `(Jxx,Jxy,Jxz,...)`. `aux_inplane_direction.py` explicitly reassembles this into a
  `(3,3,...)` matrix with `x,y,z` ordering before `eigh`; any port must replicate that reassembly exactly or
  the eigenvector "tangent" comes out permuted. usrm2's own channel order convention (`[CT, ctx..., CASCADE,
  scale, radial(3)]`, "image channels first, radial last") is a different axis-order concern (channel dim vs.
  spatial ZYX) but the same class of bug: get one of these orderings wrong and the loss trains against noise
  silently (no shape error, just wrong gradients).
- **Normal sign convention.** `compute_surface_normals_from_sdt` returns the gradient of *(outside_dist -
  inside_dist)*, i.e. normals point **outward** (away from foreground) by construction; there's no separate
  parameter to flip it. `aux_nearest_component.py`'s direction is `-∇sdf`, i.e. it points **inward**. Two
  villa functions in the same package disagree on sign for structurally the same SDT. Any usrm2 normals head
  needs one explicit, documented convention (and ideally a sign-invariant loss like `SignInvariantCosineLoss`
  during early training, switching to a signed loss once the recto/verso split is what fixes orientation).
- **Thresholds baked into aux generators.** `binary_mask = source[0] > 0` assumes the *first channel* of
  whatever `source_target` is, and a hard `>0` threshold — villa's aux targets are computed from **binary**
  masks, not from usrm2's fractional (0..255 -> 0..1 coverage) target pyramid values. Reusing this code
  against usrm2 targets means picking a threshold (e.g. 0.5) that discards the fractional/coarse-rung
  information usrm2 deliberately preserves (unified_design.md §3's whole point: "the dynamic range the user
  asked for"). An EDT of a thresholded coarse-rung target would be a much worse signal than at native rung.
- **Coverage / masking.** All of these aux losses depend on correctly zeroing background contribution
  (`ignore_index=-100` sentinel, or masked-mean denominators clamped with `.clamp_min(eps)`); usrm2 already
  has an equivalent, more general per-channel-and-per-voxel weight tensor (unified_design.md §3, §23) that
  should carry any of these ports' masking instead of inventing a parallel sentinel-value convention.
- **Cost.** These targets are computed CPU-side, per-sample, uncached (scipy EDT, Scharr, structure-tensor
  Gaussian blur all repeated every epoch). At usrm2's throughput (10.8 Mvox/s on an A100 for CT+9ctx+cascade
  alone, unified_design.md §2/§22) this would need to become either a **precomputed target pyramid** (the
  usrm2 way — build it once into a `<name>.zarr/<rung>` store like every other target) or a GPU
  implementation, not a per-sample CPU call.
- **`alphashape`/`shapely` and `betti_matching` are optional/soft dependencies** in villa's code (guarded
  imports with `HAS_ALPHASHAPE` / lazy `_load_betti_module`); porting Betti-matching topology loss brings in
  an external package usrm2 doesn't currently depend on.
- **No dense spiral/winding target exists upstream** — don't expect to find or port one; the closest concept
  (wrap-overlap band detection, spiral fitting) operates on meshes in UV/2D space, not on our voxel pyramids.
