# villa/lasagna as a source of extra channels/losses for the unified model (2026-09-21)

Scope: read-only survey of `/home/forrest/villa/lasagna` for the unified 12-rung model
(`docs/unified_design.md`). Goal: find dense per-voxel representations lasagna already computes or
derives from labels that could become extra input channels, extra output channels, or losses for our
single recto+verso UNet.

## 1. What "lasagna" is

Lasagna is not one representation, it is a stack of three, all keyed off the same sheet-normal idea:

| Layer | What it is | File(s) |
|---|---|---|
| **cos/grad_mag/dir 2D UNet output** | A per-axis-slice 2D UNet (z, y, x slices independently) predicts 4 channels per pixel: `cos` (periodic layer signal), `grad_mag` (sheet density), `dir0/dir1` (180deg-symmetric double-angle in-plane direction). `preprocess_cos_omezarr.py` | `preprocess_cos_omezarr.py`, `train_unet.py` |
| **3-axis fusion → 3D normal + fused cos/grad_mag** | The three per-axis dir encodings, each a linear constraint on the 3D surface normal, are combined via cross products of constraint rows into one estimated 3D unit normal per voxel, plus reliability weights `w_axis = sqrt(1-n_axis^2)`. cos and grad_mag are then fused across axes using those weights. | `preprocess_cos_omezarr.py:integrate`, `lasagna_3d.md` "Normal estimation algorithm" |
| **cosine-grid / mesh fit (the actual "lasagna" model)** | A learned quadmesh `(D, Hm, Wm, 3)` (D = winding count, H = column/height, W = winding-phase) fit by Adam against the cos/grad_mag/dir fields via losses (`dir`, `step`, `gradmag`, `data`, `data_plain`, `pred_dt`). This is the geometric surface-tracing model, analogous to what our UNet is trying to shortcut. | `model.py`, `fit.py`, `optimizer.py`, `lasagna_3d.md` |

Two more label-only pipelines matter because they need **no UNet, just published binary masks** (same
input our bootstrap uses):

- `labels_to_lasagna_normals.py`: binary label -> `vc_gen_normalgrids`/`vc_ngrids --fit-normals` (mesh
  normal estimation from the label surface itself) -> flat lasagna zarr `(C,Z,Y,X)` uint8, channels
  `[cos, grad_mag, nx, ny, pred_dt]`. `cos` here is just the binary mask (255 on surface), `grad_mag`
  is a constant "density" value inside the mask, `nx/ny` are the label-surface normal's x/y components
  hemisphere-encoded (nz implied >=0, flip nx/ny where the raw nz<0), `pred_dt` is a clamped Euclidean
  distance-to-surface in voxels.
- `labels_to_winding_volume.py`: binary label -> connected components -> greedy chain ordering across
  components -> skeletonize -> per-voxel **winding number** (float, continuous chain order) via
  distance-transform-based interpolation between consecutive sheet skeletons.
- `tifxyz_labels.py` / `fitted_to_unet_labels.py`: given a fitted mesh (`fitted.zarr`: `normal` (3,Z,Y,X),
  `winding` (Z,Y,X), `validity`, `density`), derive the full 8-channel UNet training-label set: `cos =
  0.5+0.5*cos(2*pi*winding)`, `grad_mag = density`, and three dir pairs from the normal projected onto
  each axis-plane (`dir_z` from `(nx,ny)`, `dir_y` from `(nx,nz)`, `dir_x` from `(ny,nz)`).

### Channels, ranges, resolution

| Channel | Encoding (stored) | Decoded range | Notes |
|---|---|---|---|
| `cos` | uint8 | [0,1] float | periodic in winding position, 1 period = 1 winding |
| `grad_mag` | uint8, scale 1000 | [0, ~a few] voxel^-1 | `\|grad(frac_winding_pos)\|`; "sheet density" |
| `dir0`, `dir1` | uint8 | [0,1] each | double-angle (180deg symmetric) in-plane normal direction, decodable to an angle via `cos2t=2*dir0-1`, `sin2t = cos2t - sqrt2*(2*dir1-1)`, `theta = atan2(sin2t,cos2t)/2` |
| `nx`, `ny` (label-normal path) | uint8, `(v-128)/127` | [-1,1] | hemisphere-encoded (nz forced >=0 by flipping nx/ny) |
| `valid` | uint8 | {0,1} | inference/label coverage mask |
| `pred_dt` | uint8, raw voxels clamped 255 | [0,255] voxels | Euclidean distance to nearest sheet/skeleton |
| `winding` (winding volume) | float32 zarr | unbounded float | continuous chain order across components, monotonically increasing along the outward wrap direction |

Resolution: everything is voxel-dense at a chosen `scaledown` (a per-axis integer, e.g. 4x), uniform in
Z/Y/X for the 3D pipeline (`lasagna_3d.md` "uniform scaledown"). It rides on whatever CT volume it was
built from — same grid, just coarser by `scaledown`. There is no fixed absolute resolution; it inherits
the source scan's voxel size, same as our rungs.

### How it's derived (label path vs CT path)

- **From CT only** (no labels): 2D UNet per axis -> fuse -> 3D normal + cos/grad_mag/dir. This is what
  `preprocess_cos_omezarr.py` and the 3D UNet in `train_unet_3d.py` do; it is itself a *learned* model,
  trained against fitted-mesh-derived labels (circular: the UNet is trained on lasagna-fit output, then
  used to seed new lasagna fits).
- **From published binary masks only** (no CT, no UNet): `labels_to_lasagna_normals.py` (mesh-fit normals
  via `vc_gen_normalgrids`) and `labels_to_winding_volume.py` (CC + skeleton + interpolated winding
  number) both start from exactly the kind of binary surface mask our bootstrap already has. This is the
  path most directly reusable by us — it needs no lasagna UNet, no CT.
- **From a fitted mesh** (`fitted.zarr`, the tifxyz quadmesh already fit to a scroll region):
  `fitted_to_unet_labels.py` / `tifxyz_labels.py` derive cos/grad_mag/dir/winding/validity densely and
  on GPU. This is the highest-quality source (an actual traced surface, not just a raw binary blob) but
  needs meshes, which for us means "wherever `.tifxyz` segments already exist" (our teachers/segments,
  see `usrm2-teachers-and-data.md`).

## 2. What the lasagna UNet is trained to predict

`train_unet_3d.py` docstring: "Targets (cos, grad_mag, 3x2 direction encoding = 8 channels) are computed
on-the-fly on GPU from fitted.zarr files." It is a 3D UNet (via the shared `vesuvius` model builder,
`NetworkFromConfig`), input = raw CT patch, output = 8 channels (cos, grad_mag, dir_z(2), dir_y(2),
dir_x(2)), losses `--w-cos`, `--w-mag`, `--w-dir` (simple weighted regression/MSE-style per channel,
masked by `valid`). This is architecturally close to our setup: single 3D UNet, CT in, several dense
probability/vector fields out, masked losses. It differs in that ours has one binary surface-probability
output per side (recto/verso) rather than a periodic phase + density + direction stack — lasagna encodes
*where in the winding cycle* a voxel sits and *which way the sheet is tilted*, not just *is this voxel
on a sheet*.

The actual `fit.py` optimizer (the "lasagna model" proper) doesn't train a network at all — it does
per-scroll gradient-descent mesh fitting against the UNet's output fields (or against fitted.zarr/labels
directly), with losses `dir`, `step`, `gradmag`, `data`, `data_plain`, `pred_dt` (see lasagna_3d.md
section "Losses"). This is a geometric solver, not something to imitate directly, but its loss
definitions are exactly the formulas we'd need if we wanted to derive dense targets from label surfaces
for our own network (see below — `fitted_to_unet_labels.py` already does this extraction).

## 3. Data products that exist or could be exported

From label surfaces (published masks or our own `.tifxyz` segments), cheaply, without any lasagna UNet:

1. **Per-voxel unit normal** (`nx, ny, nz`, hemisphere-encoded or full 3-component) — from
   `labels_to_lasagna_normals.py` (mesh normal via `vc_gen_normalgrids`) or from `tifxyz_labels.py`'s
   normal field derived from a fitted quadmesh.
2. **Winding number / fractional winding position** (continuous float, monotonic along the outward
   wrap direction) — from `labels_to_winding_volume.py` (CC + skeleton + DT-interpolation) or from a
   fitted mesh's `D` (winding) index directly.
3. **`cos` of winding phase** — trivially `0.5+0.5*cos(2*pi*winding)`, a smooth periodic re-encoding of
   (2) that is bounded and differentiable everywhere, unlike a raw winding integer.
4. **`grad_mag` / sheet density** — `|grad(winding)|`, i.e. local sheet spacing (inverse of gap between
   consecutive wraps); high where sheets are tightly packed (damage, crushed regions), low where they're
   spread out.
5. **`pred_dt` / signed or unsigned distance to nearest sheet** — cheap Euclidean DT from the binary
   mask, already how our own "narrow-band" style targets could be built without new inference.
6. **Cylinder/shell violation-depth SDF** (`cyl_sdf_volume.py`, `build_previous_shell_violation_depth_volume`) —
   a coarse (default grid step 64 vox) signed-distance-like field to a *previous* completed shell/mesh,
   via libigl `signed_distance`, encoded uint8 with a sqrt compression (`encode_violation_depth`). This is
   used as a barrier/init field during cylinder-initialized fitting, not as a training target, but it is
   exactly a "distance to already-known surface" channel.
7. **Per-axis double-angle direction encoding** (`dir0,dir1` per z/y/x slice plane) — derivable from any
   normal field (label-mesh normal, or fused-UNet normal) by the same `_encode_dir()` formula
   (`fitted_to_unet_labels.py:_encode_dir`).

None of these require CT at label-export time (only the binary/mesh label); at *inference* time on new,
unlabeled CT, they'd need a network to predict them (like lasagna's own 2D/3D UNets do), or — for the
normal/winding fields specifically — our own model's cascade/self-consistency machinery.

## 4. Fit to our model: input channels, output channels, losses

### (a) Candidate extra INPUT channels

| Channel | Computable at inference from CT alone? | Verdict |
|---|---|---|
| cos/grad_mag/dir (lasagna 2D/3D UNet output) | Only by running lasagna's own UNet as a preprocessor (another network in the input pipeline) | **Skip as input.** Section 21 of `unified_design.md` already rules out exactly this class of feature ("local filters... fail that test since the first 3x3x3 layers learn them") for anything derivable from a small receptive field; cos/grad_mag/dir are *not* small-receptive-field local filters (they need cross-slice context and a trained UNet), so the rule doesn't strictly apply, but feeding another network's output as an input channel duplicates the cascade-channel idea we already built (section 22) — better to let our own model learn this internally or use our own predicted-probability cascade, not import a second model's feature stack. |
| our own predicted surface probability, at a coarser rung | **yes, already input** | This *is* the cascade channel (section 22). No new work needed. |
| distance-to-nearest-known-surface (from segments/teacher predictions, like `cyl_sdf_volume`'s violation-depth field) | Yes, if we already have a nearby segment/prior surface (e.g. an approved slab-viewer anchor or a previous rung's confident detection) | **Plausible input for the refinement/interactive mode** mentioned in `unified_design.md` section 21 item 5 ("existing segmentation meshes as a sparse known-surface channel: later"). Lasagna's `cyl_sdf_volume.py` is a ready-made, GPU/libigl-based implementation of exactly this (mesh -> coarse uint8 SDF volume), reusable almost as-is for that later work. |
| normalized radius from umbilicus | Already planned (section 21 item 2), unrelated to lasagna | n/a |

Net: lasagna offers **no compelling new INPUT channel** beyond what's already planned, because its
per-voxel fields (cos/dir/grad_mag) are themselves model outputs requiring inference, and feeding one
network's output into another as an input is the cascade pattern we already have for our own rungs. The
one piece worth banking for later is the SDF-to-known-surface code (`cyl_sdf_volume.py`) for the
"existing segmentation meshes as sparse known-surface channel" item.

### (b) Candidate extra OUTPUT channels (dense per-voxel targets, derivable from existing labels)

These are the strongest fit, because lasagna's label-only pipelines (`labels_to_lasagna_normals.py`,
`labels_to_winding_volume.py`, `fitted_to_unet_labels.py`) already show how to turn a binary published
mask (or one of our own `.tifxyz` segments) into these dense targets with no CT and no lasagna UNet:

| Output channel | Label source | Rungs it makes sense at | What it buys |
|---|---|---|---|
| **Surface normal (nx,ny,nz)**, e.g. as a 2- or 3-channel head (hemisphere-encoded like lasagna does) | `vc_gen_normalgrids` on the binary mask, or normal-from-mesh if we have `.tifxyz` | fine rungs (0-3ish) where a normal is well-defined per voxel; degrades/undefined at coarse rungs where "surface" is many sheets thick | Orientation for downstream flattening/meshing without a separate normal-estimation pass; a differentiable, dense proxy for "which way is out" that a segmentation-only model doesn't give you; useful as an auxiliary loss even if not exported (regularizes the recto/verso boundary to be sharp and consistently oriented) |
| **Fractional winding position (cos-encoded, `0.5+0.5*cos(2*pi*winding)`)** | `labels_to_winding_volume.py` run on the published mask; needs CC + skeleton + chain ordering, which is exactly the kind of "merge two components that are really the same sheet" signal we care about | fine-to-mid rungs; winding number itself is scroll-global so only meaningful where CC chaining succeeds cleanly | **Merge avoidance for free**: two adjacent sheet fragments that a plain binary mask would connect through a crease get *different* winding numbers if the label pipeline chained them correctly, so a winding/cos output head teaches the network "these are different wraps" even when the binary mask alone is ambiguous. This is the single most valuable idea to borrow — it directly attacks the merge-vs-split failure mode a binary recto/verso head is blind to. |
| **Sheet density / grad_mag (\|grad(winding)\|)** | derived from the winding field above | fine-to-mid rungs | A continuity/spacing signal — tells the network how tightly wound the local region is (crushed/damaged vs open), which a binary mask throws away; could regularize against sheets predicted to be locally self-crossing (grad_mag blowing up or going negative) |
| **Distance-to-surface (pred_dt / narrow-band SDF)** | trivial EDT of the binary mask (`labels_to_lasagna_normals.py:_compute_pred_dt`, or `cyl_sdf_volume.py` for a mesh-based signed version) | any rung | A softer, distance-weighted target than binary occupancy — classic narrow-band segmentation trick, gives non-zero gradient far from the boundary, and a natural coarse-rung analogue (distance in rung-k voxels to the nearest label voxel, well-defined even when "surface" is many voxels thick at coarse rungs) |
| **Double-angle direction encoding per axis-plane** (`dir0/dir1` for z/y/x) | derivable from the normal field above via `_encode_dir()` | fine rungs only | Mainly useful as an auxiliary 2D-projected orientation signal per slicing plane; lower priority than the full 3D normal head since it's strictly less information and exists mainly because lasagna's *input* UNet is 2D-slice-based (a constraint that doesn't apply to us — we're already 3D) |

Practical note: all of these need either (i) the `vc_gen_normalgrids`/`vc_ngrids` C++ toolchain (Volume
Cartographer) available wherever we preprocess targets, or (ii) our own `.tifxyz` segments (if we have
per-vertex normals or a quadmesh already) to derive normal/winding fields without that toolchain.
`labels_to_winding_volume.py` is pure Python + scipy/cc3d, no VC dependency, and is the easiest one to
port/run standalone.

### (c) Candidate LOSSES / constraints

| Loss | Source formula | What it buys |
|---|---|---|
| **Orientation consistency** (predicted normal vs. label-derived normal, `1 - |dot(n_pred, n_label)|`) | `lasagna_3d.md` "normal loss" | Even without exporting a normal head, a normal-consistency auxiliary loss (computed from the gradient of the predicted probability field, compared to the label-mesh normal) sharpens boundaries and discourages locally-flat/blurry predictions — cheap because the probability gradient direction is already there |
| **Winding-density / gradmag "one wrap per unit" constraint** (`opt_loss_winding_density.py`, `WINDING_DENSITY_BARRIER_MARGIN/SCALE`) | Integrate density along the connection direction; target integral = 1.0 per wrap; barrier term discourages self-intersection when the strip goes the wrong way (along-normal instead of anti-normal) | A **merge/self-intersection penalty**: directly penalizes predictions where consecutive "wraps" of the predicted surface would cross or double back — could be adapted into a loss on our own predicted probability field's local gradient magnitude/direction between rungs, though this requires meshing the prediction first, which is heavier machinery than a voxel loss |
| **Step-distance regularization** (mesh row spacing vs target `mesh_step`) | `opt_loss_step.py` | Not directly applicable — this is a mesh-parameter regularizer, not a voxel loss; only relevant if we ever add a meshing/tracing stage downstream of the probability output |
| **pred_dt clamped two-regime L1** (`opt_loss_pred_dt.py`) | distance-to-surface loss with different weight inside/outside | If we add a distance-transform output channel (4b above), this is the ready-made loss shape (asymmetric weighting near vs far from the true surface) rather than a plain MSE |

Priority ranking for what to actually pursue, given the unified model's stated concerns (merge avoidance,
continuity, orientation, "one model to do everything"):

1. **Winding/cos output channel derived from `labels_to_winding_volume.py`**, at fine rungs where our
   published-mask or teacher-region labels already exist — highest leverage on the merge-avoidance goal,
   reuses existing label sources, pure-Python pipeline.
2. **Normal output channel (or normal-consistency auxiliary loss)**, derived the same way as (1) or via
   `vc_gen_normalgrids` — buys orientation for free and a natural downstream unrolling/flattening cue,
   but needs the VC3D binary unless we use mesh-derived normals from our own `.tifxyz` segments.
3. **Distance-to-surface channel/loss** — lowest new information (largely redundant with a well-trained
   binary head, since distance can be approximated post-hoc from a probability field) but cheapest to
   add and a reasonable narrow-band regularizer.
4. Cylinder-SDF-to-known-surface as an *input* channel — deferred, matches section 21 item 5's "later"
   note; keep `cyl_sdf_volume.py` in mind for that phase.

## 5. Pitfalls

- **Axis order / coordinate convention mismatch.** Lasagna's dense fields are `(C, Z, Y, X)` (zarr) but
  mesh/tifxyz coordinates are `(X, Y, Z)` in *fullres voxel units*, and the double-angle direction
  encoding is defined **per slicing plane** (`dir0/dir1` for z-slices means gradient in the XY plane;
  for y-slices, XZ; for x-slices, YZ) — mixing these up silently rotates the decoded angle by 90 degrees.
  If we ever import a normal or direction field, audit which axis convention it was computed in before
  wiring it into our `(Z,Y,X)` tensors and our radial-vector convention.
- **Normal sign/hemisphere ambiguity.** A surface normal has no inherent sign; lasagna resolves this with
  a hemisphere encoding (`nz` forced >= 0 by flipping `nx,ny` — see `labels_to_lasagna_normals.py`
  "Apply +z hemisphere encoding"). Our recto/verso split gives us a *natural* sign convention (normal
  points from verso side to recto side), which is actually better-defined than lasagna's arbitrary +z
  hemisphere pick — if we add a normal output we should use our own recto-outward convention, not copy
  lasagna's, to avoid a discontinuity where the label-derived normal flips sign but our recto/verso
  labels don't.
- **Direction encoding is 180-degree symmetric by construction** (`dir0/dir1`) — it cannot represent a
  signed normal at all; using it as-is would throw away exactly the sign information our verso channel
  is meant to supply. Only useful as an auxiliary/orientation loss, not as the primary orientation output.
- **Winding number requires successful chain ordering.** `labels_to_winding_volume.py`'s connected
  components -> greedy chain -> skeleton -> DT-interpolation pipeline can fail or produce garbage where
  components are noisy, poorly separated, or the mask has gaps — a wrong chain order would poison a
  winding-based loss/target with confidently wrong "this is wrap 5 not wrap 6" information, worse than no
  signal at all. Needs a validity/confidence mask (lasagna's own `valid` channel convention) and probably
  a coverage check before trusting it as ground truth, especially on damaged/crushed regions where our
  own project's history (see `usrm2-findings.md`) already flags eval-ceiling issues.
- **Thresholding and uint8 quantization.** Every stored lasagna channel is uint8 with a fixed encode
  scale (`grad_mag_encode_scale=1000`, `pred_dt` clamped to 255 voxels, direction/cos scaled by 255) —
  reusing any of this means matching decode scales exactly, and clamped fields (density, distance) lose
  information in high-density/far-distance tails; at coarse rungs where 255 voxels of distance is a small
  fraction of the field of view, `pred_dt` becomes nearly useless without rescaling by rung.
- **Coverage / resolution mismatch with our ladder.** Lasagna's fields are computed at *one* chosen
  `scaledown` per run, not as a multi-rung pyramid with our exact power-of-2 log2-nearest-snap semantics;
  normal and winding fields in particular become ill-defined at coarse rungs (a "surface normal" is
  meaningless once a 256^3 cube at rung 6+ contains dozens of wraps) — so any imported channel should be
  scoped to the fine rungs only (roughly rungs 0-3, matching where lasagna itself operates: full-res to a
  few voxels/pixel), with explicit ignore-weight at coarser rungs, exactly the same pattern the verso
  channel already uses ("weight 0 everywhere" above rung 3, section 23).
- **VC3D/libigl toolchain dependency.** `vc_gen_normalgrids`/`vc_ngrids` (used by
  `labels_to_lasagna_normals.py`) and `cyl_sdf_volume.py`'s libigl extension both require the Volume
  Cartographer C++ build and Eigen/libigl headers — not currently a usrm2 dependency. `labels_to_winding_volume.py`
  is the only one of the three "from labels" pipelines that's pure Python (scipy/cc3d), so it's the
  cheapest to actually deploy without adding a new native build dependency to our pipeline.
- **Circularity risk.** The lasagna 2D/3D UNets are trained on labels *derived from lasagna's own mesh
  fits*, which are themselves optimized against those same UNets' outputs in later iterations
  (`docs/status.md` architecture diagram: preprocess -> fit -> `fitted_to_unet_labels.py` -> retrain
  UNet). If we ever import lasagna-*predicted* (not label-derived) normal/winding channels as targets,
  we'd be training on another model's (possibly biased) predictions rather than ground truth — prefer the
  label-only derivation path (`labels_to_lasagna_normals.py`, `labels_to_winding_volume.py`) over anything
  that runs lasagna's UNet.

## Key file/function reference

- `lasagna_3d.md` — the 3D model spec; "Data modalities", "Losses (relevant subset)", "Preprocessing" sections are the primary source for this report.
- `labels_to_lasagna_normals.py` — binary label -> normals (`vc_gen_normalgrids`/`vc_ngrids --fit-normals`) -> flat `[cos, grad_mag, nx, ny, pred_dt]` zarr; hemisphere encoding at `_write_lasagna_zarr`.
- `labels_to_winding_volume.py` — binary label -> CC -> chain -> skeleton -> DT-interpolated winding number.
- `fitted_to_unet_labels.py` — `fitted.zarr` (`normal`, `winding`, `validity`, `density`) -> 8-channel UNet training labels; `_encode_dir()` is the double-angle formula.
- `tifxyz_labels.py` — GPU/CuPy version of the same derivation, `edt_torch()` for CUDA EDT via DLPack.
- `cyl_sdf_volume.py` — `build_previous_shell_violation_depth_volume()`, libigl-based signed distance to a completed shell mesh; `encode_violation_depth()`/`decode_violation_depth()` sqrt-compressed uint8 codec.
- `preprocess_cos_omezarr.py` — the 2D-UNet-per-axis preprocessing + `integrate` fusion (3D normal estimate, cos/grad_mag fusion) that produces lasagna's CT-derived channels.
- `opt_loss_winding_density.py`, `opt_loss_pred_dt.py`, `opt_loss_dir.py`, `opt_loss_step.py` — the loss shapes, useful as formulas even if we never run lasagna's optimizer.
- `docs/model.md`, `docs/status.md`, `docs/flatten.md` — architecture/status docs; `docs/status.md` has the clearest end-to-end pipeline diagram and "Not yet implemented" list (gradmag loss not yet ported to 3D, mesh growing, mask scheduling).
- `train_unet_3d.py` — the closest existing analogue to our training script: CT-in, 8-channel dense field out, masked per-channel weighted loss.
