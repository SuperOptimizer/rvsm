# Literature: dense-prediction to clean 2-manifold surface, outside Vesuvius Challenge (2026-09-21)

Scope: methods for turning a per-voxel probability/SDF volume into a single, thin, 2-manifold sheet, drawn
from medical imaging, computer vision and computer graphics, 2018-2026, no scroll-data assumptions. Framed
against usrm2's own failure modes (`docs/unified_design.md` §20): **merge_frac 0.44** (recto/verso or
adjacent-wrap sheets fusing) and **continuity 0.656** (gaps in the band), at gigavoxel scale (Paris 4 alone
is ~1.3e14 voxels — nothing here runs in one machine's RAM over the whole scroll, so "applicability" always
means chunked/region-wise, matching the existing 1024^3 region-store architecture of `usrm2/stream.py` and
`cloud/teacher_regions.py`).

---

## 1. Medial-surface / thin-structure extraction

Classical topological thinning (Lee-Kashyap-Chu 1994, re-derived and corrected since — Németh & Palágyi
~2015) peels simple points from a binary volume until only a 2D sheet/1D curve survives, preserving genus
and component count by construction; near-linear cost, shipped in `scikit-image.skeletonize_3d`, ITK, CGAL.
At gigavoxel scale only tractable per-1024^3-chunk with halo exchange, since it is a global fixed point, not
a local filter — the same region-boundary tension graph-cut methods have (§6). Structural MAT (arXiv
2605.02302, 2026) simplifies a medial axis with explicit correspondence back to the surface so it never
drifts, but it consumes a mesh, not a volume — a step after MC/DC, not a replacement.

**Mapping / pitfall**: useful only as a *diagnostic* — thin the binary prediction and use where the medial
sheet forks as a training-free localizer for L1's normal-gated repulsion term (`synthesis_future_inputs_
outputs.md` §4 L1). Not a mesh-production path: it discards thickness the SDF head (O2/O3) already carries,
and outputs a 1-voxel raster skeleton, not a manifold. Not robust to noise at a probability threshold — one
spurious voxel changes the whole skeleton's topology; never threshold-then-thin without an opening pass first.

## 2. Marching cubes and dual contouring on learned fields

Marching cubes (MC) is still the default for a probability volume (topologically-correct MC33 variants since
Nielson & Hamann 1991, e.g. VTK's `vtkDiscreteFlyingEdges3D`). At gigavoxel scale the practical algorithm is
**Flying Edges** (Schroeder et al. 2015, still the fastest exact CPU-parallel MC) or out-of-core/octree MC
(block-compressed MC for streamed volumes, classical out-of-core partitioning reaching ~2.6B cubes/pass) —
literally "MC per 1024^3 shard, stitch at the boundary," free given our shard-aligned region tiling (§17/§18
of the design doc). **Dual contouring** (Ju et al. 2002) places one vertex per active cell from a Hermite
(position+normal) sample, reproducing sharp/thin features MC smooths away. **Neural Dual Contouring** (Chen
& Tagliasacchi, arXiv 2202.01999, SIGGRAPH 2022) trains a small net to predict per-cell vertex/edge-sign
directly from signed *or unsigned* distance fields, voxels or points, and explicitly supports **open
surfaces** — built for a single-sided sheet, not just closed solids; beats Poisson and "neural marching
cubes" on feature preservation and triangle count, code released. **DCUDF2** (arXiv 2408.17284, 2024)
extracts a zero-level-set directly from an *unsigned* distance field, needing no global sign — relevant since
the sign of our SDF head is known to be unreliable near the axis (tsm's "recto_is_in" drops to 0.47 near the
core, `synthesis…` §1c O3). **MIND** (arXiv 2506.02938, 2025) goes further and extracts a **non-manifold**
"material interface" mesh directly from a UDF, built for two sheets touching or a genuine boundary — the
recto/verso seam at an outer wrap or torn edge is exactly that case, the only method here treating "two
surfaces touch" as a first-class output rather than a defect.

**Mapping**: Flying-Edges-per-shard is the mechanical replacement for the tracer's post-hoc `make_surf_sdt.py`
EDT+threshold (`vc3d_tracer_inputs.md` §1.3) once O3 exports; no new model work, stitches at the existing
shard grid. NDC is worth a pilot on the recto/verso pair specifically — consuming O3/O4 per shard to emit an
open, one-sided mesh, replacing the discrete threshold-and-triangulate step. MIND is the tool to reach for at
a merge (§6): where recto/verso are both high-probability and near-antiparallel in normal, a non-manifold
interface mesh is the geometrically correct object, not an error to smooth away.

**Pitfall**: MC on a soft probability produces genus errors and self-intersections wherever the 0.5-crossing
is ambiguous — exactly our measured merge/gap failures; Hermite methods need a normal at every active cell,
inheriting the SDF/normal head's noise floor near the axis and at thin/torn material; none of this fixes
topology on its own (§4/§5).

## 3. Poisson and screened-Poisson reconstruction from oriented points

Screened Poisson (Kazhdan & Hoppe, TOG 2013; still the field standard — `PoissonRecon`/CGAL/Open3D/MeshLab)
solves a screened Poisson equation over an adaptive octree, pulling the surface toward input points as
interpolation constraints rather than just their indicator gradient, fixing classical Poisson's over-
smoothing. Documented thin-feature requirement: correct reconstruction needs local point spacing at most
~1/10 of the local feature size (distance to the medial axis, i.e. the sheet-to-sheet gap) — a real
constraint here since the m7 band is often thicker than the true sheet, so points sampled from a soft band
inherit double thickness. Its known failure with thin, closely-packed sheets is stated plainly in current
surveys: **it tends to merge nearby thin layers and bridge holes, because it optimizes for a smooth,
watertight result** — our merge_frac problem, named, from the reconstruction side.

**Mapping / pitfall**: Poisson wants oriented *points*, not a probability volume — pipeline would be O3/O4 ->
extract points at confident, low-curvature voxels with predicted normals -> Poisson per 1024^3 region with
octree depth set from the local winding pitch (10-16 voxels, `dr_per_winding`) -> stitch. Given its documented
thin-layer-merging failure this is **worse** than direct MC/DC on our own SDF wherever wraps are close (most
of a scroll), and only worth trying where input is genuinely sparse/noisy (e.g. manually corrected patches,
`vc3d_tracer_inputs.md` §1.4) — not a general replacement.

## 4. Topology repair: hole filling, handle removal, self-intersection removal

Hole filling still runs Liepa's constrained-Delaunay triangulation + Laplacian fairing (2003), now packaged
in detect-then-repair pipelines (topology-graph repair, *J. Comp. Design & Eng.* 2021) that classify
holes/non-manifold edges/self-intersections before dispatching a fixer. "Instant Self-Intersection Repair"
(ACM TOG 2025) locally remeshes only the intersecting region from voxelized occupancy, near-interactive.
Handle/genus removal: the strongest transferable evidence is cortical-surface reconstruction (closest problem
shape to ours — a thin, folded, single 2-manifold sheet, genus-0 required): DeepCSR (WACV 2021) and
successors report topology-correction on the reconstructed implicit volume costing **>30 minutes per
correction**, which is why the field's newer line (CortexODE, CorticalFlow, 2024/2025 diffeomorphic-
deformation methods) moved to **explicit deformation of a genus-0 template**, which never changes topology
and needs no correction pass — orders of magnitude faster.

**Mapping**: the cortical-surface lesson is the most transferable finding here — **constrain the
representation instead of repairing a thresholded mesh post-hoc**, because a global topology pass on a
scroll-scale mesh (a sheet unrolling to on the order of a square meter, tens of millions of triangles) would
be the >30-min-per-defect cost times thousands of defects. If the tracer's spiral fit deforms a genus-
appropriate template (the spiral itself) to match SDF/normal fields rather than thresholding-then-repairing,
most of this section is moot for the final mesh; Liepa-style local hole-fill remains fine for small per-
region defects in any intermediate MC/DC mesh used as a deformation target.

**Pitfall**: repair algorithms cannot distinguish a true tear from a true fusion — both look like a defect —
so blind repair silently manufactures wrong geometry with no signal it did so. A confidence channel (O7)
flagging "repaired" regions for human/tracer review is worth more than a smarter repair algorithm.

## 5. Gap closing along thin structures

**Tensor voting** (Medioni, Tang & Lee; Mordohai & Medioni's survey; patented gap-filling of 3D microvascular
networks and contour completion for surface reconstruction) encodes orientation+confidence as a 2nd-order
tensor per voxel and propagates it by convolution with orientation-aware voting fields, built specifically to
bridge curve/surface gaps from sparse, locally-oriented evidence. Local, training-free, a few convolution
passes over an accumulator volume — cheap per-1024^3-region.

**Minimal-path/geodesic completion**: finds the lowest-cost path/ribbon between two known fragments through a
cost field. "Surface Reconstruction via Geodesic Interpolation" (2008-era reference) and "Neural Shortest
Path for Surface Reconstruction" (arXiv 2502.06047, 2025, learned cost field) solve exactly "two disconnected
fragments, find the connecting path" — near-linear (Dijkstra/fast marching) but needs endpoints already
identified, so it is a *targeted*, not volume-wide, gap-closer.

**Learned inpainting**: 3D shape-completion diffusion (DiffComplete NeurIPS 2023, Diffusion-SDF 2022, SC-Diff
2024) generates a *plausible* complete shape from a partial one, trained on paired data. Heaviest option (no
paired "occluded region -> true region" data exists here) and least controllable — fills with something
plausible, not necessarily true, risky for a document later read for content.

**Mapping / pitfall**: tensor voting is the best fit for our continuity metric (0.656) — operates directly on
O3/O4, needs no new labels, cheap as a post-process between MC/DC and topology repair, or as a differentiable
relaxation folded into the L1/L2 loss family (a convolutional propagation term is a soft local consistency
loss, close to what L1/L2 already estimate from central differences). Geodesic completion fits better as an
*interactive* tracer-UI tool (human clicks two fragments, system proposes the ribbon) than an automatic pass.
Learned inpainting: not recommended now — no paired data, hallucination risk, and the cascade channel (§22)
already gives the network the cheaper, safer "look at a coarser neighborhood" signal during training. Any
completion method should carry a provenance flag distinguishing "observed" from "completed" geometry all the
way to the final mesh — a confidently-wrong fill is worse than a visible gap.

## 6. Merge splitting: cutting bridges via normals or min-cut

The Boykov-Jolly volumetric graph-cut framework (foundational, still the reference for interactive volume
segmentation) treats voxels as graph nodes and finds the globally minimal cut separating two labels given
seeds — directly applicable to "two touching sheets are one connected component; the min-cut weighted so
cutting *across* a normal discontinuity is cheap and cutting *along* a sheet is expensive is the split."
Min-cut point-cloud/surface segmentation (Golovinskiy & Funkhouser; ray-discretized surface graph cuts) are
the mesh-side analogues. Max-flow/min-cut is polynomial but not cheap at gigavoxel scale — practical only
per-1024^3-region with merge candidates pre-localized (§1's medial-axis forks, or a normal-antiparallel scan).

The villa `NormalGatedRepulsionLoss` already scoped into the Phase-A roadmap (`synthesis…` §1c L1) is
functionally a soft, differentiable stand-in for this min-cut: penalizes close, near-parallel-normal
high-probability pairs (same-sheet fusion candidates) while exempting near-antiparallel pairs (the true
recto/verso of one physical sheet). "Diffusion-Driven Inter-Outer Surface Separation for Point Clouds with
Open Boundaries" (arXiv 2602.00739, 2026) validates the same idea outside training: it splits a double-
layered point cloud (from TSDF-truncation fusion, a different mechanism than ours) into two single-sided
surfaces via a diffusion classifier keyed on local normal/depth cues — clean separation on ~20k+20k points in
~10s, the closest published analogue to "split fused recto/verso" found here; its classify-by-local-normal
principle transfers even though the fusion mechanism differs.

**Mapping / pitfall**: keep L1 as the training-time answer; add a *post-hoc* min-cut only at merge candidates
flagged by §1's fork test or the `overlap` metric, using the predicted normal field as edge weight — targeted
inference-time repair, not a whole-volume solve. Any normal-based cut is only as good as the normal field
near the cut, which is precisely where normals are least reliable — a min-cut can produce a topologically
valid but geometrically wrong split (swapped recto/verso across the cut); sanity-check every cut against the
radial-vector sign convention already used to orient normals.

## 7. Mesh-based smoothing that preserves creases

The bilateral-mesh-denoising lineage (feature-preserving via normal-weighted anisotropic neighborhoods,
2000s) is still the baseline. Segmentation-Driven Feature-Preserving Mesh Denoising (arXiv 2008.01358, 2020)
first segments the mesh into piecewise-smooth patches by normal clustering, smooths within patches, sharpens
at boundaries — directly analogous to "smooth within a sheet, keep the crease at a fold." Homogeneous-MLS
anisotropic filtering (arXiv 1912.10194, 2019) is a cheaper, local-stencil alternative. Both operate
per-vertex/face on an already-extracted mesh.

**Mapping / pitfall**: a low-risk post-process after MC/DC (§2) and before topology repair (§4) — segment-
then-smooth suits a papyrus sheet's local flatness plus occasional sharp folds and needs no scroll-specific
tuning to try first. Any Laplacian-family smoothing shrinks a thin sheet toward its own medial surface if
run too long or without a feature weight — always run normal-preserving and verify sheet-to-sheet spacing
does not shrink.

## 8. Differentiable surface extraction: a training-time loss

**DMTet** (Shen et al., NeurIPS 2021, arXiv 2111.04276): a deformable tetrahedral grid encoding an SDF at its
vertices with differentiable Marching Tetrahedra, so a downstream loss backprops through the extracted mesh
into the SDF field — the core of NVIDIA `kaolin`, widely reused in text-to-3D. **FlexiCubes** (Shen et al.,
TOG/SIGGRAPH 2023, arXiv 2308.05371) generalizes this to a dual-MC extraction with learned per-cell
parameters (vertex offsets, quad-split weights, grid deformation) so mesh connectivity itself is
optimizable, beating DMTet and standard MC on reconstruction accuracy and mesh quality; TetWeave (arXiv
2505.04590, 2025) replaces the fixed grid with on-the-fly Delaunay tetrahedralization for further gains.
Neural Dual Contouring (§2) is trainable the same way but normally used as one-shot inference rather than an
in-the-loop layer. **Eikonal regularization** (`||grad SDF|| ~= 1`, standard since implicit-surface work,
shipped in villa's `SignedDistanceLoss`, `synthesis…` §1c L5) is the cheapest item here: it extracts no mesh
at all, just keeps the SDF metric well-conditioned, benefiting any *downstream* extraction for free.

**Mapping**: DMTet/FlexiCubes are built for generative/optimization settings (differentiate a rendering or
Chamfer loss back through a mesh being actively solved for) — not usrm2's setting, which predicts one dense
field per sample from an image rather than iteratively optimizing one scroll's mesh. The transferable piece
is narrower and speculative: a FlexiCubes-style extraction inside the training loop as an auxiliary loss
(extract a mesh from the batch prediction, apply a mesh-native geometric term such as discrete curvature or
planarity, backprop into the voxel logits) — genuinely new, nothing in the villa/tsm/usrm/vc3d prior surveys
used a differentiable mesh-extraction layer. Do Eikonal first (cheap, already scoped in Phase A); only pilot
an in-the-loop mesh loss if voxel-space losses (L1/L2) leave curvature/planarity defects unresolved.

**Pitfall**: DMTet/FlexiCubes optimize one shape at a time by construction — at gigavoxel scale only sensible
per-256^3-patch during training, never a whole-scroll representation. In-the-loop extraction adds real GPU
cost (the cascade channel's `self` mode already costs +50% for one extra forward, §22) while the step is
already backward-pass bound (§14: GroupNorm+SiLU and upsampling backward dominate) — any new differentiable
path competes for exactly the budget `up2x` and `ckpt_act 0` were built to win back.

## 9. Priority ranking for usrm2/VC3D

1. **MC/Flying-Edges per shard on the O3 SDF export** — mechanical, no new labels, matches the shard grid,
   replaces the tracer's `make_surf_sdt.py` round-trip with a cleaner mesh.
2. **Eikonal loss on the SDF head (L5)** — cheapest here, already scoped, conditions every downstream
   extraction for free.
3. **Tensor voting as a normal-field gap closer**, per-region between MC/DC and topology repair — targets
   continuity (0.656) directly, no training data, shares machinery with L1/L2.
4. **Normal-gated repulsion (L1)** at training time, plus a targeted post-hoc min-cut only at flagged merge
   candidates — targets merge_frac (0.44) directly.
5. **Segmentation-driven feature-preserving smoothing** as a cheap final mesh pass.
6. **Explicit-deformation topology handling** (the cortical-surface lesson: deform a genus-correct template
   rather than repair a thresholded mesh) as a design principle for the tracer's spiral-to-mesh step, not a
   specific algorithm to import.
7. **Neural Dual Contouring / MIND**, piloted on merge/seam regions where a non-manifold or open-surface-
   aware extractor is the geometrically honest answer — worth a small experiment, not yet a commitment.
8. **Screened Poisson, learned shape-completion diffusion, DMTet/FlexiCubes in-the-loop** — lowest priority:
   Poisson's documented thin-layer-merging failure works against merge_frac; diffusion completion risks
   hallucinating readable-looking content into damage; differentiable mesh-extraction losses are unmeasured
   and compete for an already backward-pass-bound GPU budget.
