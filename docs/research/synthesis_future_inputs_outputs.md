# Synthesis: future input channels, output channels and losses for the unified model (2026-09-21)

Consolidates `docs/research/{villa_lasagna,villa_vesuvius,vc3d_tracer_inputs,tsm_ideas,usrm_legacy_ideas}.md`
against `docs/unified_design.md` (sections 1-7 = the ladder, the 15-channel sample, targets/weights, sampling,
sources, code changes; 21-24 = next input channels, cascade, verso, store codecs).

Where we are (section 19-20, 23): `u1_30m6_p4` val dice r2/r3/r4 = 0.740/0.740/0.682; surface metrics on the
val box **recall@4 0.806, continuity 0.656, merge_frac 0.44, offset<=3 0.36**. `u3` (cascade 15-channel +
verso cout=2) is training; the 5090 writes verso region stores. **Merges (0.44) and gaps (continuity 0.656)
are the two dominant failures** and every recommendation below is scored against them first.

Everything marked *(speculative)* has no measurement in tsm, usrm, villa or usrm2 behind it.

---

## 1. Consolidated candidate table

Columns: source (repo:file) / label source / rungs / build cost / which failure it attacks / evidence / pitfall.
`M` = merges, `G` = gaps+continuity, `O` = orientation, `P` = recto/verso pairing, `T` = tracer, `X` = cross-scan.

### 1a. INPUT channels

| # | channel | source | label source | rungs | cost | attacks | evidence | pitfall |
|---|---|---|---|---|---|---|---|---|
| I1 | **cascade** (rung k+1 prediction, 2x up) | usrm2 §22 | self / mask | all (0 at r11) | **DONE**; +6% mask, +50% self, +35% mix (measured, `cloud/cascade_bench.py`) | G, X | measured in usrm2 | mask mode leaks the target; noise+dropout mandatory |
| I2 | **normalised radius from umbilicus** | §21.2; `usrm2/data.py:radial` already gives direction | none (geometry) | all | ~0 (one more constant-ish plane from `radial_t`) | M, G, O | none yet; strong prior (sheet pitch, curvature and damage are all radius-dependent) | needs a per-scroll outer radius to normalise by, or it is not comparable across scans; z-varying axis already handled by `axis_at` |
| I3 | **scan metadata planes** (energy, samplePixelSize, delta_beta, unsharp, sampleDetectorDistance, window_u16_min/max, mosaic) | §21.3, `docs/example_metadata_paris4_2.4um_78keV.json` | none | all | small (constant planes; log-scale wide ranges) | X | none; but Paris 4 alone spans 74/78/137 keV and 220mm/11m propagation | zero-fill for warm start like the scale plane; a plane that is constant within a scan is free capacity for the net to memorise scan identity — only worth it once >5 scans train together |
| I4 | **axis-tangent vector (3)** | tsm `fiber.py` | none | all | ~0 | O | tsm: fiber class overlap 0.90 -> 0.03-0.12 **only after** adding axis-tangent inputs | only needed if an axis-relative target is ever added; useless on its own |
| I5 | **SDF to a known surface / mesh** (sparse known-surface channel) | lasagna `cyl_sdf_volume.py:build_previous_shell_violation_depth_volume`; villa `direction_helpers.py` for the frame version | existing `.tifxyz` segments | fine rungs only | high (libigl/VC3D build) | M, G | none | §21.5 already says "later, refinement/interactive only" — keep it there |
| I6 | CT inner/outer face classification | usrm `ctfaces.py` | CT + morphology | 0-3 | CPU-bound morphology per batch | P | usrm used it as a *label-QA* source, not an input | per-scroll `BODY_THRESHOLD_BY_SCROLL` tuning; welds are genuinely ambiguous |
| I7 | lasagna cos/grad_mag/dir preprocessor output | lasagna `preprocess_cos_omezarr.py` | another model | - | very high | - | — | **reject**: importing a second network's features is the cascade pattern we already own, and it is circular (lasagna's UNet is trained on lasagna's own fits) |

### 1b. OUTPUT channels

| # | head | source | label source | rungs | cost to build targets | attacks | evidence | pitfall |
|---|---|---|---|---|---|---|---|---|
| O1 | **verso probability** (cout 1->2) | usrm2 §23 | 5090 verso region stores | 2-3 (weight 0 at >=4) | **DONE** | P, M | in progress | no verso surface to score against (`evalsurf --head verso` is a thickness readout only) |
| O2 | **UDF** (unsigned distance to recto face, clamped) | usrm `model.py` HEAD_ORDER, `labels.py`; villa `aux_distance_transform.py`; lasagna `pred_dt` | **EDT of our existing target pyramid** — no new label | 0-4 | one EDT pass per rung per store, GPU or scipy, offline into `<name>_udf.zarr/<rung>` | G, T | usrm shipped it in a 4-head model; villa ships `SignedDistanceLoss` w/ Eikonal; lasagna ships `pred_dt`. **3 of 5 docs.** Never A/B'd against a binary-only baseline anywhere. | distance does NOT 2x-mean-pool (pooling a distance field is not the distance of the pooled field) — must be recomputed per rung, in that rung's voxel units; garbage above a store's native rung where the target is already fractional |
| O3 | **SDIST** (signed, + on the papyrus side) | usrm `sdist`; tsm `rvfaces.py` `sdf_in`/`sdf_out` | same EDT + a sign from the radial vector | 0-4 | as O2 | P, T | tsm `faces30k` (two-face SDF) **dice 0.636-0.669, the best surface parameterisation it tried**; usrm dropped the single signed head because "it saturates everywhere" | the sign must come from the radial vector, never from a teacher's face choice (tsm §0); **drop supervision within a radius of the axis** — tsm measured `recto_is_in` -> 0.47 near the core |
| O4 | **surface normal (nz,ny,nx)** | lasagna `labels_to_lasagna_normals.py`; villa `aux_surface_normals.py:compute_surface_normals_from_sdt`; usrm `geom.orient_normals`; vc3d §4.1; tsm `rvfaces` PCA normals | gradient of the O2/O3 distance field, or PCA of the mask band | 0-4 | free once O2/O3 exist (Scharr of the SDT) | O, P, T | **5 of 5 docs**. usrm oriented by radial majority vote per 64x64 block, flagged `|dot(n,radial)|<0.3` unreliable | three different sign conventions in the source docs (villa outward, `aux_nearest_component` inward, lasagna +z hemisphere) — see §2 |
| O5 | **winding phase (sin,cos)** | lasagna `labels_to_winding_volume.py` (pure python, cc3d+scipy); tsm `winding_fine.py` (`f = sdf_in/P mod 1`, `P = t+g`) | CC + chain + skeleton + DT-interp of the published mask, or the O3 face pair | 0-3 only | high: a fragile chaining pipeline per region, needs a validity mask | **M** | tsm kept `winding_fine` and measured that the coarse prior aliased 6x against it (28 vs 167 voxel spacing). Its output value on surface metrics is **unmeasured everywhere.** | a wrong chain order is confidently-wrong supervision, worse than none; aliases at rung >=4 exactly as tsm's coarse field did; blocked on O3 in the tsm formulation |
| O6 | **thickness / half-thickness along the normal** | usrm `thick` head; tsm `rvfaces` thickness channel | `|recto SDT| + |verso SDT|` once verso covers a region | 2-3 | free once O1+O3 exist | P | **negative**: usrm's ablation found dropping `w_thick` to 0.0 *increased* recto dice | thick=0 means "unknown", not zero — a fragile convention that bit usrm |
| O7 | confidence / validity head | vc3d §4.6; tsm's 3-way 0/1/2 validity | teacher disagreement (recto vs m7 lineage) | all | small | T | none | we already merge the two lineages by averaging (§3); tsm would instead down-weight the disagreement band. A *head* that predicts its own uncertainty is a different, unvalidated thing |
| O8 | fiber direction (dz,dy,dx,strength) | tsm `fiber.py` | needs a fiber teacher we do not have | 0-3 | blocked | - | tsm: accuracy topped out 0.67, **no measured effect on surface dice** | needs I4; no bootstrap source in usrm2 |
| O9 | ink | tsm `labels.py` | no ink GT in the bootstrap | - | blocked | - | tsm ran it as a separate head throughout | out of scope until an ink source exists |
| O10 | in-plane tangent / structure-tensor eigenvector | villa `aux_inplane_direction.py`, `aux_structure_tensor.py` | our own prediction | 2-6 | small | O | none | §21's own rule kills it: the first 3x3x3 layers compute a structure tensor |
| O11 | double-angle `dir0/dir1` per axis-plane | lasagna `fitted_to_unet_labels.py:_encode_dir` | any normal field | - | small | - | exists only because lasagna's input UNet is 2D-slice-based | **180-degree symmetric by construction** — it throws away exactly the sign the verso channel exists to supply |

### 1c. LOSSES

| # | loss | source | needs a target? | rungs | est. step cost | attacks | evidence | pitfall |
|---|---|---|---|---|---|---|---|---|
| L1 | **normal-gated repulsion** — penalise pairs within radius tau where both p>0.5 and normals near-parallel | villa `losses.py:NormalGatedRepulsionLoss` | **no** | 2-6 | +10-15% *(estimate; villa publishes no cost)* | **M** | implemented upstream, no ablation number given | a naive O(N^2) pair loop is unaffordable at 256^3 — must be a fixed small offset set of shifted dots; needs a normal estimate (grad of p) which is noisy in flat regions |
| L2 | **planarity** pi = (l2-l1)/(l0+l1+l2) of the prediction's own structure tensor | villa `losses.py:PlanarityLoss` | no | 2-6 | +5% *(estimate)* | G | implemented upstream, unablated | punishes legitimately blobby regions (crushed/damaged); needs a gate on p |
| L3 | **soft exclusivity** relu(p_recto + p_verso - 1) | usrm2 §23 already **measures** this as `overlap` | no | 2-3 | ~0 (one elementwise term) | **P, M** | our own metric, already plumbed in `train.evaluate` | turning a metric into a loss makes it stop being a diagnostic; keep a held-out copy |
| L4 | **cascade self-consistency** — pool(p at rung k, 2x) vs the rung-(k+1) self prediction already computed in `--cascade self/mix` | usrm2 §22 (new use of an existing tensor) | no | 2-10 | **~0** in `self`/`mix` (the coarse forward already runs) | G, X | novel here *(speculative)*, but the coarse forward is measured | must not backprop into the coarse branch (it is EMA/no-grad); stop-gradient one side or it collapses to agreeing on nothing |
| L5 | **Eikonal** `||grad d|| ~ 1` on a distance head | villa `losses.py:SignedDistanceLoss`; usrm §6 "grad(udf) alignment" | no (self-consistency on O2/O3) | 0-4 | +3-5% *(estimate)* | G, T | standard; villa ships it | meaningless where the target is a pooled fraction rather than a real band |
| L6 | **normal-vs-radial alignment** `\|dot(n,radial)\|` near the band | usrm `geom.orient_normals`, lasagna orientation loss | no | 0-4 | +3% | O, P | usrm used the same test as a *validity flag* (`<0.3` = unreliable) | false in crushed/folded regions where the sheet really is not radial — gate it, do not enforce it |
| L7 | **topology: spherical-Betti or ECT-mass** | villa `betti_losses.py`, `spherical_betti_loss.py`, `ect_loss.py` | uses the existing target | one mid rung | high *(unknown; persistent homology on 256^3 is not a per-step budget)* | **M, G** | implemented upstream, unablated; ECT-mass needs no external package | `betti_matching` is a new hard dependency; pilot at one rung on 2D slices before believing it |
| L8 | **48-symmetry equivariance consistency** | tsm `equivariance.py` (diagnostic, not a loss) | no | all | +50% (second forward) or ~0 as a periodic **diagnostic** | O | tsm used it to *discover* that m7's recto/verso follows a z-axis convention | tsm deliberately chose augmentation + measurement over equivariance-by-construction. Run it as a QA probe, not a loss |
| L9 | **swap-invariant face loss** (min over the two assignments) | tsm `label_store.md:576` | - | - | - | - | tsm needed it because its convention was crop-local | **not needed here**: our radial vector is a global convention. Do not port |
| L10 | per-head loss balancing (EMA of per-head grad norm) | tsm `student.py` LossBalancer | no | all | ~0 | - | built in tsm, **shipped OFF, never validated** | apply only *if* a third head measurably dominates the gradient norm |
| L11 | feature distillation (cached encoder / DINO) | tsm `feats.py`, `dino.py` | no | - | large | - | **implemented, never measured, two bugs found in the harness** | redundant with the cascade channel, which does the same job structurally and *is* measured |
| L12 | winding-density barrier (integral of density per wrap = 1) | lasagna `opt_loss_winding_density.py` | needs a mesh | - | very high | M | a mesh-solver loss, not a voxel loss | requires meshing the prediction inside the training step. Out of scope |

Dedup notes: **normals appear in all five docs** (O4) and converge on "gradient of a signed distance field,
sign fixed externally". **Distance fields appear in four** (usrm udf/sdist, villa aux_distance_transform,
lasagna pred_dt, vc3d surf-SDT) — and vc3d *already consumes* one that `make_surf_sdt.py` builds post-hoc from
our binarised predictions, so O2/O3 is the one candidate that simultaneously serves training and replaces a
downstream pipeline stage. **Winding appears in three** (lasagna winding volume, tsm winding_fine/winding,
vc3d spiral phase) with wildly different evidence: geometric per-region = kept, learned-coarse = failed.

---

## 2. What the tracer consumes, and what we would have to emit to replace it end to end

From `vc3d_tracer_inputs.md` §1, the fitter (`scripts/spiral/fit_spiral.py`, loaded by `lasagna_data.py`)
reads exactly three dense volumetric fields plus two geometry inputs:

| tracer input | today's producer | what the unified model would emit instead |
|---|---|---|
| normal grid `nx`,`ny` (two uint8 zarrs, typically 2-4x coarser) | lasagna / image gradients | **O4**, one store, three components |
| `grad_mag` uint8 (scale 1000) | lasagna | O5's density, or drop it: it only feeds the *legacy* `dense_spacing_mode='grad_mag'` path |
| surf-SDT uint8 (OME multiscales) | `make_surf_sdt.py`, post-hoc EDT of a binarised prediction | **O3** directly, at higher fidelity than a binarisation round-trip |
| verified/unverified patches, track PCL json | VC3D UI | unchanged, not ours |

### The channel set to output, with conventions pinned

All stores follow §23/§24: `<root>/<channel>/region_<z>_<y>_<x>.zarr`, 1024^3, zarr v3 sharded (one
`c/0/0/0`), serializer `VolcompCodec(q=8)`, `compressors=None`, attrs `origin_zyx`/`voxel_um`/`rung`/
`volume`/`umbilicus`/`channels`/`done`, written by `predict.out_array`.

1. `recto` uint8 0..255 — exists.
2. `verso` uint8 0..255 — exists.
3. `sdist` uint8 — **`sd_working_voxels = (v - 128) * unit`, `unit = 0.25` working voxels, cap ±32, v=0
   reserved for no-data.** This is byte-identical to `make_surf_sdt.py`'s scheme, so `lasagna_data.py` needs
   no change at all. Sign: **positive outside the sheet body, negative inside** (the tracer's convention),
   which is the *opposite* of usrm's legacy `sdist` (positive on the papyrus side) — convert at export, and
   train in whichever sign we like internally.
4. `normal` uint8 x3 — `component = (v - 128) / 127`, components stored in **ZYX order: [nz, ny, nx]**, the
   same order `sdt_losses.py:445-483` returns after decoding. Store the **full signed 3-vector**, with the
   sign defined as **pointing from the verso face toward the recto face**, i.e. `dot(n, radial) > 0` where
   the radial vector is the one `prep.radial_t` already builds — `(0, dy, dx)/|.|` in ZYX with z-component
   exactly 0, pointing away from the scroll axis. This is strictly more information than what the tracer
   reads today (it stores only the `nz >= 0` hemisphere and reconstructs `nz = sqrt(1 - nx^2 - ny^2)`), so a
   two-line shim at export writes the tracer's `nx`/`ny` pair by flipping the vector wherever `nz < 0`. Do
   **not** adopt lasagna's +z hemisphere convention internally (villa_lasagna §5): it introduces a sign
   discontinuity exactly where our recto/verso labels have none.
5. `phase` uint8 cyclic (optional, O5) — `theta = v/256 * 2*pi`, only meaningful within a wrap, and the
   tracer would use it to turn a discrete winding search into a continuous regression (vc3d §4.3).
6. `conf` uint8 (optional, O7) — down-weights the tracer's losses in merged/noisy zones.

Pitfalls that must be honoured on any export: **axis order is ZYX everywhere** (vc3d §5.1) — usrm2 is already
ZYX internally (`data.read3`, `rung_item`'s `lo`, `prep.radial_t`), so the only translation risk is at the
mesh/tifxyz boundary, which is XYZ in fullres voxels; **units are working voxels**, so a store written at rung
k must record `voxel_um` and the consumer must rescale (a rung-3 SDT in rung-3 voxels is 2x the physical
distance of the same number at rung 2); **0 is reserved for no-data in every uint8 distance field**, not a
small distance; **air (CT == 0) is masked automatically** by the tracer, and by our §3 weight rule, so the two
agree for free.

---

## 3. Dead — do not re-try as-is

1. **Learned coarse winding at 9.6 um reverse-engineered from villa's `winding_model_9um`** (tsm §1).
   `WINDING_ENABLED = False`. Only 42% of windows within ±0.3 of the true rate against a >=60% gate; density
   peaks matched CT sheet peaks 38% of the time; a sweep over ~10 architecture conventions found no variant
   that tracked sheet spacing, and the validation harness itself passed on constant input. Two lessons:
   never reverse-engineer a foreign checkpoint from tensor shapes, and never build a coarse winding field
   from a receptive field that cannot resolve the sheet pitch.
2. **Collapsing the two-sided surface into one signed field.** tsm measured it three ways
   (`runs_orientation.md`): two-face `faces30k` dice 0.636-0.669 with 5.6-11.1% body-merges; single signed
   `body30k` dice **0.507 with 66.7% missed**; magnitude/sign `sides30k` dice **0.36 with 61.1% missed**.
   Both single-field variants are marked REJECT. usrm2 already has the right answer (two separate output
   channels, §23) — this is a lesson to *not undo*, not one to port. In particular: do not replace
   `recto`+`verso` with one `sdist` head to "save a channel".
3. **Feature distillation (`feats.py`, `dino.py`).** Implemented in tsm, ablation config written, **no
   results exist on disk**, and two real bugs were found in the surrounding plumbing (the spatial transform
   was not passed to the distillation term under one aug mode; the DINO window stitching overwrote instead of
   blending). It is also gated to signed-permutation augs only, ~1/4 of samples. The cascade channel already
   does the "learn from a coarser view of yourself" job, structurally and measurably.

Also dead-on-arrival, with weaker evidence: the `thick` head as a *priority* (usrm's ablation: dropping
`w_thick` to 0.0 slightly **increased** recto dice); fiber class-mode targets (overlap 0.90 = no class
separation at all until the direction reparameterisation plus axis-tangent inputs landed); the double-angle
`dir0/dir1` encoding (O11, 180-degree symmetric); tsm's swap-invariant face loss (L9, unnecessary given our
radial convention); and gating a fine rung on a coarse field (tsm measured a 6x aliasing error, 28 vs 167
voxels of implied sheet spacing — which is precisely why §22's cascade uses noise and dropout instead of a
hard gate).

---

## 4. Phased roadmap

### Phase A — now, no new labels: label-free losses

What: **L3** soft exclusivity, **L4** cascade self-consistency, **L1** normal-gated repulsion, **L2**
planarity. All operate on the prediction only; none needs a byte of new data on disk.

- `usrm2/train.py`: new `losses_aux(logit, wv, radial, coarse=None, ...)` returning a dict of scalars,
  summed into the existing `bce + dice` with flags `--w-excl` (default 0.0), `--w-cascade-consist`,
  `--w-repel`, `--w-planar`. Keep each default 0 so every existing run is bit-for-bit unchanged, exactly the
  discipline §22/§23 used for `--cascade off` and `--verso`.
- L3: `relu(p[:,0] + p[:,1] - 1)`, masked by `wv[:,0] * wv[:,1]` so it is silent on the (majority) samples
  with no verso store. `train.evaluate` already computes `overlap` — keep reporting it, and additionally
  report it on a held-out rung that carries **no** exclusivity weight, or the metric stops being evidence.
- L4: in `--cascade self|mix`, `prep.Cascade` already holds the rung-(k+1) EMA prediction (no-grad). Add
  `F.avg_pool3d(sigmoid(logit), 2)` vs that tensor's central block, L1 or BCE, **stop-grad on the coarse
  side** (it already is: no-grad EMA). Cost: one pool and one subtract — call it free.
- L1: estimate the normal as the normalised central difference of `sigmoid(logit)` (6 shifts, no new head),
  then sum over a fixed offset set (the 13 unique offsets at radius 1-2 plus a handful at radius 3-4, not
  an O(N^2) pair loop) of `p_i * p_j * exp(-d^2/sigma_d^2) * exp(-theta^2/sigma_theta^2)`. **Exempt
  near-antiparallel pairs** — that is the recto/verso pair of one sheet and must not be punished.
- L2: 6-component structure tensor of `sigmoid(logit)` with the villa layout, but reassembled to XYZ before
  `eigh` exactly as `aux_inplane_direction.py` does (villa_vesuvius §5, axis-order pitfall); gate on p>0.5.

Warm start: none needed — the network is unchanged, so `u3`'s checkpoint resumes with `--w-repel` switched
on mid-run. Step cost: L3+L4 ~0; L1+L2 **estimated** +15-20% combined, all GPU-side, and the run is often
loader-bound (§22), so the wall-clock cost may be near zero.

Metric that must move: **`merge_frac` 0.44 -> below 0.40** (L1, L3) and **`continuity` 0.656 -> above 0.68**
(L2, L4). If `merge_frac` does not move with L1 at a weight that visibly changes the loss curve, drop L1;
it is the least-evidenced of the four and the most expensive.

### Phase B — new dense outputs from existing masks and meshes

What: **O2/O3** distance head (+**L5** Eikonal, **L6** radial alignment), **O4** normal head. Optionally
**O5** winding, which is a separate, much riskier project.

Target generation, per rung, per store:
- New `usrm2/targets.py` entry point `dist_pyramid(src, dst, rungs)` (the module §7.1 already calls for):
  for each rung k where the source has a *near-binary* target (its native rung and at most one above), read
  the rung-k target block, threshold at 0.5, run a 3D EDT **in rung-k voxel units**, sign it by the radial
  vector at that voxel, clamp to ±32, encode uint8 with offset 128 and unit 0.25 (the tracer's scheme, §2).
  **Distance fields must not be 2x-mean-pooled** — the EDT of a pooled mask is not the pool of the EDT, so
  each rung is computed independently and the pyramid is not built by `pool`. Store as
  `<name>_sdist.zarr/<rung>`, volcomp q8, `compressors=None` (§24), same attrs as §3 plus `unit`/`cap`.
- O4 needs no store of its own: the normal is the normalised Scharr gradient of the O3 store, computed on
  the GPU in `prep.prepare` from the already-loaded sdist channel. One fewer pyramid to build and keep in
  sync.

Where the weight must be **0**:
- every rung above (native rung + 1), where the target is a pooled fraction and the "surface" is many voxels
  thick — the same pattern §23 uses for verso above rung 3;
- within a configured radius of the umbilicus axis, verbatim from tsm `rvfaces` (`recto_is_in` measured at
  ~0.47 near the core — a coin flip). tsm **dropped** these voxels rather than down-weighting them; do the
  same. `data.axis_at` already gives the axis per rung, so this is a radius test against the same tensor
  `radial_t` builds;
- where CT == 0 (already the §3 rule);
- outside the store's box (already the §3 rule).

Code: `data.Patches._rung_build` appends the sdist channel to `tgt`/`w` (they are already
`(channels,Z,Y,X)` uint8 in `rung_item`); `model.build(cout=...)` grows 2 -> 3 (sdist) or 2 -> 5 (sdist +
3 normals); `train.losses_tw` gains a per-channel loss *kind* — the probability channels keep BCE+dice, the
distance channel takes a Huber on the decoded value (usrm used Huber on `T*(pred-target)`; villa uses
Smooth-L1 + band weighting via `surface_sigma`), the normal channels take `1 - dot`. This is where §23's
single `losses_tw` stops being enough and a small per-channel dispatch is required.

Warm start: `train.warm_start`'s "head j takes source head j mod n" rule copies recto into the new sdist
slot, which is **wrong** for a distance head (a probability filter produces nonsense distances). Extend
`warm_start` with an explicit per-channel policy: `copy mod n` for probability channels, **zero-init** for
regression channels. The bias should start at the encoding's zero (128), not 0.

Step cost: the head itself is `w0` parameters per output (§23: 32 weights + 1 bias on the 30m6) — nothing.
The real cost is the loader (one more uint8 channel per sample, ~+3% of sample bytes at 3 extra channels)
and the Eikonal term's finite differences, **estimated** +5%. Target-build cost is offline and one-off.

Metric that must move: **`offset<=3` 0.36 -> above 0.45** (sub-voxel localisation is exactly what a distance
head buys) and `offset_std` down. `recall@4` should not regress; if it does, the distance head is stealing
capacity and its weight is too high (tsm's LossBalancer, L10, is the ready-made mitigation).

### Phase C — the strict manifold recto/verso loss, once verso lands

Prerequisite: verso coverage wide enough that `dice_verso` appears in `eval.jsonl` for the val box at rungs
2 and 3 (§23 — it is simply absent today), and O3/O4 from Phase B so the pairing can be expressed
geometrically rather than by voxel coincidence.

The three properties the user asked for, as three terms:
- **mutually exclusive**: L3 promoted from a soft penalty to a hard one — `p_r * p_v` (product, not the
  `relu(sum-1)` hinge), so the two channels cannot both commit anywhere.
- **paired**: for every recto voxel, a verso voxel must exist at `x - t*n` for some local thickness `t` in
  the physical range (usrm measured 15-35 voxels at 2.4 um, i.e. **rung-dependent** — 7-17 at rung 3). As a
  loss: soft-sample `p_v` along `-n` over the plausible `t` band and penalise the shortfall. This is O6
  (thickness) re-derived from the pair rather than supervised as its own head, which sidesteps usrm's
  negative `thick` ablation.
- **non-intersecting**: L1 with the antiparallel exemption removed on the *recto-recto* and *verso-verso*
  pairs and kept on *recto-verso* — two recto faces at 1 voxel with parallel normals is exactly a merge.
- **topology** (L7): pilot **ECT-mass** first, not Betti matching. ECT-mass needs no external package
  (`ect_loss.py`), runs at one mid rung, and if it does not move `merge_frac` the far heavier
  `betti_matching` dependency is not worth taking on.

Code: a new `usrm2/manifold.py` holding the three terms, flags `--w-manifold-excl/-pair/-repel`, and a
`--manifold-rungs 2,3` gate so nothing fires where verso has weight 0. Warm start: unchanged network, so any
Phase B checkpoint carries over. Step cost **estimated** +10-20% (the pairing term is a handful of
`grid_sample` reads along the normal).

Metric: `merge_frac` is the headline. Also add an **exclusivity metric on held-out rungs** and a
pair-completeness metric (fraction of recto surface points with a verso hit within the thickness band) —
once exclusivity is a loss, `overlap` is no longer independent evidence.

### Phase D — tracer-ready outputs and the corpus fine-tune

- **Export path**: `predict.out_array(..., channels=["recto","verso","sdist","nz","ny","nx"])` already writes
  the channel names into the store attrs (§23), so the stores are self-describing. Add
  `usrm2 export-tracer REGION_DIR OUT` that writes the tracer's exact layout: the surf-SDT store with OME
  multiscales metadata, and the `nx`/`ny` hemisphere pair (flip where `nz<0`). Then `fit_spiral.py` needs
  **no code change** — only new paths in its config, and `make_surf_sdt.py` drops out of the pipeline.
- **Scan metadata planes (I3)** plus **normalised radius (I2)**: both are constant-or-cheap planes appended
  after the cascade channel and before the radial vector, i.e. `[CT, ctx_1..9, CASCADE, radius, meta_1..m,
  scale, radial(3)]`. `train.warm_start` already knows how to widen the stem while keeping "image channels
  first, radial last" and pinning the scale plane (§22) — extend the same explicit-slot mapping to the new
  planes and zero-fill them, which makes the warm start exact to a float32 ulp as the cascade one was.
  `prep.prepare` builds them; `data.group_meta` reads `metadata.json` next to each volume.
- **Corpus fine-tune (42 scrolls)**: only worth doing *after* I3, because without scan-conditioning planes a
  42-scan corpus asks one model to absorb 74/78/137 keV, two Paganin settings and a 19-tile mosaic as
  unexplained nuisance variance. Sampling stays §4's `n_k^0.5` rule, per store.
- Step cost: planes are ~0 GPU; the corpus fine-tune is a data-logistics problem (§6's mirror plan), not a
  compute one.

Metric: for D the metric is **cross-scan** — per-scroll `recall@4`/`continuity` on held-out scrolls, and the
variance across them, not the mean. For the tracer export, the metric is the tracer's own convergence, which
we cannot measure inside usrm2; ship the store and ask.

---

## 5. Top 5 recommendations, in priority order

1. **Phase A now, on the live `u3` run: soft exclusivity (L3) + cascade self-consistency (L4), both at
   near-zero step cost.** These are the only two candidates in the entire survey that need no new label, no
   new head, no new store and no measurable compute — L3 is a loss form of a number `train.evaluate` already
   prints, and L4 reuses a tensor the `--cascade self|mix` forward already computed and currently throws
   away. They attack recto/verso pairing and cross-rung gaps respectively, which are two of the three stated
   failures. The whole intervention is one function in `train.py` and two flags defaulting to 0.0, so the
   risk is bounded by the flag.

2. **Phase B's signed-distance head (O3) with an Eikonal term, at rungs 0-4, targets built by an independent
   per-rung EDT of the target pyramid we already have.** This is the single best-evidenced output candidate:
   four of the five docs converge on it, tsm measured the two-face SDF as the best surface parameterisation
   it ever trained (dice 0.636-0.669 vs 0.507 and 0.36 for the single-field alternatives), usrm shipped it,
   and villa ships the loss. It costs `w0` parameters, it gives non-zero gradient away from the band (which
   is what a binary head cannot do and what `offset<=3 = 0.36` says we need), and it is the one output that
   simultaneously deletes a downstream pipeline stage (`make_surf_sdt.py`). The hard part is not the model,
   it is the discipline that distance does not pool: recompute per rung, weight 0 above native+1, weight 0
   near the axis.

3. **A signed normal head (O4) derived from that distance field, with the sign pinned to "verso -> recto"
   against `prep.radial_t`'s outward vector.** Normals are the only candidate that appears in all five
   source documents, the tracer reads a normal grid as one of its three dense inputs, and a signed normal
   formalises the `--radial-sign -1` flip trick (§23) into a real output. It is nearly free once O3 exists
   (a Scharr gradient of the sdist channel — no second pyramid). Pin the convention once, in writing, at the
   store's attrs: three components in ZYX order, `(v-128)/127`, full signed vector, `dot(n, radial) > 0` on
   the outer wraps — precisely because the source repos disagree three ways (villa outward, villa inward in
   a sibling file, lasagna +z hemisphere).

4. **Normal-gated repulsion (L1) as the dedicated merge attack, piloted at one rung before it goes
   everywhere.** `merge_frac 0.44` is the worst number on the board and nothing currently in the loss
   expresses "these two nearby sheets are different sheets" — BCE and Dice are both indifferent to a bridge.
   L1 is the cheapest formulation of that constraint that needs no new label. It is also the highest-variance
   recommendation here: villa publishes the implementation but no ablation, so it goes in behind a flag, at
   one rung, with the antiparallel exemption (or it will punish the recto/verso pair of a single sheet), and
   it comes back out if `merge_frac` does not move.

5. **Scan-metadata conditioning planes (I3) plus normalised radius (I2), landed *before* the 42-scroll
   corpus fine-tune, not after.** Paris 4 alone spans 74/78/137 keV, two Paganin `delta_beta` settings and a
   220 mm / 11 m propagation split; a corpus fine-tune without these planes asks the model to absorb all of
   that as unexplained nuisance variance, and the failure mode (a model that is mediocre everywhere instead
   of good somewhere) is expensive to diagnose after the fact. The radius plane is the other half: the
   radial vector gives direction but not how far out a voxel sits, and radius is what actually sets sheet
   pitch, curvature and damage. Both are constant-or-cheap planes and both warm-start exactly, by the same
   explicit-slot mechanism §22 already had to build for the cascade channel.

## The top 3 things NOT to do

1. **Do not collapse recto and verso into one signed field to save a channel.** tsm measured this three
   times: two-face dice 0.636-0.669, single signed field 0.507 with 66.7% missed, magnitude/sign 0.36 with
   61.1% missed — both single-field variants marked REJECT. If a signed distance head lands (recommendation
   2) it goes *beside* the two probability channels, never instead of them.

2. **Do not build a learned coarse winding field, and above all do not reverse-engineer one from a foreign
   checkpoint.** tsm's `WINDING_ENABLED = False` cost a lot to establish: 42% of windows within tolerance
   against a 60% gate, 38% peak alignment, ~10 architecture conventions swept with no variant tracking sheet
   spacing, and a validation harness that passed on constant input. If a winding signal is ever wanted, it
   comes from the geometric per-region construction (lasagna `labels_to_winding_volume.py`, tsm
   `winding_fine.py`) at fine rungs only, with a validity mask, and never as a gate on a finer rung.

3. **Do not add feature distillation (DINO / cached-encoder cosine loss), and do not import another model's
   dense fields as input channels.** The first is tsm's least-evidenced idea (implemented, never measured,
   two bugs found in the harness, gated to a quarter of samples); the second is circular by construction
   (lasagna's UNet is trained on labels derived from fits that were optimised against that same UNet). Both
   are trying to do what the cascade channel already does — and the cascade channel is implemented,
   measured (+6%/+35%/+50%) and in production.
