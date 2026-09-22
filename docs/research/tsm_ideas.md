# tsm ideas for the unified model's input/output/loss (2026-09-21)

Survey of /home/forrest/tsm (the predecessor "tiny scroll model", src/tsm/) auxiliary heads, geometric
label builders, teacher-distillation channels, and evaluation tooling, mapped onto usrm2's unified ladder
model (`docs/unified_design.md` sections 1-4: CT cube + 9 context cubes + cascade + scale plane + radial
vector in, recto/verso probability channels out, per section 22-23). Each idea is scored as an (a) INPUT
channel computable at inference, (b) OUTPUT channel with a derivable dense target, or (c) LOSS/constraint,
with rungs, expected benefit, and the pitfall tsm actually hit. Sources: `tsm/src/tsm/{winding,
winding_fine, fiber, rvfaces, equivariance, feats, dino, labels, student}.py`, `tsm/docs/{winding_status,
student_v2_plan, label_store, runs_orientation, antagonistic_code_review_2026-09-06}.md`.

## 0. What tsm actually validated, before the per-idea detail

The single most load-bearing, repeatedly-measured finding across `runs_orientation.md`: **collapsing a
two-sided surface representation into one signed field regresses topology hard.** `faces30k` (two separate
SDFs, one to the inward face `sdf_in` and one to the outward face `sdf_out`) scored Dice 0.636-0.669 with
5.6-11.1% body-merges; `body30k` (single signed SDF `min(sdf_in, -sdf_out)`, saving one channel) dropped to
Dice 0.507 with 66.7% missed; `sides30k` (magnitude/sign decomposition) to Dice 0.36 with 61.1% missed. Both
one-channel variants are explicitly marked REJECT. usrm2 already has the two-sided answer for free: recto
and verso are two *separate* output channels (section 23), not one signed field — this is the tsm lesson
already correctly applied, not one to port.

The second cross-cutting finding: recto/verso identity is **not** an intrinsic geometric property — it is
an orientation convention that a teacher assigns per training crop, and it flips under symmetry. tsm
measured this directly (`student_v2_plan.md` H.1, via `tsm.equivariance`/`dev/tta_side.py`): the m7 teacher's
recto/verso choice follows a z-axis convention, and a 180 degree in-plane rotation swaps which physical face
it paints. usrm2 sidesteps this by construction (radial vector direction defines "outward"/recto, not a
teacher's per-crop labeling), but any auxiliary target built from teacher-assigned face identity (fiber
class, thickness sign, or a face-relative winding phase) inherits this ambiguity and must be defined against
the radial vector or a global convention, never against a teacher's local face choice.

## 1. winding (coarse, 9.6 um) — do not port

**What it computed**: a scalar winding-rate density along outward CT-space rays (`WindingNet`, decoded to
`density = softplus(logit)` windings/voxel, `phase = cumsum(density)`), reverse-engineered from villa's
`scrollprize/winding_model_9um` checkpoint with no available source (`tsm/src/tsm/winding.py:1-50`).

**How trained**: not trained in tsm — architecture/weights archaeology of an external checkpoint; several
conventions (norm type, attention order, residual pre/post-act) were unverifiable from weight shapes alone
and tagged "guess".

**Status**: **FAILED and disabled** (`WINDING_ENABLED = False`, `winding.py:111`). Real-ROI validation:
only 42% of windows landed within +-0.3 of the true winding rate (gate >=60%), density peaks aligned with
CT sheet peaks only 38% of the time (`winding_status.md:84-90`). A synthetic sweep over ~10 architecture
conventions found "no variant makes the winding rate track the sheet spacing" (`winding_status.md:63-66`).
The validation harness itself had a bug (a flat/constant CT input could pass the periodicity check trivially,
`antagonistic_code_review_2026-09-06.md:253-259`).

**Mapping**: none. This is a dead end, not a candidate. If usrm2 ever wants a coarse winding/sheet-index
signal, do not attempt to reverse-engineer an undocumented external checkpoint; either get the source or
skip straight to the geometric approach below (section 2), which tsm found strictly better anyway.

**Lesson for usrm2**: reverse-engineering a foreign checkpoint from tensor shapes is a trap when multiple
architectural conventions are simultaneously unverifiable — the result can "look right" (correct band
structure, right mean rate) while being quantitatively useless per-window, and there is no way to
disambiguate without the source or paired I/O samples.

## 2. winding_fine (2.4 um, geometric) — candidate OUTPUT + input to itself

**What it computed**: a continuous fractional winding phase field derived purely geometrically from two-face
SDF labels (`sdf_in`, `sdf_out`), no CT, no network: `f = (sdf_in / P) mod 1` where the local sheet period
`P = t + g` (thickness + gap) is recovered per-box from `sdf_in - sdf_out` via EDT fill
(`tsm/src/tsm/winding_fine.py:151-339`). Output channels: `sin(2*pi*w)`, `cos(2*pi*w)`, density (wraps per
fine voxel), outward normal `(nx,ny,nz)`, geometric confidence, validity.

**How derived**: deterministic function of face-distance labels + local PCA-oriented normals; no training.

**Status — kept, with an explicit measured reason to distrust the coarse teacher**: the module docstring
records that on a crumpled crop, the face-implied sheet spacing is ~28 fine voxels while the coarse
9.6 um winding field implies ~167 (a 6x aliasing error), so the fine phase is **not gated** on agreement
with the coarse prior by default (`snap_max=None`) — "gating the (correct) fine phase on the (aliased)
coarse one would throw away good supervision and keep bad" (`winding_fine.py:55-59`).

**Mapping to unified model**:
- (b) OUTPUT: a `winding_phase` auxiliary channel pair (sin, cos) is derivable at any rung where a two-face
  or medial-SDF representation exists, by exactly this method — but usrm2 has no two-face SDF target today
  (only binary recto/verso probability). Building it would require first exporting an SDF-style target
  pyramid (see the usrm_legacy_ideas.md UDF/SDF discussion) before a winding-phase output is derivable at
  all. Rungs: only where sheet spacing is resolvable relative to the rung's voxel size (fine rungs 0-3 at
  most; at rung 4+ this aliases exactly like the coarse teacher did).
  - (c) LOSS: a cross-field consistency term (recto/verso SDF zero-crossing aligned with phase turning
    points) was proposed but never built in tsm (`student_v2_plan.md` section C) — worth doing here if an
    SDF or two-face head is ever added, as a cheap regularizer with no separate target store.
- **The concrete, low-risk lesson to port now, with no new channel required**: never trust a coarse rung's
  own field as validity ground truth for a finer rung in the unified model either. This directly confirms
  usrm2's cascade design choice (section 22) to add **noise + dropout** to the cascade channel rather than
  using it as a hard gate — tsm's winding aliasing is the same failure mode the cascade noise/dropout
  already guards against.

**Pitfall**: needs at least half a period of sheet-body and half a period of gap inside the box to recover
`t`/`g`, otherwise returns all-invalid; phase-origin convention differs from the coarse store by a roughly
constant offset that must be tracked if mixing sources.

## 3. fiber (vertical/horizontal orientation) — candidate OUTPUT, low priority

**What it computed**: per-voxel in-sheet fiber orientation, either as a 2-class soft label (vertical vs
horizontal/circumferential, distilled from a separate villa 4-class fiber teacher's softmax) or, after a
bug fix, as a sign-free axial direction vector `d` in the sheet tangent plane plus a strength scalar
(`tsm/src/tsm/fiber.py: fiber_basis`, `direction_from_class`).

**How trained**: distilled from an external teacher's softmax probabilities (not computed from CT/labels
directly); the direction variant is a closed-form transform of those probabilities plus the geometric sheet
normal — no new supervision, "nothing new is stored" (`fiber.py:31`).

**Status — a real, well-documented bug and its fix**: fiber classes are *axis-relative* (vertical =
aligned with scroll z), so any augmentation rotating the volume's z-axis silently corrupts the vt/hz labels
unless corrected. Class-mode "overlap" (both vt and hz firing — should be near 0) was measured at 0.90 in
early runs (`faces30k`), i.e. the head was not separating the classes at all. Only after adding 3 explicit
axis-tangent input channels and switching to the direction parameterization did overlap drop to 0.03-0.12
(`runs_orientation.md`). Direction mode is explicitly called "the proper fix" because it is a true
rotation-equivariant geometric field (`fiber.py:18`).

**Mapping to unified model**:
- (a) INPUT: none needed for the *target*, but the fix required exposing an **axis-tangent channel** to the
  network, analogous to how usrm2 already exposes the radial vector. usrm2 does not currently carry an axis
  channel; if a fiber-like head is ever added, this input is a hard prerequisite, not optional.
- (b) OUTPUT: a 4-channel fiber-direction head (dz, dy, dx, strength) at fine rungs (0-3, where fiber
  texture is CT-visible) is plausible and cheap (one 1x1x1 head), but the *target* would have to come from a
  distilled teacher tsm doesn't have in usrm2 (no fiber teacher checkpoint here) or be derived from CT
  texture directly (untested in tsm — tsm always used an external teacher for this).
- (c) LOSS: none standalone; only useful as a training target for the head above.

**Lesson**: any orientation target defined relative to the scroll axis (not the radial vector) needs the
axis itself as an input, or augmentation will silently break it exactly as it did here. Low priority for
usrm2 since there is no existing fiber teacher to bootstrap from, and the benefit (recall/continuity) is
unmeasured even in tsm — fiber accuracy topped out at 0.67 with no reported effect on surface Dice.

## 4. rvfaces (recto/verso face SDF builder) — candidate OUTPUT, direct fit

**What it computed**: not a loss or head itself — a geometric label builder (`tsm/src/tsm/rvfaces.py`) that
thins a voxelized recto/verso mesh band to a ~1-voxel medial sheet per face (3D EDT ridge), computes local
outward normals via windowed PCA oriented by the radial direction from the umbilicus, and produces signed
EDT distance-to-face fields `sdf_in`/`sdf_out` (sign always geometric, from the local outward normal, **never
from CT or a teacher's recto/verso label**) plus a validity and thickness channel.

**Status — validated, this is the surviving surface parameterization** (section 0 above): `faces30k` /
`faces gc1` (Dice 0.636-0.669) is the best of every surface variant tsm tried, including the single-field
alternatives it explicitly rejected. Two extra findings worth carrying:
- Near the umbilicus the in/out (recto/verso) convention degenerates to a coin flip — measured
  `recto_is_in` fraction falling to ~0.47 near the axis (`label_store.md:596-598`) — so voxels within a
  configurable radius of the core were dropped from supervision entirely, not down-weighted.
- Training loss was made **swap-invariant** (pick whichever of {sdf_in as target0, sdf_out as target0} gives
  lower loss) because "in/out" still requires a *global* outward convention that no single crop reveals on
  its own (`label_store.md:576-580`).

**Mapping to unified model**:
- (b) OUTPUT: usrm2's radial vector already gives a global, per-voxel outward direction (unlike tsm's
  crop-local convention problem), so the swap-invariance issue tsm had to work around mostly does not apply
  here — recto/verso are already assigned by radial direction, not by teacher convention. This argues FOR
  eventually building an SDF-style (signed distance to nearest face) output alongside or instead of the
  current binary/probability recto and verso channels, using exactly tsm's geometric construction (thin
  the mask to a medial sheet, EDT to it, sign by the radial vector) — this is a strict generalization of the
  probability channel (adds sub-voxel localization) and would reuse the umbilicus-degeneracy mask (drop
  supervision within some voxel radius of the axis) directly. Natural rungs: 0-4, where sub-voxel precision
  matters; coarser rungs get little from an SDF over the existing fractional-probability encoding.
  - Concretely this maps onto usrm_legacy_ideas.md's UDF/SDIST discussion (same idea, independently
    invented in usrm's predecessor too) — two independent codebases converged on signed-distance-to-face as
    the right dense target, which is corroborating evidence, not a new argument.
- (c) LOSS: the near-umbilicus dropout rule (weight 0 within a radius of the axis for any face-identity
  target) should be reused verbatim if an SDF head is added — it's a one-line weight-mask change, cheap
  insurance against training on a known-degenerate label.

**Pitfall**: two independent label builders (CT-geometric vs mesh-derived) disagreed measurably in tsm; a
`merge_face_labels` disagreement histogram was needed as a calibration/QA signal. If usrm2 ever mixes a
geometric SDF builder with the existing published-mask-derived probability targets, expect the same kind of
disagreement and budget for a similar sanity check (e.g. compare derived-SDF zero-crossing to the existing
probability band's 0.5 crossing on a held-out box before trusting it).

## 5. equivariance — LOSS/diagnostic tooling, not a channel

**What it computed**: not a training loss — an evaluation/diagnostic toolkit (`tsm/src/tsm/
equivariance.py`) implementing the 48-element octahedral group (exact signed-permutation transforms) plus
arbitrary SO(3) rotations, with per-channel "invert" rules to bring a transformed-input prediction back to
the original frame, used to measure a model's own orientation bias and to validate which symmetry transforms
preserve a one-sided prediction (recto vs verso) vs flip it.

**Status**: used to derive the H.1 finding quoted in section 0 (m7's recto/verso choice follows a z-axis
convention) — i.e. it was diagnostic infrastructure that *led to* a design decision (geometric face
labeling), not itself a trained component. `escnn`-style equivariance-by-construction was considered and
explicitly deferred in favor of measuring bias and relying on augmentation (`student_v2_plan.md:137`).

**Mapping to unified model**: (c) not a channel or head — but the measurement *method* is directly reusable
as a QA tool for usrm2's own cube-symmetry augmentation (48 symmetries, already in use per section 4 of
`unified_design.md`): periodically running the model on a symmetry-transformed patch, inverting, and
diffing against the untransformed prediction would catch orientation bias the same way tsm's
`dev/orientation_bias.py` did. Two bugs are worth avoiding on a port: (1) an average-precision helper with
broken tie-handling gave AP 1.0 vs 0.417 for differently-ordered ties on identical scores
(`antagonistic_code_review_2026-09-06.md:185`) — use a proper tie-aware AP if this is ever measured; (2) a
valid-support mask built to exclude `grid_sample` zero-padded corners under continuous rotation was
constructed but never actually applied in one script (`R23`, review:` dev/tta_side.py:411,460`) — any
continuous-rotation (non-cube-symmetry) diagnostic needs that masking wired through, not just built.

## 6. feats / dino (frozen-teacher feature distillation) — LOSS, speculative, unvalidated in tsm

**What they computed**: `feats.py` caches frozen encoder activations from tsm's own recto/ink teacher
U-Nets; `dino.py` is a bit-exact pure-PyTorch reimplementation of villa's externally-trained `dinovol_2` 3D
ViT (24-layer, 864-dim, DINO-style self-supervised checkpoint, verified bit-exact against the reference
implementation). Both feed a training-only cosine-similarity distillation loss between a projected student
encoder stage and the cached teacher/ViT features — never exported, purely a regularizer.

**Status — implemented but never measured**: an ablation config (`ablate_feats.json`, variants `live_every1
/ cached_every1 / cached_plus_dino / none`) was written specifically to test whether feature distillation
helps, but **no results exist anywhere on disk** — it was configured and apparently never run, or results
were lost. Two real bugs were found in the surrounding plumbing: (1) under one augmentation mode the spatial
transform wasn't passed through to the distillation term, so cached targets were sampled in the wrong frame
vs. the augmented student input (`antagonistic_code_review_2026-09-06.md:131`); (2) the DINO window-stitching
had an overlap-overwrite bug rather than a blend (`review:95-97`). Both encoder types are rotation- but not
arbitrary-rotation-equivariant, so the distillation term is gated to samples whose augmentation is a signed
permutation only — roughly a quarter of samples qualify under strong augmentation
(`label_store.md:1031-1032`).

**Mapping to unified model**: (c) LOSS only, and low priority — this is the weakest-evidenced idea in the
whole tsm codebase (implemented, never validated, two bugs found in the harness itself). If usrm2 wanted
this, the direct analogue would be distilling from usrm2's OWN coarser-rung predictions into the same-rung
encoder — but that is already what the **cascade channel** (unified_design.md section 22) does, as an input
rather than a loss, and it is measured (mask/self/mix cost 6-50% extra step time, in production). There is
no reason to add a second, unvalidated mechanism (feature-space distillation) to do a similar job the
cascade channel already does structurally, unless a genuinely new frozen teacher (e.g. a foundation vision
model, not tsm's own U-Nets) becomes available and worth distilling from.

## 7. labels (label-store architecture) — mostly process lessons, one direct OUTPUT idea

tsm's label store (`tsm/src/tsm/labels.py`, `docs/label_store.md`) built, per region: a recto medial-SDF +
ink store, an optional two-face store (section 4 above), an optional 3-channel fiber store (section 3), an
8-channel fine winding store (section 2), an 8-channel coarse winding store, and a human-correction overlay
with a carefully scoped "region of influence" recompute after a bug where a full-crop relabel silently
perturbed 10M of 16.7M voxels from non-reproducible builder state (`label_store.md`, section on human
corrections). This maps less onto a single channel and more onto process lessons for usrm2's own
`targets.py` (`unified_design.md` section 7.1):

- **Ink** as a distinct target: tsm always trained ink as a separate 1-channel head (BCE + soft-Dice,
  auto pos-weight) alongside surface, never folded into the surface channel. (b) OUTPUT candidate for
  usrm2 if/when ink ground truth is available for any scroll — cheap (one more head, same warm-start pattern
  as the cout 1->2 verso addition in section 23), but usrm2 currently has no ink source in its bootstrap
  plan at all, so this is deferred, not scoped.
- **Validity/ignore encoding**: tsm used a 3-way convention (0=no-data, 1=supervise, 2=ignore/uncertain)
  everywhere, distinct from usrm2's current weight-tensor product-of-masks (section 3 of
  `unified_design.md`). The two are compatible (ignore = weight 0) but tsm's explicit "ignore because the
  teacher disagreed with itself" case (recto probability in [0.2, 0.8]) is a signal usrm2 does not currently
  compute — worth considering as an extra weight term wherever multiple teacher lineages disagree (recto vs
  m7, already merged as "the student learns an average band" per section 3 of `unified_design.md` — tsm's
  approach would instead down-weight the disagreement region rather than average through it).
- **Label-store patch scoping bug**: a full-store regeneration after a human correction silently touched
  voxels far from the correction due to three independent non-reproducibility sources (a PCA band not
  stored, provenance loss when merging two builders, brick-wise halo effects). Direct lesson for usrm2's own
  `targets.py import_mask`/export pipeline: any future re-export or patch of a target pyramid must be
  provably scoped (only touch voxels within a bounded radius of what changed) and the builder's random/PCA
  state must be either stored or made deterministic — a full re-run is not a safe substitute once multiple
  target sources are merged, which usrm2 already does (recto + m7 feeding one head, section 3).
- **72 GiB allocation bug** in a label audit script (`np.repeat` over a full histogram including background,
  `review:109-113`) — a reminder that per-voxel audit/QA tooling at Paris-4 scale needs `bincount`-style
  aggregation, not per-voxel materialization, or it will blow memory budgets that look fine on a small box.

## 8. student architecture — one structural lesson, already reflected in usrm2

tsm's student combined every active head's loss into one scalar under one global gradient clip, and
explicitly flagged the risk: "the head with the largest loss (or gradient) scale owns the step"
(`train.py:42-43`). An EMA-based per-head loss-scale balancer was built but shipped OFF by default (mode=
None reproduces historical training byte-for-byte) — i.e. tsm never actually validated that the balancer
helped, only that it was safe to leave disabled. usrm2's current recto+verso combination already lives with
this exact risk on a smaller scale (BCE + soft dice, weight tensor per channel, section 23), and the more
heads get added from this survey (fiber, ink, winding-phase, SDF), the more this risk compounds. If usrm2
adds a second or third auxiliary head, budget a check for one head dominating gradient norm before assuming
the fixed per-head `loss_weights` are adequate — tsm's own mitigation (`LossBalancer`) is a small, ready-made
pattern to copy if that happens, not something to build from scratch.

## Summary table

| idea | tsm status | mapping | rungs | benefit | priority |
|---|---|---|---|---|---|
| winding (coarse, 9.6um) | FAILED, disabled | none | - | - | do not port |
| winding_fine (geometric phase) | kept, active | (b) output, needs SDF target first; confirms cascade noise/dropout design | 0-3 | continuity/merge-avoidance if SDF exists | low, blocked on SDF |
| fiber | fixed after 0.90->0.03 overlap bug | (a) needs axis input; (b) output | 0-3 | unmeasured on surface Dice | low, no teacher to bootstrap from |
| rvfaces (two-face SDF) | validated, best surface param | (b) output, direct generalization of recto/verso probability | 0-4 | sub-voxel localization, merge-avoidance | medium, natural next step |
| equivariance | diagnostic only, led to design fix | (c) QA method, not a channel | all | orientation-bias detection | low-cost, reuse the method not the code |
| feats/dino | implemented, never validated, 2 bugs found | (c) loss, redundant with cascade | - | unmeasured | lowest |
| labels (process) | mixed | process lessons for targets.py | all | avoids known regressions/bugs | free, apply now |
| student (loss balancing) | risk flagged, mitigation unused | (c) apply if heads multiply | all | prevents one head dominating gradient | apply when 3rd head added |
