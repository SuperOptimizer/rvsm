# Research corpus index

These 18 documents are copied unchanged from `/home/forrest/usrm2/docs/research/`. They are the evidence
base for rvsm's fixed recipe: the block of `Config` fields below the `# fixed recipe` comments in
`rvsm/config.py` is a default from the first commit rather than a research flag, and the justification for
each of those defaults is in one or more of these documents. The plan that froze them is
`/home/forrest/.claude/plans/flickering-cooking-iverson.md` §4.

The documents were written against usrm2's `docs/unified_design.md` (not copied here), so their section
cross-references point at that design document. Internal section numbers cited below are the ones inside
each research document. Where a document itself flags a citation as recall-based, unverified, or as an
inference rather than a measurement, that is noted; the bibliography at the end carries the same flags.

Three groups: five code-provenance surveys of sibling repositories, two synthesis documents, twelve
external-literature reviews.

---

## Code-provenance surveys

### [villa_lasagna.md](villa_lasagna.md)

A read-only survey of `/home/forrest/villa/lasagna` for dense per-voxel representations that could become
extra channels or losses. §1 describes lasagna as three stacked representations (a per-axis 2D UNet emitting
`cos`/`grad_mag`/`dir`, a 3-axis fusion into a 3D normal, and the quadmesh fit that is the "lasagna model"
proper) plus two label-only pipelines that need no UNet: `labels_to_lasagna_normals.py` and
`labels_to_winding_volume.py`. §3 enumerates the exportable products (normals, winding number, cos-of-phase,
grad_mag, `pred_dt`, a cylinder-SDF violation depth, double-angle direction). §4(a) is the load-bearing
negative: lasagna offers **no compelling new input channel**, because its per-voxel fields are themselves
model outputs and feeding one network's output to another is the cascade pattern already owned — which is
why rvsm's `planes = "radius+meta"` carries only geometry and scan metadata, and `cascade = "mix"` is the
only prediction-derived input. §4(b) ranks a winding/cos head first for merge avoidance and a normal head
second; rvsm takes neither as a head (`channels = ("recto", "verso")`, `head_names()` has no normal or phase
row) and instead derives normals at export by Scharr, per the plan's "normals derived". §4(c)'s `pred_dt`
two-regime distance loss is the ancestor of `loss_sdist = 1.0`. §5's pitfalls are the source of two
conventions rvsm keeps: the recto-outward sign convention (rather than lasagna's +z hemisphere) behind
`verso_source = "flip"`, and the "scope orientation channels to fine rungs only" rule reflected in the
midline/thickness stores being built at rungs 2-4 only. Confidence: this document is a code survey, so its
claims about what lasagna computes are verifiable by reading; its claims about what any of it would buy usrm2
are explicitly unmeasured — the priority ranking at the end of §4 is reasoning from failure modes, not from
an ablation, and §5 flags that a wrong winding chain order is "confidently wrong supervision, worse than no
signal at all".

### [villa_vesuvius.md](villa_vesuvius.md)

A survey of `/home/forrest/villa/vesuvius` (the nnU-Net-style multi-task trainer), `villa-volcomp` and
`volume-cartographer`. §1 tabulates every auxiliary target and loss found: surface normals from the SDT
gradient, structure tensor, in-plane direction, distance transform, nearest-component vector, surface frame,
ECT loss, Betti-matching losses, planarity, normal smoothness, normal-gated repulsion, plus several
neural-tracing-only items. §2 documents the shared recipe (binary mask → signed distance transform → Scharr
or eigendecomposition) and that it runs CPU-side per sample, uncached. §3 records that vc3d consumes tifxyz
meshes, not dense probability volumes. §4's priority list puts a repulsion-style merge term first, a
signed-distance head reusing the existing ramp second, and a topology loss (spherical-Betti, or ECT-mass as a
cheaper stand-in) third. This is the direct source of `loss_sdist = 1.0` and `loss_eikonal = 0.1` (villa's
`SignedDistanceLoss` ships Smooth-L1 + Eikonal + band weighting), of `loss_ect = 0.05` with `ect_n = 1`,
`ect_rung = 2`, `ect_block = 64` (villa's `ect_loss.py` "mass" variant needs no external `betti_matching`
package), and of the heteroscedastic `logvar` head at `Layout.i_log`. §5's pitfalls set two rvsm rules: one
explicit documented normal sign convention (villa's own `aux_surface_normals.py` and
`aux_nearest_component.py` disagree on sign within the same package), and precomputing targets into stores
rather than per-sample CPU EDT — which is why `targets.py`/`region_fields` builds the midline and thickness
stores offline. Confidence: every loss here is "implemented upstream, no ablation number given"; the document
says so for `NormalGatedRepulsionLoss` and `PlanarityLoss` explicitly. Neither has a corresponding rvsm
config field, which is the correct reading of that evidence.

### [vc3d_tracer_inputs.md](vc3d_tracer_inputs.md)

The downstream contract: what the VC3D spiral fitter actually reads. §1 lists the five volumetric inputs
(`normal_nx`/`normal_ny` uint8 with `(u8-128)/127` decoding, `grad_mag` uint8 scale 1000, a uint8 surf-SDT
with offset 128 and 0 reserved for no-data, verified/unverified patches, track PCLs). §2 defines spiral space
and the winding pitch (10-16 voxels). §3 walks the six fitter losses and concludes with the critical insight
that the tracer "struggles when recto/verso sheets are indistinguishable or close", when bands are too thick,
or when phase wraps are lost. §4 is the wish list that rvsm's head is built around: a normal field (4.1), a
recto/verso sheet-pair field (4.2), a phase field (4.3), a signed distance field (4.4), normalised radius
from the umbilicus (4.5) and a confidence field (4.6). rvsm takes 4.2 as `channels = ("recto", "verso")`, 4.4
as the `midline` head with `loss_sdist = 1.0`, 4.5 as the radius plane in `planes = "radius+meta"`
(`Layout.i_radius`), 4.6 as the `logvar` head, and 4.1 as a derived-at-export field rather than a head. 4.3
(phase) is deliberately not taken. §5 fixes the encoding conventions the exporter must honour — ZYX order
throughout, distances in working voxels so `voxel_um` must be recorded, 0 reserved for no-data — which the
plan's §2 store attrs and the q0 midline encoding (`code 128 + d/0.25`) implement byte-compatibly with
`make_surf_sdt.py`. Confidence: §1, §5 and §6 are a code reading with file:line references and are strong;
§4 is a wish list written by the surveying agent, with no measurement that any of these channels improves the
tracer's convergence — the document's own closing line says so ("would allow the tracer to disambiguate"),
and §5.5 notes the grad_mag encode scale of 1000 is "empirical".

### [tsm_ideas.md](tsm_ideas.md)

A survey of `/home/forrest/tsm`, the predecessor "tiny scroll model", scored per idea as input/output/loss.
§0 carries the two most load-bearing measurements in the whole corpus. First: collapsing a two-sided surface
into one signed field regresses topology hard — `faces30k` (two SDFs) Dice 0.636-0.669 at 5.6-11.1%
body-merges, versus `body30k` (one signed field) Dice 0.507 with 66.7% missed and `sides30k` Dice 0.36 with
61.1% missed, both marked REJECT. That is the direct justification for `channels = ("recto", "verso")` being
two separate probability heads with the distance head beside them, never instead of them. Second:
recto/verso identity is a per-crop orientation convention that flips under symmetry, so any face-relative
target must be defined against a global geometric convention — rvsm's is the radial vector, and
`verso_source = "flip"` is exactly that. §1 records the coarse 9.6 µm winding model as FAILED and disabled
(42% of windows within ±0.3 against a ≥60% gate, density peaks matching CT sheet peaks 38% of the time), so
no winding channel exists in rvsm. §2's measurement that the coarse winding field aliased 6× against the fine
one (28 vs 167 voxels of implied spacing) is the evidence for `cascade_noise = True` and
`cascade_drop = 0.1`: never use a coarse rung as a hard gate on a finer one. §3 (fiber) is why rvsm has no
fiber head and no axis-tangent plane. §4 (rvfaces) validates the geometric-EDT-signed-by-radial construction
behind `targets.py` and the near-umbilicus dropout rule (`recto_is_in` measured ≈0.47 near the core). §5's
equivariance work and §8's loss-balancer both stay out of the config: the balancer was "shipped OFF, never
validated", so rvsm keeps fixed loss weights. Confidence: §0-§4 rest on real in-house numbers; §6
(feats/dino) is explicitly "implemented, never measured, two bugs found in the harness", and §3's fiber
accuracy of 0.67 has "no reported effect on surface Dice".

### [usrm_legacy_ideas.md](usrm_legacy_ideas.md)

A survey of the older `usrm` (`src/usrm/`), whose `HEAD_ORDER` was `("recto", "udf", "sdist", "thick")`.
§1 documents the UDF and SDIST heads: clamped at T = 6.0 level voxels, encoded to 127, trained with Huber on
`T * (pred - target)` plus a soft-Dice variant, and records that the signed head was dropped in practice
"because it saturates everywhere" while remaining useful for ablations. That measured encoding-and-Huber
shape is the ancestor of rvsm's midline head and `loss_sdist = 1.0`, and the saturation finding is why rvsm
signs its distance to the **midline** rather than to a face. §2 (thick head) carries the one negative
ablation in the corpus: dropping `w_thick` to 0.0 *increased* recto dice, and the `thick = 0` means
"unknown" convention is called fragile. rvsm's answer is to keep `thickness` as a head at `Layout.i_thick`
but as the bounded offset parameter of the midline pair rather than a separately weighted supervised target —
it shares `loss_sdist` and has no weight of its own. §3 (CT inner/outer faces) and §5 (medial features) are
recorded as label-QA and post-processing, not channels; §4 (winding) and §6 (grid normals) are marked "not
applicable" and "not necessary" respectively, the latter because normals are derivable from the field, which
is the rvsm export rule. The recommendations section ranks UDF+SDIST medium, thick low, CT faces low. This
document also independently invented signed-distance-to-face, which tsm_ideas §4 explicitly calls
corroborating evidence rather than a new argument. Confidence: the thick ablation and the sdist saturation
are real measurements; the "mapping to unified model" subsections are inference throughout, and the document
says the medial label store existed for Paris 4 only.

---

## Synthesis documents

### [synthesis_future_inputs_outputs.md](synthesis_future_inputs_outputs.md)

The first consolidation, built only from the five in-house surveys, written when `u1_30m6_p4` scored val dice
0.740/0.740/0.682 at rungs 2/3/4 and the val box scored recall@4 0.806, continuity 0.656, merge_frac 0.44,
offset≤3 0.36. §1 is the candidate table: inputs I1-I7, outputs O1-O11, losses L1-L12, each with source,
label source, rungs, cost, which failure it attacks and its pitfall. It marks everything with no measurement
anywhere as *(speculative)*. §2 pins the tracer contract and its encodings. §3 is the dead list: learned
coarse winding, collapsing recto+verso into one signed field, feature distillation, plus weaker-evidence dead
items (the `thick` head as a priority, fiber class-mode targets, the double-angle encoding, tsm's
swap-invariant face loss, gating a fine rung on a coarse field). §4 is the A-D phased roadmap; §5 the top five
recommendations. The rvsm defaults traceable here: `loss_excl = 0.1` (L3 soft exclusivity) and
`loss_selfcons = 0.1` (L4 cascade self-consistency) are its two headline near-zero-cost recommendations;
`cascade = "mix"` with `cascade_drop = 0.1` and `cascade_noise = True` come from I1; `planes = "radius+meta"`
is I2 plus I3; `ctx = (1..9)` and the stem order `[CT, ctx, CASCADE, radius, meta, scale, radial]` are fixed
in §4 Phase D. O2/O3 become rvsm's midline/thickness heads and `loss_eikonal = 0.1`. L1 (normal-gated
repulsion), L2 (planarity), L7 (Betti) and O5 (winding) are proposed here but have no rvsm config field.
Confidence caveats are unusually explicit: every step-cost figure for L1, L2, L5 and Phase C is marked
*(estimate)*; L4 is marked *(speculative)*; L7's cost is "unknown"; O2/O3 is noted as "never A/B'd against a
binary-only baseline anywhere"; and O5's value on surface metrics is "unmeasured everywhere".

### [synthesis_v2_with_literature.md](synthesis_v2_with_literature.md)

The revision of the above after the twelve literature reviews, and the document rvsm's fixed recipe is
closest to. It introduces a per-row confidence tag: **[MH]** measured in-house, **[ML]** measured in
published literature, **[ML?]** where the surveying agent flagged the citation as recall-based or ran out of
web-search budget, **[S]** speculative. It states directly which documents the [ML?] tag applies to: most of
`lit_layered_structures` (its own `[M]` marks), `lit_optimisation_schedules` §2-4, and the arXiv IDs in
`lit_noisy_labels_self_training`. §1 re-tabulates inputs, outputs, losses and adds two new categories — §1d
training recipe (R1-R10) and §1e evaluation (E1-E5). §2 updates the tracer contract. §3 is the expanded dead
list (now 12 items, including Mamba/attention backbones, the newer optimisers, BatchNorm at batch 2, MC
dropout/EDL/deep ensembles, screened Poisson, and equivariant convolutions). §4 is the A0-D roadmap, §5 the
twelve costed experiments, §6-7 the revised recommendations. Nearly the whole fixed-recipe block of
`config.py` lands here: `sched = "wsd"` with `cooldown = 0.1` (R1), `ema = "auto"` with `ema_k = 50` (R2),
`rewarm = 800` and `new_param_lr_mult = 3.0` (R3), `calibrate = True` (R4), `self_p_lo = 0.1` →
`self_p_hi = 0.7` (R6), `aug = "full2"` (R9), the `rvsm pretrain` path (R10), `loss_skel = 0.05` with
`skel_iters = 4` (L8), `loss_affinity = 0.1` with `aff_offsets = (8, 16, 32)` (O12, offsets tuned to the
15-35-voxel sheet pitch), `loss_ect = 0.05` (L7), the `midline` + `thickness` reparameterisation (O3b),
`logvar` (O7 as a heteroscedastic sdist), `loss_eikonal = 0.1` (L5), `loss_excl`/`loss_selfcons` carried
forward, `rounds = 3` self-distillation with the plateau round gate, and `eval_every = 500` with
`heldout = 8` (E1-E5). The one deliberate deviation is that rvsm enables `loss_ect` from the first commit,
which §29.9 of the design had excluded only for attributability. Confidence: the [ML?] tag is applied to O5,
R2, R3, §3.1 and recommendation 5; every Phase-C step cost is [S]; and §6.1's whole argument is that the
project's own decision rule was untestable before Phase A0 lands.

---

## Literature reviews

### [lit_topology_merge_losses.md](lit_topology_merge_losses.md)

Four clusters of non-VC work on merges and gaps. §1 covers clDice, Skeleton Recall, clCE, homotopy warping,
DMT, TopoLoss, TopoInteraction and three 2025-2026 entries; its key finding is the split by failure mode —
clDice and Skeleton Recall are *recall*-type losses that reward a connected centerline, so **a merge bridge
is itself thin and connected and can score well under them**; they close gaps, they do not prevent merges.
§2 covers Betti matching and the Euler characteristic transform and concludes that fast ECT (arXiv:2507.23763)
is the correct pilot because it is the only method with a dedicated 3D formulation *and* a stated cost
advantage over persistent homology, with Efficient Betti Matching 3D (arXiv:2407.04683) as the fallback that
costs a hard C++ dependency. §3 covers EM connectomics and lands the corpus's best-evidenced merge idea: the
field's answer is a relational signal between voxels at an offset, not a better mask loss. §4 covers OCT,
cortex, seismic, tree rings and vessel gap-closers. §5 ranks long-range affinity first, DconnNet second, fast
ECT third, Skeleton Recall fourth. rvsm takes `loss_affinity = 0.1` with `aff_offsets = (8, 16, 32)` from §3,
`loss_skel = 0.05` / `skel_iters = 4` from C2, and `loss_ect = 0.05` with `ect_rung = 2` and `ect_block = 64`
from B5 — the block edge implements the cluster-wide pitfall that patch cropping truncates real topological
features at every crop edge, so the term must run on an interior sub-block. Confidence: C8, C9, C10, B7 are
marked "not deeply reviewed" / "unconfirmed"; B2's wall-clock numbers are explicitly "in the paper body, not
pulled here — read before committing"; several 2026-dated arXiv ids are given as returned by search.

### [lit_implicit_surfaces_manifold.md](lit_implicit_surfaces_manifold.md)

Five areas: (a) dense SDF/UDF regression, (b) joint normal/frame learning, (c) topology-preserving
nested-surface cortical reconstruction, (d) layered media and unrolling, (e) multi-task head weighting.
(a) confirms the clamped-signed-distance-plus-Eikonal recipe (`loss_sdist = 1.0`, `loss_eikonal = 0.1`) and
resolves the apparent UDF counter-argument: dropping the sign is right for a lone open sheet and wrong for a
pair where the sign is the pairing signal. It also supplies the SIREN pitfall that a ReLU conv decoder has
piecewise-constant autograd gradients, so the normal must be derived from the *stored* distance field —
which is why `head_names()` has no normal rows and the plan derives normals by Scharr at export. (c) is the
single most consequential section: DeepCSR needs a non-differentiable post-hoc topology fix, Vox2Cortex and
PialNN are flagged by later papers as crossing-prone, and CortexODE / TopoFit / the NeurIPS-2023 Coupled
Reconstruction line get non-intersection free from an invertible offset off a shared midline. The concluding
recommendation — predict distance to the midline and derive the two faces as `midline ± (t/2)·n` with `t`
bounded below — is exactly rvsm's `midline` + `thickness` head pair at `Layout.i_mid` / `Layout.i_thick`.
(d) supplies the differentiable-DP soft-ordering idea and the RGT sin/cos precedent. (e) recommends Kendall
uncertainty weighting over tsm's EMA balancer *if* balancing is ever needed, which is why rvsm's loss weights
are fixed scalars. Confidence: (b) records an explicit negative finding — no dedicated literature exists for
"a normal field for a thin, self-intersecting-prone laminar sheet"; the cross-task-consistency row in (e) is
"fragment found via search, no single canonical paper isolated"; GeoUDF is cited without an arXiv id, and
Vox2Cortex is listed as MICCAI 2022 in the table and CVPR 2022 in the bibliography.

### [lit_noisy_labels_self_training.md](lit_noisy_labels_self_training.md)

Six topics (noise-robust losses, 3D self-training, knowledge distillation, coarse-to-fine cascades and
exposure bias, equivariance/TTA as a label-free signal, small-clean-set reweighting) mapped onto four label
sources S1-S4. The findings that shape rvsm: (d) names the cascade channel a textbook exposure-bias problem
and scheduled sampling the fix, which is `self_p_lo = 0.1` annealed to `self_p_hi = 0.7` rather than a fixed
0.5; RITM confirms that training against *predicted* priors beats clean-GT priors, which is `cascade = "mix"`;
channel dropout is confirmed as a complement to, not a substitute for, annealing, which is
`cascade_drop = 0.1`; and arXiv:2507.10143's finding that unconstrained self-feedback diverges without
damping is why the one-level truncation must be preserved deliberately (`cascade_depth = 3` is a
coarse-to-fine inference sweep, not a recursion). (b) and (c) argue for agreement-gated fusion of the two
teacher lineages over flat averaging, which is the `fuse_agreement` bootstrap behind `teacher_ckpts` holding
recto and m7 and the `rw` agreement-weight store. Born-Again Networks and Noisy Student are named as the two
best-evidenced "student beats teacher" results, which is the warrant for `rounds = 3`. The single most
repeated warning across all six topics is that self-training without an external anchor amplifies correlated
teacher errors, and that agreement-based filtering cannot catch the case where both teachers agree and are
both wrong — the reason the round gate in the plan is scored on held-out regions and meshes, not on
self-agreement. Confidence: **this document carries a blanket caveat in its own scope paragraph — "arXiv/DOI
ids were not independently re-verified against the live index, so treat a stale-looking id as a paraphrase of
the paper title/venue, not a guaranteed resolvable link"** — so every citation from it is flagged
[unverified] in the bibliography below, as synthesis_v2 also records.

### [lit_scaling_multiscale_generalisation.md](lit_scaling_multiscale_generalisation.md)

Scaling laws, scale-conditioned architectures, long-context 3D, and cross-scanner domain generalisation. Its
TL;DR states that no Chinchilla-style joint law exists for 3D dense segmentation; what exists is single-axis
model-size scaling on nnU-Net-family models, where returns are positive, sub-linear and task-dependent —
STU-Net still gained at 1.4B params, while easy tasks saturate. Saturation must therefore be detected
empirically: fit `1 - dice` against `log(params)` per rung across ≥3 sizes and watch the train/val gap, not
an aggregate mean. That is the warrant for rvsm's `size` presets and the `rvsm ladder` subcommand rather than
a fixed capacity. (b) is the strongest external endorsement of the conditioning design: voxel-spacing-agnostic
training beats resampling to a canonical grid, HyperSpace shows spacing conditioning works, and CoordConv
establishes that coordinate planes are the standard fix for a translation-invariant net needing position,
with the caveat that only planes compatible with the augmentation group may be added (radius is, angle is
not) — together justifying `rungs = (2..11)` with the scale plane at `Layout.i_scale` and
`planes = "radius+meta"`. (c) is the negative that keeps rvsm on convolutions: the controlled nnU-Net
Revisited re-benchmark finds CNN variants beat Transformer and Mamba networks once training budget and
augmentation are equalised. (d) recommends conditioning on the true acquisition parameters *and* augmenting
across their plausible range in the same change. Confidence: the document flags its own biggest evidence gap
— no outside paper isolates "context beyond 256³" as the fix for thin-structure discontinuity, so
`ctx = (1..9)` is called "a plausible-but-unvalidated-by-outside-literature bet"; BioVFM-21M is
"directional evidence, not a fitted law"; and the perivascular-space spacing result is from a much narrower
spacing ratio than rvsm's ladder.

### [lit_ct_physics_augmentation.md](lit_ct_physics_augmentation.md)

Physics- and simulation-based CT augmentation, compared op-by-op against the existing `aug.py` suite. §1
identifies SinoSynth as the closest match and extracts its transferable finding: randomise the *composition
order* of the detector/scan-domain chain per sample, since real acquisition composes source → object →
detector → recon and a network exposed to one order can learn order-specific correlations. §2 covers beam
hardening and recommends tying `_bias` amplitude to `energy_keV`. §3 concludes no change is needed for rings
and stripes. §4 is the most distinctive gap: no outside paper augments for Paganin delta/beta mismatch, so
the document derives the op from the physics — a single Fourier-domain transfer-function ratio at a resampled
delta/beta, log-uniform 0.5-2×, and separately notes that blur sigmas defined in voxels mean a different
physical blur at every rung, so `sigma_vox = sigma_um / voxel_um(rung)`. §9 ranks micron-based sigmas first,
`_paganin_jitter` second, order shuffling third. All three are inside `aug = "full2"`, whose ranges the plan
centres on the scan metadata frozen into `config.json`. §8 is the quantitative anchor: physics-augmented
training reached 0.74 Dice on real CBCT with zero real CBCT labels, and SinoSynth reports synthetic-only
training beating real-but-narrow training for cross-site robustness. §6 defers procedural fibrous-material
simulation, and §7 concludes no mosaic-seam op is needed. Confidence: the `_paganin_jitter` recommendation is
an inference from Paganin's formula, not a cited augmentation result — the document says the search "mostly
returns phase-retrieval/denoising papers" and calls this "a genuinely under-studied corner even outside VC";
the CFRP paper's noise model is "not confirmed from the abstract alone" because the paper is paywalled; and
the `_class_contrast` range check against tsm's 1.3× contrast spread is flagged for a numeric check rather
than performed.

### [lit_fibre_orientation.md](lit_fibre_orientation.md)

Structure-tensor orientation fields, learned orientation regression, fibre tracking in materials CT,
diffusion-MRI ODF estimation, and papyrus/parchment CT. §1 establishes the structure tensor as the classical
workhorse and then rejects it as an input channel by the same filter-redundancy rule the design already used,
with an external corroboration: no 2018-2026 paper feeds a structure-tensor or Hessian field into a
segmentation CNN and reports a measured gain — every use is post-hoc analysis or pre-CNN cleanup. That is
why `planes = "radius+meta"` contains no orientation planes and the stem has no structure-tensor input. §2
confirms tsm's squared-cosine axial loss as the textbook choice and documents the doubled-angle / outer-product
fix for sign ambiguity. §3 surveys composites, nonwovens, wood and paper, and records that low-contrast
cellulose CT is a known structure-tensor failure regime. §4 tabulates the input/output/self-supervised roles
and finds the input role rejected, the output role plausible. §5 notes the two blockers for a fibre head: an
axis-tangent input channel is a hard prerequisite (tsm measured class overlap dropping from 0.90 to 0.03-0.12
only after adding it) and there is no fibre teacher to bootstrap from — so rvsm keeps `channels =
("recto", "verso")` with no fibre head and no axis-tangent planes, and `cin = 21` reflects that. §6's bottom
line is that the literature corroborates each piece of tsm's already-correct architecture without offering a
shortcut. Confidence: the one new option, a self-supervised structure-tensor-derived vt/hz label, is an
inference by the surveying agent with no cited implementation; the "no paper found" claims in §1 and §3 are
negative search results, not proofs of absence; and the two foundational structure-tensor citations (Bigün &
Granlund 1987, Knutsson 1989) are cited via Wikipedia and software documentation rather than the papers.

### [lit_layered_structures.md](lit_layered_structures.md)

How other fields segment, order and unroll stacked or wound layers: battery jelly rolls (§1), retinal OCT
(§2), seismic relative geologic time (§3), tree rings (§4), phase unwrapping and Laplace coordinates (§5),
touching-instance separation and ordinal regression (§6), and a ranked recommendation (§7). §0 states the
answer plainly: nobody outside VC assigns a global wrap index to a 300-wrap spiral end to end. §2's
transferable trick is ordering by *parameterisation* — He et al. regress non-negative thicknesses and cumsum
them, reaching MAE 2.82 µm against 2.83 µm for the graph method — which is the external half of the argument
for rvsm's bounded-thickness midline pair. §3 shows the winding field already exists as RGT, and RGT-Est's
sinusoidal encoding is the strongest single idea offered. §6 supplies the long-range-affinity formulation:
offsets of 15-35 voxels along the normal encode "next wrap" directly, and the mutex-watershed repulsive edge
is how you say two sheets 20 voxels apart are different wraps — this is the second independent source for
`aff_offsets = (8, 16, 32)` and `loss_affinity = 0.1`. §7's ranking puts the tracer's local geometric
channels first (normal, clipped signed distance, ideally heteroscedastic so the variance *is* the confidence
channel — rvsm's `logvar` head) and a *local* wrap identity second, never a global index. The recurring
pitfall across every field is decisive for what rvsm omits: every method surveyed assumes a fixed layer count
per column and fewer than ~50 layers with hundreds of voxels of spacing, and every learned global field
smears at density. Confidence: **this document carries per-entry verification marks and most entries are
`[M]` (cited from memory, check before quoting numbers) or `[U]` (found but not retrievable)** — only Madi
2026, Garvin 2009, He 2018, Kugelman 2018, Liu 2021, SD-LayerNet, Islam 2024, ReLayNet, Geng 2020, Bi 2021
and Zambrano-Suarez 2026 are marked `[S]` verified; RGT-Est is `[S/U]` with the id "as reported by search,
not opened". synthesis_v2 tags the whole document [ML?] for this reason.

### [lit_uncertainty_active_labelling.md](lit_uncertainty_active_labelling.md)

Ten sections covering MC dropout, deep ensembles, cheap ensembles, evidential deep learning, TTA
disagreement, calibration, uncertainty-guided pseudo-label filtering, active-learning region selection,
interactive refinement and uncertainty-weighted losses, closing with a priority table. §6 is the one that
sets a config default: Dice-trained nets are measurably overconfident (Mehrtash), and the sigmoid is not a
probability *by construction* above native rung because the target there is a pooled fraction — so fit one
global temperature **per rung**, and never calibrate against pooled-fraction targets. That is
`calibrate = True` refitting after every eval, with the temperatures stored in the checkpoint. §5 makes
symmetry-TTA the cheapest uncertainty signal but warns that a model trained on the 48-symmetry group is
taught invariance to it, so only ranked spread is meaningful — which is why `tta = 1` at inference and TTA
stays a diagnostic. §1, §2 and §4 rule out MC dropout, full deep ensembles and evidential deep learning on
cost and on collision with the tuned BCE+dice+ignore design. §8's headline is nnActive, the largest 3D
biomedical active-learning benchmark, finding that no query method reliably beats a foreground-aware random
baseline — so region selection should be cheap disagreement at region-store granularity validated against
random, not BALD or core-set machinery; rvsm's walk is a deterministic weighted shuffle, consistent with
that. §10's heteroscedastic aleatoric head is the `logvar` channel. The cross-cutting caution is that every
signal derived from the model's own weights systematically under-flags that model's confident, systematic
failures — exactly merges — so none of it substitutes for the geometric losses. Confidence: several entries
are surveys rather than primary results; the SAM-Med3D-class recommendation is explicitly a *pilot*, with
"no evidence on transfer to CT of a carbonized, low-contrast papyrus sheet — a real domain-gap risk"; and
§10's claim that a learned variance head can degenerate is stated as a mechanism requiring the
metric-must-move discipline, not as a measurement.

### [lit_surface_extraction.md](lit_surface_extraction.md)

Nine sections on turning a dense probability/SDF volume into a clean 2-manifold: medial-surface thinning,
marching cubes and dual contouring on learned fields, Poisson reconstruction, topology repair, gap closing,
merge splitting by min-cut, crease-preserving smoothing, differentiable surface extraction, and a priority
ranking. Its top recommendation is Flying-Edges-per-shard on the exported signed-distance store, which
matches the existing 1024³ region grid and mechanically replaces `make_surf_sdt.py` — that is the plan's
`export.py` / `mesh_shards` path. Second is the Eikonal loss as the cheapest item in the whole survey, since
it extracts no mesh at all and conditions every downstream extraction for free (`loss_eikonal = 0.1`). §3 is
the strongest negative: screened Poisson "tends to merge nearby thin layers and bridge holes, because it
optimizes for a smooth, watertight result" — the merge failure named from the reconstruction side — so it is
excluded. §4 carries the cortical-surface lesson to constrain the representation rather than repair a
thresholded mesh, with DeepCSR's topology correction costing over 30 minutes per defect. §5 offers tensor
voting as a training-free gap closer and warns that any completion must carry a provenance flag, because a
confidently-wrong fill is worse than a visible gap in a document whose content will be read. §6 keeps the
min-cut targeted at flagged merge candidates only. §8 flags DMTet/FlexiCubes as speculative for this setting.
Confidence: the differentiable-mesh-extraction idea in §8 is called "genuinely new" and "speculative" by the
document itself and has no rvsm counterpart; the Poisson thin-feature spacing requirement (~1/10 of local
feature size) is quoted as a documented requirement, and the "instant self-intersection repair" and
"structural MAT" entries are single-line summaries not read in depth.

### [lit_pretraining_foundation.md](lit_pretraining_foundation.md)

Masked and contrastive pretraining for 3D volumes, CT/EM foundation models, and when pretraining helps small
labelled sets. §1's central citation is Wald/Isensee CVPR 2025, which pretrains a Residual-Encoder U-Net (a
CNN, not a transformer) with MAE on ~39k unlabeled volumes and beats both the supervised nnU-Net baseline and
prior 3D SSL by roughly +3 Dice on 8 held-out datasets, with the gain concentrated at low label counts. §2
surveys CT-FM, SAM-Med3D, SegVol, Merlin, VISTA3D and the EM side (CEM500K, micro-SAM) and records that **no
published foundation model targets micro-CT of papyrus or comparable materials**, so no checkpoint import is
available. §3 establishes three patterns: CNN backbones beat ViT/Swin at these data scales; pretraining
corpus *scale* matters more than pretext sophistication, but the literature's scaling claims are keyed to
scan diversity, not raw voxel count; and gains are largest on thin, structurally hard targets, with VAMAE
reporting the gain concentrated in clDice/topology metrics when masking is made structure-aware. §4 records a
genuine gap: no paper pretrains across a deliberate multi-resolution pyramid. §5's recommendation — a small,
CNN-native, in-domain masked-cube stage at fine rungs, explicitly *not* a DINO revival and *not* a foundation
checkpoint import — is the `rvsm pretrain` subcommand's label-free masked-cube path. Confidence: the expected
gain of "roughly 1-3 Dice-equivalent points" is stated as an analogy, not a measurement on this data; pitfall
(3) tells the reader to audit how many *distinct* scans the unlabelled corpus spans before assuming the
data-scale argument applies; and pitfall (5) says the stage should ship with a real from-scratch-versus-
pretrained ablation rather than being assumed to work — which is why `pretrain` is a separate subcommand and
not part of `rvsm run`.

### [lit_optimisation_schedules.md](lit_optimisation_schedules.md)

LR schedules, EMA, batch/LR scaling, optimisers, muP, loss balancing and normalisation. §1 is the source of
`sched = "wsd"` with `cooldown = 0.1`: MiniCPM's warmup → constant plateau → short cosine decay removes the
need to commit a step budget at run start, which matches how these runs are actually managed, and MiniCPM
reports a 10% decay length suffices. §2 gives `ema = "auto"` with `ema_k = 50` — a fixed 0.999 is a ~1000-step
window, which is 1.7% of a 60k-step run and 0.5% of a 200k-step one, so the window should be a roughly
constant fraction of run length; it also warns that a flat EMA validation curve is not proof of convergence.
§2 also gives `rewarm = 800` and `new_param_lr_mult = 3.0`: Ash & Adams found that resuming at a decayed LR
generalises worse than training from scratch, the continual-pretraining line found a fresh short warmup beats
resuming on the old tail, and newly-added parameters carry no pretrained memory to protect so they can take
full-strength LR from step 0 as a separate parameter group. §3 keeps `lr = 3e-4` and AdamW: batch 2 is deep
in the noise-dominated regime where large-batch scaling rules do not apply, the optimiser step is ~1% of wall
clock, and Muon's matricisation of a 5D conv kernel is undefined. §4 keeps fixed loss weights — two NeurIPS
2022 papers find tuned fixed weights match or beat GradNorm/PCGrad/uncertainty weighting for fewer than ~5
same-family losses — and keeps GroupNorm, since BatchNorm at batch 2 in 3D is roughly 10 points worse in
GroupNorm's own ablation. Confidence: **the document states at the top that this session's web-search budget
was exhausted after the first research pass — only §1 rests on live-verified searches with confirmed URLs,
and §2-§4 are "reconstructed from training knowledge without a live re-check", with exact titles, years and
arXiv IDs to be spot-checked before being relied on verbatim.** synthesis_v2 tags R2, R3 and recommendation 5
[ML?] for this reason. §1 additionally flags that the 10% decay-length figure is tuned for LLM-scale token
budgets and is "a starting guess, not a validated constant for this domain", and the `ema_decay = 1 - k/steps`
parametrisation note in §2 explicitly warns the reader to be careful with the parametrisation.

### [lit_evaluation_metrics.md](lit_evaluation_metrics.md)

Eight sections on evaluating thin-structure, topology-critical segmentation, written against the existing
`evalsurf.py` metric set. §1 covers surface Dice / NSD, §2 Hausdorff and HD95, §3 Betti number error and
Betti matching error, §4 clDice and skeleton recall, §5 the connectomics split/merge metrics (VOI, adapted
Rand, ERL/NERL), §6 metrics under label noise, §7 bootstrapping and plateau tests, §8 adjacent domains. Its
two highest-priority items are §6's noise ceiling — run the identical metric suite teacher-versus-mesh and
report every number as "X (ceiling Y)", turning "is 0.91 saturated?" from a judgement into a comparison — and
§5's ERL, the expected traceable run length in physical units before a split or a merge, which is the closest
published analogue to the actual quantity of interest and is symmetric over both failure types where
`merge_frac` is merge-only and `continuity` is split-only and unitless. §3 notes that `merge_frac` and
`continuity` are local hand-built proxies for b0/b1 defects that cannot see a hole spanning more than a couple
of grid cells. §7 supplies the statistical discipline: bootstrap by resampling *surfaces*, not points, at
N = 200-1000, and fit a bounded sigmoid rather than a power law to call saturation. This section is the
warrant for rvsm's `eval_every = 500` cadence with `heldout = 8` stratified regions, for the plateau fit that
ends a round, and for the round gate in the plan being expressed as "within the bootstrap CI" rather than as
a bare threshold. Confidence: §4 notes clDice is defined for curvilinear structures and that the 2D-manifold
analogue (surface thinning) may be noisier than the existing grid-based continuity metric — an open question,
not a result; §7 warns not to trust a saturation call from fewer than ~10-15 checkpoints; and the pitfalls
section warns that a noise ceiling computed from one teacher is itself noisy, which is why the plan computes
teacher-versus-teacher as a second independent ceiling.

---

## Bibliography

Every external paper cited across the twelve `lit_*.md` documents, deduplicated, grouped by topic. Titles,
venues and identifiers are reproduced as given in the source documents; where a source gives no venue or no
arXiv id, that is stated rather than guessed. **[unverified]** marks entries the source documents themselves
flagged as recall-based, approximate, unretrievable, or not re-checked against a live index. Three
document-level flags apply: `lit_noisy_labels_self_training` states that its arXiv/DOI ids "were not
independently re-verified against the live index"; `lit_optimisation_schedules` §2-4 is "reconstructed from
training knowledge without a live re-check"; and `lit_layered_structures` marks entries `[M]` (from memory)
or `[U]` (not retrievable) individually.

### Topology-aware losses and persistent homology

- clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR 2021. arXiv:2003.07311. — lit_topology_merge_losses (C1), lit_evaluation_metrics (§4)
- Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures. ECCV 2024. arXiv:2404.03010. — lit_topology_merge_losses (C2), lit_evaluation_metrics (§4)
- The Centerline-Cross Entropy Loss for Vessel-Like Structure Segmentation. MICCAI 2024. papers.miccai.org/miccai-2024/770-Paper1081. — lit_topology_merge_losses (C3)
- Topology-Preserving Deep Image Segmentation (TopoLoss). NeurIPS 2019. arXiv:1906.05404. — lit_topology_merge_losses (C6/B3), lit_evaluation_metrics (§3)
- Structure-Aware Image Segmentation with Homotopy Warping. NeurIPS 2022. arXiv:2112.07812. — lit_topology_merge_losses (C4)
- Topology-Aware Segmentation Using Discrete Morse Theory. ICLR 2021 (spotlight). arXiv:2103.09992. — lit_topology_merge_losses (C5)
- Learning Topological Interactions for Multi-Class Medical Image Segmentation. ECCV 2022 (oral). arXiv:2207.09654. — lit_topology_merge_losses (C7)
- ContextLoss: Context Information for Topology-Preserving Segmentation. 2025; no venue stated. arXiv:2506.11134. — lit_topology_merge_losses (C8; "not deeply reviewed")
- Topology-Guaranteed Image Segmentation: Enforcing Connectivity, Genus, and Width Constraints. 2026; no venue stated. arXiv:2601.11409. — lit_topology_merge_losses (C9/V-b; cost profile "unconfirmed")
- TopoSculpt: Betti-Steered Topological Sculpting of 3D Fine-grained Tubular Shapes. 2025; no venue stated. arXiv:2509.03938. — lit_topology_merge_losses (C10; "not deeply reviewed")
- Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes (Betti Matching). ICML 2023, PMLR 202:32698-32727. arXiv:2211.15272. — lit_topology_merge_losses (B1), lit_evaluation_metrics (§3)
- Efficient Betti Matching Enables Topology-Aware 3D Segmentation via Persistent Homology. 2024; no venue stated. arXiv:2407.04683. — lit_topology_merge_losses (B2; wall-clock numbers "not pulled here"), lit_evaluation_metrics (§3)
- A Topological Loss Function for Deep-Learning based Image Segmentation using Persistent Homology. TPAMI 2020/2022. arXiv:2107.12689 (multi-class variant arXiv:2008.09585). — lit_topology_merge_losses (B4)
- Topology Optimization in Medical Image Segmentation with Fast Euler Characteristic. 2025, IEEE TMI. arXiv:2507.23763. — lit_topology_merge_losses (B5)
- Differentiable Euler Characteristic Transform for Shape Classification. 2023; no venue stated. arXiv:2310.07630. — lit_topology_merge_losses (B6)
- Topology-Preserving Image Segmentation with Spatial-Aware Persistent Feature Matching. 2024; no venue stated. arXiv:2412.02076. — lit_topology_merge_losses (B7; "not deeply reviewed")
- Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function. 2025; no venue stated. arXiv:2501.01022. — lit_topology_merge_losses (B8/V-c)
- DconnNet: Directional Connectivity-based Segmentation. CVPR 2023. arXiv:2304.00145. — lit_topology_merge_losses (V-a)
- VascuConNet. Medical & Biological Engineering & Computing, 2024. doi 10.1007/s11517-024-03150-8. — lit_topology_merge_losses (sources list)
- COp-Net: Deep Contour Closing Operator. 2024/2025; no venue stated. arXiv:2407.15817. — lit_topology_merge_losses (M-a)
- Seg2Link. Scientific Reports 2023. doi:10.1038/s41598-023-34232-6. — lit_topology_merge_losses (M-b)

### EM connectomics: affinities, instance separation, run-length metrics

- Maximin Affinity Learning of Image Segmentation (MALIS). NeurIPS 2009; structured-loss form: Large Scale Image Segmentation with Structured Loss based Deep Learning for Connectome Reconstruction, TPAMI 2018. arXiv:1709.02974. — lit_topology_merge_losses (E1), lit_layered_structures (§6) [unverified in lit_layered_structures]
- Local Shape Descriptors for Neuron Segmentation. Nature Methods 2023. doi:10.1038/s41592-022-01711-z. — lit_topology_merge_losses (E2), lit_layered_structures (§6) [unverified in lit_layered_structures]
- The Mutex Watershed: Efficient, Parameter-Free Image Partitioning. ECCV 2018; theory arXiv:1904.12654; Semantic Mutex Watershed 2020 (Springer LNCS); GASP arXiv:1906.11713. — lit_topology_merge_losses (E3), lit_layered_structures (§6) [unverified in lit_layered_structures]
- Semantic Instance Segmentation with a Discriminative Loss Function. CVPR-W 2017. arXiv:1708.02551. — lit_topology_merge_losses (E4), lit_layered_structures (§6) [unverified in lit_layered_structures]
- Convolutional networks can learn to generate affinity graphs for image segmentation. Neural Computation 22, 2010. — lit_layered_structures (§6) [unverified]
- Superhuman accuracy on the SNEMI3D connectomics challenge. 2017. arXiv:1706.00120. — lit_layered_structures (§6) [unverified], lit_noisy_labels_self_training (b) [unverified]
- Instance segmentation by jointly optimizing spatial embeddings and clustering bandwidth. CVPR 2019. — lit_layered_structures (§6) [unverified]
- Deep watershed transform for instance segmentation. CVPR 2017. — lit_layered_structures (§6) [unverified]
- High-precision automated reconstruction of neurons with flood-filling networks. Nature Methods 2018; arXiv:1612.02120, arXiv:1905.06236. — lit_evaluation_metrics (§5), lit_noisy_labels_self_training (b) [unverified]
- Google AI Blog, "Improving Connectomics by an Order of Magnitude", 2018. — lit_evaluation_metrics (§5)

### Implicit surfaces, distance fields, Eikonal

- DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation. CVPR 2019. arXiv:1901.05103. — lit_implicit_surfaces_manifold (a)
- Implicit Geometric Regularization for Learning Shapes (IGR). ICML 2020. arXiv:2002.10099. — lit_implicit_surfaces_manifold (a)
- Implicit Neural Representations with Periodic Activation Functions (SIREN). NeurIPS 2020. arXiv:2006.09661. — lit_implicit_surfaces_manifold (a)
- Neural Unsigned Distance Fields for Implicit Function Learning (NDF). NeurIPS 2020. arXiv:2010.13938. — lit_implicit_surfaces_manifold (a)
- NUDF: Neural Unsigned Distance Fields for high resolution 3D medical image segmentation. ISBI 2022. arXiv:2504.18344. — lit_implicit_surfaces_manifold (a)
- GeoUDF: Surface Reconstruction from 3D Point Clouds via Geometry-guided Distance Representation. ICCV 2023; no arXiv id given. — lit_implicit_surfaces_manifold (a)
- EikoNet: Solving the Eikonal Equation with Deep Neural Networks. arXiv:2004.00361; no venue stated. — lit_implicit_surfaces_manifold (a)
- Deep Eikonal Solvers. arXiv:1903.07973; no venue stated. — lit_implicit_surfaces_manifold (a)
- Neural Dual Contouring. SIGGRAPH 2022. arXiv 2202.01999. — lit_surface_extraction (§2)
- DCUDF2: zero-level-set extraction from unsigned distance fields. 2024; no venue stated. arXiv 2408.17284. — lit_surface_extraction (§2)
- MIND: non-manifold material-interface extraction from a UDF. 2025; no venue stated. arXiv 2506.02938. — lit_surface_extraction (§2)
- DMTet: Deep Marching Tetrahedra. NeurIPS 2021. arXiv 2111.04276. — lit_surface_extraction (§8)
- FlexiCubes. TOG/SIGGRAPH 2023. arXiv 2308.05371. — lit_surface_extraction (§8)
- TetWeave. 2025; no venue stated. arXiv 2505.04590. — lit_surface_extraction (§8)
- Neural Shortest Path for Surface Reconstruction. 2025; no venue stated. arXiv 2502.06047. — lit_surface_extraction (§5)
- Screened Poisson Surface Reconstruction. Kazhdan & Hoppe, ACM TOG 2013. — lit_surface_extraction (§3)
- Marching cubes lineage: topologically-correct MC33 (Nielson & Hamann 1991); Flying Edges (Schroeder et al. 2015); Dual Contouring (Ju et al. 2002). — lit_surface_extraction (§2)
- Topological thinning: Lee-Kashyap-Chu 1994; Németh & Palágyi ~2015 (approximate year given). — lit_surface_extraction (§1) [unverified]
- Structural MAT. 2026; no venue stated. arXiv 2605.02302. — lit_surface_extraction (§1)
- Instant Self-Intersection Repair. ACM TOG 2025. — lit_surface_extraction (§4)
- Liepa hole filling, 2003; topology-graph repair, J. Computational Design & Engineering 2021. — lit_surface_extraction (§4)
- Tensor voting: Medioni, Tang & Lee; Mordohai & Medioni survey; no year or venue given. — lit_surface_extraction (§5) [unverified]
- DiffComplete (NeurIPS 2023); Diffusion-SDF (2022); SC-Diff (2024). — lit_surface_extraction (§5)
- Boykov-Jolly volumetric graph cuts; Golovinskiy & Funkhouser min-cut point-cloud segmentation; no years given. — lit_surface_extraction (§6) [unverified]
- Diffusion-Driven Inter-Outer Surface Separation for Point Clouds with Open Boundaries. 2026; no venue stated. arXiv 2602.00739. — lit_surface_extraction (§6)
- Segmentation-Driven Feature-Preserving Mesh Denoising. 2020. arXiv 2008.01358. — lit_surface_extraction (§7)
- Homogeneous-MLS anisotropic filtering. 2019. arXiv 1912.10194. — lit_surface_extraction (§7)

### Nested-surface / cortical reconstruction

- DeepCSR: A 3D Deep Learning Approach for Cortical Surface Reconstruction. WACV 2021. arXiv:2010.11423. — lit_implicit_surfaces_manifold (c), lit_surface_extraction (§4)
- Vox2Cortex. CVPR 2022 per the doc's bibliography; the same doc's table gives MICCAI 2022. researchgate.net/publication/363906781. — lit_implicit_surfaces_manifold (c) [unverified venue]
- TopoFit: Rapid Reconstruction of Topologically-Correct Cortical Surfaces. 2023; no venue stated. researchgate.net/publication/370982335. — lit_implicit_surfaces_manifold (c)
- Coupled Reconstruction of Cortical Surfaces by Diffeomorphic Mesh Deformation (SurfNet lineage). NeurIPS 2023. proceedings.neurips.cc/…/ff0da832a110c6537e885cdfbac80a94; PMC11149912. — lit_implicit_surfaces_manifold (c)
- CortexODE: Learning Cortical Surface Reconstruction by Neural ODEs. IEEE TMI / MICCAI lineage 2022. arXiv:2202.08329. — lit_implicit_surfaces_manifold (c), lit_evaluation_metrics (§8), lit_surface_extraction (§4)
- PialNN: A Fast Deep Learning Framework for Cortical Pial Surface Reconstruction. 2021 (MICCAI workshop). arXiv:2109.03693. — lit_implicit_surfaces_manifold (c)
- CorticalFlow++. MICCAI 2022. — lit_evaluation_metrics (§8), lit_surface_extraction (§4)
- SegRecon: Learning joint surface reconstruction and segmentation, from brain images to cortical surface parcellation. Medical Image Analysis 2024. sciencedirect.com/…/S1361841523002347. — lit_implicit_surfaces_manifold (b)
- Improved Segmentation of Deep Sulci in Cortical Surfaces (Laplace-constrained laminar segmentation). 2023; no venue stated. arXiv:2303.00795. — lit_topology_merge_losses (L-e)
- Three-dimensional mapping of cortical thickness using Laplace's equation. Human Brain Mapping 11, 2000. — lit_layered_structures (§5) [unverified]
- Anatomically motivated modeling of cortical laminae (equivolume layering). NeuroImage 93, 2014. — lit_layered_structures (§5) [unverified]

### Layered media: OCT, seismic, tree rings, battery, phase unwrapping

- Order-constrained retinal layer regression. Scientific Reports 2023. doi:10.1038/s41598-023-35230-4. — lit_topology_merge_losses (L-a)
- Differentiable Dynamic Programming for OCT Surface Segmentation. 2022/2023. arXiv:2210.06335. — lit_topology_merge_losses (L-b)
- Globally optimal segmentation of mutually interacting surfaces using deep learning. arXiv 2007.01259 (2020); Optics Express 2022 as "Globally optimal OCT surface segmentation using a constrained IPM optimization". — lit_layered_structures (§2) [unverified]
- Assignment Flow for Order-Constrained OCT Segmentation. 2020; no venue stated. arXiv:2009.04632. — lit_topology_merge_losses (L-c)
- Uncertainty-aware retinal layer segmentation in OCT through probabilistic signed distance functions. 2024; no venue stated. arXiv:2412.04935. — lit_topology_merge_losses (L-d), lit_implicit_surfaces_manifold (d), lit_layered_structures (§2)
- Deep learning network with differentiable dynamic programming for retina OCT surface segmentation. 2023; no venue stated. PubMed 37497505. — lit_implicit_surfaces_manifold (d)
- Topology guaranteed segmentation of the human retina from OCT using convolutional neural networks. arXiv 1803.05120 (2018); extension: Structured layer surface segmentation for retina OCT using fully convolutional regression networks, Medical Image Analysis 68, 2021. — lit_layered_structures (§2) [unverified for the 2021 extension]
- Automated 3-D intraretinal layer segmentation of macular spectral-domain OCT images (the "Iowa" multi-surface graph search). IEEE TMI 28, 2009. doi:10.1109/TMI.2009.2016958. — lit_layered_structures (§2)
- Automatic segmentation of OCT retinal boundaries using recurrent neural networks and graph search. Biomedical Optics Express 9, 2018. doi:10.1364/BOE.9.005759. — lit_layered_structures (§2)
- Simultaneous alignment and surface regression using hybrid 2D-3D networks for 3D coherent layer segmentation of retina OCT images. MICCAI 2021, arXiv 2203.02390; MedIA extension arXiv 2312.01726. — lit_layered_structures (§2)
- SD-LayerNet. MICCAI 2022. arXiv 2207.00458. — lit_layered_structures (§2)
- ReLayNet. Biomedical Optics Express 8, 2017. arXiv 1704.02161. — lit_layered_structures (§2)
- Deep Relative Geologic Time: A Deep Learning Method for Simultaneously Interpreting 3-D Seismic Horizons and Faults. JGR Solid Earth 126, 2021. doi:10.1029/2021JB021882; code zfbi/rgtNet. — lit_topology_merge_losses (L-f), lit_implicit_surfaces_manifold (d), lit_layered_structures (§3)
- Learning Stratigraphically Consistent Relative Geologic Time from 3D Seismic Data via Sinusoidal Mapping (RGT-Est). 2026 preprint. arXiv:2605.01273. — lit_topology_merge_losses (L-f), lit_implicit_surfaces_manifold (d), lit_layered_structures (§3; id "as reported by search, not opened") [unverified]
- Relative geologic time (age) volumes. The Leading Edge 23, 2004. — lit_layered_structures (§3) [unverified]
- Generating a relative geologic time volume by 3D graph-cut phase unwrapping. Geophysics 77, 2012. — lit_layered_structures (§3) [unverified]
- Horizon volumes with interpreted constraints. Geophysics 80, 2015. — lit_layered_structures (§3) [unverified]
- Least-squares horizons with local slopes and multigrid correlations. Geophysics 83, 2018. — lit_layered_structures (§3) [unverified]
- Deep learning for relative geologic time and seismic horizons. Geophysics 85, 2020. doi:10.1190/geo2019-0252.1. — lit_layered_structures (§3)
- Fully reversible neural networks for large-scale 3D seismic horizon tracking. arXiv 2003.08466; no venue stated. — lit_layered_structures (§3) [unverified]
- A Deep Learning-Based Seismic Horizon Tracking Method With Uncertainty Encoding and Vertical Constraint. IEEE TGRS 2024. ieeexplore.ieee.org/document/10587310. — lit_topology_merge_losses (L-g)
- Iterative next boundary detection for instance segmentation of tree rings in microscopy images. CVPR 2023. — lit_layered_structures (§4) [unverified], synthesis_v2_with_literature (I5)
- CS-TRD: a cross-sections tree ring detection method. arXiv 2305.10809 (2023); DeepCS-TRD (2025). — lit_layered_structures (§4) [unverified]
- Tree ring segmentation performance in highly disturbed trees using deep learning. PLOS ONE 2026. — lit_layered_structures (§4)
- Automated 3D tree-ring detection and measurement from X-ray CT. Dendrochronologia 2021. — lit_layered_structures (§4) [unverified]
- Virtual unrolling of spirally-wound lithium-ion cells for correlative degradation studies and predictive current distribution modelling. Sustainable Energy & Fuels 3, 2019. — lit_layered_structures (§1) [unverified]
- 4D imaging of lithium-batteries using correlative neutron and X-ray tomography with a virtual unrolling technique. Nature Communications 11, 2020. — lit_layered_structures (§1) [unverified]
- Long-term cycling induced jelly roll deformation in commercial 18650 cells. J. Power Sources 392, 2018. — lit_layered_structures (§1) [unverified]
- Coupling X-ray computed tomography with digital volume correlation to study core collapse in lithium-ion batteries. EES Batteries 2026. doi:10.1039/d5eb00229j. — lit_layered_structures (§1)
- BS-Mamba: a battery-specific Mamba network for robust battery electrode CT image segmentation. Measurement 2026. doi:10.1016/j.measurement.2025.119496. — lit_layered_structures (§1) [unverified, abstract paywalled]
- Two-dimensional phase unwrapping: theory, algorithms and software. Wiley 1998. — lit_layered_structures (§5) [unverified]
- PhaseNet: a deep convolutional neural network for two-dimensional phase unwrapping. IEEE SPL 26, 2019. — lit_layered_structures (§5) [unverified]
- One-step robust deep learning phase unwrapping. Optics Express 27, 2019. — lit_layered_structures (§5) [unverified]
- Rank consistent ordinal regression for neural networks with application to age estimation (CORAL). Pattern Recognition Letters 140, 2020. — lit_layered_structures (§6) [unverified]
- Deep neural networks for rank-consistent ordinal regression based on conditional probabilities (CORN). Pattern Analysis and Applications 2023. — lit_layered_structures (§6) [unverified]
- Unconstrained monotonic neural networks. NeurIPS 2019. — lit_layered_structures (§6) [unverified]

### Virtual unrolling of rolled objects (non-Herculaneum) and papyrus CT

- Fully Automatic Virtual Unwrapping Method for Documents Imaged by X-Ray Tomography. ICDAR 2024. dl.acm.org/doi/10.1007/978-3-031-70543-4_14. — lit_implicit_surfaces_manifold (d)
- Virtual unrolling for analysing rolled objects (industrial CT, flexible PCB / rolled microelectronics). DTU 3D Industry Portal case study; no year given. — lit_implicit_surfaces_manifold (d)
- Revealing text in a complexly rolled silver scroll from Jerash with computed tomography and advanced imaging software. Scientific Reports 2015. nature.com/articles/srep17765. — lit_implicit_surfaces_manifold (d)
- A computational platform for the virtual unfolding of Herculaneum Papyri. Scientific Reports 2021. PMC7813886. — lit_fibre_orientation (§3)
- From invisibility to readability: Recovering the ink of Herculaneum. PLOS ONE; no year given. doi 10.1371/journal.pone.0215775. — lit_fibre_orientation (§3)

### Noisy labels, self-training, distillation, cascades and exposure bias

All entries in this group are flagged **[unverified]** by the document-level caveat in
`lit_noisy_labels_self_training`.

- Generalized Cross Entropy (GCE). NeurIPS 2018. arXiv:1805.07836. [unverified]
- Symmetric Cross Entropy (SCE). ICCV 2019. arXiv:1908.06112. [unverified]
- Bootstrapping (Reed et al. 2015, arXiv:1412.6596); dynamic bootstrapping (Arazo et al., ICML 2019, arXiv:1904.11238). [unverified]
- Early-Learning Regularization (ELR). NeurIPS 2020. arXiv:2007.00151. [unverified]
- Normalized-loss framework (APL, NCE+RCE). ICML 2020; no arXiv id given. [unverified]
- Co-teaching. NeurIPS 2018. arXiv:1804.06872; segmentation variant arXiv:2104.13766. [unverified]
- JoCoR. CVPR 2020. arXiv:2003.02752. [unverified]
- Confident Learning / cleanlab. JAIR 2021. arXiv:1911.00068. [unverified]
- Mean-Teacher-Assisted Confident Learning. 2022. PubMed 35604969. [unverified]
- Adaptive Label Correction (ALC). 2025. arXiv:2503.12218. [unverified]
- GSD-Net (geometric-structural dual guidance). 2025. arXiv:2509.02419. [unverified]
- AIO2. 2024. arXiv:2403.01641. [unverified]
- Noisy Student. CVPR 2020. arXiv:1911.04252. [unverified]
- Born-Again Networks. ICML 2018. arXiv:1805.04770. [unverified]
- UA-MT. MICCAI 2019; no arXiv id given. [unverified]
- Double-uncertainty dual mean-teacher. arXiv:2303.05126; no venue stated. [unverified]
- Cross-Teaching 3D↔2D. MICCAI 2023. arXiv:2307.16256. [unverified]
- FPL+. arXiv:2404.04971; no venue stated. [unverified]
- SRPL-SFDA. 2025. arXiv:2506.09403. [unverified]
- nnFilterMatch. 2025. arXiv:2509.19746. [unverified] — also cited by lit_uncertainty_active_labelling (§7)
- Reliable-pseudo-label co-training. arXiv:2301.04465; no venue stated. [unverified]
- Confident Learning for noisy segmentation labels. MICCAI 2020. [unverified]
- nnU-Net "3D U-Net Cascade". Nature Methods 2021; KiTS21 arXiv:2307.01984. [unverified] — also cited by lit_scaling_multiscale_generalisation
- Sparse-annotation bootstrapping (EM). PMC11195258; no venue or year given. [unverified]
- TTA-based active learning + self-training. arXiv:2308.10727; no venue stated. [unverified]
- TTA aleatoric uncertainty. Neurocomputing 2019. arXiv:1807.07356. [unverified] — also cited by lit_uncertainty_active_labelling (§5)
- Distilling the Knowledge in a Neural Network (Hinton KD). 2015. arXiv:1503.02531. [unverified]
- Structured Knowledge Distillation for Semantic Segmentation. CVPR 2019; no arXiv id given. [unverified]
- Structural & statistical texture knowledge distillation. arXiv:2305.03944; no venue stated. [unverified]
- Multi-modal → mono-modal KD. arXiv:2106.09564; no venue stated. [unverified]
- Adaptive/sample-wise temperature KD. 2026 preprint. arXiv:2605.20357. [unverified]
- Ensemble-then-distill multi-teacher KD. Fukuda & Suzuki, Interspeech 2017; "Multi-Teacher Distillation: Ensemble-Then-Distill", NeurIPS 2024. [unverified]
- Weighted ensemble of teaching assistants. arXiv:2206.12005; no venue stated. [unverified]
- Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation (multi-view theory). ICLR 2023. arXiv:2012.09816; related arXiv:2009.04120. [unverified]
- Does Knowledge Distillation Really Work? NeurIPS 2021; no arXiv id given. [unverified]
- Cascaded 3D FCN for organ segmentation. Medical Image Analysis 2019. arXiv:1803.05431. [unverified]
- CascadePSP. CVPR 2020. arXiv:2005.02551. [unverified]
- Recurrent iterative refinement / feedback networks. arXiv:1811.08043; arXiv:1705.07238; no venues stated. [unverified]
- Deep recurrence / predictive-coding feedback (divergence without damping). 2025. arXiv:2507.10143. [unverified]
- Reviving Iterative Training with Mask Guidance (RITM). arXiv:2102.06583; no venue stated. [unverified]
- Scheduled Sampling. NeurIPS 2015; no arXiv id given. [unverified]
- DAgger. AISTATS 2011; no arXiv id given. [unverified]
- Professor Forcing. NeurIPS 2016; no arXiv id given. [unverified]
- OneSeg. arXiv:2309.13671; no venue stated. [unverified]
- Mean Teacher. NeurIPS 2017. arXiv:1703.01780. [unverified]
- Hierarchical consistency-regularized mean teacher. arXiv:2105.10369; no venue stated. [unverified]
- Cross-Consistency Training (CCT). CVPR 2020. arXiv:2003.09005. [unverified]
- Unsupervised Data Augmentation (UDA). NeurIPS 2020. arXiv:1904.12848. [unverified]
- CertainTTA / budget-aware nnU-Net uncertainty. ScienceDirect S1566253525003732; arXiv:2604.11798. [unverified]
- Learning to Reweight Examples (L2RW). ICML 2018. arXiv:1803.09050. [unverified]
- MentorNet. ICML 2018. arXiv:1712.05055. [unverified]
- Meta-Weight-Net. NeurIPS 2019; CMW-Net arXiv:2202.05613. [unverified]
- Gold Loss Correction (GLC). NeurIPS 2018. arXiv:1802.05300. [unverified]
- Adaptive Early-Learning Correction for Segmentation. CVPR 2022. arXiv:2110.03740. [unverified]
- Walking on Two Legs (label correction + reweighting for noisy segmentation). ACML 2020. [unverified]
- High-quality pseudo masks from noisy/weak annotations. 2024. PubMed 39520897. [unverified]
- ScaleBiO (first-order bilevel data reweighting). 2024/2025; no venue or id given. [unverified]

### Uncertainty, calibration, active learning, interactive annotation

- Dropout as a Bayesian Approximation. Gal & Ghahramani, 2016; no venue or id given. — lit_uncertainty_active_labelling (§1)
- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017. — lit_uncertainty_active_labelling (§2)
- Uncertainty Quantification in Medical Image Segmentation: A Comprehensive Survey. PMC13514988; no year given. — lit_uncertainty_active_labelling
- Evaluating Uncertainty Quantification in Medical Image Segmentation: A Multi-Dataset, Multi-Algorithm Study. doi 10.3390/app142110020. — lit_uncertainty_active_labelling
- Uncertainty quantification and segmentation, Bayesian deep learning. Communications Medicine 2024. — lit_uncertainty_active_labelling
- Snapshot Ensembles: Train 1, Get M for Free. ICLR 2017. — lit_uncertainty_active_labelling (§3)
- Reliable uncertainty with cheaper neural network ensembles. arXiv:2403.10182; no venue stated. — lit_uncertainty_active_labelling (§3)
- Evidential Deep Learning to Quantify Classification Uncertainty. NeurIPS 2018. arXiv:1806.01768. — lit_uncertainty_active_labelling (§4)
- A Comprehensive Survey on Evidential Deep Learning. arXiv:2409.04720; no venue stated. — lit_uncertainty_active_labelling (§4)
- DuEDL: Dual-Branch Evidential Deep Learning for Scribble-Supervised Medical Image Segmentation. arXiv:2405.14444; no venue stated. — lit_uncertainty_active_labelling (§4, §9)
- Test-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty. MIDL 2018. — lit_uncertainty_active_labelling (§5)
- BayTTA: Uncertainty-aware medical image classification with optimized test-time augmentation. arXiv:2406.17640; no venue stated. — lit_uncertainty_active_labelling (§5)
- On Calibration of Modern Neural Networks. ICML 2017. — lit_uncertainty_active_labelling (§6)
- Local Temperature Scaling for Probability Calibration. ICCV 2021. arXiv:2008.05105. — lit_uncertainty_active_labelling (§6)
- Rethinking Post-Hoc Calibration in Semantic Segmentation. arXiv:2607.01902; no venue stated. — lit_uncertainty_active_labelling (§6)
- Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation. arXiv:2506.03942; no venue stated. — lit_uncertainty_active_labelling (§6)
- Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation. IEEE TMI 2020. arXiv:1911.13273. — lit_uncertainty_active_labelling (§6)
- Dual uncertainty-guided multi-model pseudo-label learning (DUMM). aimspress mbe.2024097. — lit_uncertainty_active_labelling (§7)
- Uncertainty-Guided Cross Attention Ensemble Mean Teacher. arXiv:2412.15380; no venue stated. — lit_uncertainty_active_labelling (§7)
- nnActive: A Framework for Evaluation of Active Learning in 3D Biomedical Segmentation. Nov 2025. arXiv:2511.19183. — lit_uncertainty_active_labelling (§8)
- Integrating Deep Metric Learning with Coreset for Active Learning in 3D Segmentation. arXiv:2411.15763; no venue stated. — lit_uncertainty_active_labelling (§8)
- Dataset-Aware Cold-Start Active Learning for Annotation-Efficient 3D Medical Image Segmentation (CSCS). arXiv:2606.20765; no venue stated. — lit_uncertainty_active_labelling (§8)
- Active Learning for Convolutional Neural Networks: A Core-Set Approach. ICLR 2018. — lit_uncertainty_active_labelling (§8)
- Scribble2D5. MICCAI 2022. arXiv:2205.06779. — lit_uncertainty_active_labelling (§9)
- Volumetric Medical Image Segmentation via Scribble Annotations and Shape Priors. arXiv:2310.08084; no venue stated. — lit_uncertainty_active_labelling (§9)
- SAM-Med3D. ECCV 2024. arXiv:2310.15161. — lit_uncertainty_active_labelling (§9), lit_pretraining_foundation (§2)
- 3DSAM-adapter. Medical Image Analysis; no year given. sciencedirect S1361841524002494. — lit_uncertainty_active_labelling (§9)
- ProtoSAM-3D. sciencedirect S0895611125000102; no venue name given. — lit_uncertainty_active_labelling (§9)
- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? NeurIPS 2017. — lit_uncertainty_active_labelling (§10)
- Uncertainty- and hardness-weighted loss functions for medical image segmentation. PMC12691699; no year given. — lit_uncertainty_active_labelling (§10)
- Uncertainty is not sufficient for identifying noisy labels in training data for binary segmentation of building footprints. Frontiers in Remote Sensing 2022. frsen.2022.1100012. — lit_uncertainty_active_labelling (§10)

### Scaling, architectures, scale/metadata conditioning, domain generalisation

- Training Compute-Optimal Large Language Models (Chinchilla). 2022; referenced only to state that no 3D-segmentation equivalent exists. — lit_scaling_multiscale_generalisation (a)
- STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training. 2023. arXiv:2304.06716. — lit_scaling_multiscale_generalisation (a)
- Revisiting model scaling with a U-Net benchmark for 3D medical image segmentation. Scientific Reports 2025 / PubMed 40813440. — lit_scaling_multiscale_generalisation (a)
- Scaling nnU-Net for CBCT Segmentation. 2024. arXiv:2411.17213. — lit_scaling_multiscale_generalisation (a)
- MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation. MICCAI 2023. arXiv:2303.09975. — lit_scaling_multiscale_generalisation (a)
- MedNeXt-v2. 2025. arXiv:2512.17774. — lit_scaling_multiscale_generalisation (a)
- SegVol: Universal and Interactive Volumetric Medical Image Segmentation. NeurIPS 2024. arXiv:2311.13385. — lit_scaling_multiscale_generalisation (a), lit_pretraining_foundation (§2)
- BioVFM-21M. 2025. arXiv:2505.09329. — lit_scaling_multiscale_generalisation (a)
- HyperSpace: Hypernetworks for spacing-adaptive image segmentation. MICCAI 2024. arXiv:2407.03681. — lit_scaling_multiscale_generalisation (b)
- A comprehensive framework for automated segmentation of perivascular spaces in brain MRI with the nnU-Net. 2024/2026. arXiv:2411.19564. — lit_scaling_multiscale_generalisation (b)
- Scale-Equivariant Deep Learning for 3D Data. 2023. arXiv:2304.05864. — lit_scaling_multiscale_generalisation (b)
- Truly Scale-Equivariant Deep Nets with Fourier Layers. NeurIPS 2023. arXiv:2311.02922. — lit_scaling_multiscale_generalisation (b)
- FiLM (Perez et al. 2018); no id given, cited as the general conditioning mechanism. — lit_scaling_multiscale_generalisation (b)
- CoordConv (Liu et al. 2018); no id given, noted as outside the 2021-2026 window but foundational. — lit_scaling_multiscale_generalisation (b)
- SegMamba. MICCAI 2024. papers.miccai.org/miccai-2024/676-Paper0663. — lit_scaling_multiscale_generalisation (c)
- SegMamba-V2. 2025. PubMed 40679879. — lit_scaling_multiscale_generalisation (c)
- nnMamba. 2024. arXiv:2402.03526. — lit_scaling_multiscale_generalisation (c)
- nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. MICCAI 2024. doi 10.1007/978-3-031-72114-4_47. — lit_scaling_multiscale_generalisation (c), synthesis_v2_with_literature (§3.5)
- Learning Multimodal Volumetric Features for Large-Scale Neuron Tracing. 2024. arXiv:2401.03043. Recorded as a negative/non-match result. — lit_scaling_multiscale_generalisation (c)
- DG-TTA: Out-of-domain Medical Image Segmentation through Augmentation and Descriptor-driven Domain Generalization and Test-Time Adaptation. arXiv:2312.06275; no venue stated. — lit_scaling_multiscale_generalisation (d)
- BucketAugment: Reinforced Domain Generalisation in Abdominal CT Segmentation. PubMed 38899027. — lit_scaling_multiscale_generalisation (d)
- Colormap augmentation: a novel method for cross-modality domain generalization. PMC13035743. — lit_scaling_multiscale_generalisation (d)

### CT physics, augmentation, synthetic and procedural training data

- SinoSynth: A Physics-based Domain Randomization Approach for Generalizable CBCT Image Enhancement. MICCAI 2024. arXiv:2409.18355; PMC12711319. — lit_ct_physics_augmentation (§1, §8)
- Generalizable Cone Beam CT Esophagus Segmentation Using Physics-Based Data Augmentation. arXiv:2006.15713; ~2020, no venue stated. — lit_ct_physics_augmentation (§8)
- Deep Learning with Domain Randomization in Image and Feature Spaces for Abdominal Multiorgan Segmentation. Radiology: AI 2026. PMC12476582. — lit_ct_physics_augmentation (§1)
- Physics-informed data augmentation to simulate low dose CT scans: Application to lung nodule detection. 2025. PMC13366036. — lit_ct_physics_augmentation (§1), lit_scaling_multiscale_generalisation (d)
- One Sequence to Segment Them All: Efficient Data Augmentation for CT and MRI Cross-Domain 3D Spine Segmentation. 2026. arXiv:2605.03098. — lit_ct_physics_augmentation (§1)
- On optimisation of Paganin's method for propagation-based X-ray phase-contrast imaging and tomography. 2026. arXiv:2601.07225. — lit_ct_physics_augmentation (§4)
- Development of a deep learning method for phase retrieval image enhancement in phase contrast microcomputed tomography. Journal of Microscopy 2025. PMC12265864. — lit_ct_physics_augmentation (§4)
- Investigating the robustness of a learning-based method for quantitative phase retrieval from propagation-based x-ray phase contrast measurements under laboratory conditions. arXiv:2211.01372; no venue stated. — lit_ct_physics_augmentation (§4)
- Ring artifacts correction method in x-ray computed tomography based on stripe classification and removal in sinogram images. 2025. arXiv:2505.19513. — lit_ct_physics_augmentation (§3)
- CBCT dental beam-hardening correction. arXiv:2010.03778; no venue or title given beyond the description. — lit_ct_physics_augmentation (§2)
- Paganin et al. 2002, original phase-retrieval method; no id given, noted as outside the requested window but foundational. — lit_ct_physics_augmentation (§4)
- Griem, Koeppe, Greß, Feser, Nestler, "Synthetic training data for CT image segmentation of microstructures". 2025; venue given as "Computational Materials Science (or similar Elsevier venue)". ScienceDirect S1359645425005075; SSRN 5087564; elib.dlr.de/215032. — lit_ct_physics_augmentation (§6) [unverified venue]
- Synthetic, automatically labelled training data for machine learning based X-ray CT image segmentation: Application to 3D-textile carbon fibre reinforced composites. 2025; venue given as "Composites Part A (or similar)". ScienceDirect S1359836825005578. — lit_ct_physics_augmentation (§6) [unverified venue; paper paywalled, noise model not confirmed]
- Training Generalized Segmentation Networks with Real and Synthetic Cryo-ET Data. bioRxiv 2025.01.31. PMC11838407. — lit_ct_physics_augmentation (§6)
- CryoGEM: Physics-Informed Generative Cryo-Electron Microscopy. 2024/2025; no venue or id given. — lit_ct_physics_augmentation (§5)
- MosaicNet: A deep-learning-based multi-tile biomedical image stitching method. PubMed 38082798. — lit_ct_physics_augmentation (§7)
- UnMICST: Deep learning with real augmentation for robust segmentation of highly multiplexed images of human tissues. Communications Biology 2022. — lit_ct_physics_augmentation (§7)

### Fibre orientation, structure tensor, orientation regression

- Bigün & Granlund, structure tensor (1987); Knutsson (1989). Cited via the Wikipedia structure-tensor summary and the pi2/OrientationJ documentation rather than the papers. — lit_fibre_orientation (§1) [unverified]
- Software: structure-tensor (PyPI, CuPy); fiberorient (GitHub); OrientationJ / OrientationPy (EPFL BIG); pi2 orientation documentation. — lit_fibre_orientation (§1)
- Cardiotensor: teravoxel-scale structure-tensor and tractography. 2025. arXiv:2508.07476. — lit_fibre_orientation (§1)
- Micro-CT based structure tensor analysis of fibre orientation in random fibre composites versus high-fidelity fibre identification methods. ResearchGate 338035784; no venue or year given. — lit_fibre_orientation (§3)
- Individual fibre segmentation from 3D X-ray computed tomography for characterising the fibre orientation in unidirectional composite materials. Emerson et al. 2017. ResearchGate 312299326. — lit_fibre_orientation (§1, §3)
- Instance Segmentation of Fibers from Low Resolution CT via 3D Deep Embedding Learning. arXiv:1901.01034; no venue stated. — lit_fibre_orientation (§3)
- Identification and analysis of fibers in ultra-large micro-CT scans of nonwoven textiles using deep learning. Textile Research Journal 2022. — lit_fibre_orientation (§3)
- Thermal fiber orientation tensors for digital paper physics. ScienceDirect S0020768316302335; no year given. — lit_fibre_orientation (§3)
- Computational approaches for structural analysis of wood specimens. De Gruyter 2024. doi 10.1515/rams-2024-0073. — lit_fibre_orientation (§3)
- X-ray CT structure tensor orientation mapping for FE models (STXAE), Herrmann et al. ScienceDirect S2665963821000968, 2021. — lit_fibre_orientation (§1, §3)
- Robust FOD estimation using deep constrained spherical deconvolution. 2023. arXiv:2306.02900. — lit_fibre_orientation (§2)
- Equivariant spherical CNNs for FOD estimation in neonatal dMRI. 2025. arXiv:2504.01925; PMC12343732. — lit_fibre_orientation (§2)
- Constrained spherical deconvolution, Tournier et al. 2007; no id given. — lit_fibre_orientation (§2)
- Leveraging SO(3)-steerable convolutions for pose-robust semantic segmentation. PMC7617181; Weiler et al. 2018 cited as the steerable-filter pattern, no id given. — lit_fibre_orientation (§2)
- Structure Tensor Representation for Robust Oriented Object Detection. arXiv:2411.10497; no venue stated. — lit_fibre_orientation (§2)
- Sign and Basis Invariant Networks for Spectral Graph Representation Learning (SignNet). arXiv:2202.13013; no venue stated. — lit_fibre_orientation (§2)
- Task-based Loss Functions in Computer Vision. arXiv:2504.04242; no venue stated. Plus a loss-functions survey in Artificial Intelligence Review 2025, no id given. — lit_fibre_orientation (§2)
- Auxiliary Tasks in Multi-task Learning. arXiv:1805.06334; and arXiv:2412.19547; no venues stated. — lit_fibre_orientation (§4)
- Improving Vessel Segmentation with Multi-Task Learning and Auxiliary Data. 2025. arXiv:2509.03975. — lit_fibre_orientation (§4)
- Learning-enhanced 3D fiber orientation mapping in thick cardiac tissues. 2024/2025. PMC12339308. — lit_implicit_surfaces_manifold (b)
- Bingham / matrix-Fisher orientation-uncertainty loss, NeurIPS 2020 camera-pose work; no title or id given. — lit_fibre_orientation (§2) [unverified]

### Self-supervised pretraining and foundation models

- Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis (SwinUNETR-SSL). CVPR 2022. researchgate.net/publication/359507351. — lit_pretraining_foundation (§1)
- Revisiting MAE pre-training for 3D medical image segmentation. CVPR 2025. arXiv:2410.23132. — lit_pretraining_foundation (§1), synthesis_v2_with_literature (L11/R10)
- Hi-End-MAE: Hierarchical encoder-driven masked autoencoders. arXiv:2502.08347; no venue stated. — lit_pretraining_foundation (§1)
- Swin MAE: Masked Autoencoders for Small Datasets. arXiv:2212.13805; no venue stated. — lit_pretraining_foundation (§1)
- VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis. CVPR 2024. arXiv:2402.17300. — lit_pretraining_foundation (§1, §4)
- Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis. Medical Image Analysis 2021. — lit_pretraining_foundation (§1)
- Vision Foundation Models for Computed Tomography (CT-FM). arXiv:2501.09001; no venue stated. — lit_pretraining_foundation (§2)
- Merlin: a computed tomography vision-language foundation model and dataset. Nature 2026. arXiv:2406.06512. — lit_pretraining_foundation (§2)
- VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging. CVPR 2025. — lit_pretraining_foundation (§2)
- CEM500K, a large-scale heterogeneous unlabeled cellular electron microscopy image dataset for deep learning. eLife 2021. — lit_pretraining_foundation (§2)
- RETINA: Reconstruction-based pre-trained enhanced TransUNet for EM segmentation on CEM500K. PLOS Computational Biology 2025. doi 10.1371/journal.pcbi.1013115. — lit_pretraining_foundation (§2)
- Segment Anything for Microscopy (micro-SAM). Nature Methods 2024/2025. doi 10.1038/s41592-024-02580-4. — lit_pretraining_foundation (§2)
- VAMAE: Vessel-Aware Masked Autoencoders for OCT Angiography. arXiv:2604.06583; no venue stated. — lit_pretraining_foundation (§3)
- Benefit from public unlabeled data: A Frangi filter-based pretraining network for 3D cerebrovascular segmentation. Medical Image Analysis 2024. — lit_pretraining_foundation (§3)
- Mitigating Overfitting in Medical Imaging: Self-Supervised Pretraining vs. ImageNet Transfer Learning. arXiv:2505.16773; no venue stated. — lit_pretraining_foundation (§3)
- Transfer or Self-Supervised? Bridging the Performance Gap in Medical Imaging. arXiv:2407.05592; no venue stated. — lit_pretraining_foundation (§3)

### Optimisation: schedules, EMA, optimisers, normalisation, multi-task weighting

Sections 2-4 of `lit_optimisation_schedules` are flagged by that document as reconstructed without a live
re-check; entries drawn from them are marked [unverified].

- MiniCPM: Unveiling the Potential of Small Language Models (WSD schedule). 2024. arXiv:2404.06395. — lit_optimisation_schedules (§1, live-verified)
- Scaling Laws Beyond Fixed Training Durations. arXiv:2405.18392; author list "not independently re-verified this session". — lit_optimisation_schedules (§1) [unverified authors]
- The Road Less Scheduled (Schedule-Free AdamW). arXiv:2405.15682; github.com/facebookresearch/schedule_free. — lit_optimisation_schedules (§1, live-verified)
- Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training. arXiv:2507.09846. — lit_optimisation_schedules (§1, live-verified)
- Warmup-Stable-Decay scheduling survey. emergentmind.com topic page; not a paper. — lit_optimisation_schedules (§1)
- On Warm-Starting Neural Network Training. NeurIPS 2020. — lit_optimisation_schedules (§2) [unverified]
- Re-warming for continual pretraining, "commonly cited as Gupta et al., ~2023"; exact venue not re-verified. — lit_optimisation_schedules (§2) [unverified]
- EMA-profile analyses for diffusion training, "Karras et al.-adjacent work, ~2023-2024"; no title or id given. — lit_optimisation_schedules (§2) [unverified]
- Accurate, Large Minibatch SGD. 2017. arXiv:1706.02677. — lit_optimisation_schedules (§3) [unverified]
- An Empirical Model of Large-Batch Training. 2018. arXiv:1812.06162. — lit_optimisation_schedules (§3) [unverified]
- Keskar et al. 2017, sharp-vs-flat minima; no id given. — lit_optimisation_schedules (§3) [unverified]
- Muon (Keller Jordan, 2024-2025; NanoGPT-speedrun writeups); no formal citation given. — lit_optimisation_schedules (§3) [unverified]
- SOAP. 2024. arXiv:2409.11321. — lit_optimisation_schedules (§3) [unverified]
- Shampoo (Gupta et al. 2018); Scalable Second-Order Optimization for Deep Learning (Anil et al. 2020); no ids given. — lit_optimisation_schedules (§3) [unverified]
- Lion: Symbolic Discovery of Optimization Algorithms. 2023. arXiv:2302.06675. — lit_optimisation_schedules (§3) [unverified]
- Sophia. 2023. arXiv:2305.14342. — lit_optimisation_schedules (§3) [unverified]
- Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP). 2022. arXiv:2203.03466; github.com/microsoft/mup. — lit_optimisation_schedules (§4) [unverified]
- Multi-Task Learning Using Uncertainty to Weigh Losses. CVPR 2018. arXiv:1705.07115. — lit_implicit_surfaces_manifold (e), lit_optimisation_schedules (§4) [unverified in the latter]
- AHU-MultiNet: Adaptive loss balancing based on homoscedastic uncertainty in multi-task medical image segmentation network. 2022. sciencedirect S0010482522008654. — lit_implicit_surfaces_manifold (e)
- GradNorm. ICML 2018; no id given. — lit_optimisation_schedules (§4) [unverified]
- PCGrad. NeurIPS 2020; CAGrad/IMTL 2021-2022; no ids given. — lit_optimisation_schedules (§4) [unverified]
- In Defense of the Unitary Scalarization for Deep Multi-Task Learning. NeurIPS 2022. — lit_optimisation_schedules (§4) [unverified]
- Do Current Multi-Task Optimization Methods in Deep Learning Even Help? NeurIPS 2022. — lit_optimisation_schedules (§4) [unverified]
- Group Normalization. ECCV 2018. — lit_optimisation_schedules (§4) [unverified]
- Weight standardization / micro-batch training (Qiao et al. 2019); Big Transfer (Kolesnikov et al. 2020); no ids given. — lit_optimisation_schedules (§4) [unverified]
- Cross-task consistency: a distance-transform auxiliary decoder regularizing a mask decoder. "Fragment found via search, no single canonical paper isolated", 2021-2023 medical segmentation papers. — lit_implicit_surfaces_manifold (e) [unverified]

### Evaluation metrics and statistical practice

- Deep learning to achieve clinically applicable segmentation of head and neck anatomy for radiotherapy (surface Dice). 2018/2021. — lit_evaluation_metrics (§1)
- Metrics reloaded: pitfalls and recommendations for image analysis validation. Nature Methods 2024 (preprint 2022). metrics-reloaded.dkfz.de. — lit_evaluation_metrics (§1, §6, §7)
- STAPLE (Warfield et al.); no year or id given. — lit_evaluation_metrics (§6)
- IMA++ dermoscopy 2025; federated noisy-label benchmark 2026. No titles or ids given. — lit_evaluation_metrics (§6) [unverified]
- cbDice / cl-X-Dice family (caliber-weighted clDice variants); no ids given. — lit_evaluation_metrics (§4) [unverified]
- Deep Learning Scaling is Predictable, Empirically. 2017. arXiv:1712.00409. — lit_evaluation_metrics (§7)
- (Mis)Fitting Scaling Laws: A Survey of Scaling Law Fitting Techniques in Deep Learning. OpenReview, 2024-2025. — lit_evaluation_metrics (§7)
- OCT layer segmentation surveys reporting Mean Absolute Distance in µm; no specific citation given. — lit_evaluation_metrics (§8) [unverified]
