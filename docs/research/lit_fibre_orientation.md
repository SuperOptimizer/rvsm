# Literature survey: fibre/texture orientation and anisotropy, outside the Vesuvius Challenge (2026-09-21)

Scope: structure-tensor / Hessian orientation fields, learned orientation regression, fibre tracking in
materials CT, diffusion-MRI orientation-distribution estimation, and paper/parchment/textile CT — with a
concrete read on what maps onto usrm2's recto-vs-verso problem (recto: horizontal fibres; verso: vertical
fibres, section 0 of `docs/research/tsm_ideas.md`) and the sheet-normal / cascade channels already in
`docs/unified_design.md` sections 21-23. tsm's own attempt (`tsm/src/tsm/fiber.py`) is in-repo prior art, not
"outside VC" literature, and is referenced only as a baseline to compare against.

## 1. Structure-tensor orientation fields: the standard tool, and why it isn't a candidate channel here

The structure tensor (the outer product of the image gradient, Gaussian-smoothed at an "integration scale")
is the default way to get a per-voxel orientation out of a 3D volume without training anything: its
eigenvectors give the local principal directions and its eigenvalue ratios give anisotropy/coherence
("how fibre-like is this voxel"). It is decades old (Bigün & Granlund 1987; Knutsson 1989) and remains the
workhorse in materials CT — fibre-reinforced composites (Emerson et al. 2017, "Individual fibre
segmentation from 3D X-ray CT"; the STXAE structure-tensor-orientation-mapping paper, Herrmann et al.,
ScienceDirect 2021), wood/MDF (thermal fibre-orientation tensors for paper/paperboard, Karlsson-type work;
"Computational approaches for structural analysis of wood specimens", De Gruyter 2024), and cardiac
microstructure at teravoxel scale (`cardiotensor`, Reichardt et al., arXiv:2508.07476, 2025 — chunk-based,
HPC-parallel structure-tensor + tractography pipeline built explicitly to scale past what a single
in-memory 3D FFT/convolution can do; the design pattern, not the cardiac content, is the relevant part).
GPU implementations exist and are cheap: the `structure-tensor` PyPI package has CuPy-backed
`structure_tensor_3d`/`eig_special_3d`; `fiberorient` and `OrientationPy` (the EPFL OrientationJ successor)
are CPU reference implementations used as ground truth in composites papers; `pi2` documents the same
recipe end to end. Cost at usrm2's rung-2 patch size (256^3): three first derivatives, six second-moment
smooths, one 3x3 eigendecomposition per voxel — a few hundred ms on GPU for a 256^3 cube, i.e. cheap enough
to be an input channel if it were useful.

It is not, by the standard already applied in `unified_design.md` section 21: "local filters ... fail that
test since the first 3x3x3 layers learn them." A structure tensor's eigenvector field is exactly a smoothed,
non-linear function of local gradients, computable from a handful of learned 3x3x3 convolutions plus a
non-linearity a 30M-parameter net can trivially represent. The literature agrees this is a *classical
alternative to* learning, not a *complement* to it: every deep-learning fibre paper found here either (a)
uses the structure tensor purely as a **post-hoc analysis** on a CT volume with no network in the loop
(materials-CT anisotropy quantification, cardiotensor), or (b) uses a CNN to first **segment/denoise** the
volume and only then runs the structure tensor on the clean output (composite fibre papers: "local fibre
orientation is obtained using the structure tensor calculated from image scans" after CNN segmentation).
None of the surveyed 2018-2026 work feeds a structure-tensor field *into* a segmentation CNN as an input
channel and reports a measured gain over the raw image — which corroborates rather than undercuts the
existing unified_design.md rule.

**Mapping to usrm2**: do not add a structure-tensor channel. If a real orientation *signal* is wanted, get it
from a downstream **target**, not an input filter (section 2 below) — the network can compute the same
filter itself from the CT it already has.

## 2. Learned orientation fields: unit-vector regression, circular losses, equivariant nets

Three lineages are relevant.

**(a) Diffusion MRI ODF estimation.** The closest analogue in a different modality: recovering, per voxel,
one or more fibre directions from a noisy 3D signal. Two threads matter:
- *Constrained spherical deconvolution* (Tournier et al. 2007, foundational; still the field's default) —
  deconvolves a fixed single-fibre "response function" out of the diffusion signal to get a fibre orientation
  *distribution* (not a single vector) on the sphere, with a non-negativity constraint added specifically
  because unconstrained deconvolution produces physically-impossible negative lobes. Recent work
  (arXiv:2306.02900, 2023, "Robust FOD estimation using deep CSD") replaces the deconvolution with a learned
  operator but keeps the ODF-on-a-sphere output, because a single vector cannot represent crossing fibres.
  usrm2's fibre problem is simpler — recto/verso is a **bimodal, near-orthogonal** case (horizontal vs
  vertical), not a continuum of crossing angles — so a full spherical-harmonic ODF head is over-built; the
  two-class-or-axial-direction framing tsm already used is the right complexity level.
- *Equivariant spherical CNNs* (arXiv:2504.01925 / PMC12343732, 2025, neonatal dMRI): predicting a
  rotation-equivariant FOD directly, gaining "data and parameter efficiency" specifically because the
  network no longer has to learn rotation invariance from augmented examples — the architecture guarantees
  it. This is the strongest 2018-2026 argument for orientation-equivariant machinery, but it targets a
  spherical (S^2) output built from spherical harmonics or SO(3)-steerable filters (Weiler et al. 2018
  pattern; also "Leveraging SO(3)-steerable convolutions for pose-robust semantic segmentation", PMC7617181),
  which is a substantial architecture change (steerable convolutions throughout, not a 1x1x1 head) — not a
  drop-in for usrm2's existing U-Net-style stem/decoder. Worth flagging, not worth adopting now: usrm2 already
  gets rotation robustness cheaply from the 48-symmetry cube augmentation (`aug.apply`), which is exact for
  the 24+24 group the data loader draws from; equivariant convolutions would only help for *continuous*
  rotations the augmentation doesn't cover, and papyrus fibres away from the umbilicus are locally
  axis-aligned enough (the "axis-relative" framing in `fiber.py`) that this gap is probably small.

**(b) Circular/axial regression losses outside orientation-specific literature.** General surveys
(arXiv:2504.04242, "Task-based Loss Functions in Computer Vision"; the loss-functions survey in *Artificial
Intelligence Review* 2025) confirm the standard toolkit for angle-valued targets: (i) regress `(sin θ, cos θ)`
or a unit vector directly with a squared-error or cosine loss (works when the quantity is genuinely a
direction, 2π- or π-periodic depending on whether it's polar or axial); (ii) discretize into bins and treat
as classification (used in pose-estimation work, e.g. the 360-bin orientation distributions in the
part-orientation paper found here) when multimodality or discontinuity matters; (iii) a Bingham/matrix-Fisher
distribution loss (NeurIPS 2020 camera-pose work) when calibrated *uncertainty* over orientation is wanted,
not just a point estimate. tsm's `fiber.py` already implements the correct member of family (i) for an
**axial** (sign-free, π-periodic, not 2π) quantity: `1 - (d̂·d)^2`, the squared-cosine loss, which is exactly
what the general loss-function literature recommends for axis-valued (not vector-valued) targets — this is
not a tsm-specific hack, it's the textbook answer, and tsm's choice was right.

**(c) The double-angle / doubled-vector trick for sign ambiguity.** A separate strand (oriented-object
detection, arXiv:2411.10497 "Structure Tensor Representation for Robust Oriented Object Detection"; classic
2D orientation-field smoothing) represents a π-periodic direction as `(cos 2θ, sin 2θ)` — mapping the
direction to a point on the circle traversed *twice*, which makes the sign ambiguity (θ and θ+π are the same
axis) disappear algebraically: averaging in the doubled representation is well-defined where averaging raw
unit vectors is not (two anti-parallel vectors would cancel to zero instead of reinforcing). This is the 3D
analogue of what `fiber.py`'s `direction_from_class`/`class_from_direction` do by hand via
`|d̂·t_v|`/`|d̂·t_h|` absolute-value dot products — same fix, different notation. Worth naming as the general
pattern if usrm2 ever regresses a raw direction vector (not just a class-weighted combination of two known
basis vectors as `fiber.py` does): store/compare the outer product `d d^T` (a rank-1 tensor, sign-free by
construction) rather than `d` itself, exactly the fix graph-learning papers converged on for eigenvector sign
ambiguity (arXiv:2202.13013 SignNet: `φ(v) + φ(-v)`; sorting/canonicalization against a reference tensor for
structure-tensor eigenvectors specifically).

## 3. Fibre tracking / segmentation in materials CT (paper, wood, composites, textiles)

- **Composites**: individual-fibre segmentation from 3D CT for unidirectional fibre-reinforced composites
  (Emerson et al. 2017) and structure-tensor-vs-fibre-identification comparisons (2019-2021, the STXAE
  paper) establish structure tensor as accurate for *aggregate* orientation statistics (volume-averaged fibre
  angle, used for finite-element material models) but not for *individual* fibre tracing at high fibre
  density — that needs either very high resolution (single fibre >> voxel) or instance segmentation.
  Instance-level work (arXiv:1901.01034, "Instance Segmentation of Fibers from Low Resolution CT via 3D Deep
  Embedding Learning") trains a CNN to produce an embedding per voxel and clusters it into individual fibres
  — a genuinely learned approach, but solving a different problem (fibre *count and shape*, glass-fibre
  polymers) than usrm2's (fibre *bulk direction*, papyrus).
- **Nonwoven textiles**: "Identification and analysis of fibers in ultra-large micro-CT scans of nonwoven
  textiles using deep learning" (Textile Research Journal / Tandfonline 2022) is the closest textile analogue
  — large-volume CT, deep learning, fibre-level structure — but again targets individual fibre extraction
  in a low-density fibrous mat, not two orthogonal *sheets* of tightly packed strips.
- **Wood/MDF and paper/paperboard**: structure-tensor-based fibre-orientation-tensor estimation is standard
  (thermal-conductivity-analogy method, "Thermal fiber orientation tensors for digital paper physics",
  ScienceDirect; local structure tensor for MDF, Springer 2015) but these are **aggregate physics
  characterization** papers (feeding finite-element or thermal models), not per-voxel deep-learning targets,
  and they explicitly note the same imaging problem papyrus has: "μCT scans often exhibit low contrast and
  strong artifacts due to high porosity ... and unknown additives" — i.e. even outside archaeology, cellulose
  fibre CT is a low-SNR regime where filter-only (non-learned) orientation estimation is known to be noisy.
  This is independent corroboration that a *learned* orientation signal (robust to the same noise a
  segmentation net already has to be robust to) should beat a hand-built structure-tensor filter for usrm2's
  data, not just on principle (section 1) but empirically-by-analogy from a different fibrous material.
- **Papyrus/parchment CT specifically, outside VC**: micro-CT and X-ray phase-contrast tomography (XPCT)
  studies of Herculaneum papyri (PMC7813886, "A computational platform for the virtual unfolding of
  Herculaneum Papyri", *Scientific Reports* 2021; the related PLOS ONE ink-visibility paper) describe the
  crisscross fibre structure directly and use it as a *geometric quality signal* rather than a learned
  channel: "the direction of papyrus fibers is exploited as a principal criterion to estimate the accuracy of
  flattening... once flattened, the papyrus sheet is nearly rectangular, with locally perpendicular fibers" —
  i.e. fibre orthogonality is a **post-hoc metric for unwrapping quality**, computed the same structure-tensor
  way as in materials CT, applied to an already-segmented/flattened sheet, not as a per-voxel training signal
  before segmentation. No paper found (2018-2026, outside VC) trains a network to predict fibre direction on
  raw papyrus CT and uses it as a segmentation input or auxiliary loss; the nearest thing is the
  scrollprize-affiliated nnU-Net surface model that outputs a 3-way (surface / horizontal-fibre /
  vertical-fibre) probability map for geometric tracing (Hugging Face `scrollprize/surface_recto_3dunet`) —
  but that is Vesuvius Challenge work and out of this slice's scope by the user's framing; it is noted only
  because it is direct evidence that **within** the recto/verso classification problem, horizontal-vs-vertical
  fibre probability as an auxiliary *output* (not input) channel is a live, currently-used idea, which the
  next section returns to.

## 4. Orientation as INPUT channel vs OUTPUT target vs self-supervised signal — the evidence, summarized

| role | evidence found | verdict for usrm2 |
|---|---|---|
| **input channel** (hand-built filter, e.g. structure tensor / Frangi sheetness) | No 2018-2026 paper found that adds a structure-tensor or Hessian-eigenvector field as a CNN input and reports a measured accuracy gain over raw intensity; every use is either pure post-hoc analysis (no network) or a *pre*-processing/cleanup step before the filter, never a network input | **Reject**, consistent with `unified_design.md` section 21's own filter-redundancy rule; a 3x3x3 stem already subsumes it |
| **output target** (regress a direction / ODF / class) | Diffusion-MRI CSD/FOD work (multiple, 2018-2026) treats orientation as the primary *output*; auxiliary-task multi-task-learning surveys (arXiv:1805.06334; arXiv:2412.19547) and one directly on-point paper ("with an auxiliary classification task, performance for orientation estimation is improved... training is more stable with smaller and fewer spikes", found in the orientation-auxiliary search) report gains from pairing an orientation head with a primary task, and vessel-segmentation MTL work (arXiv:2509.03975, 2025) measures a segmentation-accuracy gain from an auxiliary SDF/structural regression branch even when that branch's data is unavailable at inference | **Plausible candidate**, and it's the shape `fiber.py` already used (a direction head, not an input filter) |
| **self-supervised / geometric derivation, no new labels** | `fiber.py` itself: the direction target is a closed-form function of already-available quantities (sheet normal from `∇sdf`, scroll axis, teacher class probabilities) — "nothing new is stored" | This is the cheapest and most defensible route for usrm2 *if* a fibre signal is added: derive it from the sheet normal (already recoverable from the recto/verso probability gradient) and a scroll axis, exactly as tsm did, rather than sourcing a new external teacher |

The auxiliary-task literature's measured gains (saliency-detection auxiliary task: ~3% segmentation gain;
SDF-regression auxiliary branch: measured vessel-segmentation improvement) are the best outside evidence that
an orientation-flavoured **auxiliary output head** can help a primary segmentation task even when the
auxiliary target itself is imperfect or has a smaller receptive-field justification than the main task — this
is a point *in favour* of eventually adding a fibre-direction head to usrm2, contingent on solving the
labelling problem (below), not a point in favour of an input channel.

## 5. Concrete mapping to usrm2

**Fibre-direction channel to separate recto/verso.** Do not add one as an *input*. If pursued as an *output*:
usrm2 already has, for free, two of tsm's three prerequisites — a global axis convention is unnecessary
(usrm2 uses the radial vector for outward, not a teacher's per-crop convention, sidestepping the
`tsm_ideas.md` section 0 ambiguity), and a sheet normal is derivable from the gradient of the recto/verso
probability output (or, once it exists, an SDF-style face target per `tsm_ideas.md` section 4's mapping) the
same way `fiber.py`'s `sdf_normal` derives it from `sdf_in`. What's still missing, and is the actual blocker
per `unified_design.md`/`tsm_ideas.md`: (a) an axis-tangent input channel (tsm found this a "hard
prerequisite" once any axis-relative target is added, because augmentation silently corrupts axis-relative
labels without it — the scroll axis is not usrm2's radial vector and is not currently an input channel at
all) and (b) a source of vt/hz ground truth: tsm's was a distilled external villa 4-class fibre teacher
checkpoint that usrm2 does not have. The papyrus-CT literature (section 3) confirms structure-tensor
orientation is genuinely visible and computable directly from CT texture at usrm2's resolutions (fibre
ridges 50-200 um wide, well above the 2.4 um rung-2 voxel size and even above rung 4's 9.6 um) — so a
**self-supervised** structure-tensor-derived vt/hz label (computed offline once, at rung 2 or 3, from CT
texture directly, no external teacher) is a real alternative to tsm's distillation route and closes gap (b)
without new inference infrastructure; it would need the same noise handling section 3's wood/paper papers
flag (low-SNR cellulose CT) — likely a coherence/anisotropy-gated confidence weight (structure tensor's own
`λ2/λ1` ratio), matching how `fiber.py`'s `s = max(p_vt, p_hz)` strength already gates the loss.

**Sheet normal from local anisotropy.** Already effectively present in usrm2's pipeline via the recto/verso
probability output's spatial gradient (once a verso store exists per `unified_design.md` section 23) — no
separate anisotropy-input machinery is needed; this is the same "compute it from the network's own output,
not a hand filter on the input" pattern as the cascade channel (section 22).

**Costs.** A structure-tensor pass at rung 2/3 (256^3 or smaller after downsampling) is cheap on GPU (the
`structure-tensor` CuPy package or a hand-rolled 3x3 eigendecomposition kernel; sub-second per 256^3 cube) if
used *offline* to build a self-supervised vt/hz label store, comparable in cost to the existing target-pyramid
export pipeline (section 9 of `unified_design.md`). An equivariant-architecture route (section 2a) would be
materially more expensive — new layer types throughout, not a channel — and the evidence for it here is
about *sample efficiency on genuinely continuous rotations*, which usrm2's 48-symmetry-augmented, largely
axis-aligned fibre geometry may not need enough of to justify the cost; not recommended now.

**Pitfalls, cross-checked against the sources above:**
- *Sign ambiguity*: a structure-tensor eigenvector, or any raw regressed direction vector, has no inherent
  sign; average it directly and antiparallel estimates cancel instead of reinforcing. Fix: doubled-angle
  representation or an outer-product (`d d^T`) representation before any averaging/interpolation/loss
  (section 2c) — `fiber.py` already avoids this by construction (its targets are basis-relative
  coefficients, not raw vectors), so it only bites if usrm2 later regresses a raw direction vector directly.
- *Scale selection*: the structure tensor's Gaussian integration scale must roughly match the fibre period
  (papyrus ridges 50-200 um); too small and it responds to CT noise/grain, too large and it blurs across the
  perpendicular fibre layer and reports garbage at the recto/verso boundary — directly relevant if a
  structure-tensor label is computed at a *coarser* rung than the fibres are resolved at (rung 4+, 9.6 um+,
  where individual fibres approach or exceed the voxel pitch and orientation estimation degrades).
- *Noise in low-contrast cellulose CT*: repeatedly flagged in the wood/paper materials-CT literature (section
  3) as a known failure mode for structure-tensor methods specifically (not a learned-network problem) —
  argues for computing any structure-tensor label at the finest resolution where fibres are still
  individually resolved (rung 0-2) and treating its own eigenvalue-ratio coherence as a per-voxel confidence
  weight, exactly as `fiber.py`'s `s` strength scalar already gates its (differently-sourced) direction loss.
- *Axis-relative vs sheet-relative confound*: tsm's core lesson (`tsm_ideas.md` section 0, `fiber.py` module
  docstring) — vt/hz is defined against the **scroll axis**, not the local sheet, so any augmentation that
  rotates the volume's z-axis (usrm2's 48-symmetry cube augmentation does this routinely) silently corrupts
  the labels unless the axis itself is either supplied as an input or the swap/direction correction from
  `axis_swap_needed`/`direction_from_class` is applied. This is not a general orientation-estimation pitfall
  from the outside literature so much as a usrm2/tsm-specific geometry fact, but it is the load-bearing one:
  any new fibre-orientation work here inherits it regardless of whether the label source is a structure
  tensor, a distilled teacher, or an equivariant network.

## 6. Bottom line

Outside-VC literature does not offer a shortcut past tsm's already-correct architecture (axial direction
head, squared-cosine loss, basis built from sheet normal + scroll axis) — it corroborates each piece
independently (structure tensor as the standard classical orientation tool but not as a learned-net input;
squared-cosine/doubled-angle as the textbook axial loss; auxiliary orientation heads as a measured
segmentation aid in unrelated domains; low-SNR cellulose CT as a known structure-tensor failure mode in
non-archaeological fibrous materials too). The one genuinely new option the literature surfaces is a
**self-supervised structure-tensor-derived vt/hz label**, computed offline from CT texture directly at fine
rungs with a coherence-based confidence weight, as an alternative to tsm's blocked external-teacher
dependency — everything else (equivariant architectures, ODF/spherical outputs, individual-fibre instance
segmentation) is real but solves a harder or different problem than usrm2 currently has.

## Sources

- Bigün & Granlund, structure tensor (1987); Knutsson (1989) — foundational, cited via the Wikipedia
  structure-tensor summary and the `pi2`/`OrientationJ` documentation.
- [structure-tensor (PyPI, CuPy GPU)](https://pypi.org/project/structure-tensor/0.3.1)
- [fiberorient (GitHub)](https://github.com/scott-trinkle/fiberorient)
- [OrientationJ / OrientationPy (EPFL BIG)](https://bigwww.epfl.ch/demo/orientation/)
- [pi2: orientation via structure tensor](https://pi2-docs.readthedocs.io/en/latest/examples/ex_orientation.html)
- [Cardiotensor: teravoxel-scale structure-tensor + tractography, arXiv:2508.07476](https://arxiv.org/pdf/2508.07476)
- [Micro-CT structure tensor vs high-fidelity fibre ID in composites (ResearchGate)](https://www.researchgate.net/publication/338035784_Micro-CT_based_structure_tensor_analysis_of_fibre_orientation_in_random_fibre_composites_versus_high-fidelity_fibre_identification_methods)
- [Individual fibre segmentation from 3D X-ray CT, unidirectional composites (ResearchGate)](https://www.researchgate.net/publication/312299326_Individual_fibre_segmentation_from_3D_X-ray_computed_tomography_for_characterising_the_fibre_orientation_in_unidirectional_composite_materials)
- [Instance Segmentation of Fibers from Low-Res CT via 3D Deep Embedding Learning, arXiv:1901.01034](https://arxiv.org/pdf/1901.01034)
- [Identification and analysis of fibers in ultra-large micro-CT scans of nonwoven textiles using deep learning, Textile Research Journal 2022](https://www.tandfonline.com/doi/full/10.1080/00405000.2022.2145429)
- [Thermal fiber orientation tensors for digital paper physics (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0020768316302335)
- [Computational approaches for structural analysis of wood specimens, De Gruyter 2024](https://www.degruyterbrill.com/document/doi/10.1515/rams-2024-0073/html?lang=en)
- [X-ray CT structure tensor orientation mapping for FE models, STXAE (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2665963821000968)
- [Robust FOD estimation using deep constrained spherical deconvolution, arXiv:2306.02900](https://arxiv.org/abs/2306.02900)
- [Equivariant spherical CNNs for FOD estimation in neonatal dMRI, arXiv:2504.01925](https://arxiv.org/html/2504.01925) / [PMC12343732](https://pmc.ncbi.nlm.nih.gov/articles/PMC12343732/)
- [Leveraging SO(3)-steerable convolutions for pose-robust 3D medical segmentation (PMC7617181)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7617181/)
- [Structure Tensor Representation for Robust Oriented Object Detection, arXiv:2411.10497](https://arxiv.org/pdf/2411.10497)
- [Sign and Basis Invariant Networks for Spectral Graph Representation Learning (SignNet), arXiv:2202.13013](https://arxiv.org/pdf/2202.13013)
- [Task-based Loss Functions in Computer Vision, arXiv:2504.04242](https://arxiv.org/pdf/2504.04242)
- [Auxiliary Tasks in Multi-task Learning, arXiv:1805.06334](https://arxiv.org/pdf/1805.06334)
- [Improving Vessel Segmentation with Multi-Task Learning and Auxiliary Data, arXiv:2509.03975](https://arxiv.org/abs/2509.03975)
- [A computational platform for the virtual unfolding of Herculaneum Papyri, Scientific Reports 2021 (PMC7813886)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7813886/)
- [From invisibility to readability: Recovering the ink of Herculaneum, PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0215775)
- in-repo: `/home/forrest/tsm/src/tsm/fiber.py`; `/home/forrest/usrm2/docs/research/tsm_ideas.md` (fiber section);
  `/home/forrest/usrm2/docs/unified_design.md` sections 1-4, 21-23.
