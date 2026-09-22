# Physics- and simulation-based CT augmentation: outside-VC literature (2026-09-21)

Scope: state of the art OUTSIDE Vesuvius Challenge work on physics/simulation-based augmentation for
CT/micro-CT segmentation robustness, and on procedurally generated layered/fibrous training data. Compared
throughout against what `usrm2/aug.py` already has (see its docstring and `INTENS`): `_bias` (low-freq
multiplicative gain = cupping/beam hardening), `_ring`, `_stripe` (detector line), `_cor` (centre-of-rotation
ghost), `_haze` (dense-region decoherence blur+contrast loss), `_unsharp` (nabu's Paganin post-filter),
`_quant` (histogram-window 8-bit re-cast), `_lowres`/`_thick`/`_pool` (resolution/PSF family), `_blur`/
`_aniso_blur`/`_sharpen`, `_spectral` (1/f-shaped noise), `_tone` (monotone remap), `_class_contrast`/`_modes`
(scan-domain family from tsm), `_zjit`, `_sheetcomp` (villa's sheet compression), plus the spatial family
(rot/scale/shear/elastic) and cube-symmetry. This is already a broad, physically-motivated image-domain suite.
The question this note answers: what does the literature outside VC do that we don't, and is it worth porting.

## 1. Physics-based augmentation as domain randomization (the dominant 2023-2025 pattern)

**SinoSynth** (Xia et al., MICCAI 2024, arXiv:2409.18355) is the cleanest match to our situation: a *known,
paired* high-quality CT and an *unknown-parameter* degraded modality (CBCT vs our varying energy/propagation/
pixel-size scans). It forward-projects a planning CT into a 3D sinogram, applies randomized CBCT-specific
degradations (beam hardening curve, scatter kernel, detector blur, noise, streaking) with *shuffled order and
random occurrence*, then reconstructs with FDK back to volume space. Trained purely on this synthetic domain,
downstream networks generalized to real multi-institutional CBCT better than networks trained on real CBCT
itself (their claim). Two extra consistency losses (sinogram-domain and structure-domain) stabilize the
image-domain outcome. Takeaway for us: **randomize in a fixed order-shuffled pipeline with per-step
occurrence probability**, which `aug.py` already does at the intensity-augmentation level (`INTENS` list +
per-effect `p`), but SinoSynth additionally *shuffles pipeline order* per sample — worth doing for the
detector/scan-domain family (bias, ring, stripe, cor, haze, unsharp, quant) since real acquisition chains
compose these in the physical order (source -> object -> detector -> recon), not a fixed learned order; a
network exposed to only one composition order can pick up on order-specific correlations. Concrete change:
randomly permute the order of `_bias`, `_ring`, `_stripe`, `_cor`, `_haze`, `_unsharp`, `_quant` within
`intensity()` per sample (cheap: shuffle the sublist before the loop).

**Domain randomization in image and feature space for abdominal CT/MRI** (Zhang et al., Radiology: AI 2026,
PMC12476582) adds a second axis beyond image-space augmentation: perturbing *intermediate feature-map
statistics* (mean/std per channel, sampled from a randomized distribution) during training, not just the
input. This is the "MixStyle"/feature-statistics-randomization family. Applicability: moderate — it targets
domain shift the image augmentations can't reach (e.g. reconstruction-kernel-induced texture that survives
z-scoring). Recommendation: low priority; would require a model-side hook (perturb GroupNorm running
stats or activation stats at a mid-decoder level) rather than an `aug.py` change, and our channel/context
richness (9-cube pyramid, cascade) already gives strong cross-scale regularization. Flag as a later model-side
experiment, not an aug-suite item.

**Physics-informed low-dose CT simulation for lung nodule detection** (PMC13366036, 2025) forward-projects a
normal-dose CT, adds calibrated Poisson (quantum) + correlated Gaussian (electronic) noise in the *projection*
domain at a target dose, and reconstructs — rather than adding noise in image space. Applicability to us is
limited: we don't have projection data in the training loop (CT arrives as reconstructed uint8 volumes), and
BM18's photon counts/dose aren't in our metadata. `_noise`/`_mulnoise`/`_spectral` are image-domain
approximations already; going to true sinogram-domain noise would need a forward/back-projector in the
pipeline (ASTRA/tomopy), a much bigger build for uncertain payoff given our images are already reconstructed
and Paganin-filtered (denoised) upstream. **Not recommended** for this project; note the option for later if
teacher-noise-floor turns out to matter.

**"One Sequence to Segment Them All" cross-domain CT/MRI spine** (arXiv:2605.03098, 2026) reports that a
*small, cheap* augmentation set targeting the specific physical difference between domains (intensity range,
resolution, and a learned/simple style-transfer step) closes most of the cross-domain gap efficiency-wise
versus large generic augmentation libraries. Consistent with our design philosophy already (tsm's measured
scan-domain family) — no new action, but supports keeping our real-metadata-driven `_bias`/`_quant`/`_tone`
knobs central rather than adding many more generic photometric ops.

## 2. Beam hardening / cupping

Our `_bias` (low-frequency multiplicative gain field, "beam hardening / cupping") is already the standard
image-domain proxy used in the beam-hardening-correction literature (e.g. the CBCT dental beam-hardening
correction work, arXiv:2010.03778, which frames the artifact as a smooth low-frequency multiplicative/additive
cupping plus streaks near dense objects). The main thing outside-VC papers add that we lack: **material-
dependent (dense-object-anchored) cupping**, i.e. the gain field is not spatially arbitrary but strongest
near/inside dense regions (papyrus ink/mineral inclusions, case walls) and radially symmetric about them,
because beam hardening is caused by preferential absorption of low-energy photons passing through more
material. Our `_bias` field is a free-form low-res random field with no such anchoring. Recommendation:
optional but cheap — bias the random low-res grid `_bias` samples by CT density itself (e.g. multiply the
per-cell gain amplitude by local block-mean intensity) so cupping is stronger where more absorber is; low
priority since Paganin-filtered phase-contrast BM18 data is already less prone to classic absorption-CT
cupping than lab CT, and our current free-form field is a reasonable superset.

Energy dependence: BM18 scans at 53-137 keV across the fleet (`unified_design.md` sec 21: 74/78/137 keV on
Paris 4 alone). No paper found ties an augmentation parameter directly to `energy_keV`, but the physical
scaling is well established (attenuation coefficient roughly falls as E^-3 in the photoelectric regime, beam
hardening severity falls with increasing mean energy and with monochromatic/synchrotron beams is much weaker
than polychromatic lab CT). Recommendation: since scans span 53-137 keV, make `_bias`'s amplitude range
inversely related to `energy_keV` when training on a specific scan's metadata (lower energy -> allow larger
cupping amplitude; e.g. `max` scaled by `(78/energy_keV)`), rather than one fixed range for all scans; this
uses the metadata channel work already planned in unified_design.md sec 21.3 (scan-level conditioning planes)
— if that plane goes in, the augmentation and the conditioning signal should agree in sign, i.e. don't let the
network learn "high cupping + high energy" as an impossible combination it never sees.

## 3. Rings / streaks

Our `_ring` (concentric multiplicative gain rings about a random center) and `_stripe` (constant-column/row
gain bands) already cover the two dominant synchrotron detector artifacts: ring artifacts from
pixel-to-pixel detector gain variation (rings in reconstructed slices, from stripes in the sinogram — see the
2025 arXiv:2505.19513 stripe-classification paper) and column defects. What's missing versus the sinogram-
domain literature: **partial-ring arcs and ring-radius drift with z** (real detector rings are not perfectly
constant-radius through a stack when there is z-drift or a badly calibrated flat-field, and are not always
full 360-degree circles when a single dead pixel run is short). Low priority: our `_ring` already draws a
random number of rings `k["n"]` with random radius/width/amplitude per sample, which approximates the
population statistics even without axial drift; the network never sees enough consecutive z to need
axial-drift realism at 256^3 patch scale. **No change recommended.**

One idea worth porting: `_ring`/`_stripe` currently act as multiplicative gain in image space (post-
reconstruction). The literature (stripe-classification and multi-stage papers) reconstructs from a synthetic
*sinogram* with the stripe injected pre-reconstruction, which makes the resulting image-space ring shape
exactly circular-in-object-space *only when the rotation axis matches*, and produces the correct "streaks
converge toward the ring center" radial structure automatically. Our `_ring` already draws rings centered near
the crop (approximating this locally). This is adequate for patch-based training; true sinogram synthesis
would need a projector and is not worth building for a marginal accuracy gain here.

## 4. Phase-contrast fringe / Paganin delta-beta variation

This is the most physically distinctive gap versus generic CT augmentation. BM18 data is Paganin-filtered:
delta/beta and unsharp coefficient/sigma vary by scan (`unified_design.md` sec 21: 1000/4.0/1.2 on the 78 keV
2.4 um scan vs 500/4.0/2.5 on the 1.1 um mosaic). No outside-VC paper was found doing *augmentation* for
delta/beta mismatch specifically (searches on "Paganin simulation augmentation deep learning" mostly return
phase-retrieval/denoising papers, e.g. arXiv:2601.07225 "On optimization of Paganin's method...", PMC12265864
deep-learning phase-retrieval enhancement, arXiv:2211.01372 robustness of learned phase retrieval under lab
conditions) — this is a genuinely under-studied corner even outside VC. What the physics tells us though:

- The Paganin filter is itself a low-pass filter with strength set by delta/beta and propagation distance
  (roughly, effective blur sigma grows with sqrt(delta/beta * distance / (4*pi) )); mis-set delta/beta either
  under- or over-smooths the reconstruction and leaves residual edge-enhancement fringes (bright/dark halo at
  material interfaces) when delta/beta is too small relative to the true material ratio, or over-blurs fine
  fibre structure when too large. This is the physical justification for `_unsharp` already in `aug.py`
  ("nabu's unsharp mask... its default a=1, s=1" — this simulates the *residual* edge enhancement after an
  imperfect Paganin pass) and for `_blur`/`_aniso_blur`. What's not yet tied together: `_unsharp`'s sigma
  range (`k["s_lo"]/k["s_hi"]`, currently scan-agnostic) should scale with the *actual* `unsharp_coeff/sigma`
  values seen in the metadata fleet (1.2 to 2.5 observed), and more importantly with pixel size, since sigma
  is defined in pixels but the physical PSF is roughly fixed in microns — a fixed sigma range at rung 2
  (2.4 um) is a different physical blur than the same sigma at rung 0 (0.6 um). Recommendation: make
  `_unsharp`'s sigma bounds (and `_blur`'s) a function of rung (voxel size) rather than constant across the
  ladder, i.e. `sigma_um` fixed, `sigma_vox = sigma_um / voxel_um(rung)`; this is a small code change in
  `train.py`'s cfg builder (pass rung-dependent `k` dicts) and directly targets the propagation-distance /
  delta-beta variability documented in sec 21.3 without needing a real Paganin forward simulation.
- A genuine forward simulation (apply the actual Paganin transfer function with a randomized delta/beta to a
  "clean" volume, i.e. simulate what a different delta/beta choice at reconstruction time would have produced)
  is possible in principle (Paganin's formula is a simple Fourier-domain filter: divide by
  `1 + delta/beta * distance * lambda * |k|^2/(4*pi)` in frequency space) and would be a faithful, cheap-to-
  implement single-FFT augmentation, strictly more physical than the current `_blur`/`_unsharp` combo.
  Recommendation: **add `_paganin_jitter`** — apply the inverse-then-forward Paganin transfer function with a
  resampled delta/beta ratio (e.g. multiply/divide the effective blur by a log-uniform factor 0.5-2x per
  sample, one 3D FFT filter, isotropic in-plane / different along z is not needed since Paganin acts on the
  2D projection before reconstruction but its net effect on the reconstructed volume is close to isotropic in
  the transverse plane and separately tunable in z via `_aniso_blur`). This is new physics content beyond
  `_blur`, worth adding; medium implementation cost (one FFT-domain kernel, analogous in cost to `_spectral`).
  Cite: Paganin et al. 2002 (original phase-retrieval method, pre-2019 so out of the requested window but
  foundational and referenced by every phase-contrast-augmentation-adjacent paper found, including
  arXiv:2601.07225 2026 and PMC12265864 2025 above) plus this project's own delta/beta metadata as the
  parameter source.
- No paper models fringe *residuals at material interfaces from failed phase retrieval* (the classic bright/
  dark edge doublet when delta/beta is wrong) as an explicit augmentation; `_unsharp`'s signed amplitude
  (`k["a_lo"]` can be negative implicitly via the `+ _p(...)` signed range) already produces this doublet
  structure locally (unsharp mask overshoot at edges = the fringe signature), so this gap is effectively
  already closed by the existing op, just not parameterized by real delta/beta ranges. Action: parameterize,
  don't add a new op, beyond the `_paganin_jitter` FFT filter above if implemented.

## 5. Resolution / PSF changes and energy-dependent contrast

`_lowres`/`_thick`/`_pool` already give a broad resolution-degradation family (nearest/avg/max/min/median/
stride decimation, optional anisotropic, blocky-or-smooth upsample) that's more thorough than what's typically
published (most papers use a single Gaussian-blur-plus-downsample step). No outside-VC paper does anything
beyond this for PSF variation in CT specifically; the cryo-ET/EM literature (CryoGEM, arXiv-adjacent,
"physics-informed generative cryo-EM", 2024/2025) models the *contrast transfer function* (CTF) explicitly as
a Fourier-domain multiply with defocus/aberration parameters, analogous to the Paganin-filter idea above but
for electron optics — supports the general pattern "model the real optical transfer function in Fourier space
and randomize its physical parameter" rather than approximating with spatial blur kernels. This reinforces the
`_paganin_jitter` recommendation above as the CT-appropriate analogue, since a real CT/phase-contrast MTF is
also naturally a Fourier-domain operation. Energy-dependent contrast (different keV means different relative
attenuation of papyrus vs mineral ink vs air, so contrast ratios shift, not just blur) is only partially
covered by `_class_contrast`/`_tone`; those are scan-agnostic random remaps, not physically tied to
`energy_keV`. Given the fleet spans 53-137 keV, recommendation: same pattern as sec 2 — scale `_class_contrast`
and `_tone`'s amplitude ranges by a factor derived from `energy_keV` relative to the training scan's own
energy, OR (simpler, no new physics) just confirm the *existing* ranges already bracket the empirical
Paris3-vs-Paris4 contrast-ratio spread tsm measured (3.49 vs 2.70, i.e. ~1.3x) comfortably, which they do
(`_class_contrast`'s `lo`/`hi` and `_contrast`'s `max` are presumably tuned already — not inspected here, flag
for a numeric check against the 1.3x figure rather than guessing new bounds).

## 6. Synthetic/procedural layered and fibrous materials, sim-to-real

Three outside-VC papers are directly on point for "procedurally generate a layered/fibrous material and its
CT, train on it, transfer to real scans":

- **Griem, Koeppe, Greß, Feser, Nestler, "Synthetic training data for CT image segmentation of
  microstructures"**, *Computational Materials Science* (or similar Elsevier venue), 2025,
  ScienceDirect S1359645425005075 / SSRN 5087564 / DLR repository (elib.dlr.de/215032). Two-stage pipeline:
  (1) an algorithmic/procedural generator produces a coarse synthetic microstructure (their demonstration
  case: foam) with known ground truth, (2) a learned refinement step (image-to-image, GAN-like or
  style-transfer) makes the synthetic CT progressively more realistic, iterated so the synthetic distribution
  closes in on the real one. They validate transfer not just by visual/Dice similarity but by checking that
  *downstream physical property measurements* (e.g. porosity, permeability-type quantities derived from the
  segmentation) computed on real CT using the synthetic-trained network agree with independently measured
  material properties — a stronger transfer test than pixel accuracy alone. Applicability: high in spirit,
  low in direct portability — our targets already come from strong upstream teacher predictions (recto/m7),
  not from scratch, so we don't need de novo procedural papyrus generation for the bootstrap; but the
  *validation methodology* (check a physical/geometric invariant downstream, not just Dice) is worth adopting:
  e.g. validate augmented-vs-real transfer via sheet-count or wrap-spacing consistency, which
  `usrm2/verso.py`-style overlap metrics could extend to.
- **"Synthetic, automatically labelled training data for machine learning based X-ray CT image segmentation:
  Application to 3D-textile carbon fibre reinforced composites"**, *Composites Part A* (or similar),
  ScienceDirect S1359836825005578, 2025. Procedurally generates the 3D-textile fibre-tow geometry (a woven
  fabric model, analogous in structure to papyrus's crossed fibre layers) with automatic per-voxel ground
  truth, simulates a CT-like appearance (their paper is behind a paywall here so the exact noise/blur model
  used is not confirmed from the abstract alone, but their pipeline is "free/open-source software" end to end
  — likely tomopy/gVirtualXray/aRTist-class simulators), and trains a segmentation network purely on the
  synthetic set. Reported result: **88% pixel-wise agreement** with manual segmentation of a real CT scan,
  trained on synthetic data only, no real training examples. This is a strong sim-to-real result for a
  geometrically analogous problem (crossed fibrous layers, low material contrast) and is the closest published
  analogue to "procedurally generate papyrus-like layered/fibrous CT and train on it." Applicability: medium —
  useful validation that pure-synthetic geometry training *can* work for our problem class, but not
  actionable for us right now since (a) we already have real upstream teacher labels covering the geometry
  distribution at scale, (b) building a procedural papyrus-fibre-layer CT simulator (crossed/parallel fibre
  bundles, carbonisation-driven density variation, delamination gaps, ink) is a substantial new subsystem, not
  an aug.py change. Recommendation: **defer**; revisit only if/when fine-pitch (rung 0/1) coverage remains
  data-starved after the real fine scans (sec 7.5 of unified_design.md) are exhausted — procedural fibre-layer
  synthesis would be the natural next augmentation-adjacent project at that point, and the two papers above
  are the templates to follow (algorithmic generator + iterative realism refinement, or open-source CT
  simulator + a woven-tow geometry model swapped for a papyrus sheet-layer geometry model).
- **Training Generalized Segmentation Networks with Real and Synthetic Cryo-ET Data** (bioRxiv 2025.01.31,
  PMC11838407) is the closest cross-domain precedent for "mix a physically simulated volume with real
  low-label data": they show a network trained on a mix of physics-simulated cryo-ET tomograms (known ground
  truth, procedurally placed macromolecules with a modeled CTF/missing-wedge/noise chain) plus a modest amount
  of real annotated data generalizes far better than either alone, and that synthetic-only training
  transfers weakly without at least some real fine-tuning. Applicability: reinforces that our plan (teacher-
  label bootstrap + real scans, no synthetic-only phase) is the right order of operations; synthetic procedural
  data, if ever built, should be a *supplement* to the real teacher-label training, not a replacement, matching
  what unified_design.md already does (train on the ladder of increasingly real sources, sec 5-6).

## 7. Mosaic stitching seams

Paris 4's 1.1 um volume is a 19-tile fused mosaic (unified_design.md sec 21.3, `mosaic.*` metadata). Outside-
VC literature on stitching seams (MosaicNet, PubMed 38082798; "Seamless stitching of tile scan microscope
images"; UnMICST's "real augmentation", Nature Comms Biology 2022) is almost entirely about *removing* seams
(feathering/blending at acquisition or post-hoc) rather than augmenting a network to be robust to them, because
in microscopy/pathology the standard practice is to fix the seam before any network sees the image. UnMICST's
"real augmentation" idea (train with genuinely imperfect, seam-containing real tiles rather than synthesizing
seam artifacts) is the one directly transferable point: **don't synthesize seam artifacts; sample training
patches so that mosaic seam locations occur naturally in proportion to their real frequency** (i.e. no
augmentation is needed if the real 1.1 um data, seams included, is simply part of the training distribution —
which it already is once that volume is mirrored per unified_design.md sec 6.2). No new `aug.py` op
recommended. If seam-adjacent patches are ever a measured failure mode (a gain/contrast step at the tile
boundary, similar in effect to `_stripe`), the existing `_stripe` op (constant-column/row multiplicative gain)
already approximates a single seam and needs no new code — just confirm its width range brackets a tile-seam's
typical feather zone if this becomes a diagnosed problem later.

## 8. Cross-scanner generalization: what's actually measured

Quantified evidence that physics-style augmentation moves the needle (for calibrating expectations, not
parameter values):
- CBCT esophagus segmentation (arXiv:2006.15713, Physics-based augmentation, ~2020 but foundational and still
  cited by 2024+ CBCT-domain-randomization work): synthetic-scatter/noise-augmented training reached 0.81 Dice
  on planning CT and 0.74 on real CBCT with **zero real CBCT training data** — i.e. physics-augmented synthetic
  training alone gets most of the way to same-domain performance on the harder target domain. This is the
  strongest quantitative anchor found for "how much does physics augmentation buy you when the target domain
  has no labels," directly analogous to our 42-scroll fine-tuning problem (train mostly on Paris 4's physics,
  generalize to 41 other energy/distance/pixel-size combinations with few or no labels there).
  Sources: [Generalizable Cone Beam CT Esophagus Segmentation Using Physics-Based Data Augmentation](https://arxiv.org/pdf/2006.15713)
- SinoSynth (arXiv:2409.18355, MICCAI 2024): synthetic-only training *outperformed* training on real CBCT for
  downstream image-enhancement/segmentation networks on heterogeneous multi-institutional data — evidence that
  physics-randomized synthetic domains can beat real-but-narrow training distributions for cross-site
  robustness, consistent with our approach of deliberately widening the intensity/artifact distribution rather
  than relying on collecting more real scans at every energy/distance combination.
  Sources: [SinoSynth (arXiv)](https://arxiv.org/abs/2409.18355), [SinoSynth (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12711319/)

## 9. Recommendations summary (ranked)

1. **Tie `_unsharp`/`_blur`/`_aniso_blur` sigma ranges to physical microns, not voxels**, so the same aug
   config means the same physical blur at every rung (`sigma_vox = sigma_um / voxel_um(rung)`). Small change,
   directly targets the documented delta/beta and unsharp_coeff/sigma spread (1.2-2.5) across the fleet.
   High priority, low cost.
2. **Add a `_paganin_jitter` Fourier-domain op**: apply the Paganin transfer-function ratio for a resampled
   delta/beta (log-uniform 0.5-2x the scan's nominal value) as a single FFT low-pass/high-pass step, isotropic
   in-plane. This is new, physically grounded content the current `_blur`/`_unsharp` combo only approximates
   spatially. Medium cost (one new op, analogous to `_spectral`'s FFT machinery already in the file).
3. **Shuffle the order of the detector/scan-domain artifact family** (`_bias`, `_ring`, `_stripe`, `_cor`,
   `_haze`, `_unsharp`, `_quant`) per sample instead of the fixed `INTENS` list order, following SinoSynth's
   "randomize composition order" finding. Trivial cost (shuffle a list before applying).
4. **Scale `_bias`/`_class_contrast`/`_tone` amplitude ranges by `energy_keV`** once the scan-metadata
   conditioning planes (unified_design.md sec 21.3) land, so the augmentation distribution and the conditioning
   signal stay consistent (don't let the model see energy-cupping combinations that can't physically co-occur).
   Depends on sec 21.3 landing first; medium cost.
5. **Defer** procedural fibrous-material CT simulation (papyrus-analogue sim-to-real) until real fine-pitch
   (rung 0/1) coverage is exhausted; the woven-composite (88% agreement, S1359836825005578) and microstructure
   (Griem et al., S1359645425005075) papers are the templates if/when it's built — algorithmic geometry
   generator + either a learned realism-refinement pass or an open-source CT simulator, validated against a
   downstream physical/geometric invariant (e.g. sheet spacing), not just Dice.
6. **No change**: sinogram-domain noise/ring simulation via a real forward/back-projector (not worth the
   build — we never see raw sinograms and current image-domain `_noise`/`_ring`/`_stripe` already approximate
   the population statistics adequately for patch-level training); feature-statistics domain randomization
   (model-side, not an `aug.py` change, and our multi-rung context already regularizes across scale); mosaic
   seam synthesis (real mosaic data already in the mirror covers this; `_stripe` is a sufficient proxy if a
   failure mode is ever diagnosed).

## Sources

- [SinoSynth: A Physics-based Domain Randomization Approach for Generalizable CBCT Image Enhancement (arXiv:2409.18355, MICCAI 2024)](https://arxiv.org/abs/2409.18355)
- [Generalizable Cone Beam CT Esophagus Segmentation Using Physics-Based Data Augmentation (arXiv:2006.15713)](https://arxiv.org/pdf/2006.15713)
- [Deep Learning with Domain Randomization in Image and Feature Spaces for Abdominal Multiorgan Segmentation on CT and MRI Scans, Radiology: AI, 2026 (PMC12476582)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12476582/)
- [Physics-informed data augmentation to simulate low dose CT scans: Application to lung nodule detection, 2025 (PMC13366036)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13366036/)
- [One Sequence to Segment Them All: Efficient Data Augmentation for CT and MRI Cross-Domain 3D Spine Segmentation (arXiv:2605.03098, 2026)](https://arxiv.org/pdf/2605.03098)
- [On optimization/optimisation of Paganin's method for propagation-based X-ray phase-contrast imaging and tomography, 2026 (arXiv:2601.07225)](https://arxiv.org/pdf/2601.07225)
- [Development of a deep learning method for phase retrieval image enhancement in phase contrast microcomputed tomography, J. Microscopy, 2025 (PMC12265864)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12265864/)
- [Investigating the robustness of a learning-based method for quantitative phase retrieval from propagation-based x-ray phase contrast measurements under laboratory conditions (arXiv:2211.01372)](https://arxiv.org/pdf/2211.01372)
- [Ring artifacts correction method in x-ray computed tomography based on stripe classification and removal in sinogram images (arXiv:2505.19513, 2025)](https://arxiv.org/pdf/2505.19513)
- [Griem, Koeppe, Greß, Feser, Nestler, "Synthetic training data for CT image segmentation of microstructures", 2025 (ScienceDirect S1359645425005075; DLR repo elib.dlr.de/215032)](https://www.sciencedirect.com/science/article/pii/S1359645425005075)
- ["Synthetic, automatically labelled training data for machine learning based X-ray CT image segmentation: Application to 3D-textile carbon fibre reinforced composites", 2025 (ScienceDirect S1359836825005578)](https://www.sciencedirect.com/science/article/pii/S1359836825005578)
- [Training Generalized Segmentation Networks with Real and Synthetic Cryo-ET data, bioRxiv 2025 (PMC11838407)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11838407/)
- [CryoGEM: Physics-Informed Generative Cryo-Electron Microscopy (physics-informed simulation + noise translation, 2024/2025)](https://liner.com/review/cryogem-physicsinformed-generative-cryoelectron-microscopy)
- [MosaicNet: A deep-learning-based multi-tile biomedical image stitching method (PubMed 38082798)](https://pubmed.ncbi.nlm.nih.gov/38082798/)
- [UnMICST: Deep learning with real augmentation for robust segmentation of highly multiplexed images of human tissues, Communications Biology, 2022](https://www.nature.com/articles/s42003-022-04076-3)
