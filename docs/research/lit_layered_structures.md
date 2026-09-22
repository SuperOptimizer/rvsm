# Layered and wound structures outside the Vesuvius Challenge: how others segment, order and unroll them

Research note, 2026-09-21. Scope: methods from battery CT, OCT, seismic interpretation, dendrochronology, cortical
laminae and generic instance segmentation that segment stacked or wound layers AND give each layer an identity or an
order. For each: the mechanism that encodes order, evidence, 3D applicability, a mapping onto usrm2 (a new output
channel, an input, or a post-process) and the pitfalls at our scale (one sheet, ~300 wraps, 15-35 voxels apart,
touching / delaminated / crushed). Compare `docs/unified_design.md` sections 21-23 (cascade input, verso head) and
`docs/research/vc3d_tracer_inputs.md` section 4 (what the tracer would consume: recto/verso pair, normals, phase,
SDF, normalised radius, confidence).

Verification marks: [S] title/venue/DOI verified by web search or Crossref this session; [M] cited from memory,
title and venue believed correct, check before quoting numbers; [U] found by search but content not retrievable.

## 0. The one-paragraph answer

Nobody outside VC solves "assign a global wrap index to every surface voxel of a 300-wrap spiral" end to end. The
pieces exist in four places: (a) seismic RELATIVE GEOLOGIC TIME volumes regress exactly the field we want -- a smooth
scalar whose isosurfaces are the layers -- from a raw image with a 3D CNN, and the 2026 variant encodes it as
multi-frequency sinusoids to keep thin layers sharp; (b) OCT layer segmentation shows how to make ordering a
property of the OUTPUT PARAMETERISATION (non-negative thicknesses, cumsum) or of a GRAPH post-process
(Iowa multi-surface cut, DP along rays) instead of a loss term; (c) EM connectomics gives the standard recipe for
separating thousands of touching thin instances in 3D (affinities / embeddings -> mutex watershed), which gives
LOCAL identity with no order; (d) phase unwrapping and Laplace-coordinate methods turn a local periodic signal
(angle about the umbilicus, or a harmonic potential between the two sheet ends) into a global integer index by
post-processing. Battery jelly-roll CT, the closest physical analogue, has "virtual unrolling" by spiral fitting
but almost no learned per-layer labelling; it is an evidence gap, not a source. Recommendation at the end.

## 1. Battery jelly rolls and wound foils

Closest analogue: one electrode sandwich wound ~20-40 turns in an 18650, layers touching, delaminating and buckling
("core collapse") with age. The literature virtually unrolls by GEOMETRY, not by learned labelling.

- Kok, Robinson, Weaving, Jnawali, Pham, Iacoviello, Brett, Shearing, "Virtual unrolling of spirally-wound lithium-ion
  cells for correlative degradation studies and predictive current distribution modelling", Sustainable Energy &
  Fuels 3, 2019 [M]. Fits an Archimedean spiral to the electrode in each CT slice (centre + pitch from the visible
  current-collector foil), resamples the volume along the spiral to a flat (arc length, z, thickness) frame, then
  correlates local degradation with position along the electrode. Order = the spiral parameter; layer identity is
  the integer part of (r - r0) / pitch. Fully 3D but slice-wise. Mapping to usrm2: this is the "normalised radius ->
  winding" prior; it works because pitch is constant and the roll is round, neither true for a crushed papyrus.
  Pitfall: any local deviation (a delaminated flap, a crushed section) breaks the integer-part rule silently.
- Ziesche, Arlt, Finegan, Heenan, Tengattini, Baum, Kardjilov, Markotter, Manke, Kockelmann, Brett, Shearing,
  "4D imaging of lithium-batteries using correlative neutron and X-ray tomography with a virtual unrolling
  technique", Nature Communications 11, 2020 [M]. Same idea, refined: the spiral is fitted to the segmented foil
  and deviations from the ideal spiral are carried as a per-angle radial offset, so the unrolled map stays in
  register when the roll is not perfectly round. Mapping: a per-(z, theta) radial-offset field is a cheap
  "winding field" for the NEAR-ROUND part of a scroll -- essentially what the VC3D spiral tracer fits
  (vc3d_tracer_inputs.md section 2), so no new channel; note it as prior art for the tracer only.
- Pfrang, Kersys, Kriston, Sauer, Rahe, Kabitz, Figgemeier, "Long-term cycling induced jelly roll deformation in
  commercial 18650 cells", J. Power Sources 392, 2018 [M]. CT of jelly-roll buckling; layers tracked by hand /
  thresholding along radial lines. Shows the failure modes that matter to us (folds where one layer is locally
  wound onto its neighbour), no algorithmic contribution.
- Madi et al., "Coupling X-ray computed tomography with digital volume correlation to study core collapse in
  lithium-ion batteries", EES Batteries 2026 [S, DOI 10.1039/d5eb00229j]. DVC (3D block cross-correlation between
  two scans) gives a displacement field that carries layer correspondence between states implicitly. Mapping: a
  self-supervised auxiliary target (predict displacement between two augmented views) is a way to get "same
  sheet" consistency without labels; not an ordering.
- Yu, Li, Zhang, Lian, Sun, "BS-Mamba: a battery-specific Mamba network for robust battery electrode CT image
  segmentation", Measurement 2026 [U, DOI 10.1016/j.measurement.2025.119496]. Semantic segmentation of electrode /
  separator, state-space backbone; abstract paywalled. Only note: an SSM along a radial ray is a plausible way to
  count ~300 crossings sequentially (section 6).
- Not found: a learned per-layer instance labelling of a jelly roll, or a rolled-steel / foil-coil CT layer
  counter. Coil inspection is ultrasonic/eddy-current, not volumetric layer labelling. Both agents searched
  Crossref / OpenAlex / arXiv; NDT.net and IEEE Xplore were not searched directly.

Takeaway: the battery world gets away with geometry because pitch is constant and the layer count is small
(< 50). Our problem starts where theirs ends.

## 2. Retinal OCT: ordering built into the output or into a graph

9-11 layers that must never cross, per A-scan column; the field has three distinct mechanisms.

- Garvin, Abramoff, Wu, Russell, Burns, Sonka, "Automated 3-D intraretinal layer segmentation of macular
  spectral-domain OCT images", IEEE TMI 28, 2009 [S, DOI 10.1109/TMI.2009.2016958]. The "Iowa" multi-surface graph
  search: one cost volume per surface, surfaces found simultaneously by a single min-cut on a graph whose ARC
  STRUCTURE forbids crossing and enforces min/max separation between adjacent surfaces. Globally optimal, truly
  3D. Mapping: the post-processing ideal -- snap a soft prediction to the nearest topologically valid set of
  surfaces. Pitfall: cost scales with (surfaces x voxels); 300 surfaces x a scroll is out of reach unless the
  search is restricted to a band around a coarse prior (which the cascade channel already gives us).
- Xie, Pan, Sonka, Wu, "Globally optimal segmentation of mutually interacting surfaces using deep learning",
  arXiv 2007.01259 (2020); Optics Express 2022 as "Globally optimal OCT surface segmentation using a constrained
  IPM optimization" [M]. A CNN predicts per-column surface-position likelihoods; a differentiable primal-dual
  interior-point layer solves the constrained multi-surface problem INSIDE the network, so ordering / separation
  constraints are hard and training is end to end. 3D. Mapping: the most principled version of "learned cues +
  hard ordering", but the IPM layer is sized for ~10 surfaces.
- He, Carass, Jedynak, Solomon, Saidha, Calabresi, Prince, "Topology guaranteed segmentation of the human retina
  from OCT using convolutional neural networks", arXiv 1803.05120 (2018) [S]; extended as "Structured layer surface
  segmentation for retina OCT using fully convolutional regression networks", Medical Image Analysis 68, 2021 [M].
  Per column the net regresses NON-NEGATIVE THICKNESSES (ReLU); boundary depths are the cumulative sum, so order
  is guaranteed by the parameterisation, zero thickness allowed. MAE 2.82 um vs 2.83 um for the graph method.
  2D per B-scan but trivially vectorised. Mapping: THE transferable trick -- along a radial ray from the
  umbilicus, predict the gap to the next sheet crossing (>= 0) and cumsum; ordering costs nothing. Pitfall: the
  layer COUNT per column is fixed in OCT; on a ray through a scroll it varies with delamination and crushing, and
  a ray is only monotone where the sheet is wound around the axis the ray leaves from (bad in crushed regions
  where the sheet folds back and a ray crosses the same wrap twice).
- Kugelman, Alonso-Caneiro, Read, Vincent, Collins, "Automatic segmentation of OCT retinal boundaries using
  recurrent neural networks and graph search", Biomed. Opt. Express 9, 2018 [S, DOI 10.1364/BOE.9.005759]. An RNN
  scores boundary probability along each column, a shortest path enforces the ordered set. Hybrid pattern:
  unconstrained learned cues, cheap DP for the order. 2D. Mapping: DP along each ray is O(wraps x samples) per
  ray -- affordable at 300 wraps, unlike a 3D multi-surface cut.
- Liu, Wei, Lu, Li, Ma, Wang, Zheng, "Simultaneous alignment and surface regression using hybrid 2D-3D networks
  for 3D coherent layer segmentation of retina OCT images", MICCAI 2021, arXiv 2203.02390; MedIA extension
  arXiv 2312.01726 [S]. 2D encoder, two coupled 3D decoders: a B-scan alignment field and continuous surface
  depths, trained jointly so surfaces are coherent across the volume. One of the few genuinely 3D OCT nets.
  Mapping: couple a normal / displacement head with the index head so neighbouring rays agree.
- Fazekas, Aresta, Lachinov, Riedl, Mai, Schmidt-Erfurth, Bogunovic, "SD-LayerNet", MICCAI 2022, arXiv 2207.00458
  [S]. Surface positions -> pixel map by a differentiable rasteriser, plus a disentangled (anatomy vs appearance)
  latent for semi-supervision. Mapping: the "identity vs texture" split is the wrap-identity embedding idea.
- Islam, de Vente, Liefers, Klaver, Bekkers, Sanchez, "Uncertainty-aware retinal layer segmentation in OCT through
  probabilistic signed distance functions", arXiv 2412.04935 (2024) [S]. One SDF per layer with a Gaussian over
  it; zero level sets are the boundaries, the variance is a per-boundary confidence. Mapping: the tracer already
  wants an SDF (vc3d_tracer_inputs.md 4.4) and a confidence (4.6); predicting a heteroscedastic SDF gives both,
  and it does not need a per-wrap channel because it is "distance to the nearest recto face", not per layer.
- Roy et al., "ReLayNet", Biomed. Opt. Express 8, 2017, arXiv 1704.02161 [S]: plain per-pixel multi-class FCN with
  no ordering mechanism. The negative baseline: works for 7 classes, meaningless for 300 near-identical wraps.

Takeaway: order should live in the PARAMETERISATION (cumsum of non-negative gaps) or in a cheap DP, not in a
soft loss; and every OCT method assumes a known layer count per column, which we do not have.

## 3. Seismic horizons and relative geologic time: the winding field already exists under another name

An RGT volume is a scalar field, one value per voxel, increasing monotonically with deposition time, whose
isosurfaces are the horizons (layers). Replace "time" by "turns about the umbilicus" and it is a winding field.

- Stark, "Relative geologic time (age) volumes -- relating every seismic sample to a geologically reasonable
  horizon", The Leading Edge 23, 2004 [M]: the original. Instantaneous phase of the seismic trace is unwrapped in 3D
  so that each 2-pi of phase is one layer; RGT = unwrapped phase. This is literally phase unwrapping of a periodic
  local signal into a global count -- section 5.
- Wu and Zhong, "Generating a relative geologic time volume by 3D graph-cut phase unwrapping", Geophysics 77, 2012
  [M]: the unwrapping done as a graph cut, robust to noise and faults (cuts placed where the phase is unreliable).
- Wu and Hale, "Horizon volumes with interpreted constraints", Geophysics 80, 2015 [M]; Wu and Fomel, "Least-squares
  horizons with local slopes and multigrid correlations", Geophysics 83, 2018 [M]: RGT as the solution of a
  least-squares problem -- find the scalar field whose gradient is parallel to the locally estimated layer normal
  (structure tensor / slopes) subject to sparse interpreted control points. No network. Mapping: given our
  predicted sheet NORMALS (a channel the tracer wants anyway), a winding field is the least-squares solution of
  grad(w) parallel to n with |grad w| = 1 / pitch, anchored at a few picked wraps. A post-process, cheap on the
  surface graph, and it is exactly what the "normalised radius" channel of section 21 approximates crudely.
- Geng, Wu, Shi, Fomel, "Deep learning for relative geologic time and seismic horizons", Geophysics 85, 2020 [S,
  DOI 10.1190/geo2019-0252.1]: a 3D CNN regresses RGT directly from the amplitude volume, trained on synthetic
  folded / faulted models with known RGT. Horizons = isosurfaces. First evidence that a net can emit the ordinal
  field itself.
- Bi, Wu, Geng, Li, "Deep relative geologic time: a deep learning method for simultaneously interpreting 3-D
  seismic horizons and faults", JGR Solid Earth 126, 2021 [S, DOI 10.1029/2021JB021882, code zfbi/rgtNet]: joint
  RGT + fault-probability heads; the field is only required to be smooth BETWEEN faults, the fault channel marks
  where it may jump. Mapping: winding channel + a "discontinuity" channel (delamination, tear, crushed fold)
  so the net is not forced to hallucinate a smooth index across real breaks; both heads share the decoder like
  recto/verso do.
- Dou, Wu, Gao, Bi, "RGT-Est: learning stratigraphically consistent relative geologic time from 3D seismic data via
  sinusoidal mapping", arXiv 2605.01273 (2026 preprint; ID as reported by search, not opened) [S/U]: the RGT
  target is encoded as sin(f_i * RGT) at several frequencies and regressed with a perceptual + adversarial loss on
  a 3D HRNet at 512^3; motivation is that plain L1 regression over-smooths thin closely spaced layers. Mapping:
  the strongest single idea for us -- a phase-plus-winding-number decomposition where the fine frequency is the
  angle within a wrap and the coarse ones disambiguate the wrap; decode by unwrapping (section 5).
- Also seen, less relevant: per-horizon tracking nets (Peters, Haber, Granek, "Fully reversible neural networks
  for large-scale 3D seismic horizon tracking", arXiv 2003.08466 [U]) -- label one surface at a time, not a field.

Pitfalls: RGT is monotone in DEPTH almost everywhere; a winding field is monotone along the radial ray only where
the roll is roughly concentric, and it is 2-pi-periodic in angle with a step of exactly one wrap per turn. The
seismic nets never see 300 layers in one 512^3 box, only tens. Synthetic training data (folded models) is how
they get dense labels; our equivalent is a synthetic spiral generator with crush / delamination / tears, or the
VC3D tracer's fitted spiral as a pseudo-label on already-traced regions.

## 4. Tree rings and other concentric growth layers

- Gillert, Resente, Anadon-Rosell, Wilmking, von Lukas, "Iterative next boundary detection for instance segmentation
  of tree rings in microscopy images", CVPR 2023 [M]. Rings are found SEQUENTIALLY from the pith outward: the net
  is conditioned on the previous ring boundary and predicts the next one, so ring identity is the iteration
  count and ordering is by construction. Mapping: a "given wrap k, find wrap k+1" recurrence -- the same as the
  cascade input channel but with the PREVIOUS WRAP's surface as the extra input instead of the coarse prediction;
  identity comes free, errors accumulate over 300 iterations, and a crushed region where wrap k+1 is not adjacent
  to wrap k everywhere stalls it.
- Marichal, Passarella, Randall, "CS-TRD: a cross-sections tree ring detection method", arXiv 2305.10809 (2023),
  and DeepCS-TRD (2025) [M]: polar sampling about the pith, rays, edge chains, ring closure as a graph over ray
  crossings. Same ray-and-DP structure as OCT columns, in polar coordinates -- i.e. our umbilicus rays.
- Zambrano-Suarez et al., "Tree ring segmentation performance in highly disturbed trees using deep learning", PLOS
  ONE 2026 [S]: U-Net variants, Dice 0.77, and a clear statement that dense / disturbed rings break boundary
  detection at ~10-100x LOWER layer density than ours.
- 3D CT ring work ("Automated 3D tree-ring detection and measurement from X-ray CT", Dendrochronologia 2021 [U])
  exists but is classical. No onion / multilayer biological CT ordering paper turned up.

Takeaway: rings are closed, concentric and tens in number; the transferable bit is the polar ray + sequential
"next boundary" formulation, not the models.

## 5. Phase unwrapping and Laplace coordinates: from a local periodic signal to a global integer

The winding index is the integer part of an unwrapped angle: theta about the umbilicus is known everywhere (the
radial-vector channel already encodes it), and w = (theta + 2 pi k) / 2 pi with k the unknown integer. That is
exactly 2D/3D phase unwrapping with the sheet as the wrapped-phase carrier.

- Ghiglia and Pritt, "Two-dimensional phase unwrapping: theory, algorithms and software", Wiley 1998 [M]:
  quality-guided, branch-cut and least-squares (L2 / Lp) unwrapping. Quality-guided flood fill from a seed,
  ordered by a per-voxel reliability, is the natural post-process on the surface graph: seed at one wrap,
  propagate k across sheet adjacency, never across gaps.
- Spoorthi, Gorthi, Gorthi, "PhaseNet: a deep convolutional neural network for two-dimensional phase unwrapping",
  IEEE SPL 26, 2019 [M]; Wang, Li, Kemao, Di, Zhao, "One-step robust deep learning phase unwrapping", Optics
  Express 27, 2019 [M]: nets that predict the integer wrap count k as a CLASSIFICATION per pixel (PhaseNet) or the
  unwrapped phase as regression. Both fail on more than a few tens of fringes and on discontinuities -- the same
  regime warning as seismic. Mapping: predict k mod M (a small cyclic class count, e.g. M = 8) as a per-voxel
  output; the net only has to distinguish a wrap from its ~4 neighbours either side, which is a LOCAL question
  within its receptive field, and a global unwrapping pass resolves the multiples of M. This is the discrete
  twin of RGT-Est's multi-frequency sinusoids.
- Jones, Buchbinder, Aharon, "Three-dimensional mapping of cortical thickness using Laplace's equation", Human
  Brain Mapping 11, 2000 [M]; Waehnert et al., "Anatomically motivated modeling of cortical laminae", NeuroImage
  93, 2014 [M] (equivolume layering). A harmonic potential solved between two boundaries gives a monotone
  "depth" with non-crossing streamlines. Mapping: solve Laplace INSIDE the segmented sheet between its two ends
  (innermost and outermost edge as Dirichlet 0 / 1): the potential is monotone along the sheet, its level sets
  cut the sheet across, and rescaling by arc length gives the unrolled coordinate directly. This is the cleanest
  post-process for a CLEAN sheet mask; every merge between adjacent wraps short-circuits it, so it needs the
  instance separation of section 6 first (or a conductivity that drops to zero on predicted merges).

## 6. Separating touching thin instances: connectomics recipes, plus ordinal regression

Standard 3D tools for "thousands of touching thin objects"; they give local identity, never order.

- Turaga et al., "Convolutional networks can learn to generate affinity graphs for image segmentation", Neural
  Computation 22, 2010 [M]; Lee, Zung, Li, Jain, Seung, "Superhuman accuracy on the SNEMI3D connectomics
  challenge", arXiv 1706.00120 (2017) [M]; Funke et al., "Large scale image segmentation with structured loss based
  deep learning for connectome reconstruction", IEEE TPAMI 41, 2019 [M]. Per-voxel affinities to neighbours at
  several offsets, watershed / agglomeration afterwards; MALIS-style losses train the affinities for the
  segmentation they induce. Mapping: an affinity channel across the sheet normal ("is the voxel on the other side
  of this thin gap the SAME wrap?") is precisely the merge/split question the tracer's gap-count loss struggles
  with; long-range offsets of 15-35 voxels along the normal would encode "next wrap" directly.
- Wolf, Pape, Bailoni, Rahaman, Kreshuk, Kothe, Hamprecht, "The mutex watershed: efficient, parameter-free image
  partitioning", ECCV 2018 / TPAMI 2020 [M]: attractive short-range and repulsive long-range affinities merged
  greedily with no threshold. The repulsive long-range edges are how you say "these two sheets 20 voxels apart
  are DIFFERENT wraps" without a global objective. Recommended partitioner if affinities are added.
- Sheridan et al., "Local shape descriptors for neuron segmentation", Nature Methods 20, 2023 [M]: an auxiliary
  10-channel local-shape target (offset to local mass centre, covariance, size) that regularises affinities.
  Mapping: the sheet analogue is the local normal and thickness -- which the tracer wants anyway.
- De Brabandere, Neven, Van Gool, "Semantic instance segmentation with a discriminative loss function", CVPRW 2017
  [M]; Neven et al., "Instance segmentation by jointly optimizing spatial embeddings and clustering bandwidth",
  CVPR 2019 [M]: per-pixel embeddings pulled together within an instance, pushed apart between, clustered at test
  time. Mapping: a low-dimensional embedding head is the "wrap identity" channel; it only needs to separate a
  wrap from its neighbours (push term over a small radius), and a 1-D ordered embedding degenerates into the
  winding field. Pitfall: embeddings are not globally unique -- a cluster in one region has no relation to the
  same wrap 5 mm away; a global pass (section 5) is still needed.
- Bai and Urtasun, "Deep watershed transform for instance segmentation", CVPR 2017 [M]: learned distance-to-
  boundary energy, watershed cut. For a sheet the energy is signed distance to the sheet's mid-surface, i.e. the
  SDF channel again.
- Ordinal outputs: Cao, Mirjalili, Raschka, "Rank consistent ordinal regression for neural networks with
  application to age estimation" (CORAL), Pattern Recognition Letters 140, 2020 [M]; Shi, Cao, Raschka, "Deep
  neural networks for rank-consistent ordinal regression based on conditional probabilities" (CORN), Pattern
  Analysis and Applications 2023 [M]: K-1 binary "greater than k" outputs with shared weights and a single bias
  vector so predicted cumulative probabilities are monotone. Mapping: for the k mod M classifier above, CORN
  gives rank consistency for free and its expected value is a sub-wrap continuous index.
- Monotone networks (Wehenkel and Louppe, "Unconstrained monotonic neural networks", NeurIPS 2019 [M]) enforce
  monotonicity of an output in an INPUT (e.g. radius); not useful when the input the index must be monotone in
  (arc length along the sheet) is itself unknown.

## 7. What this suggests for usrm2

Ranked by expected value over cost, and consistent with the tracer's wish list:

1. Keep recto/verso and add the tracer's local geometric channels first: unit normal (3, or nx/ny in the recto
   hemisphere as the tracer stores it) and a signed distance to the recto face clipped at +-32 voxels, ideally
   heteroscedastic (mean + log-variance, Islam 2024) so the variance is the confidence channel. These are local,
   supervisable from existing masks (EDT of the target, normals from its gradient), and every ordering method in
   this note consumes them.
2. A LOCAL wrap-identity output, not a global index: either (a) M-way cyclic class "winding mod M" with a CORN
   head, supervised where the VC3D tracer or a fitted spiral gives k, or (b) long-range affinities along +-n at
   the sheet pitch (same wrap / next wrap / previous wrap), or (c) a 2-4 dim discriminative embedding with a
   push radius of ~3 wraps. All three are answerable inside a 256^3 receptive field; a global index is not, and
   RGT-Est and PhaseNet both say raw regression of the global value smears at our layer density. (a) is the
   simplest to add to the current head (M extra channels, weight 0 where no tracer label exists, like verso).
3. Global index by post-processing on the surface: quality-guided unwrapping of theta + the mod-M prediction
   (section 5), or the least-squares field grad(w) parallel to n with the affinities as edge weights (Wu and Fomel
   2018), or Laplace along the separated sheet. Both are graph problems on the sheet voxels, not on the volume.
4. Optional discontinuity channel (Bi 2021): "the index is allowed to jump here" -- tears, delamination flaps,
   crushed folds. Cheap to add, and it tells the unwrapper where NOT to propagate.
5. Sequential "next wrap from this wrap" as a cascade-style input (Gillert 2023) is the interactive / refinement
   mode of section 21 item 5 (known surfaces as a sparse input), not a training-time channel.

Pitfalls that recur across every field: a fixed layer count per column (OCT, tree rings) that we do not have;
monotonicity along one Cartesian axis (seismic) that a spiral only has in polar form and loses when crushed;
methods proven on < 50 layers and hundreds of voxels of spacing, never on 300 wraps at 15-35 voxels; and every
learned "global" field failing at density, which is why the local mod-M / affinity formulation plus a classical
unwrapping pass is the safer design. Training labels for any identity channel come from the tracer's fitted
spirals on traced regions or from a synthetic spiral generator, as the seismic community trains on synthetic
folded models -- there is no published wrap-labelled scroll volume to bootstrap from.
