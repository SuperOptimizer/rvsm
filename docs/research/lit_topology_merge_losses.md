# Literature: topology-aware and merge-avoidance segmentation outside Vesuvius Challenge (2026-09-21)

Scope: state-of-the-art (2020-2026, non-VC) on segmenting thin, touching, sheet-/tube-like structures where
the dominant failures are the same two usrm2 already measures — **MERGES** (a bridge of foreground fuses two
adjacent wraps, `merge_frac` 0.44) and **GAPS** (a sheet's own band breaks, `continuity` 0.656). BCE+soft-dice
gives near-zero gradient signal for either failure mode: a one-voxel bridge or a one-voxel gap changes the
loss by a vanishing amount relative to the whole volume. Every method below is read against that specific
blindness, and against `docs/research/synthesis_future_inputs_outputs.md` §1/§4/§5 (candidate table, phased
roadmap, top-5 recommendations), which already independently proposes several of the same ideas (L1 repulsion,
L7 Betti/ECT, O2-O4 distance+normal heads) sourced from villa/tsm/usrm/vc3d rather than the outside literature.
Where this survey corroborates or upgrades an existing candidate, that is called out explicitly; nothing here
duplicates §1-5's own citations to villa/tsm/usrm/vc3d source files.

Four clusters, each with a table (method / venue-year / id / optimizes / cost / evidence / 3D) followed by a
usrm2 mapping paragraph, then a cross-cluster priority list.

---

## 1. clDice, skeleton-based losses, and warping/TopoNet-style topology losses

| # | method | venue/year | id | optimizes | cost/step | evidence | 3D? |
|---|---|---|---|---|---|---|---|
| C1 | **clDice** (Shit, Paetzold, Shit et al.) | CVPR 2021 | arXiv:2003.07311 | soft-skeleton (iterative morphological erosion/dilation, k~5-10 steps) intersected with the *other* mask, both directions, combined as a harmonic-mean-style "centerline Dice" alongside voxel Dice | k extra 3x3(x3) morphological passes per side, fully differentiable, no extra network | 5 public datasets (vessels, roads, neurons, 2D+3D); improves Betti-number error/connectivity at similar or better Dice; now a standard baseline | yes, 3D vessel/neuron volumes explicitly validated |
| C2 | **Skeleton Recall Loss** (Kirchhoff, Isensee et al., MIC-DKFZ) | ECCV 2024 | arXiv:2404.03010 | replaces clDice's *soft* per-step skeletonization with a **precomputed hard GT skeleton** (offline, e.g. kimimaro) and a cheap recall-of-fixed-skeleton term, multi-class capable | near-zero extra GPU cost/step (skeleton computed once, offline); claimed up to 90% GPU-cost reduction vs clDice-style losses | SOTA connectivity on multiple 3D tubular medical benchmarks; first multi-class thin-structure connectivity loss | yes, targets 3D tubular medical volumes directly |
| C3 | **Centerline Cross-Entropy (clCE)** | MICCAI 2024 | papers.miccai.org/miccai-2024/770-Paper1081 | CE-style reformulation of clDice's centerline term, claims better topology consistency without sacrificing volumetric accuracy | similar order to clDice | vessel data; not yet independently verified at scale | vessel volumes |
| C4 | **Homotopy Warping** (Hu) | NeurIPS 2022 | arXiv:2112.07812 | warps GT toward the binarized prediction via a distance-transform-based search for minimal-Hamming-distance "critical voxels" (where flipping the label changes Betti number), penalizes only those voxels — a cheap stand-in for full persistent homology | one DT pass + a warping search per step; explicitly reported to generalize to 3D, no PH computation | outperformed prior topology losses (incl. DMT, clDice-family) on topology metrics in its own benchmarks | yes, explicit 3D claim — the most practical PH-family loss for full-volume 3D training found in this survey |
| C5 | **DMT-loss, discrete Morse theory** (Hu et al.) | ICLR 2021 (spotlight) | arXiv:2103.09992 | extracts the Morse-critical global structure (skeletons/membranes) of the **prediction's own probability map**, penalizes only voxels on that extracted structure — designed for exactly "weak spots of connections and membranes" | nontrivial Morse-complex extraction per patch, but restricted to critical structures (cheaper than full PH over the whole complex) | outperformed prior PH-family methods on Betti error/ARI/VOI on 2D **and 3D EM connectomics (membrane) benchmarks** | yes — one of few PH-family losses with explicit 3D EM membrane evaluation |
| C6 | **TopoLoss** (Hu, Li, Samaras, Chen) | NeurIPS 2019 | arXiv:1906.05404 | persistence-diagram difference between predicted probability map and GT; proves gradient hits the correct critical pixels | computing a persistence diagram per training patch is the expensive step; original work is 2D per-slice | gains in Betti-number error and accuracy on EM membrane/road/vessel-like 2D benchmarks; **explicitly motivated by neuron-membrane segmentation** | 2D demonstrated; 3D claimed possible but markedly more expensive |
| C7 | **Learning Topological Interactions** (Gupta, Hu et al.) | ECCV 2022 (oral) | arXiv:2207.09654 | a convolutional module (not a PH loss) enforcing learned containment/exclusion between *classes* — "class A must never touch class B" | cheap, purely convolutional, no skeletonization/PH | 2D and 3D, CT and ultrasound | yes |
| C8 | ContextLoss | 2025 | arXiv:2506.11134 | context-aware topology-preserving term, successor to DMT/Warping line | not deeply reviewed | new | unconfirmed |
| C9 | Topology-Guaranteed Segmentation: connectivity+genus+width | 2026 | arXiv:2601.11409 | formal constraint enforcing Betti-0, genus, **and minimum width** jointly — width constraint is new relative to C1-C6 and maps to "sheet spacing must stay >= N voxels" | constraint-satisfaction layer, likely projection/optimization per step, higher than C1-C4 | new, reports compliance vs unconstrained/clDice baselines | not confirmed |
| C10 | TopoSculpt: Betti-steered 3D tubular sculpting | 2025 | arXiv:2509.03938 | Betti-number-steered shaping specifically for 3D fine-grained tubular shapes | not deeply reviewed | new | explicit 3D, tubular |

**Mapping to usrm2.** All of C1-C6 operate on the probability channel alone (recto and verso independently);
none needs a new label source beyond the existing binary target pyramid, so all are "free" relative to O2-O5.
The split by failure mode matters more than usual here: **clDice (C1) and Skeleton Recall (C2) are recall-type
losses — they reward a continuous, connected centerline, but a merge bridge is itself thin and connected, so
neither penalizes it; a bridge can even score *well* under clDice.** They are GAPS-closers, not
MERGES-preventers, and map most directly onto `continuity` (0.656). C2 is the best cost/evidence tradeoff in
this cluster (near-zero step cost, offline skeleton reuses the same EDT machinery §7.1 of the synthesis doc
already calls for) and is a natural Phase A/B companion to L4 (cascade self-consistency). **Homotopy Warping
(C4) and DMT-loss (C5) are the ones that attack MERGES directly**, because their critical-voxel selection is
defined as exactly the voxels whose flip changes Betti number — a bridge is precisely such a voxel set. DMT-loss
(C5) is the single best-targeted match to usrm2's stated failure (it was built for "weak spots of connections
and membranes" in 3D EM, structurally identical to two touching papyrus wraps) but has the highest
implementation complexity of C1-C6; C4 is the better first pilot (simpler, explicit 3D, no PH library
dependency). C7's convolutional exclusion module is a second, cheaper implementation reference for the
synthesis doc's own L1/L3 (repulsion/exclusivity) rather than a new idea — worth reading alongside villa's
`NormalGatedRepulsionLoss` before committing to either implementation. Pitfall shared by C1/C2: the soft- or
hard-skeleton operation assumes the band is thick enough that skeletonization is stable; at rung>=4 where a
sheet is 2-3 voxels wide this can be noisy. C9's width-constraint framing is worth a direct read since "minimum
width" is close to "sheets stay >= N voxels apart," i.e. a direct merge-preventer, but its cost profile is
unconfirmed.

---

## 2. Betti-matching / persistent-homology losses, and the Euler characteristic transform

| # | method | venue/year | id | optimizes | cost/step | evidence | 3D? |
|---|---|---|---|---|---|---|---|
| B1 | **Betti Matching** (Stucki, Paetzold, Shit, Menze, Bauer) | ICML 2023 | arXiv:2211.15272 | an *induced matching* between prediction and GT persistence barcodes — fixes plain Betti-number-error losses, which can score two spatially-wrong-but-count-matching predictions as correct | standard PH software (Ripser-family) per step; no 256^3 numbers published at this stage | 6 datasets, improves topological accuracy (Betti-matching-error) while holding Dice roughly constant | mostly 2D/2D-slice in the original paper |
| B2 | **Efficient Betti Matching for 3D** (Stucki, Bürgin, Paetzold, Bauer) | 2024 | arXiv:2407.04683 | same loss, new C++/Python cubical-complex PH implementation purpose-built to make **3D** persistent homology tractable per training step | authors claim "significant speedups" over Cubical Ripser; exact wall-clock/voxel-scale numbers are in the paper body, not pulled here — read before committing | multiple datasets, improved 3D topological correctness | **explicit 3D**, the paper's stated contribution; hard new C++ dependency, not pip-trivial |
| B3 | TopoLoss (Hu et al. 2019) | NeurIPS 2019 | arXiv:1906.05404 | see C6 above — same paper, listed here for the PH-loss lineage | 2D-scale cost; 3D unconfirmed | see C6; explicitly motivated by neuron membranes | 2D demonstrated |
| B4 | **Clough et al. topological loss (cardiac)** | TPAMI 2020/2022 | arXiv:2107.12689 (+ arXiv:2008.09585 multi-class) | PH loss against an explicit prior Betti-number target, extended to multi-class via all label pairs | per-patch PH computation | cardiac MRI (ACDC), predominantly 2D slices | 2D |
| B5 | **Fast Euler Characteristic Transform topology loss** (Li, Ma, Ouyang, Paetzold, Rueckert, Kainz) | 2025 | arXiv:2507.23763 (IEEE TMI) | a "fast χ" formulation as a topology-correctness signal, explicitly positioned as cheaper than PH ("polynomial complexity... difficult for high-dimensional data") | selling point of the paper: cheaper than PH, no persistence-diagram matching | abstract confirms **experiments on both 2D and 3D datasets** | **explicit 2D+3D**, best-costed 3D-capable topology loss found in this survey |
| B6 | Differentiable Euler Characteristic Transform (DECT) — general mechanism | 2023 | arXiv:2310.07630 | differentiable ECT w.r.t. sampling directions/coordinates; foundational math for B5, not itself a segmentation loss | n/a | shape classification | n/a |
| B7 | Spatial-Aware Persistent Feature Matching | 2024 | arXiv:2412.02076 | successor/competitor to Betti Matching (B1) | not deeply reviewed | new | unconfirmed |
| B8 | Efficient connectivity-preserving instance seg. via supervoxel loss | 2025 | arXiv:2501.01022 | supervoxel-graph loss avoiding full PH, connectivity-focused, targeted at EM/connectomics-scale volumes | claimed efficient vs. full PH | new, targets large 3D volumes | **3D**, built for our scale range |

**Mapping to usrm2.** This is the exact cluster the synthesis doc's L7 already flags as "high, unknown cost,
persistent homology on 256^3 is not a per-step budget," recommending "pilot ECT-mass first, not Betti
matching." This literature now backs that call with real evidence rather than an internal guess: **B5 (fast
ECT, 2025) is the correct pilot** — it is the only method in this cluster with an explicit, dedicated 3D
formulation *and* a stated cost advantage over PH, matching the synthesis doc's existing preference exactly.
**B2 (efficient Betti matching 3D)** is the fallback if ECT under-performs — it is the first PH-proper method
with a genuine 3D-affordability claim, but it pulls in a bespoke C++ persistent-homology library as a hard
dependency, which is a heavier commitment than L7's own text anticipated; read the arXiv:2407.04683 experiments
table for actual wall-clock/voxel-count numbers before scoring it against a training budget. None of B1-B8
needs a new label (all operate on the existing binary target); all attack **both** merges and gaps
simultaneously (a spurious bridge is an extra H0/H1 feature, a break is a missing one) rather than being
lopsided toward one failure the way the skeleton-recall cluster is. Pitfall common to the whole cluster:
patch-cropping truncates real topological features at patch boundaries, producing spurious topology-loss
gradient at every crop edge — this is a known PH-loss failure mode not specific to any one paper here, and
usrm2's 256^3-or-smaller training patches make it a live risk; mitigate by computing the loss only on an
interior sub-block, away from the crop boundary.

---

## 3. EM connectomics practice for touching membranes

| # | method | venue/year | id | optimizes | cost | evidence | 3D? |
|---|---|---|---|---|---|---|---|
| E1 | **MALIS** (Turaga et al.; structured-loss form Funke/Tschopp et al.) | NeurIPS 2009 / TPAMI 2018 | arXiv:1709.02974 | trains a voxel-affinity CNN so the loss gradient hits the single weakest (maximin) edge on the path between any two voxels — directly targets the bottleneck edge that would merge or split objects | expensive: a maximin-spanning-tree / union-find search per training example every step; "constrained MALIS" reduces this by separating intra/extra-object passes but is still costlier than plain BCE | SOTA (at publication) on ISBI/CREMI/FIB-25 via 3D UNet + affinities + agglomeration; superseded in accuracy-per-compute by LSD | inherently 3D (voxel affinity graphs over 3D EM stacks) |
| E2 | **Local Shape Descriptors (LSD)** (Sheridan et al.) | Nature Methods 2023 | doi:10.1038/s41592-022-01711-z; code github.com/funkelab/lsd | an auxiliary regression head predicting local per-voxel object statistics (centroid offset, covariance/elongation/diameter, direction) within a small window, trained jointly with short-range affinities by MSE — pure extra supervisory signal, no graph search | near-free at inference (one small conv head); affinity+LSD reported **two orders of magnitude more efficient than flood-filling networks** at matching accuracy | consistent gains over affinities-only on CREMI/FIB-25/Zebrafinch (VOI, adapted Rand); the strongest, most reproducible EM result in this space | yes, designed and validated in 3D EM |
| E3 | **Mutex Watershed** (Wolf, Pape, Bailoni, Rahaman, Kreshuk, Köthe, Hamprecht) | ECCV 2018 (+ theory arXiv:1904.12654; Semantic Mutex Watershed 2020; GASP arXiv:1906.11713) | openaccess.thecvf.com/.../Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper | **inference-time graph algorithm, not a loss**: processes short-range attractive + long-range repulsive ("mutex") affinity edges strongest-to-weakest via a Kruskal/union-find variant; repulsive edges register a permanent never-merge constraint, no threshold or seed needed | near-linear in edges (sort + union-find), no learned params, deterministic — essentially free relative to training | SOTA on ISBI 2012 EM at publication when fed CNN affinities; Semantic Mutex Watershed and GASP extend/generalize it, benchmarked on CREMI | routinely run in 3D on volumetric affinity graphs (offsets e.g. ±1,±3,±9,±27 voxels per axis) |
| E4 | **Discriminative embedding loss** (De Brabandere, Neven, Van Gool) | CVPR-W 2017 | arXiv:1708.02551 | per-voxel embedding with a variance term (pull same-instance together) and a **repulsion/margin term** (push different-instance means apart by hinge margin) | cheap: O(N) variance term, O(K^2) over instance *clusters* (small K), no graph search | Cityscapes/KITTI (2D, not EM); pattern widely reused in bio-image instance segmentation (EmbedSeg-style) | extends to 3D directly (embedding per voxel), used in 3D microscopy, not connectomics proper |

**Mapping to usrm2.** The field's consistent answer to "how do you stop two touching objects fusing" is *not*
a better Dice/BCE term on the mask itself, but a **separate relational signal between nearby voxels**,
supervised or computed independently of the binary mask. The most directly portable idea is not full MALIS
(E1) — its per-step maximin graph search is exactly the cost profile the task brief is trying to avoid, and it
presupposes a per-wrap *instance* partition, which the synthesis doc already rejects building (§3, "dead:
learned coarse winding"). The practical port is a **long-range affinity auxiliary head/loss, mutex-watershed
style but training-time only**: add a handful of fixed-offset "same-face" affinity channels (offsets tuned to
the 15-35 voxel sheet spacing, not the short-range 1-3 voxel offsets typical in EM) supervised by BCE against
affinity targets derived geometrically from the existing binary target masks (same class at voxel i and i+offset
=> positive, cross a different wrap => negative) — label-free relative to new annotation, and structurally a
more rigorously evidenced version of the synthesis doc's own **L1 (normal-gated repulsion)**, which is currently
unablated anywhere in villa/tsm/usrm. This is a direct upgrade candidate for L1, not a new item: same failure
target (M), similar cost class, but backed by CREMI/FIB-25-scale evidence rather than an internal estimate.
Running the full **mutex watershed (E3)** as an inference-time graph post-process is a heavier commitment
(requires an instance/winding decomposition first, which conflicts with the same "no coarse winding" decision)
and should be deferred; its *concept* — repulsive long-range edges, zero training cost — is what to borrow, not
the algorithm itself. **LSD (E2)** is the best-evidenced auxiliary-head idea in the whole EM literature but
needs real reinterpretation before porting: its descriptor set (diameter, elongation, direction) assumes
roughly blob/tube-like objects (neurons), while usrm2's targets are locally near-planar sheets — the
descriptors would need to become local sheet normal, local curvature, and local spacing-to-nearest-parallel-
sheet rather than a literal port. Expected effect of E2-style descriptors is mainly on **gaps/continuity**
(LSD's own gains are largest in flat, boundary-ambiguous interior regions, which is close to usrm2's "band
breaks" failure), with a weaker effect on merges than the affinity/mutex approach. E4's embedding+repulsion
pattern is the weakest match here — it also presupposes a per-instance count (K in the O(K^2) term), which
recto/verso's 2-class formulation doesn't have; lowest priority of the four.

---

## 4. Layered media (OCT/cortex/geology) ordering constraints, and vessel/membrane gap-closing

| # | method | venue/year | id | optimizes | cost | evidence | 3D? |
|---|---|---|---|---|---|---|---|
| L-a | **Order-constrained retinal layer regression** (Morelle, Wintergerst, Finger, Schultz) | Sci. Reports 2023 | doi:10.1038/s41598-023-35230-4 | regresses layer heights as offsets from a reference, each constrained non-negative (softplus/cumsum) so predicted layers **cannot cross by construction** | cheap — a reparameterization of the regression head, no extra forward pass | Duke/UPenn AMD data, BM/RPE/EZ mean abs. dist. 0.63/0.85/0.44 px, improved drusen accuracy | 2D per B-scan, trivially stacks |
| L-b | **DL + differentiable dynamic programming for OCT surfaces** (Xie et al.) | 2022/2023 | arXiv:2210.06335 | UNet features feed a soft-DP layer finding the globally optimal ordered, smooth surface set under hard min-distance/no-crossing constraints, end-to-end differentiable | moderate — one DP relaxation per forward, parallelizable per column | Duke AMD MASD 1.88±1.96 um, JHU MS 2.75±0.94 um, beats plain regression and post-hoc graph search | column-wise (2D slice) |
| L-c | **Assignment Flow for order-constrained OCT** | 2020 | arXiv:2009.04632 | continuous-time assignment-flow ODE with a built-in ordering prior, geometrically guaranteeing monotone layer order | higher — ODE integration per column at inference | retinal OCT, order violations reduced to ~0 vs CNN baselines | 2D |
| L-d | **Probabilistic SDF retinal layers** | 2024 | arXiv:2412.04935 | predicts a **signed distance function per layer**; order is implicit in SDF zero-crossings, plus a calibrated uncertainty head | cheap-moderate, SDF regression head only | new, competitive surface-distance error with added uncertainty | 2D, but SDF formulation is inherently 3D-compatible — directly analogous to usrm2's own O2/O3 |
| L-e | **Laplace-equation-constrained cortical laminar segmentation** | 2023 | arXiv:2303.00795 | layer geometry = level sets of a **harmonic potential** solved between two known bounding surfaces (WM, pial) — crossing is geometrically impossible by construction, needs only *two* local bounding surfaces, not a global layer count | moderate — solves/uses a Laplace field as an auxiliary geometric prior | improves deep-sulcus segmentation continuity vs. standard 3D CNN segmentation | **true 3D**, closest geometric analogue to usrm2's radial-axis setup of anything surveyed |
| L-f | **Deep Relative Geologic Time (RGT)** (Bi, Wu, Geng et al.) + **sinusoidal-mapping follow-up** | 2021 / 2026 | RGT 2021 (cig.ustc.edu.cn); sinusoidal arXiv:2605.01273 | regresses a single scalar RGT field over the volume; every horizon is a level set of RGT, so stratigraphic order is enforced by construction; sinusoidal (sin/cos phase) reparameterization fixes RGT's periodicity/unwrapping failures at fault zones | cheap — one extra scalar head, self-supervised from local seismic dip/similarity, no manual horizon labels needed | industry-standard technique, widely cited continuity improvements; sinusoidal version fixes unwrapping artifacts | **true 3D**, volumetric from the start |
| L-g | **Vertical-constrained seismic horizon tracking with uncertainty** (Liao et al.) | IEEE TGRS 2024 | doi via ieeexplore.ieee.org/document/10587310 | explicit vertical (depth-axis) ordering constraint plus a conditional-density/uncertainty formulation, replacing point-wise losses that ignore horizon order | moderate, conditional-density head | field+synthetic seismic, reduced order-violation vs point-wise baselines | 3D volume, per-trace but jointly across the volume |
| V-a | **DconnNet** (Yang & Farsiu) | CVPR 2023 | arXiv:2304.00145 | predicts per-pixel **directional connectivity maps** (8/26-direction local connectivity, a learned analogue of graph connectivity) via a sub-path direction excitation module; adds a size-density loss for class imbalance | moderate — one extra lightweight decoder head | CHASEDB1 retinal vessels: clDice 83.3%, improved Betti-number error over UNet/AttUNet/CE-Net/clDice/Graph-Cut baselines | published 2D; 26-connectivity generalizes to 3D but not demonstrated there |
| V-b | Topology-Guaranteed connectivity+genus+width (dup. of C9) | 2026 | arXiv:2601.11409 | see C9 | higher | new | unconfirmed |
| V-c | Efficient connectivity-preserving supervoxel loss (dup. of B8) | 2025 | arXiv:2501.01022 | see B8 | claimed efficient | new, 3D-scale | 3D |
| M-a | **COp-Net, learned contour-closing operator** | 2024/2025 | arXiv:2407.15817 | a dedicated post-hoc CNN that takes a gappy cell-contour probability map and outputs a corrected, closed contour — learned morphological closing, trained end-to-end | cheap as a small second network applied post-hoc, not in the main training step | SEM cell images, improved instance separation, reduced leaking through gaps vs raw-threshold | 2D contour-based |
| M-b | **Seg2Link** | Sci. Reports 2023 | doi:10.1038/s41598-023-34232-6 | DL boundary prediction + 2D watershed + explicit cross-slice linking that reconnects the same cell across z when per-slice segmentation gaps/splits it — rule-based post-hoc gap-closer | cheap, post-hoc heuristic, no training cost | 3D brain-tissue stacks, successful reconstruction where per-slice-only methods fragment | 3D via 2D+linking, not a true 3D network |

**Mapping to usrm2.** The strongest transferable concept here is architectural, not a specific loss:
**collapsing "N ordered layers" into a single monotone scalar field (RGT, L-f) or a non-negative-offset
regression (L-a) makes crossing/merging structurally unrepresentable rather than merely penalized** — a
stronger prior than any soft loss term, and directly on-target for merges (a merge *is* two wraps losing their
order locally). But every method in the OCT/cortex/geology sub-cluster assumes a **small, fixed layer count
with a globally well-defined ordering axis** (retinal depth, WM->pial axis, geologic time). The scroll has an
unbounded, damage-broken winding count and a locally-foldable axis near crush/delamination — exactly the
assumption that killed tsm's own global winding field (synthesis doc §3, "dead: learned coarse winding," 42%
within-tolerance against a 60% gate). **L-e (Laplace-field cortical layers) is the one variant in this
sub-cluster that survives that critique**, because it only needs *two* known local bounding surfaces (like
recto+verso of one sheet, or one sheet's inner/outer neighbors), not a global count — read this one directly
before RGT if pursuing a layer-order head. A *local* (per-patch, bounded-window) reparameterization of O3's
sdist head as a monotone offset-from-local-reference (à la L-a/L-d) rather than a raw signed distance sidesteps
the aliasing failure while keeping the anti-crossing property; this belongs in Phase B/C of the synthesis doc
(needs the existing EDT target, reparameterized) rather than Phase A. Effect on GAPS is weak-to-neutral — none
of L-a through L-g were built to close a break in a single layer's own continuity. For the connectivity/gap
sub-clusters: **DconnNet's directional-connectivity idea (V-a)** — a per-voxel map of "does this voxel connect
to its neighbor in direction d," derived label-free from the existing binary target (a Sobel/structure-tensor-
cost pass) — is a materially cheaper approximate substitute for L7/B5 (Betti/ECT) and, unlike L1's
antiparallel-exemption complexity, naturally separates "connects along-sheet" from "connects across-sheet to
the neighboring wrap" *if* the direction set is pre-rotated into a normal-aligned local frame using the O4
normal head once it exists — attacks merges directly (a bridge lights up as an anomalous cross-sheet
connectivity edge). **COp-Net (M-a)** is the one candidate in the entire four-cluster survey aimed squarely at
GAPS via a dedicated, cheap, fully label-free post-hoc mechanism: train a small 3D closing network on
synthetically-gapped copies of the existing target pyramid (mask out random thin slabs, train the closer to
restore them) and run it as a refinement pass on the recto/verso probability maps — complementary to, not
competing with, the merge-focused ideas above.

---

## 5. Cross-cluster priority for usrm2

Ranked by (evidence strength) x (3D affordability) x (directness of match to MERGES specifically, since that
is the worse of the two numbers, 0.44 vs continuity 0.656):

1. **Long-range affinity auxiliary loss (E3-inspired upgrade of L1)** — label-free, cheapest evidenced
   merge-preventer here (mutex-watershed's repulsive-edge concept without the inference-time graph algorithm),
   directly supersedes villa's unablated `NormalGatedRepulsionLoss` with CREMI/FIB-25-scale backing. Pilot in
   Phase A alongside L1, same flag-gated discipline.
2. **Directional connectivity map (DconnNet, V-a)** — cheap, label-free, a materially lighter substitute for
   L7/B5 that plugs the same "no relational signal between nearby voxels" gap, attacks merges directly if
   rotated into the O4 normal frame.
3. **Fast ECT loss (B5, arXiv:2507.23763)** — the correct Phase-C topology-loss pilot per the synthesis doc's
   own stated preference ("ECT-mass first, not Betti matching"); this is now literature-backed, not a guess.
   Fall back to Efficient Betti Matching 3D (B2) only if ECT under-delivers on `merge_frac`.
4. **Skeleton Recall Loss (C2)** — near-zero step cost, strong evidence, but a GAPS-closer not a
   MERGES-preventer; pair with #1-#3 rather than substituting for them. Good Phase A/B companion to L4.
5. **Homotopy Warping (C4) or DMT-loss (C5)** — the two PH-family losses that attack both failure modes
   symmetrically and have explicit 3D framing; higher implementation cost than #1-#4, treat as a Phase C
   escalation if the cheaper options plateau.
6. **Local Shape Descriptors (E2), reinterpreted for sheets** — best-evidenced general auxiliary-head idea in
   EM connectomics, but needs real redesign (blob descriptors -> sheet normal/curvature/spacing descriptors)
   before it maps onto Phase B's O2-O4 heads; treat as a design reference for O4, not a drop-in loss.
7. **Local monotone layer-order reparameterization of O3 (à la L-a/L-e)** — the strongest *architectural* prior
   against merges in this whole survey (crossing becomes unrepresentable, not just penalized), but is the only
   item here that needs new/reparameterized supervision rather than being loss-only; belongs in Phase B/C, and
   only the *local*, bounded-window form (L-e's two-surface Laplace framing) survives the same aliasing
   critique that killed tsm's global winding field.
8. **COp-Net-style post-hoc closing pass (M-a)** — the one dedicated GAPS-only mechanism found; cheap,
   fully label-free (train on synthetically-gapped copies of the existing target), complementary refinement
   step rather than a training loss, worth prototyping once the recto/verso heads are otherwise stable.

Deprioritized / not recommended: full MALIS (E1, cost profile is the thing this survey was trying to avoid,
and needs an instance partition usrm2 has already rejected building); mutex watershed as an inference-time
graph post-process (E3, same instance-partition dependency); discriminative embedding+margin loss (E4, needs
an instance count K the 2-class recto/verso formulation doesn't have); global RGT-style layer-index fields
(L-f, same failure mode as tsm's already-dead coarse winding); Assignment Flow (L-c, ODE integration cost per
inference column, no 3D demonstration).

## Sources

- Shit et al., "clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation," CVPR 2021. arXiv:2003.07311.
- Kirchhoff, Isensee et al., "Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures," ECCV 2024. arXiv:2404.03010. github.com/MIC-DKFZ/Skeleton-Recall
- "The Centerline-Cross Entropy Loss for Vessel-Like Structure Segmentation," MICCAI 2024. papers.miccai.org/miccai-2024/770-Paper1081
- Hu, Li, Samaras, Chen, "Topology-Preserving Deep Image Segmentation," NeurIPS 2019. arXiv:1906.05404. github.com/HuXiaoling/TopoLoss
- Hu, "Structure-Aware Image Segmentation with Homotopy Warping," NeurIPS 2022. arXiv:2112.07812. github.com/HuXiaoling/Warping
- Hu et al., "Topology-Aware Segmentation Using Discrete Morse Theory," ICLR 2021 (spotlight). arXiv:2103.09992. github.com/HuXiaoling/DMT_loss
- Gupta, Hu et al., "Learning Topological Interactions for Multi-Class Medical Image Segmentation," ECCV 2022 (oral). arXiv:2207.09654. github.com/TopoXLab/TopoInteraction
- "ContextLoss: Context Information for Topology-Preserving Segmentation," 2025. arXiv:2506.11134.
- "Topology-Guaranteed Image Segmentation: Enforcing Connectivity, Genus, and Width Constraints," 2026. arXiv:2601.11409.
- "TopoSculpt: Betti-Steered Topological Sculpting of 3D Fine-grained Tubular Shapes," 2025. arXiv:2509.03938.
- Stucki, Paetzold, Shit, Menze, Bauer, "Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes," ICML 2023. arXiv:2211.15272. github.com/nstucki/Betti-matching
- Stucki, Bürgin, Paetzold, Bauer, "Efficient Betti Matching Enables Topology-Aware 3D Segmentation via Persistent Homology," 2024. arXiv:2407.04683.
- Clough, Byrne, Oksuz, Zimmer, Schnabel, King, "A Topological Loss Function for Deep-Learning based Image Segmentation using Persistent Homology," TPAMI 2020/2022. arXiv:2107.12689 (+ arXiv:2008.09585, multi-class cardiac variant).
- Li, Ma, Ouyang, Paetzold, Rueckert, Kainz, "Topology Optimization in Medical Image Segmentation with Fast Euler Characteristic," 2025 (IEEE TMI). arXiv:2507.23763.
- Röell, Rieck et al., "Differentiable Euler Characteristic Transform for Shape Classification," 2023. arXiv:2310.07630.
- "Topology-Preserving Image Segmentation with Spatial-Aware Persistent Feature Matching," 2024. arXiv:2412.02076.
- "Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function," 2025. arXiv:2501.01022.
- Turaga et al., "Maximin Affinity Learning of Image Segmentation," NeurIPS 2009; Funke, Tschopp et al., "Large Scale Image Segmentation with Structured Loss based Deep Learning for Connectome Reconstruction," TPAMI 2018. arXiv:1709.02974. github.com/TuragaLab/malis
- Sheridan et al., "Local Shape Descriptors for Neuron Segmentation," Nature Methods 2023. doi:10.1038/s41592-022-01711-z. github.com/funkelab/lsd
- Wolf, Pape, Bailoni, Rahaman, Kreshuk, Köthe, Hamprecht, "The Mutex Watershed: Efficient, Parameter-Free Image Partitioning," ECCV 2018. + theory: arXiv:1904.12654. Semantic Mutex Watershed, 2020 (Springer LNCS). GASP: arXiv:1906.11713.
- De Brabandere, Neven, Van Gool, "Semantic Instance Segmentation with a Discriminative Loss Function," CVPR-W 2017. arXiv:1708.02551.
- Morelle, Wintergerst, Finger, Schultz, order-constrained retinal layer regression, Scientific Reports 2023. doi:10.1038/s41598-023-35230-4.
- Xie et al., "Differentiable Dynamic Programming for OCT Surface Segmentation," 2022/2023. arXiv:2210.06335.
- "Assignment Flow for Order-Constrained OCT Segmentation," 2020. arXiv:2009.04632.
- "Uncertainty-aware Retinal Layer Segmentation via Probabilistic Signed Distance Functions," 2024. arXiv:2412.04935.
- "Improved Segmentation of Deep Sulci in Cortical Surfaces" (Laplace-constrained laminar segmentation), 2023. arXiv:2303.00795.
- Bi, Wu, Geng et al., "Deep Relative Geologic Time," 2021 (cig.ustc.edu.cn); sinusoidal-mapping follow-up, 2026. arXiv:2605.01273.
- Liao et al., "A Deep Learning-Based Seismic Horizon Tracking Method With Uncertainty Encoding and Vertical Constraint," IEEE TGRS 2024. doi via ieeexplore.ieee.org/document/10587310.
- Yang, Farsiu, "DconnNet: Directional Connectivity-based Segmentation," CVPR 2023. arXiv:2304.00145. github.com/Zyun-Y/DconnNet
- "VascuConNet," Medical & Biological Engineering & Computing, 2024. doi via link.springer.com/article/10.1007/s11517-024-03150-8.
- "COp-Net: Deep Contour Closing Operator," 2024/2025. arXiv:2407.15817.
- "Seg2Link," Scientific Reports 2023. doi:10.1038/s41598-023-34232-6.
