# Self-supervised pretraining and foundation models for 3D volumes (outside Vesuvius Challenge)

Scope: literature on masked/contrastive pretraining for 3D CNNs and ViTs, CT and EM foundation
models, and evidence for when pretraining helps small labelled sets at our scale (~30M params,
one-scroll supervised set, hundreds of TB unlabelled). Context: `docs/unified_design.md`
sections 1-4, 14, 21 (the ladder, one sample, 30M 3D UNet-style model); tsm's DINO-distillation
attempt (`docs/research/tsm_ideas.md` section 6) was implemented but never measured — this
survey is partly aimed at deciding whether to resurrect that idea in a better form or drop it.

## 1. Masked autoencoding for 3D volumes

**SwinUNETR self-supervised pretraining** (Tang et al., CVPR 2022; the original "Self-Supervised
Pre-Training of Swin Transformers for 3D Medical Image Analysis") pretrained on ~5,000 unlabeled
CT volumes with a combination of masked-volume inpainting, rotation prediction, and contrastive
coding, then fine-tuned on BTCV/MSD tasks. Reported gains: up to ~10 Dice points in low-label
regimes, and a systematic edge over from-scratch training across most MSD tasks (their Task03
Liver: 77.8 vs 75.3 from scratch; Task08 Hepatic Vessel: 68.5 vs 64.6; Task10 Colon: 43.4 vs
34.8 — the largest single gain, notably on a thin/sparse-structure task). Their headline claim
is roughly 40% less annotation needed to match from-scratch Dice on BTCV. This is a ViT-style
(Swin) encoder, 62M params in the base config, comparable order of magnitude to us but with far
more pretraining data (5k full CT volumes vs our two scrolls) and a much larger downstream label
set to fine-tune with than one scroll's worth of recto/verso labels.

**"Revisiting MAE pre-training for 3D medical image segmentation"** (Wald, Isensee et al., CVPR
2025, arXiv:2410.23132) is the most methodologically relevant paper for us: it explicitly argues
that most prior 3D SSL work in medical imaging used (a) too little pretraining data, (b) ViT/Swin
architectures that are known to underperform plain CNNs on 3D medical segmentation at these data
scales, and (c) weak evaluation (few, small, biased downstream benchmarks). They instead pretrain
a **Residual-Encoder U-Net** (the nnU-Net "ResEnc" CNN family, not a transformer) with MAE on
~39k unlabeled 3D brain MRI volumes, then evaluate on 8 held-out segmentation datasets. Result:
MAE-pretrained ResEnc U-Net beats both the strong supervised nnU-Net baseline and prior 3D SSL
methods (SwinUNETR-SSL, Models Genesis, etc.) by roughly **+3 Dice points** on average, with the
gain concentrated in the lower-label regimes. This is the strongest CNN-native, MAE-native
evidence we have that pretraining transfers into a from-scratch nnU-Net-class architecture rather
than only helping ViT/Swin backbones that already lag CNNs in this domain. Companion work,
**Hi-End-MAE** (arXiv:2502.08347) and **Swin MAE for small datasets** (arXiv:2212.13805), pursue
hierarchical/asymmetric MAE variants aimed specifically at data-scarce 3D medical regimes; both
report consistent but modest (1-3 Dice point) gains over vanilla MAE, i.e. the architecture of the
pretext task matters less than getting a CNN-shaped backbone and enough pretraining volumes.

**VoCo** (Wu et al., CVPR 2024, arXiv:2402.17300) is a contrastive, not reconstructive, pretext
task: it exploits the fact that CT volumes have consistent global geometry (organs appear in
roughly consistent relative positions) and trains the encoder to predict the relative spatial
position of a random crop against a set of "base crops," without any masking or decoder. Trained
on 10k-160k unlabeled CT volumes, released checkpoints, and reports gains across six downstream
tasks. The pretext task is CT-anatomy-specific (consistent organ layout) in a way that does not
transparently transfer to scroll CT, which has no fixed macro-anatomy — but the *general recipe*
(predict relative position/scale between two sampled sub-cubes of the same object) maps cleanly
onto our multi-rung, multi-scale sampling scheme (section 2 below).

**Models Genesis** (Zhou et al., MedIA 2021) is the older reference point tsm's `feats.py`/
`dino.py` lineage implicitly competes with: four generic in-volume restoration transforms
(non-linear intensity, local pixel shuffling, out-painting, in-painting) trained as a single
denoising-autoencoder pretext task, no external labels, transfers across several CT/MRI
downstream tasks with then-SOTA-for-2019 gains. It has mostly been superseded by MAE-style
masking (simpler, comparably or more effective) and contrastive volume-position tasks (VoCo);
we would not implement Models Genesis directly today, but its core insight — cheap in-volume
restoration objectives beat feature-distillation from an external teacher when no such teacher
exists — is directly relevant here, since we have no CT-domain teacher at all (unlike tsm's DINO
distillation from a natural-image/general-volume ViT).

## 2. CT, general-medical, and EM foundation models

**CT-FM** (project-lighter, 2024-2025, arXiv:2501.09001 survey context) pretrains a 3D encoder
on 148,000 CT scans with label-agnostic contrastive learning, evaluated across whole-body/tumor
segmentation, triage, retrieval, and semantic clustering tasks — general capability across many
downstream heads, at a data scale (148k scans) roughly 4-5 orders of magnitude beyond what any
lab pretrains from scratch for a single application; relevant to us mainly as evidence for scale
sensitivity (below), not as a checkpoint we could transfer (CT numbers/windowing, organ-anatomy
domain, 12-bit clinical CT dynamic range — none of it matches micro-CT of desiccated papyrus).

**SAM-Med3D** (arXiv:2310.15161) and **SegVol** are SAM-style promptable volumetric segmenters,
trained on ~22k 3D images / 143k masks (SA-Med3D-140K) and CT-focused data respectively. Both are
point/box-prompted interactive segmenters, not dense per-voxel classifiers, and both are reported
(2024 follow-ups, e.g. SAM-Med3D-MoE) to segment common organs well but to generalize poorly to
categories/textures far from their training distribution — the relevant lesson for us being that
promptable general-purpose 3D segmenters, even at 100M+ param scale, are anatomy/organ-specific in
practice and would need heavy fine-tuning (not merely a prompt) to see papyrus fiber sheets at
all; not a productive pretrained-checkpoint transfer path.

**Merlin** (Stanford, Nature 2026, arXiv:2406.06512) and **VISTA3D** (NVIDIA/MONAI, CVPR 2025) are
further evidence in the same direction: Merlin is a vision-language CT foundation model trained on
paired CT + EHR + radiology-report data (millions of tokens), and VISTA3D is a SegResNet-backbone
(CNN, not ViT) unified segmentation foundation model. Both confirm the field's current default
backbone choice for 3D medical segmentation foundation models is CNN or CNN-hybrid, not plain ViT
— consistent with the Wald/Isensee finding above, and consistent with usrm2's own architecture
choice (3D UNet-style, not a plain transformer).

**EM/microscopy models** are the closest domain analogue to micro-CT scroll data in texture
statistics (dense, high-resolution, no macro-anatomy prior, thin/sheet-like structures matter).
**CEM500K** (Conrad & Narayan, eLife 2021) is a 500k-image curated unlabeled cellular EM corpus;
reconstruction-based pretraining on it (e.g. RETINA, PMC12143494, 2025) reports faster convergence
and higher accuracy on small annotated downstream sets, consistent with the general MAE story.
**micro-SAM / "Segment Anything for Microscopy"** (Nature Methods 2024/2025) fine-tunes SAM/SAM2
on light- and electron-microscopy data for promptable cell/organelle segmentation; useful as a
recipe reference (how to adapt a natural-image foundation model to microscopy) but again
promptable/interactive, not a dense-prediction backbone we would transfer weights from — and 2D,
where our problem is inherently volumetric.

**No published foundation model targets micro-CT of papyrus or comparable desiccated/carbonized
organic-fiber materials.** The nearest analogues by imaging physics (X-ray micro-CT of low-contrast
fibrous/porous media) are materials-science and paleontology CT literature, which is thin on
public pretraining corpora at our scale and was not surfaced as a distinct foundation-model line
in this search — if this direction is wanted, that's a narrower follow-up query, not this survey.

## 3. When does pretraining help — CNN vs ViT, dataset size, negative transfer

Consistent pattern across the sources above:

- **Backbone**: at 3D-medical-imaging data scales (thousands to tens of thousands of volumes,
  not natural-image-scale billions), plain ViT/Swin encoders underperform CNN or CNN-hybrid
  encoders for dense segmentation, both from scratch and after SSL pretraining (Wald/Isensee
  2025 is explicit about this; VISTA3D and Merlin's segmentation head both use CNN backbones).
  This favors doing any pretraining on **usrm2's own CNN-style encoder**, not swapping to a ViT
  to make an off-the-shelf pretraining recipe (MAE-ViT, DINO-ViT) drop in more easily — echoing
  tsm's own DINO attempt, which distilled from an *external* ViT into a CNN student and was never
  validated, rather than pretraining the CNN itself.
- **Data scale for the pretraining corpus matters more than pretext-task sophistication**:
  Wald/Isensee's central point is that most earlier 3D SSL papers under-pretrained (hundreds to a
  few thousand volumes) and that going to tens of thousands of volumes is what produced a
  reliable, task-general gain; VoCo and CT-FM both scaled from ~10k to ~150k volumes across
  versions and reported larger downstream gains at the larger scale. usrm2's "hundreds of
  terabytes of unlabelled scroll CT" easily clears this bar in raw voxel count, though not
  necessarily in *scan diversity* (many crops of the same handful of scrolls is not the same
  as tens of thousands of independent CT acquisitions with independent noise/artifact/anatomy
  distributions) — diversity of scroll/scan identity, not just voxel count, is what the
  literature's scaling claims are actually keyed to.
- **Where the gain shows up**: gains are consistently largest in low-label regimes (SwinUNETR-SSL:
  biggest jump on the smallest/hardest task, Colon; Wald/Isensee: gain concentrated at low label
  counts) and, notably, on structurally hard targets — SwinUNETR-SSL's largest absolute gain was
  on Hepatic Vessel and Colon (thin, branching, low-volume-fraction structures), and the one
  purpose-built thin-structure pretraining paper found (VAMAE, vessel-aware MAE for OCT
  angiography, 2026) reports pretraining gains concentrated specifically in **clDice / topology**
  metrics, not just Dice, when the masking/reconstruction target is made structure-aware (vessel-
  weighted masking + multi-target reconstruction) rather than uniform random masking. The
  cerebrovascular Frangi-pretraining paper similarly reports +2-3 points clDice from pretraining.
  This is direct evidence that generic masked pretraining transfers to thin/sheet topology
  metrics, and that making the pretext task structure-aware (weight masking/loss toward the
  thin/sheet regions you actually care about, not uniform-random cubes) amplifies that transfer —
  directly relevant to recto/verso sheet topology, which is exactly this kind of thin-structure
  target.
- **Negative transfer / domain gap**: the recurring caution across the transfer-learning survey
  material is that gains shrink or reverse when the pretraining domain diverges from the
  downstream domain (natural-image-to-medical transfer is the most-cited failure case; several
  2024-2025 papers frame their contribution as "bridging" that gap by pretraining in-domain
  instead). This argues *for* in-domain pretraining (scroll CT on scroll CT) over borrowing any
  of the checkpoints above (CT-FM, Merlin, SAM-Med3D), and *against* assuming a generic 3D-medical
  SSL recipe transfers its reported Dice deltas unchanged to micro-CT of papyrus, which has none
  of clinical CT's organ-shape priors, intensity windowing, or acquisition-noise structure.

## 4. Cross-resolution pretraining

No paper surfaced in this search directly pretrains a single encoder across a deliberately
multi-resolution pyramid the way usrm2's ladder does (section 1 of `unified_design.md`: 12 rungs,
0.6 um to 1.23 mm, 9 context cubes per sample). The closest conceptual relatives are: (1) VoCo's
relative-position/relative-scale contrastive pretext task, which could be generalized from
"predict where this crop sits in the volume" to "predict which rung and where this cube sits in
the pyramid" — a natural fit for our existing sample structure, since every training sample
already carries a known rung and context-cube relationship for free, at zero extra data cost; and
(2) the general finding (Wald/Isensee, VoCo) that scale/context diversity in the pretraining
crops improves downstream robustness, which is consistent with our ladder's multi-rung sampling
already being a reasonable pretraining-time source of scale diversity if we choose to use it that
way. This is a genuine gap in the outside literature, not a place to find a ready-made recipe —
our own scale-plane + cascade-channel architecture is more relevant prior art than any external
paper here.

## 5. Concrete recommendation for usrm2

**Worth doing, but as a small, cheap, CNN-native, in-domain pretraining stage — not a ViT-DINO
revival and not a foundation-model checkpoint import.**

- **Objective**: masked-cube reconstruction (MAE-style, on our own encoder/decoder pair — the CT
  cube plus context cubes are already the exact input shape) reconstructing z-scored CT intensity
  in randomly masked sub-regions, plus optionally a VoCo-style auxiliary task predicting which
  rung/relative position a context cube came from (near-free given our existing sample structure).
  Do **not** revive frozen-teacher feature distillation (tsm's `dino.py`/`feats.py` line) — it was
  never validated even in tsm, adds an external-teacher domain-gap risk this survey argues against,
  and duplicates work the cascade channel (unified_design.md section 22) already does structurally
  and *is* measured.
- **Which rungs**: pretrain primarily at the finer rungs (0-4, ~0.6-9.6 um) where CT texture
  actually carries fiber/sheet signal worth reconstructing; coarser rungs are close to piecewise-
  smooth density and a masked-reconstruction objective there is nearly trivial (low pretext
  difficulty, low expected transfer value) — this also matches where the Wald/Isensee and VAMAE
  results locate their biggest downstream gains (thin, textured structure).
- **Structure-aware masking**: given the VAMAE/cerebrovascular evidence that structure-aware
  masking beats uniform-random masking specifically on topology metrics, weight the masking
  toward high-gradient/high-local-variance regions once a cheap recto/verso proxy mask exists
  (e.g. from the published-mask bootstrap in `docs/usrm2-unified-plan.md`), rather than uniform
  random cubes, once that signal is available; uniform random masking is an acceptable first cut.
- **Expected gain**: by analogy to the closest-matching evidence (CNN-native MAE on a from-scratch
  architecture, in-domain data, low downstream label count): roughly **1-3 Dice-equivalent points**
  at low label counts, more on topology/thin-structure metrics (clDice-style) than on Dice itself,
  with the gain shrinking as the labelled fine-tune set grows past the 42-scroll target — i.e.
  treat this as most valuable for the *early* rungs of the unified ladder and for the recto-first
  scoping phase (fewer labels), and re-evaluate whether it is still worth the pipeline complexity
  once the 42-scroll label set is large.
- **Cost**: cheap relative to the corpus available — no new labels, reuses the existing sample
  loader and rung structure, needs only a mask-and-reconstruct head swapped in ahead of the real
  training run and then discarded (encoder weights kept). The real cost is engineering/validation
  time and a second training run's worth of compute, not data acquisition.
- **Pitfalls to watch, in priority order**: (1) domain gap is *low* risk here specifically because
  we're proposing in-domain pretraining (scroll CT on scroll CT), not transfer from a CT/medical
  foundation model — the negative-transfer literature's warnings are about *cross*-domain transfer,
  which we are deliberately avoiding; (2) architecture — stay CNN, do not switch to ViT/Swin to
  make a paper's exact recipe drop in, per section 3; (3) scan-identity diversity, not just voxel
  count — if the unlabelled pretraining corpus is dominated by very few distinct scroll scans, the
  literature's scale-driven gains (keyed to thousands of independent acquisitions) may not
  materialize; audit how many *distinct* scans the "hundreds of TB" actually spans before assuming
  the data-scale argument applies; (4) do not expect this to substitute for the 42-scroll labelled
  fine-tune — every source here treats pretraining as a head start for fine-tuning, not a
  replacement for in-domain supervision, and the reported gains shrink as label counts rise; and
  (5) an unvalidated pretext task is itself a project risk — tsm's own DINO distillation shows
  that an implemented-but-never-run SSL ablation produces zero evidence either way, so this should
  ship with an actual from-scratch-vs-pretrained ablation on one rung before being adopted
  project-wide, not assumed to work by analogy to CT literature alone.

## Sources

- Tang et al., "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis," CVPR 2022. https://www.researchgate.net/publication/359507351
- Wald, Isensee et al., "Revisiting MAE pre-training for 3D medical image segmentation," CVPR 2025 / arXiv:2410.23132. https://arxiv.org/abs/2410.23132
- "Hi-End-MAE: Hierarchical encoder-driven masked autoencoders," arXiv:2502.08347. https://arxiv.org/pdf/2502.08347
- "Swin MAE: Masked Autoencoders for Small Datasets," arXiv:2212.13805. https://arxiv.org/pdf/2212.13805
- Wu et al., "VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis," CVPR 2024 / arXiv:2402.17300. https://github.com/Luffy03/VoCo
- Zhou et al., "Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis," Medical Image Analysis 2021. https://www.sciencedirect.com/science/article/abs/pii/S1361841520302048
- "Vision Foundation Models for Computed Tomography" (CT-FM), arXiv:2501.09001. https://arxiv.org/abs/2501.09001 ; https://github.com/project-lighter/CT-FM
- "SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images," arXiv:2310.15161. https://arxiv.org/pdf/2310.15161
- "Merlin: a computed tomography vision-language foundation model and dataset," Nature 2026 / arXiv:2406.06512. https://arxiv.org/html/2406.06512v1
- He et al., "VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging," CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/papers/He_VISTA3D_A_Unified_Segmentation_Foundation_Model_For_3D_Medical_Imaging_CVPR_2025_paper.pdf
- Conrad & Narayan, "CEM500K, a large-scale heterogeneous unlabeled cellular electron microscopy image dataset for deep learning," eLife 2021. https://elifesciences.org/articles/65894
- "RETINA: Reconstruction-based pre-trained enhanced TransUNet for EM segmentation on CEM500K," PLOS Comp Bio 2025. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013115
- "Segment Anything for Microscopy" (micro-SAM), Nature Methods 2024/2025. https://www.nature.com/articles/s41592-024-02580-4
- "VAMAE: Vessel-Aware Masked Autoencoders for OCT Angiography," arXiv:2604.06583. https://arxiv.org/pdf/2604.06583
- "Benefit from public unlabeled data: A Frangi filter-based pretraining network for 3D cerebrovascular segmentation," Medical Image Analysis 2024. https://www.sciencedirect.com/science/article/abs/pii/S1361841524003670
- "Mitigating Overfitting in Medical Imaging: Self-Supervised Pretraining vs. ImageNet Transfer Learning," arXiv:2505.16773. https://arxiv.org/html/2505.16773
- "Transfer or Self-Supervised? Bridging the Performance Gap in Medical Imaging," arXiv:2407.05592. https://arxiv.org/abs/2407.05592
- usrm2 internal: `docs/unified_design.md` sections 1-4, 14, 21-22; `docs/research/tsm_ideas.md` section 6 (feats/dino, unvalidated).
