# The fixed recipe: every default in `rvsm/config.py`, and what justifies it

`rvsm/config.py` holds ONE flat `Config` dataclass. The first block is what a run may sensibly change;
everything after it is the **fixed recipe** — the v2 union of `usrm2/docs/unified_design.md` §26.6 and
§29.9 — which is a default from the first commit rather than a research flag (plan Context, §4). A field
is still a field so that `small_cfg` can shrink it for the tests and so that `<out>/config.json` records
exactly what ran, but nothing in the pipeline chooses between recipes at runtime.

Citations: `§n` = `/home/forrest/usrm2/docs/unified_design.md`; `research/<file>` = the copies in
[`research/`](research/); `plan §n` = [`plan.md`](plan.md). Measured numbers are in
[`rationale.md`](rationale.md) §2.

---

## 1. Tunables (a run may set these)

| field | default | why this default | reference |
|---|---|---|---|
| `ct` | `""` | required: the CT zarr URL or a local pyramid directory with integer level names. The whole premise is "raw CT + umbilicus in" | plan Context |
| `umbilicus` | `"auto"` | a published `umbilicus.txt` is not available for every volume; `auto` derives one from rung-9 per-z centroids. Per-scroll axes were a measured requirement for multi-source training | plan §2; `usrm2-streaming.md` |
| `out` | `"out"` | everything else is derived from it: the filesystem IS the bus between the two processes | plan §1, §2 |
| `size` | `"30m6"` | the size that produced every headline number (u1/u2/u3/u4 are all 30m6 at 256³); presets are `1m` (tests), `5m`, `15m`, `30m6`, `60m`, a factor-2 ladder | §19, §30; plan §5 |
| `patch` | `256` | the production patch; a 5m/128³ result establishes the *sign* of a change, never its final numbers | research/synthesis_v2_with_literature.md §5 cost anchors |
| `batch` | `2` | batch 3 was measured at +2 %/voxel and changes the optimisation; batch 1 + accum 2 is ~15 % slower than a real batch 2 | §14 (rejected list), §26.6 |
| `ckpt_act` | `1` | `0` fits an 80 GB card **with compile** (46.8 GiB; 67 GiB and OOM without); `1` is what a 48 GB card needs once Phase A adds ~0.6-0.8 GB; `1` is the default that fits both targets | §14, §26.5, §26.6 |
| `gpus` | `(0,)` | one card is the premise | plan Context |
| `mode` | `"auto"` | picks `resident` or `timeshare` from the cards found; the user asked for defaults tuned for both 80 GB resident and 32 GB timeshare | plan Context, §1 |
| `rounds` | `3` | round 0 is the teacher bootstrap; a round is discarded if it fails the gate, so the count is an upper bound, not a commitment | plan §1 |
| `steps` | `20000` | u3 passed the label ceiling on `merge_frac` at 21k; WSD makes extension free, so a shorter default with an explicit extension is the cheaper error | §2.1 of rationale; §26.2 |
| `workers` | `4` | sampler worker processes; the A100 ran 6 and went loader-bound at ~33k on a boxed mirror, which region streaming fixed | `usrm2-runs-state.md`; §17 |
| `tifxyz` | `""` | human meshes are **optional** and eval-only, by user decision | plan Context |
| `teacher_ckpts` | `{}` | teacher weights come from **local paths in the config**, by user decision — no registry lookup, no download at train time | plan Context |
| `cache_gb` | `64.0` | the CT shard cache budget; region mode fetches 1.37 bytes per training voxel at a 0.98 hit rate, so tens of GB is many hours of training | `usrm2-streaming.md`; §16 |

`FINGERPRINT_EXCLUDE = ("steps", "eval_every", "workers", "gpus", "rounds")` — a resume may be longer, may
evaluate at a different cadence, and may run on different hardware; everything else must match, which is
usrm2's `grow` tuple made explicit (§26.2).

---

## 2. The ladder

| field | default | why | reference |
|---|---|---|---|
| `ctx` | `(1..9)` | nine coarser context cubes, fixed, pooled past the pyramid top; no finer slots, no ignore channel — the user's decision of 2026-09-19 | `usrm2-unified-plan.md`; §2 |
| `rungs` | `(2..11)` | rung k = 0.6·2^k µm; 2.4 µm (Paris 4's native pitch) is rung 2, and 11 is 1.23 mm | §1 |
| `rung_boost` | `{2: 2}` | rung 2 carries the face detail everything downstream reads; u4/u5 ran exactly `--rung-boost 2=2`. The large coarse boosts u1 used (8=4, 9=16, 10=40, 11=80) are **not** carried over: §18.2 measured that they burn a rung's whole budget in the first few hundred windows | §26.6, §29.9, §18.2 |
| `region` | `1024` | the shard IS the unit: a volcomp level's object is a 1024³ shard of 512 inner 128³ chunks, and at rung 2 a region origin is a multiple of 1024, so the walk's tiles are the shard grid | §16, §18.1 |
| `windows_per_region` | `128` | a 1024³ region is 64 windows' worth of volume at 256³. Measured: plain random windows fetch 39.5 B/voxel at a 0.24 hit rate and stall; `--region 1024 --windows-per-region 64` fetches 1.37 B/voxel at 0.98. **128 since 2026-09-22 (tnr-0):** a region costs the producer ~1 min of A100 per unit (teacher pass, then the verso pass, then the fields), so drawing each region twice as deep halves the rate the producer must sustain; `T_train` per region doubles, and `L = ceil(T_produce/T_train)·K_active + 4` (re-estimated from the logs) shrinks with it | `usrm2-streaming.md`; §17 |
| `visits_max` | `64` | `--walk mix` gives a region `round(w·regions)` visits of weight `w/visits`, capped here, so a group's share of the walk is its intended share all the way through instead of only in the prefix | §18.2 |
| `air_keep` | `0.1` | the only rejection rule rvsm keeps from usrm2's `_rung_draw` (the `fg_min`/`dense_pow` rules needed a mask and are dropped); pure-air windows still teach "no sheet here", so they are kept at a low rate rather than dropped | plan §5 `sample.py` |
| `occ_min_fine` / `occ_min_coarse` | `0.05` / `0.01` | `tiles_fraction` is the block **mean** of `ct_level > 0`, replacing the mask's block max (there is no mask); draw weight ∝ √fraction | plan §3; rationale §3.10 |
| `heldout` | `8` | 8 rung-2 regions by seed, stratified over z and radius, occupancy ≥ 0.5, excluded from the walk at **every** rung. Eight is set by the CI: the val box's 11 surfaces already give 4-10-point-wide 95 % CIs, so a smaller held-out set could not falsify a round gate | §25.6; plan §3 |

---

## 3. Optimisation

| field | default | why | reference |
|---|---|---|---|
| `lr` | `3e-4` | the peak LR of every u-series run; `--lr 3e-4` appears verbatim in both recommended flag sets | §26.6, §29.9 |
| `warmup` | `200` | cold-start linear warmup | §26.2 |
| `rewarm` | `800` | resuming at a decayed tail generalises worse than starting fresh (Ash & Adams); the literature's window is 1-5 % of the new budget, and 800/50 000 is 1.6 % | research/lit_optimisation_schedules.md; §26.2, §26.6 |
| `new_param_lr_mult` | `3.0` | a second AdamW param group at 3× LR through the stable phase and 1× during the cooldown, holding the tensors a warm start GREW (the stem when `cin` grows, the head when `cout` grows). Cheap here: the head is 1×1×1 and the stem is one 3×3×3 out of ~200 tensors | §26.2, §26.6 |
| `sched` | `"wsd"` | warmup-stable-decay: the step budget need not be committed at run start, which is how these runs are actually managed (u1→u2→u3→u4 were all warm starts or extensions). Its value is **optionality, not accuracy** — experiment 9's decision rule is "adopt WSD if it *matches* cosine at matched steps" | research/lit_optimisation_schedules.md §1; §26.2; research/synthesis_v2 §5 exp 9 |
| `cooldown` | `0.1` | the literature's 10 %; §26.2 explicitly says to treat 10 % as a starting guess and check 5/10/20 % cheaply | research/lit_optimisation_schedules.md; §26.2 |
| `ema` / `ema_k` | `"auto"` / `50` | `ema_decay = 1 - K/steps`, clamped to [0.9, 0.9999]: the averaging window is `steps/K`, so K=50 is 2 % of the run — the middle of the literature's 1-3 % — and it *stays* 2 % when the run is extended. A fixed 0.999 is 1.7 % of a 60k run and 0.5 % of a 200k one. Caveat from the same survey: a flat EMA val curve is not proof of convergence; use raw-weight dice for a cooldown trigger | research/lit_optimisation_schedules.md; §26.2 |
| `aug` | `"full2"` | ranges centred on the scan metadata: blur sigmas in **microns** not voxels (the same config is a different physical blur at every rung), `_paganin_jitter` re-filtering at a different δ/β, shuffled artefact order. Evidence it matters: histogram-matching PHerc1667 CT to Paris 4 before the teacher raises teacher recall@4 from 0.48 to 0.62 and removes its bias — the teacher is sensitive to the per-volume uint8 rescale window | §27, §27.1-27.3; `usrm2-findings.md`; research/lit_ct_physics_augmentation.md |
| `compile` | `True` | not a speed knob: `--ckpt-act 0` needs 67 GiB and OOMs without compile, and 46.8 GiB with it. Inductor fuses GroupNorm+SiLU, which eager runs at 50 GiB/s against a 2300 GiB/s memory bound | §14, §14b |

**Not a config field, but part of the recipe** (inherited from the ported `model.py` / `train.py`): NCDHW
memory format (`channels_last_3d` is a pessimisation, 96.2 ms vs 15.7 ms for compiled GroupNorm+SiLU) and
`model.up2x` in place of `F.interpolate(trilinear)` at exact 2× (backward 1762 → 651 ms). See
rationale §2.3.

---

## 4. Cascade and conditioning planes

| field | default | why | reference |
|---|---|---|---|
| `cascade` | `"mix"` | the rung-(k+1) prediction, upsampled 2×, as one extra input channel. `mix` alternates the coarse TARGET and the model's own coarse PREDICTION as the source. Measured step cost: mask 1.07×, self 1.54×, mix 1.35× | §22 |
| `self_p_lo` / `self_p_hi` | `0.1` / `0.7` | annealed scheduled sampling. A fixed 0.5 is textbook exposure bias (Bengio 2015 / OneSeg); 0.1 → 0.7 is the suggested range. The one-level truncation of the self feedback (the coarse pass's own cascade channel is always zero) is deliberate: unconstrained multi-step self-feedback diverges without damping | §26.1; research/synthesis_v2 §6.2 |
| `cascade_drop` | `0.1` | the model must not become dependent on a channel that is absent at the coarsest rung of an inference cascade | §22 |
| `cascade_noise` | `True` | jitter on the cascade channel, same exposure-bias reasoning | §22 |
| `planes` | `"radius+meta"` | **radius** (1 plane): the radial *vector* gives direction, not how far out a voxel sits, and how far out it sits is what sets sheet spacing, curvature and damage. Built by `prep.radius_t` from the *same* axis interpolation `prep.radial_t` uses, so the direction and distance channels can never disagree about where the axis is. **meta** (5 planes): energy, log10 δ/β, unsharp σ in µm, sample-detector distance, pixel pitch, each min-max normalised over a documented corpus range — they restore the absolute intensity the per-window z-score throws away. Canonical order `radius, meta`, so a checkpoint is unambiguous. Cost: +2.3-3.5 %, ~73 MiB | §21, §29.5, §29.7 |

Explicitly **not** planes: angular position about the axis and absolute z (they break the 48-symmetry
augmentation and carry little); any local filter — sharpen, Laplacian, Sobel, sheetness, structure-tensor
normals — because the first 3×3×3 layers learn them (§21).

---

## 5. Heads

`channels = ("recto", "verso")`, `aff_offsets = (8, 16, 32)`, giving
`[recto, verso | midline, thickness | logvar | aff8_{z,y,x}, aff16_*, aff32_*]` = **cout 14**, and a stem
of `[CT, ctx_1..9, cascade, radius, meta×5, scale, rz, ry, rx]` = **cin 21**.

| field | default | why | reference |
|---|---|---|---|
| `channels` | `("recto", "verso")` | ONE head with two probability rows, not two heads: `cout 1 → 2` with `warm_start` copying the recto row into the verso row. The recto heads were measurably **not** hurt by adding verso rows | §23; `usrm2-findings.md` (r3 result) |
| `aff_offsets` | `(8, 16, 32)` | offsets are in voxels **at the sample's own rung**, and the measured Paris 4 sheet pitch of 15-35 voxels is a rung-2 number: 32 brackets rung 2, 16 rung 3, 8 rung 4 — the three rungs carrying almost all the sampling mass. A single offset brackets one rung only. Midpoint-centred and even, so a flip maps the channel to itself and a permutation only permutes the three axis channels: the set survives the 48-symmetry augmentation for free | §26.1, §26.6; research/lit_topology_merge_losses.md (connectomics affinities) |
| `i_mid` / `i_thick` | (derived) | the midline+thickness **pair**, not a face distance: the cortical white/pial literature moved decisively from penalties to construction (CortexODE / TopoFit / Coupled Reconstruction get non-intersection free from an invertible offset off a shared midline; DeepCSR needs a >30-minute-per-defect post-hoc topology fix; Vox2Cortex and PialNN are flagged crossing-prone) | research/lit_implicit_surfaces_manifold.md; research/synthesis_v2 §6.3; §29.2, §29.3 |
| `i_log` | (derived) | `--sdist-hetero`: one log-variance channel turns the Huber into a Gaussian likelihood `exp(-s)·huber + 0.5·s` (Kendall & Gal 2017). The model raises `s` where it cannot localise the surface — which is exactly the `conf` channel the tracer contract asks for, obtained free rather than as a separate head | §29.2 |
| normals | **derived, never stored** | three more channels buy nothing until something reads them, and the export derives `n = ∇d/|∇d|` by a Scharr kernel from the stored field anyway. Note this is a finite difference of the decoder's **output values**, not its autograd gradient — the SIREN pitfall the literature warns about | §29.2, §29.9; plan §2 |

`RUNG_ITEM_KEYS` pins the sampled item's keys so a key added without a producer fails loudly instead of
silently vanishing; `Layout` is the ONE place the stem and head order live, which is the class of bug
usrm2 repeatedly hit as `cin` and `cout` grew (rationale §3.9).

---

## 6. Losses

Every weight below was measured for its real step cost on the same card under the same protocol (whole
training step, two passes, 5m at 128³ batch 1, bf16, `--ckpt-act 0`, RTX 5080): §26.5 for Phase A,
§29.7 for Phase B/C.

| field | default | why | reference |
|---|---|---|---|
| `loss_excl` | `0.1` | soft exclusivity `relu(p_recto + p_verso - 1)`, masked to voxels where a verso store actually covers (so an untrained verso channel is not pushed to 0). Cost +1.6-2.5 %, i.e. inside the card's noise. Under `pair construct` it is **identically zero by construction**, so it doubles as a free assertion that the pairing works. Discipline: once exclusivity is a loss, `overlap` is no longer independent evidence | research/synthesis_v2 §Phase A (L3); §26.1, §26.5, §29.3 |
| `loss_selfcons` | `0.1` | `|avgpool2(p_recto) - avgpool2(CASCADE)|` over the SELF-source samples only, coarse side detached, one-way stop-grad (collapse risk). Free: it reuses the cascade channel the step already holds. Scored only on self-source samples because the `mask` source is the coarse TARGET, and a consistency term against it is a second, blurrier copy of the supervised loss. Cost -1.2 to +3.8 % | research/synthesis_v2 §Phase A (L4); §26.1, §26.5 |
| `loss_skel` / `skel_iters` | `0.05` / `4` | Skeleton Recall Loss (Kirchhoff/Isensee, ECCV 2024), hard skeleton built on the augmented target in N+1 pooling ops. It is a **recall of a point set derived from the target**, so it cannot be gamed by widening the band (a 3-voxel and a 9-voxel prediction score identically). Half the others' weight because it is the one term that can reward a merge bridge — a bridge is itself thin and connected. **Abort the arm if `merge_frac` rises**, whatever `continuity` does. Cost +0.8-1.7 % | research/lit_topology_merge_losses.md; §26.1, §26.5, §26.6 |
| `loss_affinity` | `0.1` | weighted BCE on the affinity rows; the only Phase-A term with a real cost (+5-12 % for two offsets, ~+8-15 % for three), because it is nine extra head channels plus nine BCE maps. Scored only where **both** ends are foreground, without which the channel is >95 % trivial "one end is air" | §26.1, §26.5, §26.6 |
| `loss_sdist` | `1.0` | the clamped Huber on the midline distance (δ = 2 voxels) is the localisation supervision, so it carries weight 1 and everything else is weighted relative to it | §29.2, §29.9 |
| `loss_eikonal` | `0.1` | `(|∇d| - 1)²` over the band the target says is within 8 voxels of a surface, one-voxel patch border dropped. IGR (Gropp et al., ICML 2020): the Eikonal residual is what makes a regressed field an actual distance function *between* the voxels that pin it down. It carries no localisation of its own — a regulariser, weighted well below the Huber. Measured cost: negative (inside noise) | research/lit_implicit_surfaces_manifold.md; §29.2, §29.7, §29.9 |
| `pair_band` / `pair_tau` | `1.5` / `0.5` | `band(u) = sigmoid((half - |u|)/tau)`, `p_recto = band(m - t/2)`, `p_verso = band(m + t/2)`, `t = TMIN + softplus(raw)` with `TMIN = 2·half`. §29.3 **proves** `relu(p_r + p_v - 1) ≡ 0` for every `(m, t)` with `t ≥ 2·half`: crossing is *unrepresentable*, not discouraged, at zero extra parameter cost. `TMIN = 3.0` voxels is also the physical floor — the measured median CT-march sheet thickness on Paris 4 is ~10 voxels (p90 25) | §29.3; `usrm2-findings.md` (verso 2026-09-18) |
| `loss_ect` / `ect_n` / `ect_rung` / `ect_block` | `0.05` / `1` / `2` / `64` | the fast-ECT topology pilot (arXiv:2507.23763's "fast χ", villa's `ect_loss.py` χ variant): per direction, eight products and eight `index_add_`s with fixed gradient-free bin indices, then a cumsum — no persistence diagram, no matching, no C++ dependency. **The crop is the load-bearing detail**: a topology loss on a cropped patch sees every sheet truncated at the patch face, so the term runs only on interior sub-blocks ≥8 voxels from every face, on a fixed stride, at rung 2 only. Cost: **+18 % at `ect_n 1`, +43 % at `ect_n 4`**, against experiment 7's own "< +20 %" budget | §29.4, §29.7; research/lit_topology_merge_losses.md |

### The one deliberate deviation from §29.9

§29.9's `u5` flag set **excludes** `--loss-ect`, for attributability: it is experiment 7, a separate arm,
and the only Phase B/C flag in the resume `grow` tuple precisely so it can be switched on later on an
existing checkpoint. rvsm turns it **on** at `ect_n 1` from the first commit (plan §4, stated there as
"the one deliberate deviation from §29.9, which excluded ECT only for attributability"). The reason is
§3.8 of [`rationale.md`](rationale.md): rvsm is not running ablation arms, so the attributability argument
does not apply, and +18 % of a step is the price of a topology term at the only rung where merges are
measured. A reviewer who disagrees should note that this is the single largest cost item in the recipe and
the easiest to switch off (`loss_ect = 0.0`).

### Near-axis and no-data weighting

Two rules that are not weights but change what the losses see, and which the review checklist calls out:

- **Weight 0 within 400 µm of the umbilicus axis** (`rvsm/targets.py:57 AXIS_R_UM = 400.0`) and above
  rung 4 for distance channels: near the axis the sheet geometry degenerates and the distance target is
  meaningless (§29.1, "weight 0 within 400 um of the axis / above rung 4").
- **A distance voxel counts only at full weight.** Spatial augmentations resample the target and the
  weight together, so a voxel on the boundary between real data and `code 0` comes out with a fractional
  weight *and* an interpolated value — and interpolating across code 0 (which decodes to -32 voxels, not
  to "nothing") gives a number that is simply wrong. `losses.dist_weight` **drops** a distance voxel whose
  weight is below 0.95 rather than down-weighting it (§29.2).
- **A distance is an isometry-only target.** Rotations and flips carry it unchanged; `scale`, `shear`,
  `elastic` and `sheetcomp` do not, because they change the metric. With distance heads on, those four
  must be dropped from the augmentation config for the run, loudly (§29.2). The 48 cube symmetries and
  every intensity augmentation are unaffected.
  **Not implemented in rvsm as of this writing**: `aug.PRESETS["full2"]` includes `SPATIAL` (`rot`,
  `scale`, `shear`, `elastic`) and `SHEETCOMP` (`rvsm/aug.py:618-624`), and nothing removes them although
  `loss_sdist = 1.0` is a default. See [`review_checklist.md`](review_checklist.md) §10.

---

## 7. Rounds and inference

| field | default | why | reference |
|---|---|---|---|
| `verso_source` | `"flip"` | flipping the *teacher* does not produce verso labels (mirror flips keep the band on the same face, dice 0.77-0.81); negating the radial channel of the *student* does move both heads to the far face. So verso comes from the student run with the radial sign flipped, never from a teacher symmetry | `usrm2-findings.md` 2026-09-16 / 2026-09-18; rationale §2.6 |
| `verso_after_steps` | `10000` | the unconditional fallback if the recto gate has not fired. u3 reached recall@4 0.822 at 21k and rung-2 dice 0.744 at 12k, so 10k is roughly where the recto becomes worth flipping | `usrm2-runs-state.md`; plan §1 |
| `eval_every` | `2000` | at 32 Mvox/s the periodic cost (`evaluate` 17-19 s + `val_png` 8.4 s + checkpoint 1.6 s) is 27 s per 500 steps = 54 ms/step amortised, ~4 % of a step. **2000 since 2026-09-22:** in `rvsm run` every evaluation also scores the held-out regions with a full student pass on the PRODUCER's card (8 regions × a 1024³ pass every 500 steps was ~20 % of the producer's GPU); calibration still runs after every evaluation | §14 |
| `calibrate` | `True` | one scalar temperature per rung, fitted by golden-section search on `log T ∈ [0.2, 5]` against the run's own held-out grid, minimising the same weighted BCE the training loss uses. It moves no weight. Dice-trained networks are measurably overconfident (Mehrtash, arXiv:1911.13273) and every uncertainty-gated mechanism downstream needs the sigmoid to be a probability first. A rung is fitted only when its target is a genuine binary band (`binary_frac ≤ 0.5`): above the native rung the target is a pooled fraction, and a temperature there fits a scale to a different quantity | §26.4 |
| `infer_window` / `infer_halo` | `256` / `32` | the 5090 production settings. Larger windows are worse *and* slower: windows 320/384 change the per-window z-score enough to give dice 0.93 against window 256, and TRT builders OOM at 384/512 on 16 GB | `runpod-5090-verso.md`; `usrm2-findings.md` |
| `cascade_depth` | `3` | top-down inference: three coarse-to-fine passes per region, the cascade channel of each level fed by the level above | §22 "Inference: top-down" |
| `tta` | `1` | measured gains sit inside the eval CI at 4-8× cost: teacher 4-flip TTA recall@4 0.851 vs 0.844, student 8-flip +1 pt, and on the 5090 sym8 gives dice 0.91 against plain. Not used in production there either | `usrm2-findings.md`; `runpod-5090-verso.md` |
| `min_regions_before_train` | `8` | cold start: the trainer must not begin on a single region's worth of windows | plan §1 |
| `lookahead_extra` | `4` | the "+4" of `L = ceil(T_produce/T_train)·K_active + 4`, re-estimated every 10 min from `logs/produce.jsonl`; A100 ~12, 32 GB cards ~8 | plan §1 |
| `train_min` / `produce_max_min` | `20.0` / `10.0` | timeshare phase lengths. Both processes stay alive across a swap, so the compile cache survives and a recompile on `.cuda()` costs ~1 min — under 5 % of a 20-minute phase | plan §1 |
| `reserve_gb` | `50.0` | production pauses below this much free disk; a region store is ~9-10 MB, and at most two rounds live on disk at once | `runpod-5090-verso.md`; plan §1 |
| `round_steps` | `20000` | a round ends here if the plateau fit (`< 2 %` remaining gain) has not ended it first | plan §1 |

---

## 8. The ordered experiment list, rewritten for rvsm

`research/synthesis_v2_with_literature.md` §5 lists 12 experiments in order. Nine of them are **already
decided** in rvsm — their outcome is a fixed default — so what follows is the residual list: what a
reviewer or an operator would actually run against a live rvsm, in order, with the flag or config edit,
the cost and the decision rule. Cost anchors (from the same section, and confirmed by
`usrm2-runs-state.md`): the A100 runs 30m6 at 256³ batch 2 at ~20-32 Mvox/s, so **10k steps ≈ 3-4.5
GPU-h**; a 5m/128³ run establishes the *sign* of a change, never its final numbers.

| # | experiment | status in rvsm | what to change | cost | decision rule |
|---|---|---|---|---|---|
| 1 | evalsurf upgrade (Phase A0) | **shipped as `rvsm eval`** — ceiling replaced by `compare_stores` on held-out regions plus optional `tifxyz` meshes; ERL, Betti-0/1, bootstrap CI, HD95 all ported | — | — | gating work only; every number quoted as `value (reference) [CI]` |
| 2 | L3 + L4 + annealed cascade self-p | **default on** (`loss_excl 0.1`, `loss_selfcons 0.1`, `self_p 0.1→0.7`) | to ablate: set each to 0 | 2 arms × 10k ≈ 9 GPU-h | keep a term only if its own target metric moves outside the CI |
| 3 | long-range affinity vs normal-gated repulsion | **affinity chosen**; repulsion (L1) never implemented | to ablate: `aff_offsets = ()` , `loss_affinity = 0` | 1 arm × 10k ≈ 4.5 GPU-h | `merge_frac` and `erl_merge_um` vs the round-0 reference CI |
| 4 | L8 skeleton recall | **default on** (`loss_skel 0.05`, `skel_iters 4`) | to ablate: `loss_skel = 0` | 1 arm × 10k ≈ 4.5 GPU-h | `continuity`/ERL up; **abort if `merge_frac` rises** |
| 5 | midline sdist + Eikonal vs a face sdist | **midline chosen** (§29.9: pick midline if equal on localisation) | `targets.region_fields --kind face` and a face-parameterised head | 2 arms × 20k ≈ 18-27 GPU-h | `offset_le3` 0.397 → >0.45 and HD95 down, `recall@4` not down |
| 6 | construction vs penalty pairing | **construction chosen** (`pair_band 1.5`, `pair_tau 0.5`) | to compare: drop the constructed pair, keep `loss_excl` as the penalty | 2 arms × 20k ≈ 18-27 GPU-h | `merge_frac`, pair completeness, held-out exclusivity |
| **7** | **topology-loss pilot: fast ECT** | **on at `ect_n 1`** — the one deliberate deviation from §29.9 | `loss_ect = 0.0` for the control arm | 2 arms × 10k ≈ 9 GPU-h | adopt only if `betti0_err` improves **and** step cost < +20 %. Measured cost is +18 %, i.e. at the limit. On a null result, switch it off — do **not** escalate to Betti matching |
| **8** | teacher fusion + per-source weights | fusion by agreement is **on**; GLC per-source weights are **not built** (plan §9) | none in v1 | ~0 GPU-h for the statistics | out of scope; revisit only if round 0 underperforms the recto teacher's own 0.843 recall@4 |
| 9 | WSD + EMA window + re-warmup | **default on** (`sched wsd`, `cooldown 0.1`, `ema auto`/`ema_k 50`, `rewarm 800`, `new_param_lr_mult 3`) | `cooldown` 0.05 / 0.1 / 0.2 | free, folded into an existing round transition | adopt WSD if it **matches** cosine at matched steps; its value is optionality |
| **10** | physics augmentation + metadata planes | **both default on** (`aug full2`, `planes radius+meta`) | to ablate: `aug = "geo"`, `planes = ""` | 2 arms × 20k ≈ 18-27 GPU-h | score on **cross-scroll variance**, not the aggregate mean — and note that rvsm v1 is single-volume, so this experiment cannot be run until a second volume is in scope (plan §9) |
| **11** | in-domain masked-cube pretraining | `rvsm pretrain` is planned (commit 7) and **not yet built** | `rvsm pretrain cfg.toml` then `rvsm train --init` | ~7 GPU-h pretrain + 2 × 10k fine-tune ≈ 16 GPU-h | adopt only if the gain survives at the label counts actually available; **audit how many distinct scans the pretraining corpus spans** — voxel count is not scan diversity |
| **12** | size ladder, fitted per rung | `rvsm ladder --sizes 15m,30m6,60m` is planned (commit 7) and **not yet built** | `rvsm ladder cfg.toml --sizes 15m,30m6,60m` | 3 runs × 20k ≈ 35-50 GPU-h | fit `1 - dice` vs `log(params)` **per rung**; call saturation only when the slope flattens across ≥3 sizes **and** the train/val gap grows |

The four rows in bold are the only ones that are genuinely open for rvsm v1: **7** (the ECT deviation),
**10** (untestable until a second volume is in scope), and **11**/**12** (unbuilt). Everything else is a
settled default whose ablation is available but not scheduled.

## 9. Things the recipe deliberately does not include

From `research/synthesis_v2_with_literature.md` §3 ("Dead — do not re-try as-is") and §7 ("The things not
to do"), each already reflected in `config.py` by absence:

- **No single signed field** collapsing recto and verso. The sign carries the *pairing*; a UDF's gradient
  is undefined at the zero set, which would poison the normals. (tsm's measured dead end;
  `usrm2-findings.md`: "keep recto classification target — signed single field was a dead end".)
- **No global winding or layer-index field**, learned or regressed. Every external field that tried —
  seismic RGT, OCT dynamic programming, tree rings, phase unwrapping — assumes tens of layers with
  generous spacing and degrades at density.
- **No feature distillation, no importing another model's dense fields as inputs.** The correct version of
  that instinct is in-domain masked-cube pretraining, which ships with an ablation (experiment 11).
- **No architecture chase**: no Mamba/attention backbone (controlled re-benchmarks attribute the wins to
  recipe confounds), no equivariant or steerable convolutions (2-5× cost for rotations the exact
  48-symmetry augmentation already covers), no hypernetwork conditioning while constant planes work.
