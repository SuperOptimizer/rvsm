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
| `batch` / `accum` | `1` / `2` | the optimiser step is batch 2's (batch 3 was measured at +2 %/voxel and changes the optimisation). **Since 2026-09-22 it is taken as 1 x accum 2:** on tnr-0 (A100 80 GB, Thunder) batch 2 at ckpt_act 1 peaked at 55.5 GB and OOMed inside the resident trainer budget once fragmentation was counted, while batch 1 x accum 2 at ckpt_act 0 peaks at 27.4 GB and runs 24-25 Mvox/s against batch 2's ~10 | §14; `rvsm/train.py` RVSM_PROFILE |
| `ckpt_act` | `0` | at batch 1 (above) there is room for every activation: backward 298 ms instead of 396 ms a step, peak 27.4 GB. `1` / `2` remain for bigger presets or smaller cards (fingerprint-excluded, so a run can change it on resume) | RVSM_PROFILE on tnr-0 |
| `gpus` | `(0,)` | one card is the premise | plan Context |
| `mode` | `"auto"` | picks `resident` or `timeshare` from the cards found; the user asked for defaults tuned for both 80 GB resident and 32 GB timeshare | plan Context, §1 |
| `rounds` | `3` | round 0 is the teacher bootstrap; a round is discarded if it fails the gate, so the count is an upper bound, not a commitment | plan §1 |
| `steps` | `20000` | the GLOBAL optimiser-step budget over every round (not per round); the round gate promotes only while `round_steps` of it are left, so a promoted round always gets to train. u3 passed the label ceiling on `merge_frac` at 21k; WSD makes extension free, so a shorter default with an explicit extension is the cheaper error | §2.1 of rationale; §26.2 |
| `workers` | `6` | sampler worker processes. On the 8-core tnr-0 the loader was the bottleneck once the step got fast; with the per-visit context super-cubes, the uint8 targets and the rung-3 pool a draw is ~0.8 s (rung 2-3) per worker, and 6 of them keep a ~0.7 s step fed | `usrm2-runs-state.md`; §17 |
| `tifxyz` | `""` | human meshes are **optional** and eval-only, by user decision | plan Context |
| `teacher_ckpts` | `{}` | teacher weights come from **local paths in the config**, by user decision — no registry lookup, no download at train time | plan Context |
| `cache_gb` | `64.0` | the CT shard cache budget; region mode fetches 1.37 bytes per training voxel at a 0.98 hit rate, so tens of GB is many hours of training | `usrm2-streaming.md`; §16 |
| `vram_train_gb` / `vram_produce_gb` | `50` / `26` | the resident split of an 80 GB card (scaled to the card's usable memory). The trainer peaks at 27.4 GB (batch 1, ckpt_act 0, 30m6), the compiled bf16 teachers at 13.4 GB; both are per-process memory fractions, so the slack is headroom against fragmentation, not waste | tnr-0 |
| `pin_memory` / `gpu_prefetch` | `False` / `True` | the trainer's next batch is fetched and copied on a helper thread during the current step. Pinned loader memory is off: on Thunder's A100 every run that pinned hung inside a CUDA call, and pinned H2D was slower there (2.2 vs 2.7 GB/s) | tnr-0 |
| `ct_seed` | `""` | a local, possibly partial mirror of a URL `ct`: shards it has are hard-linked into the cache (eviction removes only the link), a gap in a level its `mirror.json` marks complete is absent without a request, and only real gaps go to the origin. On tnr-0 the Paris 4 mirror has levels 1-9 complete and level 0 at 501 of 76,800 shards, so a region costs one level-0 GET. Outside the fingerprint: it changes where bytes come from, not what they are | 2026-09-22 |

`FINGERPRINT_EXCLUDE = ("steps", "eval_every", "workers", "gpus", "rounds", "ct_seed", "ckpt_act", "compile", "vram_train_gb", "vram_produce_gb", "pin_memory", "gpu_prefetch", "verso_min_dice", "self_p_mid_step", "self_p_end", "self_p_end_step", "cascade", "cascade_drop", "self_p_lo", "self_p_hi", "loss_prob_dice", "loss_pair", "verso_regen_gain", "round_min_steps_after_verso", "verso_min_regions", "ram_trainer_gb", "ram_host_exit", "fields_batch", "verso_regen", "ckpt_every")` — a resume may be longer, may
evaluate at a different cadence, may run on different hardware and may hold the verso fallback to a different quality floor; everything else must match, which is
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
| `cascade` | `"self"` | the rung-(k+1) prediction, upsampled 2×, as one extra input channel: the EMA net's OWN coarse prediction, or zeros (`cascade_drop`). **`self` since 2026-09-23 (was `mix`)**: `mix` also fed the coarse TARGET (`mask`), and the paris4 diagnostic at step 12000 showed the student reading the target out of that channel instead of the recto out of the CT (CT-only path flat, fg 0.27 / bg 0.24; mask path 0.88 / 0.13). `mix` and `mask` remain available; `dice_mask` in the eval still scores the mask source as a diagnostic. Fingerprint-excluded so paris4 could switch on resume. Measured step cost: mask 1.07×, self 1.54×, mix 1.35× | §22 |
| `self_p_lo` / `self_p_hi` / `self_p_end` @ `self_p_mid_step` / `self_p_end_step` | `0.1` / `0.7` / `0.9` @ `20000` / `30000` | used by `cascade = "mix"` only (`self` logs self_p 1.0). 0.1 at step 0, linear to 0.7 at 20k, linear to 0.9 at 30k, held (mask source the rest). Since 2026-09-23 (was 0.1 → 0.7 over the whole 60k run): paris4's self-cascade dice fell 0.15 → 0.05 over steps 4k-10k while the mask-cascade dice was 0.83, the student leaning on the mask source at self_p ≈ 0.2. The breakpoints are fingerprint-excluded. annealed scheduled sampling. A fixed 0.5 is textbook exposure bias (Bengio 2015 / OneSeg); 0.1 → 0.7 is the suggested range. The one-level truncation of the self feedback (the coarse pass's own cascade channel is always zero) is deliberate: unconstrained multi-step self-feedback diverges without damping | §26.1; research/synthesis_v2 §6.2 |
| `cascade_drop` | `0.3` (was 0.1) | the model must not become dependent on a channel that is absent at the coarsest rung of an inference cascade; 0.3 since the switch to `self`, so a third of the samples train the CT-only path directly | §22 |
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
| `loss_prob_dice` / `loss_pair` | `1.0` / `1.0` | the weight of the learned probability heads' soft dice (their BCE stays 1), and of the constructed pair's BCE + dice. Both fingerprint-excluded so a running experiment can retune them on resume; logged in every train row as `w_prob_dice` / `w_pair` (pass-3 P3-01). The pair terms are scored only where PAIRED support exists (midline target weight > 0), so in round 0 before the verso fields exist they are off; the ECT term chooses PER SAMPLED BLOCK: the constructed band only when paired support covers the whole block, the learned recto head otherwise (pass-4 P4-05); `pair_support` (the fraction of supported voxels) is logged per row | review 2026-09-23 |
| `loss_excl` | `0.1` | soft exclusivity `relu(p_recto + p_verso - 1)`, masked to voxels where a verso store actually covers (so an untrained verso channel is not pushed to 0). Cost +1.6-2.5 %, i.e. inside the card's noise. Under `pair construct` it is **identically zero by construction**, so it doubles as a free assertion that the pairing works. Discipline: once exclusivity is a loss, `overlap` is no longer independent evidence | research/synthesis_v2 §Phase A (L3); §26.1, §26.5, §29.3 |
| `loss_selfcons` | `0.1` | `|avgpool2(p_recto) - avgpool2(CASCADE)|` over the SELF-source samples only, coarse side detached, one-way stop-grad (collapse risk). Free: it reuses the cascade channel the step already holds. Scored only on self-source samples because the `mask` source is the coarse TARGET, and a consistency term against it is a second, blurrier copy of the supervised loss. Cost -1.2 to +3.8 % | research/synthesis_v2 §Phase A (L4); §26.1, §26.5 |
| `loss_skel` / `skel_iters` | `0.05` / `4` | Skeleton Recall Loss (Kirchhoff/Isensee, ECCV 2024), hard skeleton built on the augmented target in N+1 pooling ops. It is a **recall of a point set derived from the target**, so it cannot be gamed by widening the band (a 3-voxel and a 9-voxel prediction score identically). Half the others' weight because it is the one term that can reward a merge bridge — a bridge is itself thin and connected. **Abort the arm if `merge_frac` rises**, whatever `continuity` does. Cost +0.8-1.7 % | research/lit_topology_merge_losses.md; §26.1, §26.5, §26.6 |
| `loss_skel_prec` / `skel_prec_iters` / `skel_prec_gate` | `0.0` (off) / `3` / `0.3` | soft-clDice PRECISION (Shit et al., CVPR 2021), the complement of `loss_skel`: the PREDICTION's soft skeleton (clDice's min/max-pool thinning, `2·(iters+1)` pooling ops) must lie inside the target band dilated by one voxel, `1 - sum(S·B·m) / sum(S·m)`, on the same probability heads and level-0 output as `loss_skel` (so every rung `loss_skel` applies at). `m` = the loader weight, known within one voxel (`known_within(wv, 1)`), times a hard detached gate `p >= 0.3` so early-training haze (a diffuse low-probability field whose soft skeleton is everywhere) is not scored; a channel with no gated skeleton is not averaged in. Off by default so existing runs are unchanged; fingerprint-excluded and in `LOSS_SWITCH_FIELDS`, so it can be switched on at a resume (a `loss_switch` sched line, from 0 even against a config.json that predates it); logged as `skel_prec`. Cost: ~0.55x `loss_skel`'s (CPU, below) | `losses.skel_precision`; `tests/test_losses.py::test_skeleton_precision_*` |
| `loss_affinity` | `0.1` | weighted BCE on the affinity rows; the only Phase-A term with a real cost (+5-12 % for two offsets, ~+8-15 % for three), because it is nine extra head channels plus nine BCE maps. Scored only where **both** ends are foreground, without which the channel is >95 % trivial "one end is air" | §26.1, §26.5, §26.6 |
| `loss_sdist` | `1.0` | the clamped Huber on the midline distance (δ = 2 voxels) is the localisation supervision, so it carries weight 1 and everything else is weighted relative to it | §29.2, §29.9 |
| `loss_eikonal` | `0.1` | `(|∇d| - 1)²` over the band the target says is within 8 voxels of a surface, one-voxel patch border dropped. IGR (Gropp et al., ICML 2020): the Eikonal residual is what makes a regressed field an actual distance function *between* the voxels that pin it down. It carries no localisation of its own — a regulariser, weighted well below the Huber. Measured cost: negative (inside noise) | research/lit_implicit_surfaces_manifold.md; §29.2, §29.7, §29.9 |
| `pair_band` / `pair_tau` | `1.5` / `0.5` | `band(u) = sigmoid((half - |u|)/tau)`, `p_recto = band(m - t/2)`, `p_verso = band(m + t/2)`, `t = TMIN + softplus(raw)` with `TMIN = 2·half`. §29.3 **proves** `relu(p_r + p_v - 1) ≡ 0` for every `(m, t)` with `t ≥ 2·half`: crossing is *unrepresentable*, not discouraged, at zero extra parameter cost. `TMIN = 3.0` voxels is also the physical floor — the measured median CT-march sheet thickness on Paris 4 is ~10 voxels (p90 25) | §29.3; `usrm2-findings.md` (verso 2026-09-18) |
| `loss_ect` / `ect_n` / `ect_blocks` / `ect_rung` / `ect_block` | `0.05` / `1` / `1` / `2` / `64` | `ect_n` is the number of ECT DIRECTIONS and `ect_blocks` the interior blocks per sample, drawn at random per sample from a generator seeded by the step (a block with no target weight is never drawn). Until 2026-09-23 `ect_n` was passed as the block count, the direction count stayed 1, and the block was always the first of the grid (review T07). the fast-ECT topology pilot (arXiv:2507.23763's "fast χ", villa's `ect_loss.py` χ variant): per direction, eight products and eight `index_add_`s with fixed gradient-free bin indices, then a cumsum — no persistence diagram, no matching, no C++ dependency. **The crop is the load-bearing detail**: a topology loss on a cropped patch sees every sheet truncated at the patch face, so the term runs only on interior sub-blocks ≥8 voxels from every face, on a fixed stride, at rung 2 only. Cost: **+18 % at `ect_n 1`, +43 % at `ect_n 4`**, against experiment 7's own "< +20 %" budget | §29.4, §29.7; research/lit_topology_merge_losses.md |

**Why a skeleton PRECISION term beside the recall.** `loss_skel` is a recall of the target's skeleton:
it charges a GAP (a target medial voxel the prediction does not cover) and nothing else. A spur that
leaves the sheet, a fin into the inter-wrap air or a bridge to the next wrap is thin and connected and
costs the recall nothing -- which is why §6 says to abort a `loss_skel` arm when `merge_frac` rises.
`loss_skel_prec` scores the other direction: the prediction's own (soft, differentiable) skeleton must lie
inside the target band plus one voxel, so every spur or bridge voxel outside the band lowers it, while
widening the band inside the target, a one-voxel offset and a sheet that STOPS SHORT all keep precision
~1 (stopping short is recall's job; `test_skeleton_precision_does_not_charge_a_sheet_that_stops_short`).
Together they are clDice's two halves with the recall side kept hard and cheap. Measured cost (CPU,
one thread, 64^3 patch, two probability channels, fwd + bwd, min of 20 on a shared host, 2026-09-25):
`skel_precision` (3 iterations) +0.92 s against `skel_recall`'s (4 iterations) +1.69 s and a 1m
model step's 14.5 s -- CPU `max_pool3d` is slow, so both look larger there than the GPU's +0.8-1.7 %
for `loss_skel`; the precision term is 8 differentiable 3^3 pools plus 2 for the band and the known
mask, the recall 5 plus the known mask, so expect it in the same range on the card.

### Thinned band target (`thin_band`, `thin_band_width`, `thin_band_soft`)

Off by default (`thin_band = 0`: the targets are bit-identical to before; the code is not entered).
In an m7-only run the rung-2 recto target is m7's 9.6 µm output upsampled 4x: a soft band 4-8 voxels
wide. BCE / dice pull the student toward that blur while the thin-sheet terms (`loss_skel`,
`loss_skel_prec`, the constructed pair, exclusivity, the thickness head) pull it toward a sharp sheet, and
the two fight: paris4's rung-2 scores were flat. The thinned band is **a fittable sharp target that
carries m7's continuity**: every sheet m7 found, including the ones the 2.4 µm teacher misses, but as a
sheet the student can actually draw.

**Definition** (`losses.thin_band`, `losses.thin_apply`; computed in the trainer right after the
augmentation, on the device, for every rung-2 sample of every step; nothing is cached across steps and
no store changes). With `t` the recto target probability after the augmentation, `B = t >= 0.5`, `M` the
medial surface of `B` (`targets.medial_torch`, exact Euclidean EDT -- the same code path as the
evaluation's `dice_recto_r*_thin`), `d` the exact Euclidean distance to `M` (`edt.edt2`), `w =
thin_band_width` (4) and `s = thin_band_soft` (1):

- target `t' = clamp(1 - (d - w/2 + s)/s, 0, 1)` inside `B` (1 within `w/2 - s` of `M`, fading to 0 at
  `w/2`), `t' = 0` outside `B`;
- weight: inside `B` with `t' = 0` (the band's flanks) **0** -- not punished either way; on the thinned
  sheet the loader's weight; outside `B` the loader's weight (background supervision as before); a
  weight that was 0 (unknown) stays 0.

It replaces the recto ROW, so every term reading that row sees it: BCE, dice (`loss_prob_dice`), the
constructed pair's BCE / dice, ECT, skeleton recall / precision, affinity. The skeleton of the thinned
sheet is the band's medial surface, i.e. the same skeleton the recall term had. **Rungs >= 3 are
unchanged** (there the band is 1-2 voxels wide already; a rung-3 variant at half the width was not
added). **The distance heads keep their own targets**, and that is consistent rather than a conflict:
the `midline` / `thickness` fields (§6 "Distance-field targets") are already built from the medial
surfaces of the stored recto and verso bands, so their zero level IS `M`; the thinned sheet is centred on
the same surface. They do not exist without a verso (paris4 under `VERSO_HOLD` has none, so the pair and
distance terms are off there anyway).

**With routing** (`teacher_route`, §7): only the GAP changes. There (`c_A = 0` inside the dilated base
band) the thinned base band `(thin(tb), w · keep)` becomes a DIRECT BCE / dice target where the routing
alone had none; the skeleton recall from the band and the band penalty are as before; `c_A = 1` voxels
(the fine teacher's own sharp faces) and the `outside` zone keep exactly the routing's targets. An
unrouted voxel of a rung-2 sample (a region the regeneration has not reached; its recto is still m7's
band) gets the standalone thinning.

**Placement.** The thinned sheet sits at the middle of m7's band, which is m7's coarse estimate of the
face: ±2-4 rung-2 voxels from the true one. The thinning makes the target sharp, not more accurate;
routing is what fixes the placement where the 2.4 µm teacher is confident, and the two combine (thin
m7 sheets in the gaps, the fine teacher's faces where it is sure).

**Logging and switching.** Every train row at a rung-2 sample logs `thin_zero`, the share of the recto
weight the thinning zeroed (the flanks where the weight was > 0: the standalone / unrouted path), and
`thin_gain`, the share of the recto weight after the transform that it ADDED (the routed gap's direct
target); a routed config also logs `thin_gap_w`, the share of gap voxels that now carry a direct term
(the rest are m7's flanks, left at 0). **Under full routing `thin_zero` is 0.0 by construction**: a gap
voxel's weight is already 0 before the transform (`route_apply`), so zeroing a flank there changes
nothing it counts; the gap's activity shows in `thin_gain` / `thin_gap_w`. The three fields are fingerprint-excluded and in
`LOSS_SWITCH_FIELDS`, so a resume may switch it on (a `loss_switch` sched line). The evaluation is
unchanged: the reference stays the raw band, and `dice_recto_r2_thin` / `recall_recto_r2_band` /
`precision_recto_r2_band` already score a thin student against it.

**Cost** (RTX 5080 laptop, a synthetic m7-like band: wavy sheets every 22 voxels, 5-8 voxels wide,
soft edges, 30 % of the volume; `/home/forrest/rvsm_bench/thin/`): `thin_band` on one 256³ sample
is **9.4 ms** median (medial EDT + distance EDT, 0.59 GB peak), on one 160³ sample 1.6 ms. A whole
30m6 step at 160³ (the largest patch that fits the 16 GB card; accum 2, both microbatches rung 2) was
1.094 s off and 1.021 s on -- the difference is inside the run-to-run noise (the host was also running
the CPU test suite). Against tnr-0's ~0.35 s per 256³ microbatch the 9.4 ms would be ~2.7 % if the A100
were no faster at the EDT than the 5080; it is only paid on rung-2 samples. The exact EDT is used, not
an erosion approximation.

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

- **Weight 0 within 400 µm of the umbilicus axis** (`rvsm/targets.py:118 AXIS_R_UM = 400.0`) and above
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

### Distance-field targets (`paired-v3`)

The `midline` and `thickness` field stores (`rvsm/targets.py`, q0, code 0 = no data) are built per block
from the region's recto and verso probability stores. They exist at rungs 2-4. Rungs 3-4 are
recomputed from the pooled bands, never pooled. Each block uses the existing 48-voxel halo. The user
approved this definition on 2026-09-23 as the fix for review findings T01-T04
(`docs/production_readiness_review.md`). Pass-3 findings P3-06, P3-07 and P3-08
(`docs/paris4_pass3_review.md`) tightened it.

**Orientation.** Signs are pinned to `axis.radial`, the unit vector pointing away from the umbilicus.
The recto face is the **outward** face of a sheet (larger radius) and the verso face is the inward one.
Three places already use this convention:

- `losses.pair_bands` puts the recto band at `m = +t/2`.
- `export.SIGN_CONVENTION` points the normal from verso to recto, i.e. radially outward.
- `infer`'s verso trick (`sign = -1`) negates only the radial inputs.

Take a sheet with its recto face at radial coordinate `a` and its verso face at `a - t`. Then
`d_r = x - a` and `d_v = x - (a - t)`. So `t = d_v - d_r > 0` on either side of the sheet and inside
it. Both signed distances increase outward, so the two faces' normals `∇d` point the **same** way:
`n_r · n_v ≈ +1`.

**Per-rung bounds** (rung voxels):

| rung | REACH | TMIN | TMAX |
|---|---|---|---|
| 2 | 24 | 3 | 24 |
| 3 | 12 | 3 | 12 |
| 4 | 6 | 3 | 6 |

REACH and TMAX are both 57.6 µm at every rung. TMIN is the decoder's own floor at every rung, because
`soft_thickness = 3 + softplus` and the constructed bands have half-width 1.5.

1. **Raw distances.** `d_r` and `d_v` are unclipped signed EDTs to the medial surfaces of the two bands,
   over core + halo. The ±31.75 cap is applied **only by the encoders**, after all geometry.
2. **Reach and coverage.** A face counts only within REACH of the voxel, and the code asserts
   `REACH < halo`.
   - A block with no recto face within reach is code 0.
   - A region with no verso store, or with an empty verso band, is code 0 for **both** fields. There is
     no `midline = d_r` fallback.
   - The voxel must also be more than `max(|d_r|, |d_v|) + 2` voxels inside the region's stores.
     Zero-filled air outside the stores is unobserved, not background, so a nearer face could hide
     there.
3. **Thickness.** `TMIN ≤ t ≤ TMAX` must hold. A voxel outside that range is **rejected** (code 0),
   never clamped. Coarse rungs therefore deliberately lose thin-sheet support. For example, a 10-voxel
   rung-2 sheet is 2.5 voxels at rung 4 and gets no field target there.
4. **Same-sheet pairing.** Let `p_r` and `p_v` be the voxel's nearest recto and verso points. All three
   checks must pass:
   - **Reciprocal.** The nearest recto point of `p_v` must be in the same 26-connected recto band
     component as `p_r`, or within √3 of it. The same test applies to the nearest verso point of `p_r`
     against `p_v`.
   - **Normals.** Take the central-difference gradients of σ = 1.5-smoothed `d_r` and `d_v` at `p_r`
     and `p_v`. Each gradient's magnitude must be ≥ 0.4, `n · radial` must be ≥ 0.5 for both, and
     `n_r · n_v` must be ≥ 0.95.
   - **No intervening face.** Walk the segment `p_r → p_v` at ≤ 0.5-voxel spacing against both
     **bands**. The walk may leave the recto band once and enter the verso band once. Re-entering a
     recto band, or leaving a verso band after entering one, means another observed face lies in
     between. No distance threshold is involved. The walk uses the bands because a digital medial
     surface of a curved sheet has gaps a segment can slip through.
5. **Stencil and gradient.** `midline = (d_r + d_v)/2` and `thickness = d_v - d_r` are kept only if two
   conditions hold. First, the voxel and all six of its face neighbours pass rules 1-4, which is the full
   stencil the Eikonal term uses. Second, the unquantised midline gradient norm is within [0.8, 1.2].
   Everything else is code 0, and so is everything inside the 400 µm axis exclusion.

**Generation identity.** Each store records:

- `target_def`, the per-rung `reach_vox`, `tmin_vox` and `tmax_vox`, and `thr`;
- `recto_digest` and `verso_digest`: the source stores' `zarr.json` plus each shard's relative path and
  size. This identifies a store; it is not a content hash;
- a `support` histogram of why core voxels were rejected, with the same histogram per block in
  `support_blocks`.

`targets._current` is the one up-to-date predicate. `region_fields` skips a store on it, and the
producer's scheduler (`run._next_job`) asks `targets.fields_current`. A store from an older definition,
with other parameters, or from other source stores (a new verso generation, say) is therefore
regenerated through `stores.write`'s tmp-dir-then-rename. Code changes alone do not repair cached
labels, so this is how stale stores get rebuilt.

**Why.** These are the reviews' CPU reproductions:

- **T01:** an all-air recto block was written as code 128 over the whole block, i.e. as a *valid* zero
  distance.
- **T02:** for parallel recto/verso planes at x = 80/70, the true thickness is 10 everywhere. The old
  code clipped each distance to ±31.75 before combining them and gave thickness `[3, 10, 3]` at
  x = `[0, 75, 120]`. Far from the sheet both distances saturated to the same value, so their difference
  was 0, and the old `TMIN = 3` replaced it.
- **T03:** with recto at x = 50 and 80 and verso at x = 40 and 70 (two sheets), x = 65 got `midline 5,
  thickness 3`. The inputs were `d_r = 15` to one sheet and `d_v = -5` to the other, and the negative
  difference was silently clamped to `TMIN`. The nearest real paired midline is x = 75, at signed
  distance -10. That voxel is now either code 0 or `-10 / 10`.
- **T04:** without verso, the stored "midline" was the recto-face distance, so m = 0 lay **on** the
  recto face. The pair construction instead places the recto face at `m = +t/2`. The two objectives are
  incompatible, so midline supervision now requires paired geometry.
- **P3-06:** `paired-v2` admitted thickness 2 / 1 / 0.5 at rungs 2 / 3 / 4. Those values are below
  the decoder's 3-voxel floor and cannot be represented.
- **P3-07:** `paired-v2` accepted two bad pairs:
  - Touching wraps (recto x = 20 and 22, verso x = 18) paired the outer recto through the inner one.
    The +1.5 signed-distance threshold missed the 1-voxel gap.
  - An orthogonal pair (recto x = 20, verso y = 10, radial +x) passed, with a target midline gradient
    norm of 0.707.

  Both are now code 0; the tests `test_touching_wraps_are_code_zero` and
  `test_orthogonal_faces_are_not_a_pair` cover them.
- **P3-08:** the scheduler checked only `is_done` on the highest-rung midline, so a stale definition
  was never regenerated. A region's zero-filled edge was also treated as observed background. Now
  `_next_job` shares the writer's predicate, and the coverage rule applies. The tiling test compares
  one 256-wide region with two 128-wide ones, with a competing wrap across the seam. Wherever the
  tiling has a value, it equals the whole-volume value.

**Known limits.**

- Code-0 fields remove only the direct field, thickness and Eikonal supervision. The constructed-pair
  loss still sends gradients into `m`/`t` wherever recto has weight
  (`docs/review_evidence/pass3_pair_validity.py`). Gating that loss on paired support is a training
  objective decision, not a target one.
- On a curved sheet, the digital medial surfaces fragment. Thickness on the analytic annulus is within
  ~2 voxels, the midline within 1, and about 92 % of the ideal support survives.
- The coverage rule drops a band about REACH wide along every region face. Stitching neighbouring
  regions' labels would recover it.
- The source digest identifies stores by metadata and shard sizes, not content.

### Overlap-crop consistency (off by default: `overlap_p` 0, `loss_overlap` 0)

No other loss compares two crops of the same rung. The student's receptive field (~450 voxels) is
larger than its 256 patch, so near a patch face it predicts a sheet from half its usual context, and
at inference the halo only hides part of that. This term makes the student's prediction at a voxel
near its face agree with the EMA's prediction of the same physical voxel, seen from a window that
extends past that face.

| field | default | what |
|---|---|---|
| `overlap_p` | 0.0 | probability that a training draw also carries a second window (`sample.OV_KEYS`) |
| `overlap_sub` | 0 | EMA forward on a `q`^3 sub-crop of the second window (0 = the whole window) |
| `loss_overlap` | 0.0 | weight of the term (0 = not computed, whatever `overlap_p` says) |

All three are in `FINGERPRINT_EXCLUDE`, and `loss_overlap` is in `LOSS_SWITCH_FIELDS`, so a resume may
switch the term on or off and the change is logged.

- **Second window** (`sample.Patches._overlap`, `overlap_shift`). It comes from the same visit, shifted
  along one or more axes (each axis with room with probability 1/2, at least one). The shift is even
  and between p/4 and p/2 (64 to 128 at p 256), so the two windows share at least p/2 on every axis. It
  stays inside the visit's draw box, so its context cubes come from the same super-cube. It carries
  only the input cubes, the corner and the axis; there is no target and no weight.
- **EMA view** (`rvsm/overlap.py`). The EMA net runs in eval mode, under `no_grad` and bf16 autocast,
  before the student's forward. It is the cascade's self net when one exists (already synced every
  step and compiled), otherwise its own synced copy. The input is clean: no cube symmetry, no
  augmentation, and the window's own per-patch z-score. Its cascade channel has the same source as the
  first window's. It is zero if that channel was dropped or off. If it was the self source, it is a
  slice of the same rung-(k+1) prediction at S/4 + s/2, so there is no second coarse pass. If the first
  window took the `mask` source (`mix` mode), the sample gets no term.
- **Where it is scored.** The shared voxels where the EMA window's margin (distance to its nearest
  face) is larger than the student's. With a one-way, stop-gradient term, scoring voxels where the
  student sees more would teach it the teacher's truncation. On a full-crop shift this is about 15 % of
  the patch: the band next to the face that the second window extends past.
- **Frame.** The EMA heads are pasted onto the first window's voxels in the physical frame, then given
  the first window's cube symmetry. They are appended to the target stack, so every spatial
  augmentation moves them exactly as it moves the targets. Distances are carried in the store's 0..1
  code (`losses.encode_signed` / `encode_unsigned`), because the warp clamps target channels to 0..1.
- **Loss** (`losses.overlap_loss`, stop-gradient on the EMA side, like `self_consistency`).
  - Probability heads (recto, verso): one-way binary KL(EMA || student) = BCE(student logit, EMA p)
    minus H(EMA p). It is zero exactly when they agree, and its gradient is the soft-target BCE's.
  - Midline and thickness: L1 in voxels, only where the first window's distance weight is > 0. The
    distance heads are unconstrained elsewhere. They are regressions, and L1 is robust to a teacher
    that is locally wrong.
  - The distance part is scaled by `OVERLAP_DIST_SCALE` = 0.1 (1 voxel of disagreement is about 0.1).
- **Log.** `train.jsonl` rows carry `overlap` (the unweighted term's mean over the microbatches that had
  one), `overlap_kl`, `overlap_l1`, `overlap_vox` (the scored share of the voxels), `overlap_n` and
  `w_overlap`. Profiled runs (`RVSM_PROFILE=1`) add the `overlap_ema` and `overlap` phases.

**Cost** (laptop RTX 5080 16 GB, 30m6, batch 1 x accum 2, bf16, `gn_bf16` off, compile default, the
synthetic full-recipe items of the trainer bench, 60 steps, `RVSM_PROFILE=1`). The host was shared
with other test runs, so wall times drift by up to 30 % between runs. The within-run `overlap_ema` share
is the steadier number.

| patch | overlap_p | EMA crop | step s (baseline) | overlap_ema ms/step | share of step | scored voxels | peak alloc, backward phase |
|---|---|---|---|---|---|---|---|
| 144 | 0 | - | 0.761 / 0.769 | - | - | - | 6.24 GB |
| 144 | 0.25 | full 144 | 0.804 | 34 | 4 % | 15 % | 6.32 GB |
| 144 | 0.5 | full 144 | 0.831 | 52 | 7 % | 15 % | 6.35 GB |
| 144 | 1.0 | full 144 | 0.964 | 169 | 18 % | 15 % | 6.35 GB |
| 144 | 1.0 | sub 72 | 0.901 | 37 | 4 % | 1.4 % | 6.17 GB |
| 144 | 1.0 | sub 128 | 1.025 | 119 | 12 % | 10 % | 6.17 GB |
| 160 | 0 | - | 0.982 / 0.998 | - | - | - | 7.59 GB |
| 160 | 0.25 | full 160 | 1.046 (+6 %) | 45 | 4 % | 15 % | 7.71 GB |
| 160 | 0.5 | full 160 | 1.107 (+12 %) | 70 | 7 % | 15 % | 7.75 GB |
| 160 | 1.0 | full 160 | 1.229 (+24 %) | 203 | 17 % | 15 % | 7.74 GB |
| 160 | 1.0 | sub 80 | 1.262 (+27 %) | 44 | 4 % | 1.4 % | 7.99 GB |

The full-crop EMA pass costs about one cascade self pass: ~10 % of a step per microbatch that has one.
So the added step time is about 2 x `overlap_p` x 10 %, plus the extra target channels through the
augmentation. The sub-crop is **not** worth it. A half-edge sub-crop scores ~10x fewer voxels, with a
teacher margin of at most q/2. Every sub-crop run also slowed the student's own forward by 70-120 ms,
consistent with a second input shape on the `UNet.forward` code object that the training net and the
cascade net share. Peak allocated memory grows by the five-channel field: +0.1 to +0.16 GB at 144-160,
about +0.6 to +0.9 GB at 256. The EMA pass's own transient peak is the cascade pass's, and it runs at
the same point in the step.

---

## 7. Rounds and inference

| field | default | why | reference |
|---|---|---|---|
| `verso_source` | `"flip"` | flipping the *teacher* does not produce verso labels (mirror flips keep the band on the same face, dice 0.77-0.81); negating the radial channel of the *student* does move both heads to the far face. So verso comes from the student run with the radial sign flipped, never from a teacher symmetry | `usrm2-findings.md` 2026-09-16 / 2026-09-18; rationale §2.6 |
| `verso_after_steps` | `10000` | the unconditional fallback if the recto gate has not fired. u3 reached recall@4 0.822 at 21k and rung-2 dice 0.744 at 12k, so 10k is roughly where the recto becomes worth flipping. **Before** this step the gate is attempted at every evaluation, but its held-out pass is skipped while the evaluation's fine-rung (2-4, voxel-weighted) `dice` is more than `run.GATE_SCREEN = 0.1` below `verso_gate_dice` (the grid is tiles of the same held-out regions against the same round-0 stores, so such a pass cannot pass; it costs ~13 min of trainer time on tnr-0). When it does run it is in the trainer, on `GATE_REGIONS = 2` held-out regions: the student pass stays on the card (cascade included) and only its uint8 recto reaches the host, and `compare_stores` streams the reference store in haloed 256³ blocks (Betti numbers and ERL runs become per-block sums). ~2 GB of host RAM a region; the whole-region version OOMed tnr-0's 64 GB at paris4 step 2000 (`betti_error` alone passed 52 GB on one region, the host-side cascade 12 GB). The supervisor's RAM guard (`run.ram_guard`, every 5 s) pauses the producer above 85 % of MemTotal and resumes it below 75 % | `usrm2-runs-state.md`; plan §1; tnr-0 2026-09-23 |
| `verso_min_dice` | `0.3` | the `verso_after_steps` fallback fires only once the evaluation's RUNG-2 self-cascade dice of the RECTO head alone (`dice_recto_r2`, against the immutable recto grid, so a verso grid appearing on restart cannot change its meaning) reaches this on TWO consecutive evaluations: the row AT the step and the row at the immediately preceding distinct step, one row each, both of this code's `eval_schema` -- anything else fails closed (`run.eval_streak`; both values logged in the `verso_gate` record; the held-out comparison path keeps its own screen) -- before 2026-09-23 it read the fine-rung (2-4) dice of one evaluation; until then the gate keeps waiting and logs why. paris4's self-cascade dice FELL 0.15 -> 0.11 -> 0.07 over steps 4000-8000 and a verso pass from that student at 10000 would have written garbage verso labels. Fingerprint-excluded (it gates scheduling, not the math) | tnr-0 2026-09-23 |
| `ram_trainer_gb` / `ram_host_exit` | `40.0` / `0.90` | the supervisor's heartbeat also watches the TRAINER: its own RSS past `ram_trainer_gb`, or the host past `ram_host_exit` of MemTotal, sets `RAM_EXIT`; the training loop reads it every step, logs a `stop_now` record, checkpoints only if >= 200 steps have passed since the last checkpoint, and the run exits cleanly (`sched` `ram_exit` / `exit_ram`) instead of the kernel killing the host (paris4 step 24000: +280 MB a step, 6 -> 47.7 GB, host rebooted). Every save keeps the checkpoint it replaces as `_prev` (never twice for one step). Fingerprint-excluded | tnr-0 2026-09-23 |
| `producer_recycle_fields` / `producer_recycle_rss_gb` / `producer_mallopt` / `producer_arena_max` (producer RSS creep) | `0` / `0.0` / `True` / `2` | the producer's host RSS floor grew ~0.15 GB per GPU fields region on the A100 (smaps: ~270 glibc per-thread arenas, fed by the per-region `rvsm-fcut` / `rvsm-fwrite` threads churning 11-33 MB buffers; 33 MB is just under glibc's dynamic mmap threshold). `producer_arena_max` sets `mallopt(M_ARENA_MAX)` at producer start (an env `MALLOC_ARENA_MAX` wins); `producer_mallopt` sets M_TRIM_THRESHOLD 64 MB and M_MMAP_THRESHOLD 16 MB; every fields region then calls `malloc_trim(0)` and logs `rss_gb` / `rss_trim_gb` in its `fields` line (all no-ops without glibc). On the laptop repro (512^3 region, 5080, one fields thread) the floor went 1.39 -> 1.92 GB over 10 regions by default (saturating at 2.01 by ~15), 1.2-1.4 with no steady slope over 30 regions with arena max 2, flat at 1.17-1.20 after trim with arena max 2; the thresholds alone and the trim alone did not change the slope. On the A100 MALLOC_ARENA_MAX=2 only cut it to ~0.09 GB / region (the rest is suspected in Thunder's `libthunder.so` GPU RPC client, not fixable here), so the backstop is the RECYCLE: after `producer_recycle_fields` fields regions, or once its own RSS passes `producer_recycle_rss_gb` (checked between units; spawned producer only -- a cpu-mode thread shares the supervisor's RSS), the producer stops like STOP (the unit in flight and the queued fields finish, the writer drains, the cache closes), logs `sched` / `produce` `recycle` with its `rss_gb`, stamps `recycle` in `workers/produce.json` and exits 0. `ProducerWatch` respawns such a producer AT ONCE and leaves the backoff (`fails`, `next_at`, `born`) exactly as it was: a recycle never counts toward `RESTART_FATAL` (`sched` `restart` `reason: recycle`). Committed stores are never redone: the next producer reads the disk state machine. paris4: `producer_recycle_rss_gb = 22` under the 24 GB host watchdog. Fingerprint-excluded | A100 smaps 2026-09-26; `tests/test_run_e2e.py` recycle tests |
| `VERSO_HOLD` (marker, `rvsm verso hold|release --out DIR`) | absent | a manual hold on round 0's verso: while `<out>/VERSO_HOLD` exists the verso gate still computes and logs its decision every evaluation (`sched` `verso_hold`, `would_pass`, the streak values) but `verso_on` stays false; removing it lets the next evaluation decide normally. paris4 trains recto-only under it from 2026-09-24 (user decision) | user 2026-09-24 |
| `verso_regen_gain` | `0.15` | round 0's verso may be REGENERATED once: the first evaluation whose rung-2 dice reaches `verso_min_dice + verso_regen_gain` records `state.json` `verso_regen`, and the producer rewrites every verso store made by an older checkpoint (store attr `step`) as generation 1 (`region_<z>_<y>_<x>.g1.zarr`, a new directory beside the old one -- a finished store is never rewritten in place), then its fields at generation 1 (the writer builds from the newest finished verso, `targets.source_verso`); readers (`Catalog`, the stitched rung 3-6 targets, the rung-3 pools, the grid) use the region's COMMITTED bundle generation (`stores.bundle_gen`, `stores/round_0/bundle/region_*.json`), which the producer moves only once the new verso AND all its fields are finished -- a reader never mixes a new verso with old fields (pass-4 P4-04). The grid key includes the identity of every held-out label store a reader sees, and the driver refreshes the grid at an evaluation boundary when it changes (the recto reference grid directory is never touched). The trigger FREEZES the qualifying checkpoint (`ckpt/verso_regen.pt`, its sha256 in `state.json`) and persists the whole backlog (`stores/round_0/bundle/regen_backlog.json`: every gen-0 verso made by an older checkpoint); the producer works through it whenever its lookahead window has nothing for the GPU, independently of the walk, with the frozen checkpoint (a second student slot), and marks `verso_regen_done` when every backlog region's regenerated bundle is committed (pass-4 P4-06). Fingerprint-excluded | pass-3 item 11 |
| `teacher_ckpts` keys: the teacher set; the m7-only mode and the RECTO regeneration | `{}` = recto + m7 | the KEYS of `teacher_ckpts` pick round 0's teachers (`run.teacher_names`); the producer's bank builds only those (with `{"m7": path}` no recto weights are ever loaded). Two teachers: recto = the confidence-weighted fusion, rw = their agreement `1 - |p_recto - p_m7|`. ONE teacher: recto = its probability as it is (m7: the level-2 pass upsampled 4x and masked by its coarse air mask; the rung 3-6 pools and coarse rungs 7-11 then derive from m7's native 9.6 um output -- rung 4's 4x mean pool of a 4x trilinear upsample is a mild blur of it, not measured further -- so there is no separate "m7-native coarse" target) and rw = 1 (255) everywhere -- the loader's weight is already `inside x (CT > 0) x rw`, so air needs nothing from rw, and a self-confidence weight would change the bce's pull on undecided voxels rather than keep the loss. Every teacher store records `teachers` (a store without it is the old recto + m7 fusion) and `rw` ("agreement" / "ones"). A round-0 recto whose set differs from the configured one is REGENERATED: a non-blocking `reteach` pass writes recto + rw at the region's next free generation (`stores.next_gen`, one counter per region shared with the verso regeneration, so the fields rebuilt for either -- at `targets.field_gen` = max(verso gen, recto gen) -- land in a fresh directory); generation 0 is never touched. Readers use the COMMITTED generation (`stores.bundle_state`: `{"gen": fields, "verso": g, "recto": g}`), which `run.commit_sources` moves in two independent parts: the recto + rw pair AS SOON AS the reteach has written both (the loader, the grid and the gate reference switch at once: the fields are the pair / distance losses' input only and fields from the previous recto are the same sheet a voxel or two off -- gating the switch on the fields queue left paris4 on the old targets for > 1 day, changed 2026-09-25), and the fields with their verso once every field built from the newest verso and recto is finished (a verso never moves without its fields); `recto_commit` / `fields_commit` lines; a committed name is always a finished store; the grid's `grid_sources` see the new digests and rebuild that region's items; the coarse rungs are re-fed once per generation. Order: held-out regions first (the gate/eval reference, `heldout_rows` reads the committed recto), then the window's and leased regions beside the verso passes (after every blocking pass), then the rest in walk order whenever the window is idle (`_recto_backlog`, `recto_regen` lines with the stale rectos and the fields rebuilds remaining, `recto_regen_done`; the backlog keeps at most `RECTO_FIELDS_INFLIGHT` = 1 fields rebuild queued so the window's own fields go first, and a window region's rebuild is chained after its reteach). Superseded generations are kept. Fingerprint-excluded: **paris4 switched to m7 alone at ~step 56000 (user decision 2026-09-25)**, a deliberate mid-run change of the recto target definition; the resume logs a `sched` `teacher_switch` line (old/new sets and paths), and the loss weights of `config.LOSS_SWITCH_FIELDS` (continuity terms `loss_skel`, `loss_affinity`, `loss_ect`, `loss_selfcons` fingerprint-excluded the same day, values unchanged) get a `loss_switch` line when one moves | user 2026-09-25 |
| `round_min_steps_after_verso` / `verso_min_regions` | `8000` / `200` | round 0 is promoted only once verso has been on for this many steps (`state.json` `verso_on_step`) and this many verso stores are finished; the gate's held-out rows must be COMPLETE (a row missing a metric is dropped whole). Fingerprint-excluded | pass-3 P3-03 |
| `eval_every` | `2000` | at 32 Mvox/s the periodic cost (`evaluate` 17-19 s + `val_png` 8.4 s + checkpoint 1.6 s) is 27 s per 500 steps = 54 ms/step amortised, ~4 % of a step. **2000 since 2026-09-22:** in `rvsm run` every evaluation also scores the held-out regions with a full student pass on the PRODUCER's card (8 regions × a 1024³ pass every 500 steps was ~20 % of the producer's GPU); calibration still runs after every evaluation. **Since 2026-09-26** the validation PNGs are drawn from `evaluate`'s own forwards (`train.PanelCapture`: no second read of the panel items, no whole-grid pass to pick the region panels) and rendered on a background thread (`train._PngWorker`, one render at a time, waited for at exit up to `PNG_WAIT_S`, `val_png_wait` in train.jsonl): `eval_s.val_png` is the hand-off (~0 s; it was 290 s on the A100: 36 panel items re-read and re-run, plus the whole 192-item grid at the first evaluation of each `train()` call), `eval_s.val_png_bg` the previous evaluation's render seconds. Grid items are pickled with protocol 4 (`sample._save_item`): ~0.25 s instead of ~1.2 s of GIL-held load per 240 MB item; an existing grid is converted in place by `rvsm grid-repack <grid dir> --jobs N` (`rvsm/grid_repack.py`: atomic, skips protocol-4 files, waits while the trainer's `<grid dir>/.evaluating` marker is live) | §14 |
| `ckpt_every` | `1000` | a CHECKPOINT-ONLY boundary between evaluations (0: checkpoint at evaluations only): the same `save()` (student, EMA, optimiser, step, temperatures, `_prev` rotation) and the same `state.json` write (`run.checkpoint_state`: step, cursor, walk snapshot) as an evaluation, plus STOP, but no `evaluate` / `val_png` / calibration / gate. The hook sees `kind` "eval", "ckpt" or "stop" (the RAM guard's checkpoint, which now records its walk too, so a resume after it does not replay the steps since the last evaluation). Every save logs a `ckpt` line (`step`, `at`, `s`) in `train.jsonl`. A resume from a checkpoint off the eval cadence continues at the next step with the schedule by step and evaluates at the next `eval_every` multiple. Fingerprint-excluded | 2026-09-24 |
| `calibrate` | `True` | one scalar temperature per rung, fitted by golden-section search on `log T ∈ [0.2, 5]` against the run's own held-out grid, minimising the same weighted BCE the training loss uses. It moves no weight. Dice-trained networks are measurably overconfident (Mehrtash, arXiv:1911.13273) and every uncertainty-gated mechanism downstream needs the sigmoid to be a probability first. A rung is fitted only when its target is a genuine binary band (`binary_frac ≤ 0.5`): above the native rung the target is a pooled fraction, and a temperature there fits a scale to a different quantity | §26.4 |
| `infer_window` / `infer_halo` | `256` / `32` | the 5090 production settings. Larger windows are worse *and* slower: windows 320/384 change the per-window z-score enough to give dice 0.93 against window 256, and TRT builders OOM at 384/512 on 16 GB | `runpod-5090-verso.md`; `usrm2-findings.md` |
| `--backend trt` / `teacher_bf16` | `torch` / `True` | the teachers run bf16 + `torch.compile` (max-autotune-no-cudagraphs): recto 0.342 -> 0.122 s and m7 0.136 -> 0.062 s a window on tnr-0, a teacher region ~19 s of GPU. TensorRT is gated by a measured verdict (`<plan>.verdict.json`): an engine is used only when it beats the torch module on one window. **Follow-up:** on Thunder's virtualised A100 every timed build fails (the autotuner's `cuMemHostGetDevicePointer` is unsupported: default, level 3 and level 1 with a timing cache and 16 GB workspace all fail), and the untimed level-0 engine is 8.6x slower than torch (380 vs 44 s a region), so the verdict keeps torch. A plan built with timing on a real A100-SXM4-80GB (same GPU name) would deserialise here and should be tried against the compiled bf16 teachers | tnr-0, 2026-09-22 |
| `cascade_depth` | `3` | top-down inference: three coarse-to-fine passes per region, the cascade channel of each level fed by the level above | §22 "Inference: top-down" |
| `tta` | `1` | measured gains sit inside the eval CI at 4-8× cost: teacher 4-flip TTA recall@4 0.851 vs 0.844, student 8-flip +1 pt, and on the 5090 sym8 gives dice 0.91 against plain. Not used in production there either | `usrm2-findings.md`; `runpod-5090-verso.md` |
| `min_regions_before_train` | `8` | cold start: the trainer must not begin on a single region's worth of windows | plan §1 |
| `lookahead_extra` | `4` | the "+4" of `L = ceil(T_produce/T_train)·K_active + 4`, re-estimated every 10 min from `logs/produce.jsonl`; A100 ~12, 32 GB cards ~8 | plan §1 |
| `train_min` / `produce_max_min` | `20.0` / `10.0` | timeshare phase lengths. Both processes stay alive across a swap, so the compile cache survives and a recompile on `.cuda()` costs ~1 min — under 5 % of a 20-minute phase | plan §1 |
| `reserve_gb` | `50.0` | production pauses below this much free disk; a region store is ~9-10 MB, and at most two rounds live on disk at once | `runpod-5090-verso.md`; plan §1 |
| `round_steps` | `20000` | the MINIMUM a round trains before the round gate may promote it, counted from the round's own start (`state.json` `round_step`), not the absolute step. The gate fails closed (2026-09-23 review): round 0 needs `verso_on` (self-distillation needs round 0's verso stores); then held-out comparison rows with finite precision / betti0 error are required; round 0's stats become the reference (`state.json` `round_ref`, so a restart keeps the veto) and a round >= 1 with no reference, or worse than it beyond the bootstrap CI, is not promoted. The plateau fit reads only this round's evaluations and is logged beside the decision | plan §1; review 2026-09-23 |

### Per-rung teacher routing with gap-fill (`teacher_route`, `loss_band`, `band_dilate`, `band_eps`)

Off by default (`teacher_route = {}`: every rung's recto target comes from the teacher set of
`teacher_ckpts`, exactly as above; the fingerprint, the grid key, the sampler rows and the loss are
unchanged). On, it splits the recto target by rung:

```toml
teacher_ckpts = { recto = "/home/ubuntu/.cache/rvsm/surface_recto_3dunet.pth", m7 = "/home/ubuntu/.cache/rvsm/surface_m7_nnunet.pth" }
teacher_route = { "2" = "recto", "3" = "m7", "4" = "m7" }
loss_band = 0.1        # optional; 0 = the band penalty off (the gap-fill masks apply regardless)
```

- **Rung 2** recto targets come from the FINE teacher (the 2.4 µm `recto` 3D U-Net), trusted only where
  its coverage `c_A = 1`; **rungs ≥ 3** (and the coarse rungs 7-11) keep the BASE teacher (m7, native at
  rung 4; rung 3 its 2x-upsampled pool, as in the m7-only mode). `config.route_spec` accepts only this
  shape: rung 2 is the one routable rung, every other rung named must be the one base teacher (default
  `m7`), both must be keys of `teacher_ckpts` (checked at `setup` and by the `TeacherBank`).
- **Stores.** A routed round-0 generation holds `recto` (the fine teacher, q8), `band` (the base
  teacher's rung-2 probability, q8, a third member of the teacher bundle `stores.TEACHER_BUNDLED`, so it
  follows the committed recto generation) and `rw` = c_A (0/255, encoding `coverage_u8`), written in that
  order (rw's `done` finishes the generation). Every routed store records `route`
  (`r2=<fine>;band=<base>;cA-v1`), `teachers` = [fine, base], `band_source` (`teacher`, or `store:<path>`
  when the base probability was reused) and the coverage constants.
- **The coverage rule `c_A` (`infer.route_coverage_u8`, rule `cA-v1`).** `c_A = 1` where the fine
  teacher's p ≥ 0.6 (a confident face), or p ≤ 0.2 **and** within REACH = 24 rung-2 voxels (57.6 µm,
  `targets.REACH`) of one of the fine teacher's own faces (p ≥ 0.5): confident background where the teacher
  demonstrably resolves sheets. Everywhere else `c_A = 0`: undecided voxels (0.2 < p < 0.6) and
  "background" far from any face the fine teacher found -- which is where a sheet it missed would be. The
  reach is measured on a 4x max-pooled face mask dilated by 6 coarse voxels, so it is exact to ±3 voxels
  (a 256³ instead of a 49³-kernel dilation per 1024³ region).
- **Regeneration.** Turning the route on (or changing it, or bumping the rule) changes the store identity
  (`run.teacher_set`: [fine, base, `route:<sig>`] vs `run.store_ident`), so every round-0 recto is
  regenerated through the SAME `reteach` machinery as the m7 switch: the next free generation beside the old
  one (generation 0 is never touched), held-out regions first, then the window's and leased regions beside
  the verso passes, then walk order when the window is idle, committed by `commit_sources` once its rw is
  finished. A region whose COMMITTED recto was made by the base teacher alone (paris4's m7-only stores) --
  or a routed generation with the same base -- is reused as the band (`run.band_reuse`, decoded on the
  reader thread): **no m7 rerun**, only the fine teacher runs, and the coarse rungs are not re-fed (they
  already are the base teacher's). A never-produced region runs both teachers in its first `teacher`
  pass and feeds the coarse rungs from the band. The fine and base passes run one after the other on the
  one card, each quantised to uint8 before the next; like every teacher pass the unit holds the GpuGate
  for its whole forward, so it never overlaps a verso pass or the GPU fields. Turning the route off again
  regenerates back to the plain teacher set.
- **The backlog's GPU share (`reteach_share`, default 0.25, fingerprint-excluded).** The backlog outside
  the window used to run only when the window had NO GPU unit; once verso is on the window is never
  idle, and paris4 (2026-09-26, step 85000-89000) regenerated only the regions its window revisited,
  leaving ~1350 stale. Now, unless the pass still holds a round-r `self` or any unit of a LEASED region
  (a worker waits on those), backlog reteaches are admitted while the reteach share of the last ~20 min
  of producer GPU seconds (`run.ReteachMeter`, every reteach counted) is under the target: before a pass
  (after its first-visit `teacher` units, which keep their order; at most 4) and again BETWEEN the units
  of a pass (one at a time). A first-visit teacher does not block: 61e3357 let it, and in first-visit
  territory (~18 teacher units, a ~20 min pass) nothing was admitted (paris4 10:09-10:47). Each admitted
  unit logs `recto_backlog_admit` with the share at admission. `0` is the old idle-only rule. The work
  list is rescanned and logged (`recto_regen`: stale rectos, fields rebuilds, the identity `teachers`
  with its `route:<sig>` token, `share` / `share_target` / `reteach_s` / `gpu_s`) every `RECTO_TODO_S`
  = 300 s, checked before every pass AND between its units (61e3357 checked once per pass, so a 20 min
  pass logged once), idle or not. At 0.25 with a ~15 s routed reteach
  (band reused) beside ~65 s verso passes that is ~60 regions an hour.
- **Sampler (`sample.Patches`).** A routed config adds ONE trailing target row, `band` (`target_channels`),
  that no head predicts. For a region whose committed generation has a band (`_routed`): at rung 2 the
  recto row is the fine teacher at full weight (rw is no longer a recto weight), and the band row is the
  base probability with weight 255 where `c_A = 1`, 128 where `c_A = 0` (0 = no routed target:
  air, outside the store, an unrouted region, another rung); at rungs 3-6 the recto row reads the `band`
  store (pooled exactly as the recto was) with full weight, and the rw is ignored. A region the
  regeneration has not reached yet keeps its old meaning (recto + rw), so rungs 3-6 never lose targets
  mid-switch. Round ≥ 1 never routes.
- **Trainer loss at rung 2** (`losses.route_masks` / `route_apply` / `band_penalty`; the routing row is
  taken off the batch after the augmentation, which moves it with the targets). Per voxel:
  - `c_A = 1`: BCE + dice vs the fine teacher's p, as today;
  - `c_A = 0`, inside the base band dilated by `band_dilate` (2) voxels (the GAP): weight 0 for the
    direct BCE / dice (and the pair / ECT / exclusivity / affinity terms, which read the same weight);
    the supervision there is the cascade self-consistency term and the skeleton-recall term -- both keep
    the gap's weight (`aux_losses(w_cont=...)`), and the recto row's target in the gap is the base
    probability, so the skeleton recalled there IS the m7 band's skeleton -- plus the distance heads
    (their fields, where they exist, are unchanged);
  - `c_A = 0`, outside the dilated band: BCE + dice vs the base teacher's probability (the air
    suppression of the m7-only targets, unchanged), plus `loss_band · mean relu(p - band_eps)²` over
    these voxels, so nothing is predicted outside m7's footprint where the fine teacher is not sure.
    Deliberate deviation from "every voxel outside the band": a `c_A = 1` face the fine teacher sees and
    m7 does not is NOT penalised -- it is a direct BCE target there and the two terms would fight.
  Every train row logs `route_vox` (the routed share of the batch), `route_ca`, `route_gap`, `route_out`
  (shares of the routed voxels) and `band`. Rungs 3-4 are unchanged (m7 targets). With `thin_band = 1`
  the gap gets a direct target after all: the THINNED base band (§6 "Thinned band target").
- **Evaluation.** Unchanged code, the committed generations as the reference: the held-out grid's
  `grid_sources` see the new recto / rw / band digests and rebuild exactly the held-out regions' items as
  their regeneration commits (a routed grid also carries the route in `grid_global`); `heldout_rows` /
  `rvsm eval` read the committed recto. So from the switch on, **rung-2 dice is against the fine teacher
  at its full base weight, gaps included**, rungs 3-4 against m7: a student that fills the gaps correctly
  loses rung-2 dice there. Read `dice_r2` together with `dice_r3` / `dice_r4` and the train-row shares.
- **Failure modes to watch.**
  - *Confident-but-wrong fine-teacher voxels*: `c_A = 1` is self-confidence, not correctness. A confident
    false face (a crack, a bright inclusion) is a full-weight target, and its REACH makes the background
    around it trusted too. Look at rung-2 panels where the fine teacher and m7 disagree with `c_A = 1`.
  - *Self-consistency at 0.1 may be too weak to bridge*: in a gap the direct loss is zero, and
    `loss_selfcons` (0.1, self-source samples only) plus `loss_skel` (0.05) may not pull the student to
    fill it; the gap then stays empty at rung 2 (the cascade from rung 3 is the only positive signal).
    Both weights are fingerprint-excluded (`loss_switch` on resume) if they need raising.
  - *Handoff seams at the c_A boundary*: at the edge of a trusted zone the target switches from the fine
    teacher's thin face to the thicker upsampled m7 band (outside the dilated band) or to nothing (in a
    gap); the prediction can step or tear along that boundary, and the hybrid skeleton can kink there.
  - *Regression to blur if gap zones dominate*: if `route_gap` is large, most rung-2 voxels carry only the
    soft continuity terms and the student drifts back to the coarse (upsampled m7) cascade -- blurrier than
    either teacher. Watch `route_gap`, `dice_best_r2`, and the rung-2 panels' sharpness.
  - The band reused from a committed m7 store is re-encoded at q8 (a second lossy pass, within one
  or two q8 steps of the stored m7 probability, mean error < 1 code, test-checked).
- **Producer cost.** Per routed reteach only the fine teacher runs: ~0.12 s a 256³ window compiled bf16
  on tnr-0's A100 (§7 `teacher_bf16` row: recto 0.342 → 0.122 s a window), i.e. roughly the recto half
  of the ~19 s two-teacher region (~12-15 s of GPU for a 1024³ region at window 256 / halo 32), plus the
  coverage (~1 s), the band decode on the reader thread (overlapped) and the three store writes on the
  writer thread (overlapped). A first-time routed region runs both teachers (~19 s, as the fusion did).

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
