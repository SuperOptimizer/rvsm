# Pre-production review checklist

What a reviewing agent should verify before the first real run (`rvsm run --ct <Paris 4 2.4 µm URL>
--umbilicus <json> --mode resident --gpus 0` on the A100, plan §7). Each item names the contract, where it
is defined, and the test that covers it; items with no test are called out as gaps in §10.

Conventions: `§n` = `/home/forrest/usrm2/docs/unified_design.md`; `plan §n` = [`plan.md`](plan.md);
the reasons behind each rule are in [`rationale.md`](rationale.md) and [`recipe.md`](recipe.md).

---

## 1. The channel contract

| check | where | test |
|---|---|---|
| The stem order `[CT, ctx_1..9, cascade, radius, meta×5, scale, rz, ry, rx]` and the head order `[recto, verso, midline, thickness, logvar, aff8_zyx, aff16_zyx, aff32_zyx]` are declared in **exactly one place**, `Layout` in `rvsm/config.py` | `rvsm/config.py` (`Layout.i_ct`…`Layout.i_rad`, `head_names`, `stem_names`) | `tests/test_config.py::test_default_layout_is_21_in_14_out` |
| `cin == 21`, `cout == 14`, `cout_t == 4`, `nprob == 2` for the default config | `Layout` properties | same |
| Nothing else computes a channel index by arithmetic. **Grep for it**: any literal slice into the stem or the head outside `config.py` is a finding | `rvsm/prep.py`, `rvsm/sample.py`, `rvsm/model.py`, `rvsm/train.py`, `rvsm/infer.py`, `rvsm/export.py` | `tests/test_sample_prep.py::test_prepare_channel_order_matches_the_layout`; `tests/test_losses.py::test_aux_losses_reads_every_head_index_from_the_layout`, `::test_deep_losses_slices_to_the_probability_heads_with_a_layout` |
| The loader item carries **exactly** `RUNG_ITEM_KEYS` — a key added without a producer must fail loudly, not vanish | `rvsm/config.py:RUNG_ITEM_KEYS`; `rvsm/sample.py:rung_item` | `tests/test_config.py::test_rung_item_keys_are_frozen`; `tests/test_sample_prep.py::test_rung_item_yields_exactly_the_contract`, `::test_loader_collates_the_contract` |
| The radial channels are the **last three** of the stem, so a sign flip is a slice — and flipping negates exactly those | `Layout.i_rad` | `tests/test_sample_prep.py::test_prepare_norad_zeroes_only_the_radial_channels`; `tests/test_infer_export.py::test_the_flipped_sign_negates_exactly_the_radial_channels`, `::test_flips_chan_negates_the_radial_input_channels` |
| `fingerprint()` excludes exactly `(steps, eval_every, workers, gpus, rounds, ct_seed, ckpt_act, compile, vram_train_gb, vram_produce_gb, pin_memory, gpu_prefetch, verso_min_dice, self_p_mid_step, self_p_end, self_p_end_step)` and nothing else | `rvsm/config.py:FINGERPRINT_EXCLUDE` | `tests/test_config.py::test_fingerprint_ignores_only_the_resume_fields` |
| An unknown TOML/CLI key is an **error**, not a silent ignore (a typo in a recipe field would train something else) | `rvsm/config.py:load` | `tests/test_config.py::test_unknown_key_is_an_error` |

**Why this matters:** every growth step in usrm2 — cascade (`cin` 14→15), verso (`cout` 1→2), affinities
(`cout` 2→11), distance heads — needed a bespoke warm-start rule because the order lived at the use sites
(§22, §26.1, §29.2, §29.5). See rationale §3.9.

## 2. Store rules

| check | rule | where | test |
|---|---|---|---|
| Probability stores are **q8**; distance stores (`midline`, `thickness`) are **q0, lossless** | q8 rounding turned a stored `0` into a `6`, i.e. a −30.5-voxel distance indistinguishable from a real one, and it compounds under partial-chunk writes (§29.1) | `rvsm/stores.py:out_array(q=)`; plan §2 | `tests/test_targets.py::test_encoding_round_trips_and_reserves_code_zero`, `::test_signed_distance_sign_units_and_clamp_on_a_slab` |
| The codec chain is **exactly `[volcomp]`** — `compressors=None` | zarr-python silently appends zstd; measured saving 0.2 % for a decode step on every read (§24) | `rvsm/stores.py:55` | `tests/test_stores.py::test_store_round_trip_and_pool` |
| **One write per shard**, one shard per 1024³ region, 128³ inner chunks | 514 files → 2 files, ~590 sftp ops → 7 (§18.1b) | `rvsm/stores.py:shard_shape`, `CHUNK`, `SHARD` | `tests/test_stores.py::test_store_round_trip_and_pool` |
| **Never rewrite a finished store in place.** Write to `.zarr.tmp/`, rename, set `done` **last** | an in-place repack under a live reader killed the u2 trainer at 18:01 UTC 2026-09-21 (§24) | `rvsm/stores.py:write`; plan §1, §10 | `tests/test_stores.py::test_tmp_never_left_behind`; `tests/test_infer_export.py::test_produce_writes_a_done_store` |
| `volcomp_build` (sha256 of `libvolcomp.so`) is in every store's attrs | the desk and the A100 decode the same bytes to slightly different arrays (different builds) | `rvsm/stores.py:volcomp_build`, `:58` | `tests/test_stores.py::test_store_round_trip_and_pool` |
| A store's attrs carry `origin_zyx`, `rung`, `voxel_um`, `channels`, volume URL, umbilicus, round, producer, ckpt/step, window/halo, `radial_sign`, `done` | plan §2 | `rvsm/stores.py:out_array` | `tests/test_infer_export.py::test_produce_writes_a_done_store` |
| `rungs()` **fails loudly** when `libvolcomp.so` is missing — usrm2 silently dropped the levels it could not decode | plan §10 | `rvsm/ladder.py` | `tests/test_ladder.py` |
| A region is done **iff** `zarr.json` says `done: true` — region state is derived from disk, and `rvsm ledger --rebuild` is a directory scan | plan §1 | `rvsm/stores.py:is_done` | `tests/test_regions.py::test_catalog_reflects_a_store_written_with_stores_write` |

**Verify by hand before the run:** `libvolcomp.so` on the A100 is built with **portable** flags, not
`-march=native` (the desk build SIGILLs on Xeon instances — `usrm2-streaming.md`), and the first store
written on the A100 records that build's hash.

## 3. The near-axis and no-data weighting rules

| check | rule | where | test |
|---|---|---|---|
| Voxels within **400 µm of the umbilicus axis get weight 0** | near the axis the sheet geometry degenerates and a distance target is meaningless (§29.1) | `rvsm/targets.py:57 AXIS_R_UM = 400.0` | `tests/test_targets.py::test_near_axis_voxels_get_weight_zero`; `tests/test_sample_prep.py::test_weight_is_zero_near_the_umbilicus_for_verso` |
| **`code 0` means weight 0**, everywhere a distance channel is read | code 0 is the contract's no-data marker and decodes to −32 voxels, not to "nothing" | `rvsm/losses.py:dist_weight`; `rvsm/sample.py` | `tests/test_sample_prep.py::test_code_zero_means_weight_zero_for_a_distance_channel`; `tests/test_losses.py::test_sdist_and_thickness_decode_the_store_encoding` |
| A distance voxel whose resampled weight is **< 0.95 is dropped**, not down-weighted | interpolating *across* code 0 gives a number that is simply wrong (§29.2) | `rvsm/losses.py:dist_weight` | `tests/test_losses.py::test_dist_weight_drops_partially_resampled_voxels` |
| Distance channels are **rungs 2-4 only** and are **never pooled** | a pooled distance is not a distance (plan §3) | `rvsm/targets.py` | `tests/test_targets.py::test_coarse_rungs_are_recomputed_and_never_pooled`, `::test_a_rung_above_four_is_refused` |
| A window with **no store** gets weight 0 rather than a zero target | plan §5 `sample.py` | `rvsm/sample.py` | `tests/test_sample_prep.py::test_a_window_without_a_store_has_weight_zero` |
| With distance heads on, `scale` / `shear` / `elastic` / `sheetcomp` must be **dropped from the augmentation config, loudly** | a distance is an isometry-only target: a rotation or flip carries its value unchanged, a scale/shear/elastic/sheet-compression does not, because they change the metric the distance is measured in (§29.2) | nowhere — see §10 | **none: this rule is NOT implemented in rvsm.** `aug.PRESETS["full2"]` includes `SPATIAL` (`rot`, `scale`, `shear`, `elastic`) and `SHEETCOMP` (`rvsm/aug.py:618-624`), and `loss_sdist 1.0` on the midline/thickness heads is on by default. **This is the highest-priority finding in this checklist.** |

## 4. Cascade: leak and exposure rules

| check | rule | where | test |
|---|---|---|---|
| The cascade channel at rung k holds the rung-(k+1) prediction **over the same field of view**, upsampled 2× — never a same-rung quantity | §22 | `rvsm/prep.py:Cascade`; `rvsm/infer.py:cascade_for` | `tests/test_sample_prep.py::test_cascade_modes_shapes`; `tests/test_infer_export.py::test_cascade_for_walks_one_rung_up` |
| **The self-feedback is truncated at one level**: the coarse pass's own cascade channel is always zero. Unconstrained multi-step self-feedback diverges without damping | §26.1 | `rvsm/prep.py:Cascade`, `rvsm/infer.py` | `tests/test_infer_export.py::test_cascade_for_at_depth_one_is_the_coarse_pass_upsampled` |
| `loss_selfcons` is scored **only on the SELF-source samples**, with the coarse side **detached** (one-way stop-grad) | against the `mask` source it would be a second, blurrier copy of the supervised loss; a dropped channel is all zeros | `rvsm/losses.py`; `rvsm/prep.py:Cascade.last_self` | `tests/test_losses.py::test_self_consistency_is_zero_when_the_pools_agree` |
| `self_p` anneals **0.1 → 0.7** over the run (exposure bias); `cascade_drop 0.1` blanks the channel so the model cannot become dependent on it | §26.1 | `rvsm/config.py`; `rvsm/prep.py` | `tests/test_train.py::test_twenty_steps_of_the_full_recipe` |
| At inference, `cascade_depth 3` is a genuine top-down recursion with the per-rung temperature applied at **each** level | §26.4, §22 | `rvsm/infer.py` | `tests/test_infer_export.py::test_student_fn_applies_the_temperature_to_the_probability_heads_only` |
| **No held-out leak**: the 8 held-out regions are excluded from the walk **at every rung**, and a coarse tile containing one gets weight 0 there | plan §3 | `rvsm/regions.py` | `tests/test_regions.py::test_held_out_is_stratified_distinct_and_excluded`; `tests/test_sample_prep.py::test_val_grid_is_fixed_and_over_the_held_out_regions` |

## 5. The round gate and the verso gate

| check | rule | where |
|---|---|---|
| Round 0 trains **recto only** (verso weight 0 everywhere) until the gate fires | plan §1 | `rvsm/run.py:run` (the producer writes no `verso` store until `state.json` says `verso_on`, and a channel with no store carries weight 0) — `tests/test_run_e2e.py::test_rvsm_run_two_rounds_end_to_end` |
| The verso gate is `recall@4` **and** continuity against the fused-teacher reference **within the bootstrap CI**, or `verso_after_steps = 10000` as an unconditional fallback | a 2-point move on the val box is not evidence; the 95 % CIs are 4-10 points wide (§25.6) | `rvsm/config.py:verso_after_steps`, `verso_gate_dice`; `rvsm/run.py:verso_gate` — **deviation**: recall@4 and continuity are MESH metrics and the reference is a store, so the gate uses `compare_stores` dice (recall side) and betti0 against the reference-vs-itself baseline (continuity side), with the same bootstrap over regions; recorded in `run.py`'s docstring — `tests/test_run_e2e.py::test_the_verso_gate_needs_the_dice_and_the_betti_baseline` |
| Round r+1 requires: plateau fit < 2 % remaining gain (or `round_steps`), **and** `merge_frac` and `betti0_err` not worse than the round-0 reference beyond the CI, **and** `recall@4` within CI | plan §1 | `rvsm/run.py:round_gate` (`evalsurf.fit_curve` on `logs/eval.jsonl`, then `precision` and `betti0_err` against what the round-0 gate measured; round 0 itself has no earlier round to be worse than, so its only condition is the plateau) — `tests/test_run_e2e.py::test_the_round_gate_wants_a_plateau_and_then_the_quality` |
| A round failing the gate is **discarded** — previous teacher kept, training extended (WSD makes extension free) | plan §1; §26.2 | `rvsm/run.py:round_gate` returns False and the round is not bumped; no teacher snapshot is written — `tests/test_run_e2e.py::test_the_round_gate_wants_a_plateau_and_then_the_quality` |
| At most **two rounds live on disk**; round-r stores are deleted per region once superseded **and unreferenced** | plan §1 | `rvsm/run.py:_clean_old_rounds` — `tests/test_run_e2e.py::test_old_rounds_are_deleted_only_once_superseded` |
| Every metric is quoted as `value (reference) [CI]`; per-surface rows are inspected before a pooled mean is believed | §25.7 | `rvsm/evalsurf.py` |

Supporting tests that exist today: `tests/test_evalsurf.py::test_bootstrap_ci_contains_the_point_estimate`,
`::test_fit_curve_recovers_a_planted_asymptote`, `::test_fit_curve_needs_enough_points_and_reports_a_flat_run`,
`::test_compare_stores_is_perfect_against_itself`, `::test_compare_stores_charges_a_break_in_the_skeleton`,
`::test_a_bridge_between_two_sheets_shows_up_as_a_betti0_error`. The gate **logic** that consumes them is
the gap (§10).

## 6. VRAM budget tables

The plan requires each process to set `torch.cuda.set_per_process_memory_fraction` and the run to **refuse
to start** if the table sum exceeds `vram_total − 4 GB` (plan §1). Verify the numbers against these
measurements before trusting the table:

### resident (one ≥80 GB card)

| slot | budget | measured basis |
|---|---|---|
| trainer, 30m6 / 256³ / batch 2 / `ckpt_act` 0 | ~47 GB | u1 on the A100: 46.8 GiB with compile; **67 GiB and OOM without compile** (§14) |
| trainer, same at `ckpt_act` 1 | ~57 GB | u4 on the A100: 56.5 GB (`usrm2-runs-state.md` 2026-09-22) |
| + Phase A terms (affinity targets 3×3×256³ bf16, skeleton temporaries) | +0.6-0.8 GB | §26.5 arithmetic |
| producer: teacher slot | ~13 GB TRT / ~14 GB torch bf16 | desk 5060 Ti nvml peak 13.4 GB (`usrm2-findings.md`) |
| producer: student slot, window 256 / halo 32 / batch 1 / fp16 accumulators | ~16 GB | 5090 32 GB pod ran it with room (`runpod-5090-verso.md`) |
| — teacher and student **never run concurrently**; m7 shares the student slot; after round 0 the teacher slot is released | | plan §1 |

### timeshare

| configuration | rule |
|---|---|
| two cards | GPU0 trainer (batch 1, accum 2, `ckpt_act` 2 on 32 GB), GPU1 producer, no switching |
| one card | phases via `<out>/PHASE`: train `train_min` (20 min), checkpoint, move net + optimizer to CPU, `empty_cache`; producer drains the window or `produce_max_min` (10 min); swap |
| both processes stay alive across a swap | the compile cache survives; a recompile on `.cuda()` is ~1 min, < 5 % of a phase; TRT engines are built once per host and deserialised in seconds |
| 48 GB card at batch 2 | needs `ckpt_act 1`; `batch 1 --accum 2` is the fallback, ~15 % slower than a real batch 2 (§26.6) |

**Also check:** `batch 2` is not silently raised — batch 3 was measured at +2 %/voxel and changes the
optimisation (§14); `cudnn.benchmark` stays **off** (no speed change, peak 41 → 71 GiB); the net runs
**NCDHW**, not `channels_last_3d` (96.2 ms vs 15.7 ms for compiled GroupNorm+SiLU, §14b); `model.up2x` is
used for exact 2× upsampling, not `F.interpolate` (backward 1762 → 651 ms, §14).

## 7. Resumability from disk

| check | rule | test |
|---|---|---|
| The **only** mutable metadata is `<out>/state.json` (walk cursor, round, phase, verso gate), written atomically by the trainer, plus the `PHASE` and `STOP` markers | plan §1 | `tests/test_run_e2e.py::test_state_and_markers_are_atomic_and_readable_by_anyone`, `::test_rvsm_stop_ends_the_run_within_one_unit` |
| Region state is **derived**: `rvsm ledger --rebuild` is a directory scan and must reproduce the ledger exactly | plan §1 | — (gap) |
| A resume compares `fingerprint()` and refuses a different config | `rvsm/config.py` | `tests/test_train.py::test_resume_continues_and_refuses_a_different_config` |
| A warm start copies by name, zero-inits the rest, and **reports** the new tensors; probability rows come out bit-identical | §26.1, §29.2 | `tests/test_train.py::test_warm_start_reproduces_the_source_and_reports_the_new_tensors`, `::test_warm_start_from_a_usrm2_checkpoint_maps_by_name` |
| `targets.region_fields` is resumable and **deterministic across job counts** | §29.8 (`--resume`, `--jobs`) | `tests/test_targets.py::test_done_is_resume_and_force_recomputes`, `::test_jobs_four_is_byte_identical_to_jobs_one` |
| The shard cache releases and evicts only released regions, and never evicts pinned coarse levels; a 404 leaves an `.absent` marker so nothing refetches it | plan §2 | `tests/test_stream.py::test_release_and_evict_keep_pinned_levels_and_live_regions`, `::test_404_leaves_an_absent_marker` |
| A local CT path is **symlinked, not copied** | plan §2 | `tests/test_stream.py::test_a_local_volume_is_symlinked_not_copied` |
| A run with no verso and no distance stores still trains (the cold-start case) | plan §1 | `tests/test_train.py::test_a_run_with_no_verso_and_no_distance_stores_still_trains` |

## 8. Recipe assertions worth re-deriving during review

| claim | check |
|---|---|
| `relu(p_recto + p_verso − 1) ≡ 0` under construction pairing for every `t ≥ 2·pair_band` — so a non-zero `loss_excl` means a **bug**, not a merge | §29.3 proof; `tests/test_losses.py::test_pair_bands_never_overlap_when_the_sheet_is_thick_enough` |
| The affinity channel set is invariant under the 48 cube symmetries (midpoint-centred, even offsets) | `tests/test_losses.py::test_affinity_targets_on_two_slabs_and_equivariance_under_flips`; `tests/test_sample_prep.py::test_sym_apply_t_matches_numpy_for_all_48_symmetries` |
| The skeleton term cannot be gamed by widening the band | `tests/test_losses.py::test_skeleton_of_a_slab_is_one_voxel_thick_and_recall_is_perfect` |
| WSD is **flat** until `stable_until` and zero at the end; `ema auto` is `1 − k/steps` clamped | `tests/test_train.py::test_lr_lambda_wsd_is_flat_until_stable_until_and_zero_at_the_end`, `::test_ema_auto_is_one_minus_k_over_steps_clamped` |
| Temperature calibration fits a rung **only** when its target is a genuine binary band (`binary_frac ≤ 0.5`) | `tests/test_calib.py::test_binary_frac_separates_a_hard_band_from_a_pooled_fraction`, `::test_run_fits_one_temperature_per_rung_and_skips_the_pooled_rungs` |
| Temperatures apply to the **probability heads only**, never to the distance or affinity rows | `tests/test_infer_export.py::test_student_fn_applies_the_temperature_to_the_probability_heads_only` |
| `ect(p, p) = 0` with finite gradients; `eikonal` of a linear ramp is 0 | `tests/test_losses.py::test_ect_loss_is_zero_on_itself_and_has_finite_gradients`, `::test_eikonal_of_a_linear_ramp_is_zero` |
| Export: sign convention on an outward slab, Scharr of a ramp = 1, every store written with the right q/encoding/attrs, normals **derived** and never stored | `tests/test_infer_export.py::test_tracer_fields_sign_convention`, `::test_scharr_of_a_ramp_is_one`, `::test_export_tracer_writes_the_contract`, `::test_encodings_round_trip` |
| `--sign -1` writes **only** the verso store, and refuses the field heads | `tests/test_infer_export.py::test_produce_student_verso_writes_only_the_verso_store`, `::test_produce_student_refuses_the_field_heads_at_a_negative_sign`, `::test_export_refuses_a_flipped_sign` |
| Batching and fp16 accumulators do not change the result within a store code | `tests/test_infer_export.py::test_batching_does_not_change_the_result`, `::test_fp16_accumulators_match_fp32_within_a_store_code` |

## 9. The plan's own verification steps (§8), not yet run

Before the production run, the plan requires:

1. The full CPU suite on the laptop, plus a **GPU smoke** of `rvsm produce` and `rvsm train` on the 5080 at
   `size 5m, patch 128` (16 GB). Partially scaffolded:
   `tests/test_infer_export.py::test_gpu_smoke_region`, `::test_gpu_smoke_student_region`.
2. On the A100: one region through `rvsm produce --teacher recto,m7` compared **voxel-wise** to the desk's
   existing recto teacher store for the same region — expect **dice > 0.99 within one volcomp build**.
3. One student region at `--sign -1` compared to the 5090 pod's v2 output for the same checkpoint —
   bit-identical expected on the same GPU build, **dice > 0.99 across builds**. Note the floor: two eager
   bf16+cuDNN runs of the same region agree only to dice 0.994 / max 0.11, so tolerances must be looser
   than that (`runpod-5090-verso.md`).
4. During the first run: watch `logs/produce.jsonl` (s/region per pass), `logs/train.jsonl` (Mvox/s, and
   **`train_wait_s` must stay ~0 after warm-up**), VRAM per process against the §6 table; then
   `rvsm eval --tifxyz` on the held-out regions at 10k and 20k steps, compared to u3/u4:
   **recall@4 0.822, continuity 0.705, merge_frac 0.331, ERL 409 µm at 21k** (rationale §2.1).

## 10. Known gaps

**Unbuilt at the time of writing** (see [`plan.md`](plan.md) "Progress"):

- `rvsm/run.py` is built and `test_run_e2e` passes, but it has only ever run on the CPU on a 256³
  fixture: the **spawned** producer process, `CUDA_VISIBLE_DEVICES` per role, the per-process memory
  fraction, the one-card `PHASE` timeshare and the silent-producer restart have no test and no GPU
  hours behind them. The end-to-end test runs `mode = cpu`, where the producer is a thread.
- The gates have never fired on a metric, only on their step fallbacks (`verso_after_steps`,
  `round_steps`): a 20-step student on a synthetic slab cannot pass an honest gate. The metric halves
  are unit-tested on rows, not end to end.

**Exercised only partially:**

- **TensorRT is exercised only on the fallback path**, on the laptop:
  `tests/test_infer_export.py::test_engine_for_falls_back_cleanly` checks that a missing/failing
  `tensorrt` degrades to torch bf16. No real engine has been built or compared in this repo. The upstream
  evidence says teacher engines work (137.8 s → ~48 s/region, byte-identical output) and that the
  **student** cannot be built at all (Myelin fails on the `model.up2x` gather graph) — so a reviewer
  should confirm rvsm never tries to build a student engine (rationale §2.5).
- **The recto teacher's registered window is 256³** (`rvsm/teachers.py:681`; m7 is 192³), so the synthetic
  fixtures use a tiny `fake` teacher and the real teachers are never run at test size. Any test or smoke
  run of the real recto teacher needs windows of at least 128³, and the production path uses 256³ with the
  region padded when it is thinner (`::test_run_region_pads_a_region_thinner_than_a_window`).
**Missing, and load-bearing:**

- **The isometry-only augmentation rule (§29.2) is not implemented.** usrm2 dropped `scale`, `shear`,
  `elastic` and `sheetcomp` from a run's augmentation config, loudly, whenever a distance head was on,
  because those four change the metric the distance is measured in and the resampled target is then simply
  a wrong number. rvsm has `aug = "full2"` and `loss_sdist = 1.0` both on by default, and
  `rvsm/aug.py:618-624` shows `full2 = _pre(SPATIAL, INTENSITY, CUTOUT, SCAN, TONE, THICK, POOL, VOLCOMP,
  BLANK, ZJIT, SHEETCOMP, PAGANIN, SHUFFLE)` with `SPATIAL = {rot, scale, shear, elastic}`. Nothing in
  `aug.py`, `train.py`, `sample.py` or `prep.py` removes them. Either the four must be dropped (and the
  drop asserted in `tests/test_aug.py`), or the deviation must be recorded with its own evidence. The 48
  cube symmetries and every intensity augmentation are unaffected either way.
- **Every number in this repo's docs is from usrm2.** No rvsm run has produced a validation number yet:
  real-scroll numbers are **pending the first A100 run** (plan §7-§8), and until then the u3/u4 figures in
  rationale §2.1 are the bar, not a baseline rvsm has reproduced.

**Environment preconditions** (plan §7 "Then:"):

- `libvolcomp.so` built on the A100 with **portable** flags (the desk's `-march=native` build SIGILLs on
  Xeon instances).
- The two teacher `.pth` files present at the paths in `teacher_ckpts` (`surface_recto_3dunet.pth`,
  `surface_m7_nnunet.pth` from the desk's `/vesuvius/tsm/models/`, or fetched by `rvsm teachers fetch`).
- usrm2's `u4` stopped, checkpoint kept.
- Never `pkill` on an ssh command line; kill logic lives in scripts on the host (plan §10).
