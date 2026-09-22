# rvsm implementation plan

**Status: approved 2026-09-22.** This is the plan the user approved on 2026-09-22, reproduced verbatim
below from `/home/forrest/.claude/plans/flickering-cooking-iverson.md` (nothing in section "The approved
plan" has been edited, reordered or summarised). It is the contract the implementation is being built
against; where the code has already deviated, the deviation is recorded in
[`rationale.md`](rationale.md) with its evidence, not by editing this file.

Companion documents:

- [`rationale.md`](rationale.md) — why rvsm exists at all, the measured usrm2 numbers behind each choice,
  and the alternative rejected for each decision.
- [`recipe.md`](recipe.md) — every fixed default in `rvsm/config.py`, one line of justification each,
  with a pointer into `docs/research/` or `usrm2/docs/unified_design.md`.
- [`research/README.md`](research/README.md) — the 18-document research corpus, indexed, with the
  consolidated bibliography.
- [`review_checklist.md`](review_checklist.md) — what to verify before a production run.

## User decisions recorded in the plan

These were asked and answered before the plan was approved; they are not open questions, and a reviewer
should treat a change to any of them as a change of scope, not a bug fix. They appear in the plan's
"Context" section and are repeated here so they are findable.

| # | decision | where in the plan |
|---|---|---|
| 1 | Round-0 teachers are **recto + m7, fused by agreement** (no third lineage, no ink/fiber/lasagna teachers) | Context; §5 `teachers.py`; §9 non-goals |
| 2 | Defaults are tuned to fit **both** an 80 GB resident card and a 32 GB timeshare card | Context; §1 "GPU modes" |
| 3 | Human `tifxyz` meshes are **optional**, and only in eval | Context; §3 held-out set; §4 `tifxyz` |
| 4 | **Two processes with the filesystem as the bus** — no sqlite, no planner process | Context; §1; §9 non-goals |
| 5 | In round 0, verso production starts **only once the student's recto passes a gate** | Context; §1 "Round 0 verso gate" |
| 6 | Teacher weights come from **local paths in the config**, with **TensorRT in v1** and a fallback ladder to torch | Context; §5 `trt.py` |
| 7 | Develop on the laptop 5080; the **first real run is on the A100, after u4** | Context; §7 "Then:"; §8 |
| 8 | No `vesuvius` / `dynamic-network-architectures` dependency: the teacher nets are **ported natively from tsm** | Context "Deps" |
| 9 | Every v2 improvement is a **fixed default from the first commit**, not a research flag | Context; §4 "Fixed recipe" |
| 10 | rvsm **never** looks up published masks or published stores | Context; §3; §9 non-goals |
| 11 | The one deliberate deviation from the design's §29.9 recipe: **ECT is ON at `ect_n 1`** | §4 "Fixed recipe" |

## Progress: the plan's 7 commits vs `main`

Read from `git log` on `main` at 2026-09-22. The build order in §7 is a *logical* order; the actual
history split commit 2 in three (the store writer landed on its own first, and the teacher-weight fetch
came back as a follow-up) and pulled the pure library ports of commits 4/5/7 forward into one commit,
because they have no dependency on the driver.

| plan commit (§7) | actual commit(s) on `main` | state |
|---|---|---|
| 0 — repo | `bd1cafb` Initial commit (2026-09-19) | done |
| 1 — skeleton: `pyproject.toml`, `config.py`, `ladder.py`, `axis.py`, `scanmeta.py`, fixtures | `2a899ab` *skeleton: the config contract, the rung ladder, the axis and the scan metadata* | done |
| 2 — teachers + region runner + store writer: `teachers.py`, `trt.py`, `infer.py`, `export.py` | `3f136dc` *stores: the region store writer/reader shared by producers and the sampler* → `b6a99c2` *teachers, the region runner, the tracer export: commit 2* → `4b41c5b` *teachers: fetch the published weights, and `rvsm teachers fetch`* | done; the store writer was split out ahead of it as `rvsm/stores.py`, and `4b41c5b` adds a weight-fetch path the plan did not call for (the plan assumed hand-copied `.pth` files) |
| 3 — cache + regions + sampler + prep + model | `49cb206` *cache, regions, sampler, prep and model: commit 3* | done |
| 4 — losses + aug + train | `2a44646` *losses, augmentation, calibration, distance targets and evaluation: the pure ports* (`losses.py`, `aug.py`) | **partial**: the loss and augmentation ports landed; `rvsm/train.py` and `tests/test_train.py` exist in the working tree but are not yet committed |
| 5 — student inference + verso + distance stores + calib + export | `b6a99c2` (`infer.py`, `export.py`) + `2a44646` (`calib.py`, `targets.py`) | **partial**: the ported pieces are in; the student/verso production path and the `rvsm produce --student` acceptance test are not yet committed |
| 6 — the driver: `run.py`, supervisor, Producer, lookahead, verso gate, rounds, GPU modes | — | **not started**; `rvsm/run.py` does not exist |
| 7 — evaluation v2 + pretrain + ladder + status/stop/ledger | `2a44646` (`evalsurf.py`) | **partial**: the evaluation port is in; `pretrain.py`, `rvsm ladder`, `status`/`stop`/`ledger` are not |

Not yet reached at all: the post-implementation steps at the end of §7 (push, build `libvolcomp.so` on the
A100 with portable flags, copy the teacher weights, stop u4, first `rvsm run`) and all of §8's verification.

---

## The approved plan (verbatim)

# rvsm: raw CT + umbilicus in, self-distilled recto/verso surface model out, one machine

## Context

usrm2 became a four-host pipeline (desk teachers, A100 trainer, RunPod verso loop, dl.ash2txt.org as the
exchange) and most incidents of the last two days were the coordination itself: a repack under a live reader
killed the trainer, fetch starvation, sftp operation counts, and a desk outage that cut the only hop to the
A100. The user wants a **greenfield, tiny, standalone project `rvsm`** (github.com/SuperOptimizer/rvsm, one
empty commit; local checkout /home/forrest/rvsm) that starts from **a raw CT zarr URL and an umbilicus and
nothing else**: it runs the upstream teachers itself, trains the unified multi-rung student, infers the verso
with the flipped radial sign, and keeps iterating (self-distillation, the student as its own teacher),
**on demand per 1024^3 region, on one machine**. It never looks up published masks or stores. Every v2
improvement is a **fixed default from the first commit**, not a research flag. Two GPU modes: **resident**
(all roles on one big GPU) and **timeshare** (roles alternate on one small GPU, or one role per card).

User decisions (asked and answered): round 0 teachers = recto + m7 fused by agreement; defaults tuned for
both 80 GB resident and 32 GB timeshare; human tifxyz meshes optional in eval; **two processes with the
filesystem as the bus** (no sqlite, no planner process); verso production in round 0 starts **once the
student's recto passes a gate**; teacher weights from **local paths** in the config with **TensorRT in v1**
(fallback ladder to torch); develop on the laptop 5080, **first real run on the A100 after u4**.

Deps: torch, numpy, zarr>=3, volcomp-zarr (+ `libvolcomp.so` built on the host with portable flags),
aiohttp, scipy, pillow; extras `mesh` (scikit-image, tifffile), `trt` (tensorrt, onnx). No `vesuvius` /
`dynamic-network-architectures`: the teacher nets are ported natively from tsm.

## 1. Architecture

`rvsm run cfg.toml` = one supervisor that spawns **two processes** (spawn context, `CUDA_VISIBLE_DEVICES` per
child) talking only through `<out>/`:

| process | GPU | owns |
|---|---|---|
| `train` (main) | yes | student, EMA, WSD, cascade, all losses, checkpoints, walk cursor, evaluator (held-out metrics, round gate, calibration; the eval-region inference is requested from the producer) |
| `produce` | yes | CT shard cache (fetch/evict), region occupancy + walk, teacher passes (recto, m7, fused), student passes (verso in round 0; all heads from round 1), coarse-rung pooling, distance stores, TRT engines |

Separate processes because proxied GPUs (Thunder) scale across processes and not threads. Region state is
**derived from disk**: a store is done iff `stores/round_<r>/<ch>/region_<z>_<y>_<x>.zarr/zarr.json` has
`done: true` (written to `.zarr.tmp/`, renamed, `done` set last). The only mutable metadata is
`<out>/state.json` (walk cursor, round, phase, verso gate) written atomically by the trainer, plus
`<out>/PHASE` and `<out>/STOP` markers. `rvsm ledger --rebuild` is just a directory scan.

### Region state machine

```
none -produce-> ct_cached -produce(recto+m7 fused)-> recto_done -produce(student, sign -1)-> verso_done
round r>=1:  -produce(student round r, one multi-head pass, sign +1)-> self_done(r)
```
Coarse-rung regions (k >= 3) carry a `coverage` fraction from pooling instead. Held-out regions go through the
same states but never enter the walk.

### On demand: the lookahead

The trainer owns the cursor over the deterministic walk (`region_walk`, mix mode, visits_max 64). The producer
keeps the next `L` regions ready: fetch missing CT shards (rung-2 shard + `ctx_need` shards, one keep-alive
aiohttp session, 16-32 connections), then run whichever pass the region lacks, in walk order. The trainer
never blocks: it takes the first ready region in the window (recto present suffices; a verso that lands later
earns one revisit with fresh windows); if nothing is ready it sleeps 5 s and logs `train_wait_s`. Cold start
produces `min_regions_before_train` (8) regions first. `L = ceil(T_produce / T_train) * K_active + 4`
(A100 ~12, 32 GB cards ~8), re-estimated every 10 min from the timings in `logs/produce.jsonl`.
Backpressure: no CT beyond the window, production pauses below `disk.reserve_gb`.

### GPU modes

- **resident** (one >= 80 GB card): trainer ~46-57 GB (30m6, 256^3, batch 2, ckpt-act 1-2), producer
  ~30 GB (teacher TRT ~13 GB or torch bf16 ~14 GB + student ~16 GB with fp16 accumulators; m7 shares the
  student slot; they never run concurrently). Each sets `torch.cuda.set_per_process_memory_fraction`; refuse
  to start if the table sum > `vram_total - 4 GB`. After round 0 the teacher slot is released.
- **timeshare**: two cards -> GPU0 trainer (batch 1, accum 2, ckpt-act 2 on 32 GB), GPU1 producer, no
  switching. One card -> phases via `<out>/PHASE`: train for `phase.train_min` (20), checkpoint, move
  net + optimizer to CPU, `empty_cache`; producer drains the window (or `phase.produce_max_min` = 10); swap.
  Both processes stay alive (compile cache survives; recompile on `.cuda()` ~1 min < 5% of a phase). TRT
  engines built once per host, deserialised in seconds. `mode = auto` picks from the cards found.

### Round 0 verso gate, and self-distillation rounds

Round 0 trains recto-only (verso weight 0 everywhere) until the held-out recto passes `verso_gate`
(recall@4 and continuity vs the fused teacher reference within the bootstrap CI, or `verso_after_steps`
= 10k); then the producer starts flipped-sign verso stores in walk order and the verso loss switches on
region by region. Round r+1 triggers when the plateau fit says < 2% remaining gain (or `round_steps`), and
merge_frac / betti0_err are not worse than the round-0 reference beyond the CI, and recall@4 is within CI.
Then: snapshot the calibrated EMA as `ckpt/teacher_round_<r+1>.pt`, open `stores/round_<r+1>/`, regenerate
in walk order with one multi-head pass (sign +1), prefer new stores per region as they land, delete round-r
stores per region once superseded and unreferenced (at most two rounds on disk). A round failing the gate
is discarded (previous teacher kept, training extended; WSD makes extension free).

## 2. Data model

```
<out>/config.json state.json umbilicus.json metadata.json PHASE STOP
<out>/ct/<vol>.zarr/<level>/{zarr.json, c/z/y/x[, .absent]}       # mirror layout; local path input = symlink
<out>/stores/round_<r>/<ch>/region_<z>_<y>_<x>.zarr               # rung 2, one shard, 128^3 chunks, per-region
                                                                   # mini-pyramid levels 2..6 by 2x mean pool
<out>/stores/round_<r>/<ch>/coarse.zarr/<k>  (+ coverage bitmap)  # rungs 7..11, fed one pooled block per region
<out>/ckpt/{student.pt, teacher_round_<r>.pt, best.pt, trt/*.plan}
<out>/eval/heldout.json  eval/round_<r>/step_<s>/{stores, metrics.json}
<out>/logs/{produce,train,eval,sched}.jsonl
```
Channels per region: `recto` (q8; round 0 = recto teacher fused with m7 by agreement), `rw` (q8, the
agreement weight, round 0), `verso` (q8), `midline` (q0, code 128 + d/0.25, 0 = no data), `thickness` (q0),
`conf` (q0, round >= 1). Codec chain exactly `[volcomp]` (`compressors=None`), one write per shard. Attrs:
origin_zyx, rung, voxel_um, channels, volume URL, umbilicus, round, producer, ckpt/step, window/halo,
radial_sign, volcomp_build (sha256 of libvolcomp.so), done. Normals and gmag are derived at export by Scharr
of the exported field, never stored.

CT cache: whole levels <= 48 Mvox pinned at start (rungs >= 5, ~3 GB on Paris 4), level 2 pinned if
`cache_gb` allows, levels 0-1 a rolling buffer evicted LRU by last window/job reference; 404 -> `.absent`
(air). Umbilicus: loader json | volpkg `umbilicus.txt` | `auto` (rung-9 per-z centroids), stored in rung-2
voxels. `metadata.json` fetched from beside the CT, flattened, frozen into `config.json` (planes +
augmentation ranges).

## 3. Rung ladder without masks

- Occupancy from CT: `tiles_fraction` = block mean of `ct_level > 0` at the occupancy rung per 1024^3 tile
  (replaces the mask's block max); keep >= 0.05 at rung 2 / 0.01 at rungs >= 5; draw weight x sqrt(fraction).
- Coarse targets: rungs 3-6 from each region's own mini-pyramid; rungs 7-11 from `coarse.zarr` fed by pooled
  blocks with a coverage bitmap; loader weight = coverage x (CT > 0) x rw. Distance channels never pooled
  (rungs 2-4 only, code 0 = weight 0). From round 1 the student also runs natively at rungs 3-4 over coarse
  regions (top-down cascade); pooled stores remain the fallback for rungs >= 5.
- Held-out set: 8 rung-2 regions by seed, stratified over z and radius, occupancy >= 0.5, excluded from the
  walk at every rung (a coarse tile containing one gets weight 0 there). Their round-0 stores are the fixed
  reference; tifxyz meshes are used whenever `tifxyz` is set.

## 4. CLI and config

```
rvsm run      cfg.toml | --ct URL|PATH --umbilicus PATH|auto --out DIR --gpus 0[,1] --mode auto|resident|timeshare
              [--rounds N] [--steps N] [--size 30m6] [--init ckpt.pt]
rvsm produce  --out DIR (--teacher recto,m7 | --student ckpt [--sign -1]) --region Z Y X   # one region, no state
rvsm train    cfg.toml                      # trainer alone on existing stores
rvsm eval     --out DIR [--ckpt P] [--round R] [--tifxyz DIR] [--json]
rvsm export   --out DIR --ckpt P --box Z Y X DZ DY DX --dest DIR    # tracer contract + marching cubes
rvsm calibrate --out DIR --ckpt P
rvsm pretrain cfg.toml [--steps N]          # label-free masked cubes
rvsm ladder   cfg.toml --sizes 15m,30m6,60m
rvsm status | stop | ledger --rebuild | umbilicus --ct URL --out umbilicus.json
```
`rvsm/config.py` holds ONE `Config` dataclass (TOML via stdlib tomllib, flags override, resolved copy +
fingerprint frozen in `config.json`). Tunables: `ct, umbilicus, out, size, patch, batch, ckpt_act, gpus,
mode, rounds, steps, workers, tifxyz, teacher_ckpts, cache_gb`. Fixed recipe (v2 union of §26.6 + §29.9):
ctx 1-9, rungs 2-11, rung_boost {2: 2}, region 1024, windows_per_region 64, visits_max 64, air_keep 0.1;
lr 3e-4, warmup 200, rewarm 800, new_param_lr_mult 3, sched wsd (cooldown 10%), ema auto (k 50); aug `full2`
from scan meta; cascade mix, self-p anneal 0.1->0.7, drop 0.1, noise on; planes radius + meta (5);
head `[recto, verso | midline, thickness | logvar | affinity 8,16,32 x zyx]` (normals derived) -> cin 21,
cout 14; loss_excl 0.1, loss_selfcons 0.1, loss_skel 0.05 (iters 4), loss_affinity 0.1, loss_sdist 1.0,
loss_eikonal 0.1, pair construct (band 1.5, tau 0.5), **loss_ect 0.05 at rung 2 with ect_n 1** (the one
deliberate deviation from §29.9, which excluded ECT only for attributability); calibrate after every eval;
infer window 256 halo 32, cascade_depth 3, tta 1; verso_source flip; verso_after_steps 10000; eval_every 500;
held-out 8. The checkpoint stores `asdict(cfg)` + layout + temps; resume compares everything except
`(steps, eval_every, workers, gpus, rounds)`.

## 5. Module layout and porting map (~4,300 source + ~700 tests)

| rvsm file | ~lines | ported from (file:function) / what changes |
|---|---|---|
| `config.py` | 170 | new: `Config`, `Layout` (stem order `[CT, ctx1..9, CASCADE, radius, 5 meta, scale, rz,ry,rx]`, head order, `RUNG_ITEM_KEYS`, `head_names` from `predict.py:228-252`) — the ONE place the channel contract lives |
| `ladder.py` | 130 | `data.py`: `rung_um/um_rung` (303), `open_zarr` (79), `pool2` (410), `full_level` (419), `read_rung` (461), `context` (498), `rung_shape` (769); `rungs()` takes a URL or cache dir, integer levels only. Drop: `local/remote` mapping, `read_slice` retry, µm-named groups, `chunk_index/covered` |
| `axis.py` | 120 | `umbilicus.py` whole + `data.axis/axis_at/radial/rmax_vox/radius/scale_plane` (144-171, 583-612) |
| `stream.py` | 160 | `stream.py`: `chunk_key/keys_in/all_keys/rung_need/ctx_need` (46-113), `Fetcher.get/_get` (118-178), `charge/evict` (454-464, 716-731) as `ShardCache(url, root, budget_gb)` with `fetch_region(lo2)` / `release(ordinal)`. Drop the ~350-line planner process (`Walk`, queue.jsonl, resume, meta) |
| `regions.py` | 200 | `data.py`: `occupancy_rung/occupancy` (873, CT level instead of mask), `shard_grid/region_tiles/region_list` (888-952) with `tiles_fraction` (new), `region_visits/walk_order` (1076-1102), `teacher_region_path`->`store_path`, `read_teacher`->`read_store` (1051), `fuse_agreement/binary_confidence` (1015-1048, kept for round 0), TTL cache (1476) -> `Catalog`; new `write_region_store` (levels 2-6) and `feed_coarse` (coarse.zarr + bitmap) |
| `sample.py` | 230 | `data.py`: `zscore/NORM`, `sym_decode/draw_sym/sym_apply` (250-278), `rung_item` (631), `Patches.__init__/region_state/_rung_draw/_cascade_extras/_plane_extras` (1105-1447), `val_grid`, `loader`. `_rung_draw` keeps only the CT-air rejection; `_rung_target` reads region stores (rungs 2-3 store, 4 its level, 5-6 gather, 7-11 coarse.zarr x coverage); cascade `cm` from the same store at k+1. Drop: box mode, stores-file parsing, `_replay`, `require_targets`, `fg_min/dense_pow`, `source_w`, `wtgt` |
| `prep.py` | 190 | `prep.py` near-verbatim (`prepare, radial_t, radius_t, fill_planes_, zscore_cubes_, sym_apply_t, Cascade`); `autocast` defined here (breaks the prep->train cycle) |
| `model.py` | 110 | `model.py` near-verbatim; presets `1m` (tests), `5m`, `15m`, `30m6`, `60m` |
| `aug.py` | 330 | `aug.py` 18-548 (ops, `for_rung`, `intensity` with shuffle, `apply`) + presets `none/geo/full/full2` + `get(name, meta)`; drop the 25 ablation presets |
| `scanmeta.py` | 140 | `scanmeta.py` whole + `data.scan_planes/META_RANGE` (549-580) |
| `losses.py` | 320 | `losses.py` whole (excl, selfcons, skeleton, affinity, sdist/thickness/eikonal/normals_from, pair_bands, ect); offsets are a tuple, no string parsing |
| `train.py` | 300 | `train.py`: `deep_losses/losses_tw` (30-75), `lr_lambda/ema_auto` (89-121), `param_groups/ema_update` (124-162), `evaluate` (171-242, trimmed), `val_png` (254-290, red/blue), `warm_start` (359-386: copy by name, zero-init the rest, `newp` = not copied), the step loop (786-910) verbatim in structure. Drop DDP, streaming bookkeeping, the ~250 lines of flag-recording; `args = asdict(cfg)` |
| `targets.py` | 190 | `targets.py`: encoding (58-75), `medial/signed_to/block_fields` (80-156), block/halo loop (487-519), process pool (`--jobs`); `region_fields(root, lo2, ax, jobs)` builds midline+thickness stores at rungs 2-4 from the recto+verso region stores, `done` = resume. Drop mask-pyramid `dist_pyramid`, `VersoSource`, shard markers |
| `teachers.py` | 380 | tsm `teachers.py` 80-651 (blocks, SE, `ResidualEncoder/UNetDecoder/VesuviusUNet/NNUNetResEncUNet`, `infer_*_arch`, `build_teacher`), `Normalizer` (672-718), `apply_activation`, `TEACHERS` recto + m7 only, `_extract_state/load_teacher` (796-850) with an explicit ckpt path. Drop ink/fiber/lasagna, encoder feature helpers |
| `trt.py` | 120 | `usrm2/trt.py` (`plan/build/Engine`, level 0 + no_timing ladder) for the teachers; fallback to torch bf16 on any build/load failure; engines under `ckpt/trt/` |
| `infer.py` | 300 | `cloud/verso_core_v2.py` 133-338 (`starts/gauss_t/Net/RegionInputs/run_region`, no compat shims), `predict.flips_chan` (188), `cascade_for/crop_pad/up2x_np` (300-318, 474-485), tsm `sliding.window_starts` (226) + all-zero skip; one `run_region(fn, inputs, window, halo, batch)` for `TeacherInputs` (normaliser, softmax fg) and the student's `RegionInputs` (21 channels incl. cascade depth 3, planes, per-rung temperature); m7 = `usrm2/m7.py` level-2 pass + trilinear x4 + coarse air mask. Drop `slide/slide_gpu/probs/predict/luts`, other TTA |
| `export.py` | 150 | `predict.py`: `shard_shape/out_array/put/u8/write` (115-169), `enc_signed/enc_normal/dec_normal/scharr3/tracer_fields` (594-651), `export_tracer` (654-731, one multi-head pass), `mesh_shards` (734-773) |
| `calib.py` | 90 | `calib.py` whole; grid from `sample.val_grid` over held-out regions |
| `evalsurf.py` | 290 | `evalsurf.py`: `read_surface/normals/sites/trilerp/metrics/continuity` (26-150), `surface_rows/pool/bootstrap` (248-444), `fit_curve` (531-611); `topo.py` whole (betti/euler/rasterize); new `compare_stores(a, b)` for held-out regions without meshes. Drop `--ceiling` on published masks, `table/png/curve` CLI ceremony |
| `pretrain.py` | 160 | `pretrain.py`: `rename_out/in` (38-62), `block_mask/mask_ctx_/mask_input/recon_loss` (67-149), `pretrain()` (218-430) on `Patches(label_free=True)` |
| `run.py` | 260 | new: supervisor (spawn, memory fractions, PHASE/STOP, heartbeat + restart), `Producer` loop (from `cloud/teacher_regions.py` 85-102 + `verso_run_v2` reader/GPU/writer threads without sftp), rounds driver, verso gate, `ladder()` (from `ladder.py` 57-128, 188-273: sequential runs sharing the walk seed) |
| `cli.py` | 110 | subcommands above |
| `tests/` | 700 | see §6 |

## 6. Tests (all CPU, synthetic; fixtures ported from usrm2 `tests/test_rungs.py:19-76`, `test_stream.py:27-63`)

Fixtures: a sharded zarr-v3 CT pyramid (base 256, chunks 16, shards 64) with a bright slab served by a
range-capable `http.server` subprocess + a trimmed `metadata.json`; a straight-axis umbilicus; a tiny
`VesuviusUNet` state dict registered as `TEACHERS["fake"]` (patch 32); `small_cfg` (size 1m, patch 32,
region 64, ctx 1-3, rungs 2-3, affinity (4,8), ect_block 8, steps 20, compile off).
- `test_ladder`, `test_axis`, `test_stream` (rung_need == read_rung coverage; 404 -> `.absent`; release
  evicts only released regions; coarse levels pinned), `test_regions` (occupancy drops air tiles; weighted
  shuffle; store round-trip rung 2 and rung 3 = 2x pool; `feed_coarse` block + bitmap; fuse_agreement
  bounds), `test_teachers` (arch inference from shapes, strict parity, normalisers, forward shape),
  `test_sample_prep` (exact `RUNG_ITEM_KEYS`; channel order matches `Layout`; `sym_apply_t` == numpy for
  all 48; no store -> weight 0; code 0 -> weight 0), `test_losses` (excl 0 for disjoint; pair bands never
  overlap for t >= 2·half; skeleton one voxel thick; affinity equivariant under flips; ect(p,p)=0;
  eikonal of a ramp ≈ 0), `test_train` (wsd flat until stable_until; ema auto; warm start reproduces
  outputs to 1e-5 and reports new tensors; deep_losses pooling), `test_infer_export` (teacher and student
  through the same `run_region`; thin regions padded; `flips_chan` negates the flipped normal component;
  export writes every store with the right q/encoding/attrs; sign convention on an outward slab; TRT
  fallback path when tensorrt is absent), `test_targets_calib_eval` (signed distance sign/units/clamp;
  `region_fields` jobs=2 byte-identical to jobs=1; `fit_temp` recovers a planted temperature; betti of a
  slab; `compare_stores(a,a)` perfect; `fit_curve` recovers a plateau), `test_pretrain`.
- `test_run_e2e`: `rvsm run` 20 steps on CPU, `rounds=2`, teacher `fake`, one-card timeshare with
  `phase.train_min` tiny: after round 0 recto stores are `done`, the verso gate fires (forced by config),
  verso stores appear, `ckpt.pt` has cout 14, `eval.jsonl` finite, temps written; after round 1 `midline`,
  `thickness`, `conf` stores exist for every produced region and `export/` holds the tracer contract.
  Budget < 90 s.

## 7. Build order (7 commits on rvsm main, each leaves pytest green)

1. Skeleton: `pyproject.toml` (deps + extras), `config.py`, `ladder.py`, `axis.py`, `scanmeta.py`, fixtures.
   Accept: `Config().layout()` gives cin 21 / cout 14; `test_ladder/test_axis` pass.
2. Teachers + region runner + store writer: `teachers.py`, `trt.py`, `infer.py` (TeacherInputs, run_region,
   m7), `export.py` (out_array/put). Accept: `rvsm produce --teacher fake --region 0 0 0` writes a done store
   from the served CT; TRT fallback exercised.
3. Cache + regions + sampler + prep + model: `stream.py`, `regions.py`, `sample.py`, `prep.py`, `model.py`.
   Accept: `Patches` yields rung_items at rungs 2-3 from commit-2 stores (write `feed_coarse` and its test
   first: the only genuinely new logic).
4. Losses + aug + train: Accept: `rvsm train` 20 CPU steps with the full 14-row head writes ckpt/eval/png.
5. Student inference + verso + distance stores + calib + export: Accept: `rvsm produce --student ckpt`
   writes recto/verso/midline/thickness stores; `rvsm export` writes the tracer contract.
6. The driver: `run.py` (supervisor, Producer, lookahead, verso gate, rounds, GPU modes), `cli run`.
   Accept: `test_run_e2e`.
7. Evaluation v2 + pretrain + ladder + status/stop/ledger. Accept: `rvsm eval`, `rvsm pretrain` 20 steps
   then `rvsm train --init`, `rvsm ladder` launches three sizes.

Then: push to github.com/SuperOptimizer/rvsm (commit trailers as in usrm2), build `libvolcomp.so` on the
A100 with portable flags, copy the two teacher `.pth` files from the desk (`/vesuvius/tsm/models/`), stop u4
(checkpoint kept), and `rvsm run --ct <Paris 4 2.4um URL> --umbilicus <json> --mode resident --gpus 0`.

## 8. Verification

- Unit + e2e suite above on the laptop (CPU); a GPU smoke of `rvsm produce` and `rvsm train` on the 5080
  with `size 5m, patch 128` (16 GB).
- On the A100: one region through `rvsm produce --teacher recto,m7` compared voxel-wise to the desk's
  existing recto teacher store for the same region (expect dice > 0.99 within one volcomp build); one
  student region in `--sign -1` compared to the pod's v2 output for the same checkpoint (bit-identical
  expected on the same GPU build, dice > 0.99 across builds).
- First real run: watch `logs/produce.jsonl` (s/region per pass), `logs/train.jsonl` (Mvox/s, train_wait_s
  must stay ~0 after warm-up), VRAM per process vs the budget table; `rvsm eval --tifxyz` on the held-out
  regions against the meshes at 10k and 20k steps; compare to u3/u4 numbers (recall 0.82, continuity
  0.70, merge 0.33, ERL 409 um at 21k).

## 9. Non-goals for v1

No planner process, queue replay, DDP, multi-node, multi-scroll runs; no mask-pyramid targets or published
ceilings; no sftp publishing or mirror conventions (optional `--publish rsync://` is v2); teachers recto +
m7 only; no explicit normals head, no `pair construct-only`, no Betti matching, no `--loss-warp`, no muP;
no tifxyz refinement tool, no `glc-weights`, no ablation presets.

## 10. Notes carried from usrm2 that must survive the port

- volcomp: `compressors=None`; q8 probabilities, q0 fields; shape multiples of 128; one write per shard;
  never rewrite a finished store in place; record `volcomp_build`; `rungs()` must fail loudly when the lib
  is missing (usrm2 silently dropped levels).
- Never `pkill` on an ssh command line; kill logic lives in scripts on the host.
- The desk (forlindesk2) is currently unreachable; the desk-side dist-pyramid deploy queued in
  `/home/forrest/.claude/plans/flickering-cooking-iverson-agent-a186423acb62fafb7.md` is independent of rvsm.
