# Trainer speed notes: compile mode, CUDA graphs, activation checkpointing (2026-09-24)

Measured on the laptop RTX 5080 (16 GB, WSL2), torch 2.14.0+cu130, with the real step loop
(`train.train`: every loss term, deep supervision, EMA, the self-cascade net compiled
`max-autotune-no-cudagraphs`), synthetic items, batch 1 x accum 2, bf16 autocast, NCDHW,
`TORCHINDUCTOR_COMPILE_THREADS=1`, 30 steps, a fresh inductor cache dir per mode ("cold"). Step time
is the median over the second half of the run. The allocator was capped at 13.7 GB
(`set_per_process_memory_fraction(0.86)`), because WSL spills an over-committed card into host memory
and runs about 10x slower instead of raising OOM. Bench scripts are not in the repo; the step bench is
`TR.train` with a timestamping `patches_factory`.

## What fits on 16 GB

30m6 at patch 256 **does not fit at any `ckpt_act`** (0, 2, 4 and 6 all OOM at 12-13 GB). Checkpointing
only the net's blocks leaves the full-resolution 14-head outputs, their fp32 copies and the loss
tensors, and at 256^3 those alone exceed the card. At patch 192, `ckpt_act` 0 runs but thrashes
against the cap (10.9 s/step, reserved = cap), and `ckpt_act` 2 OOMs on fragmentation. **Patch 160 is
the largest clean size**, so the comparison below uses 30m6 / patch 160.

## Compile modes (30m6, patch 160, ckpt_act 0)

| trainer mode | cold compile* | step | peak alloc | peak reserved | graph breaks | cudagraph skips |
|---|---|---|---|---|---|---|
| default (current) | 137 s | 0.883 s | 7.59 GB | 11.69 GB | 0 | n/a |
| max-autotune-no-cudagraphs | 280 s | 0.848 s (-4 %) | 7.59 GB | 11.69 GB | 0 | n/a |
| max-autotune (CUDA graphs) | **OOM at the first captured step** | - | 7.25 GB + ~6 GB graph pool | 13.69 GB (cap) | 0 | 0 |
| reduce-overhead | not measured (session ended) | | | | | |
| max-autotune, warm cache | < 10 s to the same OOM (whole run 10.3 s) | - | 7.25 GB + graph pool | 13.69 GB (cap) | 0 | 0 |

\* "cold compile" = the first two steps minus two steady steps. It includes the cascade net's own
`max-autotune-no-cudagraphs` compile, which is the same in every row. "Not enough SMs to use
max_autotune_gemm mode": on this laptop part inductor skips GEMM/conv template autotuning, so
max-autotune here tunes only the Triton pointwise and reduction kernels. On the A100 (108 SMs) the
conv templates are benchmarked too: expect a much longer cold compile there, and behind Thunder's GPU
proxy every benchmark is a round trip (the producer's max-autotune student compile once sat 25+ min in
it).

**CUDA graphs roughly double the step's device memory.** The captured forward and backward keep their
activations in the graph's private pool, and that pool does not share memory with the regular
allocator's blocks. The max-autotune run had 13.40 GB allocated when it died, against 7.25 GB for the
tensors the regular allocator counts. The default run's peak is 7.6 GB, so the graphs add about 6 GB
here. Scaled to the A100 production step (27.4 GB trainer peak at 256^3, 43 of a 47.5 GB budget in
use), that is far past the ~4.5 GB of headroom. **With today's memory budget, do not enable CUDA
graphs on the A100** unless `ckpt_act` is raised to pay for the pool (not measured), or the budget
grows.

### Graph breaks, recompiles, shapes

* 0 graph breaks and 0 cudagraph skips in every mode (`torch._dynamo.utils.counters`, and
  `TORCH_LOGS=perf_hints,graph_breaks,recompiles`).
* 2 frames / 2 unique graphs, and one "recompile" logged: `UNet.forward` is ONE code object shared by
  the training net (grad on) and the cascade net (`no_grad`), so dynamo keeps two cache entries
  (guard `GLOBAL_STATE changed: grad_mode`). It is not a per-step recompile. The eval net adds more
  entries of the same code object at an evaluation. They all count against dynamo's
  `recompile_limit` (default 8) for that code object, and past it the frame falls back to eager.
  Watch for `recompile_limit` warnings in run.log if more shapes or modes are added.
* **No per-rung shapes.** Every rung's training patch is `patch^3` rung-k voxels, so the compiled net
  sees one input shape whatever the rung mix. Only the net is compiled. The data-dependent parts (the
  ECT's rung-2 selection, the pair's paired-support mask, the aux terms) run eagerly outside it, and
  none of them recompiles or breaks the graph. Under CUDA graphs that means one captured graph for
  the forward and one for the backward, not one per rung.
* The CUDA-graph step loop needed two fixes (commit `cf95401`). Each forward calls
  `torch.compiler.cudagraph_mark_step_begin()`. The gradients live in stable zeroed buffers, zeroed in
  place (`zero_grad(set_to_none=False)` under graphs). Without the buffers, the accumulation of
  microbatch 2 read a `.grad` that the replay had overwritten (torch raised "accessing gradient
  tensor output of CUDAGraphs that has been overwritten"). Every 30m6 parameter receives a gradient
  every step, so the zero buffers change nothing the optimiser sees.
  `test_the_step_loop_runs_under_cuda_graphs` covers it with `reduce-overhead` on the 1m net.

## Activation checkpointing (`ckpt_act`) and CUDA graphs (2026-09-25)

Same bench as above (real `train.train` step loop, synthetic items, 30m6, batch 1 x accum 2, bf16,
13.7 GB allocator cap, `TORCHINDUCTOR_COMPILE_THREADS=1`), 24-30 steps, step = median of the second
half. 30m6 has six levels, so `ckpt_act` runs 0-6 (level i is checkpointed when i < ckpt_act; 6 = every
block). "Graph pool" is the cudagraph trees' private pool, read from `torch.cuda.memory_snapshot()`
(`segment_pool_id` != (0, 0)). `max_memory_allocated` does NOT count that pool, so under CUDA graphs
compare reserved, not allocated.

### Patch 160, trainer mode default

| ckpt_act | step | vs 0 | peak alloc | peak reserved |
|---|---|---|---|---|
| 0 | 0.906 s | - | 7.56 GB | 11.38 GB |
| 1 | 0.978 s | +8 % | 7.52 GB | 10.47 GB |
| 2 | 1.047 s | +16 % | 6.97 GB | 10.56 GB |
| 3 | 1.082 s | +19 % | 6.89 GB | 11.14 GB |
| 4 | 1.038 s | +15 % | 6.86 GB | 11.14 GB |
| 5 | 1.094 s | +21 % | 6.82 GB | 11.13 GB |
| 6 | 1.061 s | +17 % | 6.80 GB | 11.14 GB |

`RVSM_TRAIN_COMPILE_MODE=max-autotune` (CUDA graphs) at patch 160: **every level, 0 to 6, OOMs** at
the 13.7 GB cap in the first captured steps (microbatch 4-6). At `ckpt_act` 6 the OOM snapshot has
5.27 GB in the regular allocator and **8.42 GB in the graph pool**.

**Checkpointing buys almost no memory.** Going from 0 to 6 lowers the peak allocation by 0.76 GB (10 %)
and the reserved memory by nothing. `RVSM_PROFILE=1` shows why. The per-phase peaks at `ckpt_act` 0 / 6
are: cascade self pass 5.05 / 5.38 GB, aug 3.97 / 4.61, forward + deep losses 6.59 / 6.35,
backward 7.56 / 6.80, idle baseline 2.2 / 2.5. The step's high water is the full-resolution
working set, not the stored activations: the 14 head outputs and their fp32 copies, the loss
temporaries, the batch after the augmentation, and each conv's own transient buffers (a no-grad
cascade forward alone takes ~2.8 GB above the baseline). Inductor's partitioner already recomputes
the cheap GroupNorm/SiLU tensors, so `ckpt_act` removes only the saved conv outputs.

### Patch 128 and 144 (where CUDA graphs fit)

| patch | mode | ckpt_act | step | regular reserved | graph pool | total reserved |
|---|---|---|---|---|---|---|
| 128 | default | 0 | 0.492 s | 6.50 | - | 6.50 GB |
| 128 | default | 2 | 0.533 s | 6.51 | - | 6.51 GB |
| 128 | default | 6 | 0.570 s | 6.51 | - | 6.51 GB |
| 128 | max-autotune-no-cudagraphs | 0 | 0.485 s | 6.50 | - | 6.50 GB |
| 128 | max-autotune-no-cudagraphs | 6 | 0.555 s | 6.51 | - | 6.51 GB |
| 128 | max-autotune (graphs) | 0 | 0.475 s | 4.13 | 4.72 | 8.85 GB (+36 %) |
| 128 | max-autotune (graphs) | 2 | 0.572 s | 4.12 | 3.63 | 7.75 GB (+19 %) |
| 128 | max-autotune (graphs) | 6 | 0.574 s | 4.12 | 3.62 | 7.74 GB (+19 %) |
| 144 | default | 0 | 0.640 s | 9.02 | - | 9.02 GB |
| 144 | default | 2 | 0.753 s | 8.88 | - | 8.88 GB |
| 144 | max-autotune (graphs) | 0 | 0.633 s | 5.24 | 6.24 | 11.48 GB (+27 %) |
| 144 | max-autotune (graphs) | 2 | 0.737 s | 5.04 | 5.90 | 11.58 GB (+30 %) |

(+x % = against default mode at the same patch and `ckpt_act` 0.) Every graph run captured exactly
two nodes (the forward and its backward), with no re-records and no cudagraph skips.

* **The graph pool does not shrink with `ckpt_act` past level 2.** At 128 the pool is 4.72 GB at 0 and
  3.6 GB at both 2 and 6. At 144 checkpointing saves 0.34 GB of pool (6.24 -> 5.90 GB) and the total
  does not go down at all. The pool holds the captured forward+backward's whole transient working
  set, which checkpointing does not reduce. It cannot share blocks with the loss/aug/cascade tensors
  in the regular allocator, so the step costs pool + the rest instead of the maximum of the two.
* **The graph gain is small.** At `ckpt_act` 0: -3.5 % at patch 128, -1 % at 144 against default
  (and -2 % against `max-autotune-no-cudagraphs` at 128). With checkpointing on, the graph runs are no
  faster than the default runs at the same level (128/2: 0.572 vs 0.533 s, 144/2: 0.737 vs 0.753 s).
* **The recompute costs 8-18 %:** `ckpt_act` 2 is +8 % at patch 128, +18 % at 144 and +16 % at 160;
  `ckpt_act` 6 is +16-17 %.

### Extrapolation to the A100 (patch 256)

Reserved memory fits `B + a * V`, with V = (patch / 128)^3 the activation volume (patch 256: V = 8).
From the default-mode `ckpt_act` 0 points (6.50 GB at V = 1, 11.38 GB at V = 1.95): a = 5.1 GB,
B = 1.4 GB, so **42 GB at 256**. The same fit on peak allocation gives **27.0 GB at 256**. Both match
the host (~43 GB in use, 27.4 GB peak), so the scaling holds up.

In graph mode the total is 1.19-1.36x the default-mode total at 128-144, and at 160 it is more than
1.2x (OOM at 13.7 GB). The absolute increase is +2.35-2.7 GB at V = 1-1.42, about +1.7-1.9 GB per unit
of V. At 256 that is **+14-15 GB, so a trainer at ~56-58 GB**, and `ckpt_act` changes it by at most
~2-3 GB:

| ckpt_act at 256 (est.) | peak alloc, default | reserved, default | reserved, CUDA graphs | fits 47.5-49.5 GB? | step vs today |
|---|---|---|---|---|---|
| 0 (today) | ~27 GB | ~42-43 GB | ~56-58 GB | no | graphs -1 to -3.5 % (laptop) |
| 2 | ~24.6 GB | ~42 GB | ~54-57 GB | no | +8-18 % recompute, no graph gain on top |
| 6 | ~23.9 GB | ~42 GB | ~54-57 GB | no | +16-17 % |

The ask was ~4-8 GB for the pool; the measured need scales to ~14 GB, and **no `ckpt_act` level makes
CUDA graphs fit at patch 256 in the trainer's budget**. Even if one did, the recompute (+8-18 %)
would cost more than the laptop's graph gain (1-3.5 %). The Thunder proxy's per-launch latency could
make the graph gain larger on the host, but the host's measured GPU duty cycle (mean 88 %, p50 96 %,
trainer and producer sharing the card) bounds what removing launch gaps can recover to roughly 5-10 %.
That still does not pay for the recompute.

**Recommendation: keep `ckpt_act` 0 and `RVSM_TRAIN_COMPILE_MODE=max-autotune-no-cudagraphs`**
(what the host runs). Do not raise `ckpt_act` to make room for CUDA graphs. Graphs would need either a
smaller training patch (~200 or below) or ~15 GB more trainer budget. If the budget ever grows, measure
first: 20 steps at `max-autotune`, `ckpt_act` 0, then read `memory_reserved` and nvidia-smi.

## Producer inference under CUDA graphs (`RVSM_STUDENT_COMPILE_MODE`, fast_teacher) (2026-09-25)

The same laptop and torch. `infer.run_region` drives the real `Student.plane_fn` (5 planes: recto,
verso, midline, thickness, conf) and the real `fast_teacher` + `teacher_fn` (m7). A fixed-shape
synthetic input stands in for the CT: 21-channel windows for the student, a 320^3 noise CT for m7.
Batch 1, 8 windows per region. The student window is **192 (halo 24), not 256**: at 256 the compiled
30m6 forward allocates one (1, 96, 256^3) **fp32** buffer (6 GB, the level-0 decoder concat) and does
not fit next to a region on 16 GB. m7 runs at its production window, 192. One process per mode.
Random weights (seed 0), then `Student.reload` of a second checkpoint (seed 1) into the same
compiled module.

| path | mode | first call | fwd / window | run_region / window | peak reserved | reserved after `empty_cache` | graph nodes |
|---|---|---|---|---|---|---|---|
| student 30m6 | default | 31.8 s | 156 ms | 161 ms | 7.61 GB | 1.47 GB | - |
| student 30m6 | max-autotune-no-cudagraphs | 57.8 s | 159 ms | 160 ms | 7.61 GB | 1.47 GB | - |
| student 30m6 | max-autotune (graphs) | 15.4 s* | 157 ms | 162 ms | 9.23 GB | **7.41 GB** | 1 |
| m7 | default | 24.0 s | 162 ms | 157 ms | 5.46 GB | 1.21 GB | - |
| m7 | max-autotune-no-cudagraphs | 108.8 s | 159 ms | 160 ms | 5.46 GB | 1.21 GB | - |
| m7 | max-autotune (graphs) | 16.4 s* | 160 ms | 158 ms | 4.80 GB | **4.11 GB** | 3 roots (re-records) |

\* warm autotune cache from the no-cudagraphs run just before.

Outputs (q8 = `infer.u8_t` of the bounded planes; `midline` / `thickness` compared in float):

| comparison | student q8 max \|dq\| (voxels differing) | student fields max \|d\| | m7 q8 max \|dq\| |
|---|---|---|---|
| cudagraphs vs no-cudagraphs | **0 (bit-identical)**, also after the reload | 0 | **0**, also after 3 offload/onload cycles |
| either autotune vs default | 2 (7.8 %) | 0.03 voxel | 45 (56 %)** |

\** m7 on pure noise CT, a chaotic input where bf16 kernel-choice differences flip the softmax. This
is the autotune kernel choice, not graphs. It says nothing about real CT.

**Correctness: the student path works under CUDA graphs without a code change.** Nothing in the
`run_region` loop syncs the host (the air test `window_any` runs once, before the loop). Every
window has the same shape, and the cascade's coarse windows are padded to it. The cudagraph trees copy
each fresh `prep()` tensor into the static input. `plane_fn` consumes the graph's output (`torch.cat`
of the planes) before the next replay. `Student.reload` copies weights INTO the captured parameters,
so the replay sees the new checkpoint (same output as the no-graph module with those weights). Counts:
0 skips, 0 graph breaks, 0 recompiles, 1 captured node.

**Speed: no gain on the laptop.** The forward time per window is the same within 2 % in all three
modes (156-159 ms), because a 192^3 window of 30m6 is far past launch-bound. Behind Thunder's proxy the
launch latency is higher (the GPU distance fields were launch-bound there), so the host could gain
something. The live trial measures that.

**Memory: the cost is a permanent pool.** Without graphs, a pass's working set goes back to the
caching allocator, and the next teacher pass, the fields or the region accumulators reuse it. With
graphs it stays in the cudagraph pool for the life of the process (`empty_cache` releases nothing):
+5.9 GB held at window 192 (7.41 vs 1.47 GB after `empty_cache`). By activation volume
((256/192)^3 = 2.37) that is **~14 GB held permanently at the production window 256**, plus ~+4 GB on
the student pass's own peak (+1.6 GB at 192). The producer runs at 27.6 of its 27.7 GB budget, with
peaks of 15.4 GB (teacher pass) and 21.0 GB (verso pass). A 14 GB pool that the teacher pass cannot
reuse very likely pushes it over. That is the thrash-then-OOM pattern of the 2026-09-24 producer
stalls, which ran silent at the cap for ~30 minutes.

**m7 / fast_teacher under graphs:** the outputs are correct, but the TeacherBank's offload/onload
(`param.data` to pinned host and back) gives the parameters new device addresses. The cudagraph trees
then **re-record** the graph: 3 root nodes after 3 cycles. Torch allows 128 unexpected re-records per
function, then falls back to the no-graph path. The pool held 2.9 GB at the production window
(4.11 vs 1.21 GB). `fast_teacher`'s mode is not an environment switch (the bank always compiles
`max-autotune-no-cudagraphs`), so nothing changes in production. Do not switch it to graphs while the
bank offloads.

**Recommendation for the live trial:** turn `RVSM_STUDENT_COMPILE_MODE=max-autotune` off. Go back to
`max-autotune-no-cudagraphs`: the same kernels, bit-identical output, no permanent pool. Keep graphs
only if the host shows BOTH a verso `gpu_s` clearly below the no-graph ~51 s/region AND no
`vram_pressure` or near-cap `vram_report` lines through at least one teacher pass after a verso pass.
The memory math above says the second condition will fail.

**Lead for later (not changed):** activations run in fp32 through the whole net under autocast.
`GroupNorm` is on autocast's fp32 list, so every block output, every skip and every decoder concat is
an fp32 tensor. At window 256 the level-0 concat alone is 6 GB (compiled inference), and training is
the same. Keeping them in bf16 (casting the GroupNorm output) would roughly halve the activation
memory, in the producer and in the trainer (where it might be what makes graphs fit). That changes
numerics, so it needs its own eval.

## Compile threads

`RVSM_TRAIN_COMPILE_THREADS` sets `torch._inductor.config.compile_threads` in the trainer process
only. The environment variable is left alone because the spawned producer inherits it. With the
run's `TORCHINDUCTOR_COMPILE_THREADS=1` the trainer compiled in-process: 1 child process (the
bench's own), 0.1 GB child RSS, 3.3-4.0 GB trainer RSS. Per-worker RSS at 2-4 threads was **not
measured**. The 40 GB incident was 8 workers in the producer, about 5 GB each, so 2 trainer workers is
a reasonable first try.

## Recommendation for the A100 (43 of 47.5 GB in use)

1. Keep the trainer's default mode for now. `max-autotune-no-cudagraphs` bought 4 % here (the GEMM
   templates were not tuned on this part, so the A100 may gain more) at twice the cold compile, and
   costs no memory. It is the safe upgrade if a longer first compile is acceptable:
   `RVSM_TRAIN_COMPILE_MODE=max-autotune-no-cudagraphs`.
2. `max-autotune` / `reduce-overhead` (CUDA graphs): **not on the A100 at patch 256**, at any
   `ckpt_act`. Measured 2026-09-25 (section above): the graph pool adds ~14-15 GB at 256, checkpointing
   takes at most ~2-3 GB of it back and costs 8-18 % per step, and the graph gain was 1-3.5 %.
3. Persist the inductor cache (`TORCHINDUCTOR_CACHE_DIR` on the run disk) so a restart is warm.

Env lines (none needed to keep today's behaviour):

    # safe, no extra memory:
    export RVSM_TRAIN_COMPILE_MODE=max-autotune-no-cudagraphs
    export RVSM_TRAIN_COMPILE_THREADS=2
    export TORCHINDUCTOR_CACHE_DIR=<out>/../inductor_cache
    # CUDA graphs: do not (needs ~15 GB more than the trainer's budget at patch 256, any ckpt_act)

## Missing measurements

reduce-overhead; the graph gain ON the A100 behind the Thunder proxy (the laptop measures launch
latency without the proxy); compile-worker RSS at 2-4 threads; the eval-grid prefetch timing (commit
`23ee7c6`, tested for identical results but not timed).
