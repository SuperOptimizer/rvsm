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

## Activation checkpointing (`ckpt_act`)

Not measured at a size that fits, because of the session's time box. At 256 every level OOMs (above).
The A100 numbers in `docs/recipe.md` still stand: at batch 1, `ckpt_act` 0 runs the backward in 298
ms against 396 ms at `ckpt_act` 1, with a 27.4 GB peak.

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
2. `max-autotune` / `reduce-overhead` (CUDA graphs) remove the per-launch latency the Thunder proxy
   adds, which is where the host would gain most. But the graph pool measured here is about as large
   as the step's own activations, and the A100 has ~4.5 GB free. Enable it only after a memory
   test on the host at the production config: a short run with `RVSM_TRAIN_COMPILE_MODE=max-autotune`
   and a raised `ckpt_act` (for example 1), reading `vram_MiB` and nvidia-smi.
3. Persist the inductor cache (`TORCHINDUCTOR_CACHE_DIR` on the run disk) so a restart is warm.

Env lines (none needed to keep today's behaviour):

    # safe, no extra memory:
    export RVSM_TRAIN_COMPILE_MODE=max-autotune-no-cudagraphs
    export RVSM_TRAIN_COMPILE_THREADS=2
    export TORCHINDUCTOR_CACHE_DIR=<out>/../inductor_cache
    # CUDA graphs: only after a host memory test at the production config
    # export RVSM_TRAIN_COMPILE_MODE=max-autotune

## Missing measurements

reduce-overhead; max-autotune step time (at a size where the graph pool fits, for example patch 128); the `ckpt_act` step-time / VRAM curve at patch 160; compile-worker RSS at
2-4 threads; the eval-grid prefetch timing (commit `23ee7c6`, tested for identical results but not
timed).
