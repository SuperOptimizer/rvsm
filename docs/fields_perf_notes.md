# GPU distance fields: performance notes (2026-09-24, updated 2026-09-25)

## Per-block device profile (RTX 5080, one 224^3 rung-2 block with sheets, B = 1)

After the 2026-09-25 round 3 (items E-H below):

| part | device time | launches |
|---|---|---|
| whole block (`_block_fields_t`) | 14.9 ms device, 17.4 ms wall | 89 kernels eager; per production batch 127 API calls eager, 25 with the graphs |
| stage B: forest merge 1.9, reciprocal kernel (with the finds) 1.2, box Gaussians 1.8, normals + walk, `nonzero` | 5.6 ms | 16 (14 under the graphs: the forest is in graph A) |
| face transforms incl. signed distance, index planes and reach rules (`edt.face_edt`: `_first` x2, `_pass` x2, `_pass_face` x2) | 3.4 ms | 6 |
| medial transforms (`edt2`, no indices) | 2.2 ms | 6 |
| max-filter (medial) | 1.1 ms | 2 |
| the rest of stage A (thresholds, medial compare, dilation, coverage, thickness) | ~2.1 ms | ~57 |
| stage C: rule 5, outputs / encoding, counts (`_stage5`) | 0.4 ms | 2 (1 kernel + the counts memset) |

Production batch (host inputs to encoded cores, `gprof.py` harness): B = 1 eager 16.0 ms device /
127 API calls, graphs 15.9 ms / 25; B = 3 eager 54.4 ms / 127, graphs 55.3 ms / 25.

Round 2's table (after 88827c0):

| part | device time | launches |
|---|---|---|
| whole block (`_block_fields_t`) | 23.7 ms device, 25.9 ms wall | 190 kernels eager; per production batch 27 API calls with the graphs |
| EDT passes (`_pass` x8 + `_first` x4, inside `edt2`) | 6.3 ms | 14 |
| stage B: forest merge 1.9, reciprocal kernel (with the finds) 1.2, box Gaussians 1.8, normals + walk, `nonzero` | 5.6 ms | 16 |
| stage C: stencil, gradient, fails, encode, counts kernel | 4.3 ms | 65 (1 graph launch) |
| face-distance elementwise (sqrt, side, where) | 3.8 ms | 26 |
| max-filter (medial) | 1.1 ms | 2 |
| the rest of stage A (thresholds, fails, dilation) | ~2.5 ms | ~65 |

The same block before the round (re-measured at 5e91243^ on the same laptop): 42.0 ms device,
44.3 ms wall, 514 launches, 5 syncs per batch. The 2026-09-24 table below it is kept for reference.

| part (2026-09-24, after 59297b8) | device time |
|---|---|
| whole block (`_block_fields_t`) | 47.4 ms device, 59.8 ms wall; 514 kernel launches |
| Gaussians (`_gauss`, 6 passes over the whole volume, for the face normals) | 7.0 ms |
| labelling (`_lab_iter` 5.1 + `_lab_jump` 2.7, 16 iterations each) | 7.8 ms |
| EDT passes (`_pass` x8: medial + face, recto + verso) + `_first` x4 | 5.7 + 1.2 ms |
| one large scatter/gather (the support-count scatter-add, not the pair checks) | 4.9 ms |
| the rest (elementwise, max-pool, cat, copies) | ~20 ms over ~450 launches |

Behind Thunder's GPU proxy the same region was ~75 s at batch 3 (launch-latency bound), against
~40 s on the laptop then. Launches and syncs cost more there than device time.

## Done 2026-09-25, round 3 (all byte-identical: fixtures bitwise vs numpy, synthetic 1024^3 region vs HEAD at batch 1 and 3, graphs on and off)

Per block = the block above, B = 1. "API" = kernel launches + memcpy/memset + graph launches of the
production batch path (host inputs to encoded cores).

| item | commit | block device ms | block launches | batch API eager / graphs | notes |
|---|---|---|---|---|---|
| before | a65b30e | 23.6 | 190 | 228 / 28 | batch device 24.9 eager, 23.2 graphs |
| E: rule 5 + outputs + encoding + counts in one kernel | 58256c3 | 19.7 | 127 | 165 / 28 | stage C 4.31 ms / 65 -> 0.39 ms / 2 |
| F: face distance arithmetic + reach rules in the last EDT pass | 78334e1 | 14.9 | 89 | 127 / 28 | face part ~8.2 ms / ~44 -> 3.4 ms / 6 |
| G: graph pool per key | 8b0c4ff | 14.9 | 89 | 127 / 25 | pool 0.77 -> 0.65 GB (B = 1), 2.30 -> 1.93 GB (B = 3) |
| H: mixed / short batches replay | bec8fbc | 14.9 | 89 | 127 / 25 | batch 3: replayed 112 -> 127 of the region's batches |

- **E** (`_stage_c`, kernel `_stage5`): per voxel the pair test at itself and its six neighbours (the
  codes are only read, so every voxel sees the pre-rule-5 codes), the midline's central differences
  at the full-stencil voxels straight from dr / dv (their neighbours are pair voxels), the gradient
  rule, valid, and either the masked midline / thickness / valid volumes (`block_fields_torch`) or the
  encoded core window (the producer), plus the counts row (one reduced atomic per value and program).
  torch's float32 steps each rounded on its own (`*_rn`). The torch steps stay for the CPU and
  RVSM_FIELDS_FUSED=0.
- **F** (`edt.face_edt`, kernel `_pass_face`): the face transform's last pass computes u = sqrt(v), the
  side (y - iy) * dy + (x - ix) * dx (rounded product, rounded product, rounded sum), d, writes the
  three index planes directly (no stack) and applies the no_recto / no_verso reach rule to the codes.
- **G**: (1) the index planes are int16 (numpy's `face_distance` does the same below 2^15): 2 x 68 MB
  instead of 2 x 135 MB per block, the largest statics; (2) the union-find forest is computed in
  graph A and, with one Gaussian buffer, allocated at the end of the capture, reusing the transients'
  dead pool memory (the pool did not grow); the Gaussian's middle pass writes into the dead forest
  (eager too: `gaussian3_box(out=)`); (3) `empty_cache()` before a capture, so the pool is not
  stacked on the first batch's cached eager blocks; (4) any batch of another key drops the graphs
  (`drop_other`), not only a graph-eligible one.
- **H**: a batch with at least one face pair goes through the graphs; its faceless windows are
  computed in full (the SKIP_FACELESS equivalence: the same bytes as `_faceless`), a short batch is
  padded with air windows (`_pad_batch`). A batch without any face pair stays eager (`_faceless`).
  Stats: `graphed_faceless`, `graphed_padded`.

**Memory now** (synthetic 1024^3 region, `max_memory_reserved`):

| | graph pool per key | eager peak allocated per batch | region peak reserved, graphs | region peak reserved, eager |
|---|---|---|---|---|
| B = 1, before | 0.81 GB | 0.74 GB | 1.51 GB | 0.99 GB |
| B = 1, now | 0.65 GB | 0.59 GB | 1.31 GB | 0.91 GB |
| B = 3, before | 2.43 GB | 2.21 GB | 4.45 GB | 2.61 GB |
| B = 3, now | 1.93 GB | 1.78 GB | 2.87 GB | 2.36 GB |

The remaining gap between graphs and eager at B = 1 is the pool beside the eager cache of the
batches that still run eagerly while it lives (the faceless ones, whose recto-only transforms
allocate), plus `_SlabStore`'s pooling on the helper thread's stream (~0.18 GB cached there).

Laptop region times (synthetic 1024^3, rungs 2-4): 22.9 s (batch 1, graphs), 24.4 s (batch 1, eager),
24.0 s (batch 3, graphs), 25.9 s (batch 3, eager); +-2 s run to run. The device work per block is
now well under the store decode and the encode tail.

## Done 2026-09-25 (all byte-identical: fixtures bitwise vs numpy, synthetic 1024^3 region vs HEAD at batch 1 and 3)

Per block = the one block above, B = 1; "API" = kernel launches + memcpy/memset + graph launches of
the production batch path (host inputs to encoded cores, `gprof.py`-style harness).

| item | commit | device ms | launches | notes |
|---|---|---|---|---|
| baseline | 3773ecb | 42.0 | 514 | 5 syncs / batch |
| A: normal Gaussians over the face points' box + fused pair checks | 5e91243 | 33.6 | 247 | Gaussians 6.6 -> 1.9 ms (box + int32 indexing), pair checks ~250 -> 5 launches |
| B: union-find labelling, roots found only where compared | 1b8bacf | 29.6 | 204 | labelling 7.0 ms / 46 launches / 4 syncs -> 1.9 ms merge / 2 launches / 0 syncs |
| C: CUDA graphs for stages A and C | 38165f9 | 29.7 | API 242 -> 27 per batch | syncs 4 -> 2 per batch; memory below |
| D: support counts as one reduction kernel | 88827c0 | 23.7 | 190 (API 228 eager / 27 graphs) | the 4.7 ms scatter-add -> ~0.05 ms |

- **A** (`edt.gaussian3_box`, `targets._pair_checks_triton`): the Gaussian is computed only over each
  (field, block)'s extent of the face points that passed the reciprocal test, widened by the gradient
  stencil and, per earlier pass, by the radius -- the same loads and float64 expression per voxel as
  the whole-volume kernel, clamping to the whole volume, so the same bits. Not per-point patches: on
  a dense block the face points are ~10 % of the voxels and the 13^3 per-point support would cost more
  than the volume; the box is one "patch" per block and costs no launch (it lives on the device). On a
  sparse real block (one sheet) the box shrinks to that sheet's extent. The normals, reciprocal test
  and walk are three kernels in `_pair_checks_torch`'s float32 order (libdevice `*_rn`, no contraction).
  RVSM_FIELDS_FUSED=0 runs the op-by-op torch pair checks.
- **B** (`edt.label`, `edt.label_forest`): Playne-Hawick union-find with atomic MAX linking (roots =
  each component's largest voxel = the old fixed point, bit for bit), path halving, one union per
  voxel inside a solid band (a neighbour adjacent to an already-joined one is skipped). The pair
  checks take the forest and find the roots of the four face points per voxel.
- **C** (`targets._FieldGraphs`, RVSM_FIELDS_GRAPHS=0 to disable): graph A = block inputs + rules
  1-3, graph C = rule 5 + counts + encode; stage B (`nonzero`, pair checks) eager between them. Full
  batches only (every window has both faces, `batch` of them), captured on the second full batch of a
  (rung, batch size, parameters) key, dropped on a new key, a fields yield and at region end. The
  medial saturation flag is copied to pinned memory in graph A and read after `nonzero`'s sync; a
  saturated batch is recomputed eagerly.
  **Memory cost** (round 2; round 3's numbers above): graph pool 0.81 GB at B = 1, 2.43 GB at B = 3 while the key's graphs live, on top of
  the eager allocations (stage B, eager batches). Peak reserved over the synthetic region: batch 1
  0.99 -> 1.51 GB, batch 3 2.61 -> 4.45 GB. If the producer budget cannot take it, RVSM_FIELDS_GRAPHS=0.
- **D**: one Triton reduction per 4096 voxels and volume, 11 atomic adds per program, straight into
  the (B, 11) support rows.

Laptop region times (synthetic 1024^3, rungs 2-4): 36.4 s (batch 1) / 37.8 s (batch 3) before, ~29-31 s
after, now dominated by the store decode and the volcomp encode tail; run-to-run noise is +-2 s.
Behind the proxy the API-call count per full batch (242 -> 27) and the syncs (4 -> 2, plus the medial
checks folded into one) are what should move the ~75 s.

## Done 2026-09-24 (all byte-identical to the previous output, tested)

| item | commit | effect (laptop, synthetic 1024^3 region, batch 1) |
|---|---|---|
| fields yield the GPU to blocking passes | 3fc79dc | a teacher / self pass waits at most one fields batch |
| 5: skip faceless blocks | 483b19f | 216 of 584 blocks skipped; region 60-67 s -> 53-54 s |
| 8: pooled chunk encode | ea39269 | field store 8.2 -> 3.0 s; region 51.4 -> 41.9 s |
| 9: reach-capped EDT | 59297b8 | block 62.0 -> 59.8 ms; region 42.3 -> 40.0 s |

## Dropped

- **7, larger cores (256 + 48 halo):** cannot be byte-identical. The pair checks compare
  connected-component labels (`edt.label`) computed over the whole block window, so two face voxels
  joined only outside a 224^3 window but inside a 352^3 one are judged differently. Measured: core
  64 vs 128 (same halo) differs on 35 of 4.3 M valid rung-2 voxels of the curved test region.
- **10, int16 / int8 intermediates:** bounded by the EDT passes' 5.7 ms of 47.4 ms (at most ~3 ms a
  block), and an uncapped fallback needs int32 anyway, so it would need a second kernel variant.
- **8b, pool rungs 3 / 4 fields from rung 2:** not what the code computes (rungs 3 / 4 are recomputed
  from the pooled recto / verso), so not byte-identical.

## Next steps, ranked (2026-09-25, after round 3)

1. **Stage B** (5.6 ms, 14-16 launches, the largest part now): the forest merge (1.9 ms) could run only
   over the band voxels the candidates' face points can reach; the box Gaussians (1.8 ms) could be
   one kernel for the three passes over a box in shared memory.
2. **The rest of stage A** (~2.1 ms, ~57 launches eager; one graph launch under the graphs): the
   thresholds, the medial compare, the dilation, the coverage and thickness rules into one or two
   kernels (the coverage and thickness rules into the verso face pass's epilogue would also drop u
   from the verso transform: ~45 MB less pool per block).
3. **Medial transforms** (2.2 ms + 1.1 ms max-filter): the transforms carry an unused index output
   (`_first` writes it); a no-index `_first` and the 3^3 max-compare fused into one kernel.
4. **Graph memory**: int16 carries inside the passes (the index intermediates of the verso face
   transform are the pool's peak), and the verso u above; each ~45 MB per block.

## Round 2's next steps (2026-09-25; 1-4 done in round 3 as E, F, G, H)

1. **Fuse stage C into one kernel** (~4.3 ms, 65 launches eager): pair, the 6-neighbour stencil, the
   midline central differences (computable from dr / dv / pair at the neighbours), both fails, valid,
   the outputs and the encoding of the core, in torch's float32 order with `*_rn`. Only device time
   under the graphs (the launches are already one graph launch).
2. **Fuse the face-distance elementwise** (3.8 ms): sqrt, side, sign in the last EDT pass's epilogue.
3. **Graph memory**: share one pool between the graphs and stage B's eager buffers, or keep the box
   Gaussian's and forest's buffers static, to bring the batch-3 peak back towards 2.6 GB.
4. **Mixed batches**: at batch 3 a batch with one faceless window runs eagerly (no graph); on a region
   with many such batches, pad them through the graph or regroup the full windows.

## Older next steps (2026-09-24; 1-4 done above)

1. **Gaussians only around the selected face points** (~6 ms, ~13 % of device time, plus 6 whole-volume
   launches). The normal reads the smoothed d one voxel around each face point p_r / p_v of the
   voxels that reach the pair checks. The separable Gaussian rounds to float32 per pass, per voxel,
   so computing it on 15^3 patches around those points, in the same pass order, gives the same bits.
   Exact; moderate work. Test against the whole-volume path on every fixture.
2. **CUDA-graph the fixed-shape stages** (launch count 514 -> ~200-250; small on the laptop, the
   largest win behind the proxy). Stage A (thresholds, medial and face EDTs, the per-voxel rules up
   to the pair-check candidates) and stage C (stencil, gradient, counts, encode) are fixed-shape. Leave
   stage B (the `nonzero` selection, labelling, pair checks, walk) eager, and keep the medial
   saturation check outside the graph (fall back to eager). Costs: a graph pool of ~1-2 GB reserved
   while it lives. Free it at the end of each rung / region and on a `fields_yield`, because the
   producer budget is tight. Keep an env switch.
3. **Label loop** (~7.8 ms, 32 launches plus a sync every 8 iterations). Label only the band voxels
   the pair checks touch, or start from a cheaper seed (e.g. per-slab labels merged). Must keep the
   "largest 1 + index" fixed point, or any labelling whose equality classes are the same.
4. **The large scatter/gather** (4.9 ms): the pair checks' gathers over whole-volume index tensors.
   Gather only at the selected voxels' face points (already int32); check for full-volume
   intermediates that can shrink.
