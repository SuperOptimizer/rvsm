# GPU distance fields: performance notes (2026-09-24, updated 2026-09-25)

## Per-block device profile (RTX 5080, one 224^3 rung-2 block with sheets, B = 1)

After the 2026-09-25 round (items A-D below):

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
  **Memory cost**: graph pool 0.81 GB at B = 1, 2.43 GB at B = 3 while the key's graphs live, on top of
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

## Next steps, ranked (2026-09-25)

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
