# GPU distance fields: performance notes (2026-09-24)

## Per-block device profile (RTX 5080, one 224^3 rung-2 block with sheets, B = 1, after 59297b8)

| part | device time |
|---|---|
| whole block (`_block_fields_t`) | 47.4 ms device, 59.8 ms wall; 514 kernel launches |
| Gaussians (`_gauss`, 6 passes over the whole volume, for the face normals) | 7.0 ms |
| labelling (`_lab_iter` 5.1 + `_lab_jump` 2.7, 16 iterations each) | 7.8 ms |
| EDT passes (`_pass` x8: medial + face, recto + verso) + `_first` x4 | 5.7 + 1.2 ms |
| one large scatter/gather (pair-check gathers) | 4.9 ms |
| the rest (elementwise, max-pool, cat, copies) | ~20 ms over ~450 launches |

Behind Thunder's GPU proxy the same region is ~75 s at batch 3 (launch-latency bound), against
~40 s on the laptop now. Launches and syncs cost more there than device time.

## Done (all byte-identical to the previous output, tested)

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

## Next steps, ranked (estimates are per block on the 5080 unless stated)

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
