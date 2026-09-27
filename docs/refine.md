# rvsm refine: snapping published surfaces onto the recto/verso predictions

`rvsm/tools/refine/refine.py` (`rvsm refine ...` or `python -m rvsm.tools.refine.refine ...`) ports the
usrm2 refiner (`usrm2/refine.py`) onto rvsm's stores. It moves every point of a published tifxyz surface
along its normal onto the nearest recto probability peak. It checks each move against the verso
prediction, keeps render3d's drag anchors fixed, and writes the surface back in the frame it was read in.

## Algorithm (per iteration, search radius from `--far`: default `40,24,16,8`, one number = fixed)

1. **Grid.** Each surface is read, mapped to the fine 2.4 µm frame, and cropped to the rows and columns
   that touch the box. It is then upsampled so the grid pitch is about `--pitch` voxels (`upsample`; node
   (r, c) becomes node (r·up, c·up), so the published nodes are kept exactly).
2. **Normals.** Normals are hole-tolerant: they are computed on the hole-filled grid (`filled`) and then
   masked back. They are oriented **outward from the umbilicus**, which is also VERSO → RECTO
   (`export.SIGN_CONVENTION`). They are computed ONCE, on the published (upsampled) grid, and every node
   moves only along its own normal for the whole run: its displacement is a scalar. `--local-normal`
   recomputes them every iteration, as before. That lets nodes slide along the sheet and bunch, which
   is what the first local-following preview showed.
3. **Peaks.** The recto probability is sampled along the normal over ±r. The strongest 6 local maxima are
   kept (`local_maxima`), each placed to sub-voxel precision with a parabolic fit. A two-sample plateau
   now counts as ONE peak. usrm2 counted it twice, which let two sheets share one band.
4. **Verso term** (with `--verso`). The verso maxima on the same ray are found the same way.
   `verso_adjust` then does two things:
   - **Reject a verso-first candidate.** If a verso peak lies strictly between the vertex and a recto
     candidate, on the candidate's side and more than `--verso-margin` from both ends, the candidate is
     rejected. Reaching it would cross a sheet's back face, i.e. jump a wrap. With `--verso-block P`
     the candidate is penalised by P instead of rejected. A verso peak AT the vertex does not block: that
     case is a surface sitting on the back face of its own sheet.
   - **Pairing bonus.** If a verso peak sits one thickness T inward of the candidate (the sheet's own
     back face), the candidate earns `beta · verso strength · exp(-((gap - T)/tol)²/2)`.

   T comes from `--thickness-store` sampled at the vertex (code × 0.25 voxel; 0 = no data). Otherwise
   it is `--thickness`. Otherwise it is estimated once per run as the median recto-to-inward-verso
   spacing (2–24 voxels) of the strongest peak on each ray (`thickness_from_peaks`).
5. **Joint, order-preserving assignment** (`assign`). Along each ray, the nearest other sheet below and
   above (`ray_neighbours`, over ALL surfaces in the box) and this sheet are matched one-to-one and in
   order to the peaks. The cost is `0.08·|offset − peak| − strength`. Neighbouring sheets can therefore
   neither merge onto one band nor cross. Before the match, a candidate is also bounded by half the
   distance to the nearest other sheet on the ray (else by far), so a wide first pass cannot jump a
   wrap; after the smoothing the move is clipped to the same bound, taken over every neighbour within
   2r + 2 (that neighbour moves too, by at most r), so two neighbours cannot cross.

   **Duplicate traces are not neighbours** (`duplicate_sheets`, `--dup-gap`, `--dup-frac` 0.4).
   Published segmentations overlap: several segments trace the same wrap a few voxels apart. On the
   2.4 µm strip, 90% of the inter-sheet node pairs within 24 voxels were such duplicates while the wraps
   are ~44 voxels apart; treated as neighbours, they walled each other off (the first version of this
   pass: 'capped' 0.6–0.96, 'with_peak' 0.14–0.27, every published pair "under min gap"). Per tile, the
   wrap spacing is measured on the recto (median gap between consecutive peaks on a ray, `wrap_spacing`)
   and dup gap = 0.4 × it (at least `--min-spacing`). Two sheets whose published nodes are, in median,
   nearer than that are one wrap: they ignore each other and both snap to its band. Decided once, on
   the published grids, so a pair never switches between wall and duplicate as it moves. Among peaks within `--peak-tol` (0.15) of the strongest, the one nearest the current position
   wins, not simply the strongest. The neighbour search's lateral tolerance is 0.75 × pitch (at least
   3).
6. **Smoothing.** The per-point offsets are smoothed over the (H, W) grid with a Gaussian weighted by
   peak height (`smooth`). σ = `--sigma-vox` voxels along the sheet, default 2 × `--pitch`, so ~10
   at pitch 5 and it follows locally; usrm2's 40 flattened everything under ~100 voxels. The LAST
   iteration uses `--sigma-final` (default σ/2), so the early, wide passes stay robust and the final
   pass follows the band locally. The moves are faded to 0 over
   `--taper` voxels at the box faces, so a slab-refined surface joins its untouched remainder without a
   step. They are also faded around anchors (below). All sheets move only after all of them have been
   measured.
7. **Keeping the grid a grid.** After each move:
   - **Reparametrisation** (`--reparam`, OFF by default). For every run of nodes along a grid row, then a column, if
     some edge's length ratio to the published grid departs from the run's median by more than 15%, the
     run's nodes slide ALONG their own polyline back to the published arc-length fractions (end nodes
     fixed). This undoes bunching exactly and keeps the shape. An evenly stretched run, or the slight
     kink where moves fade out, is left alone. Off by default: with normal-only moves nodes cannot slide
     in the first place, and on a real (jagged) snap the redistribution along the jagged polyline itself
     drifted nodes 11–37 voxels sideways and cut the on-band fraction from 0.70 to 0.41 (tile 1 of the
     strip). It still evens out a smooth out-of-phase sheet (the wavy test pins it on).
   - **Tangential relaxation** (`--relax` 0.5, `--relax-iters` 2). A rest-relative Laplacian
     (L(P) − L(P_published)), projected onto the refined surface's tangent plane. It keeps the
     published parametrisation, and it is zero for a pure normal offset of a smooth sheet.
   - **Fold guard** (`--fold-guard`, on). A node is pulled back by bisection toward its previous
     position (up to 4 halvings, then fully) if a quad touching it has flipped. A flip means either
     triangle's (u × v)·n changed sign against the published grid. The same applies if one of the
     node's grid edges is now shorter than `--min-spacing` (0.4 × pitch) without having been so short
     when published. The pull-backs are counted as fold events.

   On a published wavy sheet half a wavelength out of phase with its band, the old scheme gives spacing
   CV 0.87 and 200 folded nodes. With `--reparam` it gives CV 0.06, no fold, and a fit within 0.07 voxel
   (`test_wavy_band_keeps_even_spacing_and_never_folds`). On the synthetic slab, the mean tangential
   drift is 0.03 voxel.
8. **Joint mesh solve** (`--mesh-opt`, off). After the snap, torch (on the GPU when there is one)
   optimises every sheet of the tile together with Adam (`--mesh-steps` 100, `--mesh-lr` 0.1). The
   variable is each movable node's scalar displacement along its original normal, within 3 voxels.
   The terms are:
   - data: 1 − recto (trilinear on the uint8 store);
   - verso barrier: verso at node + {1, 2} × n;
   - edge-length (|e| − |e0|)², over u, v and one diagonal;
   - bending: |L(P) − L(P0)|²;
   - anti-fold: relu(−J·sign J0);
   - inter-sheet gap: a hinge below max(thickness, `--min-spacing`) against the nearest OTHER sheet's
     node along the normal, re-found every 25 steps;
   - no-crossing: relu of the gap's flip against the published side.

   Duplicate traces (above) are excluded from the gap and no-crossing pairs. Tile and grid boundaries,
   anchors and the taper are fixed, and the fold guard has the last word.
9. **No-crossing guard** (`no_cross`, always). Each node's nearest non-duplicate neighbour node above
   and below along its normal is found on the published grids; any pair whose side has flipped after
   the snap (and the mesh solve) has both nodes' moves halved up to 6 times, then undone. Crossings in
   the final grids are 0 by construction (`tile_pairs.*.no_cross_pulled` counts the nodes it touched).
   Free tangential moves were tried first and random-walked nodes 0.4 voxel sideways.

**Anchors** (`--anchors anchors.jsonl`, render3d's append-only log). The log is replayed (`add` / `del`,
and a torn last line is ignored), and each record is matched to a surface by its `surface` name (the
tifxyz dir name, with or without `.tifxyz`, or the segment id). Each surface's anchors become a
displacement field in grid space, following usrm `ui.anchor_field`:
`field = (Σ wᵢ dᵢ / Σ wᵢ) · maxᵢ wᵢ`, where `wᵢ` is a Gaussian of σ = `--anchor-sigma` voxels. It
reproduces each drag exactly at the anchor's cell and tapers to zero away from the anchors. `|d|` is
clipped to `--anchor-clip`. The field is applied BEFORE the snap, and the snap is then scaled by
`1 − maxᵢ wᵢ`, so an anchored cell is a hard constraint and its neighbourhood blends in. The cell is
`grid_rc` of the published grid, or, when that is null, the grid point nearest `from_zyx`. Anchor
coordinates are taken as fine-frame voxels unless `--anchor-frame legacy` is given.

## Frames (the one thing that goes silently wrong)

- The stores and the umbilicus are in the fine volume `20260411134726` (2.4 µm, rung 2) frame.
- Published tifxyz come in two frames: `<id>-on-20230205180739-7.91um.tifxyz` (legacy) and, for the
  re-published segments, `<id>-on-20260411134726-2.4um.tifxyz` (fine). `--prefer` (default
  `2.4um,7.91um`) picks one variant per segment. The frame is read from the directory name, then from
  meta.json's `volume`, then from `--src-frame`. The `-45.532um` variants are skipped.
- legacy → fine is the fine volume's published `transform.json`. That is a 3×4 **xyz** affine mapping
  FINE → LEGACY, which includes a reflection and a −141° rotation, so it is not a scale + offset. Its
  landmark residual is about 6 fine voxels mean and 19 max. Legacy surfaces can therefore start further
  off than `--far`, which is the main reason to prefer the published 2.4 µm variants.
- The transform comes from, in order: `--transform transform.json` (direction auto-detected from the
  landmarks and `fixed_volume`; `--transform-direction` overrides it), `--transform` with 12 numbers,
  `--legacy-scale/--legacy-offset`, and finally the data agent's `rvsm.tools.refine.frame.Frame`,
  which fetches the transform from S3.
- The refined surface is written back in its ORIGINAL frame and at its ORIGINAL grid shape. Moved nodes
  become `src + (to_src(new) − to_src(old))`. Nodes that were not moved (outside the box or slab) stay
  bit-identical. Holes are written as -1, as vc3d writes them.

## Cross-section mode and outputs

`--z0 Z --dz 128` restricts the box to that fine z slab of the recto store and refines, jointly, every
surface under `--paths` that crosses it. Outputs are listed below.

### Memory: tiles and pieces

A 128 × 8192² slab of the 2.4 µm volume is crossed by about 51 published segments. Their full grids are
19 GB, and at pitch 4 their slab crops are about 1.1 G dense points: the sheets cross the slab at a slant,
so each crop's bounding rectangle is ~40× the band inside the box. Each store is 8.6 GB over the slab.
None of that is ever held. Instead:

- **Surfaces** are memory-mapped (`read_surface_box`) and scanned 256 rows at a time; fine-frame ones are
  ruled out by their z channel first. Only the published rows/cols that touch the box stay in memory,
  at published resolution (~0.5 GB for the whole slab). Loading all 81 candidates takes ~20 s and
  0.43 GB.
- **Tiles.** The slab is refined in yx tiles (`--tile`, default 1024 voxels) with a halo (`--halo`,
  default 160, at least 3·far + 8 and 44). The halo covers everything that couples nodes: the
  neighbour-sheet search (far + 3), the smoothing kernel (3 × `--sigma-vox`) and the anchor kernel
  (3 × `--anchor-sigma` 48). The stores are read densely only over one tile (core + halo + far + 4;
  ~240 MB per store at the defaults). The taper still fades at the SLAB's faces, never at a tile's.
- **Pieces.** Within a tile, each surface is cut into column runs of its grid that reach into the tile,
  each with its own row range (`column_pieces`), and upsampled only there. Two wraps of one segment
  are separate pieces, so they also keep each other apart as neighbour sheets. `ray_neighbours` only
  sees the other sheets' points inside a piece's bbox + reach.
- **Ownership.** A published node is taken from the tile whose core holds its published position.
  Results are kept as (row, col, value) at published resolution. After the last tile, each surface is
  re-read whole, one at a time, and written back.
- **Metrics and pictures** are gathered per tile, from core points only. Metrics are pooled per
  surface by point count, so continuity and ERL are per-tile numbers pooled, not whole-surface runs.
  The slab pictures show the whole box at stride ⌈extent/2000⌉. The crops keep, per tile, only each
  surface's cross-section segments at the crop levels plus a 5-voxel z band of its points with their
  |move| (a few MB for the whole slab). The sites are picked after the last tile, and the stores and
  CT are then read over just the 128² crop.

`--tile 48` against `--tile 0` (one tile) on the test fixture agrees to 0.05 voxel
(`test_tiled_slab_matches_one_tile`). A synthetic slab at the real tile density (45 sheets over
128 × 2048², ~465k points per tile, 4 tiles) took 2:19 at a 1.44 GB peak RSS. The real slab is 64
tiles: expect ~40 min and a ~2–2.5 GB peak. The extra is the published crops plus one full grid
(≤ 460 MB, ×3 transient) at write-back. `--tile 768` lowers the per-tile part.

Outputs under `--out`:

- `<name>/`: the refined tifxyz. `<name>.before/`: a copy of the published input.
- `png/slab_z<Z>.png`: `--slices` z slices of the slab, with before | after panels. They show CT in
  grayscale (the store's `volume`, read at a coarser rung when the picture is strided), recto tinted
  red, and verso tinted blue. Unrefined polylines (marching squares on the grid) are magenta, and
  refined ones are green over a dim magenta trace.
- `png/<name>_crop<k>.png`: `--crops` (default 6) before | after crops per surface, `--crop-px` 384 at
  `--crop-scale` 3 (a 128-voxel window). Each crop is at its own z level, spread over the slab inside
  the taper. The site alternates between the densest 64-voxel cell and the cell with the largest mean
  |move|, at least one window apart along the surface (the farthest site when the surface is too
  small). The same CT/recto/verso coloring is used; this surface is magenta → green, other sheets
  are dim. A label gives zyx and the local mean |move|. `--crop-montage` also writes
  `png/<name>_crops.png` with all of them stacked.
- `png/displacement_hist.png`: the signed normal displacement per surface.
- `refine_report.json` plus JSON lines on stdout: per-iteration stats (with_peak, capped by a
  neighbour, verso_blocked, thickness), and evalsurf metrics before and after per surface and pooled:
  recall@2/4, offset mean/std/≤3, merge_frac, continuity and ERL. Per surface (`moves`) it has:
  - the fraction of points that moved more than 2 voxels, and the mean and median |move|, with a
    WARNING when the median is < 0.5 voxel (over-smoothed) or > max far/2 (runaway);
  - the first-pass |move| percentiles, with `far_off` when the median is > 10 voxels: how far off the
    published line really was;
  - the mean tangential drift (the part of the total move not along the published normal);
  - fold events;
  - grid spacing mean and CV, before and after.

  `geometry` pools these numbers. Per tile, `tile_pairs` gives the adjacent-wrap statistics for the
  published, snapped (before the no-cross guard) and final positions: node pairs sampled every 3rd node
  to the nearest NON-duplicate sheet, pairs closer than max(thickness, min spacing), crossings, and
  `coincident` = the number of duplicate sheet pairs in the tile.
- `png/spacing_hist.png`: grid-edge length histograms, published (gray) and refined (green).

## Warnings

- **Circularity.** Metrics measured on the store you refined against only show that the optimiser did
  its job. Use `--eval-store` with a DIFFERENT store (another teacher, or the student when you refined
  on a teacher). Without it, the output is labelled `"circular": true` and a warning is printed.
- A surface whose true band is more than `--far` away (legacy transform error, wrong wrap) will not be
  found. The verso term makes it stay put rather than jump.
- The refiner assumes the published surface is the RECTO face. A surface traced on the verso face gets
  moved one thickness outward.

## Command (laptop)

Slab lo [55552, 13696, 13312], shape [128, 8192, 8192] (fine zyx). The z0/dz pick the slab; y/x come
from the store's own extent. The midline store is not an input to the refiner. The command is capped
at 6 GB so a mistake cannot take the WSL VM down:

```
VOLCOMP_LIB=/home/forrest/volume-compressor/build/release/libvolcomp.so \
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
/home/forrest/rvsm/.venv/bin/python -m rvsm.tools.refine.refine \
  --recto /home/forrest/refine/pred/recto.zarr --verso /home/forrest/refine/pred/verso.zarr \
  --thickness-store /home/forrest/refine/pred/thickness.zarr \
  --paths /home/forrest/refine/paths --transform /home/forrest/refine/transform.json \
  --umbilicus /home/forrest/refine/umbilicus/20260411134726-umbilicus-20260524235033.json \
  --z0 55552 --dz 128 --tile 1024 --halo 160 --no-ct \
  --pitch 5 --far 16 --iters 4 --slices 6 --crops 6 --crop-montage \
  --out /home/forrest/refine/refined_z55552 [--anchors anchors.jsonl] [--eval-store S]
```

These are the local-following settings: sigma defaults to 2 × pitch = 10 voxels, and to 5 on the last
pass. On the synthetic 45-sheet 128 × 2048² slab they took 2:27 at a 1.48 GB peak RSS. Expect ~45 min
and ~2–2.5 GB for the full slab.

Drop `--no-ct` for CT under the pictures. CT is read strided, one plane per picture, from the store's
`volume` attribute or `--ct`. `--ct-mask` reads full-resolution CT over each tile.
