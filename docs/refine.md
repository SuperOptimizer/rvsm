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

### Solvers (`--solver cut | label | snap`)

Steps 3-7 above are `--solver snap`, the per-node peak choice plus smoothing. It lets grid neighbours
5 voxels apart take wraps 20-40 voxels apart, and the smoothing turns that into a zigzag. The user's
verdict on that strip was: jumps forward and backward, sheet switches, local length doubled.

The two other solvers are near-isometric. Every node still moves only along its original normal, and
between 4-neighbours `|d_i − d_j| <= spacing × slope`. The slope is min(`--max-slope` 0.5,
sqrt((1 + `--max-strain` 0.10)² − 1)) = 0.458, so a slope adds at most 10% strain. After every guard,
`slope_project` restores the caps exactly. It keeps the fixed nodes and takes the mean of the largest
cap-Lipschitz minorant and the smallest majorant (min-plus on the grid). Hole edges, grid borders and
seams (`stable_nodes`) get no data term and only follow their neighbours. The box taper zone does not
move. `--far-total` 40 caps the cumulative move. A node whose placed feature is not real at the end
goes back toward its published position, as far as the caps allow.

- **`cut`** (default) is the exact optimal-surface solve (Li, Wu, Chen & Sonka 2006) by s-t min cut
  (PyMaxflow), one grid at a time (`two_surface_cut`). Per node, the recto and verso profiles along its
  normal are sampled from the stores in world space (no flattened volume). The solve finds a recto depth
  r and a verso depth w for every node that minimise Σ −log p_recto(r) + Σ −log p_verso(w), under three
  hard constraints:
  - the slope caps on r and on w;
  - t_min <= r − w <= t_max (`--t-min` 6, `--t-max` 28; the verso lies inward, n points verso → recto);
  - the neighbour-wrap bounds (the whole sheet stays between them).

  The verso's labels are shifted inward by the thickness estimate, so the window covers both faces.
  Passes are set by `--cut-depths` 24,12,6 and `--cut-steps` 2,1,1. The max-flow time grows steeply with
  the label count: 12 s for a 200 × 440 grid at 25 labels, minutes at 49.

  `--snap` picks the placement:
  - `recto`, `verso` and `mid` place the node on r, w or (r + w)/2 of the coupled solve;
  - `contrast`, `edge` and `ct` solve ONE surface with cost −score, where the score is recto − verso,
    the steepest rise of the σ=1 smoothed recto − verso, or the CT.

  Per node, the free choice of every mode on the final profile is written to
  `mode_positions/<surface>.npz`, so the modes can be compared without rerunning.
- **`label`**: the same caps and unaries on a ladder of integer offsets, solved by semi-global matching
  plus line-wise ICM (`label_solve`), coarse to fine (`--label-pitch` 20,20,10,5).

On tile 1 of the 2.4 µm strip, the snap solver scores recall@2 0.65 and offset_le3 0.88. Label scores
0.45 and 0.55, and cut (recto) scores 0.46 and 0.47. But folds drop from 16590 to ~100, moves over 30
voxels from 17787 to ~30, and strain p90 from 0.48 to 0.10. Projecting the snap result onto the same
caps drops its on-band fraction from 0.68 to 0.47. So the recall gap IS the switches: the recall
metrics are measured against the nearest recto ridge, and they reward jumping to whichever wrap is
nearest.

Per pass, the report gives:
- `at_ridge`;
- `both_faces` (a valid recto AND verso at r, w);
- `thickness_fit` percentiles;
- switches;
- folds;
- strain.

Per surface and pooled, it adds `sheet_switches`, `zigzag` / `zigzag_1vox`, `strain`, `moved_gt30`,
`no_ridge_end` and `ridge_reverted`.

**Output pitch** (`--write-pitch fine`, the default). The refined tifxyz is the crop of the surface that
touches the box, at the refinement pitch: every upsampled node, and meta scale × up. `refined.crop_rc`
and `write_up` record where it sits in the published grid. The `.before` is the same crop, upsampled,
so `render.py --compare` puts both on one layout. `published` writes the full surface at the published
pitch with the moved nodes replaced, as before. A whole 3748 × 9740 surface at 4× would be 7 GB.

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

## Folds, the verso side, a missing verso, the no-ridge revert (2026-09-29)

Measured on tile 1 (`--box 55552 13696 13312 128 1024 1024`) and the 6-tile strip
(`--box 55552 13696 13312 128 1024 6144 --tile 1024 --halo 160`) of the student slab.

- **Folds were orientation flips, not seams.** The strip's fold events grew tile by tile (132, 172, 816, 2241,
  6727, 12647); tiles 5-6 hold sheets lying almost flat in z (|n_z| ~ 0.93). There dot(n, radial) is near 0 and
  evalsurf's per-node orientation flips single nodes; neighbours then move in opposite directions. 91% of the
  strip's final folded nodes (2955 of 3250) were within 3 nodes of such a flip; only 0.6% within 8 voxels of a
  tile seam (1.3% of the nodes are). `normals` now takes the grid's own cross-product normal and flips it as a
  whole, per connected piece, by the weighted outward vote.
  Folds are now counted per CORE node (the halo of a piece is another tile's core and was counted twice), and
  `folds_final` counts the nodes still folded in the output. `geometry.fold_locations` bins the events by the
  distance to the nearest interior tile seam, to a hole or grid border, to the box's faces, and counts the
  ones on sheets nearly perpendicular to the radial direction. `png/folds_<surface>.png` maps them (gray nodes,
  red events, yellow final folds, blue seams).
- **Crossings.** `pair_stats` found its pairs with the REFINED grid's pitch as lateral tolerance while
  `no_cross` used the published one, so 4 strip pairs were counted that the guard never checked. Both use the
  published pitch now; final crossings are 0 by construction.
- **The verso lies OUTWARD here.** `export.SIGN_CONVENTION` says verso -> recto points outward. On this slab the
  recto/verso cross-correlation along the outward normal peaks at +7 to +13 voxels in every region tried, for the
  student AND for the recto teacher (`teacher_regions/{recto,verso}`); the student recto matches the teacher recto
  at lag 0 (corr 0.66). So the old coupling searched the verso on the wrong side: 80% of recto peaks have a verso
  >= 0.5 within 6-28 voxels outward, 21% inward. `--verso-side auto` (default) measures the lag on the first tile
  and uses that side for every tile (`inward` / `outward` force it). The solvers then work with n pointing verso ->
  recto. Thickness estimate: 22 -> 9 voxels; both faces found on tile 1: 13% -> 33%.
- **A missing verso is free, not a cost.** A node whose verso window holds nothing >= `--verso-thr` gets a flat
  verso cost, so faint verso noise cannot pull a recto-only column (the -log of 0.28 against 0.001 used to
  outweigh 4 voxels of a wide recto band). `verso_in_window` reports the fraction of columns with a verso face.
- **The no-ridge revert.** Per candidate (moved > 0.5 voxel) the last pass records why its placed recto is not on
  a ridge (`no_ridge_why`, see `_cut_info`): nonmovable, no_peak (nothing >= thr/2 within 8 voxels), weak_peak
  (thr/2..thr), near_miss (>= thr within `--ridge-reach`, just not at the rounded label), peak_past_neighbour,
  peak_past_window, held_by_coupling (the recto-only solve does reach it), held_by_slope. A node now keeps its
  move when a recto >= `--revert-thr` (default thr/2) is within `--ridge-reach`, or when at least
  `--revert-support` (0.5) of its neighbourhood (Gaussian, sigma 2 nodes) is on a ridge: there the slope-consistent
  interpolation of the neighbours stands instead of a dimple toward the published position.
- **Fair evaluation.** `--eval-store` metrics now count only points inside the eval store's own box (16 voxels
  in), and `python -m rvsm.tools.refine.evalrun --store S --box ... --umbilicus U RUN...` re-scores finished runs
  the same way. The only independent store over this slab is the recto teacher's 1024^3 regions
  (`forlindesk2:/vesuvius/usrm2/teacher_regions/recto/region_55296_13312_13312.zarr`, copied to
  `/home/forrest/refine/pred_eval/`); it covers y < 14336 of tile 1 (62%). The student was distilled partly from
  this teacher, so it is independent of the refinement target, not of the student's lineage.

- **The strip's +-40 voxel moves (d6dc3bf -> this).** After the verso-side fix the strip had 6852 moves over 30
  voxels (2525 before) and 4440 slope switches (1531). The moved nodes STARTED in air (87% of their published
  positions below the CT air level) and 96% ended on a recto >= 0.25; the old inward-verso coupling had held them.
  The min cut also has ties wherever a column's costs are flat, and its tie-break slid such columns to the edge
  of the window every pass, up to --far-total. Now:
  - a small cost per voxel of total move (0.01) breaks those ties toward staying;
  - `--far-evidence` (16): a node keeps more than 16 voxels of total move (up to `--far-total`) only when its
    FINAL place has a recto >= thr within 2 voxels, is not air (no CT above `--ct-air` within 3 voxels along the
    normal: the recto face itself is the papyrus/air edge), and at least 60% of the nodes within 3 grid steps
    moved over 8 voxels the same way; else it is clipped to 16 (`allowance_used`, `allowance_denied` and why).
    An earlier version demanded a ridge under EVERY pass; the wide early passes cross air on the way to the
    real sheet, so it forbade exactly the corrections the wide search is for. Per node,
    `ridge_passes/<surface>.npz` logs the recto at the placed recto per pass (published zyx, per-pass value,
    and whether every moving pass landed on a ridge);
  - `--revert-support` never rescues a node without evidence of its own: no recto >= `--revert-thr` within
    `--ridge-reach` of it AND (a recto profile flat below 0.15 over the window, OR its destination in air, CT
    below `--ct-air`, default the 2-means air/papyrus midpoint measured on the first tile, 79.7 here). Such a
    node goes back toward its published place as far as its neighbours' caps allow (`--hard-revert caps`,
    default; `full` makes the neighbours give way instead, which cost tile 1 0.035 of teacher recall@2)
    (`hard_reverted`, `hard_air`, `hard_flat`);
  - the stores end at the box: a profile label outside it is NO DATA (the node's median in-box cost), and a node
    whose +-24 voxel window leaves the box is neither "flat" nor "air". A sheet lying flat in z in a 128-deep slab
    otherwise saw the empty outside as "no ridge" and was pushed away from the face (the kinks on 20260623141135
    near yx 13938,15852: normal sign consistent, |n_z| 0.92, 66% of the windows there left the slab);
  - the later steps (the evidence clip, the revert, the guard) never re-ran the fold guard: the final repair
    loop now also halves the move of any folded node (toward its published place) before the caps and the
    crossings are re-checked (`fold_repairs`);
  - the no-crossing guard holds the nodes it pulled while the caps are restored around them (up to 8 rounds);
    the held nodes can still contradict each other (642 strip switches left that way), so `shrink_to_caps`
    then closes every violated edge by moving nodes only TOWARD their published place, alternating with the
    guard until both hold. The solve itself leaves 0 switches; `switch_stages` counts them after the solve,
    after the first guard round, after the guard rounds and at the end.
  - the flip test ignores quads the PUBLISHED grid already collapsed (oriented area under 5% of the grid's
    median quad) and the bunching test ignores noise-sized shortenings (< 5%): 20260603222816 has a crumpled
    published patch (x 17019-17080, y 14574-14610, quad areas ~0.1 vs 25, 88% inverted) whose "folds" were
    noise; a node still folded after 6 halvings is put back on its published point, tangential part included.
  Tile 1 (published / fix 1 / this): recall@2 against the recto teacher 0.407 / 0.648 / 0.639, moves > 30
  0 / 108 / 0 (none of fix 1's ended on a ridge >= thr outside air), switches 22 -> 0, final folds 44 -> 0,
  crossings 0.

PyMaxflow is the `refine` extra (`pip install 'rvsm[refine]'`); `--solver cut` fails at start-up with that hint
when it is missing.

## Speed and the production run (`rvsm refine-slab`)

cProfile of tile 1 (fix3, 854 s wall): the no-crossing pairs 420 s (the neighbour search over every node, redone by
`no_cross` in every guard / repair round and by `pair_stats` three times), max flow 252 s, the rest < 30 s each.
Changes, every one leaving the output identical (tile 1: max |difference| 0.0 over 79 M written coordinates,
identical pooled metrics and geometry):

- `pair_index`: a tile's no-crossing pairs are found ONCE on the published grids and reused by every `no_cross`
  round and `pair_stats` (the sampled pairs are a subset: the same as searching from the sampled nodes alone);
- the kd-tree queries run on every core (`workers=-1`), and the per-sheet searches (`sheet_pairs`, each pass's
  neighbour sheets) run in threads;
- `cut_cropped`: each max-flow graph covers only the bounding rectangle of the nodes being solved plus one ring
  (a piece's grid is the rectangle of a slanted band: mostly pinned nodes outside the tile);
- the per-grid solves of a pass run in `--jobs` (4) forked workers (PyMaxflow holds the GIL);
- the coupled solve no longer samples the unused single-surface profiles;
- production flags: `--no-pictures`, `--no-before`, `--no-ridge-log`, `--no-revert-diag` (the extra recto-only
  solve that splits held_by_coupling from held_by_slope), and per-stage timings in every tile's `tile_time` line
  and in the report's `timings`.

Not done, because they change results: fewer far-schedule passes when a pass moves little, skipping small
pieces, fewer kd-tree neighbours.

`rvsm refine-slab --z0 Z --dz 128 --pred-dir DIR --paths DIR --out DIR [--umbilicus U --transform T]` runs every
tile of the prediction slab with the fix3 settings (tile 1024, halo 160, pitch 5, sigma 8) and the production
flags, and writes one refined tifxyz per surface (the part inside the slab) plus `refine_report.json`. After each
tile it saves what the tile added (`<out>/tiles/tile_NNN.pkl`, then `tile_NNN.json` as the done marker); a rerun
replays finished tiles and computes the rest (`test_refine_slab_resumes_a_tile_run_to_the_same_result`). The
verso side and the CT air level are measured on the first computed tile and carried in the checkpoints.

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
