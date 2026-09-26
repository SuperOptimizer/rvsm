# rvsm refine: snapping published surfaces onto the recto/verso predictions

`rvsm/tools/refine/refine.py` (`rvsm refine ...` or `python -m rvsm.tools.refine.refine ...`) ports the
usrm2 refiner (`usrm2/refine.py`) onto rvsm's stores. It moves every point of a published tifxyz surface
along its normal onto the nearest recto probability peak. It checks each move against the verso
prediction, keeps render3d's drag anchors fixed, and writes the surface back in the frame it was read in.

## Algorithm (per iteration, search radius `far / 1.5^it`, at least 3)

1. **Grid.** Each surface is read, mapped to the fine 2.4 µm frame, and cropped to the rows and columns
   that touch the box. It is then upsampled so the grid pitch is about `--pitch` voxels (`upsample`; node
   (r, c) becomes node (r·up, c·up), so the published nodes are kept exactly).
2. **Normals.** Normals are hole-tolerant: they are computed on the hole-filled grid (`filled`) and then
   masked back. They are oriented **outward from the umbilicus**, which is also VERSO → RECTO
   (`export.SIGN_CONVENTION`).
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
   neither merge onto one band nor cross, and a move is also capped 1 voxel short of a neighbour.
6. **Smoothing.** The per-point offsets are smoothed over the (H, W) grid with a Gaussian weighted by
   peak height (`smooth`, σ = `--sigma-vox` voxels along the sheet). The moves are faded to 0 over
   `--taper` voxels at the box faces, so a slab-refined surface joins its untouched remainder without a
   step. They are also faded around anchors (below). All sheets move only after all of them have been
   measured.

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
  bit-identical. Holes are written as -1, as vc3d writes them. `--write-dense` also writes the dense
  refined crop as `<name>.dense` (meta scale × up, `crop_rc` recorded).

## Cross-section mode and outputs

`--z0 Z --dz 128` restricts the box to that fine z slab of the recto store and refines, jointly, every
surface under `--paths` that crosses it. The stores are read only in the 128³ blocks within reach of a
surface point (`read_box(near=...)`; the rest of the `np.zeros` array is never paged in), so a
128 × 8192² slab fits on the laptop. Outputs under `--out`:

- `<name>/`: the refined tifxyz. `<name>.before/`: a copy of the published input.
- `png/slab_z<Z>.png`: `--slices` z slices of the slab, with before | after panels. They show CT in
  grayscale (the store's `volume`, read at a coarser rung when the picture is strided), recto tinted
  red, and verso tinted blue. Unrefined polylines (marching squares on the grid) are magenta, and
  refined ones are green over a dim magenta trace.
- `png/<name>.png`: a 320-voxel zoom per surface (usrm2's `compare_png`).
- `png/displacement_hist.png`: the signed normal displacement per surface.
- `refine_report.json` plus JSON lines on stdout: per-iteration stats (with_peak, capped by a
  neighbour, verso_blocked, thickness), and evalsurf metrics before and after per surface and pooled:
  recall@2/4, offset mean/std/≤3, merge_frac, continuity and ERL.

## Warnings

- **Circularity.** Metrics measured on the store you refined against only show that the optimiser did
  its job. Use `--eval-store` with a DIFFERENT store (another teacher, or the student when you refined
  on a teacher). Without it, the output is labelled `"circular": true` and a warning is printed.
- A surface whose true band is more than `--far` away (legacy transform error, wrong wrap) will not be
  found. The verso term makes it stay put rather than jump.
- The refiner assumes the published surface is the RECTO face. A surface traced on the verso face gets
  moved one thickness outward.

## Command (laptop)

```
VOLCOMP_LIB=/home/forrest/volume-compressor/build/release/libvolcomp.so \
/home/forrest/rvsm/.venv/bin/python -m rvsm.tools.refine.refine \
  --recto /home/forrest/refine/pred/recto.zarr --verso /home/forrest/refine/pred/verso.zarr \
  --thickness-store /home/forrest/refine/pred/thickness.zarr \
  --paths /home/forrest/refine/paths --transform /home/forrest/refine/transform.json \
  --z0 <slab z0> --dz 128 --out /home/forrest/refine/refined [--anchors anchors.jsonl] [--eval-store S]
```
