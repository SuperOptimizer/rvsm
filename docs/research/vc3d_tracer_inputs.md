# VC3D Surface Tracer: Volumetric Inputs and Model Integration

**Research context:** Unified multi-resolution 3D surface model (usrm2) could emit additional per-voxel channels to improve the Herculaneum spiral tracer's reconstruction of papyrus geometry. This document maps the tracer's current volumetric inputs and identifies what new outputs would help.

## 1. Current volumetric inputs consumed by fit_spiral.py

The tracer reads five volumetric fields (all stored as Zarr, with sparse CUDA LRU cache backed directly by source arrays). See `scripts/spiral/lasagna_data.py` for the data-loading layer.

### 1.1 Normal grid (nx, ny components, uint8)
- **Where read:** `lasagna_data.py:44–48` (opens `normal_nx_zarr_path` and `normal_ny_zarr_path` as separate Zarr roots)
- **Shape:** 3D array, typically 2x coarser than working resolution (default `lasagna_scale=16`)
- **Format:** Two uint8 arrays (x and y components of unit surface normals):
  - Encoding: `component_u8 = (component_normalized * 127.0) + 128.0`
  - Decoding (`sdt_losses.py:479–481`): `component_norm = (u8 - 128.0) / 127.0`
  - Full normal: `nz = sqrt(clamp(1 - nx² - ny², 0))` then normalize to unit length
  - Returned as `[nz, ny, nx]` to match zyx coordinate order
  - Only positive-nz hemisphere stored (recto side by convention)
- **Cache budget:** 6 GiB default; overridable via `FIT_SPIRAL_SPARSE_NORMAL_CACHE_GB`
- **Usage:** Dense normal loss (`fit_spiral.py:286`) in geodesic alignment of fitted surface to observed normals

### 1.2 Gradient magnitude (uint8)
- **Where read:** `lasagna_data.py:56–62` (opens single Zarr root at `grad_mag_zarr_path`, group `normal_zarr_group`)
- **Shape:** Same 3D shape as normals
- **Format:** uint8 unsigned intensity gradient magnitude
- **Cache budget:** 2 GiB default; overridable via `FIT_SPIRAL_SPARSE_GRAD_CACHE_GB`
- **Value range:** 0–255; scaled by `grad_mag_encode_scale=1000.0` at runtime
- **Usage:** Legacy density integral (sheet spacing loss, `fit_spiral.py:289–290`, when `dense_spacing_mode='grad_mag'`)

### 1.3 Surface signed-distance transform (surf-SDT, uint8)
- **Where read:** `lasagna_data.py:152–277` (prepares from single Zarr with OME multiscales metadata)
- **Shape:** 3D array (full working resolution); grid scale and geometry from store's own attrs
- **Format:** uint8 encoding capped Euclidean distance (see `make_surf_sdt.py:1–54`):
  - `value = 1..255`: `sd = (value - offset) * unit_working_voxels`, offset=128 by default
  - Positive outside binarized surface, negative inside, clipped to `±cap_working_voxels`
  - Zero = no-data (reserved)
- **Cache budget:** 16 GiB default; overridable via `FIT_SPIRAL_SPARSE_SDT_CACHE_GB`
- **Decoded range:** ±~32 voxels typical (cap ~32 wv, unit ~0.25 wv/level)
- **Usage:** Distance-transform loss pulls fitted spiral sheets onto the observed surface (`sdt_losses.py:436–442`, active in both `grad_mag` and `phase` modes). Also enables phase registration by detecting SDT bands with soft-sequence alignment (`sdt_losses.py:8–18`, when `dense_spacing_mode='phase'`)

### 1.4 Verified/unverified surface patches (binary masks)
- **Where read:** Dataset `verified_patches/` and `unverified_patches/` directories (VC3D-prepared geometry, not directly from usrm2)
- **Format:** Quad-mesh geometry with valid-region masks in patch ij coordinates
- **Usage:** Patch radius and distance-transform losses anchor the spiral to interactive corrections

### 1.5 Track collection (polyline points + winding indices)
- **Where read:** Dataset `*.json` PCL files with extracted horizontal/vertical tracks
- **Format:** Lists of 3D points in scroll coordinates, each with spiral-space winding identity
- **Usage:** Track alignment loss constrains spiral-coordinate samples to lie near observed ink

## 2. What "spiral" means in this context

The tracer fits a canonical **Archimedean spiral** in cylindrical coordinates around the scroll's central axis (umbilicus). Three interlocking coordinate systems:

- **Scroll space (zyx):** The 3D volumetric grid (z down, y across width, x across depth)
- **Spiral space (z, θ, r_spiral):** z stays absolute; θ is angular phase (radians) around the axis; r_spiral encodes the winding number and fractional radius
  - `r_spiral = (winding_index + θ/(2π)) * dr_per_winding`
  - `dr_per_winding` is the radial pitch (distance between adjacent wraps, typically 10–16 voxels)
- **Cylindrical working space:** Used internally for loss gradients and transforms

**Winding**: An integer indexing concentric spiral rings (1 = innermost, incrementing outward). The same physical sheet appears at multiple winding indices as you walk radially outward; the tracer must disambiguate which winding each observation belongs to.

## 3. What the tracer's optimization actually needs

The tracer minimizes six loss components; their success depends on input quality:

1. **Patch radius loss** (`losses.py:456–482`): Fitted winding surface radii must match patch-center targets. Fails if:
   - Surface prediction (via SDT) is thick or ambiguous → hard to localize the true surface
   - Prediction has orientation artifacts (e.g., rotated merges) → radius target becomes invalid

2. **Distance-transform loss** (`losses.py:514–557`): Fitted sheet must lie near (inside or outside?) the observed surface. Fails if:
   - SDT has large gaps (interior vs. exterior ambiguity) → DT loss pulls toward air or noise
   - Thick sheet predictions → unclear which side is "correct"

3. **Dense normals loss** (`fit_spiral.py:286`): Surface normal field should point perpendicular to local sheet geometry. Fails if:
   - Normals are inverted (verso vs. recto confusion) or flipped by symmetry artifacts
   - Normals are noisy or discontinuous across sheet boundaries

4. **Dense spacing loss—phase mode** (`sdt_losses.py`): Observed SDT band positions should align to modeled winding sequence. Fails if:
   - Phase signal is ambiguous (e.g., recto and verso sheets close together, indistinguishable)
   - Merges or thick bands hide the true phase

5. **Gap/sheet-count loss** (`sdt_losses.py:401–442`): The number of observed sheets in a ray should match the model's prediction. Fails if:
   - Recto and verso sheets merge or are only partially separated
   - Thick bands count as multiple sheets when they are one, or vice versa

6. **Track alignment loss** (`losses.py:246–262`): Ink points should lie on the fitted surface.

**Critical insight:** The tracer excels at precise radial and axial position but **struggles when recto/verso sheets are indistinguishable or close** (merge), when prediction **bands are too thick** (ambiguous normal), or when **phase wraps are lost** (orientation flip).

## 4. Additional per-voxel channels a model could OUTPUT to improve tracing

### 4.1 Surface normal field (3 channels: nz, ny, nx, float32 or uint8)
- **Why help:** Current normals come from noisy image gradients. Model-predicted normals are spatially consistent and aware of local geometry.
- **Format:** Same encoding as current normals (or higher precision: float32 ∈ [–1, 1]); store positive-nz hemisphere
- **What tracer would do:** Use in dense normal loss instead of (or in addition to) image-gradient normals, reducing noise and improving phase detection

### 4.2 Recto/verso sheet-pair field (2 channels: recto_prob, verso_prob, uint8)
- **Why help:** Explicitly label which voxels belong to recto vs. verso sheet. Disambiguates merges and enables per-sheet loss weighting.
- **Format:** uint8 [0, 255] probabilities for each sheet (may sum >255 in boundary regions)
- **What tracer would do:** Separate the SDT loss into per-sheet terms; route phase matching to the correct sheet; skip samples in truly merged regions where no single sheet is clear

### 4.3 Winding/phase field (1 channel: phase_angle, uint8 or float32)
- **Why help:** Directly encode the angular position (θ) of the surface. Converts a hard discrete search over winding indices into a continuous regression.
- **Format:** uint8 cyclic [0, 255] representing θ ∈ [0, 2π), or float32 phase ∈ [–π, π]
- **What tracer would do:** Refine winding assignment from discrete snapping to sub-winding phase; improve DT loss convergence
- **Caveat:** Only valid on or near the true surface; must handle multi-sheet regions carefully

### 4.4 Signed-distance field (1 channel: sd, float32 or uint8)
- **Why help:** Direct supervision of the distance-transform loss, replacing the current post-hoc EDT from binarized predictions. Keeps finer structure (edge-aware, continuous).
- **Format:** float32 or uint8-encoded (as current SDT); positive outside, negative inside, cap at ±32 voxel
- **What tracer would do:** Replace `make_surf_sdt.py` pipeline; loss directly penalizes deviation from predicted distance, faster convergence

### 4.5 Normalized radius from umbilicus (1 channel: normalized_r, uint8)
- **Why help:** Quantify how far from the axis a voxel sits (core → inner wraps → outer wraps). Tracer can learn scale and winding pitch from this alone.
- **Format:** uint8 [0, 255] linearly mapping [0, scroll_radius] in working voxels
- **What tracer would do:** Anchor winding calibration; reduce sensitivity to spiral pitch mismatches; guide rough winding assignment before refinement

### 4.6 Density/confidence field (1 channel: confidence, uint8)
- **Why help:** Mark which voxels have high-quality observations (e.g., strong signal, single sheet) vs. low (noise, merge). Reweight losses per sample.
- **Format:** uint8 [0, 255] confidence score (or binary mask)
- **What tracer would do:** Down-weight loss gradients in low-confidence regions; avoid pulling the fit into artifact-prone zones

## 5. Axis order, units, and pitfalls

### 5.1 Axis order: ZYX (not XYZ)
- All volumetric inputs and outputs use **z-down, y-right, x-back** order.
- Normal components and spherical coords: `[nz, ny, nx]` (not `[nx, ny, nz]`)
- Zarr shapes: `(z, y, x)` throughout
- Coordinate transforms in `transforms.py` are zyx-first; a mistake here breaks all spatial losses

### 5.2 Units
- All distances in **working voxels** (the CT volume's native resolution, e.g., 1–2 um per voxel)
- Spiral pitch `dr_per_winding` is in working voxels; default ~10–16 voxels ≈ 10–30 um
- SDT cap is in working voxels (~32 typical); unit is the encoding step size (~0.25 working voxels)
- Normal grid and grad magnitude are typically 2x or 4x coarser; the loader scales coordinates accordingly

### 5.3 Chunking expectations
- SDT and normal caches use 32³-voxel chunks backed by Zarr
- A single gather's distinct chunks must fit the cache pool (6 GiB for normals)
- Large rays (dense spacing samples) can exceed the budget; the fitter reports required working-set size and env var name

### 5.4 Validation and masking
- **No-data**: Values of 0 reserved (e.g., SDT=0 = no-data, not a small distance)
- **Air masking**: Voxels where paired CT is 0 (unscanned fill) are automatically ignored
- **In-bounds checks**: Sparse cache returns zero/false for out-of-bounds access; loss terms skip invalid samples

### 5.5 Threshold sensitivity
- Normal magnitude: pairs with `(nx_u8 != 0 | ny_u8 != 0)` check for validity (threshold is implicit)
- Grad magnitude encoding scale 1000.0 is empirical; rescale if using a different magnitude source
- Phase-mode SDT band detection uses a soft sigmoid-gated inside/outside indicator; sensitivity tunable in `dense_spacing_mode='phase'` config

## 6. Where the code lives

| File | Purpose |
|------|---------|
| `scripts/spiral/fit_spiral.py:159–310` | Default config; input enable/disable switches |
| `scripts/spiral/lasagna_data.py` | Loading normals, SDT, grad_mag into sparse CUDA cache |
| `scripts/spiral/sdt_losses.py:113–195` | SDT decoding, trilinear sampling, validity masking |
| `scripts/spiral/sdt_losses.py:445–483` | Normal decoding (nx/ny from uint8 to normalized, compute nz) |
| `scripts/spiral/losses.py:230–557` | All six loss functions (radius, DT, normals, spacing, phase) |
| `scripts/spiral/soft_alignment.py` | Phase-mode band alignment (differentiable HMM) |
| `scripts/spiral/make_surf_sdt.py:1–94` | SDT generation from binary prediction; encoding scheme |

---

**Summary for model development:** The tracer's primary pain point is **ambiguity at recto/verso boundaries and thick merged regions**. Direct model outputs of sheet-pair identity (recto_prob, verso_prob), local normal vectors, and phase angle would allow the tracer to disambiguate merges and converge on thin sheet solutions. A high-confidence field would suppress losses in noise-prone zones. All outputs should use uint8 encoding with bounds and no-data markers to match current Zarr infrastructure.
