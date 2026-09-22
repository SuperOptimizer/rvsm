"""The upstream `metadata.json` that sits next to every open-data volume, flattened.

`<bucket>/<scroll>/volumes/<vol>.zarr/metadata.json` carries the acquisition and nabu-reconstruction
parameters of the scan (an example, with the motor/attenuator dumps stripped, is kept at
`docs/example_metadata.json`).  `load()` reads one -- from a local path or from an
https URL -- and returns a FLAT dict of the conditioning-relevant fields, filling documented defaults
for anything missing so a caller never has to branch (`meta["missing"]` / `meta["defaulted"]` say what
was made up).  `ranges_for()` turns that dict into augmentation-config overrides centred on the scan,
which `aug.get(name, meta=...)` merges into a preset (the augmentation preset is centred on the scan this way).

Units: the upstream file is in MILLIMETRES (`samplePixelSize` 0.0024 = 2.4 um, `sampleDetectorDistance`
220.0 = 220 mm) and `unsharp_sigma` is in detector PIXELS.  Everything this module returns is in
microns / keV / mm as named by the key suffix, so callers never have to remember the upstream units.
The two scans of the corpus that have been inspected:

    scan                       pixel   energy  distance  delta/beta  unsharp (coeff, sigma)
    2.4 um PHerc-Paris4 B_HA   2.4 um   78 keV   220 mm      1000      4.0, 1.2 px = 2.88 um
    1.1 um PHerc-Paris4 mosaic 1.1 um   (n/a)    (n/a)        500      4.0, 2.5 px = 2.75 um

i.e. the unsharp sigma is very nearly CONSTANT IN MICRONS across the fleet (2.75-2.88 um) while it is
2x apart in pixels -- which is the whole argument for defining augmentation sigmas in microns and
converting with the sample's rung (see the augmentation presets).
"""
import json
import math
import os
import urllib.request

import numpy as np

# Defaults: the 2.4 um 78 keV PHerc-Paris4 B_HA scan (docs/example_metadata.json),
# i.e. the scan the whole pipeline is calibrated on.  A field missing from a real metadata.json falls
# back to these and is listed in meta["defaulted"].
DEFAULTS = {
    "energy_kev": 78.0,          # scan.tomo.acquisition.energy
    "pixel_um": 2.4,             # detector.samplePixelSize (mm) * 1000
    "detector_pixel_um": 4.6,    # detector.sensorPixelSize (mm) * 1000
    "distance_mm": 220.0,        # sampleDetectorDistance (propagation distance)
    "source_distance_mm": 178000.0,
    "expo_time": 0.017,
    "tomo_n": 49000,
    "half_acquisition": False,
    "helical": True,             # acquisition.scanType == "helical"
    "scintillator": "GAGG_50um_refl",
    "phase_method": "Paganin",
    "delta_beta": 1000.0,        # processing.preprocessing.phase.delta_beta
    "unsharp_coeff": 4.0,        # ... .unsharp_coeff
    "unsharp_sigma_px": 1.2,     # ... .unsharp_sigma (detector pixels)
    "hist_min": -0.348,          # processing.32bitsData.histogram.*
    "hist_max": 4.913,
    "hist_p002": -0.017,         # min_0p002_percentile
    "hist_p998": 0.1307,         # max_0p998_percentile
    "used_min": -0.0559,         # postprocessing.32BitsConversion.dataset_used_min/max
    "used_max": 0.3271,
    "win_f32_lo": -0.04,         # zarr_export.target_window_f32_min/max (f32 -> uint8 window)
    "win_f32_hi": 0.22,
    "win_u16_lo": 2725.0,
    "win_u16_hi": 47214.0,
    "mosaic": False,             # the 1.1 um Paris 4 volume is a 19-tile fused mosaic
    "mosaic_tiles": 1,
}

# Corpus spread of the Paganin parameters, from the two inspected scans above.  `WIDEN` stretches the
# span multiplicatively on both sides so the sampled range is a superset of what the fleet shows
# (2x each way: the fleet is two scans, not a population, and the 42-scroll corpus is not measured yet).
DB_SPAN = (500.0, 1000.0)
COEFF_SPAN = (4.0, 4.0)
SIGMA_UM_SPAN = (2.75, 2.88)
WIDEN = 2.0


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _num(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def path_of(p):
    """'<vol>.zarr', '<vol>.zarr/' or a directory -> its metadata.json; a .json path is returned as is.

    Works for local paths and for https bucket URLs alike (the open-data layout keeps metadata.json
    directly inside the .zarr group)."""
    p = str(p)
    if p.endswith(".json"):
        return p
    return p.rstrip("/") + "/metadata.json"


def read_json(p, timeout=10.0):
    """The raw metadata.json as a dict, or None when it cannot be read (missing / unreadable / not json)."""
    p = path_of(p)
    try:
        if p.startswith(("http://", "https://")):
            with urllib.request.urlopen(p, timeout=timeout) as f:  # noqa: S310 (a bucket URL by construction)
                return json.loads(f.read().decode("utf-8"))
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return json.load(f)
    except Exception:
        return None


def flatten(raw):
    """The conditioning-relevant fields of a raw metadata.json, flat, in microns / keV / mm.

    Every key of DEFAULTS is present in the result.  `defaulted` lists the keys that the file did not
    supply, so a caller can tell a real 78 keV scan from a defaulted one."""
    raw = raw or {}
    acq = _get(raw, "scan", "tomo", "acquisition", default={}) or {}
    det = _get(acq, "detector", default={}) or {}
    ph = _get(raw, "scan", "tomo", "processing", "preprocessing", "phase", default={}) or {}
    hist = _get(raw, "scan", "tomo", "processing", "32bitsData", "histogram", default={}) or {}
    conv = _get(raw, "scan", "tomo", "processing", "postprocessing", "32BitsConversion", default={}) or {}
    exp = _get(raw, "zarr_export", default={}) or {}
    mos = _get(raw, "mosaic", default=None)

    src = {
        "energy_kev": acq.get("energy"),
        "pixel_um": None if det.get("samplePixelSize") is None else _num(det.get("samplePixelSize"), 0.0) * 1e3,
        "detector_pixel_um": None if det.get("sensorPixelSize") is None else _num(det.get("sensorPixelSize"), 0.0) * 1e3,
        "distance_mm": acq.get("sampleDetectorDistance"),
        "source_distance_mm": acq.get("sourceSampleDistance"),
        "expo_time": acq.get("expo_time"),
        "tomo_n": acq.get("tomo_N"),
        "half_acquisition": acq.get("half_acquisition"),
        "helical": None if acq.get("scanType") is None else (str(acq.get("scanType")).lower() == "helical"),
        "scintillator": det.get("scintillator"),
        "phase_method": ph.get("method"),
        "delta_beta": ph.get("delta_beta"),
        "unsharp_coeff": ph.get("unsharp_coeff"),
        "unsharp_sigma_px": ph.get("unsharp_sigma"),
        "hist_min": hist.get("min"),
        "hist_max": hist.get("max"),
        "hist_p002": hist.get("min_0p002_percentile"),
        "hist_p998": hist.get("max_0p998_percentile"),
        "used_min": conv.get("dataset_used_min"),
        "used_max": conv.get("dataset_used_max"),
        "win_f32_lo": exp.get("target_window_f32_min"),
        "win_f32_hi": exp.get("target_window_f32_max"),
        "win_u16_lo": exp.get("window_u16_min"),
        "win_u16_hi": exp.get("window_u16_max"),
        "mosaic": None if mos is None else True,
        "mosaic_tiles": None if not isinstance(mos, dict) else len(mos.get("tiles", []) or []) or None,
    }
    out, defaulted = {}, []
    for k, d in DEFAULTS.items():
        v = src.get(k)
        if v is None:
            v, miss = d, True
        else:
            miss = False
            v = bool(v) if isinstance(d, bool) else (float(v) if isinstance(d, float) else
                                                    (int(v) if isinstance(d, int) else str(v)))
        out[k] = v
        if miss:
            defaulted.append(k)
    out["unsharp_sigma_um"] = out["unsharp_sigma_px"] * out["pixel_um"]  # the physical PSF, pitch-free
    out["rung"] = int(round(math.log2(max(out["pixel_um"], 1e-6) / 0.6)))  # data.rung_of, without the import
    out["defaulted"] = defaulted
    out["missing"] = not raw
    return out


def load(p=None, timeout=10.0):
    """`flatten(read_json(p))`: the flat scan dict for a volume path / bucket URL / metadata.json.

    `p=None` or an unreadable file gives the documented DEFAULTS with `missing=True` -- never an error,
    so `--scan-meta` can always be passed."""
    return flatten(None if p is None else read_json(p, timeout=timeout))


def _span(lo, hi, w=WIDEN):
    return (lo / w, hi * w)


def ranges_for(meta, widen=WIDEN):
    """Augmentation-config overrides centred on one scan (merge into a preset; `aug.get(..., meta=)`).

    - `paganin`: the scan's OWN Paganin/unsharp parameters (so the identity is in the range) plus the
      corpus-spanned sampling ranges, widened by `widen` on both sides.  Sigmas in MICRONS.
    - `bias`: cupping is a polychromatic/low-energy effect, so its amplitude scales as 78/energy_kev
      (clamped to 0.5-2x) relative to the 78 keV calibration scan -- cupping is a polychromatic effect.  Only the amplitude moves; the field is unchanged.

    The result is a partial cfg: keys absent from the preset are ignored by `aug.apply`, and keys the
    preset does not configure are NOT switched on by this function (`aug.get` merges per op, so a
    preset without `paganin` stays without it)."""
    m = dict(DEFAULTS, **(meta or {}))
    db = _span(*DB_SPAN, w=widen)
    co = _span(*COEFF_SPAN, w=widen)
    sg = _span(*SIGMA_UM_SPAN, w=widen)
    sig_um = float(m.get("unsharp_sigma_um", m["unsharp_sigma_px"] * m["pixel_um"]))
    return {
        "paganin": {
            "energy_kev": float(m["energy_kev"]), "dist_mm": float(m["distance_mm"]),
            "db": float(m["delta_beta"]), "a": float(m["unsharp_coeff"]), "s_um": sig_um,
            "db_lo": min(db[0], float(m["delta_beta"])), "db_hi": max(db[1], float(m["delta_beta"])),
            "a_lo": min(co[0], float(m["unsharp_coeff"])), "a_hi": max(co[1], float(m["unsharp_coeff"])),
            "s_lo": min(sg[0], sig_um), "s_hi": max(sg[1], sig_um),
        },
        "bias": {"max": round(0.3 * min(2.0, max(0.5, 78.0 / max(float(m["energy_kev"]), 1e-6))), 4)},
    }


# ------------------------------------------------------------------ the five conditioning planes

# The corpus ranges every scan plane is min-max normalised over. They are DOCUMENTED CONSTANTS, not
# fitted: a plane must mean the same thing in a run trained on Paris 4 and in a run that later sees a
# 1.1 um mosaic, so the mapping may not move when the corpus grows. A value outside its range is
# clipped, which is the honest thing (the plane says "at or past the end of what we have seen").
META_RANGE = (("energy_kev", 30.0, 120.0, False),        # keV
              ("delta_beta", 10.0, 10000.0, True),       # Paganin delta/beta, log10
              ("unsharp_sigma_um", 0.0, 6.0, False),     # the physical PSF, microns
              ("distance_mm", 0.0, 1000.0, False),       # sample-detector (propagation) distance
              ("pixel_um", 0.6, 1228.8, True))           # the ladder itself: log2, rung 0 .. rung 11


def scan_planes(meta):
    """The five scan-conditioning plane VALUES of one scan's flat metadata (`load`).

    Each is min-max normalised over its documented corpus range (log10 / log2 where the range spans
    decades) and clipped to 0..1. A field the metadata.json did not supply is ZERO, not its default:
    `flatten` fills documented defaults so no caller has to branch, but a plane that says "78 keV" when
    nothing said so would be a lie the model would condition on. `meta=None` (no metadata.json beside
    the volume) gives five zeros."""
    out = np.zeros(len(META_RANGE), np.float32)
    if not meta or meta.get("missing"):
        return out
    bad = set(meta.get("defaulted") or ())
    for i, (k, lo, hi, lg) in enumerate(META_RANGE):
        if k in bad or (k == "unsharp_sigma_um" and {"unsharp_sigma_px", "pixel_um"} & bad):
            continue
        v = float(meta.get(k, 0.0) or 0.0)
        if lg:
            v, lo, hi = math.log10(max(v, 1e-9)), math.log10(lo), math.log10(hi)
        out[i] = float(min(max((v - lo) / max(hi - lo, 1e-9), 0.0), 1.0))
    return out


def fetch(ct, timeout=10.0):
    """The flat scan dict for the `metadata.json` that sits beside a CT volume -- an https URL or a local
    path, a `.zarr` group or one of its levels. NEVER raises: an unreadable or absent file gives the
    documented defaults with `missing=True`, which `scan_planes` turns into five zeros."""
    p = str(ct or "").rstrip("/")
    if p and not p.endswith(".json"):
        head, _, last = p.rpartition("/")
        if head and last.isdigit():   # '<vol>.zarr/0' -> the group beside which metadata.json sits
            p = head
    try:
        return load(p or None, timeout=timeout)
    except Exception:  # noqa: BLE001  (fetch is called on every run start; it may never be the failure)
        return flatten(None)
