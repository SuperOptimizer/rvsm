"""The scan metadata: the example file, the five conditioning planes, and the augmentation ranges."""
import json
import pathlib

import numpy as np
import pytest

from rvsm import scanmeta as S

EXAMPLE = pathlib.Path(__file__).resolve().parent.parent / "docs/example_metadata.json"


def test_example_parses():
    m = S.load(str(EXAMPLE))
    assert m["missing"] is False
    # the example is a single-tile scan, so only the mosaic keys are absent
    assert set(m["defaulted"]) == {"mosaic", "mosaic_tiles"}
    assert m["energy_kev"] == pytest.approx(78.0)
    assert m["pixel_um"] == pytest.approx(2.4)        # mm -> um
    assert m["distance_mm"] == pytest.approx(220.0)
    assert m["delta_beta"] == pytest.approx(1000.0)
    assert m["unsharp_sigma_px"] == pytest.approx(1.2)
    assert m["unsharp_sigma_um"] == pytest.approx(2.88)
    assert m["helical"] is True
    assert m["rung"] == 2


def test_planes_are_in_range():
    p = S.scan_planes(S.load(str(EXAMPLE)))
    assert p.shape == (len(S.META_RANGE),)
    assert p.dtype == np.float32
    assert (p >= 0).all() and (p <= 1).all()
    assert p[0] == pytest.approx((78.0 - 30.0) / 90.0)        # energy, linear
    assert p[4] == pytest.approx(2 / 11, abs=1e-5)            # pixel_um: log2 over rungs 0..11 -> rung 2


def test_missing_metadata_gives_zeros(tmp_path):
    assert (S.scan_planes(None) == 0).all()
    assert (S.scan_planes(S.load(str(tmp_path / "nope.json"))) == 0).all()
    m = S.load(str(tmp_path / "nope.json"))
    assert m["missing"] is True
    assert m["energy_kev"] == pytest.approx(78.0)             # the documented default is still filled in


def test_a_defaulted_field_gives_a_zero_plane(tmp_path):
    """A plane that said "78 keV" when nothing did would be a lie the model conditions on."""
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps({"scan": {"tomo": {"acquisition": {
        "sampleDetectorDistance": 220.0,
        "detector": {"samplePixelSize": 0.0024}}}}}))
    m = S.load(str(p))
    assert "energy_kev" in m["defaulted"]
    v = S.scan_planes(m)
    assert v[0] == 0.0                                        # energy was not supplied
    assert v[3] > 0.0 and v[4] > 0.0                          # distance and pixel size were


def test_fetch_beside_a_volume(ct_origin):
    for p in (ct_origin.path, ct_origin.path + "/0", ct_origin.url, ct_origin.url + "/0"):
        m = S.fetch(p)
        assert m["missing"] is False, p
        assert m["energy_kev"] == pytest.approx(78.0)


def test_fetch_never_raises(tmp_path):
    for p in (None, "", str(tmp_path / "gone.zarr"), "http://127.0.0.1:1/x.zarr", 12345):
        m = S.fetch(p, timeout=1.0)
        assert m["missing"] is True
        assert (S.scan_planes(m) == 0).all()


def test_ranges_contain_the_scans_own_values():
    m = S.load(str(EXAMPLE))
    r = S.ranges_for(m)
    pg = r["paganin"]
    assert pg["db_lo"] <= m["delta_beta"] <= pg["db_hi"]
    assert pg["a_lo"] <= m["unsharp_coeff"] <= pg["a_hi"]
    assert pg["s_lo"] <= m["unsharp_sigma_um"] <= pg["s_hi"]
    assert pg["energy_kev"] == pytest.approx(78.0)
    assert r["bias"]["max"] == pytest.approx(0.3, abs=1e-4)   # the 78 keV calibration scan: no rescaling


def test_ranges_scale_the_bias_with_energy():
    lo = S.ranges_for(dict(S.DEFAULTS, energy_kev=39.0))["bias"]["max"]
    hi = S.ranges_for(dict(S.DEFAULTS, energy_kev=200.0))["bias"]["max"]
    assert lo == pytest.approx(0.6, abs=1e-4)                 # 2x, the clamp
    assert hi == pytest.approx(0.15, abs=1e-4)                # 0.5x, the clamp
