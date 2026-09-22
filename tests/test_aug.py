"""The augmentation presets, the micron-to-voxel rung conversion, and what `apply` may and may not touch."""
import math

import pytest
import torch

from rvsm import aug, ladder, scanmeta


def test_only_four_presets_and_full2_adds_exactly_the_physics_pair():
    assert sorted(aug.PRESETS) == ["full", "full2", "geo", "none"]
    assert set(aug.get("full2")) - set(aug.get("full")) == {"paganin", "shuffle"}
    assert set(aug.get("full")) - set(aug.get("full2")) == set()
    assert aug.get("none") == {"sym": False}
    assert set(aug.get("geo")) == {"sym"} and aug.get("geo")["sym"] is True
    with pytest.raises(KeyError):
        aug.get("all3")                      # the 25 ablation presets do not exist in rvsm


def test_get_recentres_the_physics_ops_on_the_scan_metadata():
    meta = scanmeta.load()                   # the documented defaults (the 2.4 um 78 keV Paris 4 scan)
    cfg = aug.get("full2", meta=meta)
    r = scanmeta.ranges_for(meta)
    assert cfg["paganin"]["db"] == r["paganin"]["db"]
    assert cfg["paganin"]["s_um"] == r["paganin"]["s_um"]
    assert cfg["paganin"]["p"] == aug.PAGANIN["paganin"]["p"]     # merged PER OP: `p` survives
    assert "paganin" not in aug.get("geo", meta=meta)             # ... and a preset without it stays so


def test_paganin_is_the_identity_at_the_scans_own_parameters():
    """The transfer function is the RATIO of the wanted filter to the one the scan was reconstructed
    with, so pinning the sampled range to the scan's own values must give T == 1 exactly."""
    torch.manual_seed(0)
    k = dict(aug.PAGANIN["paganin"])
    k.update(db_lo=k["db"], db_hi=k["db"], a_lo=k["a"], a_hi=k["a"], s_lo=k["s_um"], s_hi=k["s_um"],
             vox_um=ladder.rung_um(2))
    c = torch.rand(2, 1, 32, 32, 32)
    y = aug._paganin_jitter(c, k)
    assert float((y - c).abs().max()) <= 1.0 / 255.0


def test_paganin_at_other_parameters_actually_changes_the_cube():
    torch.manual_seed(0)
    k = dict(aug.PAGANIN["paganin"])
    k.update(db_lo=200.0, db_hi=200.0, vox_um=ladder.rung_um(2))
    c = torch.rand(1, 1, 32, 32, 32)
    assert float((aug._paganin_jitter(c, k) - c).abs().max()) > 1.0 / 255.0


def test_for_rung_halves_the_sigmas_one_rung_coarser_and_is_the_identity_at_rung_2():
    cfg = aug.get("full2")
    assert aug.for_rung(cfg, 2) == {**cfg, "paganin": {**cfg["paganin"], "vox_um": ladder.rung_um(2)}}
    for k in aug.SIGMA_KEYS:
        if k in cfg:
            assert cfg[k] == aug.for_rung(cfg, 2)[k], k    # exactly today's numbers at rung 2
    c3 = aug.for_rung(cfg, 3)
    for name, keys in aug.SIGMA_KEYS.items():
        for q in (keys if name in cfg else ()):
            if q in cfg[name]:
                assert c3[name][q] == pytest.approx(cfg[name][q] / 2.0), (name, q)
    assert c3["paganin"]["s_lo"] == cfg["paganin"]["s_lo"]   # already in MICRONS: not converted
    assert c3["paganin"]["vox_um"] == ladder.rung_um(3)
    assert c3["ring"] == cfg["ring"] and c3["elastic"] == cfg["elastic"]  # detector / geometry: not PSFs


def test_for_rung_takes_a_per_sample_rung():
    cfg = aug.for_rung(aug.get("full2"), [2, 4])
    assert cfg["paganin"]["vox_um"] == [ladder.rung_um(2), ladder.rung_um(4)]


def _stack(B=2, S=16, nimg=2, nplane=3):
    """(x, tg) shaped like the real stem: `nimg` image cubes, `nplane` non-image planes, then rz, ry, rx."""
    torch.manual_seed(0)
    img = torch.rand(B, nimg, S, S, S)
    planes = torch.arange(nplane, dtype=torch.float32).view(1, nplane, 1, 1, 1).expand(B, nplane, S, S, S)
    rad = torch.zeros(B, 3, S, S, S)
    rad[:, 2] = 1.0                                   # a clean unit radial vector, pointing along +x
    x = torch.cat([img, planes.contiguous(), rad], 1)
    tg = (torch.rand(B, 1, S, S, S) > 0.5).float()
    return x, tg


def test_intensity_ops_leave_the_cascade_scale_and_radial_channels_untouched():
    """`nimg` is the whole point of the argument: a brightness shift applied to a dropped (all-zero)
    cascade channel would move it off the "no coarse prediction" value the model is taught to read."""
    x, tg = _stack()
    nimg = 2
    cfg = {k: v for k, v in aug.get("full2").items() if k not in ("rot", "scale", "shear", "elastic",
                                                                  "sheetcomp", "cor")}
    cfg = {k: ({**v, "p": 1.0} if isinstance(v, dict) else v) for k, v in cfg.items()}
    torch.manual_seed(1)
    y, ty = aug.apply(x.clone(), tg.clone(), cfg, nimg=nimg, rung=2)
    assert not torch.allclose(y[:, :nimg], x[:, :nimg])          # the images DID change
    assert torch.equal(y[:, nimg:], x[:, nimg:])                 # everything else is byte-identical
    assert torch.equal(ty, tg)                                   # no spatial op: the target is untouched


def test_spatial_ops_renormalise_the_radial_vector():
    x, tg = _stack(S=24)
    cfg = {"sym": True, "rot": {"p": 1.0, "max_deg": 30}, "scale": {"p": 1.0, "lo": 0.8, "hi": 1.25, "iso": 0.5},
           "shear": {"p": 1.0, "max": 0.1}, "elastic": {"p": 1.0, "sigma": 4.0, "grid": 12}}
    torch.manual_seed(3)
    y, ty = aug.apply(x.clone(), tg.clone(), cfg, nimg=2, rung=2)
    n = y[:, -3:].norm(dim=1)
    core = n[:, 4:-4, 4:-4, 4:-4]
    assert float((core - 1.0).abs().max()) < 1e-4                # unit, everywhere the vector survived
    assert not torch.allclose(y[:, -3:], x[:, -3:])              # ... and it really was rotated
    assert float(ty.min()) >= 0.0 and float(ty.max()) <= 1.0     # the target stays a probability


def test_apply_with_no_config_is_the_identity():
    x, tg = _stack()
    y, ty = aug.apply(x, tg, None)
    assert y is x and ty is tg


def test_shuffle_changes_the_composition_order_not_the_op_set():
    """With `shuffle` every sample applies the drawn ops in its own order (SinoSynth); without it they
    run in the fixed pipeline order. Both must produce a finite, differently-ordered result."""
    x, _ = _stack(B=4, S=16, nimg=1)
    ops = {"bright": {"p": 1.0, "max": 0.3}, "contrast": {"p": 1.0, "max": 1.5},
           "gamma": {"p": 1.0, "max": 1.8}}
    torch.manual_seed(7)
    a = aug.intensity(x[:, :1].clone(), dict(ops))
    torch.manual_seed(7)
    b = aug.intensity(x[:, :1].clone(), {**ops, "shuffle": True})
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.allclose(a, b)


def test_sigma_keys_cover_every_psf_op_the_presets_configure():
    cfg = aug.get("full2")
    # every PSF-type op, whether or not an rvsm preset configures it (`haze` / `unsharp` do not appear
    # in `full` or `full2`; their entries are kept so a preset that adds one converts it too).
    assert set(aug.SIGMA_KEYS) == {"blur", "sharpen", "unsharp", "aniso_blur", "haze"}
    assert {"blur", "sharpen", "aniso_blur"} <= set(cfg)
    assert math.isclose(aug.RUNG2_UM, ladder.rung_um(2))
