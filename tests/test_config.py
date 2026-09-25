"""The one place the channel contract lives."""
import pytest

from rvsm.config import RUNG_ITEM_KEYS, Config, load


def test_default_layout_is_21_in_14_out():
    L = Config().layout()
    assert L.cin == 21 and L.cout == 14
    assert (L.i_cas, L.i_planes, L.n_planes, L.i_scale, L.i_rad) == (10, 11, 6, 17, 18)
    assert (L.nprob, L.cout_t, L.i_log, L.i_aff) == (2, 4, 4, 5)
    assert L.head_names()[:5] == ["recto", "verso", "midline", "thickness", "logvar"]
    assert L.head_names()[5:8] == ["aff8_z", "aff8_y", "aff8_x"]
    assert len(L.stem_names()) == L.cin and len(L.head_names()) == L.cout


def test_rung_item_keys_are_frozen():
    assert RUNG_ITEM_KEYS == ("ct", "tgt", "w", "tch", "lo", "cyx", "sym", "rung", "norm", "cm", "cx",
                              "lo1", "cyx1", "rmax", "meta")


def test_fingerprint_ignores_only_the_resume_fields():
    a = Config(ct="x")
    assert a.fingerprint() == Config(ct="x", steps=99, eval_every=7, workers=1, gpus=(1, 2),
                                     rounds=9, ct_seed="/m", ckpt_act=0, compile=False,
                                     verso_min_dice=0.9).fingerprint()
    assert a.fingerprint() == Config(ct="x", gn_bf16=True).fingerprint()   # a logged precision switch
    assert a.fingerprint() != Config(ct="x", patch=128).fingerprint()
    assert a.fingerprint() != Config(ct="y").fingerprint()


def test_load_toml_then_overrides(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text('ct = "/a/b.zarr"\nsize = "15m"\nrungs = [2, 3]\n[phase]\ntrain_min = 1.5\n')
    c = load(str(p), {"size": "60m", "steps": None})
    assert c.ct == "/a/b.zarr" and c.size == "60m" and c.rungs == (2, 3)
    assert c.train_min == pytest.approx(1.5)
    assert c.steps == Config().steps          # a None override does not touch the default


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text('lrr = 1.0\n')
    with pytest.raises(ValueError, match="unknown config key"):
        load(str(p), {})


def test_to_json_is_serialisable():
    import json
    j = Config().to_json()
    assert json.loads(json.dumps(j))["layout"]["cin"] == 21


def test_a_resume_compares_field_values_not_the_stale_stored_hash():
    """Moving a field into FINGERPRINT_EXCLUDE changes every hash; a run's stored config must still
    resume when every field outside the exclude set matches (paris4 at step 12000, when `cascade` and
    `cascade_drop` joined it), and must still be refused when one of them differs."""
    from rvsm import config as CFG
    stored = Config(ct="x", cascade="mix", cascade_drop=0.1).to_json()
    stored["fingerprint"] = "written-by-older-code"
    stored["config"].pop("self_p_end_step")                  # a field the stored config predates
    assert CFG.stored_fingerprint(stored) == Config(ct="x").fingerprint()
    stored["config"]["patch"] = 128
    assert CFG.stored_fingerprint(stored) != Config(ct="x").fingerprint()
    assert CFG.stored_fingerprint({"fingerprint": "abc"}) == "abc"
