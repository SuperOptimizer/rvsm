"""The rung ladder: rung arithmetic, pooling above the top, context cubes, URL vs local, volcomp."""
import numpy as np
import pytest

from rvsm import ladder as L


def test_rung_um_round_trip():
    for k in range(L.NRUNGS):
        assert L.um_rung(L.rung_um(k)) == k
    assert L.rung_um(2) == pytest.approx(2.4)
    assert L.rung_um(11) == pytest.approx(1228.8)
    assert L.um_rung(9.6) == 4


def test_native_um_from_name():
    assert L.native_um("/x/20260101000000-2.400um-0.2m-78keV-masked.zarr") == pytest.approx(2.4)
    assert L.native_um("http://h/a/b-1.100um-x.zarr") == pytest.approx(1.1)
    assert L.native_um("/x/nameless.zarr") == pytest.approx(2.4)  # the documented default


def test_rungs_levels_and_shapes(ct_origin):
    pyr = L.rungs(ct_origin.path)
    assert sorted(pyr) == [2, 3, 4, 5]                      # level l of a 2.4 um volume is rung l + 2
    assert tuple(pyr[2].shape) == (ct_origin.base,) * 3
    assert tuple(pyr[5].shape) == (ct_origin.base >> 3,) * 3
    assert tuple(L.rung_shape(pyr, 5)) == (ct_origin.base >> 3,) * 3
    assert tuple(L.rung_shape(pyr, 7)) == (ct_origin.base >> 5,) * 3  # two rungs above the top


def test_rungs_url_matches_local(ct_origin):
    local = L.rungs(ct_origin.path)
    L.clear_caches()
    remote = L.rungs(ct_origin.url)
    assert sorted(local) == sorted(remote)
    for k in local:
        assert tuple(local[k].shape) == tuple(remote[k].shape)
    a = L.read_rung(remote, 3, (0, 0, 0), 32, dtype=np.uint8)
    L.clear_caches()
    b = L.read_rung(L.rungs(ct_origin.path), 3, (0, 0, 0), 32, dtype=np.uint8)
    assert np.array_equal(a, b)


def test_a_level_the_multiscales_never_learned_about_is_still_found(tmp_path, ct_origin):
    """A mirror grows coarser levels after the group's `multiscales` was written. Trusting the
    declaration alone drops them silently, and `occupancy` then scans a level three decades too big."""
    import json
    import shutil

    from pathlib import Path

    d = tmp_path / Path(ct_origin.path).name
    shutil.copytree(ct_origin.path, d)
    j = json.loads((d / "zarr.json").read_text())
    ds = j["attributes"]["ome"]["multiscales"][0]["datasets"]
    top = max(int(x["path"]) for x in ds)
    j["attributes"]["ome"]["multiscales"][0]["datasets"] = [x for x in ds if int(x["path"]) < top]
    (d / "zarr.json").write_text(json.dumps(j))      # the top level is on disk but no longer declared
    L.clear_caches()
    assert sorted(L.rungs(str(d))) == sorted(L.rungs(ct_origin.path))
    L.clear_caches()


def test_read_rung_matches_the_level(ct_origin):
    pyr = L.rungs(ct_origin.path)
    lo, p = (16, 32, 48), 32
    got = L.read_rung(pyr, 3, lo, p, dtype=np.uint8)
    want = np.asarray(pyr[3][16:48, 32:64, 48:80], np.uint8)
    assert np.array_equal(got, want)


def test_read_rung_outside_is_air(ct_origin):
    pyr = L.rungs(ct_origin.path)
    n = ct_origin.base
    cube = L.read_rung(pyr, 2, (n - 8, 0, 0), 16, dtype=np.uint8)
    assert cube[8:].max() == 0                 # past the end of the array
    assert L.read_rung(pyr, 2, (-64, -64, -64), 32, dtype=np.uint8).max() == 0


def test_above_the_top_pools_from_the_top(ct_origin):
    """Rung 6 does not exist on disk; it must be the 2x pool of rung 5, and rung 7 the pool of that."""
    pyr = L.rungs(ct_origin.path)
    top = np.asarray(pyr[5][:], np.uint8)
    n6 = top.shape[0] // 2
    got6 = L.read_rung(pyr, 6, (0, 0, 0), n6, dtype=np.uint8)
    assert np.array_equal(got6, L.pool2(top))
    got7 = L.read_rung(pyr, 7, (0, 0, 0), n6 // 2, dtype=np.uint8)
    assert np.abs(got7.astype(int) - L.pool2(L.pool2(top)).astype(int)).max() <= 1


def test_context_cubes_are_centred_and_pooled(ct_origin):
    pyr = L.rungs(ct_origin.path)
    p, lo = 32, (64, 64, 64)
    cubes = L.context(ct_origin.path, lo, p, (1, 2, 3), rung=2)
    assert len(cubes) == 3
    assert all(c.shape == (p, p, p) for c in cubes)
    # the same physical centre: the rung-2 centre 80 sits at 40 in rung 3, 20 in rung 4, 10 in rung 5
    for d, c in zip((1, 2, 3), cubes):
        centre = (np.array(lo) + p // 2) // (1 << d)
        want = L.read_rung(pyr, 2 + d, centre - p // 2, p, dtype=np.uint8)
        assert np.array_equal(c, want)


def test_pool2_pads_odd_shapes():
    v = np.full((3, 3, 3), 8, np.uint8)
    out = L.pool2(v)
    assert out.shape == (2, 2, 2)
    assert out[0, 0, 0] == 8                    # a full 2^3 block
    assert out[1, 1, 1] == 1                    # one voxel of 8 against seven of zero-pad


def test_missing_volcomp_raises(monkeypatch, ct_origin):
    """usrm2 swallowed this and silently returned a pyramid with levels missing."""
    monkeypatch.setenv("VOLCOMP_LIB", "/nonexistent/libvolcomp.so")
    L.clear_caches()
    with pytest.raises(RuntimeError, match="VOLCOMP_LIB"):
        L.rungs(ct_origin.path)

    import builtins
    monkeypatch.delenv("VOLCOMP_LIB", raising=False)
    real = builtins.__import__

    def no_volcomp(name, *a, **k):
        if name == "volcomp_zarr":
            raise ImportError("no volcomp_zarr")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_volcomp)
    monkeypatch.delitem(__import__("sys").modules, "volcomp_zarr", raising=False)
    L.clear_caches()
    with pytest.raises(RuntimeError, match="volcomp_zarr"):
        L.rungs(ct_origin.path)


def test_no_levels_raises(tmp_path):
    (tmp_path / "empty.zarr").mkdir()
    with pytest.raises(RuntimeError, match="no integer pyramid levels"):
        L.rungs(str(tmp_path / "empty.zarr"))


def test_base_rung_of_a_level_path(ct_origin):
    assert L.base_rung(ct_origin.path + "/0") == 2
    assert L.base_rung(ct_origin.path + "/2") == 4
    assert L.base_rung(ct_origin.path) == 2


def test_volcomp_level_round_trips(tmp_path, has_volcomp):
    """A real volcomp-encoded level (the codec wants 128^3 chunks) decodes through `open_zarr`."""
    if not has_volcomp:
        pytest.skip("volcomp_zarr / libvolcomp.so not available")
    import volcomp_zarr as vc
    import zarr
    from tests.conftest import slab
    d = tmp_path / "20260101000000-2.400um-v.zarr"
    (d / "0").parent.mkdir(parents=True, exist_ok=True)
    v = slab((128, 128, 128))
    a = zarr.create_array(str(d / "0"), shape=v.shape, chunks=(128,) * 3, dtype="uint8", fill_value=0,
                          compressors=None, serializer=vc.VolcompCodec(q=0), overwrite=True)
    a[:] = v
    pyr = L.rungs(str(d))
    assert sorted(pyr) == [2]
    assert np.array_equal(L.read_rung(pyr, 2, (0, 0, 0), 128, dtype=np.uint8), v)  # q=0 is exact


def test_pool2_is_the_float_mean_floored():
    """The integer-sum pool2 against the float formula it replaced, odd shapes included."""
    rng = np.random.default_rng(0)
    for shape in ((8, 8, 8), (7, 9, 5), (1, 2, 3), (33, 16, 17)):
        v = rng.integers(0, 256, size=shape, dtype=np.uint8)
        s = np.array(v.shape, np.int64)
        n = -(-s // 2)
        pv = np.pad(v, [(0, int(q)) for q in n * 2 - s])
        want = pv.reshape(n[0], 2, n[1], 2, n[2], 2).astype(np.float32).mean((1, 3, 5)).astype(np.uint8)
        assert np.array_equal(L.pool2(v), want), shape
    v = np.full((4, 4, 4), 255, np.uint8)
    assert (L.pool2(v) == 255).all()
