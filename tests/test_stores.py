import numpy as np

from rvsm import stores


def test_store_round_trip_and_pool(tmp_path, volcomp_lib):
    u = np.zeros((128, 256, 128), np.uint8)
    u[:, 100:110, :] = 255
    p = stores.store_path(str(tmp_path), "recto", (0, 1024, 2048))
    stores.write(p, u, (0, 1024, 2048), q=0)
    assert stores.is_done(p)
    a = stores.open_store(p)
    assert a.attrs["channels"] == ["recto"] and a.attrs["volcomp_q"] == 0 and a.attrs["rung"] == 2
    cube, inside = stores.read_store(a, 2, (0, 1024, 2048), (128, 256, 128))
    assert np.array_equal(cube, u) and inside.all()
    cube3, inside3 = stores.read_store(a, 3, (0, 512, 1024), (64, 128, 64))
    assert cube3.shape == (64, 128, 64) and inside3.all() and cube3[:, 50:55, :].max() == 255
    part, ins = stores.read_store(a, 2, (0, 1024 + 128, 2048), (128, 256, 128))
    assert ins[:, :128, :].all() and not ins[:, 128:, :].any() and part[:, 128:, :].max() == 0


def test_tmp_never_left_behind(tmp_path, volcomp_lib):
    p = stores.store_path(str(tmp_path), "verso", (0, 0, 0))
    stores.write(p, np.zeros((128, 128, 128), np.uint8), (0, 0, 0))
    import os
    assert not os.path.exists(p + ".tmp") and stores.is_done(p)


def test_pooled_chunk_encodes_write_the_same_store(tmp_path, volcomp_lib, monkeypatch):
    """`stores._codec`: the volcomp chunk encodes on a thread pool (zarr's event loop ran them one by one)
    write byte-identical stores -- shards and zarr.json -- to the plain codec, lossless and lossy, with
    all-zero chunks (stored as missing) among them."""
    import filecmp
    import os

    import numpy as np
    from rvsm import stores
    rng = np.random.default_rng(3)
    blk = np.zeros((256, 256, 384), np.uint8)
    blk[:128, :, :128] = rng.integers(0, 256, (128, 256, 128), dtype=np.uint8)
    blk[128:, 128:, 200:] = (rng.random((128, 128, 184)) < 0.2) * rng.integers(1, 256, (128, 128, 184))
    roots = {}
    for n in (0, 4):
        monkeypatch.setattr(stores, "ENCODE_THREADS", n)
        monkeypatch.setattr(stores, "_ENC", {})
        for q in (0, 8):
            p = str(tmp_path / f"s{n}_q{q}")
            stores.write(p, blk, (0, 128, 256), rung=2, channels=("x",), q=q)
            roots[(n, q)] = p
    for q in (0, 8):
        a, b = roots[(0, q)], roots[(4, q)]
        fa = sorted(os.path.relpath(os.path.join(d, f), a) for d, _, fs in os.walk(a) for f in fs)
        fb = sorted(os.path.relpath(os.path.join(d, f), b) for d, _, fs in os.walk(b) for f in fs)
        assert fa == fb and "zarr.json" in fa
        assert all(filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False) for f in fa)
        got = np.asarray(stores.open_store(b)[:])
        if q == 0:
            assert np.array_equal(got, blk)
    assert type(stores._codec(0)).__name__ == "PooledVolcompCodec"
