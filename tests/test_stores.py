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
