"""Shared synthetic fixtures. Everything here is CPU-only and built in a tmp_path.

The CT fixture is the one thing every other test stands on: a real sharded zarr-v3 pyramid with a bright
tilted slab in it, served over HTTP by a range-capable server, so the ladder, the axis and (later) the
shard cache are exercised against the same bytes locally and remotely.
"""
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# The build the laptop keeps; `VOLCOMP_LIB` in the environment wins over it.
LIB_CANDIDATES = ("/home/forrest/volume-compressor/build/release/libvolcomp.so",
                  str(ROOT.parent / "volume-compressor/build/release/libvolcomp.so"))

VOL = "20260101000000-2.400um-0.2m-78keV-masked.zarr"  # level 0 = 2.4 um = rung 2
BASE = 256          # edge of level 0
NLEV = 4            # levels 0..3 -> rungs 2..5
CHUNK = 16
SHARD = 64
TILT = 0.25         # the slab's dy/dx
HALF = 12           # the slab's half thickness, in level-0 voxels


@pytest.fixture(scope="session", autouse=True)
def volcomp_lib():
    """Point `VOLCOMP_LIB` at a local build when the environment has not already done so, so the whole
    suite can decode volcomp arrays. Tests that must have it use the `has_volcomp` fixture."""
    if not os.environ.get("VOLCOMP_LIB"):
        for c in LIB_CANDIDATES:
            if os.path.exists(c):
                os.environ["VOLCOMP_LIB"] = c
                break
    return os.environ.get("VOLCOMP_LIB")


@pytest.fixture(scope="session")
def has_volcomp(volcomp_lib):
    try:
        import volcomp_zarr  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(autouse=True)
def clean_caches():
    """The ladder caches decoded levels and open pyramids by path; a tmp_path per test must not inherit
    another test's entries."""
    from rvsm import ladder
    ladder.clear_caches()
    yield
    ladder.clear_caches()


def slab(shape):
    """A bright slab tilted in y as a function of x, the same in every z slice.

    Because the tilt is in x and the slab spans the whole x range, the non-air centroid of any z slice is
    exactly the volume centre: `axis.derive` must recover it. 255 on the slab, 0 (air) elsewhere."""
    Z, Y, X = shape
    y, x = np.arange(Y)[:, None], np.arange(X)[None, :]
    c = BASE / 2.0
    plane = (np.abs(y - (c + TILT * (x - c))) < HALF).astype(np.uint8) * np.uint8(255)
    return np.broadcast_to(plane, (Z, Y, X)).copy()


def _level(path, arr, use_volcomp):
    import zarr
    kw = {}
    if use_volcomp:
        try:
            import volcomp_zarr as vc
            kw = {"serializer": vc.VolcompCodec(q=0)}
        except Exception:  # noqa: BLE001
            kw = {}
    sh = tuple(min(SHARD, n) for n in arr.shape)
    a = zarr.create_array(str(path), shape=arr.shape, chunks=(CHUNK,) * 3, shards=sh,
                          dtype="uint8", fill_value=0, compressors=None, overwrite=True, **kw)
    a[:] = arr
    return a


def build_ct(root, base=BASE, nlev=NLEV, use_volcomp=False):
    """A CT mirror pyramid `<...-2.400um-...>.zarr/<integer level>` with OME multiscales, sharded."""
    from rvsm import ladder
    d = pathlib.Path(root) / VOL
    d.mkdir(parents=True, exist_ok=True)
    v = slab((base, base, base))
    for l in range(nlev):  # noqa: E741
        _level(d / str(l), v, use_volcomp)
        v = ladder.pool2(v)
    meta = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"version": "0.5", "multiscales": [{
        "version": "0.5", "name": VOL, "type": "mean",
        "axes": [{"name": q, "type": "space", "unit": "micrometer"} for q in "zyx"],
        "datasets": [{"path": str(l), "coordinateTransformations":
                      [{"type": "scale", "scale": [ladder.rung_um(2 + l)] * 3}]} for l in range(nlev)]}]}}}
    (d / "zarr.json").write_text(json.dumps(meta))
    (d / "metadata.json").write_text((ROOT / "docs/example_metadata.json").read_text())
    return d


def _serve(root):
    """A range-capable http.server subprocess over `root`; returns (popen, base url)."""
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    p = subprocess.Popen([sys.executable, str(HERE / "serve.py"), str(port), str(root)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5).read(1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.05)
    else:
        p.kill()
        pytest.fail("the test http server did not start")
    return p, f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def ct_origin(tmp_path_factory, volcomp_lib):
    """A served CT pyramid. Yields a namespace with `.url`, `.path` (both naming the zarr GROUP),
    `.root` (the served directory) and `.volcomp` (whether the levels are volcomp-encoded)."""
    import types
    root = tmp_path_factory.mktemp("origin")
    use = False
    try:
        import volcomp_zarr  # noqa: F401
        use = True
    except Exception:  # noqa: BLE001
        use = False
    try:
        d = build_ct(root, use_volcomp=use)
    except Exception:  # noqa: BLE001  (volcomp rejects 16^3 chunks: fall back to plain zarr)
        use = False
        d = build_ct(root, use_volcomp=False)
    srv, url = _serve(root)
    try:
        yield types.SimpleNamespace(url=f"{url}/{VOL}", path=str(d), root=str(root), volcomp=use,
                                    base=BASE, nlev=NLEV, vol=VOL)
    finally:
        srv.kill()
        srv.wait()


@pytest.fixture
def umbilicus(tmp_path):
    """A straight axis through the volume centre, in both published formats. Yields (json, txt, points)
    where the points are (z, y, x) in rung-2 voxels."""
    c = BASE / 2.0
    pts = [(float(z), c, c) for z in range(0, BASE + 1, 32)]
    j = tmp_path / "umbilicus.json"
    j.write_text(json.dumps({"control_points": [{"z": z, "y": y, "x": x} for z, y, x in pts]}))
    t = tmp_path / "umbilicus.txt"   # volpkg: "x, y, z" per line, 1-based
    t.write_text("\n".join(f"{x + 1}, {y + 1}, {z + 1}" for z, y, x in pts) + "\n")
    return str(j), str(t), pts


@pytest.fixture
def small_cfg(tmp_path, ct_origin, umbilicus):
    """The tiny config every CPU test trains and samples with."""
    from rvsm.config import Config
    return Config(ct=ct_origin.path, umbilicus=umbilicus[0], out=str(tmp_path / "run"),
                  size="1m", patch=32, batch=1, region=64, ctx=(1, 2, 3), rungs=(2, 3),
                  aff_offsets=(4, 8), ect_block=8, steps=20, workers=0, compile=False,
                  heldout=1, windows_per_region=4, eval_every=10)
