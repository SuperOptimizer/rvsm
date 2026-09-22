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


# --- teachers (commit 2) -------------------------------------------------------------------------
# The tiny `fake` teacher lives in tests/teachers_fixture.py; imported here so `fake_teacher` is a
# fixture of the whole suite. Keep this block at the END of the file: other commits append their own.
from tests.teachers_fixture import fake_teacher  # noqa: E402,F401


# ===================================================================================================
# commit 3 (stream / regions / sample / prep / model) fixtures -- appended; keep additions at the end.
# ===================================================================================================

@pytest.fixture
def region_cfg(small_cfg):
    """`small_cfg` with a 128^3 region.

    A store is a volcomp array and volcomp encodes 128^3 blocks only, so `stores.out_array` refuses a
    shape that is not a multiple of 128: 128 is the smallest region a real store can have. Everything
    else stays tiny (patch 32, rungs 2-3, ctx 1-3), so the fixture CT's 256^3 holds 2x2x2 regions."""
    from dataclasses import replace
    return replace(small_cfg, region=128)


@pytest.fixture
def synth_run(region_cfg, ct_origin, umbilicus, has_volcomp):
    """A run directory with SYNTHETIC rung-2 region stores, written from the fixture CT itself.

    One store per rung-2 region per channel: `recto` = the bright slab as a probability, `verso` = the
    slab shifted by one voxel in y (so the two channels are not the same array), `rw` = full agreement,
    `midline` / `thickness` = a non-zero code everywhere the slab is (code 0 is the no-data marker the
    sampler turns into weight 0). Yields a namespace with `.cfg`, `.root`, `.ax`, `.pyr`, `.regions`
    (the rung-2 records) and `.lo` (their origins)."""
    import types

    import numpy as np

    from rvsm import axis as AX, ladder, regions as RG, stores
    if not has_volcomp:
        pytest.skip("a region store is a volcomp array; no libvolcomp on this host")
    cfg = region_cfg
    root = cfg.out
    os.makedirs(root, exist_ok=True)
    ax = AX.load(cfg.umbilicus, ct=cfg.ct)
    pyr = ladder.rungs(cfg.ct)
    recs = RG.region_list(pyr, rungs=cfg.rungs, patch=cfg.patch, region=cfg.region,
                          boost=cfg.rung_boost, occ_min_fine=cfg.occ_min_fine,
                          occ_min_coarse=cfg.occ_min_coarse)
    two = [r for r in recs if r["k"] == 2]
    los = []
    for r in two:
        lo, sz = np.array(r["lo"], np.int64), np.array(r["size"], np.int64)
        ct = ladder.read_rung(pyr, 2, lo, sz, dtype=np.uint8)
        recto = (ct > 0).astype(np.uint8) * np.uint8(255)
        verso = np.roll(recto, 1, axis=1)
        code = np.where(recto > 0, np.uint8(128), np.uint8(0))  # 0 = no data, as targets.encode_signed
        for chan, blk, q in (("recto", recto, 8), ("verso", verso, 8), ("rw", recto * 0 + 255, 8),
                             ("midline", code, 0), ("thickness", code, 0)):
            stores.write(stores.store_path(root, chan, lo), blk, lo, rung=2, channels=(chan,), q=q,
                         volume=cfg.ct, umbilicus=cfg.umbilicus)
        los.append(tuple(int(v) for v in lo))
    yield types.SimpleNamespace(cfg=cfg, root=root, ax=ax, pyr=pyr, regions=recs, two=two, lo=los)
    RG.clear_pool()


# ============================================================================================
# Fixtures for the loss / augmentation / calibration / distance-target / evaluation modules.
# (Appended for rvsm/losses.py, aug.py, calib.py, targets.py, evalsurf.py and their tests.)
# ============================================================================================

@pytest.fixture
def slab_region(tmp_path, volcomp_lib):
    """A builder for the region stores `rvsm.targets` reads: two parallel sheets perpendicular to x.

    The umbilicus is put far away in -x by default, so the radial direction is +x everywhere in the
    region and the geometry is exactly one-dimensional: the recto face sits at `recto_x` and the verso
    face, one sheet thickness further IN, at `verso_x`. Then, for a voxel at x,

        d_recto = x - recto_x     d_verso = x - verso_x
        midline = x - (recto_x + verso_x) / 2        thickness = recto_x - verso_x

    which is what the tests assert against, code by code. `axis_yx` inside the region instead puts every
    voxel within the excluded radius of the axis, which is how the near-axis weight-0 rule is tested.

    Returns a namespace with `.root` (a run directory holding `stores/round_0/{recto,verso}/...`),
    `.lo`, `.ax` (3, N) control points in rung-2 voxels, and the geometry it was built with."""
    import types

    import numpy as np

    from rvsm import stores

    def build(name="r", n=128, recto_x=80, verso_x=70, half=1, verso=True,
              axis_yx=(64.0, -1000.0), lo=(0, 0, 0), round_=0):
        root = str(tmp_path / name)
        x = np.arange(n)[None, None, :]
        rec = np.where(np.abs(x - recto_x) <= half, np.uint8(255), np.uint8(0))
        rec = np.broadcast_to(rec, (n, n, n)).astype(np.uint8)
        stores.write(stores.store_path(root, "recto", lo, round_), rec, lo, rung=2,
                     channels=("recto",), q=8)
        if verso:
            v = np.where(np.abs(x - verso_x) <= half, np.uint8(255), np.uint8(0))
            stores.write(stores.store_path(root, "verso", lo, round_),
                         np.broadcast_to(v, (n, n, n)).astype(np.uint8), lo, rung=2,
                         channels=("verso",), q=8)
        zs = np.arange(0, n + 1, 16, dtype=np.float64)
        ax = np.stack([zs, np.full_like(zs, float(axis_yx[0])), np.full_like(zs, float(axis_yx[1]))])
        return types.SimpleNamespace(root=root, lo=tuple(lo), ax=ax, n=n, recto_x=recto_x,
                                     verso_x=verso_x, round_=round_)

    return build
