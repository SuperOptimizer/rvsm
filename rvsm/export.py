"""The tracer contract: what leaves rvsm for a downstream tracer, and how it is encoded.

One multi-head pass over the box produces every plane the contract asks for -- the recto and verso
probabilities, the distance field, the thickness, the confidence -- because they are all pointwise
functions of the SAME network output, and blending them together costs one forward pass instead of five
over the same voxels. `export_tracer` therefore takes the pass as a callable (`probs_multi(origin, size)
-> {name: (Z, Y, X) float32}`), which is what lets the encoding be tested with a synthetic field today
and driven by the student's region pass in commit 5.

The normal field and the gradient magnitude are DERIVED here, from the exported distance field, with a
Scharr kernel -- never read out of the network. A ReLU/SiLU conv decoder has piecewise-constant
gradients, so its analytic gradient is noisier than a finite difference of its own output; deriving the
normal from the stored field also guarantees that what the tracer reads is exactly the gradient of what
it reads.

Sign convention, stated in every store's attrs and pinned by a test: d > 0 and n point from the VERSO
face towards the RECTO face, i.e. radially OUTWARD from the scroll axis (dot(n, radial) > 0).

Encoding: q8 (lossy) for the probabilities, q0 (lossless) for every field -- their code 0 means NO DATA
and their other codes are a distance in 0.25-voxel steps or a normal component, neither of which a codec
may round.
"""
from __future__ import annotations

import os

import numpy as np

from rvsm import stores

TRACER_UNIT = 0.25    # the distance quantum, in voxels of the exported rung
TRACER_OFF = 128      # the zero code of a signed field
TRACER_CAP = 31.75    # the largest distance a code can hold (127 * 0.25)
NORMAL_SCALE = 127.0

SIGN_CONVENTION = ("d > 0 and n pointing from the VERSO face towards the RECTO face, i.e. radially "
                   "OUTWARD from the scroll axis (dot(n, radial) > 0)")


def enc_signed(d, valid):
    """voxels -> uint8, code 0 = no data."""
    c = np.rint(np.clip(d, -TRACER_CAP, TRACER_CAP) / TRACER_UNIT) + TRACER_OFF
    return np.where(valid, np.clip(c, 1, 255), 0).astype(np.uint8)


def enc_normal(n, valid):
    """a normal COMPONENT in -1..1 -> uint8, code 0 = no data."""
    c = np.rint(np.clip(n, -1.0, 1.0) * NORMAL_SCALE) + TRACER_OFF
    return np.where(valid, np.clip(c, 1, 255), 0).astype(np.uint8)


def dec_normal(u):
    return (np.asarray(u, np.float32) - TRACER_OFF) / NORMAL_SCALE


def scharr3(d):
    """Scharr gradient of a (Z, Y, X) field, (3, Z, Y, X) in ZYX order.

    Scharr is the rotationally best-behaved 3x3 first derivative; the 3D kernel is the 1D derivative
    [-1, 0, 1] along the axis times the 1D smoother [3, 10, 3] on the other two, normalised so a unit
    ramp gives 1."""
    import scipy.ndimage as ndi
    sm = np.array([3.0, 10.0, 3.0]) / 16.0
    dv = np.array([-1.0, 0.0, 1.0]) / 2.0
    out = []
    for a in range(3):
        g = np.asarray(d, np.float32)
        for b in range(3):
            g = ndi.correlate1d(g, dv if b == a else sm, axis=b, mode="nearest")
        out.append(g)
    return np.stack(out)


def tracer_fields(sd, thick=None, conf=None, valid=None):
    """(recto-face sdist, normals, |grad|, valid) from a predicted distance field.

    `sd` is the model's distance field in voxels. With `thick` it is a MIDLINE distance and the
    recto-face field is `m - t/2` -- the subtraction the contract asks for, done HERE, so that the
    training-time representation (which is what buys non-crossing sheets) never reaches the tracer and a
    midline-trained model and a face-trained one export byte-comparable stores. The normal is the Scharr
    gradient of the EXPORTED field, normalised; because d grows outward it points verso -> recto."""
    d = np.asarray(sd, np.float32)
    if thick is not None:
        d = d - 0.5 * np.asarray(thick, np.float32)
    g = scharr3(d)
    mag = np.linalg.norm(g, axis=0)
    n = g / np.maximum(mag, 1e-4)
    v = np.ones(d.shape, bool) if valid is None else np.asarray(valid, bool)
    return d, n, mag, v


def export_tracer(probs_multi, origin, size, out, rung=2, volume="", umbilicus="", attrs=None,
                  marching_cubes=False, mc_level=0.0, log=print):
    """Write the tracer contract's stores over a box. Returns {name: path}.

    `probs_multi(origin, size) -> {name: (Z, Y, X) float32}` is ONE multi-head pass: `recto` (required),
    and then whichever of `verso`, `sdist` / `midline`, `thickness` and `conf` the checkpoint can serve.
    `marching_cubes` additionally meshes the zero level of the distance field, one shard at a time.
    """
    origin = tuple(int(v) for v in origin)
    size = tuple(int(v) for v in size)
    got_f = probs_multi(origin, size)
    got_f = {str(k): np.asarray(v, np.float32) for k, v in got_f.items()}
    rec = got_f.get("recto")
    assert rec is not None, f"export_tracer: the pass produced {sorted(got_f)}, with no 'recto'"
    dch = next((c for c in ("sdist", "midline") if c in got_f), None)
    assert dch is not None, (f"export_tracer: the pass produced {sorted(got_f)}, with no distance "
                             f"channel; a checkpoint without one cannot serve the tracer contract")
    sd, th, cf, ver = got_f[dch], got_f.get("thickness"), got_f.get("conf"), got_f.get("verso")
    d, n, mag, valid = tracer_fields(sd, th)
    valid = valid & (rec > 0)      # CT == 0 is masked by both sides: the region pass already zeroed it
    os.makedirs(str(out), exist_ok=True)
    got = {}

    def w(name, u8, enc, q=0):
        p = os.path.join(str(out), f"{name}.zarr")
        stores.write(p, u8, origin, rung=int(rung), channels=(name,), q=q, volume=volume,
                     umbilicus=umbilicus,
                     attrs={"encoding": enc, "unit": "voxels_of_this_rung", "no_data": 0,
                            "axis_order": "ZYX", "sign_convention": SIGN_CONVENTION, **(attrs or {})})
        got[name] = p
        return p

    log(f"export-tracer {out}: rung {rung}, one pass for {sorted(got_f)}")
    w("recto", stores.u8(rec), "prob_u8", q=8)
    if ver is not None:
        w("verso", stores.u8(ver), "prob_u8", q=8)
    w("surf_sdist", enc_signed(d, valid), "signed_u8_off128_q0.25")
    for j, nm in enumerate(("nz", "ny", "nx")):
        w(nm, enc_normal(n[j], valid & (mag > 1e-3)), "normal_u8_off128_div127")
    w("gmag", np.clip(np.rint(mag * NORMAL_SCALE), 0, 255).astype(np.uint8), "gradmag_u8_x127")
    if cf is not None:
        w("conf", stores.u8(cf), "conf_u8")
    if th is not None:
        w("thickness", np.clip(np.rint(th / TRACER_UNIT), 0, 255).astype(np.uint8), "unsigned_u8_q0.25")
    if marching_cubes:
        got["mesh"] = mesh_shards(d, valid, origin, os.path.join(str(out), "mesh"), level=mc_level,
                                  log=log)
    log(f"export-tracer {out}: wrote {sorted(got)}")
    return got


def mesh_shards(d, valid, origin, out, level=0.0, shard=stores.SHARD, log=print):
    """Marching cubes on the ZERO LEVEL of a distance field, one shard at a time, one .obj per shard.

    The shard grid is the store's own, and each shard is extracted with a ONE-VOXEL halo on its high
    faces so neighbouring shards' triangles meet: marching cubes places a vertex BETWEEN two samples, so
    without that overlap there is a missing cell between shards. Vertices are written in GLOBAL ZYX
    voxels of this rung, which is the frame the rest of the contract uses (`v z y x` in the .obj, so an
    .obj reader's x is our z -- stated here and in the store attrs rather than silently swapped, because
    a swap is exactly the bug this contract exists to prevent).

    Invalid voxels are pushed to +cap, so the surface never closes over a no-data region."""
    from skimage import measure
    os.makedirs(str(out), exist_ok=True)
    f = np.where(valid, np.asarray(d, np.float32), TRACER_CAP)
    Z, Y, X = f.shape
    paths = []
    for z in range(0, Z, shard):
        for y in range(0, Y, shard):
            for x in range(0, X, shard):
                blk = f[z:min(z + shard + 1, Z), y:min(y + shard + 1, Y), x:min(x + shard + 1, X)]
                if blk.min() > level or blk.max() < level or min(blk.shape) < 2:
                    continue
                try:
                    v, tri, _, _ = measure.marching_cubes(blk, level=float(level))
                except (ValueError, RuntimeError) as e:  # noqa: PERF203
                    log(f"  shard {z},{y},{x}: marching cubes failed ({e!r})")
                    continue
                v = v + np.array([z + origin[0], y + origin[1], x + origin[2]], np.float32)
                p = os.path.join(str(out), f"shard_{z + origin[0]}_{y + origin[1]}_{x + origin[2]}.obj")
                with open(p, "w") as fh:
                    fh.write("# rvsm export-tracer: vertices are GLOBAL ZYX voxels of this rung\n")
                    for q in v:
                        fh.write(f"v {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}\n")
                    for q in tri:
                        fh.write(f"f {q[0] + 1} {q[1] + 1} {q[2] + 1}\n")
                paths.append(p)
                log(f"  shard {z},{y},{x}: {len(v)} vertices, {len(tri)} triangles -> {os.path.basename(p)}")
    return str(out) if paths else None
