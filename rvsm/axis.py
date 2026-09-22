"""The scroll axis (umbilicus), and the radial channels built from it.

The radial unit vector and the normalised-radius plane are two of the student's input channels, so a run
needs an axis for its volume before it can sample anything. rvsm accepts, in order of preference:

- a published umbilicus in either format in circulation: the loader's own
  `{"control_points": [{"z":..., "y":..., "x":...}, ...]}` and the volpkg `umbilicus.txt` ("x, y, z" per
  line, one point per z slice, 1-BASED);
- `auto`: DERIVED from the CT itself, as the centroid of the non-air voxels of each z slice at a coarse
  rung (rung 7, 76.8 um, is a few MB for a whole scroll). The scroll is a roll, so the centroid of its
  cross-section is the umbilicus to within the accuracy the radial channel needs.

UNITS. Control points are RUNG-2 voxels everywhere in rvsm (`axis_at(ax, k)` divides by `2^(k-2)`). A
point read off level 0 of a volume whose level 0 is rung 4 is therefore multiplied by 4 on the way in;
`load(..., ct=...)` does that from the volume's name.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from rvsm import ladder


def parse(text):
    """Published umbilicus text -> [(z, y, x), ...] in the units of the file. Accepts the loader's json
    (an object with `control_points`, or a bare list of triples) and the volpkg `umbilicus.txt`."""
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        j = json.loads(text)
        pts = j["control_points"] if isinstance(j, dict) else j
        if pts and isinstance(pts[0], dict):
            return [(float(p["z"]), float(p["y"]), float(p["x"])) for p in pts]
        return [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    out = []
    for line in text.splitlines():
        v = [q for q in re.split(r"[,\s]+", line.strip()) if q]
        if len(v) >= 3:
            try:
                x, y, z = float(v[0]), float(v[1]), float(v[2])
            except ValueError:
                continue
            out.append((z - 1, y - 1, x - 1))  # umbilicus.txt is 1-based, x first
    if not out:
        raise ValueError("no control points in the umbilicus text")
    return out


def read(path):
    """The control points of an umbilicus file, in the units of the file."""
    with open(path) as f:
        return parse(f.read())


def write(path, pts):
    """[(z, y, x), ...] in rung-2 voxels -> the loader's json, written atomically."""
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = str(path) + ".part"
    with open(tmp, "w") as f:
        json.dump({"control_points": [{"z": float(z), "y": float(y), "x": float(x)} for z, y, x in pts]}, f)
    os.replace(tmp, path)
    return str(path)


def derive(ct, rung=7, thresh=0, step=1):
    """The axis of a scroll from its own CT: per-z centroid of the non-air voxels at `rung`, in RUNG-2
    voxels. A z slice with no papyrus is skipped (the ends of a scan); one control point per `step`
    slices."""
    pyr = ladder.rungs(ct)
    k = int(rung) if rung in pyr else max(r for r in pyr if r <= rung)
    a = ladder.full_level(pyr, k)
    if a is None:
        a = ladder._read(pyr[k], (slice(None),) * 3)
    f = 2.0 ** (k - 2)  # rung-k voxels -> rung-2 voxels; a rung-k cell centre is (i + 0.5) * f - 0.5
    pts = []
    for z in range(0, a.shape[0], step):
        m = a[z] > thresh
        if int(m.sum()) < 16:
            continue
        ys, xs = np.nonzero(m)
        pts.append(tuple((v + 0.5) * f - 0.5 for v in (z, float(ys.mean()), float(xs.mean()))))
    if len(pts) < 2:
        raise RuntimeError(f"{ct}: no non-air slices at rung {k}; cannot derive an axis")
    return pts


def load(spec, ct=None, rung=7):
    """(3, N) control points sorted by z, in RUNG-2 voxels: `spec` is a path to either umbilicus format,
    or "auto"/empty to derive one from `ct`. Points read from a file are in the volume's own level-0
    voxels and are rescaled when level 0 is not rung 2."""
    spec = str(spec or "auto")
    if spec == "auto":
        if not ct:
            raise ValueError("umbilicus='auto' needs a CT volume to derive the axis from")
        pts = derive(ct, rung=rung)
    else:
        if not os.path.exists(spec):
            raise FileNotFoundError(f"no umbilicus at {spec}: every volume needs one (or pass 'auto')")
        f = 2.0 ** (ladder.um_rung(ladder.native_um(ladder.pyramid_base(str(ct)))) - 2) if ct else 1.0
        pts = [(z * f, y * f, x * f) for z, y, x in read(spec)]
    return np.array(sorted(pts), np.float64).T


def axis_at(ax, rung):
    """The control points (given in rung-2 voxels) expressed in rung-k voxels."""
    return np.asarray(ax, np.float64) / (2.0 ** (int(rung) - 2))


def radial(ax, origin, shape):
    """Unit vectors (3,Z,Y,X) pointing away from the scroll axis in the xy plane (z component 0).
    `ax`, `origin` and `shape` are all in the same rung's voxels (see `axis_at`)."""
    shape = tuple(int(v) for v in ladder.shape3(shape))
    z = np.arange(shape[0]) + origin[0]
    cy, cx = np.interp(z, ax[0], ax[1]), np.interp(z, ax[0], ax[2])
    dy = (np.arange(shape[1]) + origin[1])[None, :, None] - cy[:, None, None]
    dx = (np.arange(shape[2]) + origin[2])[None, None, :] - cx[:, None, None]
    n = np.sqrt(dy * dy + dx * dx) + 1e-6
    return np.stack([np.zeros(shape, np.float32), (dy / n).astype(np.float32), (dx / n).astype(np.float32)])


def rmax_vox(ax, shape):
    """The largest radius any voxel of a box of `shape` (corner at the origin) can have from `ax`, in the
    same voxels: the far corner of the cross-section, over the z range of the box.

    This is the r_max the radius plane divides by, so the plane is `r / r_max` in 0..1 for every voxel of
    the scroll and means the same thing for a scroll of any size."""
    Z, Y, X = (int(v) for v in ladder.shape3(shape))
    z = np.arange(Z, dtype=np.float64)
    cy, cx = np.interp(z, ax[0], ax[1]), np.interp(z, ax[0], ax[2])
    dy = np.maximum(np.abs(cy), np.abs(Y - cy))
    dx = np.maximum(np.abs(cx), np.abs(X - cx))
    return float(np.sqrt(dy * dy + dx * dx).max())


def radius(ax, origin, shape, rmax):
    """The (1,Z,Y,X) normalised-radius plane, `clip(r / rmax, 0, 1)`, built from the same axis
    interpolation as `radial` -- so the direction channel and the distance channel can never disagree
    about where the axis is."""
    shape = tuple(int(v) for v in ladder.shape3(shape))
    z = np.arange(shape[0]) + origin[0]
    cy, cx = np.interp(z, ax[0], ax[1]), np.interp(z, ax[0], ax[2])
    dy = (np.arange(shape[1]) + origin[1])[None, :, None] - cy[:, None, None]
    dx = (np.arange(shape[2]) + origin[2])[None, None, :] - cx[:, None, None]
    r = np.sqrt(dy * dy + dx * dx) / max(float(rmax), 1e-6)
    return np.clip(r, 0.0, 1.0).astype(np.float32)[None]


def scale_plane(rung, shape):
    """The constant scale channel of a sample at rung k: (k - 2) / 9, i.e. 0 at 2.4 um and 1 at rung 11."""
    return np.full(tuple(int(v) for v in ladder.shape3(shape)), (int(rung) - 2) / 9.0, np.float32)


def ensure(out, spec, ct=None, rung=7, force=False):
    """Make sure `<out>/umbilicus.json` exists (deriving from the CT when `spec` is "auto"), and return
    its path. Everything downstream reads that one file, in rung-2 voxels."""
    path = os.path.join(str(out), "umbilicus.json")
    if os.path.exists(path) and not force:
        return path
    ax = load(spec, ct=ct, rung=rung)
    return write(path, list(zip(*ax)))
