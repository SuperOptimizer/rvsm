"""The legacy 7.91 um PHerc Paris 4 frame <-> the fine 2.4 um volume (20260411134726), and a visual check.

The fine volume's `transform.json` (S3 only; the volcomp mirror does not carry it) holds a 3x4 affine
`M = [A | b]` fitted to 24 landmark pairs. It maps FINE (moving) voxel coordinates, (x, y, z) order,
to LEGACY (fixed, `PHercParis4-20230205180739_masked`) voxel coordinates:

    legacy_xyz = A @ fine_xyz + b          fine_xyz = A^-1 @ (legacy_xyz - b)

It is NOT a pure scale + offset: A is a similarity with a reflection (det < 0) and a -141 degree turn about
z; its singular values are 0.3066-0.3073 (fine voxel = 1/3.257 legacy voxel, vs the nominal 2.4/7.91 =
1/3.296), with a slight tilt of the z axis (~0.9 deg). Landmark residual: 1.9 legacy voxels mean
(~6 fine voxels), 5.8 max. The fine volume's zarr axes are (z, y, x), so a fine point indexes as
ct[z, y, x] of the transformed (x, y, z).

tifxyz (VC3D): x.tif / y.tif / z.tif float32 grids, -1 = invalid; the published meshes come in the legacy
frame (`<id>-on-20230205180739-7.91um.tifxyz`) and, for re-published segments, already in the fine
frame (`<id>-on-20260411134726-2.4um.tifxyz`).

    python -m rvsm.tools.refine.frame --seg /home/forrest/refine/paths/20231016151002 --z 60000 \\
        --png /home/forrest/refine/check_frame.png
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

import numpy as np

FINE_VOL = "20260411134726-2.400um-0.2m-78keV-masked"
TRANSFORM_URL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/"
                 f"{FINE_VOL}.zarr/transform.json")
CT_URL = ("https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/volumes/"
          f"{FINE_VOL}.zarr")


class Frame:
    """legacy (x, y, z at 7.91 um) <-> fine (x, y, z at 2.4 um) via the published affine."""

    def __init__(self, transform=TRANSFORM_URL):
        if "://" in str(transform):
            with urllib.request.urlopen(transform, timeout=30) as f:   # noqa: S310
                t = json.loads(f.read())
        else:
            t = json.load(open(transform))
        self.raw = t
        M = np.asarray(t["transformation_matrix"], np.float64)
        self.A, self.b = M[:, :3], M[:, 3]
        self.Ai = np.linalg.inv(self.A)

    def to_legacy(self, fine_xyz):
        return np.asarray(fine_xyz, np.float64) @ self.A.T + self.b

    def to_fine(self, legacy_xyz):
        return (np.asarray(legacy_xyz, np.float64) - self.b) @ self.Ai.T

    def residuals(self):
        F, Mv = (np.asarray(self.raw[k], np.float64) for k in ("fixed_landmarks", "moving_landmarks"))
        return (np.linalg.norm(self.to_legacy(Mv) - F, axis=1),
                np.linalg.norm(self.to_fine(F) - Mv, axis=1))

    def describe(self):
        U, S, Vt = np.linalg.svd(self.A)
        R = U @ Vt
        rl, rf = self.residuals()
        return {"det": float(np.linalg.det(self.A)), "singular": S.tolist(),
                "fine_per_legacy": (1 / S).tolist(),
                "rot_xy_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                "z_tilt_deg": float(np.degrees(np.arccos(abs(R[2, 2])))),
                "reflection": bool(np.linalg.det(R) < 0),
                "legacy_to_fine_A": self.Ai.tolist(), "legacy_to_fine_t": (-self.Ai @ self.b).tolist(),
                "resid_legacy_mean": float(rl.mean()), "resid_legacy_max": float(rl.max()),
                "resid_fine_mean": float(rf.mean()), "resid_fine_max": float(rf.max())}


def read_tifxyz(d):
    """(H, W, 3) float32 xyz grid of a tifxyz dir, NaN where invalid."""
    import tifffile
    g = np.stack([tifffile.imread(os.path.join(d, n + ".tif")).astype(np.float32) for n in "xyz"], -1)
    g[(g <= 0).any(-1)] = np.nan
    return g


def fine_grid(seg_dir, frame=None):
    """The segment's mesh in fine xyz: the published fine-frame tifxyz if present, else the legacy one
    transformed. Returns (grid, source)."""
    names = sorted(os.listdir(seg_dir))
    fine = [n for n in names if n.endswith(".tifxyz") and "20260411134726" in n]
    if fine and all(os.path.exists(os.path.join(seg_dir, fine[0], n + ".tif")) for n in "xyz"):
        return read_tifxyz(os.path.join(seg_dir, fine[0])), "published fine"
    leg = [n for n in names if n.endswith(".tifxyz") and "20230205180739" in n]
    g = read_tifxyz(os.path.join(seg_dir, leg[0]))
    f = frame or Frame()
    return f.to_fine(g.reshape(-1, 3)).reshape(g.shape).astype(np.float32), "legacy transformed"


def z_crossings(g, z0):
    """Points (x, y) where the mesh's grid edges cross z = z0 (linear interpolation along u and v edges):
    the mesh's cross-section with that plane."""
    out = []
    for a, b in ((g[:, :-1], g[:, 1:]), (g[:-1], g[1:])):
        za, zb = a[..., 2], b[..., 2]
        m = np.isfinite(za) & np.isfinite(zb) & ((za - z0) * (zb - z0) <= 0) & (za != zb)
        t = ((z0 - za[m]) / (zb[m] - za[m]))[:, None]
        out.append(a[m][:, :2] * (1 - t) + b[m][:, :2] * t)
    return np.concatenate(out) if out else np.zeros((0, 2))


def z_segments(g, z0):
    """Marching squares on the mesh grid: (N, 2, 2) line segments ((x, y), (x, y)) where each grid cell
    crosses z = z0. Only crossings inside the SAME cell are joined, so adjacent wraps never connect."""
    P = [g[:-1, :-1], g[:-1, 1:], g[1:, 1:], g[1:, :-1]]          # cell corners, in ring order
    pts, ok = [], []
    for a, b in zip(P, P[1:] + P[:1]):
        za, zb = a[..., 2], b[..., 2]
        m = np.isfinite(za) & np.isfinite(zb) & ((za - z0) * (zb - z0) < 0)
        t = np.where(m, (z0 - za) / np.where(zb != za, zb - za, 1), 0)[..., None]
        pts.append(a[..., :2] * (1 - t) + b[..., :2] * t)
        ok.append(m)
    pts, ok = np.stack(pts, -2), np.stack(ok, -1)                  # (H-1, W-1, 4, 2), (H-1, W-1, 4)
    two = ok.sum(-1) == 2
    idx = np.argsort(~ok[two], axis=-1, kind="stable")[:, :2]
    sel = pts[two]
    return np.take_along_axis(sel, idx[..., None], 1)


def z_range(g):
    z = g[..., 2]
    return float(np.nanmin(z)), float(np.nanmax(z))


def check(seg_dir, z0, png, ct=CT_URL, zoom=1024, frame=None):
    """Draw the segment's cross-section at fine z = z0 on the CT: legacy mesh transformed (red), the
    published fine mesh (green, when present), at rung 4 over the segment's footprint and at rung 2 in a
    `zoom`^2 crop. Returns a dict of numbers, including the CT contrast AT the transformed contour vs
    the same points shifted off it (a mesh on a sheet sits on bright papyrus)."""
    from PIL import Image, ImageDraw
    from rvsm import ladder
    f = frame or Frame()
    names = sorted(os.listdir(seg_dir))
    leg = [n for n in names if n.endswith(".tifxyz") and "20230205180739" in n][0]
    gl = read_tifxyz(os.path.join(seg_dir, leg))
    gt = f.to_fine(gl.reshape(-1, 3)).reshape(gl.shape)
    fin = [n for n in names if n.endswith(".tifxyz") and "20260411134726" in n]
    gp = read_tifxyz(os.path.join(seg_dir, fin[0])) if fin and os.path.exists(
        os.path.join(seg_dir, fin[0], "z.tif")) else None
    ct_t = z_crossings(gt, z0)
    ct_p = z_crossings(gp, z0) if gp is not None else np.zeros((0, 2))
    assert len(ct_t), f"the mesh does not cross fine z={z0} (fine z range {z_range(gt)})"
    pyr = ladder.rungs(ct)
    # ---- overview at rung 4 (9.6 um) over the contour's footprint
    k, s = 4, 4
    lo = np.floor(ct_t.min(0) - 400).astype(int)
    hi = np.ceil(ct_t.max(0) + 400).astype(int)
    oy, ox = lo[1] // s, lo[0] // s
    ov = ladder.read_rung(pyr, k, (z0 // s, oy, ox), (1, (hi[1] - lo[1]) // s, (hi[0] - lo[0]) // s),
                          dtype=np.uint8)[0]
    # ---- zoom at rung 2 around the contour point nearest the footprint centre
    c = ct_t[np.argmin(np.linalg.norm(ct_t - ct_t.mean(0), axis=1))]
    zy, zx = int(c[1]) - zoom // 2, int(c[0]) - zoom // 2
    zm = ladder.read_rung(pyr, 2, (int(z0), zy, zx), (1, zoom, zoom), dtype=np.uint8)[0]

    def contrast(pts, img, oy, ox, sc):
        """mean CT at the points, and at the points shifted by d rung-2 voxels in 8 directions."""
        res = {}
        for d in (0, 8, 16, 32):
            vals = []
            for ang in (np.arange(8) * np.pi / 4 if d else [0.0]):
                q = pts + d * np.array([np.cos(ang), np.sin(ang)])
                iy = ((q[:, 1] - oy) / sc).astype(int)
                ix = ((q[:, 0] - ox) / sc).astype(int)
                m = (iy >= 0) & (iy < img.shape[0]) & (ix >= 0) & (ix < img.shape[1])
                vals.append(img[iy[m], ix[m]].astype(float))
            v = np.concatenate(vals)
            res[d] = float(v.mean()) if v.size else float("nan")
        return res

    sg = z_segments(gt, z0)
    tt = np.linspace(0, 1, 32)[None, :, None]
    dt = (sg[:, :1] * (1 - tt) + sg[:, 1:] * tt).reshape(-1, 2)
    stats = {"z0": int(z0), "n_cross_transformed": int(len(ct_t)), "n_cross_published": int(len(ct_p)),
             "contrast_zoom_r2": contrast(dt, zm, zy, zx, 1.0)}
    if len(ct_p):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(ct_p).query(ct_t)
        stats["transformed_vs_published_dist_fine_vox"] = {
            "median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "max": float(d.max())}

    def draw(img, seg_list, oy, ox, sc, wd):
        im = Image.fromarray(img).convert("RGB")
        dr = ImageDraw.Draw(im)
        for segs, col in seg_list:
            for (x0, y0), (x1, y1) in segs:
                dr.line([((x0 - ox) / sc, (y0 - oy) / sc), ((x1 - ox) / sc, (y1 - oy) / sc)], fill=col,
                        width=wd)
        return im
    st_ = z_segments(gt, z0)
    sp_ = z_segments(gp, z0) if gp is not None else np.zeros((0, 2, 2))
    A = draw(ov, [(sp_, (0, 255, 0)), (st_, (255, 0, 0))], lo[1], lo[0], s, 1)
    B = draw(zm, [(sp_, (0, 255, 0)), (st_, (255, 0, 0))], zy, zx, 1.0, 2)
    h = max(A.height, B.height)
    W = Image.new("RGB", (A.width + B.width + 10, h + 40), (255, 255, 255))
    W.paste(A, (0, 40))
    W.paste(B, (A.width + 10, 40))
    ImageDraw.Draw(W).text((5, 5), f"{os.path.basename(seg_dir)} at fine z={z0}: red = legacy 7.91um mesh "
                           f"through transform.json, green = published 2.4um tifxyz. left rung 4 "
                           f"(9.6um) footprint, right rung 2 zoom {zoom}^2 at y={zy} x={zx}", fill=(0, 0, 0))
    W.save(png)
    stats["png"] = png
    stats["zoom_origin_zyx"] = [int(z0), zy, zx]
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--z", type=int, required=True, help="fine z of the cross-section")
    ap.add_argument("--png", default="/home/forrest/refine/check_frame.png")
    ap.add_argument("--transform", default=TRANSFORM_URL)
    ap.add_argument("--ct", default=CT_URL)
    a = ap.parse_args(argv)
    f = Frame(a.transform)
    print(json.dumps(f.describe(), indent=1))
    print(json.dumps(check(a.seg, a.z, a.png, ct=a.ct, frame=f), indent=1))


if __name__ == "__main__":
    main()
