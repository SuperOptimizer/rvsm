"""3D augmentation, batched on the GPU: `apply` is called once per step on the assembled batch.

Spatial augs (rot/scale/shear/elastic/sheet compression) resample CT + target trilinearly and transform
the three radial unit-vector channels by the same map: a direction transforms as M^-1 d (M is the
out->in affine of `grid_sample`), then it is renormalised. The elastic field deliberately does NOT
rotate the vectors by its local Jacobian: the displacements are small and smooth and the global rotation
part is already covered by `rot`.

Intensity augs touch the IMAGE channels only (`nimg`): the cascade channel, the conditioning planes, the
scale plane and the radial vector are not images, and a brightness shift would move a dropped (zero)
cascade channel off the "no coarse prediction" value the model is taught to read.

The 48 cube symmetries stay in the sampler (`cfg["sym"]`), and so does the raw-uint8 stage that must
precede the z-score (`window`, `volcomp`, `blank`).

SIGMAS ARE IN MICRONS, not voxels. Every PSF-type range below was calibrated at rung 2 (2.4 um);
`for_rung` rewrites them for another rung as sigma_vox(k) = sigma_vox(2) * 2^(2-k), i.e. the same
physical blur at every rung and EXACTLY the calibrated numbers at rung 2. The scan-metadata hook
(`get(name, meta=...)` -> `rvsm.scanmeta.ranges_for`) recentres the physics ops on the scan actually
being trained on, which is what puts the IDENTITY inside the Paganin range.

Four presets only -- `none`, `geo`, `full`, `full2` -- and `full2` is the fixed default of every rvsm
run. The 25 ablation presets of the research pipeline are gone: rvsm has no augmentation flags.
"""
import math

import torch
import torch.nn.functional as F

from rvsm import ladder


def _p(c, lo, hi, n=1):  # per-sample scalars shaped to broadcast over (B,C,Z,Y,X)
    return torch.empty(c.shape[0], n, 1, 1, 1, device=c.device, dtype=c.dtype).uniform_(lo, hi)


def _m(b, dev, p):
    return (torch.rand(b, device=dev) < p).float().view(b, 1, 1, 1, 1)


def _eye(b, dev):
    return torch.eye(3, device=dev).expand(b, 3, 3).clone()


def _rot(b, dev, k, m):
    a = torch.randn(b, 3, device=dev)
    a = a / a.norm(dim=1, keepdim=True)
    th = (torch.empty(b, device=dev).uniform_(-1, 1) * k["max_deg"] * math.pi / 180 * m.view(b)).view(b, 1, 1)
    z = torch.zeros(b, device=dev)
    K = torch.stack([z, -a[:, 2], a[:, 1], a[:, 2], z, -a[:, 0], -a[:, 1], a[:, 0], z], 1).view(b, 3, 3)
    return _eye(b, dev) + th.sin() * K + (1 - th.cos()) * (K @ K)


def _scale(b, dev, k, m):
    s = torch.exp(torch.empty(b, 3, device=dev).uniform_(math.log(k["lo"]), math.log(k["hi"])))
    s = torch.where(torch.rand(b, 1, device=dev) < k["iso"], s[:, :1], s)  # isotropic or per-axis
    return torch.diag_embed(1 / (1 + (s - 1) * m.view(b, 1)))  # out->in is the inverse zoom


def _shear(b, dev, k, m):
    o = torch.empty(b, 3, 3, device=dev).uniform_(-k["max"], k["max"]) * (1 - torch.eye(3, device=dev))
    return _eye(b, dev) + o * m.view(b, 1, 1)


def _elastic(grid, k, m):
    b, S = grid.shape[0], grid.shape[1:4]
    c = max(2, min(S) // k["grid"])
    d = torch.randn(b, 3, c, c, c, device=grid.device)
    d = F.interpolate(d, size=S, mode="trilinear", align_corners=False)
    return grid + (d * (2 * k["sigma"] / min(S)) * m).permute(0, 2, 3, 4, 1)


def warp(x, tg, M, el=None):
    """Resample (B,4,Z,Y,X) input + (B,1,Z,Y,X) target by the out->in matrix M (B,3,3), xyz order."""
    b, dev, S = x.shape[0], x.device, x.shape[2:]
    grid = F.affine_grid(torch.cat([M, torch.zeros(b, 3, 1, device=dev)], 2), (b, 1, *S), align_corners=False)
    if el is not None:
        grid = _elastic(grid, *el)
    g = dict(mode="bilinear", padding_mode="reflection", align_corners=False)
    o, tg = F.grid_sample(x.float(), grid, **g), F.grid_sample(tg.float(), grid, **g)
    ni = o.shape[1] - 3  # image channels first, the radial vector is the last 3
    v = torch.einsum("bij,bjzyx->bizyx", M.inverse(), o[:, [ni + 2, ni + 1, ni]])[:, [2, 1, 0]]  # zyx <-> xyz
    n = v.norm(dim=1, keepdim=True)
    return torch.cat([o[:, :ni], v / n.clamp_min(1e-6) * (n > 1e-3)], 1), tg.clamp(0, 1)


def spatial(x, tg, cfg):
    b, dev = x.shape[0], x.device
    sc = cfg.get("sheetcomp")
    if not any(k in cfg for k in ("rot", "scale", "shear", "elastic")):
        return _sheetcomp(x, tg, sc, _m(b, dev, sc["p"])) if sc else (x, tg)
    M = _eye(b, dev)
    for name, f in (("rot", _rot), ("scale", _scale), ("shear", _shear)):
        if name in cfg:
            M = M @ f(b, dev, cfg[name], _m(b, dev, cfg[name]["p"]))
    el = cfg.get("elastic")
    x, tg = warp(x, tg, M, (el, _m(b, dev, el["p"])) if el else None)
    return _sheetcomp(x, tg, sc, _m(b, dev, sc["p"])) if sc else (x, tg)


def _gamma(c, k):
    lo, hi = c.amin((2, 3, 4), True), c.amax((2, 3, 4), True)
    n = ((c - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    return n.pow(torch.exp(_p(c, -1, 1) * math.log(k["max"]))) * (hi - lo) + lo


def _contrast(c, k):
    m = c.mean((2, 3, 4), True)
    return m + (c - m) * torch.exp(_p(c, -1, 1) * math.log(k["max"]))


def _bright(c, k):
    return c + _p(c, -k["max"], k["max"])


def _noise(c, k):
    return c + torch.randn_like(c) * _p(c, 0, k["max"])


def _mulnoise(c, k):
    return c * (1 + torch.randn_like(c) * _p(c, 0, k["max"]))


def _blur1(c, s):  # separable gaussian; s is one sigma for the whole batch or one per axis
    for d, sd in enumerate((s, s, s) if isinstance(s, float) else s):
        if sd < 0.05:
            continue
        r = max(1, int(2 * sd))
        g = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=c.device, dtype=c.dtype) / sd) ** 2)
        g = g / g.sum()
        pad = [0] * 6
        pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = r
        sh = [1, 1, 1, 1, 1]
        sh[2 + d] = -1
        C = c.shape[1]
        c = F.conv3d(F.pad(c, pad, mode="replicate"), g.view(sh).expand(C, 1, *sh[2:]).contiguous(), groups=C)
    return c


def _blur(c, k):
    return _blur1(c, float(torch.empty(1).uniform_(k["lo"], k["hi"])))


def _sharpen(c, k):
    return c + float(torch.empty(1).uniform_(0, k["max"])) * (c - _blur1(c, k["sigma"]))


def _lu(lo, hi):  # log-uniform: the pitch spread between scans is multiplicative
    return float(torch.empty(1).uniform_(math.log(lo), math.log(hi)).exp())


def _lowres(c, k):
    f = _lu(k["lo"], k["hi"])
    S = c.shape[2:]
    d = F.interpolate(c, size=[max(2, int(s / f)) for s in S], mode="nearest")
    return F.interpolate(d, size=S, mode="trilinear", align_corners=False)


def _bias(c, k):  # low-frequency multiplicative gain (beam hardening / cupping)
    f = torch.randn(c.shape[0], 1, k["grid"], k["grid"], k["grid"], device=c.device, dtype=c.dtype)
    return c * (1 + F.interpolate(f, size=c.shape[2:], mode="trilinear", align_corners=False) * _p(c, 0, k["max"]))


def _clip(c, k):
    m, s = c.mean((2, 3, 4), True), c.std(dim=(2, 3, 4), keepdim=True)
    a = _p(c, k["lo"], k["hi"]) * s
    return torch.maximum(torch.minimum(c, m + a), m - a)


def _boxes(c, k):  # set k["n"] random boxes to the air value
    b, dev, S = c.shape[0], c.device, c.shape[2:]
    hit = torch.zeros_like(c, dtype=torch.bool)
    for _ in range(k["n"]):
        sel = torch.ones(b, 1, 1, 1, 1, dtype=torch.bool, device=dev)
        for d, s in enumerate(S):
            n = torch.randint(max(1, int(k["lo"] * s)), max(2, int(k["hi"] * s)), (b, 1), device=dev)
            o = (torch.rand(b, 1, device=dev) * (s - n)).long()
            i = torch.arange(s, device=dev)
            sh = [b, 1, 1, 1, 1]
            sh[2 + d] = s
            sel = sel & ((i >= o) & (i < o + n)).view(sh)
        hit |= sel
    return torch.where(hit, c.amin((2, 3, 4), True), c)


# --- scan-domain family (ported from tsm's ScanDomainFamily): what differs BETWEEN scans, not generic photometric jitter.  tsm measured
# PHercParis4 vs PHercParis3 at 2.4 um (24 x 256^3 blocks): 8-bit export window [-0.04,0.22] vs
# [-0.03,0.19] f32 (~ +-12% of the window); papyrus std 20.1 vs 27.1 grey at air/papyrus modes
# 42.5/112.5 vs 41.5/114.5 (contrast ratio 3.49 vs 2.70); ACF 1/e length ratios z 0.80 y 0.88
# x 0.99 (13.3/10.9/10.8 vs 10.6/9.6/10.7 vox); PSD log-log slope -4.26 vs -4.39, noise ratio
# 0.67 vs 0.60.  `window` runs on the raw uint8 in the sampler on the raw uint8 -- an affine remap of the
# z-scored CT is a no-op after the re-z-score, only its CLIPPING is real.  The rest run here on the
# z-scored CT: unitless spreads/sigmas are kept as fractions of the per-sample statistics, and
# multiplicative gains multiply the grey level, i.e. they act about the per-sample air value.


def _air(c):
    return c.amin((2, 3, 4), True)


def _aniso_blur(c, k):  # z resolution drawn independently of the in-plane resolution
    sz = float(torch.empty(1).uniform_(k["z_lo"], k["z_hi"]))
    return _blur1(c, (sz,) + (float(torch.empty(1).uniform_(k["yx_lo"], k["yx_hi"])),) * 2)


def _modes(c, k):  # (air, papyrus) modes per sample: 2-means (Lloyd, 10 it) on a strided subsample
    v = c.flatten(1)
    v = v[:, :: max(1, v.shape[1] // k["sub"])]
    a, p = v.amin(1, True), v.amax(1, True)
    for _ in range(10):
        m = (v < (a + p) / 2).float()
        a = (v * m).sum(1, True) / m.sum(1, True).clamp_min(1)
        p = (v * (1 - m)).sum(1, True) / (1 - m).sum(1, True).clamp_min(1)
    return a.view(-1, 1, 1, 1, 1), p.view(-1, 1, 1, 1, 1)


def _class_contrast(c, k):
    """Rescale the spread of the papyrus mode only, smoothstep-blended about the air/papyrus
    midpoint over k["band"] of the mode gap (tsm: +-10 of the 70 grey levels between the modes)."""
    a, mp = _modes(c, k)
    band = ((mp - a) * k["band"]).clamp_min(1e-6)
    w = ((c - (a + mp) / 2 + band) / (2 * band)).clamp(0, 1)
    w = w * w * (3 - 2 * w)
    return c + w * (mp + (c - mp) * _p(c, k["lo"], k["hi"]) - c)


def _spectral(c, k):  # white noise shaped by |k|^(beta/2) in Fourier space; sigma in sample stds
    beta = float(torch.empty(1).uniform_(k["beta_lo"], k["beta_hi"]))
    S = c.shape[2:]
    f = sum(torch.fft.fftfreq(n, device=c.device).pow(2).view([-1 if i == j else 1 for j in range(3)])
            for i, n in enumerate(S)).sqrt()
    f = torch.where(f > 0, f.clamp_min(1e-12) ** (0.5 * beta), torch.zeros_like(f))
    w = torch.fft.ifftn(torch.fft.fftn(torch.randn_like(c), dim=(2, 3, 4)) * f, dim=(2, 3, 4)).real
    w = w / w.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)
    return c + w * _p(c, k["lo"], k["hi"]) * c.std(dim=(2, 3, 4), keepdim=True)


def _gain(c, g):  # a multiplicative detector gain acts on the grey level, i.e. about the air value
    a = _air(c)
    return a + (c - a) * g


def _ring(c, k):  # concentric rings about a centre k["c_*"] crop widths away: gently curved stripes
    b, dev, (_, H, W) = c.shape[0], c.device, c.shape[2:]
    m = float(max(H, W))
    u = lambda lo, hi: torch.empty(b, 1, 1, device=dev, dtype=c.dtype).uniform_(lo, hi)  # noqa: E731
    R, ang = u(k["c_lo"], k["c_hi"]) * m, u(0, 2 * math.pi)
    y = torch.arange(H, device=dev, dtype=c.dtype).view(1, H, 1) - (H - 1) / 2 + R * ang.cos()
    x = torch.arange(W, device=dev, dtype=c.dtype).view(1, 1, W) - (W - 1) / 2 + R * ang.sin()
    d, g = (y * y + x * x).sqrt(), torch.ones(b, H, W, device=dev, dtype=c.dtype)
    for _ in range(k["n"]):
        amp = u(k["a_lo"], k["a_hi"]) * torch.where(u(0, 1) < 0.5, -1.0, 1.0)
        g = g + amp * torch.exp(-0.5 * ((d - R - u(-0.5, 0.5) * m) / u(k["w_lo"], k["w_hi"])) ** 2)
    return _gain(c, g.view(b, 1, 1, H, W))


def _stripe(c, k):  # a detector line: constant y and/or x planes of multiplicative gain
    b, dev, (_, H, W) = c.shape[0], c.device, c.shape[2:]
    g = torch.ones(b, 1, 1, H, W, device=dev, dtype=c.dtype)
    for d, n in ((3, H), (4, W)):
        wd = torch.randint(k["w_lo"], k["w_hi"] + 1, (b, 1), device=dev)
        o = (torch.rand(b, 1, device=dev) * (n - wd)).long()
        i = torch.arange(n, device=dev)
        sh = [b, 1, 1, 1, 1]
        sh[d] = n
        sel = ((i >= o) & (i < o + wd)).view(sh) & (torch.rand(b, 1, 1, 1, 1, device=dev) < 0.5)
        amp = _p(c, k["a_lo"], k["a_hi"]) * torch.where(torch.rand(b, 1, 1, 1, 1, device=dev) < 0.5, -1.0, 1.0)
        g = torch.where(sel, g * (1 + amp), g)
    return _gain(c, g)


def _tone(c, k):
    """Random monotone piecewise-linear remap of the [0,1]-normalized CT (GIN/Bezier-style
    nonlinear intensity remap for cross-scanner transfer), mapped back to the original range."""
    b, n = c.shape[0], k["knots"]
    lo, hi = c.amin((2, 3, 4), True), c.amax((2, 3, 4), True)
    u = ((c - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1).flatten(1)
    v = torch.empty(b, n + 1, device=c.device, dtype=c.dtype).uniform_(1 - k["max"], 1 + k["max"]).cumsum(1)
    v = torch.cat([torch.zeros(b, 1, device=c.device, dtype=c.dtype), v], 1)
    v = v / v[:, -1:]  # monotone knot values (positive increments), equispaced in u
    i = (u * (n + 1)).clamp(0, n + 1 - 1e-4)
    j, t = i.long(), i - i.floor()
    y = torch.gather(v, 1, j) * (1 - t) + torch.gather(v, 1, j + 1) * t
    return y.view_as(c) * (hi - lo) + lo


def _thick(c, k):  # anisotropic low resolution along ONE random axis (thicker slices)
    f = _lu(k["lo"], k["hi"])
    S = list(c.shape[2:])
    T = list(S)
    d = int(torch.randint(3, (1,)))
    T[d] = max(2, int(S[d] / f))
    return F.interpolate(F.interpolate(c, size=T, mode="nearest"), size=S, mode="trilinear", align_corners=False)


def _pool(c, k):  # a coarser voxel grid, by a pooling op: avg/gauss = detector integration (a real coarser
    # pitch), max/min = fat/thin sheets (a morphological close/open), median = a filtered reconstruction,
    # stride = decimation without a prefilter (aliasing). Kernel integer, optionally anisotropic; the
    # way back up is nearest (blocky) or trilinear (smooth).
    op = k["ops"][int(torch.randint(len(k["ops"]), (1,)))]
    r = int(torch.randint(k["k_lo"], k["k_hi"] + 1, (1,)))
    ks = [r, r, r]
    if torch.rand(1) < k["aniso"]:
        ks[int(torch.randint(3, (1,)))] = 1
    S = c.shape[2:]
    pad = [(-n) % q for n, q in zip(S, ks)]
    x = F.pad(c, [0, pad[2], 0, pad[1], 0, pad[0]], mode="replicate") if any(pad) else c
    if op == "avg":
        d = F.avg_pool3d(x, ks, ks)
    elif op == "max":
        d = F.max_pool3d(x, ks, ks)
    elif op == "min":
        d = -F.max_pool3d(-x, ks, ks)
    elif op == "gauss":
        d = _blur1(x, tuple(0.5 * (q - 1) for q in ks))[:, :, ::ks[0], ::ks[1], ::ks[2]]
    elif op == "median":
        d = x.unfold(2, ks[0], ks[0]).unfold(3, ks[1], ks[1]).unfold(4, ks[2], ks[2]).flatten(5).median(-1).values
    else:  # stride
        d = x[:, :, ::ks[0], ::ks[1], ::ks[2]]
    up = "nearest" if torch.rand(1) < k["nearest"] else "trilinear"
    P = [n + p for n, p in zip(S, pad)]
    return F.interpolate(d, size=P, mode=up, **({} if up == "nearest" else {"align_corners": False}))[:, :, :S[0], :S[1], :S[2]]


def _zjit(c, k):  # the z-score itself is uncalibrated: jitter its scale and offset (speculative)
    return c * _p(c, k["s_lo"], k["s_hi"]) + _p(c, -k["b"], k["b"])


def _sheetcomp(x, tg, k, m):
    """villa's SheetCompressionTransform: push the sheets together by compressing the low-intensity
    gaps.  Displacement = cumsum of the gap weight along a random axis, smoothed across it; the
    target and the radial channels ride the same grid (a pure 1D compression, no vector rotation)."""
    b, dev, S = x.shape[0], x.device, x.shape[2:]
    d = int(torch.randint(3, (1,)))
    c = x[:, :1]
    lo, hi = c.amin((2, 3, 4), True), c.amax((2, 3, 4), True)
    gap = 1 - ((c - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)  # 1 = air gap, 0 = sheet
    dis = gap.cumsum(2 + d) * _p(c, k["lo"], k["hi"]) * m
    dis = _blur1(dis, tuple(0.0 if i == d else k["smooth"] for i in range(3)))
    g = F.affine_grid(torch.cat([_eye(b, dev), torch.zeros(b, 3, 1, device=dev)], 2), (b, 1, *S),
                      align_corners=False).clone()
    g[..., 2 - d] = g[..., 2 - d] + dis[:, 0] * (2.0 / S[d])  # out->in: advance faster through air = gaps shrink
    gs = dict(mode="bilinear", padding_mode="border", align_corners=False)
    o, tg = F.grid_sample(x, g, **gs), F.grid_sample(tg, g, **gs)
    ni = o.shape[1] - 3
    n = o[:, ni:].norm(dim=1, keepdim=True)
    return torch.cat([o[:, :ni], o[:, ni:] / n.clamp_min(1e-6) * (n > 1e-3)], 1), tg.clamp(0, 1)


# pipeline order (tsm's, extended): resolution -> sharpen -> class contrast -> photometric ->
# noise -> detector artefacts -> tone -> boxes.  `window`/`volcomp`/`blank` are worker-side (the sampler).
# With cfg["shuffle"] (the `full2` preset) that fixed order is replaced by a per-sample permutation.
# --- ESRF/nabu reconstruction family: what the recon pipeline itself does differently per scan
# (pitch 1.13-45.5 um, 53-137 keV, propagation 0.2/1.2/11 m; uint8 ESRF products vs uint16 legacy).


def _haze(c, k):
    """Documented "decoherence" in dense papyrus: 1-4 smooth blobs (a thresholded low-pass noise
    field, radius k["r_*"] vox) inside which the CT is blurred AND its local contrast shrinks
    toward the local mean, smooth-blended at the blob edge."""
    b, dev, S = c.shape[0], c.device, c.shape[2:]
    g = max(2, int(min(S) / _lu(k["r_lo"], k["r_hi"])))
    f = F.interpolate(torch.randn(b, 1, g, g, g, device=dev, dtype=c.dtype), size=S,
                      mode="trilinear", align_corners=False)
    m = ((f / f.std() - k["thr"]) / k["edge"]).clamp(0, 1)
    m = m * m * (3 - 2 * m)
    lo, mu = _blur1(c, _lu(k["s_lo"], k["s_hi"])), _blur1(c, k["r_lo"] / 4)  # blurred / local mean
    return c + m * (mu + (lo - mu) * _p(c, k["k_lo"], k["k_hi"]) - c)


def _unsharp(c, k):  # nabu's unsharp mask I' = (1+a) I - a Gauss(I, s); its default is a=1, s=1
    return c + _p(c, k["a_lo"], k["a_hi"]) * (c - _blur1(c, _lu(k["s_lo"], k["s_hi"])))


# --- Paganin jitter -------------------------------------------
# nabu's phase step is a Fourier-domain LOW-PASS (Paganin 2002) followed by an unsharp mask, both
# applied to the projections; their net effect on the reconstruction is close to isotropic, so we
# model the pair as one 3D transfer function.  With f the spatial frequency in cycles/um,
#
#     H(f; db) = 1 / (1 + pi * db * lambda * D * f^2)          Paganin, a Lorentzian low-pass
#     U(f; a, s) = 1 + a * (1 - exp(-2 pi^2 s^2 f^2))          nabu's unsharp, I' = (1+a)I - a G_s*I
#
# with db = delta/beta, lambda = 1.2398e-3 / E[keV] um, D = sampleDetectorDistance in um, s the
# unsharp sigma in um.  (Paganin's own form is 1/(1 + db*lambda*D*|k|^2/(4 pi)) for the ANGULAR
# wavenumber k = 2 pi f, which is the same thing.)  A scan is reconstructed once, with its own
# (db0, a0, s0); to ask "what would this cube look like had nabu been run with (db', a', s')" we
# apply the RATIO of the two transfer functions to the reconstructed cube:
#
#     T(f) = H(f; db') U(f; a', s') / (H(f; db0) U(f; a0, s0))
#          = (1 + L0^2 f^2) / (1 + L1^2 f^2) * U(f; a', s') / U(f; a0, s0),   L^2 = pi db lambda D
#
# so the scan's own parameters give T == 1 exactly: the IDENTITY IS IN THE RANGE, which is what makes
# this safe to switch on for a scan whose metadata we have.  Everything enters as L/vox and s/vox, so
# the frequency grid is in cycles/VOXEL and the op is scale-aware for free: a coarser rung has a
# larger voxel, hence a proportionally smaller sigma in voxels, for the same physics.
# T is clamped to [1/gmax, gmax] (a small db' is a deconvolution and would otherwise amplify the noise
# floor without bound) and the result is kept inside the sample's own tone range widened by `keep`.


def _pscalar(c, lo, hi, log=False):  # (B,1,1,1) per-sample scalar, to broadcast over an (Z,Y,X) freq grid
    u = torch.empty(c.shape[0], 1, 1, 1, device=c.device, dtype=torch.float32)
    if log:
        return u.uniform_(math.log(max(lo, 1e-12)), math.log(max(hi, 1e-12))).exp()
    return u.uniform_(lo, hi)


def _vox_um(k, c):  # the sample's voxel pitch in um: a float, or one per sample
    v = k.get("vox_um", RUNG2_UM)
    if isinstance(v, (int, float)):
        return torch.full((c.shape[0], 1, 1, 1), float(v), device=c.device, dtype=torch.float32)
    return torch.as_tensor(v, device=c.device, dtype=torch.float32).reshape(-1, 1, 1, 1).expand(c.shape[0], 1, 1, 1)


def _paganin_jitter(c, k):
    """Re-filter the cube as if nabu's Paganin delta/beta and unsharp (coeff, sigma) had been different.

    One rfftn/irfftn; see the block comment above for the maths.  `k` carries the scan's own parameters
    (`energy_kev`, `dist_mm`, `db`, `a`, `s_um`) and the sampling ranges (`db_lo/db_hi` log-uniform,
    `a_lo/a_hi` uniform, `s_lo/s_hi` log-uniform, sigmas in MICRONS); `vox_um` (set by `for_rung`) is the
    sample's voxel pitch."""
    S = c.shape[2:]
    vox = _vox_um(k, c)
    f2 = sum(torch.fft.fftfreq(n, device=c.device, dtype=torch.float32).pow(2)
             .view([-1 if i == j else 1 for j in range(3)])
             for i, n in enumerate(S[:2]))  # cycles^2 / voxel^2, the two full-length axes
    f2 = f2 + torch.fft.rfftfreq(S[2], device=c.device, dtype=torch.float32).pow(2).view(1, 1, -1)
    lam = 1.2398e-3 / max(float(k.get("energy_kev", 78.0)), 1e-6)          # um
    L2 = math.pi * lam * float(k.get("dist_mm", 220.0)) * 1e3 / vox.pow(2)  # L^2 per unit delta/beta, vox^2
    db1 = _pscalar(c, k["db_lo"], k["db_hi"], log=True)
    a1, s1 = _pscalar(c, k["a_lo"], k["a_hi"]), _pscalar(c, k["s_lo"], k["s_hi"], log=True)
    db0 = float(k.get("db", 1000.0))
    u = lambda a, s: 1 + a * (1 - torch.exp(-2 * math.pi ** 2 * (s / vox).pow(2) * f2))  # noqa: E731
    T = (1 + L2 * db0 * f2) / (1 + L2 * db1 * f2) * u(a1, s1) / u(float(k.get("a", 4.0)),
                                                                 float(k.get("s_um", 2.88)))
    g = float(k.get("gmax", 4.0))
    y = torch.fft.irfftn(torch.fft.rfftn(c.float(), dim=(2, 3, 4)) * T.clamp(1 / g, g).unsqueeze(1),
                         s=S, dim=(2, 3, 4)).to(c.dtype)
    lo, hi = c.amin((2, 3, 4), True), c.amax((2, 3, 4), True)
    m = (hi - lo) * float(k.get("keep", 0.25))  # the filter overshoots at edges; keep the tone range sane
    return y.clamp(min=lo - m).clamp(max=hi + m)


def _quant(c, k):
    """nabu's histogram window + 8-bit cast, with the per-scan window unknown: clip to a pair of
    the sample's own percentiles and round to 256 levels."""
    v = c.flatten(1).sort(1).values
    n = v.shape[1]
    q = lambda f: v.gather(1, (f.view(-1, 1) * (n - 1)).long()).view(-1, 1, 1, 1, 1)  # noqa: E731
    lo, hi = q(_p(c, k["lo_lo"], k["lo_hi"])), q(_p(c, k["hi_lo"], k["hi_hi"]))
    u = ((c - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    return (u * 255).round() / 255 * (hi - lo) + lo


def _cor(x, k, m):
    """Centre-of-rotation ghosting: a faint copy shifted along the in-plane radial direction (the
    radial channels give it), with an amplitude that grows with the in-plane distance from the
    patch centre and vanishes on it."""
    b, dev, S = x.shape[0], x.device, x.shape[2:]
    ni = x.shape[1] - 3
    c, d = x[:, :ni], x[:, ni:]
    o = torch.stack([d[:, 2] / S[2], d[:, 1] / S[1], d[:, 0] / S[0]], -1) * 2 * (_p(c, k["lo"], k["hi"]) * m)
    g = F.affine_grid(torch.cat([_eye(b, dev), torch.zeros(b, 3, 1, device=dev)], 2), (b, 1, *S),
                      align_corners=False)
    gh = F.grid_sample(c, g + o, mode="bilinear", padding_mode="border", align_corners=False)
    ax = [(torch.arange(n, device=dev, dtype=c.dtype) - (n - 1) / 2) for n in S[1:]]
    r = ((ax[0].view(1, 1, -1, 1) ** 2 + ax[1].view(1, 1, 1, -1) ** 2).sqrt() / (max(S[1:]) / 2)).clamp(0, 1)
    a = _p(c, k["a_lo"], k["a_hi"]) * m * r
    return torch.cat([c * (1 - a) + gh * a, d], 1)


INTENS = [("bias", _bias), ("lowres", _lowres), ("thick", _thick), ("pool", _pool), ("blur", _blur),
          ("aniso_blur", _aniso_blur), ("haze", _haze), ("sharpen", _sharpen),
          ("paganin", _paganin_jitter), ("unsharp", _unsharp),
          ("class_contrast", _class_contrast), ("gamma", _gamma), ("contrast", _contrast),
          ("bright", _bright), ("noise", _noise), ("mulnoise", _mulnoise), ("spectral_noise", _spectral),
          ("ring", _ring), ("stripe", _stripe), ("tone", _tone), ("quant", _quant), ("zjit", _zjit),
          ("clip", _clip), ("airbox", _boxes), ("cutout", _boxes)]


# --- sigmas in microns, not voxels ----------------------------
# The PSF-type ops below take a Gaussian sigma.  Their ranges were calibrated on 2.4 um data (rung 2),
# so their numbers ARE sigma_um / 2.4; `for_rung` rewrites them for another rung as
# sigma_vox(k) = sigma_um / rung_um(k) = sigma_vox(rung 2) * 2.4 / rung_um(k) = sigma_vox(2) * 2^(2-k),
# i.e. the same physical blur at every rung, and EXACTLY today's numbers at rung 2.
# Not converted: `ring`/`stripe` widths (a detector line is a fixed number of DETECTOR pixels, not a
# fixed length in the sample), `elastic`/`sheetcomp` smoothing (geometry, not a PSF), `haze`'s r_*
# (a blob count divisor, not a length), and `paganin`'s s_lo/s_hi (already in microns).
RUNG2_UM = ladder.rung_um(2)  # 2.4 um; rung k is 0.6 * 2^k um
SIGMA_KEYS = {"blur": ("lo", "hi"), "sharpen": ("sigma",), "unsharp": ("s_lo", "s_hi"),
              "aniso_blur": ("z_lo", "z_hi", "yx_lo", "yx_hi"), "haze": ("s_lo", "s_hi")}
# SinoSynth (arXiv:2409.18355) randomises the COMPOSITION ORDER of its degradation chain per sample,
# not just each step's occurrence: a fixed order lets the network learn order-specific correlations
# that no real acquisition chain guarantees.  `cfg["shuffle"]` turns that on for the intensity list.


def for_rung(cfg, rung=2):
    """`cfg` with the voxel-defined sigmas rescaled from rung 2 to `rung` (a scalar, or one per sample).

    A per-sample `rung` scales the scalar-sigma ops by the batch's median rung (they draw one sigma for
    the whole batch anyway) and gives `paganin` the true per-sample pitch, which it applies exactly."""
    ks = [float(rung)] if isinstance(rung, (int, float)) else [float(v) for v in rung]
    kmed = sorted(ks)[len(ks) // 2]
    out, f = dict(cfg), RUNG2_UM / ladder.rung_um(kmed)
    for name, keys in SIGMA_KEYS.items():
        k = cfg.get(name)
        if k:
            out[name] = {**k, **{q: k[q] * f for q in keys if q in k}}
    if cfg.get("paganin"):
        out["paganin"] = {**cfg["paganin"], "vox_um": ladder.rung_um(ks[0]) if len(ks) == 1 else [ladder.rung_um(v) for v in ks]}
    return out


def _gate(c, m, f, k):
    """Apply op `f` to the samples `m` (a CPU bool (B,)) selects. The whole batch selected is just
    `f(c)`; a partial selection blends with `where`. `m` lives on the HOST on purpose: deciding
    whether to run an op from a device tensor is a device->host sync per op per slot (576 of them
    per batch for the shuffled 24-op chain), and on a GPU behind an RPC link (Thunder) those syncs
    were ~300 ms of a ~440 ms augmentation pass."""
    if not bool(m.any()):
        return c
    if bool(m.all()):
        return f(c, k)
    return torch.where(m.to(c.device).view(-1, 1, 1, 1, 1), f(c, k), c)


def intensity(c, cfg):
    ops = [(n, f, cfg[n]) for n, f in INTENS if cfg.get(n)]
    b = c.shape[0]
    if not cfg.get("shuffle"):
        for name, f, k in ops:   # an aug that selected no sample of this batch costs nothing
            c = _gate(c, (torch.rand(b) < k["p"]), f, k)
        return c
    n = len(ops)
    sel = torch.stack([(torch.rand(b) < k["p"]) for _, _, k in ops], 1)  # (B, n), on the host
    slot = torch.rand(b, n).argsort(1)  # slot[b, t] = the op sample b applies t-th
    for t in range(n):
        for i, (name, f, k) in enumerate(ops):
            c = _gate(c, sel[:, i] & (slot[:, t] == i), f, k)   # the samples that put op i in slot t
    return c


def apply(x, tg, cfg, nimg=None, rung=None):
    """(B,C,Z,Y,X) input (image channels, then the 3 radial-vector channels) + (B,T,Z,Y,X) target -> augmented
    pair (float32). Intensity augs act on every image channel with the same per-sample parameters.

    `nimg`: how many leading channels are IMAGE channels. The default (C - 3, everything but the radial
    vector) is what the 13/14-channel stacks always used. A cascade run passes the cube count, so the
    intensity augs leave the CASCADE channel and the scale plane alone: they are not images, and a
    brightness shift would move a dropped (zero) cascade channel off the "no coarse prediction" value the
    model is taught to read. Spatial augs always act on every channel.

    `rung`: the ladder rung the batch was sampled at (an int, or one per sample), which converts the
    PSF-type sigmas from microns to voxels (`for_rung`) and gives `paganin` its voxel pitch. The default
    is rung 2 (2.4 um), where the conversion is the identity and the effective ranges are exactly today's."""
    if not cfg:
        return x, tg
    if rung is not None:
        cfg = for_rung(cfg, rung)
    ni = x.shape[1] - 3
    nim = ni if nimg is None else int(nimg)
    x, tg = spatial(x.float(), tg.float(), cfg)
    if any(cfg.get(name) for name, _ in INTENS):  # no intensity aug configured (e.g. "geo"): the cat below
        x = torch.cat([intensity(x[:, :nim], cfg), x[:, nim:]], 1)  # would only copy the whole batch
    if cfg.get("cor"):  # needs the radial channels for the shift direction, so not in `intensity`
        x = _cor(x, cfg["cor"], _m(x.shape[0], x.device, cfg["cor"]["p"]))
    if cfg.get("norad"):
        x[:, ni:] = 0
    return x, tg


SPATIAL = {"rot": {"p": 0.5, "max_deg": 30}, "scale": {"p": 0.3, "lo": 0.8, "hi": 1.25, "iso": 0.5},
           "shear": {"p": 0.2, "max": 0.1}, "elastic": {"p": 0.3, "sigma": 4.0, "grid": 12}}
INTENSITY = {"gamma": {"p": 0.3, "max": 1.8}, "contrast": {"p": 0.3, "max": 1.5},
             "bright": {"p": 0.3, "max": 0.3}, "noise": {"p": 0.3, "max": 0.2},
             "mulnoise": {"p": 0.2, "max": 0.15}, "blur": {"p": 0.2, "lo": 0.5, "hi": 1.5},
             "sharpen": {"p": 0.2, "max": 1.0, "sigma": 1.0}, "lowres": {"p": 0.25, "lo": 1.5, "hi": 4.0},
             "bias": {"p": 0.3, "max": 0.3, "grid": 4}, "clip": {"p": 0.15, "lo": 1.5, "hi": 3.0},
             "airbox": {"p": 0.15, "n": 1, "lo": 0.1, "hi": 0.4}}
CUTOUT = {"cutout": {"p": 0.25, "n": 3, "lo": 0.05, "hi": 0.25}}
SPATIAL = {"rot": {"p": 0.5, "max_deg": 30}, "scale": {"p": 0.3, "lo": 0.8, "hi": 1.25, "iso": 0.5},
           "shear": {"p": 0.2, "max": 0.1}, "elastic": {"p": 0.3, "sigma": 4.0, "grid": 12}}
INTENSITY = {"gamma": {"p": 0.3, "max": 1.8}, "contrast": {"p": 0.3, "max": 1.5},
             "bright": {"p": 0.3, "max": 0.3}, "noise": {"p": 0.3, "max": 0.2},
             "mulnoise": {"p": 0.2, "max": 0.15}, "blur": {"p": 0.2, "lo": 0.5, "hi": 1.5},
             "sharpen": {"p": 0.2, "max": 1.0, "sigma": 1.0}, "lowres": {"p": 0.25, "lo": 1.5, "hi": 4.0},
             "bias": {"p": 0.3, "max": 0.3, "grid": 4}, "clip": {"p": 0.15, "lo": 1.5, "hi": 3.0},
             "airbox": {"p": 0.15, "n": 1, "lo": 0.1, "hi": 0.4}}
CUTOUT = {"cutout": {"p": 0.25, "n": 3, "lo": 0.05, "hi": 0.25}}
# scan-domain defaults: calibrated on PHercParis4 vs PHerc1667 (see dev_scan_stats in the commit
# message / tests); ranges are wide enough that augmented Paris4 covers the 1667 statistics.
SCAN = {"window": {"p": 0.5, "lo_lo": -12.0, "lo_hi": 12.0, "hi_lo": 200.0, "hi_hi": 300.0},
        "class_contrast": {"p": 0.6, "lo": 0.5, "hi": 2.0, "band": 0.14, "sub": 4096},
        "aniso_blur": {"p": 0.3, "z_lo": 0.0, "z_hi": 1.5, "yx_lo": 0.0, "yx_hi": 1.5},
        "spectral_noise": {"p": 0.3, "lo": 0.02, "hi": 0.25, "beta_lo": -1.0, "beta_hi": 1.0},
        "ring": {"p": 0.2, "n": 2, "a_lo": 0.02, "a_hi": 0.08, "w_lo": 2.0, "w_hi": 8.0,
                 "c_lo": 4.0, "c_hi": 20.0},
        "stripe": {"p": 0.05, "a_lo": 0.01, "a_hi": 0.05, "w_lo": 1, "w_hi": 3}}
TONE = {"tone": {"p": 0.3, "knots": 4, "max": 0.7}}
THICK = {"thick": {"p": 0.3, "lo": 1.0, "hi": 4.0}, "lowres": {"p": 0.25, "lo": 1.0, "hi": 4.0}}
POOL_OPS = ["avg", "max", "min", "median", "gauss", "stride"]
POOL = {"pool": {"p": 0.3, "ops": POOL_OPS, "k_lo": 2, "k_hi": 4, "aniso": 0.3, "nearest": 0.5}}
VOLCOMP = {"volcomp": {"p": 0.3, "q": [4.0, 12.0]}}  # the real codec residual (worker, CPU)
BLANK = {"blank": {"p": 0.03}}  # an all-air patch with target 0 (the sampler)
ZJIT = {"zjit": {"p": 0.3, "s_lo": 0.9, "s_hi": 1.1, "b": 0.1}}
SHEETCOMP = {"sheetcomp": {"p": 0.3, "lo": 0.1, "hi": 0.3, "smooth": 5.0}}
HAZE = {"haze": {"p": 0.3, "r_lo": 10.0, "r_hi": 40.0, "thr": 0.6, "edge": 0.8,
                 "s_lo": 1.0, "s_hi": 3.0, "k_lo": 0.3, "k_hi": 0.7}}
UNSHARP = {"unsharp": {"p": 0.3, "a_lo": 0.0, "a_hi": 1.5, "s_lo": 0.5, "s_hi": 3.0}}
QUANT = {"quant": {"p": 0.5, "lo_lo": 0.005, "lo_hi": 0.05, "hi_lo": 0.95, "hi_hi": 0.995}}
COR = {"cor": {"p": 0.3, "lo": 0.3, "hi": 1.5, "a_lo": 0.05, "a_hi": 0.15}}
# Paganin jitter: the scan's own parameters are the 2.4 um 78 keV PHerc-Paris4 B_HA defaults
# (rvsm.scanmeta.DEFAULTS), so the identity sits inside the range; the sampled ranges are the corpus span
# (delta/beta 500-1000, unsharp 4.0 at 2.75-2.88 um) widened 2x each way (rvsm.scanmeta.ranges_for, which
# recentres all of this on the run's own scan).  s_lo/s_hi are MICRONS.
PAGANIN = {"paganin": {"p": 0.3, "energy_kev": 78.0, "dist_mm": 220.0, "db": 1000.0, "a": 4.0, "s_um": 2.88,
                       "db_lo": 250.0, "db_hi": 2000.0, "a_lo": 2.0, "a_hi": 8.0,
                       "s_lo": 1.375, "s_hi": 5.76, "gmax": 4.0, "keep": 0.25}}
SHUFFLE = {"shuffle": True}  # SinoSynth-style per-sample composition order for the intensity ops


def _pre(*ds):
    return {"sym": True, **{k: v for d in ds for k, v in d.items()}}


# `full`  = every op that was in production before the physics-augmentation work.
# `full2` = `full` + the Paganin delta/beta + unsharp jitter and the per-sample composition order,
#            i.e. the two ops whose ranges are set by the scan's own metadata. It is the default.
PRESETS = {
    "none": {"sym": False},
    "geo": _pre(),                      # the 48 cube symmetries only
    "full": _pre(SPATIAL, INTENSITY, CUTOUT, SCAN, TONE, THICK, POOL, VOLCOMP, BLANK, ZJIT, SHEETCOMP),
    "full2": _pre(SPATIAL, INTENSITY, CUTOUT, SCAN, TONE, THICK, POOL, VOLCOMP, BLANK, ZJIT, SHEETCOMP,
                  PAGANIN, SHUFFLE),
}


def get(name, meta=None, rung=None):
    """The preset `name`, optionally recentred on one scan's metadata and one rung.

    `meta`: a `rvsm.scanmeta.load()` dict. Its `ranges_for` overrides are merged PER OP into the ops the
    preset already configures, so a preset without `paganin` does not grow one and `full` keeps its
    shape. `rung`: see `apply` / `for_rung`; the ranges are calibrated at rung 2 and converted from
    microns for anything else."""
    if name not in PRESETS:
        raise KeyError(f"unknown augmentation preset {name!r}; rvsm has {sorted(PRESETS)}")
    cfg = PRESETS[name]
    if meta is not None:
        from rvsm import scanmeta
        r = scanmeta.ranges_for(meta)
        cfg = {k: ({**v, **r[k]} if k in r and isinstance(v, dict) else v) for k, v in cfg.items()}
    return cfg if rung is None else for_rung(cfg, rung)
