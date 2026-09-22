"""Compact uint8 loader samples -> the model input, built on the GPU.

The loader worker only READS (`sample.rung_item`): the CT cube and the nine context cubes stay uint8, the
target and the weight stay uint8 (255 = 1.0), and every per-voxel float -- the z-score of each cube, the
radius plane, the scale plane, the radial unit vector and the cube symmetry -- happens here, on the
device, once the batch has arrived. At 256^3 that turns ~1 GB of float32 per sample in the worker into
~200 MB of uint8 and moves ~13 s of CPU per sample onto the card.

The channel ORDER is `config.Layout`'s and nothing else's:

    [CT, ctx_1..ctx_n, CASCADE, radius, meta_0..meta_4, scale, rz, ry, rx]

`prepare` fills that stack in place, in that order, and asserts the result against the layout it was
given -- so a stem change is one edit in config.py and a loud failure everywhere else.

`autocast` lives here rather than in train.py: prep is what the Cascade's self-prediction forward needs,
and defining it here is what breaks usrm2's prep -> train import cycle.
"""
import contextlib

import time

import torch

from rvsm import ladder, model as M
from rvsm.sample import sym_decode

CASCADE_MODES = ("off", "mask", "self", "mix")


def autocast(dev):
    """bf16 autocast on cuda, a no-op anywhere else. Every forward in rvsm goes through this."""
    dev = torch.device(dev) if not isinstance(dev, torch.device) else dev
    return torch.autocast("cuda", torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()


def sym_apply_t(sym, x, tg):
    """`sample.sym_apply` on tensors: the cube symmetry `sym` (0..47) applied to a (B,C,Z,Y,X) input and
    a (B,T,Z,Y,X) target. The LAST 3 channels of x are the radial VECTOR and are permuted and negated
    with the axes exactly as the CPU path does; every other channel -- the image cubes, the cascade
    channel, the planes and the scale plane -- is a plain spatial field and is only permuted and
    flipped. The channel count is read off the tensor, so any stem width works."""
    perm, flip = sym_decode(sym)
    d = x.dim() - 3  # 2 with a batch dimension, 1 without
    ax = tuple(range(d)) + tuple(d + int(q) for q in perm)
    x, tg = x.permute(ax), tg.permute(ax)
    dims = [d + i for i, f in enumerate(flip) if f]
    if dims:
        x, tg = x.flip(dims), tg.flip(dims)
    ni = x.shape[d - 1] - 3
    idx = torch.as_tensor([ni + int(q) for q in perm], device=x.device)
    sh = [1] * x.dim()
    sh[d - 1] = 3
    sgn = torch.tensor([-1.0 if f else 1.0 for f in flip], device=x.device, dtype=x.dtype).view(sh)
    v = torch.index_select(x, d - 1, idx) * sgn
    return torch.cat([x.narrow(d - 1, 0, ni), v], d - 1).contiguous(), tg.contiguous()


def radial_t(cyx, lo, shape, dtype=torch.float32, out=None):
    """`axis.radial` on the device: (B,3,Z,Y,X) unit vectors pointing away from the scroll axis in the xy
    plane (z component 0). `cyx` (B,2,Z) is the axis (y, x) at each z of the cube and `lo` (B,3) its
    corner; the differences are formed in float64 as they are on the CPU, only the normalisation is
    float32. `out`, when given, is written in place: at 256^3 that is 200 MB of card not allocated
    twice, and only one full-size temporary (the norm) is ever made."""
    Z, Y, X = (int(v) for v in shape)
    dev = cyx.device
    ay = torch.arange(Y, device=dev, dtype=torch.float64) + lo[:, 1, None]
    ax = torch.arange(X, device=dev, dtype=torch.float64) + lo[:, 2, None]
    dy = (ay[:, None, :] - cyx[:, 0][:, :, None]).to(dtype)[..., None]   # (B,Z,Y,1)
    dx = (ax[:, None, :] - cyx[:, 1][:, :, None]).to(dtype)[:, :, None]  # (B,Z,1,X)
    n = (dy * dy + dx * dx).sqrt_().add_(1e-6)
    if out is None:
        out = torch.empty((cyx.shape[0], 3, Z, Y, X), device=dev, dtype=dtype)
    out[:, 0] = 0
    torch.div(dy, n, out=out[:, 1])
    torch.div(dx, n, out=out[:, 2])
    return out


def radius_t(cyx, lo, shape, rmax, dtype=torch.float32, out=None):
    """`axis.radius` on the device: (B,1,Z,Y,X) of `clip(r / r_max, 0, 1)`. Same axis interpolation as
    `radial_t`, so the direction channel and the distance channel cannot disagree about where the axis
    is; `rmax` (B,) is in the sample's own rung voxels."""
    Z, Y, X = (int(v) for v in shape)
    dev = cyx.device
    ay = torch.arange(Y, device=dev, dtype=torch.float64) + lo[:, 1, None]
    ax = torch.arange(X, device=dev, dtype=torch.float64) + lo[:, 2, None]
    dy = (ay[:, None, :] - cyx[:, 0][:, :, None]).to(dtype)[..., None]
    dx = (ax[:, None, :] - cyx[:, 1][:, :, None]).to(dtype)[:, :, None]
    r = (dy * dy + dx * dx).sqrt_().div_(rmax.to(dtype).clamp_min(1e-6).view(-1, 1, 1, 1)).clamp_(0, 1)
    if out is None:
        return r[:, None]
    out[:, 0] = r
    return out


def n_planes_b(b):
    """How many PLANE channels a compact sample carries: the radius field plus the scan-metadata values."""
    return (1 if b.get("rmax") is not None else 0) + (int(b["meta"].shape[-1]) if "meta" in b else 0)


def fill_planes_(x, j, cyx=None, lo=None, rmax=None, meta=None, dtype=torch.float32):
    """Write the PLANE channels into `x[:, j:j+n]` in place and return n.

    The order is `Layout`'s: the radius FIELD first (one channel), then the constant scan planes. Both
    are plain spatial channels with no vector part, so `sym_apply_t` permutes and flips them like the
    cascade channel and the scale plane and never negates them."""
    n = 0
    if rmax is not None:
        radius_t(cyx, lo, x.shape[2:], rmax, dtype, out=x[:, j:j + 1])
        n += 1
    if meta is not None:
        m = meta.to(x.device).to(dtype)
        nm = int(m.shape[-1])
        x[:, j + n:j + n + nm] = m.reshape(m.shape[0], nm, 1, 1, 1)
        n += nm
    return n


def zscore_cubes_(img, norm, dtype):
    """z-score (B,C,Z,Y,X) image cubes in place with the sample's (mean, std): std 0 = the per-patch
    z-score, which is `sample.zscore`'s meaning with NORM unset."""
    B = img.shape[0]
    glob = (norm[:, 1] > 0).view(B, 1, 1, 1, 1)
    m = torch.where(glob, norm[:, 0].view(B, 1, 1, 1, 1).to(dtype), img.mean((2, 3, 4), keepdim=True))
    s = torch.where(glob, norm[:, 1].view(B, 1, 1, 1, 1).to(dtype),
                    img.std((2, 3, 4), keepdim=True, correction=0) + 1e-3)
    return img.sub_(m).div_(s)


def shapes(item):
    """(stem channels, head-target channels) of one compact sample."""
    return (int(item["ct"].shape[0]) + 4 + (1 if item.get("cm") is not None else 0) + n_planes_b(item),
            int(item["tgt"].shape[0]))


def batch1(item):
    """One compact sample -> a batch of one (what the validation grid and the tests want)."""
    return {k: (v[None] if torch.is_tensor(v) else v) for k, v in item.items()}


def prepare(b, dev, dtype=torch.float32, norad=False, non_blocking=True, cascade=None, layout=None):
    """A collated batch of compact samples (`sample.rung_item`) -> (x, target, weight) on `dev`.

    `x` is (B, cin, Z, Y, X) in exactly `Layout`'s order: every image cube z-scored with the sample's
    norm, the CASCADE channel (a probability in 0..1, NOT z-scored -- it is not an image channel), the
    radius plane, the five scan planes, the constant scale plane (k - 2) / 9 and the radial unit vector.
    Then the worker's cube symmetry is applied to the whole stack (every sample of a batch has the same
    patch shape, and `draw_sym` only draws permutations that keep it). `target` and `weight` come back as
    floats in 0..1. `norad=True` zeroes the radial channels. `cascade`: a `Cascade` saying where the
    channel's values come from; None means the `mask` source with no noise and no dropout, which is what
    validation and the tests want. `layout`, when given, is asserted against the result."""
    to = lambda t: t.to(dev, non_blocking=non_blocking)  # noqa: E731
    ct, tgt, w = to(b["ct"]), to(b["tgt"]), to(b["w"])
    lo, cyx, norm, rung = to(b["lo"]), to(b["cyx"]), to(b["norm"]), to(b["rung"])
    B, C, S = ct.shape[0], ct.shape[1], ct.shape[2:]
    casc = 1 if b.get("cm") is not None else 0
    npl = n_planes_b(b)
    x = torch.empty((B, C + 4 + casc + npl) + tuple(S), dtype=dtype, device=ct.device)  # filled in place
    img = x[:, :C]
    img.copy_(ct)
    zscore_cubes_(img, norm, dtype)
    if casc:
        cascade = cascade if cascade is not None else Cascade("mask", drop=0.0, noise=False)
        x[:, C:C + 1] = cascade.channel(b, x, norm, dtype, norad=norad, non_blocking=non_blocking)
    if npl:
        fill_planes_(x, C + casc, cyx, lo, dtype=dtype,
                     rmax=(to(b["rmax"]).reshape(-1) if b.get("rmax") is not None else None),
                     meta=(to(b["meta"]) if "meta" in b else None))
    x[:, C + casc + npl] = ((rung.to(dtype) - 2) / 9.0).view(B, 1, 1, 1)
    if norad:
        x[:, C + casc + npl + 1:] = 0
    else:
        radial_t(cyx, lo, S, dtype, out=x[:, C + casc + npl + 1:])
    if layout is not None:
        assert x.shape[1] == layout.cin and C == 1 + layout.nctx and casc == 1 \
            and npl == layout.n_planes and C + casc == layout.i_planes, \
            f"prepare built {x.shape[1]} channels ({C} cubes, {casc} cascade, {npl} planes), " \
            f"layout wants {layout.cin} ({1 + layout.nctx}, 1, {layout.n_planes})"
    tgt, w = tgt.to(dtype) / 255.0, w.to(dtype) / 255.0
    sym = [int(v) for v in b["sym"].reshape(-1).tolist()]
    if any(sym):  # one symmetry per sample, so the batch is done a sample at a time (B is 1 or 2)
        nt = tgt.shape[1]
        xs, ts = zip(*[sym_apply_t(v, x[i:i + 1], torch.cat([tgt[i:i + 1], w[i:i + 1]], 1))
                       for i, v in enumerate(sym)])
        x, tw = torch.cat(xs), torch.cat(ts)
        tgt, w = tw[:, :nt], tw[:, nt:]
    return x.contiguous(), tgt, w


class Cascade:
    """The source of the CASCADE input channel.

    mode:
      "off"   the channel is zero.
      "mask"  the rung-(k+1) TARGET block over the patch footprint (`cm`, read by the worker), upsampled
              2x. Without noise that is a blurred copy of the rung-k target -- a leak -- so `noise` is on
              by default: a one-voxel erosion or dilation (p 0.3) and 32^3 block dropout (p 0.2) on top
              of the blur the 2x pool + 2x upsample already applies.
      "self"  the model's OWN rung-(k+1) prediction, computed here with the EMA weights (no grad,
              autocast bf16) from the cubes the sample already carries: its CT cube is ctx_1, its
              contexts are ctx_2..ctx_9 plus the tenth cube `cx`, its scale plane is (k+1-2)/9, its
              radial vector is recomputed at rung k+1 from `cyx1`/`lo1`, and its own cascade channel is
              ZERO (one-level truncation). The central half of the output is the patch footprint.
      "mix"   per sample, "self" with probability `self_p`, else "mask" (+ noise). The production mode:
              "mask" alone leaks the target, "self" alone never shows the model a coarse prediction
              better than its own, and the mixture brackets what inference actually feeds it.

    `drop`: probability that a sample's channel is zeroed altogether, so inference WITHOUT a coarse
    prediction stays in distribution. The top rung is always zero: there is no rung above it."""

    def __init__(self, mode="off", self_p=0.5, drop=0.1, noise=True, net=None, seed=None, fwd=None):
        self.mode = str(mode or "off")
        assert self.mode in CASCADE_MODES, f"cascade {mode}: one of {CASCADE_MODES}"
        self.self_p, self.drop, self.noise, self.net = float(self_p), float(drop), bool(noise), net
        # `fwd`: a compiled forward of `net` (the self pass is a whole extra 256^3 forward; eager it was
        # the largest single cost of a training step on the A100). `net` stays the module whose
        # state_dict `sync` copies the EMA into: a compiled wrapper renames every key.
        self.fwd = fwd
        self.clock, self._ms = False, {}   # RVSM_PROFILE: ms spent in the self / mask sources
        self.gen = None if seed is None else torch.Generator().manual_seed(int(seed))
        # (B,) 1 where the last `channel()` call took the SELF source for that sample. The cascade
        # self-consistency loss scores ONLY those: the `mask` source is the coarse TARGET, so a
        # consistency term against it would be a second, blurrier copy of the supervised loss, and a
        # dropped channel is all zeros.
        self.last_self = None

    def _tick(self, name, t0):
        if self.clock:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._ms[name] = self._ms.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def take_clock(self):
        d, self._ms = {f"  of which cascade_{k}": v for k, v in self._ms.items()}, {}
        return d

    @property
    def on(self):
        return self.mode != "off"

    def sync(self, ema):
        """Point the self-mode net at the current EMA weights (one pair of foreach kernels, not a
        300-tensor python copy loop)."""
        if self.net is None or self.mode not in ("self", "mix"):
            return
        own = self.net.state_dict()
        es, vs = [], []
        for k, v in own.items():
            if k in ema:
                es.append(v), vs.append(ema[k])
        if es:
            torch._foreach_copy_(es, vs)

    def _rand(self, n=()):
        return torch.rand(n, generator=self.gen) if self.gen is not None else torch.rand(n)

    def _mask(self, b, i, dev, dtype, S):
        """The `mask` source for sample i: the coarse target block, optionally roughened, upsampled 2x."""
        c = b["cm"][i:i + 1].to(dev).to(dtype)[:, None] / 255.0
        if self.noise:
            r = float(self._rand())
            if r < 0.15:
                c = -torch.nn.functional.max_pool3d(-c, 3, stride=1, padding=1)   # erosion by one voxel
            elif r < 0.30:
                c = torch.nn.functional.max_pool3d(c, 3, stride=1, padding=1)     # dilation by one voxel
        u = M.up2x(c, tuple(S))
        if self.noise and float(self._rand()) < 0.20:  # block dropout: the model must not lean on it
            bs = [min(32, max(int(s) // 2, 1)) for s in S]
            for _ in range(4):
                o = [int(self._rand() * max(int(s) - q, 1)) for s, q in zip(S, bs)]
                u[..., o[0]:o[0] + bs[0], o[1]:o[1] + bs[1], o[2]:o[2] + bs[2]] = 0
        return u

    def coarse_input(self, b, i, dev, dtype, norad=False):
        """The rung-(k+1) model input of sample i, (1, C, Z, Y, X): exactly what `prepare` would build
        for a sample at rung k+1 over the same centre, with a ZERO cascade channel."""
        cubes = torch.cat([b["ct"][i, 1:], b["cx"][i]]).to(dev)[None]
        S, C = cubes.shape[2:], cubes.shape[1]
        npl = n_planes_b(b)
        x = torch.empty((1, C + 5 + npl) + tuple(S), dtype=dtype, device=dev)
        x[:, :C].copy_(cubes)
        zscore_cubes_(x[:, :C], b["norm"][i:i + 1].to(dev), dtype)
        x[:, C] = 0                                                   # its own cascade channel: truncated
        cyx1, lo1 = b["cyx1"][i:i + 1].to(dev), b["lo1"][i:i + 1].to(dev)
        if npl:  # the planes at rung k+1: the scan values are the same, r_max halves with the voxel size
            fill_planes_(x, C + 1, cyx1, lo1, dtype=dtype,
                         rmax=(b["rmax"][i:i + 1].to(dev).reshape(-1) * 0.5
                               if b.get("rmax") is not None else None),
                         meta=(b["meta"][i:i + 1].to(dev) if "meta" in b else None))
        x[:, C + 1 + npl] = (float(int(b["rung"][i]) + 1) - 2) / 9.0
        if norad:
            x[:, C + 2 + npl:] = 0
        else:
            radial_t(cyx1, lo1, S, dtype, out=x[:, C + 2 + npl:])
        return x

    @torch.no_grad()
    def _self(self, b, i, dev, dtype, S):
        """The `self` source: one extra forward of the EMA net at rung k+1, central half upsampled 2x."""
        x = self.coarse_input(b, i, dev, dtype)
        was = self.net.training
        self.net.eval()
        with autocast(dev):
            y = (self.fwd or self.net)(x.to(memory_format=M.memfmt()))
        self.net.train(was)
        y = y[0] if isinstance(y, (list, tuple)) else y
        p = torch.sigmoid(y.float())[:, :1].to(dtype)
        sl = tuple(slice(int(s) // 4, int(s) // 4 + max(int(s) // 2, 1)) for s in S)
        return M.up2x(p[(slice(None), slice(None)) + sl], tuple(S))

    def channel(self, b, x, norm, dtype=torch.float32, norad=False, non_blocking=True):
        """(B,1,Z,Y,X) cascade channel for the batch."""
        dev, B, S = x.device, x.shape[0], x.shape[2:]
        out = torch.zeros((B, 1) + tuple(S), dtype=dtype, device=dev)
        if not self.on:
            return out
        rung = b["rung"].reshape(-1).tolist()
        sel = torch.zeros(B, dtype=dtype, device=dev)
        self.last_self = sel
        for i in range(B):
            if int(rung[i]) + 1 >= ladder.NRUNGS:   # the top rung: no rung above it, no coarse prediction
                continue
            if self.drop > 0 and float(self._rand()) < self.drop:
                continue
            if self.mode == "self" or (self.mode == "mix" and float(self._rand()) < self.self_p):
                assert self.net is not None and b.get("cx") is not None, \
                    "cascade self mode needs a net and the tenth context cube"
                t0 = time.perf_counter()
                out[i:i + 1] = self._self(b, i, dev, dtype, S)
                self._tick("self", t0)
                sel[i] = 1
            else:
                t0 = time.perf_counter()
                out[i:i + 1] = self._mask(b, i, dev, dtype, S)
                self._tick("mask", t0)
        return out
