"""ONE region runner, for the teachers and for the student.

Everything rvsm infers -- a teacher's probability in round 0, the student's verso, the student's
multi-head pass from round 1 -- is the same computation: read a 1024^3 region's inputs ONCE, slide a
window over it, blend the outputs with a Gaussian, mask by the CT, and hand back one or more planes. The
only thing that differs is what a window's input tensor is made of, so that is the only thing this module
abstracts: an `Inputs` object with `prep(o) -> (1, C, w, w, w)` and `window_any(o) -> bool`.

Why the inputs are read once per REGION and not once per window: the context cubes of a 256^3 window at
rungs 3..11 come from nine different levels, and reading them per window re-decodes the top of the
pyramid tens of thousands of times. A region reads one super-cube per rung and slices it.

Why the accumulators are fp16: a 1024^3 region with 14 planes is 14 GiB in fp32 and 7 in fp16, which is
the difference between the producer fitting beside the trainer on one card and not. The error a fp16
accumulator adds is far below the 1/255 the store quantises to anyway (`test_infer_export` pins that).

The zero-window skip is not an optimisation of the average case, it is the reason a region outside the
scroll costs nothing: a window whose CT is all air cannot produce a sheet, so it is never forwarded, and
the blend leaves air at exactly 0.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rvsm import ladder, teachers

RUNG = 2            # region coordinates are always rung-2 voxels
CASCADE_HALO = 16   # rung-(k+1) voxels of margin around a coarse prediction's footprint


# --------------------------------------------------------------------------- #
# Window geometry and the blend kernel
# --------------------------------------------------------------------------- #
def starts(n, w, stride):
    """Window starts covering [0, n): first at 0, last exactly at n - w (so the far face is complete)."""
    s = list(range(0, max(n - w, 0) + 1, stride))
    if s[-1] != n - w:
        s.append(n - w)
    return s


def offsets(shape, window, halo):
    """Every (z, y, x) window start of a padded region."""
    stride = window - 2 * halo
    ss = [starts(int(shape[a]), int(window), stride) for a in range(3)]
    return [(z, y, x) for z in ss[0] for y in ss[1] for x in ss[2]]


GAUSS_SCALE = float(1 << 12)


def gauss_t(w, dev, dtype=torch.float32):
    """The separable Gaussian window weight (sigma = w/6), as a (w, w, w) tensor."""
    g = torch.exp(-0.5 * ((torch.arange(w, device=dev, dtype=torch.float64) - (w - 1) / 2) / (w / 6)) ** 2)
    return (g[:, None, None] * g[None, :, None] * g[None, None, :]).to(dtype)


# --------------------------------------------------------------------------- #
# What a window's input tensor is made of
# --------------------------------------------------------------------------- #
class Inputs:
    """The protocol `run_region` drives. Implementations hold a region's inputs, read once.

    Attributes: `roi` (the padded rung-`rung` CT of the region as a tensor, the air mask and the pad
    record in one), `shape` (its padded (Z, Y, X)), `window`, `dev`.

    `prep(o) -> (1, C, w, w, w)` float tensor on `dev`: the network input of the window at `o`.
    `window_any(o) -> bool`: False when the window is pure air, and then it is never forwarded.
    """

    roi: torch.Tensor
    shape: tuple
    window: int
    dev: torch.device

    def prep(self, o):
        raise NotImplementedError

    def window_any(self, o):
        w = self.window
        return bool(self.roi[o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w].any())


class TeacherInputs(Inputs):
    """A teacher sees the CT and nothing else, through its OWN normaliser.

    `ct` is the region's CT block at the teacher's level, already read (numpy uint8 or a tensor). A block
    thinner than one window on any axis is zero-padded (air) to a full window, exactly as usrm2's
    `slide` did, and the caller crops the result back.
    """

    def __init__(self, ct, normalizer, window, device="cpu"):
        self.dev = torch.device(device)
        self.window = int(window)
        self.norm = normalizer or teachers.Normalizer("none")
        a = ct.detach().cpu().numpy() if torch.is_tensor(ct) else np.asarray(ct)
        if any(s < self.window for s in a.shape):
            a = np.pad(a, [(0, max(self.window - s, 0)) for s in a.shape])
        self.roi = torch.from_numpy(np.ascontiguousarray(a)).to(self.dev)
        self.shape = tuple(self.roi.shape)

    def window_ct(self, o):
        w = self.window
        return self.roi[o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w]

    def prep(self, o, out_dtype=torch.float32):
        return self.norm(self.window_ct(o))[None, None].to(out_dtype)


class StudentInputs(Inputs):
    """The student's region inputs -- the `cin` stem of `config.Layout`, read ONCE per region.

    What is read once, and why it has to be: the rung-`rung` CT of the box (`roi`, which is also the air
    mask and the pad record), and ONE context super-cube per context rung, sized to cover every window's
    context box. A 256^3 window at rung 2 asks for nine coarse cubes centred on the same point; reading
    them per window re-decodes the top of the pyramid once per window, and at 64 windows a region that
    is the whole cost of the pass. The super-cube of rung `k + d` is the union of every window's box
    there, which is only `window + spread / 2^d` voxels on a side -- for d >= 3 barely more than one
    window.

    The umbilicus is interpolated once for the padded region's z range, and `prep` slices it; the radial
    vector and the radius plane are then built on the DEVICE by `rvsm.prep`'s own kernels, which is what
    makes a window's input the same tensor the trainer would have built (`test_infer_export` pins
    `prep(o)` against `prep.prepare` on the matching `sample.rung_item`, to 1e-5).

    `prep(o)` stacks, in `Layout.stem_names()` order and nothing else's:

        [CT z-scored, nctx z-scored context crops, cascade, radius, n_meta meta planes, scale,
         rz, ry, rx]

    `sign` multiplies the three radial channels and NOTHING else: that is the whole verso trick. A
    recto-trained student fed the negated radial vector places its band on the other face of the sheet,
    so round 0's verso store is this same pass at `sign=-1` (`rvsm produce --student --sign -1`).

    THE CASCADE CHANNEL is a top-down pass, not a stored input: the box's own footprint is predicted at
    rung k+1 first (an eighth of the work, plus a 16-voxel halo), upsampled 2x and fed in, recursively
    for `cascade_depth` rungs and zero above that -- which is exactly the `cascade_drop` case the model
    was trained on. `head0(k) -> fn` is the caller's per-rung head-0 window function (a `Student`'s), and
    the recursion re-enters THIS class at rung k+1, so a coarse pass is built by the same code and cannot
    drift from the fine one. `cascade=<array>` supplies it directly; `cascade_depth=0` leaves it zero.
    """

    def __init__(self, ct, ax, lo, size, layout, meta=None, rung=RUNG, ctx=None, window=256, halo=32,
                 sign=1.0, cascade=None, cascade_depth=0, head0=None, norm=None, device="cpu",
                 pyr=None, rmax_um=None, batch=1):
        from rvsm import axis as AX, scanmeta as SM
        self.dev = torch.device(device)
        self.ct, self.ax = ct, np.asarray(ax, np.float64)
        self.layout = layout
        self.rung, self.window, self.halo = int(rung), int(window), int(halo)
        self.sign, self.batch = float(sign), int(batch)
        self.ctx = tuple(int(q) for q in (ctx if ctx is not None else range(1, layout.nctx + 1)))
        assert len(self.ctx) == layout.nctx, \
            f"StudentInputs: {len(self.ctx)} context offsets for a layout with {layout.nctx}"
        self.pyr = pyr if pyr is not None else ladder.rungs(ct)
        self.head0, self.cascade_depth = head0, int(cascade_depth)
        lo = np.asarray(lo, np.int64)
        size = ladder.shape3(size)
        self.lo, self.size = tuple(int(v) for v in lo), tuple(int(v) for v in size)
        w = self.window

        # ---- the CT of the region, padded to at least one full window (air), as TeacherInputs does
        roi = ladder.read_rung(self.pyr, self.rung, lo, size, dtype=np.uint8)
        if any(s < w for s in roi.shape):
            roi = np.pad(roi, [(0, max(w - s, 0)) for s in roi.shape])
        self.roi = torch.from_numpy(np.ascontiguousarray(roi)).to(self.dev)
        self.shape = tuple(self.roi.shape)
        self.offs = offsets(self.shape, w, self.halo)

        # ---- ONE context super-cube per context rung, covering every window's context box
        cs = [sorted({self.lo[a] + int(o[a]) + w // 2 for o in self.offs}) for a in range(3)]
        self.ctx_cubes = {}
        for d in self.ctx:
            lo_d = [(min(cs[a]) >> d) - w // 2 for a in range(3)]
            sz_d = [(max(cs[a]) >> d) - w // 2 + w - lo_d[a] for a in range(3)]
            cube = ladder.read_rung(self.pyr, self.rung + d, lo_d, sz_d, dtype=np.uint8)
            self.ctx_cubes[d] = (lo_d, torch.from_numpy(np.ascontiguousarray(cube)).to(self.dev))

        # ---- the axis over the padded region's z range, interpolated once (float64, as `rung_item`'s)
        a_k = AX.axis_at(self.ax, self.rung)
        z = np.arange(self.shape[0]) + self.lo[0]
        self.cyx = torch.from_numpy(np.ascontiguousarray(
            np.stack([np.interp(z, a_k[0], a_k[1]), np.interp(z, a_k[0], a_k[2])]))).to(self.dev)

        # ---- the per-sample plane numbers: r_max at this rung, and the five scan values
        if rmax_um is None:
            kn = min(self.pyr)
            rmax_um = AX.rmax_vox(AX.axis_at(self.ax, kn), ladder.rung_shape(self.pyr, kn)) \
                * ladder.rung_um(kn)
        self.rmax_um = float(rmax_um)
        self.rmax = torch.tensor([self.rmax_um / ladder.rung_um(self.rung)], dtype=torch.float32,
                                 device=self.dev)
        self.meta_np = (np.asarray(SM.scan_planes(SM.fetch(ct)), np.float32) if meta is None
                        else np.asarray(meta, np.float32))
        self.meta = torch.from_numpy(np.ascontiguousarray(self.meta_np))[None].to(self.dev)
        self.norm = torch.tensor([[0.0, 0.0] if norm is None else [float(norm[0]), float(norm[1])]],
                                 dtype=torch.float32, device=self.dev)

        # ---- the cascade channel: the rung-(k+1) prediction over this footprint, upsampled
        if cascade is None and self.cascade_depth > 0 and head0 is not None:
            cascade = cascade_for(self._at_rung, self.rung, self.lo, self.size, self.cascade_depth)
        if cascade is None or torch.is_tensor(cascade):
            self.cascade = cascade
        else:
            self.cascade = np.ascontiguousarray(cascade, np.float32)

    # ---- the top-down cascade ------------------------------------------------------------------
    def _at_rung(self, k1, o1, s1, depth1):
        """Head-0 probability over a box at rung `k1`, cascading `depth1` rungs above it: the same class
        one rung up, driven through the same `run_region`."""
        sub = StudentInputs(self.ct, self.ax, o1, s1, self.layout, meta=self.meta_np, rung=int(k1),
                            ctx=self.ctx, window=self.window, halo=self.halo, sign=self.sign,
                            cascade_depth=int(depth1), head0=self.head0, device=self.dev,
                            pyr=self.pyr, rmax_um=self.rmax_um, batch=self.batch)
        p = run_region(self.head0(int(k1)), sub, tuple(int(v) for v in s1), self.window, self.halo,
                       batch=self.batch, planes=1, offs=sub.offs,
                       out_dtype=(torch.float16 if self.dev.type == "cuda" else torch.float32))
        del sub
        if self.dev.type == "cuda":
            return p[0]                  # fp16 on the card: the cascade never visits the host
        return p[0].float().cpu().numpy()

    # ---- one window ----------------------------------------------------------------------------
    def window_ct(self, o):
        w = self.window
        return self.roi[o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w]

    def ctx_window(self, d, o):
        """The context cube of rung `rung + d` for the window at `o`, sliced out of that rung's
        super-cube at the SAME centre the sampler uses (`ladder.context`: centre >> d, minus half a
        window)."""
        w = self.window
        lo_d, cube = self.ctx_cubes[int(d)]
        i = [((self.lo[a] + int(o[a]) + w // 2) >> int(d)) - w // 2 - lo_d[a] for a in range(3)]
        return cube[i[0]:i[0] + w, i[1]:i[1] + w, i[2]:i[2] + w]

    def prep(self, o, out_dtype=torch.float32):
        from rvsm import prep as P
        w, L = self.window, self.layout
        C = 1 + L.nctx
        x = torch.empty((1, L.cin, w, w, w), dtype=out_dtype, device=self.dev)
        img = x[:, :C]
        img[0, 0] = self.window_ct(o).to(out_dtype)
        for i, d in enumerate(self.ctx):
            img[0, 1 + i] = self.ctx_window(d, o).to(out_dtype)
        P.zscore_cubes_(img, self.norm, out_dtype)              # per cube, exactly as the loader's is
        if torch.is_tensor(self.cascade):
            c = self.cascade[o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w]
            x[:, L.i_cas] = 0
            x[0, L.i_cas, :c.shape[0], :c.shape[1], :c.shape[2]] = c.to(out_dtype)
        else:
            x[:, L.i_cas] = torch.from_numpy(crop_pad(self.cascade, o, (w, w, w))).to(self.dev).to(out_dtype)
        lo_w = torch.tensor([[self.lo[a] + int(o[a]) for a in range(3)]], dtype=torch.int64,
                            device=self.dev)
        cyx_w = self.cyx[None, :, o[0]:o[0] + w]
        n = P.fill_planes_(x, L.i_planes, cyx_w, lo_w, rmax=self.rmax,
                           meta=(self.meta if L.n_meta else None), dtype=out_dtype)
        assert n == L.n_planes, f"StudentInputs: built {n} planes, layout wants {L.n_planes}"
        x[:, L.i_scale] = (self.rung - 2) / 9.0
        P.radial_t(cyx_w, lo_w, (w, w, w), out_dtype, out=x[:, L.i_rad:L.i_rad + 3])
        if self.sign != 1.0:
            x[:, L.i_rad:L.i_rad + 3] *= self.sign
        return x


# --------------------------------------------------------------------------- #
# The region pass
# --------------------------------------------------------------------------- #
def run_region(fn, inputs, size, window, halo, batch=1, planes=1, acc_dtype=torch.float16,
               prep_dtype=torch.float32, offs=None, out_dtype=torch.float32, bounded=None):
    """The blended output of one region: (planes, Z, Y, X) with `size` = (Z, Y, X).

    `fn(x)` takes a (B, C, w, w, w) tensor and returns (B, planes, w, w, w). Windows whose CT is all air
    are skipped, and the result is zeroed wherever the CT is 0, so what comes back is defined exactly
    where the scroll is.

    ACCUMULATION. The Gaussian weight sum is always fp32. `bounded[i]` says plane i is a probability in
    [0, 1] (default: every plane -- the teachers' and the recto passes'); only those may use an
    `acc_dtype` fp16 accumulator, with the Gaussian scaled by 2^12 so a window corner (~1.5e-6
    unscaled, an fp16 subnormal) is a normal fp16 and <= 8 overlapping centres stay <= 32768. Any
    other plane -- a distance, a thickness, a log-variance -- accumulates in fp32 unscaled: scaled in
    fp16, a 31.75-voxel midline overflowed to inf (pass-3 review P3-11). The division is fp32."""
    w, dev = int(window), inputs.dev
    P = int(planes)
    bnd = [True] * P if bounded is None else [bool(b) for b in bounded]
    assert len(bnd) == P, (bnd, P)
    half = acc_dtype == torch.float16
    ib = [k for k in range(P) if bnd[k]] if half else []
    iu = [k for k in range(P) if k not in ib]
    g32 = gauss_t(w, dev, torch.float32)
    g16 = (g32 * GAUSS_SCALE).to(torch.float16) if ib else None
    todo = [o for o in (offsets(inputs.shape, w, halo) if offs is None else offs) if inputs.window_any(o)]
    acc_b = torch.zeros((len(ib),) + tuple(inputs.shape), dtype=torch.float16, device=dev) if ib else None
    acc_u = torch.zeros((len(iu),) + tuple(inputs.shape), dtype=torch.float32, device=dev) if iu else None
    wsum = torch.zeros(tuple(inputs.shape), dtype=torch.float32, device=dev)
    for i in range(0, len(todo), max(1, int(batch))):
        ob = todo[i:i + max(1, int(batch))]
        x = torch.cat([inputs.prep(o, prep_dtype) for o in ob])
        with torch.no_grad():
            p = fn(x)
        del x
        for o, pj in zip(ob, p):
            sl = (slice(o[0], o[0] + w), slice(o[1], o[1] + w), slice(o[2], o[2] + w))
            if ib:
                acc_b[(slice(None),) + sl] += pj[ib].to(torch.float16) * g16
            if iu:
                acc_u[(slice(None),) + sl] += pj[iu].float() * g32
            wsum[sl] += g32
        del p
    Z, Y, X = (int(v) for v in size)
    # normalised in z-slabs straight into the output dtype: the whole-volume float32 temporaries of
    # `where(keep, acc / wsum)` were 4 GB per plane on top of the accumulators (20+ GB for five heads)
    out = torch.empty((P, Z, Y, X), dtype=out_dtype, device=dev)
    for z in range(0, Z, 64):
        e = min(z + 64, Z)
        ws = wsum[z:e, :Y, :X].clamp_min(1e-30)
        keep = (wsum[z:e, :Y, :X] > 0) & (inputs.roi[z:e, :Y, :X] > 0)
        zero = torch.zeros((), device=dev)
        if ib:
            q = acc_b[:, z:e, :Y, :X].float() / (ws * GAUSS_SCALE)[None]
            out[ib, z:e] = torch.where(keep[None], q, zero).to(out_dtype)
        if iu:
            q = acc_u[:, z:e, :Y, :X] / ws[None]
            out[iu, z:e] = torch.where(keep[None], q, zero).to(out_dtype)
    del acc_b, acc_u, wsum
    return out


# --------------------------------------------------------------------------- #
# Flip TTA
# --------------------------------------------------------------------------- #
def flips_chan(fn, n=8, vec=(), radial=True):
    """Flip TTA for a function whose output KEEPS a channel axis, (B, C, Z, Y, X).

    Flipping spatial axis d flips the world along it, which has three consequences and each one is a bug
    when it is forgotten. On the INPUT side the radial vector's component d is negated (`radial=False`
    for a CT-only teacher, which has no radial channels to negate). On the OUTPUT side the spatial axes
    are 2 + d, not 1 + d. And a VECTOR output has to be flipped as a vector: `vec` lists the
    (cz, cy, cx) output-plane triples that are ZYX vectors, whose component d is negated too. Without
    that last step a normals TTA average silently cancels the field."""
    fl = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)][:int(n)]

    def go(t):
        out = 0
        for f in fl:
            x = torch.flip(t, [2 + d for d in f]).clone()
            if radial:
                ni = x.shape[1] - 3
                for d in f:
                    x[:, ni + d] = -x[:, ni + d]
            y = torch.flip(fn(x), [2 + d for d in f]).clone()
            for tri in vec:
                for d in f:
                    y[:, tri[d]] = -y[:, tri[d]]
            out = out + y
        return out / len(fl)
    return go


# --------------------------------------------------------------------------- #
# The top-down cascade channel (the student's; the teachers have none)
# --------------------------------------------------------------------------- #
def crop_pad(a, off, shape):
    """`a[off : off + shape]` zero-padded where it runs past the end: a region padded to a full window
    asks for windows the cascade array does not cover."""
    out = np.zeros(tuple(int(v) for v in shape), np.float32)
    if a is None:
        return out
    s = tuple(slice(int(o), min(int(o) + int(n), int(d))) for o, n, d in zip(off, shape, a.shape))
    blk = a[s]
    out[:blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
    return out


def up2x_np(a):
    """2x trilinear upsample of a (Z, Y, X) float32 array -- the same interpolation training uses.

    A TENSOR stays a tensor on its device, in its own dtype (the kernel accumulates in fp32 either
    way): a CUDA region pass keeps its cascade on the card. On the host the rung-2 cascade of a 1024^3
    region was a 5 GB float32 upsample plus a 4 GB crop of it -- 12 GB RSS peaks in the trainer's gate
    and in every producer student pass."""
    if torch.is_tensor(a):
        return F.interpolate(a[None, None], size=tuple(2 * int(q) for q in a.shape), mode="trilinear",
                             align_corners=False)[0, 0]
    t = torch.from_numpy(np.ascontiguousarray(a, np.float32))[None, None]
    up = F.interpolate(t, size=tuple(2 * int(q) for q in a.shape), mode="trilinear", align_corners=False)
    return up[0, 0].numpy()


def cascade_for(at_rung, k, o, s, depth, halo=CASCADE_HALO):
    """The rung-(k+1) prediction over the footprint of the box (o, s), upsampled 2x onto that box's own
    grid; None when nothing above may be predicted (depth exhausted, or the top of the ladder).

    `at_rung(k1, o1, s1, depth1) -> (Z, Y, X)` is the caller's own recursive region pass."""
    if depth <= 0 or int(k) + 1 >= ladder.NRUNGS:
        return None
    o1 = [max(int(v) // 2 - halo, 0) for v in o]
    s1 = [(int(v) + 1) // 2 + 2 * halo for v in s]
    up = up2x_np(at_rung(int(k) + 1, o1, s1, int(depth) - 1))
    a = [int(o[i]) - 2 * o1[i] for i in range(3)]
    crop = up[a[0]:a[0] + int(s[0]), a[1]:a[1] + int(s[1]), a[2]:a[2] + int(s[2])]
    if torch.is_tensor(crop):
        return crop.contiguous()
    return np.ascontiguousarray(crop)


# --------------------------------------------------------------------------- #
# The teacher pass
# --------------------------------------------------------------------------- #
def fast_teacher(net, device, compile=True, mode="max-autotune-no-cudagraphs"):
    """A teacher module as the producer runs it: bf16 autocast (the teachers load in fp32, and an fp32
    forward was the whole cost of a teacher region) and, on CUDA, `torch.compile` in `mode`. The raw
    module stays the caller's: ONNX export for a TensorRT plan wants it uncompiled.

    The compiled forward is traced for the one window shape it sees, so the first region pays the
    compile (minutes with max-autotune) and every later one runs the tuned kernels."""
    from rvsm import prep as P
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    run = torch.compile(net, mode=mode, dynamic=False) if (compile and dev.type == "cuda") else net

    def go(x):
        with torch.no_grad(), P.autocast(dev):
            return run(x)
    go.raw = net
    return go


def teacher_fn(net, spec, tta=1):
    """`(B, 1, w, w, w)` CT -> `(B, P, w, w, w)` probability: the teacher's activation, then its
    foreground channel (all channels when `fg_channel` is None)."""
    fg = spec.fg_channel

    def go(x):
        y = spec.select(net(x))
        p = teachers.apply_activation(y.float(), spec.activation)
        return p if fg is None else p[:, int(fg):int(fg) + 1]
    return flips_chan(go, int(tta), radial=False) if int(tta) > 1 else go


def teacher_read(ct, lo, size, spec, pyr=None):
    """The CT a teacher pass over the region (`lo`, `size`, rung 2) reads, as (roi uint8, a): the block
    at the teacher's own level, with `spec.margin` voxels of context for a coarse teacher, and its
    corner `a` at that level. Split out of `teacher_region` so a producer can read the NEXT region's CT
    (the volcomp decode is seconds) on a thread while the GPU runs the current one."""
    if isinstance(spec, str):
        spec = teachers.TEACHERS[spec]
    pyr = ladder.rungs(ct) if pyr is None else pyr
    lvl = int(spec.level)
    lo, size = np.asarray(lo, np.int64), np.asarray(size, np.int64)
    if lvl == 0:
        return ladder.read_rung(pyr, RUNG, lo, size, dtype=np.uint8), lo
    f = 1 << lvl
    assert not (lo % f).any() and not (size % f).any(), \
        f"teacher {spec.name} runs at level {lvl}: the box must be a multiple of {f}"
    o, s = lo // f, size // f
    m = int(spec.margin)
    shape_l = ladder.rung_shape(pyr, RUNG + lvl)
    a = np.maximum(o - m, 0)
    b = np.minimum(o + s + m, shape_l)
    return ladder.read_rung(pyr, RUNG + lvl, a, b - a, dtype=np.uint8), a


def teacher_region(ct, lo, size, spec, ckpt, device=None, backend="torch", tta=1, window=None, halo=None,
                   batch=1, engine_dir=None, acc_dtype=torch.float16, net=None, roi=None,
                   as_tensor=False, fast=None):
    """One teacher over one region: the foreground probability as (Z, Y, X) float32 at RUNG 2.

    `lo` / `size` are rung-2 voxels. A teacher whose `level` is 0 (recto) runs on the region as it is. A
    teacher that works coarser (m7: level 2, 9.6 um) runs on the level-`level` CT with `spec.margin`
    voxels of context on every side -- a 1024^3 region is only 256^3 there, thinner than the window it
    was trained at -- and its probability is trilinearly upsampled back to rung 2 and masked by the
    COARSE air mask (the loader masks again with the fine CT, so a coarse mask here is safe and cheap).

    `roi` is `teacher_read`'s result when the caller read the CT ahead of time. `as_tensor` returns the
    probability as a float16 tensor ON THE DEVICE instead of a host array, so a producer can fuse and
    quantise there and move only uint8 across the bus. `fast` is `fast_teacher(net, ...)`: what the
    torch path runs when no TensorRT engine is in use.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if net is None:   # `net` is a teacher the CALLER already loaded: a producer loads each one once
        net, spec = teachers.load_teacher(spec.name if hasattr(spec, "name") else spec, ckpt, device=dev)
    elif isinstance(spec, str):
        spec = teachers.TEACHERS[spec]
    lvl = int(spec.level)
    w = int(window or spec.patch[0])
    h = int(halo if halo is not None else max(1, w // 8))
    eng = None
    if str(backend) == "trt":
        from rvsm import trt as trt_mod
        eng = trt_mod.engine_for(net, spec.name, w, 1, engine_dir or ".", device=str(dev))
        if eng is not None:
            batch = 1
    fn = teacher_fn(eng if eng is not None else (fast if fast is not None else net), spec, tta=tta)
    lo, size = np.asarray(lo, np.int64), np.asarray(size, np.int64)
    blk, a = teacher_read(ct, lo, size, spec) if roi is None else roi
    odt = torch.float16 if as_tensor else torch.float32
    if lvl == 0:
        inp = TeacherInputs(blk, spec.normalizer, w, device=dev)
        p = run_region(fn, inp, tuple(size), w, h, batch=batch, planes=1, acc_dtype=acc_dtype,
                       out_dtype=odt)[0]
        return p if as_tensor else p.float().cpu().numpy()
    f = 1 << lvl
    o, s = lo // f, size // f
    inp = TeacherInputs(blk, spec.normalizer, w, device=dev)
    c = o - np.asarray(a, np.int64)
    pc = run_region(fn, inp, tuple(int(v) for v in blk.shape), w, h, batch=batch, planes=1,
                    acc_dtype=acc_dtype, out_dtype=odt)[0]
    pc = pc[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]]
    # upsample and mask ON THE DEVICE: the same trilinear on the CPU was ~115 s of a ~125 s m7 region
    up = F.interpolate(pc[None, None], size=tuple(int(v) for v in size), mode="trilinear",
                       align_corners=False)[0, 0]
    del pc
    coarse = inp.roi[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]] == 0
    air = coarse.repeat_interleave(f, 0).repeat_interleave(f, 1).repeat_interleave(f, 2)
    up.masked_fill_(air[:size[0], :size[1], :size[2]], 0.0)
    del air
    return up if as_tensor else up.cpu().numpy().astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Fusing the two round-0 teachers
# --------------------------------------------------------------------------- #
def binary_confidence(p):
    """1 - H2(p)/ln 2 for p in [0, 1]: 1 where a source commits (p = 0 or 1), 0 where it is undecided
    (p = 0.5). The normalised binary entropy is the cheapest per-voxel confidence that needs nothing but
    the probability itself."""
    q = np.clip(np.asarray(p, np.float32), 1e-6, 1 - 1e-6)
    h = -(q * np.log(q) + (1 - q) * np.log1p(-q)) / np.log(2.0)
    return (1.0 - h).astype(np.float32)


def fuse_agreement(ps, pm, ws=1.0, wm=1.0, floor=0.05):
    """(probability, weight) of two teachers over the same voxel:

        p = (a_s p_s + a_m p_m) / (a_s + a_m),   a_x = w_x * (confidence(p_x) + floor)
        w = 1 - |p_s - p_m|

    a CONFIDENCE-weighted mean (the source that commits carries the voxel) whose loss weight is the
    sources' AGREEMENT, so a voxel the two lineages disagree about is down-weighted smoothly instead of
    being decided by lineage order. `floor` keeps a voxel both sources call 0.5 from dividing by zero.
    Both outputs are float32 in [0, 1]. The pitfall this cannot see: where the two agree AND are both
    wrong, it fuses confidently into the same error."""
    ps, pm = np.asarray(ps, np.float32), np.asarray(pm, np.float32)
    a_s = float(ws) * (binary_confidence(ps) + float(floor))
    a_m = float(wm) * (binary_confidence(pm) + float(floor))
    p = (a_s * ps + a_m * pm) / np.maximum(a_s + a_m, 1e-6)
    return p.astype(np.float32), (1.0 - np.abs(ps - pm)).astype(np.float32)


def u8_t(p):
    """`stores.u8` on a tensor: round-half-even to 0..255, uint8, on whatever device `p` is on."""
    return torch.round(p.float() * 255).clamp_(0, 255).to(torch.uint8)


def fuse_agreement_u8(ps, pm, ws=1.0, wm=1.0, floor=0.05, chunk=32):
    """`fuse_agreement` on DEVICE tensors, straight to the two uint8 stores: (u8 probability, u8 weight),
    both still on the device. Done in z-slabs of `chunk` so the float32 temporaries of a 1024^3 region
    stay a few hundred MB instead of a dozen GB. The same formula as the numpy version, in float32."""
    P = torch.empty(ps.shape, dtype=torch.uint8, device=ps.device)
    W = torch.empty(ps.shape, dtype=torch.uint8, device=ps.device)
    ln2 = float(np.log(2.0))

    def conf(q):
        q = q.clamp(1e-6, 1 - 1e-6)
        return 1.0 - (-(q * torch.log(q) + (1 - q) * torch.log1p(-q)) / ln2)

    for z in range(0, int(ps.shape[0]), int(chunk)):
        a, b = ps[z:z + chunk].float(), pm[z:z + chunk].float()
        a_s = float(ws) * (conf(a) + float(floor))
        a_m = float(wm) * (conf(b) + float(floor))
        P[z:z + chunk] = u8_t((a_s * a + a_m * b) / (a_s + a_m).clamp_min(1e-6))
        W[z:z + chunk] = u8_t(1.0 - (a - b).abs())
    return P, W


# --------------------------------------------------------------------------- #
# The student pass
# --------------------------------------------------------------------------- #
# WHAT A CHECKPOINT HOLDS. `rvsm train` writes, and everything here reads:
#
#     {"cfg": asdict(Config), "layout": Layout.to_json(), "ema": state_dict, "model": state_dict,
#      "temps": {rung: T}, "step": int}
#
# `cfg` is the contract: the layout (and therefore cin/cout, the head order, the context offsets) is
# DERIVED from it, never trusted from the stored copy, so a checkpoint whose recorded layout disagrees
# with its own config fails loudly at build time instead of silently mapping the wrong head. `ema` is
# what inference runs -- the raw weights are only ever the training state. The reader below accepts the
# older spellings (`args`, `config`, `model`, `state_dict`) so a hand-made or third-party checkpoint can
# still be driven, and says which key it used.
CKPT_STATE_KEYS = ("ema", "model", "state_dict", "net")
CKPT_CFG_KEYS = ("cfg", "config", "args")


def save_student(path, state, cfg, temps=None, step=0, **extra):
    """Write a student checkpoint in the format `student_fn` reads. `state` is the EMA state dict."""
    import os
    from dataclasses import asdict
    d = {"cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
         "layout": cfg.layout().to_json(), "ema": dict(state), "step": int(step),
         "temps": {int(k): float(v) for k, v in (temps or {}).items()}, **extra}
    if os.path.dirname(str(path)):
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    torch.save(d, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", str(path))
    return str(path)


def load_student_ckpt(path, map_location="cpu"):
    """(raw dict, Config, Layout, state dict, {rung: T}, step) of a student checkpoint."""
    from rvsm import calib as CAL
    from rvsm.config import Config, _coerce, _TYPES
    st = torch.load(str(path), map_location=map_location, weights_only=False)
    raw = next((st[k] for k in CKPT_CFG_KEYS if isinstance(st.get(k), dict)), None)
    if isinstance(raw, dict) and isinstance(raw.get("config"), dict):
        raw = raw["config"]   # a TRAINER checkpoint stores `cfg.to_json()` = {config, layout, fingerprint}
    assert raw is not None, f"{path}: no config in the checkpoint (looked for {CKPT_CFG_KEYS})"
    cfg = Config(**{k: _coerce(k, v) for k, v in raw.items() if k in _TYPES})
    sd = next((st[k] for k in CKPT_STATE_KEYS if isinstance(st.get(k), dict) and st[k]), None)
    assert sd is not None, f"{path}: no weights in the checkpoint (looked for {CKPT_STATE_KEYS})"
    temps = CAL.temps_of(st.get("temps") or raw.get("temps") or {})
    return st, cfg, cfg.layout(), sd, temps, int(st.get("step", 0))


class Student:
    """One student checkpoint, loaded once, as a set of per-rung window functions.

    Every plane the tracer contract asks for is a POINTWISE function of the same raw head output, so a
    multi-head pass is one forward per window and not five over the same voxels:

        recto / verso   sigmoid(logit / T(rung))     the probability heads -- the ONLY ones the
                                                     per-rung temperature touches (a temperature is a
                                                     calibration of a probability; dividing a distance
                                                     in voxels by 1.13 would just be wrong)
        midline         the head, raw, in voxels
        thickness       TMIN + softplus(head)        it cannot go below the minimum physical thickness
        conf            1 / (1 + exp(logvar / 2))    from the heteroscedastic log-variance

    The affinity heads are training-only and are never read here. `plane_fn(names, rung)` builds the
    callable `run_region` drives; `head0(rung)` is the one-plane version the cascade recursion uses.
    """

    FIELDS = ("midline", "thickness", "conf")

    def __init__(self, ckpt, device=None, compile=True, mode="max-autotune-no-cudagraphs", temps=True):
        from rvsm import model as M
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt = str(ckpt)
        st, cfg, layout, sd, tmps, step = load_student_ckpt(ckpt, map_location=self.dev)
        self.cfg, self.layout, self.step = cfg, layout, step
        self.temps = tmps if temps else {}
        self.use_temps = bool(temps)
        net = M.build(cfg.size, cin=layout.cin, cout=layout.cout, ckpt_act=0,
                      add_skip=int(st.get("add_skip", 0)), deep=0, verbose=False).to(self.dev)
        net.load_state_dict(sd)
        net.eval()
        self.raw = net
        self.net = (torch.compile(net, mode=mode) if (compile and self.dev.type == "cuda") else net)
        self.planes = tuple(str(c) for c in layout.channels) + self.FIELDS

    # ---- the planes ----------------------------------------------------------------------------
    def temp(self, rung):
        """The temperature the probability heads are divided by at `rung` (1.0 when uncalibrated)."""
        from rvsm import calib as CAL
        return CAL.temp_for(self.temps, int(rung), use=self.use_temps)

    def plane_names(self, heads="all"):
        """`heads` -> the plane names, in `self.planes` order. "all" is every one; a name the checkpoint
        cannot serve is an error, because a silently dropped plane is a store that never appears."""
        if heads is None or (isinstance(heads, str) and str(heads) == "all"):
            return list(self.planes)
        names = [str(q) for q in ([heads] if isinstance(heads, str) else heads)]
        bad = [n for n in names if n not in self.planes]
        assert not bad, f"{self.ckpt}: cannot serve plane(s) {bad} (has {list(self.planes)})"
        return names

    def piece(self, name, rung):
        """(B, cout, ...) raw head output -> the (B, 1, ...) plane `name`."""
        from rvsm import losses as L
        ch = [str(c) for c in self.layout.channels]
        if name in ch:
            T = float(self.temp(rung))
            j = ch.index(name)
            return lambda y, j=j, T=T: torch.sigmoid(y[:, j:j + 1].float() / T)
        if name == "midline":
            return lambda y, j=self.layout.i_mid: y[:, j:j + 1].float()
        if name == "thickness":
            return lambda y, j=self.layout.i_thick: L.soft_thickness(y[:, j:j + 1].float(), L.TMIN)
        if name == "conf":
            return lambda y, j=self.layout.i_log: 1.0 / (
                1.0 + torch.exp(0.5 * y[:, j:j + 1].float().clamp(-8, 8)))
        raise KeyError(f"{self.ckpt}: no plane {name!r} (has {list(self.planes)})")

    def plane_fn(self, heads="all", rung=RUNG):
        """`fn(x) -> (B, P, w, w, w)`: ONE forward, every named plane read off that one output."""
        from rvsm import model as M, prep as P
        names = self.plane_names(heads)
        parts = [self.piece(n, rung) for n in names]

        def go(x):
            with torch.no_grad(), P.autocast(self.dev):
                y = self.net(x.contiguous(memory_format=M.memfmt()))
            y = y[0] if isinstance(y, (list, tuple)) else y
            return torch.cat([f(y) for f in parts], 1)
        go.plane_names = names
        return go

    def head0(self, rung=RUNG):
        """The first probability head at `rung` -- what the top-down cascade predicts one rung up."""
        return self.plane_fn([str(self.layout.channels[0])], rung)

    def __call__(self, x):
        return self.plane_fn()(x)


def student_fn(ckpt_path, device=None, compile=True, mode="max-autotune-no-cudagraphs", temps=True):
    """A `Student` for a checkpoint: the net built from its own cfg/layout, the EMA weights loaded, the
    per-rung temperature applied to the PROBABILITY heads only, ready to serve plane stacks."""
    return Student(ckpt_path, device=device, compile=compile, mode=mode, temps=temps)


def student_region(student, ct, ax, lo, size, sign=1.0, heads="all", meta=None, device=None,
                   window=None, halo=None, cascade_depth=None, batch=1, rung=RUNG, tta=1, pyr=None,
                   acc_dtype=torch.float16, umbilicus=None, as_tensor=False):
    """One student pass over one region: `{plane name: (Z, Y, X) float32}` in rung-`rung` voxels
    (`as_tensor`: float16 tensors left on the device, for a producer that encodes there).

    `student` is a `Student` or a checkpoint path; `lo` / `size` the box in rung-`rung` voxels; `ax` the
    umbilicus control points in rung-2 voxels (`axis.load`). `sign=-1` negates the radial input channels,
    which is how a recto-trained student produces the VERSO band (round 0's verso store).

    Every plane comes out of ONE sliding-window pass -- one forward per window, every head read off that
    output -- and every plane is blended with the same Gaussian and zeroed wherever the CT is air.
    """
    st = student if isinstance(student, Student) else student_fn(student, device=device)
    names = st.plane_names(heads)
    w = int(window if window is not None else st.cfg.infer_window)
    h = int(halo if halo is not None else st.cfg.infer_halo)
    d = int(cascade_depth if cascade_depth is not None else st.cfg.cascade_depth)
    inp = StudentInputs(ct, ax, lo, size, st.layout, meta=meta, rung=int(rung), ctx=st.cfg.ctx,
                        window=w, halo=h, sign=sign, cascade_depth=d, head0=st.head0, device=st.dev,
                        pyr=pyr, batch=batch)
    fn = st.plane_fn(names, int(rung))
    if int(tta) > 1:
        fn = flips_chan(fn, int(tta), radial=True)
    bounded = [n in set(str(c) for c in st.layout.channels) or n == "conf" for n in names]
    out = run_region(fn, inp, tuple(int(v) for v in ladder.shape3(size)), w, h, batch=batch,
                     bounded=bounded,
                     planes=len(names), offs=inp.offs, acc_dtype=acc_dtype,
                     out_dtype=(torch.float16 if as_tensor else torch.float32))
    if as_tensor:
        return {n: out[i] for i, n in enumerate(names)}
    return {n: np.ascontiguousarray(out[i].float().cpu().numpy(), np.float32)
            for i, n in enumerate(names)}
