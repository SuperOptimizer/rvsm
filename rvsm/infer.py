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
    """The student's region inputs -- the 21-channel stem of `Config.layout()`.

    NOT IMPLEMENTED IN COMMIT 2, deliberately: every piece it needs belongs to another commit, and a
    half-built version would be the one copy of the channel contract that drifts. The signature it will
    carry, so `run_region` and the producer can already be written against it:

        StudentInputs(ct, lo, size, window, ax, rung=2, ctx=(1..9), planes=..., sign=-1.0,
                      cascade=None, device="cuda")

        lo/size   the region box in rung-`rung` voxels
        ax        the umbilicus control points (rung-2 voxels, `rvsm.axis`)
        ctx       the context offsets: one super-cube per rung is read here, once per region
        planes    the conditioning planes (radius + scan metadata) from `rvsm.scanmeta`
        sign      the radial sign: +1 recto, -1 the verso flip of round 0
        cascade   the rung-(k+1) prediction upsampled onto this grid (`cascade_for`), or None

    `prep` will stack, in `Layout.stem_names()` order: the z-scored CT window, the nine z-scored context
    windows sliced out of the super-cubes, the cascade channel, the radius plane, the five metadata
    planes, the scale plane and the three radial components (the last three: the channels a flip TTA
    must negate). Commit 3 supplies `rvsm.prep`, commit 5 fills this in.
    """

    def __init__(self, *a, **kw):
        raise NotImplementedError(
            "StudentInputs lands in commit 5, on top of rvsm.prep / rvsm.model (commit 3)")


# --------------------------------------------------------------------------- #
# The region pass
# --------------------------------------------------------------------------- #
def run_region(fn, inputs, size, window, halo, batch=1, planes=1, acc_dtype=torch.float16,
               prep_dtype=torch.float32, offs=None, out_dtype=torch.float32):
    """The blended output of one region: (planes, Z, Y, X) with `size` = (Z, Y, X).

    `fn(x)` takes a (B, C, w, w, w) tensor and returns (B, planes, w, w, w). Windows whose CT is all air
    are skipped, and the result is zeroed wherever the CT is 0, so what comes back is defined exactly
    where the scroll is. The accumulators are `acc_dtype` (fp16 by default: see the module docstring);
    the division is done in fp32."""
    w, dev = int(window), inputs.dev
    g = gauss_t(w, dev, acc_dtype)
    todo = [o for o in (offsets(inputs.shape, w, halo) if offs is None else offs) if inputs.window_any(o)]
    acc = torch.zeros((int(planes),) + tuple(inputs.shape), dtype=acc_dtype, device=dev)
    wsum = torch.zeros(tuple(inputs.shape), dtype=acc_dtype, device=dev)
    for i in range(0, len(todo), max(1, int(batch))):
        ob = todo[i:i + max(1, int(batch))]
        x = torch.cat([inputs.prep(o, prep_dtype) for o in ob])
        with torch.no_grad():
            p = fn(x).to(acc_dtype)
        del x
        for o, pj in zip(ob, p):
            acc[:, o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w] += pj * g
            wsum[o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w] += g
        del p
    keep = ((wsum > 0) & (inputs.roi > 0))[None]
    out = torch.where(keep, acc.float() / wsum.float().clamp_min(1e-6)[None], torch.zeros((), device=dev))
    del acc, wsum
    Z, Y, X = (int(v) for v in size)
    return out[:, :Z, :Y, :X].to(out_dtype)


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
    """2x trilinear upsample of a (Z, Y, X) float32 array -- the same interpolation training uses."""
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
    return np.ascontiguousarray(up[a[0]:a[0] + int(s[0]), a[1]:a[1] + int(s[1]), a[2]:a[2] + int(s[2])])


# --------------------------------------------------------------------------- #
# The teacher pass
# --------------------------------------------------------------------------- #
def teacher_fn(net, spec, tta=1):
    """`(B, 1, w, w, w)` CT -> `(B, P, w, w, w)` probability: the teacher's activation, then its
    foreground channel (all channels when `fg_channel` is None)."""
    fg = spec.fg_channel

    def go(x):
        y = spec.select(net(x))
        p = teachers.apply_activation(y.float(), spec.activation)
        return p if fg is None else p[:, int(fg):int(fg) + 1]
    return flips_chan(go, int(tta), radial=False) if int(tta) > 1 else go


def teacher_region(ct, lo, size, spec, ckpt, device=None, backend="torch", tta=1, window=None, halo=None,
                   batch=1, engine_dir=None, acc_dtype=torch.float16):
    """One teacher over one region: the foreground probability as (Z, Y, X) float32 at RUNG 2.

    `lo` / `size` are rung-2 voxels. A teacher whose `level` is 0 (recto) runs on the region as it is. A
    teacher that works coarser (m7: level 2, 9.6 um) runs on the level-`level` CT with `spec.margin`
    voxels of context on every side -- a 1024^3 region is only 256^3 there, thinner than the window it
    was trained at -- and its probability is trilinearly upsampled back to rung 2 and masked by the
    COARSE air mask (the loader masks again with the fine CT, so a coarse mask here is safe and cheap).
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, spec = teachers.load_teacher(spec.name if hasattr(spec, "name") else spec, ckpt, device=dev)
    lvl = int(spec.level)
    w = int(window or spec.patch[0])
    h = int(halo if halo is not None else max(1, w // 8))
    eng = None
    if str(backend) == "trt":
        from rvsm import trt as trt_mod
        eng = trt_mod.engine_for(net, spec.name, w, 1, engine_dir or ".", device=str(dev))
        if eng is not None:
            batch = 1
    fn = teacher_fn(eng if eng is not None else net, spec, tta=tta)
    pyr = ladder.rungs(ct)
    lo, size = np.asarray(lo, np.int64), np.asarray(size, np.int64)
    f = 1 << lvl
    if lvl == 0:
        ct_blk = ladder.read_rung(pyr, RUNG, lo, size, dtype=np.uint8)
        inp = TeacherInputs(ct_blk, spec.normalizer, w, device=dev)
        p = run_region(fn, inp, tuple(size), w, h, batch=batch, planes=1, acc_dtype=acc_dtype)
        return p[0].float().cpu().numpy()
    assert not (lo % f).any() and not (size % f).any(), \
        f"teacher {spec.name} runs at level {lvl}: the box must be a multiple of {f}"
    o, s = lo // f, size // f
    m = int(spec.margin)
    shape_l = ladder.rung_shape(pyr, RUNG + lvl)
    a = np.maximum(o - m, 0)
    b = np.minimum(o + s + m, shape_l)
    roi = ladder.read_rung(pyr, RUNG + lvl, a, b - a, dtype=np.uint8)
    inp = TeacherInputs(roi, spec.normalizer, w, device=dev)
    c = o - a
    pc = run_region(fn, inp, tuple(int(v) for v in (b - a)), w, h, batch=batch, planes=1,
                    acc_dtype=acc_dtype)[0]
    pc = pc[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]]
    up = F.interpolate(pc[None, None].float().cpu(), size=tuple(int(v) for v in size), mode="trilinear",
                       align_corners=False)[0, 0].numpy()
    coarse = roi[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]] == 0
    air = np.repeat(np.repeat(np.repeat(coarse, f, 0), f, 1), f, 2)[:size[0], :size[1], :size[2]]
    return np.where(air, 0.0, up).astype(np.float32)


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
