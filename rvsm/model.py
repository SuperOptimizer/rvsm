"""The student: a plain 3D U-Net, `cin` stem channels in (config.Layout), `cout` logits out.

Nothing about the architecture is conditional on the task: every head is one 1x1 convolution on the
finest decoder stage, and which head means what is `Layout.head_names()`'s business alone. The presets
are the size ladder -- `30m6`'s six levels with every width scaled by 1/sqrt(2) and by sqrt(2), rounded
to a multiple of 8 (GroupNorm takes min(8, c) groups and an off-grid width loses tensor-core alignment),
which is a clean factor-2 ladder in parameters and therefore three points evenly spaced in log(params)
for a log-log fit of val loss.

`1m` is the tests' net. `5m` is the laptop's. `30m6` is the default and `60m` the top of the ladder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    "1m": (16, 32, 64, 128),                       # the CPU tests
    "5m": (32, 64, 128, 256),                      # the laptop 5080 at patch 128
    "15m": (24, 48, 88, 184, 272, 272),            # the ladder's bottom rung
    "30m6": (32, 64, 128, 256, 384, 384),          # the default: ~500-voxel theoretical receptive field
    "60m": (48, 88, 184, 360, 544, 544),           # the ladder's top rung
}

# Memory format of the weights and of the input. channels_last_3d gives the convolutions cudnn's NDHWC
# tensor-core kernels, but at 256^3 this net is bound by the normalisations and activations, and those are
# far slower in that layout (measured on an A100: GroupNorm+SiLU fwd+bwd 96 ms channels_last vs 16 ms
# contiguous; the whole 30m6 step 1152 ms vs 679 ms). So the net runs in plain NCDHW.
CHANNELS_LAST = False


def memfmt():
    return torch.channels_last_3d if CHANNELS_LAST else torch.contiguous_format


def up2x(x, size):
    """Trilinear upsample of (B,C,Z,Y,X) to `size`, with a fast path at an exact factor of 2.

    `F.interpolate(..., mode="trilinear", align_corners=False)` at 2x is, per axis,
        out[2m] = 0.75 in[m] + 0.25 in[m-1],   out[2m+1] = 0.75 in[m] + 0.25 in[m+1]
    with the out-of-range neighbour clamped to the edge. Written that way -- two weighted sums and an
    interleave per axis -- every operation is a GATHER, so the backward is a gather too; aten's
    `upsample_trilinear3d_backward` is a scatter-add and costs ~1.1 s per call at the decoder's widest
    stage on an A100 against ~0.1 s for this form. The values agree with `F.interpolate` to 2e-7
    relative in float32. Any other ratio falls back to `F.interpolate`, so the semantics never change."""
    if tuple(int(v) for v in size) != tuple(2 * int(v) for v in x.shape[2:]):
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    for d in range(3):
        a, n = 2 + d, x.shape[2 + d]
        prev = torch.cat([x.narrow(a, 0, 1), x.narrow(a, 0, n - 1)], a)
        nxt = torch.cat([x.narrow(a, 1, n - 1), x.narrow(a, n - 1, 1)], a)
        x = torch.stack([0.75 * x + 0.25 * prev, 0.75 * x + 0.25 * nxt], a + 1).flatten(a, a + 1)
    return x


def block(cin, cout):
    layers = []
    for c in (cin, cout):
        layers += [nn.Conv3d(c, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU()]
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, widths=PRESETS["1m"], cin=4, cout=1, ckpt_act=0, add_skip=0, deep=0):
        """ckpt_act: recompute the activations of the blocks at the first `ckpt_act` levels (the
        full-resolution ones hold most of the memory) in the backward pass; True/-1 = every level.
        add_skip: at the first `add_skip` levels the decoder ADDS the skip to a 1x1 projection of the
        upsampled tensor instead of concatenating, so the widest full-resolution tensor is w0 channels
        and not w0 + w1.
        deep: also predict the `cout` maps at decoder levels 1..deep; `forward` returns
        [logits_level0, logits_level1, ...] while training and level 0 otherwise."""
        super().__init__()
        self.deep = min(int(deep), len(widths) - 2)
        self.ckpt_act = len(widths) if ckpt_act is True or ckpt_act < 0 else int(ckpt_act)
        self.add_skip = int(add_skip)
        w = list(widths)
        self.enc = nn.ModuleList([block(cin if i == 0 else w[i - 1], w[i]) for i in range(len(w))])
        self.down = nn.ModuleList([nn.Conv3d(c, c, 3, stride=2, padding=1) for c in w[:-1]])
        self.dec = nn.ModuleList([block(w[i] if i < self.add_skip else w[i] + w[i + 1], w[i])
                                  for i in range(len(w) - 1)])
        self.proj = nn.ModuleList([nn.Conv3d(w[i + 1], w[i], 1) if i < self.add_skip else nn.Identity()
                                   for i in range(len(w) - 1)])
        self.head = nn.Conv3d(w[0], cout, 1)
        self.deep_heads = nn.ModuleList([nn.Conv3d(w[i], cout, 1) for i in range(1, self.deep + 1)])

    def _run(self, m, x, level):
        rg = any(t.requires_grad for t in (x if isinstance(x, tuple) else (x,)))
        if level < self.ckpt_act and self.training and rg:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(m, x, use_reentrant=False)
        return m(x)

    def forward(self, x):
        skips = []
        if self.ckpt_act and self.training and not x.requires_grad:
            x = x.requires_grad_()  # checkpointed blocks need a grad path through their input
        for i, e in enumerate(self.enc):
            x = self._run(e, x, i)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        outs = {}
        for i in range(len(self.dec) - 1, -1, -1):
            x = self._run(self._stage(i), (x, skips[i]), i)  # upsample + concat INSIDE the checkpointed
            if 1 <= i <= self.deep:                          # segment: the wide concat is never stored
                outs[i] = self.deep_heads[i - 1](x)
        y = self.head(x)
        if self.deep and self.training:
            return [y] + [outs[i] for i in range(1, self.deep + 1)]
        return y

    def _stage(self, i):
        dec, proj, add = self.dec[i], self.proj[i], i < self.add_skip

        def f(pair):
            x, skip = pair
            x = up2x(x, skip.shape[2:])
            return dec(proj(x) + skip) if add else dec(torch.cat([x, skip], 1))
        return f


def params(size="1m", cin=4, cout=1, add_skip=0, deep=0):
    """Parameter count of a preset WITHOUT allocating it (built on the `meta` device). The ladder fits
    val loss against log(params), and the count depends on the run's own cin/cout, never the name."""
    with torch.device("meta"):
        m = UNet(PRESETS[size], cin=int(cin), cout=int(cout), add_skip=int(add_skip), deep=int(deep))
    return int(sum(p.numel() for p in m.parameters()))


def build(size="1m", cin=4, cout=1, ckpt_act=0, add_skip=0, deep=0, verbose=True):
    if size not in PRESETS:
        raise KeyError(f"unknown size {size!r}: one of {sorted(PRESETS)}")
    m = UNet(PRESETS[size], cin=int(cin), cout=int(cout), ckpt_act=ckpt_act, add_skip=int(add_skip),
             deep=int(deep)).to(memory_format=memfmt())
    if verbose:
        n = sum(p.numel() for p in m.parameters())
        print(f"rvsm UNet {size} widths={PRESETS[size]} in={cin} heads={cout} params={n / 1e6:.2f}M")
    return m
