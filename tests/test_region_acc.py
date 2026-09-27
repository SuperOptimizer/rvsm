"""run_region's rolling z-window accumulators against the whole-box path they replace.

The round-1 self pass blends five planes over a 1024^3 box; holding whole-box accumulators for all of
them (18 GiB) OOMed the producer. The rolling path keeps only min(w, Z) z-rows of accumulator and
flushes the rest as the z-major window order passes them, which must change nothing: same terms, same
order, same dtypes. Pinned here on a synthetic three-plane problem (a probability, a thickness near
200, a distance near 1e-3) with a fine-CT margin, an air gap wider than a window (skipped windows, rows
the buffer never reaches) and a caller's own window order."""
import torch

from rvsm import infer

W, H = 16, 2


class Synth(infer.Inputs):
    """A padded region whose `prep` hands the window its global z coordinate and its start, so a
    window's output depends on both where the voxel is and which window predicts it (the blend
    matters). The CT is air on a z band wider than a window."""

    def __init__(self, box, core, gap):
        self.dev, self.window = torch.device("cpu"), W
        self.core, self.spread = core, True
        self.shape = tuple(b + 2 * c for b, c in zip(box, core))
        self.roi = torch.ones(self.shape, dtype=torch.uint8)
        self.roi[gap[0]:gap[1]] = 0
        self.roi[:, :, -3:] = 0

    def prep(self, o, out_dtype=torch.float32):
        z = torch.arange(o[0], o[0] + W, dtype=out_dtype)[:, None, None].expand(W, W, W)
        wid = torch.full((W, W, W), float(o[0] * 7 + o[1] * 3 + o[2]) / 100.0, dtype=out_dtype)
        return torch.stack([z, wid])[None]

    def window_any(self, o):
        return bool(self.roi[o[0]:o[0] + W, o[1]:o[1] + W, o[2]:o[2] + W].any())


def synth_fn(x):
    z, wid = x[:, 0], x[:, 1]
    prob = torch.sigmoid(torch.sin(z / 7.0) * 3 + 0.2 * torch.cos(wid))
    thick = 200.0 + 5.0 * torch.sin(z / 5.0) + 0.3 * torch.sin(wid)
    tiny = 0.001 * (1.0 + 0.5 * torch.sin(z / 3.0)) + 1e-5 * torch.cos(wid)
    return torch.stack([prob, thick, tiny], 1)


def test_rolling_accumulators_match_the_whole_box_path():
    box, core = (80, 24, 20), (4, 3, 0)
    inp = Synth(box, core, gap=(30, 70))          # 40 z-rows of air: > 2 W, window rows skipped
    kw = dict(planes=3, bounded=[True, False, False], acc_dtype=torch.float16, batch=3)
    s_old, s_new = {}, {}
    old = infer.run_region(synth_fn, inp, box, W, H, rolling=False, stats=s_old, **kw)
    new = infer.run_region(synth_fn, inp, box, W, H, stats=s_new, **kw)
    assert s_new["windows"] == s_old["windows"] > 0
    assert s_old["acc_depth"] == box[0] and s_new["acc_depth"] == W
    # 1 fp16 + 2 fp32 planes + fp32 wsum, over the z depth
    assert s_new["acc_bytes"] == W * box[1] * box[2] * (2 + 4 * 2 + 4)
    assert s_old["acc_bytes"] == s_new["acc_bytes"] * box[0] // W
    assert bool(torch.isfinite(new).all())
    # the task's tolerances (1/512 probability, 0.05 voxel regression) -- and in fact bit-identical
    assert float((new[0] - old[0]).abs().max()) <= 1 / 512
    assert float((new[1:] - old[1:]).abs().max()) <= 0.05
    assert torch.equal(new, old)

    # the planes are right, not merely unchanged: against a float64 blend, the regression planes are
    # far inside 0.05 voxel (the tiny plane to its own relative precision) and air is exactly 0
    keep = (inp.roi[core[0]:core[0] + box[0], core[1]:core[1] + box[1], core[2]:core[2] + box[2]] > 0)
    live = keep & (new[1] != 0)
    assert int(live.sum()) > 1000
    assert float(new[1][live].min()) > 190 and float(new[1][live].max()) < 210
    assert float(new[2][live].min()) > 4e-4 and float(new[2][live].max()) < 2e-3
    assert bool((new[:, ~keep] == 0).all())
    unreached = slice(30 - core[0] + W, 70 - core[0] - W)                # rows no window reaches
    assert unreached.stop - unreached.start == 8 and bool((new[:, unreached] == 0).all())

    # a caller's own window order (not z-major) is sorted into the same blend
    offs = infer.core_offsets(infer.spread_offsets(inp.shape, W, H, box), W, core, box, H)
    rev = infer.run_region(synth_fn, inp, box, W, H, offs=offs[::-1], **kw)
    assert float((rev[0] - old[0]).abs().max()) <= 1 / 512
    assert float((rev[1:] - old[1:]).abs().max()) <= 1e-4


def test_rolling_box_thinner_than_a_window_and_single_plane():
    """The teacher/m7 shape: one bounded plane, and a box thinner than the window (depth = Z)."""
    box, core = (10, 20, 20), (3, 0, 2)
    inp = Synth(box, core, gap=(0, 0))

    def one(x):
        return synth_fn(x)[:, :1]
    st = {}
    a = infer.run_region(one, inp, box, W, H, rolling=False)
    b = infer.run_region(one, inp, box, W, H, stats=st)
    assert st["acc_depth"] == box[0]
    assert torch.equal(a, b)
