"""Per-rung temperature calibration.

A dice-trained sigmoid is measurably overconfident (Mehrtash et al., arXiv:1911.13273), and above the
native rung ours is not a probability *by construction*: there the target is a pooled FRACTION of a
binary band, not a binary event. So one scalar temperature is fitted PER RUNG on the run's own held-out
grid, by minimising the same weighted BCE the training loss uses, and only on the rungs whose target
really is a binary band -- calibrating against a pooled fraction would be fitting a scale to a different
quantity.

    p = sigmoid(logit / T)      T > 1 softens (the usual direction), T < 1 sharpens

The temperatures live in the checkpoint beside the config (`temps = {2: 1.13, ...}`); inference divides
the logits by the one for the rung it is predicting at, and the evaluator reads the same value through
`temp_for`. Nothing is retrained and no weight moves: calibrating rewrites only that dict, so an
uncalibrated checkpoint and a calibrated one give the same logits.

`run(net, grid_iter, layout)` takes an ITERABLE OF PREPARED BATCHES -- `(x, t, w, rung)`, exactly what
the sampler's validation grid yields once `prep` has built the stem. This module therefore does not
import the sampler, the model or the config loader: it is the fit, and nothing else.
"""
import math

import torch
import torch.nn.functional as F

BINARY_FRAC = 0.5   # a rung counts as a BINARY band when at most this share of its weighted target mass
                    # is strictly between `BINARY_EPS` and 1 - `BINARY_EPS`
BINARY_EPS = 0.02
T_LO, T_HI = 0.2, 5.0


def temps_of(d):
    """{rung: T} of a checkpoint's stored temperatures, ints for keys (json makes them strings)."""
    t = (d or {}).get("temps") if isinstance(d, dict) and "temps" in d else (d or {})
    return {int(k): float(v) for k, v in (t or {}).items()}


def temp_for(d, rung, use=True):
    """The temperature to divide the logits by at `rung`: 1.0 when there is none or `use` is False."""
    if not use:
        return 1.0
    return float(temps_of(d).get(int(rung), 1.0))


def bce_at(logit, tgt, w, T):
    """Weighted BCE of `sigmoid(logit / T)` against `tgt` -- the metric `fit_temp` minimises."""
    b = F.binary_cross_entropy_with_logits(logit / float(T), tgt, reduction="none")
    return float((b * w).sum() / w.sum().clamp_min(1e-6))


def fit_temp(logit, tgt, w, lo=T_LO, hi=T_HI, iters=40):
    """The T in [lo, hi] minimising `bce_at`, by golden-section search on log T.

    The BCE of a sigmoid in one temperature is convex in log T for a fixed set of logits, so a
    golden-section search is exact to the bracket width and needs no gradients and no optimiser state."""
    g = (math.sqrt(5) - 1) / 2
    a, b = math.log(lo), math.log(hi)
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = bce_at(logit, tgt, w, math.exp(c)), bce_at(logit, tgt, w, math.exp(d))
    for _ in range(int(iters)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = bce_at(logit, tgt, w, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = bce_at(logit, tgt, w, math.exp(d))
    return float(math.exp((a + b) / 2))


def binary_frac(tgt, w):
    """Share of the weighted target mass strictly inside (eps, 1 - eps): 0 for a hard band, large for a
    pooled fraction."""
    mid = ((tgt > BINARY_EPS) & (tgt < 1 - BINARY_EPS)).to(tgt.dtype)
    return float((mid * w).sum() / w.sum().clamp_min(1e-6))


@torch.no_grad()
def collect(net, grid_iter, layout=None, device=None, stride=(2, 4, 4)):
    """{rung: (logit, tgt, weight)} over the grid, CHANNEL 0 only -- the recto band is the one channel
    every validation patch carries and the only one with a published meaning.

    `grid_iter` yields prepared `(x, t, w, rung)` batches; `w` may be None (all ones). Everything is
    moved to the CPU as it is collected, so a long grid costs host memory and not VRAM -- which is why
    only every `stride`-th voxel (z, y, x) is kept: the whole of eight held-out regions' grid was ~40 GB
    of float32 on a 64 GB host, and one temperature per rung does not need 16 M voxels per window. (An
    axis shorter than 64 voxels is kept whole.)"""
    out = {}
    net.eval()
    for item in grid_iter:
        x, t = item[0], item[1]
        w = item[2] if len(item) > 2 and item[2] is not None else None
        rung = int(item[3]) if len(item) > 3 else 2
        if device is not None:
            x, t = x.to(device), t.to(device)
            w = None if w is None else w.to(device)
        lg = net(x)
        lg = lg[0] if isinstance(lg, (list, tuple)) else lg
        lg = lg.float()
        if layout is not None:
            lg = lg[:, :layout.cout_t]
        w = torch.ones_like(t) if w is None else w
        sl = (slice(None), slice(0, 1)) + tuple(slice(None, None, int(q) if n >= 64 else 1)
                                                for q, n in zip(stride, lg.shape[2:]))
        a, b, c = lg[sl].detach().cpu(), t[sl].detach().cpu(), w[sl].detach().cpu()
        if float(c.sum()) <= 0:
            continue
        out.setdefault(rung, []).append((a, b, c))
    return {k: tuple(torch.cat([q[i] for q in v]) for i in range(3)) for k, v in out.items()}


def keep(per, lg, t, w, rung, stride=(2, 4, 4)):
    """Add one window's channel-0 (logit, target, weight) to a `collect`-style dict, strided the same
    way `collect` strides it (axes shorter than 64 kept whole). Lets an evaluation that already ran the
    forward hand its logits to `run(per=...)` instead of calibration running the grid a second time."""
    sl = (slice(None), slice(0, 1)) + tuple(slice(None, None, int(q) if n >= 64 else 1)
                                            for q, n in zip(stride, lg.shape[2:]))
    a, b, c = lg[sl].detach().float().cpu(), t[sl].detach().float().cpu(), w[sl].detach().float().cpu()
    if float(c.sum()) > 0:
        per.setdefault(int(rung), []).append((a, b, c))
    return per


def stack(per):
    """`keep`'s lists -> `collect`'s {rung: (logit, tgt, weight)}."""
    return {k: tuple(torch.cat([q[i] for q in v]) for i in range(3)) for k, v in per.items()}


def run(net, grid_iter, layout=None, device=None, all_rungs=False, per=None):
    """Fit one temperature per rung over `grid_iter` and return `{"temps": {rung: T}, "rungs": [row, ...]}`.

    A rung whose target is a pooled FRACTION (`binary_frac` above `BINARY_FRAC`) is reported but NOT
    given a temperature unless `all_rungs`: a temperature fitted against a fraction is not a calibration.
    The caller writes `temps` into the checkpoint; nothing here touches a file. `per` is a precollected
    `stack(keep(...))` (the trainer's evaluation collects it on its own forward), and then `net` and
    `grid_iter` are not used."""
    per = collect(net, grid_iter, layout=layout, device=device) if per is None else per
    rows, temps = [], {}
    for k in sorted(per):
        lg, tg, w = per[k]
        bf = binary_frac(tg, w)
        T = fit_temp(lg, tg, w)
        row = {"rung": k, "n_patches": int(lg.shape[0]), "binary_frac": round(bf, 4),
               "bce_T1": round(bce_at(lg, tg, w, 1.0), 6), "bce_T": round(bce_at(lg, tg, w, T), 6),
               "T": round(T, 4), "binary": bool(bf <= BINARY_FRAC)}
        if row["binary"] or all_rungs:
            temps[k] = round(T, 4)
        else:
            row["skipped"] = "target is a pooled fraction, not a binary band"
        rows.append(row)
    return {"temps": temps, "rungs": rows}
