"""In-domain masked-cube pretraining (MAE-style) of the trunk `rvsm train` uses.

`docs/recipe.md` experiment 11. The objective: take the SAME sample the rung loader builds for `train`
-- the CT cube, the nine context cubes, the cascade slot, the radius and scan planes, the scale plane
and the radial vector -- blank out a large fraction of the CT cube, and ask the SAME trunk to put the
z-scored CT back. **No store is read**, so this runs over the whole mirrored scroll before a single
teacher pass has finished: `sample.Patches(label_free=True)` draws windows anywhere the CT is not air.

What makes the weights reusable is that nothing about the trunk changes. `model.build` is called with
the config's own `size`, `ckpt_act` and `Layout.cin` -- the cascade slot included, held at ZERO, which
is exactly the in-distribution "no coarse prediction" value -- and only the 1x1 OUTPUT head is
repurposed. In the checkpoint that head is stored under `recon_head.*`, so `train --init`'s
`warm_start` copies every trunk tensor by name, finds no `head.*` to copy, and reports the segmentation
head as NEW (which is what gives it `new_param_lr_mult`). `tests/test_pretrain.py` asserts exactly that.

Why the context cubes are masked too: context cube j sits at rung k + ctx[j] over the same centre, so
its central 2^-ctx[j] box is a coarser copy of the CT cube. Left alone it is a free low-frequency answer
key, and the model would learn to upsample rather than to invent texture. The same voxel mask is
therefore max-pooled down and pasted into each context channel's own footprint (`mask_ctx=False` turns
that off).

Everything else -- WSD schedule, EMA, bf16 autocast, `torch.compile`, activation checkpointing, the
atomic save, the fingerprint-checked resume -- is `train.py`'s, imported from it rather than copied:
one schedule, one EMA rule, one autocast, for the pretraining and the fine-tuning halves alike.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rvsm import aug as A
from rvsm import model as M
from rvsm import prep
from rvsm.prep import autocast
from rvsm.train import ema_auto, ema_update, lr_lambda, wsd_stable_until

# The defaults of the masking task (usrm2's, unchanged: 50-75 % of 32^3 blocks, half of the samples
# masked structure-aware).
BLOCK = 32
MASK_LO, MASK_HI = 0.5, 0.75
SHEET_P = 0.5
SHEET_PCT = 0.7

# The checkpoint stores the reconstruction head under these names, so a fine-tuning warm start drops it
# instead of loading a reconstruction head into the segmentation head. Everything else -- enc.*, down.*,
# dec.*, proj.* -- keeps the state-dict keys `train` expects, byte for byte.
HEAD_RENAME = (("head.", "recon_head."), ("deep_heads.", "recon_deep_heads."))


def rename_out(sd):
    """UNet state dict -> checkpoint keys (the output head becomes `recon_head.*`)."""
    out = {}
    for k, v in sd.items():
        for a, b in HEAD_RENAME:
            if k.startswith(a):
                k = b + k[len(a):]
                break
        out[k] = v
    return out


def rename_in(sd):
    """Checkpoint keys -> UNet state dict (the inverse of `rename_out`, for a resume)."""
    out = {}
    for k, v in sd.items():
        for a, b in HEAD_RENAME:
            if k.startswith(b):
                k = a + k[len(b):]
                break
        out[k] = v
    return out


# ------------------------------------------------------------------------------- the masking


def quantile_(x, q, cap=1 << 20):
    """Per-sample quantile of (B,1,Z,Y,X), computed on a strided subsample so a 256^3 cube does not hit
    `torch.quantile`'s 16M-element limit (and costs a sort of 1M values, not 16M)."""
    f = x.reshape(x.shape[0], -1)
    if f.shape[1] > cap:
        f = f[:, :: max(f.shape[1] // cap, 1)]
    return torch.quantile(f.float(), float(q), dim=1)


def block_mask(ct, block=BLOCK, lo=MASK_LO, hi=MASK_HI, sheet_p=SHEET_P, pct=SHEET_PCT, gen=None):
    """A (B,1,Z,Y,X) float mask, 1 where the CT voxel is MASKED, built out of `block`^3 blocks.

    Per sample a masking ratio r is drawn uniformly from [lo, hi] and round(r * nblocks) blocks are
    hidden. With probability `sheet_p` the blocks are drawn STRUCTURE-AWARE: a block is sampled with
    probability proportional to its foreground fraction, foreground being "z-scored CT above the `pct`
    quantile of this cube" -- a one-line proxy for "on a sheet". Sheet-heavy blocks are then what
    disappears, so the model has to reconstruct sheet texture and cannot score well by interpolating
    through air. Otherwise the blocks are drawn uniformly (plain MAE).

    Returns `(mask, ratios, sheet)`: `ratios` is the ACHIEVED masked fraction of BLOCKS per sample and
    `sheet` the per-sample bool, which is what the tests check.
    """
    B, S = ct.shape[0], ct.shape[2:]
    g = [max(int(math.ceil(int(s) / block)), 1) for s in S]
    n = int(np.prod(g))
    thr = quantile_(ct, pct).view(B, 1, 1, 1, 1)
    fg = F.avg_pool3d((ct > thr).to(ct.dtype), block, stride=block, ceil_mode=True).reshape(B, n)
    r = torch.rand(B, generator=gen).to(ct.device) * (hi - lo) + lo
    sheet = torch.rand(B, generator=gen).to(ct.device) < sheet_p
    sel = torch.zeros(B, n, device=ct.device, dtype=ct.dtype)
    ratios = []
    for i in range(B):
        k = min(max(int(round(float(r[i]) * n)), 1), n)
        w = (fg[i] + 1e-3) if bool(sheet[i]) else torch.ones(n, device=ct.device, dtype=ct.dtype)
        idx = torch.multinomial(w.float(), k, replacement=False,
                                generator=(gen if gen is not None and gen.device == w.device else None))
        sel[i, idx] = 1.0
        ratios.append(k / n)
    m = sel.view(B, 1, *g)
    for d in range(3):                       # blocks -> voxels, then crop the ceil_mode overhang
        m = m.repeat_interleave(block, dim=2 + d)
    m = m[:, :, : int(S[0]), : int(S[1]), : int(S[2])].contiguous()
    return m, torch.tensor(ratios), sheet


def mask_ctx_(x, m, ctx):
    """Paste the CT mask into the context channels, at each one's own scale, in place.

    Context channel j (`x[:, 1 + j]`) is the cube at rung k + ctx[j]: the same voxel count, 2^ctx[j]
    times coarser, the same centre. The CT cube's footprint inside it is therefore the CENTRAL box of
    side S / 2^ctx[j], and the mask pooled down by that factor (MAX pool: a coarse voxel that sees any
    masked fine voxel is masked) is what has to be blanked there. Past the point where the footprint is
    under one voxel there is nothing left to leak and the channel is left alone.
    """
    S = [int(v) for v in m.shape[2:]]
    for j, off in enumerate(ctx):
        f = 1 << int(off)
        sz = [s // f for s in S]
        if min(sz) < 1:
            break
        mj = F.max_pool3d(m, f)
        o = [(s - q) // 2 for s, q in zip(S, sz)]
        sl = (slice(None), slice(1 + j, 2 + j)) + tuple(slice(a, a + q) for a, q in zip(o, sz))
        x[sl] = x[sl] * (1 - mj)
    return x


def mask_input(x, ctx=(), nimg=None, block=BLOCK, lo=MASK_LO, hi=MASK_HI, sheet_p=SHEET_P,
               pct=SHEET_PCT, mask_ctx=True, gen=None):
    """`(x, target, mask, ratios, sheet)`: blank the CT channel and, with `mask_ctx`, the context
    channels' footprint.

    `x` is the finished model input (`prep.prepare` + `aug.apply`): channel 0 the z-scored CT, channels
    1..nimg-1 the context cubes, then the cascade slot, the planes and the radial vector. The target is
    the CT channel BEFORE masking; masked voxels are set to 0, which is the mean of the z-scored cube --
    the "no information" value, exactly what a dropped cascade channel is.
    """
    tgt = x[:, :1].clone()
    m, ratios, sheet = block_mask(tgt, block, lo, hi, sheet_p, pct, gen)
    x[:, :1] = x[:, :1] * (1 - m)
    if mask_ctx and ctx:
        nc = (int(nimg) - 1) if nimg else len(ctx)
        mask_ctx_(x, m, tuple(ctx)[:nc])
    return x, tgt, m, ratios, sheet


def recon_loss(pred, tgt, m, kind="l1"):
    """Reconstruction loss on the MASKED voxels only: an unmasked voxel is a copy, not a prediction."""
    d = (pred - tgt).abs() if kind == "l1" else (pred - tgt) ** 2
    return (d * m).sum() / m.sum().clamp_min(1.0)


# ------------------------------------------------------------------------------- the sample


def _log(path, rec, echo=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if echo:
        print(os.path.basename(path), rec, flush=True)


def _batch(item):
    return item if item["rung"].ndim else prep.batch1(item)


def _scan_meta(cfg):
    """The scan's own metadata, when there is a `metadata.json` beside the CT: the augmentation ranges
    must be the SAME ones the fine-tuning run will see, or the pretrained trunk is adapted to a
    different distribution than the one it is handed over to."""
    try:
        from rvsm import scanmeta as SM
        return SM.fetch(cfg.ct)
    except Exception as e:   # noqa: BLE001  -- no metadata beside the CT: the preset's own ranges
        print(f"[pretrain] scan metadata unavailable ({e!r}); using the preset ranges", flush=True)
        return None


def label_free_source(cfg, out, ax=None, seed=0):
    """`(patches_factory, held-out grid)` for a pretraining run: the label-free sampler over the whole
    CT and a fixed grid over the held-out regions, which is scored with a FIXED mask so the eval number
    moves because the model improved and not because other blocks were hidden."""
    from rvsm import axis as AX, ladder, regions as RG, sample
    ct = str(cfg.ct)
    pyr = ladder.rungs(ct)
    ax = AX.load(cfg.umbilicus, ct=ct) if ax is None else ax
    recs = RG.region_list(pyr, rungs=cfg.rungs, patch=cfg.patch, region=cfg.region,
                          boost=cfg.rung_boost, occ_min_fine=cfg.occ_min_fine,
                          occ_min_coarse=cfg.occ_min_coarse)
    held = RG.held_out(recs, n=cfg.heldout, ax=ax)

    def factory():
        ds = sample.Patches(cfg, root=str(out), ct=ct, ax=ax, label_free=True, heldout=held, seed=seed)
        return sample.loader(ds, workers=cfg.workers, batch=cfg.batch)

    grid = []
    try:
        grid = sample.val_grid(cfg, held, root=str(out), ct=ct, ax=ax, limit=2, label_free=True)
    except AssertionError as e:   # noqa: BLE001  -- a held-out region smaller than a patch: no grid
        print(f"[pretrain] no validation grid ({e}); training without one", flush=True)
    return factory, grid


def build_x(b, dev, layout, acfg):
    """One collated sample -> the model input, augmented, with the cascade slot at zero.

    `prep.prepare` builds the cascade channel from the sample's `cm` block, which a label-free sample
    carries as zeros -- so the slot is present (the stem has the fine-tuning run's exact `cin`) and
    holds the "no coarse prediction" value, with no coarse pass and no leak.
    """
    x, tg, _ = prep.prepare(b, dev, layout=layout)
    ks = [int(v) for v in b["rung"].reshape(-1).tolist()]
    x, _ = A.apply(x, tg, acfg, nimg=layout.i_cas, rung=ks)
    return x.contiguous(), ks


@torch.no_grad()
def evaluate(net, grid, dev, layout, acfg, args):
    """Masked L1/L2 on the held-out grid with a FIXED mask (seeded per rung) and no augmentation."""
    was = net.training
    net.eval()
    tot, n, per = 0.0, 0, {}
    for item in grid:
        b = _batch(item)
        x, _, _ = prep.prepare(b, dev, layout=layout)
        k = int(b["rung"].reshape(-1)[0])
        g = torch.Generator().manual_seed(1234 + k)
        x, tgt, m, _, _ = mask_input(x.contiguous(), args["ctx"], layout.i_cas, args["block"],
                                     args["mask_lo"], args["mask_hi"], args["sheet_p"],
                                     args["sheet_pct"], args["mask_ctx"], gen=g)
        with autocast(dev):
            p = net(x.to(memory_format=M.memfmt()))
        p = (p[0] if isinstance(p, (list, tuple)) else p).float()
        v = float(recon_loss(p, tgt, m, args["loss"]))
        tot, n = tot + v, n + 1
        per.setdefault(k, []).append(v)
    net.train(was)
    out = {"recon": tot / max(n, 1)}
    for k in sorted(per):
        out[f"recon_r{k}"] = float(np.mean(per[k]))
    return out


# ------------------------------------------------------------------------------------ the run


def pretrain(cfg, out=None, steps=None, device=None, block=BLOCK, lo=MASK_LO, hi=MASK_HI,
             sheet_p=SHEET_P, pct=SHEET_PCT, mask_ctx=True, loss="l1", resume=False,
             patches_factory=None, val_items=None, accum=1):
    """Masked-cube pretraining of the trunk. Returns the checkpoint path (`<out>/ckpt.pt`).

    The checkpoint is what `rvsm train --init` reads: the trunk under its usual names, the
    reconstruction head under `recon_head.*`, the config and the layout beside them.
    """
    out = Path(out or cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    layout = cfg.layout()
    nsteps = int(cfg.steps if steps is None else steps)
    ctx = tuple(int(v) for v in cfg.ctx)
    args = {"block": int(block), "mask_lo": float(lo), "mask_hi": float(hi), "sheet_p": float(sheet_p),
            "sheet_pct": float(pct), "mask_ctx": bool(mask_ctx), "loss": str(loss), "ctx": ctx}

    # cout 1: the reconstruction is ONE channel. Every other build argument is the fine-tuning run's,
    # which is what makes the stem and the trunk transfer without a reshape.
    net = M.build(cfg.size, cin=layout.cin, cout=1, ckpt_act=cfg.ckpt_act, gn_bf16=cfg.gn_bf16,
                  verbose=False).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.01)
    S, C = wsd_stable_until(nsteps, cfg.cooldown)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(nsteps, int(cfg.warmup), sched=cfg.sched, stable_until=S, cooldown=C))
    decay = ema_auto(nsteps, cfg.ema_k) if str(cfg.ema) == "auto" else float(cfg.ema)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    step = 0

    ck = out / "ckpt.pt"
    if resume and ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        got = (st.get("cfg") or {}).get("fingerprint")
        assert got == cfg.fingerprint(), \
            f"resume: the checkpoint's config fingerprint is {got}, this run's is {cfg.fingerprint()}"
        net.load_state_dict(rename_in(st["model"]))
        opt.load_state_dict(st["opt"])
        ema = {k: v.to(dev) for k, v in rename_in(st["ema"]).items()}
        step = int(st["step"])
        for _ in range(step):
            sched.step()
        print(f"[pretrain] resumed {ck} at step {step}", flush=True)

    acfg = A.get(cfg.aug, meta=_scan_meta(cfg))
    if patches_factory is None or val_items is None:
        factory, grid = label_free_source(cfg, out)
        patches_factory = patches_factory or factory
        val_items = grid if val_items is None else val_items
    grid = list(val_items or [])
    evnet = M.build(cfg.size, cin=layout.cin, cout=1, gn_bf16=cfg.gn_bf16, verbose=False).to(dev)
    model = torch.compile(net) if cfg.compile else net

    def save():
        tmp = ck.with_suffix(".tmp")
        torch.save({"model": rename_out(net.state_dict()), "ema": rename_out(ema),
                    "opt": opt.state_dict(), "step": step, "cfg": cfg.to_json(),
                    "layout": layout.to_json(), "pretrain": args, "cin": layout.cin, "cout": 1}, tmp)
        tmp.replace(ck)

    _log(str(out / "logs" / "pretrain.jsonl"),
         {"step": step, "size": cfg.size, "patch": cfg.patch, "batch": cfg.batch, "cin": layout.cin,
          "steps": nsteps, "mask": [args["mask_lo"], args["mask_hi"]], "block": args["block"],
          "sheet_p": args["sheet_p"], "ctx": list(ctx), "ema": decay,
          "fingerprint": cfg.fingerprint()})

    t0, micro, rung_n, nvox = time.time(), 0, {}, 0
    rl = None
    for item in patches_factory():
        if step >= nsteps:
            break
        b = _batch(item)
        x, ks = build_x(b, dev, layout, acfg)
        for k in ks:
            rung_n[k] = rung_n.get(k, 0) + 1
        x, tgt, m, ratios, _ = mask_input(x, ctx, layout.i_cas, args["block"], args["mask_lo"],
                                          args["mask_hi"], args["sheet_p"], args["sheet_pct"],
                                          args["mask_ctx"])
        x = x.to(memory_format=M.memfmt())
        nvox += int(np.prod(x.shape[2:])) * x.shape[0]
        with autocast(dev):
            pred = model(x)
            pred = pred[0] if isinstance(pred, (list, tuple)) else pred
            rl = recon_loss(pred.float(), tgt, m, args["loss"])
        (rl / max(int(accum), 1)).backward()
        micro += 1
        if micro < int(accum):
            continue
        micro = 0
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        ema_update(ema, net, decay)
        step += 1

        if step % 20 == 0:
            dt = max(time.time() - t0, 1e-6)
            _log(str(out / "logs" / "pretrain.jsonl"),
                 {"step": step, "recon": float(rl.detach()), "ratio": round(float(ratios.mean()), 3),
                  "lr": sched.get_last_lr()[0], "vox_s": round(nvox / dt),
                  "vram_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else 0,
                  "rung": {str(k): rung_n[k] for k in sorted(rung_n)}})
            rung_n, nvox, t0 = {}, 0, time.time()
        if step % max(int(cfg.eval_every), 1) == 0 or step >= nsteps:
            if grid:
                evnet.load_state_dict(ema)
                _log(str(out / "logs" / "eval.jsonl"),
                     {"step": step, **evaluate(evnet, grid, dev, layout, acfg, args)})
            save()
            t0 = time.time()
    save()
    return str(ck)
