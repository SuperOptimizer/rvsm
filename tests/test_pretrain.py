"""Masked-cube pretraining: the mask itself, three CPU steps, and the hand-over to `rvsm train`.

The two things that can silently break this stage are (a) a mask that does not actually hide anything
(or hides it in the context channels' answer key) and (b) a checkpoint whose trunk does not reach the
fine-tuning run. Both are asserted here; the rest is a smoke test that the loop runs and stays finite.
"""
import numpy as np
import pytest
import torch

from rvsm import pretrain as PT


def _cube(B=2, S=64, seed=0):
    """A (B,1,S,S,S) z-scored-looking cube with a bright slab: a foreground the quantile can find."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 1, S, S, S, generator=g) * 0.1
    x[:, :, :, S // 2 - 4:S // 2 + 4] += 3.0
    return x


def test_block_mask_ratio_and_alignment():
    """The masked fraction lands in [lo, hi] and the mask is constant inside every block."""
    ct = _cube(B=4, S=64)
    m, ratios, sheet = PT.block_mask(ct, block=16, lo=0.5, hi=0.75, sheet_p=0.0)
    assert m.shape == ct.shape
    assert set(np.unique(m.numpy())) <= {0.0, 1.0}
    assert float(ratios.min()) >= 0.5 - 1e-6 and float(ratios.max()) <= 0.75 + 1e-6
    got = m.mean((1, 2, 3, 4)).numpy()
    assert np.allclose(got, ratios.numpy(), atol=1e-6), (got, ratios)
    blocks = m[:, 0].reshape(4, 4, 16, 4, 16, 4, 16)      # (B, gz, 16, gy, 16, gx, 16)
    per = blocks.amax((2, 4, 6)) - blocks.amin((2, 4, 6))
    assert float(per.abs().max()) == 0.0, "a block is masked whole or not at all"
    assert not bool(sheet.any()), "sheet_p=0 must draw uniformly"


def test_block_mask_sheet_aware_lands_on_the_slab():
    """With sheet_p=1 the hidden blocks are the ones holding the slab, not the air around it."""
    ct = _cube(B=1, S=64, seed=3)
    fg = (ct > ct.quantile(0.7)).float()
    hit_u, hit_s = [], []
    for s in range(12):
        g = torch.Generator().manual_seed(s)
        mu, _, _ = PT.block_mask(ct, block=16, lo=0.5, hi=0.5, sheet_p=0.0, gen=g)
        ms, _, sheet = PT.block_mask(ct, block=16, lo=0.5, hi=0.5, sheet_p=1.0, gen=g)
        assert bool(sheet.all())
        hit_u.append(float((mu * fg).sum() / fg.sum()))
        hit_s.append(float((ms * fg).sum() / fg.sum()))
    assert np.mean(hit_s) > np.mean(hit_u) + 0.05, (np.mean(hit_s), np.mean(hit_u))


def test_mask_ctx_blanks_the_footprint_at_each_scale():
    """The CT mask reaches every context channel, at its own scale, over the CENTRAL footprint only."""
    S, ctx = 32, (1, 2, 3)
    x = torch.ones(1, 1 + len(ctx) + 1, S, S, S)
    m = torch.zeros(1, 1, S, S, S)
    m[:, :, :, :, : S // 2] = 1.0                      # mask the lower half in x
    PT.mask_ctx_(x, m, ctx)
    for j, off in enumerate(ctx):
        f = 1 << off
        sz = S // f
        o = (S - sz) // 2
        ch = x[0, 1 + j]
        inner = ch[o:o + sz, o:o + sz, o:o + sz]
        assert float(inner[:, :, : sz // 2].max()) == 0.0, f"ctx_{off} footprint not blanked"
        assert float(inner[:, :, sz // 2:].min()) == 1.0, f"ctx_{off} blanked past the mask"
        outside = ch.clone()
        outside[o:o + sz, o:o + sz, o:o + sz] = 1.0
        assert float(outside.min()) == 1.0, f"ctx_{off} touched outside the CT cube's footprint"


def test_mask_input_zeroes_the_ct_and_keeps_the_target():
    x = torch.randn(2, 6, 32, 32, 32) + 5.0
    before = x[:, :1].clone()
    x, tgt, m, ratios, _ = PT.mask_input(x, ctx=(1, 2), nimg=3, block=16, lo=0.6, hi=0.6, sheet_p=0.0)
    assert torch.equal(tgt, before), "the target is the CT BEFORE masking"
    assert float((x[:, :1] * m).abs().max()) == 0.0, "every masked voxel is 0"
    assert torch.equal(x[:, :1] * (1 - m), before * (1 - m)), "an unmasked voxel is untouched"
    assert float(PT.recon_loss(tgt, tgt, m)) == 0.0


@pytest.mark.parametrize("kind", ["l1", "l2"])
def test_recon_loss_masked_only(kind):
    tgt = torch.zeros(1, 1, 8, 8, 8)
    pred = torch.ones_like(tgt)
    m = torch.zeros_like(tgt)
    m[..., :4] = 1.0
    assert float(PT.recon_loss(pred, tgt, m, kind)) == pytest.approx(1.0)
    m2 = torch.zeros_like(tgt)
    assert float(PT.recon_loss(pred, tgt, m2, kind)) == 0.0   # no masked voxel, no loss


def test_three_cpu_steps_and_the_warm_start(region_cfg, tmp_path):
    """Three label-free steps run, stay finite, and the checkpoint warm-starts `rvsm train`."""
    from dataclasses import replace

    from rvsm import train as TR

    cfg = replace(region_cfg, steps=3, eval_every=3, out=str(tmp_path / "pre"))
    ck = PT.pretrain(cfg, out=cfg.out, steps=3, device="cpu")
    st = torch.load(ck, map_location="cpu", weights_only=False)
    assert int(st["step"]) == 3
    assert st["cin"] == cfg.layout().cin and st["cout"] == 1
    assert any(k.startswith("recon_head.") for k in st["ema"]), sorted(st["ema"])[:5]
    assert not any(k.startswith("head.") for k in st["ema"]), "the head must be renamed away"
    for v in st["ema"].values():
        if v.is_floating_point():
            assert torch.isfinite(v).all()
    rows = [l for l in open(tmp_path / "pre" / "logs" / "pretrain.jsonl")]
    assert rows, "the run logged nothing"

    # the hand-over: every TRUNK tensor copies by name, the segmentation head is new, and the
    # reconstruction head is what a plain strict=False load would report as unexpected.
    import rvsm.model as M
    layout = cfg.layout()
    net = M.build(cfg.size, cin=layout.cin, cout=layout.cout, verbose=False)
    src, newp = TR.warm_start(st["ema"], net, layout, src_layout=st.get("layout"))
    trunk = [k for k in net.state_dict() if not k.startswith(("head.", "deep_heads."))]
    assert not [k for k in trunk if k not in src], \
        f"trunk keys the warm start did not copy: {[k for k in trunk if k not in src][:5]}"
    assert newp and all(k.startswith(("head.", "deep_heads.")) for k in newp), sorted(newp)[:5]
    miss = net.load_state_dict(st["ema"], strict=False)
    assert [k for k in miss.unexpected_keys if k.startswith("recon_head.")], miss.unexpected_keys[:5]
    got = net.load_state_dict(src, strict=False)
    assert not got.unexpected_keys
    # and the copied trunk really is the pretrained one
    a = dict(net.state_dict())
    for k in trunk[:8]:
        assert torch.equal(a[k], st["ema"][k].to(a[k].dtype))


def test_train_init_from_a_pretrain_checkpoint(synth_run, tmp_path):
    """`rvsm pretrain` then `rvsm train --init` is the acceptance criterion of the stage."""
    from dataclasses import replace

    cfg = replace(synth_run.cfg, steps=2, eval_every=2)
    pre = str(tmp_path / "pre")
    ck = PT.pretrain(cfg, out=pre, steps=2, device="cpu")
    from rvsm import train as TR
    run = str(tmp_path / "ft")

    def items():
        from rvsm import sample
        ds = sample.Patches(cfg, root=synth_run.root, ct=cfg.ct, ax=synth_run.ax, label_free=False)
        it = iter(ds)
        for _ in range(2):
            yield next(it)

    out = TR.train(replace(cfg, out=run), out=run, init=ck, patches_factory=items, device="cpu",
                   steps=2)
    st = torch.load(out, map_location="cpu", weights_only=False)
    assert int(st["step"]) == 2
    assert st["model"]["head.weight"].shape[0] == cfg.layout().cout
