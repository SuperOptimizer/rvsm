#!/usr/bin/env python3
"""Engine / sliding-window lab for tools/m7_wholevol.py (runs on one GPU; the other keeps producing).

  build   --window W --batch B --level L [--tag T]   a TensorRT fp16 plan from the tsm m7 ONNX
  time    --plan P [--n N] [--graph]                  ms per launch and Mvox/s per window-voxel
  gate    --vol STEM --center z,y,x --half H          the quality table: every (plan, window, stride)
          --configs plan:window:stride,...              against the reference config (the first one),
                                                        dice at p>=0.5 and corr over CT>0 voxels
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m7_wholevol as M  # noqa: E402

ONNX = "/vesuvius/tsm/models/trt/m7_b1_fp16.onnx"
LAB = os.path.join(M.WORK, "lab")


def build(window, batch, level, tag=""):
    import onnx
    import tensorrt as trt
    os.makedirs(LAB, exist_ok=True)
    name = f"m7_p{window}_b{batch}_L{level}{tag}"
    plan = os.path.join(LAB, name + ".plan")
    m = onnx.load(ONNX, load_external_data=True)
    for vi in list(m.graph.input) + list(m.graph.output):
        d = vi.type.tensor_type.shape.dim
        for i, v in enumerate([batch, d[1].dim_value, window, window, window]):
            d[i].dim_value = int(v)
    del m.graph.value_info[:]
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    p = trt.OnnxParser(net, log)
    assert p.parse(m.SerializeToString()), [str(p.get_error(i)) for i in range(p.num_errors)]
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    cfg.builder_optimization_level = int(level)
    tc_path = os.path.join(LAB, "timing.cache")
    tc = cfg.create_timing_cache(open(tc_path, "rb").read() if os.path.exists(tc_path) else b"")
    cfg.set_timing_cache(tc, ignore_mismatch=False)
    t = time.time()
    ser = b.build_serialized_network(net, cfg)
    assert ser is not None, "build failed"
    open(plan, "wb").write(bytes(ser))
    open(tc_path, "wb").write(bytes(cfg.get_timing_cache().serialize()))
    # the layer summary: which kernels / whether InstanceNorm fused
    rt = trt.Runtime(log)
    eng = rt.deserialize_cuda_engine(bytes(ser))
    insp = eng.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info.get("Layers", [])
    kinds = {}
    for L in layers:
        k = L.get("LayerType", "?") if isinstance(L, dict) else "?"
        kinds[k] = kinds.get(k, 0) + 1
    print(json.dumps({"plan": plan, "build_s": round(time.time() - t, 1), "MiB": ser.nbytes >> 20,
                      "n_layers": len(layers), "layer_types": kinds}), flush=True)
    return plan


class Eng:
    def __init__(self, plan, dev):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.e = self.rt.deserialize_cuda_engine(open(plan, "rb").read())
        self.c = self.e.create_execution_context()
        self.xs = tuple(self.e.get_tensor_shape("x"))
        self.ys = tuple(self.e.get_tensor_shape("y"))
        self.x = torch.zeros(self.xs, dtype=torch.float32, device=dev)
        self.y = torch.zeros(self.ys, dtype=torch.float32, device=dev)
        self.c.set_tensor_address("x", self.x.data_ptr())
        self.c.set_tensor_address("y", self.y.data_ptr())
        self.s = torch.cuda.Stream(dev)
        self.w = self.xs[-1]
        self.b = self.xs[0]
        self.graph = None

    def run(self):
        assert self.c.execute_async_v3(self.s.cuda_stream)

    def capture(self):
        g = torch.cuda.CUDAGraph()
        self.run()
        self.s.synchronize()
        with torch.cuda.graph(g, stream=self.s):
            self.run()
        self.graph = g

    def go(self):
        if self.graph is not None:
            with torch.cuda.stream(self.s):
                self.graph.replay()
        else:
            self.run()


def cmd_time(a):
    dev = torch.device("cuda:0")
    e = Eng(a.plan, dev)
    if a.graph:
        e.capture()
    for _ in range(3):
        e.go()
    e.s.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record(e.s)
    for _ in range(a.n):
        e.go()
    t1.record(e.s)
    t1.synchronize()
    ms = t0.elapsed_time(t1) / a.n
    w, b = e.w, e.b
    st = w - 48
    print(json.dumps({"plan": os.path.basename(a.plan), "window": w, "batch": b, "graph": bool(a.graph),
                      "ms_per_launch": round(ms, 2), "ms_per_window": round(ms / b, 2),
                      "Mvox_s_window": round(b * w ** 3 / ms / 1e3, 1),
                      "Mvox_s_at_halo24": round(b * st ** 3 / ms / 1e3, 1),
                      "mem_MiB": torch.cuda.max_memory_allocated() >> 20}), flush=True)


def box_pred(ct, eng, window, stride, dev):
    """Blended fg probability over a CT box (its own even grid, Gaussian normalised), float32."""
    shp = ct.shape
    ss = [M.starts(n, window, stride) for n in shp]
    g = torch.from_numpy(M.gauss1(window).astype(np.float32)).to(dev)
    g3 = g[:, None, None] * g[None, :, None] * g[None, None, :]
    acc = torch.zeros(shp, dtype=torch.float32, device=dev)
    wsum = torch.zeros(shp, dtype=torch.float32, device=dev)
    cg = torch.from_numpy(np.ascontiguousarray(ct)).to(dev)
    for z in ss[0]:
        for y in ss[1]:
            for x in ss[2]:
                win = cg[z:z + window, y:y + window, x:x + window]
                sl = (slice(z, z + window), slice(y, y + window), slice(x, x + window))
                wsum[sl] += g3
                if not bool(win.any()):
                    continue
                with torch.cuda.stream(eng.s):
                    eng.x[0, 0].copy_(win)
                    eng.x.clamp_(M.CLO, M.CHI).sub_(M.MEAN).div_(M.STD)
                    eng.run()
                    p = torch.sigmoid(eng.y[0, 1] - eng.y[0, 0])
                    acc[sl] += p * g3
                eng.s.synchronize()
    return (acc / wsum.clamp_min(1e-12)).cpu().numpy(), len(ss[0]) * len(ss[1]) * len(ss[2])


def cmd_gate(a):
    from concurrent.futures import ThreadPoolExecutor
    dev = torch.device("cuda:0")
    vols = {v["stem"]: v for v in M.load_vols()}
    v = vols[a.vol]
    codec = M.Codec()
    ct_src = M.CTSource(v, codec, ThreadPoolExecutor(16))
    c = np.array([int(x) for x in a.center.split(",")])
    lo, hi = c - a.half, c + a.half
    ct = ct_src.block(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    fg = ct > 0
    m = a.margin                                 # score the inner box only (no box-edge effects)
    inner = (slice(m, -m),) * 3
    rows = []
    ref = None
    for cfg in a.configs.split(","):
        plan, window, stride = cfg.split(":")
        window, stride = int(window), int(stride)
        eng = Eng(plan, dev)
        t = time.time()
        p, nwin = box_pred(ct, eng, window, stride, dev)
        el = time.time() - t
        p = np.where(fg, p, 0)[inner]
        f = fg[inner]
        if ref is None:
            ref = p
        a_, b_ = p >= 0.5, ref >= 0.5
        dice = 2 * (a_ & b_).sum() / max(1, a_.sum() + b_.sum())
        corr = float(np.corrcoef(p[f], ref[f])[0, 1])
        mad = float(np.abs(p[f] - ref[f]).mean() * 255)
        rows.append({"plan": os.path.basename(plan), "window": window, "stride": stride,
                     "step": round(stride / window, 3), "windows": nwin, "s": round(el, 1),
                     "dice_vs_ref": round(float(dice), 4), "corr_vs_ref": round(corr, 5), "mad_255": round(mad, 3),
                     "fg_frac": round(float(a_.mean()), 5)})
        print(json.dumps(rows[-1]), flush=True)
        del eng
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--window", type=int, default=192)
    b.add_argument("--batch", type=int, default=1)
    b.add_argument("--level", type=int, default=3)
    b.add_argument("--tag", default="")
    t = sub.add_parser("time")
    t.add_argument("--plan", required=True)
    t.add_argument("--n", type=int, default=20)
    t.add_argument("--graph", action="store_true")
    g = sub.add_parser("gate")
    g.add_argument("--vol", required=True)
    g.add_argument("--center", required=True)
    g.add_argument("--half", type=int, default=320)
    g.add_argument("--margin", type=int, default=96)
    g.add_argument("--configs", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.window, a.batch, a.level, a.tag)
    elif a.cmd == "time":
        cmd_time(a)
    else:
        cmd_gate(a)


if __name__ == "__main__":
    main()
