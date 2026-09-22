"""TensorRT fp16 engines for a fixed-shape network, with torch as the fallback.

Any module with a fixed `(1, C, w, w, w)` input can be traded for an engine: `engine_for` exports the
module to ONNX once, builds a strongly-typed fp16 plan for THIS GPU, caches it under
`<dir>/<name>_p<w>_b1_fp16_<gpu>.plan`, and returns a callable with the module's own signature. It is
worth ~2.4x the torch bf16 throughput on the recto teacher, and it is never worth a failed run: every
import, build and load error returns None (logged once) and the caller keeps its torch module.

The build ladder exists because virtualised GPUs (Thunder, some cloud hosts) break TensorRT's autotuner:
`builder_optimization_level=0` picks tactics without timing them, and `set_tactic_sources(0)` drops the
cuBLAS/cuDNN tactic sources whose pinned-memory probes are what actually fails.
"""
from __future__ import annotations

import glob
import os
import time

import torch

_LOGGED = set()


def _once(msg):
    """Log a fallback reason once per process: an engine that cannot be built cannot be built."""
    if msg not in _LOGGED:
        _LOGGED.add(msg)
        print(f"[trt] {msg}; using torch", flush=True)


def _gpu_tag():
    try:
        return torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
    except Exception:  # noqa: BLE001
        return "cpu"


def plan(name, window, dir_, gpu=None):
    """The plan path for `name` at `window` on this GPU (existing or to-be-built)."""
    gpu = gpu or _gpu_tag()
    return os.path.join(str(dir_), f"{name}_p{int(window)}_b1_fp16_{gpu}.plan")


def _export_onnx(net, path, window, cin, device):
    import torch.onnx  # noqa: F401  (import for the side effect of a clear error when it is missing)
    x = torch.zeros((1, int(cin)) + (int(window),) * 3, dtype=torch.float32, device=device)
    tmp = path + ".tmp"
    with torch.no_grad():
        torch.onnx.export(net, (x,), tmp, input_names=["x"], output_names=["y"], opset_version=17,
                          dynamo=False)
    os.replace(tmp, path)
    return path


def build(onnx_path, plan_path, window, cin=1, batch=1, workspace_gb=8.0, level=None, no_timing=False):
    """Parse the ONNX graph with its I/O pinned to [batch, C, window^3] and build a fp16 engine.

    Strongly typed, so the precision is the graph's own. `level`: builder optimization level (0 = pick
    tactics without timing them); `no_timing` also drops the cuBLAS/cuDNN tactic sources."""
    import onnx
    import tensorrt as trt
    m = onnx.load(onnx_path, load_external_data=True)
    for vi in list(m.graph.input) + list(m.graph.output):
        d = vi.type.tensor_type.shape.dim
        if len(d) == 5:
            for i, v in enumerate([batch, d[1].dim_value or int(cin), window, window, window]):
                d[i].dim_value = int(v)
    del m.graph.value_info[:]
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, log)
    assert parser.parse(m.SerializeToString()), \
        "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if level is not None:
        cfg.builder_optimization_level = int(level)
    if no_timing:
        cfg.set_tactic_sources(0)
    t = time.time()
    ser = b.build_serialized_network(net, cfg)
    assert ser is not None, f"TensorRT build failed for {onnx_path}"
    os.makedirs(os.path.dirname(plan_path) or ".", exist_ok=True)
    open(plan_path + ".tmp", "wb").write(bytes(ser))
    os.replace(plan_path + ".tmp", plan_path)
    print(f"[trt] built {os.path.basename(plan_path)} ({ser.nbytes >> 20} MiB) in {time.time() - t:.0f}s",
          flush=True)
    return plan_path


class Engine:
    """A deserialised plan called like a module: fp32 in, fp32 out, at the one shape it was built for."""

    def __init__(self, path, dev="cuda"):
        import tensorrt as trt
        self.dev = torch.device(dev)
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.in_shape = tuple(self.engine.get_tensor_shape("x"))
        self.out_shape = tuple(self.engine.get_tensor_shape("y"))
        self.out = torch.empty(self.out_shape, dtype=torch.float32, device=self.dev)

    def __call__(self, x):
        assert tuple(x.shape) == self.in_shape, f"engine wants {self.in_shape}, got {tuple(x.shape)}"
        x = x.float().contiguous()
        self.ctx.set_tensor_address("x", x.data_ptr())
        self.ctx.set_tensor_address("y", self.out.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream(self.dev).cuda_stream)
        return self.out


def engine_for(net, name, window, cin, dir_, device="cuda", workspace_gb=8.0):
    """An `Engine` for `net` at `(1, cin, window^3)`, or None when TensorRT cannot serve it.

    An existing plan for this (name, window, GPU) is deserialised; otherwise the module is exported to
    ONNX and built, trying the plain build first and then the two ladder rungs that survive a
    virtualised GPU. EVERY failure -- no tensorrt, no onnx, no CUDA, a parser error, a bad plan -- is a
    None return with one logged line, because the caller's torch bf16 path is always available."""
    if not torch.cuda.is_available():
        _once("no CUDA device")
        return None
    try:
        import tensorrt  # noqa: F401
    except Exception as e:  # noqa: BLE001
        _once(f"tensorrt is not importable ({e.__class__.__name__}: {e})")
        return None
    dir_ = str(dir_)
    p = plan(name, window, dir_)
    if not os.path.exists(p):  # a plan built for this GPU under another driver's name still counts
        alt = sorted(glob.glob(os.path.join(dir_, f"{name}_p{int(window)}_b1_fp16_*.plan")))
        p = alt[-1] if alt else p
    try:
        if not os.path.exists(p):
            os.makedirs(dir_, exist_ok=True)
            onnx_path = os.path.join(dir_, f"{name}_p{int(window)}_b1_fp16.onnx")
            if not os.path.exists(onnx_path):
                _export_onnx(net, onnx_path, window, cin, device)
            p = plan(name, window, dir_)
            for kw in ({}, {"level": 0}, {"level": 0, "no_timing": True}):
                try:
                    build(onnx_path, p, int(window), cin=int(cin), workspace_gb=workspace_gb, **kw)
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
            else:
                raise last
        eng = Engine(p, device)
    except Exception as e:  # noqa: BLE001
        _once(f"{name} p{window}: {e.__class__.__name__}: {e}")
        return None
    try:
        x = torch.randn(eng.in_shape, dtype=torch.float32, device=device)

        def ref(t):
            with torch.no_grad():
                return net(t)
        use = verdict(p, eng, ref, x)
    except Exception as e:  # noqa: BLE001
        _once(f"{name} p{window}: the engine-vs-torch benchmark failed ({e.__class__.__name__}: {e})")
        return None
    if not use:
        _once(f"{name} p{window}: the engine is slower than torch on this GPU")
        return None
    return eng


def _time(fn, x, reps=3):
    """Median seconds of `fn(x)` after one warm-up call, synchronised."""
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    fn(x)
    sync()
    ts = []
    for _ in range(int(reps)):
        t = time.perf_counter()
        fn(x)
        sync()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def verdict(plan_path, eng, ref, x, reps=3):
    """Whether the engine at `plan_path` beats the torch module `ref` on one window, decided ONCE and
    recorded next to the plan as `<plan>.verdict.json`.

    A plan built by the untimed rungs of the ladder (optimization level 0 on a GPU whose autotuner
    cannot time tactics) runs whatever kernels it picked blind: on Thunder's A100 the recto engine came
    out 8.6x SLOWER than torch (380 s against 44 s per 1024^3 region). An engine is only worth using when
    it is faster, so it has to be measured, not assumed."""
    import json
    vp = str(plan_path) + ".verdict.json"
    if os.path.exists(vp):
        try:
            with open(vp) as f:
                return bool(json.load(f)["use"])
        except (OSError, ValueError, KeyError):
            pass
    te, tt = _time(eng, x, reps), _time(ref, x, reps)
    use = te < tt
    rec = {"use": use, "trt_s": round(te, 4), "torch_s": round(tt, 4), "ts": time.time()}
    with open(vp + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(vp + ".tmp", vp)
    print(f"[trt] {os.path.basename(str(plan_path))}: engine {te * 1e3:.0f} ms, torch {tt * 1e3:.0f} ms "
          f"per window -> {'engine' if use else 'torch'}", flush=True)
    return use
