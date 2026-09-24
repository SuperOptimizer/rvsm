"""The producer's pass order over one window: the passes a sampler worker is blocked on come first."""
import os

from rvsm.run import _gpu_order


def test_blocking_passes_first_walk_order_kept():
    units = [((0, 0, 0), "verso"), ((0, 0, 1), "verso"), ((0, 0, 2), "teacher"),
             ((0, 0, 3), "verso"), ((0, 0, 4), "teacher"), ((0, 0, 5), "self")]
    got = _gpu_order(units)
    assert [u[1] for u in got] == ["teacher", "teacher", "self", "verso", "verso", "verso"]
    assert [u[0] for u in got if u[1] == "verso"] == [(0, 0, 0), (0, 0, 1), (0, 0, 3)]
    assert [u[0] for u in got if u[1] != "verso"] == [(0, 0, 2), (0, 0, 4), (0, 0, 5)]


def test_empty_and_single():
    assert _gpu_order([]) == []
    assert _gpu_order([((1, 2, 3), "verso")]) == [((1, 2, 3), "verso")]


# --------------------------------------------------------------------------- the GPU gate (fields vs passes)

import threading  # noqa: E402
import time  # noqa: E402

from rvsm import run as RUN  # noqa: E402


class _Fields(threading.Thread):
    """A fake GPU fields job: `n` batches of `dt` s each under `gate.fields_hold`, calling its yield
    point between batches (as `targets._fields_torch` does); records when each batch ran."""

    def __init__(self, gate, n=40, dt=0.02, since=None):
        super().__init__(daemon=True)
        self.gate, self.n, self.dt = gate, n, dt
        self.since = time.time() if since is None else since
        self.runs, self.started = [], threading.Event()

    def run(self):
        with self.gate.fields_hold(self.since, tag={"region": [0, 0, 0]}) as h:
            self.started.set()
            for i in range(self.n):
                if i:
                    h.yield_point(lambda: None)
                assert self.gate.holds()
                t = time.time()
                time.sleep(self.dt)
                self.runs.append((t, time.time()))


def _gate(**kw):
    logs = []
    return RUN.GpuGate(RUN.TracedLock("gpu"), log=logs.append, **kw), logs


def test_a_blocking_pass_waits_at_most_one_fields_batch():
    """Fields in flight, then a teacher pass arrives: the fields yield at their next batch boundary, the
    pass gets the card at once, no fields batch overlaps it, the fields resume after it and log the
    yield."""
    gate, logs = _gate()
    f = _Fields(gate, n=40, dt=0.02)
    f.start()
    assert f.started.wait(5)
    time.sleep(0.1)
    gate.set_pending(["teacher", "verso"])
    t0 = time.time()
    gate.pass_acquire("teacher")
    waited = time.time() - t0
    p0 = time.time()
    time.sleep(0.2)                                 # the pass
    p1 = time.time()
    gate.set_pending(["verso"])                     # only a verso left: the fields' job is young ...
    gate.release()
    time.sleep(0.2)
    assert not any(a < p1 and b > p0 for a, b in f.runs), "a fields batch ran beside the pass"
    n_mid = len(f.runs)
    assert n_mid < 40 and f.is_alive(), "the fields must not resume while a verso is pending"
    gate.set_pending(())                            # the window is drained
    f.join(10)
    assert not f.is_alive() and len(f.runs) == 40
    assert waited < 0.5, f"the teacher pass waited {waited:.2f} s behind the fields"
    y = [r for r in logs if r.get("kind") == "fields_yield"]
    assert len(y) == 1 and y[0]["for"] == ["teacher"] and y[0]["region"] == [0, 0, 0]
    assert gate.lock.holder() is None


def test_fields_do_not_start_while_a_gpu_unit_is_pending():
    """No fields while a blocking pass is pending; while only verso passes are pending, not until the
    fields job has waited FIELDS_DEFER_S (a clock here), and then it runs to the end without yielding to
    the verso passes."""
    now = [1000.0]
    gate, logs = _gate(defer_s=600.0, now=lambda: now[0])
    gate.set_pending(["self"])
    f = _Fields(gate, n=5, dt=0.01, since=1000.0)
    f.start()
    assert not f.started.wait(0.3)
    gate.set_pending(["verso", "verso"])            # the blocking pass is done; young fields still wait
    assert not f.started.wait(0.3)
    now[0] = 1000.0 + 601.0                         # the fields job is old: it may run beside verso work
    assert f.started.wait(6.0)
    f.join(5)
    assert len(f.runs) == 5 and not [r for r in logs if r.get("kind") == "fields_yield"]


def test_a_pass_that_is_not_blocking_does_not_make_the_fields_yield():
    """A verso pass registered while old fields run waits for them (verso is never blocking)."""
    gate, logs = _gate(defer_s=0.0)
    f = _Fields(gate, n=10, dt=0.02)
    f.start()
    assert f.started.wait(5)
    gate.set_pending(["verso"])
    gate.pass_acquire("verso")
    assert not f.is_alive() and len(f.runs) == 10    # it got the card only when the fields were done
    gate.set_pending(())
    gate.release()
    assert not [r for r in logs if r.get("kind") == "fields_yield"]


# --------------------------------------------------------------------------- the producer's VRAM report

def test_vram_report_splits_the_card_between_bank_student_and_passes(tmp_path):
    """`vram_report`: allocated / reserved, the bank's and the student's own parameter + buffer bytes,
    and each job kind's last pass peak; nothing at all without CUDA (cap None)."""
    import json
    import types

    import pytest
    import torch
    out = str(tmp_path)
    assert RUN.vram_report(out, None, None, {}, None, "start") is None
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    t = torch.nn.Conv3d(4, 8, 3).cuda()                     # 8*4*27 + 8 floats
    t.register_buffer("b", torch.zeros(1000, device="cuda"))
    s = torch.nn.Linear(256, 256).cuda()
    bank = RUN.TeacherBank.__new__(RUN.TeacherBank)
    bank.items = [["recto", None, "x", t], ["m7", None, "y", None]]    # m7 not loaded yet
    slot = types.SimpleNamespace(st=types.SimpleNamespace(raw=s))
    RUN._cuda_peak(reset=True)
    x = torch.zeros(64, 1 << 20, device="cuda")                       # a 256 MB pass
    del x
    peak = RUN._cuda_peak()
    assert peak >= 256 << 20
    rec = RUN.vram_report(out, bank, slot, {"teacher": peak}, 8 << 30, "periodic")
    g = float(1 << 30)
    assert rec["bank_gb"] == round(((8 * 4 * 27 + 8) * 4 + 4000) / g, 3) and rec["bank_loaded"]
    assert rec["student_gb"] == round((256 * 256 + 256) * 4 / g, 3)
    assert rec["peak_gb"]["teacher"] >= 0.25 and rec["cap_gb"] == 8.0
    assert rec["allocated_gb"] <= rec["reserved_gb"]
    with open(os.path.join(out, "logs", "produce.jsonl")) as f:
        got = [json.loads(line) for line in f]
    assert got[-1]["kind"] == "vram_report" and got[-1]["why"] == "periodic"
    assert RUN.module_bytes(None) == 0


# --------------------------------------------------------------------------- the teacher bank off the card

def _fake_bank(dev, backend="torch"):
    import torch
    bank = RUN.TeacherBank.__new__(RUN.TeacherBank)
    bank.cfg, bank.out, bank.device, bank.backend = None, "", dev, backend
    bank.fast, bank.on_device, bank._host = {}, True, {}
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv3d(1, 64, 3, padding=1), torch.nn.BatchNorm3d(64),
                              torch.nn.Conv3d(64, 64, 3, padding=1)).eval()
    bank.items = [["recto", None, "x", net.to(dev)], ["m7", None, "y", None]]
    return bank


def test_the_teacher_bank_parks_its_weights_in_host_memory_and_back():
    """`offload` gives the weights' VRAM back (the parameters are the same objects, now pinned host
    tensors) and a compiled forward runs again after `onload` WITHOUT a recompile and with the same
    output; a second cycle reuses the host copy. `probs_u8` onloads by itself."""
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from torch._dynamo.utils import counters
    bank = _fake_bank("cuda")
    net = bank.items[0][3]
    fn = torch.compile(net, dynamic=False)
    x = torch.rand(1, 1, 16, 16, 16, device="cuda")
    with torch.no_grad():
        y0 = fn(x)
    g0 = int(counters["stats"]["unique_graphs"])
    params = list(net.parameters())
    fp = bank.footprint()
    assert fp == RUN.module_bytes(net) > 0
    for _ in range(2):
        before = torch.cuda.memory_allocated()
        s = bank.offload()
        assert s is not None and not bank.on_device and bank.footprint() == 0
        assert all(p.device.type == "cpu" and p.is_pinned() for p in net.parameters())
        assert before - torch.cuda.memory_allocated() >= fp        # the weights' VRAM is given back
        assert bank.offload() is None                    # already parked
        assert bank.onload() is not None and bank.on_device
        assert bank.onload() is None
        assert [id(p) for p in net.parameters()] == [id(p) for p in params]
        with torch.no_grad():
            y1 = fn(x)
        assert torch.equal(y0, y1)
    assert int(counters["stats"]["unique_graphs"]) == g0, "the compiled teacher recompiled"
    assert len(bank._host["recto"]) == len(list(net.parameters())) + len(list(net.buffers()))


def test_the_bank_is_parked_only_when_no_teacher_pass_is_left(tmp_path, monkeypatch):
    """`_bank_park`: back on the card before a teacher pass, off it before a verso / self pass once no
    teacher unit is left in the window's pass, and left alone while one is; a TensorRT or CPU bank never
    moves."""
    calls = []

    class B:
        def onload(self):
            calls.append("on")
            return 0.1

        def offload(self):
            calls.append("off")
            return 0.2
    out = str(tmp_path)
    b = B()
    assert RUN._bank_park(out, b, "verso", [((0, 0, 0), "verso"), ((0, 0, 1), "teacher")]) is None
    assert calls == []
    assert RUN._bank_park(out, b, "teacher", [((0, 0, 1), "teacher")]) == 0.1
    assert RUN._bank_park(out, b, "verso", [((0, 0, 2), "verso")]) == 0.2
    assert calls == ["on", "off"]
    import json
    with open(os.path.join(out, "logs", "produce.jsonl")) as f:
        kinds = [json.loads(line)["kind"] for line in f]
    assert kinds == ["bank_onload", "bank_offload"]
    assert not _fake_bank("cpu").movable()
    assert _fake_bank("cpu").offload() is None
    import torch
    if torch.cuda.is_available():
        assert not _fake_bank("cuda", backend="trt").movable()
        monkeypatch.setattr(RUN, "TEACHER_OFFLOAD", False)
        assert not _fake_bank("cuda").movable()
