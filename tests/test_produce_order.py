"""The producer's pass order over one window: the passes a sampler worker is blocked on come first."""
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
