"""timing.py's request-driven ("server") window: the arrival process, not a batch."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PATH = (Path(__file__).resolve().parents[1]
        / "src" / "forgekernel" / "card_template" / "eval" / "timing.py")


def _load(monkeypatch, fake_torch):
    """timing.py imports torch at module scope; load it against a stub."""
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    spec = importlib.util.spec_from_file_location("timing_under_test", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeTorch:
    class cuda:
        @staticmethod
        def synchronize():
            pass

        class Event:
            def __init__(self, enable_timing=False):
                pass

            def record(self):
                pass

            def elapsed_time(self, other):
                return 1.0


def test_time_server_samples_one_per_request_seamlessly(monkeypatch):
    timing = _load(monkeypatch, _FakeTorch())
    calls = []

    def fn():
        calls.append(1)

    samples = timing.time_server(fn, qps=200, warmup_seconds=0.0, benchmark_seconds=0.15)
    assert len(samples) == 30          # 0.15 s * 200 qps
    assert len(calls) == 30
    assert all(s >= 0 for s in samples)


def test_time_server_refuses_a_nonpositive_rate(monkeypatch):
    timing = _load(monkeypatch, _FakeTorch())
    for rate in (0, -1):
        with pytest.raises(ValueError):
            timing.time_server(lambda: None, qps=rate)


def test_server_window_is_off_by_default(monkeypatch):
    timing = _load(monkeypatch, _FakeTorch())
    assert timing.SERVER_QPS is None
    assert timing.server_window(lambda: None) is None
    assert timing.server_reading(lambda: None) is None


def test_server_window_reports_rate_and_median(monkeypatch):
    timing = _load(monkeypatch, _FakeTorch())
    block = timing.server_window(lambda: None, qps=100)
    assert block["server_qps"] == 100
    assert block["server_median_ms"] >= 0
    assert block["server_samples_ms"] == 300


def test_mirrored_averages_the_server_reading(monkeypatch):
    timing = _load(monkeypatch, _FakeTorch())
    first = {"median_ms": 2.0, "min_ms": 1.9, "device_median_ms": 1.0,
             "host_enqueue_ms": 0.1, "server_median_ms": 3.0, "server_qps": 100}
    second = {"median_ms": 2.1, "min_ms": 2.0, "device_median_ms": 1.2,
              "host_enqueue_ms": 0.1, "server_median_ms": 5.0, "server_qps": 100}
    out = timing._mirrored(first, second)
    assert out["server_median_ms"] == pytest.approx(4.0)
    assert out["server_readings_ms"] == [3.0, 5.0]
    assert out["device_median_ms"] == pytest.approx(1.1)


def test_mirrored_leaves_server_absent_when_unmeasured(monkeypatch):
    timing = _load(monkeypatch, _FakeTorch())
    first = {"median_ms": 2.0, "min_ms": 1.9, "device_median_ms": 1.0, "host_enqueue_ms": 0.1}
    second = {"median_ms": 2.0, "min_ms": 1.9, "device_median_ms": 1.0, "host_enqueue_ms": 0.1}
    out = timing._mirrored(first, second)
    assert "server_median_ms" not in out


def test_paired_performance_emits_server_keys_only_when_measured(monkeypatch):
    timing = _load(monkeypatch, _FakeTorch())

    def side(dev, srv=None):
        d = {"median_ms": dev * 2, "min_ms": dev * 2, "device_median_ms": dev,
             "host_enqueue_ms": dev * 0.1, "host_bound_ratio": 0.1,
             "device_readings_ms": [dev, dev]}
        if srv is not None:
            d.update({"server_median_ms": srv, "server_qps": 100,
                      "server_readings_ms": [srv, srv]})
        return d

    off = {"kernel": side(1.0), "ref": side(1.2), "ref_error": None}
    perf = timing.paired_performance(off)
    assert "server_median_ms" not in perf and "vs_ref_server" not in perf

    on = {"kernel": side(1.0, 1.5), "ref": side(1.2, 3.0), "ref_error": None}
    perf = timing.paired_performance(on)
    assert perf["server_median_ms"] == 1.5
    assert perf["ref_server_ms"] == 3.0
    assert perf["vs_ref_server"] == pytest.approx(2.0)
