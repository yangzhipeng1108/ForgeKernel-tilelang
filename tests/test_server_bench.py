"""Tests for the server-mode bench collector (A2) — no GPU.

``collect`` keeps the two CUDA windows behind injected callables, so every test supplies fake
sample lists and verifies the orchestration: medians are the reading, the server window is
skipped without an offered rate, a missing window stays None (never a zero), student/reference
ratios use ref/student (>1 = faster), and the record fields carry only measured keys. The real
GPU path (``collect_gpu``) is the thin adapter over card_template timing; it is exercised by
shape only (imports torch lazily), not run here.
"""

from __future__ import annotations

import pytest

from forgekernel.server_bench import BenchReading, collect
from forgekernel.timing_audit import estimate_queue, queue_estimate_of


def test_collect_medians_are_the_reading():
    r = collect(device_fn=lambda: [1.0, 2.0, 3.0],
                server_fn=lambda: [4.0, 6.0, 8.0], qps=100.0)
    assert r.device_median_ms == 2.0
    assert r.device_samples_ms == 3
    assert r.server_median_ms == 6.0
    assert r.server_samples_ms == 3
    assert r.server_qps == 100.0


def test_server_window_skipped_without_qps():
    called = {"server": False}

    def server_fn():
        called["server"] = True
        return [1.0]

    r = collect(device_fn=lambda: [2.0], server_fn=server_fn, qps=None)
    assert r.server_median_ms is None
    assert r.server_qps is None
    assert r.server_samples_ms == 0
    assert called["server"] is False   # never run at a fabricated rate


def test_server_window_skipped_on_nonpositive_qps():
    r = collect(device_fn=lambda: [2.0], server_fn=lambda: [5.0], qps=0.0)
    assert r.server_median_ms is None


def test_missing_window_stays_none_not_zero():
    r = collect(device_fn=lambda: [3.0])
    assert r.device_median_ms == 3.0
    assert r.server_median_ms is None
    fields = r.record_fields()
    assert "server_median_ms" not in fields       # unmeasured keys absent, not zero
    assert fields["device_median_ms"] == 3.0


def test_empty_samples_yield_none():
    r = collect(device_fn=lambda: [], server_fn=lambda: [], qps=10.0)
    assert r.device_median_ms is None
    assert r.server_median_ms is None
    assert r.device_samples_ms == 0


def test_vs_ref_ratio_is_ref_over_student():
    r = collect(device_fn=lambda: [2.0], ref_device_fn=lambda: [4.0],
                server_fn=lambda: [8.0], ref_server_fn=lambda: [8.0], qps=50.0)
    # student 2 ms vs ref 4 ms -> ref/student = 2.0 (student 2x faster)
    assert r.vs_ref_device == pytest.approx(2.0)
    assert r.vs_ref_server == pytest.approx(1.0)


def test_ratio_none_when_either_side_missing():
    r = collect(device_fn=lambda: [2.0])   # no reference
    assert r.vs_ref_device is None
    assert r.vs_ref_server is None


def test_server_ratio_skipped_when_server_not_run():
    # reference server supplied but no qps -> student server not run; ratio must be None
    r = collect(device_fn=lambda: [2.0], server_fn=lambda: [5.0],
                ref_device_fn=lambda: [3.0], ref_server_fn=lambda: [5.0], qps=None)
    assert r.server_median_ms is None
    assert r.vs_ref_server is None


def test_record_fields_carry_only_measured_keys():
    r = collect(device_fn=lambda: [2.0], ref_device_fn=lambda: [4.0],
                server_fn=lambda: [6.0], ref_server_fn=lambda: [6.0], qps=80.0)
    f = r.record_fields()
    assert f["device_median_ms"] == 2.0
    assert f["server_median_ms"] == 6.0
    assert f["server_qps"] == 80.0
    assert f["ref_device_ms"] == 4.0
    assert f["ref_server_ms"] == 6.0
    assert f["vs_ref_device"] == pytest.approx(2.0)
    assert f["vs_ref_server"] == pytest.approx(1.0)


def test_feeds_f4_queue_estimate():
    # service 2 ms, served 8 ms at 200 req/s -> rho=0.4, wait=6 ms
    r = collect(device_fn=lambda: [2.0], server_fn=lambda: [8.0], qps=200.0)

    class Rec:
        verdict = "KEEP"
        vs_ref_server = 1.5
        device_median_ms = r.device_median_ms
        server_median_ms = r.server_median_ms
        server_qps = r.server_qps

    est = queue_estimate_of([Rec()])
    assert est.service_ms == 2.0
    assert est.server_median_ms == 8.0
    assert est.utilization == pytest.approx(0.4)
    assert est.queue_wait_ms == pytest.approx(6.0)


def test_collect_gpu_imports_lazily_and_is_callable():
    # the GPU adapter must exist and be importable without running torch; we don't invoke it
    from forgekernel.server_bench import collect_gpu
    assert callable(collect_gpu)
