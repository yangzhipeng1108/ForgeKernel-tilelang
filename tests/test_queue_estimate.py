"""Tests for server-mode queue estimation (F4) — no GPU.

The server window reads service + queueing; the device window reads ~service. Little's law
(together with the offered rate) splits them. These pin the arithmetic and the qualitative
verdicts (saturated / queue-bound / service-bound) and that a missing input never fabricates a
zero (an unmeasured split must not read as "no queue").
"""

from __future__ import annotations

import pytest

from forgekernel.timing_audit import (
    QueueEstimate,
    estimate_queue,
    queue_estimate_of,
    queue_note,
)


class _Rec:
    def __init__(self, verdict="KEEP", vs_ref_server=None, device_median_ms=None,
                 server_median_ms=None, server_qps=None):
        self.verdict = verdict
        self.vs_ref_server = vs_ref_server
        self.device_median_ms = device_median_ms
        self.server_median_ms = server_median_ms
        self.server_qps = server_qps


def test_utilization_is_rate_times_service():
    # service 2 ms = 0.002 s, offered 100 req/s -> rho = 0.2
    est = estimate_queue(service_ms=2.0, server_median_ms=3.0, server_qps=100.0)
    assert est.utilization == pytest.approx(0.2)
    assert est.queue_wait_ms == pytest.approx(1.0)


def test_wait_clamped_at_zero_on_noise():
    # server median below device median is cross-window noise, not a negative wait
    est = estimate_queue(service_ms=5.0, server_median_ms=4.0, server_qps=10.0)
    assert est.queue_wait_ms == 0.0


def test_missing_inputs_stay_none():
    est = estimate_queue(service_ms=None, server_median_ms=None, server_qps=None)
    assert est.utilization is None
    assert est.queue_wait_ms is None
    assert not est.measured
    # service present but no rate -> no utilisation; still no wait without server median
    est2 = estimate_queue(service_ms=2.0, server_median_ms=None, server_qps=None)
    assert est2.utilization is None
    assert est2.queue_wait_ms is None


def test_saturated_when_rho_at_or_above_one():
    # service 10 ms, 150 req/s -> rho = 1.5
    est = estimate_queue(service_ms=10.0, server_median_ms=50.0, server_qps=150.0)
    assert est.utilization == pytest.approx(1.5)
    assert est.saturated
    assert "SATURATED" in queue_note(est)


def test_queue_bound_when_wait_exceeds_service():
    # service 2 ms, server median 8 ms -> wait 6 ms > service
    est = estimate_queue(service_ms=2.0, server_median_ms=8.0, server_qps=200.0)
    assert est.queue_bound
    note = queue_note(est)
    assert "queue-bound" in note
    assert "wait" in note


def test_service_bound_when_wait_small():
    est = estimate_queue(service_ms=10.0, server_median_ms=10.5, server_qps=50.0)
    assert not est.queue_bound
    assert not est.saturated
    assert "service-bound" in queue_note(est)


def test_note_empty_when_unmeasured():
    assert queue_note(QueueEstimate(None, None, None, None, None)) == ""


def test_queue_estimate_of_picks_best_server_keep():
    recs = [
        _Rec(verdict="KEEP", vs_ref_server=1.1, device_median_ms=2.0,
             server_median_ms=3.0, server_qps=100.0),
        _Rec(verdict="KEEP", vs_ref_server=1.3, device_median_ms=4.0,
             server_median_ms=6.0, server_qps=80.0),
        _Rec(verdict="REVERT", vs_ref_server=1.9, device_median_ms=1.0,
             server_median_ms=1.0, server_qps=10.0),
    ]
    est = queue_estimate_of(recs)
    # best KEEP is the 1.3 -> service 4.0, server 6.0, qps 80
    assert est.service_ms == 4.0
    assert est.server_median_ms == 6.0
    assert est.server_qps == 80.0
    assert est.utilization == pytest.approx(80.0 * 0.004)


def test_queue_estimate_of_none_when_no_server_keep():
    recs = [_Rec(verdict="KEEP", vs_ref_server=None), _Rec(verdict="REVERT", vs_ref_server=1.5)]
    est = queue_estimate_of(recs)
    assert not est.measured
    assert est.service_ms is None
