"""Tests for the three deploy gates — no GPU.

The gate *logic* (pass/fail/None, the parity threshold, the paged-layout shape check, capture
"raised means fail") is pure; the real capture/run/compare hides behind injected callables. So
every test here runs on the CPU with fakes, and only the default GPU callables need hardware.
"""

from __future__ import annotations

import types

import pytest

from forgekernel.serve import GATE_GRAPH, GATE_PAGED, GATE_PARITY, PARITY_MIN_COS
from forgekernel.serve.checks import (
    GateResult,
    graph_capturable,
    make_graph_capture,
    paged_kv_ok,
    parity_ok,
    run_all_checks,
)


# ---------------------------------------------------------------- cuda graph

def test_graph_passes_when_capture_does_not_raise():
    r = graph_capturable(lambda: None)
    assert r.name == GATE_GRAPH and r.passed is True


def test_graph_fails_when_capture_raises():
    def boom():
        raise RuntimeError("cudaErrorStreamCaptureInvalidated")
    r = graph_capturable(boom)
    assert r.passed is False
    assert "StreamCapture" in r.detail


def test_make_graph_capture_warmups_outside_and_replays():
    calls = {"n": 0, "captured": 0}

    class FakeGraph:
        def __init__(self):
            self.replayed = False
        def replay(self):
            self.replayed = True
            calls["n"] += 1

    class FakeCtx:
        def __enter__(self):
            calls["captured"] += 1
        def __exit__(self, *a):
            return False

    fake_cuda = types.SimpleNamespace(
        synchronize=lambda: None,
        CUDAGraph=lambda: FakeGraph(),
        graph=lambda g: FakeCtx(),
    )
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    fn_calls = {"n": 0}
    cap = make_graph_capture(lambda: fn_calls.__setitem__("n", fn_calls["n"] + 1), (),
                             torch=fake_torch)
    cap()
    # warmup + capture body call fn twice; one replay
    assert fn_calls["n"] == 2
    assert calls["captured"] == 1
    assert calls["n"] == 1


# ---------------------------------------------------------------- paged kv

def test_paged_passes_with_block_table_and_page_size():
    r = paged_kv_ok({"block_table": object(), "page_size": 32})
    assert r.passed is True


def test_paged_fails_without_block_table():
    r = paged_kv_ok({"kv": object(), "page_size": 32})
    assert r.passed is False
    assert "dense KV" in r.detail


def test_paged_fails_on_bad_page_size():
    r = paged_kv_ok({"block_table": object(), "page_size": 0})
    assert r.passed is False


def test_paged_accepts_an_object_with_attributes():
    inputs = types.SimpleNamespace(block_table=object(), page_size=16)
    assert paged_kv_ok(inputs).passed is True


def test_paged_rejects_non_2d_block_table():
    bt = types.SimpleNamespace(dim=lambda: 1)
    r = paged_kv_ok({"block_table": bt, "page_size": 32})
    assert r.passed is False


# ---------------------------------------------------------------- parity

def test_parity_passes_above_threshold():
    r = parity_ok("a", "b", cosine_fn=lambda a, b: 0.9999)
    assert r.passed is True


def test_parity_fails_below_threshold():
    r = parity_ok("a", "b", cosine_fn=lambda a, b: 0.5)
    assert r.passed is False


def test_parity_exactly_at_threshold_passes():
    r = parity_ok("a", "b", cosine_fn=lambda a, b: PARITY_MIN_COS)
    assert r.passed is True


def test_parity_undefined_cosine_fails_not_passes():
    r = parity_ok("a", "b", cosine_fn=lambda a, b: None)
    assert r.passed is False


# ---------------------------------------------------------------- bundle

def test_run_all_checks_unmeasured_gates_are_none():
    out = run_all_checks(None)                       # nothing supplied
    assert out == {GATE_GRAPH: None, GATE_PAGED: None, GATE_PARITY: None}


def test_run_all_checks_all_green():
    out = run_all_checks({"block_table": object(), "page_size": 32},
                         capture_fn=lambda: None,
                         kernel_out="k", server_out="s",
                         cosine_fn=lambda a, b: 1.0)
    assert out == {GATE_GRAPH: True, GATE_PAGED: True, GATE_PARITY: True}


def test_run_all_checks_mixed():
    out = run_all_checks({"block_table": object(), "page_size": 32},
                         capture_fn=lambda: (_ for _ in ()).throw(RuntimeError("no")),
                         kernel_out="k", server_out="s",
                         cosine_fn=lambda a, b: 0.5)
    assert out[GATE_GRAPH] is False
    assert out[GATE_PAGED] is True
    assert out[GATE_PARITY] is False


def test_gate_result_dict():
    assert GateResult(GATE_PAGED, True).as_dict() == {GATE_PAGED: True}
