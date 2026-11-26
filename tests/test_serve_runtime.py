"""Tests for the production runtime — no real server, no GPU.

Every function is driven through an injected runner / popen / http_get / clock, so the
launch-wait, bench-run-and-parse, gate-dispatch and one-shot orchestration logic are all tested
on the CPU. The only untested leaves are the real ``subprocess.Popen`` of the model and the real
CUDA work behind the gate callables, which need hardware.
"""

from __future__ import annotations

import types

import pytest

from forgekernel.serve import GATE_GRAPH, GATE_PAGED, GATE_PARITY
from forgekernel.serve.harness import ServeConfig
from forgekernel.serve.runtime import (
    BenchResult,
    ServerHandle,
    make_gate_runners,
    run_bench,
    run_checks,
    run_deployment,
    start_server,
    wait_for_health,
)


def _cfg(**kw) -> ServeConfig:
    base = dict(model_path="/models/m", serve_flags=["--tp", "2"], port=30000)
    base.update(kw)
    return ServeConfig(**base)


# ---------------------------------------------------------------- server launch + health

class _Proc:
    def __init__(self, exit_after=None):
        self._polled = 0
        self.stopped = False
        self._exit_after = exit_after

    def poll(self):
        self._polled += 1
        if self._exit_after is not None and self._polled > self._exit_after:
            return 1
        return None

    def terminate(self):
        self.stopped = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.stopped = True


def test_start_server_launches_and_returns_handle_without_wait():
    seen = {}

    def fake_popen(cmd, env):
        seen["cmd"] = cmd
        seen["env"] = env
        return _Proc()

    h = start_server(_cfg(env={"X": "1"}), popen=fake_popen, wait_healthy=False)
    assert isinstance(h, ServerHandle)
    assert h.base_url == "http://127.0.0.1:30000"
    assert seen["cmd"][:2] == ["sglang", "serve"]
    assert seen["env"]["X"] == "1"


def test_wait_for_health_returns_when_200():
    # fails twice then succeeds
    calls = {"n": 0}

    def http_get(url):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("not up")
        return 200

    ticks = iter(range(1000))
    slept = {"n": 0}

    def sleep(s):
        slept["n"] += 1

    wait_for_health("http://127.0.0.1:30000", http_get=http_get,
                    clock=lambda: next(ticks), sleep=sleep)
    # two failed attempts -> two sleeps before the third (successful) attempt
    assert slept["n"] == 2


def test_wait_for_health_raises_on_timeout():
    def http_get(url):
        raise ConnectionError("never up")
    ticks = iter(range(1000))
    with pytest.raises(TimeoutError):
        wait_for_health("http://x", http_get=http_get, clock=lambda: next(ticks),
                        sleep=lambda s: None, timeout_s=10)


def test_wait_for_health_fails_fast_if_server_dead():
    def http_get(url):
        raise ConnectionError("x")
    proc = _Proc(exit_after=0)   # poll() returns 1 immediately
    with pytest.raises(RuntimeError) as ei:
        wait_for_health("http://x", http_get=http_get, sleep=lambda s: None, proc=proc)
    assert "exited" in str(ei.value)


# ---------------------------------------------------------------- bench run + parse

def test_run_bench_default_parses_and_ok():
    def runner(argv):
        assert argv[:3] == ["python3", "-m", "sglang.bench_serving"]
        return 0, '{"output_throughput": 730.0}'
    r = run_bench(_cfg(), concurrency=16, runner=runner)
    assert r.ok and r.summary["output_throughput"] == 730.0


def test_run_bench_nonzero_exit_keeps_stdout_but_not_ok():
    r = run_bench(_cfg(), concurrency=16, runner=lambda a: (1, '{"output_throughput": 0}'))
    assert not r.ok
    assert r.returncode == 1


def test_run_bench_empty_output_is_not_ok():
    r = run_bench(_cfg(), concurrency=16, runner=lambda a: (0, ""))
    assert not r.ok


# ---------------------------------------------------------------- gate dispatch

def test_run_checks_only_named_gates_unmeasured_is_none():
    cfg = _cfg(run_gates=(GATE_GRAPH, GATE_PAGED, GATE_PARITY))
    runners = {GATE_GRAPH: lambda h, c: True}       # only graph has a runner
    out = run_checks(ServerHandle("http://x"), cfg, gate_runners=runners)
    assert out[GATE_GRAPH] is True
    assert out[GATE_PAGED] is None
    assert out[GATE_PARITY] is None


def test_run_checks_gate_none_stays_none():
    cfg = _cfg(run_gates=(GATE_PAGED,))
    out = run_checks(ServerHandle("http://x"), cfg,
                     gate_runners={GATE_PAGED: lambda h, c: None})
    assert out[GATE_PAGED] is None


def test_make_gate_runners_skips_absent_inputs():
    cfg = _cfg(run_gates=(GATE_GRAPH, GATE_PAGED, GATE_PARITY))
    runners = make_gate_runners(cfg, capture_fn=None, kernel_inputs=None,
                                kernel_out=None, server_out=None)
    assert runners == {}
    runners2 = make_gate_runners(cfg, kernel_inputs={"block_table": object(), "page_size": 32})
    assert GATE_PAGED in runners2 and GATE_GRAPH not in runners2


# ---------------------------------------------------------------- one-shot run

def test_run_deployment_sweeps_and_stops_server():
    procs = {"n": 0}

    def start(cmd, env):
        procs["n"] += 1
        return ServerHandle("http://127.0.0.1:30000", proc=_Proc(), own=True)

    def bench(cfg, concurrency):
        return 0, '{"output_throughput": %d}' % concurrency

    cfg = _cfg(concurrencies=(16, 32))
    out = run_deployment(cfg, start=start, bench=bench)
    assert procs["n"] == 1
    assert [r.concurrency for r in out["readings"]] == [16, 32]
    assert [r.tok_s for r in out["readings"]] == [16.0, 32.0]


def test_run_deployment_with_gates():
    cfg = _cfg(concurrencies=(16,), run_gates=(GATE_GRAPH, GATE_PAGED, GATE_PARITY))
    runners = {GATE_GRAPH: lambda h, c: True, GATE_PAGED: lambda h, c: True,
               GATE_PARITY: lambda h, c: True}
    out = run_deployment(cfg,
                         start=lambda cmd, env: ServerHandle("http://x", proc=_Proc()),
                         bench=lambda cfg, c: (0, '{"output_throughput": 100}'),
                         gate_runners=runners)
    r = out["readings"][0]
    assert r.gates == {GATE_GRAPH: True, GATE_PAGED: True, GATE_PARITY: True}
    assert r.deployable


def test_run_deployment_base_url_override_does_not_start():
    started = {"n": 0}

    def start(cmd, env):
        started["n"] += 1
        return ServerHandle("http://y")

    out = run_deployment(_cfg(concurrencies=(16,)), start=start,
                         bench=lambda cfg, c: (0, '{"output_throughput": 5}'),
                         base_url_override="http://127.0.0.1:30000")
    assert started["n"] == 0
    assert out["readings"][0].tok_s == 5.0
