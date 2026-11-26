"""Tests for the serving harness and reading schema — no GPU, no server.

``run_once``/``run_sweep`` keep the server behind injected callables (``start_server`` /
``run_bench`` / ``run_checks``), so every test here is pure CPU: it verifies the commands the
harness builds (and the three bench_serving traps it must not reintroduce), the summary parsing
(the last JSON line, degrade-to-None on garbage), the reading's gate logic (unchecked gate is
not a pass; ``deployable`` requires all three checked-and-passed), and the baseline comparison
(refuses a cross-concurrency tok/s ratio). The real GPU/server body is the callables' job, not
anything that decides the result.
"""

from __future__ import annotations

import json

import pytest

from forgekernel.serve import (
    GATE_GRAPH,
    GATE_PAGED,
    GATE_PARITY,
    ServingReading,
    compare_to_baseline,
)
from forgekernel.serve.harness import (
    SERVE_BASE_URL,
    STABLE_CONCURRENCY,
    ServeConfig,
    bench_command,
    parse_bench_summary,
    run_once,
    run_sweep,
    serve_command,
    shell_preview,
)


def _cfg(**kw) -> ServeConfig:
    base = dict(model_path="/models/deepseek-v4.1-flash",
                serve_flags=["--tp", "2", "--page-size", "32"])
    base.update(kw)
    return ServeConfig(**base)


# ---------------------------------------------------------------- command assembly

def test_serve_command_fixes_model_host_port_and_passes_flags():
    cmd = serve_command(_cfg())
    assert cmd[:2] == ["sglang", "serve"]
    assert "--model-path" in cmd and "/models/deepseek-v4.1-flash" in cmd
    assert cmd[cmd.index("--port") + 1] == "30000"
    assert "--page-size" in cmd and "32" in cmd        # caller flags pass through


def test_bench_command_applies_the_three_traps_fixes():
    cmd = bench_command(_cfg(), concurrency=16)
    assert cmd[:3] == ["python3", "-m", "sglang.bench_serving"]
    assert cmd[cmd.index("--base-url") + 1] == SERVE_BASE_URL   # 127.0.0.1, not a hostname
    assert cmd[cmd.index("--dataset-path") + 1] == "/dataset/random"  # random needs a path
    assert cmd[cmd.index("--max-concurrency") + 1] == "16"
    assert cmd[cmd.index("--num-prompts") + 1] == "16"


def test_stable_concurrency_is_sixteen():
    assert STABLE_CONCURRENCY == 16


def test_shell_preview_is_quoted():
    line = shell_preview(["echo", "a b"])
    assert line == "echo 'a b'"


# ---------------------------------------------------------------- summary parsing

def test_parse_takes_the_last_json_line():
    out = ("Running benchmark...\n"
           '{"output_throughput": 730.2, "median_ttft_ms": 6400}\n')
    d = parse_bench_summary(out)
    assert d["output_throughput"] == 730.2


def test_parse_last_parseable_wins_over_earlier_json():
    out = '{"x": 1}\nnoise\n{"output_throughput": 522, "median_tps": 33}\n'
    assert parse_bench_summary(out)["output_throughput"] == 522


def test_parse_garbage_returns_empty_not_crash():
    assert parse_bench_summary("not json\nstill not {json") == {}
    assert parse_bench_summary("") == {}


# ---------------------------------------------------------------- reading + gates

def test_reading_from_fake_summary_maps_fields():
    out = json.dumps({"output_throughput": 730.0, "median_ttft_ms": 6400.0,
                      "median_tps": 45.6})

    def start(cmd, env):
        return object()

    def bench(cmd):
        return out

    r = run_once(_cfg(backend="flashinfer"), start_server=start, run_bench=bench,
                 concurrency=16)
    assert r.concurrency == 16
    assert r.tok_s == 730.0
    assert r.ttft_s == 6400.0
    assert r.tps_per_user == 45.6
    assert r.backend == "flashinfer"
    assert r.gates == {}            # no run_gates -> nothing checked


def test_unmeasured_run_yields_none_not_zero():
    r = run_once(_cfg(), start_server=lambda c, e: object(),
                 run_bench=lambda c: "no json here", concurrency=16)
    assert r.tok_s is None and r.ttft_s is None and r.tps_per_user is None


def test_gates_only_run_when_named():
    called = {"checks": 0}

    def checks(handle, cfg):
        called["checks"] += 1
        return {GATE_GRAPH: True, GATE_PAGED: True, GATE_PARITY: True}

    cfg = _cfg(run_gates=(GATE_GRAPH, GATE_PAGED, GATE_PARITY))
    r = run_once(cfg, start_server=lambda c, e: object(), run_bench=lambda c: "{}",
                 run_checks=checks, concurrency=16)
    assert called["checks"] == 1
    assert r.gates == {GATE_GRAPH: True, GATE_PAGED: True, GATE_PARITY: True}


def test_unchecked_gate_is_not_a_pass():
    r = ServingReading(concurrency=16, tok_s=1.0, ttft_s=1.0, tps_per_user=1.0,
                       gates={GATE_GRAPH: True})   # paged/parity absent
    assert r.gate(GATE_PAGED) is None
    assert r.all_checked_gates_pass is True       # only checked ones matter here
    assert r.deployable is False                  # but deployable needs all three TRUE


def test_deployable_requires_all_three_gates_true():
    ok = ServingReading(16, 1.0, 1.0, 1.0,
                        gates={GATE_GRAPH: True, GATE_PAGED: True, GATE_PARITY: True})
    assert ok.deployable
    bad = ServingReading(16, 1.0, 1.0, 1.0,
                        gates={GATE_GRAPH: True, GATE_PAGED: False, GATE_PARITY: True})
    assert not bad.deployable
    assert bad.blocked_gates == [GATE_PAGED]


def test_paged_only_checked_gate_can_still_block():
    # a fast kernel that fails CUDA graph is not deployable and reports why
    r = ServingReading(16, 999.0, 1.0, 1.0,
                       gates={GATE_GRAPH: False, GATE_PAGED: True, GATE_PARITY: True})
    assert r.blocked_gates == [GATE_GRAPH]
    assert not r.deployable


# ---------------------------------------------------------------- baseline comparison

def test_compare_reports_tok_ratio_at_matched_concurrency():
    cand = ServingReading(16, 730.0, 6400.0, 46.0)
    base = ServingReading(16, 522.0, 24400.0, 33.0)
    v = compare_to_baseline(cand, base)
    assert v["tok_s_ratio"] == pytest.approx(730.0 / 522.0)
    assert v["ttft_ratio"] == pytest.approx(6400.0 / 24400.0)


def test_compare_refuses_cross_concurrency_ratio():
    cand = ServingReading(16, 730.0, 6400.0, 46.0)
    base = ServingReading(64, 1200.0, 9000.0, 20.0)
    v = compare_to_baseline(cand, base)
    assert "tok_s_ratio" not in v
    assert "concurrency mismatch" in v["reason"]


def test_compare_carries_blocked_gates():
    cand = ServingReading(16, 999.0, 1.0, 1.0,
                          gates={GATE_GRAPH: False, GATE_PAGED: True, GATE_PARITY: True})
    base = ServingReading(16, 522.0, 6400.0, 33.0)
    v = compare_to_baseline(cand, base)
    assert v["deployable"] is False
    assert v["blocked_gates"] == [GATE_GRAPH]


# ---------------------------------------------------------------- sweep

def test_sweep_starts_server_once_and_benches_each_concurrency():
    starts = {"n": 0}
    bench_calls = []

    def start(cmd, env):
        starts["n"] += 1
        return object()

    def bench(cmd):
        cc = int(cmd[cmd.index("--max-concurrency") + 1])
        bench_calls.append(cc)
        return json.dumps({"output_throughput": float(cc)})

    cfg = _cfg(concurrencies=(16, 32, 64))
    readings = run_sweep(cfg, start_server=start, run_bench=bench)
    assert starts["n"] == 1
    assert bench_calls == [16, 32, 64]
    assert [r.concurrency for r in readings] == [16, 32, 64]
    assert [r.tok_s for r in readings] == [16.0, 32.0, 64.0]


# ---------------------------------------------------------------- CLI dry-run

def test_cli_dry_run_prints_commands_without_running(capsys):
    from forgekernel.serve.__main__ import main
    rc = main(["--model-path", "/m", "--serve-flag=--tp", "--serve-flag=2",
               "--concurrency", "16", "--concurrency", "32", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sglang serve --model-path /m" in out
    assert out.count("sglang.bench_serving") == 2
    assert "--max-concurrency 32" in out

