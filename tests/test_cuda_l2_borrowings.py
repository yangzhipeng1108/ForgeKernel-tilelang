"""The three ideas borrowed from CUDA-L2, at the level each one lives at.

1. ``[[comparison]]`` baselines — task parsing + the report rows (report only).
2. The server gate window — "server:<qps>" parsing, the archive refusal, the
   baseline's render, and the record's reading.
3. The fast-skip band — a candidate far behind the best skips profiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgekernel import task as task_mod
from forgekernel.baseline import RefBaseline
from forgekernel.inner.trajectory import (
    GATE_WINDOW_DEVICE,
    GATE_WINDOW_E2E,
    GATE_WINDOW_SERVER,
    IterationRecord,
    gate_ms,
    gate_window_name,
    parse_gate_window,
)
from forgekernel.task import TaskError, load_task

SHAPES = [{"name": "small", "axes": {"m": 128, "n": 256}},
          {"name": "large", "axes": {"m": 4096, "n": 256}}]
MINIMAL = '[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
REF_FILES = {"reference/run.py": "def run(x, w):\n    return x\n",
             "reference/cublaslt.py": "def run(x, w):\n    return x\n"}


def _kb(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for kb in ("hw_kb/cutedsl/sm100",):
        (root / kb).mkdir(parents=True, exist_ok=True)
    return root


def _task(tmp_path: Path, toml: str, *, files: dict | None = None,
          name: str = "gemm") -> Path:
    d = tmp_path / "tasks" / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text(toml)
    (d / "description.md").write_text("c = a @ b\n")
    (d / "shapes.json").write_text(json.dumps(SHAPES))
    for rel, text in (files or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return d


# --- 1. comparison baselines ------------------------------------------------------


def test_comparisons_parse_in_order(tmp_path):
    toml = (MINIMAL + '[reference]\nrun = "reference/run.py:run"\n'
            + '[[comparison]]\nname = "cuBLASLt-heuristic"\nrun = "reference/cublaslt.py:run"\n'
            + '[[comparison]]\nname = "cuBLASLt-AutoTuning"\nrun = "reference/cublaslt.py:run"\n')
    t = load_task(_task(tmp_path, toml, files=REF_FILES), kb_root=_kb(tmp_path))
    assert [c.name for c in t.comparisons] == ["cuBLASLt-heuristic", "cuBLASLt-AutoTuning"]
    assert all(c.run_fn == "run" for c in t.comparisons)
    assert any("2 comparison baseline(s)" in n for n in t.notes)


def test_comparisons_default_empty(tmp_path):
    """A task with no [[comparison]] must behave exactly as before."""
    t = load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path))
    assert t.comparisons == ()
    assert not any("comparison baseline" in n for n in t.notes)


def test_comparison_run_outside_the_task_is_refused(tmp_path):
    toml = MINIMAL + '[[comparison]]\nname = "x"\nrun = "../elsewhere.py:run"\n'
    with pytest.raises(TaskError) as e:
        load_task(_task(tmp_path, toml, files=REF_FILES), kb_root=_kb(tmp_path))
    assert any("leaves the task directory" in p for p in e.value.problems)


def test_comparison_duplicate_name_is_refused(tmp_path):
    toml = (MINIMAL
            + '[[comparison]]\nname = "a"\nrun = "reference/cublaslt.py:run"\n'
            + '[[comparison]]\nname = "a"\nrun = "reference/cublaslt.py:run"\n')
    with pytest.raises(TaskError) as e:
        load_task(_task(tmp_path, toml, files=REF_FILES), kb_root=_kb(tmp_path))
    assert any("duplicate comparison name" in p for p in e.value.problems)


def test_comparison_unknown_key_is_refused(tmp_path):
    toml = MINIMAL + '[[comparison]]\nname = "a"\nrun = "reference/cublaslt.py:run"\nbogus = 1\n'
    with pytest.raises(TaskError) as e:
        load_task(_task(tmp_path, toml, files=REF_FILES), kb_root=_kb(tmp_path))
    assert any("unknown key 'bogus'" in p for p in e.value.problems)


def test_comparisons_round_trip_through_dump(tmp_path):
    toml = (MINIMAL + '[[comparison]]\nname = "cuBLAS"\nrun = "reference/cublaslt.py:run"\n')
    cfg = {"task": {"language": "cutedsl", "arch": "sm100"},
           "comparison": [{"name": "cuBLAS", "run": "reference/cublaslt.py:run"}]}
    assert "[[comparison]]" in task_mod.dump_task_toml(cfg)


def test_report_rows_are_median_vs_median_and_never_gated(tmp_path):
    """comparison_rows reads archives; a missing one is a 'not measured' row."""
    from forgekernel import launch

    op_dir = tmp_path / "ops" / "gemm"
    op_dir.mkdir(parents=True)
    (op_dir / "comparison_cuBLAS.json").write_text(json.dumps(
        {"op": "gemm", "ms": 1.0, "ms_device": 0.8, "role": "library"}))

    class _Run:
        operators_dir = tmp_path / "ops"

    class _Comp:
        def __init__(self, name):
            self.name = name

    class _Task:
        name = "gemm"
        comparisons = (_Comp("cuBLAS"), _Comp("cuBLASLt-AutoTuning"))

    best = {"median_ms": 0.5, "window": "device"}
    rows = launch.comparison_rows(_Run(), _Task(), best)
    assert rows[0]["name"] == "cuBLAS" and rows[0]["status"] == "measured"
    assert rows[0]["median_ms"] == 0.8 and rows[0]["speedup"] == pytest.approx(1.6)
    assert rows[0]["basis"] == "median-vs-median"
    assert rows[1]["status"] == "not measured"


def test_report_rows_empty_without_best_or_comparisons():
    from forgekernel import launch

    class _Task:
        comparisons = ()

    class _Run:
        operators_dir = Path(".")

    assert launch.comparison_rows(_Run(), _Task(), {"median_ms": 1.0}) == []


# --- 2. the server window ---------------------------------------------------------


def test_parse_gate_window():
    assert parse_gate_window("device") == (GATE_WINDOW_DEVICE, None)
    assert parse_gate_window("e2e") == (GATE_WINDOW_E2E, None)
    assert parse_gate_window("server:100") == (GATE_WINDOW_SERVER, 100.0)
    assert parse_gate_window("server: 12.5") == (GATE_WINDOW_SERVER, 12.5)
    assert gate_window_name("server:100") == "server"
    assert gate_window_name("device") == "device"


@pytest.mark.parametrize("spec", ["server:", "server:abc", "server:0", "server:-1", "bogus"])
def test_parse_gate_window_refuses_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_gate_window(spec)


def test_gate_ms_reads_the_server_window():
    rec = IterationRecord(iter=1, decision_id="d", change="c", correctness=True,
                          verdict="KEEP", bwd_median_ms=2.0, device_median_ms=1.0,
                          server_median_ms=3.0, server_qps=100)
    assert gate_ms(rec, "device") == 1.0
    assert gate_ms(rec, "e2e") == 2.0
    assert gate_ms(rec, "server:100") == 3.0


def test_gate_ms_server_window_unmeasured_is_none():
    rec = IterationRecord(iter=1, decision_id="d", change="c", correctness=True,
                          verdict="KEEP", bwd_median_ms=2.0, device_median_ms=1.0)
    assert gate_ms(rec, "server:100") is None


def test_target_ms_server_window():
    b = RefBaseline(op="gemm", ms=1.0, ms_device=0.9,
                    server_median_ms=0.8, server_qps=100)
    assert b.target_ms(1.2, "server:100") == pytest.approx(0.8 / 1.2)


def test_target_ms_server_refuses_a_rate_mismatch():
    b = RefBaseline(op="gemm", ms=1.0, server_median_ms=0.8, server_qps=100)
    with pytest.raises(ValueError, match="was measured at 100 qps"):
        b.target_ms(1.2, "server:200")


def test_target_ms_server_refuses_an_unmeasured_archive():
    b = RefBaseline(op="gemm", ms=1.0, ms_device=0.9)
    with pytest.raises(ValueError, match="no server-window reading"):
        b.target_ms(1.2, "server:100")


def test_render_shows_the_server_line_and_marker():
    b = RefBaseline(op="gemm", ms=1.0, ms_device=0.9,
                    server_median_ms=0.8, server_qps=100)
    gating = b.render("server:100")
    assert "server median" in gating and "100 qps" in gating
    assert "the target anchors on this number" in gating
    not_gating = b.render("device")
    assert "server median" in not_gating
    assert "the target does not anchor on this number" in not_gating


def test_eval_result_server_fields_parse():
    from forgekernel.loop.parse import parse_eval
    js = json.dumps({"correctness": {"pass": True},
                     "performance": {"bwd_median_ms": 2.0, "device_median_ms": 1.0,
                                     "server_median_ms": 0.9, "ref_server_ms": 1.2,
                                     "vs_ref_server": 1.3333, "server_qps": 100}})
    ev = parse_eval(f"some log\n=== JSON ===\n{js}\n")
    assert ev.server_median_ms == 0.9
    assert ev.ref_server_ms == 1.2
    assert ev.vs_ref_server == pytest.approx(1.3333)
    assert ev.server_qps == 100


# --- 3. the fast-skip band --------------------------------------------------------


def test_fast_skip_is_off_by_default():
    from forgekernel.loop.driver import LoopContext, _fast_skip_reason
    ctx = LoopContext.__new__(LoopContext)
    ctx.gate_window = "device"
    ctx.watermark_margin = 0.015
    ctx.fast_skip = False
    ev = IterationRecord(iter=1, decision_id="d", change="c", correctness=True,
                         verdict="KEEP", device_median_ms=10.0)
    assert _fast_skip_reason(ctx, ev, 1.0) is None


def test_fast_skip_fires_only_far_behind_the_best():
    from forgekernel.loop.driver import FAST_SKIP_NOISE_MULTIPLE, LoopContext, _fast_skip_reason
    ctx = LoopContext.__new__(LoopContext)
    ctx.gate_window = "device"
    ctx.watermark_margin = 0.015
    ctx.fast_skip = True

    def ev(ms):
        return IterationRecord(iter=1, decision_id="d", change="c", correctness=True,
                               verdict="KEEP", device_median_ms=ms)

    best = 1.0
    cutoff = best * (1 + 0.015 * FAST_SKIP_NOISE_MULTIPLE)
    assert _fast_skip_reason(ctx, ev(cutoff), best) is None          # inside the band
    assert _fast_skip_reason(ctx, ev(cutoff * 1.01), best) is not None   # past it
    assert _fast_skip_reason(ctx, ev(0.5), best) is None             # a faster candidate
    assert _fast_skip_reason(ctx, ev(10.0), None) is None            # no bar to be behind


def test_fast_skip_never_applies_to_an_unmeasured_gate_window():
    from forgekernel.loop.driver import LoopContext, _fast_skip_reason
    ctx = LoopContext.__new__(LoopContext)
    ctx.gate_window = "server:100"
    ctx.watermark_margin = 0.015
    ctx.fast_skip = True
    ev = IterationRecord(iter=1, decision_id="d", change="c", correctness=True,
                         verdict="KEEP", device_median_ms=10.0)
    assert _fast_skip_reason(ctx, ev, 1.0) is None
