"""A6: multi-baseline comparison archive + report rows (CUDA-L2's side-by-side tables)."""
import json
from types import SimpleNamespace

from forgekernel.baseline import RefBaseline
from forgekernel import launch


def _fake_run(op_dir):
    return SimpleNamespace(operators_dir=op_dir)


def _comp(name, slug_run="x"):
    return SimpleNamespace(name=name, run=slug_run)


def test_save_comparison_then_rows_reads_it_back(tmp_path):
    op_dir = tmp_path / "ops" / "demo"
    # two baselines archived as the harness would
    launch.save_comparison_baseline(op_dir, "cuBLASLt-AutoTuning",
                                    RefBaseline(op="demo", ms=2.4, ms_device=2.0,
                                                ref_name="cuBLASLt-AutoTuning"))
    launch.save_comparison_baseline(op_dir, "torch.matmul",
                                    RefBaseline(op="demo", ms=3.6, ms_device=3.0,
                                                ref_name="torch.matmul"))
    task = SimpleNamespace(name="demo", comparisons=(_comp("cuBLASLt-AutoTuning"), _comp("torch.matmul")))
    best = {"median_ms": 1.0, "window": "device"}
    rows = launch.comparison_rows(_fake_run(op_dir.parent), task, best)
    by = {r["name"]: r for r in rows}
    assert by["cuBLASLt-AutoTuning"]["status"] == "measured"
    # device-window: 2.0 / 1.0 = 2.0x, 3.0 / 1.0 = 3.0x
    assert by["cuBLASLt-AutoTuning"]["speedup"] == 2.0
    assert by["torch.matmul"]["speedup"] == 3.0
    assert by["torch.matmul"]["basis"] == "median-vs-median"


def test_missing_archive_is_reported_not_measured(tmp_path):
    op_dir = tmp_path / "ops" / "demo"
    op_dir.mkdir(parents=True)
    task = SimpleNamespace(name="demo", comparisons=(_comp("cuBLAS"),))
    best = {"median_ms": 1.0, "window": "device"}
    rows = launch.comparison_rows(_fake_run(op_dir.parent), task, best)
    assert rows == [{"name": "cuBLAS", "status": "not measured"}]


def test_no_best_or_no_comparisons_yields_no_rows(tmp_path):
    op_dir = tmp_path / "ops" / "demo"
    task = SimpleNamespace(name="demo", comparisons=(_comp("cuBLAS"),))
    assert launch.comparison_rows(_fake_run(op_dir), task, None) == []
    emptytask = SimpleNamespace(name="demo", comparisons=())
    assert launch.comparison_rows(_fake_run(op_dir), emptytask,
                                  {"median_ms": 1.0, "window": "device"}) == []


def test_renderer_marks_unmeasured_with_reason_and_formats_measured():
    rows = [{"name": "cuBLAS", "status": "measured", "median_ms": 2.0,
             "speedup": 2.0, "basis": "median-vs-median"},
            {"name": "Triton", "status": "not measured"}]
    lines = launch.render_comparison_table(rows)
    text = "\n".join(lines)
    assert "| baseline | reading |" in text
    assert "2.0000x" in text
    assert "not measured" in text          # the reason rides the basis column
    # empty rows -> no table at all
    assert launch.render_comparison_table([]) == []
