"""Same-operator runs share one reconnaissance analysis (loop/driver.py).

A run that is not the first for its operator starts from what earlier runs already
measured, instead of re-deriving the reference's architecture from scratch. The
analysis lives in the operator's own tree (``ctx.workspace``), one file per
language."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import FakeBackend, _ctx  # noqa: E402

from forgekernel.loop import driver  # noqa: E402


def _run(tmp_path, name="op", language="cutedsl"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    ctx = _ctx(d, FakeBackend([], []))
    ctx.language = language
    return ctx


def test_a_first_run_has_no_analysis_and_is_told_where_it_will_land(tmp_path):
    ctx = _run(tmp_path)
    binding = driver._family_analysis_binding(ctx)
    assert "none yet" in binding and "first cutedsl run" in binding
    assert "analysis" in binding and "cutedsl.md" in binding


def test_the_report_is_appended_once_and_is_idempotent(tmp_path):
    ctx = _run(tmp_path)
    ctx.recon_path = ctx.workspace / "recon" / "report.md"
    ctx.recon_path.parent.mkdir(parents=True)
    ctx.recon_path.write_text("# Recon\n\nThe reference is a two-stage GEMM.\n",
                              encoding="utf-8")
    driver._write_family_analysis(ctx)
    dest = driver.family_analysis_path(ctx)
    assert dest.is_file()
    text = dest.read_text(encoding="utf-8")
    assert "The reference is a two-stage GEMM." in text
    assert str(ctx.status_path) in text  # provenance marker

    # a second sync (a resume) must not append the same run twice
    driver._write_family_analysis(ctx)
    assert dest.read_text(encoding="utf-8").count("two-stage GEMM") == 1


def test_two_runs_of_one_operator_accumulate_into_one_file(tmp_path):
    ctx1 = _run(tmp_path)
    ctx1.recon_path = ctx1.workspace / "recon" / "report.md"
    ctx1.recon_path.parent.mkdir(parents=True)
    ctx1.recon_path.write_text("# Recon A\n\nfirst run\n", encoding="utf-8")
    driver._write_family_analysis(ctx1)

    # a later run of the same operator: same tree, its own status file/report
    ctx2 = _run(tmp_path)
    ctx2.status_path = ctx2.workspace / "status2.jsonl"
    ctx2.recon_path = ctx2.workspace / "recon" / "report2.md"
    ctx2.recon_path.write_text("# Recon B\n\nsecond run\n", encoding="utf-8")
    driver._write_family_analysis(ctx2)

    text = driver.family_analysis_path(ctx2).read_text(encoding="utf-8")
    assert "first run" in text and "second run" in text


def test_one_language_does_not_read_another(tmp_path):
    ctx_cu = _run(tmp_path, language="cutedsl")
    ctx_cu.recon_path = ctx_cu.workspace / "recon" / "report.md"
    ctx_cu.recon_path.parent.mkdir(parents=True)
    ctx_cu.recon_path.write_text("# Recon\n\ncutedsl reading\n", encoding="utf-8")
    driver._write_family_analysis(ctx_cu)

    ctx_tl = _run(tmp_path, language="tilelang")
    assert driver.family_analysis_path(ctx_tl) != driver.family_analysis_path(ctx_cu)
    assert "none yet" in driver._family_analysis_binding(ctx_tl)


def test_a_missing_report_writes_nothing(tmp_path):
    ctx = _run(tmp_path)
    ctx.recon_path = ctx.workspace / "recon" / "report.md"
    driver._write_family_analysis(ctx)  # no file on disk: a no-op, never raises
    assert not driver.family_analysis_path(ctx).exists()
