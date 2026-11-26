"""The search-loop and dual-eval notes wired into the student prompt."""
from types import SimpleNamespace

from forgekernel.inner.trajectory import IterationRecord, append_record
from forgekernel.loop import prompt as promptmod


def _ctx(tmp_path, gate_window="device"):
    return SimpleNamespace(status_path=tmp_path / "status.jsonl", gate_window=gate_window)


def _rec(iter, change, verdict, *, off=None, srv=None, qps=None, correctness=True):
    return IterationRecord(
        iter=iter, decision_id=f"d{iter}", change=change, correctness=correctness,
        verdict=verdict, vs_ref_device=off, vs_ref_server=srv, server_qps=qps)


def test_search_loop_note_is_silent_until_something_is_evaluated(tmp_path):
    ctx = _ctx(tmp_path)
    assert promptmod._search_loop_note(ctx) == ""          # no status file at all
    append_record(ctx.status_path, _rec(1, "raise occupancy", "NEUTRAL"))
    assert promptmod._search_loop_note(ctx) == ""          # only a neutral scaffold row


def test_search_loop_note_advises_after_evaluated_rows(tmp_path):
    ctx = _ctx(tmp_path)
    append_record(ctx.status_path, _rec(1, "raise occupancy", "REVERT"))
    append_record(ctx.status_path, _rec(2, "retile 128x128", "KEEP", off=1.10))
    note = promptmod._search_loop_note(ctx)
    assert "Search-loop advisory" in note
    assert "retile" in note            # the paying technique is named


def test_dual_eval_note_flags_a_mode_split_when_both_windows_measured(tmp_path):
    ctx = _ctx(tmp_path)
    # offline win, server loss -> the deployment-regression warning
    append_record(ctx.status_path, _rec(1, "retile", "KEEP", off=1.30, srv=1.05, qps=200))
    note = promptmod._dual_eval_note(ctx)
    assert "Dual-evaluation" in note and "DIVERGENCE" in note


def test_dual_eval_note_is_silent_with_one_mode(tmp_path):
    ctx = _ctx(tmp_path)
    append_record(ctx.status_path, _rec(1, "retile", "KEEP", off=1.20))
    assert promptmod._dual_eval_note(ctx) == ""
