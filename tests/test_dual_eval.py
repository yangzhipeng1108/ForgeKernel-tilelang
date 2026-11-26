"""Offline vs server dual evaluation: the two modes, and the divergence warning."""
from forgekernel.inner.trajectory import IterationRecord
from forgekernel import timing_audit as ta


def _rec(iter, verdict, *, off=None, srv=None, qps=None):
    return IterationRecord(
        iter=iter, decision_id=f"d{iter}", change="x", correctness=True, verdict=verdict,
        vs_ref_device=off, vs_ref_server=srv, server_qps=qps)


def test_both_modes_read_the_best_kept_ratio_each():
    records = [_rec(1, "KEEP", off=1.10, srv=1.25, qps=100),
               _rec(2, "KEEP", off=1.20, srv=1.18, qps=100),
               _rec(3, "REVERT", off=1.50, srv=1.60, qps=100)]  # reverted: ignored
    dual = ta.offline_vs_server(records)
    assert dual.offline_ratio == 1.20           # best kept offline
    assert dual.server_ratio == 1.25            # best kept server
    assert dual.server_qps == 100
    assert dual.both_measured


def test_divergence_flags_a_mode_split_each_direction():
    # server much stronger than offline (CUDA-L2's +19% offline / +26% server shape)
    up = ta.offline_vs_server([_rec(1, "KEEP", off=1.19, srv=1.26, qps=200)])
    assert up.diverges and up.divergence > 0
    # a kernel that wins offline but LOSES in server mode -- the deployment regression
    down = ta.offline_vs_server([_rec(1, "KEEP", off=1.30, srv=1.05, qps=200)])
    assert down.diverges and down.divergence < 0
    note = ta.dual_eval_note(down)
    assert "DIVERGENCE" in note and "WEAKER" in note and "200" in note


def test_within_noise_is_not_a_divergence():
    dual = ta.offline_vs_server([_rec(1, "KEEP", off=1.20, srv=1.22, qps=50)])
    assert not dual.diverges
    note = ta.dual_eval_note(dual)
    assert "within noise" in note and "holds" in note


def test_one_mode_only_has_no_divergence_and_no_note():
    off_only = ta.offline_vs_server([_rec(1, "KEEP", off=1.20)])
    assert not off_only.both_measured
    assert off_only.divergence is None
    assert ta.dual_eval_note(off_only) == ""
    srv_only = ta.offline_vs_server([_rec(1, "KEEP", srv=1.20, qps=80)])
    assert not srv_only.both_measured and ta.dual_eval_note(srv_only) == ""


def test_no_records_is_graceful():
    dual = ta.offline_vs_server([])
    assert dual.offline_ratio is None and dual.server_ratio is None
    assert ta.dual_eval_note(dual) == ""
