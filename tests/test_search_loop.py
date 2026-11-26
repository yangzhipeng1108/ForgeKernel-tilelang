"""The CUDA-L2-style search loop: technique arms, measured reward, UCB propose."""
from forgekernel.inner.trajectory import IterationRecord
from forgekernel.loop import search_loop as sl


def _rec(iter, change, verdict, *, ratio_dev=None, correctness=True, ratio_e2e=None,
         ratio_server=None):
    return IterationRecord(
        iter=iter, decision_id=f"d{iter}", change=change, correctness=correctness,
        verdict=verdict, vs_ref_device=ratio_dev, vs_ref=ratio_e2e,
        vs_ref_server=ratio_server)


def test_technique_of_canonicalises_phrasings_to_one_arm():
    assert sl.technique_of("raise occupancy to hide latency") == "occupancy"
    assert sl.technique_of("more warps per block via launch bounds") == "occupancy"
    assert sl.technique_of("retile the block to 128x128") == "retile"
    assert sl.technique_of("use cp.async double buffer") == "prefetch"
    assert sl.technique_of("add a software pipeline with num_stages=3") == "pipeline"
    # order tie-break: a text naming retile and occupancy lands on retile (listed first)
    assert sl.technique_of("retile to fix occupancy") == "retile"
    # nothing matches -> other, never dropped
    assert sl.technique_of("rename variables and clean comments") == "other"
    assert sl.technique_of("") == "other"


def test_reward_is_improvement_over_the_frontier_not_absolute_ratio():
    # best_before None: the first KEEP has no frontier to beat, so it is a hold (not its
    # whole ratio — that would reward an easy kernel for being easy)
    keep = _rec(1, "retile", "KEEP", ratio_dev=1.10)
    assert sl.reward_of(keep, None) == sl.REWARD_KEEP_HOLD
    # best_before 1.10: a KEEP at 1.13 earns only the 0.03 improvement
    keep2 = _rec(2, "retile", "KEEP", ratio_dev=1.13)
    assert abs(sl.reward_of(keep2, 1.10) - 0.03) < 1e-9
    # a KEEP that did not beat the frontier earns 0 (kept, but not progress here)
    keep3 = _rec(3, "retile", "KEEP", ratio_dev=1.05)
    assert sl.reward_of(keep3, 1.10) == 0.0


def test_revert_and_correctness_break_are_negative():
    rev = _rec(1, "swizzle", "REVERT", correctness=True)
    brk = _rec(2, "swizzle", "REVERT", correctness=False)
    assert sl.reward_of(rev, 1.0) == sl.REWARD_REVERT
    assert sl.reward_of(brk, 1.0) == sl.REWARD_CORRECTNESS_BREAK
    # a wrong kernel must cost more than a mere revert
    assert sl.REWARD_CORRECTNESS_BREAK < sl.REWARD_REVERT < 0


def test_keep_without_a_measured_ratio_earns_hold_not_zero_penalty():
    keep = _rec(1, "retile", "KEEP", ratio_dev=None)
    assert sl.reward_of(keep, None) == sl.REWARD_KEEP_HOLD


def test_build_stats_aggregates_evidence_per_arm_in_order():
    records = [
        _rec(1, "retile to 64x64", "KEEP", ratio_dev=1.05),
        _rec(2, "retile to 128x128", "KEEP", ratio_dev=1.12),
        _rec(3, "raise occupancy", "REVERT"),
        _rec(4, "raise occupancy with launch bounds", "KEEP", ratio_dev=1.20),
    ]
    stats, rows = sl.build_stats(records, gate_window="device")
    assert stats["retile"].n == 2 and stats["retile"].keeps == 2
    assert stats["occupancy"].n == 2 and stats["occupancy"].keeps == 1
    assert stats["occupancy"].reverts == 1
    # rewards scored against the advancing frontier: it4 improves on it2's 1.12
    assert rows[3].reward == 1.20 - 1.12 or abs(rows[3].reward - 0.08) < 1e-9
    # best single gain on retile is it2's 0.07 (1.12 - 1.05)
    assert abs(stats["retile"].best_gain - 0.07) < 1e-9


def test_ucb_prefers_untried_arms_then_exploits():
    stats = {"a": sl.ArmStats("a", n=10, reward_sum=1.0),
             "b": sl.ArmStats("b", n=1, reward_sum=0.01),
             "c": sl.ArmStats("c", n=0)}
    scores = sl.ucb_scores(stats)
    assert scores["c"] == float("inf")             # never tried -> first
    assert scores["b"] > scores["a"]               # under-sampled beats a saturated arm here
    # with no tries at all, nobody is infinite; all score 0
    empty = sl.ucb_scores({"x": sl.ArmStats("x"), "y": sl.ArmStats("y")})
    assert empty == {"x": 0.0, "y": 0.0}


def test_propose_ranks_untried_first_and_flags_priced_nulls():
    records = [
        _rec(1, "raise occupancy", "REVERT"),
        _rec(2, "raise occupancy more", "REVERT"),
        _rec(3, "retile 128x128", "KEEP", ratio_dev=1.10),
    ]
    prop = sl.propose(records, gate_window="device")
    # occupancy tried twice, never kept -> priced null
    assert "occupancy" in prop.priced_null
    assert "retile" not in prop.priced_null
    # untried arms lead the ranking
    assert prop.ranked[0] in prop.untried
    assert prop.best_arm == "retile"
    # the trace keeps every row, arms sum to the record count
    total = sum(s.n for s in prop.stats.values())
    assert total == 3


def test_render_table_and_advice_are_grounded_and_safe_on_empty():
    prop = sl.propose([], gate_window="device")
    table = sl.render_table(prop)
    assert "technique" in table and "| occupancy | 0 |" in table
    advice = sl.render_advice(prop)
    assert "untried" in advice.lower() or "no evaluated" in advice.lower()
    # a populated table shows a real gain for the paying arm (needs a frontier to beat)
    prop2 = sl.propose([_rec(1, "retile", "KEEP", ratio_dev=1.02),
                        _rec(2, "retile again", "KEEP", ratio_dev=1.08)], gate_window="device")
    assert "+0.060" in sl.render_table(prop2)
    adv = sl.render_advice(prop2)
    assert "retile" in adv and "Best paying" in adv


def test_gate_window_selects_the_measured_ratio_for_reward():
    rec = _rec(1, "retile", "KEEP", ratio_dev=1.10, ratio_server=1.40, ratio_e2e=1.02)
    assert abs(sl.reward_of(rec, 1.0, "device") - 0.10) < 1e-9
    assert abs(sl.reward_of(rec, 1.0, "server:100") - 0.40) < 1e-9
    assert abs(sl.reward_of(rec, 1.0, "e2e") - 0.02) < 1e-9
