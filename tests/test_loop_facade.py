"""Facade contract for the loop/driver.py split (docs/driver-split-spec.md).

driver.py grew into a God module and is being split, step by step, into leaf
modules under forgekernel/loop/ that driver.py re-exports. Every step MUST keep
the facade contract in docs/driver-split-spec.md intact:

  1. forgekernel.loop.driver keeps exposing every name the tests import at the
     same dotted path, plus the names loop/__init__.py re-exports.
  2. The extracted modules are importable on their own and do not form an
     import cycle (extracted modules may import driver only lazily, at call
     time, via shims).

This test freezes the set of names imported from forgekernel.loop.driver across
the whole test suite. If an extraction (or a rename) drops one of these names,
the test fails with the exact missing names -- so a refactor can never silently
break a caller. When a name is intentionally renamed/removed, update the frozen
set here as part of that change (the diff makes the surface change reviewable).
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

TESTS_DIR = pathlib.Path(__file__).resolve().parent

# Frozen at commit that completed split steps 2-4 (harness/placeholder/pricing).
# Regenerate with: python tests/test_loop_facade.py  (prints the set).
FROZEN_DRIVER_IMPORTS = frozenset(
    {
        "AGENT_ROLE_TEACHER",
        "AgentError",
        "BLACKBOX_GUARD_NAME",
        "BackendError",
        "FAST_SKIP_NOISE_MULTIPLE",
        "LoopContext",
        "MetaResult",
        "NoRunError",
        "PRICING_SEEDED_MODULES",
        "PricedTrees",
        "ProbeBattle",
        "RECON_SEEDED_MODULES",
        "REF_STRUCT_PROMPT_CHARS",
        "_INFEASIBLE_MENTION_BLOCK",
        "_JOB_ENV_CHECK_CMD",
        "_KB_CERT_MARK",
        "_LEVER_PRICING_CMD",
        "_ORCH_CENSUS_CMD",
        "_PMTL_OK",
        "_PMTL_WAIVED",
        "_PM_TIMELINE_RE",
        "_PendingPatch",
        "_RECON_STATE_NAME",
        "_SASS_OK",
        "_SCHED_NOT_APPLICABLE",
        "_SCHED_OK",
        "_STALL_LEDGER_RE",
        "_STALL_OK",
        "_STALL_PRICE_CMD",
        "_STALL_PRICE_LOGS",
        "_STALL_WAIVED",
        "_STRUCT_OK",
        "_TIMELINE_CMD",
        "_admit_paired",
        "_apply_phase_close",
        "_axis_battle_block",
        "_bank_evidence",
        "_bank_timeline",
        "_banked_dossier_header_text",
        "_banked_lever_table",
        "_banked_timeline_of",
        "_best_keep_profiler",
        "_bindings",
        "_blackbox_guard_source",
        "_check_profile_covers_the_kernel",
        "_cited_levers",
        "_cited_price_jobs",
        "_cluster_scale",
        "_collect_verdict_near_misses",
        "_confirm_apparent_target",
        "_confirm_suspicious_watermark",
        "_confirm_workload_medians",
        "_current_phase",
        "_declared_construction",
        "_distill_prompt_path",
        "_enforce_single_ref",
        "_ensure_ref_pricing",
        "_evidence_gate_msg",
        "_fast_skip_reason",
        "_fruitless_reopen_streak",
        "_gate_regression",
        "_green_at_ref",
        "_green_on_resume",
        "_infeasible_refusal",
        "_inferred_rows_block_for_distill",
        "_is_timeline_request",
        "_judge_patch",
        "_kb_certify_lint_gate",
        "_latest_timeline",
        "_launch_cluster_from_log",
        "_launch_contradictions",
        "_launch_contradictions_open",
        "_lever_price_block",
        "_lever_price_submit",
        "_lever_pricing_jobs_text",
        "_lineage_rows",
        "_live_records",
        "_load_recon_state",
        "_mark_lever_citations",
        "_merge_lever_tables",
        "_meta_workload_readings",
        "_missing_recon_evidence",
        "_ncu_mode",
        "_ncu_profiler",
        "_needs_authoring",
        "_note_reopen",
        "_on_cluster",
        "_orch_diff_views",
        "_orch_summary",
        "_paired_parity_stop",
        "_parse_cause_class",
        "_parse_gate_contract",
        "_parse_list",
        "_parse_phase_verdict",
        "_parse_serial_census",
        "_persist_gap_queue",
        "_phantom_gap_reanchor",
        "_phase_exhaustion",
        "_pick_ref_dossier",
        "_placeholder_ledger",
        "_plateau_diagnoses",
        "_pm_interval",
        "_pop_orch_summary",
        "_previous_gate_reading",
        "_price_log_settled",
        "_price_log_verdict",
        "_price_revision_note",
        "_price_role_pass_killed",
        "_price_student_trees",
        "_quarantine_leaky_kb",
        "_recon_artifacts_written",
        "_recon_prompt",
        "_ref_for_decision",
        "_ref_overlap_banked",
        "_ref_overlap_block_for_distill",
        "_ref_price_status",
        "_ref_pricing_key",
        "_ref_workload_pricing",
        "_reopen_gate_on",
        "_reprice_verdict",
        "_resolve_resume",
        "_resolve_resume_in_place",
        "_resolve_target",
        "_restore_kept_probes",
        "_resume_anchor",
        "_run_stall_pricing",
        "_run_timeline",
        "_sass_serial_status",
        "_save_recon_state",
        "_save_ref_orch_dossier",
        "_save_ref_role_exposure",
        "_save_ref_stall_ledger",
        "_seed_target_ms",
        "_seed_ws_stall_tools",
        "_set_resume_note",
        "_source_recon_distill_addendum",
        "_stall_price_cmd",
        "_student_prompt",
        "_table_noise_pct",
        "_target_workloads_mismatch",
        "_teacher",
        "_timeline_feedback",
        "_timeline_on",
        "_timeline_probe_stale",
        "_timeline_renderable",
        "_timeline_short_coverage",
        "_timeline_student_note",
        "_too_few_workloads",
        "_validate_milestone_file",
        "_verify_phase_verdict",
        "_verify_reopen_price",
        "_without_census_lines",
        "align_segment_table",
        "apply_window_credit",
        "archive_is_whole",
        "job_env_preflight",
        "kb_axis_debt",
        "kb_harvest_debt",
        "kb_read_dir",
        "load_gap_queue",
        "parse_gap_queue",
        "parse_lever_prices",
        "render_prompt",
        "run_best_confirmation",
        "run_distill_step",
        "run_inner_iteration",
        "run_inner_loop",
        "run_kb_certify",
        "run_kb_harvest",
        "run_meta_step",
        "run_recon_step",
        "run_two_layer",
    }
)

# Modules extracted so far (steps 2-4). Each must import on its own.
EXTRACTED_MODULES = (
    "forgekernel.loop.harness_section",
    "forgekernel.loop.placeholder",
    "forgekernel.loop.pricing",
    "forgekernel.loop.prompt",
    "forgekernel.loop.comparisons",
    "forgekernel.loop.reopen",
    "forgekernel.loop.shared",
    "forgekernel.loop.timeline",
    "forgekernel.loop.gap_queue",
    "forgekernel.loop.best_confirmation",
)


def _collect_imported_driver_names() -> set[str]:
    """Names imported from forgekernel.loop.driver across tests/, via AST."""
    names: set[str] = set()
    for path in TESTS_DIR.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "forgekernel.loop.driver"
            ):
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.name)
    return names


def test_driver_facade_superset_of_imports():
    """Every name the test suite imports from driver is actually present."""
    driver = importlib.import_module("forgekernel.loop.driver")
    wanted = _collect_imported_driver_names()
    missing = sorted(n for n in wanted if not hasattr(driver, n))
    assert not missing, (
        "driver.py dropped names still imported by the test suite: %s" % missing
    )


def test_frozen_set_matches_live_imports():
    """The live import set did not drift from the frozen contract.

    A diff here means a test now imports a name not in the frozen set (add it)
    or stopped importing one (drop it) -- either way, update FROZEN_DRIVER_IMPORTS
    so the facade surface change is an explicit, reviewable diff.
    """
    live = _collect_imported_driver_names()
    added = sorted(live - FROZEN_DRIVER_IMPORTS)
    removed = sorted(FROZEN_DRIVER_IMPORTS - live)
    assert not added and not removed, (
        "driver import surface changed; update FROZEN_DRIVER_IMPORTS.\n"
        "  newly imported: %s\n  no longer imported: %s" % (added, removed)
    )


@pytest.mark.parametrize("modname", EXTRACTED_MODULES)
def test_extracted_module_imports_cleanly(modname):
    """An extracted module imports on its own (no cycle at import time)."""
    importlib.import_module(modname)


def test_loop_package_all_resolves():
    """Every name in loop.__all__ resolves on the package."""
    pkg = importlib.import_module("forgekernel.loop")
    missing = sorted(n for n in pkg.__all__ if not hasattr(pkg, n))
    assert not missing, "loop.__all__ names not resolvable: %s" % missing


if __name__ == "__main__":
    live = _collect_imported_driver_names()
    for name in sorted(live):
        print(name)
    print(f"# {len(live)} names")
