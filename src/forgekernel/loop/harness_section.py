"""The card-owned evaluation harness: hygiene and edit-refusal.

Extracted from ``loop/driver.py`` (see ``docs/driver-split-spec.md``). These two steps
run inside the inner loop around a delivery, but they are leaf-like -- each is called
once, from ``run_inner_iteration`` -- so they carry no orchestration state and are safe
to live in their own module.

They depend only on:
- ``IterationRecord`` / ``append_record`` from :mod:`forgekernel.inner.trajectory`,
- :mod:`forgekernel.loop.harness_guard` for the drift description,
- ``ctx`` (``LoopContext``) for the guard mode, the anchor, the backend and the
  kernel file name.

They do **not** import from :mod:`forgekernel.loop.driver`: a tiny local ``_log`` and a
one-line active-cluster read keep this module a leaf, so ``driver`` can import it
back with no cycle. Behaviour is byte-for-byte the functions that lived in ``driver``;
``tests/test_loop_driver.py`` (the 411-test gate) is what proves it.
"""
from __future__ import annotations

from forgekernel.inner.trajectory import IterationRecord, append_record
from forgekernel.loop import harness_guard


def _log(msg: str) -> None:
    print(f"[loop] {msg}", flush=True)


def _active_cluster(ctx: "LoopContext") -> str:  # noqa: F821 -- forward ref, ctx duck-typed
    """The cluster submissions currently resolve to (the last one the backend
    submitted to), falling back to home. "" on backends that cannot say."""
    return getattr(ctx.backend, "last_cluster", "") or ctx.home_cluster


def _harness_hygiene(ctx: LoopContext, iter_no: int) -> None:
    """Put the lineage's harness back to the launch tree's before the student turn.

    The path check below sees what ONE delivery changed; three things reach the lineage
    without being a delivery: an iteration that ends with no record keeps its probe
    commits while there is no green tree to reset to, a teacher turn runs in the same
    working directory and a reset keeps the untracked files it wrote, and the next
    ``git add -A`` commits whatever is there. Restored here in a commit of their own,
    so they are neither judged by nor blamed on the next delivery."""
    if ctx.harness_guard == "off" or not ctx.harness_anchor:
        return
    be = ctx.backend
    drift_of = getattr(be, "harness_drift", None)
    restore = getattr(be, "restore_harness", None)
    if drift_of is None or restore is None:
        return
    try:
        drift = drift_of(ctx.harness_anchor)
    except Exception as e:  # noqa: BLE001 — a hygiene step must not become the failure
        _log(f"iter {iter_no}: harness guard: cannot read the lineage's harness ({e})")
        return
    if not drift:
        return
    shown = harness_guard.describe(drift)
    if ctx.harness_guard != "revert":
        _log(f"iter {iter_no}: WARNING harness guard: the lineage's {shown} differ from the "
             f"launch tree (harness_guard = warn: left as is; jobs run the card's copies)")
        return
    try:
        restore(ctx.harness_anchor, drift,
                f"harness guard: restore the card-owned {shown} to the launch tree")
    except Exception as e:  # noqa: BLE001
        _log(f"iter {iter_no}: harness guard: could not restore {shown} ({e})")
        return
    _log(f"iter {iter_no}: harness guard: restored {shown} to the launch tree "
         f"({str(ctx.harness_anchor)[:10]}) before the student turn — card-owned evaluation "
         f"files that reached the workspace outside a delivery")


def _harness_edit_revert(ctx: LoopContext, iter_no: int, record, sha: str, base_sha: str,
                         metric_before) -> "IterationRecord | None":
    """REVERT a delivery whose commits (probe commits included) edit the card-owned
    evaluation harness, before any job is bought; None = nothing to refuse."""
    if ctx.harness_guard == "off" or not base_sha:
        return None
    changes_of = getattr(ctx.backend, "harness_changes", None)
    if changes_of is None:
        return None
    try:
        touched = changes_of(base_sha, sha) or []
    except Exception:  # noqa: BLE001 — bookkeeping, never a reason to stop
        return None
    if not touched:
        return None
    shown = harness_guard.describe(touched)
    if ctx.harness_guard != "revert":
        _log(f"iter {iter_no}: WARNING {record.decision_id} edits the card-owned {shown} "
             f"(harness_guard = warn: kept; jobs run the card's copies)")
        return None
    err = (f"harness guard: the delivery edits card-owned evaluation files ({shown}). "
           f"gpu_entry.sh, eval/, ref_bin/ and .gitattributes belong to the card — jobs run "
           f"the card's copies whatever the tree holds, and only {ctx.kernel_file} (plus "
           f"notes/) is the student's to change. Reverted without an eval; deliver the "
           f"decision again without touching them.")
    _log(f"iter {iter_no}: {record.decision_id} REVERT ({err})")
    ctx.backend.reset(base_sha)
    # correctness None: nothing was evaluated, so the lineage's green state is not
    # moved (a False would drop a never-evaluated workspace into scaffold mode)
    rec = IterationRecord(
        iter=iter_no, decision_id=record.decision_id, change=record.change,
        correctness=None, verdict="REVERT", metric_before=metric_before,
        error=err, profiler={}, cluster=_active_cluster(ctx), base_sha=base_sha,
    )
    append_record(ctx.status_path, rec)
    return rec
