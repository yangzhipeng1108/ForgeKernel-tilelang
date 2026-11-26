"""Reopen budget — reopen streak tracking and cause-class parsing.

Split out of loop/driver.py (docs/driver-split-spec.md step 5, first slice). This
is the self-contained reopen-budget core: the patch-class enums, the fruitless
reopen streak, the reopen cap prompt block, the infeasible-mention guard, the
cause-class enum + parser, and the teacher (kb-recipe) prompt. The heavier
certify/harvest/pricing pieces of the old "reopen budget" section stay in driver
for now — they are coupled to the timeline/pricing core.

Only driver-side dependency is LoopContext (annotation-only, postponed) ->
TYPE_CHECKING. render_prompt / _meta_prompt / _bindings come from the already
extracted prompt module (which does not import reopen).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from forgekernel.loop.prompt import _bindings, _meta_prompt, render_prompt

if TYPE_CHECKING:
    from forgekernel.loop.driver import LoopContext

__all__ = [
    "_MILESTONE_PATCH_CLASSES",
    "_REOPEN_PATCH_CLASSES",
    "_fruitless_reopen_streak",
    "_reopen_capped_block",
    "_INFEASIBLE_MENTION_BLOCK",
    "_CAUSE_ENUM",
    "_CAUSE_ENUM_AUDIT",
    "_INFEASIBLE_MENTION",
    "_parse_cause_class",
    "_teacher_prompt",
]


_MILESTONE_PATCH_CLASSES = ("additive", "structural", "split")
_REOPEN_PATCH_CLASSES = ("structural", "split")


def _fruitless_reopen_streak(meta_records) -> int:
    """How many reopen transactions in a row have closed without a new best.

    A transaction is what _judge_patch prices: one milestone patch, or a held
    one (HOLD_PATCH / still PENDING) plus every patch that joined it until a
    KEEP_PATCH / REVERT_PATCH closed it — the closing verdict is written on its
    last record only. A transaction reopens when any member is structural /
    split. KEEP_PATCH of any class resets the count (the tree moved, a reopen's
    premise may be new); a REVERT_PATCH transaction that reopened adds one; an
    additive-only REVERT_PATCH leaves it. A still-open transaction is not
    counted, and a patch that never landed (malformed: violations on record)
    is not a member of anything."""
    streak, open_, reopens = 0, False, False
    for r in meta_records:
        if r.patch_class not in _MILESTONE_PATCH_CLASSES or r.milestone_violations:
            continue
        if not open_:
            open_, reopens = True, False
        reopens = reopens or r.patch_class in _REOPEN_PATCH_CLASSES
        if r.outcome == "KEEP_PATCH":
            streak, open_ = 0, False
        elif r.outcome == "REVERT_PATCH":
            streak += 1 if reopens else 0
            open_ = False
    return streak


def _reopen_capped_block(streak: int, cap: int, *, bounce: bool = False) -> str:
    """The teacher-facing note once the reopen budget is spent."""
    head = ("\n## Reopen budget exhausted (driver check)\n"
            + ("You classified `R1` (structural reopen), but this" if bounce else "This")
            + f" run's last {streak} reopens in a row (structural / split patches) all ended in "
            f"REVERT_PATCH without a new best (budget {cap}). **Reopening delivered decisions is no "
            "longer accepted**: R1 diagnoses and structural / split patches are both refused "
            "(recorded as `REOPEN_CAPPED`; the milestone is not changed and the student redoes no "
            "decision). A reopen means the student re-delivers a whole group of already-delivered "
            "decisions on the best tree, and these readings say that route pays nothing on this "
            "tree.\n")
    ask = ("Routes still open: add on the current tree a new decision that does **not** reopen "
           "delivered decisions (R2 / G1 / G2 / G4, additive), or take the KB route of G3 / G5; if "
           "`reachability` reads BLOCKED and the target is out of reach without a reopen, report "
           "`INFEASIBLE` per the stop rules. Any KEEP_PATCH (a new best) restores the reopen "
           "budget.\n")
    tail = ("Resend a **complete** diagnosis along the routes above; if you submit R1 again, this "
            "step is refused and no patch is applied.\n") if bounce else ""
    return head + ask + tail


_INFEASIBLE_MENTION_BLOCK = (
    "\n## INFEASIBLE has only one channel (driver check)\n"
    "Your diagnosis mentions INFEASIBLE, but `cause_class` is not `INFEASIBLE`. The driver reads "
    "a stop request only from `cause_class`: INFEASIBLE written into the milestone or the "
    "evidence does not stop the loop; it only makes the student report EXHAUSTED and you diagnose "
    "yet another class with no lesion, idling until the budget runs out. Pick one, then resend a "
    "**complete** diagnosis:\n"
    "1. **You judge this run's target out of reach and request a stop** → "
    "`cause_class: INFEASIBLE`. The driver stops the whole loop at once (no more student rounds, "
    "no patch), and this run finishes on its current best; your diagnosis text and the "
    "`reachability` line go into the stop record. This is the end of this run, not a request "
    "for help.\n"
    "2. **It is not a stop request** (only a citation / negation, e.g. \"no ground for "
    "INFEASIBLE\") → resend under the original class; `deficient_locus` / `patch_direction` "
    "must point at a direction the student can build.\n")


# the diagnose cause_class is a closed enum; the agent occasionally editorializes in the
# field ("R1 (atypical…"). Extract just the enum token, tolerating trailing prose.
# INFEASIBLE is the plateau diagnosis's stop request, not a cause: the teacher
# judges the target out of reach and the run ends there — the run's verdict,
# recorded for a human to read (see _write_infeasible_notice), never a request
# for a human to hand the teacher a new direction. The milestone template tells the teacher to
# "declare INFEASIBLE and stop", but a declaration written only into the
# milestone had no channel to the driver — the run kept diagnosing "G4 @ none"
# against an EXHAUSTED student until the budget ran out.
_CAUSE_ENUM = re.compile(r"cause_class\s*[:：]\s*(G[1-5]|R[12]|INFEASIBLE)")
# The M0 audit adds one verdict the plateau diagnosis has no use for: OK — the
# milestone survived the reconciliation and there is nothing to patch. It has
# no INFEASIBLE: at M0 the distance to target is the expected state.
_CAUSE_ENUM_AUDIT = re.compile(r"cause_class\s*[:：]\s*(G[1-5]|R[12]|OK)")
_INFEASIBLE_MENTION = re.compile(r"INFEASIBLE")


def _parse_cause_class(text: str, *, allow_ok: bool = False) -> str:
    """The clean cause_class enum (G1-G5 / R1 / R2 / INFEASIBLE on the plateau
    path, G1-G5 / R1 / R2 / OK when ``allow_ok``), or "" if none present."""
    m = (_CAUSE_ENUM_AUDIT if allow_ok else _CAUSE_ENUM).search(text)
    return m.group(1) if m else ""


def _teacher_prompt(ctx: LoopContext, diagnosis: str, probe_feedback: str = "") -> str:
    extra = f"\n\n## DIAGNOSIS\n{diagnosis}\n"
    if probe_feedback:
        extra += probe_feedback
    return render_prompt(_meta_prompt(ctx, "kb_recipe_prompt.md"), _bindings(ctx), extra)
