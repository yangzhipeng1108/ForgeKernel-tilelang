"""The placeholder ledger: a decision landed in a conservative/interim form must be
declared and later restored, and no gate row may cross while a placeholder is open.

Extracted from ``loop/driver.py`` (see ``docs/driver-split-spec.md`` step 3). The
functions are byte-for-byte the ones that lived in ``driver`` -- the 411-test gate in
``tests/test_loop_driver.py`` is what proves it.

Dependencies (no import from ``driver``, so no cycle):
- ``re`` for the tag/prose regexes,
- ``forgekernel.inner.trajectory.load_records``,
- the gate-row regex ``_GATE_ROW`` and the record / turn / ctx types (annotations
  only, kept lazy by ``from __future__ import annotations``).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from forgekernel.inner.trajectory import load_records

if TYPE_CHECKING:
    from forgekernel.loop.driver import LoopContext
    from forgekernel.loop.parse import Record

# A gate row (its decision id names a gate) is what a placeholder debt blocks.
_GATE_ROW = re.compile(r"gate", re.IGNORECASE)

#
# POLICY §3 ③ lets a decision land in a conservative form only as a failure
# response, and only on the promise that the real form comes back after first
# green. The promise had no keeper: scaffold rows carry no timing, the M0 gate
# sets no performance line, and "after first green" names no iteration — when
# every sub-step is scaffolded, the first green IS the gate row. A scalar
# stand-in for the one decision the milestone had spelled out in full crossed
# the gate as the phase's best, dragged the register split with it (spill,
# then whole-function wgmma serialization), and the audit hours later was the
# first thing to notice. The ledger makes the promise a debt the driver holds:
# the row that lands a placeholder declares it in its change text,
#     [PLACEHOLDER: D0.9 → <the form the milestone specifies>]
# the row that restores the real form declares
#     [RESTORED: D0.9]
# and a gate row is refused — not committed, not evaluated — while anything is
# open. The open set rides every student turn. REVERT rows are skipped: the
# tree they described was rolled back.
_PLACEHOLDER_TAG = re.compile(r"\[PLACEHOLDER:\s*([^\]]+)\]", re.I)
_RESTORED_TAG = re.compile(r"\[RESTORED:\s*([^\]]+)\]", re.I)
_DECISION_IDS = re.compile(r"D\d+\.\d+")
_TAG_ARROW = re.compile(r"→|->|=>")
# Prose that reads as a placeholder declaration without the tag. The row is
# bounced for the tag (or a rewording) before it is committed: the ledger never
# depends on the driver guessing which member of a package is the stand-in.
_PLACEHOLDER_PROSE = re.compile(
    r"first[ -]?green\s+form|sanctioned[^.\n]{0,40}?(?:interim|placeholder|stand-in|first[ -]?green)|"
    r"transitional\s+(?:form|implementation|version)|placeholder|"
    r"stand-in\s+(?:form|solve|solver|implementation)|"
    r"interim\s+(?:form|solve|solver|implementation|path|version)|"
    r"after\s+green[^.;\n]{0,24}?swap|swap(?:ped)?\s+back\b[^.;\n]{0,24}?(?:real\s+form|D\d+\.\d+)|"
    r"swap(?:ped)?\s+(?:back|in)\s+(?:after|once|when)\b|"
    r"(?:until|once|after)\s+(?:correctness\s+is\s+|first\s+)?green\b|replaces?\s+it\s+once", re.I)


# A mention of the ledger itself, or a denial, is not a declaration: "placeholder
# ledger cleared" on a gate row / "no placeholder" on any row must not be bounced
# for a tag.
_PLACEHOLDER_PROSE_NEG = re.compile(
    r"no\s+placeholders?|placeholder[- ]free|not\s+a\s+(?:placeholder|stand-in)|non-placeholder|"
    r"without\s+(?:a\s+|any\s+)?placeholders?|ledger", re.I)


def _placeholder_ids(tag_body: str, row_decision_id: str) -> list[str]:
    """Decision ids a `[PLACEHOLDER: …]` / `[RESTORED: …]` tag names — the part
    before the arrow; a tag naming none stands for every member of the row."""
    head = _TAG_ARROW.split(tag_body, maxsplit=1)[0]
    ids = _DECISION_IDS.findall(head)
    return ids or _DECISION_IDS.findall(row_decision_id or "")


def _lineage_roots(rows) -> set:
    """The tree(s) the run's first row(s) built on — the launch baseline. A chain
    that reaches one of these has been walked to its root; any other sha no row
    owns is a break in the chain (a driver-side commit that was not reset, a
    row whose sha never made it into the record)."""
    if not rows:
        return set()
    first = min(r.iter for r in rows)
    return {getattr(r, "base_sha", "") or "" for r in rows if r.iter == first} - {""}


def _lineage_rows(rows, head: str) -> list:
    """The landed rows on the ancestry of ``head`` (each row's ``base_sha`` is the
    tree it built on), in iteration order. A head no row owns — the baseline, a
    probe commit, a pre-tag record — falls back to every non-REVERT row, and so
    does a chain that BREAKS before reaching the run's root (a ``base_sha`` no
    row owns that is not the launch baseline): rows above the break may still
    be ancestors, so they are replayed. The ledger errs towards refusing a gate
    row, never towards passing one."""
    live = [r for r in rows if r.verdict != "REVERT"]
    by_sha = {r.sha: r for r in live if r.sha}
    if not head or head not in by_sha:
        return live
    chain, seen, cur = [], set(), head
    while cur and cur in by_sha and cur not in seen:
        seen.add(cur)
        chain.append(by_sha[cur])
        cur = getattr(by_sha[cur], "base_sha", "") or ""
    if cur and cur not in seen and cur not in _lineage_roots(rows):
        return live  # broken chain: the rows above the break are not provably dropped
    return sorted(chain, key=lambda r: r.iter)


def _placeholder_ledger(records, head: str = "") -> dict[str, str]:
    """Open placeholders after replaying the rows on ``head``'s lineage in order:
    decision id -> the declared target form ('' when the tag named none). A
    workspace reset (REVERT_PATCH, a kb certification, a REVERT) rewinds the
    lineage, and a restore that landed on a branch the reset dropped no longer
    closes anything — nor does a placeholder the reset dropped stay open."""
    open_: dict[str, str] = {}
    for r in _lineage_rows(list(records), head):
        change = r.change or ""
        for m in _PLACEHOLDER_TAG.finditer(change):
            parts = _TAG_ARROW.split(m.group(1), maxsplit=1)
            target = parts[1].strip() if len(parts) > 1 else ""
            for did in _placeholder_ids(m.group(1), r.decision_id):
                open_[did] = target
        for m in _RESTORED_TAG.finditer(change):
            for did in _placeholder_ids(m.group(1), r.decision_id):
                open_.pop(did, None)
    return open_


def _placeholder_items(open_: dict[str, str]) -> str:
    return "; ".join(d + (f" (→ {t})" if t else "") for d, t in open_.items())


def _workspace_head(ctx: LoopContext) -> str:
    try:
        return ctx.backend.head() or ""
    except Exception:  # noqa: BLE001 — lineage is bookkeeping, never a precondition
        return ""


def _placeholder_note(ctx: LoopContext) -> str:
    """Standing student note: the open placeholders, until the ledger is clear."""
    try:
        open_ = _placeholder_ledger(load_records(ctx.status_path), _workspace_head(ctx))
    except (OSError, ValueError):
        return ""
    if not open_:
        return ""
    return (f"- **Placeholder ledger not cleared (no gate row accepted)**: {_placeholder_items(open_)}. "
            f"These decisions are currently in a placeholder / conservative form; when you swap "
            f"in the real form the milestone specifies, submit it as a normal decision row whose "
            f"change text opens with `[RESTORED: {', '.join(open_)}]` (you may probe first). A gate "
            f"row can be submitted only once the ledger is clear.\n")


def _placeholder_bounce(ctx: LoopContext, turn: Record,
                        head: str = "") -> tuple[str, str, str] | None:
    """Why a record block is not accepted yet: ("gate-refused" | "untagged", log line,
    feedback for the re-prompt), or None when it may land. ``head`` is the tree the
    iteration builds on (captured before its probe commits), so the ledger is read
    on the lineage the gate row would close."""
    change = turn.change or ""
    if _GATE_ROW.search(turn.decision_id or ""):
        open_ = _placeholder_ledger(load_records(ctx.status_path), head)
        if open_:
            ids = ", ".join(open_)
            return (
                "gate-refused",
                f"gate row {turn.decision_id} REFUSED — open placeholder(s) "
                f"{_placeholder_items(open_)}; [RESTORED: {ids}] must land first",
                f"- **Gate row `{turn.decision_id}` not accepted**: placeholder ledger not cleared — "
                f"{_placeholder_items(open_)}. A placeholder form must not cross a gate (the build "
                f"phase has no performance reading and the gate sets no performance line, so it "
                f"would become this phase's \"best\" as is). First swap these decisions back to the "
                f"real form the milestone specifies, submitted as normal decision rows whose change "
                f"text opens with `[RESTORED: {ids}]` (you may probe the local contract first), then "
                f"submit the gate row on its own.\n")
        if (m := _PLACEHOLDER_TAG.search(change)):
            # the debt declared ON the gate row: a shape that once crossed an
            # M0 gate (a stand-in landed with the gate row itself)
            ids = ", ".join(_placeholder_ids(m.group(1), turn.decision_id)) or "?"
            return (
                "gate-refused",
                f"gate row {turn.decision_id} REFUSED — the row itself declares "
                f"[PLACEHOLDER: {m.group(1).strip()}]; a gate row cannot land a stand-in",
                f"- **Gate row `{turn.decision_id}` not accepted**: this row itself declares a "
                f"placeholder `[PLACEHOLDER: {m.group(1).strip()}]`. A gate row must not carry a "
                f"placeholder: first land {ids} in the real form the milestone specifies as a normal "
                f"decision row (or record it in the ledger and, once swapped back, "
                f"`[RESTORED: {ids}]`), then submit the gate row on its own.\n")
        if _RESTORED_TAG.search(change):
            return None
        # a gate row whose prose reads as a placeholder is bounced for the tag
        # like any other row (below) — it must not slip past the ledger untagged
    elif _PLACEHOLDER_TAG.search(change) or _RESTORED_TAG.search(change):
        return None
    if (m := _PLACEHOLDER_PROSE.search(change)):
        seen = change[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
        if _PLACEHOLDER_PROSE_NEG.search(seen):
            return None
        return (
            "untagged",
            f"{turn.decision_id}: change text reads as a placeholder declaration "
            f"(…{seen.strip()}…) without a [PLACEHOLDER: …] tag — bounced for the tag",
            f"- Your change text reads as landing a decision in a placeholder / transitional form "
            f"first (\"…{seen.strip()}…\"), but it carries no `[PLACEHOLDER: D<x>.<y> → <real form>]` "
            f"tag. A placeholder must be recorded in the ledger: put that tag in the first sentence "
            f"of the change (which decision, and what it will become), and before the gate row it "
            f"must be swapped back with `[RESTORED: D<x>.<y>]`; if this is not a placeholder, reword "
            f"it and resend the record block.\n")
    return None
