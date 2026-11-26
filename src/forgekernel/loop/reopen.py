"""Reopen budget + phase exhaustion + lever pricing.

Split out of loop/driver.py (docs/driver-split-spec.md step 5). Two rings:

1. Reopen-budget core (a true leaf): the patch-class enums, the fruitless reopen
   streak, the reopen cap prompt block, the infeasible-mention guard, the
   cause-class enum + parser, and the teacher (kb-recipe) prompt.
2. Phase exhaustion + lever pricing: the phase-verdict machinery, live-record
   filtering, the lever price / ablation / priced notes, and the lever-pricing
   run (_run_lever_pricing) with its refusal/normalise/bank/persist helpers. This
   ring is what turns the price table into a phase-level fact and drives the
   priced lever experiments.

Imports come only from already-extracted modules (prompt, pricing, comparisons,
shared) plus stdlib/job_env/meta_shapes — none of which import reopen, so no
top-level cycle. BackendError is defined early in driver (before this module is
imported back) -> imported at module load. LoopContext is annotation-only ->
TYPE_CHECKING. Driver helpers still only in driver (the certify/harvest/agent
core) are lazy shims so this module never imports driver at load time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from forgekernel.inner.trajectory import load_records
from forgekernel.loop import meta_shapes
from forgekernel.loop.comparisons import (
    _edits_outside_notes,
    _log,
    _note_discarded_edits,
    _note_env_failure,
    _note_env_ok,
)
from forgekernel.loop.job_env import env_failure
from forgekernel.loop.pricing import (
    _ABLATE_FAIL_RE,
    _LEVER_MAX_ABLATIONS,
    _LEVER_MAX_SUBSETS,
    _LEVER_NOISE_DEFAULT_PCT,
    _LEVER_PRICE_RETRIES,
    _LEVER_PRICING_CMD,
    _LEVER_SPEC_NAME,
    _lever_price_submit,
    _price_log_lost_reason,
    _price_log_verdict,
    _price_role_pass_killed,
    _stamp_price_row,
    parse_lever_prices,
)
from forgekernel.loop.prompt import _bindings, _meta_prompt, render_prompt
from forgekernel.loop.shared import (
    _DECISION_ID,
    _LEVER_DESIGN_BLOCK,
    TRUNCATION_STORM_BUDGET,
)
from forgekernel.loop.driver import BackendError  # defined early in driver; safe at load
from forgekernel.loop.parse import parse_probe_verdict

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
    "_PHASE_EXHAUSTED_CLASSES",
    "_table_noise_pct",
    "_PHASE_VERDICT_LINE",
    "_current_phase",
    "_delivered_decisions",
    "_REOPEN_CLOSURE_RE",
    "_live_records",
    "_phase_exhaustion",
    "_phase_exhausted_block",
    "_parse_phase_verdict",
    "_verify_phase_verdict",
    "_phase_close_note",
    "_phase_priced_note",
    "_lever_price_lost_block",
    "_lever_price_block",
    "_ablation_block",
    "_fmt_pct",
    "_merge_lever_tables",
    "_run_lever_pricing",
    "_lever_spec_refused",
    "_normalise_ablations",
    "_lever_refusal_block",
    "_banked_lever_table",
    "_persist_lever_table",
]


def _agent_turn(*args, **kwargs):  # lazy shim -> driver._agent_turn
    from forgekernel.loop import driver

    return driver._agent_turn(*args, **kwargs)


def _bank_evidence(*args, **kwargs):  # lazy shim -> driver._bank_evidence
    from forgekernel.loop import driver

    return driver._bank_evidence(*args, **kwargs)


def _evidence_dir(*args, **kwargs):  # lazy shim -> driver._evidence_dir
    from forgekernel.loop import driver

    return driver._evidence_dir(*args, **kwargs)


def _lever_pricing_jobs_text(*args, **kwargs):  # lazy shim -> driver._lever_pricing_jobs_text
    from forgekernel.loop import driver

    return driver._lever_pricing_jobs_text(*args, **kwargs)


def _milestone_dag(*args, **kwargs):  # lazy shim -> driver._milestone_dag
    from forgekernel.loop import driver

    return driver._milestone_dag(*args, **kwargs)


def _seed_pricing_toolkits(*args, **kwargs):  # lazy shim -> driver._seed_pricing_toolkits
    from forgekernel.loop import driver

    return driver._seed_pricing_toolkits(*args, **kwargs)



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


# --- phase exhaustion -----------------------------------------------------------
#
# The price table is a batch measurement of exactly the knobs a phase asks the
# student to A/B one iteration at a time. When every lever a phase owns reads
# null on the best tree, the phase's remaining walk can only re-measure the
# table — but nothing consumed the table: the driver handed it to the teacher
# and the student, who never sees it, kept walking the phase (observed on a
# dsa_sparse run: five levers null at the M0 audit, then six iterations and
# two meta steps of D1.1 wait-depth A/Bs inside the noise band, one of them
# −32%). The reading below turns the table into a phase-level fact and makes
# the diagnosis answer it: close the phase (the student submits the gate row
# next, judged by the gate's own lines) or name the ONE decision of the phase
# no lever priced and walk that. Unanswered twice, the driver closes — a gate
# row that fails just returns the student to normal work with its diagnosis,
# while a phase walked for nothing is a budget the run never gets back.
_PHASE_EXHAUSTED_CLASSES = ("null", "noop", "harmful")


def _table_noise_pct(table: list) -> float:
    """The noise floor the pricing job read its table at (the tool's `PRICE
    noise_floor=` line, banked on the stack row); the tool's own default when
    a table predates the field."""
    for t in table or []:
        if t.get("kind") == "stack" and t.get("noise_pct") is not None:
            return float(t["noise_pct"])
    return _LEVER_NOISE_DEFAULT_PCT
# the value stays on the line: `\s*` would run through a newline and read the
# NEXT field's first token as the verdict of an empty `phase_verdict:` line
_PHASE_VERDICT_LINE = re.compile(r"phase_verdict[ \t]*[:：][ \t]*([^\n]*)")


def _current_phase(records, phases) -> int | None:
    """The phase the student is walking: the lowest of ``phases`` (the k of every
    ``D<k>.<n>`` the milestone declares) whose ``M<k>-gate`` has no KEEP row.
    None when every declared gate is closed (or nothing is declared)."""
    closed = {int(m.group(1)) for r in records
              if r.verdict == "KEEP" and (m := re.match(r"M(\d+)-gate$", r.decision_id or ""))}
    return next((k for k in sorted(set(phases)) if k not in closed), None)


def _delivered_decisions(records) -> set:
    """Decision ids with a KEEP row (``D1.1+D1.3`` counts both)."""
    out: set = set()
    for r in records:
        if r.verdict == "KEEP":
            out.update(p.strip() for p in (r.decision_id or "").split("+")
                       if _DECISION_ID.match(p.strip()))
    return out


_REOPEN_CLOSURE_RE = re.compile(r"closure \[([^\]]*)\] is reopened")


def _live_records(records, voided=None) -> list:
    """The rows that still describe a standing tree — what every phase /
    parity reading is computed on.

    Two kinds of row are on file but void (the student's ledger is append-only):
    rows inside a segment a REVERT_PATCH rolled back (``voided``: [lo, hi) index
    ranges), and KEEP rows for decisions a later in-place REOPEN row reopened
    (the row itself tells the student those KEEPs are void; the gate KEEP of a
    reopened phase is void with them — its closure was premised on the old
    values). A REOPEN-REVERT row restores what its REOPEN voided."""
    voided = voided or []
    stack: list = []          # (reopen row index, decision ids, phase ks) still standing
    for i, r in enumerate(records):
        if r.decision_id == "REOPEN":
            m = _REOPEN_CLOSURE_RE.search(r.change or "")
            ids = {p.strip() for p in (m.group(1).split(",") if m else []) if p.strip()}
            ks = {int(mm.group(1)) for d in ids if (mm := re.match(r"D(\d+)\.", d))}
            stack.append((i, ids, ks))
        elif r.decision_id == "REOPEN-REVERT" and stack:
            stack.pop()
    out = []
    for i, r in enumerate(records):
        if any(lo <= i < hi for lo, hi in voided):
            continue
        if r.verdict == "KEEP" and stack:
            parts = {p.strip() for p in (r.decision_id or "").split("+")}
            gate = re.match(r"M(\d+)-gate$", r.decision_id or "")
            if any(i < ri and ((parts & ids) or (gate is not None and int(gate.group(1)) in ks))
                   for ri, ids, ks in stack):
                continue
        out.append(r)
    return out


def _phase_exhaustion(ctx: LoopContext, records, table: list) -> dict | None:
    """The phase-exhaustion reading of a price table, or None.

    Non-None when the phase the student is walking has at least one lever on
    the table and every one of them is null / noop: {"phase", "gate",
    "levers", "priced" (phase decisions some lever moved), "unpriced" (phase
    decisions no lever moved and no KEEP delivered), "delivered"}. A phase with
    an independent or coupled lever is not exhausted (the lever IS the way
    forward); a phase past the milestone's last gate has nothing to close.
    """
    levers = [t for t in table if t.get("kind") == "lever"]
    if not levers:
        return None
    dag = _milestone_dag(ctx)
    by_phase: dict = {}
    for d in dag:
        if (m := re.match(r"D(\d+)\.", d.id)):
            by_phase.setdefault(int(m.group(1)), []).append(d.id)
    k = _current_phase(records, by_phase)
    if k is None:
        return None
    phase_ids = sorted(by_phase[k])
    own = [lv for lv in levers if set(lv.get("decisions") or []) & set(phase_ids)]
    if not own:
        return None
    if any(str(lv.get("cls")) not in _PHASE_EXHAUSTED_CLASSES for lv in own):
        return None
    # A subset (the teacher's, or the tool's all-pos) that moves one of this
    # phase's decisions and reads above the noise floor IS a way forward: the
    # single-add / remove classes of its members can all read null while the
    # combination pays (the reading rule handed to the teacher says the subset
    # wins over the class column — the phase reading must agree with it).
    noise = _table_noise_pct(table)
    for v in table:
        if v.get("kind") != "variant" or not (set(v.get("decisions") or []) & set(phase_ids)):
            continue
        if v.get("delta_pct") is not None and v["delta_pct"] > noise:
            return None
    priced = sorted({d for lv in own for d in lv["decisions"] if d in phase_ids})
    delivered = sorted(_delivered_decisions(records) & set(phase_ids))
    unpriced = [d for d in phase_ids if d not in priced and d not in delivered]
    return {"phase": f"M{k}", "gate": f"M{k}-gate",
            "levers": [lv["name"] for lv in own],
            "priced": priced, "unpriced": unpriced, "delivered": delivered}


def _phase_exhausted_block(ex: dict) -> str:
    unpriced = (", ".join(f"`{d}`" for d in ex["unpriced"])
                if ex["unpriced"] else "(none: every undelivered decision of this phase has a priced lever)")
    return (
        f"\n## Phase levers exhausted (the driver read this off the pricing table)\n"
        f"The student is working on **{ex['phase']}**. This phase's levers in the pricing table, "
        f"{', '.join(f'`{n}`' for n in ex['levers'])} (moving {', '.join(f'`{d}`' for d in ex['priced'])}), "
        f"**all measured null / noop** on the best tree — any further point-by-point A/B in this phase "
        f"would only re-measure that table (one point per iter, each inside the noise band). "
        f"Delivered in this phase: {', '.join(ex['delivered']) or '(none)'}; "
        f"decisions with no priced lever and not delivered: {unpriced}.\n"
        f"The output block **must** add one line (driver check: a missing line is bounced once, "
        f"missing again is treated as close):\n"
        f"```\n"
        f"phase_verdict: close | continue:<D id>\n"
        f"```\n"
        f"- `close`: the remaining decisions of this phase count as settled (the pricing table is "
        f"their measurement); next round the student **submits the `{ex['gate']}` row directly** "
        f"(still submitted on its own, still judged by the gate's own acceptance bar; a gate FAIL is "
        f"attributed as a FAIL, without going back to tune this phase's knobs). If the gap sits in a "
        f"next-phase decision (supply organisation, per-element refinement, ...), close is the way to "
        f"get the student there fastest; to pull a next-phase decision forward, close first and then "
        f"adjust it in the patch.\n"
        f"- `continue:<D id>`: may only name one decision from the \"no priced lever\" list above, and "
        f"`evidence` must say which lever of it the pricing table has not measured; the driver "
        f"validates the id and lets the student continue on it.\n"
        f"This line sits beside `cause_class` and the other fields and does not replace them: G3/G5 "
        f"still add to the KB, R1 still prices first.\n")


def _parse_phase_verdict(text: str) -> str:
    m = _PHASE_VERDICT_LINE.search(text or "")
    if not m:
        return ""
    # teachers gloss the token ("close (M1 fully measured)"), quote it (`close`) and
    # space the continue form ("continue: D1.4" / "continue:`D1.4`"): cut at
    # either paren family (as _REOPEN_PRICE_SRC and the locus parser do), drop
    # the quoting, close the gap after "continue:", then keep the first token.
    raw = re.split(r"[（(]", m.group(1))[0].strip().strip("`'\"* ")
    raw = re.sub(r"^continue\s*[:：]\s*[`'\"*]*", "continue:", raw, flags=re.I)
    tok = raw.split()[0] if raw.split() else ""
    return tok.strip("`'\"*.;，。").strip()


def _verify_phase_verdict(verdict: str, ex: dict) -> str | None:
    """None when ``verdict`` answers the exhaustion reading; else why not."""
    if not verdict:
        return "no phase_verdict line"
    v = verdict.lower()
    if v == "close":
        return None
    if v.startswith("continue:"):
        did = verdict.split(":", 1)[1].strip().strip("`'\"*")
        if did in ex["unpriced"]:
            return None
        if did in ex["priced"]:
            return f"continue:{did} names a decision the table already priced null"
        if did in ex["delivered"]:
            return f"continue:{did} names a decision already delivered (KEEP row)"
        return f"continue:{did} is not a decision of {ex['phase']} without a priced lever"
    return f"phase_verdict {verdict!r} is neither close nor continue:<D id>"


def _phase_close_note(ex: dict, verdict: str, job: str = "") -> str:
    """The student-facing notice for a phase verdict (rides every turn until
    the gate row lands; see LoopContext.phase_close)."""
    priced = ", ".join(f"`{d}`" for d in ex["priced"]) or "(none)"
    head = (f"- **Phase close (driver read of the pricing table, teacher verdict `{verdict}`)**: the "
            f"pricing round measured {ex['phase']}'s levers {', '.join(f'`{n}`' for n in ex['levers'])} "
            f"on the best tree, all null / noop"
            f"{f' (job {job})' if job else ''}; the decisions they move, {priced}, are settled by "
            f"that measurement — **no more point-by-point A/B**. ")
    if verdict.startswith("continue:"):
        walk = verdict.split(":", 1)[1].strip()
        return (head + f"The teacher ruled that only `{walk}` is still worth doing in this phase "
                f"(it has a lever the pricing table has not measured): do `{walk}` this round, and "
                f"once it is delivered, **submit the `{ex['gate']}` row directly next round** "
                f"(the gate row is still submitted on its own and judged by the gate's own acceptance "
                f"bar; a FAIL is attributed as a FAIL).\n")
    left = ", ".join(f"`{d}`" for d in ex["unpriced"]) or "(none)"
    return (head + f"The other undelivered decisions of this phase, {left}, count as settled too; "
            f"**submit the `{ex['gate']}` row directly this round** "
            f"(the gate row is still submitted on its own and judged by the gate's own acceptance "
            f"bar; a FAIL is attributed as a FAIL, without going back to tune this phase's knobs). "
            f"Once the gate passes, move to the next phase in phase order.\n")


def _phase_priced_note(pc: dict, verdict: str) -> str:
    """The reduced notice after a closed phase's gate row FAILED: the phase is
    open again, but the table that closed it still describes this best tree."""
    levers = ", ".join(f"`{n}`" for n in pc.get("levers", [])) or "(the levers on the pricing table)"
    return (f"- **Phase levers already priced (driver)**: the `{pc['gate']}` row was submitted but "
            f"judged {verdict}, so {pc['phase']} is open again; but the pricing round's reading on this "
            f"best tree — {pc['phase']}'s levers {levers} all null / noop — still holds: **do not go "
            f"back to point-by-point A/B of these knobs** (every point would land inside the noise "
            f"band); act on the teacher's attribution of the gate FAIL instead. This notice is "
            f"withdrawn automatically when the best tree changes (any KEEP lands) or a structural "
            f"patch lands.\n")


def _lever_price_lost_block(note: str) -> str:
    return (f"\n## The pricing job produced no table\n{note}**None of the levers designed this time "
            f"was priced**: a citation of one may only be marked `est:<x>%`, never `priced_by=`; a "
            f"structural reopen still needs a `reopen_price` that cites a measured source.\n")


def _lever_price_block(table: list) -> str:
    """The measured price table for the diagnose prompt, with the reading rules."""
    levers = [t for t in table if t.get("kind") == "lever"]
    stack = next((t for t in table if t.get("kind") == "stack"), None)
    rows = ["| lever | add-one | remove-one | class | Δspin(ns, +L) | Δspin(ns, all−L) |",
            "|---|---|---|---|---|---|"]
    for lv in levers:
        rows.append(f"| `{lv['name']}` | {_fmt_pct(lv['add_pct'])} | {_fmt_pct(lv['remove_pct'])} | "
                    f"**{lv['cls']}** | {lv.get('spin_add_ns')} | {lv.get('spin_all_minus_ns')} | "
                    f"{', '.join(lv.get('decisions') or []) or '—'} |")
    rows[0] += " decisions |"
    rows[1] += "---|"
    block = ("\n## Pricing table (measured by the driver on the best tree: add-one / remove-one / "
             "all-on, same-job paired ratios)\n")
    block += meta_shapes.scope_note(stack)
    if levers:
        block += "\n".join(rows) + "\n"
    else:
        block += "(No candidate levers this step; only the ablation table follows.)\n"
    trees = {str(lv.get("priced_on")) for lv in levers if lv.get("priced_on")}
    if len(trees) == 1:
        (tree,) = trees
        jobs = sorted({str(lv.get("job")) for lv in levers if lv.get("job")})
        block += (f"(Measured tree = best `{tree[:10]}`" + (f", job {', '.join(jobs)}" if jobs else "")
                  + ". **The price tags hold only for this tree**: once best moves, the driver "
                  "re-measures every lever the milestone cites on the new tree automatically, and "
                  "citations that reversed / faded are marked SUSPENDED — see the \"Price-tag review\" "
                  "block.)\n")
    variants = [t for t in table if t.get("kind") == "variant"]
    if variants:
        vrows = ["| subset | members | vs base | Δspin(ns) | question it answers |", "|---|---|---|---|---|"]
        for v in variants:
            vrows.append(f"| `{v['name']}` | {', '.join(v.get('levers') or [])} | "
                         f"{_fmt_pct(v.get('delta_pct'))} | {v.get('spin_ns')} | {v.get('question') or '—'} |")
        block += ("\n**Subsets (the combinations you named in the design turn; `all-pos` is the "
                  "driver's automatic \"all-on minus the harmful levers\", re-measured in a second pass "
                  "interleaved with base and read against that pass's own base)**:\n"
                  + "\n".join(vrows) + "\n")
    if stack:
        block += (f"\nall-on = {_fmt_pct(stack.get('all_pct'))}; sum of add-ones = "
                  f"{_fmt_pct(stack.get('sum_singles_pct'))}; "
                  f"coupling = {_fmt_pct(stack.get('coupling_pp'))} (negative = they eat each other, "
                  f"positive = they enable each other). "
                  f"base {stack.get('base_ms')} ms → all {stack.get('all_ms')} ms; "
                  f"spin exposure {stack.get('spin_base_ns')} → {stack.get('spin_all_ns')} ns; "
                  f"tensor {stack.get('tensor_base')} → {stack.get('tensor_all')}%.\n")
        if stack.get("dropped"):
            pos = stack.get("all_pos_pct")
            block += (f"**Harmful levers {', '.join(f'`{d}`' for d in stack['dropped'])} "
                      f"(negative add-one, and all-on does not depend on them) went into all-on**, "
                      f"and they skew every `remove` reading (remove is measured on the all-on that "
                      f"contains them) — "
                      + (f"all-on without them, `all-pos` = {_fmt_pct(pos)} (members "
                         f"{', '.join(stack.get('all_pos_levers') or [])}); all-on and the class column "
                         f"go by it.\n"
                         if pos is not None else
                         "the driver could not measure `all-pos` (fewer than two members left), so "
                         "all-on and the class column are not trustworthy; go by the add-ones and "
                         "the subsets.\n"))
    block += (
        "\n**How to read it (hard rules)**: `independent` = this group pays on its own; fix in "
        "add order. `coupled` = ≈ 0 alone but taking it out of all-on loses — you split a group "
        "that must go in together: it is the companion of another candidate in the table, and the "
        "diagnosis **must merge it with the candidate that depends on it into one ⊗ package** "
        "(price the package with −remove); it must not be judged useless on its own. "
        "`dependent` = the add-one pays but taking it out of all-on loses nothing — someone else "
        "swallowed the gain; rank it last. `null` = ≈ 0 both ways; drop it. `harmful` = negative "
        "add-one **and** taking it out of all-on loses nothing — the direction is backwards and must "
        "not be proposed as a direction again (base never had it; its reverse is not a gain). A "
        "negative add-one whose removal from all-on does lose is `coupled`, not harmful — it costs "
        "on its own but all-on cannot do without it; treat it as a ⊗ package. `noop` = the edit did "
        "not change the cubin; the spelling is ineffective. `broken` = the variants containing it "
        "failed (did not compile / were incorrect): this lever **was not priced**, and the all-on "
        "and remove-one readings that contain it are n/a too — the direction is not falsified, the "
        "spelling is rejected; either rewrite it and propose it again, or mark it unpriced in the "
        "diagnosis. `unstacked` = the add-one has a reading but the all-on variant failed "
        "correctness, so its coupling with the others is unmeasured. **Every price the diagnosis "
        "cites must be a measured value from this table** (add for an independent item; −remove, a "
        "subset delta or an all-on difference for a package); write `priced_by=<lever or subset name "
        "from the table>` to point at it; a quantity the table does not have may only be marked "
        "`est:<x>%` and must not be cited as measured. "
        "The class is derived mechanically by the driver from add-one / remove-one: `coupled` only "
        "says that **some** lever depends on it, not which one — the subset deltas you specified "
        "are the judge; when a subset and the class disagree, the subset wins. "
        "The ceiling of the pricing table: the spin exposure that all-on still leaves belongs to "
        "something outside this batch of candidates; the diagnosis must say whose it is.\n")
    block += meta_shapes.shape_block(table)
    block += _ablation_block(table)
    return block


def _ablation_block(table: list) -> str:
    """The ablation rows of a price table (design-turn `ablations` or a JSON
    verdict probe) with their reading rules, or nothing."""
    abl = [t for t in table if t.get("kind") == "ablation"]
    if not abl:
        return ""
    noise = next((a.get("noise_pct") for a in abl if a.get("noise_pct") is not None),
                 _LEVER_NOISE_DEFAULT_PCT)
    rows = ["| ablation | Δ(NCU kernel clock) | status | correctness | Δdevice | decisions | hypothesis |",
            "|---|---|---|---|---|---|---|"]
    for a in abl:
        rows.append(f"| `{a['name']}` | {_fmt_pct(a.get('delta_pct'))} | {a.get('status') or '—'} | "
                    f"{a.get('correctness')} | {_fmt_pct(a.get('device_delta_pct'))} | "
                    f"{', '.join(a.get('decisions') or []) or '—'} | {a.get('hypothesis') or '—'} |")
    return (
        "\n**Ablation table (in the same job the driver removes / degrades a mechanism already on the "
        "tree and re-measures the NCU kernel clock in the same pass as base; "
        f"noise floor {noise:.2f}%)**:\n" + "\n".join(rows) + "\n"
        "**How to read it (hard rules)**: Δ positive and above the noise floor = faster without it = "
        "**the cost this mechanism charges now** (hypothesis holds); inside the band = this mechanism "
        "is not the current main term (hypothesis falsified); **Δ negative beyond the noise = slower "
        "without it = this mechanism is helping** (hypothesis falsified, and it is not something to "
        "reopen); `n/a` = no reading (status: `hung` the variant hung, `nodata` ncu captured nothing, "
        "`budget` the ablation budget ran out), **not 0**. "
        "An ablation reading is the upper bound on **how much fixing this mechanism can win back at "
        "most**, not the gain of any particular fix: it can convict, and it can price an R1 "
        "(`reopen_price: <Δ> via ablation:<name>`, recorded by the driver as an upper bound), but it "
        "must not be cited as the gain of an additive patch (a patch's gain goes through a measured "
        "candidate lever). `correctness=fail` is expected (an ablation may compute wrong results); "
        "Δdevice has a reading only when the ablation happens to stay correct.\n")


def _fmt_pct(x):
    return "n/a" if x is None else f"{x:+.2f}%"


def _merge_lever_tables(records, current: list) -> list:
    """The levers this run has priced: earlier steps' tables (the reflection
    shows them as priced, so a citation of one must verify) under the current
    step's (which wins on a name clash). Only lever rows travel; the stack row
    is the current step's own."""
    merged: dict = {}
    for r in records:
        for lv in (getattr(r, "lever_prices", None) or []):
            if lv.get("kind") in ("lever", "variant", "ablation") and lv.get("name"):
                merged[lv["name"]] = dict(lv, step=r.meta_step)
    out = list(merged.values())
    names = {lv["name"] for lv in current if lv.get("kind") in ("lever", "variant", "ablation")}
    out = [lv for lv in out if lv["name"] not in names] + list(current)
    return out


def _run_lever_pricing(ctx: LoopContext, priced: "PricedTrees", bindings: dict,
                       diagnose_extra: str, prompt_file: str = "diagnose_prompt.md") -> list:
    """The lever pricing round: one design turn, one GPU job, a measured table.

    Runs at every teacher entry that has a priced tree — the plateau diagnosis
    and the M0 audit alike (``prompt_file`` is the entry's own prompt, so the
    design turn reads the audit's question at an audit).

    Design turn: the teacher reads the same evidence the diagnosis will read
    (ledger, dossier diff, role exposure) and writes ``notes/levers.json`` — up
    to ``lever_pricing_max_levers`` composable exact-text edits of the best
    tree's kernel file, each a lever it would otherwise have to estimate a
    price for. The driver seeds ``lever_pricing_run.py``, commits, runs it on
    the best tree and parses the LEVER / STACK lines. No table (design turn
    wrote nothing, job refused an edit, budget spent) is logged and the
    diagnosis proceeds as before — the round never blocks a step.
    """
    if ctx.lever_pricing_rounds <= 0 or ctx.stall_pricing is False:
        return []
    be = ctx.backend
    best_ref = priced.best_sha or priced.head_sha
    if not best_ref:
        return []
    banked = _banked_lever_table(ctx, best_ref)
    if banked:
        # the same best tree was priced earlier in this run (a retried meta step,
        # an audit followed by a plateau on the same first green): the readings
        # are on the record, another design turn + job would only re-buy them
        _log(f"meta/lever-price: reusing the table priced on best {best_ref[:10]} "
             f"earlier this run ({len([t for t in banked if t.get('kind') == 'lever'])} lever(s))")
        return banked
    if ctx.now() > ctx.deadline_epoch - 3600:
        _log("meta/lever-price: < 60 min of budget left — skipping the pricing round")
        return []
    spec_path = Path(ctx.workspace) / "notes" / _LEVER_SPEC_NAME
    pre_head = be.head()
    if lost := _edits_outside_notes(be, pre_head):
        _note_discarded_edits(ctx, "meta/lever-price (checkout of the best tree)", lost)
    try:
        be.reset(best_ref)
    except BackendError as e:
        _log(f"meta/lever-price: could not check out best {best_ref[:10]} ({e}) — skipping")
        return []
    table: list = []
    storm_budget = [TRUNCATION_STORM_BUDGET]
    try:
        _seed_pricing_toolkits(ctx.workspace, be)
        design_extra = diagnose_extra + _LEVER_DESIGN_BLOCK.replace(
            "{MAX_LEVERS}", str(ctx.lever_pricing_max_levers)).replace(
            "{KERNEL_FILE}", str(ctx.kernel_file)).replace(
            "{PRICING_JOBS}", _lever_pricing_jobs_text(ctx))
        for attempt in range(1 + _LEVER_PRICE_RETRIES):
            try:
                spec_path.unlink()
            except OSError:
                pass
            out = _agent_turn(
                be, render_prompt(_meta_prompt(ctx, prompt_file), bindings,
                                  extra=design_extra),
                label="meta/lever-design", storm_budget=storm_budget,
                # a legal design turn either wrote the lever file or declined
                left_a_trace=lambda o: (spec_path.is_file()
                                        or re.search(r"^LEVERS:\s*none\b", o, re.M) is not None))
            if not spec_path.is_file():
                _log("meta/lever-price: the teacher declared no composable levers "
                     "(or wrote no levers.json) — diagnosing without a price table")
                return []
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                levers = spec.get("levers") or []
                ablations = spec.get("ablations") or []
            except (OSError, json.JSONDecodeError) as e:
                _log(f"meta/lever-price: levers.json unreadable ({e}) — skipping")
                return []
            if not levers and not ablations:
                _log("meta/lever-price: levers.json names no lever and no ablation "
                     "— diagnosing without a price table")
                return []
            levers = levers[:ctx.lever_pricing_max_levers]
            known_ids = {d.id for d in _milestone_dag(ctx)}
            ablations = _normalise_ablations(ctx, ablations, known_ids)
            for lv in levers:
                # one token on the LEVER line, so the parser and every later
                # citation (`priced_by=` / `lever:<name>`) see the same name
                lv["name"] = re.sub(r"\s+", "_", str(lv.get("name", "")).strip()) or "unnamed"
                # the milestone decisions this lever moves the value of — the
                # link the phase-exhaustion reading needs (a table row alone
                # cannot say which phase it priced); unknown ids are dropped
                # loudly, never bounced (a mis-typed id is not worth a turn)
                ids = [str(d).strip() for d in (lv.get("decisions") or [])]
                bad = [d for d in ids if d not in known_ids]
                if bad:
                    _log(f"meta/lever-price: lever {lv['name']} names decision(s) not in "
                         f"the milestone {bad} — dropped from its attribution")
                lv["decisions"] = [d for d in ids if d in known_ids]
                if not lv["decisions"]:
                    _log(f"meta/lever-price: lever {lv['name']} names no milestone decision "
                         f"— priced, but it cannot count toward any phase's exhaustion")
            # the teacher's subsets: names one token, members lever names
            subsets = []
            for sub in (spec.get("variants") or [])[:_LEVER_MAX_SUBSETS]:
                if not isinstance(sub, dict):
                    continue
                sub = dict(sub)
                sub["name"] = re.sub(r"\s+", "_", str(sub.get("name", "")).strip())
                sub["levers"] = [re.sub(r"\s+", "_", str(m).strip()) for m in (sub.get("levers") or [])]
                subsets.append(sub)
            spec["variants"] = subsets
            spec["levers"] = levers
            spec["ablations"] = ablations
            shapes = meta_shapes.card_meta_workloads(ctx.workspace)
            spec["rounds"] = meta_shapes.rounds_for(ctx.lever_pricing_rounds, len(shapes))
            spec["kernel_file"] = str(ctx.kernel_file)
            # The tool's own build check, run here on the same tree first: a
            # subset that shadows a lever, a member that is not a stack lever,
            # an `old` that does not apply — each used to cost a GPU job to
            # learn (PRICE-FAIL lever-edit). Skipped only when the kernel file
            # cannot be read locally; the job then judges as before.
            if fail := _lever_spec_refused(ctx, spec):
                _log(f"meta/lever-price: levers.json refused before the job — {fail}")
                if attempt < _LEVER_PRICE_RETRIES:
                    design_extra += _lever_refusal_block(ctx, fail)
                    continue
                return []
            spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
            be.commit(f"meta-price: {len(levers)} lever(s) on {best_ref[:10]}")
            branch = f"{ctx.branch_prefix}-leverprice"
            be.push_branch(branch)
            solo = [lv["name"] for lv in levers if lv.get("in_stack", True) is False]
            _log(f"meta/lever-price: pricing {len(levers)} lever(s) "
                 f"[{', '.join(lv.get('name', '?') for lv in levers)}] on best {best_ref[:10]} "
                 + (f"(one job per workload of {', '.join(shapes)}, merged: " if shapes else "(one job: ")
                 + f"base / +each / all / all-each x {spec['rounds']} rounds"
                 + (f"; {len(subsets)} teacher subset(s) [{', '.join(sb['name'] for sb in subsets)}]"
                    if subsets else "")
                 + (f"; solo-only [{', '.join(solo)}]" if solo else "")
                 + (f"; {len(ablations)} ablation(s) [{', '.join(a['name'] for a in ablations)}] "
                    f"on the NCU clock" if ablations else "") + ")")
            try:
                log, price_job = _lever_price_submit(ctx, be, branch, _LEVER_PRICING_CMD,
                                                     "meta/lever-price")
            except BackendError as e:
                _log(f"meta/lever-price: job failed ({e}) — diagnosing without a price table")
                return []
            parsed, stack, fail = parse_lever_prices(log)
            if _price_log_verdict(log) is None:
                # No verdict: the archive was read before the job finished
                # writing (observed in an offline replay: a job log came back 60 lines
                # long, cut mid-variant, and an empty table was banked as if
                # measured) or the tool never ran. Not a table — whatever LEVER
                # rows the cut log holds are the timing copy, incomplete.
                job = str(getattr(be, "last_job_id", "") or "")
                why = _price_log_lost_reason(log)
                _log(f"meta/lever-price: job {job or '?'} log carries no PRICE-OK / PRICE-FAIL / "
                     f"ABLATE verdict ({why}) — diagnosing without a price table")
                ctx.lever_price_lost = (
                    f"Pricing job {job or '?'} was bought but its log has no PRICE-OK / PRICE-FAIL / "
                    f"ABLATE verdict line ({why}); this entry **has no measured price table**. ")
                return []
            if (killed := _price_role_pass_killed(log)) and _price_log_verdict(log) == "PRICE-TIMED":
                # the timing copy is whole; the kill took the role pass (and any
                # ablation pass after it) — priced from the timing copy, role
                # columns n/a, rather than bought again to be killed again
                _log(f"meta/lever-price: job {getattr(be, 'last_job_id', '') or '?'}: {killed} after "
                     f"the timing table (PRICE-TIMED) — the role pass was cut by the job timeout; "
                     f"pricing from the timing copy (spin / tensor columns n/a)")
            if fail and (why := env_failure(log)) is not None:
                # The base IS the best tree, which passed its own eval: an import
                # that dies in the pricing job is the job's environment (the tool's
                # bare `python3`, loop/job_env.py). The tool still says
                # "base-failed-correctness", and a retry design turn read that as
                # its tree or its levers.json — neither can fix an interpreter.
                _note_env_failure(ctx, "lever-price", why)
                job = str(getattr(be, "last_job_id", "") or "")
                _log(f"meta/lever-price: PRICE-FAIL {fail} — job environment ({why}), not "
                     f"the base tree and not levers.json: no design retry, diagnosing "
                     f"without a price table")
                ctx.lever_price_lost = (
                    f"Pricing job {job or '?'} could not start in the job environment (`{why}`) — the "
                    f"base tree's own eval passed, so this is neither the base tree nor levers.json "
                    f"but the job environment (the operator tree's gpu_entry.sh / eval/, or the "
                    f"image); ")
                return []
            if fail:
                _log(f"meta/lever-price: PRICE-FAIL {fail}")
                if attempt < _LEVER_PRICE_RETRIES:
                    design_extra += _lever_refusal_block(ctx, fail)
                    continue
                return []
            _note_env_ok(ctx, "lever-price")
            if (afail := _ABLATE_FAIL_RE.search(log)):
                # the lever table stands; only the ablation pass is lost
                _log(f"meta/lever-price: ABLATE-FAIL {afail.group(1).strip()}")
            for lv in parsed:
                lv.setdefault("kind", "lever")
            attribution = {lv["name"]: lv["decisions"] for lv in levers}
            attribution.update({a["name"]: a["decisions"] for a in ablations})
            for lv in parsed:
                if lv["kind"] in ("lever", "ablation"):
                    lv["decisions"] = list(attribution.get(lv["name"], []))
                else:
                    # a subset moves whatever its members move
                    lv["decisions"] = sorted({d for m in lv.get("levers", [])
                                              for d in attribution.get(m, [])})
            table = list(parsed)
            if stack:
                stack["kind"] = "stack"
                table.append(stack)
            for lv in table:
                # provenance on every row: a price is a reading on ONE tree, and
                # a later citation of it (a reopen, a milestone rung) is checked
                # against the tree it is about to be applied to
                lv["priced_on"] = best_ref
                lv["job"] = price_job
            _log("meta/lever-price: " + "; ".join(
                f"{lv['name']} add={_fmt_pct(lv['add_pct'])} remove={_fmt_pct(lv['remove_pct'])} "
                f"[{lv['cls']}]" for lv in parsed if lv["kind"] == "lever")
                + (f"; all={_fmt_pct(stack['all_pct'])} coupling={_fmt_pct(stack['coupling_pp'])}"
                   if stack else "")
                + (f"; all_pos={_fmt_pct(stack['all_pos_pct'])} (dropped {','.join(stack['dropped'])})"
                   if stack and stack.get("all_pos_pct") is not None else "")
                + meta_shapes.scope_log(stack))
            if variant_rows := [lv for lv in parsed if lv["kind"] == "variant"]:
                _log("meta/lever-price: subsets " + "; ".join(
                    f"{lv['name']}={{{','.join(lv['levers'])}}} {_fmt_pct(lv['delta_pct'])}"
                    for lv in variant_rows))
            if abl_rows := [lv for lv in parsed if lv["kind"] == "ablation"]:
                _log("meta/lever-price: ablations " + "; ".join(
                    f"{lv['name']} {_fmt_pct(lv['delta_pct'])} (ncu, correctness={lv['correctness']})"
                    for lv in abl_rows))
            _persist_lever_table(ctx, table, best_ref, price_job, spec_levers=levers)
            break
    finally:
        # The pricing commits are `add -A` snapshots of whatever the design turns
        # touched; the reset below throws all of it away so the student line stays
        # teacher-free. Anything outside notes/ in there was never part of the
        # pricing spec — a teacher's job-environment fix once rode one pricing
        # job this way and then vanished without a word.
        if lost := _edits_outside_notes(be, best_ref):
            _note_discarded_edits(ctx, "meta/lever-price", lost)
        try:
            be.reset(pre_head)
        except BackendError as e:
            _log(f"meta/lever-price: could not restore head {pre_head[:10]} ({e})")
    return table


def _lever_spec_refused(ctx: LoopContext, spec: dict) -> str | None:
    """The tool's PRICE-FAIL reason for ``spec`` on the workspace's kernel file
    (the tree is checked out at best when this runs), or None when the job
    would accept it. None too when the file cannot be read here."""
    from forgekernel.skills.lever_pricing_run import (build_ablation_variants, build_variants,
                                                       check_ablations, check_subsets,
                                                       lever_name, stack_names)
    try:
        src = (Path(ctx.workspace) / str(spec.get("kernel_file") or ctx.kernel_file)
               ).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    levers = spec.get("levers") or []
    try:
        subsets = (check_subsets(spec.get("variants") or [], stack_names(levers),
                                 [lever_name(lv) for lv in levers]) if levers else [])
        if levers:
            build_variants(src, levers, subsets)
        abl = check_ablations(spec.get("ablations") or [],
                              [lever_name(lv) for lv in levers] + [sb["name"] for sb in subsets])
        build_ablation_variants(src, abl)
    except (ValueError, KeyError, TypeError) as e:
        return f"lever-edit ({e})"
    return None


def _normalise_ablations(ctx: LoopContext, ablations: list, known_ids: set) -> list:
    """The teacher's ablations the way the tool and the table will see them:
    at most _LEVER_MAX_ABLATIONS, one-token names, `decisions` restricted to
    milestone ids (unknown ones dropped loudly, as for levers)."""
    out = []
    for a in list(ablations)[:_LEVER_MAX_ABLATIONS]:
        if not isinstance(a, dict):
            continue
        a = dict(a)
        a["name"] = re.sub(r"\s+", "_", str(a.get("name", "")).strip()) or "unnamed"
        ids = [str(d).strip() for d in (a.get("decisions") or [])]
        bad = [d for d in ids if d not in known_ids]
        if bad:
            _log(f"meta/lever-price: ablation {a['name']} names decision(s) not in "
                 f"the milestone {bad} — dropped from its attribution")
        a["decisions"] = [d for d in ids if d in known_ids]
        a["hypothesis"] = " ".join(str(a.get("hypothesis", "")).split())
        out.append(a)
    return out


def _lever_refusal_block(ctx: LoopContext, fail: str) -> str:
    return (f"\n## The pricing job rejected the lever table\n`{fail}`. Every `old` must occur "
            f"exactly once in {ctx.kernel_file}, and still exactly once each after all levers are "
            f"stacked in order; every subset name in `variants` must be unique, **must not equal any "
            f"lever's name**, and its members must be lever names from the table (with `in_stack` not "
            f"switched off); `ablations` holds at most {_LEVER_MAX_ABLATIONS} entries, whose names "
            f"must not equal a lever or subset name, whose `old` must likewise occur exactly once, "
            f"and whose edit must not be a no-op. "
            f"Fix levers.json, then end the turn.\n")


def _banked_lever_table(ctx: LoopContext, best_ref: str) -> list:
    """The last table this run priced on ``best_ref`` (evidence/lever_prices.jsonl),
    or [] when that tree was never priced."""
    ev = _evidence_dir(ctx, create=False)
    if ev is None or not (ev / "lever_prices.jsonl").is_file():
        return []
    found: list = []
    try:
        for line in (ev / "lever_prices.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("reprice"):
                continue        # a price-check's partial table, not a teacher table
            if rec.get("best_ref") == best_ref and rec.get("table"):
                found = [_stamp_price_row(dict(lv), str(rec.get("best_ref") or ""),
                                          str(rec.get("job") or "")) for lv in rec["table"]]
    except (OSError, ValueError) as e:
        _log(f"meta/lever-price: could not read lever_prices.jsonl ({e}) — pricing afresh")
        return []
    return found


def _persist_lever_table(ctx: LoopContext, table: list, best_ref: str, job: str,
                         spec_levers: list | None = None, reprice: bool = False) -> None:
    """Bank a priced table under the tree it was read on. ``spec_levers`` (the
    teacher's lever recipes: name + edits + in_stack) ride along so a later new
    best can re-price a cited lever without a design turn. A ``reprice`` table
    (cited levers only, no stack, no roles) is history for the price-check and
    never the teacher's table: _banked_lever_table skips it."""
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    rec = {"ts": ctx.now(), "best_ref": best_ref, "job": job, "table": table}
    if reprice:
        rec["reprice"] = True
    if spec_levers is not None:
        rec["spec_levers"] = [{k: lv.get(k) for k in ("name", "edits", "in_stack", "expect", "decisions")
                               if k in lv} for lv in spec_levers]
    try:
        with (ev / "lever_prices.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        _log(f"meta/lever-price: could not persist lever_prices.jsonl: {e}")
    else:
        _bank_evidence(ctx, "lever prices")
