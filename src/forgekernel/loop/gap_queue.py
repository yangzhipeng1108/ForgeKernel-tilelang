"""Out-of-run gap queue — recon candidates that no in-run check will verify.

Split out of loop/driver.py (docs/driver-split-spec.md step 8, first slice). This
is the self-contained gap-queue core: the row model (GapRow), parsing/loading/
persisting the queue, the recon-completion state, and the agent turn / recon prompt
that drives the recon step. The heavier recon/pricing/distill/two_layer engine that
shared the old "out-of-run gap queue" section stays in driver — it is the
orchestration core.

Imports come from already-extracted modules (shared, prompt, comparisons) plus
stdlib/lang_mode — none import gap_queue, so no top-level cycle. AgentError,
ReconResult and _recon_state_path are defined in driver before this module is
imported back -> module-load import. LoopContext is annotation-only -> TYPE_CHECKING.
_bank_evidence / _evidence_dir / _teacher stay in driver (defined later) -> lazy
shims (each verified not to call back into this module).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from forgekernel.loop import lang_mode
from forgekernel.loop.comparisons import _log
from forgekernel.loop.prompt import _bindings, _meta_prompt, render_prompt
from forgekernel.loop.shared import TRUNCATION_STORM_BUDGET
from forgekernel.loop.driver import AgentError, ReconResult, _recon_state_path  # defined early

if TYPE_CHECKING:
    from forgekernel.loop.driver import LoopContext

__all__ = [
    "_GAP_QUEUE_NAME",
    "_GAP_ARCHIVE_SUBDIR",
    "_TABLE_ROW_RE",
    "_EXPR_TAG_RE",
    "_EXPR_GUESS_RE",
    "_VERIFY_TAG_RE",
    "GapRow",
    "_row_verify",
    "parse_gap_queue",
    "_gap_queue_text",
    "load_gap_queue",
    "_gap_archive_path",
    "_persist_gap_queue",
    "_load_recon_state",
    "TRUNCATION_MAX_CHARS",
    "_read_or_none",
    "_agent_turn",
    "_recon_prompt",
    "TRUNCATION_BACKOFF_S",
]


def _bank_evidence(*args, **kwargs):  # lazy shim -> driver._bank_evidence
    from forgekernel.loop import driver

    return driver._bank_evidence(*args, **kwargs)


def _evidence_dir(*args, **kwargs):  # lazy shim -> driver._evidence_dir
    from forgekernel.loop import driver

    return driver._evidence_dir(*args, **kwargs)


def _teacher(*args, **kwargs):  # lazy shim -> driver._teacher
    from forgekernel.loop import driver

    return driver._teacher(*args, **kwargs)


# --- out-of-run gap queue ------------------------------------------------------
#
# The recon candidates table tags every decision with a VERIFY channel; rows
# tagged `none` — and rows whose remark guesses at expressibility ("may not be
# expressible in the DSL") without an [EXPR:] ladder verdict — are claims nobody
# in the run will
# ever check. On one sparse-attention run such a row (the cp.async cache bits)
# was worth 22 % and sat unread for the whole run. The queue makes them
# visible: banked beside the evidence at recon completion, surfaced in the
# loop summary, and archived for human review under docs/kb-provenance/gaps/.
#
# It never reaches hw_kb. The rows are the reference's structure as the recon
# read it (a DEPBAR depth at a buffer release, a barrier handshake, cache
# bits) — narrated provenance, which the KB discipline keeps out of every
# agent read set precisely because hw_kb is read in full by every later
# student: a "[pending certification]" entry would hand the next same-operator run an
# unverified reference-level structural prior through a door kb_lint cannot
# see (no reference token, no line number — just structure). The only carrier
# into hw_kb is a probe-certified recipe.
_GAP_QUEUE_NAME = "gap_queue.md"
_GAP_ARCHIVE_SUBDIR = "gaps"           # docs/kb-provenance/gaps/<op>.md
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
_EXPR_TAG_RE = re.compile(r"\[EXPR:")
_EXPR_GUESS_RE = re.compile(
    r"(?:not|n't)\s+(?:be\s+)?expressible"
    r"|(?:cannot|can't|may\s+not|might\s+not)\s+be\s+expressed"
    r"|inexpressible|unexpressible", re.I)
# Both spellings the recon writes: a bare VERIFY column (`none`) and the
# inline tag the prompt itself prescribes (`[VERIFY: none]`, parked in a
# `tag` / `source` column). Real tables are mostly the latter: of five runs'
# candidates.md only one carried the column; the other four (sparse,
# fp8, bwd-GQA, fwd-MHA) would have parsed to zero rows.
_VERIFY_TAG_RE = re.compile(r"\[VERIFY:\s*([A-Za-z-]+)\s*\]", re.I)


@dataclass
class GapRow:
    decision: str
    value: str
    verify: str
    remark: str
    reason: str     # "verify-none" | "expressibility-guess" | "inferred"

    _WHY = {"expressibility-guess": "unverified expressibility guess, no [EXPR:] verdict",
            "inferred": ("inferred: the discriminating observation gave no positive reading "
                         "(readings a byte/count identity cannot separate)"),
            "verify-none": "no machine-read verification in this run"}

    def line(self) -> str:
        return (f"- **{self.decision}** = {self.value} — VERIFY `{self.verify}`"
                f"{'; notes: ' + self.remark if self.remark else ''}"
                f" ({self._WHY.get(self.reason, self.reason)})")


def _row_verify(header: list[str], cols: dict, text: str) -> str:
    """The row's VERIFY channel: the `[VERIFY: x]` tag wherever it sits in the
    row, else the bare value of a VERIFY column (`none`, `` `none` ``,
    `none（...）`), else ""."""
    tm = _VERIFY_TAG_RE.search(text)
    if tm:
        return tm.group(1).lower()
    vcol = next((h for h in header if h.upper().startswith("VERIFY")), None)
    if vcol is None:
        return ""
    bare = cols[vcol].strip("`").strip()
    return re.split(r"[\s（(`]", bare, maxsplit=1)[0].lower() if bare else ""


def parse_gap_queue(candidates_text: str) -> list[GapRow]:
    """Rows of every candidates table that are either VERIFY `none` (a bare
    column or the inline `[VERIFY: none]` tag) or guess at expressibility
    without an [EXPR:] ladder verdict. Duplicates (the same decision in
    several candidate sections) collapse on (decision, value)."""
    out: list[GapRow] = []
    seen: set = set()
    header: list[str] | None = None
    for raw in candidates_text.splitlines():
        m = _TABLE_ROW_RE.match(raw.strip())
        if m is None:
            header = None
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if header is None:
            header = cells
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue                                   # the |---|---| rule
        cols = {h: (cells[i] if i < len(cells) else "") for i, h in enumerate(header)}
        text = " ".join(cells)
        verify = _row_verify(header, cols, text)
        decision = cols.get(header[0], "").strip("*").strip()
        value = cols.get(header[1], "") if len(header) > 1 else ""
        remark = next((cols[h] for h in header if h.strip().lower() in
                       ("note", "notes", "remark", "remarks", "comment", "comments")), "")
        if verify == "none":
            reason = "verify-none"
        elif verify == "inferred":
            # Observed in a dsa_sparse recon: the KV-ring stage axis had two readings
            # with identical bytes, the discriminator it wrote printed nothing,
            # and the row shipped as [VERIFY: sass-derived] "no ambiguity". Under the
            # rule an empty discriminator leaves the row `inferred`; it comes
            # here so distill sees BOTH readings as candidates, never a verdict.
            reason = "inferred"
        elif _EXPR_GUESS_RE.search(text) and not _EXPR_TAG_RE.search(text):
            reason = "expressibility-guess"
        else:
            continue
        key = (decision, value)
        if key in seen:
            continue
        seen.add(key)
        out.append(GapRow(decision, value, verify, remark, reason))
    return out


def _gap_queue_text(rows: list[GapRow], op: str = "") -> str:
    body = "\n".join(r.line() for r in rows)
    return (f"# Out-of-run gap queue (recon-table rows this run did not verify"
            f"{': ' + op if op else ''})\n\n"
            "Rows with VERIFY `none` have no machine-read acceptance; rows that say \"may not "
            "be expressible\" without an `[EXPR:]` ladder verdict are unverified guesses; rows "
            "with VERIFY `inferred` are structural readings whose discriminating observation "
            "gave no positive reading (the two readings are byte-identical). None of the three "
            "may be used as a conclusion. These rows are the recon's reading of the reference's "
            "structure (narrated provenance) and **do not enter hw_kb**: only a probe-certified "
            "recipe carries knowledge across runs; to verify one, start a separate probe, with "
            "the evidence under `docs/kb-provenance/`.\n\n"
            f"{body}\n")


def load_gap_queue(ctx: LoopContext) -> list[GapRow]:
    """The queue as parsed from the candidates table now (not the banked
    file): the table is the SSOT and a re-affirmed recon may have closed rows."""
    if ctx.recon_path is None:
        return []
    candidates = ctx.recon_path.parent / "candidates.md"
    try:
        return parse_gap_queue(candidates.read_text(encoding="utf-8"))
    except OSError:
        return []


def _gap_archive_path(ctx: LoopContext) -> Path | None:
    """docs/kb-provenance/gaps/<op>.md — the human-review copy, outside every
    agent read set (the provenance archive's discipline), reviewed with the KB
    changes."""
    op = getattr(ctx, "op", "") or ""
    if not op or ctx.kb_dir is None:
        return None
    # Anchored on the KB's repo root (hw_kb/ and docs/kb-provenance/ are
    # siblings — a reviewer commits them together).
    return lang_mode.kb_repo_root(ctx.kb_dir) / "docs" / "kb-provenance" / _GAP_ARCHIVE_SUBDIR / f"{op}.md"


def _persist_gap_queue(ctx: LoopContext) -> list[GapRow]:
    rows = load_gap_queue(ctx)
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return rows
    targets = [ev / _GAP_QUEUE_NAME]
    archive = _gap_archive_path(ctx)
    if archive is not None and archive.parent.parent.is_dir():   # docs/kb-provenance exists
        targets.append(archive)
    touched = False
    for dst in targets:
        try:
            if rows:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(_gap_queue_text(rows, getattr(ctx, "op", "") or ""),
                               encoding="utf-8")
                touched = True
            elif dst.exists():
                dst.unlink()
                touched = True
        except OSError as e:
            _log(f"meta/recon: could not persist {dst}: {e}")
    if touched:
        _bank_evidence(ctx, "gap queue")
    if rows:
        _log(f"meta/recon: gap queue — {len(rows)} unverified candidates row(s) "
             f"({sum(r.reason == 'expressibility-guess' for r in rows)} expressibility "
             f"guess(es) without an [EXPR:] verdict) -> "
             + ", ".join(str(t) for t in targets) + " (never hw_kb)")
    return rows


def _load_recon_state(ctx: LoopContext) -> ReconResult | None:
    """A prior session's completion marker, if it still describes what is on disk.

    Trust is conditional: the marker must name this run's report path, and both
    artifacts must still exist, be non-empty, and be no older than when the marker
    was written (a later mtime is a legitimate re-affirmation or edit; an earlier
    one means the file was swapped out from under the marker). Anything off —
    missing files, unparsable JSON, a foreign path — and the marker is ignored,
    which simply re-runs reconnaissance: the pre-fix behavior.
    """
    state_path = _recon_state_path(ctx)
    if state_path is None or not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        report = Path(state["report"])
        mtimes = {report: float(state["report_mtime"]),
                  report.parent / "candidates.md": float(state["candidates_mtime"])}
        rounds = int(state["rounds"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if report != ctx.recon_path:
        return None
    for path, recorded in mtimes.items():
        try:
            st = path.stat()
        except OSError:
            return None
        if st.st_size == 0 or st.st_mtime + 1.0 < recorded:  # +1s: fs granularity
            return None
    return ReconResult(rounds=rounds, report=ctx.recon_path, finished=True)


# A truncated stream from the gateway ends in a well-formed-looking tiny final
# message and rc=0 — indistinguishable, at the exit-code level, from a deliberate
# turn. Charged as ordinary rounds, one bad gateway hour spent 8 of a recon's 12
# GPU rounds on nothing. The tell is the combination: an output
# too short to be real work AND no legal trace of the turn on disk. Deliberate
# no-op turns write artifacts or emit requests; deliberate rambling is long.
TRUNCATION_MAX_CHARS = 400
TRUNCATION_BACKOFF_S: tuple[float, ...] = (60.0, 120.0, 300.0)


def _read_or_none(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _agent_turn(be, prompt: str, *, left_a_trace, label: str,
                storm_budget: list[int]) -> str:
    """run_agent with bounded FREE retries for truncated turns.

    ``left_a_trace(out)`` answers "did this turn end legally?" (a parseable
    request in the output, or the artifact it was asked to write changed on
    disk). A turn that is both traceless and shorter than TRUNCATION_MAX_CHARS
    is retried after a backoff — truncation correlates with gateway congestion,
    so immediate re-fire mostly re-truncates — WITHOUT charging the caller's
    round budget: budget rounds are measurement opportunities, and spending
    them on turns that never happened destroyed 8/12 of a reconnaissance.

    ``storm_budget`` is a mutable one-element list shared across a phase. When
    it runs dry the channel is systematically sick; AgentError converts the run
    to stop=agent_failed — the transient class the watchdog already restarts —
    with a diagnosis in the log instead of a silently spent budget.
    """
    be = _teacher(be)  # every caller is a teacher phase (recon / distill / lever design)
    out = be.run_agent(prompt)
    from forgekernel.loop import driver as _drv
    for pause in _drv.TRUNCATION_BACKOFF_S:
        if len(out.strip()) >= TRUNCATION_MAX_CHARS or left_a_trace(out):
            return out
        if storm_budget[0] <= 0:
            raise AgentError(
                f"{label}: truncation storm — {TRUNCATION_STORM_BUDGET} free "
                "retries consumed this phase and turns still come back tiny and "
                "traceless; the channel is failing systematically (see "
                "TRUNCATION_* in loop/driver.py)")
        storm_budget[0] -= 1
        _log(f"{label}: turn looks truncated ({len(out.strip())} chars, no legal "
             f"trace) -> free retry in {pause:.0f}s, not charged to the round "
             f"budget ({storm_budget[0]} storm-budget left)")
        time.sleep(pause)
        out = be.run_agent(prompt)
    return out


def _recon_prompt(ctx: LoopContext, extra: str = "") -> str:
    """The reconnaissance brief, with the source-mode addendum when the teacher
    also holds the oracle.

    recon_prompt.md is written for the blackbox teacher ("you will not be given
    the source"). A source-mode run that reconnoitres gets the same brief — the
    evidence gates, the tool contracts and the write-up are identical — plus a
    block saying the source IS available and what reconnaissance is for when it
    is: measuring what the source states only as a template (the values the
    binary actually took, the SASS-level schedule, the sync topology) rather
    than re-deriving what the source states plainly. Appended, not branched
    into the file, so the blackbox rendering stays byte-identical.
    """
    addendum = ""
    if ctx.oracle_mode == "source" and ctx.oracle_path:
        addendum = (
            "\n\n## Source-mode addendum: you also hold the reference's source\n\n"
            f"This is a **source run**; the passage above about not being given the source "
            f"does not apply: `ORACLE` = `{ctx.oracle_path}`"
            " (comma-separated, all readable). Reconnaissance therefore has a different purpose "
            "— not to replace the source but to **measure the facts the source gives only as "
            "templates / parameters and that are fixed only in the binary**: the tile / stage / "
            "warp-role values this shape actually takes, the real smem and register usage, the "
            "SASS-level issue order and commit/wait choreography of the K loop, the real "
            "topology and arrival counts of the mbarriers / named barriers, whether the supply "
            "is serial or prefetched, the DEPBAR depth, the rasterization group width.\n\n"
            "- **Read the source first, then decide the probes**: the source tells you what to "
            "ask, the probe tells you how the binary answers; when the two disagree, **the "
            "binary's reading is the fact and the source only explains the intent** — write "
            "both in the report and mark the disagreement.\n"
            "- The evidence gates, the tool contracts and the two deliverables (report + "
            "candidates table) are exactly the same as for a blackbox run — the candidates table "
            "may hold a single candidate (the machine in the source), but every value must "
            "still carry a job number or be marked \"not measured\".\n"
            "- Every fact in the report must still trace to a job number; \"the source says "
            "so\" is not a measurement source.\n"
        )
    return render_prompt(_meta_prompt(ctx, "recon_prompt.md"), _bindings(ctx),
                         extra=addendum + extra)
