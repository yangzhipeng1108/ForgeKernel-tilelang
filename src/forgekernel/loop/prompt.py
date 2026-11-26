"""Prompt rendering — student/teacher/meta prompt construction.

Split out of loop/driver.py (docs/driver-split-spec.md step 1). This section
builds every prompt the loop sends: the oracle/arch/precision bindings, the
family-analysis and window-contract notes, the prep-contract enforcement, and
the render entry points ``render_prompt`` / ``_student_prompt`` / ``_meta_prompt``.

The few driver helpers this module still needs are imported lazily (inside the
functions that call them) so there is no import cycle with driver.py:

    _log, _orch_summary, _read_ref_orch_dossier, _timeline_student_note

``LoopContext`` is used only in annotations (postponed via __future__), so it is
imported under TYPE_CHECKING for static tools. ``_placeholder_note`` and
``_price_revision_note`` live in the already-extracted placeholder/pricing
modules, which do not import prompt — importing them at module load is safe.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from forgekernel.inner.trajectory import gate_ms, load_records
from forgekernel.loop import lang_mode, precision_gate
from forgekernel.loop.placeholder import _placeholder_note
from forgekernel.loop.pricing import _price_revision_note

if TYPE_CHECKING:
    from forgekernel.loop.driver import LoopContext

__all__ = [
    "_oracle_binding",
    "_arch_ref_binding",
    "_verdict_probe_brief",
    "kb_read_dir",
    "_meta_prompt",
    "FAMILY_ANALYSIS_DIR",
    "family_analysis_path",
    "_write_family_analysis",
    "_is_cuda",
    "_is_tilelang",
    "_family_analysis_binding",
    "_bindings",
    "_ref_for_window",
    "_window_contract_binding",
    "_window_student_note",
    "_precision_reading",
    "_sass_precision",
    "_enforce_prep_contract",
    "_PREP_ERROR_PREFIX",
    "_prep_structural_stop",
    "_best_keep_record",
    "render_prompt",
    "_student_prompt",
]


def _log(msg: str) -> None:  # lazy shim -> driver._log
    from forgekernel.loop import driver

    return driver._log(msg)


def _orch_summary(*args, **kwargs):  # lazy shim -> driver._orch_summary
    from forgekernel.loop import driver

    return driver._orch_summary(*args, **kwargs)


def _read_ref_orch_dossier(*args, **kwargs):  # lazy shim -> driver._read_ref_orch_dossier
    from forgekernel.loop import driver

    return driver._read_ref_orch_dossier(*args, **kwargs)


def _timeline_student_note(*args, **kwargs):  # lazy shim -> driver._timeline_student_note
    from forgekernel.loop import driver

    return driver._timeline_student_note(*args, **kwargs)


# --- prompt rendering -------------------------------------------------------

def _oracle_binding(ctx: LoopContext) -> str:
    """What the teacher is told about the reference implementation's source.

    Under "blackbox" the source is withheld on purpose, and saying so matters: a
    teacher told only that no oracle is "provided" reads that as a missing input and
    either stalls or invents one. It is not missing — the reference is a binary that
    can be run and profiled but not read, which is the situation this mode exists to
    reproduce, and the reconnaissance report is the input that replaces it.
    """
    if ctx.oracle_mode == "blackbox":
        return ("(withheld by design — this run distills WITHOUT reading reference "
                "source. The reference is a closed binary: you may run it, time it and "
                "profile it, but there is no source to read and none will be supplied. "
                "The reconnaissance report is the input that replaces it.)")
    if ctx.oracle_path:
        return str(ctx.oracle_path)
    return "(oracle not provided)"


def _arch_ref_binding(ctx: LoopContext) -> str:
    """The architecture reference, or why there isn't one.

    Absence has two very different causes and the teacher has to tell them apart. On an
    operator whose reference *is* the library — a GEMM measured against cuBLAS — the
    two references coincide and the library baseline is exactly what the run
    reproduces; template A.4 asks that the coincidence be stated rather than left to
    look like an omission. On an operator that has a separate reference which simply
    could not be measured, anchoring to the library instead understates the target.
    """
    if ctx.arch_baseline:
        return ctx.arch_baseline.render(ctx.gate_window)
    if ctx.oracle_mode == "blackbox":
        return ("(none declared, and correctly so: this run's reference IS the library "
                "baseline above — the same implementation is both the thing being "
                "reconnoitred and the thing being reproduced. Anchor target_ms and the "
                "performance ladder to it, and state in §0 that the architecture "
                "reference and the library baseline coincide here.)")
    return ("(no architecture reference declared — target falls back to the library "
            "baseline. If this operator's reference genuinely IS the library, say so "
            "in §0; otherwise the target understates what the milestone teaches.)")


def _verdict_probe_brief(ctx: LoopContext, name: str) -> str:
    """The verdict-probe protocol section, or nothing when the budget is zero.

    A zero-budget run must use the instructions it would have used before the
    mechanism existed (the control-run principle _distill_prompt_path states):
    leaving the section in with a "0 rounds" budget both changes the control
    prompt and invites PROBE_REQUEST turns that can only spin on the
    budget-spent path. The round count is substituted here rather than left to
    render_prompt because the section arrives as another binding's value, after
    the token pass may already have moved on.
    """
    if ctx.max_verdict_probe_rounds <= 0:
        return ""
    text = _meta_prompt(ctx, name).read_text(encoding="utf-8")
    return text.replace("{MAX_VERDICT_PROBE_ROUNDS}", str(ctx.max_verdict_probe_rounds))


def kb_read_dir(ctx: LoopContext) -> Path:
    """The one KB tree this run reads and writes: ``hw_kb/<language>/<kb_arch>/``.

    hw_kb/ is one tree per architecture with no shared root. A hardware recipe
    earned on one architecture is not evidence about another -- the MMA trunk,
    the register file it was budgeted against and the asynchrony it pipelines
    against all change -- so the run gets its own tree and nothing else: the
    architecture-neutral entries it needs are transcribed into that tree (with a
    ``Certification source`` line saying where their readings came from), which is what lets
    the read set be a single directory.

    Recipe lookup (``KB:<entry>`` citations), the three leak-lint gates (loop
    start, certify, harvest) and the quarantine all go through this function, so
    what the run may read and what is held to the leak rules are the same set by
    construction. ``ctx.kb_dir`` stays the root: ``{KB_DIR}/TEMPLATE.md`` and the
    harvest are per-KB, not per-architecture.
    """
    return Path(ctx.kb_dir) / ctx.kb_arch


def _meta_prompt(ctx: LoopContext, name: str) -> Path:
    """The prompt file ``name`` for this run's language mode (loop/lang_mode.py): the
    CUDA mode reads ``meta/cuda/<name>`` where that overlay exists; every other run,
    and every language-neutral prompt, reads ``meta/<name>`` exactly as before."""
    return lang_mode.prompt_path(ctx.meta_dir, name, ctx.language)


FAMILY_ANALYSIS_DIR = "analysis"


def family_analysis_path(ctx: LoopContext) -> Path:
    """Where this operator's cross-run reconnaissance analysis lives.

    Operators come in families: the same op is run again on another arch, or again
    after a fix, and every run re-derives the reference's architecture from scratch
    — the same measurement, the same wrong turns, the same cost. The operator tree
    (``ctx.workspace``) is shared by every run of that op, so accumulating one
    analysis file per language there lets a run that is not the first start from
    what earlier runs already established.

    One file per language (``analysis/<language>.md``): a cutedsl run's readings
    are not a tilelang run's — the DSL changes what is expressible and what the
    SASS looks like — so the family is "same op, same language", not "same op".
    Resolved from the workspace, not a global operators root: the run already
    holds its own tree, and a path derived from it cannot be written into some
    other checkout (or a test's real repo) by accident.
    """
    return Path(ctx.workspace) / FAMILY_ANALYSIS_DIR / f"{ctx.language}.md"


def _write_family_analysis(ctx: LoopContext) -> None:
    """Append this run's recon report to the operator family's analysis file.

    Best-effort and idempotent: the section is keyed on the run's status path, so a
    resumed run does not append twice, and any failure (a read-only tree, a missing
    report) is logged and swallowed — the analysis is a convenience for the next
    run, never a gate on this one. The body is the report verbatim under a header
    that says which run wrote it; the reader is a later teacher deciding whether it
    is looking at the same reference at all, so provenance has to travel with it.
    """
    if ctx.recon_path is None or not Path(ctx.recon_path).is_file():
        return
    try:
        dest = family_analysis_path(ctx)
        dest.parent.mkdir(parents=True, exist_ok=True)
        marker = f"<!-- run: {ctx.status_path} -->"
        existing = dest.read_text(encoding="utf-8") if dest.is_file() else (
            f"# {ctx.op} — family reconnaissance analysis ({ctx.language})\n\n"
            f"Accumulated across runs of this operator at the same language. Each "
            f"section is one run's reconnaissance report, appended by the driver. A "
            f"later run reads this to avoid re-deriving what an earlier one measured; "
            f"read the nearest section whose reference matches yours.\n")
        if marker in existing:
            return
        try:
            report = Path(ctx.recon_path).read_text(encoding="utf-8").strip()
        except OSError:
            return
        block = (f"\n{marker}\n## Reconnaissance report — {Path(ctx.status_path).parent.name}\n\n"
                 f"{report}\n")
        dest.write_text(existing.rstrip() + "\n" + block, encoding="utf-8")
        _log(f"meta/recon: appended this run's report to the family analysis {dest}")
    except Exception as e:  # noqa: BLE001 — a convenience record never blocks the run
        _log(f"meta/recon: family analysis not written ({e.__class__.__name__}: {e})")


def _is_cuda(ctx: LoopContext) -> bool:
    """A CUDA-mode run: its student edits a .cu the card builds with nvcc."""
    return lang_mode.is_cuda(ctx.language) or lang_mode.is_cuda_kernel_file(ctx.kernel_file)


def _is_tilelang(ctx: LoopContext) -> bool:
    """A TileLang-mode run: its student edits a ``*_tl.py`` the card compiles.

    The language and the file are checked together because either can be the one
    that names the mode (an unset ``[loop] language`` follows the file). A run is
    never both: ``lang_mode.language_problem`` refuses the combination, so a
    ``_tl.py`` under the CUDA mode does not reach here, and the timeline /
    census / job-env code below can branch on this without ordering it against
    ``_is_cuda``."""
    return lang_mode.is_tilelang(ctx.language) or lang_mode.is_tilelang_kernel_file(ctx.kernel_file)


def _family_analysis_binding(ctx: LoopContext) -> str:
    """The operator family's accumulated analysis, for a teacher prompt.

    A first run gets the honest absence plus the path it would live at, so a teacher
    that wants to write (or a human reading the prompt) sees where it belongs rather
    than a dangling token."""
    try:
        path = family_analysis_path(ctx)
    except Exception:  # noqa: BLE001 — a missing operators root is not a run error
        return "(no operator family analysis available)"
    if path.is_file():
        return (f"{path} — earlier runs of this operator at {ctx.language} appended "
                f"their reconnaissance reports here; read the sections whose reference "
                f"matches yours before re-deriving its architecture")
    return (f"(none yet — this is the first {ctx.language} run of {ctx.op}; its report "
            f"will be appended to {path} for later runs)")


def _bindings(ctx: LoopContext) -> dict[str, str]:
    repo_root = ctx.meta_dir.parent
    return {
        "{MILESTONE_PATH}": str(ctx.milestone),
        "{MILESTONE}": str(ctx.milestone),
        "{KB_DIR}": str(ctx.kb_dir),
        "{KB_ARCH}": ctx.kb_arch,
        # The read set, rendered from the same function the lookup and the lint
        # gates use, so a prompt cannot name a tree the loop does not enforce.
        "{KB_READ_DIR}": str(kb_read_dir(ctx)),
        "{STATUS_JSONL}": str(ctx.status_path),
        "{TRAJECTORY_PATH}": str(ctx.status_path),
        "{WORKSPACE}": str(ctx.workspace),
        "{KERNEL_FILE}": ctx.kernel_file,
        "{LANGUAGE}": ctx.language,
        "{META_MILESTONE_PATH}": str(repo_root / "meta_milestone.md"),
        "{MILESTONES_TEMPLATE_PATH}": str(repo_root / "docs" / "milestones-template.md"),
        # Absolute on purpose: the teacher runs with cwd = the student workspace,
        # so a bare relative docs/ path would land the oracle-provenance archive
        # inside the tree the student reads and the kbprobe branch force-pushes.
        "{PROVENANCE_DIR}": str(lang_mode.provenance_dir(repo_root, ctx.language)),
        "{META_TRAJECTORY_PATH}": str(ctx.meta_status_path) if ctx.meta_status_path else "(none)",
        "{ORACLE}": _oracle_binding(ctx),
        "{RECON_REPORT}": (str(ctx.recon_path) if ctx.recon_path
                           else "(no reconnaissance report)"),
        "{CANDIDATES_PATH}": (str(ctx.recon_path.parent / "candidates.md")
                              if ctx.recon_path else "(no candidates file)"),
        "{RECON_DIR}": str(ctx.recon_dir) if ctx.recon_dir else "(no reconnaissance tree)",
        # The operator family's accumulated analysis (this op at this language, every
        # earlier run). Rendered from the path so a first run reads the honest absence.
        "{FAMILY_ANALYSIS}": _family_analysis_binding(ctx),
        "{SIMULATOR_DIR}": str(ctx.simulator_dir) if ctx.simulator_dir else "(no simulator)",
        "{MAX_RECON_ROUNDS}": str(ctx.max_recon_rounds),
        "{VERDICT_PROBE_BRIEF}": _verdict_probe_brief(ctx, "verdict_probe_brief.md"),
        "{VERDICT_PROBE_BRIEF_AUDIT}": _verdict_probe_brief(
            ctx, "verdict_probe_brief_audit.md"),
        "{CANDIDATE}": ctx.candidate or "(not pinned — take the first in the candidates table)",
        "{REF_BASELINE}": (ctx.ref_baseline.render(ctx.gate_window) if ctx.ref_baseline
                           else "(library baseline not measured)"),
        "{ARCH_REF}": _arch_ref_binding(ctx),
        "{GATE_WINDOW}": ctx.gate_window,
        "{WINDOW_CONTRACT}": _window_contract_binding(ctx),
        # the run's configured equivalence ratio and the target the driver derived
        # from it (anchor / ratio): §0's target_ms must be THIS number, not the
        # 1.0x reading (a self-evolution run asks for >1)
        "{EQUIVALENCE_RATIO}": f"{ctx.equivalence_ratio:g}",
        "{TARGET_MS}": (f"{ctx.target_ms:.4f}" if ctx.target_ms else "(pending measurement)"),
    }


def _ref_for_window(ctx: LoopContext):
    """The reference whose per-kernel table the ledger is read against: the arch
    ref when it banked one, else the library baseline, else the arch ref."""
    if ctx.arch_baseline and ctx.arch_baseline.kernels:
        return ctx.arch_baseline
    return ctx.ref_baseline if ctx.ref_baseline else ctx.arch_baseline


def _window_contract_binding(ctx: LoopContext) -> str:
    """The timed-window ledger for the teacher: declared boundary (both
    directions), the reference's per-kernel table and the sitting best KEEP's,
    plus the credit or why there is none. Rendered from disk state every time it
    is bound, so a regex added to window.toml mid-run prices the next entry."""
    from forgekernel.window_contract import baseline_label, render_window_ledger
    ref = _ref_for_window(ctx)
    student = _best_keep_record(load_records(ctx.status_path), ctx.gate_window) \
        if ctx.status_path and Path(ctx.status_path).exists() else None
    return render_window_ledger(
        ctx.window_contract,
        ref_kernels=ref.kernels if ref else None,
        ref_ms=ref.device_anchor_ms if ref else None,
        ref_label=baseline_label(ref),
        ref_profiler=ref.ncu if ref else None,
        student_kernels=student.kernels if student else None,
        student_ms=gate_ms(student, ctx.gate_window) if student else None,
        student_label=f"best KEEP iter {student.iter}" if student else "",
        student_profiler=student.profiler if student else None,
        student_prep=student.prep if student else None)


def _window_student_note(ctx: LoopContext) -> str:
    """Standing student note: which extra work its window carries relative to
    the reference's (inputs it must rebuild, outputs only it produces), how the
    pricing reads its launches, and which inputs reach its timed call from the
    bench's prep step outside the window (and that the step is fenced).
    Measurement convenience must not steer the design: fusing a rebuild into
    the main kernel can be the real lever, so the note asks the student to SAY
    when it does, not to avoid it."""
    contract = ctx.window_contract
    if contract is None or not contract.declared:
        return ""
    from forgekernel.window_contract import (KIND_REF_INPUT, PAYER_STUDENT,
                                             window_findings)
    if window_findings(contract):
        return ""
    items = contract.paid_by(PAYER_STUDENT)
    prep = contract.prep_entries
    if not items and not prep:
        return ""
    lines = ["\n## Timed-window contract (window.toml)"]
    if items:
        lines.append("Compared with the reference's timed call, your timed call pays for the extra "
                     "work below (the reference gets it outside its window / does not have to "
                     "produce it):")
        for a in items:
            what = ("an input the reference gets outside its window; you must rebuild it inside yours"
                    if a.kind == KIND_REF_INPUT
                    else "an output the reference does not produce; you must produce it inside your window")
            how = f" (derived from {a.derived_from})" if a.derived_from else ""
            if a.student_kernel_regex:
                tag = (f"; the finish-time pricing deducts this part by matching kernel names "
                       f"against `{a.student_kernel_regex}`")
            else:
                tag = "; no pricing regex configured yet"
            lines.append(f"- {a.name}: {what}{how}{tag}")
        lines.append("This work is contract tax: it does not go away as the kernel gets better, and "
                     "it is not a reason to KEEP or REVERT. Whether it is a separate launch or fused "
                     "into the main kernel is decided by performance, not by the pricing — if you fuse "
                     "it (or rename the kernel), say so in the decision note: the pricing then stops "
                     "applying and the finish reports only the raw ratio.")
    if prep:
        lines.append("The inputs below are produced by the bench's prep step **outside** the timed "
                     "window and reach your timed call directly (symmetric with the reference's "
                     "autograd path, which also gets them for free from the forward):")
        for a in prep:
            spec = f"{tuple(a.shape) if a.shape else '?'} {a.dtype or '?'}"
            how = f", derived from {a.derived_from}" if a.derived_from else ""
            lines.append(f"- {a.name}: {spec}{how}{f'; {a.note}' if a.note else ''}")
        lines.append("The prep step may produce only the tensors above, matching the declaration in "
                     "name / shape / dtype item by item — producing more or fewer, or a mismatched "
                     "shape or dtype, REVERTs the round as a correctness failure; the prep time is "
                     "recorded for the teacher every round. Moving in-window work into prep does not "
                     "improve the ratio; it only exposes the work in the ledger.")
    lines.append("")
    return "\n".join(lines)


def _precision_reading(ctx: LoopContext, ev, iter_no: int) -> dict:
    """The precision gate (loop/precision_gate.py): summarize the eval JSON's
    ``precision`` block for the record, and under ``[loop] precision_contract = strict``
    turn a failed check into a correctness failure (``ev`` is changed in place).
    ``official`` records and logs only -- the bench's own verdict stays what the board
    would say."""
    contract = ctx.precision_contract
    summary = precision_gate.summarize(contract, ev.precision)
    if not summary:
        return summary
    if summary.get("status") == "unreported":
        if contract == precision_gate.STRICT and ctx.precision_bench_reports:
            if ev.correctness:
                ev.correctness = False
                ev.error = precision_gate.MISSING_BLOCK
            return summary
        if not ctx.precision_unreported_warned:
            ctx.precision_unreported_warned = True
            _log(f"iter {iter_no}: WARNING precision ({contract}): the eval JSON carries no "
                 f"`precision` block — the bench does not report one (copy "
                 f"eval/precision.py from src/forgekernel/card_template and call PrecisionRecorder; a "
                 f"leaderboard card needs the current eval/leaderboard.py), so the "
                 f"precision_contract cannot be applied; rows record `unreported`")
        return summary
    if (why := precision_gate.gate_failure(contract, summary)) and ev.correctness:
        ev.correctness = False
        ev.error = why
    elif summary.get("status") == "fail":
        how = ("the eval already failed correctness" if why else "recorded only")
        _log(f"iter {iter_no}: precision ({contract}, {how}): "
             f"{precision_gate.describe(summary)}")
    return summary


def _sass_precision(ctx: LoopContext, student: "dict | None", iter_no: int) -> dict:
    """The SASS cross-check of the precision gate (warning only): the census's MMA
    operand / accumulator types against the reference dossier's and the card's input
    dtypes (loop/precision_gate.sass_findings). Logged when it finds a drop; {} when the
    census carried no MMA reading."""
    try:
        if not student or ctx.precision_contract == "off":
            return {}
        cache = ctx.__dict__.setdefault("_sass_precision_ref", {})
        if "ref" not in cache:
            cache["ref"] = ((_orch_summary(_read_ref_orch_dossier(ctx)) or {})
                            .get("mma_precision"))
            dtypes = []
            try:
                prob = json.loads((Path(ctx.workspace) / precision_gate.LEADERBOARD_PROBLEM)
                                  .read_text(encoding="utf-8"))
                dtypes = [str(v.get("dtype")) for v in
                          (prob.get("definition") or {}).get("inputs", {}).values()]
            except (OSError, ValueError, AttributeError):
                pass
            cache["dtypes"] = dtypes
        out = precision_gate.sass_findings(student, cache["ref"], cache["dtypes"])
    except Exception as e:  # noqa: BLE001 — a cross-check, never a reason to stop
        _log(f"iter {iter_no}: sass precision check skipped ({type(e).__name__}: {e})")
        return {}
    for w in out.get("warnings") or ():
        _log(f"iter {iter_no}: WARNING SASS precision: {w} (census {out['student']['ops']}; "
             f"recorded only — the precision gate reads the error ratio)")
    return out


def _enforce_prep_contract(ctx: LoopContext, ev):
    """Hold the bench's prep report to window.toml; a mismatch is a REVERT.

    The prep step is the one place work legitimately leaves the timed window,
    and the declaration (symmetrize entries with shape / dtype) is what bounds
    it. The bench ships to the job without window.toml, so the comparison is
    made here, on the eval JSON, where the student cannot edit it away: an
    undeclared extra tensor, a missing one, a wrong shape or dtype turns the
    eval into a correctness FAIL with the mismatch as its error text.
    """
    contract = ctx.window_contract
    if not ev.correctness or contract is None or not contract.declared:
        return ev
    from forgekernel.window_contract import check_prep_report, window_findings
    if not contract.prep_entries or window_findings(contract):
        return ev
    problems = check_prep_report(contract, ev.prep)
    if not problems:
        return ev
    return replace(ev, correctness=False, error="prep contract: " + "; ".join(problems))


_PREP_ERROR_PREFIX = "correctness FAIL: prep contract: "


def _prep_structural_stop(records) -> str | None:
    """The inner loop's own stop reason for a prep-contract failure that is the
    bench's, not the kernel's: ``prep_unreported`` the first time the bench
    reports no prep block at all; ``prep_contract`` when the same mismatch
    (shape / dtype / undeclared tensor / mutation) repeats verbatim on two
    consecutive iterations — a student edit that did not change the verdict
    text did not touch the cause."""
    if not records:
        return None
    last = records[-1]
    if last.verdict != "REVERT" or not last.error.startswith(_PREP_ERROR_PREFIX):
        return None
    from forgekernel.window_contract import PREP_UNREPORTED
    if PREP_UNREPORTED in last.error:
        return "prep_unreported"
    if len(records) >= 2 and records[-2].verdict == "REVERT" and records[-2].error == last.error:
        return "prep_contract"
    return None


def _best_keep_record(records, gate_window: str):
    """The sitting best KEEP record (same identity rule as ``_best_keep_profiler``)."""
    kept = [r for r in records
            if r.verdict == "KEEP" and gate_ms(r, gate_window) is not None]
    return min(kept, key=lambda r: gate_ms(r, gate_window)) if kept else None


def render_prompt(template_path: Path, bindings: dict[str, str], extra: str = "") -> str:
    text = template_path.read_text(encoding="utf-8")
    for token, value in bindings.items():
        text = text.replace(token, value)
    return text + extra


def _student_prompt(ctx: LoopContext, iter_no: int, probe_feedback: str = "") -> str:
    extra = f"\n\n## Driver context\n- This is iteration {iter_no}.\n"
    if ctx.fresh_recipe:
        extra += (f"- A new KB recipe was just certified into the KB: `{ctx.fresh_recipe}`, for the "
                  f"hardware interface your last segment got stuck on. **Read and apply it first** "
                  f"(it is GPU-probe certified and is the direct way past your current blocker).\n")
    if ctx.reopen_note:
        extra += ctx.reopen_note
    if ctx.g5_note:
        extra += ctx.g5_note
    if ctx.phase_close:
        # Standing until the gate row lands: the phase's remaining levers were
        # priced null on the best tree, so walking them one iteration at a time
        # would only re-measure the table (a dsa_sparse run spent five
        # iterations and two meta steps doing exactly that).
        extra += ctx.phase_close["note"]
    if note := _price_revision_note(ctx):
        # Standing until the teacher's next entry prices a fresh table on the
        # current best; read from the evidence file, so a driver restart keeps it.
        extra += note
    if note := _timeline_student_note(ctx):
        extra += note
    if ctx.sass_serial_note:
        # Re-prompted verbatim, un-anchored: the compiler's whole-function verdict
        # persists across the change (it re-fires on every recompile), so it must
        # not be mistaken for yesterday's perf reading. Cleared only when a KEEP
        # prices pipelined again.
        extra += ctx.sass_serial_note
    if note := _placeholder_note(ctx):
        # Standing until every [PLACEHOLDER:] has its [RESTORED:] — the gate row is
        # refused before then, so the student sees the debt on every turn.
        extra += note
    if ctx.resume_note:
        # Standing for the whole post-patch segment (unlike host_bound_note): every
        # turn of the segment operates under the same rolled-back premise.
        extra += ctx.resume_note
    if ctx.host_bound_note:
        # One-shot: it describes the LAST evaluated round, so it must not survive
        # into a turn whose measurement it no longer describes.
        extra += ctx.host_bound_note
        ctx.host_bound_note = ""
    if note := _window_student_note(ctx):
        # Standing: the contract does not change between turns, and the rebuild
        # launches must stay recognisable for the whole run to be priced.
        extra += note
    if probe_feedback:
        extra += probe_feedback
    return render_prompt(_meta_prompt(ctx, "student_prompt.md"), _bindings(ctx), extra)


