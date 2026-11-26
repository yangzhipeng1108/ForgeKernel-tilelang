"""Timeline probe — the per-segment cycle account.

Split out of loop/driver.py (docs/driver-split-spec.md step 7, first slice). This
is the self-contained timeline probe: running recon/timeline_probe.py, parsing and
banking the table, deciding when it is due/stale, rendering the timeline tables and
block, and the student-facing notes (cuda / tilelang). The meta-loop engine
(run_meta_step, resume/reopen/gate/recon logic) that shared the old "timeline probe"
section stays in driver for now — it is the orchestration core, not a leaf.

Only three driver helpers remain: _bank_evidence, _evidence_dir, _read_milestone
(lazy shims, each verified not to call back into this module). Everything else comes
from already-extracted modules (prompt, comparisons, pricing) plus stdlib/job_env/
meta_shapes. BackendError is defined early in driver (before this module is imported
back) -> module-load import. LoopContext is annotation-only -> TYPE_CHECKING.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from forgekernel.inner.trajectory import GATE_WINDOW_DEVICE, gate_window_name
from forgekernel.loop import meta_shapes
from forgekernel.loop.comparisons import _log, _note_env_failure, _note_env_ok
from forgekernel.loop.job_env import env_failure
from forgekernel.loop.pricing import _ms_of_tree, _node_dirty_reason
from forgekernel.loop.prompt import _is_cuda, _is_tilelang
from forgekernel.loop.driver import BackendError  # defined early in driver; safe at load

if TYPE_CHECKING:
    from forgekernel.loop.driver import LoopContext

__all__ = [
    "_TIMELINE_CMD",
    "_timeline_cmd",
    "_TIMELINE_OVERLAY_MSG",
    "_TIMELINES_NAME",
    "_TIMELINE_REQUESTS",
    "_LOCUS_SEG_LINE",
    "_BOTTLENECK_LATENCY",
    "_is_timeline_request",
    "_timeline_on",
    "_timeline_markers",
    "_timeline_unparsed_markers",
    "_parse_locus_segment",
    "_timeline_short_coverage",
    "_timeline_probe_stale",
    "_timeline_preamble_fence_hit",
    "_timeline_probe_ref",
    "_run_timeline",
    "_fmt_cyc",
    "_timeline_renderable",
    "_bank_timeline",
    "_banked_timelines",
    "_banked_timeline_usable",
    "_latest_timeline",
    "_timeline_probe_digest",
    "_banked_timeline_of",
    "_timeline_due",
    "_ref_region_lines",
    "_timeline_tables",
    "_timeline_block",
    "_timeline_student_note",
    "_timeline_student_note_cuda",
    "_timeline_student_note_tilelang",
    "_timeline_feedback",
]


def _bank_evidence(*args, **kwargs):  # lazy shim -> driver._bank_evidence
    from forgekernel.loop import driver

    return driver._bank_evidence(*args, **kwargs)


def _evidence_dir(*args, **kwargs):  # lazy shim -> driver._evidence_dir
    from forgekernel.loop import driver

    return driver._evidence_dir(*args, **kwargs)


def _read_milestone(*args, **kwargs):  # lazy shim -> driver._read_milestone
    from forgekernel.loop import driver

    return driver._read_milestone(*args, **kwargs)


_TIMELINE_CMD = "cmd:python3 notes/timeline_probe.py"


def _timeline_cmd(ctx: LoopContext) -> str:
    """The probe job's command: a CUDA-mode run names its kernel.cu (the probe
    stamps `FK_SEG` markers in C++ and runs the bench under ``--cuda-run``); a
    TileLang run names its ``*_tl.py`` (so the probe reads that file, not the
    ``kernel.py`` its default assumes); a CuTe DSL run uses exactly
    ``_TIMELINE_CMD``."""
    if _is_cuda(ctx) or _is_tilelang(ctx):
        return f"{_TIMELINE_CMD} --kernel {ctx.kernel_file}"
    return _TIMELINE_CMD


_TIMELINE_OVERLAY_MSG = ("timeline probe: harness copy of notes/timeline_probe.py overlaid for one "
                         "probe job (the tree under judgment is the parent, unchanged)")
_TIMELINES_NAME = "timelines.jsonl"
_TIMELINE_REQUESTS = ("timeline", "notes/timeline", "notes/timeline_probe.py")
_LOCUS_SEG_LINE = re.compile(r"locus_segment\s*[:：]\s*([^\n]*)")
_BOTTLENECK_LATENCY = re.compile(r"^\s*bottleneck\s*[:：]\s*latency\b", re.M | re.I)


def _is_timeline_request(path: str) -> bool:
    """`PROBE_REQUEST: timeline` however the agent quoted it (backticks, quotes,
    a trailing full stop) — the parser's `\\S+` keeps the punctuation."""
    return str(path or "").strip().strip("`'\"“”「」").rstrip(".,;:：。").lower() in _TIMELINE_REQUESTS


def _timeline_on(ctx: LoopContext) -> bool:
    mode = str(getattr(ctx, "timeline_probe", "off") or "off").lower()
    if mode == "on":
        return True
    if mode == "auto":
        return _BOTTLENECK_LATENCY.search(_read_milestone(ctx)) is not None
    return False


def _timeline_markers(ctx: LoopContext) -> "tuple[bool, list]":
    """(the kernel keeps the no-op `_seg` definition, its marker site names) — the
    `#define FK_SEG` line and `FK_SEG("...")` calls of a CUDA-mode kernel.

    A TileLang kernel is Python, so it keeps the Python spelling (`def _seg` +
    `_seg("...", i)` calls) and reads through the same parser as the CuTe DSL one;
    only the CUDA mode's C++ spelling needs ``cuda=True``."""
    from forgekernel.skills.timeline_probe import has_marker_def, markers
    try:
        src = (Path(ctx.workspace) / str(ctx.kernel_file)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False, []
    if _is_cuda(ctx) and not _is_tilelang(ctx):
        return has_marker_def(src, cuda=True), markers(src, cuda=True)
    return has_marker_def(src), markers(src)


def _timeline_unparsed_markers(ctx: LoopContext) -> list:
    """The kernel's `_seg(` calls whose site name is not a string literal — the
    probe's site table cannot see them and the stamped variant reads them as no-ops
    (timeline_probe.unparsed_markers); the student note names them."""
    from forgekernel.skills.timeline_probe import unparsed_markers
    try:
        src = (Path(ctx.workspace) / str(ctx.kernel_file)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    if _is_cuda(ctx) and not _is_tilelang(ctx):
        return unparsed_markers(src, cuda=True)
    return unparsed_markers(src)


def _parse_locus_segment(text: str) -> str:
    m = _LOCUS_SEG_LINE.search(text or "")
    return m.group(1).strip() if m else ""


def _timeline_short_coverage(tl: dict) -> bool:
    """A renderable table whose stamps do not span the whole device call.

    Two spellings, one condition: the probe's own ``partial`` class, and a table
    banked before the TIMELINE-COVERAGE line existed that ``table_usable`` read
    through on its span (``ok`` false, coverage unknown). Both leave part of the
    device window outside the table, so both get the caveat where the numbers
    are shown (see ``timeline_probe.coverage_class`` for what a short span does
    and does not say: segment cycles are exact, the rows do not add up to the
    call).
    """
    if tl.get("coverage") == "partial":
        return True
    return tl.get("coverage") in (None, "unknown") and not tl.get("ok") and bool(tl.get("roles"))


def _timeline_probe_stale(ctx: LoopContext) -> "str | None":
    """The harness copy of the probe when the tree's ``notes/timeline_probe.py``
    differs from it (or is missing); None when the two are identical.

    The job runs the TREE's copy (``_TIMELINE_CMD``), and nothing in the harness
    refreshes it: an operator tree keeps whatever probe it was scaffolded with,
    so a probe change lands in a job only once someone copies the file over
    (a GDN bwd tree's copy once differed from the harness's). The driver
    reads such a log through its legacy rule (``table_usable``); this names the
    gap so the operator and the student can close it."""
    from forgekernel.skills import timeline_probe as probe_mod
    src = Path(probe_mod.__file__)
    tree = Path(ctx.workspace) / "notes" / "timeline_probe.py"
    try:
        if tree.is_file() and tree.read_bytes() == src.read_bytes():
            return None
    except OSError:
        pass
    return str(src)


def _timeline_preamble_fence_hit(ctx: LoopContext) -> "str | None":
    """The card-fence token the probe's injected preamble would trip, read from
    the workspace's ``eval/bench.py`` (the file the stamped job runs); None when
    the card has no literal ``_FORBIDDEN`` fence or nothing in the preamble
    matches it. Never raises: an unreadable bench is the job's problem."""
    try:
        from forgekernel.skills.timeline_probe import preamble_fence_hit
        bench = Path(ctx.workspace) / "eval" / "bench.py"
        if not bench.is_file():
            return None
        return preamble_fence_hit(bench.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a self-check must not become the failure
        return None


def _timeline_probe_ref(ctx: LoopContext, ref: str, label: str) -> str:
    """The commit the timeline job archives for ``ref``: ``ref`` itself when its
    notes/timeline_probe.py is the harness copy, else ``ref`` with the harness
    copy overlaid as a child commit (backend ``with_file``; a backend without
    it probes the tree's own copy, as before).

    The job runs the TREE's copy of the probe and nothing refreshed it, so a
    probe fix never reached a tree committed before it: one run's
    plateau timeline would have run on a best sha carrying a probe one fix old
    and read dead. What is banked and logged stays ``ref`` — only the probe
    file differs, and the probe is the instrument, not the thing measured."""
    be = ctx.backend
    with_file = getattr(be, "with_file", None)
    if not ref or with_file is None:
        return ref
    from forgekernel.skills import timeline_probe as probe_mod
    try:
        text = Path(probe_mod.__file__).read_text(encoding="utf-8")
        out = str(with_file(ref, "notes/timeline_probe.py", text, _TIMELINE_OVERLAY_MSG) or ref)
    except (BackendError, OSError) as e:
        _log(f"timeline[{label}]: could not overlay the harness probe on {ref[:10]} ({e}) — "
             f"probing the tree's own copy")
        return ref
    if out != ref:
        _log(f"timeline[{label}]: notes/timeline_probe.py at {ref[:10]} is not the harness copy — "
             f"probing {ref[:10]} with the harness probe overlaid ({out[:10]}; the tree under "
             f"judgment is unchanged)")
    return out


def _run_timeline(ctx: LoopContext, ref: str | None, label: str, *,
                  branch: str | None = None, workload: str | None = None) -> dict | None:
    """Run the timeline probe on ``ref`` (a pushed ``branch`` when given),
    bank and log the table. None when the job ran no probe at all; a dict
    with ``ok`` False when the probe failed closed (banked, not rendered).
    ``workload``: one of a group card's other meta workloads (loop/meta_shapes.py),
    banked under its name -- every machine reading stays on the representative's."""
    be = ctx.backend
    hit = _timeline_preamble_fence_hit(ctx)
    if hit:
        # The card's kernel fence would refuse the driver's own preamble: every
        # stamped bench dies before the kernel imports and the job can only
        # report no-slab-dumped (observed: three GPU jobs on a GDN run, zero
        # tables). A configuration fault, caught here for the price of a regex.
        _log(f"timeline[{label}]: preamble hits the card fence: {hit!r} — skipping the "
             f"probe; give the fence a word boundary (`\\bos\\s*\\.\\s*environ`) or keep "
             f"that token out of recon/timeline_probe._PREAMBLE, then re-run")
        return None
    stale = _timeline_probe_stale(ctx)
    if stale and not getattr(ctx, "_timeline_stale_logged", False):
        # once per loop: the tree's probe is not this harness's, so its verdict
        # lines (TIMELINE-COVERAGE among them) may be missing or older
        _log(f"timeline: notes/timeline_probe.py in the tree differs from the harness copy "
             f"({stale}) — each probe job gets the harness copy overlaid on the probed ref "
             f"(logged per probe; a backend without with_file runs the tree's copy, whose "
             f"verdict lines may predate this harness and are read through the legacy span "
             f"rule); copy the harness file over notes/timeline_probe.py and commit it to "
             f"refresh the tree itself")
        try:
            ctx._timeline_stale_logged = True
        except AttributeError:
            pass
    push_ref = getattr(be, "push_ref", None)
    if branch is None and ref and (banked := _banked_timeline_of(ctx, ref, workload)) is not None:
        # TIMELINE REUSE: the teacher entry re-reads the best tree, which the
        # gate row / new-best probe (or the previous entry) usually read
        # already — in one large batch of runs, 1672 of 1887 entry probes were of
        # a sha with a renderable table banked. Same tree, same
        # harness probe: the banked table is that reading. A student-requested
        # probe (``branch`` given) always runs.
        _log(f"timeline[{label}] @ {str(ref)[:10]}: reusing the banked table "
             f"({banked.get('label')}, job {banked.get('job') or '?'}) — same tree, same probe")
        return banked["table"]
    if branch is None:
        branch = f"{ctx.branch_prefix}-timeline"
        try:
            if ref and push_ref is not None:
                push_ref(_timeline_probe_ref(ctx, ref, label), branch)
            else:
                be.push_branch(branch)
                ref = be.head()
        except BackendError as e:
            _log(f"timeline[{label}]: could not push {str(ref)[:10]} ({e})")
            return None
    elif ref and push_ref is not None and (probe_ref := _timeline_probe_ref(ctx, ref, label)) != ref:
        # the caller pushed ``branch`` at ``ref`` already: re-point it at the overlay
        try:
            push_ref(probe_ref, branch)
        except BackendError as e:
            _log(f"timeline[{label}]: could not push the overlay {probe_ref[:10]} ({e}) — "
                 f"probing the tree's own copy")
    log = None
    for rerun in range(2):
        try:
            log = be.submit_gpu(branch, _timeline_cmd(ctx)
                                + (f" --workload {workload}" if workload else ""))
        except BackendError as e:
            _log(f"timeline[{label}]: job failed ({e})")
            return None
        if rerun == 0 and _node_dirty_reason(None, log):
            # the job's GPU held an orphan (the probe's own clock-stamped bench
            # timed out; its launch target is still on the GPU): the SAME design
            # is re-probed once — same contract as the ablation / lever-price
            # probes
            _log(f"timeline[{label}]: {_node_dirty_reason(None, log)} — re-probing "
                 f"once")
            continue
        break
    if log is None:
        return None
    from forgekernel.skills.timeline_probe import parse_timeline
    tl = parse_timeline(log)
    job = str(getattr(be, "last_job_id", "") or "")
    if tl is None:
        _log(f"timeline[{label}] @ {str(ref)[:10]}: the job printed no TIMELINE line "
             f"(is notes/timeline_probe.py in the tree?)")
        return None
    # The log line reads the SAME verdict the retrieval does. It used to read
    # ``tl["ok"]``, which stopped being the question when a partial table became
    # renderable: the probe no longer prints TIMELINE-OK for one, so a short-
    # coverage run logged "FAIL — None" and the annotation below never printed.
    if _timeline_renderable(tl):
        roles = "; ".join(f"{r} period={_fmt_cyc(v.get('period'))} busy={_fmt_cyc(v.get('busy'))} "
                          f"wait={_fmt_cyc(v.get('wait'))}" for r, v in tl["roles"].items())
        # A partial table is not a failure: name the shortness here, so a run
        # that reads 0.6 span on every probe is visibly "usable, short" rather
        # than a wall of FAILs that hide the table.
        short = _timeline_short_coverage(tl)
        _log(f"timeline[{label}] @ {str(ref)[:10]}: OK mode={tl['mode']} sites={tl['sites']} "
             f"perturbation={tl.get('perturbation')} span={tl.get('span_ratio')}"
             + (f" (table usable; stamps bracket {tl.get('span_ratio')} of the device window — "
                f"segment cycles exact, rows do not add up to the call)" if short else "")
             + f" — {roles}")
        _note_env_ok(ctx, "timeline")
    else:
        # "base-bench-failed-or-incorrect" over a tree whose own eval passed, with an
        # import error in the log, is the probe's `python3` (loop/job_env.py) — say
        # so here and to whoever reads the feedback, or markers get "fixed" forever
        if (why := env_failure(log)) is not None:
            tl["env"] = why
            _note_env_failure(ctx, "timeline", why)
        _log(f"timeline[{label}] @ {str(ref)[:10]}: FAIL — {tl.get('fail')}"
             + (f" [job environment: {why}]" if why else ""))
    _bank_timeline(ctx, ref or "", job, label, tl, workload)
    return tl


def _fmt_cyc(x) -> str:
    return "n/a" if x is None else f"{int(round(x))}"


def _timeline_renderable(tl: dict) -> bool:
    """Whether a parsed or banked table can be handed to the teacher / student.

    A thin read of ``table_usable`` (the one place the rule lives — see its
    docstring for why ``ok`` and "worth diagnosing from" are separate questions).
    Kept as a name because call sites read better for it, not because the rule
    is re-decided here.
    """
    from forgekernel.skills.timeline_probe import table_usable
    return table_usable(roles=tl.get("roles") or {}, ok=bool(tl.get("ok")),
                        coverage=tl.get("coverage"), span_ratio=tl.get("span_ratio"),
                        sm_src=tl.get("sm_src"))


def _bank_timeline(ctx: LoopContext, sha: str, job: str, label: str, tl: dict,
                   workload: str | None = None) -> None:
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    rec = {"ts": ctx.now(), "sha": sha, "job": job, "label": label, "ok": bool(tl.get("ok")),
           # a group card's other meta workload: a reading beside the representative's,
           # never the one the loop's timeline decisions read (_latest_timeline)
           **({"workload": workload} if workload else {}),
           # ``tl["usable"]`` when the parse produced one (it always does since the
           # field landed; ``_timeline_renderable`` is the fallback for a dict a
           # test or an older path built by hand).
           "usable": bool(tl.get("usable")) if "usable" in tl else _timeline_renderable(tl),
           "coverage": tl.get("coverage"),
           "mode": tl.get("mode"), "ms": tl.get("base_ms"), "table": tl,
           # the instrument the table was read with: reuse requires the same one
           "probe": _timeline_probe_digest()}
    try:
        with (ev / _TIMELINES_NAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        _log(f"timeline: could not persist {_TIMELINES_NAME}: {e}")
    else:
        _bank_evidence(ctx, "timeline table")


def _banked_timelines(ctx: LoopContext) -> list:
    ev = _evidence_dir(ctx, create=False)
    if ev is None or not (ev / _TIMELINES_NAME).is_file():
        return []
    try:
        return [json.loads(ln) for ln in (ev / _TIMELINES_NAME).read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    except (OSError, ValueError) as e:
        _log(f"timeline: could not read {_TIMELINES_NAME} ({e})")
        return []


def _banked_timeline_usable(rec: dict) -> bool:
    """A banked record the driver can render. Records written before ``usable``
    existed are judged on the table they carry, so a resumed run's evidence
    does not go dark mid-run."""
    if "usable" in rec:
        return bool(rec.get("usable"))
    return _timeline_renderable(rec.get("table") or {})


def _latest_timeline(ctx: LoopContext, sha: str | None = None) -> dict | None:
    """The last banked *renderable* timeline (of ``sha`` when given).

    Renderable, not ``ok``: a partial-coverage table is a real table and the
    whole point of the split (see ``_timeline_renderable``). Returning None
    here would silence the meta entries."""
    for rec in reversed(_banked_timelines(ctx)):
        if rec.get("workload"):
            continue      # another meta workload of a group card: not the loop's reading
        if _banked_timeline_usable(rec) and (sha is None or rec.get("sha") == sha):
            return rec
    return None


def _timeline_probe_digest() -> str:
    """A short digest of the harness probe every timeline job runs (overlaid by
    ``_timeline_probe_ref``); '' when it cannot be read."""
    from forgekernel.skills import timeline_probe as probe_mod
    try:
        return hashlib.sha256(Path(probe_mod.__file__).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _banked_timeline_of(ctx: LoopContext, sha: str, workload: str | None = None) -> dict | None:
    """The latest banked renderable timeline of ``sha`` read with today's
    harness probe (on ``workload``: a group card's other meta workload; None =
    the card's own reading), or None. Records banked before the ``probe`` field,
    or by another probe version, never match: their table may be one fix old."""
    digest = _timeline_probe_digest()
    if not digest:
        return None
    for rec in reversed(_banked_timelines(ctx)):
        if (rec.get("sha") == sha and rec.get("probe") == digest
                and rec.get("workload") == workload
                and isinstance(rec.get("table"), dict) and _banked_timeline_usable(rec)):
            return rec
    return None


def _timeline_due(ctx: LoopContext, new_ms) -> bool:
    """A new best is re-read when it beats the tree of the last *renderable*
    timeline by the re-price margin (the same bound that keeps the price-check
    to one job per real move); always when nothing is banked yet.

    Renderable, not ``ok``: a partial-coverage table is a real reading of a
    real tree, so it anchors the margin like any other. (Whether a stale
    anchor should also force a re-read is a separate question — it fires on
    every KEEP and would defeat the margin, so it is not decided here.)"""
    last = _latest_timeline(ctx)
    if last is None:
        return True
    last_ms = _ms_of_tree(ctx, str(last.get("sha") or ""))
    if last_ms is None and gate_window_name(ctx.gate_window) == GATE_WINDOW_DEVICE:
        last_ms = last.get("ms")          # the probe's own base variant, a device reading
    if new_ms is None or not last_ms:
        return True
    return new_ms <= last_ms * (1.0 - ctx.reprice_min_gain_pct / 100.0)


def _ref_region_lines(ctx: LoopContext) -> str:
    """The reference's recon-time stall share per SASS region (the [tensor]
    line and the REGION rows of ref_role_exposure.txt) — the blackbox-safe
    reference column: the teacher maps regions to segments and reads
    cycles ≈ share × the reference's chunk period."""
    ev = _evidence_dir(ctx, create=False)
    p = (ev / "ref_role_exposure.txt") if ev is not None else None
    if p is None or not p.is_file():
        return ""
    try:
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines()
                 if ln.startswith("[tensor]") or ln.startswith("REGION ")]
    except OSError:
        return ""
    return "\n".join(lines[:12])


def _timeline_tables(tl: dict) -> str:
    """The per-role tables of a timeline (shared by the teacher block, the
    student note and the probe feedback)."""
    out = [f"Checks: mode={tl.get('mode')}, numeric={tl.get('numeric')}, perturbation {tl.get('perturbation')}×, "
           f"span/device {tl.get('span_ratio')}, non-monotonic {tl.get('mono_bad')}/{tl.get('mono_tot')}, "
           + (f"row restarts {tl.get('restarts')}/{tl.get('restart_pairs')}, " if tl.get('restart_pairs') is not None else "") +
           f"sites {tl.get('sites')}, steady rows {tl.get('rows')}, SM {tl.get('sm_mhz')} MHz"
           f" ({tl.get('sm_src', 'default')})."]
    pert = tl.get("perturbation")
    if pert is not None and pert > 1.03:
        out.append(f"**Note**: instrumentation stretched device time by {(pert - 1) * 100:.1f}%, so every "
                   "segment's absolute value is inflated by about that ratio; the relative structure "
                   "between segments, the waits and the phases are trustworthy — scale absolute cycle "
                   "counts down by that ratio.")
    if _timeline_short_coverage(tl):
        # Say it in the rendered table, not only in the log — and say only what
        # the span can say. It is the longest slot's max-minus-min over all its
        # stamps, so a short one means the stamps do not reach both ends of the
        # device window; which stretch is missing (an unmarked prologue /
        # epilogue, sites mode=waits dropped, a launch with no stamps) it cannot
        # tell. The segment cycles are stamp differences inside a row and are
        # exact; what the reader must not do is add the rows up into the call,
        # or scale the segments up by 1/coverage (one GDN bwd tree's
        # mode=all and mode=waits tables read 0.947 vs 0.60 with the same
        # per-row periods). Also the caveat for a table banked by a probe that
        # predates TIMELINE-COVERAGE and was read through on its span.
        out.append(f"{meta_shapes.TL_COVERAGE_NOTE} (partial)**: this table's stamps cover only "
                   f"{tl.get('span_ratio')} of device time (mode={tl.get('mode')}). The part they do not "
                   "cover **is not in this table** — it may be a stretch of the marked kernel that no "
                   "pair of sites brackets (mode=waits keeps only `.begin`/`.end` wait pairs), or another "
                   "launch in the same call that carries no stamps; the probe cannot tell which. "
                   "**Segment cycles are in-row stamp differences; read the absolute values as they "
                   "are** (scaled only by the perturbation above), and do not inflate them by "
                   "1/coverage. Only, the segments do not add up to the whole call: do not use this "
                   "table as the time account of the whole call.")
    for w in tl.get("warn") or []:
        out.append(f"{meta_shapes.TL_SELFCHECK_NOTE}**: {w}")
    out += ["", "| role | warps | period per block | busy | wait | wrap (unmarked tail) | period IQR |",
            "|---|---|---|---|---|---|---|"]
    for role, r in tl.get("roles", {}).items():
        out.append(f"| {role} | {r.get('warps')} | {_fmt_cyc(r.get('period'))} | {_fmt_cyc(r.get('busy'))} | "
                   f"{_fmt_cyc(r.get('wait'))} | {_fmt_cyc(r.get('wrap'))} | {r.get('iqr')} |")
    for role, r in tl.get("roles", {}).items():
        out += ["", f"**{role}** {meta_shapes.TL_SEGMENT_LEDGER_HEADING}",
                "| segment | kind | median | p25 | p75 | n |", "|---|---|---|---|---|---|"]
        for sg in r.get("segments", []):
            out.append(f"| `{sg['from']}` → `{sg['to']}` | {sg['kind']} | {_fmt_cyc(sg['median'])} | "
                       f"{_fmt_cyc(sg['p25'])} | {_fmt_cyc(sg['p75'])} | {sg['n']} |")
    if tl.get("phase"):
        out += ["", "Phase (each site's offset from the block's earliest stamp; aligned across roles — "
                "who is late, who waits for whom)",
                "| role | site | offset |", "|---|---|---|"]
        for role, sites in tl["phase"].items():
            for site, off in sorted(sites.items(), key=lambda kv: (kv[1] is None, kv[1] or 0)):
                out.append(f"| {role} | `{site}` | {_fmt_cyc(off)} |")
    return "\n".join(out) + "\n"


def _timeline_block(tl: dict, ref_lines: str = "") -> str:
    """The teacher's timeline evidence block."""
    block = ("\n## Timeline (measured by the driver with %clock64 stamps on the student tree — cycles "
             "per block; the reference is not instrumented)\n"
             + _timeline_tables(tl))
    if ref_lines:
        block += ("\nReference sampling share per SASS region (from recon, available to blackbox runs "
                  "too; you map region ↔ segment through the milestone timeline, cycles ≈ share × the "
                  "reference's period per block):\n```\n" + ref_lines + "\n```\n")
    block += (
        "\n**How to read it (hard rules)**: the diagnosis output block must carry "
        "`locus_segment: <role>.<segment>` (the segment the gap falls in; if it cannot be placed in "
        "a segment, write `none` and say why), and `deficient_locus` points at the decision that owns "
        "that segment. A large wait segment is not the wait's fault — look in the phase table at what "
        "the awaited role is doing at that moment; the lesion is on the side being waited for. Only a "
        "large busy segment is this role's own instruction stream. "
        "busy + wait + wrap ≈ period; wrap is the unmarked tail from the last site to the next "
        "block's first site — when it is large, have the student add markers there before judging. "
        "A `mode=waits` table keeps only the wait sites (the fallback when full-site instrumentation "
        "perturbs too much): its busy segments are merged spans between adjacent waits, and "
        "cross-role phases are slightly off (the waiting side reads ~10% high). An `.end` site after "
        "`bar.sync` can fire early (DEFER_BLOCKING), and the rendezvous time then lands in the next "
        "segment — segments with a wide p25/p75 spread are this case. When one role is split into "
        "two `@wg<k>` rows, the skew between the two warpgroups is itself a reading (who arrives "
        "first, who waits for whom).\n")
    return block


def _timeline_student_note(ctx: LoopContext) -> str:
    if not _timeline_on(ctx):
        return ""
    has_def, names = _timeline_markers(ctx)
    from forgekernel.skills.timeline_probe import NSITE
    if _is_cuda(ctx):
        return _timeline_student_note_cuda(ctx, has_def, names, NSITE)
    if _is_tilelang(ctx):
        return _timeline_student_note_tilelang(ctx, has_def, names, NSITE)
    lines = ["\n## Timeline probe (enabled for this run)",
             "- Convention: keep `def _seg(site, row=0): return None` in the kernel (the driver turns it "
             "into stamping only in the probe build); put `_seg(\"<role>.<segment>.<stem>.begin\", i)` / "
             "`\".end\"` before and after every blocking wait, and one marker after each chain commit, "
             "after each `wait_group(0)` drain, and at each role's entry; names use the segment names of "
             "the milestone timeline and **must be string literals** "
             "(the probe builds its site table from the source text: computed names such as "
             "`\"…%d…\" % k` or `NAMES[k]` cannot be read and are treated as no-ops in the probe build; "
             "to split sites per unrolled step, write N literal calls under branches such as "
             "`if cutlass.const_expr(k == 0)`), `i` = chunk / tile counter "
             "(not passed outside loops). In a persistent kernel (one CTA runs several work items / "
             "tile batches across the grid) a counter that keeps increasing over the CTA's whole life "
             "is the more convenient `i` — if the row restarts per work item, the probe automatically "
             "reads it as time-monotonic stretches, restarts do not count toward wrap/period, and the "
             "low-row-number segments only read the last consistent stretch of the loop. Send "
             "`PROBE_REQUEST: timeline` to get a table of the current tree (counts as one probe).",
             f"- **Site cap: {NSITE}**: beyond it the probe keeps only the `.begin`/`.end` wait pairs "
             f"(the other markers cannot be read) — as the marker count nears the cap, merge sites of "
             f"the same segment."]
    if has_def and names:
        lines.append(f"- Markers in the current tree ({len(names)}): " + ", ".join(f"`{n}`" for n in names[:24])
                     + ("…" if len(names) > 24 else ""))
        if len(names) > NSITE:
            lines.append(f"- **{len(names)} markers now > cap {NSITE}**: full-table mode will be "
                         f"skipped and only wait pairs measured; trim to {NSITE} or fewer.")
    else:
        lines.append("- **The current tree has no markers** (or lacks the `_seg` definition): the "
                     "driver cannot measure your timeline; add them before the next gate row.")
    if has_def and (bad := _timeline_unparsed_markers(ctx)):
        # observed: six computed site names the table could not see
        # killed the stamped variant at trace time on three trees, and the failure
        # line never said why — say it here, every turn, before the next probe
        lines.append(f"- **{len(bad)} `_seg(` calls have site names that are not string literals; "
                     f"the probe cannot read them (treated as no-ops in the probe build)**: "
                     + ", ".join(f"`{c}`" for c in bad[:8]) + ("…" if len(bad) > 8 else "")
                     + "; to split sites per unrolled step, write one literal call per step.")
    stale = _timeline_probe_stale(ctx)
    if stale:
        lines.append(f"- **The tree's `notes/timeline_probe.py` does not match the harness version (or "
                     f"is missing)**: copy `{stale}` over `notes/timeline_probe.py` and commit it with "
                     f"your next submission — the job runs the copy in the tree, and an old version's "
                     f"self-check lines (e.g. no `TIMELINE-COVERAGE`) leave the driver reading the "
                     f"table by the old rules.")
    last = _latest_timeline(ctx)
    if last is not None:
        tl = last.get("table") or {}
        lines.append(f"- Latest table (@`{str(last.get('sha'))[:10]}`, {last.get('label')}):")
        if _timeline_short_coverage(tl):
            # The student reads these cycle counts as the size of its own
            # segments. A short-coverage table's absolutes are not that, and the
            # teacher's copy already carries this note — same table, same caveat.
            lines.append(f"  - {meta_shapes.TL_COVERAGE_NOTE}**: this table's stamps cover only "
                         f"{tl.get('span_ratio')} of device time (mode={tl.get('mode')}); the uncovered "
                         f"part is not in the table (stretches no marker brackets, sites that mode=waits "
                         f"dropped, or launches in the same call without stamps); **segment cycles are "
                         f"in-row stamp differences; read the absolute values as they are** — do not "
                         f"inflate them by coverage, and do not take the sum of the segments as the whole "
                         f"call; to make the table cover everything, put sites in the prologue / epilogue "
                         f"and in the other kernels too.")
        for role, r in tl.get("roles", {}).items():
            top = sorted([sg for sg in r.get("segments", []) if sg.get("median") is not None],
                         key=lambda sg: -sg["median"])[:3]
            lines.append(f"  - {role}: period {_fmt_cyc(r.get('period'))}, busy {_fmt_cyc(r.get('busy'))}, "
                         f"wait {_fmt_cyc(r.get('wait'))}; largest three segments "
                         + ", ".join(f"`{sg['from']}`→`{sg['to']}` {_fmt_cyc(sg['median'])} ({sg['kind']})"
                                     for sg in top))
    return "\n".join(lines) + "\n"


def _timeline_student_note_cuda(ctx: LoopContext, has_def: bool, names: list, nsite: int) -> str:
    """``_timeline_student_note`` for a CUDA-mode run: the C++ marker convention
    (``timeline_probe.CU_MARKER_*``), the same site rules, cap, staleness and
    latest-table lines."""
    lines = ["\n## Timeline probe (enabled for this run)",
             f"- Convention: keep one line `#define FK_SEG(...) ((void)0)` in `{ctx.kernel_file}` (the "
             f"driver replaces it with stamping only in the probe build); "
             "put `FK_SEG(\"<role>.<segment>.<stem>.begin\", i);` / `\".end\"` before and after every "
             "blocking wait, and one marker after each chain commit, after each wait drain, and at each "
             "role's entry; names use the segment names of the milestone timeline and **must be string "
             "literals** "
             "(the probe builds its site table from the source text: names computed by macro pasting or "
             "array lookup cannot be read and are treated as no-ops in the probe build; to split sites "
             "per unrolled step, write N literal calls), `i` = chunk / tile counter (outside loops write "
             "`FK_SEG(\"<role>.entry\");`; no parentheses inside `i`). In a persistent kernel a counter "
             "that keeps increasing over the CTA's whole life is the more convenient `i` — if the row "
             "restarts per work item, the probe reads it as time-monotonic stretches. The probe build "
             "adds a readout function to your `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` block; keep "
             "that block in its standard form. Send `PROBE_REQUEST: timeline` to get a table of the "
             "current tree (counts as one probe).",
             f"- **Site cap: {nsite}**: beyond it the probe keeps only the `.begin`/`.end` wait pairs "
             f"(the other markers cannot be read) — as the marker count nears the cap, merge sites of "
             f"the same segment."]
    if has_def and names:
        lines.append(f"- Markers in the current tree ({len(names)}): " + ", ".join(f"`{n}`" for n in names[:24])
                     + ("…" if len(names) > 24 else ""))
        if len(names) > nsite:
            lines.append(f"- **{len(names)} markers now > cap {nsite}**: full-table mode will be "
                         f"skipped and only wait pairs measured; trim to {nsite} or fewer.")
    else:
        lines.append("- **The current tree has no markers** (or lacks the `#define FK_SEG` line): the "
                     "driver cannot measure your timeline; add them before the next gate row.")
    if has_def and (bad := _timeline_unparsed_markers(ctx)):
        lines.append(f"- **{len(bad)} `FK_SEG(` calls have site names that are not string literals; "
                     f"the probe cannot read them (treated as no-ops in the probe build)**: "
                     + ", ".join(f"`{c}`" for c in bad[:8]) + ("…" if len(bad) > 8 else "")
                     + "; to split sites per unrolled step, write one literal call per step.")
    stale = _timeline_probe_stale(ctx)
    if stale:
        lines.append(f"- **The tree's `notes/timeline_probe.py` does not match the harness version (or "
                     f"is missing)**: copy `{stale}` over `notes/timeline_probe.py` and commit it with "
                     f"your next submission.")
    last = _latest_timeline(ctx)
    if last is not None:
        tl = last.get("table") or {}
        lines.append(f"- Latest table (@`{str(last.get('sha'))[:10]}`, {last.get('label')}):")
        if _timeline_short_coverage(tl):
            lines.append(f"  - {meta_shapes.TL_COVERAGE_NOTE}**: this table's stamps cover only "
                         f"{tl.get('span_ratio')} of device time (mode={tl.get('mode')}); segment cycles "
                         f"are in-row stamp differences; read the absolute values as they are, and do "
                         f"not inflate them by coverage.")
        for role, r in tl.get("roles", {}).items():
            top = sorted([sg for sg in r.get("segments", []) if sg.get("median") is not None],
                         key=lambda sg: -sg["median"])[:3]
            lines.append(f"  - {role}: period {_fmt_cyc(r.get('period'))}, busy {_fmt_cyc(r.get('busy'))}, "
                         f"wait {_fmt_cyc(r.get('wait'))}; largest three segments "
                         + ", ".join(f"`{sg['from']}`→`{sg['to']}` {_fmt_cyc(sg['median'])} ({sg['kind']})"
                                     for sg in top))
    return "\n".join(lines) + "\n"


def _timeline_student_note_tilelang(ctx: LoopContext, has_def: bool, names: list,
                                    nsite: int) -> str:
    """``_timeline_student_note`` for a TileLang-mode run.

    The probe speaks TileLang now: the marker convention is the DSL's unchanged
    (``def _seg(site, row=0): return None`` / ``_seg("<role>.<seg>.<event>", i)``), and the
    stamped build turns the definition into a ``@macro`` that emits the same PTX through
    ``T.import_source`` + ``T.call_extern``. Verified end to end on an RTX 6000D:
    ``TIMELINE-OK mode=all``, ``numeric=identical``.

    What a student must NOT do is reach for the DSL spelling (``import cutlass`` /
    ``@dsl_user_op`` / ``cute.arch.*``): a TileLang kernel cannot execute it, and rewriting
    the kernel around a probe that does not need it costs rounds for nothing."""
    lines = ["\n## Timeline probe (enabled for this run)",
             "- Convention: keep `def _seg(site, row=0): return None` in the kernel — the probe "
             "replaces it in its own stamped build only. Put `_seg(\"<role>.<segment>.stem.begin\", "
             "i)` / `\".end\"` around every blocking wait and one marker after each chain commit.",
             "- **TileLang is supported**: do not add an `import cutlass` / `@dsl_user_op` / "
             "`cute.arch.*` spellings — the probe rewrites the marker definition itself. A kernel "
             "that compiles and runs is all the probe needs; do not restructure it for the probe."]
    if has_def and names:
        lines.append(f"- Markers in the current tree ({len(names)}): " + ", ".join(
            f"`{n}`" for n in names[:24]) + ("…" if len(names) > 24 else ""))
    elif not has_def:
        lines.append("- The tree carries no `_seg` definition yet: add "
                     "`def _seg(site, row=0): return None` beside the kernel's other module-level "
                     "helpers and mark the wait pairs.")
    stale = _timeline_probe_stale(ctx)
    if stale:
        lines.append(f"- `notes/timeline_probe.py` does not match the harness version (or is "
                     f"missing): copy `{stale}` over it with your next submission.")
    return "\n".join(lines) + "\n"


def _timeline_feedback(tl: dict | None) -> str:
    if tl is None:
        return ("\n## Timeline probe\n- The job printed no TIMELINE line (the probe did not run); check "
                "that `notes/timeline_probe.py` is in the tree.\n")
    if not _timeline_renderable(tl):
        if tl.get("env"):
            return (f"\n## Timeline probe\n- Did not run: a **job environment** problem (`{tl['env']}`), "
                    f"not a problem with the tree or the markers — adding markers or changing the kernel "
                    f"will not make it pass. Do not send `PROBE_REQUEST: timeline` again until the "
                    f"environment is fixed; take segment readings from the pricing table or a verdict "
                    f"probe instead.\n")
        return (f"\n## Timeline probe\n- Failed: {tl.get('fail')}. Add the missing markers / fix it, then "
                f"send `PROBE_REQUEST: timeline` again.\n")
    return "\n## Timeline (measured on the current tree)\n" + _timeline_tables(tl)
