"""Two-layer loop driver — the orchestration that was previously hand-driven.

The driver chains the primitives (scaffold / submit / trajectory / ncu / resume)
into the inner student loop and the outer meta loop, driving sub-agents and GPU
jobs through a single injected ``backend`` seam. All control flow here is
deterministic given the backend, so it is unit-tested with a fake; the real
backend (agent CLI + GPU jobs + git) lives in ``forgekernel.loop.cli``.

Inner loop per iteration:
  render student prompt -> agent -> (probe sub-loop until a record) -> commit +
  push -> GPU ``all`` (correctness+perf) -> GPU ``ncu`` (profiler) -> append
  IterationRecord. relaxed-revert: a perf regression is KEPT (scaffolding), only a
  correctness break REVERTs (and rolls the workspace back). Pre-green — before the
  lineage's first correctness PASS, e.g. while M0 is under construction — committed
  sub-steps are NOT evaluated (NEUTRAL rows, POLICY §3 build-phase semantics, executed): a full
  eval of a half-built kernel decides nothing, and the deferred verification is
  spent by the gate row ("M0-gate"), which always gets the full eval.

Outer loop: when the inner loop plateaus, run meta diagnose + patch, compute the
prefix-reuse resume point, reset the workspace there, and resume the inner loop.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shlex
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Protocol

from forgekernel.inner.ncu import kernels_from_ncu, profiler_from_ncu
from forgekernel.skills.sass_census import parse_census_line, parse_serial_line
from forgekernel.loop.job_env import env_failure
from forgekernel.loop import harness_guard, precision_gate
from forgekernel.loop.kb_lint import lint_kb_dir
from forgekernel.loop import lang_mode
from forgekernel.inner.trajectory import (
    GATE_WINDOW_DEFAULT,
    GATE_WINDOW_DEVICE,
    GATE_WINDOW_SERVER,
    GATE_WINDOWS,
    IterationRecord,
    append_record,
    constant_output_flag,
    gate_ms,
    gate_window_name,
    load_records,
)
from forgekernel.meta_trajectory import (
    MetaRecord,
    append_meta_record,
    load_meta_records,
    reflect_summary,
    update_last_outcome,
)
from forgekernel.window_contract import PERF_WORKLOADS
from forgekernel.loop.parse import (
    describe_workload_readings,
    ENTRY_DONE_RE,
    Exhausted,
    JOB_END_KILLED,
    ProbeRequest,
    ProbeVerdict,
    Record,
    detect_plateau,
    job_abend,
    parse_eval,
    parse_probe_verdict,
    parse_student_turn,
    ReconRequest,
)
from forgekernel.baseline import RefBaseline, workloads_mismatch
from forgekernel.resume import Decision, compute_reopen, compute_resume
# a group card's meta readings, one workload per job
from forgekernel.loop import meta_shapes


class DriverBackend(Protocol):
    """The side-effecting seam: agent spawn, GPU submit, git. Faked in tests."""

    def run_agent(self, prompt: str) -> str: ...
    def submit_gpu(self, branch: str, mode: str) -> str: ...  # returns gpu_entry stdout text
    def commit(self, message: str) -> str: ...  # returns the new HEAD sha
    def push_branch(self, branch: str) -> None: ...
    def reset(self, ref: str) -> None: ...
    def head(self) -> str: ...  # current HEAD sha (baseline for meta resume)


AGENT_ROLE_STUDENT = "student"
AGENT_ROLE_TEACHER = "teacher"

# ENTRY_DONE_RE (the operator entry script's closing line) now lives in parse,
# where parse_probe_verdict demands it behind a PASS; imported above so loop.cli
# and the tests keep finding it here.


def archive_is_whole(log: str, task: str = "") -> bool:
    """Is ``log`` a complete job archive: the entry script's closing line is its
    LAST non-empty line and names ``task`` (the submitted gpu_entry task, when
    the caller knows it)?

    Anywhere else the line proves nothing: a ``cmd`` job that wraps another
    entry run (orch_census_run streams ``bash gpu_entry.sh ncu`` line by line
    and prints its ORCH-* verdict after it; teacher probes shell out the same
    way) carries the inner ``task=ncu finished`` mid-log, and an archive read
    at that point is exactly the truncated tail this check exists to refuse.
    """
    lines = [ln for ln in log.splitlines() if ln.strip()]
    if not lines:
        return False
    m = ENTRY_DONE_RE.match(lines[-1].strip())
    return bool(m) and (not task or m["task"] == task)


def _backend_takes_role(be) -> bool:
    """Does ``be.run_agent`` accept ``role=``? Backends that predate the split
    (test fakes, single-model runs) take only the prompt and serve both roles."""
    try:
        return "role" in inspect.signature(be.run_agent).parameters
    except (TypeError, ValueError):
        return False


class _TeacherView:
    """``be`` with ``run_agent`` bound to the teacher role; everything else passes
    through. Teacher turns (recon, distill, diagnose/patch, KB certify/harvest,
    lever design) go here so a run can use the teacher on one model and the
    student on another (``--teacher-agent-cmd``); a backend without a teacher
    command serves the turn from its only agent, as before."""

    def __init__(self, be):
        self._be = be

    def run_agent(self, prompt: str) -> str:
        if _backend_takes_role(self._be):
            return self._be.run_agent(prompt, role=AGENT_ROLE_TEACHER)
        return self._be.run_agent(prompt)

    def __getattr__(self, name):
        return getattr(self._be, name)


def _teacher(be):
    """The teacher-role view of a backend (idempotent)."""
    return be if isinstance(be, _TeacherView) else _TeacherView(be)


class AgentError(RuntimeError):
    """The agent backend failed unrecoverably (after its own retries).

    Raised by the real backend when the agent CLI keeps failing; the loop catches
    it and stops cleanly instead of crashing mid-run.
    """


class BackendError(RuntimeError):
    """A backend side-effect (GPU submit, git push/commit/reset) failed after
    retries. The loop catches it and stops cleanly rather than crashing."""


class NoRunError(BackendError):
    """A GPU submit that ended with NO job producing output: every job lost its
    runner, or the GPU would not hold the board's clock lock (loop.cli raises the
    concrete kinds). Nothing ran, so nothing may be
    charged to the payload — a caller that counts attempts against a tree
    (_ensure_ref_pricing: "the same tree would hang again") must not count one
    of these. Its own type because recognising the case by a word in the
    message is control flow keyed on prose: a no-run verdict worded differently
    slipped past it, wrote the attempt marker, and (same tree, same key)
    forfeited the reference pricing for good."""


@dataclass
class ProbeBattle:
    """One student probe path's GPU-round bill, kept honest per *fight*.

    A probe that needs many rounds before its PASS is a knowledge gap that never
    surfaces as a G3: the student wins, so the inner loop never stops ``stuck``
    and the meta layer never diagnoses it. A fight is the consecutive rounds on
    one path up to a PASS; it routinely spans iterations (max_probe_rounds caps
    one iteration, not the fight), so ``open_rounds`` carries across them. The
    bill (``rounds`` + evidence) is always ONE winning fight, taken whole: a
    cheap win followed by later re-probes must not add up to a phantom epic, and
    a re-won fight must not cite the previous fight's verdict — rounds, pass
    iter, verdict and log tail travel together or not at all."""
    path: str
    open_rounds: int = 0  # rounds in the fight currently in progress (no PASS yet)
    rounds: int = 0       # the billed fight: the dearest winning fight's rounds
    passed: bool = False
    pass_iter: int | None = None
    verdict_line: str = ""
    log_tail: str = ""  # bounded tail of the billed fight's winning job log


@dataclass
class LoopContext:
    """Driver configuration plus M0 audit / stall pricing / timeline toggles.

    **STRUCT channel**: on blackbox runs the per-segment construction table of
    the reference is a mandatory reconnaissance evidence channel
    (``_struct_recon`` gates it on ``oracle_mode``), decoded by the vendored
    ``struct_tools`` and banked as the ref's construction table for the distill
    brief — see ``_read_ref_struct_table``.
    """
    op: str
    workspace: Path
    milestone: Path
    kb_dir: Path
    status_path: Path
    meta_dir: Path
    kernel_file: str
    branch_prefix: str
    target_ms: float
    deadline_epoch: float
    backend: DriverBackend
    now: Callable[[], float]
    language: str = "cutedsl"  # kernel implementation language (governs the student prompt)
    # Which hw_kb/<language>/<arch>/ tree this run reads AND writes -- see kb_read_dir. One
    # tree per architecture, no shared root (hw_kb/<language>/README.md): a recipe earned on
    # Hopper is not evidence about Blackwell, so a run sees its own tree only.
    # From loop.toml [loop].kb_arch (or --kb-arch); "sm90" is the default.
    kb_arch: str = "sm90"
    # Which timing window every verdict reads: best, target stop, plateau, rollback
    # anchor, digest ranking. "device" (pipelined batch ≈ device time — the
    # kernel-quality claim, the default) or "e2e" (per-call+sync, host
    # tax included; per-task opt-in). Per-task config because neither is right in
    # general — see GATE_WINDOWS in inner/trajectory.py. target_ms must be anchored
    # in this same window; the CLI derives it that way and refuses a device gate
    # over a baseline that never measured the device window.
    gate_window: str = GATE_WINDOW_DEFAULT
    # Skip the profiling half of an eval for a candidate that is clearly SLOWER
    # than the sitting best (see FAST_SKIP_NOISE_MULTIPLE / _fast_skip_reason).
    # Off by default: the mechanism line of the gate wants an NCU table on every
    # committed row, and only runs that pay more in GPU time than in diagnosis
    # should turn it on (loop.toml [loop] fast_skip = true).
    fast_skip: bool = False
    meta_status_path: Path | None = None  # meta-loop trajectory (meta agent's memory)
    fresh_recipe: str | None = None  # a just-certified KB recipe to point the student at
    # In-place reopening of structural / split patches: keep the best green tree
    # and redo only the dependency closure of the changed entries on it, instead
    # of resetting to the checkpoint before the earliest affected decision and
    # rebuilding everything after it (a sparse-attention run once spent six of ten
    # meta steps on whole-M0 rebuilds, two of them regressing to 3.9 ms). The closure
    # is announced to the student as a REOPEN row in status.jsonl plus a driver
    # context note. Off per task via loop.toml [loop] reopen_in_place = false.
    reopen_in_place: bool = True
    reopen_note: str = ""
    g5_note: str = ""  # G5 in-segment reordering directive for the student (cleared with fresh_recipe)
    # A phase the price table read as exhausted and the diagnosis closed:
    # {"gate": "M1-gate", "phase": "M1", "note": <student-facing text>}. The
    # note rides every student turn until the gate row is submitted (any
    # verdict) or a structural resume replaces the milestone premise. None
    # when no phase is closed this way. See _phase_exhaustion.
    phase_close: dict | None = None
    # status.jsonl index ranges [lo, hi) of post-patch segments a REVERT_PATCH
    # rolled back. The rows stay on file (the student's ledger is append-only)
    # but describe a tree that no longer exists: the same-stage gate check
    # must not take a reverted segment's gate as the "previous" reading, or a
    # rebuild that regressed and was reverted lowers the bar for the next one.
    # In-memory like the pending transaction itself; a driver restart starts
    # both empty.
    voided_ranges: list = field(default_factory=list)
    # Reference tokens the certify-time KB lint scans for, on top of kb_lint's
    # static list — derived from the resolved oracle tree by the CLI at startup.
    kb_lint_tokens: list[str] = field(default_factory=list)
    # Diagnosis carried into the NEXT student turn when the last evaluated round was
    # host-bound (the student's own host path, never the reference's). Set by
    # _check_host_bound, consumed and cleared by _student_prompt.
    host_bound_note: str = ""
    # Compiler-fallback note standing until the chains un-serialize: the last
    # evaluated KEEP's wgmma serialization (DEPBAR-0x0 waits per wgmma ~1).
    # Set from the sass_census fields the orchestration census emits; cleared
    # when a later KEEP prices pipelined again.
    sass_serial_note: str = ""
    # Standing notice for every student turn of a post-patch segment whose workspace
    # was reset by _resolve_resume: names the checkpoint and the invalidated
    # decisions, because the student otherwise reads the trajectory's old records as
    # proof those decisions are "delivered" and reports EXHAUSTED without landing a
    # single row (a GDN run burned 4 meta steps on exactly this). Set at each
    # meta resume, overwritten by the next one.
    resume_note: str = ""
    decisions_dag: list[Decision] = field(default_factory=list)
    oracle_path: str | Path | None = None
    # Which channel the teacher distills through: "source" reads oracle_path,
    # "blackbox" withholds it and distills from black-box reconnaissance of the
    # reference instead, "none" does not distill. Kept separate from oracle_path
    # being None because those used to be the same thing, and "forbidden to read
    # source" then silently meant "do not distill, and do not measure the baseline
    # that distillation needs" — the loop would start against whatever milestone
    # happened to be on disk.
    oracle_mode: str = "source"
    recon_path: Path | None = None  # the reconnaissance report distilled from (blackbox
                                    # always; a source-mode run that wired a recon tree)
    recon_dir: Path | None = None   # the teacher-only tree recon scripts are written into
    # A second backend, pointed at the reconnaissance tree. Reconnaissance must not run
    # through the student's submission tree: the scripts and everything they learn about
    # the reference would ride the student's job tarball, and what the student is meant
    # to receive is a milestone, not the analysis behind it.
    recon_backend: "DriverBackend | None" = None
    max_recon_rounds: int = 12
    simulator_dir: Path | None = None  # hardware calculator; pure-python, no GPU needed
    candidate: str | None = None       # which candidate architecture this run implements
    ref_baseline: "RefBaseline | None" = None  # library baseline: "better than the library?"
    # The timed-window CONTENT contract (operators/<op>/window.toml, see
    # forgekernel.window_contract): which inputs the reference's timed call
    # receives that the student's does not, and how each is classed. The CLI
    # refuses to start a run with a reference and no consistent declaration;
    # the driver renders it to the teacher next to both per-kernel tables and
    # prices the declared rebuilds at confirmation. None = not declared (the
    # binding then says so out loud rather than staying silent).
    window_contract: "WindowContract | None" = None
    arch_baseline: "RefBaseline | None" = None  # the implementation the milestone teaches;
    # target_ms anchors here. Separate from ref_baseline because they answer different
    # questions and are ~20% apart on this operator — deriving the stop threshold from
    # the library one let the loop plan to finish at three quarters of the oracle.
    max_probe_rounds: int = 4
    max_patch_rounds: int = 3
    # GPU verdict-probe budget per diagnose step (0 disables). Before this existed the
    # diagnose agent's only verification granularity was a whole inner segment judged
    # on scalar best-progress: a poisoned best waterline read three correct patches as
    # useless (an attention fwd run, 7.5h), and an unfalsified recon hypothesis steered
    # a 33.6h bwd-MHA run. A verdict probe collapses the hypothesis space at the cost
    # of one GPU job instead.
    max_verdict_probe_rounds: int = 4
    # INFEASIBLE GATE: the fewest diagnoses (this one included; the distillation
    # and a clean M0 audit do not count) before a stop request can be accepted.
    # 1 = the reachability check alone gates the stop (see _infeasible_refusal).
    # loop.toml [loop] infeasible_min_diagnoses / --min-meta-before-infeasible.
    infeasible_min_diagnoses: int = 1
    ncu_retries: int = 2  # retry NCU when a passing kernel yields an empty profiler
    # LEVER PRICING ROUND (plateau diagnosis AND the M0 audit). Before the
    # teacher names a cause_class it designs N composable levers on the best tree
    # and the driver prices them in ONE job — base / +each / all / all-each — so
    # the diagnosis reads MEASURED prices, and a lever that
    # is worthless alone but required by the others (single-add ≈ 0, all-minus
    # < 0) is seen as a package member rather than dropped. A dsa_sparse run
    # named one locus per audit off the ledger; three ablation probes on its
    # best tree afterwards priced three levers in two hours, one of which (ring
    # acquire granularity, the largest) was never a ledger line at all. The audit
    # is included because that is where structural reopens are decided: the
    # same run flipped its stage split axis at the M0 audit on an inferred
    # masking window, with no price on either the flip or the levers it passed
    # over. The value is the number of interleaved bench ROUNDS the pricing job
    # times each variant (the in-job repeat count); 0 disables the round. A
    # PRICE-FAIL (an edit that does not apply) re-prompts the design turn
    # _LEVER_PRICE_RETRIES times, independently of this value. A table priced
    # on the same best tree earlier in the run is reused, not re-bought.
    lever_pricing_rounds: int = 3
    lever_pricing_max_levers: int = 5
    # A PRICE IS A READING ON ONE TREE. Every row the pricing round banks carries
    # the best sha it was measured on (`priced_on`), and when a later KEEP takes
    # the watermark the driver re-prices — one job, base / +each only, the
    # recipes banked with the table — every lever the milestone cites
    # (`lever:<name>`) whose price was read on another tree. A citation whose
    # price reversed or faded on the new best is marked in the milestone and
    # suspended for the student until the teacher's next entry reads the
    # revision (observed on a GDN run: a +25.6% rung priced on the first green was
    # walked two trees later, where the same lever read −11.6% and serialised
    # every wgmma; the gate row and the next lever were built on that tree).
    # Off via loop.toml [loop] reprice_on_new_best = false. A new best that
    # beats the tree the citation was last priced on by less than
    # reprice_min_gain_pct (same window, same-cluster rows) is not re-priced —
    # one job per watermark move would otherwise ride every KEEP of a climbing
    # segment; the teacher's next entry prices the current best anyway.
    reprice_on_new_best: bool = True
    reprice_min_gain_pct: float = 3.0
    # TIMELINE PROBE (recon/timeline_probe.py): the student's `_seg(...)`
    # markers rewritten into %clock64 stamps in a probe build — per role, per
    # segment, cycles per chunk, who waits on whom — the unit a latency-bound
    # kernel is judged in (the stall / role account only says who waits). "on":
    # every teacher entry (M0 audit, plateau), every gate row, every new best
    # beyond reprice_min_gain_pct, and on request (`PROBE_REQUEST: timeline`
    # from the student or the teacher); "auto": on when the milestone declares
    # `bottleneck: latency`; "off": never. The reference is never instrumented.
    timeline_probe: str = "off"
    # NO REOPEN WITHOUT A PRICE. An R1 diagnosis (or any structural / split patch)
    # must cite a price the driver measured: a verdict probe run this step, a
    # lever on this step's price table, or a job-certified hw_kb entry
    # (`reopen_price: <delta> via probe:<path>#<n> | lever:<name> | KB:<entry>`).
    # Without one the diagnosis is bounced once with the three routes, then
    # refused — recorded UNPRICED_REOPEN, nothing patched. Off via loop.toml
    # [loop] reopen_requires_price = false (control runs); auto-off when the run
    # has no in-step way to measure (no verdict probes and no pricing round).
    reopen_requires_price: bool = True
    # REOPEN BUDGET. A reopen redoes a whole closure of delivered decisions on the
    # best tree (often the entire M0 package), so a run of them that keep coming
    # back REVERT_PATCH spends iteration after iteration re-delivering what was
    # already built — and a re-delivery read on a lucky node looks like a gain.
    # Once this many reopen transactions IN A ROW have closed without a new best
    # (REVERT_PATCH; a KEEP_PATCH of any patch class resets the count; a held
    # transaction counts once, when it closes), the teacher is told reopens are
    # closed and a structural / split patch is refused — recorded REOPEN_CAPPED,
    # milestone restored, nothing reopened. Additive / KB patches and the
    # INFEASIBLE stop request are untouched. 0 = no budget (the old behavior).
    # loop.toml [loop] reopen_max_fruitless.
    reopen_max_fruitless: int = 2
    # UNCHANGED-TREE READING. A row whose commit changes nothing outside notes/
    # (a gate acceptance row submitted alone, a decision re-filed "as built", a
    # pricing-only row) is not re-measured: the driver archives it with the
    # reading of the evaluated KEEP on that same tree (IterationRecord.reading_of)
    # — no eval / NCU job, and no extra noise draw for best / patch judgment to
    # select on. "all" (default) | "gate" (gate rows only) | "off" (re-measure
    # every row, the old behavior). loop.toml [loop] reuse_unchanged_reading.
    reuse_unchanged_reading: str = "all"
    # DECLARED GATE CLOSE. A delivery whose change text carries `[CLOSES: M<k>-gate]`
    # and that KEEPs on a full eval gets its Mk-gate row archived by the driver
    # right after it, on the same tree and the same reading — the separate
    # student turn (plus smoke probe and eval) a no-code gate row costs is not
    # spent. The gate stays its own ledger row and every gate consumer (phase
    # close, same-stage regression, the reopened-M0 audit, the timeline bank)
    # reads it as before. Not on a pre-green row (no eval to close on), an
    # unfinished `part k of n` step, or over an open placeholder. loop.toml
    # [loop] declared_gate_close = false turns the tag into plain text.
    declared_gate_close: bool = True
    # [loop] harness_guard (loop/harness_guard.py): "revert" refuses a delivery whose
    # commits edit the card-owned evaluation harness (gpu_entry.sh, eval/, ref_bin/,
    # .gitattributes) before any job is bought, and puts a drifted lineage back to the
    # launch tree's harness at the start of every iteration; "warn" logs only; "off".
    # harness_anchor is the launch tree (run_two_layer sets it).
    harness_guard: str = "revert"
    harness_anchor: str = ""
    # [loop] precision_contract (loop/precision_gate.py): "strict" makes a failed
    # precision check a correctness failure, "official" records it only, "off" does not
    # read it. The CLI resolves the card kind's default (leaderboard card: official,
    # own bench: strict); "off" here keeps scripted contexts neutral.
    precision_contract: str = "off"
    # The card's bench is wired to report the precision block (precision_gate.bench_findings
    # found nothing missing at launch): under strict an eval that comes back without one
    # is then a correctness failure, not "unreported" -- the block's absence is something
    # the kernel, running in the bench's process, could have caused.
    precision_bench_reports: bool = False
    precision_unreported_warned: bool = False
    plateau_window: int = 3
    max_iters: int = 60
    max_consecutive_reverts: int = 3
    # kb-tripwire: a student probe that needs >= this many GPU rounds before its
    # PASS is queued for the end-of-run KB harvest write-up (run_kb_harvest).
    # 0 disables. The threshold is on the *fight* (cumulative rounds per probe
    # path), not on one iteration's sub-loop.
    kb_tripwire_rounds: int = 6
    probe_battles: dict[str, ProbeBattle] = field(default_factory=dict)
    # Force a meta audit the first time a segment goes green (= the M0 gate lands).
    # The gate verifies the student against the milestone's OWN declared values — it
    # cannot verify the milestone. Every iteration after M0 builds on the foundation,
    # so a foundation defect audited at the M0 close costs one meta turn; discovered
    # at the next plateau it costs the whole segment built on top of it (observed on a
    # bwd run: bankconf 73M was severe-flagged early and scheduled for reckoning at D2.4
    # per the milestone — 26 iterations and ~10h later).
    m0_audit: bool = True
    # Per-SASS stall pricing: a fourth recon evidence channel (STALL-OK on the ref)
    # and a driver-run pricing job on the best student kernel before every plateau
    # diagnosis. One fwd run's meta chose two loci the aggregate digest could
    # not falsify — both priced at ~1pp once measured — and burned 10h proving it;
    # the ledger is the evidence face that diagnosis was missing. Off per task via
    # loop.toml [loop] stall_pricing = false.
    stall_pricing: bool = True
    # Set by the pricing round when a job was bought but its log carried no
    # verdict (archive read short / tool never ran): the diagnosis prompt says
    # so once, in place of the table, instead of looking like pricing never ran.
    lever_price_lost: str = ""
    # Consecutive job-environment failures per harness channel (ncu / timeline /
    # stall-price / lever-price), see loop/job_env.py: a tool that cannot start on
    # the image is not a reading about the tree. Counted so the log says how long
    # a channel has been blind; cleared by that channel's next good reading.
    env_broken: dict = field(default_factory=dict)
    # Set when a reset threw away teacher edits outside notes/ (a pricing round,
    # a verdict probe): the next diagnosis prompt says so once, so a teacher that
    # "fixed" a job-environment file in passing does not believe it landed.
    discarded_edits: str = ""
    # Per-iteration orchestration census: the NCU job of every evaluated
    # iteration also renders the tree's SASS contract axes (orch_census_run,
    # vendored with the pricing toolkit), recorded as IterationRecord.orch_summary.
    # A KEEP whose axes differ from the previous KEEP's is billed for the KB
    # harvest (kb_axis_debt) — the harvest channel cheap wins never had (the
    # cp.async cache-mode line shipped twice at ~4 % and was priced at 22 % later;
    # only probe fights and stuck diagnoses used to reach the harvest). Rides the
    # stall_pricing seeding; off per task via loop.toml [loop] orch_census = false.
    orch_census: bool = True
    # Single-ref contract: whether the declared reference rides the student eval job
    # (bench times it back-to-back with the student — legitimate only when the
    # comparator IS this run's ref). False when the ref is measured through a
    # separate submission tree (ref_mode tasks — blackbox runs where the ref package
    # is readable source): the bench must then time ONLY the student, and any in-job
    # comparator reading is a smuggled second reference (the retired "instrument"
    # concept) — dropped loudly, never recorded, never gated on.
    ref_rides_job: bool = True
    # Where the ref-side and student-side pricing evidence (stall ledger,
    # segment account, orchestration dossier, role exposure) is banked. A
    # blackbox run banks it beside the recon report; a source-mode run — which
    # never reconnoitres and used to have nowhere to put it, so its role and
    # dossier tables were one-sided forever — under operators/<op>/evidence/.
    evidence_dir: Path | None = None
    # Source-mode runs: a teacher-side tree (its own git repo, carrying eval/ncu_run.py
    # like the student's) the driver prices ONCE with the same pricing recipe to
    # produce the ref side of those tables. loop.toml [loop] ref_price_op.
    ref_price_dir: Path | None = None
    ref_price_backend: "DriverBackend | None" = None
    # The launch target stall_price_run.py opens the profiler window around in
    # that tree, when it is not the student contract (eval/ncu_run.py): loop.toml
    # [loop] ref_price_app = ["python3", "dsa_ref.py", "ncu"]. None = default.
    ref_price_app: list | None = None
    # PM-sampling interval (cycles) for the frozen pricing recipe on BOTH trees:
    # 8192 resolves phase structure on ~ms kernels; a sub-0.5 ms kernel starves
    # at it (too few samples -> PMTL-FAIL by construction) and needs 2048.
    # 0 = pick from the run's target reading. loop.toml [loop] ref_price_pm_interval.
    ref_price_pm_interval: int = 0
    ref_price_submit_failures: int = 0   # transient BackendErrors this session (not attempts)
    # the same, per non-representative workload of a group card (_ref_workload_pricing)
    ref_wl_price_submit_failures: dict = field(default_factory=dict)
    ref_price_note: str = ""             # this session's transient-failure note (markers carry the rest)
    # Probe files an iteration's reset would otherwise erase (probe-blocked /
    # probe-undelivered): re-written into the workspace when the next student
    # iteration starts — the meta step's own verdict-probe commit+reset would
    # sweep an untracked file away in between.
    kept_probes: dict = field(default_factory=dict)
    # ref_in_job: the same-job residual the last paired re-anchor measured for
    # the current best (best_ms, residual) — the residual the ledger gate must
    # judge on, not the cross-node absolute gap.
    paired_residual_memo: "tuple[float, float] | None" = None
    # The comparator the bench times INSIDE the student job is the declared
    # reference itself (same build, same image) — not a stand-in. The separate
    # ref tree still measures the launch baseline, the NCU signature and the
    # reconnaissance; but the same-job reading is the one verdict currency that
    # is node- and period-matched, so with this on the single-ref contract keeps
    # it and the loop judges parity by it (loop.toml [loop] ref_in_job).
    ref_in_job: bool = False
    # --equivalence-ratio: a same-job ratio (ref / student) at or above it is parity.
    equivalence_ratio: float = 1.0
    # shas whose same-job parity was re-measured and NOT confirmed: never re-bought
    # (sha -> account, with the cluster it was measured on).
    paired_memo: dict = field(default_factory=dict)
    # the account of a same-job parity stop: which sha stopped the run and the
    # samples already bought — the finish confirmation confirms THAT sha and
    # counts these samples instead of re-measuring the absolute best.
    paired_stop: dict | None = None
    # A KEEP that beats the running best by more than this fraction is re-measured
    # before its reading becomes the watermark (0 disables). One eval job is one
    # node draw: a single-job outlier that takes the watermark poisons every later
    # meta transaction — one fwd run's 0.2585 (re-measured 0.282, +9%) held
    # best for 9h and had two rebuilds judged "no gain" against a number that was
    # never real. Within-noise improvements (the common case) cost nothing extra.
    watermark_margin: float = 0.03
    # Machine awareness. Records are stamped with the machine their eval ran on
    # (backend.last_cluster), and every absolute-ms comparison — running best,
    # plateau, patch judgment, the target stop — reads only rows from the machine
    # submissions currently go to: one unchanged kernel's absolute latency moves
    # more between machines than most single-round gains (the same DSA kernel:
    # 2.1441 ms on one, 2.234 ms on another), so a mixed ledger reads a slower
    # machine as a REVERT streak and a faster one as a new best. home_cluster is
    # the machine the run started on; unstamped rows are taken to be from it. ""
    # (the default) means everything is one machine. On another machine (a resume
    # elsewhere) the target is restated through the reanchor_ref channel below
    # (_cluster_scale).
    home_cluster: str = ""
    # Phantom-gap re-anchor (the ref-side mirror of watermark_margin, which only
    # ever re-measures the STUDENT): before a gap-driven meta diagnosis, a residual
    # within this fraction of target is checked against a fresh same-period ref
    # measurement. The stop target derives from a ref measured once at launch, and
    # the H100 pool is bimodal (~±6% absolute), so a residual inside the band is as
    # likely to be anchor drift as kernel deficit — a bwd-GQA run's meta
    # chased a +3.6% residual through a 20h structural rebuild; the finish confirm
    # measured the same sha at 1.0271x. 0 disables. loop.toml [loop] reanchor_band.
    reanchor_band: float = 0.06
    # The measurement channel for the re-anchor: returns (fresh gate-window ref
    # samples in ms, the cluster they ran on) — same path the finish confirmation
    # uses. Wired by the CLI from the task's declared ref; None on a target-ms-only
    # run (no ref to measure).
    reanchor_ref: "Callable[[], tuple[list[float], str]] | None" = None
    # Prompt note set when the fresh anchor CONFIRMS the gap. Read by run_two_layer
    # and passed into every run_meta_step of the SAME meta entry (a malformed-patch
    # retry re-runs the diagnosis and must see the same re-anchored residual, not
    # fall back to the stale one), then cleared.
    reanchor_note: str | None = None
    # The launch reading the transport pairs with: the fastest-of-N gate-window
    # sample of the SAME channel reanchor_ref measures (baseline_cli keeps the
    # fastest), plus the cluster it ran on. None/"" when the target was not derived
    # from a measured baseline (--target-ms, a toml constant): a number that was
    # never node-sampled has no launch-period node conditions to transport from.
    target_anchor_ms: float | None = None
    target_anchor_cluster: str = ""
    # (best at the last confirmed-gap check, its note): a re-check with an
    # unchanged best cannot change the answer except by further drift — reuse the
    # verdict instead of re-buying ref jobs at every subsequent plateau.
    reanchor_memo: "tuple[float, str] | None" = None


# --- prompt rendering (loop/prompt.py) -------------------------------------
# Moved to loop/prompt.py (docs/driver-split-spec.md step 1); imported back so
# every name it exposed from driver is still reachable at forgekernel.loop.driver.*
from forgekernel.loop.prompt import (  # noqa: F401 -- re-export
    _oracle_binding,
    _arch_ref_binding,
    _verdict_probe_brief,
    kb_read_dir,
    _meta_prompt,
    FAMILY_ANALYSIS_DIR,
    family_analysis_path,
    _write_family_analysis,
    _is_cuda,
    _is_tilelang,
    _family_analysis_binding,
    _bindings,
    _ref_for_window,
    _window_contract_binding,
    _window_student_note,
    _precision_reading,
    _sass_precision,
    _enforce_prep_contract,
    _PREP_ERROR_PREFIX,
    _prep_structural_stop,
    _best_keep_record,
    render_prompt,
    _student_prompt,
)
# --- inner loop -------------------------------------------------------------

# A gate closure row ("M0-gate") vs an ordinary decision ("D0.7"). Gate rows always get
# the full eval — they are where a scaffold phase's deferred verification is spent.
_GATE_ROW = re.compile(r"gate", re.IGNORECASE)
# The ref's per-segment construction table travels WHOLE into the distill prompt:
# a real table (five regions, seven segments, hand-offs, precision, warp rows) is
# ~9k characters; a 4k cut fell inside the decisive region's first hand-off line.
REF_STRUCT_PROMPT_CHARS = 12000


# --- placeholder ledger --------------------------------------------------------

# Moved to loop/placeholder.py (docs/driver-split-spec.md step 3); imported back so
# every name it exposed from driver is still reachable at forgekernel.loop.driver.*
from forgekernel.loop.placeholder import (  # noqa: F401 -- re-export
    _PLACEHOLDER_TAG,
    _RESTORED_TAG,
    _DECISION_IDS,
    _TAG_ARROW,
    _PLACEHOLDER_PROSE,
    _PLACEHOLDER_PROSE_NEG,
    _placeholder_ids,
    _lineage_roots,
    _lineage_rows,
    _placeholder_ledger,
    _placeholder_items,
    _workspace_head,
    _placeholder_note,
    _placeholder_bounce,
)
def _best_ms(records, gate_window: str = GATE_WINDOW_DEFAULT) -> float | None:
    vals = [gate_ms(r, gate_window) for r in records
            if gate_ms(r, gate_window) is not None]
    return min(vals) if vals else None


def _gate_ratio(rec, gate_window: str = GATE_WINDOW_DEFAULT) -> float | None:
    """The record's same-job ratio (ref / student) in the gate window; None when
    the reference did not ride that job or that window was not measured."""
    window = gate_window_name(gate_window)
    if window == GATE_WINDOW_DEVICE:
        return rec.vs_ref_device
    if window == GATE_WINDOW_SERVER:
        return getattr(rec, "vs_ref_server", None)
    return rec.vs_ref


def _best_ratio(records, gate_window: str = GATE_WINDOW_DEFAULT) -> float | None:
    """The best same-job ratio among the measured KEEP rows, or None."""
    vals = [_gate_ratio(r, gate_window) for r in records
            if r.verdict == "KEEP" and r.correctness and _gate_ratio(r, gate_window)]
    return max(vals) if vals else None


# --- machine-aware comparisons (loop/comparisons.py) -----------------------
# Moved to loop/comparisons.py (docs/driver-split-spec.md comparisons step);
# imported back so every name it exposed from driver is still reachable at
# forgekernel.loop.driver.*
from forgekernel.loop.comparisons import (  # noqa: F401 -- re-export
    FAST_SKIP_NOISE_MULTIPLE,
    _ANCHORS_FILENAME,
    _CENSUS_LINE_RE,
    _KERNELS_PROF_KEY,
    _NCU_REGRESSION_BAND,
    _ORCH_CENSUS_CMD,
    _ORCH_CENSUS_RUNNER,
    _ORCH_PROF_KEY,
    _PROBE_LOG_TAIL_CHARS,
    _PROBE_LOG_TAIL_LINES,
    _REANCHOR_ATTEMPTS,
    _SASS_PRECISION_PROF_KEY,
    _SASS_PRECISION_RE,
    _VERDICT_PROBE_TAIL_CHARS,
    _VERDICT_PROBE_TAIL_LINES,
    _ablation_probe_feedback,
    _active_cluster,
    _anchors_path,
    _best_good_ref_here,
    _best_keep_ncu_ns,
    _best_keep_profiler,
    _check_host_bound,
    _check_profile_covers_the_kernel,
    _cluster_scale,
    _confirm_suspicious_watermark,
    _edits_outside_notes,
    _enforce_single_ref,
    _env_counts,
    _fast_skip_reason,
    _home_ref_ms,
    _keep_probe_across_reset,
    _load_anchors,
    _log,
    _ncu_mode,
    _ncu_profiler,
    _note_discarded_edits,
    _note_env_failure,
    _note_env_ok,
    _on_cluster,
    _parse_serial_census,
    _pop_kernels,
    _pop_orch_summary,
    _probe_feedback,
    _probe_label,
    _probe_log_suffix,
    _rec_cluster,
    _restore_kept_probes,
    _restore_probe_after_reset,
    _run_ablation_probe,
    _say_target_suspended,
    _target_workloads_mismatch,
    _verdict_probe_feedback,
    _without_census_lines,
)
# --- the card-owned evaluation harness (loop/harness_guard.py) ------------------------

# Moved to loop/harness_section.py (docs/driver-split-spec.md step 2); imported back so
# every name it exposed from driver is still reachable at forgekernel.loop.driver.*
from forgekernel.loop.harness_section import (  # noqa: F401 -- re-export
    _harness_hygiene,
    _harness_edit_revert,
)
def run_inner_iteration(
    ctx: LoopContext, iter_no: int, *, metric_before: float | None,
    last_good_ref: str | None = None, green: bool | None = True,
) -> IterationRecord | None:
    """One inner iteration. Returns the record, or None if the student is exhausted.

    ``green`` is the lineage's correctness state: True once the kernel has passed
    correctness since the last workspace reset, False while it is under construction,
    None when unknown (the next full eval decides). Pre-green (False), a committed
    decision is NOT evaluated — see the scaffold branch below."""
    be = ctx.backend
    entry_cluster = _active_cluster(ctx)
    _restore_kept_probes(ctx)
    _harness_hygiene(ctx, iter_no)
    prompt = _student_prompt(ctx, iter_no)
    _log(f"iter {iter_no}: spawning student agent…")
    try:
        base_sha = be.head() or ""       # the tree this iteration builds on
    except Exception:  # noqa: BLE001 — lineage is bookkeeping, never a precondition
        base_sha = ""

    record: Record | None = None
    last_probe: str = ""
    last_probe_passed = False
    last_pv, last_log = None, ""
    gate_refused = ""   # the last gate-row refusal (open placeholders), for the record
    for _ in range(ctx.max_probe_rounds):
        out = be.run_agent(prompt)
        try:
            turn = parse_student_turn(out)
        except ValueError:
            # student emitted no parseable result (prose only / truncated) — re-prompt
            # with a strict format reminder instead of aborting the loop.
            _log(f"iter {iter_no}: unparseable student output -> re-prompt")
            prompt = _student_prompt(
                ctx, iter_no,
                probe_feedback="- Your last reply had NO parseable result. End your turn with "
                               "EXACTLY one of: the record block (decision_id: / change: / "
                               "files_touched:), 'PROBE_REQUEST: <path>', or 'EXHAUSTED: <reason>' "
                               "— and nothing after it.\n")
            continue
        if isinstance(turn, Exhausted):
            _log(f"iter {iter_no}: student EXHAUSTED ({turn.reason})")
            return None
        if isinstance(turn, Record):
            if bounce := _placeholder_bounce(ctx, turn, base_sha):
                # A gate row over an open placeholder, or a placeholder landed
                # without its tag: neither is committed — the student is re-prompted
                # with the debt (see the placeholder ledger above).
                kind, why, feedback = bounce
                _log(f"iter {iter_no}: {why}")
                if kind == "gate-refused":
                    gate_refused = why
                prompt = _student_prompt(ctx, iter_no, probe_feedback=feedback)
                continue
            record = turn
            break
        if _is_timeline_request(turn.path):
            # a timeline of the student's own tree: one probe round, no verdict
            if not _timeline_on(ctx):
                prompt = _student_prompt(
                    ctx, iter_no,
                    probe_feedback="- This run has the timeline probe off (loop.toml [loop] "
                                   "timeline_probe), so `PROBE_REQUEST: timeline` is unavailable; "
                                   "send an ordinary probe or submit a record block.\n")
                continue
            be.commit("probe: timeline")
            probe_branch = f"{ctx.branch_prefix}-i{iter_no}probe"
            be.push_branch(probe_branch)
            tl = _run_timeline(ctx, be.head(), f"iter{iter_no}:student", branch=probe_branch)
            prompt = _student_prompt(ctx, iter_no, probe_feedback=_timeline_feedback(tl))
            continue
        # ProbeRequest: commit + run on GPU, feed the verdict back to the student
        last_probe, last_probe_passed = turn.path, False
        be.commit(f"probe: {turn.path}")
        probe_branch = f"{ctx.branch_prefix}-i{iter_no}probe"
        be.push_branch(probe_branch)
        log = be.submit_gpu(probe_branch, f"cmd:python3 {turn.path}")
        pv = parse_probe_verdict(log)
        last_pv, last_log = pv, log
        _log(f"iter {iter_no}: probe {turn.path} -> PROBE {_probe_label(pv)} "
             f"(max_diff={pv.diff_text}){_probe_log_suffix(pv)}")
        battle = ctx.probe_battles.setdefault(turn.path, ProbeBattle(path=turn.path))
        battle.open_rounds += 1
        if pv.passed:
            last_probe_passed = True
            fight = battle.open_rounds
            battle.open_rounds = 0  # a PASS closes the fight; re-probes start a new one
            # Bill the dearest winning fight, whole: on a tie the later fight wins
            # (its PASS certifies the probe file as it now ships). A cheaper re-win
            # leaves the billed fight's record intact — still internally coherent.
            if fight >= battle.rounds:
                battle.passed = True
                battle.rounds = fight
                battle.pass_iter = iter_no
                battle.verdict_line = f"PROBE PASS (max_diff={pv.diff_text})"
                tail = "\n".join(log.strip().splitlines()[-_PROBE_LOG_TAIL_LINES:])
                battle.log_tail = tail[-_PROBE_LOG_TAIL_CHARS:]
                if ctx.kb_tripwire_rounds and fight >= ctx.kb_tripwire_rounds:
                    _log(f"iter {iter_no}: kb-tripwire: {turn.path} passed after "
                         f"{fight} GPU rounds (>= {ctx.kb_tripwire_rounds}) — "
                         f"queued for the end-of-run KB harvest write-up")
        prompt = _student_prompt(
            ctx, iter_no,
            probe_feedback=_probe_feedback(
                turn.path, pv, log,
                on_pass="Implement the full decision and emit the record block."),
        )
    if record is None and last_probe_passed:
        # The budget counts agent turns, probe turns included, so a probe that
        # PASSes on the last turn has no turn left to deliver the decision it
        # certified. Observed on dsa_sparse runs: eight
        # rounds, r20 PASS 2/2, recorded as "could not certify", two hours of
        # kernel work reset. One delivery-only turn closes that hole — a probe
        # request on it is not honoured (no GPU), the record block is.
        _log(f"iter {iter_no}: {last_probe} passed on the last probe round — "
             f"one delivery-only turn")
        # A fresh subprocess: it needs the verdict table and log tail the
        # normal on_pass path carries, not just "it passed".
        out = be.run_agent(_student_prompt(
            ctx, iter_no,
            probe_feedback=_probe_feedback(
                last_probe, last_pv, last_log,
                on_pass=("This iteration's probe rounds are used up: **this turn can only "
                         "deliver** — implement the full decision and end with a record block. "
                         "Another PROBE_REQUEST will not be executed."))
            + "- This turn can only deliver (probe rounds used up).\n"))
        try:
            turn = parse_student_turn(out)
        except ValueError:
            turn = None
        if isinstance(turn, Record):
            if bounce := _placeholder_bounce(ctx, turn, base_sha):
                _log(f"iter {iter_no}: {bounce[1]}")
                if bounce[0] == "gate-refused":
                    gate_refused = bounce[1]
            else:
                record = turn
    if record is not None:
        # A landed decision ends every kept fight: the probe files are in the
        # student's tree now (or the student dropped them), and a keep that
        # outlived the iteration it was made for re-wrote stale text into
        # every later workspace (and every tree a later reset restored).
        ctx.kept_probes.clear()
    if record is None:
        # budget exhausted with no landed decision (probe never certified, or the
        # student never emitted a parseable decision). Objective REVERT, roll back.
        kept = _keep_probe_across_reset(ctx, last_probe) if last_probe else None
        if last_good_ref:
            be.reset(last_good_ref)
        if kept is not None:
            _restore_probe_after_reset(ctx, last_probe, kept)
        if gate_refused:
            # The student spent the iteration re-submitting a gate row over an
            # open placeholder. Say which debt blocked it — "no parseable
            # decision" would send the next iteration looking for a format bug.
            did, change = "gate-refused", "(gate row refused: placeholder ledger not clear)"
            err = gate_refused
        elif last_probe and last_probe_passed:
            # The fight was WON; only the delivery is missing. Say so — the next
            # iteration's student reads this row, and "could not certify" sent
            # the next iteration straight past a certified probe into a -17.9% commit.
            did, change = "probe-undelivered", "(probe passed; decision not delivered in budget)"
            err = (f"{last_probe} PASSED on the last probe round but the decision was "
                   f"not delivered — the probe file is kept in the workspace (uncommitted); "
                   f"the next iteration continues from that PASS, it does not re-fight it")
        elif last_probe:
            did, change = "probe-blocked", "(probe budget exhausted)"
            err = (f"{last_probe} did not pass in {ctx.max_probe_rounds} probe rounds — "
                   f"the probe file is kept in the workspace (uncommitted) so the fight "
                   f"can continue next iteration")
        else:
            did, change = "unparseable", "(no parseable decision emitted)"
            err = f"no parseable decision from the student in {ctx.max_probe_rounds} rounds"
        rec = IterationRecord(
            iter=iter_no, decision_id=did, change=change,
            correctness=False, verdict="REVERT", metric_before=metric_before,
            error=err, profiler={}, cluster=_active_cluster(ctx),
        )
        append_record(ctx.status_path, rec)
        _log(f"iter {iter_no}: {did} -> REVERT ({err})")
        return rec

    sha = be.commit(f"{record.decision_id}: {record.change[:60]}")
    if (rec := _harness_edit_revert(ctx, iter_no, record, sha, base_sha,
                                    metric_before)) is not None:
        return rec
    for m in _PLACEHOLDER_TAG.finditer(record.change or ""):
        _log(f"iter {iter_no}: placeholder opened — [{m.group(1).strip()}] "
             f"(gate rows are refused until [RESTORED: …] lands)")
    for m in _RESTORED_TAG.finditer(record.change or ""):
        _log(f"iter {iter_no}: placeholder restored — [{m.group(1).strip()}]")
    declared = _declared_construction(record.change) if green is None else None
    if (green is False or declared) and not _GATE_ROW.search(record.decision_id):
        # SCAFFOLD (POLICY §3 build-phase semantics, executed): the kernel has not passed
        # correctness in this lineage, so a full eval of this sub-step can only measure
        # "still under construction" — the verdict is decided before the job runs, NCU
        # yields nothing on a broken bench, and the REVERT it would record is
        # bookkeeping, not information. The decision's own [VERIFY:] declaration backs
        # the skip: "none" is never per-step-measurable, and the ncu-* launch facts are
        # all harvested by the single full eval the gate row triggers. The commit still
        # lands (the sha is a resume checkpoint); the record says NOT-evaluated
        # (NEUTRAL, correctness None) rather than pretending a verdict.
        #
        # An UNKNOWN lineage (green None: fresh workspace, or a checkpoint no record
        # owns) is normally decided by its first full eval — a source-mode run that starts
        # from a working kernel must not lose its first readings to scaffolding. But a
        # delivery whose change OPENS with the prompt's own construction declaration
        # (`part k of n`, k < n; `[PLACEHOLDER: …]`) has told the driver the tree is
        # mid-build: evaluating it buys a REVERT that says "under construction" (the
        # observed: a GDN run paid two such evals, the second right after a budget resume).
        from forgekernel.milestone import parse_verify_tags
        tag = parse_verify_tags(_read_milestone(ctx)).get(record.decision_id, "none")
        rec = IterationRecord(
            iter=iter_no, decision_id=record.decision_id, change=record.change,
            correctness=None, verdict="NEUTRAL", metric_before=metric_before, sha=sha,
            cluster=_active_cluster(ctx), base_sha=base_sha,
        )
        append_record(ctx.status_path, rec)
        why = (f"declared construction: {declared}; lineage state unknown"
               if declared else "pre-green")
        _log(f"iter {iter_no}: {record.decision_id} SCAFFOLD ({why}, [VERIFY: {tag}] "
             f"-> full eval deferred to the gate row)")
        if ctx.declared_gate_close and (m := _CLOSES_TAG.search(record.change or "")):
            _log(f"iter {iter_no}: [CLOSES: {_closes_gate_id(m)}] not honoured — a "
                 f"scaffold row has no eval to close on; the gate row is the student's")
        return rec
    if (base := _unchanged_tree_reading(ctx, record, sha, base_sha)) is not None:
        # Nothing the eval reads changed since a tree this trajectory already
        # measured: archive the row on that reading instead of re-measuring it.
        rec = _archive_reused_reading(ctx, iter_no, record, sha, base_sha, base,
                                      metric_before)
        return _declared_gate_close(ctx, record, rec, iter_no) or rec
    branch = f"{ctx.branch_prefix}-i{iter_no}"
    be.push_branch(branch)
    _log(f"iter {iter_no}: decision {record.decision_id} committed; evaluating on GPU…")
    ev = parse_eval(be.submit_gpu(branch, "all"))
    ev = _enforce_single_ref(ctx, ev)
    ev = _enforce_prep_contract(ctx, ev)
    precision = _precision_reading(ctx, ev, iter_no)
    cluster = _active_cluster(ctx)
    if cluster != entry_cluster:
        # The eval failed over mid-iteration: the best it is about to be judged
        # against was read on another cluster. Re-read it from this cluster's
        # rows (usually nothing yet) — the watermark check and the record's
        # metric_before must not compare across clusters.
        metric_before = _best_ms(_on_cluster(load_records(ctx.status_path), ctx, cluster),
                                 ctx.gate_window)
        _log(f"iter {iter_no}: eval ran on {cluster!r} (the iteration started on "
             f"{entry_cluster!r}) — best re-read from that machine's rows: "
             f"{metric_before}")
    # NCU on EVERY committed iteration (KEEP and REVERT) — negative iterations need
    # measured hardware truth too, so the meta agent never has to guess a diagnosis.
    # NCU is best-effort on the GPU side: a PASSING kernel with an empty profiler means
    # NCU itself failed (e.g. a lost job), so retry; a correctness-broken kernel yields
    # no NCU legitimately (the bench crashes), so don't waste GPU retrying that.
    skip = _fast_skip_reason(ctx, ev, metric_before)
    if skip:
        # A candidate this far behind the bar cannot take the watermark, so its
        # NCU table would only serve a diagnosis of a row that loses anyway. It
        # still gets a KEEP/REVERT verdict and a record -- what it does not get is
        # a second GPU pass (see FAST_SKIP_NOISE_MULTIPLE).
        _log(f"iter {iter_no}: fast-skip: {skip}")
        profiler = {}
    else:
        profiler = _ncu_profiler(ctx, branch, correctness=ev.correctness)
    orch_summary = _pop_orch_summary(profiler)
    kernels = _pop_kernels(profiler)
    serial_raw = _parse_serial_census(profiler)
    sass_mp = profiler.pop(_SASS_PRECISION_PROF_KEY, None) if profiler else None
    if sass := _sass_precision(ctx, sass_mp or (orch_summary or {}).get("mma_precision"),
                               iter_no):
        precision = {**precision, "sass": sass}

    if not ev.correctness:
        # relaxed-revert keeps perf regressions, but a correctness break is a hard REVERT.
        # Carry the bench's own verdict text into the record: the student and the meta
        # diagnoser debug from this string, and "correctness FAIL" alone once hid a
        # harness-level rejection (forbidden-op scan) behind 14h of phantom-bug chasing.
        detail = f"correctness FAIL: {ev.error}" if ev.error else "correctness FAIL"
        _log(f"iter {iter_no}: {record.decision_id} REVERT ({detail})")
        if last_good_ref:
            be.reset(last_good_ref)
        rec = IterationRecord(
            iter=iter_no, decision_id=record.decision_id, change=record.change,
            correctness=False, verdict="REVERT", metric_before=metric_before,
            error=detail, profiler=profiler, cluster=cluster,
            orch_summary=orch_summary, base_sha=base_sha, kernels=kernels,
            prep=ev.prep or {}, precision=precision,
        )
        append_record(ctx.status_path, rec)
        return rec

    ev = _confirm_suspicious_watermark(ctx, be, branch, ev, metric_before,
                                       profiler=profiler)
    rec = IterationRecord(
        iter=iter_no, decision_id=record.decision_id, change=record.change,
        correctness=True, verdict="KEEP", bwd_median_ms=ev.bwd_median_ms, mfu=ev.mfu,
        ref_median_ms=ev.ref_median_ms, vs_ref=ev.vs_ref,
        device_median_ms=ev.device_median_ms, ref_device_ms=ev.ref_device_ms,
        vs_ref_device=ev.vs_ref_device,
        server_median_ms=getattr(ev, "server_median_ms", None),
        ref_server_ms=getattr(ev, "ref_server_ms", None),
        vs_ref_server=getattr(ev, "vs_ref_server", None),
        server_qps=getattr(ev, "server_qps", None),
        host_enqueue_ms=ev.host_enqueue_ms, host_bound_ratio=ev.host_bound_ratio,
        ref_host_bound_ratio=ev.ref_host_bound_ratio,
        device_window_basis=ev.device_window_basis,
        metric_before=metric_before, profiler=profiler, sha=sha, cluster=cluster,
        orch_summary=orch_summary, base_sha=base_sha, kernels=kernels,
        prep=ev.prep or {}, n_workloads=ev.n_workloads, workloads=ev.workloads,
        precision=precision,
    )
    append_record(ctx.status_path, rec)
    _log_axis_change(ctx, rec)
    _check_host_bound(ctx, iter_no, ev)
    # NCU reports device time, so the reconciliation compares against the device
    # window when the bench measured one — against the e2e window a host-heavy
    # call shrinks the kernel share and false-fires the alarm.
    if (warn := _check_profile_covers_the_kernel(
            profiler, ev.device_median_ms or ev.bwd_median_ms, ev)):
        _log(f"iter {iter_no}: WARNING {warn}")
    _log(f"iter {iter_no}: {record.decision_id} KEEP  "
         f"metric={gate_ms(ev, ctx.gate_window)} ({ctx.gate_window}) mfu={ev.mfu}"
         + (f" vs_ref={ev.vs_ref}" if ev.vs_ref is not None else "")
         + (f" vs_ref_device={ev.vs_ref_device}" if ev.vs_ref_device is not None else "")
         + (f"  (best was {metric_before})" if metric_before else "")
         + (f"  [cluster {cluster}]" if cluster and cluster != ctx.home_cluster else "")
         + (f"  {precision_gate.describe(precision)}" if precision else ""))
    if ev.workloads:
        _log(f"iter {iter_no}: gate = geomean over {ev.n_workloads} workloads: "
             + describe_workload_readings(ev.workloads))
    if _too_few_workloads(ctx, ev) and str(ctx.status_path) not in _SINGLE_WORKLOAD_WARNED:
        _SINGLE_WORKLOAD_WARNED.add(str(ctx.status_path))
        _log(f"iter {iter_no}: WARNING the gate reading covers {ev.n_workloads} workload(s), "
             f"not {PERF_WORKLOADS} — a kernel kept on it can be tuned to that input set "
             f"(timing.measure_windows_workloads)")
    new_ms = gate_ms(ev, ctx.gate_window)
    ctx.sass_serial_note = _sass_serial_status(ctx, serial_raw, exclude_iter=iter_no)
    if new_ms is not None and (metric_before is None or new_ms < metric_before):
        # the tree the milestone's cited prices were read on just changed under
        # them: re-price those citations here, on the tree they now apply to
        try:
            _reprice_cited_levers(ctx, rec)
        except (BackendError, OSError, ValueError) as e:
            _log(f"iter {iter_no}: price-check skipped ({e})")
    if _timeline_on(ctx):
        # a delivery that closes its phase is read like the gate row it files
        is_gate = bool(_GATE_ROW.search(rec.decision_id or "")) or bool(
            ctx.declared_gate_close and _CLOSES_TAG.search(record.change or ""))
        is_best = new_ms is not None and (metric_before is None or new_ms < metric_before)
        why = "gate row" if is_gate else ("new best" if is_best and _timeline_due(ctx, new_ms) else "")
        if why:
            try:
                _run_timeline(ctx, rec.sha, f"iter{iter_no}:{why}")
            except (BackendError, OSError, ValueError) as e:
                _log(f"iter {iter_no}: timeline skipped ({e})")
        elif is_best:
            _log(f"iter {iter_no}: new best within {ctx.reprice_min_gain_pct:g}% of the last "
                 f"timeline's tree — not re-read")
    return _declared_gate_close(ctx, record, rec, iter_no) or rec


# --- rows that buy no eval ------------------------------------------------------
#
# Two shapes of row carry no new information for a GPU job to measure:
#   * a commit that changes nothing outside notes/ on top of a tree this
#     trajectory already evaluated — the gate acceptance row "submitted on its
#     own with no code change", a reopened decision re-filed as built, a
#     pricing-only row. Its eval re-reads the same bytes; the reading it buys is
#     a second noise draw of one tree, and best / patch judgment select on the
#     maximum of the draws, so a lucky one moves "the best" for nothing.
#   * the gate row of a delivery that declares it closes its phase
#     (`[CLOSES: Mk-gate]`): the delivery's own full eval already read the tree
#     the gate accepts, so a separate gate turn would only re-read it.
# Both rows still land in the ledger as their own rows (the student's ledger,
# phase close, the same-stage gate check and the reopened-M0 audit read them by
# decision id); they carry the evaluated row's reading and say so (reading_of).

# Paths a commit may touch and still be the same tree for the eval: the
# student's probe scripts. Everything else (the kernel, any other file the job
# ships) counts as a change.
_READING_NEUTRAL_PREFIXES = ("notes/",)
_REUSE_MODES = ("all", "gate", "off")
# `[CLOSES: M2-gate]` on a delivery: this tree completes phase M2 (see
# LoopContext.declared_gate_close). "M2 gate" / "M2-Gate" spellings are read too.
_CLOSES_TAG = re.compile(r"\[CLOSES:\s*(M\d+)\s*[- ]?\s*gate\s*\]", re.I)


def _closes_gate_id(m: re.Match) -> str:
    return f"{m.group(1).upper()}-gate"


def _unchanged_tree_reading(ctx: LoopContext, record: Record, sha: str,
                            base_sha: str) -> IterationRecord | None:
    """The evaluated KEEP whose reading ``sha`` may carry, or None (evaluate).

    Applies when the commit changes nothing outside notes/ relative to the tree
    the iteration built on (``base_sha``), and that tree has a KEEP on file
    that passed correctness and read the gate window, on the cluster the run is
    submitting to now. A backend that cannot list a commit's changes (no
    ``commit_paths``) always evaluates."""
    mode = str(ctx.reuse_unchanged_reading or "off").lower()
    if mode not in _REUSE_MODES or mode == "off" or not sha or not base_sha or sha == base_sha:
        return None
    if mode == "gate" and not _GATE_ROW.search(record.decision_id or ""):
        return None
    lister = getattr(ctx.backend, "commit_paths", None)
    if lister is None:
        return None
    try:
        paths = lister(base_sha, sha)
    except Exception:  # noqa: BLE001 — unknown means evaluate, never a crash
        return None
    if paths is None or any(not p.startswith(_READING_NEUTRAL_PREFIXES) for p in paths):
        return None
    try:
        records = load_records(ctx.status_path)
    except (OSError, ValueError):
        return None
    for r in reversed(records):
        if (r.sha == base_sha and r.verdict == "KEEP" and r.correctness
                and gate_ms(r, ctx.gate_window) is not None):
            return r if _rec_cluster(r, ctx) == _active_cluster(ctx) else None
    return None


def _archive_reused_reading(ctx: LoopContext, iter_no: int, record: Record, sha: str,
                            base_sha: str, base: IterationRecord,
                            metric_before: float | None) -> IterationRecord:
    """Land ``record`` as a KEEP carrying ``base``'s reading (no GPU job)."""
    origin = base.reading_of if base.reading_of is not None else base.iter
    rec = replace(base, iter=iter_no, decision_id=record.decision_id, change=record.change,
                  metric_before=metric_before, sha=sha, base_sha=base_sha, error="",
                  reading_of=origin)
    append_record(ctx.status_path, rec)
    if base.sha in ctx.paired_memo:
        # the same tree's same-job parity was already re-measured and refused
        ctx.paired_memo[sha] = ctx.paired_memo[base.sha]
    _log(f"iter {iter_no}: {record.decision_id} changes nothing outside notes/ on "
         f"{base_sha[:10]} — KEEP on the reading of iter {origin} "
         f"(metric={gate_ms(rec, ctx.gate_window)} {ctx.gate_window}); no eval "
         f"([loop] reuse_unchanged_reading={ctx.reuse_unchanged_reading})")
    if _timeline_on(ctx) and (_GATE_ROW.search(record.decision_id or "") or (
            ctx.declared_gate_close and _CLOSES_TAG.search(record.change or ""))):
        # the gate row banks a table of the tree it accepts; on an unchanged
        # tree that is the evaluated row's table when one is banked
        try:
            _run_timeline(ctx, base.sha, f"iter{iter_no}:gate row")
        except (BackendError, OSError, ValueError) as e:
            _log(f"iter {iter_no}: timeline skipped ({e})")
    return rec


def _declared_gate_close(ctx: LoopContext, record: Record, rec: IterationRecord,
                         iter_no: int) -> IterationRecord | None:
    """The Mk-gate row a `[CLOSES: Mk-gate]` delivery files, archived on the
    delivery's tree and reading; None when the tag is absent or not honoured
    (the student then submits the gate row as before).

    Holds the gate row's own preconditions: the delivery KEEPs on a measured
    reading, it is not an unfinished `part k of n` step, and the placeholder
    ledger on its tree is clear (a gate row is refused over an open one)."""
    if not ctx.declared_gate_close:
        return None
    m = _CLOSES_TAG.search(record.change or "")
    if not m:
        return None
    gid = _closes_gate_id(m)
    why = ""
    if _GATE_ROW.search(record.decision_id or ""):
        why = "the row is a gate row itself"
    elif rec.verdict != "KEEP" or not rec.correctness or gate_ms(rec, ctx.gate_window) is None:
        why = "the delivery did not KEEP on a measured reading"
    elif (d := _declared_construction(record.change)) or _PLACEHOLDER_TAG.search(record.change or ""):
        why = f"the delivery declares construction ({d or 'a placeholder'})"
    else:
        open_ = _placeholder_ledger(load_records(ctx.status_path), rec.sha)
        if open_:
            why = f"open placeholder(s) {_placeholder_items(open_)}"
    if why:
        _log(f"iter {iter_no}: [CLOSES: {gid}] not honoured — {why}; the gate row is "
             f"the student's to submit")
        return None
    origin = rec.reading_of if rec.reading_of is not None else rec.iter
    records = load_records(ctx.status_path)
    gate = replace(
        rec, iter=iter_no + 1, decision_id=gid,
        change=(f"{gid} closed by the driver on the delivery of iter {iter_no} "
                f"({record.decision_id}, [CLOSES: {gid}]): same tree {rec.sha[:10]}, "
                f"reading of iter {origin} — no separate gate turn or eval"),
        metric_before=_best_ms(_on_cluster(records, ctx), ctx.gate_window),
        base_sha=rec.sha, reading_of=origin)
    append_record(ctx.status_path, gate)
    _log(f"iter {iter_no}: {record.decision_id} declares [CLOSES: {gid}] — {gid} archived "
         f"as iter {iter_no + 1} on {rec.sha[:10]} (reading of iter {origin})")
    return gate


def _sass_serial_status(ctx: LoopContext, serial_raw: dict,
                        exclude_iter: int | None = None) -> str:
    """The standing compiler-fallback note, or '' when the chains are pipelined.

    ``serial_raw`` is the census's wgmma fields for THIS KEEP (the ordinates the
    census printed). Reads the previous KEEP's ratio off the trajectory so a
    change that *introduced* serialization reads as a regression ("pipelined ->
    serialized") rather than a standing state — the thing a GDN run once
    could not see and forbade a correct direction over.

    ``exclude_iter`` is THIS iteration: its record is already on status.jsonl
    when the note is written, and reading it back as "the previous KEEP" made
    every verdict a standing state (prev == cur), never a regression.
    """
    from forgekernel.skills.sass_census import serialization_verdict
    if not serial_raw:
        return ctx.sass_serial_note  # census absent this round: leave the last note
    # the previous KEEP's ratio (the status.jsonl rows, gate-window agnostic)
    prev = None
    try:
        for r in reversed(load_records(ctx.status_path)):
            if exclude_iter is not None and getattr(r, "iter", None) == exclude_iter:
                continue
            raw = getattr(r, "profiler", None) or {}
            if raw.get("wgmma_serial_ratio") is not None:
                prev = raw
                break
    except Exception:  # noqa: BLE001 — a note, never a crash
        prev = None
    v = serialization_verdict(prev, serial_raw)
    return v if v is not None else ""


def _paired_ratio(rec, gate_window: str) -> float | None:
    """The same-job ref/student ratio of a record in the gate window (> 1 =
    student faster), or None when the job carried no ref reading."""
    if gate_window == "device":
        ratio, ms, ref = rec.vs_ref_device, rec.device_median_ms, rec.ref_device_ms
    else:
        ratio, ms, ref = rec.vs_ref, rec.bwd_median_ms, rec.ref_median_ms
    if ratio is None and ms and ref:
        ratio = ref / ms
    return ratio





def _persist_paired(ctx: LoopContext, record: dict) -> None:
    """Append a same-job parity account to <op_dir>/paired_verdict.jsonl."""
    try:
        path = Path(ctx.milestone).parent / "paired_verdict.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        _log(f"paired: could not persist the account ({e})")


# Two same-job ratios that disagree by more than this represent no single
# period (a node-mode flip mid-run, a killed-and-resumed job): the same
# floor the docstrings quote for a same-job ratio, not the 10% band the
# absolute path allows two different nodes.
_PAIRED_SPREAD_MAX = 0.05


def _admit_paired(conf: dict | None) -> "tuple[float | None, str | None]":
    """(median same-job ratio, None) when the re-measure is admissible, else
    (None, why). Admission: >= 2 usable ratios, spread within
    _PAIRED_SPREAD_MAX — one draw is still one draw, and two draws that
    disagree by more than the noise floor decide nothing either way."""
    if not conf:
        return None, "no re-measure came back"
    ratios = [s["ratio"] for s in conf.get("jobs", []) if s.get("ratio") is not None]
    if len(ratios) < 2:
        return None, f"{len(ratios)} usable same-job ratio(s), need >= 2"
    spread = (max(ratios) - min(ratios)) / min(ratios)
    if spread > _PAIRED_SPREAD_MAX:
        return None, (f"ratios {sorted(round(r, 4) for r in ratios)} spread {spread:.1%} "
                      f"> {_PAIRED_SPREAD_MAX:.0%} (no single period)")
    return statistics.median(ratios), None


def _paired_parity_stop(ctx: LoopContext, rec, iter_no: int) -> bool:
    """Same-job parity as a stop: a KEEP whose in-job ratio meets the
    equivalence ratio is re-measured (2 jobs, same sha) and stops the run when
    the median same-job ratio still meets it.

    Only on ref_in_job runs. The absolute check above it compares one period's
    reading against a target derived from another period's draws — nodes vary by
    ~6% — and cannot decide anything inside that band;
    the same-job ratio is measured on one card in one minute and can. One draw
    is still one draw (same-job ratios carry a 1-5% floor), so the claim is
    repeated before it ends the run, and a sha whose re-measure disagreed is
    memoised so a later plateau does not re-buy the same two jobs.
    """
    if not ctx.ref_in_job or rec.verdict != "KEEP" or not rec.sha:
        return False
    ratio = _paired_ratio(rec, ctx.gate_window)
    eq = ctx.equivalence_ratio
    if ratio is None or ratio < eq or rec.sha in ctx.paired_memo:
        return False
    _log(f"iter {iter_no}: same-job ratio {ratio:.4f} >= {eq} ({ctx.gate_window} "
         f"window) — parity on paper; re-measuring {rec.sha[:10]} (2 jobs) before stopping")
    try:
        conf = run_best_confirmation(ctx, jobs=2, max_extra=1, ref=rec.sha)
    except Exception as e:  # noqa: BLE001 — a starved queue must not kill the run
        conf = None
        _log(f"iter {iter_no}: paired re-measure failed ({e})")
    med, why = _admit_paired(conf)
    record = {"ts": ctx.now(), "iter": iter_no, "sha": rec.sha, "window": ctx.gate_window,
              "cluster": _active_cluster(ctx),
              "ratio": round(ratio, 4), "equivalence_ratio": eq,
              "jobs": [s.get("job") for s in (conf or {}).get("jobs", [])],
              "samples": list((conf or {}).get("jobs", [])),
              "median_ratio": med}
    if med is None:
        # not admissible: neither a stop nor a verdict — not memoised either,
        # so a later plateau may try again under better conditions
        record["verdict"] = "remeasure_unavailable"
        record["why"] = why
        _persist_paired(ctx, record)
        _log(f"iter {iter_no}: paired re-measure inadmissible ({why}) — no verdict; continuing")
        return False
    if med >= eq:
        record["verdict"] = "target"
        ctx.paired_stop = record
        _persist_paired(ctx, record)
        _log(f"iter {iter_no}: same-job median ratio {med:.4f} over "
             f"{len(record['jobs'])} job(s) >= {eq} — parity confirmed on the same "
             f"card; stopping as target")
        return True
    record["verdict"] = "unconfirmed"
    ctx.paired_memo[rec.sha] = record
    _persist_paired(ctx, record)
    _log(f"iter {iter_no}: paired parity NOT confirmed (median ratio {med:.4f} < {eq}) "
         f"— the single reading was a lucky draw; continuing")
    return False


def run_inner_loop(ctx: LoopContext, start_iter: int, *,
                   segment_records_before: int | None = None,
                   gate_contract: dict | None = None,
                   initial_green: bool | None = True,
                   initial_good_ref: str | None = None,
                   audit_on_green: bool = False,
                   audit_on_gate: str | None = None) -> str:
    """Iterate until a stop condition. Returns the reason.

    ``initial_good_ref``: the rollback anchor a green segment enters on when it is
    NOT its entry HEAD — a resumed lineage whose HEAD no record verified (a
    delivery committed before the process died mid-eval, a kept probe on top of
    the last KEEP). The anchor is the last KEEP then, HEAD stays the tree the next
    delivery builds on and its first eval decides it.

    ``initial_green`` seeds the lineage's correctness state (see run_inner_iteration):
    run_two_layer passes None for a fresh workspace and the resumed-to checkpoint's
    state after a meta reset; the default True preserves evaluate-everything behavior
    for direct callers. Green flips True on the first correctness PASS and never flips
    back within a segment — a post-green break is a real regression and stays a hard
    REVERT, not a return to scaffolding.

    ``audit_on_green``: stop with reason "first_green" the first time this segment
    turns green (= its M0 gate lands), so run_two_layer can run the forced M0 audit
    before optimization stacks a whole segment on an unverified foundation. Target
    still wins: a kernel that is already at target has nothing left to audit for.

    ``audit_on_gate``: the same stop, keyed on a gate row instead of the green
    flip — for a segment that ENTERS green (an in-place reopen keeps the best
    tree) but redoes foundation decisions on it: its M0 close is the first
    KEEP of that gate, not a correctness transition, and without this the
    rebuilt foundation would only be read at the next plateau."""
    iter_no = start_iter
    green = initial_green
    entered_not_green = initial_green is not True
    # Rollback anchor for hard REVERTs (run_inner_iteration resets the workspace to
    # it on a correctness break / dead probe budget). A green segment enters on a
    # verified tree, so its entry HEAD is the first anchor; scaffolding/unknown
    # segments start with none — construction has no verified state to restore, and
    # resetting mid-build would erase progress, not recover any. Every KEEP advances
    # the anchor. This assignment IS the revert contract's production half: it was
    # never written, so the reset branch was dead code and REVERT was a ledger row
    # while the broken commit stayed in the tree as the next iteration's foundation.
    last_good_ref: str | None = (initial_good_ref if initial_good_ref is not None
                                 else (ctx.backend.head() if initial_green else None))
    consecutive_reverts = 0
    # iter numbers taken by rows the driver archived without a turn (a gate a
    # delivery closed): they keep status rows one-iter-per-row, but they are not
    # iterations the max_iters budget pays for
    free_rows = 0
    while True:
        if ctx.now() >= ctx.deadline_epoch:
            return "budget"
        if iter_no - start_iter - free_rows >= ctx.max_iters:
            return "max_iters"
        records = load_records(ctx.status_path)
        try:
            rec = run_inner_iteration(
                ctx, iter_no,
                metric_before=_best_ms(_on_cluster(records, ctx), ctx.gate_window),
                last_good_ref=last_good_ref,
                green=green,
            )
        except AgentError as e:
            _log(f"agent failed unrecoverably: {e}")
            return "agent_failed"
        except BackendError as e:
            _log(f"backend failed unrecoverably: {e}")
            return "backend_error"
        except Exception as e:  # unattended-run safety net: never crash, stop cleanly
            _log(f"unexpected error, stopping cleanly: {e!r}")
            return "error"
        if rec is None:
            return "exhausted"
        if rec.iter > iter_no:
            # the delivery closed its gate: the iteration filed two rows, and the
            # one returned (and checked below) is the gate row it closed
            free_rows += rec.iter - iter_no
            iter_no = rec.iter
        if ctx.phase_close and rec.decision_id == ctx.phase_close["gate"]:
            if rec.verdict == "KEEP":
                # The closed phase's gate row is on file and passed: the notice
                # described the walk up to it, and the gate's own lines judged it.
                _log(f"iter {iter_no}: {rec.decision_id} submitted (KEEP) — "
                     f"phase-close notice for {ctx.phase_close['phase']} retired")
                ctx.phase_close = None
            elif ctx.phase_close.get("stage") != "gate-failed":
                # The gate row failed: the phase is open again, but the table that
                # closed it still describes this best tree. Without a standing
                # note the student walks straight back into the null knobs it
                # priced (the iterations the close was meant to save), and the
                # next meta step re-asks the question on the same table. Keep the
                # "do not re-walk" half until the tree moves (a KEEP) or a
                # structural patch replaces the premise.
                ctx.phase_close["stage"] = "gate-failed"
                ctx.phase_close["note"] = _phase_priced_note(ctx.phase_close, rec.verdict)
                _log(f"iter {iter_no}: {rec.decision_id} submitted ({rec.verdict}) — "
                     f"{ctx.phase_close['phase']} stays open; its levers stay priced null "
                     f"on this best tree (notice reduced to 'do not re-walk')")
        elif (ctx.phase_close and ctx.phase_close.get("stage") == "gate-failed"
                and rec.verdict == "KEEP"):
            _log(f"iter {iter_no}: {rec.decision_id} KEEP moved the tree — the priced-null "
                 f"notice for {ctx.phase_close['phase']} retired")
            ctx.phase_close = None
        if stop := _prep_structural_stop(load_records(ctx.status_path)):
            # Same family as gate_unmeasured below: the bench is the card's
            # frozen file, so a prep block it never reports — or a prep
            # mismatch that repeats verbatim — does not heal with the next
            # student edit. Left as per-iteration REVERTs it would run
            # three-REVERT stuck, a meta diagnosis, and repeat to the budget.
            _log(f"iter {iter_no}: {rec.error} — structural, stopping ({stop}); fix "
                 f"eval/bench.py / window.toml, not the kernel")
            return stop
        if (gate_window_name(ctx.gate_window) == "device" and rec.verdict == "KEEP"
                and rec.bwd_median_ms is not None and rec.device_median_ms is None):
            # Structural bench-contract violation: the bench timed the e2e window
            # but not the device window this run gates on — that never heals on
            # its own, and letting it run means every KEEP carries no gate
            # reading, so target is never reached, plateau never fires, and the
            # loop grinds to its wall-clock budget with nothing steering it.
            # Checked HERE, after the record is appended, so the trajectory and
            # the pushed commit agree on resume; and only for a KEEP that DID
            # measure e2e — a wholly missing perf block (job killed mid-eval) is
            # transient and stays the tolerated no-metric row it always was, in
            # both windows. Its own stop reason, because "backend_error" reads
            # as GPU flake and sends the operator restarting instead of fixing
            # eval/bench.py (see src/forgekernel/card_template/eval/timing.py).
            _log(f"iter {iter_no}: KEEP measured bwd_median_ms but no "
                 f"device_median_ms while gate_window=device — the bench does not "
                 f"time the gate window; stopping (fix eval/bench.py, not the GPU)")
            return "gate_unmeasured"
        if rec.correctness is not None:  # scaffold NEUTRAL rows carry no verdict on green
            green = True if rec.correctness else (False if green is None else green)
        if rec.verdict == "KEEP" and rec.sha:
            last_good_ref = rec.sha  # the anchor follows the last verified state
        if rec.verdict == "REVERT":
            consecutive_reverts += 1
            flag = constant_output_flag(load_records(ctx.status_path))
            if flag:
                # the diagnosis reaches student and meta through digest["flags"];
                # this line is for the human reading the log at 2am
                _log(f"iter {iter_no}: CONSTANT-OUTPUT flag — bit-identical diff "
                     f"signature across iters {flag['iters']}; the scored outputs "
                     f"are not responding to code changes (see digest flags)")
            if consecutive_reverts >= ctx.max_consecutive_reverts:
                return "stuck"
        else:
            consecutive_reverts = 0
        records = load_records(ctx.status_path)
        best = _best_ms(_on_cluster(records, ctx), ctx.gate_window)
        if best is not None and (
                wl_why := _target_workloads_mismatch(ctx, _on_cluster(records, ctx))):
            # the cluster scale's in-job readings divide the same mismatched pair,
            # so neither the absolute hit nor its restatement is read at all;
            # the same-job parity stop below keeps its ratio currency
            _say_target_suspended(ctx, f"iter {iter_no}", wl_why)
        elif best is not None:
            scale = _cluster_scale(ctx)
            if scale is None:
                if getattr(ctx, "_scale_suspended_on", None) != _active_cluster(ctx):
                    ctx._scale_suspended_on = _active_cluster(ctx)
                    _log(f"iter {iter_no}: target stop suspended on cluster "
                         f"{_active_cluster(ctx)!r} — the reference has no reading "
                         f"there, so the home target cannot be restated")
            elif best <= _resolve_target(ctx) * scale and not ctx.ref_in_job:
                # On a ref_in_job run this absolute comparison is a cross-node
                # number that must not be reasoned from: the
                # target derives from the reference tree's node while the
                # student reads on its own node (one self-evolution run's
                # reference itself read 0.278 ms on one and 0.249 ms on the
                # other). There `_paired_parity_stop` below is the only stop
                # currency, and no fresh-ref jobs are bought for the absolute hit.
                if scale != 1.0:
                    _log(f"iter {iter_no}: target reached on cluster "
                         f"{_active_cluster(ctx)!r} against the home target scaled "
                         f"x{scale:.4f} ({_resolve_target(ctx) * scale:.4f} ms)")
                # An absolute hit is a claim about two readings drawn hours apart
                # on a bimodal cluster, so it buys a same-period check rather than
                # a stop. False (a node draw) keeps the run climbing with the
                # residual threaded into the next diagnosis; None means the check
                # itself was unavailable, and a run that cannot check must still
                # be able to finish.
                confirmed = _confirm_apparent_target(ctx, best, iter_no)
                if confirmed is not False:
                    if confirmed is None:
                        _log(f"iter {iter_no}: target stop NOT verified same-period "
                             f"(no usable re-measurement) — stopping on the launch "
                             f"anchor alone; treat the ratio as unconfirmed")
                    return "target"
                # a node draw, not a win: fall through and keep climbing
        if _paired_parity_stop(ctx, rec, iter_no):
            return "target"
        if segment_records_before is not None:
            # Post-patch segment: hold each gate row against the previous
            # segment's same gate. Checked before first_green so a rebuild whose
            # M0 already lost stops here instead of walking M1 to a plateau.
            msg = _gate_regression(records, rec, segment_records_before,
                                   gate_contract or {}, ctx.gate_window,
                                   voided=ctx.voided_ranges)
            if msg:
                _log(f"iter {iter_no}: gate regression — {msg}")
                return "gate_regression"
        if audit_on_green and entered_not_green and green is True:
            return "first_green"
        if audit_on_gate and rec.verdict == "KEEP" and rec.decision_id == audit_on_gate:
            return "first_green"
        # plateau is scoped to the post-resume SEGMENT (records since start_iter), not the
        # whole history. Otherwise a meta patch's first non-improving iteration re-fires the
        # pre-meta plateau instantly, so each patch gets only ONE iteration to prove itself
        # and meta intervenes every iteration. Segment-scoping gives each patch a real window.
        if detect_plateau(_on_cluster(records[start_iter:], ctx),
                          window=ctx.plateau_window, gate_window=ctx.gate_window):
            # A standing plateau reading with a declared package in flight is not a
            # stop: the package's measured intermediates are exactly the rows that
            # read as "no new best" (its payoff lands at part n), so an n-step
            # package all but guarantees one mid-package plateau — and stopping here
            # burns a meta step plus a diagnose round on construction the judgment
            # lane already knows to hold (_judge_patch HOLDs this same declaration).
            # The stop lane defers to the same audited plea under the same forfeit
            # rules: package finished / moved on / over its KEEP budget / stale
            # trail flips _open_package_tail to None, and the standing plateau
            # routes to meta on the very next reading. The plea buys continuation
            # only, never exemption — the eventual judgment still prices the whole
            # segment, and budget/max_iters walls above are untouched.
            pkg = _open_package_tail(records[start_iter:])
            if pkg is None:
                return "plateau"
            k, n = pkg
            _log(f"iter {iter_no}: plateau reading with a declared package in flight "
                 f"(part {k} of {n} is the segment's latest delivery) -> continuing "
                 f"construction instead of stopping")
        iter_no += 1


# --- meta loop --------------------------------------------------------------

@dataclass
class MetaResult:
    diagnosis: str
    patch_class: str
    changed_entries: list[str]
    resume_after: str | None
    invalidated: list[str]
    certified: bool | None = None
    milestone_violations: list[str] = field(default_factory=list)
    recipe: str | None = None  # a just-certified KB recipe path (G3 branch)
    gate_contract: dict = field(default_factory=dict)
    # In-place reopen plan (structural / split): the dependency closure of the
    # changed entries, and what stays built. Empty for additive / kb-example.
    reopened: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    # G5 only: the in-segment reordering the diagnose prompt promises the
    # student ("KB recipe + student intra-segment instruction reordering"). The KB branch alone delivered
    # the recipe and nothing to the student; this rides to the next student
    # turns as ctx.g5_note so the second half of the promise exists.
    student_directive: str | None = None
    # The lever pricing table this diagnosis read (plateau path), for the ledger.
    lever_prices: list = field(default_factory=list)
    # The phase the diagnosis closed on an exhausted price table (see
    # _phase_exhaustion): {"gate", "phase", "note"}; None when no phase closed.
    phase_close: dict | None = None


@dataclass
class KbCertifyResult:
    certified: bool
    probe: str | None
    rounds: int
    recipe: str | None = None  # RECIPE_CERTIFIED path — the student is pointed at it
    job: str = ""              # the certifying probe's job id (the entry's provenance pointer)


def _parse_list(text: str, field_name: str) -> list[str]:
    m = re.search(rf"{field_name}\s*:\s*\[([^\]]*)\]", text)
    if not m:
        return []
    # The reply is YAML-ish prose and the teacher quotes ids as often as not
    # (`changed_entries: ["D0.1", "D0.3"]`). A quoted id matches no DAG node, so
    # compute_resume / compute_reopen saw nothing to invalidate or reopen and the
    # landing check (which keys on the id shape) skipped it: an attn_bwd R1
    # rebuild was planned as "resume after the tip, nothing invalidated" and only
    # restarted from the baseline because no record owned that tip.
    items = (x.strip().strip("\"'`").strip() for x in m.group(1).split(","))
    return [x for x in items if x]


def _parse_field(text: str, field_name: str) -> str:
    m = re.search(rf"{field_name}\s*:\s*(\S+)", text)
    return m.group(1) if m else ""


def _parse_line(text: str, field_name: str) -> str:
    """Grab the rest of a ``field: ...`` line (multi-word values like patch_direction)."""
    m = re.search(rf"{field_name}\s*:\s*(.+)", text)
    return m.group(1).strip() if m else ""


# the reachability line, like cause_class below, must tolerate a full-width colon —
# an agent typing CJK punctuation (`reachability：`) would otherwise burn the one
# re-prompt the gate is allowed.
_REACH_LINE = re.compile(r"reachability\s*[:：]\s*(.+)")


def _parse_reachability(text: str) -> str:
    m = _REACH_LINE.search(text)
    return m.group(1).strip() if m else ""


# INFEASIBLE GATE: a stop request is accepted only on the diagnosis's own
# reachability disproof (diagnose_prompt §1c) — `-> BLOCKED` with both numbers
# and floor_opt above target x 1.20, the margin §1c itself sets for BLOCKED.
# OPEN means the disproof failed: the optimistic floor sits inside the target,
# which is a reason to keep walking, not to stop. Before this gate the first
# INFEASIBLE ended the run whatever the line said, and stops on an OPEN line
# ("every lever priced null" = levers exhausted, not target out of reach) cost
# runs most of their budget.
_REACH_VERDICT = re.compile(r"(?:->|→|=>)\s*\**\s*(BLOCKED|OPEN)\b", re.I)
_REACH_VERDICT_BARE = re.compile(r"\b(BLOCKED|OPEN)\b", re.I)
_REACH_UNIT_MS = {"ms": 1.0, "µs": 1e-3, "μs": 1e-3, "us": 1e-3}
_REACH_BLOCKED_MARGIN = 1.20


def _reach_quantity(line: str, name: str) -> float | None:
    """The first ``<name>=<number> <unit>`` on the line, in ms (µs / us accepted)."""
    m = re.search(rf"{name}\s*[=:：]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*(ms|µs|μs|us)\b", line)
    if not m:
        return None
    return float(m.group(1)) * _REACH_UNIT_MS[m.group(2)]


def _parse_reach_verdict(line: str) -> str:
    """BLOCKED / OPEN from the reachability line: the arrowed verdict (the last
    one, if the teacher restated it), else the single bare token; '' if neither
    or both appear bare."""
    arrowed = _REACH_VERDICT.findall(line)
    if arrowed:
        return arrowed[-1].upper()
    bare = {t.upper() for t in _REACH_VERDICT_BARE.findall(line)}
    return bare.pop() if len(bare) == 1 else ""


def _plateau_diagnoses(records) -> int:
    """Diagnoses this run has made that could have carried a stop request: every
    meta record except the distillation (AUTHORED) and a clean M0 audit."""
    return sum(1 for r in records if r.outcome not in ("AUTHORED", "AUDIT_CLEAN"))


def _infeasible_refusal(text: str, *, diagnoses: int, min_diagnoses: int,
                        target_ms: float | None) -> str:
    """Why an INFEASIBLE diagnosis is NOT accepted as a stop ('' = accepted).

    ``diagnoses`` counts this one; ``target_ms`` is the driver's stop threshold
    (None/0 = unknown, then the teacher's stated target alone is the bar)."""
    if diagnoses < min_diagnoses:
        return (f"this is diagnosis step {diagnoses} of this run; the loop accepts a stop "
                f"request only after at least {min_diagnoses} diagnosis steps")
    line = _parse_reachability(text)
    if not line:
        return "missing the `reachability:` line"
    verdict = _parse_reach_verdict(line)
    if verdict != "BLOCKED":
        return (f"the reachability verdict is "
                f"{verdict or 'unparseable (the line must end with -> BLOCKED or -> OPEN)'}; "
                f"only BLOCKED can support a stop")
    floor = _reach_quantity(line, "floor_opt")
    stated = _reach_quantity(line, "target")
    if floor is None or stated is None:
        return ("the BLOCKED line lacks a machine-readable `floor_opt=<number> ms` or "
                "`target=<number> ms`")
    bar = max(stated, target_ms or 0.0)
    if floor <= bar * _REACH_BLOCKED_MARGIN:
        src = ("the driver's stop threshold" if target_ms and target_ms > stated
               else "the target you wrote")
        return (f"floor_opt={floor:.6g} ms does not exceed target×{_REACH_BLOCKED_MARGIN:.2f}="
                f"{bar * _REACH_BLOCKED_MARGIN:.6g} ms (target taken from {src}: {bar:.6g} ms)")
    return ""


def _infeasible_refused_block(why: str) -> str:
    return (
        "\n## INFEASIBLE stop request not accepted (driver check)\n"
        f"Not satisfied: {why}.\n"
        "The driver accepts a stop request only when **all** of the following hold: (1) the "
        "`reachability` line reads `BLOCKED` and writes out `floor_opt=<number> ms` and "
        "`target=<number> ms`; (2) `floor_opt > target × "
        f"{_REACH_BLOCKED_MARGIN:.2f}` (target = the larger of the value you wrote and the "
        "driver's stop threshold); (3) the loop's configured minimum number of diagnosis steps "
        "has been reached. `OPEN` only says the disproof failed — the optimistic floor is still "
        "inside the target, so some routes have not been walked yet; it is no ground for a stop. "
        "\"Every in-tree lever priced null\" means the levers are exhausted, not that the target "
        "is out of reach.\n"
        "Resend a **complete** diagnosis under the usual classes (G1–G5 / R1 / R2); if you can add "
        "to `evidence` a floor construction that meets the conditions above (term-by-term "
        "arithmetic, `[MEASURED]` first), you may raise INFEASIBLE again together with the "
        "corrected `reachability` line. If it still does not meet them, the stop request is "
        "refused: this step applies no patch and re-diagnoses on the same state.\n")


# `reopen_price: <delta> via probe:<path>#<n> | lever:<name> | KB:<entry>` — the
# measured price an R1 / structural patch cites. Only the source is machine-
# checked (it must be something the driver measured this step, or a certified KB
# entry); the delta is the teacher's reading of it and travels to the record.
_REOPEN_PRICE_LINE = re.compile(r"reopen_price\s*[:：]\s*(.+)")
# the name stops at whitespace, list punctuation or EITHER paren family:
# teachers append fullwidth glosses ("lever:ring（independent）")
_REOPEN_PRICE_SRC = re.compile(r"\b(probe|lever|kb|ablation)\s*[:：]\s*([^\s,;，；()（）]+)", re.I)
_NO_READING = {"", "unreported", "n/a", "none"}
# A KB entry counts as a price source only when it is GPU-certified: its text
# carries a probe verdict, a job pointer or a [PRICED] tag. A transcribed
# recipe with none of these is a description, not a measurement. Job ids are
# 5-7 digits on the Hopper pools and 10 on the Blackwell pool (the sm100 tree's
# KEEP-census entries cite nothing else); the width matches kb_lint._JOB_ID.
_KB_CERT_MARK = re.compile(r"PROBE PASS|\[PRICED|GPU-certified|✅ Verified|probe_kb_|"
                           r"(?<![A-Za-z0-9_])job\s+\*{0,2}[0-9]{5,12}(?![0-9])")
_KB_NOT_ENTRIES = {"index", "README", "TEMPLATE"}


def _parse_reopen_price(text: str) -> str:
    m = _REOPEN_PRICE_LINE.search(text or "")
    return m.group(1).strip() if m else ""


def _reopen_gate_on(ctx: LoopContext) -> bool:
    """The no-reopen-without-a-price rule applies when the run has an in-step way
    to measure one: verdict probes or the pricing round. A control run with
    neither could only ever cite a KB entry, so the rule stands down there."""
    return bool(ctx.reopen_requires_price
                and (ctx.max_verdict_probe_rounds > 0
                     # the pricing round only runs on a priced tree
                     or (ctx.lever_pricing_rounds > 0 and ctx.stall_pricing)))


def _verify_reopen_price(cite: str, probe_readings: list, lever_table: list,
                         ctx: LoopContext, probes_run: list | None = None,
                         best_ref: str | None = None) -> str | None:
    """None when ``cite`` names a price the driver measured (or a certified KB
    entry); otherwise the reason it does not. ``probe_readings`` are this step's
    verdict-probe contract lines, ``lever_table`` this step's price table,
    ``probes_run`` the probe paths submitted this step (a probe that ran but
    printed no verdict line is "no reading", not "not run"), ``best_ref`` the
    tree the reopen would be applied to — a lever price read on another tree
    is stale and does not verify (see _reprice_cited_levers)."""
    if not cite:
        return "no reopen_price line"
    m = _REOPEN_PRICE_SRC.search(cite)
    if not m:
        return "reopen_price names no probe:/lever:/KB:/ablation: source"
    kind, name = m.group(1).lower(), m.group(2).strip("`'\"")
    if kind == "ablation":
        # an ablation delta is what the mechanism charges today — the ceiling
        # on what a reopen that fixes it can return. Only a POSITIVE delta
        # above the pass's noise floor is a price: n/a (hung, unbuilt) is no
        # reading, within-noise says the mechanism is not the term, negative
        # says removing it is slower — it is helping, and there is nothing to
        # return by reopening it.
        for lv in lever_table:
            if lv.get("kind") == "ablation" and lv.get("name") == name:
                d = lv.get("delta_pct")
                noise = lv.get("noise_pct") if lv.get("noise_pct") is not None else _LEVER_NOISE_DEFAULT_PCT
                if d is None:
                    return (f"ablation {name} has no measured delta (n/a: status="
                            f"{lv.get('status') or 'unknown'})")
                if d < -noise:
                    return (f"ablation {name} reads {d:+.2f}%: removing the mechanism is SLOWER — "
                            f"it is helping, not charging; not a price")
                if d <= noise:
                    return (f"ablation {name} reads {d:+.2f}%, within the {noise:.2f}% noise floor — "
                            f"the mechanism is not the term; not a price")
                return None
        return f"ablation {name} is not on the price table (this step or an earlier one)"
    if kind == "probe":
        path, _, idx = name.partition("#")
        path = path.lstrip("./")
        rows = [r for r in probe_readings if str(r.get("path", "")).lstrip("./") == path]
        if not rows:
            if any(str(p).lstrip("./") == path for p in (probes_run or [])):
                return f"probe {path} ran but printed no PROBE PASS/FAIL line (no reading)"
            return f"probe {path} was not run this step"
        if idx:
            rows = [r for r in rows if str(r.get("contract")) == idx]
            if not rows:
                return f"probe {path} printed no contract #{idx}"
        if not any(str(r.get("reading", "")).strip().lower() not in _NO_READING for r in rows):
            return f"probe {path} printed no reading (max_diff unreported / n/a)"
        return None
    if kind == "lever":
        for lv in lever_table:
            if lv.get("kind") == "lever" and lv.get("name") == name:
                if lv.get("add_pct") is None and lv.get("remove_pct") is None:
                    return f"lever {name} has no measured add / remove"
                return _stale_price(lv, name, best_ref)
            if lv.get("kind") == "variant" and lv.get("name") == name:
                if lv.get("delta_pct") is None:
                    return f"subset {name} has no measured delta"
                return _stale_price(lv, name, best_ref)
        return f"lever {name} is not on the price table (this step or an earlier one)"
    name = re.sub(r"\.md$", "", name.split("/")[-1])
    if name in _KB_NOT_ENTRIES:
        return f"KB:{name} is not a recipe entry"
    try:
        hit = next(kb_read_dir(ctx).rglob(f"{name}.md"), None)
        text = hit.read_text(encoding="utf-8") if hit else ""
    except OSError:
        hit, text = None, ""
    if hit is None:
        return f"KB entry {name} not found under {kb_read_dir(ctx)}"
    if not _KB_CERT_MARK.search(text):
        return (f"KB entry {name} carries no GPU certification (no PROBE PASS / "
                f"[PRICED] / job pointer) — a transcribed recipe is not a price")
    return None


def _stale_price(row: dict, name: str, best_ref: str | None) -> str | None:
    """None when ``row``'s price was read on ``best_ref`` (or either side is
    unknown — rows from before provenance was banked); else why it does not
    price a reopen of that tree."""
    on = str(row.get("priced_on") or "")
    if not best_ref or not on or on == best_ref:
        return None
    return (f"lever {name} was priced on tree {on[:10]}, not on the current best "
            f"{best_ref[:10]} — a stale price is not a price for this tree; list it in "
            f"levers.json again so the driver re-prices it here, then cite the new reading")


_UNPRICED_REOPEN_BLOCK = (
    "\n## A reopen must be priced first (driver check)\n"
    "You classified `R1` (structural reopen), but the diagnosis carries no price the driver "
    "measured. **A reopen without a measured price must not rewrite the milestone** — an observed "
    "failure mode: an audit inferred from \"where the wait instruction sits\" that the reference "
    "hid a wait, flipped the stage-split axis on that basis and had the whole group of decisions "
    "redone; the reopen itself was never priced, while the producer lever that was really worth "
    "double-digit percent never made it onto the candidates table. Take any one of the three "
    "routes, then resend a **complete** diagnosis:\n"
    "1. **Price it with a verdict experiment**: send `PROBE_REQUEST: notes/<name>.py` that builds "
    "the **minimal variant** of the reopen on the best tree (change only this one structural "
    "variable; do not deliver the full implementation), measured as a same-job pair, with the "
    "script printing `PROBE PASS/FAIL max_diff=<reading>`; then write "
    "`reopen_price: <±x.x%> via probe:notes/<name>.py#<contract index>` in the diagnosis.\n"
    "2. **Cite the pricing table**: the lever behind the reopen is already on this run's pricing "
    "table (this step or earlier, marked `priced` in the memory; **a price tag holds only for "
    "the best tree it was measured on** — a tag from an old table, read before best moved, is "
    "judged stale and must be re-listed in levers.json for a re-measure) → "
    "`reopen_price: <add or −remove> via lever:<name in the table>`; or the ablation table's "
    "reading for removing that mechanism → `reopen_price: <Δ> via ablation:<name>` (the driver "
    "records it as a ceiling: fixing it returns at most this much). If no ablation exists yet, "
    "write the ablation as JSON and send `PROBE_REQUEST: notes/<name>.json` (format: see "
    "`ablations` in the pricing round); the driver runs it with a frozen recipe, so you need "
    "not write a script.\n"
    "3. **Cite a certified KB entry**: the structural change has a job-certified KB recipe → "
    "`reopen_price: <measured delta in the entry> via KB:<entry name>`.\n"
    "If you can do none of these, land the diagnosis in a non-reopen class (G1/G2/G4/G5, or an "
    "additive R2) and write the reopen direction into `evidence` as an [ESTIMATE] candidate. "
    "If you submit an unpriced R1 again, the driver refuses this step and applies no patch.\n")


# --- reopen budget -------------------------------------------------------------

# --- reopen budget (loop/reopen.py) ----------------------------------------
# Moved to loop/reopen.py (docs/driver-split-spec.md step 5, first slice);
# imported back so every name it exposed from driver is still reachable at
# forgekernel.loop.driver.*
# --- shared constants (loop/shared.py) -------------------------------------
# Defined once in loop/shared.py so phase/timeline/gap can share them; imported
# back here early so every use below sees them.
from forgekernel.loop.shared import (  # noqa: F401 -- re-export
    _DECISION_ID,
    _LEVER_DESIGN_BLOCK,
    TRUNCATION_STORM_BUDGET,
)

from forgekernel.loop.reopen import (  # noqa: F401 -- re-export
    _MILESTONE_PATCH_CLASSES,
    _REOPEN_PATCH_CLASSES,
    _fruitless_reopen_streak,
    _reopen_capped_block,
    _INFEASIBLE_MENTION_BLOCK,
    _CAUSE_ENUM,
    _CAUSE_ENUM_AUDIT,
    _INFEASIBLE_MENTION,
    _parse_cause_class,
    _teacher_prompt,
)


def _kb_certify_lint_gate(ctx: LoopContext, diagnosis: str, *, fix_rounds: int = 2) -> list[str]:
    """Lint what the teacher just wrote into the run's KB tree, bouncing findings
    back for a bounded number of fix turns. Returns whatever findings survive.

    This is the second enforcement point (loop start is the other):
    without it, a leaky recipe certified mid-run sits in the run's tree — which
    the student reads in full — for every remaining iteration, and the next start
    gate only finds out after the run's result is already contaminated. The tree
    was lint-clean at loop start, so any finding here is the teacher's new
    content. Scoped to kb_read_dir: another architecture's tree is not in this
    run's read set, so its findings are not this run's to fix."""
    findings = lint_kb_dir(kb_read_dir(ctx), extra_tokens=ctx.kb_lint_tokens)
    for _ in range(fix_rounds):
        if not findings:
            return []
        _log(f"meta/kb: leak lint — {len(findings)} finding(s) in the fresh recipe, "
             f"bouncing back to the teacher")
        listing = "\n".join(f"- {f}" for f in findings)
        prov_dir = lang_mode.provenance_dir(ctx.meta_dir.parent, ctx.language)
        _teacher(ctx.backend).run_agent(_teacher_prompt(ctx, diagnosis, probe_feedback=(
            f"\n\n## KB LEAK LINT FAILED\n"
            f"What you just wrote into the KB contains banned provenance narrative (lint "
            f"findings below). Following the Sources discipline of {ctx.kb_dir}/TEMPLATE.md, "
            f"**edit these files in place**: move the reference solution's file names / function "
            f"names / line numbers / paraphrases / footprints to {prov_dir}/<entry name>.md, and "
            f"leave only the archive-pointer line in the entry. When done, output "
            f"RECIPE_CERTIFIED: <recipe-path> as the final line.\n"
            f"{listing}\n")))
        findings = lint_kb_dir(kb_read_dir(ctx), extra_tokens=ctx.kb_lint_tokens)
    return findings


def _quarantine_leaky_kb(ctx: LoopContext, findings: list[str]) -> None:
    """Move entries that failed the certify lint out of the student-readable tree.

    The run's tree (kb_read_dir) is what every student iteration reads; leaving
    a leaky entry there while merely not announcing it would not contain
    anything. Files are moved (not deleted) to <repo>/kb_quarantine/ for the
    human to inspect; a moved tracked entry shows up as a deletion in git status,
    so the next run's kb_blockers guard forces a human through here before
    anything restarts.

    A finding's path is relative to the lint root's parent (lint_kb_dir's
    rel_to), i.e. ``<arch>/<entry>.md`` under hw_kb/; the quarantine hangs off
    the repo root, because a quarantine inside hw_kb/ would still sit in the
    tree this is trying to take the entry out of."""
    qdir = lang_mode.kb_repo_root(ctx.kb_dir) / "kb_quarantine"
    for rel in sorted({f.split(":", 1)[0] for f in findings}):
        src = kb_read_dir(ctx).parent / rel
        if src.is_file():
            qdir.mkdir(exist_ok=True)
            dst = qdir / src.name
            src.rename(dst)
            _log(f"meta/kb: QUARANTINED leaky entry {rel} -> {dst} — review by hand")


def run_kb_certify(ctx: LoopContext, diagnosis: str) -> KbCertifyResult:
    """G3 fix: teacher writes a hw_kb recipe + probe, GPU-certifies it (PASS/FAIL loop).

    Mirrors the student probe sub-loop but for the teacher: the recipe is only
    accepted once its certification probe returns PROBE PASS on real hardware.

    The certification's `kb-probe:` commits live on the -kbprobe branch (pushed
    per round, provenance); the student's lineage does not inherit them. The
    workspace is reset to the pre-certification head afterwards and the probe
    files re-materialised untracked, so the student's next KEEP is built on a
    tree the records know — its base_sha pairs with its censused ancestor and
    kb_axis_debt bills exactly the KEEP that applies the fresh recipe — and the
    student line stays teacher-free, as after a verdict probe.
    """
    be = ctx.backend
    pre_head = be.head()
    probes: dict = {}
    try:
        return _kb_certify_rounds(ctx, diagnosis, probes)
    finally:
        if probes:
            _detach_kb_probes(ctx, pre_head, probes)


def _detach_kb_probes(ctx: LoopContext, pre_head: str, probes: dict) -> None:
    be = ctx.backend
    if lost := _edits_outside_notes(be, pre_head):
        _note_discarded_edits(ctx, "meta/kb (certification reset)", lost)
    try:
        be.reset(pre_head)
    except BackendError as e:
        _log(f"meta/kb: could not reset the workspace to {pre_head[:10]} after the "
             f"certification ({e}) — the kb-probe commits stay in the student lineage")
        return
    kept = []
    for path, text in probes.items():
        if text is None:
            continue
        dst = ctx.workspace / path
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text)
            kept.append(path)
        except OSError as e:
            _log(f"meta/kb: could not re-materialise {path}: {e}")
    _log(f"meta/kb: workspace reset to the pre-certification head {pre_head[:10]}; "
         f"probe file(s) {kept or list(probes)} re-materialised untracked "
         f"(the -kbprobe branch keeps the commits)")


def _kb_certify_rounds(ctx: LoopContext, diagnosis: str, probes: dict) -> KbCertifyResult:
    be = ctx.backend
    prompt = _teacher_prompt(ctx, diagnosis)
    for r in range(ctx.max_probe_rounds):
        # safe variant: the teacher's finalize turn is naturally prose, and a
        # stray RECON_REQUEST must not parse here (recon runs in another tree).
        turn = parse_student_turn_safe(_teacher(be).run_agent(prompt))
        if not isinstance(turn, ProbeRequest):
            if leaks := _kb_certify_lint_gate(ctx, diagnosis):
                _quarantine_leaky_kb(ctx, leaks)
                _log("meta/kb: recipe REJECTED — leak lint findings survived the fix rounds")
                return KbCertifyResult(False, None, r)
            _log(f"meta/kb: teacher finalized recipe (no probe) after {r} rounds")
            return KbCertifyResult(True, None, r)
        be.commit(f"kb-probe: {turn.path}")
        probes[turn.path] = _keep_probe_across_reset(ctx, turn.path)
        kb_branch = f"{ctx.branch_prefix}-kbprobe"
        be.push_branch(kb_branch)
        kb_log = be.submit_gpu(kb_branch, f"cmd:python3 {turn.path}")
        pv = parse_probe_verdict(kb_log)
        # the job id is the entry's provenance pointer (index / status cell
        # `job N`, the harvest's `[MEASURED, job N]`): a certification whose
        # job is not on the log cannot be cited, and the loop log is the only
        # place it is ever written
        job = str(getattr(be, "last_job_id", "") or "")
        _log(f"meta/kb: recipe probe {turn.path} -> PROBE {_probe_label(pv)} "
             f"(max_diff={pv.diff_text}){_probe_log_suffix(pv)} (round {r + 1}, "
             f"job {job or '?'})")
        if pv.passed:
            # finalize: teacher marks the recipe verified in the KB index
            final_out = _teacher(be).run_agent(_teacher_prompt(
                ctx, diagnosis,
                probe_feedback=_probe_feedback(
                    turn.path, pv, kb_log, what="recipe probe",
                    on_pass="Finalize: mark the recipe verified in the KB index and "
                            "emit RECIPE_CERTIFIED: <recipe-path>. Cite the certifying "
                            f"job in the entry's status line and its index cell: "
                            f"`PROBE PASS max_diff={pv.diff_text}, job {job or '?'}` "
                            f"(round {r + 1} of this certification).")))
            recipe = _parse_field(final_out, "RECIPE_CERTIFIED")
            if leaks := _kb_certify_lint_gate(ctx, diagnosis):
                _quarantine_leaky_kb(ctx, leaks)
                _log("meta/kb: recipe REJECTED — leak lint findings survived the fix rounds")
                return KbCertifyResult(False, None, r + 1, job=job)
            _log(f"meta/kb: recipe CERTIFIED and available to the student ({recipe or '?'}; "
                 f"job {job or '?'})")
            return KbCertifyResult(True, turn.path, r + 1, recipe=recipe or None, job=job)
        prompt = _teacher_prompt(
            ctx, diagnosis,
            probe_feedback=_probe_feedback(turn.path, pv, kb_log, what="recipe probe"))
    _log("meta/kb: could not certify a recipe within the probe budget")
    return KbCertifyResult(False, None, ctx.max_probe_rounds)


def kb_harvest_debt(ctx: LoopContext) -> list[ProbeBattle]:
    """The probe victories expensive enough to owe a KB write-up, dearest first."""
    if not ctx.kb_tripwire_rounds:
        return []
    debt = [b for b in ctx.probe_battles.values()
            if b.passed and b.rounds >= ctx.kb_tripwire_rounds]
    return sorted(debt, key=lambda b: -b.rounds)


@dataclass
class AxisBattle:
    """One KEEP that moved a SASS contract axis — a harvest item without a fight.

    The kb-tripwire bills probe fights; a stuck stop bills a diagnosis. A
    one-round KEEP that changes the built structure (sync topology, in-flight
    depth, supply order, cache-policy bits — the orch_tools summary axes) had
    no channel at all, and that is exactly how a 22 % knob shipped twice at
    ~4 % and evaporated with both runs (dsa_sparse_fwd). The
    evidence is the pair of per-iteration census readings around the KEEP plus
    its own eval — the KEEP's NCU job IS the certification, the same way a
    probe's PASS is for a fight. Computed from status records, so a resumed
    run bills the same KEEPs as the one that made them.

    "Moved an axis" is read on the count-free contract signature
    (`_contract_signature`), not on raw summary equality: the
    counting fields (mbarrier object / tier / per-role counts, nbar widths,
    itc candidates, per-loop DEPBAR lines) still move with codegen after every
    real fix (11 residual `!` on the grouped-gemm pages), and billing those
    would hand the teacher compiler noise to write recipes for. The raw axes
    that changed alongside ride as ``noise_axes`` for the reader."""
    iter: int
    decision_id: str
    sha: str
    axes: dict            # axis -> {"before": ..., "after": ...}
    ms_before: float | None
    ms_after: float | None
    change: str
    prev_iter: int | None = None
    noise_axes: list = field(default_factory=list)

    @property
    def delta_pct(self) -> float | None:
        if self.ms_before and self.ms_after:
            return round(100.0 * (self.ms_after - self.ms_before) / self.ms_before, 2)
        return None


def _axis_changes(prev: dict, cur: dict) -> tuple[dict, list[str]]:
    """({axis: {before, after}} over the contract axes whose count-free
    signature differs, [raw summary keys that differ but are not a contract
    move]). The before/after readings are the raw axis values (the evidence
    the teacher reads), the trigger is the signature."""
    from forgekernel.skills.orch_tools import _DIFF_KEYS
    a = json.loads(json.dumps(prev))
    b = json.loads(json.dumps(cur))
    sa, sb = _contract_signature(a), _contract_signature(b)
    moved = {k: {"before": a.get(k), "after": b.get(k)}
             for k in _CONTRACT_DIFF_AXES if sa.get(k) != sb.get(k)}
    noise = [k for k in _DIFF_KEYS if k not in moved and a.get(k) != b.get(k)]
    return moved, noise


def _census_ancestor(rec, by_sha: dict):
    """The nearest censused KEEP in ``rec``'s own lineage: walk base_sha ->
    record until one carries a census. None when the chain leaves the records
    (a baseline / meta-patched tree) — then there is nothing to compare with."""
    sha = rec.base_sha
    seen: set = set()
    while sha and sha in by_sha and sha not in seen:
        seen.add(sha)
        p = by_sha[sha]
        if p.verdict == "KEEP" and p.orch_summary:
            return p
        sha = p.base_sha
    return None


def kb_axis_debt(records, gate_window: str = GATE_WINDOW_DEFAULT) -> list[AxisBattle]:
    """The KEEPs whose census differs from the previous censused KEEP's IN THEIR
    LINEAGE, in iteration order. Records without a summary (census skipped,
    pre-census runs) are transparent: the comparison is between the two nearest
    KEEPs that carry one, and the first censused KEEP has nothing to compare
    against (it is the baseline, not a move).

    Lineage, not adjacency: after a meta reopen / reset the next KEEP is built
    on an earlier tree, and diffing it against the rolled-back KEEP would bill
    the rollback to its change record. A record that knows the tree it was
    built on (``base_sha``) is paired through that chain; records from before
    the field existed fall back to the previous censused KEEP."""
    debt: list[AxisBattle] = []
    prev_adjacent = None
    by_sha: dict = {}
    for r in records:
        if r.sha:
            by_sha[r.sha] = r
        if r.verdict != "KEEP" or not r.orch_summary:
            continue
        prev = _census_ancestor(r, by_sha) if r.base_sha else prev_adjacent
        if prev is not None:
            axes, noise = _axis_changes(prev.orch_summary, r.orch_summary)
            if axes:
                debt.append(AxisBattle(
                    iter=r.iter, decision_id=r.decision_id, sha=r.sha, axes=axes,
                    ms_before=gate_ms(prev, gate_window), ms_after=gate_ms(r, gate_window),
                    change=r.change, prev_iter=prev.iter, noise_axes=noise))
        prev_adjacent = r
    return debt


def _log_axis_change(ctx: LoopContext, rec) -> None:
    """Name the axis a just-landed KEEP moved, at the moment it lands (the bill
    itself is settled from the records at harvest time)."""
    if not rec.orch_summary:
        return
    try:
        records = load_records(ctx.status_path)
        debt = [b for b in kb_axis_debt(records, ctx.gate_window) if b.iter == rec.iter]
    except Exception as e:  # noqa: BLE001 — a log line, never a precondition
        _log(f"iter {rec.iter}: kb-axis: could not compare census ({e.__class__.__name__}: {e})")
        return
    for b in debt:
        d = f" ({b.ms_before} -> {b.ms_after} ms, {b.delta_pct:+.2f}%)" if b.delta_pct is not None else ""
        _log(f"iter {rec.iter}: kb-axis: {rec.decision_id} moved SASS contract axes "
             f"{sorted(b.axes)}{d} — queued for the end-of-run KB harvest write-up")


def _axis_battle_block(b: AxisBattle) -> str:
    d = f"{b.ms_before} -> {b.ms_after} ms"
    if b.delta_pct is not None:
        d += f" ({b.delta_pct:+.2f}%)"
    axes = "\n".join(
        f"  - `{k}`:\n      before (iter {b.prev_iter}): {json.dumps(v['before'], ensure_ascii=False)}\n"
        f"      after  (iter {b.iter}): {json.dumps(v['after'], ensure_ascii=False)}"
        for k, v in b.axes.items())
    noise = (f"- Counting axes that moved at the same time (codegen follow-on; not billed, "
             f"no recipe): {', '.join(b.noise_axes)}\n"
             if b.noise_axes else "")
    return (f"### iter {b.iter} {b.decision_id} (sha {b.sha[:10] if b.sha else '?'}) — "
            f"gate window {d}\n- Contract axes changed:\n{axes}\n{noise}"
            f"- Student change record (first 600 chars):\n```\n{b.change[:600]}\n```")


def _harvest_lint_findings(ctx: LoopContext) -> list[str]:
    try:
        return lint_kb_dir(kb_read_dir(ctx), extra_tokens=ctx.kb_lint_tokens)
    except Exception as e:  # noqa: BLE001 — a lint crash must not sink the harvest
        _log(f"meta/kb-harvest: kb_lint unavailable ({e.__class__.__name__}: {e})")
        return []


def _kb_deadend_block(ctx: LoopContext) -> str:
    """The KB's ABSENT / BROKEN entries, handed to the teacher as dead-end candidates.

    A recipe that records "this interface compiles to nothing" or "this one reads
    the wrong byte" is, for the operator being optimized, a decision-level dead
    end: an approach that cannot pay no matter how well it is written. Today that
    knowledge sits only in hw_kb and reaches the milestone's ``## 4 Known dead
    ends`` only if a human copies it over — so the next run on the same operator
    re-derives the absence the hard way. This block closes that loop: it lists the
    entries the KB already marks ABSENT / BROKEN, and asks the teacher to move the
    ones that bear on this operator's decisions into the milestone's dead-end
    table (under §6's rules — decision granularity, retry precondition). It is a
    candidate list, never an instruction to copy wholesale: a mechanism the
    operator never touches is not its dead end.

    Reads the run's own tree (``kb_read_dir(ctx)``) and its parent's ``_shared/``
    so an architecture-neutral entry is seen too. Empty string when nothing in the
    tree is marked absent or broken, so a healthy KB adds nothing to the prompt.
    """
    from forgekernel.loop import kb_state
    roots = [kb_read_dir(ctx)]
    shared = Path(ctx.kb_dir) / "_shared"
    if shared.is_dir():
        roots.append(shared)
    rows: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.rglob("*.md")):
            try:
                declared = kb_state.declared_state(entry.read_text(encoding="utf-8"))
            except OSError:
                continue
            if declared is None or declared.state not in (kb_state.ABSENT, kb_state.BROKEN):
                continue
            try:
                name = entry.relative_to(Path(ctx.kb_dir))
            except ValueError:
                name = entry
            rows.append(f"- {declared.mark} `{name}` — {declared.state} "
                        f"({declared.instruction})")
    if not rows:
        return ""
    return (
        f"\n\n## Dead-end candidate list (KB entries already marked ABSENT / BROKEN)\n\n"
        f"These interfaces are recorded as not-there or wrong-result on this target. "
        f"For each one, judge whether it bears on a decision in `{{MILESTONE_PATH}}`:\n"
        f"if the operator would otherwise reach for it, add ONE row to the milestone's "
        f"`## 4. Known dead ends` table (decision / symptom / trigger / retry "
        f"precondition) at decision granularity (§6) naming the KB entry as the "
        f"symptom's source; if the operator never touches it, leave the milestone "
        f"alone. Do NOT copy the whole list in, and do NOT restate these as new "
        f"hw_kb files — they already exist.\n\n" + "\n".join(rows) + "\n"
    )


def run_kb_harvest(ctx: LoopContext) -> list[str]:
    """End-of-run KB write-up for expensive probe victories (the kb-tripwire)
    and for KEEPs that moved a SASS contract axis (kb_axis_debt).

    The G3 path certifies fresh recipes with its own GPU loop; this path exists
    for the fights G3 never sees — the student won, so the run has no ``stuck``
    stop and no diagnosis, and the lesson lives only in probe notes and commit
    messages. The winning probe already returned PROBE PASS on real hardware, so
    that run IS the certification evidence: one teacher write-up turn citing the
    probe as provenance. The axis items are the same shape with a different
    evidence face: the census before/after and the KEEP's own eval reading.
    Zero GPU jobs, zero meta steps, and any failure here degrades to the status
    quo (lessons stay in the notes).

    Returns the recipe paths the teacher reported via ``RECIPE_RECORDED:`` lines.
    """
    debt = kb_harvest_debt(ctx)
    try:
        axis_debt = kb_axis_debt(load_records(ctx.status_path), ctx.gate_window)
    except Exception as e:  # noqa: BLE001 — never let the axis bill block the probe bill
        _log(f"meta/kb-harvest: axis bill unavailable ({e.__class__.__name__}: {e})")
        axis_debt = []
    # The gap queue (unverified recon rows) is deliberately NOT a harvest item:
    # it is the reference's structure as narrated, and hw_kb takes only
    # probe-certified recipes (see the gap-queue section below).
    if not debt and not axis_debt:
        return []
    # Everything from here on is best-effort, and the caller sits BEFORE the
    # harvest report: an exception escaping this function would
    # take down the one code path that keeps uncommitted hw_kb recipes from
    # being a `git checkout` away from gone. So the promise is total — a missing
    # prompt template (version skew) degrades exactly like a dead agent.
    try:
        extra = ""
        if debt:
            battles = "\n\n".join(
                f"### {b.path} — {b.rounds} GPU rounds before PASS (iter {b.pass_iter})\n"
                f"- Final verdict: {b.verdict_line}\n"
                f"- Tail of the winning job log:\n```\n{b.log_tail}\n```"
                for b in debt)
            extra += (f"\n\n## Probe battle list (expensive probe wins in this run, most costly "
                      f"first)\n\n{battles}\n")
        if axis_debt:
            moves = "\n\n".join(_axis_battle_block(b) for b in axis_debt)
            extra += (f"\n\n## Contract-axis change list (KEEP patches that changed a SASS contract "
                      f"axis, in iter order; readings = that iter's own eval + the dossier census "
                      f"before and after)\n\n{moves}\n")
        extra += _kb_deadend_block(ctx)
        prompt = render_prompt(
            _meta_prompt(ctx, "kb_harvest_prompt.md"), _bindings(ctx), extra=extra)
        _log(f"meta/kb-harvest: {len(debt)} expensive probe victor(ies) at threshold "
             f"{ctx.kb_tripwire_rounds} rounds + {len(axis_debt)} axis-moving KEEP(s) "
             f"— one write-up turn, no GPU")
        out = _teacher(ctx.backend).run_agent(prompt)
        # What the teacher wrote is the next run's exam paper: the same lint the
        # loop start enforces runs here, once, with the findings —
        # otherwise a harvest that slipped narrative into hw_kb blocks every
        # later `forge loop` start until someone hand-edits the entry.
        findings = _harvest_lint_findings(ctx)
        if findings:
            _log(f"meta/kb-harvest: {len(findings)} lint finding(s) in the write-up "
                 f"-> one re-prompt: {findings[0]}")
            out = _teacher(ctx.backend).run_agent(prompt + (
                "\n\n## What you wrote into the KB last time failed kb_lint (fix it and resend)\n"
                "None of the lines below may stay in any file under `{KB_READ_DIR}` "
                "(index.md included): run names / dates, iter numbers, commit / cubin shas, "
                "where the reference implementation came from. Move the narrative to the "
                "merge-request description or `docs/kb-provenance/`; evidence brackets carry "
                "only job numbers / probe paths / readings.\n"
                + "\n".join(f"- {f}" for f in findings[:40]) + "\n").replace(
                    "{KB_READ_DIR}", str(kb_read_dir(ctx))))
            left = _harvest_lint_findings(ctx)
            if left:
                _log(f"meta/kb-harvest: {len(left)} lint finding(s) remain after the "
                     f"re-prompt — the next loop start will refuse them: {left[0]}")
    except Exception as e:  # noqa: BLE001 — degrade to the status quo, never raise
        _log(f"meta/kb-harvest: failed ({e.__class__.__name__}: {e}) — "
             f"lessons stay in the probe notes")
        return []
    recipes = re.findall(r"^RECIPE_RECORDED:\s*(\S+)", out, re.M)
    if recipes:
        for r in recipes:
            _log(f"meta/kb-harvest: recorded {r}")
    else:
        _log("meta/kb-harvest: no RECIPE_RECORDED line from the teacher — "
             "check the KB diff by hand (NO_RECIPE or unparseable turn)")
    return recipes


# --- dossier contract signature ------------------------------------------------------

# Moved to loop/pricing.py (docs/driver-split-spec.md step 4); imported back so
# every name it exposed from driver is still reachable at forgekernel.loop.driver.*
from forgekernel.loop.pricing import (  # noqa: F401 -- re-export
    _CONTRACT_DIFF_AXES,
    _L2_HINT_RE,
    _CACHE_DESCRIPTOR_FAMS,
    _cache_class,
    _SUPPLY_TIGHT_RE,
    _contract_signature,
    _LEVER_LINE_RE,
    _STACK_LINE_RE,
    _VARIANT_LINE_RE,
    _PRICE_OK_RE,
    _PRICE_FAIL_RE,
    _PRICE_NOISE_RE,
    _PRICE_TIMED_RE,
    _PRICE_SCOPE_RE,
    _ABLATE_OK_RE,
    _TOOL_TRACEBACK,
    _ABLATE_LINE_RE,
    _ABLATE_NOISE_RE,
    _ABLATE_FAIL_RE,
    _NODE_DIRTY_MARKER,
    _price_log_verdict,
    _price_role_pass_killed,
    _price_log_settled,
    _price_log_lost_reason,
    _node_dirty_reason,
    _lever_price_job,
    _lever_price_submit,
    _LEVER_MAX_ABLATIONS,
    _LEVER_NOISE_DEFAULT_PCT,
    _LEVER_SPEC_NAME,
    _LEVER_PRICING_CMD,
    _LEVER_PRICE_RETRIES,
    _LEVER_MAX_SUBSETS,
    _pct_or_none,
    parse_lever_prices,
    _float_or_none,
)
# --- phase exhaustion (loop/reopen.py) -------------------------------------
# Moved to loop/reopen.py (docs/driver-split-spec.md step 5, second ring);
# imported back so every name it exposed from driver is still reachable at
# forgekernel.loop.driver.*
from forgekernel.loop.reopen import (  # noqa: F401 -- re-export
    _PHASE_EXHAUSTED_CLASSES,
    _table_noise_pct,
    _PHASE_VERDICT_LINE,
    _current_phase,
    _delivered_decisions,
    _REOPEN_CLOSURE_RE,
    _live_records,
    _phase_exhaustion,
    _phase_exhausted_block,
    _parse_phase_verdict,
    _verify_phase_verdict,
    _phase_close_note,
    _phase_priced_note,
    _lever_price_lost_block,
    _lever_price_block,
    _ablation_block,
    _fmt_pct,
    _merge_lever_tables,
    _run_lever_pricing,
    _lever_spec_refused,
    _normalise_ablations,
    _lever_refusal_block,
    _banked_lever_table,
    _persist_lever_table,
)


# --- price staleness ----------------------------------------------------------

# Moved to loop/pricing.py (docs/driver-split-spec.md step 4); imported back so
# every name it exposed from driver is still reachable at forgekernel.loop.driver.*
from forgekernel.loop.pricing import (  # noqa: F401 -- re-export
    _LEVER_NAME,
    _LEVER_CITE_RE,
    _CITE_JOB_RE,
    _REPRICE_TAG,
    _REVISIONS_NAME,
    _REVISION_SUSPENDS,
    _REVISION_FLAGGED,
    _stamp_price_row,
    _priced_best,
    _cited_levers,
    _cited_price_jobs,
    _lever_price_history,
    _ms_of_tree,
    _price_legs,
    _reprice_verdict,
    _append_revisions,
    _unretired_revisions,
    _retire_revisions,
    _legs_text,
    _reprice_marker,
    _mark_lever_citations,
    _revision_line,
    _price_revision_note,
    _price_revision_teacher_block,
    _reprice_cited_levers,
)
# --- timeline probe -------------------------------------------------------------
#
# A latency-bound kernel (a serial recurrence: one CTA walks its chunks one
# after another, nothing hides an exposed cycle) is judged in cycles per chunk
# per segment, and the stall / role account cannot say that: it says WHO waits,
# not WHERE the cycles sit. A GDN run spent 15 iterations and three meta
# steps on register accounting while the aux role's inverse level and gate
# epilogue were ~2400 cycles/chunk over the reference and the state roles waited
# ~1300 cycles/chunk on them — a per-segment clock account read that off one
# job. recon/timeline_probe.py is that instrument (the student's `_seg(...)`
# markers rewritten into %clock64 stamps in a probe build; the reference is never
# instrumented, so blackbox runs use the same path). The driver runs it at every
# teacher entry, at gate rows, at every new best beyond reprice_min_gain_pct,
# and on request, banks the table, puts it first among the teacher's evidence
# and makes the diagnosis name the segment (`locus_segment:`).
# --- timeline probe (loop/timeline.py) -------------------------------------
# Moved to loop/timeline.py (docs/driver-split-spec.md step 7, first slice);
# imported back so every name it exposed from driver is still reachable at
# forgekernel.loop.driver.*
from forgekernel.loop.timeline import (  # noqa: F401 -- re-export
    _TIMELINE_CMD,
    _timeline_cmd,
    _TIMELINE_OVERLAY_MSG,
    _TIMELINES_NAME,
    _TIMELINE_REQUESTS,
    _LOCUS_SEG_LINE,
    _BOTTLENECK_LATENCY,
    _is_timeline_request,
    _timeline_on,
    _timeline_markers,
    _timeline_unparsed_markers,
    _parse_locus_segment,
    _timeline_short_coverage,
    _timeline_probe_stale,
    _timeline_preamble_fence_hit,
    _timeline_probe_ref,
    _run_timeline,
    _fmt_cyc,
    _timeline_renderable,
    _bank_timeline,
    _banked_timelines,
    _banked_timeline_usable,
    _latest_timeline,
    _timeline_probe_digest,
    _banked_timeline_of,
    _timeline_due,
    _ref_region_lines,
    _timeline_tables,
    _timeline_block,
    _timeline_student_note,
    _timeline_student_note_cuda,
    _timeline_student_note_tilelang,
    _timeline_feedback,
)




def _lever_pricing_jobs_text(ctx: LoopContext) -> str:
    """How many GPU jobs the design turn's levers.json buys (_LEVER_DESIGN_BLOCK's
    {PRICING_JOBS}): one, or -- a group card listing its meta workloads -- one per
    meta workload, each timing that workload alone (_lever_price_submit)."""
    meta = meta_shapes.card_meta_workloads(ctx.workspace)
    if not meta:
        return "one GPU job"
    return (f"multi-workload tree: **one GPU job for each** of the operator tree's {len(meta)} meta "
            f"workloads ({', '.join(f'`{w}`' for w in meta)}, the representative first) — the same "
            f"levers.json (same levers, subsets and ablations), each job "
            f"timing only its own workload; the driver merges the tables by geomean and attaches "
            f"the per-workload readings under the table "
            f"(when a workload's job produced no table, the text above the table says which ones "
            f"were actually merged)")


def run_meta_step(ctx: LoopContext, *, best_before: float | None = None,
                  milestone_before: str = "", audit: bool = False,
                  reanchor_note: str | None = None) -> MetaResult:
    """Diagnose, then either certify a hw_kb recipe (G3) or patch the milestone.

    Records a MetaRecord (the meta agent's memory): the diagnosis, the patch class +
    changed entries, and the outcome — terminal for the G3 KB branch, PENDING for a
    milestone patch (judged after its inner segment runs, see run_two_layer). Prior
    meta records are reflected back into the diagnose prompt so the meta agent does
    not re-propose a direction that already regressed.

    ``audit``: the forced M0 reconciliation variant. Uses m0_audit_prompt.md (the
    question is "is the MILESTONE consistent with the hardware truth", not "why did
    the segment stall") and accepts the extra verdict ``OK`` — no defect, no patch,
    recorded AUDIT_CLEAN. Any G/R verdict flows into the normal KB / patch machinery.
    """
    be = ctx.backend
    bindings = _bindings(ctx)
    bindings["{DIAGNOSIS_PATH}"] = "(inline below)"

    prior_records = load_meta_records(ctx.meta_status_path) if ctx.meta_status_path else []
    meta_step = len(prior_records)
    reflection = reflect_summary(prior_records)
    # INJECT the NCU report inline: the meta agent runs in the student workspace and
    # cannot fetch it via ./forge, so the diagnose prompt must carry the hardware truth
    # itself — otherwise meta diagnoses blind and over-relies on the perf numbers.
    from forgekernel.inner.ncu import (render_ncu_reconciliation, render_ncu_report,
                                       render_ncu_trace)
    records = load_records(ctx.status_path)
    # the phase / parity readings must not see rows of trees that are gone
    # (a reverted segment, KEEPs an in-place REOPEN voided) — see _live_records
    live_records = _live_records(records, ctx.voided_ranges)
    # The arch ref's signature rides along so ratio-based flags (wavefront
    # volume) can fire — a volume tax is invisible to per-iteration differencing
    # AND to absolute thresholds; only student-vs-ref exposes it.
    arch_ncu = ctx.arch_baseline.ncu if (ctx.arch_baseline and ctx.arch_baseline.has_ncu) else None
    # The reconciliation reads any archived signature: the arch ref (the machine
    # being reproduced) wins when both exist; the library baseline covers runs
    # that only measured ref_baseline.json. The whole BASELINE travels, not just
    # its ncu dict: the block prints role + ref_name so the reader can tell which
    # reference is speaking, and the isomorphism verdict keys on the role — a
    # library kernel is a different implementation, and "same tensor work" against
    # it is a coincidence, not a construction target (see baseline.py on why the
    # two roles must not be collapsed).
    recon_baseline = ctx.arch_baseline if arch_ncu else (
        ctx.ref_baseline if ctx.ref_baseline and ctx.ref_baseline.has_ncu else None)
    best_keep = _best_keep_profiler(records, ctx.gate_window)
    recon = None
    if recon_baseline is not None and best_keep is not None:
        best_prof, best_iter = best_keep
        recon = render_ncu_reconciliation(
            best_prof, recon_baseline.ncu, student_iter=best_iter,
            ref_role=recon_baseline.role, ref_name=recon_baseline.ref_name)
    ncu_view = (f"\n\n## NCU hardware ground truth (inline, per-iteration bottleneck verdicts — "
                f"primary evidence, read before judging)\n"
                f"{render_ncu_report(records, baseline=arch_ncu)}"
                f"\n\n### Metric evolution table\n{render_ncu_trace(records)}\n")
    if recon:
        # Two-sided reconciliation: the digest above never quoted the ref side,
        # so "tensor occupancy gap ≈ the whole residual" — the decisive plateau
        # signal — was invisible to every diagnosis (observed on fp8 and GDN runs).
        ncu_view += (f"\n### Two-sided NCU reconciliation (student best KEEP vs reference "
                     f"signature, read before judging)\n"
                     f"{recon}\n")
    # The timed-window ledger: what each side's stopwatch covers, both
    # decompositions, and the priced contract tax. Without it a residual that is
    # entirely a rebuild the reference never pays reads as a kernel defect and
    # the diagnosis goes looking for it in register ledgers.
    ncu_view += f"\n{_window_contract_binding(ctx)}\n"
    diagnose_extra = (f"{ncu_view}\n## Your past meta steps (reflect; do not repeat failed "
                      f"directions)\n{reflection}\n")
    # REOPEN BUDGET (see LoopContext.reopen_max_fruitless): read off the meta
    # memory, so a driver restart keeps the count
    reopen_streak = _fruitless_reopen_streak(prior_records)
    reopens_capped = 0 < ctx.reopen_max_fruitless <= reopen_streak
    if reopens_capped:
        diagnose_extra += _reopen_capped_block(reopen_streak, ctx.reopen_max_fruitless)
    if not audit and reanchor_note:
        # Passed by run_two_layer for every diagnosis of the SAME meta entry —
        # including malformed-patch retries, which re-run the diagnosis and would
        # otherwise reason from the stale residual again. A later meta entry is a
        # new period: the caller measures afresh instead of forwarding this note.
        diagnose_extra += reanchor_note
    priced = None
    if ctx.stall_pricing:
        # The aggregate digest above ranks bottleneck CLASSES; only a per-instruction
        # ledger localizes a gap dispersed through mainloop interleave quality (the
        # observed on a fwd run: two loci chosen off the digest, ~1pp each once priced, 10h
        # of construction rolled back). Price the best kernel NOW, before a locus
        # is named, and put the ref's recon-time ledger beside it.
        #
        # Every teacher entry prices — the audit included. The audit used to be
        # paper-level ("no GPU job at M0 time"), which left it holding the ref's
        # dossier against the blueprint's WORDING while the student's built
        # structure went unread: a segment whose every stop is an M0 rebuild
        # (first_green -> audit -> foundation patch -> rebuild -> first_green ...)
        # never reaches the plateau path, so the student dossier was produced
        # once per run or never, and the audit kept re-legislating acceptance
        # text at the same locus while the `!` structural rows stood.
        _ensure_ref_pricing(ctx)
        priced = _price_student_trees(ctx, records)
        student_ledger, student_segments = priced.ledger, priced.segments
        ref_ledger = _read_ref_stall_ledger(ctx)
        ref_segments = _read_ref_segment_ledger(ctx)
        aligned = (align_segment_table(ref_segments, student_segments)
                   if ref_segments and student_segments else None)
        orch_diff = _orch_diff_views(ctx, priced)
        role_view = _role_exposure_view(ctx, priced)
        if role_view:
            # The priced face by role: where the segment table says WHICH
            # SEGMENT waits and the dossier says WHICH STRUCTURE differs, this
            # says WHICH ROLE owns the non-tensor exposure, in ns, both sides.
            diagnose_extra += role_view
        if orch_diff:
            # The structural face beside the priced one: the stall ledger says
            # where time leaks, this says which sync structure differs — the
            # known failure mode ("[GATHER: prefetched] in the blueprint,
            # never built") is a `!` row here, statically, before any timing
            # argument. `=` rows are equally load-bearing: a structure already
            # isomorphic to the ref is not where the residual lives.
            diagnose_extra += orch_diff
    tl_table = None
    if _timeline_on(ctx):
        # the per-segment clock account of the tree under judgment: rendered
        # before the stall ledgers, because on a latency-bound kernel it is the
        # evidence the locus is read from; a probe that produced nothing is
        # logged, not shown — but a partial-coverage table IS shown (dropping
        # it once left three meta entries diagnosing from aggregate NCU ratios
        # alone)
        tl_ref = (priced.best_sha or priced.head_sha) if priced is not None else ctx.backend.head()
        tl_table = _run_timeline(ctx, tl_ref, "audit" if audit else "plateau")
        if tl_table and _timeline_renderable(tl_table):
            diagnose_extra += _timeline_block(tl_table, _ref_region_lines(ctx))
        else:
            tl_table = None
    if not audit and ctx.stall_pricing:
        if aligned:
            # The comparison the hard rule needs, pre-aligned: same anchor, ref
            # share next to student share. Two raw ledgers made the reader do the
            # alignment, and the reading that matters (share 23.5 vs 35.4 at the
            # SAME segment, dispatch- vs wait-dominated) only exists after it.
            diagnose_extra += (
                "\n## Per-instruction stall ledger — same-position segment comparison (read "
                "before judging — `deficient_locus` must match the segments where the student "
                "exceeds the reference)\n"
                "Anchors aligned on the schedule skeleton: loop-top wait → first HG run → DEPBAR → "
                "EW segments per depth → EXIT.\n"
                "% = the segment's share of the loop body's stall samples; stalls = the dominant "
                "stall of the largest segment inside the anchor "
                "(`category:percent`). Reading: a same-segment share clearly above the "
                "reference's, with wait classes "
                "(barrier/long_sb/short_sb) dominating on the student side while dispatch / "
                "selection classes dominate the reference's same segment ⇒ the structure is "
                "right and the intra-segment instruction scheduling is worse (G5 shape; do not "
                "touch the structure).\n"
                f"```\n{aligned}\n```\n"
                f"\n### student {priced.ledger_tree} kernel raw segment ledger (measured this "
                f"time)\n"
                f"```\n{student_segments[:4000]}\n```\n")
        else:
            diagnose_extra += (
                "\n## Per-instruction stall ledger (read before judging — `deficient_locus` "
                "must match the ledger's big items)\n"
                "Ledger rows are sorted by cost, descending, accumulated to ≥90% coverage. The "
                "defect locus you name must explain the big items at the top of the ledger; if you "
                "consider them immovable and pick a locus outside the ledger, `evidence` must "
                "argue explicitly why every item at the top of the ledger is immovable.\n"
                f"\n### student {priced.ledger_tree} kernel (measured this time)\n"
                + (f"```\n{student_ledger}\n```\n" if student_ledger
                   else "(Ledger unavailable this round — the collection job failed or its "
                        "self-check did not pass. Fall back to diagnosing from the aggregate "
                        "signature, and lower `confidence` by one notch accordingly.)\n")
                + "\n### Reference (collected at recon time)\n"
                + (f"```\n{ref_ledger[:6000]}\n```\n" if ref_ledger
                   else "(Recon left no reference ledger.)\n"))
    if audit and (ref_dossier := _read_ref_orch_dossier(ctx)):
        # Paper-level contract check: the audit holds the blueprint's
        # [SYNC:]/[GATHER:]/[PHASE:] adjudications against the reference's
        # machine-read dossier — whether the blueprint transcribed the
        # reference's structure or guessed it (the two inertial wrong answers
        # §2g names). The BUILT structure is audited by the dossier diff block
        # above (the judged tree's own dossier, priced this entry); this block
        # is the blueprint's side of that reading.
        diagnose_extra += (
            "\n## Reference orchestration dossier (machine-read at recon time, static SASS "
            "transcription — for the audit's ruling on the blueprint)\n"
            "Check the milestone's [SYNC:]/[GATHER:]/[PHASE:] rulings one by one against it: "
            "where the sync topology / supply organisation the blueprint declares agrees with "
            "this dossier, record CLEAN; where it disagrees, or the blueprint is vague while the "
            "dossier has a clear reading, bounce it with a mechanism attribution — guessing a "
            "symmetric rendezvous for the sync topology and per-row inline loads for the supply "
            "were the two inertial answers from before there was a dossier; with the dossier "
            "there is nothing left to guess.\n"
            f"```\n{ref_dossier[:6000]}\n```\n")
    if priced is not None:
        # a group card: the same readings on its other meta workloads (one job each)
        diagnose_extra += _meta_workload_readings(ctx, priced, "audit" if audit else "plateau")
    lever_table: list = []
    if priced is not None:
        # plateau OR audit: the audit is where structural reopens are decided,
        # and a reopen is chosen against the levers it passes over — those must
        # carry measured prices there too, not only after a later plateau.
        lever_table = _run_lever_pricing(
            ctx, priced, bindings, diagnose_extra,
            prompt_file="m0_audit_prompt.md" if audit else "diagnose_prompt.md")
        if lever_table:
            diagnose_extra += _lever_price_block(lever_table)
        elif ctx.lever_price_lost:
            # a job was bought and its table lost (archive / tool): said once,
            # so the teacher does not read this entry as "pricing never ran"
            # and cite priced_by= for levers it just designed
            diagnose_extra += _lever_price_lost_block(ctx.lever_price_lost)
            ctx.lever_price_lost = ""
    if getattr(ctx, "discarded_edits", ""):
        # said once: a pricing turn (just now) or an earlier verdict-probe reset
        # threw away edits outside notes/ that the teacher may believe landed
        diagnose_extra += f"\n## Edits discarded by a reset\n{ctx.discarded_edits}\n"
        ctx.discarded_edits = ""
    if revs := [rv for rv in _unretired_revisions(ctx) if rv.get("verdict") in _REVISION_FLAGGED]:
        # what the inner loop found when the best moved under the milestone's
        # cited prices: the teacher reads it here and must act on the suspended
        # rungs; a fresh table on the current best supersedes (retires) it
        diagnose_extra += _price_revision_teacher_block(revs)
        # shown once: the milestone's RE-PRICED / SUSPENDED markers are the
        # durable record, the standing student note is retired here whether or
        # not this entry lists levers again
        _retire_revisions(ctx)
    # every lever this run has priced: a citation of one from an earlier step
    # (the reflection lists them as priced) verifies against the same readings
    known_levers = _merge_lever_tables(prior_records, lever_table)
    # PHASE EXHAUSTION: the table priced on the current best is a batch
    # measurement of the phase's own knobs; when they all read null the phase's
    # remaining walk is a re-measurement, and the diagnosis must say so (close
    # the phase, or name the one unpriced decision still worth walking).
    exhaustion = _phase_exhaustion(ctx, live_records, lever_table) if lever_table else None
    if exhaustion and ctx.phase_close and ctx.phase_close.get("gate") == exhaustion["gate"]:
        _log(f"meta: {exhaustion['phase']} is already closed on the price table — "
             + ("its gate row failed on this same best tree and the table has not "
                "changed; not asking again (the diagnosis attributes the FAIL)"
                if ctx.phase_close.get("stage") == "gate-failed"
                else f"awaiting its {exhaustion['gate']} row, not asking again"))
        exhaustion = None
    if exhaustion:
        _log(f"meta: {exhaustion['phase']} levers {exhaustion['levers']} all null/noop on the "
             f"best tree -> phase exhausted; demanding a phase_verdict "
             f"(unpriced: {exhaustion['unpriced'] or 'none'})")
        diagnose_extra += _phase_exhausted_block(exhaustion)
    prompt_file = "m0_audit_prompt.md" if audit else "diagnose_prompt.md"
    enum_help = ("`OK` `G1` `G2` `G3` `G4` `G5` `R1` `R2`" if audit
                 else "`G1` `G2` `G3` `G4` `G5` `R1` `R2` `INFEASIBLE`")
    diagnosis, cause = "", ""
    probe_rounds = 0
    probe_readings: list = []       # every contract line the probes printed, for the ledger
    reach_demanded = False
    price_demanded = False
    capped_demanded = False
    phase_demanded = False
    seg_demanded = False
    infeasible_asked = False
    infeasible_bounces: list[str] = []   # refusal reasons put back to the teacher this step
    diagnoses = _plateau_diagnoses(prior_records) + 1
    stop_line = None if audit else _resolve_target(ctx)
    # VERDICT PROBES: before committing to a cause_class the diagnose agent may run
    # bounded GPU experiments against the student's own tree (PROBE_REQUEST, the same
    # marker and submission plumbing as the student's probe sub-loop and the KB
    # certification — the agent decides what to ask, the driver owns the GPU). This is
    # the cheap falsification channel the meta loop lacked: without it a diagnosis
    # hypothesis was only ever tested at whole-inner-segment granularity, judged on
    # scalar best-progress. Probe turns and malformed-enum turns each draw on their own
    # budget; the loop bound is the sum, so a stubborn prober still terminates.
    pre_probe_head = be.head()
    probes_run: list = []           # paths, so a probe with no verdict line is "ran, no reading"
    for _ in range(max(1, ctx.max_patch_rounds) + ctx.max_verdict_probe_rounds):
        out = _teacher(be).run_agent(render_prompt(
            _meta_prompt(ctx, prompt_file), bindings, extra=diagnose_extra))
        cause = _parse_cause_class(out, allow_ok=audit)
        # A turn carrying a valid cause_class IS the final diagnosis, even when the
        # PROBE_REQUEST marker also appears in its text: the probe rules ask the
        # diagnosis to cite probe names + readings in `evidence`, and the marker regex
        # is a whole-text search — reading such a citation as a request would burn a
        # GPU job re-running an old script and discard a valid diagnosis (the same
        # trap the student-turn parser documents for a stray RECON_REQUEST echo).
        turn = None if cause else parse_student_turn_safe(out)
        if isinstance(turn, ProbeRequest) and _is_timeline_request(turn.path):
            if not _timeline_on(ctx):
                diagnose_extra += ("\n## Timeline probe not enabled\nThis run's loop.toml has "
                                   "[loop] timeline_probe off, so `PROBE_REQUEST: timeline` is not "
                                   "available; use a verdict probe or the pricing table.\n")
                continue
            if tl_table is not None:
                # the entry already read this tree: the block is in the prompt
                diagnose_extra += ("\n## Timeline\nThis step's entry already read the table on "
                                   "the same tree (the \"Timeline\" block above); not re-run — "
                                   "diagnose from it.\n")
                continue
            if probe_rounds >= ctx.max_verdict_probe_rounds:
                _log("meta: verdict-probe budget spent -> demanding a diagnosis")
                diagnose_extra += ("\n## Verdict-experiment budget exhausted\nNo more PROBE_REQUEST "
                                   "may be sent. Give the final diagnosis from the evidence you have; "
                                   "mark every unverified hypothesis in it explicitly as unverified.\n")
                continue
            probe_rounds += 1
            tl_ref = (priced.best_sha or priced.head_sha) if priced is not None else be.head()
            tl_table = _run_timeline(ctx, tl_ref, f"meta{meta_step}:teacher")
            if tl_table and _timeline_renderable(tl_table):
                diagnose_extra += _timeline_block(tl_table, _ref_region_lines(ctx))
            else:
                diagnose_extra += _timeline_feedback(tl_table)
                tl_table = None
            continue
        if isinstance(turn, ProbeRequest):
            if probe_rounds >= ctx.max_verdict_probe_rounds:
                _log("meta: verdict-probe budget spent -> demanding a diagnosis")
                diagnose_extra += ("\n## Verdict-experiment budget exhausted\nNo more PROBE_REQUEST "
                                   "may be sent. Give the final diagnosis from the evidence you have; "
                                   "mark every unverified hypothesis in it explicitly as unverified.\n")
                continue
            probe_rounds += 1
            probes_run.append(turn.path)
            probe_branch = f"{ctx.branch_prefix}-metaprobe"
            if turn.path.lower().endswith(".json"):
                # an ablation spec instead of a script: the driver's frozen
                # pricing recipe runs it (NCU clock, base re-timed alongside),
                # so the experiment cannot fail on the teacher's harness code
                # — the NO VERDICT shape a hand-written probe keeps taking
                rows, why, probe_log = _run_ablation_probe(ctx, be, turn.path, probe_branch)
                _log(f"meta: ablation probe {turn.path} "
                     f"({probe_rounds}/{ctx.max_verdict_probe_rounds}) -> "
                     + ("; ".join(f"{r['name']} {_fmt_pct(r['delta_pct'])}" for r in rows)
                        if rows else f"NO VERDICT ({why})"))
                diagnose_extra += _ablation_probe_feedback(
                    turn.path, rows, why, probe_log, probe_rounds, ctx.max_verdict_probe_rounds)
                noise = next((r.get("noise_pct") for r in rows if r.get("noise_pct") is not None),
                             _LEVER_NOISE_DEFAULT_PCT)
                probe_readings.extend(
                    {"path": turn.path, "round": probe_rounds, "contract": r["name"],
                     # the hypothesis "this mechanism is charging": confirmed only by a
                     # positive delta above the floor; within the floor or NEGATIVE
                     # (removing it is slower — it helps) falsifies it
                     "verdict": ("PASS" if r["delta_pct"] is not None and r["delta_pct"] > noise
                                 else "FAIL" if r["delta_pct"] is not None else "NO VERDICT"),
                     "reading": _fmt_pct(r["delta_pct"]) if r["delta_pct"] is not None else "unreported",
                     "note": (f"ablation (ncu clock, correctness={r.get('correctness')}, "
                              f"status={r.get('status')}): {r.get('hypothesis', '')}"
                              + (" — removing it is SLOWER: the mechanism helps"
                                 if r["delta_pct"] is not None and r["delta_pct"] < -noise else ""))}
                    for r in rows)
                # the rows join the step's price table: `ablation:<name>` cites
                # them, the record banks them, later steps see them as priced
                lever_table.extend(rows)
                known_levers.extend(rows)
                continue
            be.commit(f"meta-probe: {turn.path}")
            be.push_branch(probe_branch)
            probe_log = be.submit_gpu(probe_branch, f"cmd:python3 {turn.path}")
            # audit=False: these lines answer pre-registered hypotheses, and the
            # max_diff field carries the reading as printed. The certification
            # audit (error-norm ceiling, float-only) was downgrading a PASS with a
            # reading table to FAIL — i.e. telling the teacher its hypothesis had
            # been falsified when the experiment had confirmed it.
            pv = parse_probe_verdict(probe_log, audit=False)
            _log(f"meta: verdict probe {turn.path} "
                 f"({probe_rounds}/{ctx.max_verdict_probe_rounds}) -> "
                 f"PROBE {_probe_label(pv)} (max_diff={pv.diff_text})"
                 f"{_probe_log_suffix(pv)}")
            diagnose_extra += _verdict_probe_feedback(
                turn.path, pv, probe_log, probe_rounds, ctx.max_verdict_probe_rounds)
            probe_readings.extend(
                {"path": turn.path, "round": probe_rounds, "contract": c.index,
                 "verdict": c.verdict, "reading": c.reading, "note": c.note}
                for c in pv.contracts)
            continue
        diagnosis = out
        if cause:
            # REACHABILITY GATE (plateau only, ledger channel present): the floor
            # arithmetic (diagnose_prompt §1c) is what turns "levers exhausted" into
            # "architecture cannot reach target" while the evidence is fresh — the
            # observed: a DSA run derived it 6 meta steps late and lost the rebuild to the
            # shutdown. One re-prompt, then the diagnosis is taken as-is: a missing
            # line must never cost more than one turn of the shared budget.
            if (not audit and ctx.stall_pricing and not reach_demanded
                    and not _parse_reachability(out)):
                reach_demanded = True
                _log("meta: diagnosis omitted the reachability line -> re-prompt")
                diagnose_extra += (
                    "\n## Format requirement (binding)\nThe output block lacks a `reachability:` "
                    "line. Do the reachability refutation of diagnosis step 1c first (the "
                    "construction of `floor_opt` and its term-by-term arithmetic go in "
                    "`evidence`), then resend the **complete** diagnosis "
                    "(every field, not just the one missing line).\n")
                continue
            # INFEASIBLE is a stop request: none of the patch-shaping gates below
            # (segment / reopen price / phase verdict) apply to it.
            if cause.upper() == "INFEASIBLE":
                # INFEASIBLE GATE: one bounce naming the conditions; a request
                # that still fails them is refused after the loop.
                why = _infeasible_refusal(out, diagnoses=diagnoses,
                                          min_diagnoses=ctx.infeasible_min_diagnoses,
                                          target_ms=stop_line)
                if why and not infeasible_bounces:
                    infeasible_bounces.append(why)
                    infeasible_asked = True  # the refusal block already names the channel
                    _log(f"meta: INFEASIBLE stop request not accepted ({why}) -> re-prompt")
                    diagnose_extra += _infeasible_refused_block(why)
                    continue
                break
            # INFEASIBLE CHANNEL: a diagnosis that talks INFEASIBLE under another
            # class is how a run spins — the verdict lives in the milestone, the
            # driver never hears it, and every later meta step re-diagnoses a
            # "none" locus until the budget is gone. One bounce makes the teacher
            # choose: the stop request, or the class it wrote (a quote / a negation).
            #
            # Not when the diagnosis's own reachability line reads OPEN: the gate
            # refuses an INFEASIBLE stop on OPEN anyway (_infeasible_refusal), so the
            # bounce could only buy a turn that ends where this one did. §1c makes
            # every diagnosis argue reachability against the stop rule, so the word
            # is in nearly all of them as a negation — in one large batch of
            # runs, 1157 of 1163 such diagnoses read OPEN and 695 teacher
            # turns went to the bounce.
            if (not audit and not infeasible_asked
                    and _INFEASIBLE_MENTION.search(out)
                    and _parse_reach_verdict(_parse_reachability(out)) != "OPEN"):
                infeasible_asked = True
                _log(f"meta: diagnosis mentions INFEASIBLE under cause_class {cause} "
                     f"-> re-prompt (stop request or not)")
                diagnose_extra += _INFEASIBLE_MENTION_BLOCK
                continue
            # LOCUS SEGMENT: a diagnosis read against a timeline names the segment
            # the gap sits in — the reading that makes the table binding.
            if tl_table is not None and not seg_demanded and not _parse_locus_segment(out):
                seg_demanded = True
                _log("meta: diagnosis omitted the locus_segment line -> re-prompt")
                diagnose_extra += (
                    "\n## Format requirement (binding)\nThis step has a timeline table, and the "
                    "output block lacks a `locus_segment:` line. Use the table to place the lesion "
                    "in a specific segment: `locus_segment: <role>.<segment>` (if it cannot be "
                    "placed in a segment, write `none` and explain in evidence), "
                    "then resend the **complete** diagnosis (every field, not just the one missing "
                    "line).\n")
                continue
            # REOPEN BUDGET GATE: the reopen budget is spent — an R1 cannot land,
            # so one bounce asks for a class that can (before the price gate
            # would spend turns pricing a reopen the driver is going to refuse)
            if cause.upper() == "R1" and reopens_capped and not capped_demanded:
                capped_demanded = True
                _log(f"meta: R1 diagnosed with the reopen budget spent ({reopen_streak} "
                     f"fruitless reopen(s) in a row) -> re-prompt")
                diagnose_extra += _reopen_capped_block(reopen_streak, ctx.reopen_max_fruitless,
                                                       bounce=True)
                continue
            # REOPEN PRICE GATE (plateau and audit): an R1 is a structural reopen,
            # and a reopen without a measured price is a known dsa_sparse failure —
            # the stage split axis flipped on an inferred masking window, six
            # decisions redone for 2%. One bounce naming the three ways to price
            # it (verdict probe / price table / certified KB); a second unpriced
            # R1 is refused below, after the loop.
            if (cause.upper() == "R1" and _reopen_gate_on(ctx) and not price_demanded
                    and (why := _verify_reopen_price(_parse_reopen_price(out), probe_readings,
                                                     known_levers, ctx, probes_run,
                                                     best_ref=_priced_best(priced)))):
                price_demanded = True
                _log(f"meta: R1 reopen carries no measured price ({why}) -> re-prompt")
                diagnose_extra += _UNPRICED_REOPEN_BLOCK + f"(driver check result: {why})\n"
                continue
            # PHASE VERDICT GATE: the exhaustion reading was put to the teacher;
            # an answer that is neither close nor a walkable decision costs one
            # bounce, then the driver closes the phase itself (below).
            if (exhaustion and not phase_demanded
                    and (why := _verify_phase_verdict(_parse_phase_verdict(out), exhaustion))):
                phase_demanded = True
                _log(f"meta: phase_verdict for {exhaustion['phase']} missing or invalid "
                     f"({why}) -> re-prompt")
                diagnose_extra += (
                    f"\n## Phase ruling (binding)\nThe driver has ruled that {exhaustion['phase']}'s "
                    f"levers are all null on the pricing table, but the diagnosis's `phase_verdict` "
                    f"is not valid ({why}). Add `phase_verdict: close` "
                    f"(the student submits the `{exhaustion['gate']}` row directly next round) or "
                    f"`phase_verdict: continue:<D id>` "
                    f"(only a decision with no priced lever: "
                    f"{', '.join(exhaustion['unpriced']) or 'none — this phase has no such decision, so only close is possible'}), "
                    f"and resend the **complete** diagnosis. If it is missing or invalid again, the "
                    f"driver treats it as close.\n")
                continue
            break
        # ENUM GATE: cause_class must be exactly one enum token (no prose). Re-prompt.
        _log("meta: diagnosis had no valid cause_class enum -> re-prompt")
        diagnose_extra += (f"\n## Format requirement (binding)\nThe `cause_class` field must be "
                           f"exactly one of the enum values: "
                           f"{enum_help}, with no prose / commentary after it.\n")
    if probe_rounds:
        # Probe commits are add -A snapshots of whatever the diagnose agent touched —
        # not just notes/<name>.py — and every downstream branch (patch resume via
        # head(), the audit-clean and kb-example returns) continues from HEAD, so
        # without this reset a stray teacher edit to the kernel would enter the
        # student lineage silently under a "meta-probe:" message, unrecorded and
        # ungated. The -metaprobe branch keeps the scripts for provenance and the
        # readings are already in the prompt; the student line stays teacher-free.
        if lost := _edits_outside_notes(be, pre_probe_head):
            _note_discarded_edits(ctx, "meta (verdict-probe reset)", lost)
        be.reset(pre_probe_head)
        _log(f"meta: workspace reset to pre-probe head {pre_probe_head} "
             f"(probe commits preserved on the -metaprobe branch)")
    # the id alone: teachers append a parenthesised gloss ("D0.19 (C side…", or with a
    # fullwidth paren and no space) that the \S+ field parser kept, so the ledger and
    # memory rows read "D0.19(C"
    locus = re.split(r"[（(]", _parse_field(diagnosis, "deficient_locus"))[0]
    direction = _parse_line(diagnosis, "patch_direction")
    reachability = _parse_reachability(diagnosis)
    if reachability:
        _log(f"meta: reachability -> {reachability}")
    _log(f"meta{'/m0-audit' if audit else ''}: diagnosed -> {cause or '?'} @ {locus or '?'}")

    def _record(rec: MetaRecord) -> None:
        # every record of this step carries the stop requests refused on the
        # way (audit trail: a bounced INFEASIBLE that became a patch is still
        # visible in meta_status.jsonl)
        rec.infeasible_refusals = list(infeasible_bounces)
        if ctx.meta_status_path:
            append_meta_record(ctx.meta_status_path, rec)

    refused = ""
    if cause.upper() == "INFEASIBLE":
        refused = _infeasible_refusal(diagnosis, diagnoses=diagnoses,
                                      min_diagnoses=ctx.infeasible_min_diagnoses,
                                      target_ms=stop_line)
    if refused:
        # INFEASIBLE GATE, second strike: the request still fails the check after
        # the bounce. Not a stop: nothing is patched, the refusal enters the
        # memory the next diagnosis reads, and the caller re-diagnoses the same
        # state (meta budget permitting) — the UNPRICED_REOPEN shape.
        if refused not in infeasible_bounces:
            infeasible_bounces.append(refused)
        _log(f"meta: INFEASIBLE stop request REFUSED ({refused}); nothing patched, "
             f"the run continues (refusals this step: {len(infeasible_bounces)})")
        _record(MetaRecord(
            meta_step=meta_step, cause_class="INFEASIBLE", deficient_locus=locus,
            patch_class="infeasible-refused", changed_entries=[], patch_direction=direction,
            outcome="INFEASIBLE_REFUSED", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table))
        return MetaResult(diagnosis, "infeasible-refused", [], None, [],
                          lever_prices=lever_table)

    if cause.upper() == "INFEASIBLE":
        # The stop request: nothing is patched and no segment resumes — the
        # caller ends the run, and the stop is the run's verdict.
        notice = _write_infeasible_notice(ctx, meta_step=meta_step, diagnosis=diagnosis,
                                          reachability=reachability, best=best_before)
        _record(MetaRecord(
            meta_step=meta_step, cause_class="INFEASIBLE", deficient_locus=locus,
            patch_class="infeasible-stop", changed_entries=[], patch_direction=direction,
            outcome="INFEASIBLE_STOP", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table))
        _log("meta: teacher declared INFEASIBLE -> stopping the run; the stop is the "
             "run's verdict (stop=infeasible; no guard restarts it); notice -> "
             + (str(notice) if notice else "<unwritten, see this log>"))
        return MetaResult(diagnosis, "infeasible-stop", [], None, [],
                          lever_prices=lever_table)

    # The phase verdict, resolved: the teacher's answer, or close by default —
    # the student submits the gate row next either way (a `continue` narrows
    # the walk to the one decision named; the rest of the phase stays priced).
    phase_verdict, phase_close = "", None
    if exhaustion:
        answered = _parse_phase_verdict(diagnosis)
        if (why := _verify_phase_verdict(answered, exhaustion)):
            phase_verdict = "close (default: unanswered)"
            _log(f"meta: phase_verdict for {exhaustion['phase']} still {why} after the bounce "
                 f"-> closed by default")
        else:
            phase_verdict = answered
        phase_close = {"gate": exhaustion["gate"], "phase": exhaustion["phase"],
                       "verdict": phase_verdict, "levers": list(exhaustion["levers"]),
                       "note": _phase_close_note(exhaustion, phase_verdict)}
        _log(f"meta{'/m0-audit' if audit else ''}: {exhaustion['phase']} phase_verdict = "
             f"{phase_verdict} -> the student "
             + (f"walks {phase_verdict.split(':', 1)[1]} then submits {exhaustion['gate']}"
                if phase_verdict.startswith("continue:")
                else f"submits {exhaustion['gate']} next"))

    if audit and cause.upper() == "OK":
        # The milestone survived the reconciliation: every severe/elevated flag is
        # either scheduled with a sane price in the milestone or a student-side
        # matter already routed. Nothing to patch; the segment resumes where it was.
        _record(MetaRecord(
            meta_step=meta_step, cause_class="OK", deficient_locus="",
            patch_class="audit-clean", changed_entries=[], patch_direction=direction,
            outcome="AUDIT_CLEAN", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table,
            phase_verdict=phase_verdict))
        return MetaResult(diagnosis, "audit-clean", [], None, [],
                          lever_prices=lever_table, phase_close=phase_close)

    reopen_price = _parse_reopen_price(diagnosis)
    if cause.upper() == "R1" and reopens_capped:
        # the bounce named the budget; the teacher insisted: nothing is patched
        _log(f"meta{'/m0-audit' if audit else ''}: R1 @ {locus or '?'} REFUSED — reopen "
             f"budget spent ({reopen_streak} fruitless reopen(s) in a row, "
             f"[loop] reopen_max_fruitless = {ctx.reopen_max_fruitless}); nothing patched")
        _record(MetaRecord(
            meta_step=meta_step, cause_class=cause, deficient_locus=locus,
            patch_class="reopen-capped", changed_entries=[], patch_direction=direction,
            outcome="REOPEN_CAPPED", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table,
            reopen_price=reopen_price, phase_verdict=phase_verdict))
        return MetaResult(diagnosis, "reopen-capped", [], None, [],
                          lever_prices=lever_table, phase_close=phase_close)
    if cause.upper() == "R1" and _reopen_gate_on(ctx):
        why = _verify_reopen_price(reopen_price, probe_readings, known_levers, ctx, probes_run,
                                   best_ref=_priced_best(priced))
        if why:
            # The bounce above already named the routes; the teacher insisted (or
            # spent the turns without pricing). No reopen without a price: nothing
            # is patched, the refusal enters the memory the next diagnosis reads.
            _log(f"meta{'/m0-audit' if audit else ''}: R1 @ {locus or '?'} REFUSED — "
                 f"no measured price ({why}); nothing patched")
            _record(MetaRecord(
                meta_step=meta_step, cause_class=cause, deficient_locus=locus,
                patch_class="unpriced-reopen", changed_entries=[], patch_direction=direction,
                outcome="UNPRICED_REOPEN", probe_rounds=probe_rounds,
                probe_readings=probe_readings, lever_prices=lever_table,
                reopen_price=reopen_price, phase_verdict=phase_verdict))
            return MetaResult(diagnosis, "unpriced-reopen", [], None, [],
                              lever_prices=lever_table, phase_close=phase_close)

    if cause.upper() in ("G3", "G5"):
        # interface gap (G3) or in-segment ILP (G5: structure right, instruction-
        # level scheduling behind the ref's) -> the fix is a GPU-verified hw_kb
        # recipe, NOT a milestone edit; a KB addition is additive (no milestone
        # checkpoint invalidated) and the structure stays frozen.
        kb = run_kb_certify(ctx, diagnosis)
        _record(MetaRecord(
            meta_step=meta_step, cause_class=cause, deficient_locus=locus,
            patch_class="kb-example", changed_entries=[], patch_direction=direction,
            outcome="KB_CERTIFIED" if kb.certified else "KB_UNCERTIFIED",
            certified=kb.certified, probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table,
            phase_verdict=phase_verdict))
        directive = None
        if cause.upper() == "G5":
            directive = (
                f"- The previous meta step ruled **G5 intra-segment ILP** (the structure is all "
                f"right, the intra-segment instruction scheduling is worse), locus = "
                f"`{locus or '?'}`, direction: {direction or '(not given)'}. "
                "**Do not touch the M0 structure and do not change the milestone**: reorder "
                "instructions inside that segment (chain depth / interleaving two adjacent rows / "
                "the distance between a long-latency instruction and its consumer / the work "
                "distribution across a drain)"
                + (", together with the KB recipe just certified" if kb.certified
                   else " (this step's KB recipe could not be certified; only this direction is given)")
                + ". Acceptance = NCU `t_ns` goes down + that segment's stall share converges "
                "toward the reference; the single-shot absolute performance line is not the "
                "judge.\n")
        return MetaResult(diagnosis, "kb-example", [], None, [], certified=kb.certified,
                          recipe=kb.recipe, student_directive=directive,
                          lever_prices=lever_table, phase_close=phase_close)

    # HARNESS GATE: the patch must actually reach the file, and the patched milestone
    # must stay §9-conformant. Re-prompt the patch agent with the violations until clean
    # or budget spent (symmetric to the student's probe sub-loop and the KB certify loop).
    at_entry = _read_milestone(ctx)
    patch_out, violations = "", []
    for _ in range(max(1, ctx.max_patch_rounds)):
        extra = f"\n\n## DIAGNOSIS\n{diagnosis}\n"
        if violations:
            extra += ("\n## Violations in your previous patch (fix them and resend)\n"
                      + "\n".join(f"- {v}" for v in violations) + "\n")
        patch_out = _teacher(be).run_agent(render_prompt(_meta_prompt(ctx, "patch_prompt.md"), bindings, extra))
        violations = _patch_landing_violations(
            at_entry, _read_milestone(ctx), _parse_list(patch_out, "changed_entries"))
        violations += _validate_milestone_file(ctx.milestone, baseline=at_entry)
        if not violations:
            break
        _log(f"meta: patch has {len(violations)} violation(s) -> re-prompt: {violations[0]}")
    for w in _dependency_warnings_file(ctx.milestone):
        # advisory in the patch mode (a ⇐ hole / cycle / one-sided ⇒): logged
        # so the closure arithmetic below can be read against it, never
        # re-prompted — see milestone.dependency_warnings
        _log(f"meta: milestone DAG warning — {w}")
    changed = _parse_list(patch_out, "changed_entries")
    patch_class = _parse_field(patch_out, "patch_class")
    if patch_class in ("structural", "split") and reopens_capped:
        # a non-R1 diagnosis whose patch reopens delivered decisions anyway: the
        # same budget — the landed patch is withdrawn before any student turn
        _write_milestone(ctx, at_entry)
        _log(f"meta: {patch_class} patch @ {locus or '?'} REFUSED — reopen budget spent "
             f"({reopen_streak} fruitless reopen(s) in a row); milestone restored, "
             f"nothing patched")
        _record(MetaRecord(
            meta_step=meta_step, cause_class=cause, deficient_locus=locus,
            patch_class="reopen-capped", changed_entries=changed, patch_direction=direction,
            outcome="REOPEN_CAPPED", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table,
            reopen_price=reopen_price, phase_verdict=phase_verdict))
        return MetaResult(diagnosis, "reopen-capped", [], None, [],
                          lever_prices=lever_table, phase_close=phase_close)
    if (patch_class in ("structural", "split") and _reopen_gate_on(ctx)
            and (why := _verify_reopen_price(reopen_price, probe_readings, known_levers, ctx,
                                             probes_run, best_ref=_priced_best(priced)))):
        # The diagnosis was not R1 (that gate ran above) but the patch reopens the
        # closure anyway: same rule. The landed patch is withdrawn — an unpriced
        # reopen never reaches the student — and the step is recorded as such.
        _write_milestone(ctx, at_entry)
        _log(f"meta: {patch_class} patch @ {locus or '?'} REFUSED — a reopen without a "
             f"measured price ({why}); milestone restored, nothing patched")
        _record(MetaRecord(
            meta_step=meta_step, cause_class=cause, deficient_locus=locus,
            patch_class="unpriced-reopen", changed_entries=changed, patch_direction=direction,
            outcome="UNPRICED_REOPEN", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table,
            reopen_price=reopen_price, phase_verdict=phase_verdict))
        return MetaResult(diagnosis, "unpriced-reopen", [], None, [],
                          lever_prices=lever_table, phase_close=phase_close)
    gate_contract = _parse_gate_contract(patch_out)
    if gate_contract:
        _log("meta: patch declares gate contract "
             + ", ".join(f"{g} ≤ {v * 100:+.0f}%" for g, v in gate_contract.items()))
    resume_after, invalidated = _compute_resume_plan(ctx, changed)
    reopened, kept = ((_compute_reopen_plan(ctx, changed))
                      if patch_class in ("structural", "split") else ([], []))
    if violations:
        _log(f"meta: WARNING patch still invalid after {ctx.max_patch_rounds} "
             f"round(s): {violations[:3]}")
    after = _read_milestone(ctx)
    _record(MetaRecord(
        meta_step=meta_step, cause_class=cause, deficient_locus=locus,
        patch_class=patch_class, changed_entries=changed, patch_direction=direction,
        milestone_before=milestone_before, resume_after=resume_after,
        best_before=best_before, outcome="PENDING", milestone_violations=violations,
        patch_landed=after != at_entry, milestone_delta=_line_delta(at_entry, after),
        probe_rounds=probe_rounds, probe_readings=probe_readings,
        lever_prices=lever_table, reopen_price=reopen_price,
        gate_contract=gate_contract, phase_verdict=phase_verdict))
    return MetaResult(diagnosis, patch_class, changed, resume_after, invalidated,
                      gate_contract=gate_contract,
                      certified=None, milestone_violations=violations,
                      reopened=reopened, kept=kept,
                      lever_prices=lever_table, phase_close=phase_close)


def _apply_phase_close(ctx: LoopContext, meta: "MetaResult") -> None:
    """Carry a meta step's phase verdict to the student's turns. A structural /
    split patch replaces the premise the verdict was read on (the phase's
    decisions are being redone), so it retires any standing notice instead."""
    if meta.patch_class in ("structural", "split"):
        if ctx.phase_close:
            _log(f"meta: phase-close notice for {ctx.phase_close['phase']} retired — "
                 f"a {meta.patch_class} patch reopens the phase's premise")
        ctx.phase_close = None
        return
    if meta.phase_close:
        ctx.phase_close = dict(meta.phase_close)
        _log(f"meta: phase-close notice for {meta.phase_close['phase']} rides the next "
             f"student turns until {meta.phase_close['gate']} is submitted")


# a milestone entry id the patch agent can claim to have touched; §-sections and prose
# labels ("Worked Example") are not checkable this way, so only these are.


def _patch_landing_violations(before: str, after: str, declared: list[str]) -> list[str]:
    """Did the patch reach the file, and does the file contain what the reply claims?

    Nothing else writes the milestone: the driver never applies the `patch:` text from
    the agent's reply, so an agent that *reports* its patch instead of *making* it leaves
    the milestone untouched. Every downstream signal still looks healthy — the milestone
    is schema-clean (it is unchanged), the student correctly answers that nothing is left
    to do, and the loop charges a whole inner segment to a patch that never existed. This
    happened twice in one run (meta steps 3 and 4, ~4h) before anything noticed.
    """
    if after == before:
        return ["The patch never reached the file: the milestone file is byte-identical to "
                "before the patch. You must rewrite the milestone file itself with Edit/Write — the "
                "`patch:` text in your reply is never applied by anyone."]
    # An id in `before` but not `after` was deleted, which a structural patch may do.
    # An id in neither was invented for the report: the student can never act on it.
    return [f"`changed_entries` declares {e}, but the milestone file does not contain it "
            f"(neither a new nor an existing entry)"
            for e in (d.strip() for d in declared)
            if _DECISION_ID.match(e) and e not in after and e not in before]


def _line_delta(before: str, after: str) -> str:
    """"+38/-9 lines" — enough to audit a patch step later without storing both texts."""
    import difflib
    diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm=""))
    add = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    rem = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
    return f"+{add}/-{rem} lines"


def _validate_milestone_file(milestone, authoring: bool = False,
                             baseline: str | None = None) -> list[str]:
    """Violations in the milestone file — a missing or unreadable file is itself one.

    Absence is the pre-distillation state, not a conformant milestone. Reporting it
    as clean would tell _needs_authoring the milestone is already written, so the
    loop would start with nothing for the student to read; it would equally let a
    patch that deleted the file pass the post-patch gate.

    ``authoring`` selects the validator's strict mode (orchestration tags
    mandatory). Resume gates and patch validation stay in the default mode:
    a schema addition must not make every pre-existing milestone read as "not
    distilled yet" — that path re-authors v1 over the accumulated meta patches
    and re-measures baselines the resume contract promises to reuse — and a
    patch round must spend its budget on the diagnosed problem, not on
    retrofitting tags.
    """
    from forgekernel.milestone import validate_milestone
    p = Path(milestone)
    try:
        return validate_milestone(p.read_text(encoding="utf-8"),
                                  authoring=authoring, baseline=baseline)
    except OSError as e:
        return [f"milestone file unreadable ({p}): {type(e).__name__}"]


def _dependency_warnings_file(milestone: Path) -> list[str]:
    from forgekernel.milestone import dependency_warnings
    try:
        return dependency_warnings(Path(milestone).read_text(encoding="utf-8"))
    except OSError:
        return []


def _seed_target_ms(ctx: LoopContext) -> None:
    """If the authored milestone §0 lacks a machine-readable target_ms, seed it from
    the CLI --target-ms so the target SSOT lives in the milestone from the start and
    the meta agent has a concrete number to re-baseline.

    Seeded together with ``target_window:`` — the number alone cannot say which
    timing window it is anchored in, and _resolve_target only trusts a §0 target
    whose annotation matches the run's gate window."""
    from forgekernel.milestone import parse_target_ms
    try:
        p = Path(ctx.milestone)
        if not p.exists():
            return
        text = p.read_text(encoding="utf-8")
        if parse_target_ms(text) is not None:
            return  # meta already wrote one
        seeded = re.sub(r"(## 0\.[^\n]*\n)",
                        rf"\1target_ms: {ctx.target_ms}\ntarget_window: {ctx.gate_window}\n",
                        text, count=1)
        if seeded != text:
            p.write_text(seeded, encoding="utf-8")
            _log(f"meta/distill: seeded target_ms: {ctx.target_ms} "
                 f"(target_window: {ctx.gate_window}) into §0")
    except OSError:
        pass


def _resolve_target(ctx: LoopContext) -> float:
    """Stop threshold: the milestone's machine-readable target_ms (SSOT) if present
    AND anchored in this run's gate window, else the CLI-derived seed.

    Reading it from the milestone is what makes the meta agent's §0 target
    re-baseline actually move the bar (it was inert before). The window check is
    what keeps that SSOT honest: a teacher following the e2e-era prompts writes
    ``arch_ref_ms/0.95`` — an e2e number — into §0, and a device-gated loop
    comparing device readings against it gates ~one host tax loose, so the first
    mediocre KEEP stops the run as "target". An unannotated target counts as e2e
    (every pre-annotation milestone is e2e-anchored by construction); a missing or
    mismatched annotation on a device-gated run falls back to ctx.target_ms, which the
    CLI derived from the ref's device reading."""
    from forgekernel.milestone import parse_target_ms, parse_target_window
    try:
        p = Path(ctx.milestone)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            t = parse_target_ms(text)
            if t is not None:
                window = parse_target_window(text) or "e2e"
                if window == ctx.gate_window:
                    return t
                _log(f"milestone §0 target_ms {t} is anchored in the {window} "
                     f"window but this run gates on {ctx.gate_window} — using the "
                     f"CLI-derived {ctx.gate_window} threshold {ctx.target_ms} "
                     f"instead (re-baseline §0 with a matching target_window: line)")
    except OSError:
        pass
    return ctx.target_ms


def _persist_reanchor(ctx: LoopContext, record: dict) -> None:
    """Append the re-anchor account to <op_dir>/reanchor.jsonl — the audit trail.

    A run that stops as "target" off a re-anchor, or a diagnosis that reasoned
    from a re-anchored residual, must be auditable after the fact the way a
    finish confirmation is (best_confirmation.json): which samples, which
    cluster, which estimator, which verdict. Failure to write is logged, never
    raised — the account also travels in the run summary / prompt note."""
    try:
        path = Path(ctx.milestone).parent / "reanchor.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        _log(f"meta/re-anchor: could not persist the account ({e})")


def _write_infeasible_notice(ctx: LoopContext, *, meta_step: int, diagnosis: str,
                             reachability: str, best: float | None) -> Path | None:
    """Write <op_dir>/infeasible_stop.md — the human-facing record of an INFEASIBLE stop.

    The run ends on the teacher's word alone, and that word is the pipeline's
    own verdict — right or wrong, it is evidence about the system under test.
    The notice carries what a human needs to read it without opening the
    transcripts: where the run stands against its stop line, the teacher's own
    reachability line (an OPEN there contradicts the stop request and is worth
    reading first), and the diagnosis verbatim.

    What it must never do is invite the human to overrule the verdict and hand
    the teacher a new direction on this run: choosing the residual owner or the
    reopen is exactly the meta decision under test, so a human doing it turns
    the system into "pipeline + human meta" and voids the run's numbers. The
    notice names the legitimate routes instead (evidence / a harness mechanism
    that applies to every run, tested on a fresh run / a separately labelled
    human-guided run).
    Failure to write is logged, never raised — the stop itself must not depend
    on the disk.
    """
    eq = ctx.equivalence_ratio
    try:
        target = _resolve_target(ctx)
    except Exception:  # noqa: BLE001 — a notice must not fail on a malformed §0
        target = None
    lines = [
        "# Loop stopped: the teacher declared INFEASIBLE (the end of this run)",
        "",
        f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ctx.now()))}",
        f"- meta step: {meta_step}",
        f"- best: {best if best is not None else '?'} ms; target: {target or '?'} ms"
        f" (stop line = reference/student ≥ {eq:g})",
    ]
    memo = ctx.paired_residual_memo
    if memo and best is not None and memo[0] == best:
        lines.append(f"- Latest same-card paired re-measure: +{memo[1]:.1%} from the stop line")
    lines += [
        f"- The teacher's reachability line: {reachability or '(not given)'}",
        "",
        "## The teacher's diagnosis, verbatim",
        "",
        "```text",
        diagnosis.strip(),
        "```",
        "",
        "## What this stop is",
        "",
        "- This run finishes here; the result is recorded at the current best (the final "
        "re-measure is in best_confirmation.json). "
        "`stop=infeasible` is a ruling: the run is not restarted automatically.",
        "- This is the verdict of the pipeline under test itself; right or wrong, it is recorded "
        "as evidence about this run.",
        "- If you disagree with the verdict, **do not** change direction on this run and continue "
        "it: overruling the teacher's final ruling, picking the residual owner or reopening a "
        "direction is exactly the meta decision under test; once a human makes it, the system "
        "under test becomes \"pipeline + human meta\" and this run's numbers are void. "
        "Neither this notice, the diagnosis, nor your judgement goes into the milestone / oracle / "
        "KB, nor into the inputs of the next run.",
        "",
        "## Legitimate routes",
        "",
        "1. Record it as evidence of the pipeline's verdict (run records / the evidence trail "
        "of a change).",
        "2. If you judge it a harness defect, fix it as a mechanism that treats every run alike, "
        "and test it on a new run.",
        "3. If you only want to know whether the target is reachable in engineering terms: open a "
        "separate experiment clearly labelled \"human-guided\"; its results do not enter "
        "ForgeKernel's evaluation numbers.",
        "",
    ]
    try:
        path = Path(ctx.milestone).parent / "infeasible_stop.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
    except OSError as e:
        _log(f"meta: could not write the INFEASIBLE notice ({e})")
        return None


# How long a same-period target check stays good for: a target transported under
# one period's machine conditions says nothing about a period hours later.
_FALSE_WIN_MEMO_TTL_S = 4 * 3600.0


def _false_win_replay(ctx: LoopContext, best: float) -> None:
    """Re-thread the standing false-win note for the next meta entry.

    run_two_layer consumes ``ctx.reanchor_note`` once per meta entry, and
    ``_phantom_gap_reanchor`` returns on ``gap <= 0`` before its own memo replay
    — so after a false win the SECOND meta entry would diagnose against the
    stale target again. The memo is keyed by best, like the phantom-gap one.
    """
    memo = ctx.reanchor_memo
    if memo and memo[0] == best and not ctx.reanchor_note:
        ctx.reanchor_note = memo[1]


def _fresh_ref_target(ctx: LoopContext, iter_no: int, anchor: float,
                      target: float) -> "tuple | None":
    """This period's ref through ``ctx.reanchor_ref``, transported onto the
    target: (samples, cluster, spread, fresh_min, drift, fresh_target), or None
    when no single number represents the period (too few samples, samples that
    straddle node modes, a cluster the anchor was not drawn on)."""
    try:
        samples, fresh_cluster = ctx.reanchor_ref()
    except Exception as e:  # noqa: BLE001 — a starved queue must not kill the run
        _log(f"iter {iter_no}: target check — fresh ref measurement failed ({e})")
        return None
    samples = [s for s in samples if s and s > 0]
    if len(samples) < 2:
        _log(f"iter {iter_no}: target check — {len(samples)} usable ref sample(s), "
             f"need >= 2 (a lone draw must not certify a win)")
        return None
    spread = (max(samples) - min(samples)) / min(samples)
    if spread > 0.10:
        _log(f"iter {iter_no}: target check — ref spread {spread:.1%} > 10% "
             f"(samples straddle node modes; no single number represents this period)")
        return None
    if fresh_cluster and ctx.target_anchor_cluster and \
            fresh_cluster != ctx.target_anchor_cluster:
        _log(f"iter {iter_no}: target check — fresh ref ran on {fresh_cluster!r} but "
             f"the anchor is from {ctx.target_anchor_cluster!r}; absolute readings "
             f"do not transport across clusters")
        return None
    fresh_min = min(samples)
    drift = fresh_min / anchor          # min vs min: the archive keeps the fastest
    return samples, fresh_cluster, spread, fresh_min, drift, target * drift


def _confirm_apparent_target(ctx: LoopContext, best: float,
                             iter_no: int) -> bool | None:
    """An apparent target hit, verified same-period before it may end the run.

    The phantom-gap instrument (``_phantom_gap_reanchor``) only ever guarded the
    FALSE-GAP direction — it returns early on ``gap <= 0`` because "already met:
    the inner loop stops on its own". That left the FALSE-WIN direction
    unguarded, and on a bimodal cluster a false win is the more expensive error:
    a self-ref run stopped as `target` at iteration 0 having changed no
    executable code, because its launch anchor was drawn on a slower node than
    the student's job happened to land on. The anchor and the student reading
    were 5.1% apart on byte-identical code, against a 5% goal.

    So an absolute hit buys a measurement instead of a stop. Which measurement
    depends on what the run can pay in:

    * The in-job comparator IS the declared reference (``ref_rides_job`` /
      ``ref_in_job`` — the only runs on which ``_enforce_single_ref`` keeps the
      in-job readings): the student is re-measured 2 jobs and the verdict is the
      median SAME-JOB ratio against ``equivalence_ratio``. Ref and student ran on
      one card in one minute; no ref jobs are bought. (``ref_in_job`` runs do not
      reach here from run_inner_loop at all — their absolute hit defers to
      ``_paired_parity_stop`` — but a direct caller gets the same rule.)
    * Otherwise (blackbox: the ref is measured through its own tree): re-measure
      the ref this period, transport the target by the drift, re-measure the
      student, compare absolute ms. The two sides run as separate jobs, so this
      is the weakest currency the run has — it catches a launch anchor that
      drifted, not a student job that happens to see a faster period than the
      ref's. The record names the currency so a reader can weigh the verdict.

    Returns True to stop, False when the hit was a node draw (the run keeps
    climbing, the residual threaded into the next diagnosis), and None when it
    could not be checked at all — the caller stops on the absolute reading then,
    loudly, because refusing to stop on an un-checkable run would mean never
    stopping.

    A false win is memoised by the TARGET it transported, not only by the best
    that triggered it: every later KEEP that is still short of that fresh target
    (within _FALSE_WIN_MEMO_TTL_S) re-uses the verdict instead of re-buying jobs.
    """
    target = _resolve_target(ctx)
    if not target or target <= 0:
        return None
    paired = bool(ctx.ref_rides_job or ctx.ref_in_job)
    anchor = ctx.target_anchor_ms
    if not paired and (ctx.reanchor_ref is None or not anchor):
        return None
    memo = getattr(ctx, "_false_win_memo", None)
    if isinstance(memo, dict):
        if memo["best"] == best:
            _false_win_replay(ctx, best)
            _log(f"iter {iter_no}: best {best} unchanged since the last same-period "
                 f"target check said node-draw — not re-buying it; continuing")
            return False
        age = ctx.now() - memo["ts"]
        if best > memo["fresh_target"] and 0 <= age < _FALSE_WIN_MEMO_TTL_S:
            # A new best that is still short of the target this period's check
            # transported cannot have changed the answer: re-use it, otherwise
            # every KEEP under a 5%-stale target re-buys the same jobs.
            residual = (best - memo["fresh_target"]) / memo["fresh_target"]
            note = (
                f"\n## False win (carried over from the contemporaneous conversion "
                f"{age / 60:.0f} min ago)\n"
                f"The new best {best} ms looks like it meets the target under the launch-time "
                f"anchor, but it is still above the target converted from the contemporaneous "
                f"re-measure, {memo['fresh_target']:.4f} ms ({memo['currency']}). "
                f"**True residual = +{residual:.1%}**; the target is not met yet; no re-measure "
                f"was bought again.\n"
                f"Base the diagnosis and the prescription on this residual; do not write the "
                f"converted target back into §0.\n")
            ctx.reanchor_note = note
            ctx.reanchor_memo = (best, note)
            _log(f"iter {iter_no}: best {best} meets the launch-anchor target but is "
                 f"still +{residual:.1%} over the same-period target "
                 f"{memo['fresh_target']:.4f} ms transported {age / 60:.0f} min ago — "
                 f"not re-buying; continuing")
            return False
    _log(f"iter {iter_no}: best {best} meets target {target:.4f} on the launch "
         f"anchor — re-measuring "
         + ("the student same-card (the ref rides the job)"
            if paired else "ref AND student same-period")
         + " before stopping")
    fresh = None
    if not paired:
        fresh = _fresh_ref_target(ctx, iter_no, anchor, target)
        if fresh is None:
            return None
    try:
        conf = run_best_confirmation(ctx, jobs=2, max_extra=1)
    except Exception as e:  # noqa: BLE001
        conf = None
        _log(f"iter {iter_no}: target check — student re-measure failed ({e})")
    if not conf:
        return None
    student_fresh = conf["median_ms"]
    record = {"ts": ctx.now(), "iter": iter_no, "best_ms": best,
              "stale_target_ms": target, "anchor_ms": anchor,
              "student_fresh_ms": student_fresh,
              "student_jobs": [s.get("job") for s in conf.get("jobs", [])],
              "check": "apparent-target"}
    med_ratio = None
    if paired:
        med_ratio, why = _admit_paired(conf)
        if med_ratio is None:
            # The flag promised an in-job reference the jobs did not deliver (a
            # misjudged ref_rides_job, a bench without the comparator): fall back
            # to the fresh-ref channel rather than certify on nothing.
            _log(f"iter {iter_no}: target check — no admissible same-job ratio ({why}); "
                 f"falling back to a fresh ref measurement")
            if ctx.reanchor_ref is None or not anchor:
                return None
            fresh = _fresh_ref_target(ctx, iter_no, anchor, target)
            if fresh is None:
                return None
    if med_ratio is not None:
        eq = ctx.equivalence_ratio
        confirmed = med_ratio >= eq
        # this period's parity line in the student's own ms: ratio = ref / student,
        # so parity sits at student * ratio / eq
        fresh_target = student_fresh * med_ratio / eq
        residual = eq / med_ratio - 1.0
        record.update({"currency": "same-job", "median_ratio": round(med_ratio, 4),
                       "equivalence_ratio": eq, "fresh_target_ms": fresh_target})
        how = (f"same-job median ratio {med_ratio:.4f} over "
               f"{len(conf.get('jobs', []))} job(s) vs equivalence {eq}")
    else:
        samples, fresh_cluster, spread, fresh_min, drift, fresh_target = fresh
        confirmed = student_fresh <= fresh_target
        residual = (student_fresh - fresh_target) / fresh_target
        record.update({"currency": "cross-job-absolute", "samples_ms": samples,
                       "fresh_cluster": fresh_cluster, "estimator": "min-vs-min",
                       "drift": round(drift, 4), "fresh_target_ms": fresh_target})
        how = (f"student {student_fresh} ms vs {fresh_target:.4f} ms (ref this period "
               f"{fresh_min} ms over {len(samples)} job(s), drift x{drift:.4f}; each side "
               f"on its own node — the weakest currency this run has)")
    if confirmed:
        record["verdict"] = "target"
        _persist_reanchor(ctx, record)
        _log(f"iter {iter_no}: target CONFIRMED same-period — {how}")
        return True
    record["verdict"] = "false_win"
    ctx._false_win_memo = {"best": best, "fresh_target": fresh_target,
                           "ts": ctx.now(), "currency": record["currency"]}
    if med_ratio is not None:
        note = (
            f"\n## False win intercepted (same-card re-measure)\n"
            f"The launch-time anchor made best {best} ms look like it meets the target, but the "
            f"same-job re-measure ({len(conf.get('jobs', []))} job(s), "
            f"reference and student run back to back on the same card) has a median ratio of "
            f"{med_ratio:.4f} < the equivalence ratio {eq}. "
            f"**True residual = +{residual:.1%}** (by the same-job ratio); the target is not met "
            f"yet.\n"
            f"Base the diagnosis and the prescription on this residual; do not change the "
            f"absolute-millisecond target.\n")
    else:
        note = (
            f"\n## False win intercepted (contemporaneous re-measure)\n"
            f"The launch-time anchor made best {best} ms look like it meets the target, but a "
            f"contemporaneous re-measure shows that was a node draw: "
            f"the ref's fastest this period is {fresh_min} ms ({len(samples)} runs, spread "
            f"{spread:.1%}; "
            f"launch-time anchor {anchor} ms, drift x{drift:.4f}) → converted target "
            f"{fresh_target:.4f} ms; "
            f"student same-period re-measure {student_fresh} ms. **True residual = "
            f"+{residual:.1%}**; the target is not met yet.\n"
            f"Base the diagnosis and the prescription on this residual; do not write the target "
            f"converted here back into §0 (it contains the drift factor; "
            f"writing it back would count the drift twice).\n")
    ctx.reanchor_note = note
    ctx.reanchor_memo = (best, note)
    _persist_reanchor(ctx, record)
    _log(f"iter {iter_no}: target NOT confirmed — {how} (residual +{residual:.1%}); "
         f"the absolute hit was a node draw, continuing")
    return False


def _phantom_gap_reanchor(ctx: LoopContext, best: float | None) -> dict | None:
    """Re-anchor the ref before a gap-driven diagnosis whose residual sits inside
    the node-noise band — the one place a stale anchor changes DECISIONS.

    In-loop readings compare fresh student jobs against a target derived from a
    ref measured once at launch, hours and many node draws ago. For the loop's
    day-to-day choices that staleness is harmless: best-tracking is
    student-vs-student, and the error direction on the stop check is conservative
    (a fast-node launch anchor makes the loop under-claim, never over-claim). But
    the meta layer reasons from the ABSOLUTE residual: it prices loci against it,
    frames the reachability disproof with it, and sizes prescriptions by it — and
    a residual inside the bimodality band can be entirely the anchor. Live case:
    a bwd-GQA run's meta read "+3.61%" at the M3 gate and opened a second
    structural rebuild (15 iters, ~20h, erased); the finish confirmation — the
    run's only same-period measurement — put the very sha it started from at
    1.0271x. The gap never existed.

    The transport is estimator- and provenance-matched, and two-sided before it
    may stop the run:

    * drift = min(fresh samples) / target_anchor_ms — min against min: the
      launch archive keeps the FASTEST of its samples (baseline_cli), so a
      median on the fresh side would read the median-vs-min offset (~2% on
      quiet nodes) as drift and hand out stop=target for free;
    * the fresh samples must pass admission (>= 2 usable, spread <= 10% — a
      lost job leaving one slow sample, or samples straddling the node
      modes, must not rewrite the bar the run stops on) and come from the SAME
      machine as the launch anchor (absolute readings do not transport across
      machines at all);
    * transport is refused entirely when the target is not the CLI seed paired
      with the anchor — a §0 re-baseline already reflects the conditions of ITS
      own measurement, so multiplying it by drift-since-launch counts the drift
      twice — and when the target was never derived from a measured baseline
      (--target-ms, toml constants: target_anchor_ms is None);
    * a transported target that the standing best appears to meet is still only
      half a claim — the best is itself an hours-old draw — so the student is
      re-measured same-period (run_best_confirmation, 2 jobs) before the run may
      stop, independent of whether the finish confirmation is enabled
      (confirm_jobs=0 runs included).

    Returns the account dict when the run should stop as "target". On a
    CONFIRMED gap it stashes ctx.reanchor_note — run_two_layer threads it
    through every diagnosis of this meta entry, malformed-patch retries included
    — plus a memo so an unchanged best does not re-buy ref jobs at every later
    plateau. Every measured check is persisted to reanchor.jsonl. Any
    measurement failure degrades to the old behavior, loudly.
    """
    band = ctx.reanchor_band
    if best is None or band <= 0 or ctx.reanchor_ref is None:
        return None
    cluster = _active_cluster(ctx)
    if cluster and ctx.home_cluster and cluster != ctx.home_cluster:
        # Another machine: the best is this machine's, the target is restated
        # from a reference measured on it (_cluster_scale), and the launch
        # anchor is the first machine's — there is no launch-period drift to
        # transport.
        _log(f"meta/re-anchor: run is on cluster {cluster!r} (home "
             f"{ctx.home_cluster!r}) — the phantom-gap check is a home-cluster "
             f"instrument; skipping")
        return None
    target = _resolve_target(ctx)
    if not target or target <= 0:
        return None
    if wl_why := _target_workloads_mismatch(ctx):
        _say_target_suspended(ctx, "meta/re-anchor", wl_why)
        if ctx.ref_in_job:
            # the absolute gap cannot say whether a residual exists; the
            # same-card ratio can, and it is the currency this run stops on
            return _paired_reanchor(ctx, best, None)
        return None
    gap = (best - target) / target
    if gap <= 0:
        return None  # already met: the inner loop stops on its own
    if ctx.ref_in_job:
        # The reference rides the student's own jobs: there is no cross-channel
        # drift to transport, only a same-card ratio to re-measure. The absolute
        # band does not gate this — the absolute gap is exactly the cross-node
        # number this path exists to stop reasoning from.
        return _paired_reanchor(ctx, best, gap)
    # beyond the band, drift cannot explain the residual and the diagnosis is
    # legitimately gap-driven.
    if gap > band:
        return None
    if target != ctx.target_ms:
        _log(f"meta/re-anchor: residual +{gap:.1%} is inside the {band:.0%} band "
             f"but §0 target {target} is not the launch-derived seed "
             f"{ctx.target_ms} — a re-baselined target already reflects its own "
             f"measurement period, and transporting launch drift onto it would "
             f"count the drift twice. Skipping (residual is suspect).")
        return None
    anchor = ctx.target_anchor_ms
    if not anchor:
        _log(f"meta/re-anchor: residual +{gap:.1%} is inside the {band:.0%} band "
             f"but the target was not derived from a measured baseline "
             f"(--target-ms / toml constant) — no launch-period anchor to "
             f"transport drift from. Skipping (residual is suspect).")
        return None
    if ctx.reanchor_memo and ctx.reanchor_memo[0] == best:
        # nothing moved since the last confirmed-gap check; the answer cannot
        # have changed except by further drift — re-issue the note, spend nothing.
        ctx.reanchor_note = ctx.reanchor_memo[1]
        _log("meta/re-anchor: best unchanged since the last confirmed-gap check "
             "— reusing its verdict (no new ref jobs)")
        return None
    try:
        samples, fresh_cluster = ctx.reanchor_ref()
    except Exception as e:  # a starved job must not kill the run; degrade loudly
        _log(f"meta/re-anchor: fresh ref measurement failed ({e}) — diagnosing "
             f"against the stale anchor (residual is suspect)")
        return None
    samples = [s for s in samples if s and s > 0]
    if len(samples) < 2:
        _log(f"meta/re-anchor: admission failed — {len(samples)} usable "
             f"sample(s), need >= 2 (a lone draw must not rewrite the bar the "
             f"run stops on); diagnosing against the stale anchor")
        return None
    spread = (max(samples) - min(samples)) / min(samples)
    if spread > 0.10:
        _log(f"meta/re-anchor: admission failed — spread {spread:.1%} > 10% "
             f"(samples straddle node modes; no single number represents this "
             f"period); diagnosing against the stale anchor")
        return None
    if fresh_cluster and ctx.target_anchor_cluster and \
            fresh_cluster != ctx.target_anchor_cluster:
        _log(f"meta/re-anchor: fresh samples ran on cluster {fresh_cluster!r} "
             f"but the launch anchor is from {ctx.target_anchor_cluster!r} "
             f"— absolute readings do not transport across machines; "
             f"diagnosing against the stale anchor")
        return None
    fresh_min = min(samples)
    drift = fresh_min / anchor  # min vs min: the launch archive keeps the fastest
    fresh_target = target * drift
    _log(f"meta/re-anchor: residual +{gap:.1%} <= band {band:.0%} — fresh ref "
         f"min {fresh_min} ms over {len(samples)} job(s) (spread {spread:.1%}; "
         f"launch anchor {anchor} ms, drift x{drift:.4f}) -> target "
         f"{fresh_target:.4f} ms (was {target})")
    record = {"ts": ctx.now(), "best_ms": best, "stale_target_ms": target,
              "gap_vs_stale": round(gap, 4), "anchor_ms": anchor,
              "anchor_cluster": ctx.target_anchor_cluster,
              "samples_ms": samples, "fresh_cluster": fresh_cluster,
              "estimator": "min-vs-min", "drift": round(drift, 4),
              "fresh_target_ms": fresh_target}

    def _confirmed(residual: float, how: str) -> None:
        note = (
            f"\n## Contemporaneous ref re-anchor (read before judging — the residual has been "
            f"recomputed against the new anchor)\n"
            f"The launch-time anchor is stale (nodes are bimodal, ~±6%); the ref was just "
            f"measured contemporaneously {len(samples)} times: "
            f"fastest {fresh_min} ms (spread {spread:.1%}; launch-time anchor {anchor} ms, both "
            f"read as the fastest value) → drift-converted target {fresh_target:.4f} ms. {how}has "
            f"a true residual against the new anchor of **+{residual:.1%}** (on paper against the "
            f"old anchor: +{gap:.1%}).\n"
            f"The diagnosis and the prescription must go by the recomputed residual: the "
            f"on-paper prices of the defect loci you name must add up to explain it; do not "
            f"propose a structural reopen (R1) whose expected gain is below it.\n"
            f"**You must not** write the target converted here back into §0 — it contains the "
            f"drift factor since launch, and once written back the next period would multiply "
            f"the drift in again (counting it twice). A re-baseline may only be based on raw ref "
            f"readings you measured contemporaneously yourself.\n")
        ctx.reanchor_note = note
        ctx.reanchor_memo = (best, note)
        _persist_reanchor(ctx, record)

    if best > fresh_target:
        record["verdict"] = "confirmed_gap"
        _confirmed((best - fresh_target) / fresh_target, f"best {best} ")
        return None
    # The stale-anchor account says the target is already met — but `best` is
    # itself an hours-old node draw. Measure the student in the SAME period
    # before ending the run (the finish confirmation may be disabled).
    _log(f"meta/re-anchor: best {best} meets the re-anchored target on paper — "
         f"re-measuring the student same-period before stopping")
    try:
        conf = run_best_confirmation(ctx, jobs=2, max_extra=1)
    except Exception as e:
        conf = None
        _log(f"meta/re-anchor: student re-measure failed ({e})")
    if not conf:
        record["verdict"] = "student_remeasure_unavailable"
        _log("meta/re-anchor: no same-period student reading — NOT stopping on "
             "half a claim; diagnosing with the re-anchored residual instead")
        _confirmed(0.0, f"best {best} (an old reading; no same-period student re-measure "
                        f"was available, so the residual is doubtful) ")
        return None
    student_fresh = conf["median_ms"]
    record["student_fresh_ms"] = student_fresh
    record["student_jobs"] = [s.get("job") for s in conf.get("jobs", [])]
    if student_fresh <= fresh_target:
        record["verdict"] = "target"
        _persist_reanchor(ctx, record)
        _log(f"meta/re-anchor: student same-period {student_fresh} ms <= "
             f"re-anchored target {fresh_target:.4f} — the residual was the "
             f"stale anchor, not the kernel; stopping as target")
        return record
    # the OLD best reading was the lucky draw; the honest residual uses the fresh one
    record["verdict"] = "confirmed_gap_after_student_remeasure"
    _confirmed((student_fresh - fresh_target) / fresh_target,
               f"student same-period re-measure {student_fresh} ms (the old best reading "
               f"{best} was a fast-node draw) ")
    return None


def _paired_reanchor(ctx: LoopContext, best: float, gap: float | None) -> dict | None:
    """The ref_in_job form of the phantom-gap check: re-measure the best sha
    (2 jobs) and read the same-job ratio.

    Median ratio at or above the equivalence ratio: the residual was the
    cross-node anchor, not the kernel — stop as target. Below it: the honest
    residual for the diagnosis is the paired one, measured to the bar the run
    stops on (eq/ratio - 1, as the apparent-target path reads it), threaded into
    the meta entry the same way the absolute re-anchor's note is; memoised per
    best so an unchanged best re-buys nothing.

    The residual used to be measured to parity (1/ratio - 1) whatever eq was: an
    run with a 1.05x bar at ratio 0.985 was told its residual was +1.5% — inside
    the same-job floor, "prefer parity, no reopen" — when the run still stood
    +6.6% short of its stop line. At eq = 1.0 the two forms coincide.

    ``gap`` None: the absolute target is read over other workloads than the
    gate reading (_target_workloads_mismatch), so there is no absolute residual
    to quote — the same-card ratio is the whole answer.
    """
    eq = ctx.equivalence_ratio
    sha = _best_good_ref_here(load_records(ctx.status_path), ctx)
    if ctx.reanchor_memo and ctx.reanchor_memo[0] == best:
        ctx.reanchor_note = ctx.reanchor_memo[1]
        _log("meta/re-anchor: best unchanged since the last same-job check — "
             "reusing its verdict (no new jobs)")
        return None
    prior = ctx.paired_memo.get(sha) if sha else None
    if prior and prior.get("cluster", "") == _active_cluster(ctx) \
            and prior.get("median_ratio") is not None:
        # the inner loop already re-measured this very sha (parity on paper,
        # unconfirmed) on this cluster: those two jobs answer this question
        med, conf, reused = prior["median_ratio"], None, True
        _log(f"meta/re-anchor: best {sha[:10]} was re-measured same-card at iter "
             f"{prior.get('iter')} (median ratio {med:.4f}) — reusing, no new jobs")
    else:
        reused = False
        what = (f"absolute residual +{gap:.1%} is a cross-node number" if gap is not None
                else "the absolute target covers other workloads than the gate reading")
        _log(f"meta/re-anchor: {what} (ref_in_job) — re-measuring the best "
             f"{(sha or '?')[:10]} same-card (2 jobs)")
        try:
            conf = run_best_confirmation(ctx, jobs=2, max_extra=1, ref=sha)
        except Exception as e:  # noqa: BLE001
            conf = None
            _log(f"meta/re-anchor: same-card re-measure failed ({e}) — diagnosing "
                 f"against the absolute residual (suspect)")
        med, why = _admit_paired(conf)
    record = {"ts": ctx.now(), "best_ms": best, "best_sha": sha,
              "gap_vs_absolute_target": round(gap, 4) if gap is not None else None,
              "estimator": "paired-in-job",
              "equivalence_ratio": eq, "median_ratio": med,
              "cluster": _active_cluster(ctx),
              "student_jobs": (prior["jobs"] if reused
                               else [s.get("job") for s in (conf or {}).get("jobs", [])])}
    if reused:
        record["reused_from_iter"] = prior.get("iter")
    if med is None:
        record["verdict"] = "student_remeasure_unavailable"
        record["why"] = why
        _persist_reanchor(ctx, record)
        _log(f"meta/re-anchor: same-card re-measure inadmissible ({why}) — "
             f"diagnosing against the absolute residual (suspect)")
        return None
    if med >= eq:
        record["verdict"] = "target"
        if not reused:
            ctx.paired_stop = {"sha": sha, "samples": list(conf.get("jobs", [])),
                               "median_ratio": med, "jobs": record["student_jobs"]}
        _persist_reanchor(ctx, record)
        _log(f"meta/re-anchor: same-job median ratio {med:.4f} >= {eq} over "
             f"{len(record['student_jobs'])} job(s) — the residual was the "
             f"cross-node anchor, not the kernel; stopping as target")
        return record
    residual = eq / med - 1.0          # to the stop line, not to parity
    parity = 1.0 / med - 1.0
    above_parity = eq > 1.0 + 1e-9
    record["verdict"] = "confirmed_gap"
    record["paired_residual"] = round(residual, 4)
    if above_parity:
        record["parity_residual"] = round(parity, 4)
    note = (
        f"\n## Same-card paired re-measure (read before judging — the residual has been "
        f"recomputed from the same-job ratio)\n"
        f"This run's reference runs back to back inside the student's job (ref_in_job); the "
        f"absolute milliseconds compare two different sets of nodes"
        + (f" (the old on-paper residual +{gap:.1%} is a cross-node number)" if gap is not None else
           " (the absolute target and the gate reading cover different workloads, so the "
           "on-paper residual cannot be read)")
        + f". The best was just re-measured on the same card "
        f"{len(record['student_jobs'])} times: median same-job ratio **{med:.4f}** "
        f"(reference/student; ≥ {eq:g} reaches this run's stop line) → true residual to the "
        f"stop line = **+{residual:.1%}**.\n"
        f"The diagnosis and the prescription must go by this residual: the on-paper prices of "
        f"the defect loci you name must add up to explain it; "
        f"do not propose a structural reopen (R1) whose expected gain is below it. ")
    if above_parity:
        note += (
            f"This run's target is above parity: a ratio of 1.0 only ties the reference, the "
            f"stop line is {eq:g}× — getting close to or reaching parity does not mean the work "
            f"is done; do not call it parity on the strength of the noise floor.\n")
    else:
        note += ("When the residual falls inside the same-job noise floor (1–2%), prefer calling "
                 "it parity over continuing to hunt for a structural defect.\n")
    ctx.reanchor_note = note
    ctx.reanchor_memo = (best, note)
    ctx.paired_residual_memo = (best, residual)
    _persist_reanchor(ctx, record)
    _log(f"meta/re-anchor: same-job median ratio {med:.4f} < {eq} — paired "
         f"residual +{residual:.1%} to the {eq:g} line"
         + (f" (parity {parity:+.1%})" if above_parity else "")
         + f" (absolute account said +{gap:.1%})")
    return None


def _milestone_dag(ctx: LoopContext) -> list[Decision]:
    """Decision DAG for prefix reuse: an explicit ctx.decisions_dag wins, else parse
    the milestone §4 dependencies (the SSOT once cli stopped passing a fixed empty dag)."""
    if ctx.decisions_dag:
        return ctx.decisions_dag
    from forgekernel.milestone import parse_milestone_dag
    try:
        p = Path(ctx.milestone)
        if p.exists():
            return parse_milestone_dag(p.read_text(encoding="utf-8"))
    except OSError:
        pass
    return []


def _compute_resume_plan(ctx: LoopContext, changed: list[str]) -> tuple[str | None, list[str]]:
    dag = _milestone_dag(ctx)
    if dag and changed:
        plan = compute_resume(dag, changed)
        return plan.resume_after, plan.invalidated
    return None, []


def _compute_reopen_plan(ctx: LoopContext, changed: list[str]) -> tuple[list[str], list[str]]:
    """(reopen, kept): the dependency closure of the changed entries over the
    milestone DAG, and the decisions outside it. Empty when there is no DAG or
    nothing changed."""
    dag = _milestone_dag(ctx)
    if dag and changed:
        plan = compute_reopen(dag, changed)
        return plan.reopen, plan.kept
    return [], []


def _ref_for_decision(records, decision_id: str) -> str | None:
    """The commit sha of the last landed iteration implementing this decision.

    Delivery granularity is the student's choice (template A.6), so a decision may
    have landed inside a bundled commit whose id joins its members with `+`
    (D2.3+D2.4). An exact match would miss it, and the caller's fallback is a full
    restart to the baseline ref — silently discarding every kept iteration because
    the checkpoint was filed under the bundle's name. Membership is the match:
    that sha contains the decision.
    """
    for r in reversed(records):
        if r.sha and decision_id in (p.strip() for p in r.decision_id.split("+")):
            return r.sha
    return None


# inner-loop stop reasons that hand control to the meta loop instead of ending the run.
_META_ACTIONABLE = ("plateau", "stuck", "exhausted", "gate_regression")

# Same-stage gate check for a post-patch segment. A rebuild's M0-gate reading
# is measured before M1-M3 are redone, so comparing it with the checkpoint
# best is predetermined; comparing it with the PREVIOUS segment's SAME gate is
# not — that is the reading the driver had on record and never consulted
# (observed on an ixbwd run: M0-gate 14.43 vs the prior segment's 12.31, −17%, four
# more iterations spent walking M1 before the plateau judged it). The
# tolerance is the harness's own re-measurement noise; a teacher who expects
# a gate to regress on purpose (an exchange-face rewrite that pays at M1)
# declares it per gate in the patch output (`gate_contract:`), and the check
# holds that gate to the declared band instead — and kills at the first gate
# the declaration misses.
_GATE_ID = re.compile(r"^M\d+-gate$")
_GATE_REGRESSION_TOLERANCE = 0.015
_GATE_CONTRACT_RE = re.compile(r"(M\d+-gate)\s*[≤<=]+\s*([+-]?\d+(?:\.\d+)?)\s*%")


def _parse_gate_contract(text: str) -> dict[str, float]:
    """``gate_contract: M0-gate ≤ +15%; M1-gate ≤ 0%`` -> {"M0-gate": 0.15, ...}.

    Values are the allowed regression of that gate against the previous
    segment's same gate (a fraction; negative = must improve by that much)."""
    m = re.search(r"gate_contract\s*[:：]\s*(.+)", text)
    if not m:
        return {}
    return {gate: float(val) / 100.0 for gate, val in _GATE_CONTRACT_RE.findall(m.group(1))}


def _previous_gate_record(records, gate_id: str, before: int, gate_window: str,
                          voided: "list[tuple[int, int]] | None" = None):
    """The last measured KEEP row of ``gate_id`` in the pre-segment history that
    still describes a standing tree — rows inside a reverted segment
    (``voided``: [lo, hi) index ranges) are skipped, so the reading comes from
    the lineage the current tree was actually reset to."""
    voided = voided or []
    for i in range(min(before, len(records)) - 1, -1, -1):
        if any(lo <= i < hi for lo, hi in voided):
            continue
        r = records[i]
        if (r.decision_id == gate_id and r.verdict == "KEEP"
                and gate_ms(r, gate_window) is not None):
            return r
    return None


def _previous_gate_reading(records, gate_id: str, before: int, gate_window: str,
                           voided: "list[tuple[int, int]] | None" = None):
    r = _previous_gate_record(records, gate_id, before, gate_window, voided)
    return None if r is None else gate_ms(r, gate_window)


# Same-stage tolerance on absolute ms across two jobs. The same-job ratio
# (_GATE_REGRESSION_TOLERANCE above) is one card, one minute; an absolute
# reading carries the node population on top — one unchanged commit moves ~6%
# between jobs (IterationRecord.vs_ref) — so a cross-job absolute comparison is
# only read same-cluster and only outside that band.
_GATE_REGRESSION_TOLERANCE_MS = 0.06


def _gate_regression(records, rec, before: int, contract: dict, gate_window: str,
                     voided: "list[tuple[int, int]] | None" = None) -> str | None:
    """A message when ``rec`` (a gate KEEP) regressed past its band, else None.

    Read on the same-job ref/student ratio when both gate rows carry one
    (node-insensitive); else on absolute ms, same-cluster rows only, inside the
    cross-job band — two clusters' absolute readings are not a comparison."""
    gate = rec.decision_id or ""
    if rec.verdict != "KEEP" or not _GATE_ID.match(gate):
        return None
    prev = _previous_gate_record(records, gate, before, gate_window, voided)
    if prev is None:
        return None
    allowed = float(contract.get(gate, 0.0))
    band = (f"declared band {allowed * 100:+.0f}%" if gate in contract
            else "same-stage parity")
    cur_r, prev_r = _paired_ratio(rec, gate_window), _paired_ratio(prev, gate_window)
    if cur_r is not None and prev_r is not None and prev_r > 0:
        # > 1 = student faster: a regression is the ratio dropping past the band
        if cur_r >= prev_r * (1.0 - allowed - _GATE_REGRESSION_TOLERANCE):
            return None
        return (f"{gate} same-job ratio {cur_r:.4f} vs previous segment's {gate} "
                f"{prev_r:.4f} ({(cur_r / prev_r - 1) * 100:+.1f}%; {band} + "
                f"{_GATE_REGRESSION_TOLERANCE * 100:.1f}% noise) — the rebuild is behind "
                f"at the same stage; judging the patch now")
    if rec.cluster and prev.cluster and rec.cluster != prev.cluster:
        return None
    cur, prev_ms = gate_ms(rec, gate_window), gate_ms(prev, gate_window)
    if cur is None or prev_ms is None or prev_ms <= 0:
        return None
    if cur <= prev_ms * (1.0 + allowed + _GATE_REGRESSION_TOLERANCE_MS):
        return None
    return (f"{gate} {cur:.4f} ms vs previous segment's {gate} {prev_ms:.4f} ms "
            f"({(cur / prev_ms - 1) * 100:+.1f}%; {band} + "
            f"{_GATE_REGRESSION_TOLERANCE_MS * 100:.0f}% cross-job noise) — the rebuild is "
            f"behind at the same stage; judging the patch now")


def _resolve_resume(ctx: LoopContext, meta: "MetaResult", records, baseline_ref: str) -> str:
    """Map a meta patch to the git ref the workspace resumes from (replaces reset("HEAD")).

    - additive / kb-example: nothing invalidated -> keep all student state (current HEAD);
    - structural / split with a reusable prefix: reset to that decision's checkpoint sha;
    - foundation patch (earliest affected decision is the first -> resume_after None) or a
      resume point with no recorded sha: full restart to the loop's baseline ref.
    """
    if meta.patch_class in ("additive", "kb-example", "unpriced-reopen", "reopen-capped",
                            "infeasible-refused"):
        return ctx.backend.head()
    if meta.resume_after is None:
        return baseline_ref
    return _ref_for_decision(records, meta.resume_after) or baseline_ref


def _resolve_resume_in_place(ctx: LoopContext, meta: "MetaResult", records,
                             baseline_ref: str) -> "tuple[str, list[str]]":
    """(resume ref, reopened decisions): in-place reopening when it applies,
    the prefix-reuse reset otherwise.

    In place = the best green KEEP tree stays; the student redoes the
    dependency closure of the changed entries on it (announced as a REOPEN
    row). Applies to structural / split patches whose closure is non-empty,
    when a green best tree exists to reopen on and the run has not opted out.
    Pre-green (no measured KEEP) there is nothing to keep, so the old reset
    stands; additive / kb-example patches invalidate nothing either way.
    """
    if (not ctx.reopen_in_place or meta.patch_class not in ("structural", "split")
            or not meta.reopened):
        return _resolve_resume(ctx, meta, records, baseline_ref), []
    best = _best_good_ref(records, ctx.gate_window)
    if best is None:
        return _resolve_resume(ctx, meta, records, baseline_ref), []
    return best, list(meta.reopened)


def _reopen_audit_gate(reopened: list) -> str | None:
    """The gate whose next KEEP closes a reopened foundation: ``M0-gate`` when
    the closure holds any M0 decision (the segment enters green, so the
    green-flip audit trigger cannot fire), else None — the audit reads the
    foundation only, and a reopen above M0 leaves it as built."""
    return "M0-gate" if any(re.match(r"D0\.", d or "") for d in reopened) else None


def _note_reopen(ctx: LoopContext, meta: "MetaResult", ref: str) -> None:
    """Announce an in-place reopen: one NEUTRAL status row the student reads
    (its earlier KEEP rows for the closure are void from here on) and the
    driver-context note for the segment's student turns."""
    reopened = ", ".join(meta.reopened)
    kept = ", ".join(meta.kept) or "(none)"
    changed = ", ".join(meta.changed_entries) or "?"
    change = (f"REOPEN on {ref[:10]}: milestone patch changed {changed}; the dependency "
              f"closure [{reopened}] is reopened — earlier KEEP rows for these decisions "
              f"are void, redo them on this tree per the patched milestone. Kept as built "
              f"(do not rebuild): [{kept}].")
    records = load_records(ctx.status_path)
    append_record(ctx.status_path, IterationRecord(
        iter=len(records), decision_id="REOPEN", change=change,
        correctness=None, verdict="NEUTRAL", sha=""))
    ctx.reopen_note = (
        f"- **Reopened in place (REOPEN)**: the previous meta step's patch changed `{changed}`. "
        f"The workspace was **not** rolled back; it is still the best KEEP tree so far "
        f"`{ref[:10]}`. The newest `decision_id: REOPEN` row in status lists the reopened "
        f"decisions [{reopened}]: their earlier KEEP records are **void**; **redo** them on the "
        f"current tree per the patched milestone (commit under each one's own decision_id as "
        f"usual; bundling is allowed; gate rows are still committed separately). "
        f"The decisions not listed [{kept}] **remain valid; do not rebuild them**. Do not touch "
        f"code outside the reopened set.\n")
    _log(f"meta: reopen in place on {ref[:10]} — redo [{reopened}], keep [{kept}]")
def _set_resume_note(ctx: LoopContext, meta: "MetaResult", resume_ref: str,
                     baseline_ref: str) -> None:
    """Tell the student its workspace was rolled back, and to what.

    Without this the student's only delivery ledger is the trajectory, whose old
    records survive the reset — so after a foundation patch it reads every decision
    as already delivered and reports EXHAUSTED into an empty segment. The note is
    the reset's visible half: which checkpoint the code is at, which decisions the
    patch invalidated, and that old records for those rows describe a rolled-back
    implementation, not a delivery.

    The note describes the WORKSPACE, not this patch: a patch class that resets
    nothing leaves the standing note alone (clearing it would restore the failure
    mode), and only a reset onto a tree where everything is delivered again — the
    two REVERT paths — clears it.
    """
    if meta.patch_class in ("additive", "kb-example"):
        # These do NOT reset the workspace (_resolve_resume returns head()), so an
        # earlier rollback is still in force and its note still describes the tree.
        # Clearing here would reinstate the very failure mode the note exists to fix:
        # a structural rollback the student exhausted out of without re-delivering,
        # then an additive step, and the survived old records read as deliveries
        # again. The note describes the workspace, not this patch — leave it.
        return
    if resume_ref == baseline_ref:
        # Both a foundation patch (resume_after None, whether or not the DAG parsed)
        # and a resume point whose sha is not on record land here: _resolve_resume
        # falls back to baseline for either. The tree is the stub, so naming a
        # checkpoint it is not at — with an invalidation list that omits everything
        # before that checkpoint — would tell the student D0.1..D0.13 are still
        # delivered and send it to write D0.14 on top of a stub.
        ctx.resume_note = (
            "- **The outer meta loop rolled the workspace back to the baseline (stub kernel)** "
            "(the milestone was just patched). **Every decision** of the milestone must be "
            "delivered again per the patched milestone; the old records in the trajectory "
            "belong to the rolled-back implementation and are not evidence of delivery — "
            "do not report EXHAUSTED on their strength. Start from the earliest decision.\n")
        return
    if not meta.invalidated:
        # compute_resume matched no decision: ResumePlan(tip, []) means nothing was
        # invalidated and the reset is a no-op (a patch touching only prose entries
        # gets this). Manufacturing a rollback here would spend the segment
        # re-delivering rows that are already delivered; leave any standing note,
        # which still describes the tree, alone.
        return
    inv = ", ".join(meta.invalidated)
    ctx.resume_note = (
        f"- **The outer meta loop rolled the workspace back to the checkpoint of decision "
        f"{meta.resume_after}** (the milestone was just patched). "
        f"Rolled-back / invalidated decisions: {inv} — these must be **delivered again** per "
        f"the patched milestone. Their old records in the trajectory belong to the rolled-back "
        f"implementation and are not evidence of delivery; do not report EXHAUSTED on their "
        f"strength. Start from the earliest undelivered decision on the invalidated list.\n")


# a milestone patch counts as helpful only if the post-patch inner segment set a new
# best beyond this relative margin; otherwise it is rolled back (transactional).
# Aligned with detect_plateau's noise_floor: the harness's re-measurement span is
# ~1.37% (four samples of one kernel), and a margin below it lets one
# lucky draw certify a useless patch — KEEP_PATCH has no later rollback, so the
# milestone edit would be permanent. Sub-noise gains are unmeasurable and therefore
# unattributable; a patch whose true gain hides under the noise cannot be credited.
_PATCH_IMPROVE_EPS = 0.015

# Runs (by status path) already told their gate covers fewer than PERF_WORKLOADS
# workloads — the warning is said once per run, not on every KEEP.
_SINGLE_WORKLOAD_WARNED: set[str] = set()


def _too_few_workloads(ctx: LoopContext, ev: EvalResult) -> bool:
    """A plain bench's gate below PERF_WORKLOADS workloads. A board or task card times
    what its suite says (a task: every input group of every shape, one workload block
    per shape), so the count is not its contract."""
    return ev.n_workloads < PERF_WORKLOADS and \
        not precision_gate.is_leaderboard_card(ctx.workspace)

# A ⊗ companion group whose members only pay off together cannot be priced one
# member at a time: mounted alone each member is cost without gain, so per-iter
# pricing plus the transactional rollback below erased every incremental attempt
# at one (a bwd-GQA rebuild died mid-package to exactly this). The student
# prompt already lets a declared group ride slower intermediates ("companion-group
# intermediate states") and the ledger already carries the declaration — the
# judgment just never read it. A KEEP whose change opens with a `part k of n` declaration is such an
# intermediate: while one is the segment's latest delivery, the patch is under
# construction, not refuted — HOLD, like a metric-less segment.
# The declaration is a bounded plea, not a pardon: n is capped (a coupled group
# larger than this should be re-cut per the split discipline), the group forfeits
# the exemption once its KEEP budget (n rows) is spent or once a later KEEP moves
# on without declaring, and a trailing run of failed attempts longer than the
# split discipline tolerates goes back to normal pricing.
_PKG_DECL = re.compile(r"\bpart\s+(\d+)\s*(?:of|/)\s*(\d+)\b", re.IGNORECASE)
_PKG_MAX_STEPS = 4
_PKG_TRAIL_SLACK = 3
# A declaration must OPEN the change (the prompt's contract: "the first sentence
# of change") — a passing mention deeper in the prose ("mirrors the ref's part 2
# of 4 split") must not turn an ordinary KEEP into an in-flight package. The opening = the
# first line's first sentence.
_PKG_SENT_END = re.compile(r"(?<=[.!?。！？])\s")


def _pkg_decl(change: str) -> re.Match | None:
    head = (change or "").split("\n", 1)[0]
    return _PKG_DECL.search(_PKG_SENT_END.split(head, 1)[0])


def _declared_construction(change: str) -> str | None:
    """What the delivery's opening declares about the tree being mid-build, or
    None: an unfinished package step (``part k of n``, k < n, the prompt's
    multi-step build package contract) or an opened placeholder (``[PLACEHOLDER: …]``). Read
    by run_inner_iteration to scaffold such a row while the lineage's green state
    is still unknown, instead of buying a full eval that can only say
    "under construction"."""
    m = _pkg_decl(change)
    if m and int(m.group(1)) < int(m.group(2)):
        return f"part {m.group(1)} of {m.group(2)}"
    head = (change or "").split("\n", 1)[0]
    p = _PLACEHOLDER_TAG.search(head)
    if p:
        return f"[PLACEHOLDER: {p.group(1).strip()[:60]}]"
    return None


def _green_on_resume(records, head: str) -> bool | None:
    """The lineage's correctness state when a loop process starts over an existing
    trajectory (a budget / cert-expiry resume): what ``run_two_layer`` seeds the
    first inner segment with instead of "unknown".

    The checkpoint's own record decides first (``_green_at_ref``: a KEEP sha is
    green, a scaffold NEUTRAL sha is mid-construction). When no record owns HEAD
    (a driver seed commit, a kept probe), the trajectory's latest correctness
    verdict stands in: a PASS means green; a FAIL on a lineage that has never
    been green means still under construction (a GDN run resumed after an
    iteration-0 REVERT + five scaffold rows and paid a full eval of build step 6 —
    "non-finite output", REVERT — because this read None); a FAIL after an
    earlier KEEP is a post-green break whose reset target is unknown here, so it
    stays None and the next full eval decides, as before."""
    g = _green_at_ref(records, head)
    if g is not None:
        return g
    ever_green = any(r.verdict == "KEEP" and r.correctness for r in records)
    for r in reversed(records):
        if r.correctness is True:
            # green only with a verified tree to fall back on: HEAD itself was never
            # evaluated (it may be the very delivery whose eval the process died in),
            # so it must not become the rollback anchor — _resume_anchor names the
            # last KEEP instead, and the first full eval decides HEAD as before
            return True if _resume_anchor(records, head) else None
        if r.correctness is False:
            return None if ever_green else False
    if any(r.verdict == "NEUTRAL" and r.correctness is None and r.sha for r in records):
        return False      # scaffold rows only: construction, never evaluated green
    return None


def _resume_anchor(records, head: str) -> str | None:
    """The rollback anchor for a lineage that resumes green (``_green_on_resume``
    True), or None when there is no verified tree to fall back on.

    ``head`` when a KEEP record owns it, else the latest KEEP sha — never an unowned
    HEAD. A HEAD no record owns is at best the last KEEP plus commits no eval
    verified (a delivery the process died evaluating, a kept probe); entering a green
    segment on it would make every later correctness break reset onto a tree that may
    itself be the broken one, and keep resetting there (observed on an MLA run: a delivery
    committed at eval time became the anchor, then four consecutive REVERTs to it)."""
    for r in reversed(records):
        if r.sha and r.sha == head:
            return head if (r.verdict == "KEEP" and r.correctness) else None
    for r in reversed(records):
        if r.verdict == "KEEP" and r.correctness and r.sha:
            return r.sha
    return None


def _open_package_tail(segment) -> tuple[int, int] | None:
    """(k, n) when the segment's latest delivery is a declared, still-unfinished,
    within-budget multi-iter package step; None otherwise."""
    # The budget counts every declared KEEP in the segment, package identity
    # unexamined: a re-cut mid-segment (abandon pkg A, open pkg B with no plain
    # KEEP between) burns A's rows against B. Deliberately conservative — the
    # error direction is "pricing resumes early", never "plea extended".
    pkg_keeps = sum(1 for r in segment
                    if r.verdict == "KEEP" and _pkg_decl(r.change))
    for offset, r in enumerate(reversed(segment)):
        if r.verdict != "KEEP":
            continue
        m = _pkg_decl(r.change)
        if not m:
            return None        # the latest delivery moved on: no package in flight
        k, n = int(m.group(1)), int(m.group(2))
        if not (1 <= k < n <= _PKG_MAX_STEPS):
            return None        # finished (k >= n), malformed, or over the cap
        if pkg_keeps > n:
            return None        # declared budget spent: n rows of grace, no more
        if offset > _PKG_TRAIL_SLACK:
            return None        # stale: too many failed attempts since the step
        return (k, n)
    return None


@dataclass
class _PendingPatch:
    best_before: float | None
    milestone_before: str
    good_ref: str
    # status-record count when the patch landed: rows appended after this index are
    # the post-patch segment, the only evidence the judgment may read.
    records_before: int = 0
    # The cluster best_before was read on. A segment that ran elsewhere is
    # priced against that cluster's own pre-patch best, or held if it has none.
    cluster: str = ""
    # per-gate allowed regression vs the previous segment (see _gate_regression)
    gate_contract: dict = field(default_factory=dict)
    # decisions reopened in place by this patch (empty for a prefix-reuse reset);
    # a REVERT_PATCH voids the REOPEN row so the student reads the old KEEPs as
    # standing again.
    reopened: list = field(default_factory=list)


def _read_milestone(ctx: LoopContext) -> str:
    try:
        p = Path(ctx.milestone)
        return p.read_text(encoding="utf-8") if p.exists() else ""
    except OSError:
        return ""


def _write_milestone(ctx: LoopContext, text: str) -> None:
    if text:  # never clobber with an empty snapshot (no prior milestone existed)
        Path(ctx.milestone).write_text(text, encoding="utf-8")


def _best_good_ref(records, gate_window: str = GATE_WINDOW_DEFAULT) -> str | None:
    """Commit sha of the best (min gate-window ms) KEEP iteration — the checkpoint
    a rolled-back patch returns the workspace to."""
    kept = [r for r in records
            if r.verdict == "KEEP" and gate_ms(r, gate_window) is not None and r.sha]
    return min(kept, key=lambda r: gate_ms(r, gate_window)).sha if kept else None


def _green_at_ref(records, ref: str) -> bool | None:
    """The correctness state of the workspace once reset to ``ref``.

    A sha owned by a KEEP record was a passing kernel (green). One owned by a
    scaffold NEUTRAL record is mid-construction (not green) — resuming there must
    re-enter scaffold mode, not evaluate half-built sub-steps. A ref no record owns
    (the baseline, a probe/kb commit) is unknown: the next full eval decides.
    """
    for r in reversed(records):
        if r.sha and r.sha == ref:
            return bool(r.correctness) if r.correctness is not None else False
    return None


def _judge_patch(ctx: LoopContext, pending: _PendingPatch) -> str:
    """Attribute the last milestone patch against its post-patch inner segment.

    Helpful (a new best beyond EPS — in same-job ratio when both sides carry one,
    else in gate-window ms) -> KEEP_PATCH. Unhelpful with metric evidence ->
    REVERT_PATCH: restore the milestone snapshot AND reset the workspace to the
    pre-patch best-good checkpoint, so milestone and kernel stay consistent (as if
    this meta step never happened).

    A segment with NO metric reading at all -> HOLD_PATCH: the patch is
    unattributable, not refuted. Pre-green construction (scaffold rows, failed gate
    attempts, certified probes) is invisible to the metric, and rolling it back
    converts "ran out of runway" into "never happened" — three runs lost whole
    construction ladders to exactly that (a rewrite whose repair cycle outlives the
    segment can then never survive a wall). Nothing is restored; the transaction
    stays OPEN so a later patch joins it and one joint judgment — with metric
    evidence — settles both. The first-green path above makes the same call for
    rebuild patches cut short at M0; this is that rule extended to every stop with
    a metric-less segment behind it.

    A measured segment whose LATEST delivery declares itself an unfinished
    multi-iter package step (`part k of n`, see _open_package_tail) is also held,
    not refuted: a ⊗ companion group's members pay off only together, so its
    measured intermediates are construction rows that happen to carry a reading.
    The declaration is bounded — n capped, KEEP budget of n rows, forfeited the
    moment a later KEEP moves on — so it is a plea the ledger can audit, not a
    shield against pricing.
    """
    records = load_records(ctx.status_path)
    cluster = _active_cluster(ctx)
    best_before = pending.best_before
    if cluster != pending.cluster:
        # The segment ran on a different cluster than the one best_before was
        # read on. Price it against THAT cluster's pre-patch best; with none,
        # the patch is unattributable (not refuted) — the same call a
        # metric-less segment gets.
        best_before = _best_ms(_on_cluster(records[:pending.records_before], ctx, cluster),
                               ctx.gate_window)
        if best_before is None and pending.best_before is not None:
            if ctx.meta_status_path:
                update_last_outcome(ctx.meta_status_path, outcome="HOLD_PATCH",
                                    best_after=None)
            _log(f"meta: patch HOLD_PATCH (segment ran on cluster {cluster!r}, the "
                 f"pre-patch best {pending.best_before} was read on "
                 f"{pending.cluster!r}, and {cluster!r} has no pre-patch reading to "
                 f"price against — transaction open)")
            return "HOLD_PATCH"
    best_after = _best_ms(_on_cluster(records, ctx, cluster), ctx.gate_window)
    # The verdict currency: the same-job ratio when both sides carry one. Absolute
    # ms is one node draw per row, and best_before is the minimum of every draw so
    # far — a lucky one stands as the bar. Live case: a best read 1.5% fast, a
    # rebuild that lifted the same-job ratio 1.040 -> 1.072 read 0.14% faster in
    # ms than that draw, and was rolled back as "no gain". The ratio cancels the
    # node; before-rows inside a segment an earlier REVERT_PATCH voided describe
    # trees that are gone and do not set the bar.
    voided = [i for lo, hi in ctx.voided_ranges for i in range(lo, hi)]
    before_rows = [r for i, r in enumerate(records[:pending.records_before]) if i not in voided]
    ratio_before = _best_ratio(_on_cluster(before_rows, ctx, cluster), ctx.gate_window)
    ratio_after = _best_ratio(_on_cluster(records[pending.records_before:], ctx, cluster),
                              ctx.gate_window)
    if ratio_before and ratio_after:
        improved = ratio_after > ratio_before * (1 + _PATCH_IMPROVE_EPS)
        reading = (f"same-job ratio {ratio_before:.4f} -> {ratio_after:.4f}; "
                   f"best {best_before} -> {best_after} ms")
    else:
        improved = best_after is not None and (
            best_before is None or best_after < best_before * (1 - _PATCH_IMPROVE_EPS))
        reading = f"best {best_before} -> {best_after}"
    if improved:
        if ctx.meta_status_path:
            update_last_outcome(ctx.meta_status_path, outcome="KEEP_PATCH", best_after=best_after)
        _log(f"meta: patch KEEP_PATCH ({reading}"
             + (f" on cluster {cluster}" if cluster and cluster != ctx.home_cluster else "")
             + ")")
        return "KEEP_PATCH"
    segment = records[pending.records_before:]
    # Attributable evidence = at least one measured KEEP in the segment (either
    # timing window: a KEEP whose gate-window reading is missing is a bench-contract
    # violation, not a construction row, and must not turn a refuted patch into a
    # held one). Scaffold NEUTRALs, correctness-FAIL gate attempts and probe-blocked
    # rows measure nothing about the patch's worth.
    measured = any(r.verdict == "KEEP" and
                   (r.device_median_ms is not None or r.bwd_median_ms is not None)
                   for r in segment)
    if not measured:
        if ctx.meta_status_path:
            update_last_outcome(ctx.meta_status_path, outcome="HOLD_PATCH", best_after=best_after)
        _log(f"meta: patch HOLD_PATCH (no measured KEEP in the {len(segment)}-row "
             f"post-patch segment — pre-green construction preserved, transaction open)")
        return "HOLD_PATCH"
    pkg = _open_package_tail(segment)
    if pkg is not None:
        k, n = pkg
        if ctx.meta_status_path:
            update_last_outcome(ctx.meta_status_path, outcome="HOLD_PATCH", best_after=best_after)
        _log(f"meta: patch HOLD_PATCH (declared multi-iter package in flight — "
             f"part {k} of {n} is the segment's latest delivery; pricing deferred "
             f"to package completion, transaction open)")
        return "HOLD_PATCH"
    _write_milestone(ctx, pending.milestone_before)
    ctx.backend.reset(pending.good_ref)
    # every row of the rolled-back segment describes a tree that is gone
    ctx.voided_ranges.append((pending.records_before, len(records)))
    if pending.reopened:
        # The REOPEN row told the student its KEEPs for the closure were void;
        # with the patch reverted and the tree back at the pre-patch best, they
        # stand again — say so in the same ledger the student reads.
        append_record(ctx.status_path, IterationRecord(
            iter=len(records), decision_id="REOPEN-REVERT",
            change=(f"the milestone patch behind the last REOPEN row was reverted; "
                    f"tree restored to {pending.good_ref[:10]}; the earlier KEEP rows for "
                    f"[{', '.join(pending.reopened)}] stand again, the reopened segment's "
                    f"rows are void."),
            correctness=None, verdict="NEUTRAL", sha=""))
        ctx.reopen_note = ""
    # The workspace is back on the pre-patch good tree: nothing the reverted patch
    # invalidated is rolled back any more, so its standing note must not survive into
    # a student turn that runs on the restored tree.
    ctx.resume_note = ""
    if ctx.meta_status_path:
        update_last_outcome(ctx.meta_status_path, outcome="REVERT_PATCH", best_after=best_after)
    _log(f"meta: patch REVERT_PATCH (no gain: "
         + (f"same-job ratio {ratio_before:.4f} -> {ratio_after:.4f} within "
            f"{_PATCH_IMPROVE_EPS:.1%}; " if ratio_before and ratio_after else "")
         + f"best_before={best_before} best_after={best_after}); "
         f"milestone + workspace rolled back")
    return "REVERT_PATCH"


def _needs_authoring(ctx: LoopContext) -> bool:
    """True when the milestone has not been distilled yet and the teacher has an input
    to distill it from. This is the entry to the distillation loop: meta authors
    milestone v1 before the inner loop starts.

    The input is oracle source under oracle_mode="source" and black-box reconnaissance
    of the reference under "blackbox"; "none" has none and skips distillation. Asking
    only "is there an oracle path" conflated "the teacher may not read source" with
    "there is nothing to distill", so the mode is what is checked.

    "Not distilled yet" is anything that fails the §9 gate — including the file not
    existing at all, which is how a freshly scaffolded operator starts (nothing
    writes a placeholder milestone; run_distill_step creates the file)."""
    if ctx.oracle_mode == "none":
        return False
    if ctx.oracle_mode == "source" and not ctx.oracle_path:
        return False
    return bool(_validate_milestone_file(ctx.milestone))


@dataclass
class ReconResult:
    """What the reconnaissance step produced."""
    rounds: int
    report: Path | None = None
    finished: bool = False


_RECON_STATE_NAME = "recon_complete.json"


def _recon_state_path(ctx: LoopContext) -> Path | None:
    """Where "reconnaissance is finished" is remembered across process restarts.

    Lives next to status.jsonl — the loop's own state directory — and NOT in the
    reconnaissance tree: the recon backend commits with `git add -A`, so anything
    parked there would ride the next probe commit into the tree's history and from
    there into every future factory clone. student/ is gitignored in the operator
    repo, so the marker can never reach a student clone or a GPU tarball either.
    """
    if ctx.recon_path is None:
        return None
    return ctx.status_path.parent / _RECON_STATE_NAME


def _save_recon_state(ctx: LoopContext, rounds: int) -> None:
    """Persist recon completion so a restart does not re-litigate a finished phase.

    Observed on a bwd-MHA run: a watchdog restart put the new process's
    session_start after the previous session's finished report, the freshness gate
    classified that report as stale residue, and the teacher — convinced the phase
    was over — yielded turn after turn until all 12 rounds had been spent on
    re-affirmation theater. The completion is a fact about the experiment, not
    about the process, so it must live on disk. Written only when the write-up
    stands complete (artifacts fresh, evidence delivered or waived).
    """
    state_path = _recon_state_path(ctx)
    if state_path is None:
        return
    candidates = ctx.recon_path.parent / "candidates.md"
    try:
        state = {
            "report": str(ctx.recon_path),
            "report_mtime": ctx.recon_path.stat().st_mtime,
            "candidates_mtime": candidates.stat().st_mtime,
            "rounds": rounds,
        }
    except OSError:
        return  # artifacts vanished between the completeness check and here
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _log(f"meta/recon: completion persisted to {state_path}")
    _persist_gap_queue(ctx)


# --- out-of-run gap queue (loop/gap_queue.py) ------------------------------
# Moved to loop/gap_queue.py (docs/driver-split-spec.md step 8, first slice);
# imported back so every name it exposed from driver is still reachable at
# forgekernel.loop.driver.*
from forgekernel.loop.gap_queue import (  # noqa: F401 -- re-export
    _GAP_QUEUE_NAME,
    _GAP_ARCHIVE_SUBDIR,
    _TABLE_ROW_RE,
    _EXPR_TAG_RE,
    _EXPR_GUESS_RE,
    _VERIFY_TAG_RE,
    GapRow,
    _row_verify,
    parse_gap_queue,
    _gap_queue_text,
    load_gap_queue,
    _gap_archive_path,
    _persist_gap_queue,
    _load_recon_state,
    TRUNCATION_MAX_CHARS,
    _read_or_none,
    _agent_turn,
    _recon_prompt,
    TRUNCATION_BACKOFF_S,
)


def run_recon_step(ctx: LoopContext) -> ReconResult:
    """Black-box reconnaissance: the teacher measures the reference it may not read.

    Under oracle_mode="blackbox" the teacher has no source, so the input it distils
    from has to be produced rather than opened. It writes a script, names it, and the
    driver runs it against the reference and hands back the output — the same shape as
    the student's probe sub-loop and the teacher's KB certification, for the same
    reason: the agent decides *what* to ask, and the driver owns submission, polling,
    retry of lost jobs and log recovery, none of which an agent should reimplement.

    Two properties are load-bearing:

    * It runs in ``recon_backend``, a separate submission tree. Reconnaissance output
      is precisely what a student must not see, and the student's job receives its
      operator tree as a tarball.
    * The script is committed before submission. Delivery is ``git archive``, which
      ships committed files only — an uncommitted script reaches the node as absence,
      and the failure looks like a broken script rather than a missing one.

    * SASS evidence is gated, not hoped for. The driver seeds the vendored
      ``sass_tools`` into ``notes/`` so no probe hand-rolls a parser, and a finish
      is only accepted once a job log carried the tool's ``SASS-OK`` verdict or the
      report owns the absence with a ``SASS-WAIVED:`` line (a report once shipped
      without the opcode histogram and no mechanism objected).

    The loop ends when the teacher stops asking — accepted only with both write-up
    artifacts on disk, since a truncated turn is indistinguishable from a deliberate
    stop — or when the round budget runs out; a teacher that has learned nothing
    still has to write its report, and saying so honestly is more useful than a
    report padded to look complete.
    """
    be, rbe = ctx.backend, ctx.recon_backend
    if rbe is None:
        _log("meta/recon: no reconnaissance backend configured — skipping")
        return ReconResult(rounds=0, finished=False)
    if (prior := _load_recon_state(ctx)) is not None:
        _log(f"meta/recon: completed in a previous session ({_RECON_STATE_NAME} "
             "validates against the on-disk write-up) — skipping straight to distillation")
        _write_family_analysis(ctx)
        return prior
    _seed_sass_tools(ctx, rbe)
    # Freshness floor for the write-up gate: artifacts must have been (re)written
    # after this moment. Without it, whatever report.md/candidates.md happen to sit
    # at the recon path — a previous run's, or a previous OPERATOR's (the default
    # recon path ships with a populated GEMM report in-tree) — satisfies the
    # existence check, and a truncated teacher turn finishes reconnaissance
    # against wrong data. Reusing a prior tree stays legal: the teacher just has
    # to re-affirm the write-up by touching it this session.
    session_start = time.time()
    sass_ok = False  # flipped by the first job log carrying a SASS-OK verdict line
    sched_ok = False  # ditto, SCHED-OK (schedule transcription delivered)
    live_ok = False   # ditto, LIVE-OK (register-liveness ledger delivered)
    stall_ok = False  # ditto, STALL-OK (per-instruction stall ledger delivered)
    struct_ok = False  # ditto, STRUCT-OK (per-segment construction table delivered)
    near_miss: dict = {}  # channel -> a "*-OK" line that failed the full-field check
    launch_facts: dict = {}  # cluster shape read off a job log (see _launch_cluster_from_log)
    storm_budget = [TRUNCATION_STORM_BUDGET]
    prompt = _recon_prompt(ctx)
    for r in range(ctx.max_recon_rounds):
        out = _agent_turn(
            be, prompt, label="meta/recon", storm_budget=storm_budget,
            left_a_trace=lambda o: (
                isinstance(parse_student_turn_safe(o, allow_recon=True), ReconRequest)
                or _recon_artifacts_written(ctx, since=session_start)))
        turn = parse_student_turn_safe(out, allow_recon=True)
        if not isinstance(turn, ReconRequest):
            # "No further request" is how a teacher signals it is done — but a
            # truncated turn looks exactly the same. Observed verbatim: a teacher
            # ended a turn with "Writing probe3:" and nothing else; the loop read
            # that as a deliberate finish, 10 of 12 budget rounds were forfeited,
            # and four architecture quantities shipped as "not measured". A finish is
            # therefore only accepted once both write-up artifacts exist on disk;
            # any other request-less turn is re-prompted, and the round budget is
            # charged to it — a teacher that keeps stalling converges on the same
            # write-up guard the budget path uses.
            if _recon_artifacts_written(ctx, since=session_start):
                # The overlap channels (ROLE with tensor, PMTL) are delivered by
                # the driver's own frozen pricing recipe on the ref tree, not by
                # the teacher's improvised probes — it runs here, once, before
                # the teacher is held to them.
                _ensure_ref_pricing(ctx, label="meta/recon")
                missing = _missing_recon_evidence(ctx, sass_ok, sched_ok, live_ok,
                                                  stall_ok, struct_ok, near_miss=near_miss)
                if missing:
                    # Rounds remain, so "deliver or waive" is a real choice here —
                    # a teacher once knew the one round that would close four gaps
                    # and was allowed to finish without running it.
                    _log(f"meta/recon: write-up exists but {'/'.join(missing)} "
                         "evidence is neither delivered nor waived — re-prompting")
                    prompt = _recon_prompt(ctx, extra=_evidence_gate_msg(
                                               missing, near_miss, ctx))
                    continue
                contras = _launch_contradictions_open(ctx, launch_facts)
                if contras:
                    _log(f"meta/recon: RECON-CONTRADICTION cluster: NCU read "
                         f"cluster={launch_facts['cluster']} but the write-up denies it "
                         f"({contras[0][:100]}) — re-prompting")
                    prompt = _recon_prompt(ctx, extra=_contradiction_msg(launch_facts,
                                                                          contras))
                    continue
                _log(f"meta/recon: teacher finished reconnaissance after {r} run(s)")
                _require_recon_artifacts(ctx, be, sass_ok, sched_ok, live_ok,
                                         stall_ok, struct_ok, since=session_start,
                                         near_miss=near_miss, launch=launch_facts)
                _save_recon_state(ctx, rounds=r)
                _write_family_analysis(ctx)
                return ReconResult(rounds=r, report=ctx.recon_path, finished=True)
            _log("meta/recon: turn ended with neither a RECON_REQUEST nor the "
                 "write-up artifacts (written this session) — re-prompting")
            prompt = _recon_prompt(ctx, extra=(
                "\n\n## The previous turn did not end in a legal way\n\n"
                "Your previous turn neither sent `RECON_REQUEST: notes/<name>.py` "
                "nor wrote the two deliverables to disk. There are only two legal endings:\n"
                "1. **More reconnaissance** → write the script into the recon tree and end "
                "with `RECON_REQUEST: <real path>`;\n"
                "2. **Reconnaissance done** → first write both deliverables, the recon report "
                "and the candidate architecture table, then end the turn.\n"
                "If you meant to continue last time but the turn was cut off, pick up where "
                "you left off."))
            continue
        # A request naming a file that is not on disk would still cost a full GPU job
        # to learn "can't open file" — observed verbatim: a teacher echoed the
        # prompt's own `notes/<name>.py` placeholder and a job was burned on it.
        # Round budget is charged to the teacher's mistake, not the cluster.
        # Only checkable when the reconnaissance tree's path is known.
        if ctx.recon_dir is not None and not (ctx.recon_dir / turn.path).is_file():
            _log(f"meta/recon: requested script {turn.path} does not exist — re-prompting")
            prompt = _recon_prompt(ctx, extra=(
                f"\n\n## The previous RECON_REQUEST was rejected\n\n"
                f"You asked to run `{turn.path}`, but there is no such file under "
                f"`{ctx.recon_dir}`. **Write the script into the recon tree first, then send "
                f"the request with its real path** (the `<name>` in "
                f"`RECON_REQUEST: notes/<name>.py` is a placeholder; replace it with the file "
                f"name you actually wrote)."))
            continue
        rbe.commit(f"recon: {turn.path}")
        try:
            log = rbe.submit_gpu("main", f"cmd:python3 {turn.path}")
        except BackendError as e:
            # A starved queue or a dead job must not end the whole run (a probe once
            # sat Queued past the poll bound and the uncaught raise threw away 3.5h of
            # reconnaissance). The round is charged and the teacher is told — the same
            # trust model as a probe that failed to build.
            log = (f"[loop] GPU job failed to return: {e}\n"
                   f"The round is charged to the budget and its result is unavailable. "
                   f"Either schedule that measurement again in the next round, or wrap up "
                   f"from the notes you have — marking the facts you could not measure "
                   f"honestly as \"not measured\".")
            _log(f"meta/recon: submit_gpu failed ({e}) — round charged, teacher informed")
        else:
            from forgekernel.skills.stall_price_run import expand_bundle
            log = expand_bundle(log)   # job log archives clip the head; see the tool
            sass_ok = sass_ok or _SASS_OK.search(log) is not None
            sched_ok = (sched_ok or _SCHED_OK.search(log) is not None
                        or _SCHED_NOT_APPLICABLE.search(log) is not None)
            live_ok = live_ok or _LIVE_OK.search(log) is not None
            _collect_verdict_near_misses(log, near_miss)
            cluster = _launch_cluster_from_log(log)
            if cluster is not None:
                # A later round that profiled a helper / setup kernel must not demote
                # the write-up's launch facts to that kernel's shape: the reading a
                # clustered launch produced stands (compared by CTA count per cluster,
                # so a (2,8,1) reading beats a (16,1,1) one on size, not on dim0).
                # The failure case was a report whose §1 denied cluster while a round
                # carried a helper's (0,0,0) — that round used to overwrite the
                # reference's. A genuine contradiction is still caught: whichever
                # reading wins, cluster > 1 contradicts the denial, and
                # LAUNCH-DISPUTED owns the disagreement from the report's side.
                prev = launch_facts.get("cluster") or (0, 0, 0)
                if cluster[0] * cluster[1] * cluster[2] >= prev[0] * prev[1] * prev[2]:
                    launch_facts.update(cluster=cluster, round=r, source=(
                        "LAUNCH-OK" if _LAUNCH_OK.search(log) else
                        "launch__cluster_dim_* metric in the job log"))
                    _log(f"meta/recon: launch cluster={cluster} read from round {r}")
            if _STALL_OK.search(log) is not None:
                stall_ok = True
            if _STRUCT_OK.search(log) is not None:
                struct_ok = True
            _save_ref_stall_ledger(ctx, log)   # banks what self-checked, per table
            if _STRUCT_OK.search(log) is not None:
                _save_ref_struct_table(ctx, log)
            if _ORCH_OK.search(log) is not None:
                _save_ref_orch_dossier(ctx, log)
            if _ROLE_ANY.search(log) is not None:
                _save_ref_role_exposure(ctx, log)
            _save_ref_pm_timeline(ctx, log)
            _save_recon_log(ctx, turn.path, r, log)
            _log(f"meta/recon: ran {turn.path} ({len(log)} chars back)")
            # The next round's `rbe.commit` would carry these, but the LAST round
            # has no next one — and a job ships a commit, so an unbanked ledger
            # never reaches the node it is meant to inform.
            _bank_evidence(ctx, f"round {r} ledgers")
        prompt = _recon_prompt(ctx, extra=(
            f"\n\n## Output of the previous recon run ({turn.path})\n\n"
            f"```\n{log[-6000:]}\n```\n\n"
            "Read it, then either send the next RECON_REQUEST or write the two deliverables "
            "and finish.\n"
            "If this run measured nothing (the script errored, a metric name was misspelled, "
            "no kernel matched), **fix the script before going on** — one failed run does not "
            "cost you a conclusion, but treating it as \"this fact is unavailable\" does."))
    _log(f"meta/recon: round budget ({ctx.max_recon_rounds}) exhausted")
    _ensure_ref_pricing(ctx, label="meta/recon")
    _require_recon_artifacts(ctx, be, sass_ok, sched_ok, live_ok, stall_ok,
                             struct_ok, since=session_start, near_miss=near_miss,
                             launch=launch_facts)
    # The exhausted path can end in two states the log's WARNING branch already
    # distinguishes: write-up complete (persist — a restart must not re-litigate
    # it) or still incomplete (don't — a fresh session's recon turns are the only
    # remaining chance to finish the write-up).
    if (_recon_artifacts_written(ctx, since=session_start)
            and not _missing_recon_evidence(ctx, sass_ok, sched_ok, live_ok,
                                            stall_ok, struct_ok, near_miss=near_miss)
            and not _launch_contradictions_open(ctx, launch_facts)):
        _save_recon_state(ctx, rounds=ctx.max_recon_rounds)
    return ReconResult(rounds=ctx.max_recon_rounds, report=ctx.recon_path, finished=False)


def _recon_artifacts_written(ctx: LoopContext, since: float | None = None) -> bool:
    """Both write-up artifacts exist — the only acceptable evidence of a finish.

    ``since`` (a POSIX timestamp) additionally requires both files to have been
    written after that moment. Existence alone once let a run finish against a
    stale report already on disk — including the wrong operator's, since the
    default recon path ships populated in-tree.

    When the report path is unknown there is nothing to hold a finish against, so
    the request-less turn is taken at its word (the pre-fix behavior).
    """
    if ctx.recon_path is None:
        return True
    candidates = ctx.recon_path.parent / "candidates.md"
    for p in (ctx.recon_path, candidates):
        try:
            mtime = p.stat().st_mtime
        except OSError:  # missing (FileNotFoundError) or unreadable
            return False
        if since is not None and mtime < since:
            return False
    return True


# The verdict line only sass_tools.analyze prints, and only after its self-checks
# (right kernel, instruction floor, tensor families, no swallowed profiler errors)
# all pass. The full field pattern is required so an incidental "SASS-OK" in prose
# does not open the gate; a teacher printing the line by hand without the tool is
# fabricating evidence, and the transcript audit — not this regex — is the
# backstop for that (same trust model as probe verdicts).
#
# The kernel field is `.+?`, NOT `\S+`: a demangled C++ template name carries
# spaces ("void sparse_attn_fwd_kernel<KernelTemplate<512, 1>, ..."), and \S+
# stops at the first one so the trailing full-field anchor never matches — a
# false NEGATIVE on the tool's own genuine output (a sparse recon burned its
# write-up turns against exactly this; the teacher rightly refused to waive a
# channel it had delivered). The anti-cheat force lives in the line anchor plus
# the trailing numeric fields, which `.+?` preserves.
_SASS_OK = re.compile(r"^SASS-OK kernel=.+? instructions=\d+ opcodes=\d+", re.M)
# The explicit escape hatch: one report line owning the absence and its reason.
# Machine-checkable, and it surfaces on the report's first screen instead of in a
# late section.
_SASS_WAIVED = re.compile(r"^SASS-WAIVED:\s*\S", re.M)

# The schedule-transcription and liveness-ledger verdicts, printed only by
# sched_live_tools.report after its own self-checks. Same trust model and same
# full-field patterns as SASS-OK: prose cannot open the gate. These two became
# mandatory after a bwd run where every counting fact was delivered and the
# milestone still could not say which chain flies over which elementwise segment
# (the schedule) or what register peak that overlap costs (the liveness) — the
# pair of facts its M1 died without.
#
# Three forms, one per MMA ISA the tool transcribes (sched_live_tools picks the
# grammar by census): Hopper wgmma keeps its original fields; Blackwell tcgen05
# and per-warp mma.sync carry `isa=` and their own counts. A dsa_sparse
# B200 teacher's hand-written HMMA tool printed the mma_sync form and this regex
# (then HGMMA-only) refused it for a day — the report had to double-write a
# DELIVERED-MISMATCH line to close a channel that WAS delivered.
_SCHED_OK = re.compile(
    r"^SCHED-OK (?:isa=\S+ )?loop=\S+ (?:hgmma_runs=\d+ depbars=\d+"
    r"|mma_runs=\d+ barriers=\d+ ldsm=\d+"
    r"|utcmma_runs=\d+ commits=\d+ mbar_waits=\d+)", re.M)
_SCHED_WAIVED = re.compile(r"^SCHED-WAIVED:\s*\S", re.M)
# sched_live_tools models the Hopper warpgroup-MMA schedule (chains by accumulator
# register, WARPGROUP.DEPBAR.LE wait depths). A tcgen05 kernel has neither — the
# accumulator lives in TMEM and the waits are mbarrier — and an mma.sync (HMMA)
# kernel has no warpgroup chain to wait on at all; on either the tool refuses
# rather than reporting zeros, and says so on its own verdict line. That is a
# delivered answer about the instrument, not a missing channel: it opens the SCHED
# gate the way a waiver does, so a run is not held hostage by a tool that cannot
# see its ISA (a gdn_prefill B200 run's whole 12-round recon budget went
# to the tcgen05 case; a dsa_sparse B200 run hit the HMMA case six times). The tool only prints the line when the page HAS an MMA of another
# ISA — a page with no MMA at all is a wrong page, not a wrong instrument, and
# still fails. The teacher still owes the schedule through another channel; the
# recon prompt and the tool's own output both say which.
_SCHED_NOT_APPLICABLE = re.compile(
    r"^SCHED-NOTAPPLICABLE isa=\S+ tcgen05_mma=\d+ wgmma=\d+ mma_sync=\d+ reason=\S+",
    re.M)
_LIVE_OK = re.compile(r"^LIVE-OK peak=\d+ guessed_frac=", re.M)
_LIVE_WAIVED = re.compile(r"^LIVE-WAIVED:\s*\S", re.M)

# §1 launch configuration: the cluster shape is a launch ATTRIBUTE, not a grid
# dimension, and the report used to infer it from the grid. The dsa_sparse
# B200 reference launches a flat 128-CTA grid with launch__cluster_dim_x=16; the
# report's §1 said "no cluster", the candidates table reverse-engineered a dense
# GEMM tile with cluster=(1,1,1), the first milestone copied it and the student's
# first kernel was 16x slower than the reference. The driver now reads the
# cluster off every job log — the seeded launch_tools verdict line, or the raw
# NCU metric a probe printed — and holds the write-up against it.
_LAUNCH_OK = re.compile(
    r"^LAUNCH-OK kernel=.+? grid=\(\d+,\d+,\d+\) block=\(\d+,\d+,\d+\) ctas=\d+ "
    r"cluster=\((\d+),(\d+),(\d+)\) cluster_max_active=(\d+|n/a)", re.M)
_CLUSTER_METRIC = {d: re.compile(rf"\blaunch__cluster_dim_{d}\b[^\d\n-]{{0,40}}?(\d+)")
                   for d in "xyz"}
# What a report / candidates table says when it did NOT read the metric. The gate
# reads the DECLARATION, not any sentence containing the word: the brief requires
# §1 and each candidate's first line to write the shape, so the denial is a cluster
# field whose value is 1 / none / n/a, plus the explicit "no cluster" phrase when it
# is a claim about the launch (a correction or an alternative candidate that says it
# is not the measured launch is not a denial — a review measured the old
# pattern bouncing "§1 correction: the earlier 'no cluster' reading was wrong",
# "Candidate B — no-cluster variant" and even the tool's own LAUNCH-PARTIAL text, three
# distinct ways of losing a write-up turn for saying the right thing).
_ONE = r"(?:1|one|single|none|n/?a|no)"
# One value of a shape field: `1`, `none`, `1x1x1`, `(1,1,1)`, `1, 1, 1`, `single CTA`.
# The tail anchor is what separates a declaration from a sentence that merely starts
# with a 1-shaped number (`cluster=16, one group per 16 CTAs` is not a declaration of
# one); a parenthetical aside after the value is allowed (`none (each CTA independent)`).
_ONE_VALUE = (r"\s*[*`]*\s*(?:\(?\s*" + _ONE + r"(?:\s*[,x×]\s*" + _ONE + r"){0,2}\s*\)?)"
              r"\s*(?:CTAs?|ctas?)?\s*")
_FIELD_TAIL = (_ONE_VALUE
               + r"(?:\s*[（(][^）)]{0,40}[）)])?\s*[*`]*"
               # a trailing aside: `none — see the LAUNCH-OK line` (ASCII hyphens stay legal in it)
               + r"(?:\s*[—–]\s*[^—–\n]{0,60}?)?"
               + r"\s*[*`]*\s*[。.；;，,]?\s*$")
# The cluster field: the key itself — not `cluster_dim_x`, not `clusterDim.x` — and
# whatever separator the writer used (ASCII or full-width colon / equals, a
# parenthesis, a markdown cell). `cluster = 1` and `cluster shape: none` both match;
# the value side is decided in `_cluster_field_value`.
_CLUSTER_KEY = re.compile(
    r"(?:^|[\s|*`()])cluster[_ ]?(?:shape|size|dims?)?(?![_A-Za-z0-9])"
    r"\s*[*`]*\s*[:=(]?\s*[*`]*\s*", re.I)
_NO_CLUSTER_CLAIM = re.compile(
    r"\b(?:no|without|non)[- ]?cluster\b(?!\s*(?:barrier|wait|arrive|broadcast|sync))"
    r"|\bnot\s+clustere?d\b"
    r"|\b(?:does\s+not|doesn't)\s+use\s+(?:a\s+)?cluster\b"
    r"(?!\s*(?:barrier|wait|arrive|broadcast|sync))", re.I)


def _a_one_shaped_cluster_field(line: str) -> bool:
    """``line`` declares the launch's cluster as one CTA / none: the cluster field
    followed by a value that is only 1 / one / none — ``cluster=(1,1,1)``,
    ``cluster shape: none``, ``| cluster | none |``.

    The gate reads the DECLARATION rather than any sentence containing "cluster":
    the brief requires §1 and each candidate's first line to write the shape, so a
    line that says what the cluster IS is the claim to hold the report to. A
    sentence about something else — a ``no cluster barrier`` in the mainloop, an
    alternative candidate that is not the measured launch, a correction, a quoted
    tool warning — is not a denial (a review measured the old line-wide
    pattern bouncing all four of those, burning a write-up turn each)."""
    value = _cluster_field_value(line or "")
    if value is None:
        return False
    return bool(re.match(_FIELD_TAIL, value.strip(), re.I))


def _cluster_field_value(line: str) -> str | None:
    """The value the line gives the cluster field, or None when the line has no
    cluster field. `| cluster | none | other |` yields `none` (the first cell)."""
    m = _CLUSTER_KEY.search(line)
    if not m:
        return None
    rest = line[m.end():].strip()
    if rest.startswith("|"):
        rest = rest[1:]
    rest = rest.strip()
    if "|" in rest:
        rest = rest.split("|")[0]
    if not rest:
        return None
    # The field's value must be an ENTIRE value, not the start of a sentence about
    # something else: `cluster=16, one group per 16 CTAs` is not a 1-shaped declaration.
    return rest if re.match(_FIELD_TAIL, rest, re.I) else None


# An aside that states the shape is being corrected, or that the shape belongs to
# something other than the measured launch (an alternative candidate, another
# kernel, a quoted tool warning), is not a claim to hold the report to. It cannot
# excuse a shape DECLARATION (see _launch_contradictions): a correction that still
# declares `cluster=(1,1,1)` gets bounced until it declares what was measured.
_NO_CLUSTER_EXEMPT = re.compile(
    r"correction|corrected|earlier|previous(?:ly)?|wrong|incorrect|mistaken|erroneous|untrue"
    r"|absence is not evidence|does(?:\s+not|n't)\s+(?:mean|imply|show)|is\s+not\s+evidence"
    r"|LAUNCH-PARTIAL",
    re.I)
# An alternative candidate that states what IT is not. The brief builds every
# candidate on the measured launch, so this is a statement about the candidate, not
# a denial of the measurement.
_ALTERNATIVE_CANDIDATE = re.compile(
    r"\bcandidate\s*[A-Za-z0-9]?\s*[—\-–:][^\n|]{0,80}?\b(?:no|without|non)[- ]?cluster", re.I)


def _launch_cluster_from_log(log: str) -> "tuple[int, int, int] | None":
    """The cluster shape a job log carries: the launch_tools verdict line, or the
    raw ``launch__cluster_dim_*`` metric a probe printed (a page with only ``_x``
    reads as (x,1,1)). None when the log says nothing about it.

    A log with several kernels is read through ``launch_tools.kernel_section``, so
    the metrics come from the LAST profiled kernel rather than whichever helper
    appeared first: a dsa_sparse page carried ``cluster_dim_x=16`` for the
    reference and a helper's ``(0,0,0)`` above it, and taking the first turned the
    gate off (a "no cluster" write-up finished cleanly against a clustered
    reference). The chosen section is used for the LAUNCH-OK line too — a verdict
    line printed for a section that is not this one must not certify it."""
    try:
        from forgekernel.skills.launch_tools import kernel_section
        section = kernel_section(log or "")
    except Exception:  # noqa: BLE001 — a reader must not become the failure
        section = log or ""
    m = _LAUNCH_OK.search(section)
    if m is not None:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    dims = {}
    for d, rx in _CLUSTER_METRIC.items():
        mm = rx.search(section)
        if mm is not None:
            dims[d] = int(mm.group(1))
    if "x" not in dims:
        return None
    return (dims["x"], dims.get("y", 1), dims.get("z", 1))


def _launch_contradictions(ctx: LoopContext, launch: "dict | None") -> list:
    """Report / candidates lines that deny a cluster the NCU launch metrics read
    (cluster > 1 in ``launch``). Each entry quotes the line and its file.

    Two kinds of denial, with different exemptions:

    * a shape DECLARATION (``cluster=(1,1,1)``, ``cluster shape: none``): the line
      says what the launch's cluster IS, contradicted by the measurement. Nothing
      excuses it — a "correction" that still declares one CTA is still the wrong
      declaration, and the fix is to write the measured shape;
    * a prose claim (``no cluster``, ``without cluster``): excused when the line is
      correcting the old reading, is about something else (a barrier), or is an
      alternative candidate stating what it is not.
    """
    if not launch or not launch.get("cluster") or max(launch["cluster"]) <= 1:
        return []
    if ctx.recon_path is None:
        return []
    found = []
    for p in (ctx.recon_path, ctx.recon_path.parent / "candidates.md"):
        try:
            text = p.read_text()
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            declared = _a_one_shaped_cluster_field(line)
            prose = bool(_NO_CLUSTER_CLAIM.search(line)) and not (
                _NO_CLUSTER_EXEMPT.search(line) or _ALTERNATIVE_CANDIDATE.search(line))
            if declared or prose:
                found.append(f"{p.name}:{n}: `{line.strip()[:160]}`")
    return found


def _contradiction_msg(launch: dict, found: list) -> str:
    c = launch["cluster"]
    src = launch.get("source", "job log")
    return (
        "\n\n## Finish refused: RECON-CONTRADICTION cluster\n\n"
        f"The NCU launch metrics read **cluster=({c[0]},{c[1]},{c[2]})** ({src}, round "
        f"{launch.get('round', '?')}), but the report / candidates table has these lines "
        "denying a cluster:\n\n"
        + "\n".join(f"- {f}" for f in found) +
        "\n\nThe cluster shape is a **launch attribute** (`launch__cluster_dim_x/y/z`), not "
        "a grid dimension; a flat grid can just as well be a cluster launch. Rewrite §1 and "
        "the candidates table to the measured cluster (the first line of every candidate "
        "states the cluster shape and CTA count first, and the tile derivation takes it as "
        "its premise), or, if you believe the metric was misread, write a line "
        "`LAUNCH-DISPUTED: <the log line carrying the reading + your criterion>` in the "
        "report."
    )


_LAUNCH_DISPUTED = re.compile(r"^LAUNCH-DISPUTED:\s*\S", re.M)


def _launch_contradictions_open(ctx: LoopContext, launch: "dict | None") -> list:
    """``_launch_contradictions`` unless the report owns the disagreement with a
    ``LAUNCH-DISPUTED:`` line — the teacher may have a reason to distrust the
    metric (a wrong kernel profiled), and that reason belongs on the record,
    not in an unresolvable re-prompt loop."""
    found = _launch_contradictions(ctx, launch)
    if not found or ctx.recon_path is None:
        return found
    try:
        if _LAUNCH_DISPUTED.search(ctx.recon_path.read_text()) is not None:
            return []
    except OSError:
        pass
    return found

# The per-instruction stall-pricing verdict, printed only by stall_tools after its
# self-checks (samples floor, mis-aim duration floor, ledger coverage >= 90%).
# Sedimented from a forensic analysis that used to be a manual between-run step: the
# static channels above transcribe structure, this one prices measured time — the
# only channel that can see a gap dispersed through mainloop interleave quality.
# kernel=.+? for the same demangled-name-with-spaces reason as _SASS_OK.
_STALL_OK = re.compile(r"^STALL-OK kernel=.+? samples=\d+ rows=\d+ coverage=\d", re.M)
_STALL_WAIVED = re.compile(r"^STALL-WAIVED:\s*\S", re.M)

# The per-segment construction-table verdict, printed only by struct_tools
# after its self-checks (barrier-fenced region, decisive MMA/LDSM/STSM family,
# integral warp attribution). The channel that says how a phase is BUILT — the
# SASS histogram says what opcodes exist, this says which warps do a phase on
# what tile granule in what precision with what hand-off. Sedimented from a
# gdn_prefill_fwd blackbox run: the whole gap sat in the solver's
# triangular inverse, the recon inferred its construction ("four 16x16 blocks")
# from the instruction mix, and the student built the wrong shape. Mandatory on
# blackbox runs only, where no source channel can fill it.
_STRUCT_OK = re.compile(
    r"^STRUCT-OK kernel=\S+ regions=\d+ segments=\d+ warps=\S+ precision=\S+ granule=\S+", re.M)
_STRUCT_WAIVED = re.compile(r"^STRUCT-WAIVED:\s*\S", re.M)

# Full-field verdict patterns per channel, for the near-miss audit below.
_VERDICT_PATTERNS = {"SASS": _SASS_OK, "SCHED": _SCHED_OK,
                     "LIVE": _LIVE_OK, "STALL": _STALL_OK, "STRUCT": _STRUCT_OK}

# The third legal evidence state, next to delivered and waived: delivered but
# unrecognized by the acceptance pattern. The teacher quotes the tool's actual
# verdict line after this marker; the report line is machine-checkable, shows on
# the first screen, and reads as an acceptance-format defect for the human audit
# — NOT as a missing channel. Without it, a checker false-negative plus a
# teacher honestly refusing to waive a delivered channel is an unresolvable
# re-prompt loop (observed in a sparse recon).
_DELIVERED_MISMATCH = {n: re.compile(rf"^{n}-DELIVERED-MISMATCH:\s*\S", re.M)
                       for n in ("SASS", "SCHED", "LIVE", "STALL", "ROLE", "PMTL", "STRUCT")}


def _collect_verdict_near_misses(log: str, near: dict) -> None:
    """Record verdict-shaped lines that fail their full-field pattern.

    A line that STARTS like a tool verdict ("SASS-OK ...") but does not satisfy
    the full-field regex is almost never fabrication — it is an acceptance-format
    mismatch between the tool's real output and this file's pattern (observed in
    a sparse recon: a demangled kernel name with spaces). Without this audit the
    driver re-prompts "evidence missing", the teacher truthfully insists it was
    delivered, and the deadlock burns every write-up turn. Recording the nearest
    miss lets the gate message show BOTH sides of the disagreement on the first
    turn instead of never."""
    # Only the tool verdicts whose acceptance regex can lag the tool's real
    # print. ROLE / PMTL are deliberately NOT audited here: a `ROLE-OK` line
    # without `tensor=` is not a format mismatch, it is the exposure column
    # the narrowing rejects on purpose (the seeded tool prints ROLE-PARTIAL for
    # it; only an older copy prints ROLE-OK), and a DELIVERED-MISMATCH row
    # must never close those channels — the driver's own pricing or a waiver do.
    for name, pat in _VERDICT_PATTERNS.items():
        if pat.search(log) is not None:
            near.pop(name, None)
            continue
        m = re.search(rf"^{name}-OK\b.*", log, re.M)
        if m is not None:
            # Last observation wins: a round-1 garbage line (partial echo,
            # truncation) must not shadow the real tool verdict a later round
            # prints — the gate note quotes this line verbatim, and preserving
            # the WRONG side of the disagreement would instruct the teacher to
            # enshrine garbage in the DELIVERED-MISMATCH record.
            near[name] = m.group(0)[:240]
# The ledger body stall_tools brackets for verbatim lifting into prompts.
_STALL_LEDGER_RE = re.compile(
    r"=== STALL LEDGER BEGIN ===\n(.*?)\n=== STALL LEDGER END ===", re.S)
# The per-segment account (PC-keyed join of stall cost onto the schedule): the
# row ledger keys on source lines and the student's scheduling rows (HGMMA /
# DEPBAR / barrier — the @cute.jit decorator has no line) pile into one
# unresolved bucket, invisible to a diagnosis. Segment rows carry `SEG` machine
# lines the aligned comparison table parses.
_SEGMENT_LEDGER_RE = re.compile(
    r"=== SEGMENT LEDGER BEGIN ===\n(.*?)\n=== SEGMENT LEDGER END ===", re.S)

# The PM-sampling phase-timeline verdict (pm_timeline_tools): the stall ledger
# prices WHERE time leaks per instruction; the timeline reads WHEN — phase
# structure a per-PC sum disperses (each phase touches the same PCs). Optional
# channel: probes call it for phase-shaped hypotheses; nothing gates on it.
_PMTL_OK = re.compile(
    r"^PMTL-OK kernel=\S+ series=\d+ stall_series=\d+ stall_samples=\d+", re.M)
_PMTL_WAIVED = re.compile(r"^PMTL-WAIVED:\s*\S", re.M)
_PM_TIMELINE_RE = re.compile(
    r"=== PM TIMELINE BEGIN ===\n(.*?)\n=== PM TIMELINE END ===", re.S)
# The ref's timeline is banked beside its role table: together they are the
# ref's overlap account (WHEN each role waits, and how much of that waiting the
# tensor pipe hides). Static SASS cannot read either — see the ROLE comment.
_REF_PM_TIMELINE_NAME = "ref_pm_timeline.txt"

# The orchestration-dossier verdict (orch_tools): the machine form of §2g —
# role-partitioned regions, barrier identity (named ids + mbarrier objects via
# address def-chains), arrive→wait edges, and the remaining §2g bullets.
# Advisory this generation: recognised and banked, but not in the finish gate —
# a brand-new acceptance pattern hard-gating recon is how the near-miss deadlock
# started, and the near-miss machinery only softens that, it does not unprice
# it. The dossier body is banked for the distill/diagnose steps the same way
# the ref stall ledger is.
_ORCH_OK = re.compile(
    r"^ORCH-OK regions=\d+ loops=\d+ nbar_ids=\d+ mbar_objs=\d+ "
    r"resolved=\d+ edges=\d+ itc_candidates=\d+", re.M)
_ORCH_DOSSIER_RE = re.compile(
    r"=== ORCH DOSSIER BEGIN ===\n(.*?)\n=== ORCH DOSSIER END ===", re.S)
_ORCH_SUMMARY_RE = re.compile(r"^ORCH-SUMMARY-JSON: (\{.*\})$", re.M)
_REF_ORCH_DOSSIER_NAME = "ref_orch_dossier.txt"

# The role exposure ledger (role_exposure_tools): the dossier's role-labelled
# regions x the PC-sampled stall ledger x the PM tensor plateau, joined into
# one per-role table — the teacher had been re-deriving it by hand in a
# verdict probe every meta step. Banked like the dossier (ref side at recon,
# student side off the pricing job) and diffed driver-side from the ROLE-JSON
# core.
#
# ROLE-OK carries the tensor term (the exposure column is real); ROLE-PARTIAL
# is the same table without it (shares only, exposed_ns null). Both are banked
# — the shares are evidence — but only ROLE-OK is the overlap account, and
# only ROLE-OK on the REF closes the recon channel: a dsa_sparse run
# reconnoitred with a bare --set source collection, its ref table came back
# tensor=unavailable, the old single pattern accepted it, and every teacher
# entry of a 17h run diagnosed against a ref exposure column of nulls. The
# blackbox teachers of four runs guessed the gather/MMA overlap wrong at v1;
# this is the measurement they never had.
_ROLE_OK = re.compile(
    r"^ROLE-OK roles=\d+ samples=\d+ resolved=[\d.]+ tensor=[\d.]+", re.M)
# Pre-split tools (a recon tree seeded before this change) print ROLE-OK with
# no tensor field; that line is a partial table, not the account.
_ROLE_ANY = re.compile(r"^ROLE-(?:OK|PARTIAL) roles=\d+ samples=\d+ resolved=[\d.]+", re.M)
_ROLE_WAIVED = re.compile(r"^ROLE-WAIVED:\s*\S", re.M)
# The two overlap channels (ROLE / PMTL) are checked against the BANKED
# evidence rather than a per-log flag: the driver's own pricing job delivers
# them, and a later ROLE-PARTIAL must not shadow an earlier ROLE-OK. They have
# no near-miss / DELIVERED-MISMATCH state (see _collect_verdict_near_misses):
# a "ROLE-OK ..." line without the tensor field is the pre-split tool's output,
# i.e. the exposure column the narrowed pattern rejects on purpose.
_ROLE_BLOCK_RE = re.compile(
    r"=== ROLE EXPOSURE BEGIN ===\n(.*?)\n=== ROLE EXPOSURE END ===", re.S)
_REF_ROLE_EXPOSURE_NAME = "ref_role_exposure.txt"
_STUDENT_ROLE_EXPOSURE_NAME = "student_role_exposure.txt"
_REF_PRICE_JOB_LOG_NAME = "ref_price_job.log"   # the ref pricing attempts: see _REF_PRICE_MARKER_NAMES
_STUDENT_ORCH_DOSSIER_NAME = "student_orch_dossier.txt"        # tree under judgment (HEAD)
_STUDENT_BEST_ORCH_DOSSIER_NAME = "student_orch_dossier_best.txt"  # best-so-far KEEP


def _orch_summary(dossier_text: str | None) -> dict | None:
    """The comparable core a banked dossier carries as its ORCH-SUMMARY-JSON
    line; None when the text predates the line or is unparsable."""
    if not dossier_text:
        return None
    m = _ORCH_SUMMARY_RE.search(dossier_text)
    if m is None:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


@dataclass
class PricedTrees:
    """What one teacher entry priced: the tree under judgment (HEAD) and, when
    it is a different commit, the best-so-far KEEP.

    ``ledger``/``segments`` come from the best tree when one exists (the
    ledger rules in the diagnose prompt speak of "student best"), else from
    HEAD. ``head_dossier``/``best_dossier`` are the orchestration dossiers of
    the respective trees; when HEAD is the best, both name the same text."""
    head_sha: str | None = None
    best_sha: str | None = None
    ledger: str | None = None
    segments: str | None = None
    head_dossier: str | None = None
    best_dossier: str | None = None
    roles: str | None = None          # role exposure block of the ledger-source tree
    # Which tree the ledger/segments/roles were priced on ("best" or "HEAD"):
    # the diagnose labels say "student best" and must not when there is no best.
    ledger_tree: str = "best"
    # Pricing jobs that failed or came back without their tables, one line each.
    # Two jobs can half-fail; the surviving reading then rides into the prompt
    # under labels written for the both-succeeded case unless the gap is named.
    missing: list = field(default_factory=list)

    @property
    def head_is_best(self) -> bool:
        return bool(self.best_sha) and self.best_sha == self.head_sha


def _price_student_trees(ctx: LoopContext, records) -> PricedTrees:
    """Price the trees a teacher entry must read: best (ledger source) and the
    tree under judgment (HEAD) — one job when they coincide, two when not.

    HEAD is the tree the stop was decided on: at first_green it is the M0 that
    just closed, at plateau the segment's last KEEP. Its dossier is what an
    audit needs (did the rebuild carry the contract the previous patch wrote?)
    and the best-so-far dossier is what the plateau ledger already priced; the
    diff between the two says what the construction actually changed
    structurally. A missing best (pre-green plateau) prices HEAD alone, as
    before."""
    be = ctx.backend
    # the same tree the rollback point and the stall pricing use: the best
    # KEEP on the active machine, else anywhere (a best from another machine is
    # a verified kernel the student is not standing on)
    out = PricedTrees(head_sha=be.head(),
                      best_sha=_best_good_ref_here(records, ctx))
    primary = out.best_sha or out.head_sha
    banks = [_STUDENT_BEST_ORCH_DOSSIER_NAME] if out.best_sha else []
    if not out.best_sha or out.head_is_best:
        banks.append(_STUDENT_ORCH_DOSSIER_NAME)
    out.ledger_tree = "best" if out.best_sha else "HEAD"
    out.ledger, out.segments, dossier, out.roles = _run_stall_pricing(
        ctx, ref=primary, bank_as=tuple(banks),
        label="best" if out.best_sha else "head")

    def short(sha):
        return (sha or "?")[:10]

    if out.ledger is None and dossier is None:
        out.missing.append(f"{out.ledger_tree} {short(primary)}: pricing job failed or its "
                           "self-check did not pass — no ledger, no segment ledger, no dossier, "
                           "no role ledger")
    elif dossier is None:
        out.missing.append(f"{out.ledger_tree} {short(primary)}: dossier unavailable "
                           "(ledger present)")
    elif out.ledger is None:
        out.missing.append(f"{out.ledger_tree} {short(primary)}: ledger self-check failed, "
                           "no ledger (dossier present)")
    if out.best_sha:
        out.best_dossier = dossier
    if not out.best_sha or out.head_is_best:
        out.head_dossier = dossier
    else:
        _, _, out.head_dossier, _ = _run_stall_pricing(
            ctx, ref=out.head_sha, bank_as=(_STUDENT_ORCH_DOSSIER_NAME,),
            label="head")
        if out.head_dossier is None:
            out.missing.append(f"HEAD {short(out.head_sha)}: pricing job failed or its "
                               "self-check did not pass — the tree judged this time has no "
                               "dossier; the ref vs HEAD / best → HEAD faces are absent")
    return out


def _meta_workload_readings(ctx: LoopContext, priced: PricedTrees, label: str) -> str:
    """A group card's other meta workloads at a teacher entry (loop/meta_shapes.py): the
    student's stall / SASS pricing, the reference's (once per run) and the timeline probe
    on each, on the tree the representative's readings were taken on, one job each --
    rendered beside the representative's. Every machine reading (the dossier diff gates,
    the segment table, the timeline decisions) stays on the representative. A dossier
    whose summary matches the representative's on every axis is the same SASS structure
    and is said so, not listed again. '' for a one-workload card."""
    meta = meta_shapes.card_meta_workloads(ctx.workspace)
    if len(meta) < 2 or not ctx.stall_pricing:
        return ""
    sha = priced.best_sha or priced.head_sha
    rep_summary = _orch_summary(priced.best_dossier if priced.best_sha else priced.head_dossier)
    rows = []
    for wl in meta[1:]:
        row: dict = {"wl": wl}
        rows.append(row)
        if ctx.now() > ctx.deadline_epoch - 1800:
            row["skipped"] = ("less than 30 minutes of budget left; no job was bought for "
                              "this workload")
            continue
        ledger, _segs, dossier, roles = _run_stall_pricing(
            ctx, ref=sha, bank_as=(), label=f"{priced.ledger_tree} {wl}", workload=wl)
        summary = _orch_summary(dossier)
        row.update(ledger=ledger, roles=roles)
        if ledger is None and dossier is None:
            row["missing"] = "stall / SASS pricing job failed or its self-check did not pass"
        if summary is not None and rep_summary is not None:
            if summary == rep_summary:
                row["same_as_rep"] = True
            else:
                row["diff_rep"] = _orch_diff_text(rep_summary, summary, names=(meta[0], wl))
        if (ref_log := _ref_workload_pricing(ctx, wl, label)):
            rm = _ROLE_BLOCK_RE.search(ref_log)
            row["ref_roles"] = rm.group(1) if rm else None
            dm = _ORCH_DOSSIER_RE.search(ref_log)
            ref_summary = _orch_summary(dm.group(1)) if dm else None
            if ref_summary is not None and summary is not None:
                row["diff_ref"] = _orch_diff_text(ref_summary, summary)
        if _timeline_on(ctx):
            tl = _run_timeline(ctx, sha, f"{label} {wl}", workload=wl)
            if tl and _timeline_renderable(tl):
                row["timeline"] = _timeline_tables(tl)
    return meta_shapes.workload_readings_block(
        meta[0], rows, meta_shapes.card_workload_axes(ctx.workspace), str(sha or "")[:10],
        rep_roles=priced.roles)


def _ref_workload_pricing(ctx: LoopContext, wl: str, label: str) -> str | None:
    """The reference tree's stall / SASS pricing on one of a group card's other meta
    workloads (``--workload``, before ``--app``): one job per workload on the tree
    _ensure_ref_pricing prices, its log banked as evidence ``ref_price_<wl>.log`` and read
    back at every later entry. None when the run prices no reference tree, the tree's card
    copy does not list the workload, or no usable reading came back.

    Only a log that carries a reading (_ref_workload_usable: a role table or a dossier) is
    banked as the workload's; one that came back without (ROLE-FAIL, a truncated or lost
    archive, a job that failed after launch) is an attempt, kept as
    ``ref_price_<wl>.attempt<n>.log`` under the key it ran on (_ref_pricing_key), and the
    workload is priced again at a later entry -- the rule _ensure_ref_pricing keeps for the
    representative: a log with a verdict that was not a reading is retried only once the
    tree, the launch target or the app spelling changed (the same tree reproduces it); one
    with no verdict at all (the archive lost it) also on an unchanged tree; at most
    _REF_WL_PRICE_ATTEMPTS attempts per workload per run. A job that never ran (NoRunError,
    the job id unmoved) is no attempt, bounded per session
    (_REF_PRICE_MAX_SUBMIT_FAILURES per workload)."""
    if ctx.ref_price_dir is None or ctx.ref_price_backend is None:
        return None
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return None
    path = ev / f"ref_price_{wl}.log"
    if path.is_file():
        try:
            banked = path.read_text()
        except OSError:
            return None
        if _ref_workload_usable(banked):
            return banked
        # banked before this rule (every finished job was): an attempt on an unknown key
        _ref_workload_attempt(ctx, ev, wl, "legacy", banked, label)
        path.unlink(missing_ok=True)
    entry, _named = _ref_price_entry(ctx)
    if (entry is None or not entry.is_file()
            or wl not in meta_shapes.card_meta_workloads(ctx.ref_price_dir)):
        return None
    attempts = _ref_workload_attempts(ev, wl)
    if len(attempts) >= _REF_WL_PRICE_ATTEMPTS:
        return None
    key = _ref_pricing_key(ctx)
    if attempts:
        head = re.match(r"\[ref-price\] key=(\S+) verdict=(\S+)",
                        attempts[-1].read_text(errors="replace"))
        if head and head.group(1) == key and head.group(2) != "none":
            return None   # same tree, a verdict that was no reading: it would repeat itself
    if ctx.ref_wl_price_submit_failures.get(wl, 0) >= _REF_PRICE_MAX_SUBMIT_FAILURES:
        return None
    be = ctx.ref_price_backend
    cmd = f"{_stall_price_cmd(ctx)} --workload {wl}"
    if ctx.ref_price_app:
        cmd += " --app " + " ".join(shlex.quote(a) for a in ctx.ref_price_app)
    head_before = getattr(be, "last_job_id", "")
    try:
        _seed_pricing_toolkits(ctx.ref_price_dir, be)
        be.commit("ref-price: snapshot the tree the pricing job archives")
        branch = getattr(be, "current_branch", lambda: "main")()
        _log(f"{label}/ref-price[{wl}]: pricing the reference tree on workload {wl} "
             f"(attempt {len(attempts) + 1}/{_REF_WL_PRICE_ATTEMPTS})")
        log = be.submit_gpu(branch, cmd)
    except BackendError as e:
        if (getattr(be, "last_job_id", "") not in ("", head_before)
                and not isinstance(e, NoRunError)):
            # it ran and did not come back (a hung reference kernel's poll timeout):
            # the same tree would hang again
            _ref_workload_attempt(ctx, ev, wl, key, f"[ref-price] job failed after launch: {e}\n",
                                  label, verdict="job-failed")
        else:
            ctx.ref_wl_price_submit_failures[wl] = ctx.ref_wl_price_submit_failures.get(wl, 0) + 1
            _log(f"{label}/ref-price[{wl}]: nothing ran ({e}) — not an attempt "
                 f"({ctx.ref_wl_price_submit_failures[wl]}/{_REF_PRICE_MAX_SUBMIT_FAILURES} "
                 f"this session); no reference column for {wl} this entry")
        return None
    from forgekernel.skills.stall_price_run import expand_bundle
    log = expand_bundle(log)
    if not _ref_workload_usable(log):
        _ref_workload_attempt(ctx, ev, wl, key, log, label)
        return None
    try:
        path.write_text(log)
    except OSError as e:
        _log(f"{label}/ref-price[{wl}]: could not persist {path.name}: {e}")
    _bank_evidence(ctx, f"ref-price {wl}")
    return log


# attempts at one non-representative workload's reference pricing per run
# (== len(_REF_PRICE_MARKER_NAMES), the representative's)
_REF_WL_PRICE_ATTEMPTS = 3
_REF_WL_VERDICT_RE = re.compile(r"^(?:STALL|ROLE|ORCH|PMTL)-(?:OK|PARTIAL|FAIL)\b.*$", re.M)


def _ref_workload_usable(log: str) -> bool:
    """A reference pricing log the other-workload section can read: a role table
    (ROLE-OK / ROLE-PARTIAL) or a dossier with a parsable summary."""
    if _ROLE_ANY.search(log) is not None and _ROLE_BLOCK_RE.search(log) is not None:
        return True
    dm = _ORCH_DOSSIER_RE.search(log)
    return dm is not None and _orch_summary(dm.group(1)) is not None


def _ref_workload_attempts(ev: Path, wl: str) -> list:
    """The workload's failed attempts (ref_price_<wl>.attempt<n>.log), oldest first."""
    found = []
    for q in ev.glob(f"ref_price_{wl}.attempt*.log"):
        m = re.fullmatch(rf"ref_price_{re.escape(wl)}\.attempt(\d+)\.log", q.name)
        if m:
            found.append((int(m.group(1)), q))
    return [q for _, q in sorted(found)]


def _ref_workload_attempt(ctx: LoopContext, ev: Path, wl: str, key: str, log: str,
                          label: str, verdict: str | None = None) -> None:
    """Bank one reference pricing attempt on ``wl`` that left no reading. The header
    names the key it ran on and its verdict (``none`` = the log carries no verdict
    line at all: an archive that lost it, retried even on an unchanged tree)."""
    if verdict is None:
        lines = _REF_WL_VERDICT_RE.findall(log)
        verdict = "none" if not lines else _tok_reason(" / ".join(lines))
    n = len(_ref_workload_attempts(ev, wl)) + 1
    dst = ev / f"ref_price_{wl}.attempt{n}.log"
    try:
        dst.write_text(f"[ref-price] key={key} verdict={verdict}\n" + log)
    except OSError as e:
        _log(f"{label}/ref-price[{wl}]: could not persist {dst.name}: {e}")
    _log(f"{label}/ref-price[{wl}]: no reading came back ({verdict[:160]}) — attempt "
         f"{n}/{_REF_WL_PRICE_ATTEMPTS}; no reference column for {wl} this entry"
         + ("" if verdict == "none" else ", retried once the tree changes"))
    _bank_evidence(ctx, f"ref-price {wl} attempt marker")


def _tok_reason(text: str) -> str:
    return re.sub(r"\s+", "_", text.strip())[:300] or "none"


_ORCH_DIFF_RULES = (
    "An axis-by-axis comparison of the machine-read summaries of both sides' SASS dossiers: "
    "loop roles, mbarrier object count / tiers / per-role distribution, ring shape, "
    "NamedBarrier verdict and rendezvous width, cross-role arrive→wait edges, DEPBAR "
    "in-flight depth, reduction placement, supply issue order, cache-policy bits. Reading "
    "it: a `!` row = a structural difference of the blueprint or the build from the "
    "reference (a contract the milestone declares that mismatches here ⇒ the build did not "
    "deliver it, which takes precedence over any intra-segment tuning); a `=` row = the "
    "axis is already isomorphic — do not prescribe anything more on that axis.\n"
    "**Routing rule (hard)**: a `!` row in ref vs HEAD that falls on a contract axis the "
    "milestone already declares is this diagnosis's first candidate locus — an undelivered "
    "structure takes precedence over revising acceptance wording / gate phrasing / A-B "
    "forms; changing the wording will not turn a `!` into a `=`. Only when every declared "
    "axis is `=`, or a `!` has been judged by evidence to be \"structure the reference does "
    "not explain / an environment difference\", may the locus land on the wording layer. "
    "The best→HEAD diff answers \"did this build actually touch the structure the previous "
    "patch asked for\": no change on that axis ⇒ the student did not build it (a "
    "student-side defect: add a KB recipe rather than rewording the contract again); "
    "changed but ref vs HEAD is still `!` ⇒ it was built in the wrong direction (cite the "
    "same axis in both diffs and state verbatim where they differ).\n")


def _orch_diff_text(a: dict, b: dict, names=("ref", "student")) -> str:
    from forgekernel.skills.orch_tools import diff_summaries
    lines: list = []
    diff_summaries(a, b, out=lines.append, names=names)
    return "\n".join(lines)


def _orch_diff_views(ctx: LoopContext, priced: PricedTrees) -> str | None:
    """The dossier diff block for a teacher prompt, or None when no pair of
    summaries is available.

    Up to three faces: ref vs HEAD (what the judged tree still lacks), ref vs
    best (what the best tree lacks; only when best is a different commit), and
    best -> HEAD (what this construction changed structurally). Every side
    must carry a summary line: the ref's from its banked recon-time dossier,
    the student's from the pricing jobs that just ran. Diffs are computed
    driver-side (orch_tools.diff_summaries) — no SASS page needed."""
    ref = _orch_summary(_read_ref_orch_dossier(ctx))
    head = _orch_summary(priced.head_dossier)
    best = (_orch_summary(priced.best_dossier)
            if priced.best_sha and not priced.head_is_best else None)

    def short(sha):
        return (sha or "?")[:10]

    def verdict_note(side: str, summ: dict | None) -> str:
        v = (summ or {}).get("verdict", "ORCH-OK")
        if v == "ORCH-OK":
            return ""
        return (f"**{side} page verdict `{v}`**: the page has no sync sites / no loop region "
                "— none of the barrier topology the blueprint declares was built; that by "
                "itself is the heaviest `!`, ahead of every row below.\n")

    faces: list = []
    if ref is not None and head is not None:
        faces.append((f"ref (recon time) vs HEAD {short(priced.head_sha)} (the tree judged "
                      "this time)",
                      verdict_note("HEAD", head) + _orch_diff_text(ref, head)))
    if ref is not None and best is not None:
        faces.append((f"ref (recon time) vs student best {short(priced.best_sha)}",
                      verdict_note("best", best) + _orch_diff_text(ref, best)))
    if best is not None and head is not None:
        faces.append((f"student best {short(priced.best_sha)} → HEAD "
                      f"{short(priced.head_sha)} (what structure this segment's build / "
                      "rebuild changed; `!` = changed, `=` = untouched)",
                      _orch_diff_text(best, head, names=("best", "HEAD"))))
    gaps = list(priced.missing)
    if ref is None and (head is not None or best is not None):
        gaps.append("ref: recon left no comparable dossier summary — the ref vs HEAD / "
                    "ref vs best faces are absent")
    if not faces and not gaps:
        return None
    block = ("\n## Orchestration dossier diff (static structure side by side — read it "
             "before you judge)\n" + _ORCH_DIFF_RULES)
    if gaps:
        # Named, not silent: a half-failed pricing pair used to hand the teacher
        # one surviving face under the both-succeeded labels.
        block += ("\n### Faces absent this time (a pricing job half-failed — only the "
                  "readable faces remain below; an absent tree must not be treated as read)\n"
                  + "\n".join(f"- {g}" for g in gaps) + "\n")
    for title, text in faces:
        block += f"\n### {title}\n```\n{text[:5000]}\n```\n"
    return block


def _read_ref_orch_dossier(ctx: LoopContext) -> str | None:
    ev = _evidence_dir(ctx)
    if ev is None:
        return None
    try:
        return (ev / _REF_ORCH_DOSSIER_NAME).read_text()
    except OSError:
        return None


def _verdict_evidence(ctx: LoopContext, ok: bool, waiver: "re.Pattern[str]",
                      mismatch: "re.Pattern[str] | None" = None,
                      mismatch_observed: bool = False) -> bool:
    """Delivered (a tool verdict came back from a job), explicitly waived in the
    report, or explicitly owned as delivered-but-unrecognized — silence is not
    one of the options, for any channel.

    The mismatch line only counts when the driver ITSELF recorded a verdict-
    shaped near-miss for the channel this session. Without that condition the
    line is a free bypass weaker than a waiver: a teacher could write
    "SASS-DELIVERED-MISMATCH: whatever" for a channel no job ever produced a
    verdict-shaped line for, and the human audit would read it as a checker
    defect instead of a missing delivery."""
    if ok:
        return True
    if ctx.recon_path is None or ctx.recon_dir is None:
        return True  # nothing configured to hold a finish against
    try:
        report = ctx.recon_path.read_text()
    except OSError:
        return False
    if waiver.search(report) is not None:
        return True
    return (mismatch is not None and mismatch_observed
            and mismatch.search(report) is not None)


def _struct_recon(ctx: LoopContext) -> bool:
    """The construction-table channel is mandatory on blackbox runs: no source,
    no other channel, so the teacher must decode how the reference BUILDS its
    hot phase from the binary. A source-mode run reads the source for that, so the
    channel stays optional there (waived only if the SASS really has no
    barrier-fenced construction)."""
    return ctx.oracle_mode == "blackbox"


def _missing_recon_evidence(ctx: LoopContext, sass_ok: bool, sched_ok: bool,
                            live_ok: bool, stall_ok: bool = True,
                            struct_ok: bool = True,
                            near_miss: dict | None = None) -> list:
    """Channel names whose evidence is neither delivered nor waived."""
    near = near_miss or {}
    missing = []
    if not _verdict_evidence(ctx, sass_ok, _SASS_WAIVED,
                             _DELIVERED_MISMATCH["SASS"], "SASS" in near):
        missing.append("SASS")
    if not _verdict_evidence(ctx, sched_ok, _SCHED_WAIVED,
                             _DELIVERED_MISMATCH["SCHED"], "SCHED" in near):
        missing.append("SCHED")
    if not _verdict_evidence(ctx, live_ok, _LIVE_WAIVED,
                             _DELIVERED_MISMATCH["LIVE"], "LIVE" in near):
        missing.append("LIVE")
    if ctx.stall_pricing and not _verdict_evidence(
            ctx, stall_ok or _ref_ledger_banked(ctx), _STALL_WAIVED,
            _DELIVERED_MISMATCH["STALL"], "STALL" in near):
        missing.append("STALL")
    if _struct_recon(ctx) and not _verdict_evidence(
            ctx, struct_ok, _STRUCT_WAIVED,
            _DELIVERED_MISMATCH["STRUCT"], "STRUCT" in near):
        missing.append("STRUCT")
    # The ref's overlap account: role table WITH tensor, and the phase timeline.
    # Both come off one pricing job (stall_price_run.py --pm) the driver runs on
    # the ref tree itself; the teacher only ever sees them as banked files.
    # the overlap channels have no mismatch state: banked by the driver's own
    # pricing, or waived (see _collect_verdict_near_misses)
    if ctx.stall_pricing and not _verdict_evidence(
            ctx, _ref_overlap_banked(ctx), _ROLE_WAIVED):
        missing.append("ROLE")
    if ctx.stall_pricing and not _verdict_evidence(
            ctx, _ref_timeline_banked(ctx), _PMTL_WAIVED):
        missing.append("PMTL")
    return missing


def _sass_evidence(ctx: LoopContext, sass_ok: bool,
                   near_miss: dict | None = None) -> bool:
    """SASS was either delivered (a SASS-OK verdict came back from a job) or
    explicitly waived in the report — silence is not one of the options.

    A blackbox attention-fwd run shipped with the opcode histogram missing after two
    probes failed on complementary halves (wrong parser format, wrong kernel), and
    its own report names the single round that would have closed four open gaps.
    Prose discipline in the prompt did not force that round; this check does.
    """
    return _verdict_evidence(ctx, sass_ok, _SASS_WAIVED,
                             _DELIVERED_MISMATCH["SASS"],
                             "SASS" in (near_miss or {}))


_SASS_GATE_MSG = (
    "\n\n## Finish refused: SASS evidence missing\n\n"
    "The recon report and the candidates table both exist, but this reconnaissance had no "
    "successful SASS extraction (no `SASS-OK` verdict line in any job output), and the "
    "report has no `SASS-WAIVED:` waiver line either. The opcode histogram is the only "
    "channel that can settle the (bN,stages) factorization, the register split and the "
    "consumer organization — **a silent absence is not allowed**. Pick one:\n\n"
    "1. **Deliver**: write a probe that does `import sass_tools` (the driver has seeded it "
    "into `notes/`) and runs `sass_tools.analyze(sass_text, profiled_kernel, launch_pairs)` "
    "through kernel selection → parsing → self-check; when everything passes it prints the "
    "`SASS-OK` line. End with `RECON_REQUEST: notes/<name>.py`.\n"
    "2. **Waive**: if SASS is genuinely unavailable (permission denied, both the runtime "
    "and the static channel tried), add a line `SASS-WAIVED: <the concrete reason and the "
    "channels tried>` to the recon report, then end the turn."
)

_SCHED_LIVE_GATE_MSG = (
    "\n\n## Finish refused: {names} evidence missing\n\n"
    "What is missing is the mainloop's **schedule-order transcription** (SCHED) and/or its "
    "**register live-range bill** (LIVE) — measurement showed this pair is indispensable: "
    "with every count-type footprint delivered, the milestone still could not say \"which "
    "chain is in flight across which elementwise segment\" or \"how many registers this "
    "coexistence costs\", and M1 died on exactly those two questions; while delivering the "
    "schedule order without the live ranges induced a port that measured 6% slower with "
    "10.8M spills (an A/B measurement of a schedule-order port). Pick one:\n\n"
    "1. **Deliver**: write a probe that does `import sched_live_tools` (the driver has "
    "seeded it into `notes/` together with sass_tools) and, once it has the SASS page text, "
    "calls `sched_live_tools.report(sass_text)` — it produces both deliverables in one go "
    "and self-checks them, printing the two lines `SCHED-OK` and `LIVE-OK` when everything "
    "passes. It can share one collection with the SASS extraction. "
    "End with `RECON_REQUEST: notes/<name>.py`.\n"
    "2. **Waive**: when genuinely unavailable, add `SCHED-WAIVED: <reason>` / "
    "`LIVE-WAIVED: <reason>` (whichever is missing) to the recon report, then end the "
    "turn.\n\n"
    "**The SCHED channel has three syntaxes by MMA instruction family; the tool picks one "
    "after its own census**: Hopper wgmma (HGMMA chains + `WARPGROUP.DEPBAR.LE` depth, "
    "verdict line `SCHED-OK loop=… hgmma_runs=N depbars=N`); Blackwell tcgen05 (chains "
    "grouped by TMEM accumulator operand, hand-off objects = `UTCBAR` commits and mbarrier "
    "waits, verdict line `SCHED-OK isa=tcgen05 loop=… utcmma_runs=N commits=N "
    "mbar_waits=N …`, with the page's HMMA side path transcribed too); mma.sync/HMMA "
    "(chains grouped by accumulator base, hand-off objects = `DEPBAR.LE SB`, named/cluster "
    "barriers, mbarriers, verdict line `SCHED-OK isa=mma_sync loop=… mma_runs=N "
    "barriers=N ldsm=N …`). The driver accepts all three verdict lines; the hand-off "
    "object in force before each elementwise segment is in `handoff_before_each_ew=[...]`, "
    "and the report must read it per segment as \"this segment runs after which commit / "
    "wait\". If your notes/ still has the old tool (printing only `SCHED-NOTAPPLICABLE`), "
    "that also counts as delivery, but write the architecture's own sync topology into "
    "`SCHED-WAIVED: <...>`. The LIVE half is architecture-independent; deliver it as "
    "usual.\n"
)


_STALL_GATE_MSG = (
    "\n\n## Finish refused: STALL evidence missing\n\n"
    "What is missing is the reference kernel's **per-instruction stall pricing ledger** — "
    "aggregate signatures cannot locate a gap dispersed across the quality of the "
    "mainloop's interleaving: an observed failure mode is meta picking two loci in a row, "
    "each worth only ~1pp, and 10 hours of building all rolled back; whereas one "
    "per-instruction pricing pass takes tens of seconds to pin 30% of the stalls on a "
    "few wait_group sites. Pick one:\n\n"
    "1. **Deliver**: write a probe that collects a `.ncu-rep` with PC sampling using "
    "`--set source` (or `--set full`) (it can share one collection with the SASS "
    "extraction), then does `import stall_tools` (the driver has seeded it into `notes/`) "
    "and calls `stall_tools.price_report(rep_path)` — when the self-checks (sample count, "
    "main-kernel calibration, ledger coverage ≥90%) all pass it prints the `STALL-OK` "
    "line; right after, **in the same probe, also call "
    "`stall_tools.segment_ledger(rep_path)`** — the segment ledger (the PC-keyed segment "
    "table) is printed only by it, and without that call the reference side of "
    "diagnose's same-position segment comparison is always absent. "
    "End with `RECON_REQUEST: notes/<name>.py`.\n"
    "2. **Waive**: when genuinely unavailable (e.g. PC sampling is empty on this image and "
    "that was verified by measurement), add `STALL-WAIVED: <the concrete reason and the "
    "channels tried>` to the recon report, then end the turn.\n"
)


def _near_miss_notes(missing: list, near_miss: dict | None) -> list:
    """One reconciliation note per missing channel whose logs carried a
    verdict-shaped line the full-field pattern rejected — both sides of the
    disagreement, verbatim, plus the third legal closing state."""
    notes = []
    for ch in missing:
        if near_miss and ch in near_miss:
            notes.append(
                f"⚠ {ch} reconciliation: the job log has a line that starts with "
                f"`{ch}-OK` but fails the driver's full-field check:\n"
                f"    `{near_miss[ch]}`\n"
                f"    If this is the tool's real output, the gap is in **reconciling the "
                f"acceptance check with the print format**, not in the delivery — do not "
                f"write a waiver line; instead add a line to the recon report\n"
                f"    `{ch}-DELIVERED-MISMATCH: <paste the line above verbatim>`\n"
                f"    It is a legal finishing state parallel to a waiver; human review "
                f"treats it as a defect of the acceptance check.")
    return notes


_STRUCT_GATE_MSG = (
    "\n\n## Finish refused: the reference's per-segment construction table is missing\n\n"
    "What is missing is the **per-segment construction table of the reference's hot "
    "region** — how many warps are working, how large a block each handles, what internal "
    "precision, how adjacent segments hand off. The histogram / schedule order / stall "
    "ledger cannot say \"how this segment is built\": an observed failure mode is a gap "
    "sitting entirely inside one warp-cooperative construction (e.g. a triangular "
    "inverse), the recon **inferring** the block shape from the instruction order, the "
    "milestone and the certified KB copying the inference, and the student building the "
    "whole run to the wrong shape; whereas the same SASS + per-PC execution counts "
    "**mechanically read out** the real block granule, number of doubling levels, internal "
    "precision and reduction method. Pick one:\n\n"
    "1. **Deliver**: write a probe (it can use a `.ncu-rep` collected with "
    "`--set source`, sharing one collection with the SASS extraction) that passes the "
    "source page as CSV to `struct_tools.source_csv_to_sass_and_counts()` to get the SASS "
    "text + per-instruction execution counts, then does `import struct_tools` and calls "
    "`struct_tools.report(sass_text, counts, kernel=..., ncta=<grid.x>)` — when the "
    "self-checks (named-barrier segmentation, determinate families, integral warp "
    "reductions) all pass it prints the `STRUCT-OK` line; the report must include the "
    "construction table in full and spell out **which structural decision each segment "
    "of the hot region is building**. End with `RECON_REQUEST: notes/<name>.py`.\n"
    "2. **Waive**: the only legal condition is that the SASS really has no construction "
    "segment fenced by named barriers (no `BAR.SYNC <id>, <threads>`) — add "
    "`STRUCT-WAIVED: <the concrete reason>` to the recon report, then end the turn. "
    "Without that line the finish is bounced."
)


def _evidence_gate_msg(missing: list, near_miss: dict | None = None,
                       ctx: "LoopContext | None" = None) -> str:
    parts = []
    if "SASS" in missing:
        parts.append(_SASS_GATE_MSG)
    rest = [m for m in missing if m not in ("SASS", "STALL", "ROLE", "PMTL", "STRUCT")]
    if rest:
        parts.append(_SCHED_LIVE_GATE_MSG.format(names="/".join(rest)))
    if "STALL" in missing:
        parts.append(_STALL_GATE_MSG)
    if "STRUCT" in missing:
        parts.append(_STRUCT_GATE_MSG)
    overlap = [m for m in missing if m in ("ROLE", "PMTL")]
    if overlap:
        note, option1 = (_ref_price_status(ctx, overlap) if ctx is not None
                         else ("(no ctx)", "waive"))
        parts.append(_OVERLAP_GATE_MSG.format(names="/".join(overlap),
                                              note=note, option1=option1))
    notes = _near_miss_notes(missing, near_miss)
    if notes:
        parts.append("\n\n" + "\n\n".join(notes))
    return "".join(parts)


_OVERLAP_GATE_MSG = (
    "\n\n## Finish refused: the reference overlap ledger is missing ({names})\n\n"
    "The reference's role exposure ledger must carry the tensor duty cycle "
    "(`ROLE-OK … tensor=<pct>`), and the phase timeline must be signed off with "
    "`PMTL-OK` — these two are the **only** channels that show \"how the reference hides "
    "its supply behind the MMA\"; static SASS cannot read it (an observed failure mode: "
    "with every exposed value on the reference side null, several consecutive blackbox "
    "runs guessed wrong at the same spot). The driver produces them by running the frozen "
    "recipe `python3 notes/stall_price_run.py --pm` on the recon tree (the PM sampling "
    "interval is chosen automatically from the reference's duration, 8192 / 2048 cycles, "
    "and can be set with loop.toml `[loop] ref_price_pm_interval`); you do not send a "
    "probe for them.\n\n**Status**: {note}.\n\nPick one:\n\n"
    "1. **Deliver**: {option1}.\n"
    "2. **Waive**: add `{names}-WAIVED: <the concrete reason and the channels tried>` to "
    "the recon report (one line per channel), then end the turn. After a waiver the "
    "driver no longer runs pricing for that channel; the distillation step is told "
    "explicitly that the reference overlap ledger does not exist, and the milestone's "
    "[SYNC:]/[GATHER:] rulings must be written as candidates, not as settled."
)


# Every module the driver seeds into a tree's notes/ — the reconnaissance tree
# (_seed_sass_tools) and the student / ref-pricing trees (_seed_pricing_toolkits).
# On a closed-source run scripts/blackbox-guard.sh scans notes/ before EVERY job,
# so a forbidden token in any one of these refuses every job of that tree, not
# just the probe that imports it: timeline_probe.py's stdlib byte-compile import
# did exactly that to two Blackwell reference trees, cutting a
# reconnaissance short. tests/test_blackbox_guard.py runs the guard over
# all of them.
RECON_SEEDED_MODULES = (
    "forgekernel.skills.sass_tools", "forgekernel.skills.sched_live_tools",
    "forgekernel.skills.stall_tools", "forgekernel.skills.sass_census",
    "forgekernel.skills.pm_timeline_tools", "forgekernel.skills.pm_timeline_run",
    "forgekernel.skills.orch_tools", "forgekernel.skills.role_exposure_tools",
    "forgekernel.skills.stall_price_run", "forgekernel.skills.struct_tools",
    "forgekernel.skills.launch_tools",
)
PRICING_SEEDED_MODULES = (
    "forgekernel.skills.stall_tools", "forgekernel.skills.stall_price_run",
    "forgekernel.skills.sass_tools", "forgekernel.skills.sched_live_tools",
    "forgekernel.skills.pm_timeline_tools", "forgekernel.skills.pm_timeline_run",
    "forgekernel.skills.orch_tools", "forgekernel.skills.role_exposure_tools",
    "forgekernel.skills.orch_census_run", "forgekernel.skills.lever_pricing_run",
    "forgekernel.skills.sass_census", "forgekernel.skills.timeline_probe",
    "forgekernel.skills.job_env_check",
)


def _seed_sass_tools(ctx: LoopContext, rbe) -> None:
    """Copy the vendored SASS toolkit into the reconnaissance tree and commit it.

    Probe delivery is ``git archive`` of the committed tree, so the toolkit must be
    a committed file *inside* that tree; and it must be the driver that puts it
    there, because teacher-written one-shot parsers in two probes each
    got a different half wrong. ``python3 notes/<probe>.py`` puts ``notes/`` on
    ``sys.path``, so probes reach it with a bare ``import sass_tools``.
    """
    if ctx.recon_dir is None:
        return
    import importlib
    changed = []
    # role_exposure_tools / stall_price_run ride along: the ref's overlap
    # account is priced in this tree with the frozen recipe, and recon_prompt
    # §2f/§2g tell the teacher both are seeded (a teacher once had to
    # copy them in by hand). launch_tools: §1's cluster shape off the NCU
    # launch metrics, never off the grid shape.
    for vendor in map(importlib.import_module, RECON_SEEDED_MODULES):
        text = Path(vendor.__file__).read_text()
        name = Path(vendor.__file__).name
        dst = ctx.recon_dir / "notes" / name
        if dst.exists() and dst.read_text() == text:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        changed.append(name)
    # The closed-source run's in-job tripwire rides the same way, for the same
    # reason: the ref tree's gpu_entry.sh is what runs on the node, and the node
    # only ever sees `git archive` of that tree — the harness repo's scripts/ is
    # not there. Hand-copying the guard's body into each gpu_entry.sh let the
    # copies inherit each other's bugs. Seeded, gpu_entry.sh
    # sources `$W/notes/blackbox-guard.sh` and carries no logic of its own.
    if ctx.oracle_mode == "blackbox":
        src = _blackbox_guard_source()
        if src is not None:
            text = src.read_text()
            dst = ctx.recon_dir / "notes" / BLACKBOX_GUARD_NAME
            if not (dst.exists() and dst.read_text() == text):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(text)
                changed.append(BLACKBOX_GUARD_NAME)
        else:
            _log(f"meta/recon: {BLACKBOX_GUARD_NAME} not found beside this checkout "
                 f"— the ref tree's gpu_entry.sh must carry its own copy")
    if changed:
        rbe.commit(f"recon: vendor {', '.join(changed)} (driver-seeded)")


BLACKBOX_GUARD_NAME = "blackbox-guard.sh"


def _blackbox_guard_source() -> "Path | None":
    """scripts/blackbox-guard.sh of the checkout this driver runs from. The harness
    runs from its source tree (cli._repo_root is the same resolution); under a
    wheel install there is no scripts/ and the seeding logs and skips."""
    p = Path(__file__).resolve().parents[3] / "scripts" / BLACKBOX_GUARD_NAME
    return p if p.is_file() else None


def _save_recon_log(ctx: LoopContext, script_path: str, round_no: int,
                    log: str) -> None:
    """Persist each probe's fed-back output next to the probe that produced it.

    Findings once had to be re-fished from the job logs at write-up time
    because no job's output had been written to any file.
    Summarising into the report stays the teacher's job; keeping the raw evidence
    is now the driver's.
    """
    if ctx.recon_dir is None:
        return
    dst = ctx.recon_dir / "notes" / f"{Path(script_path).stem}.run{round_no}.log"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(log)
    except OSError as e:
        _log(f"meta/recon: could not persist job log to {dst}: {e}")


_REF_STALL_LEDGER_NAME = "ref_stall_ledger.txt"
_REF_SEGMENT_LEDGER_NAME = "ref_segment_ledger.txt"
_REF_STRUCT_TABLE_NAME = "ref_struct_table.txt"
_STRUCT_BEGIN = "=== STRUCT TABLE BEGIN ==="
_STRUCT_END = "=== STRUCT TABLE END ==="


def _save_ref_struct_table(ctx: LoopContext, log: str) -> None:
    """Keep the ref's construction table where the distill step can read it.

    The STRUCT channel's table is the only place the reference's per-phase
    construction (warps / tile granule / internal precision / hand-off) is
    written down on blackbox runs. Banked verbatim off the job log like the
    stall ledger, so the milestone can cite the ref's build rather than the
    teacher's paraphrase of it.
    """
    if ctx.recon_path is None:
        return
    m = re.search(r"=== STRUCT TABLE BEGIN ===\n(.*?)\n=== STRUCT TABLE END ===",
                  log, re.S)
    if m is None:
        return
    try:
        (ctx.recon_path.parent / _REF_STRUCT_TABLE_NAME).write_text(m.group(1))
    except OSError as e:
        _log(f"meta/recon: could not persist {_REF_STRUCT_TABLE_NAME} to {e}")


def _read_ref_struct_table(ctx: LoopContext) -> str | None:
    if ctx.recon_path is None:
        return None
    try:
        return (ctx.recon_path.parent / _REF_STRUCT_TABLE_NAME).read_text()
    except OSError:
        return None


EVIDENCE_FILE_NAMES = (
    _REF_STALL_LEDGER_NAME, _REF_SEGMENT_LEDGER_NAME, _REF_ORCH_DOSSIER_NAME,
    _REF_ROLE_EXPOSURE_NAME, _REF_PM_TIMELINE_NAME, _STUDENT_ROLE_EXPOSURE_NAME,
    _STUDENT_ORCH_DOSSIER_NAME, _STUDENT_BEST_ORCH_DOSSIER_NAME,
    # the attempt markers: inherited, they would silence a fresh run's ref pricing
    "ref_price_job.log", "ref_price_job.2.log", "ref_price_job.3.log")


def _evidence_dir(ctx: LoopContext, create: bool = False) -> Path | None:
    """The directory pricing evidence is banked in and read back from: the
    configured one, else (older callers / tests) the recon report's directory,
    else None — a ctx with neither has no channel for it."""
    ev = ctx.evidence_dir or (ctx.recon_path.parent if ctx.recon_path else None)
    if ev is not None and create:
        try:
            ev.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _log(f"meta: could not create evidence dir {ev}: {e}")
            return None
    return ev


def _bank_evidence(ctx: LoopContext, why: str) -> str:
    """Commit what the loop just wrote into the reconnaissance tree.

    Everything the loop banks there — ref ledgers, dossiers, timelines, lever
    prices — is written by the driver and committed by nobody. Two things break
    when it stays uncommitted: a GPU job ships a COMMIT, so the next job does not
    carry the evidence the teacher just paid for; and every run's preflight
    refuses to start on a dirty reference tree, so the loop wedges its own
    restart and needs a human to commit before it can resume (mla_paged_decode
    and attn_fwd runs both lost hours to exactly this).

    Bookkeeping never ends a run: any failure is logged and swallowed. Returns
    the reason it did nothing ("" when it committed).
    """
    rbe = getattr(ctx, "recon_backend", None)
    if rbe is None or _evidence_dir(ctx) is None:
        return "no reconnaissance tree"
    try:
        # `dirty` is optional: a backend without it (test fakes, older backends) still
        # banks, it just cannot skip a no-op commit first.
        probe = getattr(rbe, "dirty", None)
        if callable(probe) and not probe():
            return "nothing to bank"
        rbe.commit(f"recon: bank {why}")
    except Exception as e:     # noqa: BLE001 — bookkeeping must not end a run
        _log(f"meta/recon: could not bank evidence ({why}): {e}")
        return str(e)
    _log(f"meta/recon: banked evidence in the reconnaissance tree ({why})")
    return ""


def _save_ref_stall_ledger(ctx: LoopContext, log: str) -> None:
    """Keep the ref's priced ledger where the diagnose step can re-read it.

    The recon logs already persist next to their probes, but the diagnose step
    needs THIS table specifically and must not go fishing through probe logs for
    it — the same reason the recon report exists at a fixed path. The segment
    account is banked the same way when the log carries one: it is the ref side
    of the diagnose step's aligned comparison table.
    """
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    # The two tables have separate self-checks and are banked separately: the
    # row ledger keys on source lines and needs STALL-OK (a library kernel
    # without LINEINFO fails its coverage floor — deep_gemm ref: 85.6%
    # < 90%), the segment account keys on PCs and needs only its own JOIN-OK.
    # Tying both to STALL-OK threw away a clean ref segment account.
    for pattern, name, ok in (
            (_STALL_LEDGER_RE, _REF_STALL_LEDGER_NAME, _STALL_OK.search(log) is not None),
            (_SEGMENT_LEDGER_RE, _REF_SEGMENT_LEDGER_NAME, True)):
        m = pattern.search(log)
        if m is None or not ok or (name == _REF_SEGMENT_LEDGER_NAME
                                   and "JOIN-OK" not in m.group(1)):
            continue
        dst = ev / name
        try:
            dst.write_text(m.group(1))
        except OSError as e:
            _log(f"meta/recon: could not persist {name} to {dst}: {e}")


def _save_ref_orch_dossier(ctx: LoopContext, log: str) -> None:
    """Bank the ref's orchestration dossier next to the stall ledger.

    Same contract as _save_ref_stall_ledger: the distill step's [SYNC:]/
    [GATHER:] adjudications and a later student-vs-ref dossier diff need THIS
    table at a fixed path, not a fishing trip through probe logs."""
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    pick = _pick_ref_dossier(log)
    if pick is None:
        return
    text, note = pick
    if note:
        _log(f"meta/recon: ref dossier {note}")
    dst = ev / _REF_ORCH_DOSSIER_NAME
    # A later probe log carrying only bystander kernels' blocks must not
    # replace a banked bigger kernel: a gdn_prefill_bwd run banked the
    # main kernel (9112 instrs) off one probe, silently, and the next probe's
    # aux-kernel blocks (prepare_h, 6232 instrs) overwrote it — the blueprint
    # for the whole run would have been a 20% side kernel. Same preference as
    # _pick_ref_dossier within one log: an OK verdict, then the largest parsed
    # kernel, wins across logs too; an equal or larger kernel still supersedes.
    kept = _banked_dossier_header(dst)
    new = _banked_dossier_header_text(text)
    if (kept is not None and new is not None and kept[2] == "OK"
            and kept[1] > new[1]):
        _log(f"meta/recon: ref dossier kept banked kernel={kept[0]} "
             f"instrs={kept[1]} over the smaller kernel={new[0]} "
             f"instrs={new[1]} from this log")
        return
    try:
        dst.write_text(text)
    except OSError as e:
        _log(f"meta/recon: could not persist {_REF_ORCH_DOSSIER_NAME} "
             f"to {dst}: {e}")


_BANKED_HEADER_RE = re.compile(
    r"^# banked: block \d+/\d+ kernel=(.+?) instrs=(\d+) verdict=(\S+)")


def _banked_dossier_header_text(text: str) -> tuple[str, int, str] | None:
    """(kernel, instrs, verdict) from a banked dossier's ``# banked:`` line.

    Read from the FIRST LINE only. The kernel name in that line is ncu's fully
    demangled signature, and a CUTLASS or Triton kernel's runs to hundreds of
    characters — a fixed character window would push ``instrs=``/``verdict=`` out of
    range, both sides would parse as None, and the guard against overwriting a bigger
    kernel's dossier would silently fall back to overwriting unconditionally. The
    writer always emits this line first, so the first line is the whole question.
    """
    m = _BANKED_HEADER_RE.match(text.split("\n", 1)[0])
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def _banked_dossier_header(path: Path) -> tuple[str, int, str] | None:
    try:
        return _banked_dossier_header_text(path.read_text())
    except OSError:
        return None


_DOSSIER_INSTRS_RE = re.compile(r"^\[parse\] (\d+) instructions", re.M)
_DOMINANT_LAUNCH_RE = re.compile(
    r"^\[stall_price\] dominant launch index=(\S+) name=(.+)$", re.M)


def _pick_ref_dossier(log: str) -> "tuple[str, str] | None":
    """The dossier block a reference log should be banked from, with its
    kernel identity stamped on top.

    A log can carry several blocks — a chain replay prices several kernels,
    a retried collection prints twice — and the FIRST one is not "the" ref:
    banking it made a bystander kernel's contract axes the blueprint for the
    whole run (every later `!` a phantom). Preference: ORCH-OK verdict, then
    the largest parsed kernel; ties keep the later block (a retry supersedes).
    The banked text is prefixed with one ``# banked:`` line naming the block,
    its instruction count, its verdict and the dominant-launch kernel name the
    pricing run printed before it (or the STALL-OK kernel), so a later reader
    can check the blueprint is the kernel under judgment. Returns
    (text, note) — note is "" when the log carried exactly one block."""
    blocks = list(_ORCH_DOSSIER_RE.finditer(log))
    if not blocks:
        return None
    stall = _STALL_OK.search(log)
    stall_kernel = None
    if stall is not None:
        km = re.match(r"STALL-OK kernel=(.+?) samples=", stall.group(0))
        stall_kernel = km.group(1) if km else None

    def describe(m):
        body = m.group(1)
        im = _DOSSIER_INSTRS_RE.search(body)
        instrs = int(im.group(1)) if im else 0
        summ = _orch_summary(body) or {}
        verdict = summ.get("verdict") or ("ORCH-OK" if _ORCH_OK.search(
            log[m.end():m.end() + 400]) else "?")
        name = None
        for dm in _DOMINANT_LAUNCH_RE.finditer(log, 0, m.start()):
            name = dm.group(2).strip()      # the nearest one before the block
        return instrs, verdict, name or stall_kernel or "?"

    described = [(m, *describe(m)) for m in blocks]
    best = max(range(len(described)),
               key=lambda i: (described[i][2] == "ORCH-OK", described[i][1], i))
    m, instrs, verdict, name = described[best]
    # The verdict is stamped without its ORCH- prefix: the banked text is the
    # dossier BODY, and readers assert the trailing verdict line stays outside.
    header = (f"# banked: block {best + 1}/{len(blocks)} kernel={name} "
              f"instrs={instrs} verdict={verdict.replace('ORCH-', '')}\n")
    note = ""
    if len(blocks) > 1:
        others = ", ".join(f"block {i + 1}: {d[3]} instrs={d[1]} {d[2]}"
                           for i, d in enumerate(described) if i != best)
        note = (f"banked block {best + 1}/{len(blocks)} (kernel={name}, "
                f"instrs={instrs}, {verdict}); not banked: {others}")
    return header + m.group(1), note


def _save_ref_role_exposure(ctx: LoopContext, log: str) -> None:
    """Bank the ref's role exposure ledger beside its dossier and segment
    account — the ref side of the diagnose step's per-role comparison.

    A partial table (no tensor term) is banked only where no full one is: a
    later --set-source probe must not downgrade the account a pricing job
    delivered."""
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    pick = _pick_role_block(log)
    if pick is None:
        return
    block, full = pick
    if not full and _ref_overlap_banked(ctx):
        return
    dst = ev / _REF_ROLE_EXPOSURE_NAME
    try:
        dst.write_text(block)
    except OSError as e:
        _log(f"meta/recon: could not persist {_REF_ROLE_EXPOSURE_NAME} "
             f"to {dst}: {e}")


def _pick_role_block(log: str) -> "tuple[str, bool] | None":
    """(body, carries-tensor) of the LAST role block in a log, judged by the
    verdict line that follows that block — a recon probe may print several
    (a rig round-trip, then the real table) and a whole-log ROLE-OK search
    would let the first block ride on a later block's verdict."""
    blocks = list(_ROLE_BLOCK_RE.finditer(log))
    if not blocks:
        return None
    m = blocks[-1]
    tail = log[m.end():m.end() + 400]
    full = _ROLE_OK.search(tail) is not None
    if not full and _ROLE_ANY.search(tail) is None:
        # no verdict after the block at all (clipped log): trust the core
        from forgekernel.skills.role_exposure_tools import parse_core
        core = parse_core(m.group(1))
        full = bool(core) and core.get("tensor_active") is not None
    return m.group(1), full


def _save_ref_pm_timeline(ctx: LoopContext, log: str) -> None:
    """Bank the ref's PM phase timeline (only on PMTL-OK — a FAIL block is a
    diagnostic, not a timeline)."""
    ev = _evidence_dir(ctx, create=True)
    if ev is None or _PMTL_OK.search(log) is None:
        return
    m = _PM_TIMELINE_RE.search(log)
    if m is None:
        return
    dst = ev / _REF_PM_TIMELINE_NAME
    try:
        dst.write_text(m.group(1))
    except OSError as e:
        _log(f"meta/recon: could not persist {_REF_PM_TIMELINE_NAME} to {dst}: {e}")


def _read_ref_pm_timeline(ctx: LoopContext) -> str | None:
    ev = _evidence_dir(ctx)
    if ev is None:
        return None
    try:
        return (ev / _REF_PM_TIMELINE_NAME).read_text()
    except OSError:
        return None


def _ref_overlap_banked(ctx: LoopContext) -> bool:
    """True iff the banked ref role table carries the tensor term — i.e. the
    ref's overlap account exists, not just its stall shares."""
    from forgekernel.skills.role_exposure_tools import parse_core
    core = parse_core(_read_ref_role_exposure(ctx))
    return bool(core) and core.get("tensor_active") is not None


def _ref_timeline_banked(ctx: LoopContext) -> bool:
    return _read_ref_pm_timeline(ctx) is not None


def _ref_ledger_banked(ctx: LoopContext) -> bool:
    """The driver's own pricing job delivers STALL-OK too; the recon loop's
    per-log flag never sees that job, so the gate reads the banked file."""
    return _read_ref_stall_ledger(ctx) is not None


def _read_ref_role_exposure(ctx: LoopContext) -> str | None:
    ev = _evidence_dir(ctx)
    if ev is None:
        return None
    try:
        return (ev / _REF_ROLE_EXPOSURE_NAME).read_text()
    except OSError:
        return None


_ROLE_RULES = (
    "A join of three tables: the dossier's role partition cut along back-edge regions × "
    "the PC-sampling stall ledger × the tensor-pipe duty cycle of the PM timeline. share = "
    "that role's fraction of all stall samples (measured); exposed = share × "
    "(1 − tensor_active), an **estimate** that apportions the time not hidden by the "
    "tensor pipe to roles by stall share (a role waiting while the tensor pipe is busy is "
    "hidden; the sampler cannot tell them apart); Δexposed ns = student − ref. "
    "Reading it: ① first see how far tensor_active differs between the sides — that is "
    "the total of all non-tensor exposure; ② then see which role has the largest "
    "Δexposed — that is the **owner** of the exposure, and `deficient_locus` should land on "
    "that role's decisions (copy-producer ⇒ supply organization / issue order; "
    "mma-consumer ⇒ exchange surface, bookkeeping, wait→use distance; straight ⇒ per-tile "
    "scheduling / prologue / epilogue); ③ the difference in the same role's dominant stall "
    "class (student wait/barrier-dominated vs ref dispatch/selection-dominated) tells a G5 "
    "from an R1 shape. With tensor_source=ncu-aggregate, exposed is apportioned over the "
    "whole kernel rather than the mainloop — reference only; when the two sides' sources "
    "differ, align the measurement basis before comparing.\n")


def _role_exposure_view(ctx: LoopContext, priced: "PricedTrees") -> str | None:
    """The per-role exposure comparison block for a teacher prompt, or None
    when either side lacks a ROLE-JSON core."""
    from forgekernel.skills.role_exposure_tools import align_roles, parse_core
    ref = parse_core(_read_ref_role_exposure(ctx))
    stu = parse_core(priced.roles)
    if ref is None and stu is None:
        return None
    block = ("\n## Role exposure ledger (dossier × stall ledger × tensor timeline — read it "
             "before you judge)\n" + _ROLE_RULES)
    partial = [n for n, c in (("ref", ref), ("student", stu))
               if c is not None and c.get("tensor_active") is None]
    if partial:
        block += (f"\n**Note: the {', '.join(partial)} side's ledger has no tensor duty "
                  f"cycle (ROLE-PARTIAL) — the exposed column is empty; only stall shares can "
                  f"be compared, not exposed ns; any conclusion of the form \"ref exposes X ns "
                  f"/ student exposes Y ns\" does not hold on that side.**\n")
    if ref is not None and stu is not None:
        block += ("\n### ref (recon time) vs student "
                  f"{(priced.best_sha or priced.head_sha or '?')[:10]}\n"
                  f"```\n{align_roles(ref, stu)}\n```\n")
    else:
        side = "student" if stu is not None else "ref"
        core = stu if stu is not None else ref
        block += (f"\n### Only the {side} side has a ledger (the other side's job printed no "
                  f"ROLE-OK/ROLE-PARTIAL)\n"
                  f"```\n{align_roles(core, core, names=(side, side))}\n```\n")
    if priced.roles:
        block += f"\n### student ledger, verbatim\n```\n{priced.roles[:3000]}\n```\n"
    return block


def _read_ref_stall_ledger(ctx: LoopContext) -> str | None:
    ev = _evidence_dir(ctx)
    if ev is None:
        return None
    try:
        return (ev / _REF_STALL_LEDGER_NAME).read_text()
    except OSError:
        return None


def _read_ref_segment_ledger(ctx: LoopContext) -> str | None:
    ev = _evidence_dir(ctx)
    if ev is None:
        return None
    try:
        return (ev / _REF_SEGMENT_LEDGER_NAME).read_text()
    except OSError:
        return None


_SEG_LINE_RE = re.compile(
    r"^SEG idx=\d+ kind=(\S+) depth=(\S+)(?: imm=(\S+))? pct=([\d.]+) stalls=(\S*)",
    re.M)
# Skeleton anchors, canonical order. Kinds that wait at the loop top before the
# first HGMMA run collapse into one anchor; EW segments aggregate by the DEPBAR
# depth in force; DEPBARs aggregate by their own immediate.
_SEG_WAIT_KINDS = ("SYNCS", "BAR", "SLEEP", "FENCE", "WG")
_SEG_ANCHOR_ORDER = ("loop-top wait", "first HG run", "DEPBAR 0x1", "EW@0x1",
                     "DEPBAR 0x0", "EW@0x0", "EW@none", "HG rest", "BRA", "EXIT", "other")


def _anchor_shares(seg_text: str) -> dict:
    """SEG machine lines -> {anchor: (pct_sum, dominant stalls of biggest row)}."""
    out: dict = {}

    def add(name: str, pct: float, stalls: str) -> None:
        cur = out.setdefault(name, [0.0, "", -1.0])
        cur[0] += pct
        if pct > cur[2]:
            cur[1], cur[2] = stalls, pct

    seen_hg = False
    for m in _SEG_LINE_RE.finditer(seg_text):
        kind, depth, imm, pct, stalls = (m.group(1), m.group(2), m.group(3) or "",
                                         float(m.group(4)), m.group(5))
        if kind == "HG":
            add("first HG run" if not seen_hg else "HG rest", pct, stalls)
            seen_hg = True
        elif not seen_hg and kind in _SEG_WAIT_KINDS:
            add("loop-top wait", pct, stalls)
        elif kind == "DEPBAR":
            add(f"DEPBAR {imm or '?'}", pct, stalls)
        elif kind == "EW":
            add(f"EW@{depth}", pct, stalls)
        elif kind in ("BRA", "EXIT"):
            add(kind, pct, stalls)
        else:
            add("other", pct, stalls)
    return {k: (v[0], v[1]) for k, v in out.items()}


def align_segment_table(ref_seg: str, student_seg: str) -> str | None:
    """Same-position comparison of the two segment accounts, anchor by anchor.

    The diagnose hard rule holds the locus against the ledger's big rows — which
    presupposes the two sides can be held NEXT to each other. Two raw ledgers
    leave the alignment to the reader, and the reading that matters (same
    segment, share X vs share Y, wait-dominated vs dispatch-dominated) only
    exists after alignment. Returns None when either side has no SEG lines.
    """
    ref = _anchor_shares(ref_seg)
    stu = _anchor_shares(student_seg)
    if not ref or not stu:
        return None
    keys = [k for k in _SEG_ANCHOR_ORDER if k in ref or k in stu]
    keys += [k for k in sorted(set(ref) | set(stu)) if k not in keys]
    lines = ["anchor          | ref %body | student %body | ref dominant stall | student dominant stall"]
    for k in keys:
        r, s = ref.get(k), stu.get(k)
        lines.append(f"{k:<15} | {(f'{r[0]:.1f}' if r else '—'):>9} | "
                     f"{(f'{s[0]:.1f}' if s else '—'):>12} | "
                     f"{(r[1] if r else '—'):<16} | {(s[1] if s else '—')}")
    return "\n".join(lines)


def _seed_ws_stall_tools(ctx: LoopContext) -> None:
    """Vendor the pricing toolkit + frozen collection runner into the student ws.

    Committed once at loop start so EVERY subsequent commit — including whichever
    becomes the best — carries the recipe: pricing runs against the best sha via
    `git archive`, which ships committed files only. Same driver-not-agent rule
    as the recon seeding: teacher-written parsers each got a half wrong.
    """
    if not ctx.stall_pricing:
        return
    _seed_pricing_toolkits(ctx.workspace, ctx.backend)


_JOB_ENV_CHECK_NAME = "job_env_check.py"
_JOB_ENV_CHECK_CMD = f"cmd:python3 notes/{_JOB_ENV_CHECK_NAME}"
JOB_ENV_PREFLIGHT_POLICIES = ("refuse", "warn", "off")


def job_env_preflight(ctx: LoopContext, policy: str = "refuse") -> str | None:
    """Before any iteration: can the harness's own channels run on this job image?

    Two short jobs on the workspace as it stands: the seeded job_env_check.py (can the
    job's bare ``python3`` — what the driver's tools re-spawn — import torch?) and the
    tree's NCU job exactly as every evaluated iteration will submit it. Only a
    job-ENVIRONMENT failure counts (loop/job_env.py): a scaffold kernel that raises is
    the tree's business, a launch target that cannot import its profiler binding is
    not. A fused-norm-rope run would have failed both checks before its first
    iteration, and nothing said so for 2.5 days — its reference tree had a gate like
    this (no NCU signature, no distillation), its student tree had none.

    Returns the failures as one line, or None (all clear, not judged, or off).
    """
    if policy == "off":
        return None
    be = ctx.backend
    branch = f"{ctx.branch_prefix}-jobenv"
    problems, seen = [], []
    try:
        _seed_ws_stall_tools(ctx)   # idempotent; the check script rides the pricing toolkit
        checks = []
        if (Path(ctx.workspace) / "notes" / _JOB_ENV_CHECK_NAME).is_file():
            # a CUDA-mode card also needs the toolchain its kernel is built with;
            # a TileLang card needs the TileLang install and its own toolchain
            if _is_cuda(ctx):
                tools_cmd = f"{_JOB_ENV_CHECK_CMD} --cuda"
            elif _is_tilelang(ctx):
                tools_cmd = f"{_JOB_ENV_CHECK_CMD} --tilelang"
            else:
                tools_cmd = _JOB_ENV_CHECK_CMD
            checks.append(("tools", tools_cmd))
        checks.append(("ncu", _ncu_mode(ctx)))
        be.push_branch(branch)
        for channel, mode in checks:
            out = be.submit_gpu(branch, mode) or ""
            why = env_failure(out)
            if why is None and channel == "tools" and "JOB-ENV-FAIL" in out:
                why = next(ln.strip() for ln in out.splitlines() if "JOB-ENV-FAIL" in ln)
            if why is not None:
                problems.append(f"{channel}: {why}")
                _note_env_failure(ctx, channel, why)
            elif channel == "tools" and (m := re.search(r"^JOB-ENV-OK .*$", out, re.M)):
                seen.append(m.group(0))
    except BackendError as e:
        _log(f"job-env preflight: job failed ({e}) — the environment was not judged")
        return None
    if problems:
        _log(f"job-env preflight: FAIL — {'; '.join(problems)} "
             f"(policy={policy}{': refusing to start' if policy == 'refuse' else ''})")
        return "; ".join(problems)
    _log("job-env preflight: OK" + (f" ({seen[0]})" if seen else "")
         + "; the NCU launch target starts on this image")
    return None


def _seed_pricing_toolkits(root: Path, backend) -> None:
    """Vendor the pricing toolkit into ``root/notes`` and commit it there.

    sass/sched tools ride along: the segment ledger joins stall samples onto
    the parsed schedule inside the pricing job, and vendored-flat imports
    (notes/ on sys.path) are how every probe already consumes them.
    orch_tools rides along for the student side of the dossier diff: the
    blueprint's sync/supply adjudications came from the ref's dossier, and
    whether the built SASS actually carries them is checkable in-job,
    statically, before any perf number. The same seeding serves the reference
    tree a source-mode run prices (_ensure_ref_pricing).
    """
    import importlib
    changed = []
    for vendor in map(importlib.import_module, PRICING_SEEDED_MODULES):
        text = Path(vendor.__file__).read_text()
        name = Path(vendor.__file__).name
        dst = root / "notes" / name
        if dst.exists() and dst.read_text() == text:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        changed.append(name)
    if changed:
        backend.commit(f"notes: vendor {', '.join(changed)} (driver-seeded "
                       f"recon toolkits)")


def _ref_price_markers(ev: Path) -> list:
    return [ev / n for n in _REF_PRICE_MARKER_NAMES if (ev / n).is_file()]


def _ref_price_entry(ctx: LoopContext) -> "tuple[Path | None, str]":
    """(the launch-target file the pricing recipe will run, how it was named)
    — ref_price_app's script when configured, else eval/ncu_run.py."""
    d = ctx.ref_price_dir
    if ctx.ref_price_app:
        script = next((a for a in ctx.ref_price_app if a.endswith(".py")), None)
        return (d / script if script else None), f"ref_price_app = {ctx.ref_price_app}"
    return d / "eval" / "ncu_run.py", "eval/ncu_run.py"


def _ref_pricing_key(ctx: LoopContext) -> str:
    """What a retry must see changed: the tree's head, the launch target's
    text, the app spelling. The failures a pricing job can come back with
    (no LINEINFO, PM sections crashing, the sample floor) are deterministic
    on one tree — a retry on the same key reproduces the first job."""
    import hashlib
    h = hashlib.sha1()
    # Not the head sha: the snapshot commit before every job is --allow-empty
    # and moves the head whether or not anything changed. The launch target
    # and its eval/ siblings are what a teacher edits to fix a pricing job.
    entry, _ = _ref_price_entry(ctx)
    files = []
    if entry is not None:
        files.append(entry)
    ev_dir = ctx.ref_price_dir / "eval"
    if ev_dir.is_dir():
        files.extend(sorted(q for q in ev_dir.iterdir() if q.is_file()))
    for q in files:
        try:
            h.update(str(q.relative_to(ctx.ref_price_dir)).encode())
            h.update(q.read_bytes())
        except (OSError, ValueError):
            pass
    h.update(repr(ctx.ref_price_app).encode())
    h.update(str(_pm_interval(ctx)).encode())
    return h.hexdigest()[:16]


def _report_waives(ctx: LoopContext, *patterns) -> bool:
    if ctx.recon_path is None:
        return False
    try:
        text = ctx.recon_path.read_text()
    except OSError:
        return False
    return all(p.search(text) is not None for p in patterns)


def _ref_price_missing(ctx: LoopContext) -> list:
    return [n for n, ok in (("ROLE", _ref_overlap_banked(ctx)),
                            ("PMTL", _ref_timeline_banked(ctx))) if not ok]


def _ensure_ref_pricing(ctx: LoopContext, label: str = "meta") -> None:
    """Price the reference tree with the frozen recipe — the ref's overlap account.

    The role-exposure and dossier tables have two sides; the ref side used to
    come from wherever the teacher happened to produce it. A source-mode run has no
    recon step, so its tables were one-sided (an offline check on grouped
    gemm showed the teacher reading a "ref" column that was the student column
    copied verbatim). A blackbox run's teacher priced the ref inside its own
    recon probes — improvised collections that, on one dsa_sparse run,
    fought ncu for two rounds and banked a ROLE table with no tensor term, so
    the ref's exposure column was null for the whole 17h run and the one
    measurement that separates gather/MMA overlap from its absence never
    reached a teacher. Four blackbox runs guessed that overlap wrong at v1.

    So the driver owns it: ``ctx.ref_price_dir`` (loop.toml ``ref_price_op``;
    a blackbox run defaults it to the recon tree) gets the toolkit seeded, the
    tree committed (the job archives the branch, not the working tree), and
    ``stall_price_run.py --pm`` run there — one job yields the stall ledger,
    segment account, dossier, role table WITH tensor and the phase timeline —
    and whatever verdict blocks come back are banked. Called at the end of
    reconnaissance (so the account exists before v1 is authored) and at every
    teacher entry; a no-op once the account is banked, once the teacher has
    waived what is missing, and on an unchanged tree after a job that came
    back without it (a retry re-runs only when the tree, the launch target or
    the app spelling changed; at most three attempts per run).
    """
    if not ctx.stall_pricing or ctx.ref_price_dir is None \
            or ctx.ref_price_backend is None:
        return
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    missing = _ref_price_missing(ctx)
    if not missing:
        return
    if _report_waives(ctx, *[_ROLE_WAIVED if m == "ROLE" else _PMTL_WAIVED for m in missing]):
        return  # the teacher owned the absence; no job behind its back
    if ctx.now() > ctx.deadline_epoch - 1800:
        _log(f"{label}/ref-price: < 30 min of budget left — not pricing the reference tree")
        return
    entry, named = _ref_price_entry(ctx)
    if entry is None or not entry.is_file():
        # Re-checked on every call, not once at preflight: a teacher that adds
        # the launch target during reconnaissance is priced at the next finish.
        _log(f"{label}/ref-price: launch target {named} not found in "
             f"{ctx.ref_price_dir} — the pricing recipe has no profiler-window entry; "
             f"the ref's overlap account stays missing")
        return
    markers = _ref_price_markers(ev)
    key = _ref_pricing_key(ctx)
    if markers:
        if markers[-1].read_text().startswith(f"[ref-price] key={key}"):
            return  # same tree, same target, same spelling: the job would repeat itself
        if len(markers) >= len(_REF_PRICE_MARKER_NAMES):
            return
        if "ROLE" not in missing and _marker_pm_interval(markers[-1]) == _pm_interval(ctx):
            # only the timeline is missing and the sampling interval is what the
            # last attempt ran with — the same collection would starve again
            return
    if ctx.ref_price_submit_failures >= _REF_PRICE_MAX_SUBMIT_FAILURES:
        return
    be = ctx.ref_price_backend
    cmd = _stall_price_cmd(ctx)
    if ctx.ref_price_app:
        cmd += " --app " + " ".join(shlex.quote(a) for a in ctx.ref_price_app)
    marker = ev / _REF_PRICE_MARKER_NAMES[len(markers)]
    head_before = getattr(be, "last_job_id", "")
    try:
        _seed_pricing_toolkits(ctx.ref_price_dir, be)
        # The job archives the BRANCH: a launch target the teacher just wrote
        # into the working tree is not in it until committed (--allow-empty,
        # so an unchanged tree costs nothing).
        be.commit("ref-price: snapshot the tree the pricing job archives")
        branch = getattr(be, "current_branch", lambda: "main")()
        _log(f"{label}/ref-price: pricing the reference tree {ctx.ref_price_dir.name} "
             f"@ {branch} (attempt {len(markers) + 1}/{len(_REF_PRICE_MARKER_NAMES)}, "
             f"{named})")
        log = be.submit_gpu(branch, cmd)
    except BackendError as e:
        # A job that launched and produced nothing usable is an attempt; a
        # verdict that nothing ran (NoRunError) never is, whatever it says —
        # a submit that is retried also moves the job id, so the id
        # moving proves launch, not a run.
        ran = (getattr(be, "last_job_id", "") not in ("", head_before)
               and not isinstance(e, NoRunError))
        if ran:
            # The job ran and did not come back usable (poll timeout on a hung
            # reference kernel, log lost after success): that is an attempt —
            # the same tree would hang again.
            ctx.ref_price_note = ""
            try:
                marker.write_text(f"[ref-price] key={key} app={ctx.ref_price_app!r} "
                                  f"pm_interval={_pm_interval(ctx)}\n"
                                  f"[ref-price] job failed after launch: {e}\n")
            except OSError:
                pass
            _log(f"{label}/ref-price: job failed after launch ({e}) — counted as an "
                 f"attempt; role/dossier tables stay one-sided")
            _bank_evidence(ctx, "ref-price attempt marker")
        else:
            # Nothing ran — never submitted (git push failed), or submitted and
            # never produced output (queue starved, job lost, never
            # got a node): not an attempt, bounded per session.
            ctx.ref_price_submit_failures += 1
            ctx.ref_price_note = f"the pricing job did not run: {e}"
            _log(f"{label}/ref-price: nothing ran ({e}) — not an attempt "
                 f"({ctx.ref_price_submit_failures}/{_REF_PRICE_MAX_SUBMIT_FAILURES} "
                 f"this session)")
        return
    ctx.ref_price_note = ""
    from forgekernel.skills.stall_price_run import expand_bundle
    log = expand_bundle(log)
    banked = []
    _save_ref_stall_ledger(ctx, log)
    if _STALL_OK.search(log) is not None:
        banked.append("ledger")
    if (ev / _REF_SEGMENT_LEDGER_NAME).is_file():
        banked.append("segments")
    if _ORCH_OK.search(log) is not None:
        _save_ref_orch_dossier(ctx, log)
        banked.append("dossier")
    if _ROLE_ANY.search(log) is not None:
        _save_ref_role_exposure(ctx, log)
        banked.append("roles+tensor" if _ROLE_OK.search(log) is not None
                      else "roles (PARTIAL: no tensor term)")
    _save_ref_pm_timeline(ctx, log)
    if _PMTL_OK.search(log) is not None:
        banked.append("timeline")
    try:
        marker.write_text(f"[ref-price] key={key} app={ctx.ref_price_app!r} "
                          f"pm_interval={_pm_interval(ctx)}\n" + log)
    except OSError as e:
        _log(f"{label}/ref-price: could not persist {marker.name}: {e} — "
             f"the reference will be priced again at the next teacher entry")
    # The snapshot commit above ran BEFORE the job; the ledgers, dossier, role
    # table, timeline and marker were written after it and committed by nobody —
    # the exact four files that once held an mla_paged_decode resume at the
    # dirty-tree preflight.
    _bank_evidence(ctx, f"ref-price {', '.join(banked) or 'attempt marker'}")
    _log(f"{label}/ref-price: banked {', '.join(banked) or 'nothing (no OK verdict line)'}"
         f" (job {getattr(be, 'last_job_id', '') or '?'})")
    if _ref_price_missing(ctx):
        _log(f"{label}/ref-price: still missing {'/'.join(_ref_price_missing(ctx))} — "
             f"no retry on an unchanged tree")


def _ref_price_status(ctx: LoopContext, missing: list) -> "tuple[str, str]":
    """(what happened, what the teacher can do) for the overlap gate message,
    derived from what is on disk — the markers and the banked files — so it
    is right after a resume, after a job that banked half the account, and
    after the attempts are spent. ``missing`` is the gate's ROLE/PMTL list."""
    if not ctx.stall_pricing or ctx.ref_price_dir is None or ctx.ref_price_backend is None:
        return ("this run has no reference pricing tree configured (`[loop] ref_price_op`); "
                "the driver has not run pricing",
                "configure `ref_price_op` in loop.toml (default for a blackbox run = the recon "
                "tree) and restart, or waive")
    entry, named = _ref_price_entry(ctx)
    ev = _evidence_dir(ctx)
    markers = _ref_price_markers(ev) if ev is not None else []
    if ctx.ref_price_note:
        return (ctx.ref_price_note, "wait for the driver to retry the submission at the next "
                f"finish attempt (at most {_REF_PRICE_MAX_SUBMIT_FAILURES} times this "
                "session), or waive")
    if entry is None or not entry.is_file():
        if ctx.ref_price_app:
            return (f"the entry file `{named}` points to is not in the recon tree; the driver "
                    f"has not run pricing",
                    f"fix `ref_price_app` in loop.toml (the driver uses it and does **not** "
                    f"fall back to eval/ncu_run.py), or write that entry file into the recon "
                    f"tree; then simply end the turn again")
        return (f"the recon tree has no `eval/ncu_run.py` and loop.toml has no "
                f"`ref_price_app`; the driver has not run pricing",
                "add `eval/ncu_run.py` to the recon tree (a few lines: build the inputs, "
                "`torch.cuda.cudart().cudaProfilerStart()`, call the reference kernel once, "
                "`cudaProfilerStop()`; open the profiler only around the kernel call), then "
                "**simply end the turn again** — the driver submits and re-prices at the "
                "finish check; you do not need to send a probe")
    if not markers:
        return ("the driver has not run pricing yet (budget too low, or this session's "
                "submission failures hit the cap)", "waive, or wait for the next finish attempt")
    last = markers[-1].read_text()
    verdicts = " / ".join(ln for ln in last.splitlines()
                          if re.match(r"^(ROLE|PMTL|STALL)-(OK|PARTIAL|FAIL)\b|\[ref-price\] job failed", ln))
    what = (f"the driver has run pricing {len(markers)} time(s) (entry {named}); last "
            f"verdict lines: {verdicts[:300] or 'none'}; still missing {'/'.join(missing)}")
    if missing == ["PMTL"]:
        n = _pm_interval(ctx)
        return (what, f"the role ledger is in; only the phase timeline is missing (PM sampling "
                      f"interval this time: {n} cycles): with a reference under 0.5 ms, 8192 "
                      f"collects too few samples — write `[loop] ref_price_pm_interval = 2048` "
                      f"in loop.toml and restart, and the driver retries once with the new "
                      f"interval; if the interval is already small enough and it still collects "
                      f"nothing, waive with `PMTL-WAIVED:`")
    same_key = last.startswith(f"[ref-price] key={_ref_pricing_key(ctx)}")
    if len(markers) >= len(_REF_PRICE_MARKER_NAMES):
        return (what, "pricing attempts are used up and the driver will not retry: decide by "
                      "the reason in the verdict lines — if it cannot be fixed, waive")
    if same_key:
        return (what, "rerunning the same tree with the same entry would only reproduce the "
                      "same verdict: the driver will not retry automatically. Ending the turn "
                      "again after changing the entry file or `ref_price_app` triggers a "
                      "retry; otherwise waive")
    return (what, "the recon tree has changed; simply end the turn again and the driver "
                  "re-prices")


_STALL_PRICE_CMD = "cmd:python3 notes/stall_price_run.py --pm"
_PM_INTERVAL_DEFAULT = 8192      # == stall_price_run.PM_INTERVAL
_PM_INTERVAL_FAST = 2048
_PM_FAST_KERNEL_MS = 0.5


def _pm_interval(ctx: LoopContext) -> int:
    """The PM-sampling interval the pricing recipe runs with: the configured
    one; else 2048 when the run's target reading says its kernels are sub-0.5 ms
    (the recipe's own note: fast kernels starve at 8192); else 8192."""
    if ctx.ref_price_pm_interval and int(ctx.ref_price_pm_interval) > 0:
        return int(ctx.ref_price_pm_interval)
    try:
        target = _resolve_target(ctx)
    except Exception:  # noqa: BLE001 — no milestone / target seed yet
        target = None
    if target and target < _PM_FAST_KERNEL_MS:
        return _PM_INTERVAL_FAST
    return _PM_INTERVAL_DEFAULT


def _stall_price_cmd(ctx: LoopContext) -> str:
    """The frozen recipe with the interval this run needs (the student's and
    the reference's tree run the same problem in the same duration class); the
    flag is only spelled when it departs from the script's default. `--app`
    (the ref pricing appends it) must stay LAST: the script takes everything
    after it as the launch target."""
    n = _pm_interval(ctx)
    return _STALL_PRICE_CMD if n == _PM_INTERVAL_DEFAULT else f"{_STALL_PRICE_CMD} --pm-interval {n}"


def _marker_pm_interval(marker: Path) -> int:
    """The PM interval a pricing attempt ran with (its marker's first line);
    attempts from before the field existed ran the script's default."""
    try:
        m = re.search(r"pm_interval=(\d+)", marker.read_text().splitlines()[0])
    except (OSError, IndexError):
        m = None
    return int(m.group(1)) if m else _PM_INTERVAL_DEFAULT


# STALL-PRICE REUSE: every teacher entry prices the best KEEP (and HEAD when it
# differs), but the best rarely moves between entries — in one large batch of
# runs stall pricing was ~6% of loop wall time and 553 of its 1026
# whole readings were a sha the run had already priced (one run: 19 of 22
# jobs on one best). The ledger,
# segment account and dossier are readings of the tree, so one whole reading
# per (run, sha, recipe) serves every later entry of this process; parsing and
# banking still run on each use, only the job is skipped. Keyed by the run's
# status path so tests and runs sharing a process never see each other's.
_STALL_PRICE_LOGS: "dict[tuple[str, str, str], tuple[str, str]]" = {}

_REF_PRICE_MAX_SUBMIT_FAILURES = 3
_REF_PRICE_MARKER_NAMES = ("ref_price_job.log", "ref_price_job.2.log", "ref_price_job.3.log")


def _run_stall_pricing(ctx: LoopContext, ref: str | None = None, *,
                       bank_as: "tuple[str, ...]" = (_STUDENT_ORCH_DOSSIER_NAME,),
                       label: str = "best", workload: str | None = None) \
        -> "tuple[str | None, str | None, str | None, str | None]":
    """Price one student tree (``ref``; the best-so-far KEEP when None);
    return (row ledger, segment account, orchestration dossier, role exposure).
    ``workload``: one of a group card's other meta workloads (``--workload``;
    loop/meta_shapes.py), read beside the representative's and banked as nothing
    (``bank_as`` = () from its caller) -- the banked dossiers stay the card's own.

    The collection rides the PM-sampling sections (``--pm``): the role
    exposure ledger scales stall shares by the mainloop tensor plateau, which
    only the timeline reads; stall_price_run falls back to the plain recipe
    when the combined collection crashes, and the role ledger then scales by
    the whole-kernel aggregate instead (labelled in its own line).

    One GPU job per plateau diagnosis (~1-3 min) buys the diagnose step the one
    evidence face the aggregate digest cannot provide: where the time measurably
    leaks, per instruction, summed. The segment account (PC-keyed, schedule-
    joined) rides the same job — the row ledger's line keying loses the
    scheduling rows to the no-line bucket. The student's orchestration dossier
    (stall_price_run renders the SASS page off the same collection) rides too:
    its ORCH-SUMMARY-JSON line is what the dossier diff against the ref's
    recon-time dossier is computed from, and the body is banked beside the ref's
    for later audits. A job failure degrades to the pre-ledger behavior instead
    of blocking the diagnosis — the ledger is evidence, not a precondition.
    """
    if not ctx.stall_pricing:
        return None, None, None, None
    be = ctx.backend
    if ref is None:
        records = load_records(ctx.status_path)
        ref = _best_good_ref_here(records, ctx)
    branch = f"{ctx.branch_prefix}-stallprice"
    cmd = _stall_price_cmd(ctx) + (f" --workload {workload}" if workload else "")
    push_ref = getattr(be, "push_ref", None)
    key = (str(ctx.status_path), ref, cmd) if ref and push_ref is not None else None
    hit = _STALL_PRICE_LOGS.get(key) if key else None
    if hit is not None:
        log, job = hit
        _log(f"meta/stall-price[{label}]: reusing the pricing of {ref[:10]} (job {job}) — "
             "same tree, same recipe")
    else:
        try:
            if key:
                push_ref(ref, branch)
            else:
                # no best sha yet (pre-green plateau) or a backend without push_ref:
                # price the workspace HEAD — worse than best is still measured truth.
                be.push_branch(branch)
                ref = be.head()
            log = be.submit_gpu(branch, cmd)
        except BackendError as e:
            _log(f"meta/stall-price[{label}]: pricing job failed ({e}) — "
                 "diagnosing without a ledger")
            return None, None, None, None
        # job log archives keep ~300 lines; the bundle at the tail carries the
        # whole analysis (see stall_price_run.expand_bundle).
        from forgekernel.skills.stall_price_run import expand_bundle
        log = expand_bundle(log)
        job = getattr(be, "last_job_id", "") or "?"
        if key and _STALL_LEDGER_RE.search(log) and _STALL_OK.search(log) \
                and (dm := _ORCH_DOSSIER_RE.search(log)) is not None \
                and _orch_summary(dm.group(1)) is not None:
            # only a whole reading is kept: a job that came back without its
            # ledger, self-check or dossier is priced afresh next entry
            _STALL_PRICE_LOGS[key] = (log, job)
    roles = None
    rm = _ROLE_BLOCK_RE.search(log)
    if rm is not None and _ROLE_ANY.search(log) is not None:
        roles = rm.group(1)
        full = _ROLE_OK.search(log) is not None
        _log(f"meta/stall-price[{label}]: role exposure "
             f"{'OK' if full else 'PARTIAL (no tensor term — PM sections missing)'} @ "
             f"{ref[:10] if ref else '?'}")
        if (ev := _evidence_dir(ctx, create=True)) is not None and (
                _STUDENT_ORCH_DOSSIER_NAME in bank_as
                or _STUDENT_BEST_ORCH_DOSSIER_NAME in bank_as):
            try:
                (ev / _STUDENT_ROLE_EXPOSURE_NAME).write_text(roles)
            except OSError as e:
                _log(f"meta/stall-price[{label}]: could not persist "
                     f"{_STUDENT_ROLE_EXPOSURE_NAME}: {e}")
    dossier = None
    dm = _ORCH_DOSSIER_RE.search(log)
    # Banked on a parsable summary, NOT on the trailing ORCH-OK line: the
    # summary carries its own verdict, and a dossier that fails for
    # no-sync-sites is the "blueprint declared, nothing built" reading this
    # channel exists to deliver — requiring ORCH-OK dropped exactly that case.
    if dm is not None and (summ := _orch_summary(dm.group(1))) is not None:
        dossier = dm.group(1)
        _log(f"meta/stall-price[{label}]: orch dossier "
             f"{summ.get('verdict', 'ORCH-OK')} @ {ref[:10] if ref else '?'}")
        if (ev := _evidence_dir(ctx, create=True)) is not None:
            for name in bank_as:
                dst = ev / name
                try:
                    dst.write_text(dossier)
                except OSError as e:
                    _log(f"meta/stall-price[{label}]: could not persist {name}: {e}")
    if roles is not None or dossier is not None:
        _bank_evidence(ctx, f"stall-price[{label}] dossier")
    m = _STALL_LEDGER_RE.search(log)
    ok = _STALL_OK.search(log) is not None
    if m is None:
        why = env_failure(log)
        if why is not None:
            _note_env_failure(ctx, "stall-price", why)
        _log(f"meta/stall-price[{label}]: job returned no ledger"
             + (f" (job environment: {why})" if why else "") + " — diagnosing without one")
        return None, None, dossier, roles
    _note_env_ok(ctx, "stall-price")
    _log(f"meta/stall-price[{label}]: ledger {'OK' if ok else 'SELF-CHECK-FAIL'} "
         f"@ {ref[:10] if ref else '?'} (job {job})")
    # HEAD, not tail: ledger rows print most-expensive first, and the diagnose hard
    # rule holds the locus against the TOP of the ledger. A tail slice would drop
    # exactly those rows, and only in the flat-distribution (dispersed-gap) case
    # this mechanism exists for — the ledger would fail first where it matters most.
    # The opcode-family table at the tail is the acceptable loss.
    body = m.group(1)[:9000]
    if not ok:
        body = ("(WARNING: ledger self-check failed — `STALL-OK` not printed; sample count / "
                "calibration / coverage fell short. Use it only as supporting evidence; "
                "conclusions must be downgraded.)\n" + body)
    seg = _SEGMENT_LEDGER_RE.search(log)
    seg_body = None
    if seg is not None:
        # Verdict before slice: JOIN-OK is segment_ledger's LAST line and the SEG
        # rows above it are unbounded, so a many-segment kernel's block overruns
        # the head-slice — checking after slicing would read every successful join
        # as unsafe on exactly the kernels this table exists for. An unsafe join
        # is still not evidence: no JOIN-OK in the full block drops the table.
        if "JOIN-OK" in seg.group(1):
            seg_body = seg.group(1)[:6000]
    return body, seg_body, dossier, roles


def _require_recon_artifacts(ctx: LoopContext, be, sass_ok: bool = True,
                             sched_ok: bool = True, live_ok: bool = True,
                             stall_ok: bool = True, struct_ok: bool = True,
                             since: float | None = None,
                             near_miss: dict | None = None,
                             launch: dict | None = None) -> None:
    """The reconnaissance only exists as what it wrote down — enforce that it did.

    Observed live: a teacher spent all 12 rounds probing and the loop moved on to
    distillation with {RECON_REPORT} pointing at a file nobody had written, so the
    entire reconnaissance would have reached the milestone as a dangling path. The
    findings were all on disk — in the probe docstrings — but the distill prompt
    reads the report, not the notes. Probing turns are over when this runs: the
    teacher gets write-up turns, and a further RECON_REQUEST is answered with this
    same instruction rather than a job.
    """
    if ctx.recon_path is None:
        return
    candidates = ctx.recon_path.parent / "candidates.md"
    writeup_storm = [TRUNCATION_STORM_BUDGET]
    for _ in range(3):
        lacking = _missing_recon_evidence(ctx, sass_ok, sched_ok, live_ok,
                                          stall_ok, struct_ok, near_miss=near_miss)
        contras = _launch_contradictions_open(ctx, launch)
        if _recon_artifacts_written(ctx, since=since) and not lacking and not contras:
            return
        missing = " and ".join(f"`{p}`" for p in (ctx.recon_path, candidates)
                              if not p.exists())
        gaps = []
        if contras:
            c = launch["cluster"]
            _log(f"meta/recon: RECON-CONTRADICTION cluster: NCU read cluster={c} "
                 f"but the write-up denies it ({contras[0][:100]})")
            gaps.append(f"RECON-CONTRADICTION cluster: the NCU launch metrics read cluster="
                        f"({c[0]},{c[1]},{c[2]}), but the report / candidates table still has "
                        f"lines denying a cluster ({'; '.join(contras[:3])}) — rewrite §1 and "
                        f"the candidates table to the measured cluster (the first line of each "
                        f"candidate states the cluster shape and CTA count first), or write "
                        f"`LAUNCH-DISPUTED: <the log line carrying the reading + criterion>`.")
        if missing:
            gaps.append(f"These deliverables are not written yet: {missing}.")
        elif not _recon_artifacts_written(ctx, since=since):
            # both files exist but predate this session — stale residue from a
            # previous run (possibly a different operator's, the default recon
            # path ships populated). The teacher must re-affirm them.
            gaps.append("Both deliverables are on disk but **were not written by this "
                        "reconnaissance** (stale residue) — check that their content belongs "
                        "to this operator, and update them in this session.")
        if lacking:
            # GPU rounds are gone, so delivery is off the table — the only remaining
            # legal state is an owned waiver, not a silent absence.
            waivers = " / ".join(f"`{n}-WAIVED: <the concrete reason and the channels tried>`"
                                 for n in lacking)
            gaps.append(f"This reconnaissance lacks {'/'.join(lacking)} evidence, and the "
                        f"report has no matching waiver line — the GPU budget is spent and "
                        f"delivery is no longer possible, so you **must** add {waivers} to "
                        f"the recon report.")
            # Both sides of any checker/printer disagreement, verbatim — a
            # deadlock (checker false-negative × honest refusal to
            # waive) is unfindable without this.
            gaps.extend(_near_miss_notes(lacking, near_miss))
        _log(f"meta/recon: write-up incomplete ({'; '.join(gaps)[:120]}) — write-up turn")
        _agent_turn(be, _recon_prompt(ctx, extra=(
            f"\n\n## Recon rounds are over — now only write the deliverables\n\n"
            f"The GPU budget is used up; **you can no longer send RECON_REQUEST** (one sent "
            f"will not be executed).\n"
            + "\n".join(gaps) + "\n"
            f"All your findings are in the recon tree's `notes/probe*.py` docstrings and "
            f"`notes/*.log` — read them now and complete the deliverables. Write as much as "
            f"was measured, and mark what was not honestly as \"not measured\"; an honest "
            f"hole is more useful than a fabricated whole.")),
            label="meta/recon-writeup", storm_budget=writeup_storm,
            left_a_trace=lambda o: _recon_artifacts_written(ctx, since=since))
    if not (_recon_artifacts_written(ctx, since=since)
            and _sass_evidence(ctx, sass_ok, near_miss)):
        _log("meta/recon: WARNING — write-up turns exhausted and artifacts are still "
             "incomplete; distillation will see whatever exists")


def parse_student_turn_safe(out: str, *, allow_recon: bool = False):
    """parse_student_turn, but prose with no marker is a finished turn, not an error.

    The student is required to end in one of three shapes and a violation is a
    protocol error worth re-prompting. The teacher's reconnaissance turn is different:
    "no further request" is how it signals it is done, and that is most naturally
    written as prose — but the caller only accepts the signal once the write-up
    artifacts exist, because a truncated turn produces the same absence.
    """
    try:
        return parse_student_turn(out, allow_recon=allow_recon)
    except ValueError:
        return None


def _source_recon_distill_addendum(ctx: LoopContext) -> str:
    """What a source-mode run's distill brief gains when reconnaissance ran.

    distill_prompt.md is the control run's brief and stays untouched; the
    addendum is appended only when a report exists on disk, so a source-mode run
    without a recon tree renders exactly what it always did. The brief's own
    text tells the teacher to transcribe the issue order, the exchange-surface
    atoms and the segment budget "from the source at cost ≈ 0" — the addendum
    redirects those transcriptions to the measured page wherever the report
    carries one, and keeps the source for what only the source says (why, and
    the parts no probe covered).
    """
    if ctx.oracle_mode != "source" or ctx.recon_path is None:
        return ""
    if not ctx.recon_path.is_file():
        return ""
    candidates = ctx.recon_path.parent / "candidates.md"
    return (
        "\n## Recon report (supplementary input in source mode)\n\n"
        f"Before distillation this run reconnoitred the reference binary: `RECON_REPORT` = "
        f"`{ctx.recon_path}` (candidates table `{candidates}`, recon tree "
        f"`{ctx.recon_path.parent}`, probe logs in its `notes/`). "
        "**Read it first, then the ORACLE.**\n\n"
        "- Every item above that says \"transcribe from the ORACLE source\" — tile / stage / "
        "warp-role values, exchange-surface access atoms, issue order and the segmented "
        "live-set budget, wait-target relations, barrier topology, inter-CTA dispatch order — "
        "**where the recon report has a measured reading, transcribe the measurement**, "
        "tagged `[MEASURED, job N]`; use the source only to explain intent and to fill what "
        "the recon did not cover. When the source's template parameters disagree with the "
        "binary's reading, write the binary's and record the disagreement in the decision.\n"
        "- The two places above that say \"without a recon report, do it yourself / "
        "transcribe directly from the source\" **do not apply** to this run: the load-bearing "
        "ruling on inter-CTA dispatch order follows the report's **§3b residency ruling**; "
        "the intermediate-buffer ledger of a multi-kernel operator (including the \"who "
        "allocates, who zeroes\" column) follows the report's **§3c measured ledger**, and "
        "the source only fills the ledger's empty columns.\n"
        "- For quantities the recon report marks \"not measured\", go back to the source; "
        "for those the source does not have either, mark `[ESTIMATE]` + derivation as the "
        "rules say.\n"
        "- The reference's NCU signature, SASS footprint, stall ledger and orchestration "
        "dossier are all in the recon tree; anchor mechanism-verification lines on them "
        "first.\n"
        "- The red line is unchanged: do not transcribe the ORACLE's implementation body; "
        "the recon deliverables are likewise readable only by you — the student cannot see "
        "them.\n"
    )


def _distill_prompt_path(ctx: LoopContext) -> Path:
    """Which distillation brief the teacher gets.

    The source-mode prompt is left untouched rather than branched inside, because the
    control run has to use the same instructions it would have used without this
    experiment existing. A single prompt carrying "if you have source ... otherwise ..."
    would differ from the status quo for both modes, and then the comparison is against
    something nobody ever ran.
    """
    if ctx.oracle_mode == "blackbox":
        p = _meta_prompt(ctx, "distill_blackbox_prompt.md")
        if p.exists():
            return p
        _log("meta/distill: blackbox mode but no distill_blackbox_prompt.md — falling "
             "back to the source prompt, which references an oracle this run lacks")
    return _meta_prompt(ctx, "distill_prompt.md")


def _ref_overlap_block_for_distill(ctx: LoopContext) -> str:
    """The ref's measured overlap account for the distill prompt: role table
    with tensor + phase timeline when banked, an explicit absence note when
    not. The dossier says WHAT the ref's producer/consumer code is; this says
    how much of each role's waiting the tensor pipe hides — the quantity every
    blackbox v1 on dsa_sparse guessed, four runs in a row, because no teacher
    had it on the page."""
    from forgekernel.skills.role_exposure_tools import align_roles, parse_core
    roles = _read_ref_role_exposure(ctx)
    core = parse_core(roles)
    timeline = _read_ref_pm_timeline(ctx)
    banked = core is not None and core.get("tensor_active") is not None
    if not banked and not (ctx.stall_pricing and (ctx.oracle_mode == "blackbox"
                                                 or ctx.ref_price_dir is not None)):
        # A source-mode run that neither reconnoitres nor prices its ref reads the
        # overlap off the source; the absence note is a blackbox-mode rule
        # (and the control run's distill prompt must stay byte-identical).
        return ""
    if not banked:
        return (
            "\n## Reference overlap ledger: **absent**\n"
            "The reference's role exposure ledger has no tensor duty cycle (or does not exist "
            "at all), and "
            + ("there is no phase timeline either" if timeline is None
               else "there is a phase timeline") +
            ". This means \"how the reference hides its supply behind the MMA\" is "
            "**unmeasured**: every [SYNC:]/[GATHER:]/[PHASE:] ruling in the milestone may only "
            "be written as a candidate carrying [ESTIMATE], never as settled; the M0 gate must "
            "not turn the overlap shape (supply issue order, tight-chain, spin share) into a "
            "hard line.\n")
    block = ("\n## Reference overlap ledger (measured by the recon-time pricing job: role × "
             "stall × tensor duty cycle)\n"
             "exposed = that role's stall share × (1 − tensor_active), an estimate of how much "
             "waiting the reference leaves **exposed** outside tensor work. Read this table "
             "first when ruling on the supply structure: which role exposes how many ns, how "
             "large the spin share is; if the student's v1 writes these roles' exposure an "
             "order of magnitude larger than the reference's, it is guessing.\n"
             f"```\n{align_roles(core, core, names=('ref', 'ref'))}\n```\n")
    if timeline is not None:
        block += ("\n### Reference phase timeline (PM sampling: WHEN, not WHERE)\n"
                  f"```\n{timeline[:4000]}\n```\n")
    return block


def _inferred_rows_block_for_distill(ctx: LoopContext) -> str:
    """Candidates rows the recon left `[VERIFY: inferred]`: structural readings
    whose discriminating observation printed nothing (dsa_sparse: the
    KV-ring stage axis, 32 key x 512 d vs 64 key x 256 d, byte-identical). The
    distill step must carry both readings as [ESTIMATE] candidates — the v1
    that treated the byte identity as a verdict cost a 13-decision reopen."""
    rows = [r for r in load_gap_queue(ctx) if r.reason == "inferred"]
    if not rows:
        return ""
    return ("\n## Inferred rows of the recon table (`[VERIFY: inferred]`, the discriminating "
            "observation gave no positive reading)\n"
            "The structural quantities below had more than one reading in the recon with "
            "identical counts/bytes, and the discriminating experiment printed no reading that "
            "points to one of them. The corresponding decisions **must** list every reading as "
            "a candidate with the numbers tagged `[ESTIMATE]`, and state in the five-tuple "
            "\"the student's first probe decides the axis\"; **they must not be written as "
            "settled and must not enter the M0 Gate**.\n"
            + "\n".join(r.line() for r in rows) + "\n")


def run_distill_step(ctx: LoopContext) -> "MetaResult":
    """Initial distillation: meta reads the ORACLE (open-source code) + POLICY (§9) and
    writes milestone v1 from scratch. Schema-gated (re-prompt until §9-conformant), same
    harness as a patch. Not transactional — there is no prior milestone to roll back to.

    Two hardenings from a dsa_indexer_fwd run:
    - each round's draft is snapshotted, so a revision round that leaves the file
      missing restores the last draft instead of validating a hole;
    - a round that ends with the milestone still nonexistent did no schema work at
      all — the agent session died before its Write (API timeout, truncated stream;
      the CLI exits 0 with the error as its final text, so the backend's rc-based
      retry never fires). Re-prompting such a round with "file unreadable" adds no
      information, so two of them in a row raise AgentError: the run stops as
      agent_failed (transient, watchdog-restartable), not distill_malformed."""
    be = ctx.backend
    bindings = _bindings(ctx)
    meta_step = len(load_meta_records(ctx.meta_status_path)) if ctx.meta_status_path else 0
    out, violations, draft, dead_rounds = "", [], None, 0
    storm_budget = [TRUNCATION_STORM_BUDGET]
    base_extra = (_source_recon_distill_addendum(ctx) + _ref_overlap_block_for_distill(ctx)
                  + _inferred_rows_block_for_distill(ctx))
    if ref_dossier := _read_ref_orch_dossier(ctx):
        # The recon-time orchestration dossier travels INTO the distill prompt
        # rather than waiting in the recon tree: the [SYNC:]/[GATHER:]/[PHASE:]
        # adjudications are exactly what it machine-reads, and a blueprint that
        # guesses them with this on disk repeats the failure §2g exists to end
        # (topology guessed symmetric, supply guessed inline — the recon report
        # prose kept losing the fine grain the dossier keeps).
        base_extra += (
            "\n## Reference orchestration dossier (machine-read at recon time, static SASS "
            "transcription)\n"
            "The milestone's [SYNC:]/[GATHER:]/[PHASE:] rulings must match this dossier one "
            "by one: sync topology (for each barrier object, who arrives and who waits, the "
            "granularity, the release chain), supply issue order (bulk prefetch vs per-row "
            "inline, tight-chain count), in-flight depth (the DEPBAR immediate table), "
            "cache-policy bits. Where an axis of the dossier is UNRESOLVED or absent, rule "
            "from the recon report as before — but on every axis where the dossier has a "
            "clear reading, the ruling cites it instead of guessing.\n"
            f"```\n{ref_dossier[:6000]}\n```\n")
    if ref_struct := _read_ref_struct_table(ctx):
        # The ref's per-phase construction table — the only place a blackbox run
        # has the reference's BUILD (warps / tile granule / precision / hand-off).
        # Not a [SYNC:]/[GATHER:] axis, so it travels alongside the dossier: the
        # phase contracts the milestone writes (esp. an internal phase like a
        # triangular solver) must match this construction, not an inference from
        # the instruction mix.
        base_extra += (
            "\n## Reference per-segment construction table (decoded mechanically at recon "
            "time; a blackbox run's only per-segment construction facts)\n"
            "This is **how** the reference's hot region **is built**, segment by segment — "
            "how many warps per segment, what block size it addresses, what internal "
            "precision, how segments hand off. The milestone's rulings on **a segment's "
            "internal construction** (block granule / doubling levels / reduction placement) "
            "must follow this table; the recon report's prose only paraphrases it — when the "
            "two conflict, the table wins and the disagreement is marked.\n"
            f"```\n{ref_struct[:REF_STRUCT_PROMPT_CHARS]}\n```\n")
    for _ in range(max(1, ctx.max_patch_rounds)):
        extra = base_extra
        if violations:
            extra += ("\n## SCHEMA violations (the milestone you wrote does not conform to "
                      "the §9 skeleton; you must fix it and write it again)\n"
                      + "\n".join(f"- {v}" for v in violations) + "\n")
        pre = _read_or_none(ctx.milestone)
        out = _agent_turn(
            be, render_prompt(_distill_prompt_path(ctx), bindings, extra),
            label="meta/distill", storm_budget=storm_budget,
            # the only legal trace of a distill turn is the milestone changing on
            # disk — a tiny output that left the draft byte-identical is a
            # truncated stream, not a revision, and re-prompting it with the same
            # violation list would burn one of only max_patch_rounds rounds
            left_a_trace=lambda o: _read_or_none(ctx.milestone) != pre)
        if not Path(ctx.milestone).exists():
            if draft is not None:
                Path(ctx.milestone).write_text(draft, encoding="utf-8")
                _log("meta/distill: milestone.md missing after a revision round -> "
                     "restored previous draft and re-prompting")
            else:
                dead_rounds += 1
                if dead_rounds >= 2:
                    raise AgentError(
                        "two distill rounds in a row ended with no milestone file — "
                        "the agent sessions are dying before they write "
                        f"(last output tail: {out[-160:]!r})")
        else:
            dead_rounds = 0
        violations = _validate_milestone_file(ctx.milestone, authoring=True)
        try:
            draft = Path(ctx.milestone).read_text(encoding="utf-8")
        except OSError:
            pass  # nothing written yet this round; keep the last good draft
        if not violations:
            break
        _log(f"meta/distill: milestone still has {len(violations)} violation(s) -> re-prompt")
    _log(f"meta/distill: authored milestone v1 (violations={violations[:3] or 'none'})")
    _seed_target_ms(ctx)
    if ctx.meta_status_path:
        append_meta_record(ctx.meta_status_path, MetaRecord(
            meta_step=meta_step, cause_class="DISTILL", deficient_locus="(initial)",
            patch_class="author", changed_entries=[], patch_direction="author milestone v1 from oracle",
            outcome="AUTHORED", milestone_violations=violations))
    return MetaResult(out, "author", [], None, [], certified=None, milestone_violations=violations)


def run_two_layer(ctx: LoopContext, max_meta: int = 3) -> dict:
    """Inner loop -> (plateau / stuck / exhausted) -> meta patch -> resume, bounded by max_meta.

    Milestone patches are transactional: after a patch's inner segment runs, the patch
    is attributed (best-before vs best-after). A patch that produced no gain is rolled
    back — milestone snapshot restored AND workspace reset — and recorded REVERT_PATCH,
    which the next diagnosis reflects on so the meta agent does not re-propose it.
    A patch whose segment produced no metric reading at all is HELD instead (see
    _judge_patch): pre-green construction survives the wall, and the transaction stays
    open for a joint judgment once a metric exists.
    """
    meta_steps = 0
    start_iter = len(load_records(ctx.status_path))
    # Seed the stall-pricing toolkit BEFORE capturing baseline_ref: the seed is a
    # workspace commit, and foundation patches reset the workspace to baseline_ref —
    # a seed committed after that capture is discarded by the first such reset, and
    # every later pricing job dies file-not-found (exactly the blindness this
    # mechanism exists to prevent). Seeding first puts the recipe below the reset
    # floor, so every lineage commit (the future best included) carries it.
    _seed_ws_stall_tools(ctx)
    baseline_ref = ctx.backend.head()  # foundation patches restart from here, not HEAD
    if not ctx.harness_anchor:
        ctx.harness_anchor = baseline_ref or ""
    pending: _PendingPatch | None = None
    # Correctness state of the segment about to run: None = unknown (a fresh workspace —
    # the first full eval decides; a scaffolded operator enters scaffold mode on its
    # first FAIL), then re-derived from whatever checkpoint each meta reset lands on.
    seg_green: bool | None = None
    resume_anchor: str | None = None   # first segment only; see _resume_anchor
    if start_iter:
        # A process starting over an existing trajectory (budget / certificate
        # resume) is not a fresh workspace: the records say whether the tree it
        # resumes on has ever been green. Seeding "unknown" here made the first
        # post-resume construction row a paid REVERT (observed on a GDN run).
        resumed = load_records(ctx.status_path)
        seg_green = _green_on_resume(resumed, baseline_ref)
        if seg_green:
            # The anchor is the last KEEP, not HEAD: a HEAD no record owns carries
            # commits no eval verified (see _resume_anchor).
            resume_anchor = _resume_anchor(resumed, baseline_ref)
        if seg_green is not None:
            _log(f"resume: lineage at {str(baseline_ref)[:10]} reads "
                 f"{'green' if seg_green else 'under construction'} from the "
                 f"{start_iter} recorded iteration(s)"
                 + ("" if seg_green else " -> scaffold mode until the gate row")
                 + (f"; rollback anchor = last KEEP {str(resume_anchor)[:10]} (HEAD "
                    f"carries commits no record verified)"
                    if seg_green and resume_anchor != baseline_ref else ""))
    # An in-place reopen whose closure touches M0 enters green on the best tree
    # and rebuilds the foundation there; its M0 close is the next M0-gate KEEP,
    # and the audit is keyed on that row (see run_inner_loop audit_on_gate).
    audit_gate: str | None = None
    # distillation entry: if the milestone is an empty skeleton, meta authors v1 from
    # the oracle BEFORE the first inner loop (the loop then revises it on plateau).
    if _needs_authoring(ctx):
        # Distillation without a measured ref writes gates it cannot anchor: every
        # number would be an [ESTIMATE], including target_ms, which is the stop
        # threshold. Measure first (`forge <op> baseline`) — that is the whole point
        # of doing this before v1 rather than backfilling after M0.x.
        if ctx.ref_baseline is None:
            _log("meta/distill: no ref baseline and the operator does not declare one "
                 "-> refusing to distill. Add [loop] ref_cmd to operators/<op>/loop.toml "
                 "so the loop can measure it, or run `forge <op> baseline --ref-cmd ...`")
            return {"stop": "no_ref_baseline", "meta_steps": 0}
        _log(f"meta/distill: ref baseline {ctx.ref_baseline.ms} ms "
             f"(job {ctx.ref_baseline.job or '?'}, ncu={'yes' if ctx.ref_baseline.has_ncu else 'no'})"
             " -> authoring v1 anchored to it")
        # Reconnaissance first. Under blackbox it is the input that stands in for
        # the source the teacher is not allowed to read. A source-mode run reconnoitres
        # too whenever a recon tree is wired (the default; loop.toml recon_op = ""
        # declares a source-only control run): the oracle was assumed to answer
        # the same questions at no cost, and for templated references it does
        # not — the SASS-level issue order, the sync topology and the real
        # resource numbers are what the teacher had to guess, and guessed wrong
        # (grouped GEMM: -30% at M0).
        if ctx.oracle_mode == "blackbox" or ctx.recon_backend is not None:
            recon = run_recon_step(ctx)
            _log(f"meta/recon: {recon.rounds} run(s), "
                 f"{'finished' if recon.finished else 'budget exhausted'}"
                 f"{f', report {recon.report}' if recon.report else ''}")
        distilled = run_distill_step(ctx)
        # v1 is the one write with no rollback behind it: a patch that will not conform
        # is restored from the pre-patch snapshot, but there is no earlier milestone to
        # fall back to here. And v1 shapes everything downstream — later patches edit it
        # in place, so its defects propagate into every version. Refuse to start rather
        # than spend a GPU budget optimizing against a milestone that failed its gate.
        if distilled.milestone_violations:
            _log(f"meta/distill: v1 still non-conformant after {ctx.max_patch_rounds} "
                 f"round(s) -> refusing to start: {distilled.milestone_violations[:3]}")
            return {"stop": "distill_malformed", "meta_steps": 0,
                    "violations": distilled.milestone_violations}
    while True:
        # Force the M0 audit whenever the segment starts non-green (fresh workspace
        # or a reset into scaffolding): its first green IS an M0 close, including
        # the rebuilds after foundation patches.
        reason = run_inner_loop(
            ctx, start_iter, initial_green=seg_green, initial_good_ref=resume_anchor,
            audit_on_green=ctx.m0_audit and seg_green is not True,
            audit_on_gate=audit_gate if ctx.m0_audit else None,
            segment_records_before=(pending.records_before if pending else None),
            gate_contract=(pending.gate_contract if pending else None))
        _log(f"inner loop stopped: {reason} (meta_steps={meta_steps})")
        audit_gate = None   # one audit per reopen, whatever the stop was
        resume_anchor = None   # later segments enter on a checkpoint a record owns
        if reason == "first_green":
            # The M0 gate verified the student against the milestone's own declared
            # values; nothing yet verified the MILESTONE against the hardware. Audit
            # now, before a whole segment is built on the foundation. The open
            # pending patch (if any) is deliberately NOT judged here: its segment
            # was cut short at first green, and judging a rebuild patch at M0 is
            # predetermined (best_after < best_before) — it would revert correct
            # foundations for running out of runway, not for being wrong.
            if meta_steps >= max_meta:
                _log("meta/m0-audit: meta budget exhausted -> skipping audit")
                seg_green = True
                start_iter = len(load_records(ctx.status_path))
                continue
            records = load_records(ctx.status_path)
            best_before = _best_ms(_on_cluster(records, ctx), ctx.gate_window)
            milestone_before = _read_milestone(ctx)
            meta = run_meta_step(ctx, best_before=best_before,
                                 milestone_before=milestone_before, audit=True)
            _apply_phase_close(ctx, meta)
            if meta.patch_class == "audit-clean":
                _log("meta/m0-audit: AUDIT_CLEAN — milestone consistent with the "
                     "hardware truth -> resuming the segment")
                seg_green = True
                start_iter = len(load_records(ctx.status_path))
                continue
            if meta.patch_class == "kb-example":
                # student-side defect: recipe certified, same workspace resumes green
                meta_steps += 1
                ctx.fresh_recipe = meta.recipe
                ctx.g5_note = meta.student_directive or ""
                seg_green = True
                start_iter = len(load_records(ctx.status_path))
                continue
            if meta.patch_class in ("unpriced-reopen", "reopen-capped"):
                # the audit named a reopen it could not price (or the reopen budget
                # is spent): nothing landed, the refusal is in the meta memory, the
                # segment resumes as built — a priced version of the same reopen
                # can come at the next stop
                meta_steps += 1
                _log("meta/m0-audit: UNPRICED_REOPEN — reopen refused for want of a "
                     "measured price -> resuming the segment unpatched"
                     if meta.patch_class == "unpriced-reopen" else
                     "meta/m0-audit: REOPEN_CAPPED — reopen budget spent -> resuming the "
                     "segment unpatched")
                seg_green = True
                start_iter = len(load_records(ctx.status_path))
                continue
            if meta.milestone_violations:
                # a malformed audit patch never landed: restore and resume unpatched
                # rather than burning patch retries here — if the defect is real it
                # will resurface at the next actionable stop with a plateau behind it.
                _write_milestone(ctx, milestone_before)
                if ctx.meta_status_path:
                    update_last_outcome(ctx.meta_status_path, outcome="REVERT_PATCH")
                _log("meta/m0-audit: patch REVERT_PATCH (malformed, never landed) "
                     "-> resuming unpatched")
                seg_green = True
                start_iter = len(load_records(ctx.status_path))
                continue
            meta_steps += 1
            resume_ref, reopened = _resolve_resume_in_place(ctx, meta, records, baseline_ref)
            ctx.backend.reset(resume_ref)
            seg_green = _green_at_ref(records, resume_ref)
            ctx.fresh_recipe = None
            if reopened:
                # In place: the tree is the best green KEEP, nothing rolled
                # back — the REOPEN row + reopen_note describe the workspace;
                # a standing rollback note would contradict them.
                ctx.resume_note = ""
                _note_reopen(ctx, meta, resume_ref)
                audit_gate = _reopen_audit_gate(reopened)
            else:
                ctx.reopen_note = ""
                _set_resume_note(ctx, meta, resume_ref, baseline_ref)
            ctx.g5_note = ""
            if pending is None:
                pending = _PendingPatch(best_before, milestone_before,
                                        _best_good_ref_here(records, ctx)
                                        or baseline_ref,
                                        records_before=len(load_records(ctx.status_path)),
                                        cluster=_active_cluster(ctx),
                                        gate_contract=dict(meta.gate_contract),
                                        reopened=list(reopened))
            else:
                # the prior patch's segment never got to prove itself; the audit patch
                # joins its open transaction — one joint judgment at the next
                # actionable stop, one rollback point restoring the state before BOTH.
                _log("meta/m0-audit: patch joins the open milestone transaction")
                pending.gate_contract.update(meta.gate_contract)
                pending.reopened = sorted(set(pending.reopened) | set(reopened))
            start_iter = len(load_records(ctx.status_path))
            continue
        # judge the previous milestone patch now that its inner segment has run.
        # HOLD_PATCH keeps the transaction open: the segment had no metric reading,
        # so the patch stays landed (and on a budget stop, on disk for the next run)
        # and the next patch joins it for one joint judgment.
        if pending is not None:
            if _judge_patch(ctx, pending) != "HOLD_PATCH":
                pending = None
        # inner stop reasons the meta loop can act on. All three are milestone defects:
        # plateau = levers exhausted, stuck = student can't move, exhausted = the decision
        # graph ran out below target. The phase order ends at Mk (meta_milestone.md §3) —
        # the student has no free phase to fall back on, so "the milestone ran out" must
        # route to the agent that can extend it rather than end the run with the meta
        # budget unspent.
        if reason not in _META_ACTIONABLE:
            return {"stop": reason, "meta_steps": meta_steps}
        records = load_records(ctx.status_path)
        best_before = _best_ms(_on_cluster(records, ctx), ctx.gate_window)
        good_ref = _best_good_ref_here(records, ctx) or baseline_ref
        # Before spending the meta budget on a gap-driven diagnosis, make sure the
        # gap exists: stall pricing, the verdict probes and the prescription all
        # reason from the residual, and a residual inside the noise band may be
        # nothing but the launch-time anchor (see _phantom_gap_reanchor). This
        # runs BEFORE the meta-budget return: a run whose target is genuinely
        # met must never be recorded as "meta_budget" for want of budget to
        # check the residual with.
        reanchored = _phantom_gap_reanchor(ctx, best_before)
        if reanchored is not None:
            return {"stop": "target", "meta_steps": meta_steps,
                    "reanchor": reanchored}
        if meta_steps >= max_meta:
            return {"stop": "meta_budget", "meta_steps": meta_steps}
        # The note (if a re-anchor confirmed the gap) rides every diagnosis of
        # this meta entry — the malformed-patch retry below re-runs the diagnosis
        # and must see the same re-anchored residual, not the stale one.
        note, ctx.reanchor_note = ctx.reanchor_note, None
        while True:
            milestone_before = _read_milestone(ctx)
            meta = run_meta_step(ctx, best_before=best_before,
                                 milestone_before=milestone_before,
                                 reanchor_note=note)
            meta_steps += 1
            _apply_phase_close(ctx, meta)
            if meta.patch_class == "infeasible-stop":
                # the teacher's stop request ends the run here, budget unspent:
                # the stop is the run's verdict, the notice its record.
                return {"stop": "infeasible", "meta_steps": meta_steps,
                        "notice": str(Path(ctx.milestone).parent / "infeasible_stop.md")}
            if meta.patch_class in ("unpriced-reopen", "reopen-capped"):
                # nothing landed and nothing to judge: the refusal is now in the
                # reflection, so retry the META step (not the segment — resuming
                # the same plateaued state is predetermined), budget permitting
                if meta_steps >= max_meta:
                    return {"stop": "meta_budget", "meta_steps": meta_steps}
                _log("meta: UNPRICED_REOPEN — reopen refused for want of a measured "
                     "price -> retrying the meta step on the same state"
                     if meta.patch_class == "unpriced-reopen" else
                     "meta: REOPEN_CAPPED — reopen budget spent -> retrying the meta "
                     "step on the same state")
                continue
            if meta.patch_class == "infeasible-refused":
                # same shape: the refused stop request is in the reflection,
                # nothing landed — re-diagnose the same state, budget permitting
                if meta_steps >= max_meta:
                    return {"stop": "meta_budget", "meta_steps": meta_steps}
                _log("meta: INFEASIBLE_REFUSED — stop request refused by the reachability "
                     "check -> retrying the meta step on the same state")
                continue
            if meta.patch_class == "kb-example" or not meta.milestone_violations:
                break
            # a malformed patch that never became §9-conformant never lands: roll back —
            # and retry the META step, not the inner segment. The rollback restores
            # exactly the state whose segment already stalled, so re-running the student
            # on it is predetermined (a re-plateau or an instant EXHAUSTED): GPU evals
            # and agent turns for no information — the redundant turn the landing gate
            # alone could not save. The next diagnose sees this
            # REVERT_PATCH through the reflection and must propose differently.
            _write_milestone(ctx, milestone_before)
            if pending is None:
                ctx.backend.reset(good_ref)
                ctx.resume_note = ""  # restored tree: the never-landed patch invalidated nothing
            # else: an open held transaction means the workspace tip IS the pre-green
            # construction HOLD_PATCH preserved — and good_ref predates the whole chain
            # (a metric-less segment contributes no measured KEEP), so resetting here
            # would wipe exactly that work over a patch that never even landed. Restore
            # the milestone only (same call the m0-audit malformed branch makes) and
            # leave the workspace where the held segment left it.
            if ctx.meta_status_path:
                update_last_outcome(ctx.meta_status_path, outcome="REVERT_PATCH")
            _log("meta: patch REVERT_PATCH (malformed milestone, never landed) "
                 "-> retrying the meta step on the restored state")
            if meta_steps >= max_meta:
                return {"stop": "meta_budget", "meta_steps": meta_steps}
        resume_ref, reopened = _resolve_resume_in_place(ctx, meta, records, baseline_ref)
        ctx.backend.reset(resume_ref)
        seg_green = _green_at_ref(records, resume_ref)
        if reopened:
            ctx.resume_note = ""      # in place: nothing rolled back (see the audit path)
            _note_reopen(ctx, meta, resume_ref)
            audit_gate = _reopen_audit_gate(reopened)
        else:
            ctx.reopen_note = ""
            _set_resume_note(ctx, meta, resume_ref, baseline_ref)
        if meta.patch_class == "kb-example":
            # point the student at the just-certified recipe so the G3 loop actually closes
            # (a passive KB addition was ignored by the student in the live run).
            ctx.fresh_recipe = meta.recipe
            ctx.g5_note = meta.student_directive or ""
        else:  # a milestone patch is transactional; a KB addition is not
            ctx.fresh_recipe = None
            ctx.g5_note = ""
            if pending is None:
                pending = _PendingPatch(best_before, milestone_before, good_ref,
                                        records_before=len(load_records(ctx.status_path)),
                                        cluster=_active_cluster(ctx),
                                        gate_contract=dict(meta.gate_contract),
                                        reopened=list(reopened))
            else:
                # the prior patch was HELD (metric-less segment): this patch joins its
                # open transaction — one joint judgment, one rollback point restoring
                # the state before BOTH (mirrors the m0-audit join above).
                _log("meta: patch joins the open milestone transaction (prior patch held)")
                pending.gate_contract.update(meta.gate_contract)
                pending.reopened = sorted(set(pending.reopened) | set(reopened))
        start_iter = len(load_records(ctx.status_path))


# --- best confirmation (loop/best_confirmation.py) -------------------------
# Moved to loop/best_confirmation.py (docs/driver-split-spec.md tail leaf);
# imported back so every name it exposed from driver is still reachable at
# forgekernel.loop.driver.*
from forgekernel.loop.best_confirmation import (  # noqa: F401 -- re-export
    CONFIRM_NODE_BAND_PCT,
    _keep_reading_of,
    _job_node,
    _pct_off,
    _confirm_workload_medians,
    run_best_confirmation,
    apply_window_credit,
)
