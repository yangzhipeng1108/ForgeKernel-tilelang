"""Machine-aware comparisons — NCU/SASS/census comparisons and probe feedback.

Split out of loop/driver.py (docs/driver-split-spec.md step 5-comparisons). This
section compares the student's measurements against the reference across the ncu
(SASS precision, census, orchestrator profiling) and probe signals: env counts,
fast-skip reasons, host-bound checks, cluster scale, workload mismatches, and the
verdict/ablation probe feedback. Constants (tail sizes, bands, prof keys) live
here and move with their users.

Driver helpers still needed here are imported lazily (inside the function that
uses them) so no import cycle forms with driver.py:

    _ablation_block, _best_good_ref, _lever_spec_refused, _milestone_dag,
    _normalise_ablations, _orch_summary, _seed_pricing_toolkits

``BackendError`` is defined early in driver.py (before this section is imported
back), so it is imported at module load, like pricing.py does. ``LoopContext`` is
annotation-only (postponed) -> TYPE_CHECKING. The prompt/pricing helpers used here
(_best_keep_record, _is_cuda, _is_tilelang; _ABLATE_FAIL_RE, _lever_price_submit,
_price_log_*; parse_lever_prices) come straight from those extracted modules,
neither of which imports comparisons.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from forgekernel.baseline import workloads_mismatch
from forgekernel.inner.ncu import kernels_from_ncu, profiler_from_ncu
from forgekernel.inner.trajectory import gate_ms, gate_window_name, load_records
from forgekernel.loop.job_env import env_failure
from forgekernel.loop.parse import ProbeVerdict, parse_eval
from forgekernel.loop.pricing import (
    _ABLATE_FAIL_RE,
    _lever_price_submit,
    _price_log_lost_reason,
    _price_log_verdict,
    parse_lever_prices,
)
from forgekernel.loop.prompt import _best_keep_record, _is_cuda, _is_tilelang
from forgekernel.skills.sass_census import parse_census_line, parse_serial_line

if TYPE_CHECKING:
    from forgekernel.loop.driver import LoopContext

from forgekernel.loop.driver import BackendError  # defined early in driver; safe at load

__all__ = [
    "FAST_SKIP_NOISE_MULTIPLE",
    "_ANCHORS_FILENAME",
    "_CENSUS_LINE_RE",
    "_KERNELS_PROF_KEY",
    "_NCU_REGRESSION_BAND",
    "_ORCH_CENSUS_CMD",
    "_ORCH_CENSUS_RUNNER",
    "_ORCH_PROF_KEY",
    "_PROBE_LOG_TAIL_CHARS",
    "_PROBE_LOG_TAIL_LINES",
    "_REANCHOR_ATTEMPTS",
    "_SASS_PRECISION_PROF_KEY",
    "_SASS_PRECISION_RE",
    "_VERDICT_PROBE_TAIL_CHARS",
    "_VERDICT_PROBE_TAIL_LINES",
    "_ablation_probe_feedback",
    "_active_cluster",
    "_anchors_path",
    "_best_good_ref_here",
    "_best_keep_ncu_ns",
    "_best_keep_profiler",
    "_check_host_bound",
    "_check_profile_covers_the_kernel",
    "_cluster_scale",
    "_confirm_suspicious_watermark",
    "_edits_outside_notes",
    "_enforce_single_ref",
    "_env_counts",
    "_fast_skip_reason",
    "_home_ref_ms",
    "_keep_probe_across_reset",
    "_load_anchors",
    "_log",
    "_ncu_mode",
    "_ncu_profiler",
    "_note_discarded_edits",
    "_note_env_failure",
    "_note_env_ok",
    "_on_cluster",
    "_parse_serial_census",
    "_pop_kernels",
    "_pop_orch_summary",
    "_probe_feedback",
    "_probe_label",
    "_probe_log_suffix",
    "_rec_cluster",
    "_restore_kept_probes",
    "_restore_probe_after_reset",
    "_run_ablation_probe",
    "_say_target_suspended",
    "_target_workloads_mismatch",
    "_verdict_probe_feedback",
    "_without_census_lines",
]


def _ablation_block(*args, **kwargs):  # lazy shim -> driver._ablation_block
    from forgekernel.loop import driver

    return driver._ablation_block(*args, **kwargs)


def _call_ncu_profiler(*args, **kwargs):
    """Call _ncu_profiler through driver so a test monkeypatching
    driver._ncu_profiler takes effect on this module's internal callers."""
    from forgekernel.loop import driver

    return driver._ncu_profiler(*args, **kwargs)


def _best_good_ref(*args, **kwargs):  # lazy shim -> driver._best_good_ref
    from forgekernel.loop import driver

    return driver._best_good_ref(*args, **kwargs)


def _lever_spec_refused(*args, **kwargs):  # lazy shim -> driver._lever_spec_refused
    from forgekernel.loop import driver

    return driver._lever_spec_refused(*args, **kwargs)


def _milestone_dag(*args, **kwargs):  # lazy shim -> driver._milestone_dag
    from forgekernel.loop import driver

    return driver._milestone_dag(*args, **kwargs)


def _normalise_ablations(*args, **kwargs):  # lazy shim -> driver._normalise_ablations
    from forgekernel.loop import driver

    return driver._normalise_ablations(*args, **kwargs)


def _orch_summary(*args, **kwargs):  # lazy shim -> driver._orch_summary
    from forgekernel.loop import driver

    return driver._orch_summary(*args, **kwargs)


def _seed_pricing_toolkits(*args, **kwargs):  # lazy shim -> driver._seed_pricing_toolkits
    from forgekernel.loop import driver

    return driver._seed_pricing_toolkits(*args, **kwargs)


# --- machine-aware comparisons ----------------------------------------------
#
# A run resumed on another machine is a measurement problem: absolute latency is a
# property of (kernel, machine), and the loop's verdicts are all absolute-ms
# comparisons. So: every record carries the machine ("cluster") its eval ran on,
# and the comparisons below read same-machine rows only. NCU kernel time is deliberately NOT partitioned — it replays at
# locked base clocks, so it is the one reading that travels.


def _active_cluster(ctx: LoopContext) -> str:
    """The cluster submissions currently resolve to (the last one the backend
    submitted to), falling back to home. "" on backends that cannot say."""
    return getattr(ctx.backend, "last_cluster", "") or ctx.home_cluster


def _rec_cluster(rec, ctx: LoopContext) -> str:
    return getattr(rec, "cluster", "") or ctx.home_cluster


def _on_cluster(records, ctx: LoopContext, cluster: str | None = None) -> list:
    """The rows measured on ``cluster`` (default: the active one). On one machine
    every row and the active cluster resolve to the same string — possibly "" —
    and this is the identity."""
    c = _active_cluster(ctx) if cluster is None else cluster
    return [r for r in records if _rec_cluster(r, ctx) == c]


def _best_good_ref_here(records, ctx: LoopContext) -> str | None:
    """Rollback checkpoint: the best KEEP on the active cluster, else the best
    anywhere — a checkpoint from another cluster is still a verified kernel,
    just not a comparable reading."""
    return (_best_good_ref(_on_cluster(records, ctx), ctx.gate_window)
            or _best_good_ref(records, ctx.gate_window))


_ANCHORS_FILENAME = "cluster_anchors.json"
_REANCHOR_ATTEMPTS = 2


def _home_ref_ms(ctx: LoopContext) -> float | None:
    """The launch reading of the declared reference, in the gate window — the
    same channel the in-job readings and reanchor_ref measure, so a foreign
    cluster's fresh reading divides by a like-for-like number. (The target may
    derive from the architecture ref instead; the cluster speed ratio is what
    is being transported, and one reference kernel measures it as well as
    another.) Falls back to the architecture ref when the run has no library
    baseline."""
    ref = ctx.ref_baseline or ctx.arch_baseline
    if ref is None:
        return None
    # The PLAIN device reading, not device_anchor_ms: the anchor prefers a
    # CUDA-graph replay when one exists, and neither the in-job ref reading
    # nor measure_ref_fresh measures that window — dividing a plain-window
    # numerator by a graph-window denominator would loosen the scaled target
    # by the host tax, inside the noise band, and hand out stop=target for
    # a mediocre KEEP. Same field the phantom-gap anchor uses.
    return ref.ms_device if gate_window_name(ctx.gate_window) == "device" else ref.ms


def _anchors_path(ctx: LoopContext) -> Path:
    return Path(ctx.status_path).parent / _ANCHORS_FILENAME


def _load_anchors(ctx: LoopContext) -> dict:
    try:
        return json.loads(_anchors_path(ctx).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _target_workloads_mismatch(ctx: LoopContext, records=None) -> str | None:
    """Why this run's gate readings cannot be held against its absolute target.

    The target, the cluster scale and the phantom-gap transport divide a gate
    reading by an archived reference reading (architecture / library baseline).
    On a multi-workload card the gate reading is the geomean over the bench's
    workloads (timing.workloads_performance); an archive read over another input
    set is a different quantity, and the "gap" is then the workloads' cost mix,
    not the kernel, in either direction. The CLI refuses such a fresh start
    (_target_workloads_problems); this is the runtime half — resumed runs, and
    the workload NAMES only an eval reveals. None when comparable or before
    the first KEEP.
    """
    records = load_records(ctx.status_path) if records is None else records
    best = _best_keep_record(records, ctx.gate_window)
    if best is None:
        return None
    names = [str(w.get("name")) for w in best.workloads or [] if isinstance(w, dict)]
    for ref in (ctx.arch_baseline, ctx.ref_baseline):
        if ref is None:
            continue
        why = workloads_mismatch(getattr(ref, "n_workloads", 1), getattr(ref, "workloads", []),
                                 best.n_workloads, names)
        if why:
            return f"{getattr(ref, 'role', 'reference')} baseline: {why}"
    return None


def _say_target_suspended(ctx: LoopContext, where: str, why: str) -> None:
    """Log the absolute-target suspension once per distinct reason."""
    if getattr(ctx, "_target_wl_warned", None) == why:
        return
    ctx._target_wl_warned = why
    _log(f"{where}: absolute target suspended — {why}. No stop, re-anchor or residual "
         f"is read off it until the reference is re-archived as the geomean over the "
         f"bench's workloads; same-job ratios are unaffected")


def _cluster_scale(ctx: LoopContext) -> float | None:
    """What the target must be multiplied by on the active cluster; 1.0 at
    home; None when the reference cannot be anchored there (the target stop is
    then suspended on that cluster rather than judged against home's number).

    Three sources, cheapest first — every numerator is a MIN, because the
    launch reading it divides by is the fastest of its samples (baseline_cli
    keeps the fastest): a median numerator over a min denominator reads the
    median-vs-min offset (~2% on quiet nodes) as cluster slowness and loosens
    the target by exactly that, inside the noise band.
      1. same-job reference readings already recorded on this cluster (a run
         whose ref rides the job pays nothing: the ratio travels by
         construction, and the min firms up as rows accumulate);
      2. a fresh re-anchor persisted from earlier in this run
         (``cluster_anchors.json`` next to status.jsonl);
      3. one fresh measurement through ``ctx.reanchor_ref`` (the channel the
         phantom-gap re-anchor also uses; it follows the student's stay) —
         GPU jobs, so it runs at most _REANCHOR_ATTEMPTS times per cluster per
         run, under the same admission as the phantom-gap check (>= 2 usable
         samples, spread <= 10%, and the samples must have landed on THIS
         cluster). Min against min: the launch reading it is paired with is
         the fastest of its samples (baseline_cli keeps the fastest).
    """
    cluster = _active_cluster(ctx)
    if not cluster or cluster == ctx.home_cluster:
        return 1.0
    home = _home_ref_ms(ctx)
    if not home:
        return None
    key = "ref_device_ms" if gate_window_name(ctx.gate_window) == "device" else "ref_median_ms"
    seen = [getattr(r, key) for r in
            _on_cluster(load_records(ctx.status_path), ctx, cluster)
            if getattr(r, key, None)]
    if seen:
        return min(seen) / home
    anchors = _load_anchors(ctx)
    entry = anchors.get(cluster) or {}
    if entry.get("ref_ms"):
        return float(entry["ref_ms"]) / home
    attempts = int(entry.get("attempts", 0))
    if ctx.reanchor_ref is None or attempts >= _REANCHOR_ATTEMPTS:
        return None
    _log(f"cluster {cluster!r}: no reference anchor yet — re-measuring the "
         f"reference here (attempt {attempts + 1}/{_REANCHOR_ATTEMPTS})")
    fresh = None
    try:
        samples, ran_on = ctx.reanchor_ref()
        samples = [s for s in samples if s and s > 0]
        if len(samples) < 2:
            _log(f"cluster {cluster!r}: re-anchor admission failed — "
                 f"{len(samples)} usable sample(s), need >= 2")
        elif (max(samples) - min(samples)) / min(samples) > 0.10:
            _log(f"cluster {cluster!r}: re-anchor admission failed — spread "
                 f"{(max(samples) - min(samples)) / min(samples):.1%} > 10%")
        elif ran_on and ran_on != cluster:
            _log(f"cluster {cluster!r}: re-anchor samples ran on {ran_on!r} — "
                 f"the reference channel did not follow the stay; unusable here")
        else:
            fresh = min(samples)
    except Exception as e:  # a failed anchor must not take the loop down
        _log(f"cluster {cluster!r}: reference re-anchor failed ({e})")
    entry = {"attempts": attempts + 1, "home_ref_ms": home,
             "window": ctx.gate_window, "ref_ms": fresh}
    anchors[cluster] = entry
    try:
        _anchors_path(ctx).write_text(json.dumps(anchors, indent=1), encoding="utf-8")
    except OSError:
        pass
    if not fresh:
        _log(f"cluster {cluster!r}: reference re-anchor yielded no reading — the "
             f"target stop is suspended on this cluster")
        return None
    _log(f"cluster {cluster!r}: reference re-anchored at {fresh} ms "
         f"({ctx.gate_window} window) vs {home} ms at home — the target scales "
         f"x{fresh / home:.4f} while the run is here")
    return fresh / home


def _log(msg: str) -> None:
    print(f"[loop] {msg}", flush=True)


def _enforce_single_ref(ctx: LoopContext, ev):
    """Drop in-job reference readings on runs whose ref does not ride the job.

    On such runs (ref measured through a separate submission tree) any comparator
    the bench timed in-job is by construction NOT the ref — a stand-in library kept
    "to calibrate node speed". That concept is retired: it made every vs_* number
    ambiguous (one fwd run's same-job 1.0418x read as "beat FA" when the
    denominator was cuDNN). Dropped here, at the single entry point where eval logs
    become records, so no verdict, gate, digest, or report downstream ever sees a
    second reference. Loud once per run — a bench still timing a stand-in is a
    config leftover the task author should remove.
    """
    if ctx.ref_rides_job or ctx.ref_in_job or not any(
            v is not None for v in (ev.ref_median_ms, ev.vs_ref,
                                    ev.ref_device_ms, ev.vs_ref_device)):
        # ref_in_job: the in-job comparator IS the declared reference (the
        # image-baked library the ref tree also measures). Dropping it here
        # threw away the only node-matched reading the run had — a
        # grouped-GEMM run chased a +2.3% gap against anchors read on other nodes while
        # its own jobs read the student at parity with deep_gemm on the same
        # card, every iteration.
        return ev
    if not getattr(ctx, "_second_ref_warned", False):
        _log("WARNING single-ref contract: this run's ref is measured out-of-job, but "
             "the bench timed an in-job comparator — dropping those readings (remove "
             "the stand-in library from eval/bench.py)")
        ctx._second_ref_warned = True
    return replace(ev, ref_median_ms=None, vs_ref=None,
                   ref_device_ms=None, vs_ref_device=None)


# NCU replays with clocks locked to base frequency, so its per-kernel time is
# node-invariant where the bench reading is not. A new watermark whose NCU time
# REGRESSED past this band against the sitting best is a worse kernel on a fast
# node — the free cross-signal that convicts it sits in the same record.
_NCU_REGRESSION_BAND = 1.05

# Fast-skip band, as a multiple of the noise floor: a candidate slower than the
# sitting best by more than this may skip the profiling half of its eval.
#
# Borrowed from CUDA-L2, whose whole inner signal is one scalar (execution speed)
# and which therefore never spends a second GPU pass on a candidate that already
# lost. ForgeKernel cannot copy that — the three line gate (correctness /
# performance / mechanism) is exactly what stops "fast but wrong" and "fast on a
# lucky node" — so what is borrowed is the cheap case only: a candidate far
# outside the noise band in the WRONG direction cannot become the best, and its
# NCU table only matters to a diagnosis nobody will run on it. Wide on purpose
# (4x the noise floor, so ~6% at the default 1.5%): a candidate close enough that
# a mechanism reading might explain a later win still gets profiled.
FAST_SKIP_NOISE_MULTIPLE = 4.0


def _fast_skip_reason(ctx: LoopContext, ev, metric_before: float | None) -> str | None:
    """Why this eval may skip the rest of its GPU work, or None to proceed.

    Applies only to a KEEP-eligible candidate that is clearly SLOWER than the
    sitting best (never to a correctness fail, and never when there is no bar to
    be slower than). The number is the gate window's, so the comparison is the one
    the loop actually decides on.
    """
    if not getattr(ctx, "fast_skip", False) or metric_before is None:
        return None
    ms = gate_ms(ev, ctx.gate_window)
    if ms is None:
        return None
    noise = ctx.watermark_margin or 0.015        # the loop's own noise floor
    cutoff = metric_before * (1 + noise * FAST_SKIP_NOISE_MULTIPLE)
    if ms <= cutoff:
        return None
    return (f"{ms} ms is {ms / metric_before:.3f}x the sitting best {metric_before} ms — "
            f"past {FAST_SKIP_NOISE_MULTIPLE:g}x the {noise:.1%} noise floor, so it cannot "
            f"take the watermark; skipping the profile")


def _best_keep_ncu_ns(records, gate_window: str) -> float | None:
    """NCU kernel time of the sitting best KEEP, if it was profiled."""
    kept = [r for r in records
            if r.verdict == "KEEP" and gate_ms(r, gate_window) is not None]
    if not kept:
        return None
    t = min(kept, key=lambda r: gate_ms(r, gate_window)).profiler.get("time_ns")
    return float(t) if t else None


def _best_keep_profiler(records, gate_window: str) -> tuple[dict, int] | None:
    """The sitting best KEEP's NCU profiler dict + its iter, or None if IT lacks one.

    Strict on identity, like ``_best_keep_ncu_ns`` above: NCU is best-effort, so
    the best KEEP can land with ``profiler={}`` (retries exhausted). Falling back
    to an older profiled KEEP would reconcile a kernel that no longer exists —
    and could stamp OVERLAP-BOUND on it — so the answer is None, not a substitute.
    """
    kept = [r for r in records
            if r.verdict == "KEEP" and gate_ms(r, gate_window) is not None]
    if not kept:
        return None
    best = min(kept, key=lambda r: gate_ms(r, gate_window))
    return (best.profiler, best.iter) if best.profiler else None


def _confirm_suspicious_watermark(ctx: LoopContext, be, branch: str, ev, metric_before,
                                  profiler: dict | None = None):
    """Re-measure a KEEP whose reading is about to take the watermark on suspect
    evidence, BEFORE it does.

    One eval job is one node draw. An outlier that takes the watermark is not one
    bad number — it poisons every later meta transaction: rebuilds get attributed
    "no gain" against a best that was never real, and equal-or-better segments
    cannot accumulate. Within-noise improvements — the common case — trigger
    nothing and cost nothing.

    Two triggers:
    * the reading beats the best by more than the noise floor (a jump this size
      is exactly what one lucky node draw can fake), or
    * the reading takes the watermark while the record's own NCU kernel time
      REGRESSED past the sitting best's by more than _NCU_REGRESSION_BAND — a
      worse kernel on a fast node. This one costs nothing to detect and catches
      fakes whose device-reading jump stays under the margin.

    Protocol: one confirmation job; agreement within the margin confirms the
    jump, and the watermark takes the median of the two draws — not the first,
    faster one: keeping it let a reading 1.5% fast stand as the bar every later
    patch was priced against — except under the NCU trigger, where
    ms agreement alone proves nothing (see below) and the branch is re-profiled
    to clear the kernel clock too. Disagreement beyond it gets a third job and
    the median of the three becomes the gate-window reading (the other window's
    fields keep the first job's values — the watermark only reads the gate
    window). A failed FIRST confirmation keeps the original reading: a flaky
    queue must not veto a measured KEEP. But once a dissenting confirmation is
    in hand, that excuse is spent — a failed tie-break settles on the median of
    what was measured rather than re-instating the unconfirmed jump.
    """
    margin = ctx.watermark_margin
    ms = gate_ms(ev, ctx.gate_window)
    if not margin or metric_before is None or ms is None or ms >= metric_before:
        return ev
    why = None
    ncu_suspect_ns = None  # set iff the NCU-regression trigger raised the alarm
    if ms < metric_before * (1 - margin):
        why = (f"beats best {metric_before} by more than {margin:.0%}")
    else:
        new_t = (profiler or {}).get("time_ns")
        best_t = _best_keep_ncu_ns(load_records(ctx.status_path), ctx.gate_window)
        if new_t and best_t and float(new_t) > best_t * _NCU_REGRESSION_BAND:
            why = (f"takes the watermark while its NCU kernel time regressed "
                   f"{float(new_t) / best_t:.3f}x vs the sitting best "
                   f"({new_t} vs {best_t:.0f} ns) — worse kernel on a fast node?")
            ncu_suspect_ns = best_t
    if why is None:
        return ev
    _log(f"watermark: {ms} {why} — confirming before it takes the watermark")
    key = "device_median_ms" if gate_window_name(ctx.gate_window) == "device" else "bwd_median_ms"
    readings = [ms]

    def _confirmed():
        both = statistics.median(readings)
        if both != ms:
            _log(f"watermark: confirmed — recording median {both} of {readings}, "
                 f"not the first draw")
        return replace(ev, **{key: both})
    for _ in range(2):  # at most two extra jobs: confirm, then tie-break
        try:
            ev_n = _enforce_single_ref(ctx, parse_eval(be.submit_gpu(branch, "all")))
            ms_n = gate_ms(ev_n, ctx.gate_window) if ev_n.correctness else None
        except BackendError as e:
            ms_n, e_txt = None, str(e)
        else:
            e_txt = None
        if ms_n is None:
            what = f"failed ({e_txt})" if e_txt else "run unusable"
            if len(readings) > 1:
                # The dissenting confirmation already measured the jump as
                # unconfirmed; losing the tie-break job must not hand the
                # watermark back to the outlier.
                _log(f"watermark: tie-break {what} — settling on what was measured")
                break
            _log(f"watermark: confirmation {what} — keeping the single reading")
            return ev
        readings.append(ms_n)
        _log(f"watermark: confirmation job {getattr(be, 'last_job_id', '') or '?'} "
             f"read {ms_n}")
        if len(readings) == 2 and abs(ms_n - ms) <= margin * ms:
            if ncu_suspect_ns is None:
                return _confirmed()  # two independent draws agree — the jump is real
            # ms agreement cannot clear the NCU trigger: on a bimodal queue the
            # confirmation draws from the same fast-node pool the suspect did,
            # so band-consistent readings are near coin-flip likely even when
            # the kernel really regressed. The alarm came from the kernel
            # clock; only the kernel clock clears it.
            try:
                t_n = _call_ncu_profiler(ctx, branch, correctness=True).get("time_ns")
            except BackendError:
                t_n = None
            if t_n and float(t_n) <= ncu_suspect_ns * _NCU_REGRESSION_BAND:
                return _confirmed()  # kernel time recovered too — the jump is real
            _log(f"watermark: ms agrees but the re-profiled kernel time "
                 f"({t_n or 'unavailable'} ns) does not clear the regression vs "
                 f"{ncu_suspect_ns:.0f} ns — tie-breaking")
    final = statistics.median(readings)
    _log(f"watermark: recording median {final} of {readings} — the single-job jump "
         f"did not confirm")
    return replace(ev, **{key: final})


_PROBE_LOG_TAIL_LINES = 25
_PROBE_LOG_TAIL_CHARS = 2000


def _probe_label(pv: ProbeVerdict) -> str:
    """How a verdict reads in the loop log. NO VERDICT is its own state on purpose:
    it means the job never got far enough to judge anything."""
    if pv.passed:
        return "PASS"
    return "FAIL" if pv.found else "NO VERDICT"


def _probe_feedback(path: str, pv: ProbeVerdict, log: str, *, what: str = "probe",
                    on_pass: str = "") -> str:
    """One probe round's feedback, split by failure mode.

    Every non-PASS used to read as "PROBE FAIL (max_diff=None) — iterate the probe",
    which sent the agent looking for a numerical bug when the job had in fact failed
    to build, or when the driver had rejected its PASS. Each of those now says so.
    """
    # Every branch carries a bounded log tail. The FAIL branch used to return the
    # verdict counts alone, which silently discarded whatever diagnostic payload the
    # probe had printed (sanitizer reports, shape dumps, per-rung readings) in exactly
    # the case the agent needed it — a probe's stdout is its instrument, not noise.
    # The PASS branch then did the same for longer: a probe is told to put its
    # candidate sweep on the FACT channel and keep the verdict lines to contracts
    # (gpu_probe_ladder), so a PASS is the round with the MOST payload — and it came
    # back as one line. One fp4 probe swept six relaunch variants, named the one
    # that survives, PASSed, and the student re-derived that answer eight hours later.
    tail = "\n".join(log.strip().splitlines()[-_PROBE_LOG_TAIL_LINES:])
    tail = tail[-_PROBE_LOG_TAIL_CHARS:]
    if pv.passed:
        return (f"- Your {what} {path} returned PROBE PASS (max_diff={pv.diff_text}). "
                f"{on_pass} Tail of the job log (the FACTS / readings your probe "
                f"printed live here and are not re-sent — carry them into the "
                f"decision):\n{tail}\n")
    if not pv.found:
        return (f"- Your {what} {path} produced NO verdict: {pv.reason}. This is NOT a "
                f"numerical failure — do not go hunting for one. Make the probe run and "
                f"print exactly one PROBE PASS/FAIL line per contract it checks. Tail of "
                f"the job log:\n{tail}\n")
    # A multi-contract probe names the contracts that failed: "2 of 5 reported
    # FAIL" sent the agent re-reading all five when the log already said which two.
    table = f"Per-contract verdicts:\n{pv.contract_table()}\n" if pv.summary else ""
    if pv.reason:
        return (f"- Your {what} {path} did NOT certify — the driver rejected the verdict: "
                f"{pv.reason}. Fix this and re-emit PROBE_REQUEST. {table}Tail of the job "
                f"log:\n{tail}\n")
    return (f"- Your {what} {path} returned PROBE FAIL (max_diff={pv.diff_text}). "
            f"Iterate the probe. {table}Tail of the job log (your probe's own diagnostics "
            f"live here):\n{tail}\n")


def _probe_log_suffix(pv: ProbeVerdict) -> str:
    """What follows the verdict in a loop-log line: the per-contract tally when the
    probe printed several lines (it names the failing contracts and their readings),
    and the driver's reason when it is not already told by that tally.

    The tally replaces the reason only when a contract FAILed — the reason then is
    the generic "k of n lines reported FAIL". An all-PASS probe the audit rejected
    has a tally of "n/n PASS", which next to a FAIL label explains nothing: the
    audit's reason (ceiling, unparseable, missing max_diff) must stay on the line.
    """
    a_contract_failed = pv.n_pass < len(pv.contracts)
    parts = [pv.summary] if pv.summary else []
    if pv.reason and not (pv.summary and a_contract_failed):
        parts.append(pv.reason)
    return "".join(f" — {p}" for p in parts)


# A verdict probe's payload is its printed readings (tables, per-variant ratios), not
# the PASS bit — so the feedback carries a much longer log tail than the certification
# probes, whose only question is "did it certify".
_VERDICT_PROBE_TAIL_LINES = 150
_VERDICT_PROBE_TAIL_CHARS = 8000


def _run_ablation_probe(ctx: LoopContext, be, path: str, probe_branch: str):
    """Run a JSON ablation spec (``notes/<x>.json``: ``ablations`` only) through
    the frozen pricing recipe on the tree as it stands. (rows, why, log): the
    ablation rows of the table (decisions attributed), the reason there are
    none, the job log.

    Any ``levers`` in the file are ignored: candidates are priced at the
    design turn on the best tree, a probe answers a cause question on the
    current one. The spec is normalised in place (kernel_file, rounds,
    roles off) and committed with the toolkit, like a script probe."""
    spec_path = Path(ctx.workspace) / path
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [], f"ablation spec unreadable ({e})", ""
    if not isinstance(raw, dict):
        return [], "ablation spec is not a JSON object", ""
    known_ids = {d.id for d in _milestone_dag(ctx)}
    ablations = _normalise_ablations(ctx, raw.get("ablations") or [], known_ids)
    if not ablations:
        return [], "the spec names no `ablations`", ""
    spec = {"ablations": ablations, "levers": [], "roles": False,
            "ablation_rounds": int(raw.get("ablation_rounds") or 0) or None,
            "kernel_file": str(ctx.kernel_file)}
    if spec["ablation_rounds"] is None:
        del spec["ablation_rounds"]
    if fail := _lever_spec_refused(ctx, spec):
        return [], f"spec refused before the job — {fail}", ""
    try:
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
        _seed_pricing_toolkits(ctx.workspace, be)
        be.commit(f"meta-probe: {path}")
        be.push_branch(probe_branch)
        # a dirty node or a lost archive is bought once more (_lever_price_job); a group
        # card's probe runs on each of its meta workloads, merged
        log, _job = _lever_price_submit(ctx, be, probe_branch,
                                        f"cmd:python3 notes/lever_pricing_run.py {path}",
                                        f"meta: ablation probe {path}")
        rows, _stack, fail = parse_lever_prices(log)
    except BackendError as e:
        return [], f"job failed ({e})", ""
    if _price_log_verdict(log) is None:
        # a cut archive is not "no ABLATE line": the probe ran, its reading was
        # never read — the round says so instead of scoring the probe n/a
        return [], (f"the job log carries no PRICE/ABLATE verdict "
                    f"({_price_log_lost_reason(log)}) — the probe's reading was not read"), log
    rows = [r for r in rows if r.get("kind") == "ablation"]
    attribution = {a["name"]: a["decisions"] for a in ablations}
    for r in rows:
        r["decisions"] = list(attribution.get(r["name"], []))
        r["probe"] = path
    if fail:
        return rows, f"PRICE-FAIL {fail}", log
    if (afail := _ABLATE_FAIL_RE.search(log)):
        return rows, f"ABLATE-FAIL {afail.group(1).strip()}", log
    return rows, (None if rows else "no ABLATE line in the job log"), log


def _ablation_probe_feedback(path: str, rows: list, why: str | None, log: str,
                             round_no: int, budget: int) -> str:
    """One JSON ablation probe's feedback: the ablation table with its reading
    rules, or the reason there is none plus the log tail."""
    head = f"\n## Verdict experiment {round_no}/{budget}: `{path}` (ablation, driver-frozen recipe)\n"
    if rows:
        body = _ablation_block(rows)
        body += ("Citing it: write the ablation name + Δ in `evidence`; for R1 use "
                 "`reopen_price: <Δ> via ablation:<name>`"
                 f" (or `via probe:{path}#<name>`).\n")
        if why:
            body += f"(driver note: {why})\n"
        return head + body
    tail = "\n".join(log.strip().splitlines()[-_VERDICT_PROBE_TAIL_LINES:])[-_VERDICT_PROBE_TAIL_CHARS:]
    return (head + f"No ablation reading: {why}. This is not a numerical failure — fix the JSON "
            f"(every `ablations` item needs name / hypothesis / edits, its `old` must occur exactly "
            f"once in the kernel, and its name must not equal a lever's name) and send "
            f"PROBE_REQUEST again, or send a script probe instead. Job log tail:\n{tail}\n")


def _verdict_probe_feedback(path: str, pv: ProbeVerdict, log: str,
                            round_no: int, budget: int) -> str:
    """One diagnose-step verdict-probe round's feedback.

    Unlike a certification probe, FAIL here is not a defect in the probe — PASS/FAIL
    is the experiment's *answer* to a pre-registered hypothesis, and either answer is
    evidence. Only the no-verdict case is a protocol problem, and even then the log
    tail still flows back: a probe that crashed after printing half its table has
    still measured something.
    """
    tail = "\n".join(log.strip().splitlines()[-_VERDICT_PROBE_TAIL_LINES:])
    tail = tail[-_VERDICT_PROBE_TAIL_CHARS:]
    head = (f"\n## Verdict experiment {round_no}/{budget}: `{path}` — "
            f"PROBE {_probe_label(pv)} (max_diff={pv.diff_text})\n")
    if pv.summary:
        # Several pre-registered contracts: each is its own answer. The single
        # PASS/FAIL bit above is "all passed"; a control contract that could not
        # run must not read as the decisive hypothesis being falsified.
        head += (f"Per-contract verdicts ({pv.summary.split(';')[0]}; each row is one "
                 f"hypothesis's answer, readings recorded exactly as the script printed them):"
                 f"\n```\n{pv.contract_table()}\n```\n")
    if not pv.found:
        head += ("(The log has no `PROBE PASS/FAIL` verdict line — a verdict experiment must "
                 "pre-register a falsifiable criterion: print one line "
                 "`PROBE PASS/FAIL max_diff=<reading>` per hypothesis, so the script lands the "
                 "verdict in the log instead of you interpreting it afterwards. Fix this next "
                 "round.)\n")
    elif pv.reason:
        head += f"(the driver's audit of the verdict line: {pv.reason})\n"
    return head + f"\n### Job log tail (your experiment's readings)\n```\n{tail}\n```\n"


def _ncu_profiler(ctx: LoopContext, branch: str, *, correctness: bool) -> dict:
    """Submit the NCU job and parse the profiler, retrying when a PASSING kernel comes
    back empty (best-effort NCU failed on the GPU side). A correctness-broken kernel
    yields no NCU legitimately, so it is not retried. Neither is a launch target that
    cannot start on the image (loop/job_env.py): the retry would replay the same
    import error, and the empty digest is the job's blindness, not the kernel's."""
    mode = _ncu_mode(ctx)
    out = ctx.backend.submit_gpu(branch, mode)
    prof = profiler_from_ncu(_without_census_lines(out))
    tries = 0
    while (not prof and correctness and tries < ctx.ncu_retries
           and env_failure(out) is None):
        tries += 1
        _log(f"ncu empty on a passing kernel -> retry {tries}/{ctx.ncu_retries}")
        out = ctx.backend.submit_gpu(branch, mode)
        prof = profiler_from_ncu(_without_census_lines(out))
    if correctness:
        if prof:
            _note_env_ok(ctx, "ncu")
        elif (why := env_failure(out)) is not None:
            _note_env_failure(ctx, "ncu", why)
    # Operand-source census (opt-in per operator): when the op's ncu task ran
    # sass_census over the kernel dump, its fields join the digest so milestone
    # gate lines can anchor [VERIFY: sass-derived] the way they anchor bankconf_st.
    # Absent line -> absent fields; digests of census-less operators are unchanged.
    if prof:
        # The whole decomposition (every launch of the window, by share) rides
        # under a private key the iteration pops into IterationRecord.kernels —
        # the timed-window contract prices the rebuild launches off it.
        prof[_KERNELS_PROF_KEY] = kernels_from_ncu(_without_census_lines(out))
        prof.update(parse_census_line(out))
        # The compiler's wgmma verdict (orch_census_run prints WGMMA-SERIAL off
        # the same dump): waits per wgmma, plus whether the log carries ptxas's
        # own "serialized due to" advisory. Flat digest fields, so the NCU
        # thresholds flag ratio ~1 as the binding pathology of the round.
        prof.update(parse_serial_line(out))
        # The orchestration census (orch_census_run) rides the same log. Parked
        # under a private key the iteration pops into IterationRecord.orch_summary
        # — it must not join the flat profiler digest.
        if (summ := _orch_summary(out)) is not None:
            prof[_ORCH_PROF_KEY] = summ
        # The kernel's MMA operand / accumulator types (the precision gate's SASS
        # cross-check), printed even when the dump is too small for a dossier.
        if (m := _SASS_PRECISION_RE.search(out)) is not None:
            try:
                prof[_SASS_PRECISION_PROF_KEY] = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return prof


def _note_env_failure(ctx: LoopContext, channel: str, why: str) -> None:
    """Say — every time, with a running count — that a harness channel died on the
    job's environment rather than on the tree (loop/job_env.py). A fused-norm-rope
    run once went 2.5 days with its NCU, timeline and stall channels dead on two
    imports while every log line read like a transient or a tree failure."""
    counts = _env_counts(ctx)
    n = counts.get(channel, 0) + 1
    counts[channel] = n
    _log(f"ENV-BROKEN {channel} (x{n} in a row): {why} — this tool cannot start on the "
         f"job image, so the job is no reading about the tree and a retry replays it. "
         f"Fix the job environment: the card's gpu_entry.sh / eval/ (overlaid on every "
         f"job when listed in [loop] job_env_files) or the image")


def _note_env_ok(ctx: LoopContext, channel: str) -> None:
    if _env_counts(ctx).pop(channel, None):
        _log(f"ENV-OK {channel}: the channel reads again")


def _env_counts(ctx) -> dict:
    """ctx.env_broken, created on a context built without it (test doubles)."""
    counts = getattr(ctx, "env_broken", None)
    if counts is None:
        counts = {}
        try:
            ctx.env_broken = counts
        except AttributeError:
            pass
    return counts


def _edits_outside_notes(be, base: str) -> list:
    """Tracked paths outside notes/ whose content differs between ``base`` and the
    working tree — what a hard reset to ``base`` throws away that is not a probe
    script or a pricing spec. [] when the backend cannot say (test doubles) or git
    fails: this is a visibility aid, never a reason to stop a round."""
    lister = getattr(be, "changed_paths", None)
    if lister is None or not base:
        return []
    try:
        return sorted(p for p in lister(base) if p and not p.startswith("notes/"))
    except Exception:  # noqa: BLE001
        return []


def _note_discarded_edits(ctx: LoopContext, where: str, paths: list) -> None:
    """Log — and tell the next diagnosis once — that a reset is discarding edits
    outside notes/. The reset itself is right (the student line stays teacher-free);
    the silence was not: a teacher once fixed the job environment in a pricing
    turn, wrote itself a memory that it was fixed, and the fix was gone 21 minutes
    later without a line anywhere."""
    shown = ", ".join(paths[:8]) + (f" (+{len(paths) - 8} more)" if len(paths) > 8 else "")
    _log(f"{where}: discarding edits outside notes/ with the reset: {shown} — the student "
         f"line stays teacher-free, so these reach no job. A job-environment fix belongs in "
         f"the card and reaches every job through [loop] job_env_files")
    ctx.discarded_edits = (
        f"{where}: the reset discarded changes outside notes/: {shown}. The student lineage does "
        f"not take the teacher's changes, so these files reached **no** later job. If they are a "
        f"job-environment fix (gpu_entry.sh / eval/), make it in the card directory and list it "
        f"under [loop] job_env_files in loop.toml, so the driver overlays it on every job; if it "
        f"is a kernel change, write it into the milestone for the student to land.")


_ORCH_PROF_KEY = "_orch_summary"
_KERNELS_PROF_KEY = "_kernels"
_ORCH_CENSUS_CMD = "cmd:python3 notes/orch_census_run.py"
_ORCH_CENSUS_RUNNER = "orch_census_run.py"
# The census runner prints its lines AFTER the NCU CSV (the job log archive
# keeps only the tail). The CSV parser reads every line past the header as a
# row, and a JSON line has enough commas to land a phantom kernel in the digest.
_CENSUS_LINE_RE = re.compile(r"^(?:ORCH-CENSUS|ORCH-SUMMARY-JSON:|\[orch-census\]|WGMMA-SERIAL"
                             r"|SASS-PRECISION-JSON:).*$\n?", re.M)
_SASS_PRECISION_RE = re.compile(r"^SASS-PRECISION-JSON: (\{.*\})$", re.M)
_SASS_PRECISION_PROF_KEY = "_sass_precision"


def _without_census_lines(log: str) -> str:
    return _CENSUS_LINE_RE.sub("", log)


def _ncu_mode(ctx: LoopContext) -> str:
    """The per-iteration profile job: the op's plain ``ncu`` task, or the same
    task wrapped by the vendored orchestration census when the runner is in the
    workspace (seeded with the pricing toolkit) and the run has not opted out."""
    if getattr(ctx, "orch_census", False) and (
            Path(ctx.workspace) / "notes" / _ORCH_CENSUS_RUNNER).exists():
        if _is_cuda(ctx) or _is_tilelang(ctx):
            # no JIT dump for a .cu / a TileLang kernel file: the runner reads the
            # build of this kernel file rather than a CuTe DSL cubin directory
            return f"{_ORCH_CENSUS_CMD} --kernel {ctx.kernel_file}"
        return _ORCH_CENSUS_CMD
    return "ncu"


def _pop_orch_summary(prof: dict) -> dict:
    """Detach the census summary from a profiler dict (in place); {} when absent."""
    summ = prof.pop(_ORCH_PROF_KEY, None) if prof else None
    return summ if isinstance(summ, dict) else {}


def _pop_kernels(prof: dict) -> list:
    """Detach the per-kernel table from a profiler dict (in place); [] when absent."""
    rows = prof.pop(_KERNELS_PROF_KEY, None) if prof else None
    return rows if isinstance(rows, list) else []


def _parse_serial_census(prof: dict) -> dict:
    """The WGMMA serialization fields of a profiler dict (a view, not a pop).

    ``orch_census_run`` prints the ``WGMMA-SERIAL`` line off the same dump the
    orchestration census uses, so it rides the same job; ``_ncu_profiler`` folds
    the parsed fields into the flat digest, where ``wgmma_serial_ratio`` is read
    by the NCU thresholds and the trace columns like any other metric. This
    returns that subset for the student-facing verdict.
    """
    return {k: prof[k] for k in ("wgmma_census_kernel", "wgmma_total", "wgmma_depbar0",
                                 "wgmma_tight", "wgmma_serial_ratio",
                                 "wgmma_ptxas_serialized") if k in (prof or {})}


def _check_profile_covers_the_kernel(prof: dict, bwd_median_ms: float | None,
                                     ev=None) -> str | None:
    """Warn when profiled kernel time cannot account for the timed window.

    NCU reports whichever launches it captured. If the kernel under test never
    reached the CSV, what remains is some setup kernel — and nothing downstream can
    tell, because a profile of the wrong kernel looks exactly like a profile of a
    fast one. That happened for a whole 27-iteration run: the timed kernel went
    2871 ms -> 4 ms while the recorded profile sat at 117 us / 40 registers, and every
    NCU-anchored gate line was silently being compared against the preprocess pass.

    Two changes since, both aimed at the same false-positive:

    * the share is computed from ALL profiled kernels (``total_time_ns``), not the
      dominant one, so a legitimately decomposed operator no longer trips it;
    * a second, physically different cause of the same shortfall — the GPU sitting
      idle behind a host bottleneck — is separated out using the bench's enqueue
      clock, because "the profile is aimed wrong" and "the host is the pipeline"
      call for opposite fixes.

    Always a warning, never a stop: how many of the bench's calls the profiled
    launches correspond to is not recorded anywhere, so the share is a smell, not a
    measurement. 20% is deliberately loose.
    """
    from forgekernel.timing_audit import classify, ncu_reconciliation_warning
    verdict = classify(host_enqueue_ms=getattr(ev, "host_enqueue_ms", None),
                       device_median_ms=getattr(ev, "device_median_ms", None),
                       ratio=getattr(ev, "host_bound_ratio", None),
                       basis=getattr(ev, "device_window_basis", None))
    return ncu_reconciliation_warning(prof, bwd_median_ms, verdict) or None


def _check_host_bound(ctx: LoopContext, iter_no: int, ev) -> None:
    """Audit the device window's precondition on an evaluated round, both sides.

    The two sides need opposite handling, and conflating them is how a measurement
    defect becomes a lesson:

    * STUDENT host-bound is a real defect — per-call recompiles, python-side
      allocation, redundant launches, a synchronize inside the call. It inflates the
      student's own gate reading and the deployment pays it too, so it is logged AND
      handed to the student as the next round's diagnosis.
    * REFERENCE host-bound is a defect of the measurement. It is logged for the
      operator and deliberately NOT given to the student: "the reference's python
      wrapper is thick" reads to a kernel writer as "a thin python path wins", which
      teaches metric-gaming. The fix belongs to whoever archives the baseline —
      re-measure the reference through a host-free window.
    * Under a HOST-FREE basis (graph replay / CUPTI kernel span, the bench's
      device_window_basis) neither verdict applies: the reading cannot contain
      host time, so the enqueue clock is not divided into it. A thick student
      wrapper is then logged and handed over as a deployment NOTE — its own
      line, explicit that the gate is unaffected — instead of as a gate defect
      (a dsa_sparse card set host_bound_ratio to null for exactly this
      reason and the driver recomputed 3–6x from the raw timings anyway).
    """
    from forgekernel.timing_audit import (classify, host_path_feedback, host_path_note,
                                          student_feedback, student_warning)
    basis = getattr(ev, "device_window_basis", None)
    student = classify(host_enqueue_ms=getattr(ev, "host_enqueue_ms", None),
                       device_median_ms=getattr(ev, "device_median_ms", None),
                       ratio=getattr(ev, "host_bound_ratio", None), basis=basis)
    if student.host_free:
        ref_enqueue = getattr(ev, "ref_host_enqueue_ms", None)
        if note := host_path_note(ev.host_enqueue_ms, ev.device_median_ms, basis=basis,
                                  ref_host_enqueue_ms=ref_enqueue):
            _log(f"iter {iter_no}: NOTE {note}")
        ctx.host_bound_note = host_path_feedback(ev.host_enqueue_ms, ev.device_median_ms,
                                                basis=basis, ref_host_enqueue_ms=ref_enqueue)
    else:
        if warn := student_warning(student, device_ms=ev.device_median_ms):
            _log(f"iter {iter_no}: WARNING {warn}")
        ctx.host_bound_note = student_feedback(student, device_ms=ev.device_median_ms)

    ref = classify(ratio=getattr(ev, "ref_host_bound_ratio", None))
    if ref.host_bound:
        _log(f"iter {iter_no}: WARNING HOST-BOUND IN-JOB REFERENCE "
             f"(host_bound_ratio={ref.ratio:.2f}): the ref's device reading is its own "
             f"python dispatch cost, so vs_ref_device this round is not a "
             f"kernel-vs-kernel ratio. Re-measure the reference through a host-free "
             f"window (graph replay / NCU kernel time). NOT reported to the student: "
             f"a thick reference wrapper is a measurement fact, and feeding it to a "
             f"kernel writer teaches metric-gaming.")


def _keep_probe_across_reset(ctx: LoopContext, probe_path: str) -> str | None:
    """The probe file's text before a workspace reset wipes the iteration's
    commits — a fight that spans iterations (ProbeBattle) needs the probe to
    survive the reset that ends the iteration, or the next student starts the
    apparatus from scratch (observed: five rounds lost to rig debugging)."""
    try:
        return (ctx.workspace / probe_path).read_text()
    except OSError:
        return None


def _restore_probe_after_reset(ctx: LoopContext, probe_path: str, text: str) -> None:
    """Write the probe back after the reset AND remember it: a meta step between
    two student iterations commits its own verdict probe with `git add -A` and
    resets to the pre-probe head, which erases an untracked file — so the next
    student iteration re-writes it from ctx.kept_probes before it starts."""
    ctx.kept_probes[probe_path] = text
    dst = ctx.workspace / probe_path
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
    except OSError as e:
        _log(f"could not keep {probe_path} across the reset: {e}")


def _restore_kept_probes(ctx: LoopContext) -> None:
    """Re-materialise the kept probes a reset erased. A file that is present
    with OTHER text is the student's later edit (or its deliberate rewrite):
    it is left alone and the kept copy is dropped — the keep exists to survive
    a reset, never to overrule the student's tree."""
    for path, text in list(ctx.kept_probes.items()):
        dst = ctx.workspace / path
        if dst.is_file():
            try:
                same = dst.read_text() == text
            except OSError:
                same = False
            if not same:
                ctx.kept_probes.pop(path, None)
                _log(f"kept probe {path} differs from the workspace copy — keeping "
                     f"the workspace's, dropping the kept copy")
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text)
            _log(f"restored kept probe {path} into the workspace")
        except OSError as e:
            _log(f"could not restore kept probe {path}: {e}")


