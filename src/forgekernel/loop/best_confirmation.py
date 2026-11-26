"""Best confirmation — confirming a new best against a suspicious KEEP reading.

Split out of loop/driver.py (docs/driver-split-spec.md, tail leaf). When the inner
loop reports a new best, run_best_confirmation re-measures it and compares the
sample against the KEEP reading of the same sha across nodes/windows; a sample too
far from that reading flags the KEEP as disputed (the machine, not the kernel, is
the suspect). apply_window_credit reconciles the NCU table with the timed window.

A clean leaf: the only driver dependency is BackendError (defined early in driver,
module-load import); LoopContext is annotation-only -> TYPE_CHECKING. Everything
else comes from already-extracted modules (comparisons, prompt) plus trajectory/
parse/stdlib.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from forgekernel.inner.trajectory import gate_ms, gate_window_name, load_records
from forgekernel.loop.comparisons import (
    _best_good_ref_here,
    _enforce_single_ref,
    _log,
)
from forgekernel.loop.parse import describe_workload_readings, parse_eval
from forgekernel.loop.prompt import _best_keep_record, _ref_for_window
from forgekernel.loop.driver import BackendError  # defined early in driver; safe at load

if TYPE_CHECKING:
    from forgekernel.loop.driver import LoopContext

__all__ = [
    "CONFIRM_NODE_BAND_PCT",
    "_keep_reading_of",
    "_job_node",
    "_pct_off",
    "_confirm_workload_medians",
    "run_best_confirmation",
    "apply_window_credit",
]


# A confirmation sample this far from the KEEP reading of the SAME sha is not
# noise: nodes differ by ~±6% absolute and a same-job ratio carries a 1-5% floor,
# so two honest draws of one commit sit within ~12% of each other at the very
# worst. Beyond that the MACHINE is the suspect (live case: a probe's orphaned
# launch target held the GPU, three confirmation jobs read 23.3 ms against a KEEP
# of 10.0 ms). The sample is reported with its deviation and the confirmation
# flags the KEEP reading as disputed; a human decides which to trust.
CONFIRM_NODE_BAND_PCT = 15.0


def _keep_reading_of(records, sha: str, window: str) -> float | None:
    """The gate-window reading the trajectory holds for ``sha`` (its latest
    KEEP row) — the number a confirmation sample is checked against."""
    for rec in reversed(list(records)):
        if rec.verdict != "KEEP" or not rec.sha or not sha:
            continue
        if not (rec.sha.startswith(sha) or sha.startswith(rec.sha)):
            continue
        ms = rec.device_median_ms if window == "device" else rec.bwd_median_ms
        if ms:
            return float(ms)
    return None


def _job_node(be, job: str) -> str:
    """Where a confirmation sample ran: the backend's machine when it names one (every
    job on one machine is the SAME node draw), else the job itself."""
    machine = getattr(be, "machine", "")
    if isinstance(machine, str) and machine:
        return machine
    return f"job:{job}" if job and job != "?" else ""


def _pct_off(ms: float | None, keep_ms: float | None) -> float | None:
    if not ms or not keep_ms:
        return None
    return 100.0 * (ms / keep_ms - 1.0)


def _confirm_workload_medians(samples: list[dict], window: str) -> list[dict]:
    """Per-workload medians across confirmation samples (multi-workload cards).

    Workloads are matched by name, in the order the first sample lists them;
    a workload some sample did not report is summarized over the ones that did.
    [] when no sample carried per-workload readings.
    """
    ms_key, ratio_key = (("device_median_ms", "vs_ref_device") if window == "device"
                         else ("bwd_median_ms", "vs_ref"))
    order: list[str] = []
    by: dict[str, list[dict]] = {}
    for s in samples:
        for w in s.get("workloads") or ():
            name = w.get("name")
            if name not in by:
                order.append(name)
                by[name] = []
            by[name].append(w)
    out = []
    for name in order:
        rows = by[name]
        item = {"name": name, "n": len(rows)}
        mss = [r[ms_key] for r in rows if isinstance(r.get(ms_key), (int, float))]
        rts = [r[ratio_key] for r in rows if isinstance(r.get(ratio_key), (int, float))]
        if mss:
            item[ms_key] = statistics.median(mss)
        if rts:
            item[ratio_key] = statistics.median(rts)
        out.append(item)
    return out


def run_best_confirmation(ctx: LoopContext, jobs: int = 3, max_extra: int = 2,
                          ref: str | None = None,
                          seed_samples: list | None = None) -> dict | None:
    """Post-stop confirmation: re-measure the best KEEP sha ``jobs`` times.

    One eval job is one node draw — the cluster's H100 nodes are bimodal (~±6%
    absolute) and even same-job ratios carry a 1-5% noise floor, so a single draw
    cannot separate a real win from a lucky node (live case: one fwd run's
    single-job 0.2585 ms best re-measured at 0.282 ms, three jobs, spread 0.7%).
    Per-iteration verdicts stay single-job for throughput; only the run's final
    claim pays for repetition. Runs after the loop stops (any reason) so it never
    competes with iterations for wall-clock budget. Off per task via loop.toml
    [loop] confirm_jobs = 0.

    When the ref rides the job the claim is the median same-job ratio. When it does
    not (blackbox runs — single-ref contract, no stand-in library in the job), the
    jobs yield an absolute median only; the caller re-anchors the ref through its
    own channel over the same period and forms the ratio median-vs-median.
    """
    if jobs <= 0:
        return None
    # ``ref``: the sha to re-measure; default the standing best. The same-job
    # parity stop names its own sha — the row that read parity need not be the
    # absolute best (a slow-node draw can be at parity while a fast-node draw
    # holds the watermark).
    records = load_records(ctx.status_path)
    ref = ref or _best_good_ref_here(records, ctx)
    if not ref:
        _log("confirm: no green KEEP iteration on record — nothing to confirm")
        return None
    keep_ms = _keep_reading_of(records, ref, ctx.gate_window)
    be = ctx.backend
    push_ref = getattr(be, "push_ref", None)
    if push_ref is None:
        _log("confirm: backend cannot push refs — skipping (HEAD may not be the best sha)")
        return None
    branch = f"{ctx.branch_prefix}-confirm"
    push_ref(ref, branch)
    # ``seed_samples``: readings of THIS sha already bought (the same-job parity
    # stop's re-measure) count toward ``jobs`` instead of being bought again.
    samples: list[dict] = [dict(s) for s in (seed_samples or []) if s.get("ms")]
    if samples:
        _log(f"confirm: {len(samples)} same-period sample(s) of {ref[:10]} already "
             f"in hand — buying {max(0, jobs - len(samples))} more")
    consecutive_errors = 0
    attempts = 0
    while len(samples) < jobs and attempts < jobs + max_extra:
        attempts += 1
        try:
            log = be.submit_gpu(branch, "all")
        except BackendError as e:
            consecutive_errors += 1
            _log(f"confirm: job failed ({e})")
            if consecutive_errors >= 2:  # infra is down, not unlucky — stop burning jobs
                break
            continue
        consecutive_errors = 0
        ev = _enforce_single_ref(ctx, parse_eval(log))
        job = getattr(be, "last_job_id", "") or "?"
        if gate_window_name(ctx.gate_window) == "device":
            ms, ref_ms, ratio = ev.device_median_ms, ev.ref_device_ms, ev.vs_ref_device
        else:
            ms, ref_ms, ratio = ev.bwd_median_ms, ev.ref_median_ms, ev.vs_ref
        if ratio is None and ms and ref_ms:
            ratio = ref_ms / ms
        if not ev.correctness or ms is None:
            # a correctness break on the best sha is a red flag, not just a bad sample.
            # A missing RATIO is deliberately not a rejection: a run whose bench
            # carries no in-job ref (misjudged ref_rides_job, target-ms-only) would
            # otherwise reject every sample and burn the whole job budget for a None —
            # an absolute median is still the claim repetition exists to firm up.
            why = ev.error or f"no {ctx.gate_window}-window reading in the log"
            _log(f"confirm: job {job} unusable ({why})")
            continue
        sample = {"job": job, "ms": ms, "ref_ms": ref_ms, "ratio": ratio,
                  "node": _job_node(be, job)}
        if ev.workloads:
            sample["workloads"] = ev.workloads
        off = _pct_off(ms, keep_ms)
        if off is not None and abs(off) > CONFIRM_NODE_BAND_PCT:
            sample["off_keep_pct"] = round(off, 2)
            _log(f"confirm: job {job} {ctx.gate_window}={ms} is {off:+.1f}% off the KEEP "
                 f"reading {keep_ms} of {ref[:10]} — outside the ±{CONFIRM_NODE_BAND_PCT:.0f}% "
                 f"node band: the reading stands, and the machine ({sample['node'] or '?'}) "
                 f"is worth a look (another process on the GPU, clocks)")
        samples.append(sample)
        _log(f"confirm: job {job} {ctx.gate_window}={ms}"
             + (f" ref={ref_ms} ratio={ratio:.4f}" if ratio is not None else ""))
    if not samples:
        _log("confirm: no usable sample — the claim stays single-job; check the jobs above")
        return None
    med_ms = statistics.median(s["ms"] for s in samples)
    ratios = [s["ratio"] for s in samples if s["ratio"] is not None]
    short = " (SHORT of requested — treat as provisional)" if len(samples) < jobs else ""
    out = {"sha": ref, "window": ctx.gate_window, "requested": jobs,
           "jobs": samples, "median_ms": med_ms, "keep_ms": keep_ms,
           "node_band_pct": CONFIRM_NODE_BAND_PCT}
    med_off = _pct_off(med_ms, keep_ms)
    out["keep_disputed"] = bool(med_off is not None and abs(med_off) > CONFIRM_NODE_BAND_PCT)
    if out["keep_disputed"]:
        _log(f"confirm: median {med_ms} ms is {med_off:+.1f}% off the KEEP reading {keep_ms} "
             f"of {ref[:10]} — the confirmation DISPUTES the trajectory's reading; the "
             f"number below is what fresh jobs measured, the KEEP row is the one to doubt")
    nodes = sorted({s["node"] for s in samples if s.get("node")})
    if len(samples) > 1 and len(nodes) == 1:
        _log(f"confirm: all {len(samples)} sample(s) ran on one machine ({nodes[0]}) — that "
             f"is ONE node draw, not {len(samples)}; their spread says nothing about "
             f"node-to-node variation")
    out["nodes"] = nodes
    if per_wl := _confirm_workload_medians(samples, ctx.gate_window):
        # the headline ratio is the geomean over the card's workloads; keep each
        # workload's own median beside it, so a win carried by one workload and a
        # loss on another stays visible in the record
        out["workloads"] = per_wl
        _log("confirm: per-workload medians: " + describe_workload_readings(per_wl))
    if ratios:
        out["median_ratio"] = statistics.median(ratios)
        _log(f"confirm: median ratio {out['median_ratio']:.4f} over {len(ratios)} job(s) "
             f"@ {ref[:10]} ({ctx.gate_window} window){short}")
    else:
        if ctx.ref_rides_job:
            _log("confirm: no job carried an in-job ref reading (ref_rides_job "
                 "misconfig or bench defect) — falling back to the absolute median")
        _log(f"confirm: median {med_ms} ms over {len(samples)} job(s) @ {ref[:10]} "
             f"({ctx.gate_window} window; ref measured separately){short}")
    return out


def apply_window_credit(ctx: LoopContext, confirm: dict) -> dict | None:
    """Price the declared contract tax into the confirmation record.

    Reads the confirmed sha's banked per-kernel table (falling back to the
    sitting best KEEP's) and the reference's, the contract's ``price`` regexes
    on each side, and the ref median the confirmation settled on (same-job or
    median-vs-median), and writes ``confirm["window_credit"]``: per-side matched
    shares, the credited readings and the credited ratio next to the raw one.
    The raw ``median_ratio`` is untouched — the target stop, the KEEP ledger and
    every cross-run ranking keep reading it; the credit says how much of the
    residual no kernel-quality lever can remove.

    Before any share is applied the table is reconciled with the timed window
    it is meant to describe (``timing_audit.ncu_coverage``, the mis-aimed-profile
    arithmetic): a card whose ncu task does not confine the profiler to the
    timed call carries the bench's setup launches in its table, and a share read
    off that table is diluted (or a same-named prep launch inflates it). Such a
    credit, and one whose regex matched nothing, is recorded as **unusable**
    with its reason — never as a 0-share credited ratio. Returns the credit
    dict, or None when nothing was priced (the log says why either way).
    """
    from forgekernel.window_contract import credited_ratio, window_credit
    contract = ctx.window_contract
    if contract is None or not contract.declared:
        return None
    if not contract.priced_entries:
        if contract.unpriced_entries:
            _log("confirm: window contract has price-class asymmetries without a "
                 "kernel regex — credited ratio not computed "
                 f"({', '.join(a.name for a in contract.unpriced_entries)})")
        return None
    records = load_records(ctx.status_path)
    sha = confirm.get("sha") or ""
    rec = next((r for r in records if sha and r.sha == sha and r.kernels), None)
    if rec is None:
        best = _best_keep_record(records, ctx.gate_window)
        rec = best if best is not None and best.kernels else None
    ref = _ref_for_window(ctx)
    ref_ms = confirm.get("ref_median_ms")
    if ref_ms is None:
        refs = [s.get("ref_ms") for s in confirm.get("jobs", []) if s.get("ref_ms")]
        ref_ms = statistics.median(refs) if refs else None
    credit = window_credit(
        contract, rec.kernels if rec else None, ref.kernels if ref else None,
        student_profiler=rec.profiler if rec else None,
        # the KEEP row's own reading: the table was profiled on that tree
        student_ms=gate_ms(rec, ctx.gate_window) if rec else None,
        ref_profiler=ref.ncu if ref else None,
        ref_ms=ref.device_anchor_ms if ref else None)
    if not credit:
        return None
    out = {**credit, "source_iter": rec.iter if rec else None}
    if not credit["usable"]:
        _log("confirm: window credit UNUSABLE — " + "; ".join(credit["why"])
             + ". No credited ratio is reported; the raw ratio stands alone.")
        confirm["window_credit"] = out
        return out
    cr = credited_ratio(credit, confirm.get("median_ms"), ref_ms)
    if cr:
        out.update(cr)
        _log(f"confirm: window credit — student side {cr['student_fraction'] * 100:.1f}% "
             f"({', '.join(credit['student']['priced']) or 'none'}), reference side "
             f"{cr['reference_fraction'] * 100:.1f}% "
             f"({', '.join(credit['reference']['priced']) or 'none'}); "
             f"raw ratio {cr['raw_ratio']:.4f} → credited {cr['credited_ratio']:.4f} "
             f"(student {confirm.get('median_ms')} → {cr['credited_student_ms']:.4f} ms, "
             f"ref {ref_ms} → {cr['credited_ref_ms']:.4f} ms). "
             f"The raw ratio remains the verdict.")
    else:
        _log("confirm: window credit — shares priced, but no ref median to form a "
             "credited ratio against")
    if credit.get("reconciliation"):
        _log("confirm: window credit — NCU table vs timed window: "
             + ", ".join(f"{k} {v:.2f}" for k, v in credit["reconciliation"].items()))
    confirm["window_credit"] = out
    return out
