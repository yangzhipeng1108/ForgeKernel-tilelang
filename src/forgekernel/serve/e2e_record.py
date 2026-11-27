"""End-to-end serving record: the result of driving a deployed model through a live server.

This is the *deployment-side* schema, deliberately separate from the optimisation loop's
``IterationRecord`` (which measures one kernel callable offline). Here a reading is what a
*served* model pays under a real request load -- tokens/second, time-to-first-token, per-user
throughput -- plus the three deploy gates that decide whether a faster raw kernel is actually
*deployable*: it must be CUDA-graph capturable, it must consume the server's paged KV layout,
and it must be numerically on par with the server's native kernel. A reading missing any gate
is not "optimal in deployment", however fast its kernel is offline.

The shape mirrors ``server_bench.BenchReading`` (medians + the window they came from) but adds
the serving numbers and the gates, so a single object answers "did this beat the baseline *in
deployment*" without the caller re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# The deployment gates. A reading is "deployable" only if every gate it was asked to check
# passed; a gate that was not checked is reported as None (unmeasured), never silently True --
# an unchecked gate must not read as "passed".
GATE_GRAPH = "cuda_graph"
GATE_PAGED = "paged_kv"
GATE_PARITY = "parity"
GATES = (GATE_GRAPH, GATE_PAGED, GATE_PARITY)

# A parity cosine at or above this is "the same kernel output" for a serving swap. 0.999 is
# the same threshold the kernel KB uses for a verified attention kernel (see the sm120
# attention entries), so a card that passed there passes here.
PARITY_MIN_COS = 0.999


@dataclass(frozen=True)
class ServingReading:
    """One served-model measurement at a fixed concurrency, plus the deploy gates.

    ``tok_s`` is the headline throughput (completion tokens per second over the run);
    ``ttft_s`` the median time-to-first-token; ``tps_per_user`` the per-request decode rate.
    ``concurrency`` is the client concurrency the reading was taken at -- a tok/s without its
    concurrency is comparable to nothing, so it is required, not defaulted."""

    concurrency: int
    tok_s: float | None
    ttft_s: float | None
    tps_per_user: float | None
    input_len: int | None = None
    output_len: int | None = None
    num_prompts: int | None = None
    backend: str | None = None          # the server backend this reading was taken against
    gates: dict[str, bool | None] = field(default_factory=dict)  # GATES -> True/False/None
    # The two run facts that swing TTFT by tens of times (RTX 6000D: prefix cache moves a 512K
    # TTFT from 239s to 4.4s, 54x), so a reading without them is not comparable to one that had
    # the other setting. ``None`` = not recorded (the comparison then cannot make the claim).
    prefix_cache: bool | None = None    # was prefix/KV-prefix caching enabled for this run
    context_len: int | None = None      # the prompt length the run was measured at

    def gate(self, name: str) -> bool | None:
        return self.gates.get(name)

    @property
    def all_checked_gates_pass(self) -> bool:
        """True iff every gate that WAS checked passed (unchecked gates do not fail it)."""
        checked = [v for v in self.gates.values() if v is not None]
        return all(checked) if checked else True

    @property
    def blocked_gates(self) -> list[str]:
        """The names of gates that were checked and FAILED -- why a fast kernel is not deployable."""
        return [name for name in GATES if self.gates.get(name) is False]

    @property
    def deployable(self) -> bool:
        """All three gates checked-and-passed. An unchecked gate makes this False (we cannot
        claim deployable without having tested it)."""
        return all(self.gates.get(name) is True for name in GATES)


def _parse_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def serving_reading_from_bench(args, summary: dict, *, backend: str | None = None,
                               gates: dict | None = None, prefix_cache: bool | None = None,
                               context_len: int | None = None) -> ServingReading:
    """Build a :class:`ServingReading` from a ``bench_serving`` run's args and JSON summary.

    ``args`` is what we ASKED for (concurrency, input/output length, prompt count) -- those are
    facts about the run, and the summary does not echo them, so they come from our side.
    ``summary`` is the server's own result JSON (``sglang.bench_serving`` prints one); its key
    names are the upstream contract, read defensively so a version bump that renames a field
    degrades to None rather than crashing the harness. ``gates`` is the deploy-gate verdicts
    from :mod:`forgekernel.serve.checks`, threaded through unchanged. ``prefix_cache`` is the
    server's prefix-caching state (a TTFT-affecting fact the summary never carries); ``context_len``
    defaults to the input length this run was measured at."""
    input_len = getattr(args, "random_input_len", None)
    return ServingReading(
        concurrency=int(getattr(args, "max_concurrency", 0) or 0),
        tok_s=_parse_float(summary.get("output_throughput")),
        ttft_s=_parse_float(summary.get("median_ttft_ms", summary.get("mean_ttft_ms"))),
        tps_per_user=_parse_float(summary.get("median_tps", summary.get("mean_tps"))),
        input_len=input_len,
        output_len=getattr(args, "random_output_len", None),
        num_prompts=getattr(args, "num_prompts", None),
        backend=backend,
        gates=dict(gates or {}),
        prefix_cache=prefix_cache,
        context_len=context_len if context_len is not None else input_len,
    )


def compare_to_baseline(candidate: ServingReading, baseline: ServingReading) -> dict:
    """The deployment verdict: how the candidate's tok/s and TTFT compare at the same load.

    A raw ratio is honest only at matched concurrency (tok/s at C=16 against tok/s at C=64 is a
    queueing artefact, not a win), so a mismatch is reported as a reason, not a number. The
    verdict also carries the blocked gates, because a 2x tok/s on a kernel that cannot pass
    CUDA graph capture is not deployable and must not be reported as a win."""
    out: dict = {"blocked_gates": candidate.blocked_gates,
                 "deployable": candidate.deployable}
    if candidate.concurrency != baseline.concurrency:
        out["reason"] = (f"concurrency mismatch: candidate C={candidate.concurrency} vs "
                         f"baseline C={baseline.concurrency}; a tok/s ratio across "
                         f"concurrencies is a queueing artefact, not a win")
        return out
    # Prefix caching and context length swing TTFT by tens of times (RTX 6000D: a 512K TTFT of
    # 239s becomes 4.4s with the cache on, 54x), so a tok/s/TTFT ratio across a differing
    # prefix-cache state or a differing context length compares two different workloads, not a
    # faster kernel. Reject like the concurrency mismatch instead of reporting a bogus win.
    if (candidate.prefix_cache is not None and baseline.prefix_cache is not None
            and candidate.prefix_cache != baseline.prefix_cache):
        out["reason"] = (f"prefix-cache mismatch: candidate "
                         f"{'on' if candidate.prefix_cache else 'off'} vs baseline "
                         f"{'on' if baseline.prefix_cache else 'off'}; TTFT/throughput across a "
                         f"differing prefix-cache state is not comparable")
        return out
    if (candidate.context_len is not None and baseline.context_len is not None
            and candidate.context_len != baseline.context_len):
        out["reason"] = (f"context-length mismatch: candidate {candidate.context_len} vs "
                         f"baseline {baseline.context_len}; a ratio across differing prompt "
                         f"lengths measures different workloads, not a faster kernel")
        return out
    if candidate.tok_s and baseline.tok_s:
        out["tok_s_ratio"] = candidate.tok_s / baseline.tok_s
    if candidate.ttft_s and baseline.ttft_s:
        out["ttft_ratio"] = candidate.ttft_s / baseline.ttft_s
    return out
