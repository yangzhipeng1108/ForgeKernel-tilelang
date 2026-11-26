"""Server-mode bench collector (A2): drive device + server windows, assemble one reading.

The training loop's gate reads the *device* window (kernel quality, no offered load). A served
deployment pays the *server* window (service + queueing behind a fixed request rate). A kernel
can win one and lose the other, so a run that never collects the server reading cannot see a
deployment regression. This module is the collector that produces both readings for one kernel
so they land in a single trajectory row, and so ``timing_audit.queue_estimate_of`` can split the
served latency into service vs wait.

The GPU is kept behind two injected callables (``device_fn`` / ``server_fn``) exactly like
``search_loop.evolve`` keeps evaluation behind ``evaluate``: in production they time the real
kernel (``card_template.eval.timing.time_device`` / ``time_server``), and in tests they are plain
functions returning synthetic samples. So the whole collection + aggregation + record-assembly
path is deterministic and CPU-testable; only the two timers need a GPU.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchReading:
    """One kernel's device + server readings, ready to fill a trajectory row and feed F4.

    ``device_median_ms`` is the back-to-back service time (no offered load);
    ``server_median_ms`` is the served latency (service + queueing) at ``server_qps``. Either is
    None when its window was not collected — an unmeasured window is never a measured zero.
    ``vs_ref_device`` / ``vs_ref_server`` are ref/student ratios (same definition as the gate's
    ``vs_ref``: reference median over student median, >1 means the student is faster)."""

    device_median_ms: float | None
    device_samples_ms: int
    server_median_ms: float | None
    server_samples_ms: int
    server_qps: float | None
    ref_device_ms: float | None = None
    ref_server_ms: float | None = None

    @property
    def vs_ref_device(self) -> float | None:
        return _ratio(self.ref_device_ms, self.device_median_ms)

    @property
    def vs_ref_server(self) -> float | None:
        return _ratio(self.ref_server_ms, self.server_median_ms)

    def record_fields(self) -> dict:
        """The subset of an IterationRecord this reading fills (only measured keys).

        A window that was not collected contributes NO key, so the row reads as "not measured"
        rather than "zero" — the same discipline the card template enforces on its eval JSON."""
        out: dict = {}
        if self.device_median_ms is not None:
            out["device_median_ms"] = self.device_median_ms
        if self.server_median_ms is not None:
            out["server_median_ms"] = self.server_median_ms
            if self.server_qps is not None:
                out["server_qps"] = self.server_qps
        if self.ref_device_ms is not None:
            out["ref_device_ms"] = self.ref_device_ms
        if self.ref_server_ms is not None:
            out["ref_server_ms"] = self.ref_server_ms
        if self.vs_ref_device is not None:
            out["vs_ref_device"] = self.vs_ref_device
        if self.vs_ref_server is not None:
            out["vs_ref_server"] = self.vs_ref_server
        return out


def _ratio(ref_ms, student_ms) -> float | None:
    if not ref_ms or not student_ms:
        return None
    return ref_ms / student_ms


def _median(samples) -> tuple[float | None, int]:
    """(median, count) of a sample list; (None, 0) when empty — never a fabricated zero."""
    vals = [float(s) for s in samples if s is not None]
    if not vals:
        return None, 0
    return statistics.median(vals), len(vals)


def collect(*, device_fn=None, server_fn=None, ref_device_fn=None, ref_server_fn=None,
            qps: float | None = None) -> BenchReading:
    """Collect one kernel's device + server windows and assemble a :class:`BenchReading`.

    ``device_fn()`` / ``server_fn()`` return the student kernel's per-call samples (ms) in each
    window; the ``ref_*`` variants do the same for the in-job reference. Any callable left None
    skips that window (and its reading is None). ``server_fn`` is ignored when ``qps`` is None —
    a request-driven window with no offered rate cannot be read, so it is not run rather than
    run at a fabricated rate.

    The aggregation is medians (matching the gate: "the median is the reading"), and the student
    and reference are measured in the *same* order so a thermal or clock drift that affects both
    windows is folded into the ratio rather than the absolute numbers."""
    dev_med, dev_n = (None, 0)
    if device_fn is not None:
        dev_med, dev_n = _median(device_fn())
    ref_dev_med = None
    if ref_device_fn is not None:
        ref_dev_med, _ = _median(ref_device_fn())

    srv_med, srv_n = (None, 0)
    rate = qps
    if server_fn is not None and qps is not None and qps > 0:
        srv_med, srv_n = _median(server_fn())
    else:
        rate = None

    ref_srv_med = None
    if ref_server_fn is not None and rate is not None:
        ref_srv_med, _ = _median(ref_server_fn())

    return BenchReading(
        device_median_ms=dev_med, device_samples_ms=dev_n,
        server_median_ms=srv_med, server_samples_ms=srv_n, server_qps=rate,
        ref_device_ms=ref_dev_med, ref_server_ms=ref_srv_med,
    )


def collect_gpu(*, make_fn, make_ref_fn, qps: float | None = None,
                device_kwargs: dict | None = None,
                server_kwargs: dict | None = None) -> BenchReading:
    """The real-GPU path: time a kernel with the card template's windows, then aggregate.

    ``make_fn()`` returns the student callable to time (it enqueues the kernel and the timing
    helper synchronizes); ``make_ref_fn()`` the reference. This is the ONLY function that needs
    torch: it imports ``card_template.eval.timing`` lazily so importing this module (and running
    its tests) never pulls a GPU stack. ``time_device`` is the service window, ``time_server``
    the request-driven one; the same aggregation as :func:`collect` then applies.

    It is the thin seam between the tested collector and the hardware: everything that decides
    *what is recorded* is in :func:`collect`; this only supplies the sample lists from CUDA."""
    from forgekernel.card_template.eval import timing  # torch lives here; import lazily

    dkw = device_kwargs or {}
    skw = server_kwargs or {}

    def device_fn():
        return timing.time_device(make_fn(), **dkw)

    def ref_device_fn():
        return timing.time_device(make_ref_fn(), **dkw)

    server_fn = ref_server_fn = None
    if qps is not None and qps > 0:
        def server_fn():
            return timing.time_server(make_fn(), qps=qps, **skw)

        def ref_server_fn():
            return timing.time_server(make_ref_fn(), qps=qps, **skw)

    return collect(device_fn=device_fn, server_fn=server_fn,
                   ref_device_fn=ref_device_fn, ref_server_fn=ref_server_fn, qps=qps)
