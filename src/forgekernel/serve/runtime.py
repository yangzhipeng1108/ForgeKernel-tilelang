"""Production runtime: the real bodies behind the harness's injected callables.

The harness (:mod:`forgekernel.serve.harness`) drives ``start_server`` / ``run_bench`` /
``run_checks`` as injected callables so its decision logic is unit-testable. This module is the
*production* set of those three, built on ``subprocess`` + ``urllib`` (no third-party deps), and
the one function that wires them into a one-shot run: :func:`run_deployment`.

Every function here is split so the *process-and-parse* logic is testable with a fake command
runner / fake clock, while the parts that truly need a live server + GPU (spawning the model,
capturing a CUDA graph of the real kernel, running the real kernel for parity) are the leaves.
Nothing imports torch at module load.
"""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .harness import ServeConfig, bench_command, parse_bench_summary, serve_command
from .e2e_record import ServingReading, serving_reading_from_bench


# The health endpoint the deployment doc polls (curl -sf .../health until 200). ~205s cold start
# for a large model, polled every few seconds; the timeout is generous enough for a cold model.
HEALTH_PATH = "/health"
HEALTH_POLL_S = 5.0
HEALTH_TIMEOUT_S = 600.0


@dataclass
class ServerHandle:
    """A live server process the harness holds across a sweep.

    ``proc`` is the ``subprocess.Popen`` of ``sglang serve`` (None if the caller manages the
    server out of band -- e.g. connecting to one already running). ``base_url`` is what gates and
    the bench client talk to. ``stop()`` terminates the process if we own it."""

    base_url: str
    proc: object | None = None
    own: bool = True

    def stop(self, *, timeout: float = 30.0) -> None:
        if self.proc is None or not self.own:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001 - best-effort teardown
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass


def _default_popen(cmd, env, **kw):
    return subprocess.Popen(cmd, env=env, **kw)


def start_server(cfg: ServeConfig, *, popen=None, wait_healthy=True,
                 http_get=None, clock=None, sleep=None) -> ServerHandle:
    """Launch ``sglang serve`` and (optionally) block until ``/health`` returns 200.

    ``popen`` / ``http_get`` / ``clock`` / ``sleep`` are injectable so the launch-and-wait logic
    is testable without a real model (a fake popen returns a dummy proc; a fake http_get raises
    until the Nth call to model cold start). ``base_url`` is built from the client-facing
    127.0.0.1 (the deploy doc's rule), not the ``--host`` bind address."""
    import os
    popen = popen or _default_popen
    env = {**os.environ, **cfg.env}
    cmd = serve_command(cfg)
    proc = popen(cmd, env)
    base_url = f"http://127.0.0.1:{cfg.port}"
    handle = ServerHandle(base_url=base_url, proc=proc, own=True)
    if wait_healthy:
        wait_for_health(base_url, http_get=http_get, clock=clock, sleep=sleep, proc=proc)
    return handle


def wait_for_health(base_url: str, *, http_get=None, clock=None, sleep=None,
                    proc=None, timeout_s: float = HEALTH_TIMEOUT_S,
                    poll_s: float = HEALTH_POLL_S) -> float:
    """Poll ``GET {base_url}/health`` until 200 or ``timeout_s``; return elapsed seconds.

    Raises :class:`TimeoutError` on timeout, and fails fast if the server process has exited
    (a dead server never becomes healthy -- waiting the full timeout would waste minutes)."""
    http_get = http_get or _default_http_get
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    url = base_url.rstrip("/") + HEALTH_PATH
    start = clock()
    while True:
        if proc is not None and getattr(proc, "poll", lambda: None)() not in (None,):
            raise RuntimeError("server exited before becoming healthy")
        try:
            status = http_get(url)
            if status == 200:
                return clock() - start
        except (urllib.error.URLError, ConnectionError):
            pass
        if clock() - start >= timeout_s:
            raise TimeoutError(f"server not healthy after {timeout_s:.0f}s at {url}")
        sleep(poll_s)


def _default_http_get(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 - fixed health URL
            return int(resp.status)
    except urllib.error.HTTPError as e:
        return int(e.code)


@dataclass
class BenchResult:
    argv: list[str]
    stdout: str
    returncode: int

    @property
    def summary(self) -> dict:
        return parse_bench_summary(self.stdout)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and bool(self.summary)


def run_bench(cfg: ServeConfig, *, concurrency: int, runner=None) -> BenchResult:
    """Run ``bench_serving`` at ``concurrency`` and return argv + stdout + returncode.

    ``runner(argv) -> (returncode, stdout)`` is injectable (a fake returns canned output). The
    default runs the subprocess and captures stdout; a non-zero exit is preserved in the result
    so the caller records an unmeasured reading rather than a fabricated one."""
    argv = bench_command(cfg, concurrency=concurrency)
    if runner is None:
        cp = subprocess.run(argv, capture_output=True, text=True)  # noqa: S603 - argv, no shell
        return BenchResult(argv=argv, stdout=cp.stdout, returncode=cp.returncode)
    rc, out = runner(argv)
    return BenchResult(argv=argv, stdout=out or "", returncode=int(rc))


def run_checks(handle: ServerHandle, cfg: ServeConfig, *, gate_runners: dict) -> dict:
    """Run the configured deploy gates against the live server; return ``{gate: bool|None}``.

    ``gate_runners`` maps a gate name (``cuda_graph`` / ``paged_kv`` / ``parity``) to a callable
    ``(handle, cfg) -> bool|None`` -- the pieces that need the real kernel and GPU live there
    (built in the caller from :mod:`forgekernel.serve.checks`). A gate with no runner is None
    (unmeasured), never a fabricated pass. Only gates named in ``cfg.run_gates`` are consulted."""
    out: dict[str, bool | None] = {}
    for name in cfg.run_gates:
        fn = gate_runners.get(name)
        out[name] = None if fn is None else bool(fn(handle, cfg)) if fn(handle, cfg) is not None else None
    return out


def make_gate_runners(cfg: ServeConfig, *, capture_fn=None, kernel_inputs=None,
                      kernel_out=None, server_out=None, cosine_fn=None) -> dict:
    """Build the ``gate_runners`` dict from concrete GPU callables / tensors, one per gate.

    A gate whose inputs are None gets no runner, so :func:`run_checks` records it unmeasured.
    The heavy work (capture the graph, produce the two outputs) is done once by the caller and
    passed in, keeping this module free of torch."""
    from .e2e_record import GATE_GRAPH, GATE_PAGED, GATE_PARITY
    from .checks import graph_capturable, paged_kv_ok, parity_ok

    runners: dict = {}
    if capture_fn is not None:
        runners[GATE_GRAPH] = lambda h, c: graph_capturable(capture_fn).passed
    if kernel_inputs is not None:
        runners[GATE_PAGED] = lambda h, c: paged_kv_ok(kernel_inputs).passed
    if kernel_out is not None and server_out is not None:
        runners[GATE_PARITY] = lambda h, c: parity_ok(kernel_out, server_out, cosine_fn=cosine_fn).passed
    return runners


def run_deployment(cfg: ServeConfig, *, start=None, bench=None, gate_runners=None,
                   base_url_override: str | None = None) -> dict:
    """One-call deployment run: start the server, bench each concurrency, run the gates.

    ``bench(cfg, concurrency) -> (returncode, stdout)`` is the injectable bench step (default:
    a real ``run_bench``). Returns ``{"readings": [ServingReading...], "server_health_s":
    float|None}``. The server is stopped at the end if we started it. ``base_url_override``
    connects to an already-running server (skip start/stop) -- useful when the operator launches
    the model out of band."""
    start = start or (lambda cmd, env: start_server(cfg))

    def _bench(cfg_, concurrency):
        if bench is not None:
            return bench(cfg_, concurrency)
        r = run_bench(cfg_, concurrency=concurrency)
        return r.returncode, r.stdout

    readings: list[ServingReading] = []
    if base_url_override:
        handle = ServerHandle(base_url=base_url_override, proc=None, own=False)
    else:
        handle = start(serve_command(cfg), dict(cfg.env))

    try:
        for c in cfg.concurrencies:
            _rc, stdout = _bench(cfg, c)
            summary = parse_bench_summary(stdout)

            class _A:  # minimal args shim for the reading builder
                max_concurrency = c
                random_input_len = cfg.input_len
                random_output_len = cfg.output_len
                num_prompts = c

            gates: dict[str, bool | None] = {}
            if cfg.run_gates and gate_runners is not None:
                res = run_checks(handle, cfg, gate_runners=gate_runners) or {}
                for name in cfg.run_gates:
                    gates[name] = res.get(name)
            readings.append(serving_reading_from_bench(_A, summary, backend=cfg.backend, gates=gates))
    finally:
        if isinstance(handle, ServerHandle):
            handle.stop()
    return {"readings": readings, "server_health_s": None}
