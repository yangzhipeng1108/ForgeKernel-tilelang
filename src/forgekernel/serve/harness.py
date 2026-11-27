"""The end-to-end serving harness: build the commands, run them (behind injected callables),
parse the result, run the deploy gates, and return a :class:`ServingReading`.

The GPU/server work is kept behind three injected callables exactly as ``server_bench.collect``
keeps timing behind ``device_fn``/``server_fn``:

* ``start_server(cmd, env) -> handle``  -- launch the served model (``sglang serve ...``);
* ``run_bench(cmd) -> str``             -- run ``python -m sglang.bench_serving ...`` and return
  its stdout (the JSON summary is printed on the last line);
* ``run_checks(handle, cfg) -> dict``   -- run the deploy gates (cuda_graph / paged_kv / parity)
  and return ``{gate: True/False/None}``.

In production these wrap ``subprocess`` + a live server; in tests they are plain fakes, so the
whole command-assembly / parsing / verdict path is verified without a GPU or a server. The one
thing that needs real hardware is the body of those callables, nothing that decides the result.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field

from .e2e_record import ServingReading, serving_reading_from_bench


# The three known traps the deployment doc records for bench_serving (porting/6000D/...:115):
# bind the client to 127.0.0.1 (not a hostname the container cannot resolve), the random dataset
# needs a real local --dataset-path, and C=16 is the stable concurrency. Encoded here so the
# harness cannot silently reintroduce them.
SERVE_BASE_URL = "http://127.0.0.1:30000"
STABLE_CONCURRENCY = 16


@dataclass
class ServeConfig:
    """Everything the harness needs to drive one deployment run.

    ``serve_flags`` are appended verbatim to ``sglang serve`` (backend choices, page size,
    tp/ep ...); ``env`` is the server process environment; ``concurrencies`` the client loads to
    sweep. ``image`` / ``model_path`` are recorded so a reading knows what it was taken against.
    A ``bench_extra_flags`` list threads any client-side flag the caller needs.

    On a PCIe-only SM12X box (RTX 6000D, no NVLink) put ``NCCL_P2P_LEVEL=SYS`` in ``env`` -- it is
    the cheapest small-message decode lever there (see hw_kb/tilelang/sm120/sm120_pcie_collectives.md)."""

    model_path: str
    image: str = "lmsysorg/sglang:dev"
    host: str = "0.0.0.0"
    port: int = 30000
    serve_flags: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    backend: str | None = None
    concurrencies: tuple[int, ...] = (STABLE_CONCURRENCY,)
    input_len: int = 10000
    output_len: int = 2000
    dataset_path: str = "/dataset/random"
    python: str = "python3"
    bench_extra_flags: list[str] = field(default_factory=list)
    run_gates: tuple[str, ...] = ()
    # prefix caching + context length: the two run facts that swing TTFT by tens of times on a
    # Blackwell/PCIe box (RTX 6000D 512K TTFT 239s -> 4.4s with the cache, 54x). Recorded on every
    # reading so compare_to_baseline can refuse a cross-workload ratio. None = not asserted.
    prefix_cache: bool | None = None


def serve_command(cfg: ServeConfig) -> list[str]:
    """The ``sglang serve`` argv. Flags are passed through unchanged (the caller owns them);
    the harness only fixes the invariant parts (model path, host, port)."""
    cmd = ["sglang", "serve", "--model-path", cfg.model_path,
           "--host", cfg.host, "--port", str(cfg.port)]
    cmd.extend(cfg.serve_flags)
    return cmd


def bench_command(cfg: ServeConfig, *, concurrency: int) -> list[str]:
    """The ``python -m sglang.bench_serving`` argv for one concurrency.

    Applies the three traps' fixes: base-url binds 127.0.0.1 (``SERVE_BASE_URL``), the random
    dataset carries a real ``--dataset-path``, and the caller is expected to sweep a stable
    concurrency (``STABLE_CONCURRENCY``) rather than an unstable small one."""
    return [
        cfg.python, "-m", "sglang.bench_serving",
        "--backend", "vllm-chat",
        "--dataset-name", "random",
        "--dataset-path", cfg.dataset_path,
        "--random-input-len", str(cfg.input_len),
        "--random-output-len", str(cfg.output_len),
        "--num-prompts", str(concurrency),
        "--max-concurrency", str(concurrency),
        "--base-url", SERVE_BASE_URL,
        *cfg.bench_extra_flags,
    ]


def parse_bench_summary(stdout: str) -> dict:
    """The JSON summary from a ``bench_serving`` run: the last non-empty line that parses.

    The tool prints human lines then a final JSON object; a version that prints extra trailing
    text still yields the object as long as it is the last parseable line. An unparseable run
    returns ``{}`` so the caller records a reading with None metrics rather than crashing --
    an unmeasured run must read as "not measured", never as a zero."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


class _Args:
    """Shim so :func:`serving_reading_from_bench` can read the run facts from our config."""

    def __init__(self, cfg: ServeConfig, concurrency: int):
        self.max_concurrency = concurrency
        self.random_input_len = cfg.input_len
        self.random_output_len = cfg.output_len
        self.num_prompts = concurrency


def run_once(cfg: ServeConfig, *, start_server, run_bench, run_checks=None,
             concurrency: int = STABLE_CONCURRENCY) -> ServingReading:
    """Drive one deployment reading: start the server, bench at ``concurrency``, run the gates.

    ``start_server(cmd, env)`` -> handle; ``run_bench(cmd)`` -> stdout; ``run_checks(handle,
    cfg)`` -> ``{gate: bool|None}``. The handle is what a real gate needs (a client bound to the
    live server); a fake start_server may return anything the fake run_checks ignores. Gates are
    only run when ``cfg.run_gates`` names them, so a snapshot config skips them and records None
    (unmeasured), never a fabricated pass."""
    handle = start_server(serve_command(cfg), dict(cfg.env))
    stdout = run_bench(bench_command(cfg, concurrency=concurrency))
    summary = parse_bench_summary(stdout)
    gates: dict[str, bool | None] = {}
    if run_checks is not None and cfg.run_gates:
        result = run_checks(handle, cfg) or {}
        for name in cfg.run_gates:
            gates[name] = result.get(name)
    return serving_reading_from_bench(_Args(cfg, concurrency), summary,
                                      backend=cfg.backend, gates=gates,
                                      prefix_cache=cfg.prefix_cache)


def run_sweep(cfg: ServeConfig, *, start_server, run_bench, run_checks=None) -> list[ServingReading]:
    """Run :func:`run_once` over every concurrency in ``cfg.concurrencies``, one server start.

    The server is started once and benched at each load, so a sweep measures the same deployment
    under different client pressure rather than restarting per point (a restart re-captures CUDA
    graphs and is not the comparison anyone wants)."""
    handle = start_server(serve_command(cfg), dict(cfg.env))
    readings: list[ServingReading] = []
    for c in cfg.concurrencies:
        stdout = run_bench(bench_command(cfg, concurrency=c))
        summary = parse_bench_summary(stdout)
        gates: dict[str, bool | None] = {}
        if run_checks is not None and cfg.run_gates:
            result = run_checks(handle, cfg) or {}
            for name in cfg.run_gates:
                gates[name] = result.get(name)
        readings.append(serving_reading_from_bench(_Args(cfg, c), summary,
                                                   backend=cfg.backend, gates=gates,
                                                   prefix_cache=cfg.prefix_cache))
    return readings


def shell_preview(cmd: list[str]) -> str:
    """The command as a copy-pasteable shell line (for docs / logs / dry-run output)."""
    return " ".join(shlex.quote(part) for part in cmd)
