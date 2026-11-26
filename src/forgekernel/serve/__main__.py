"""``python -m forgekernel.serve`` -- one-click deployment run (or a dry-run preview).

Two jobs, matching the harness's split between "what would run" and "run it":

* ``--dry-run`` (default off): print the ``sglang serve`` and each ``bench_serving`` command the
  config would issue, then exit. This is the CPU-only path -- no server, no GPU -- useful to check
  a recipe before committing a machine to a 200s cold start.
* otherwise: run :func:`forgekernel.serve.runtime.run_deployment` (start server, sweep
  concurrencies, gates, stop) and print a one-line verdict per concurrency.

Config comes from flags; the heavy gate runners (cuda_graph / paged_kv / parity) are only built
when ``--gates`` names them AND the caller supplies the kernel, so a plain bench sweep runs with
no torch import. The real gate wiring lives in the operator's own entry (it holds the kernel);
this CLI covers the command/bench path end to end.
"""

from __future__ import annotations

import argparse
import sys

from .harness import ServeConfig, bench_command, serve_command, shell_preview


def _config_from_args(a: argparse.Namespace) -> ServeConfig:
    return ServeConfig(
        model_path=a.model_path,
        image=a.image,
        host=a.host,
        port=a.port,
        serve_flags=list(a.serve_flag or []),
        backend=a.backend,
        concurrencies=tuple(int(x) for x in (a.concurrency or ["16"])),
        input_len=a.input_len,
        output_len=a.output_len,
        dataset_path=a.dataset_path,
        python=a.python,
        run_gates=tuple(a.gates or []),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forgekernel.serve",
                                description="Deploy a served model and decide 'optimal in deployment'.")
    p.add_argument("--model-path", required=True, help="model weights path the server loads")
    p.add_argument("--image", default="lmsysorg/sglang:dev")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--serve-flag", action="append", metavar="FLAG",
                   help="a flag appended to 'sglang serve' (repeatable), e.g. --serve-flag=--tp --serve-flag=2")
    p.add_argument("--backend", default=None, help="label for the reading, e.g. flashinfer")
    p.add_argument("--concurrency", action="append", metavar="C",
                   help="client concurrency to sweep (repeatable); default 16")
    p.add_argument("--input-len", type=int, default=10000)
    p.add_argument("--output-len", type=int, default=2000)
    p.add_argument("--dataset-path", default="/dataset/random")
    p.add_argument("--python", default="python3")
    p.add_argument("--gates", action="append", metavar="GATE",
                   help="deploy gate to run (cuda_graph/paged_kv/parity); needs an operator entry to supply the kernel")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands instead of running (no server, no GPU)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _config_from_args(args)

    if args.dry_run:
        print("sglang serve:")
        print("  " + shell_preview(serve_command(cfg)))
        print("bench_serving (one per concurrency):")
        for c in cfg.concurrencies:
            print("  " + shell_preview(bench_command(cfg, concurrency=c)))
        return 0

    from .runtime import run_deployment
    out = run_deployment(cfg)
    for r in out["readings"]:
        tok = "?" if r.tok_s is None else f"{r.tok_s:.1f}"
        ttft = "?" if r.ttft_s is None else f"{r.ttft_s:.0f}ms"
        verdict = "deployable" if r.deployable else "gates-not-all-run"
        print(f"C={r.concurrency:>3}  tok/s={tok:>8}  ttft={ttft:>8}  {verdict}  {r.gates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
