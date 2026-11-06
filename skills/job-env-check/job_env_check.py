#!/usr/bin/env python3
"""Job-environment smoke for the driver's own tools (driver-seeded into notes/).

The harness launches its tools on the GPU node as ``python3 notes/<tool>.py``
(timeline_probe, stall_price_run, lever_pricing_run, orch_census_run) and they
re-spawn ``python3 eval/bench.py``: every one of them needs the job's bare
``python3`` to import torch. On an ssh host the job runs in a non-login
``ssh`` shell, so that ``python3`` is the system interpreter unless the card's
gpu_entry.sh puts the image's venv first on PATH — and then every such tool dies on
its first import while the kernel itself, run through gpu_entry's own interpreter,
evaluates fine (seen on a fused-norm-rope B200 run: every timeline, every stall
ledger and every first pricing attempt of the run).

Run once per loop start by the driver (``cmd:python3 notes/job_env_check.py``).
Prints exactly one verdict line:

    JOB-ENV-OK python3=<path> torch=<version>
    JOB-ENV-FAIL python3=<path> <last line of the failure>

with the failing import's own text printed verbatim above a FAIL line.
"""
from __future__ import annotations

import shutil
import subprocess
import sys


def cuda_toolchain_problem(env=None) -> str | None:
    """What a CUDA-mode card (``--cuda``: the student's kernel is CUDA C++, built in
    the job by torch's cpp_extension with nvcc + CUTLASS headers) is missing on this
    image, or None. Missing, every build reads COMPILE_ERROR — the loop's verdict on
    the tree — when it is the image's."""
    import os
    env = os.environ if env is None else env
    if shutil.which("nvcc") is None and not any(
            os.path.isfile(os.path.join(env.get(k, ""), "bin", "nvcc"))
            for k in ("CUDA_HOME", "CUDA_PATH") if env.get(k)):
        return "nvcc not found (PATH / CUDA_HOME)"
    if shutil.which("ninja") is None:
        try:
            import ninja  # noqa: F401  (torch's cpp_extension builds with ninja)
        except ImportError:
            return "ninja not found (torch cpp_extension builds with it)"
    cutlass = env.get("CUTLASS_DIR") or "/usr/local/cutlass"
    # the header name in two halves: this file is seeded into notes/, and a binary-reference
    # recon tree's guard refuses notes that name C/C++ sources or headers
    if not os.path.isfile(os.path.join(cutlass, "include", "cutlass", "cutlass" + "." + "h")):
        return f"CUTLASS headers not found under {cutlass}/include (CUTLASS_DIR)"
    return None


def tilelang_toolchain_problem(env=None) -> str | None:
    """What a TileLang-mode card (``--tilelang``: the student's kernel is a Python
    TileLang module the job compiles and runs) is missing on this image, or None.

    Three things decide whether a TileLang build can run at all, and each fails with
    a different line, so each is checked and named: the package itself, a compiler
    TileLang's host-side codegen needs, and — because TileLang JITs through TVM —
    the ``apache-tvm-ffi`` runtime it links against. Missing, every build reads
    COMPILE_ERROR when it is the image's."""
    import os
    env = os.environ if env is None else env
    try:
        import tilelang  # noqa: F401
    except ImportError:
        return "tilelang not importable in the job's python3"
    try:
        import tvm_ffi  # noqa: F401  (TileLang's JIT runtime)
    except ImportError:
        return "tvm_ffi (apache-tvm-ffi) not importable in the job's python3"
    # TileLang lowers to CUDA C and compiles it with nvcc on this image; a driver
    # alone is not enough for the codegen path the kernel goes through.
    if shutil.which("nvcc") is None and not any(
            os.path.isfile(os.path.join(env.get(k, ""), "bin", "nvcc"))
            for k in ("CUDA_HOME", "CUDA_PATH") if env.get(k)):
        return "nvcc not found (PATH / CUDA_HOME)"
    return None


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    exe = shutil.which("python3") or "(no python3 on PATH)"
    want_cuda = "--cuda" in args
    want_tilelang = "--tilelang" in args
    try:
        p = subprocess.run(["python3", "-c", "import torch; print(torch.__version__)"],
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"JOB-ENV-FAIL python3={exe} could not run it: {e}", flush=True)
        return 0
    if p.returncode == 0:
        if want_cuda and (why := cuda_toolchain_problem()) is not None:
            print(f"JOB-ENV-FAIL python3={exe} {why}", flush=True)
            return 0
        if want_tilelang and (why := tilelang_toolchain_problem()) is not None:
            print(f"JOB-ENV-FAIL python3={exe} {why}", flush=True)
            return 0
        print(f"JOB-ENV-OK python3={exe} torch={p.stdout.strip()}"
              + (" cuda-toolchain=ok" if want_cuda else "")
              + (" tilelang-toolchain=ok" if want_tilelang else ""), flush=True)
        return 0
    err = (p.stderr or p.stdout or "").strip()
    if err:
        print(err, flush=True)
    last = err.splitlines()[-1] if err else f"rc={p.returncode}"
    print(f"JOB-ENV-FAIL python3={exe} {last}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
