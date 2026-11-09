#!/usr/bin/env python3
"""Measure the `[to be measured]` fields in hw_kb/tilelang/ and print a fill-in report.

Why this exists: the four TileLang mechanism entries under `hw_kb/tilelang/sm90/` were written
with their evidence blocks explicitly empty. Filling them by hand invites two failures -- taking a
reading on a machine whose arch does not match the entry's directory, and taking a reading without
saying which TileLang version emitted it. This script makes both impossible to forget: it refuses
when the arch does not match, and it stamps the version into every line it prints.

What it does NOT do: it does not edit the entries. A reading that rewrites the document it is
quoted in cannot be reviewed. It prints, per entry, the exact text to paste into `## Measured on`
and into the `[to be measured]` slots.

Usage:
    measure_kb_entries.py --arch sm90 [--out report.md]
    measure_kb_entries.py --arch sm90 --compile-only      # no GPU needed: fills the lower facts

Exit: 0 all probes produced a verdict; 1 at least one FAIL/HANG; 2 wrong arch or bad usage.

Each probe runs in its own process (see `gpu_probe_ladder.md`): a kernel that hangs takes the job
clock and returns nothing, so one rung per child with its own budget is what keeps a launch
failure from looking like a missing measurement.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
KB = REPO / "hw_kb" / "tilelang"

# NCU short keys this script asks for. These are a subset of inner/ncu.py:KEY_METRICS -- the
# ones the four entries cite. Keeping the subset here (rather than importing forgekernel) means
# the script runs on a job image where the package is not installed; the names are checked
# against ncu.py by tests/test_tilelang_mode.py so the two cannot drift.
NCU_SHORT = {
    "gpu__time_active.sum": "time_ns",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "smem_wavefronts_st",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "smem_wavefronts_ld",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "bankconf_st",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "bankconf_ld",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "tensor_pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "stall_long_sb",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "launch__shared_mem_per_block_dynamic": "smem_dynamic",
    "launch__registers_per_thread": "regs_per_thread",
}
NCU_ARG = ",".join(NCU_SHORT)

# The KB directory name is NOT the CUDA target name. `sm90` is a directory; the target that
# emits Hopper code is `sm_90a` (the `a` is the architecture-accelerated variant, and `sm_90`
# without it is refused by tilelang: "Target {'kind': 'cuda', 'arch': 'sm90'} is not supported").
# Keeping this map here rather than deriving it means a new tree has to be added deliberately.
#
# sm120 is the sharpest case: `sm_120` *lowers* fine and then fails in nvcc with
# "warpgroup_reg_dealloc requires a target with CTA register reconfiguration, such as sm_90a",
# because the pipeline's producer warpgroup needs setmaxnreg. Only the `a` form runs. Measured
# on an RTX 6000D (driver 610.43.02 / CUDA 13.3): sm_120a executes, sm_120 fails to compile,
# sm_100a compiles but fails to load with CUDA_ERROR_NO_BINARY_FOR_GPU.
KB_ARCH_TO_TARGET = {"sm90": "sm_90a", "sm100": "sm_100a", "sm120": "sm_120a"}

# Rung name -> what the entry needs from it. Each maps to a function in the probe body below.
RUNGS = {
    "pipeline": "software_pipeline_stage_budget.md: num_stages sweep, smem_dynamic per stage",
    "copy_swizzle": "tiled_copy_scope_and_swizzle.md: smem wavefronts with/without annotate_layout",
    "gemm_policy": "gemm_operand_layout_and_policy.md: tensor_pct under each GemmWarpPolicy",
    "probe_harness": "gpu_probe_ladder.md: the harness itself returns a verdict",
}


# --------------------------------------------------------------------------- the probe body
# Written to a temp file and executed as a child, so the parent can time it out.
PROBE_BODY = r'''
import json, re, sys
from pathlib import Path
import tilelang
import tilelang.language as T

M = 4096
N = 4096
K = 4096
BM, BN, BK = 128, 128, 32
THREADS = 128


def _compile(fn, target, out_idx=None):
    """Compile-only returns a lowerer; a GPU rung returns something callable.

    These are two different code paths on purpose. `tilelang.compile(...,
    execution_backend="cython")` needs a C++ compiler even when no GPU is present, so a
    compile-only box must go through `compile_kernel_source`, which is the documented
    no-device path. Using one path for both would make the compile-only rung fail on a
    machine that has tilelang but no compiler -- the exact box this mode exists for.
    """
    if COMPILE_ONLY:
        from tilelang.tools.compile_only import compile_kernel_source
        return _Lowered(compile_kernel_source(fn, target={"kind": "cuda", "arch": target}))
    return tilelang.compile(fn, target={"kind": "cuda", "arch": target},
                            out_idx=out_idx, execution_backend="cython")


COMPILE_ONLY = False      # set by the CLI; read by `_compile` inside the child process


class _Lowered:
    """Stands in for a compiled kernel so `_kernel_source` has one shape to handle."""

    def __init__(self, src):
        self._src = src

    def get_kernel_source(self):
        return self._src


def _kernel_source(k):
    return k.get_kernel_source()


@T.prim_func
def copy_kernel(x: T.Tensor((M, N), "float32"), y: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(M, BM), threads=THREADS) as bx:
        _tile_in = T.alloc_fragment((BM, N), "float32")
        _tile_out = T.alloc_fragment((BM, N), "float32")
        T.copy(x[bx * BM, 0], _tile_in)
        T.copy(_tile_in, _tile_out)
        T.copy(_tile_out, y[bx * BM, 0])


def _matmul(stages, policy):
    @T.prim_func
    def k(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
          C: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=THREADS) as (bx, by):
            a_s = T.alloc_shared((BM, BK), "float16")
            b_s = T.alloc_shared((BK, BN), "float16")
            acc = T.alloc_fragment((BM, BN), "float32")
            T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                T.copy(A[by * BM, kk * BK], a_s)
                T.copy(B[kk * BK, bx * BN], b_s)
                T.gemm(a_s, b_s, acc, policy=policy)
            T.copy(acc, C[by * BM, bx * BN])
    return k


def _launch_bounds(src):
    """The emitted thread count. Differs from the traced `threads=` when the pipeline is on:
    a producer warpgroup is appended, so threads=128 + num_stages=3 launches 256."""
    m = re.search(r"__launch_bounds__\(([^)]*)\)", src)
    return m.group(1) if m else "?"


def run_pipeline(arch):
    facts = {"stages": {}}
    for stages in (0, 2, 3, 4):
        try:
            k = _compile(_matmul(stages, T.GemmWarpPolicy.Square), arch, out_idx=[2])
            src = _kernel_source(k)
            # The markers that actually appear in Hopper output. `cp.async` is the Ampere-era
            # spelling and TileLang does not emit it here -- an earlier version of this probe
            # looked for `cp.async` and reported "no pipeline" for a kernel whose stage count
            # was clearly working. TMA + mbarrier is what a staged Hopper feed looks like.
            facts["stages"][str(stages)] = {
                "lowered": True,
                "chars": len(src),
                "launch_bounds": _launch_bounds(src),
                "tma": src.count("tma"),
                "mbarrier": src.count("mbarrier"),
                "wgmma": src.count("wgmma"),
            }
        except Exception as exc:                                  # noqa: BLE001
            facts["stages"][str(stages)] = {"lowered": False,
                                            "error": f"{type(exc).__name__}: {exc}"}
    return {"verdict": "PASS" if any(v.get("lowered") for v in facts["stages"].values()) else "FAIL",
            "facts": facts}


def run_copy_swizzle(arch):
    try:
        k = _compile(copy_kernel, arch, out_idx=[1])
        src = _kernel_source(k)
        return {"verdict": "PASS",
                "facts": {"lowered": True, "chars": len(src),
                          "launch_bounds": _launch_bounds(src),
                          "has_tma": "tma" in src.lower(),
                          "note": "annotate_layout variant is the entry's follow-up axis"}}
    except Exception as exc:                                      # noqa: BLE001
        return {"verdict": "FAIL", "facts": {"error": f"{type(exc).__name__}: {exc}"}}


def run_gemm_policy(arch):
    """Report the policy sweep *and* whether the policies changed anything at all.

    The second half matters more than the first: on most tile shapes all three policies lower
    to byte-identical source, so a GPU job comparing them would be measuring noise. Marking
    `inert` here stops that job from being spent."""
    facts = {}
    sources = {}
    for name in ("Square", "FullRow", "FullCol"):
        policy = getattr(T.GemmWarpPolicy, name)
        try:
            k = _compile(_matmul(3, policy), arch, out_idx=[2])
            src = _kernel_source(k)
            sources[name] = src
            facts[name] = {"lowered": True, "chars": len(src),
                           "launch_bounds": _launch_bounds(src)}
        except Exception as exc:                                  # noqa: BLE001
            facts[name] = {"lowered": False, "error": f"{type(exc).__name__}: {exc}"}
    ref = sources.get("Square")
    if ref is not None:
        for name, src in sources.items():
            facts[name]["same_as_square"] = (src == ref)
        facts["identical_across_all"] = len(set(sources.values())) == 1
    ok = any(v.get("lowered") for v in facts.values() if isinstance(v, dict))
    return {"verdict": "PASS" if ok else "FAIL", "facts": facts}


def run_probe_harness(arch):
    """The harness's own smoke test: does a kernel compile, run and come back with the right
    numbers on this box?

    `out_idx=[1]` makes the kernel allocate and *return* `y`, so it is called with the inputs
    only -- passing a pre-made `y` as well is an arity error (met: "Expected 2 inputs, got 3").
    The copy kernel is the right subject because a wrong result there is unambiguous."""
    import torch
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    k = _compile(copy_kernel, arch, out_idx=[1])
    y = k(x)
    torch.cuda.synchronize()
    max_diff = float((x - y).abs().max())
    return {"verdict": "PASS" if max_diff == 0.0 else "FAIL",
            "facts": {"max_diff": max_diff, "out_shape": list(y.shape)}}


HANDLERS = {"pipeline": run_pipeline, "copy_swizzle": run_copy_swizzle,
            "gemm_policy": run_gemm_policy, "probe_harness": run_probe_harness}

if __name__ == "__main__":
    rung, target, out = sys.argv[1], sys.argv[2], sys.argv[3]
    COMPILE_ONLY = sys.argv[4] == "1"
    globals()["COMPILE_ONLY"] = COMPILE_ONLY
    try:
        result = HANDLERS[rung](target)
    except Exception as exc:                                      # noqa: BLE001
        result = {"verdict": "FAIL", "facts": {"error": f"{type(exc).__name__}: {exc}"}}
    Path(out).write_text(json.dumps(result), encoding="utf-8")
'''


# --------------------------------------------------------------------------- parent
def tilelang_version() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import tilelang; print(tilelang.__version__)"],
            capture_output=True, text=True, timeout=120)
        return out.stdout.strip() or "unknown"
    except Exception:                                              # noqa: BLE001
        return "unknown"


def gpu_arch() -> str | None:
    """Return the KB tree name for the visible GPU (`sm90`), or None when there is no GPU.

    Deliberately the KB name and not the CUDA target: the comparison the caller makes is
    "--arch names a directory, and that directory is where this reading will be written", so
    both sides have to be in the directory's vocabulary.
    """
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}"
    except Exception:                                              # noqa: BLE001
        return None


def run_rung(body: Path, rung: str, target: str, budget: float, outdir: Path,
             compile_only: bool) -> dict:
    out = outdir / f"rung-{rung}.json"
    if out.exists():
        out.unlink()
    log = outdir / f"rung-{rung}.log"
    try:
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.run([sys.executable, str(body), rung, target, str(out),
                                   "1" if compile_only else "0"],
                                  stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=budget, check=False)
    except subprocess.TimeoutExpired:
        return {"verdict": "HANG", "facts": {"budget_s": budget, "log": str(log)}}
    if not out.exists():
        # A child that dies before writing its verdict used to leave nothing behind, which made
        # "the probe crashed" indistinguishable from "the probe was never run". The tail of its
        # own log is the only thing that says which.
        tail = ""
        if log.exists():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = " | ".join(line.strip() for line in lines[-4:] if line.strip())[:400]
        return {"verdict": "FAIL",
                "facts": {"error": f"rung wrote no verdict (exit {proc.returncode})",
                          "tail": tail, "log": str(log)}}
    result = json.loads(out.read_text(encoding="utf-8"))
    result.setdefault("facts", {})["log"] = str(log)
    return result


def ncu_cmd(arch_probe: str) -> str:
    return (f"ncu --metrics {NCU_ARG} --target-processes all --csv "
            f"--kernel-name-base demangled python <kernel_tl.py> ")


def render(arch: str, version: str, results: dict, compile_only: bool) -> str:
    lines = [f"# hw_kb/tilelang measurements — {arch}", "",
             f"- TileLang: `{version}`",
             f"- arch: `{arch}`",
             f"- mode: {'compile-only (L1)' if compile_only else 'GPU (L1+L3)'}",
             "", "Paste the block under each entry's `## Measured on`. The `ncu` line is what",
             "turns a lower fact into a certification; without it the entry stays pending.", ""]

    def facts_of(rung: str) -> dict:
        r = results.get(rung) or {}
        return r.get("facts") or {}

    lines += ["## software_pipeline_stage_budget.md", "",
              "```", f"- TileLang version: `{version}`", f"- arch: `{arch}`"]
    pl = facts_of("pipeline").get("stages") or {}
    if pl:
        for stage, v in pl.items():
            if v.get("lowered"):
                lines.append(f"- num_stages={stage}: lowered, {v['chars']} chars, "
                             f"launch_bounds({v.get('launch_bounds')}), "
                             f"tma={v.get('tma')}, mbarrier={v.get('mbarrier')}, "
                             f"wgmma={v.get('wgmma')}")
            else:
                lines.append(f"- num_stages={stage}: REFUSED — {v.get('error', '')[:160]}")
        st = pl.get("0") or {}
        nz = pl.get("2") or {}
        if st.get("lowered") and nz.get("lowered"):
            lines.append(f"- pipeline on/off is visible without a GPU: stages=0 has "
                         f"tma={st.get('tma')}/mbarrier={st.get('mbarrier')}, "
                         f"stages=2 has tma={nz.get('tma')}/mbarrier={nz.get('mbarrier')}")
    if not compile_only:
        lines.append(f"- smem_dynamic per stage: to be read from `{ncu_cmd(arch)}` "
                     f"(launch__shared_mem_per_block_dynamic)")
        lines.append("- axis that moved: `stall_long_sb` / `dram_pct`, before → after")
    lines += ["- Certification: [fill with the PROBE PASS job id or the before → after above]",
              "```", ""]

    lines += ["## tiled_copy_scope_and_swizzle.md", "",
              "```", f"- TileLang version: `{version}`", f"- arch: `{arch}`"]
    cs = facts_of("copy_swizzle")
    if cs:
        lines.append(f"- baseline lowered: {cs.get('lowered')} ({cs.get('chars')} chars), "
                     f"tma={'yes' if cs.get('has_tma') else 'no'}")
    if not compile_only:
        lines.append(f"- smem_wavefronts_st/ld, bankconf_st/ld: `{ncu_cmd(arch)}`; "
                     f"take the pair (default layout, annotate_layout) in one job")
    lines += ["- Certification: [fill]", "```", ""]

    lines += ["## gemm_operand_layout_and_policy.md", "",
              "```", f"- TileLang version: `{version}`", f"- arch: `{arch}`"]
    gp = facts_of("gemm_policy")
    for name in ("Square", "FullRow", "FullCol"):
        v = gp.get(name)
        if not v:
            continue
        lines.append(f"- policy {name}: " + (f"lowered, {v['chars']} chars, "
                                            f"launch_bounds({v.get('launch_bounds')}), "
                                            f"same_as_Square={v.get('same_as_square')}"
                                            if v.get("lowered")
                                            else f"REFUSED — {v.get('error', '')[:160]}"))
    if gp.get("identical_across_all"):
        lines.append("- **the policy is INERT on this tile shape**: all three lower to "
                     "byte-identical source. Do not spend a GPU job sweeping it — change the "
                     "tile (e.g. (128,256,32) with threads=256) if the milestone prices policy.")
    if not compile_only:
        lines.append(f"- tensor_pct per policy: `{ncu_cmd(arch)}`; wavefronts show the feed cost")
    lines += ["- Certification: [fill]", "```", ""]

    lines += ["## gpu_probe_ladder.md", "",
              "```", f"- TileLang version: `{version}`", f"- arch: `{arch}`"]
    ph = results.get("probe_harness") or {}
    lines.append(f"- harness verdict: {ph.get('verdict', 'not run')}"
                 + (f", max_diff={facts_of('probe_harness').get('max_diff'):.3g}"
                    if facts_of("probe_harness").get("max_diff") is not None else ""))
    lines += ["- Certification: [fill]", "```", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True,
                    help=f"{' | '.join(sorted(KB_ARCH_TO_TARGET))} — must match the KB directory")
    ap.add_argument("--budget", type=float, default=600.0, help="seconds per rung (default 600)")
    ap.add_argument("--compile-only", action="store_true",
                    help="skip the GPU rung; every other rung still runs (needs tilelang only)")
    ap.add_argument("--out", help="write the report here instead of stdout")
    args = ap.parse_args()

    if args.arch not in KB_ARCH_TO_TARGET:
        print(f"error: --arch must be one of {sorted(KB_ARCH_TO_TARGET)}, "
              f"got {args.arch!r}", file=sys.stderr)
        return 2
    if not (KB / args.arch / "index.md").exists():
        print(f"error: no KB tree at {KB / args.arch}", file=sys.stderr)
        return 2

    if not args.compile_only:
        have = gpu_arch()
        if have is None:
            print("error: --arch given but no GPU is visible.\n"
                  "       A reading taken on the wrong arch certifies the wrong entry, so this\n"
                  "       is refused rather than warned about. Use --compile-only for lower facts.",
                  file=sys.stderr)
            return 2
        if have != args.arch:
            print(f"error: --arch {args.arch} but the GPU is {have}.\n"
                  f"       Writing an {args.arch} entry from an {have} reading is how a KB becomes\n"
                  f"      untrustworthy.", file=sys.stderr)
            return 2

    # `tempfile` rather than TMPDIR/TEMP: neither is set on the Linux boxes this runs on, and
    # the old fallback of "." put the probe body in the caller's cwd -- which the child, started
    # with a different cwd, then could not find.
    outdir = Path(tempfile.mkdtemp(prefix="fk-kb-measure-"))
    body = outdir / "fk_tilelang_probe_body.py"
    body.write_text(PROBE_BODY, encoding="utf-8")

    version = tilelang_version()
    results: dict[str, dict] = {}
    rungs = [r for r in RUNGS if not (args.compile_only and r == "probe_harness")]
    for rung in rungs:
        print(f"  running rung {rung} ({RUNGS[rung]}) ...", file=sys.stderr)
        results[rung] = run_rung(body, rung, KB_ARCH_TO_TARGET[args.arch], args.budget, outdir,
                              args.compile_only)
        print(f"    -> {results[rung].get('verdict')}", file=sys.stderr)

    report = render(args.arch, version, results, args.compile_only)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    else:
        print(report)

    verdicts = {r: results[r].get("verdict") for r in rungs}
    return 0 if all(v == "PASS" for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
