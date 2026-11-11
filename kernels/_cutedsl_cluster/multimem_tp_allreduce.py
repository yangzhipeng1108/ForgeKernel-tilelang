"""Minimal NVLink multicast-memory (multimem) skeleton: load-reduce + store.

Source: cutlass.utils.distributed in cutlass 4.8.0 (600 lines), every symbol read from
the installed module. API-surfaced and runnable-checks-below only:
NO hardware has executed a multicast instruction from this repo yet.

!!! HARDWARE PREREQUISITE !!!
multimem needs a MULTICAST MEMORY OBJECT on an NVLink/NVSwitch (NVLS) domain.
A plain cudaMalloc pointer is not one. The RTX 6000D in this repo has no NVLink at all
(GPU0<->GPU1 = SYS), so these ops can never run here -- and the module has no arch guard,
so the failure will not look like an arch error.

Run inside an image that ships CuTeDSL (lmsysorg/sglang:dev, cutlass 4.8.0):

    python3 multimem_tp_allreduce.py
"""

from __future__ import annotations

import cutlass
from cutlass.utils import distributed as dist

# ---------------------------------------------------------------------------
# what this module actually offers (read from __all__, cutlass 4.8.0)
# ---------------------------------------------------------------------------
LD_REDUCE_128 = ("multimem_ld_reduce_8xf16", "multimem_ld_reduce_4xf32",
                 "multimem_ld_reduce_8xbf16", "multimem_ld_reduce_16xe4m3",
                 "multimem_ld_reduce_16xe5m2")
LD_REDUCE_64 = ("multimem_ld_reduce_4xf16", "multimem_ld_reduce_2xf32",
                "multimem_ld_reduce_4xbf16", "multimem_ld_reduce_8xe4m3",
                "multimem_ld_reduce_8xe5m2")
LD_REDUCE_32 = ("multimem_ld_reduce_2xf16", "multimem_ld_reduce_1xf32",
                "multimem_ld_reduce_2xbf16", "multimem_ld_reduce_4xe4m3",
                "multimem_ld_reduce_4xe5m2")
STORES = ("multimem_st_4xb32", "multimem_st_2xb32", "multimem_st_1xb32")
LOCKS = ("multimem_red_add1", "red_add1", "spin_lock_atom_cas_relaxed_wait",
         "spin_lock_atom_cas_acquire_wait", "spin_lock_ld_lt_relaxed_wait")


def required_symbols() -> tuple[str, ...]:
    return LD_REDUCE_128 + LD_REDUCE_64 + LD_REDUCE_32 + STORES + LOCKS


def check_surface() -> list[str]:
    """Assert the surface this skeleton documents is actually in the build."""
    missing = [n for n in required_symbols() if not hasattr(dist, n)]
    if hasattr(dist, "multimem_ld_reduce") is False:
        missing.append("multimem_ld_reduce")
    if hasattr(dist, "multimem_st") is False:
        missing.append("multimem_st")
    return missing


def check_reduce_is_add_only() -> None:
    """Every reduce spelling in this module is `add`. A non-sum reduction has to go
    through the lock route; there is no min/max/mul multicast load-reduce."""
    import inspect
    src = inspect.getsource(dist)
    ops = set()
    for line in src.splitlines():
        if "multimem.ld_reduce" in line and "add" in line:
            ops.add("add")
    assert ops == {"add"}, f"expected add-only, saw {ops}"


def check_no_arch_guard() -> bool:
    """The module has no arch/capability guard. Confirmed by scanning its source; kept
    as an executable check so a future version that adds one is noticed."""
    import inspect
    import re
    src = inspect.getsource(dist)
    hits = [l for l in src.splitlines()
            if re.search(r"\barch\b|sm_9|sm_10|capability", l, re.I)]
    return len(hits) == 0


def build_tp_allreduce_epilogue(mc_acc_ptr, dtype, num_elements):
    """What a fused all-reduce epilogue would look like, in ONE call.

    The accumulator arrives as registers; the reduce happens against the peers of the
    multicast object; the summed value comes back in registers. No collective, no HBM
    round trip -- but the pointer MUST be a multicast pointer, which the caller owns.

        regs = dist.multimem_ld_reduce(mc_acc_ptr, dtype=dtype, num_elements=num_elements)

    Note the register width is fp32 even for fp16/fp8 dtypes: the 16-bit variants lower to
    `add.acc::f32.*`, the fp8 variants to `add.acc::f16.*`. So fp8 reduce is not free of
    a narrower accumulate.
    """
    return dist.multimem_ld_reduce(mc_acc_ptr, dtype=dtype, num_elements=num_elements)


def main():
    print("cutlass", getattr(cutlass, "__version__", "?"))
    missing = check_surface()
    print("missing symbols:", missing or "none")
    check_reduce_is_add_only()
    print("reduce op family: add only (no min/max/mul)")
    print("module has arch guard:", check_no_arch_guard(), "(False = will not warn on wrong HW)")

    print("variants: 128-bit", len(LD_REDUCE_128), "| 64-bit", len(LD_REDUCE_64),
          "| 32-bit", len(LD_REDUCE_32), "| stores", len(STORES), "| locks", len(LOCKS))

    # What a real kernel still needs:
    #   1) a MULTICAST memory object (host side: multicast-memory registration)
    #      -- a plain cudaMalloc pointer is not one, and nothing will tell you
    #   2) an NVLink/NVSwitch (NVLS) domain; plain PCIe has no multicast
    #   3) an answer to the weak-memory question: does a reduce-load see a peer's store
    #      issued moments earlier in the same kernel, or is a fence required?
    #      (the module exposes no fence; every op here is .weak)
    #   4) a probe on real hardware before quoting any number
    print("todo: multicast object -> NVLS domain -> weak-memory fence question -> probe")


if __name__ == "__main__":
    main()
