# CLC and warp-group cooperation on sm120: CLC compiles to nothing, warp policy is already done

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp8 e4m3, tile `(128,128,128)`,
> `sm_120a`, emitting-source and post-`nvcc` inspection included.

## Purpose

Two "advanced scheduling" families remain on the checklist:

- **CLC (Cluster Launch Control)** — Blackwell's hardware dynamic task pool: a resident CTA asks the
  hardware for the next CTA coordinates instead of computing them, so scheduling survives preemption
  and uneven work without a hand-written persistent loop. Primitives: `clc_try_cancel`,
  `clc_is_canceled`, `clc_get_first_ctaid_x/y/z` (`tilelang.cuda.language.cluster`).
- **Cooperative warp-group / warp-policy** — several warpgroups in one block splitting a big tile so
  no single warpgroup carries the whole register load. The handle in TileLang is
  `T.gemm(..., policy=GemmWarpPolicy.{Square,FullRow,FullCol})`.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- fp8 e4m3, `BM=BN=BK=128`, `num_stages=2`, 4096³

## Finding 1: CLC exists as an API but compiles to a no-op on sm120 — and silently corrupts results

The API is real and callable:

```python
res = T.alloc_shared((16,), "uint32")
bar = T.alloc_barrier(1)
T.clc_try_cancel(res, bar)          # emits tl::clc_try_cancel(...)  — and that is all it does
```

The emitted CUDA does contain the call:

```cuda
tl::clc_try_cancel((&(((uint*)res)[0])), (&(bar[0])));
```

But the C++ template body (`src/tl_templates/cuda/cluster.h`) is:

```cpp
template <bool kDependentFalse = false>
TL_DEVICE void clc_try_cancel(void *result_ptr, void *mbar_ptr) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  ... asm volatile("clusterlaunchcontrol.try_cancel.async..."); ...
#else
  static_assert(kDependentFalse, "tl::clc_try_cancel requires sm_100a or a compatible ...");
#endif
}
```

`kDependentFalse` is a **dependent-false** static assert: it never fires, the function body is
**empty**, and no `clusterlaunchcontrol` instruction is emitted. Verified two ways:

- compiled SASS: `cuobjdump -sass` on the CLC kernel → **0** matches for `CLC|LAUNCHCONTROL`;
- compiled PTX/CUDA: preprocessed source → **0** `clusterlaunchcontrol`.

And the macro really is off for this target. **Root cause, pinned precisely (corrected):** the
enable condition chain is

```
__CUDA_ARCH__ == 1200  AND  __CUDA_ARCH_FEAT_SM120_ALL
  → CUTLASS_ARCH_MMA_SM120A_ENABLED   (cutlass/arch/config.h:160)
  → CUTLASS_ARCH_CLC_ENABLED          (cutlass/arch/config.h:225)
  → the asm in cluster.h runs; otherwise the #else static_assert no-op
```

A device-pass probe (`#include <cutlass/arch/config.h>`, `__global__`, `-arch=sm_120a`) prints
**`SM120A_ENABLED=1 CLC_ENABLED=1`** — so the arch *does* support CLC and the macros *do* turn on when
config.h is evaluated inside a device TU. Yet the compiled kernel's SASS has **0**
`clusterlaunchcontrol`. The reason: **TileLang's `src/tl_templates/cuda/cluster.h` only includes
`common.h`, never `cutlass/arch/config.h`.** So in the actual kernel TU `CUTLASS_ARCH_CLC_ENABLED` is
never defined, `#if defined(...)` is false, and the empty `#else` body is what compiles.

So the earlier wording ("the arch macro is off because config.h keys off TunableOp macros") was
**imprecise**: the arch is fine and the macro is reachable — TileLang simply does not include the header
that defines it. It is a **TileLang include omission, not an arch limitation** — and therefore fixable in
TileLang (add `#include <cutlass/arch/config.h>` to `cluster.h`), not by anything a user does at compile
time. Try as you like with `-include`, it does not help: `-include` evaluates config.h before
`__CUDA_ARCH__` is set (host pass), so the `__CUDA_ARCH__ == 1200` guard is false there; the header must be
pulled in from within the device TU, which only TileLang's own source controls. Verified: forcing
`-D__CUDA_ARCH_FEAT_SM120_ALL` and `-include config.h` both still yield **0** `clusterlaunchcontrol` in the
tilelang-compiled cubin.

**The consequence is worse than "no-op": the kernel returns wrong results.** With the CLC call in
place the pipeline still *waits* on the barrier the CLC query was supposed to signal, the wait is
never satisfied by a real transaction, and the accumulator is read early:

| kernel | result |
|---|---|
| 4096³ fp8, no CLC | `md = 0.1248` (exact, fp16 ulp at \|C\|≈140) |
| 4096³ fp8, `clc_try_cancel(res, bar)` inserted | **`md = 8498`** (garbage, no NaN) |

The mangled result appears only because we inserted a dead CLC call next to a live barrier; it is
not a real scheduling speedup or slowdown, it is corruption.

## Finding 2: warp-group cooperation is already the default, and the policy knob is measured

"Multiple warpgroups splitting one tile" is not something you add — `T.gemm` partitions the
`(BM,BN)` tile across all warps of the consumer group by `GemmWarpPolicy`, and `Square` (the
default) is the best of the three on this arch. This was already measured in
`gemm_operand_layout_and_policy.md`:

| policy | throughput | `smem_wavefronts_ld` |
|---|---|---|
| **Square** (default) | **133.3 TFLOP/s** | **3.36e7** |
| FullRow | 128.6 | 4.19e7 |
| FullCol | 127.6 | 4.19e7 |

`Square` pulls ~25 % fewer shared-memory wavefronts; the other two read the operands out of shared
more often. So the knob is live (the emitted partition differs) but the default already wins.

The low-level `warpgroup_arrive` / `warpgroup_commit_batch` / `warpgroup_wait` primitives also
exist (`tilelang/language/builtin.py`), but they are the **wgmma commit-group protocol**, and this
arch has no wgmma (warp-level `HMMA` only) — the pipeline uses mbarrier phase-check/trywait
instead. Using them by hand here would be recreating a protocol the arch does not run.

## When needed (trigger)

- **CLC: not usable on this build, but it is a TileLang include bug, not the arch.** The API compiles
  and links but emits nothing, and next to a live barrier it corrupts the output. **The arch supports
  CLC** (`sm_120a` → `SM120A_ENABLED=1` → `CLC_ENABLED=1` when config.h is in the device TU), but
  `cluster.h` does not `#include <cutlass/arch/config.h>`, so the macro is never defined where it
  matters. Fixable in TileLang (one include); a user cannot fix it with `-include`/`-D` (both verified
  still 0). Re-probe after a TileLang bump that touches `src/tl_templates/cuda/cluster.h`.
- **Cooperative warp-group: nothing to do.** The default `Square` partition is optimal and is
  already applied. A `policy=` sweep is a legitimate measurement, but the answer here is "keep the
  default."
- The whole point of CLC (surviving preemption / uneven work without a manual persistent loop) has
  no measured benefit on a compute-bound GEMM anyway; and the manual persistent loop is already a
  tie (`block_scheduling_is_a_tie.md`), so even a working CLC would land inside noise for a dense
  GEMM. So fixing the include would make CLC *correct*, not necessarily *faster* — the tie on the
  persistent-loop analogue predicts it stays a tie.

## Minimal runnable example (showing the trap)

```python
import tilelang
import tilelang.language as T

M = N = K = 4096
BM = BN = BK = 128
DT = "float8_e4m3"


@T.prim_func
def with_clc(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT), C: T.Tensor((M, N), "float16")):
    with T.ClusterKernel(T.ceildiv(N, BN), T.ceildiv(M, BM), cluster_dims=1, threads=256) as (bx, by):
        a_s = T.alloc_shared((BM, BK), DT)
        b_s = T.alloc_shared((BK, BN), DT)
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)
        res = T.alloc_shared((16,), "uint32")
        bar = T.alloc_barrier(1)
        T.clc_try_cancel(res, bar)        # compiles to NOTHING on sm_120a; corrupts `acc` below
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            T.copy(A[by * BM, k * BK], a_s)
            T.copy(B[k * BK, bx * BN], b_s)
            T.gemm(a_s, b_s, acc)          # default policy=Square is already the cooperative split
        T.copy(acc, C[by * BM, bx * BN])


kernel = tilelang.compile(with_clc, out_idx=[2], target={"kind": "cuda", "arch": "sm_120a"})
# Compiles, runs, returns garbage. Delete the clc_try_cancel line to get an exact result.
```

## What TileLang decides for you here (and what it does not)

**Decided by the pass:** the cooperative warp split of `T.gemm` (default `Square`), and — the
problem — that `clc_try_cancel` lowers to an empty function because the arch macro is off.

**Decided by you:** `policy=` (if you want to re-measure), and whether to insert CLC calls at all
(here: don't).

## Gotchas

- **`static_assert(kDependentFalse, ...)` is a silent, non-firing assertion.** It compiles clean and
  is designed to let an unsupported intrinsic degrade to a no-op rather than fail the build. A
  compile success therefore says nothing about whether the instruction runs — check the SASS
  (`cuobjdump -sass | grep -i clc`) before believing any CLC result.
- **A dead CLC call next to a live barrier corrupts output** (`md 0.12 → 8498`). If you experiment,
  remove the barrier wait too, or the result is meaningless either way.
- **`CUTLASS_ARCH_CLC_ENABLED` must be *included* to be defined.** `config.h` sets it from
  `SM120A_ENABLED` inside a device TU (`__CUDA_ARCH__==1200`), so it *is* on for `sm_120a` — but only if
  that header is in the TU. `cluster.h` omits it, so the macro is undefined where `clc_try_cancel` lives.
  Confirm with a device-side `printf` probe (not a host `#error`) — a host-side check sees `__CUDA_ARCH__`
  undefined and wrongly reports OFF.
- **The warp-policy default is already correct.** Do not "add cooperation"; `T.gemm` does it, and
  `Square` wins.

## Sources

1. TileLang 0.1.15: `tilelang/cuda/language/cluster.py` (`clc_*`, `block_rank_in_cluster`),
   `src/tl_templates/cuda/cluster.h` (the `dependent_false` no-op body),
   `3rdparty/cutlass/include/cutlass/arch/config.h:225` (`CUTLASS_ARCH_CLC_ENABLED` gate),
   `tilelang/language/builtin.py` (`warpgroup_arrive` / `commit_batch` / `wait`).
2. `gemm_operand_layout_and_policy.md` (the `Square` vs `FullRow`/`FullCol` measurement),
   `block_scheduling_is_a_tie.md` (persistent already ties, so CLC's target symptom is absent).
3. This repo's probes on an RTX 6000D, clock-locked, `cuobjdump`/`nvcc -E` verified.
