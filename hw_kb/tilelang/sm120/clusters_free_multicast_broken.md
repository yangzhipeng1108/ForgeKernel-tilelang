# Thread-block clusters on sm120: launch is free, TMA multicast is 60,000× slower

> Status: **measured on hardware, clock-locked**, fp8 e4m3, tile `(128,128,128)`, 4096³.
> Correctness `md = 0.125` (= fp16 ulp at |C|≈140, i.e. exact) in every row below.

## Purpose

Decide whether thread-block **clusters** buy anything on this arch. Two separable questions:

1. does launching the same kernel as a `ClusterKernel` (co-scheduled CTAs) cost or save time?
2. does the cluster's real feature — **TMA multicast** (`T.copy_cluster(..., cluster_mask=)`,
   one CTA's DRAM load broadcast to the whole cluster, cutting operand traffic) — pay?

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 2000 MHz, 156 SMs
- 4096³ fp8 GEMM, `cluster_dims=2` (and `(2,1,1)`, `(4,1,1)`), `threads=256`
- Timing: CUDA events over 5 launches after 3 warm-ups

## Finding 1: a cluster launch with no multicast is free (a tie with plain)

| kernel | TFLOP/s |
|---|---|
| plain `T.Kernel` | 374.7 |
| `T.ClusterKernel`, cluster_dims=2 | 371.9 |
| `T.ClusterKernel`, cluster_dims=(2,1,1) | 373.9 |
| `T.ClusterKernel`, cluster_dims=(4,1,1) | 375.4 |

All within noise. Cluster co-scheduling neither helps nor hurts a compute-bound GEMM that does
not use cross-CTA shared memory. (This matches `block_scheduling_is_a_tie.md`: scheduling
structure that does not create work or cut DRAM traffic cannot move a `dram_pct≈13 %` kernel.)

## Finding 2: TMA multicast is numerically correct but ~60,000× slower — unusable

Same `ClusterKernel`, only the A-tile load changes from `T.copy` to
`T.copy_cluster(..., cluster_mask=3)`:

| load path | latency / launch | TFLOP/s | md |
|---|---|---|---|
| `T.copy` (A loaded per CTA) | **0.348 ms** | 395.3 | 0.125 |
| `T.copy_cluster(cluster_mask=3)` | **22 083 ms** | ≈0.0 | 0.125 |
| …with `tl.disable_warp_specialized=True`, stages=2 | 22 173 ms | ≈0.0 | 0.125 |
| …with `tl.disable_warp_specialized=True`, stages=0 | 24 895 ms | ≈0.0 | 0.125 |

The output is **bit-for-bit the same** as plain (the multicast does deliver the right data), but a
single launch takes ~22 seconds instead of 0.35 ms — a factor of ~6×10⁴. It is not a deadlock
(a deadlock would never return the correct result); the kernel completes, it just spends ~2 s per
launch spinning before its completion signal lands.

It is not a warp-specialization interaction: the slowdown reproduces with WS disabled and with the
pipeline off (stages=0, the simplest kernel). It is the multicast instruction path itself.

## Why: the emitted code, and the source's own admission

The generated CUDA is structurally right — but the barrier it signals is **CTA-local**, not
cluster-scoped:

```cuda
if (tl::block_rank_in_cluster() == 0) {
    tl::tma_load_multicast(A_desc, mbarrier[(k & 1)], ..., (uint16_t)(3), ...);
} else if (((3 >> tl::block_rank_in_cluster()) & 1) == 0) {
    tl::tma_load(A_desc, mbarrier[(k & 1)], ..., ...);           // never taken for rank 1, mask=3
}
tl::tma_load(B_desc, mbarrier[(k & 1)], ...);                     // B per CTA
```

`cp.async.bulk.tensor...multicast::cluster` (`copy_sm90.h`) completes against a barrier in the
**`shared::cluster`** address space. The pipeline materializes `mbarrier` in CTA-local shared
memory, so the cross-CTA completion backpressure is not expressible — and TileLang says so in its
own source (`src/cuda/transform/auto_schedule/role_based_scheduler.cc`):

> `// TODO: support warp specialization for cluster kernels — the materialized barriers are
> CTA-local, so cluster-wide multicast backpressure cannot be expressed yet.`

That TODO gates the auto-WS path, and the same CTA-local-barrier limitation shows up as the
pathological latency on the default pipeline path.

## When needed (trigger)

- **Do not use `T.copy_cluster(..., cluster_mask=…)` (TMA multicast) on sm120 with 0.1.15.** It is
  correct but ~4 orders of magnitude slower; there is no configuration measured here that makes it
  viable.
- **Do not use `T.ClusterKernel` as a speed move, either** — without multicast it ties plain, and
  the only reason to form a cluster (multicast / DSMEM / 2-SM MMA) is either broken-slow (multicast)
  or inapplicable (2-SM `tcgen05` MMA does not exist on this warp-level `HMMA` part).
- Clusters are architecturally present (they launch, `block_rank_in_cluster()` works), so this is a
  **codegen limitation, not a hardware absence** — worth re-probing after a TileLang upgrade.

## Minimal runnable example (the broken path, for reproduction only)

```python
import tilelang
import tilelang.language as T

M = N = K = 4096
BM = BN = BK = 128
DT = "float8_e4m3"


@T.prim_func
def mc(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT), C: T.Tensor((M, N), "float16")):
    with T.ClusterKernel(T.ceildiv(N, BN), T.ceildiv(M, BM),
                         cluster_dims=2, threads=256) as (bx, by):
        a_s = T.alloc_shared((BM, BK), DT)
        b_s = T.alloc_shared((BK, BN), DT)
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            T.copy_cluster(A[by * BM, k * BK], a_s, cluster_mask=3)   # <- the 22-second line
            T.copy(B[k * BK, bx * BN], b_s)
            T.gemm(a_s, b_s, acc)
        T.copy(acc, C[by * BM, bx * BN])


kernel = tilelang.compile(mc, out_idx=[2], target={"kind": "cuda", "arch": "sm_120a"})
# runs, returns the correct matrix, takes ~22 s per launch on an RTX 6000D.
```

## What TileLang decides for you here (and what it does not)

**Decided by the pass:** the producer-warpgroup split, the multicast-vs-plain dispatch per CTA rank
(`if block_rank==0 multicast else maybe plain`), and — the bug — that the barrier is CTA-local.

**Decided by you:** only `cluster_mask` and `cluster_dims`; there is no knob to put the barrier in
cluster scope, which is exactly the missing piece.

## Gotchas

- **Correct ≠ usable.** The multicast kernel’s `md` matches plain exactly; if you only check
  correctness you will ship a 22-second kernel. Always time it.
- **It is not WS.** Disabling warp specialization does not help (22 173 ms), and disabling the
  pipeline does not help (24 895 ms). The cost is in the multicast completion wait.
- **`cluster_dims=(2,1,1)` / `(4,1,1)` launch fine and tie plain** — the cluster machinery itself
  works; only the multicast data path is broken.
- **`block_rank_in_cluster()` is available** (`tilelang.cuda.language.cluster`), so a manual SM-to-SM
  `copy_cluster(..., dst_block=…)` with an explicit `remote_barrier` is the theoretical workaround;
  not attempted here and out of scope for this measurement.

## Sources

1. TileLang 0.1.15: `tilelang/language/copy_op.py` (`copy_cluster`, `cluster_mask` / `remote_barrier`),
   `tilelang/language/kernel.py` (`ClusterKernel`, `cudaLaunchAttributeClusterDimension`),
   `src/tl_templates/cuda/copy_sm90.h` (`tma_load_multicast` → `...multicast::cluster`),
   `src/cuda/transform/auto_schedule/role_based_scheduler.cc` (the CTA-local-barrier TODO quoted above).
2. `block_scheduling_is_a_tie.md` (why structure that does not cut DRAM is free on this kernel).
3. This repo's probes on an RTX 6000D, clock-locked, emitted-source + latency measured.
