# SM120 (RTX 6000D) GEMM tuning: the whole map on one page

Everything below is measured on hardware (RTX 6000D, driver 610.43.02, CUDA 13.3, TileLang 0.1.15,
`sm_120a`), SM clock locked at 2000 MHz. This page is the decision tree over the 10 recipes in this
directory; each row links to the file with the numbers.

## Measured on

- TileLang `0.1.15`, target `sm_120a`, NVIDIA RTX 6000D, driver 610.43.02, CUDA 13.3, 156 SMs
- SM clock locked at 2000 MHz; timing by CUDA-graph replay / `do_bench` events over many iterations
- This is a summary page: every number is copied from the linked recipe, which carries its own
  measured configuration (dtype, tile, stages, problem shape).

## The verdict table: 8 axes tried, 2 pay

| # | Axis | Verdict | Best / worst reading | Recipe |
|---|---|---|---|---|
| 1 | **dtype** | ✅ **2.4×** | f16 141 → fp8 e4m3 297 TFLOP/s (BK 32→128) | `gemm_dtype_throughput.md` |
| 2 | **split-K** | ✅ **up to 5.6×** (starved grids only) | 512²×16k: 49.1 → 277.4 TFLOP/s at SK=8 (fp32 C, locked) | `split_k_fills_a_starved_grid.md`, `stream_k_is_absent_and_loses.md` |
| 3 | tile size (BM/BN) | ❌ wall | `(128,128)` optimal; `(256,256)` → 27 (−17×, 2.5e5 bank conflicts) | `tiled_copy_scope_and_swizzle.md` |
| 4 | `num_stages` | ❌ plateau | 2 → 4 costs 33 KiB smem for +0.8 pts `tensor_pct` | `software_pipeline_stage_budget.md` |
| 5 | warp specialization | ❌ already on | explicit `role_based` 2–8× **slower** (fp8 381→170, f16 157→20) | `warp_specialization_is_already_on.md` |
| 6 | block scheduling | ❌ tie | raster / L2 swizzle / persistent all ±2 % (24 cells) | `block_scheduling_is_a_tie.md` |
| 7 | eviction policy | ❌ no effect | three policies within ±8 % noise (compute-bound) | `gemm_operand_layout_and_policy.md` |
| 8 | shared layout swizzle | ❌ no effect | `T.gemm` already swizzles; manual adds nothing | `tiled_copy_scope_and_swizzle.md` |
| 9 | thread-block clusters | ⚠️ launch free, multicast broken | cluster ties plain; `copy_cluster` 22 083 ms vs 0.348 ms | `clusters_free_multicast_broken.md` |
| 10 | how you time it | ✅ protocol | graph replay 137 vs single-event median 122 vs NCU 265 for the same kernel | `gpu_probe_ladder.md` |
| 11 | **Stream-K** | ❌ absent, loses to split-K | no API in 0.1.15; hand-written is 20–33 % below split-K on starved grids | `stream_k_is_absent_and_loses.md` |
| 12 | **CLC** (Blackwell dynamic task pool) | ❌ compiles to a no-op | `static_assert(dependent_false)` fallback; 0 CLC in SASS; corrupts output if paired with a barrier | `clc_compiles_to_nothing.md` |
| 13 | cooperative warp-group | ✅ already default | `GemmWarpPolicy.Square` is the default and reads 25 % fewer smem wavefronts | `gemm_operand_layout_and_policy.md`, `clc_compiles_to_nothing.md` |
| 14 | **2:4 sparsity (fine)** | ✅ **~1.95×** on genuinely sparse data | `T.gemm_sp` → Ampere `mma.sp`; 4096³ 262 vs 134 TF; BK=64 (E buffer caps smem) | `sparse_gemm_two_mechanisms.md` |
| 15 | **block-tile skipping (coarse)** | ✅ **~linear** time saving | launch only nonzero tiles; 25/50/75 % → 0.28/0.54/0.70× time, tile-exact | `sparse_gemm_two_mechanisms.md` |

## Decision tree

```
Is the GEMM's tile grid (M/BM)*(N/BN) >= num_sms (156)?
├─ YES (well-filled, e.g. 4096³, 8192²×1k)
│     → split-K HURTS (372 → 144 monotone). Do not.
│     → dtype: use fp8 e4m3 if numerics allow  (the only 2×+ win)
│     → everything else is closed: (128,128) tile, stages=2, no WS, no scheduler knob.
│     → you are done; ~370-395 TFLOP/s is the ceiling for this tile geometry.
│
└─ NO (tile-starved, e.g. 512²×16k = 10 % fill, 1024²×8k = 41 % fill)
      → split-K is the fix.  SK ≈ ceil(156 / num_tiles), land near ~100 % fill.
        512²×16k  → SK=8 : 49 → 277 TFLOP/s (5.6x)
        768²×8k   → SK=4 : 107 → 293 TFLOP/s
        1024²×8k  → SK=2 : 190 → 323 TFLOP/s
      → do NOT over-split: past ~one wave it reverses (SK=16 drops).
      → dtype still applies (fp8), and split-K + fp8 combine.
      → NOT Stream-K: same idea, but it flushes per tile so it lands 20–33 % under
        split-K (222 vs 277, 232 vs 293, 267 vs 323). Skip it.
```

## The three traps that produced wrong answers first

1. **`out_idx=[2]` + `atomic_add` = wrong after the first call.** The output is zeroed only on the
   first launch; split-K then accumulates onto the previous result (`run1 md=3e-4`, `run2 md=6e2`,
   `cos 1.0 → 0.5`). Zero C yourself every launch. Cost more debugging time than any perf question
   in this directory.
2. **Multicast is correct but unusably slow** — `md` matches plain exactly, so a correctness-only
   check ships a 22-second kernel. Always time it.
3. **Three timing methods, three numbers for one kernel** (224 / 122 / 265 / 137 TFLOP/s). Tune on
   **CUDA-graph replay**; use NCU only to answer *why*. An unlocked clock makes ±30 % look like
   signal.

## Non-obvious arch facts, restated

- `sm_120` compiles-lowers then **fails at nvcc**; only `sm_120a` runs. `sm_100a` compiles but will
  not load (`NO_BINARY_FOR_GPU`). This is the first thing to check on a compile/load failure.
- The math is **warp-level `HMMA`**, not wgmma/tcgen05. That is *why* warp specialization is a
  subtraction of compute warps here, and why cluster 2-SM MMA has nothing to attach to.
- `T.Pipelined` already appends a producer warpgroup (`threads=256` → `__launch_bounds__(384)`);
  `threads=` is the *consumer* count.
- The card is **compute-bound** (`tensor_pct ≈ 72 %`, `dram_pct ≈ 13 %`). Any axis that only
  reorders or shrinks DRAM/cache traffic (scheduling, swizzle, eviction) is therefore free-to-tie.
  The two axes that pay are the ones that cut the *math* cost (dtype) or fill idle SMs (split-K).

## Minimal fast kernel (the default recommendation)

```python
import tilelang
import tilelang.language as T

M = N = K = 4096
BM = BN = BK = 128                 # (128,128) tile; BK=128 for fp8
DT = "float8_e4m3"


@T.prim_func
def gemm(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT), C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=256) as (bx, by):
        a_s = T.alloc_shared((BM, BK), DT)
        b_s = T.alloc_shared((BK, BN), DT)
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            T.copy(A[by * BM, k * BK], a_s)
            T.copy(B[k * BK, bx * BN], b_s)
            T.gemm(a_s, b_s, acc)
        T.copy(acc, C[by * BM, bx * BN])


kernel = tilelang.compile(gemm, out_idx=[2], target={"kind": "cuda", "arch": "sm_120a"})
# ~370-395 TFLOP/s on an RTX 6000D. Adding WS / swizzle / persistent / cluster / bigger tile
# / more stages all measure the same or worse. fp8 + split-K (if starved) are the only improvements.
```

## Sources

1. The ten recipe files in this directory; every number is a hardware reading.
2. TileLang 0.1.15 installed package (the pass source cited in each file).
3. `gpu_probe_ladder.md` for the measurement protocol that produced these numbers.
