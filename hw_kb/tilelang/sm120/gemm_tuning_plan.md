# SM120 (RTX 6000D) GEMM tuning plan — the consolidated result

> This is the English, single-document summary of the 15 measured axes in this directory. Every
> number is a hardware reading on an RTX 6000D, SM clock locked at 1995 MHz, TileLang 0.1.15,
> target `sm_120a`. Per-axis detail lives in the recipe files linked from the table.
>
> One line: **of 15 axes, 4 pay (fp8 2.4×, split-K up to 5.6×, 2:4 sparsity 1.95×, block-tile
> skipping ~linear), 1 is already the default, and the rest are ties, losses, or bugs.** A square
> dense GEMM is at the ceiling of its `(128,128)` tile geometry; further gains come from dtype,
> sparsity, or a different kernel class — not from scheduling.

## Measured on

- TileLang `0.1.15`, target `sm_120a`, NVIDIA RTX 6000D (driver 610.43.02, CUDA 13.3), 2 devices
- 156 SMs, `shared_memory_per_block_optin = 101376` B (99 KiB), warp-level `HMMA` (no wgmma/tcgen05)
- SM clock locked at 2000 MHz (`nvidia-smi -i 0 -lgc 2000,2000`; reads 1995)
- Timing: CUDA-graph replay / `do_bench` events over many iterations (see §Timing)

## The 15-axis table

| # | Axis | Verdict | Reading | Recipe |
|---|---|---|---|---|
| 1 | **dtype** | ✅ **2.4×** | f16 141 → fp8 e4m3 297 TFLOP/s (BK 32→128) | `gemm_dtype_throughput.md` |
| 2 | **split-K** | ✅ **up to 5.6×** (starved only) | 512²×16k: 49 → 277 TFLOP/s (SK=8) | `split_k_fills_a_starved_grid.md` |
| 3 | **2:4 sparsity** | ✅ **1.95×** | 4096³: 262 vs 134 TF via `T.gemm_sp` → Ampere `mma.sp` | `sparse_gemm_two_mechanisms.md` |
| 4 | **block-tile skipping** | ✅ **~linear** | 25/50/75 % tiles → 0.28/0.54/0.70× time | `sparse_gemm_two_mechanisms.md` |
| 5 | tile size BM/BN | ❌ wall | `(128,128)` optimal; `(256,256)` → 27 TF (17× bank conflicts) | `tiled_copy_scope_and_swizzle.md` |
| 6 | `num_stages` | ❌ plateau | 2→4 costs 33 KiB smem for +0.8 pts `tensor_pct` | `software_pipeline_stage_budget.md` |
| 7 | warp specialization | ❌ already on | explicit `role_based` 2–8× slower | `warp_specialization_is_already_on.md` |
| 8 | block scheduling (raster/L2 swizzle/persistent) | ❌ tie | 24 cells within ±2 % | `block_scheduling_is_a_tie.md` |
| 9 | Stream-K | ❌ absent + slower | no API in 0.1.15; hand-written 20–33 % below split-K | `stream_k_is_absent_and_loses.md` |
| 10 | eviction policy | ❌ no effect | three policies within ±8 % noise | `gemm_operand_layout_and_policy.md` |
| 11 | shared layout swizzle | ❌ no effect | `T.gemm` already swizzles | `tiled_copy_scope_and_swizzle.md` |
| 12 | warp policy (cooperative) | ✅ default | `Square` is default and reads 25 % fewer smem wavefronts | `gemm_operand_layout_and_policy.md` |
| 13 | cluster launch | ⚠️ free | `T.ClusterKernel` ties plain | `clusters_free_multicast_broken.md` |
| 14 | cluster TMA multicast | ❌ bug | 22 083 ms vs 0.348 ms (≈6×10⁴ slower, correct) | `clusters_free_multicast_broken.md` |
| 15 | CLC (Blackwell dynamic scheduling) | ❌ no-op | 0 CLC in SASS; corrupts output beside a barrier | `clc_compiles_to_nothing.md` |

## The four axes that pay

### dtype — fp8 = 2.4×

4096³, tile `(128,128,BK)`, stages=2: f16 141 → bf16 144 (≈ tie, a numerics choice) → fp8 e4m3
BK=128 **297** (≈2.1×), fp8 e5m2 BK=64 306. `cos = 1.0` on every row; fp8 `max_diff ~1e-4` is
`randn`-friendly only — validate real activations with cosine. fp8 does **not** reopen the tile
axis (BM/BN > 128 still collapses); its gain is per-`mma` byte count, not tile size.

### split-K — up to 5.6×, only on a starved grid

Trigger: `tiles = (M/BM)×(N/BN) < 156` (fill below 100 %). `SK ≈ ceil(156 / tiles)`, landing near
one wave; over-splitting reverses the win. Full grid → split-K is a loss (only adds reduction
cost). No `T.SplitK`; hand-write `T.Kernel(nt, mt, SK)` with `T.atomic_add` into fp32 C.
**Trap:** `out_idx` zeroes C only once; an `atomic_add` kernel is wrong from the second call —
zero C yourself every launch.

### 2:4 sparsity — 1.95×

`T.gemm_sp(A_sparse, E, B, C)` lowers to Ampere `mma_sp_sync` on this arch (not wgmma/tcgen05).
Contract: `A_sparse (M,K/2)`, `E (M,K/8) uint8`, logical `K = 2*A[-1]`. 262 vs 134 TF at 4096³.
**1.95× not 2×**; `BK=64` required (E buffer overflows smem at 128); a wrong `E` is silently wrong
(`0x00` → cos 1.0, `0x88` → 0.5, `0xAA` → 0). Note dense fp8 (~300–380 TF) already beats
f16-2:4-sparse (~262 TF).

### block-tile skipping — ~linear

Launch only nonzero output tiles, decode `(m,n)` from a compacted `int32` list. 25/50/75 % of tiles
→ 0.28/0.54/0.70× time, nonzero tiles bit-identical to dense. Pure grid shaping; composes with
dtype / split-K / tile size.

## Decision tree

```
Is the operand data 2:4 sparse?          -> yes: T.gemm_sp (1.95x, BK=64, build E correctly)
Is there a block-level nonzero mask?     -> yes: launch only nonzero tiles (~linear)
Does fp8 pass numerics?                  -> yes: fp8 e4m3 + BK=128 (2.4x, biggest single win)
grid fill = (M/BM)*(N/BN) / 156
  < 100%  (starved: 512^2x16k, LM head, decode) -> split-K, SK ~= ceil(156/tiles), ~one wave
  >= 100% (square large GEMM)                   -> stop; split-K hurts; ~370-395 TF is the cap
Everything else is closed: (128,128) tile, stages=2, no WS, no L2 swizzle/persistent,
no eviction hint, no cluster multicast, no CLC, no hand-written Stream-K.
```

## Recommended base kernel

```python
import tilelang, tilelang.language as T
M = N = K = 4096
BM = BN = BK = 128
DT = "float8_e4m3"

@T.prim_func
def gemm(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT), C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=256) as (bx, by):
        a_s = T.alloc_shared((BM, BK), DT); b_s = T.alloc_shared((BK, BN), DT)
        acc = T.alloc_fragment((BM, BN), "float32"); T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            T.copy(A[by * BM, k * BK], a_s); T.copy(B[k * BK, bx * BN], b_s)
            T.gemm(a_s, b_s, acc)
        T.copy(acc, C[by * BM, bx * BN])

kernel = tilelang.compile(gemm, out_idx=[2], target={"kind": "cuda", "arch": "sm_120a"})
# ~370-395 TFLOP/s. Only fp8, split-K (starved) and sparsity improve on it.
```

## Timing

One kernel, three answers: single-launch event median 122, NCU `gpu__time_duration` 265 (L2 warm),
**CUDA-graph replay 137 (sustained)**. Tune on graph replay; use NCU only for *why*; lock the clock
(±30 % noise otherwise). See `gpu_probe_ladder.md`.

## Three traps that produced wrong answers first

1. **`out_idx` + `atomic_add`** — C zeroed only on the first call; run 2 onward accumulates onto the
   previous result (`md 3e-4 → 6e2`, `cos 1.0 → 0.5`). Zero C yourself for any accumulating kernel.
2. **multicast is correct but 6×10⁴ slower** — `md` matches plain; a correctness-only check ships a
   22-second kernel. Always time it.
3. **three timing methods give three numbers** — tune on graph replay, never on single-launch medians.

## Arch facts that make the map what it is

- `sm_120` compiles-lowers then fails at `nvcc`; `sm_100a` compiles but will not load. Only
  `sm_120a` runs — check this first on any compile/load failure.
- Math is warp-level `HMMA`. That is *why* warp specialization subtracts compute warps here (default
  `T.Pipelined` already appends a producer warpgroup: `threads=256` → `__launch_bounds__(384)`),
  cluster 2-SM MMA has nothing to attach to, and CLC falls back to an empty `static_assert`.
- The card is **compute-bound** (`tensor_pct ≈ 72 %`, `dram_pct ≈ 13 %`), so every axis that only
  reorders or cuts DRAM/cache traffic is a tie. Only axes that cut math cost (dtype, sparsity) or
  fill idle SMs (split-K) win.

## Version-blocked axes to re-probe after an upgrade

CLC only needs `CUTLASS_ARCH_CLC_ENABLED` defined for `sm_120a` (the arch list already names
SM120A); cluster multicast only needs a cluster-scope barrier. Both are one fix away — re-measure
after bumping TileLang.

## Sources

1. The 15 recipe files in this directory; every number is a hardware reading.
2. TileLang 0.1.15 installed package (the pass source cited in each file).
3. `gpu_probe_ladder.md` for the measurement protocol.
