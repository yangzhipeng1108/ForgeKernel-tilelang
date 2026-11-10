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

CLC needs TileLang's `src/tl_templates/cuda/cluster.h` to `#include <cutlass/arch/config.h>` — on
`sm_120a` the arch supports CLC (a device-side probe shows `CLC_ENABLED=1`), but cluster.h omits the
header so the macro is undefined in the kernel TU. This is a TileLang include bug, not an arch limit;
fixable by a TileLang source change (not by any user compile flag). Cluster multicast needs a
cluster-scope barrier (`role_based_scheduler.cc:523`: "materialized barriers are CTA-local"). Both
wait on a TileLang bump that touches those two files.

**Still blocked as of this writing**: `pip index versions tilelang` on the probe box reports
**0.1.15 as the newest release** (older: 0.1.14 … 0.1.6.post2), and the installed package *is* 0.1.15.
So there is **nothing to upgrade to yet** — the CLC no-op and the 6×10⁴-slower multicast stand until a
future release ships the `sm_120a` CLC define / a cluster-scope multicast barrier. Do not spend a probe
cycle re-testing them on 0.1.15; re-check `pip index versions tilelang` first, and only re-run the two
recipes (`clc_compiles_to_nothing.md`, `clusters_free_multicast_broken.md`) when the version moves.

## Attention, decode and LM head (added after the GEMM map)

The 15-axis GEMM map above is the core, but the same measurement discipline was extended to the other
kernels. The one-line summary of the whole map, GEMM included:

| kernel | bottleneck | what wins | what ties |
|---|---|---|---|
| dense GEMM (`gemm_*`) | tensor core, `dram 13 %` | fp8 2.1×, split-K on starved grids, 2:4 sparse 1.95× | schedule / WS / cluster / CLC |
| prefill attention, non-causal (`flash_attention_is_compute_bound_too.md`) | tensor core, `dram 8.8 %` | fp8 QK^T 1.2–1.4×, small-SEQ grid shaping (1.36×) | schedule / WS / register-cut |
| **prefill attention, causal** (`causal_attention_skips_the_masked_half.md`) | tensor core | **tile-skip 1.95×, SDPA parity**; **fp8 QK^T 1.2–1.4×** (`fp8_qk_ports_split_k_ties.md`) | split-K |
| decode MHA (`decode_is_the_one_bandwidth_bound_case.md`) | **HBM 90 %** | (nothing — 880 GB/s wall) | schedule / split-KV / fp8-KV (all tie or lose) |
| decode MLA (`mla_absorbed_decode_beats_mha.md`) | tensor core (after absorb) | **latent cache, 3.8× over same-shape MHA / 1.16× over GQA/MQA**; `Dc=512` two-pass 96.3 TF | — |
| decode GQA/MQA vs MLA (`mla_absorbed_decode_beats_mha.md`, `decode_is_the_one_bandwidth_bound_case.md`) | tensor core / HBM | GQA/MQA sharing (1.40× vs SDPA); MLA only when MHA & high `HQ` | fp8 QK (+4%) |
| LM head (`lm_head_is_wide_not_starved.md`) | tensor core → HBM at `M16` | **vocab block-sparse 2.4–4× end-to-end** | split-K (−18 %), schedule |
| porting levers to attention (`fp8_qk_ports_split_k_ties.md`) | — | fp8 QK^T with native fp16 V | split-K (tie) |

The single rule that survives every kernel: **a lever pays only if it reduces work/bytes on the binding
resource, and only if nothing else already did.** Every win in the table is a work/byte reduction
(fp8, split-K-on-starved-grid, 2:4, causal skip, MLA absorb); **no re-schedule axis — swizzle,
persistent, warp specialization, cluster, CLC, L2, eviction — has ever paid, in any kernel, at either
bottleneck.**

## Sources

1. The 21 recipe files in this directory; every number is a hardware reading.
2. TileLang 0.1.15 installed package (the pass source cited in each file); `pip index versions tilelang`
   confirms 0.1.15 is newest, so the CLC/cluster axes stay blocked.
3. `gpu_probe_ladder.md` for the measurement protocol.
