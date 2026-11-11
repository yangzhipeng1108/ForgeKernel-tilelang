# Tuning the sm120 NVFP4 GEMM: what moves the number and what does not

Measured on RTX 6000D (sm_120a), cutlass-dsl 4.8.0, `DenseGemmKernel` from
`flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x`.

The short version: **128x128 x tile_k 128 is already the optimum**, the stage count
cannot be raised (hard-capped at 4 in the framework), and the 4096^3 drop is wave
quantization, not tiles or L2. The one real, free win is `max_active_clusters` = 128
for the N=4096 shapes.

## Tile shape sweep (M,N,K = 4096^3 | 2048x4096x4096)

| mma_tiler_mn | 4096^3 | 2048x4096x4096 | verdict |
|---|---|---|---|
| **128 x 128** | 508 TF/s | **850 TF/s** | **best overall** |
| 128 x 64 | 512 TF/s | 705 TF/s | only for very tall M |
| 64 x 128 | 511 TF/s | 705 TF/s | same, and md degrades (2.75) |
| 128 x 256 | 32 TF/s | 34 TF/s | **collapses** — see below |
| 256 x 128 | 34 TF/s | 36 TF/s | collapses |
| 256 x 256 | runtime error | runtime error | not usable |
| 128 x 192 / 192 x 128 | compile error | compile error | op needs 128-aligned tiles |

The 128x256 and 256x128 **15x collapse** is not a slow tile, it is a dead pipeline:
the larger tile leaves room for a single A/B stage, so the mainloop degenerates to
load-run-load with no overlap. On a 99 KB smem budget a 128x256 fp4 tile plus its
SF blocks and epilogue already spends most of the budget.

`192` tiles fail to compile because the block-scaled MMA atom is 128 rows wide and
the SF partition helpers assert whole 128-row blocks.

## Stage count cannot be swept

`DenseGemmKernel._compute_stages` ends with `ab_stage = max(1, min(raw_ab_stage, 4))`.
The cap is hard-coded, so no constructor argument reaches it. Raising it means
patching the framework source; `epi_stage` is capped at 4 the same way.

## Why 4096^3 is ~40% slower than 2048x4096x4096 (it is not N, not L2, not tiles)

Global-memory sweeps, M=4096 K=4096 unless noted, `max_active_clusters=156`:

| shape | B matrix | TF/s |
|---|---|---|
| N=1024 | 2 MB | 832 |
| N=2048 | 4 MB | **885** |
| N=3072 | 6 MB | 767 |
| N=4096 | 8 MB | 505 |
| 4096x2048x4096 | 8 MB | 838 |
| 8192x4096x4096 | 8 MB | 508 |
| 4096x4096x2048 | 4 MB | **953** |

B never exceeds 8 MB, far under L2, so a cache explanation does not fit. M from
2048 to 8192 changes nothing. The variable that matters is the **tile count versus
the 156 SMs**:

- 2048x4096 -> 16 x 32 = 512 tiles / 156 = 3.28 waves
- 4096x4096 -> 32 x 32 = 1024 tiles / 156 = **6.56 waves**

6.56 waves means the last wave runs 1024 - 6*156 = 88 CTAs on 156 SMs, i.e. a
**44%-idle tail**. That is wave quantization, and it is why the number lands near
505: 6 full waves of work plus a 56%-occupied seventh.

## max_active_clusters: the one free win

| shape | mac=64 | **mac=128** | mac=156 (default) | mac=312 | mac=624 |
|---|---|---|---|---|---|
| 2048x4096x4096 | 238 | **882** | 850 | 860 | 826 |
| 4096x2048x4096 | 236 | **886** | 838 | 855 | 824 |
| 4096x4096x2048 | 264 | 869 | **953** | 927 | 901 |
| 4096^3 | 236 | 471 | **505** | 506 | 506 |
| 8192x4096x4096 | 218 | 472 | 508 | 506 | **512** |

- **mac=128 beats the default 156 on every N=2048 / N=4096-K=2048 shape** (+4% at
  2048x4096x4096). 128 is a clean power of two, so the scheduler's tile waves divide
  evenly.
- mac=64 is always catastrophic (2-4x slower): half the SMs idle.
- For the 4096-deep shapes, 156 or above wins; 128 under-fills.

Recommended: **mac = 128 when N <= 2048 or K <= 2048, else 156**.
Already the case in `kernels/gemm/nvfp4_gemm/sm120/kernel.py`.

## Options that change nothing (measured, 4096^3 / 2048x4096x4096)

| option | 4096^3 | 2048x4096x4096 |
|---|---|---|
| `use_prefetch=True` | 505 | 851 |
| `enable_pdl=False` | 505 | 848 |
| `swap_ab=True` | 506 | 808 (worse) |
| `tile_k=128` | 506 | 851 |
| `tile_k=256` | 512 | 528 (worse) |
| `load_path="cpasync"` | 505 | 562 (worse) |

## The trap: `single_work_tile_per_cta` reports 5669 TF/s and is wrong

| shape | reported | md |
|---|---|---|
| 4096^3 | **5669 TF/s** | **1.02e+03** |
| 2048x4096x4096 | **2954 TF/s** | **936** |

It computes one tile per CTA and returns, so it is 10x "faster" and returns garbage.
Any throughput figure above ~1000 TF/s on this card is this bug — same class as the
split-K `M == 1` fabrication in
[`split_k_only_when_m_is_small`](split_k_only_when_m_is_small.md). Always gate a
timing run on `md`.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB, 156 SMs
- `lmsysorg/sglang:dev` container: cutlass-dsl **4.8.0**, torch 2.14.1+cu130
- `DenseGemmKernel(sf_vec_size=16, mma_tiler_mn=..., cluster_shape_mn=(1,1))`, NVFP4
  operands with unit UE4M3 scales, bf16 output, 30 timed iterations after a warmup
