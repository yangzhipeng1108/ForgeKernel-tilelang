# SM120 (consumer Blackwell, RTX 6000D / RTX 50-series) interface index

Demand-driven, exactly like `sm90/` and `sm100/`: a recipe is written when a student is
blocked on that interface, follows `../TEMPLATE.md`, and is marked verified **only** on a
GPU probe that returned `PROBE PASS` on an sm120 part. This file is a ledger of what has been
earned on this architecture, not a plan.

## Why this tree exists at all, and what it is NOT

The RTX 6000D is **compute capability (12, 0) = sm_120a**. That is *consumer* Blackwell, and
most of `sm100/` does not apply to it. The single most useful fact in this tree:

> **sm120 has no `wgmma` and no `tcgen05`.** Its tensor-core path is warp-level `mma.sync`
> (HMMA) plus the warp-level block-scaled family (`mx*.block_scale`). Any `sm100/` entry
> whose mechanism is "issue the UMMA / allocate TMEM / drive the 2-CTA pair" is **not
> applicable**, no matter how good the entry is.

The DSL says so itself when you try: constructing `tcgen05.MmaMXF4NVF4Op` on this card raises

```
expects arch to be one of [sm_100a, sm_100f, sm_103a, sm_107a, sm_110a], but got sm_120a
suggestion: tcgen05 MMA requires datacenter Blackwell (sm_100 / sm_103).
            On consumer Blackwell (sm_120 / sm_121) use the warp-level
            cute.nvgpu.warp.MmaMXF4Op / MmaMXF4NVF4Op / MmaMXF8Op
```

So this tree is **not** a renamed copy of `sm100/`. Nothing is pre-filled from it. The entries
here name mechanisms that exist only, or differently, on sm120.

## Find by symptom first

| What you see / what you are about to do | Look first at |
|---|---|
| **Want a correct NVFP4 GEMM right now**, or your own fp4 kernel compiles and runs but every number is NaN / the CuTeDSL compiler core-dumps with no Python traceback | [`nvfp4_gemm_computes_correctly`](nvfp4_gemm_computes_correctly.md): the shipped kernel that is exact on this card, the rank-3 SF tensor requirement that causes the `(128,128) vs (2,1,3)` error, and the five silent-corruption traps (thread-count tiler core-dump, `Field.SFA` rejection, rank mismatch in list operands, sub-byte store, swizzled smem vs universal copy) |
| **"Does sm120 support X?"** for any primitive; or you are about to read an `sm100/`/`sm90/` entry and try its mechanism here | [`capability_census`](capability_census.md): the 39-row table of every warp-MMA / copy / TMA / cluster symbol, with the five families that are **absent** (tcgen05, wgmma, int8 MMA, 16x16x8b ldmatrix, and the sm100 smem size), and the real TMA constructor names (`make_tiled_tma_atom`, not `make_tma_copy`) |
| **Want MXFP4 or MXFP8** (UE8M0 scales), or you have NVFP4 working and need to know the difference between the three block-scaled instructions | [`blockscaled_three_paths`](blockscaled_three_paths.md): NVFP4 / MXFP4 / MXFP8 in one kernel keyed by `sf_vec_size`, all three `md=0` (MXFP8 1.85e-3 from the fp16 store), and the pre-permuted SF layout that silently corrupts if fed row-major |
| **Plain fp16 / bf16 GEMM**, or fp8 fails with `expects the 'ab_dtype' Op parameter to be one of Float16 or BFloat16` | [`warp_mma_bf16_and_fp16`](warp_mma_bf16_and_fp16.md): fp16 and bf16 PASS from 512³ to 8192³; fp8 is **not** in the sm120 example (a correction); and the shipped bf16 skinny-M kernel that does not compile against cutlass 4.8.0 |
| **A ported kernel now needs too much smem** / `requires N bytes of shared memory`; or `make_tma_copy` is "missing"; or a copy atom constructs but fails IR verify | [`tma_and_copy_family`](tma_and_copy_family.md): **101376 B** smem (vs 232448 on sm100), the real TMA builder names, and `LdMatrix16x16x8bOp` constructing then failing verification |
| **Driving a GEMM through TMA end to end**, or you want the five-step descriptor→prefetch→partition→copy→store recipe the shipped sm120 example uses | [`tma_gemm_end_to_end`](tma_gemm_end_to_end.md): fp16 PASS at 1024³/2048³/4096³ (peak 208 TFLOP/s), and the exact `make_tiled_tma_atom` / `prefetch_descriptor` / `tma_partition` / `cute.copy` sequence |
| **Enabling split-K** to fill a starved grid, or a GEMM with `split_k_slices > 1` that runs fast but returns wrong numbers | [`split_k_only_when_m_is_small`](split_k_only_when_m_is_small.md): **correct iff M ≤ 128** — the split-K grid hardcodes the M dimension to 1 so it drops every M-tile above the first (cos → 1/√M_tiles at a fantasy TF/s). Also the bf16-only atomic C and the compile-in-the-timing-loop trap |
| **Planning to use clusters, or to split work across the two GPUs** | [`clusters_and_two_gpus`](clusters_and_two_gpus.md): clusters boot to (16,1) on one card, but are device-local by definition — no cross-card multicast exists here; P2P works at **45.5 GB/s** (SYS, no NVLink) |
| **Running tensor-parallel across the two cards**, or an `all_reduce` that is far slower than the link should allow | [`two_card_parallelism`](two_card_parallelism.md): column-parallel NVFP4 GEMM is **1.93x of 2x** (no per-layer collective); but **NCCL defaults to Socket (~12 GB/s)** and only reaches ~34 GB/s (all_reduce) / ~44 GB/s (send) once `NCCL_P2P_LEVEL=SYS` forces the PCIe peer path |
| **Splitting a layer's input dim (row-parallel TP) expecting ~2x**, or TP=2 turned out *slower* than one card | [`row_parallel_tp_costs_the_allreduce`](row_parallel_tp_costs_the_allreduce.md): measured **0.36x–0.68x** at every M — the all-reduce eats 66–86% because the fp4 GEMM (~850 TF/s) vastly outruns the link (~34 GB/s). TP=2 here is a capacity decision (130 GB > 85 GB), not a speed one |
| **About to run NVFP4 (`e2m1` x `ue4m3` scale) on the 6000D**; or a kernel that should be fp4 compiles but the numbers are garbage with no hang; or you want to know why TileLang's `T.gemm_blockscaled` gives a wrong answer here; or `tcgen05.MmaMXF4NVF4Op` refuses sm120 and you do not know what to use instead | [`nvfp4_warp_mma_works`](nvfp4_warp_mma_works.md): the instruction `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X`; `MmaMXF4NVF4Op` constructs on sm_120a with shape `(16,8,64)`, sf_vec 16; **the SF is a pointer set on the trait (`Field.SFA/SFB`), never a `cute.gemm` argument** — that is the exact difference from the broken TileLang path; the closed-form acceptance probe (A=B=1.0, SF=2.0, K=256 -> 1024) |

## Status of the known sm120 interfaces

| Interface | State on this card | Where |
|---|---|---|
| `cute.nvgpu.warp.MmaMXF4NVF4Op` (NVFP4) | **op verified**, full GEMM not yet gated | [`nvfp4_warp_mma_works`](nvfp4_warp_mma_works.md) |
| NVFP4 GEMM, end to end | **verified exact on this card**: `md=0`, `cos=1.0` across 128..256 x 128..256 x 128..1024, closed-form and random; use the shipped sm120 kernel rather than hand-building the smem plumbing | [`nvfp4_gemm_computes_correctly`](nvfp4_gemm_computes_correctly.md) |
| NVFP4 GEMM, as a shippable module | `kernels/gemm/nvfp4_gemm/sm120/kernel.py` (CuTe DSL, `.py`) wraps the verified kernel; **855 TF/s** at 2048×4096×4096, **505** at 4096³, md=2; beats TileLang fp8 375 / f16 158.8 | [`nvfp4_gemm_computes_correctly`](nvfp4_gemm_computes_correctly.md) |
| `cute.nvgpu.warp.MmaMXF8Op` (fp8, `(16,8,32)`, UE8M0 scale) | **MXFP8 GEMM verified**: `md=1.85e-3`, `cos=1.0` | [`blockscaled_three_paths`](blockscaled_three_paths.md) |
| `cute.nvgpu.warp.MmaMXF4Op` (fp4, `(16,8,64)`, UE8M0 scale) | **MXFP4 GEMM verified exact**: `md=0` | [`blockscaled_three_paths`](blockscaled_three_paths.md) |
| `cute.nvgpu.warp.MmaF16BF16Op` (fp16 / bf16) | **verified PASS** at 512³, 2048³, 8192³ via the shipped GeForce example | [`warp_mma_bf16_and_fp16`](warp_mma_bf16_and_fp16.md) |
| `cute.nvgpu.warp.MmaFP8Op` (plain fp8, `(16,8,32)`) | op constructs (arch-neutral) but **no shipped sm120 kernel drives it**; no per-tensor-scaled fp8 GEMM here | [`warp_mma_bf16_and_fp16`](warp_mma_bf16_and_fp16.md) |
| `cpasync.*` TMA + bulk-copy family | **all present**; constructor is `make_tiled_tma_atom`, **not** `make_tma_copy`; full fp16 GEMM through TMA **PASS** at 1024³–4096³ | [`tma_and_copy_family`](tma_and_copy_family.md), [`tma_gemm_end_to_end`](tma_gemm_end_to_end.md) |
| split-K (`split_k_slices > 1`) | **correct only when M ≤ 128** (one M-tile); ~1.9× on a starved grid, silently wrong above that | [`split_k_only_when_m_is_small`](split_k_only_when_m_is_small.md) |
| cluster / barrier primitives | **clusters boot to (16,1)**; single-device only | [`clusters_and_two_gpus`](clusters_and_two_gpus.md) |
| two-card parallel (NCCL + TP GEMM) | **column-parallel 1.93x**; NCCL all_reduce ~34 GB/s but **only with `NCCL_P2P_LEVEL=SYS`** (default Socket ~12 GB/s) | [`two_card_parallelism`](two_card_parallelism.md) |
| row-parallel TP | **0.36x–0.68x — slower than one card**; all-reduce 66–86% of the time; TP=2 is for capacity, not speed | [`row_parallel_tp_costs_the_allreduce`](row_parallel_tp_costs_the_allreduce.md) |
| `cutlass.utils.blockscaled_layout.sm120_make_smem_layout_sfa/sfb` | exist; the pair to use on sm120 (the `*_tmem_*` ones are sm100) | [`nvfp4_warp_mma_works`](nvfp4_warp_mma_works.md) |
| `tcgen05.*` (UMMA, TMEM, 2-CTA) | **not present on this card** — rejected at construction | [`capability_census`](capability_census.md) |
| `wgmma.*` (Hopper) | **not present on this card** — namespace absent | [`capability_census`](capability_census.md) |

## Official examples, by architecture (this saves the most time)

The CUTLASS DSL examples split by target and pick the wrong one silently looks like a bug:

| example | target | runs on 6000D? |
|---|---|---|
| `CuTeDSL/blackwell/tutorial_gemm/nvfp4_gemm_0.py` | tcgen05 / sm100 | **no** — refuses sm120 |
| `CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py` | sm103 | **no** |
| `CuTeDSL/blackwell_geforce/dense_gemm.py` | sm120 | **yes** — fp16 **and bf16** verified PASS here; but **fp16/bf16 only**: `--a_dtype Float8E4M3` fails with `expects the 'ab_dtype' Op parameter to be one of Float16 or BFloat16`. See [`warp_mma_bf16_and_fp16`](warp_mma_bf16_and_fp16.md) |

So there is **no shipped fp4 *example* for this card** — but the image does ship a working fp4
*kernel*: `flashinfer.gemm.kernels.sm12x_raw.Sm120BlockScaledGemmKernel`, verified exact here.
See [`nvfp4_gemm_computes_correctly`](nvfp4_gemm_computes_correctly.md). The same family's
`dense_blockscaled_gemm_sm120_b12x.DenseGemmKernel` also covers MXFP4 and MXFP8 — see
[`blockscaled_three_paths`](blockscaled_three_paths.md).

## What is deliberately absent

- **No cross-GPU multicast / multimem entry.** Clusters themselves are real here
  ([`clusters_and_two_gpus`](clusters_and_two_gpus.md)) but a cluster is device-local by
  definition, and the two cards are SYS-linked (no NVLink), so there is no NVLink multimem to
  write about — unreachable, not unmeasured.
- **No plain per-tensor-scaled fp8 GEMM.** The op exists; no shipped sm120 kernel drives it,
  and the example rejects fp8 outright. MXFP8 (block-scaled) is the fp8 path that works.
- No `sm100` content copied over. If a mechanism is arch-neutral (DSL algebra, probe
  methodology) it belongs in every tree, but it has to be **re-earned or explicitly
  re-certified** here with a "Certification source" line — location is not provenance.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB per card, no NVLink between them
- `lmsysorg/sglang:dev` container: cutlass **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- TileLang readings quoted in this tree come from `~/tlvenv` (tilelang 0.1.15)
