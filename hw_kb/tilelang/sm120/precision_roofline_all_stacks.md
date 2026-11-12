# Every precision on this box, three stacks, one protocol

> Status: **measured on hardware**, RTX 6000D (`sm_120a`, 156 SMs), free-running clocks.
> Every number is a 4096³ matmul unless the shape column says otherwise, timed with a plain
> wall-clock loop (warmup 5, 40 iterations, one `synchronize` at each end). The correctness
> column is a cosine against an fp32 reference, so a fast-but-wrong kernel cannot pass.

## Purpose

The stock question is "what is this card's TF/s per precision, and what fraction of nameplate
do we reach". Answering it earlier went wrong three separate times, each for a different
reason, so this note pins the whole table down at once:

1. the nameplate row `74 / 148 / 296 / 593` was read as FP64/FP32/**FP16-dense**/FP16-sparse,
2. `do_bench`'s default `backend="event"` reports fp8 ~42 % high on this part,
3. NVFP4 numbers were compared across different shapes (M=2048 vs 4096³).

With those three fixed, the tensor-core rows all land at 85–97 % of the derived peak and the
table is internally consistent — that consistency is the evidence that the peak column is
right. Once fp64/fp32 were actually measured, a fourth problem surfaced: **the plate's `74`
cannot be double-precision throughput** (measured 1.06 TF/s, a 70× gap), so the bottom of the
`74 / 148 / 296 / 593` chain is not what it looks like either. See "Where the nameplate sits".

## Measured on

- TileLang `0.1.15` (host venv) and NVIDIA cuBLAS / CuTeDSL (inside `lmsysorg/sglang:dev`)
- `sm_120a`, 156 SMs, 600 W cap, clocks free-running (load rises to ~2385 MHz and hits the cap)
- 4096³, operands dtype as listed, fp32 or fp16 accumulate, cosine gate against fp32

| precision | implementation | TF/s | % of peak | notes |
|---|---|---|---|---|
| FP16 dense | **CuTeDSL (hand)** | **143.1** | 97 % | fastest fp16 on the box |
| FP16 dense | cuBLAS | 133.1 | 90 % | vendor reference |
| FP16 dense | TileLang `T.gemm` | 131.1 | 89 % | BK=64, stages=2 |
| BF16 dense | TileLang `T.gemm` | 132.2 | 89 % | same as f16, as expected |
| TF32 | cuBLAS | 65.1 | 96 % | half of fp16 |
| FP8 e4m3 | **cuBLAS `_scaled_mm`** | **285.3** | 96 % | fastest fp8 on the box |
| FP8 e4m3 | CuTeDSL (hand) | 281.1 | 95 % | `MmaFP8Op (16,8,32)` caps it |
| FP8 e4m3 | TileLang `T.gemm` | 272.9 | 92 % | BK=128 |
| FP8 e5m2 | TileLang `T.gemm` | 271.6 | 92 % | BK=64 |
| FP16 2:4 sparse | TileLang `T.gemm_sp` | **262.9** | 89 % | 2.00× its own dense |
| FP8 2:4 sparse | TileLang `T.gemm_sp` | **522.0** | 88 % | 1.91× its own dense |
| NVFP4 | CuTeDSL 2048×4096×4096 | **847.6** | 143 % ⚠ | see below: M=2048 only |
| NVFP4 | CuTeDSL 4096³ | 504.9 | 85 % | wave-quantized |
| NVFP4 | CuTeDSL 8192³ | 548.5 | 93 % | large shape recovers |
| **NVFP4** | **TileLang `T.gemm_blockscaled` 4096²×8192** | **865.2** | **146 % of 592** | fixed; operands L2-resident |
| NVFP4 | TileLang `T.gemm_blockscaled` 4096³ | 835.9 | 141 % | fixed; A+B+C = 128 MB, resident |
| NVFP4 | TileLang `T.gemm_blockscaled` 8192³ / 16384³ | 545 / 548 | 92 % | fixed; output evicts the operands |
| FP64 | cuBLAS | **1.06** | — | CUDA cores only, no fp64 tensor core |
| FP32 (true) | cuBLAS (`allow_tf32=False`) | **44.93** | 30 % of 148 | CUDA cores |
| MXFP4 | CuTeDSL MX path (UE8M0 SF) | 509.6 | — | **correct**, cos 1.000000, md 0.25 (fp16 store) |
| MXFP8 | CuTeDSL MX path (UE8M0 SF) | 256.5 | — | **correct**, cos 1.000000, md 0.0078 (fp16 store) |

Derived peak column: FP16 = 148, FP8 = 296, FP16-sparse = 296, NVFP4 = 592. See
"Where the nameplate sits" for why these are the right denominators.

> **The NVFP4 peak is wrong.** Three independent measurements exceed 592 TF/s: CuTeDSL at
> M=2048 (847.6), and the fixed TileLang kernel at 4096²×8192 (865.2) and 4096³ (835.9), all
> `cos` 1.000000. A 46 % overshoot is not wave quantization — it means the `296 × 2` derivation
> does not hold for FP4 on this part. Use **865 TF/s as the working NVFP4 floor**, and read the
> FP16/FP8 rows of this table as the ones the nameplate actually pins down.
>
> The TileLang rows carry a caveat no other row does: the same kernel drops to 540-560 TF/s as
> soon as the output is large enough to evict the fp4 operands from L2, with no change in
> bandwidth utilisation (167 GB/s of 1273) and no throttling (97 W of 600, clock flat). Quote
> the shape and the L2 state or the number means nothing. Details in
> `nvfp4_gemm_throughput_and_tile_sweep.md`.

## The three traps, in the order they bit

### 1. `296` is the FP8 dense peak, not FP16

The plate reads `74 / 148 / 296 / 593`. The natural reading is a 1:2 chain —
FP64 → FP32 → FP16-dense → FP16-sparse — but that cannot be right, because on that reading
FP16 measures **48 % of peak** while *NVIDIA's own cuBLAS* measures **45 %**. No vendor library
leaves half the machine idle on a plain 4096³ fp16 GEMM.

Read `296` as **fp8 dense** instead and everything snaps into place:

| reading | fp16 dense would be | measured best | verdict |
|---|---|---|---|
| 296 = FP16 dense | 296 | 143.1 = 48 % | implausible — cuBLAS is also 45 % |
| **296 = FP8 dense** | 148 | 143.1 = **97 %** | **fits** |

So the derived peaks are FP16 = 296/2 = **148**, FP8 = **296**, and the 2:1 dense:sparse ratio
makes FP16-sparse = 296. The `593` row is then FP8-sparse (296 × 2), which the
`522.0` fp8-sparse reading fills at 88 %.

### 2. `do_bench(backend="event")` reads fp8 ~42 % high

The same TileLang fp8 kernel at 4096³, timed five ways in one process:

| method | µs | TF/s |
|---|---|---|
| `do_bench` **event** (default) | 365.2 | **376.3** |
| `do_bench` wall | 520.2 | 264.2 |
| `do_bench` cudagraph | 517.7 | 265.5 |
| manual wall loop | 512.6 | 268.1 |
| wall + 256 MB flush | 527.8 | 260.4 |

Four honest methods agree at 260–268; only `event` sits at 376. The cause is the L2 flush
`do_bench` issues between iterations: it evicts the operands, but the flush itself runs
*between* the timing events, so the kernel is charged cold reads while the flush is free.
At 8192³ (operands 134 MB, far past the 117 MB L2) the gap collapses to 4 %, which confirms
the mechanism.

Any older fp8 reading in this KB taken with `do_bench` default (e.g. the 297 and 306 in
`gemm_dtype_throughput.md`) is therefore inflated by roughly this factor. The
measurements here use wall-clock and are the ones to compare against cuBLAS.

### 3. NVFP4 depends hard on M

| shape | NVFP4 TF/s |
|---|---|
| 2048×4096×4096 | 847.6 |
| 4096³ | 504.9 |
| 4096×4096×16384 | 515.7 |
| 8192³ | 548.5 |

The headline 847.6 exists **only at M=2048**, where the tile grid is 16×32 = 512 blocks over
156 SMs (3.28 waves). At 4096³ the grid is 1024 blocks = 6.56 waves and the tail costs ~40 %.
8192³ recovers to 93 % because the tail is amortized. So `847.6` and `504.9` are the same
kernel at different fills, not two kernels — quote whichever matches the shape you ship.

## Sparse: 2:4 pays almost exactly 2×

Both sparse rows are **2.00× / 1.91× their own dense measurement**, run through the same
`T.gemm_sp` and timed the same way. That is the hardware ratio, so on this part sparse is a
real 2× and not a scheduling artifact.

The metadata encoding is the trap: for a group of 4 with kept indices `i0 < i1`, the selector
byte is `(i1 << 2) | i0`. A "sorted rank" encoding (`i0*2 + (i1-i0-1)`) runs at full speed
and returns **cos = 0.16** — fast and wrong, exactly the failure mode a timing-only benchmark
would miss. `E` is one `uint8` per 8 elements of A (`get_e_factor(f16, uint8) = 8`), and A is
compressed to `K/2`.

## NVFP4 in TileLang does not run

`T.gemm_blockscaled` compiles and then faults with `misaligned address` at 4096³. This matches
the source-level finding in `nvfp4_blockscaled_is_broken_in_tilelang.md`: the scale-word offset
in `mma_sm120_macro_generator._scale_word_offset` disagrees with the hardware layout at 2046 of
2048 positions, so the SF gather walks outside the tile. Use the CuTeDSL
`DenseGemmKernel` path (`nvfp4_gemm_computes_correctly.md`) instead; it is the only working
NVFP4 on this box.

## Where the nameplate sits

| precision | nameplate (derived) | best measured | % |
|---|---|---|---|
| FP64 | 74 | **1.06** | **1.4 %** ⚠ |
| FP32 (CUDA core) | 148 | **44.93** | 30 % |
| TF32 | ~68 | 65.1 | 96 % |
| FP16 dense | 148 | 143.1 | 97 % |
| FP8 dense | 296 | 285.3 | 96 % |
| FP16 2:4 sparse | 296 | 262.9 | 89 % |
| FP8 2:4 sparse | 593 | 522.0 | 88 % |
| NVFP4 | 592 | 847.6 (M=2048) | 143 % ⚠ |

Two rows are outside the 85–97 % band, and they are different problems.

**FP64 at 1.06 TF/s is not a tuning gap — the 74 on the plate cannot be fp64.** If it were,
4096³ would finish in 1.9 ms; cuBLAS takes **129 ms**, 70× longer. This is a consumer die with
no fp64 tensor core, so fp64 falls back to the CUDA cores and lands at ~1/140 of the fp16
throughput. Whatever `74` refers to, it is not double-precision GEMM on one of these cards.
Anyone sizing a milestone on "74 TF/s fp64" is off by two orders of magnitude — measure this
row before trusting the plate.

FP32 at 44.9 TF/s (true fp32, `allow_tf32=False`) is the honest CUDA-core number; the ~97 in the
old note was a lane-count estimate that ignores instruction issue. 30 % of the 148 figure is
expected for core-bound work, and TF32 at 65.1 shows the tensor path is the only way up.

**NVFP4 at M=2048 (143 %)** is the opposite kind of outlier. Either the 2× chain understates the
fp4 path (the `MmaMXF4NVF4Op` unit may sustain more than 2× FP8 on this arch), or 847.6 is the
one shape where fill hides enough tail to exceed a nominal peak. Do not use it as the fp4
denominator; quote 504.9/548.5 against 592 for shipped shapes and treat 847.6 as a small-M
curiosity.

## MXFP4 / MXFP8: correct once the SF is shipped in MMA-atom order (earlier "wrong" was the probe, not the kernel)

> **Correction.** An earlier revision of this file claimed the CuTeDSL MX path "runs but is
> wrong" (cos 0.03 / 0.04). That was the *probe feeding the wrong scale layout*, not a kernel
> defect. Re-measured with rank-3 tensors and the SF pre-permuted into MMA-atom order, both MX
> paths are numerically correct at every shape up to 4096³. The section below is rewritten.

Same `DenseGemmKernel` (`sf_vec_size=32`, `Float8E8M0FNU` scales, `MmaMXF4Op` / `MmaMXF8Op`
chosen from `a_dtype`), tile `(128,128)`, `max_active_clusters=156`, unit UE8M0 scales,
fp16 output. Numeric gate vs a dequantised fp32 reference, and a compiled-then-timed run
(compile once, 30 CUDA-event iterations after warmup):

| path | 4096³ | 2048×4096×4096 | max abs diff | cos |
|---|---|---|---|---|
| NVFP4 (`MmaMXF4NVF4Op`, UE4M3 SF) | 504.4 | 957.7 | 0.25 | 1.000000 |
| **MXFP4** (`MmaMXF4Op`, UE8M0 SF) | **509.6** | 622.7 | 0.25 | 1.000000 |
| **MXFP8** (`MmaMXF8Op`, UE8M0 SF) | **256.5** | 228.1 | 0.0078 | 1.000000 |

The `md≈0.25` at 4096³ is the fp16 epilogue store (same residual NVFP4 carries), not the MMA;
cos is 1.0 across all shapes tested — `(128,128,256)`, `(256,256,512)`, `(512,512,1024)`,
`(1024,1024,2048)`, `(4096,4096,4096)`. MXFP4 at 509 TF/s already matches NVFP4 (504) on the
square shape; MXFP8 at 257 is below dense fp8 (285) because the MX atom does `k=32` per step
against NVFP4's `k=64`.

### What the wrong probe did (so it is not repeated)

The earlier probe handed a **row-major `(M, K/sf_vec)`** scale buffer to the kernel and used
rank-2 `(M,K)` operands. On this path that produces plausible-but-wrong numbers **with no
error** — `blockscaled_three_paths.md` names this "the single most dangerous failure mode". The
fix is exactly three things, all in `_probes/probe_sm120_blockscaled_three_gate.py`:

1. **All five tensors rank-3** `(M,K,1)` / `(N,K,1)` / `(M,N,1)`; a rank-2 view fails inside
   `tile_atom_to_shape_SF` or silently mis-addresses.
2. **SF pre-permuted into MMA-atom order**: reshape to `(l, mn/128, sf_k/4, 32, 4, 4)` then
   `permute(3,4,1,5,2,0)`, zero-padded to 128-row / 4-K-block boundaries. Feeding row-major
   scales is what made cos collapse.
3. **`sf_vec_size` fixes `tile_k`**: 16 → 128, 32 → 256 (default `sf_vec_size*8`). A `tile_k`
   that is not a multiple of the k-atom is rejected at plan time.

The "closed-form 25 % / 78 % correct" numbers from the old probe were the row-major layout
mis-scatter, not the instruction. With the atom-order SF, the same all-ones gate returns 100 %.

### Practical stance (unchanged in spirit, changed in fact)

MX is now **usable**, not "do not use". NVFP4 remains the validated production 4-bit path
(fp4 + e4m3 scale, sf_vec 16) at 505/958 TF/s; MXFP4 is an equal-speed alternative that trades
the finer UE4M3 block scale for a coarser UE8M0 power-of-two scale (cheaper to quantise, less
accurate per block). MXFP8 lands under dense fp8. Size work on the NVFP4 rows; use the MX rows
only when a quantiser already emits UE8M0 scales.

## How to reproduce

Probes live at
`C:\Users\yangzhipeng\AppData\Local\Temp\opencode\p21_tl_dense.py` (dense f16/bf16/fp8),
`p22_tl_sparse.py` (2:4 f16/fp8 with the selector gate), `p23_tl_nvfp4.py` (TileLang NVFP4),
`p20_nvfp4_shapes.py` (CuTeDSL NVFP4 across shapes), `p24_nongemm.py` (RMSNorm/RoPE/attention/
LM head), `p25_fp64_fp32_mx.py` (FP64, true FP32), and `p16/p17/p19` (cuBLAS + timing backends).

**For MX use the gate probe, not p25.** `p25_fp64_fp32_mx.py` feeds row-major `(M,K/sf_vec)`
scales and rank-2 operands, which is what produced the old "cos 0.03" reading — it is the wrong
layout, not a broken kernel. The correct MX path is gated in
`hw_kb/cutedsl/sm120/_probes/probe_sm120_blockscaled_three_gate.py` (rank-3 tensors + SF
pre-permuted `permute(3,4,1,5,2,0)`), which returns cos 1.0 for NVFP4 / MXFP4 / MXFP8. All gate
on cosine before reporting a speed.
