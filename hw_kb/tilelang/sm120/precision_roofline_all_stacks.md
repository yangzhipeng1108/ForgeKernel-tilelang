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
| **NVFP4** | **TileLang `T.gemm_blockscaled` 4096²×8192** | **865.2** | **146 % of 592** | fixed; single launch, L2-resident |
| NVFP4 | TileLang `T.gemm_blockscaled` 4096³ / 8192³ | 792 / 714 | 134 / 121 % | fixed; single launch |
| NVFP4 | TileLang `T.gemm_blockscaled` back-to-back | 543–561 | 92–95 % | fixed; batch of 20, sustained |
| FP64 | cuBLAS | **1.06** | — | CUDA cores only, no fp64 tensor core |
| FP32 (true) | cuBLAS (`allow_tf32=False`) | **44.93** | 30 % of 148 | CUDA cores |
| MXFP4 | CuTeDSL MX path | (511) | — | **runs-but-wrong**, cos 0.03 |
| MXFP8 | CuTeDSL MX path | (234) | — | **runs-but-wrong**, cos 0.04 |

Derived peak column: FP16 = 148, FP8 = 296, FP16-sparse = 296, NVFP4 = 592. See
"Where the nameplate sits" for why these are the right denominators.

> **The NVFP4 peak is wrong.** Three independent measurements exceed 592 TF/s: CuTeDSL at
> M=2048 (847.6), and the fixed TileLang kernel at 4096²×8192 (865.2, `cos` 1.000000) and
> 4096³ (792). A 46 % overshoot is not wave quantization 閳?it means the `296 脳 2` derivation
> does not hold for FP4 on this part. Use **865 TF/s as the working NVFP4 floor**, and read the
> FP16/FP8 rows of this table as the ones the nameplate actually pins down.
>
> Note the TileLang rows carry a protocol caveat that no other row does: the same kernel
> sustains only 543-561 TF/s when launched back to back, because there is a ~0.15 ms gap
> between launches. Single-launch is the right basis for comparing against cuBLAS/CuTeDSL,
> which are also measured single-shot, but decode loops must use the sustained figure.
> Details in `nvfp4_gemm_throughput_and_tile_sweep.md`.

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

## MXFP4 / MXFP8: reachable but not correct — a closed-form gate localises why

The CuTeDSL MX path (`DenseGemmKernel` with `sf_vec_size=32`, `Float8E8M0FNU` scales,
`MmaMXF4Op` / `MmaMXF8Op` chosen internally from `a_dtype`) compiles and runs at
**511 TF/s (MXFP4)** and **234 TF/s (MXFP8)** at 4096³, but **cosine is 0.03 and 0.04**. These
two rows carry **no throughput claim**.

Three experiments pinned it down without guessing:

1. **Closed-form gate** — all-ones operands, unit e8m0 scale, so every `C` entry must be `K`:
   MXFP4 gets **25 %** of entries right, MXFP8 **78 %**. A working kernel gives 100 %.
2. **The SF layout is not the cause** — with all-ones scales, *any* permutation still yields
   all-ones output. Re-deriving the scatter from `BlockScaledBasicChunk(sf_vec).layout`
   (`((32,4),(sf_vec,4))` stride `((16,4),(0,1))`, tiled `(2,1,3)`) changed the result by
   exactly nothing. Rule that out.
3. **`tile_k` is the lever, but not a free one** — `tile_k` defaults to `sf_vec_size*8 = 256`,
   giving 16 K-tiles at K=4096. Forcing `tile_k=128` moves MXFP4 to 75 % correct but with
   exploded values (a garbage scale is now being read), and drops MXFP8 to 25 %. So the
   operand tile and the SF atom disagree.

What that points at: the kernel builds a `tile_k=256` TMA descriptor from `a.shape=(M,K,1)`, and
a packed fp4/fp8 buffer handed over as `(M,K)` does not satisfy it — the NVFP4 wrapper gets away
with the same call because its atom and `sf_vec_size=16` line up. The fix is a correctly packed
A/B layout (or `K/2` as the inner extent), which needs a validated example to diff against;
**flashinfer ships none for MX**. Recorded as a real gap rather than an assumed match.

Practical stance: **do not use MX on this stack today.** NVFP4 (fp4 + e4m3 scale, sf_vec 16) is
the validated 4-bit path at 848/505 TF/s, and fp8 covers the 8-bit case at 285. The MX numbers
would land between them once the operand layout is solved, but nothing should be sized on them
yet.

## How to reproduce

Probes live at
`C:\Users\yangzhipeng\AppData\Local\Temp\opencode\p21_tl_dense.py` (dense f16/bf16/fp8),
`p22_tl_sparse.py` (2:4 f16/fp8 with the selector gate), `p23_tl_nvfp4.py` (TileLang NVFP4),
`p20_nvfp4_shapes.py` (CuTeDSL NVFP4 across shapes), `p24_nongemm.py` (RMSNorm/RoPE/attention/
LM head), `p25_fp64_fp32_mx.py` (FP64, true FP32, MXFP4/MXFP8), and `p16/p17/p19` (cuBLAS +
timing backends). All gate on cosine before reporting a speed.
