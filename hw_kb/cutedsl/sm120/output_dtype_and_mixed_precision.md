# Output dtype decides MXFP8 accuracy; mixed fp4 x fp8 builds but has no kernel

Two separate results, both measured on RTX 6000D (sm_120a), cutlass-dsl 4.8.0.

## P2: the MXFP8 error was the output store, not the arithmetic

[`blockscaled_three_paths`](blockscaled_three_paths.md) recorded MXFP8 at `md=1.85e-3`
against NVFP4/MXFP4 at `md=0`, and guessed it was an fp16-store residue. Holding the
operands fixed (e4m3 x e4m3, UE8M0 scales, `sf_vec_size=32`, 128x128 tile) and changing
only the **C tensor dtype**:

| C dtype | 128x128x256 rel | 512^3 rel | 2048x4096x4096 rel | cos |
|---|---|---|---|---|
| fp16 | 2.36e-4 | 3.05e-4 | **3.64e-4** | 0.99999994 |
| bf16 | 2.64e-3 | 2.42e-3 | **2.86e-3** | 0.99999857 |
| fp32 | — | — | — | does not launch |

The error is **entirely the store width**: fp16 carries a 10-bit mantissa, bf16 only 8,
so bf16 is ~8x worse at the same relative scale — 2.86e-3 vs 3.64e-4, a **7.9x
reduction** by switching to fp16. The accumulator is f32 in both cases and is not the
limiting factor. Relative error is scale-invariant across the three shapes, confirming
the error is per-element rounding, not accumulation drift.

So the honest statement is: **MXFP8 on sm120 is accurate to ~3.6e-4 relative if you
store fp16; the 1.85e-3 figure was bf16 output.** fp16 and bf16 are the same width and
the same speed here, so there is no reason to pick bf16 for a GEMM output.

### fp32 output is not available on this card

Passing a float32 C is the natural way to see the true arithmetic error, and it fails:
`allocated 134144 bytes, max 101376` — the f32 epilogue stage alone overflows the 99 KB
smem budget. This is the same constraint as
[`tma_and_copy_family`](tma_and_copy_family.md). To get f32 out you would have to cut
the ab-stage count, which costs more than the precision is worth.

## P3: mixed A=fp4 x B=fp8 builds, but no shipped kernel runs it

`MmaMXF8F6F4Op(a_dtype, b_dtype, Float32, Float8E8M0FNU)` accepts **mixed** operand
pairs (its `__post_init__` explicitly rejects same-dtype pairs and points at
`MmaMXF4Op` / `MmaMXF4NVF4Op` / `MmaMXF8Op` instead). Constructing it with
`(Float4E2M1FN, Float8E4M3FN)`, `(Float8E4M3FN, Float4E2M1FN)` and
`(Float4E2M1FN, Float8E5M2)` at `sf_vec_size=32`, then building a tiled MMA and both
SF smem layouts, **succeeds on sm_120a**:

```
fp4 x e4m3 vec32: tiled_mma OK  sfa=512 sfb=512
e4m3 x fp4 vec32: tiled_mma OK  sfa=512 sfb=512
fp4 x e5m2 vec32: tiled_mma OK  sfa=512 sfb=512
```

But `DenseGemmKernel` cannot execute it: it asserts A and B share a dtype and fails at
compile with `Type mismatch: Float4E2M1FN != Float8E4M3FN`. There is no shipped sm120
kernel for mixed operands, so running it means hand-writing the mainloop — i.e. the
load-path work that [`from_scratch_tma_load_path_works`](from_scratch_tma_load_path_works.md)
records as blocked.

**Verdict: mixed precision is architecturally supported and reachable from the DSL,
but not usable today.** The gate to upgrade it from "constructs" to "computes" is the
same one that blocks the from-scratch NVFP4 kernel.

For reference, the legal mixed-precision matrix on this card (from
`MmaMXF8F6F4Op`'s own assertions):

- allowed: `fp4 x {e4m3, e5m2}` and `{e4m3, e5m2} x fp4`, plus fp6 combinations, f32 acc, UE8M0 SF
- rejected: same-dtype fp4/fp4 and fp8/fp8 (use the dedicated ops), and e4m3 x e5m2 (same width)

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB
- `lmsysorg/sglang:dev` container: cutlass-dsl **4.8.0**, torch 2.14.1+cu130
- P2 via `DenseGemmKernel(sf_vec_size=32, mma_tiler_mn=(128,128))`,
  `flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x`
- P3 gate is compile-only (the tiled MMA and SF layouts); no end-to-end kernel exists
