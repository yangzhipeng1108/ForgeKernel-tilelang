# Three block-scaled paths on sm120: NVFP4, MXFP4, MXFP8 — all compute correctly

`nvfp4_gemm_computes_correctly.md` proved one of them. This entry covers the other two, and
records that all three live in **one shipped kernel** rather than three: `sf_vec_size` alone
selects which warp-MMA atom the kernel builds.

That matters because "does sm120 support fp4?" is the wrong question. The card has three
*distinct* block-scaled instructions, differing in element width and in scale-factor format,
and they are not interchangeable — a quantizer that emits UE8M0 scales cannot feed an op that
wants UE4M3.

## The result

`flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x.DenseGemmKernel` (aliased
`Sm120B12xBlockScaledDenseGemmKernel`) run at M=N=128, K=256, tile (128, 128):

| path | `sf_vec_size` | A/B | scale dtype | MMA atom | max abs diff | cos |
|---|---|---|---|---|---|---|
| NVFP4 | 16 | e2m1 | **UE4M3** | `MmaMXF4NVF4Op` | **0** | 1.000000 |
| MXFP4 | 32 | e2m1 | **UE8M0** | `MmaMXF4Op` | **0** | 1.000000 |
| MXFP8 | 32 | e4m3 | **UE8M0** | `MmaMXF8Op` | 1.85e-3 | 1.000000 |

NVFP4 and MXFP4 are bit-exact because both operands were snapped onto the exact e2m1 grid and
every scale was a power of two, so the reference is computable in fp32 without rounding. MXFP8
is 1.85e-3 off, which is the fp16 output round-trip on values up to ~5.8 — the accumulator is
fp32 and the epilogue stores fp16, so that residual is the store, not the MMA.

## Which op `sf_vec_size` selects

The dispatch is not a guess; the constructor builds the atom off this table:

```
sf_vec_size == 16  ->  warp.MmaMXF4NVF4Op     e2m1 A/B, UE4M3 SF, k-atom 64
sf_vec_size == 32  ->  warp.MmaMXF4Op         e2m1 A/B, UE8M0 SF, k-atom 64
(sf_vec_size == 32, fp8 A/B) -> warp.MmaMXF8Op  e4m3 A/B, UE8M0 SF, k-atom 32
```

All three carry `admissible_archs = [sm_120a, sm_120f, sm_121a, sm_121f]`. The corresponding
`tcgen05` block-scaled op lists `[sm_100a, sm_100f, sm_103a, sm_107a, sm_110a]` and **refuses
to construct here** — the two families are disjoint, so an sm100 fp4 kernel never ports by
changing an arch flag.

## What the caller has to get right

- **All five tensors are rank-3** `(M,K,1)`, `(N,K,1)`, `(M,N,1)`. Same constraint as the NVFP4
  entry: a rank-2 view fails inside `tile_atom_to_shape_SF` with
  `expects target shape and order operands have same rank, but got (128,256) vs (2,1,3)`.
- **SF is shipped pre-permuted into MMA atom order**, not row-major. The repack is
  `(l, mn/128, sf_k/4, 32, 4, 4)` then `permute(3, 4, 1, 5, 2, 0)`. Feeding a row-major
  `(M, K/sf_vec)` buffer produces plausible-but-wrong output, never an error — the single most
  dangerous failure mode on this path.
- **`alpha` is a rank-1 `(1,)` fp32 tensor**, and `max_active_clusters` is a python int, not a
  tensor. The signature is `kernel(a, b, sfa, sfb, c, alpha, max_active_clusters, stream)`.
- **`sf_vec_size` also fixes `tile_k`**: 16 → 128, 32 → 256 (the default is `sf_vec_size * 8`).
  A tile_k that is not a multiple of the k-atom is rejected at plan time.

## Why the mixed picture with the "official" example

The shipped `blackwell_geforce/dense_gemm.py` example accepts only `Float16` / `BFloat16` for
`ab_dtype`; asking for `Float8E4M3` fails with `expects the 'ab_dtype' Op parameter to be one
of Float16 or BFloat16`. So the *example* covers no fp8 at all — see
`warp_mma_bf16_and_fp16.md`. The *kernel* in `dense_blockscaled_gemm_sm120_b12x.py` does cover
fp8, but only the block-scaled MXFP8 variant, with UE8M0 scales. There is no plain
per-tensor-scaled fp8 GEMM in this image's sm120 path.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB, no NVLink
- `lmsysorg/sglang:dev`: cutlass **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- Probe: `_probes/probe_sm120_blockscaled_three_gate.py`
