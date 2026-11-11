# The plain warp-MMA GEMM on sm120: fp16 and bf16 only

The block-scaled entries cover fp4 and mxfp8. This is the ordinary path — the one every
non-quantized kernel uses — and it has its own limits worth knowing before assuming "sm120 is
just a smaller sm100".

## The result

`blackwell_geforce/dense_gemm.py`, the only sm120 example that ships, passes on this card:

| dtype | shape (M,N,K,L) | tile (M,N,K) | time | verdict |
|---|---|---|---|---|
| fp16 | 512, 512, 512, 1 | 128, 128, 64 | 94.9 µs | **PASS** |
| bf16 | 2048, 2048, 2048, 1 | 128, 128, 64 | 200.3 µs | **PASS** |
| fp16 | 8192, 8192, 8192, 1 | 128, 256, 64 | 27867.7 µs | **PASS** |

The 8192³ run is ~39 TFLOP/s of fp32-accumulated fp16; small for a datacenter part, in line
for a consumer geForce Blackwell. It is not a tuned number — the example does one warmup — but
it confirms correctness at scale, which is what the entry is for.

## fp8 is *not* in the example — a correction

A previous reading of this tree said the example was "fp16/fp8 only". It is **fp16/bf16
only**. Asking it for fp8:

```
--a_dtype Float8E4M3 --b_dtype Float8E4M3 ...
error: expects the 'ab_dtype' Op parameter to be one of Float16 or BFloat16
```

The rejection is in `_setup_attributes`, which hardcodes the `MmaF16BF16Op` atom and only
branches on `a_dtype.width in {16, 8}` for the *load* path — not for the MMA. So a width-8
operand would type-check the copy but then hit an fp16 atom. The example never intended to do
fp8.

The real fp8 story on sm120 is the block-scaled MXFP8 path
(`blockscaled_three_paths.md`), which uses `MmaMXF8Op` with UE8M0 scales. There is **no
per-tensor-scaled fp8 GEMM** exposed for this card in this image: `MmaFP8Op` (the SM89-shaped
(16,8,32) op) *constructs*, and is arch-neutral, but no shipped sm120 kernel drives it. If you
need plain fp8, it is a build-it-yourself job; if you can live with block scales, MXFP8 is
already there and verified.

## The dead-end that is worth recording

An earlier attempt tried `flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk` as the bf16
reference. It is a real, tuned, skinny-M kernel — and it **does not compile** against this
image's cutlass 4.8.0:

```
CompilerDiagnosticError: error: compiler pass failed
  --> .../dense_bf16_gemm_warp_splitk.py:479:0  in function `__call__`  (CUTLASS diagnostic)
```

It fails on every shape, including `1x4096x4096`, which is exactly its target (it caps M at
32 and only supports output tiles 16/32 — this is a decode-class kernel, not a general GEMM).
The lesson is the same as the TileLang fp4 one: a kernel being present in the image is not a
guarantee it lowered against the cutlass that is installed. Verify before trusting.

Its input contract is still worth recording, because the same shape rules apply to any
hand-written skinny-M warp kernel:

- `b` must be shape `(K, N)` with stride `(1, K)` — packed **column-major**; row-major b is
  rejected before the build.
- `a` row-major `(M, K)` stride `(K, 1)`; `out` row-major `(M, N)` stride `(N, 1)`.
- all tensors 32-byte aligned, bf16 only (fp16 rejected outright).

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB, no NVLink
- `lmsysorg/sglang:dev`: cutlass **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- `dense_gemm.py` from `flashinfer/data/cutlass/examples/python/CuTeDSL/blackwell_geforce/`
- Probe (fp8 refusal + dense_gemm PASS are reproducible via the example CLI):
  `_probes/probe_sm120_capability_census.py` confirms the op set;
  the PASS rows come from running the example directly.
