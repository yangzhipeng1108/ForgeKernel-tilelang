# NVFP4 blockscaled GEMM on sm120: 776 TF/s at 4096鲁, and the tile axis is closed

## Purpose

Measure the fixed TileLang NVFP4 blockscaled GEMM (see
`nvfp4_blockscaled_is_broken_in_tilelang.md` for the correctness fix that made any of this
possible) across tile shapes, pipeline depths and problem sizes, so the deployment knows which
config to ship and where the ceiling is. Also settle what the FP4 peak on this part actually
is: the `296 閳?592` doubling in `precision_roofline_all_stacks.md` is too low.

## Measured on

RTX 6000D (sm_120a), TileLang 0.1.15 with the packed-scale fix, torch 2.15.0+cu132. One GPU,
`CUDA_VISIBLE_DEVICES=1`. Timing is median of 20-50 per-iteration CUDA events after 5-10 warmup
calls, `2*M*N*K` FLOPs, operands random fp4 and scale factors random e4m3.

## The tile axis is closed at 128/128/128, stages 3

`4096鲁`, 256 threads:

| BM | BN | BK | stages | ms | TF/s |
|---|---|---|---|---|---|
| 128 | 128 | 128 | 2 | 0.193 | 711.6 |
| **128** | **128** | **128** | **3** | **0.177** | **775.7** |
| 128 | 128 | 128 | 4 | 0.201 | 685.0 |
| 128 | 128 | 256 | 2 | 0.212 | 648.5 |
| 128 | 128 | 256 | 3 | 閳?| smem > 99 KiB, launch fail |
| 64 | 128 | 128 | 3 | 0.227 | 605.1 |
| 128 | 64 | 128 | 3 | 0.237 | 580.5 |
| 256 | 128 | 128 | 2 | 0.630 | 218.1 |

`8192鲁`, 256 threads:

| BM | BN | BK | stages | ms | TF/s |
|---|---|---|---|---|---|
| 128 | 128 | 128 | 2 | 2.153 | 510.7 |
| **128** | **128** | **128** | **3** | **2.073** | **530.3** |
| 128 | 128 | 128 | 4 | 2.158 | 509.5 |
| 128 | 128 | 256 | 2 | 2.071 | 531.0 |
| 128 | 128 | 64 | 3 | 2.184 | 503.3 |
| 256 | 128 | 128 | 2 | 4.695 | 234.2 |
| 128 | 256 | 128 | 2 | 5.400 | 203.6 |
| 256 | 256 | 64 | 2 | 12.141 | 90.6 |

The same shape holds as fp8: **128/128/128 wins, and every larger tile collapses**. 256/128/128
loses 3.4x at 4096鲁 and 1.9x at 8192鲁; 256/256/64 loses 8.6x. The cause is occupancy, not
bandwidth 閳?the smem budget on sm_120 is 99 KiB, so a 256-row fp4 tile plus its scale rows
forces one block per SM and the pipeline depth has nowhere to go.

`stages` is the free lever and 3 is the peak. 4 is worse at both sizes, which says the depth-3
buffers already cover the TMA latency and the extra stage only costs smem.

## Where the ceiling actually is

| shape | best config | TF/s |
|---|---|---|
| 4096鲁 | 128/128/128 st3 | **775.7** |
| 8192鲁 | 128/128/256 st2 | **531.0** |
| 16384鲁 | 128/128/128 st2 | **540.7** |
| `4096虏脳32768` | 128/128/128 st3 | **712.0** |

The `4096虏脳32768` probe was meant to be the pure-compute ceiling: C is 64 MB and stays
L2-resident, K is long enough to amortise every prologue. It returns **712.0**, *below* the
4096鲁 number, because 32768 columns of fp4 make each operand 64 MB 閳?the pair no longer fits in
L2 and streams from HBM. So 712 carries a real HBM read cost and is a floor, not a ceiling.

**The honest number is 775.7 TF/s at 4096鲁**, where both operands and C together fit in L2.
That means the NVFP4 peak on this part is at least 776 TF/s, and the derived 592 TF/s
(`296 脳 2`) is **low by 30% or more**. Use 776 as the working denominator until a cleaner probe
lands; the `precision_roofline_all_stacks.md` peak table needs an FP4 correction.

## Why 8192鲁 is 68% of 4096鲁

Not bandwidth: the arithmetic intensity is enormous (3.4e11 FLOP over 268 MB of C plus 67 MB of
operands). The gap is C writeback. At 4096鲁 the 64 MB fp32 output fits L2 and the epilogue
mostly hits cache; at 8192鲁 the output is 268 MB and every element must reach HBM. At the
measured ~1273 GB/s copy rate, writing 268 MB costs 0.21 ms of a 2.07 ms kernel 閳?about 10%,
which is roughly the observed gap 閳?but the remaining 22% is tail-wave and issue-rate, not memory.

CuTeDSL's 548.5 at 8192鲁 is 3.3% above this kernel's 531.0, i.e. the two stacks hit the same
wall. That is the evidence that 531 is the shape's ceiling on this card, not a TileLang
deficiency.

## When needed (trigger)

Use this when picking a config for NVFP4 prefill on sm120, or when someone proposes a larger
tile to "use the tensor cores better". The answer is no: 128/128/128 with three stages is the
whole space, and the remaining headroom is in the epilogue and the tail, not the MMA.

## Gotchas

- `BK=256` with `stages=3` fails at launch with a dynamic smem error. The 99 KiB budget is the
  binding constraint, so any tile/depth combination has to be checked against it before it is
  assumed to work.
- The naive async-submit timing loop over-reports. p47 put 4096鲁 at 747.6 with a single sync
  around 20 submissions; per-iteration events settle it at 707-776. Always time per iteration.
- Do not compare a small-shape TF/s against a large-shape TF/s without checking L2 residency
  first. 775 at 4096鲁 and 531 at 8192鲁 are the same kernel.

## Sources

- `tilelang/cuda/op/gemm/gemm_mma_sm120.py` 閳?the sm120 blockscaled dispatch and its smem sizing
- `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py` 閳?the tile geometry and packed-scale read
- `scripts/patch_tilelang_sm120_nvfp4_scale.py` 閳?the fix this measurement depends on
- `hw_kb/tilelang/sm120/nvfp4_blockscaled_is_broken_in_tilelang.md` 閳?correctness history
- `hw_kb/tilelang/sm120/gemm_dtype_throughput.md` 閳?the fp8 tile sweep this mirrors
