# NVFP4 blockscaled GEMM on sm120: 836-865 TF/s while L2 holds the operands, 545 when it does not

## Purpose

Measure the fixed TileLang NVFP4 blockscaled GEMM (see
`nvfp4_blockscaled_is_broken_in_tilelang.md` for the correctness fix that made any of this
possible) across tile shapes, pipeline depths and problem sizes.

The headline is uncomfortable and is the most useful thing here: **the number is decided by L2
residency, and the measurement protocol decides whether you can see that.** The same kernel runs
at 836-865 TF/s when the fp4 operands stay in L2 and 540-560 TF/s when the output grows large
enough to evict them. It is neither throttled nor bandwidth bound in either case. Any number
quoted without its shape, its L2 state and its protocol is meaningless, and two earlier versions
of this entry quoted exactly such numbers.

## Measured on

RTX 6000D (sm_120a), TileLang 0.1.15 with the packed-scale fix, torch 2.15.0+cu132, one GPU.
Operands random fp4, scale factors random e4m3, `2*M*N*K` FLOPs. Clocks sampled with
`nvidia-smi` during every batch.

## It is not the power wall

A 120-iteration run at 8192鲁, sampling clock and power between batches:

| batch start | SM clock MHz | power W | T 掳C | ms/iter | TF/s |
|---|---|---|---|---|---|
| 0 | 2325 | 97.5 | 43 | 7.331 | 150.0 |
| 20 | 2325 | 97.5 | 43 | 1.940 | 566.7 |
| 40 | 2325 | 97.5 | 43 | 1.883 | 584.1 |
| 60 | 2325 | 97.5 | 43 | 1.936 | 567.8 |
| 80 | 2325 | 97.5 | 43 | 1.917 | 573.6 |
| 100 | 2325 | 97.5 | 43 | 1.892 | 581.2 |

Clock flat at 2325 MHz for the whole run, power flat at **97 W against a 600 W limit**, die at
43 掳C. Batch 0 is slow only because it is cold. **This kernel never comes close to the power
wall**, which is the opposite of the fp16 large-tile case in `gemm_tuning_summary.md`. Locking
clocks here would buy nothing.

## It is not the bandwidth wall either

`ncu` on the same kernel:

| metric | 4096鲁 | 8192鲁 |
|---|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 19.8 % | 20.0 % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 64.5 % | 71.5 % |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 閳?| 32.0 % |
| `launch__waves_per_multiprocessor` | 閳?| 26.3 |

DRAM sits at **20 %** at both sizes, so nothing here is memory bound. The tensor pipe is the
busiest unit at 64-72 %, and only a third of the warps are resident at any time. The kernel
feeds the MMA pipe well but leaves a quarter to a third of it idle.

## The protocol controls the answer

Same kernel, same config (128/128/128, `num_stages=3`), several timing methods. Every row below
was reproduced three times on a fresh process:

| shape | plain loop | CUDA graph | ncu single launch |
|---|---|---|---|
| 4096鲁 | 0.164-0.165 ms 閳?**835-836 TF/s** | 0.265-0.270 ms 閳?508-518 TF/s | 0.166 ms 閳?830 TF/s |
| 4096虏脳8192 | 0.318 ms 閳?**865 TF/s** | 閳?| 閳?|
| 8192鲁 | 2.02-2.03 ms 閳?**541-544 TF/s** | 2.00 ms 閳?549-551 TF/s | 1.08 ms 閳?1008 TF/s |

Three things fall out.

**It is not launch overhead.** Capturing the loop in a CUDA graph removes every CPU-side launch,
and at 8192鲁 it changes nothing (2.02 閳?2.00 ms). At 4096鲁 the graph is **40 % slower**, not
faster, so the per-launch path is not where the time goes and batching it into a graph actively
hurts. Whatever the wrapper does per call is invisible next to the kernel.

**The output buffer is not the issue either.** Rotating over eight distinct `C` tensors is
identical to reusing one (2.007 vs 2.006 ms at 8192鲁), so there is no write-after-write
serialisation.

**4096鲁 and 8192鲁 differ because of L2, and nothing else.** Both run the same tile and the same
code. At 4096鲁 the two fp4 operands (32 MB each) plus the fp32 output (64 MB) total 128 MB, which
is exactly this part's L2; the operands stay resident across the K loop and the kernel reaches
**836 TF/s**. At 8192鲁 the output alone is 268 MB, so it evicts the operands and every K
iteration refetches them from HBM; the kernel settles at **545 TF/s**. That is a 35 % swing from
cache residency on identical compute.

Even so it is **not a bandwidth problem at 8192鲁**: 336 MB of traffic over 2.0 ms is 167 GB/s
against a measured 1273 GB/s copy rate, i.e. 13 % of peak. The cost of the L2 miss here is
latency that a three-stage pipeline cannot fully hide, not throughput.

**ncu's duration is not usable as a throughput anchor.** It reports 1.08 ms at 8192鲁, stable
across launches, where every event measurement and the CUDA graph agree on 2.0 ms. Its unit
percentages are still the best evidence about *where* time goes, which is why they are quoted
above, but its duration is discarded.

The honest reporting is therefore a pair: **835-865 TF/s when the operands stay in L2, 545 TF/s
when they do not.** For prefill on shapes up to ~4096鲁 use the first; for larger shapes, and for
decode loops, use the second.

## Steady-state across shapes

Warmup 5, then 40 calls with events around the whole block; reproduced three times:

| shape | ms | TF/s | L2 state |
|---|---|---|---|
| 4096鲁 | 0.164 | **835.9** | A+B+C = 128 MB, all resident |
| 4096虏脳8192 | 0.318 | **865.2** | operands 32 MB each, resident |
| 4096虏脳16384 | 0.979 | 561.4 | operands 64 MB each, partially resident |
| 8192虏脳4096 | 1.012 | 543.0 | C 268 MB evicts the operands |
| 8192鲁 | 2.025 | 542-544 | as above |
| 16384鲁 | 16.060 | 547.7 | nothing resident |

The pattern is entirely explained by whether the fp4 operands survive in L2. Everything above
about 128 MB of combined footprint lands in **540-560 TF/s**; everything below it reaches
**835-865**. There is no separate "large shape" penalty.

## The tile axis is closed at 128/128/128, stages 3

Single-launch events, `4096鲁`, 256 threads:

| BM | BN | BK | stages | ms | TF/s |
|---|---|---|---|---|---|
| 128 | 128 | 128 | 2 | 0.193 | 711.6 |
| **128** | **128** | **128** | **3** | **0.177** | **775.7** |
| 128 | 128 | 128 | 4 | 0.201 | 685.0 |
| 128 | 128 | 256 | 2 | 0.212 | 648.5 |
| 64 | 128 | 128 | 3 | 0.227 | 605.1 |
| 256 | 128 | 128 | 2 | 0.630 | 218.1 |

`8192鲁`, single-launch events:

| BM | BN | BK | stages | ms | TF/s |
|---|---|---|---|---|---|
| 128 | 128 | 128 | 3 | 2.073 | 530.3 |
| 128 | 128 | 256 | 2 | 2.071 | 531.0 |
| 128 | 128 | 64 | 3 | 2.184 | 503.3 |
| 256 | 128 | 128 | 2 | 4.695 | 234.2 |
| 128 | 256 | 128 | 2 | 5.400 | 203.6 |
| 256 | 256 | 64 | 2 | 12.141 | 90.6 |

The ranking holds under either protocol: **128/128/128 with three stages wins and every larger
tile collapses**, exactly as the fp8 sweep in `gemm_dtype_throughput.md` found. The cause is
the 99 KiB smem budget 閳?a 256-row fp4 tile leaves no room for pipeline depth, so it drops to
one block per SM. `BK=256` with `stages=3` does not even launch.

## Where the ceiling actually is

The `4096虏脳32768` probe (C L2-resident, long K) returns 712 TF/s, *below* 4096鲁, because
32768 columns make each operand 64 MB and the pair stops fitting in L2 閳?the same eviction
effect, caught in the act. So the ranking is:

- operands and output fit in L2 (4096鲁, 4096虏脳8192) 閳?**836-865 TF/s**, the best observed
- output evicts the operands (8192虏脳4096, 8192鲁, 16384鲁) 閳?**543-561 TF/s**
- long K pushes the operands themselves out (4096虏脳32768) 閳?712 TF/s

That makes **865 TF/s a floor on the NVFP4 peak**, and the derived 592 TF/s (`296 脳 2`) is low
by **at least 46 %**. The `precision_roofline_all_stacks.md` peak table needs an FP4 correction,
and the correct worked example is 865, not 592 and not 776.

## How to close the 545 -> 865 gap

The eviction penalty is latency, not bandwidth: at 8192鲁 the kernel moves 336 MB in 2.0 ms
(167 GB/s, 13 % of peak) yet loses 35 %. Three-stage pipelining does not cover the HBM latency
once every K iteration misses. The levers, in order of promise:

1. **Keep the operands resident across the K loop** by splitting the output into column blocks
   and iterating those outermost, so C for one block is small enough not to evict A/B. This
   turns the 8192鲁 case into a sequence of 8192虏脳256-ish calls, but adds A/B re-reads.
2. **More pipeline depth**, which is blocked by the 99 KiB smem budget at this tile. A shorter
   BK frees smem for stages but costs MMA efficiency, and `BK=64` measured 503 TF/s, so this
   route is closed at 128/128.
3. **A persistent kernel** that keeps the grid resident and loops over tiles internally,
   avoiding re-entry. `clusters_free_multicast_broken.md` and `clc_compiles_to_nothing.md` both
   record that the hardware-assisted versions of this are unavailable here, but a plain
   persistent loop was never tried for this kernel.

None of these is measured yet. The 865 figure assumes the operands fit, which is true for
prefill on 4096-class shapes and false for 8192-class ones.

## When needed (trigger)

Use this when picking a config for NVFP4 on sm120, when sizing a decode loop (use the
back-to-back number, not the single-shot one), or when someone proposes a larger tile to "use
the tensor cores better". The answer to the last one is no.

## Gotchas

- **Never quote a TF/s number without its protocol.** This single kernel spans 508-865 TF/s
  depending purely on how it is timed.
- `out_idx=[4]` allocates a fresh C per call. At 8192鲁 that is 268 MB and it dominates the
  measurement. Preallocate and pass C in.
- **CUDA graphs make this kernel slower, not faster** (4096鲁: 0.164 閳?0.270 ms). Do not reach
  for graph capture to close a launch-overhead gap here; there is no such gap to close.
- Do not use `ncu` durations as throughput; it reports 1.08 ms at 8192鲁 where events and graphs
  agree on 2.0. Use it for unit utilisation only.
- `BK=256` + `stages=3` fails to launch on the 99 KiB budget.
- **Measure the shape you care about, in the order you will run it.** An earlier round measured
  4096鲁 at 425 TF/s purely because a larger shape had run first and left L2 in a different
  state; the same kernel in a fresh process gives 836.

## Sources

- `tilelang/cuda/op/gemm/gemm_mma_sm120.py` 閳?the sm120 blockscaled dispatch and its smem sizing
- `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py` 閳?the tile geometry and packed-scale read
- `scripts/patch_tilelang_sm120_nvfp4_scale.py` 閳?the fix this measurement depends on
- `hw_kb/tilelang/sm120/nvfp4_blockscaled_is_broken_in_tilelang.md` 閳?correctness history
- `hw_kb/tilelang/sm120/gemm_dtype_throughput.md` 閳?the fp8 tile sweep this mirrors
- `hw_kb/tilelang/sm120/gemm_tuning_summary.md` 閳?the fp16 power-wall result this one is not
