# NVFP4 blockscaled GEMM on sm120: 865 TF/s single-shot, 546 TF/s back-to-back

## Purpose

Measure the fixed TileLang NVFP4 blockscaled GEMM (see
`nvfp4_blockscaled_is_broken_in_tilelang.md` for the correctness fix that made any of this
possible) across tile shapes, pipeline depths and problem sizes.

The headline is uncomfortable and is the most useful thing here: **the throughput depends
almost entirely on how you time it.** A single launch reaches 865 TF/s; the same kernel called
back to back settles at 546 TF/s, and the gap is not throttling, not bandwidth and not the
tile geometry. Any number quoted without its protocol is meaningless, and several earlier
entries in this KB quoted such numbers.

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

Same kernel, same config (128/128/128, `num_stages=3`), three timing methods:

| shape | single launch, best of 20 events | back-to-back, batch of 20 | ncu single launch |
|---|---|---|---|
| 4096鲁 | 0.173 ms 閳?**792 TF/s** | 0.323 ms 閳?**425 TF/s** | 0.166 ms 閳?830 TF/s |
| 4096虏脳8192 | 0.318 ms 閳?**865 TF/s** | 0.491 ms 閳?560 TF/s | 閳?|
| 8192鲁 | 1.540 ms 閳?714 TF/s | 2.013 ms 閳?**546 TF/s** | 1.09 ms 閳?1008 TF/s |

Two separate effects are visible.

**There is a fixed gap between consecutive launches.** At 4096鲁 the kernel itself is 0.173 ms
but a batch of 20 averages 0.323 ms, so ~0.15 ms per call is spent between launches. That
overhead is roughly constant in shape, so it hurts small shapes hardest and barely touches
8192鲁 (1.54 閳?2.01).

**The coin-flip in the event numbers is real, not noise.** Single-launch 8192鲁 ranged 1.54 to
2.06 ms across 20 samples with a stable clock, which is why "median of 20" and "min of 20" give
534 and 714 for the same run.

**ncu disagrees with every event measurement and should not be used as a throughput anchor.**
It reports 1.09 ms at 8192鲁, which no event run reproduces. Its per-unit percentages are still
the best evidence about *where* the time goes, which is why they are quoted above, but its
duration is not comparable.

The honest reporting is therefore a pair: **single-launch best-case 865 TF/s at 4096虏脳8192,
and sustained back-to-back 425-560 TF/s depending on shape, settling at ~546 for the large
shapes.** For prefill (one big GEMM) use the first; for decode (many small GEMMs in a row) use
the second.

## Steady-state across shapes, back-to-back

Batch of 20, preallocated `C`, median of 3 batches, clock/power sampled per shape:

| shape | ms | TF/s | clk | W |
|---|---|---|---|---|
| 4096鲁 | 0.323 | 425.1 | 2325 | 97.5 |
| 4096虏脳8192 | 0.491 | 560.0 | 2325 | 97.5 |
| 4096虏脳16384 | 0.979 | 561.4 | 2325 | 97.5 |
| 8192虏脳4096 | 1.012 | 543.0 | 2325 | 97.5 |
| 8192鲁 | 2.013 | 546.2 | 2325 | 97.5 |
| 16384鲁 | 16.060 | 547.7 | 2325 | 97.4 |

Apart from the small 4096鲁 (hurt by the per-launch gap) everything lands in **543-561 TF/s**,
which is the sustained ceiling for this kernel on this card.

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
32768 columns make each operand 64 MB and the pair stops fitting in L2. So the ranking is:

- operands and C together fit in L2 (4096虏脳8192) 閳?**865 TF/s**, the best observed
- operands fit but C streams (8192虏脳4096) 閳?817 single-shot
- neither fits (16384鲁) 閳?548, and sustained is the same 546

That makes **865 TF/s a floor on the NVFP4 peak**, and the derived 592 TF/s (`296 脳 2`) is low
by **at least 46 %**. The `precision_roofline_all_stacks.md` peak table needs an FP4 correction,
and the correct worked example is 865, not 592 or 776.

## When needed (trigger)

Use this when picking a config for NVFP4 on sm120, when sizing a decode loop (use the
back-to-back number, not the single-shot one), or when someone proposes a larger tile to "use
the tensor cores better". The answer to the last one is no.

## Gotchas

- **Never quote a TF/s number without its protocol.** This single kernel spans 425-865 TF/s
  depending purely on how it is timed.
- `out_idx=[4]` allocates a fresh C per call. At 8192鲁 that is 268 MB and it dominates the
  measurement. Preallocate and pass C in.
- Do not use `ncu` durations as throughput; it reports 1.09 ms at 8192鲁 where events say 1.54.
  Use it for unit utilisation only.
- `BK=256` + `stages=3` fails to launch on the 99 KiB budget.
- The per-launch gap (~0.15 ms) is what makes small shapes look bad. If you need 4096鲁
  throughput, keep the kernel resident rather than relaunching.

## Sources

- `tilelang/cuda/op/gemm/gemm_mma_sm120.py` 閳?the sm120 blockscaled dispatch and its smem sizing
- `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py` 閳?the tile geometry and packed-scale read
- `scripts/patch_tilelang_sm120_nvfp4_scale.py` 閳?the fix this measurement depends on
- `hw_kb/tilelang/sm120/nvfp4_blockscaled_is_broken_in_tilelang.md` 閳?correctness history
- `hw_kb/tilelang/sm120/gemm_dtype_throughput.md` 閳?the fp8 tile sweep this mirrors
- `hw_kb/tilelang/sm120/gemm_tuning_summary.md` 閳?the fp16 power-wall result this one is not
