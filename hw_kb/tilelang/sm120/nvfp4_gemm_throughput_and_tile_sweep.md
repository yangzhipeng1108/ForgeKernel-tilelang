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

A 120-iteration run at 8192³, sampling clock and power between batches:

| batch start | SM clock MHz | power W | T °C | ms/iter | TF/s |
|---|---|---|---|---|---|
| 0 | 2325 | 97.5 | 43 | 7.331 | 150.0 |
| 20 | 2325 | 97.5 | 43 | 1.940 | 566.7 |
| 40 | 2325 | 97.5 | 43 | 1.883 | 584.1 |
| 60 | 2325 | 97.5 | 43 | 1.936 | 567.8 |
| 80 | 2325 | 97.5 | 43 | 1.917 | 573.6 |
| 100 | 2325 | 97.5 | 43 | 1.892 | 581.2 |

Clock flat at 2325 MHz for the whole run, power flat at **97 W against a 600 W limit**, die at
43 °C. Batch 0 is slow only because it is cold. **This kernel never comes close to the power
wall**, which is the opposite of the fp16 large-tile case in `gemm_tuning_summary.md`. Locking
clocks here would buy nothing.

## It is not the bandwidth wall either

`ncu` on the same kernel:

| metric | 4096³ | 8192³ |
|---|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 19.8 % | 20.0 % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 64.5 % | 71.5 % |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | → | 32.0 % |
| `launch__waves_per_multiprocessor` | → | 26.3 |

DRAM sits at **20 %** at both sizes, so nothing here is memory bound. The tensor pipe is the
busiest unit at 64-72 %, and only a third of the warps are resident at any time. The kernel
feeds the MMA pipe well but leaves a quarter to a third of it idle.

## The protocol controls the answer

Same kernel, same config (128/128/128, `num_stages=3`), several timing methods. Every row below
was reproduced three times on a fresh process:

| shape | plain loop | CUDA graph | ncu single launch |
|---|---|---|---|
| 4096³ | 0.164-0.165 ms → **835-836 TF/s** | 0.265-0.270 ms → 508-518 TF/s | 0.166 ms → 830 TF/s |
| 4096²×8192 | 0.318 ms → **865 TF/s** | → | → |
| 8192³ | 2.02-2.03 ms → **541-544 TF/s** | 2.00 ms → 549-551 TF/s | 1.08 ms → 1008 TF/s |

Three things fall out.

**It is not launch overhead.** Capturing the loop in a CUDA graph removes every CPU-side launch,
and at 8192³ it changes nothing (2.02 → 2.00 ms). At 4096³ the graph is **40 % slower**, not
faster, so the per-launch path is not where the time goes and batching it into a graph actively
hurts. Whatever the wrapper does per call is invisible next to the kernel.

**The output buffer is not the issue either.** Rotating over eight distinct `C` tensors is
identical to reusing one (2.007 vs 2.006 ms at 8192³), so there is no write-after-write
serialisation.

**4096³ and 8192³ differ because of L2, and nothing else.** Both run the same tile and the same
code. At 4096³ the two fp4 operands (32 MB each) plus the fp32 output (64 MB) total 128 MB, which
is exactly this part's L2; the operands stay resident across the K loop and the kernel reaches
**836 TF/s**. At 8192³ the output alone is 268 MB, so it evicts the operands and every K
iteration refetches them from HBM; the kernel settles at **545 TF/s**. That is a 35 % swing from
cache residency on identical compute.

Even so it is **not a bandwidth problem at 8192³**: 336 MB of traffic over 2.0 ms is 167 GB/s
against a measured 1273 GB/s copy rate, i.e. 13 % of peak. The cost of the L2 miss here is
latency that a three-stage pipeline cannot fully hide, not throughput.

**ncu's duration is not usable as a throughput anchor.** It reports 1.08 ms at 8192³, stable
across launches, where every event measurement and the CUDA graph agree on 2.0 ms. Its unit
percentages are still the best evidence about *where* time goes, which is why they are quoted
above, but its duration is discarded.

The honest reporting is therefore a pair: **835-865 TF/s when the operands stay in L2, 545 TF/s
when they do not.** For prefill on shapes up to ~4096³ use the first; for larger shapes, and for
decode loops, use the second.

## Steady-state across shapes

Warmup 5, then 40 calls with events around the whole block; reproduced three times:

| shape | ms | TF/s | L2 state |
|---|---|---|---|
| 4096³ | 0.164 | **835.9** | A+B+C = 128 MB, all resident |
| 4096²×8192 | 0.318 | **865.2** | operands 32 MB each, resident |
| 4096²×16384 | 0.979 | 561.4 | operands 64 MB each, partially resident |
| 8192²×4096 | 1.012 | 543.0 | C 268 MB evicts the operands |
| 8192³ | 2.025 | 542-544 | as above |
| 16384³ | 16.060 | 547.7 | nothing resident |

The pattern is entirely explained by whether the fp4 operands survive in L2. Everything above
about 128 MB of combined footprint lands in **540-560 TF/s**; everything below it reaches
**835-865**. There is no separate "large shape" penalty.

## The tile axis is closed at 128/128/128, stages 3

Single-launch events, `4096³`, 256 threads:

| BM | BN | BK | stages | ms | TF/s |
|---|---|---|---|---|---|
| 128 | 128 | 128 | 2 | 0.193 | 711.6 |
| **128** | **128** | **128** | **3** | **0.177** | **775.7** |
| 128 | 128 | 128 | 4 | 0.201 | 685.0 |
| 128 | 128 | 256 | 2 | 0.212 | 648.5 |
| 64 | 128 | 128 | 3 | 0.227 | 605.1 |
| 256 | 128 | 128 | 2 | 0.630 | 218.1 |

`8192³`, single-launch events:

| BM | BN | BK | stages | ms | TF/s |
|---|---|---|---|---|---|
| 128 | 128 | 128 | 3 | 2.073 | 530.3 |
| 128 | 128 | 256 | 2 | 2.071 | 531.0 |
| 128 | 128 | 64 | 3 | 2.184 | 503.3 |
| 256 | 128 | 128 | 2 | 4.695 | 234.2 |
| 128 | 256 | 128 | 2 | 5.400 | 203.6 |
| 256 | 256 | 64 | 2 | 12.141 | 90.6 |

The ranking holds under either protocol: **128/128/128 with three stages wins and every larger
tile collapses**. The cause is **register pressure on the accumulator, not smem** — a 192×192
fp32 fragment is 147 KB of registers per block, past the 255-register limit, so it spills. See
"Why the tile axis is closed" below. `BK=256` with `stages=3` does exceed smem and does not
launch, but that is the only smem-limited case found.

## Where the ceiling actually is

The `4096²×32768` probe (C L2-resident, long K) returns 712 TF/s, *below* 4096³, because
32768 columns make each operand 64 MB and the pair stops fitting in L2 — the same eviction
effect, caught in the act. So the ranking is:

- operands and output fit in L2 (4096³, 4096²×8192) → **836-865 TF/s**, the best observed
- output evicts the operands (8192²×4096, 8192³, 16384³) → **543-561 TF/s**
- long K pushes the operands themselves out (4096²×32768) → 712 TF/s

That makes **865 TF/s a floor on the NVFP4 peak**, and the derived 592 TF/s (`296 × 2`) is low
by **at least 46 %**. The `precision_roofline_all_stacks.md` peak table needs an FP4 correction,
and the correct worked example is 865, not 592 and not 776.

## How to close the 545 -> 865 gap: three levers, all measured, all closed

The eviction penalty is latency, not bandwidth: at 8192³ the kernel moves 336 MB in 2.0 ms
(167 GB/s, 13 % of peak) yet loses 35 %. Three-stage pipelining does not cover the HBM latency
once every K iteration misses. Three candidate fixes were proposed and every one was run.

**Lever 1 — block the output columns so the operand slice stays resident.** Grid becomes
(M-tiles, panels) and each block walks a panel's N tiles. It loses at every panel size:

| variant | 8192³ ms | TF/s | 4096³ ms | TF/s |
|---|---|---|---|---|
| plain | 2.014 | 545.9 | 0.165 | 834.6 |
| panel N=512 | 2.078 | 529.1 | 0.185 | 744.9 |
| panel N=1024 | 2.127 | 516.9 | 0.183 | 753.1 |
| panel N=2048 | 2.271 | 484.2 | 0.527 | 260.6 |

The arithmetic is unforgiving: each panel re-reads A and B, and 33 MB × 2 × 16 panels costs
more than the eviction it avoids. **Re-reading the operands is worse than letting them be
evicted**, which is worth knowing before anyone proposes tiling for cache again.

**Lever 2 — deeper pipelining.** The earlier claim that the 99 KiB budget blocked `stages > 3`
was **wrong**. Measured dynamic smem, 128/128/128:

| stages | BK | smem B | 8192³ TF/s | 4096³ TF/s |
|---|---|---|---|---|
| 3 | 128 | 52,224 | 541.4 | 833.7 |
| 4 | 128 | 69,632 | 545.4 | 719.9 |
| 4 | 64 | 34,816 | 517.2 | 584.8 |
| 6 | 64 | 52,224 | 520.6 | 583.1 |
| 8 | 64 | 69,632 | 423.0 | 409.3 |

Four stages fit comfortably (69 KB of a 101,376 B budget) and buy nothing at 8192³ while costing
14 % at 4096³. Smaller BK is worse at every depth. **Deeper pipelining is not the lever**, and
the smem budget was never the constraint it was said to be.

**Lever 3 — a persistent kernel.** Grid sized to the SM count (156) or twice that, each block
walking a strided tile list. Identical to plain: 2.027 / 2.021 ms at 8192³, 0.167 ms at 4096³.
No scheduling cost to recover, so no gain.

**Lever 4 — cache eviction hints.** `T.copy` takes
`eviction_policy in {"evict_normal", "evict_first", "evict_last"}`, which maps to the TMA cache
hint. The idea is to pin the operands (`evict_last`) and stream the output (`evict_first`). A
single run looked promising → 542.3 → 555.2 at 8192³ — so it was reproduced three times in
fresh processes, and the effect dissolved into the run-to-run spread:

| variant | run 1 | run 2 | run 3 | best |
|---|---|---|---|---|
| base | 541.7 | 550.9 | 549.4 | 550.9 |
| A/B `evict_last` | 535.7 | 550.7 | 543.8 | 550.7 |
| A/B + SF `evict_last` | 541.1 | 553.4 | 544.2 | 553.4 |
| A/B `last`, SF `first` | 539.5 | 544.5 | 549.2 | 549.2 |
| all `evict_last` | 540.0 | 545.5 | 548.7 | 548.7 |

**The base config alone spans 541.7-550.9, i.e. ±1.7 %.** The apparent 2.4 % gain from
`evict_last` is smaller than the noise of the control, so it is not a result. `evict_first` on
the output changes nothing at all, which suggests the C store is not competing for L2 residency
in the first place.

> **Method note.** This is the trap that nearly produced a false positive: a single 2.4 % run
> looks like a finding, and the only defence is repeating the control. Anything smaller than the
> control's own spread has to be reported as no effect.

## The 545 ceiling is real

Four independent attacks on the L2-eviction penalty — column panels, deeper pipelining,
persistent grids, eviction hints — all fail. None of them changes how much data the K loop
must refetch, and the one that could in principle (panels, by making the working set small
enough to stay resident) pays more in re-reads than it saves. The conclusion is that **545 TF/s
is this hardware's throughput for this kernel when the operands cannot stay in L2**, and there
is no configuration knob on the current structure that moves it.

Getting past it would need a different shape of solution — an algorithm that does not re-read
the operands per K iteration at large N, such as a fused producer-consumer split across blocks
(split-K with partial-sum reduction, `split_k_fills_a_starved_grid.md`) or a layout that keeps
each operand slice hot by construction. Neither is measured, and both change the numerical
contract, so they are out of scope for a drop-in GEMM.

## Why the tile axis is closed: registers, not smem

The retry of large tiles with `stages=3`, now that the smem headroom is known to be real:

| BM | BN | stages | smem B | 4096³ TF/s | 8192³ TF/s |
|---|---|---|---|---|---|
| 128 | 128 | 3 | 52,224 | **836.2** | **537.3** |
| 192 | 128 | 3 | 66,048 | 601.2 | 546.6 |
| 128 | 192 | 3 | 66,048 | 511.3 | 536.0 |
| 192 | 192 | 3 | 78,336 | **120.2** | 135.7 |
| 256 | 128 | 3 | 79,872 | 204.4 | 230.1 |
| 128 | 256 | 3 | 79,872 | 166.2 | 189.0 |
| 256 | 256 | 2 | 69,632 | 76.2 | 78.0 |

192×192 collapses to 120 TF/s while using only 78 KB of smem, so the collapse is **not** the
shared-memory budget. The accumulator is: a 192×192 fp32 fragment is 147 KB of registers per
block, and with 256 threads that is 576 B/thread — far past the 255-register limit, so it spills.
`256×128` (128 KB accumulator, 512 B/thread) spills too. Only 128×128 keeps the accumulator at
64 KB, i.e. 256 B/thread, which fits. **The tile axis is closed by the register file, and the
smem budget is irrelevant at these shapes.**

## When needed (trigger)

Use this when picking a config for NVFP4 on sm120, when sizing a decode loop (use the
back-to-back number, not the single-shot one), or when someone proposes a larger tile to "use
the tensor cores better". The answer to the last one is no.

## Gotchas

- **Never quote a TF/s number without its protocol.** This single kernel spans 508-865 TF/s
  depending purely on how it is timed.
- `out_idx=[4]` allocates a fresh C per call. At 8192³ that is 268 MB and it dominates the
  measurement. Preallocate and pass C in.
- **CUDA graphs make this kernel slower, not faster** (4096³: 0.164 → 0.270 ms). Do not reach
  for graph capture to close a launch-overhead gap here; there is no such gap to close.
- Do not use `ncu` durations as throughput; it reports 1.08 ms at 8192³ where events and graphs
  agree on 2.0. Use it for unit utilisation only.
- `BK=256` + `stages=3` fails to launch on smem. That is the only smem-limited combination
  found; `BK=128` fits at `stages=4` (69,632 B of 101,376) and still loses, so do not blame smem
  for the large-tile collapse — blame the accumulator's register footprint.
- **Measure the shape you care about, in the order you will run it.** An earlier round measured
  4096³ at 425 TF/s purely because a larger shape had run first and left L2 in a different
  state; the same kernel in a fresh process gives 836.
- Column-panel blocking and persistent grids both lose. If a later reader is tempted to tile for
  cache, the numbers above say the re-read cost exceeds the eviction cost on this part.

## Sources

- `tilelang/cuda/op/gemm/gemm_mma_sm120.py` — the sm120 blockscaled dispatch and its smem sizing
- `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py` — the tile geometry and packed-scale read
- `scripts/patch_tilelang_sm120_nvfp4_scale.py` — the fix this measurement depends on
- `hw_kb/tilelang/sm120/nvfp4_blockscaled_is_broken_in_tilelang.md` — correctness history
- `hw_kb/tilelang/sm120/gemm_dtype_throughput.md` — the fp8 tile sweep this mirrors
- `hw_kb/tilelang/sm120/gemm_tuning_summary.md` — the fp16 power-wall result this one is not
