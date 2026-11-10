# Head dim decides the pipeline-stage verdict: `stages=2` fits at `D=64`, not at `D=128`

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp16, `cos = 1.00000` on every runnable cell.
> Correctness `cos = 1.00000` against `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`.

## Purpose

Every attention recipe in this map locked `D = 128` (LLaMA / Qwen / DeepSeek hidden head). But the
pipeline-stage budget is set by the shared-memory footprint, which scales with `D` — so the "`stages = 1`
only" verdict from `flash_attention_is_compute_bound_too.md` is a statement about `D = 128`, not about
attention in general. This file sweeps `D ∈ {64, 96, 128}` and checks when `stages = 2` becomes viable.

Short answer: **`D = 64` lets `stages = 2` fit, and it wins (293.5 vs 299.3 µs); `D = 128` cannot fit a
second stage at all (SMEM cap).** `D = 96` is awkward: it is not a multiple that lets the wide tiles run,
so most cells are SMEM-limited and only `BM 64 / BN 128 / stages 1` compiles. The stage verdict is a
function of `(BM + 2·BN)·D`, so **the head dimension, not the kernel, decides it.**

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- Causal FA-2, `H = 16`, `SEQ = 4096`, `D ∈ {64, 96, 128}`, fp16, `BM/BN` and `stages` swept
- Per-stage SMEM = `(BM + 2·BN)·D·2` bytes (Q tile + K tile + V tile), double-buffered by `stages`
- Baseline: the `D=128, BM64, BN128, stages1` cell from `causal_attention_skips_the_masked_half.md`

## Finding 1 — `stages=2` fits and wins at `D=64`

| `D` | `BM` | `BN` | `stages` | time (µs) | eff TF | cos |
|---|---|---|---|---|---|---|
| **64** | 64 | 128 | 1 | 299.3 | 114.8 | 1.00000 |
| **64** | 64 | 128 | **2** | **293.5** | **117.1** | 1.00000 |
| 64 | 64 | 64 | 2 | 385.0 | 89.2 | 1.00000 |
| 64 | 128 | 128 | 1 | 418.0 | 82.2 | 1.00000 |
| 64 | 128 | 64 | 2 | 403.4 | 85.2 | 1.00000 |
| 64 | 128 | 128 | 2 | — | — | SMEM cap |

At `D = 64` the per-stage footprint is half of `D = 128`, so `BM64 / BN128` double-buffers within the
99 KiB cap and `stages = 2` **edges out** `stages = 1` (293.5 vs 299.3 µs, +1.9 %). It is a small but real
win, and the opposite of the `D = 128` verdict. Note the absolute efficiency is *lower* at `D = 64`
(117 vs 149 eff TF) — smaller `D` is less arithmetic per byte — but the stage choice still flips.

## Finding 2 — `D=128` cannot fit `stages=2` in any tile

Every `D = 128, stages = 2` cell fails on the SMEM cap:

| `D` | `BM` | `BN` | `stages` | result |
|---|---|---|---|---|
| 128 | 64 | 128 | 2 | SMEM cap (`164 800` B request) |
| 128 | 128 | 64 | 2 | SMEM cap |
| 128 | 128 | 128 | 1 | SMEM cap |
| 128 | 64 | 128 | 1 | **459.8 µs, 149.4 TF** (the only good cell) |

At `D = 128` the `BM64 / BN128` single stage already uses most of the budget; a second stage is `2×` that
and overflows. This is why every `D = 128` recipe in the map says "`stages = 1`".

## Finding 3 — `D=96` is a dead zone (not a multiple that tiles cleanly)

| `D` | `BM` | `BN` | `stages` | time (µs) | eff TF |
|---|---|---|---|---|---|
| 96 | 64 | 128 | 1 | 384.3 | 134.1 |
| 96 | 64 | 128 | 2 | SMEM cap | — |
| 96 | 64 | 64 | 2 | 459.4 | 112.2 |
| 96 | 128 | 128 | 1 | SMEM cap | — |
| 96 | 128 | 64 | 2 | 504.7 | 102.1 |

`D = 96` sits between: the wide tiles overflow, and only `BM64 / BN128 / stages1` (384.3 µs) is
competitive. It is not a useful operating point — if a model uses `D = 96`, expect to fall back to the
`stages = 1` narrow configuration and accept ~134 eff TF (below both `D = 64` and `D = 128`). This is a
shape that wants padding to `D = 128` if the numerics allow.

## Finding 4 — fp8 flips the stage verdict again: at `D=64`, `stages=2` wins for fp16 but `stages=1` for fp8

The stage choice is not just a function of `D` — **the dtype moves it too**, because fp8 halves the K
tile so the double-buffer's benefit changes:

| `D` | dtype | `stages` | time (µs) | eff TF | cos |
|---|---|---|---|---|---|
| 64 | fp16 | 1 | 299.0 | 114.9 | 1.00000 |
| 64 | fp16 | **2** | **293.5** | **117.1** | 1.00000 |
| 64 | **fp8 QK** | **1** | **255.1** | **134.7** | 0.99938 |
| 64 | fp8 QK | 2 | 264.7 | 129.8 | 0.99938 |
| 128 | fp16 | 1 | 459.9 | 149.4 | 1.00000 |
| 128 | fp8 QK | 1 | 380.1 | 180.8 | 0.99941 |

Two flips:

1. **dtype beats the stage win.** `D=64` fp8 `stages=1` (255.1 µs) is faster than `D=64` fp16 `stages=2`
   (293.5 µs) — fp8 QK is worth more than the double-buffer, so the fp16 `stages=2` conclusion is moot once
   fp8 is on.
2. **fp8 `stages=1` beats fp8 `stages=2`** (255.1 vs 264.7) — the *opposite* of the fp16 verdict at the
   same `D`. With the K tile halved, the second pipeline stage adds buffering overhead without hiding more
   latency, so the fp16 `stages=2` preference does not carry to fp8.

So the operating point at `D=64` is **fp8 QK + `stages=1`** (255.1 µs, 1.15× over the fp16 best), not
`fp16 + stages=2`. **The stage verdict is a function of `(BM + 2·BN)·D·2·bytes_per_elem·stages` — `D` *and*
dtype both move it.** Do not port a stage conclusion across either.

## When needed (trigger)

- **The pipeline-stage verdict depends on `D` and dtype.** At `D = 64`: fp16 prefers `stages = 2`, but
  **fp8 prefers `stages = 1`** (and fp8 `stages=1` is faster than fp16 `stages=2`). At `D = 128`
  `stages = 2` does not fit at all. Compute `(BM + 2·BN)·D·2·bytes·stages` against the ~99 KiB cap.
- **For `D=64`, the best cell is fp8 QK + `stages=1`** (255.1 µs, 134.7 eff TF, `cos 0.9994`), beating the
  fp16 double-buffered cell by 1.15×.
- **`D = 96` is an awkward head dim**: narrow to `BM64 / BN128 / stages1` (~134 eff TF) or pad to 128.
- **`BM 64 / BN 128` is the robust tile across all `D`** — it is the only wide cell that compiles at
  `D = 96` and the best cell at `D = 64` and `D = 128`.

## Gotchas

- **The SMEM formula is `(BM + 2·BN)·D·bytes_per_elem` per stage** (Q + K + V tiles); the P tile `BM·BN·2`
  is separate and small. fp8 halves `bytes_per_elem` on K (and Q), which is *why* fp8 changes the stage
  boundary. Use the formula to predict fit before compiling — the cap errors report the exact bytes.
- **`cos = 1.0` on every cell including the ones that run slower** — this is a pure performance sweep, no
  numerics change; do not read the `D=96` fallback as a correctness problem.
- **Do not carry a stage verdict across dtype.** fp16 `stages=2` at `D=64` does *not* imply fp8
  `stages=2` — fp8 wants `stages=1` there. The two are different kernels' worth of SMEM budget.
- **fp8 QK keeps V and P fp16** (see `fp8_qk_ports_split_k_ties.md`) — the "fp8" here is `Q,K` only; the
  V tile and P tile stay 2-byte, which is part of why the SMEM math is asymmetric.

## Sources

1. TileLang `0.1.15`: `T.Pipelined(num_stages=)` and the SMEM-cap error text (`Failed to set the allowed
   dynamic shared memory size to …`) that pins the per-stage footprint.
2. `flash_attention_is_compute_bound_too.md` (the `D=128 stages=1` baseline this generalizes),
   `causal_attention_skips_the_masked_half.md` (the `BM64/BN128` causal tile), `software_pipeline_stage_budget.md`
   (the stage-budget concept, measured there for a GEMM).
3. This repo's probes on an RTX 6000D, clock-locked; per-stage SMEM computed and matched to the cap-error
   byte counts.
