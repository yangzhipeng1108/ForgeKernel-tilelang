# LM head (skinny GEMM): split-K does not help, block-sparsity would — it is the wide-N case, not the starved case

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp16, `cos = 1.00000` on every cell.
> Correctness `cos = 1.00000` against a float32 `X @ W^T` reference.

## Purpose

The LM head is `logits = X (M, K) @ W_vocab^T` with `M = batch·seq` tiny (decode) and `N = vocab`
huge (128k+). It *looks* like the textbook skinny GEMM that `split_k_fills_a_starved_grid.md` says to
fix with split-K — small `M`, enormous `K`-per-output would be the mental model. This file checks
whether split-K and the other starvation remedies actually apply, and where the real limit is.

Short answer: **no — split-K loses here, for the same reason it loses on the filled 4096³ GEMM.** The
`N = vocab` dimension makes the grid *wide*, not starved (`2.5–6.4 waves` at `M = 16–64`), so split-K
only adds atomic traffic. At the tiniest `M = 16` the kernel is finally **bandwidth-bound** (`945 GB/s`,
83 % of the 1145 GB/s peak) — and the lever there is **bytes** (the `W_vocab` weight, 209 MB), which is
where block-sparsity over the vocabulary would apply, not split-K.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- Shapes: `(M, N, K) ∈ {(64, 128256, 4096), (64, 151936, 3584), (256, 128256, 4096), (16, 50257, 2048)}`
- Reference: float32 `X @ W^T`; fp16 inputs, fp32 accumulator and output
- Baseline contrast: `split_k_fills_a_starved_grid.md` (the starved-GEMM case where split-K won 4–5.6×)

## Finding 1 — split-K is a monotonic loss on every LM-head shape

| shape | SK 1 (µs / TF) | SK 4 | SK 8 |
|---|---|---|---|
| `M64 N128256 K4096` | **939 / 71.6** | 1113 / 60.4 | 1288 / 52.2 |
| `M64 N151936 K3584` | **987 / 70.6** | 1188 / 58.7 | 1424 / 48.9 |
| `M256 N128256 K4096` | **3565 / 75.4** | 4259 / 63.2 | 4985 / 54.0 |
| `M16 N50257 K2048` | **230 / 14.3** | 231 / 14.2 | 230 / 14.3 |

Strictly worse with more splits (down to 52 TF at SK 8), except the `M16` case where it is flat at
~14 TF — split-K neither helps nor hurts there, it just adds atomics for nothing. This is **not** a
contradiction of `split_k_fills_a_starved_grid.md`; it is the same rule applied to the right knob:

| case | grid | waves (÷156) | split-K |
|---|---|---|---|
| 512²×16384 (the starved GEMM) | 16 tiles | **0.10** | **+4×** |
| LM head `M16 N50257 BN128` | 393 | **2.5** | tie |
| LM head `M64 N128256 BN128` | 1002 | **6.4** | −18 % |

`split-K` only pays when the grid is **under one wave** (`< 156` tiles). A vocab-sized `N` guarantees
hundreds to thousands of tiles, so the LM head is never starved. **The word "skinny" refers to `M`, but
starvation is a property of the whole grid (`M·N` tiles); `N = vocab` fills it.**

## Finding 2 — the tiny-`M` LM head is bandwidth-bound, and wide `N`/`BK` is the only knob

At `M = 16, N = 50257, K = 2048` the kernel moves `209 MB` (almost all of it `W_vocab`) and reads
`945 GB/s` — **83 % of the measured 1145 GB/s peak** — so it is the second bandwidth-bound case in the
map after MHA decode. And, exactly as in `decode_is_the_one_bandwidth_bound_case.md`, the tile sweep
just walks up to the wall:

| `BM` | `BN` | `BK` | time (µs) | BW (GB/s) | TF |
|---|---|---|---|---|---|
| 16 | 128 | 64 | 221.4 | 945 | 14.9 |
| 16 | 128 | 128 | 220.1 | 950 | 15.0 |
| 16 | 256 | 64 | 225.1 | 929 | 14.6 |
| 32 | 256 | 64 | 227.2 | 920 | 14.5 |

Everything sits at 920–950 GB/s. The wide-`N`/`BK=128` cell is marginally best (950 GB/s) but the
spread is 3 %. For contrast, `torch`'s fp32 `linear` on the same shape is **1066 µs / 3.1 TF** — the
hand-written fp16 kernel is already 4.8× faster, so there is no low-hanging fruit here either.

## Finding 3 — the real lever is bytes: block-sparsity over the vocabulary

`BW` is at the wall (83 %) and `intensity = 15.7` FLOP/byte — the weight matrix dominates. The way to go
faster is to **not read all of `W_vocab`**: a softmax-gated vocabulary, an `argmax`-only decode (top-k
logits), or a candidate shortlist means only a subset of the `N` rows is ever needed. That is the
**block-tile-skipping** mechanism from `sparse_gemm_two_mechanisms.md` applied along `N` (launch only the
selected vocab-block tiles from a compacted list) — the same "reduce the bytes" pattern that carries
`fp8`, `split-K` on starved grids, `2:4`, `causal tile-skip`, and `MLA`. Measured here only as the
mechanism being available (tile-skip is linear in skipped tiles, `sparse_gemm_two_mechanisms.md`); the
vocabulary-selection policy itself was not part of this probe.

## When needed (trigger)

- **Never split-K an LM head.** `N = vocab` gives a wide grid (2.5–6.4 waves); split-K is a −18 % loss or
  a flat tie. The starvation test is `M·N/BM/BN < 156` tiles — vocab-sized `N` never passes it.
- **A `M ≤ 16` LM head is bandwidth-bound** (`~945 GB/s`, 83 % of peak); expect the tile sweep to walk up
  to the wall and stop. There is no schedule/tile win past it.
- **If an LM head must be faster, cut the bytes** — block-sparse over `N` (skip unneeded vocab tiles),
  exactly as `sparse_gemm_two_mechanisms.md` measured for dense tiles. That is the only lever the
  bandwidth wall leaves.

## Gotchas

- **`M < 16` does not compile** (`T.gemm` needs `M % 16 == 0`) — a true `M = 1` decode LM head must pad
  the query rows to 16 or fold the batch/seq axis.
- **fp32 output dominates the write traffic at small `M`**: `M·N·4` bytes of logits can rival the
  `N·K·2` weight read (here 8.2 MB output vs 209 MB weight — weight still wins, but at larger `M` the
  output write becomes the second-order term; cast to fp16 output if the consumer allows).
- **"skinny" is about `M`, starvation is about `M·N`.** Do not infer split-K from `M` alone — compute
  the tile count first (the whole point of Finding 1).

## Sources

1. TileLang `0.1.15`: `T.gemm` `M % 16` gate, `T.atomic_add` for the split-K variant, the SMEM-cap
   error for `BN 512` / `BK 128` at `BN 256`.
2. `split_k_fills_a_starved_grid.md` (the grid-fill `< 156` test this file re-applies), `sparse_gemm_two_mechanisms.md`
   (block-tile skipping, linear in skipped tiles — the byte-reduction lever Finding 3 points at),
   `decode_is_the_one_bandwidth_bound_case.md` (the other bandwidth-bound case, same "wall, no schedule" behavior).
3. This repo's probes on an RTX 6000D, clock-locked; grid waves and peak-BW ratios computed and compared
   to measured time so the split-K verdict is attributed to grid fill and nothing else.
