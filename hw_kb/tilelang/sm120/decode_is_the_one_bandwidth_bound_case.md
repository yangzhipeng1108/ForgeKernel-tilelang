# Decode (MHA, q = 16) is the one bandwidth-bound case — and the scheduling axes still tie

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp16, NCU census included.
> Correctness `cos = 1.00000` against `torch.nn.functional.scaled_dot_product_attention`.

## Purpose

Every dense-kernel recipe in this map so far concluded "compute-bound, so scheduling is a tie":
GEMM at `dram_pct ≈ 13 %`, prefill attention at `dram_pct ≈ 8.8 %`. The natural question is whether
a **decode** shape — tiny query tile, huge KV, weights streamed once — finally flips the balance and
makes L2 / swizzle / split-K pay off. This file answers it with a measured `dram_pct`.

Short answer: **decode is the first case that is bandwidth-bound** (`dram_pct ≈ 90 %`, real read BW
`880 GB/s` against a measured `1145 GB/s` peak), **and the scheduling axes still tie anyway** — because
the grid is only 32 blocks and swizzle only permutes *between* blocks. The real lever is byte count
(fp8 KV) but the fp8→fp16 conversion eats the savings. Recorded as two findings: "bandwidth-bound does
not imply schedulable" and "fp8 KV cache loses to its own upcast".

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- MHA decode: `H = 32` heads (no sharing), `Q_LEN = 16`, `KV_LEN = 8192`, `D = 128`, fp16, causal = off
- Baseline: `torch.nn.functional.scaled_dot_product_attention` (SDPA), same fp16 inputs
- Peak HBM read measured separately: `1145 GB/s` (`tensor.sum()` over 537 MB); copy `1273 GB/s`

## Why decode is different from prefill

Arithmetic intensity decides it:

| case | FLOP/byte | `dram_pct` | bound by |
|---|---|---|---|
| dense GEMM (`gemm_*.md`) | high | ~13 % | tensor cores |
| prefill attention, `SEQ 2048` | ~27 | 8.8 % | tensor cores |
| **MHA decode, `q 16, KV 8192`** | **16** | **90 %** | **HBM** |
| GQA decode (`HKV 8`, G = 4) | 63 | ~30 % | mixed |

Decode reuses almost nothing: the query is 16 rows but the KV tile is streamed in full for every
head, so bytes dominate. GQA (shared KV heads) raises intensity back up — 4 KV heads reused 8× —
which is why the GQA case at `BW ≈ 300 GB/s` was *not* bandwidth-bound and beat SDPA (1.40×).

## Finding 1 — bandwidth-bound, but swizzle still ties (grid too small to permute)

The MHA decode kernel moves `134 MB` of KV in `152.6 µs` → `880 GB/s`, **77 % of the measured
1145 GB/s peak**. NCU agrees it is the bottleneck (`dram 90 %` of theoretical peak). So by the usual
heuristic this is exactly where L2 scheduling should help. It does not:

| `T.use_swizzle` | TFLOP/s | time (µs) |
|---|---|---|
| off (baseline) | 14.0 | 152.9 |
| `panel_size=4`, row | 14.0 | 153.1 |
| `panel_size=8`, row | 14.0 | 153.4 |
| `panel_size=4`, column | 14.0 | 153.0 |

Identical to three significant figures. **The reason is structural: the grid is `H = 32` blocks.**
Swizzle reorders block *execution* to improve L2 reuse across blocks; with 32 blocks on 156 SMs
(0.21 waves) there are not enough blocks for the reorder to matter, and each block reads a disjoint
KV slice anyway, so there is no cross-block reuse to exploit.

Thread count does not move it either (`threads` 128 / 256 / 512 → 878–881 GB/s), and neither does
tile width beyond a floor:

| `threads` | `BN` | `stages` | BW (GB/s) |
|---|---|---|---|
| 128 | 64 | 1 | 748 |
| 128 | 128 | 1 | 878 |
| 128 | 64 | 2 | **881** |
| 256 / 512 | 64 / 128 | 1 / 2 | 877–881 |

The wall is `~880 GB/s`. Everything converges there.

## Finding 2 — split-KV cannot beat a bandwidth wall either

The standard fix for a starved grid is split-K; porting it to decode (split the KV range across
blocks, combine partial `(acc, l)` with `O = Σ acc_s / Σ l_s` after absolute rescale by `exp2(m_i)`)
does raise the grid from 32 to 512 blocks — and buys 3 %:

| split | grid | TFLOP/s | time (µs) | cos |
|---|---|---|---|---|
| 1 | 32 | 14.1 | 152.6 | 1.00000 |
| 4 | 128 | 14.5 | 147.9 | 1.00000 |
| 8 | 256 | 14.5 | 148.3 | 1.00000 |
| 16 | 512 | 14.5 | 147.8 | 1.00000 |

Correct (`cos = 1.00000`) but flat: **once HBM is saturated, adding blocks adds no bandwidth.** This
is the mirror image of `split_k_fills_a_starved_grid.md`, where split-K won 5.6× because the GEMM was
*compute*-bound with idle SMs. Same mechanism, opposite outcome, decided by which resource is saturated.

## Finding 3 — fp8 KV cache is slower, not faster (the upcast tax)

Halving KV bytes (fp16 → fp8 e4m3) should halve the bandwidth-bound time. It does not — it is *slower*:

| KV dtype | time (µs) | TFLOP/s | KV bytes | effective BW | cos |
|---|---|---|---|---|---|
| fp16 | **152.6** | **14.1** | 134.2 MB | 880 GB/s | 1.00000 |
| fp8 e4m3 | 191.7 | 11.2 | 67.1 MB | **350 GB/s** | 0.99927 |

The bytes halved but the effective bandwidth *dropped to 350 GB/s*: the kernel copies fp8 into shared
as fp8, then upcasts to fp16 shared (`T.copy(K8, Ks)`) because `T.gemm` will not take the fp8 KV
directly here. That upcast pass is now the serial bottleneck — you saved 67 MB of HBM traffic and
paid for it with an in-SMEM conversion stage that stalls the pipeline. **Net loss of 26 %.** This is a
concrete instance of the fp8 rule from `gemm_dtype_throughput.md`: fp8 pays off *only when it feeds
the MMA directly*, never when it must be upcast first.

(`BN = 128` fp8 does not fit `num_stages=2` — Q+K+V+P over the cap; `BN = 64` is the largest that runs.)

## What this closes

- **"Bandwidth-bound ⇒ scheduling helps" is false here.** The binding constraint on whether an L2 /
  swizzle axis can help is *grid size* (`#blocks ≫ SMs`), not the compute/bandwidth ratio. Decode has
  a saturated HBM *and* a grid too small to permute, so both scheduling and split-K tie.
- **fp8 KV cache needs an fp8 MMA path to pay off.** With TileLang `0.1.15`'s `T.gemm` (no direct
  fp8-KV attention GEMM that also does the PV in fp8) the upcast tax exceeds the bandwidth saving.
- The one real decode win is still the one from `flash_attention_is_compute_bound_too.md`: at small
  shapes the hand-written kernel beats SDPA (GQA decode 1.40×, `HQ 128` 3.79×), and GQA in particular
  is *not* bandwidth-bound (KV reuse lifts intensity to 63), so keep the shared-KV structure.

## When needed (trigger)

- **Decode with MHA (no KV sharing, `q ≪ KV`) is HBM-bound at ~880 GB/s.** Expect scheduling, swizzle,
  persistent, and split-K to all tie; there is no occupancy or L2 knob to reach for.
- **GQA / MQA decode is the case to prefer**: shared KV lifts intensity (63 for G = 4), the kernel is
  no longer bandwidth-bound, and the hand-written FA already beats SDPA (1.40× measured).
- **Do not ship an fp8 KV cache through an upcast path.** It measured 26 % *slower*. If fp8 KV is
  required, it needs a GEMM that consumes fp8 operands directly (an sm120 fp8 attention path), which
  `T.gemm` as used here does not provide.

## Gotchas

- **`T.gemm` needs `M % 16 == 0`**: a true decode with `q = 1` will not compile (`M must be divisible
  by 16, but got 1`). Pad the query tile to 16 rows (what was done here) or fold the decode query into
  the head dimension.
- **split-KV combine is `O = Σ acc_s / Σ l_s`**, *only after* rescaling each partial by `exp2(m_i)` to a
  common absolute scale. Forgetting the rescale (or taking `exp(l)`) gives `nan` or a wrong matrix.
- **Measuring peak bandwidth matters**: `dram_pct 90 %` is against *theoretical* peak; the *measured*
  achievable peak on this part is `1145 GB/s` (read) / `1273 GB/s` (copy). 880 GB/s is 77 % of
  achievable — so NCU's 90 % and the "there is 23 % headroom" reading are both true; the headroom is
  just not reachable through scheduling (see Finding 1).
- **A naive streaming-read microbenchmark with an in-SMEM reduction is not a bandwidth probe** — it
  measured 1 GB/s (shared-memory bound). Use `tensor.sum()` for a clean read-bandwidth number.

## Sources

1. TileLang `0.1.15`: `T.gemm` `M % kMPerWarp` divisibility check, `T.copy` fp8→fp16 in-SMEM upcast,
   `T.use_swizzle`, `T.Pipelined(num_stages=)`; the smem-cap error text for `BN = 128`, `stages = 2`.
2. `flash_attention_is_compute_bound_too.md` (the prefill case, `dram 8.8 %`, and the small-shape wins
   that carry over here), `split_k_fills_a_starved_grid.md` (same split-K mechanism, opposite result
   because the GEMM there was compute-bound), `gemm_dtype_throughput.md` (fp8 only helps when it feeds
   the MMA directly — Finding 3 is the counterexample).
3. This repo's probes on an RTX 6000D, clock-locked, NCU census via `ncu -k regex:main_kernel`; peak
   HBM measured with `tensor.sum()` over a 512 MB fp16 buffer.
