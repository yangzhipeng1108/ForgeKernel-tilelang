# Porting GEMM levers to attention: split-K ties, but fp8 QK^T ports at 1.2–1.4× (do not upcast V)

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp16/fp8, `cos = 1.00000` (f16) and
> `0.9994` (fp8 QK) against `torch.nn.functional.scaled_dot_product_attention`.

## Purpose

The GEMM map's two winners — **split-K** (grid shaping) and **fp8** (halve operand bytes) — are the natural
things to port to attention. This file measures both on the causal and non-causal FA, and corrects an
earlier conclusion: a first pass that used an **fp8 V upcast** made fp8 look like a ~1 % tie, which is
wrong. Done correctly — **Q and K in fp8, V and P left in fp16** — fp8 QK^T is a solid **1.2–1.4× on every
shape**, causal included.

Short answer: **split-K does not port** (the prefill grid is always full); **fp8 QK^T does port, at 1.2–1.4×,
but the gain is destroyed if you quantize or upcast `V`/`P`.** The earlier "+1 %" was an artifact of
routing `V` through an fp8→fp16 conversion.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- Causal and non-causal FA-2 forward, `H = 16`, `SEQ ∈ {2048, 4096, 8192}`, `D = 128`, `BM = 64`, `BN = 128`
- Baseline: fp16 FA (`causal_attention_skips_the_masked_half.md` for causal); `cos` vs SDPA
- split-K here splits the **k-block range** of each query tile (partials combined `O = Σ acc_s / Σ l_s`
  after rescaling each by `exp2(m_i)`).

## Finding 1 — split-K is a tie (the prefill grid is never starved)

Causal FA, `SEQ 4096`:

| split | time (µs) | eff TF | cos |
|---|---|---|---|
| 1 | 448.4 | 153.2 | 1.00000 |
| 2 | **436.8** | **157.3** | 1.00000 |
| 4 | 455.5 | 150.9 | 1.00000 |

±2.6 % — a tie. The attention grid is `(SEQ/BM)·H = 64·16 = 1024` blocks = **6.6 waves**; split-K only
pays below **one** wave (`split_k_fills_a_starved_grid.md`). Prefill is never starved, so there is nothing
to fill. (Split-K ties in decode too, where the grid is `H = 32` — but decode is bandwidth-bound, so it
ties for a different reason: `decode_is_the_one_bandwidth_bound_case.md`.)

## Finding 2 — fp8 QK^T ports at 1.2–1.4×, on causal **and** non-causal

The correct configuration: **`Q, K` in fp8 e4m3, `V` and `P` in fp16.** Across the length sweep:

| SEQ | mode | fp16 (µs) | fp8 QK (µs) | speedup | cos |
|---|---|---|---|---|---|
| 2048 | causal | 152.2 | 126.6 | **1.20×** | 0.9994 |
| 2048 | full | 239.9 | 198.7 | **1.21×** | 0.9992 |
| 4096 | causal | 459.7 | 380.1 | **1.21×** | 0.9994 |
| 4096 | full | 892.3 | 639.2 | **1.40×** | 0.9992 |
| 8192 | causal | 1929.5 | 1379.7 | **1.40×** | 0.9994 |
| 8192 | full | 3928.1 | 2901.1 | **1.35×** | 0.9993 |

fp8 QK^T is a win everywhere — **and notably it survives causal skipping**, which an earlier draft got
wrong. The full breakdown of *which* operand dtypes (SEQ 4096, causal):

| config | time (µs) | eff TF | cos |
|---|---|---|---|
| fp16 QK, fp16 PV | 459.7 | 149.5 | 1.00000 |
| **fp8 QK, fp16 PV (native V)** | **380.9** | **180.4** | **0.99941** |
| fp8 QK, fp8 P × fp8 V | 409.2 | 167.9 | 0.99885 |
| fp8 QK, fp8 P × fp16 V | — | — | **rejected** (mixed dtype) |

The **fp8 QK + native fp16 V** row is the winner (1.21×). Quantizing `P` to fp8 as well is **slower**
(409 vs 381 µs) *and* less accurate (0.99885 vs 0.99941) — the softmax probabilities lose resolution and
the PV GEMM gains nothing. And `P fp8 × V fp16` is rejected outright by the mixed-dtype limit.

**Why the earlier "+1 %" was wrong**: that probe cast `V` fp8→fp16 in SMEM before the PV GEMM, paying the
upcast tax twice (once on the load path, once on the conversion) and erasing the QK saving. With `V` kept
natively fp16 there is no upcast at all, and the fp8 saving on `Q`/`K` stands.

## Finding 3 — the rule, corrected

| lever | binds on | pays in attention? | condition |
|---|---|---|---|
| split-K | idle SMs (grid < 1 wave) | **no** — grid is 6.6 waves | prefill grid always full |
| **fp8 QK^T** | operand bytes of `Q`, `K` | **yes, 1.2–1.4×** | **`V` and `P` stay fp16; never upcast/quantize them** |
| fp8 PV | `V` operand bytes | **no** — slower and less accurate | `P` loses softmax resolution; mixed dtype rejected |
| causal tile-skip | wasted FLOPs on masked tiles | **yes, 1.95×** | independent of dtype |

So the earlier blanket claim "fp8 does not port to attention" is **false** — it ports well when applied to
the right operand pair (`Q`/`K`) and left off the wrong ones (`V`/`P`). The refined rule is: **porting a
dtype lever requires keeping the sensitive operand in high precision; the gain lives in the large, error-
tolerant operand (here the long `K` stream in the QK GEMM).** This is the same shape as GEMM's fp8 rule
(fp8 pays on the fat operands, keep accumulation fp32) — it just took the PV path to see where the line is.

## When needed (trigger)

- **Do not port split-K to attention.** The grid is `(SEQ/BM)·H` (6.6 waves at SEQ 4096) — tie.
- **Do port fp8 to attention QK^T**: `Q, K` in fp8 e4m3, `V` and `P` in fp16, `cos ≈ 0.9994`, worth
  1.2–1.4× on **both** causal and non-causal. This is a real win that the earlier V-upcast probe masked.
- **Do not quantize `P` or `V` to fp8.** P-in-fp8 is slower and less accurate; P-fp8 × V-fp16 is rejected.
- **The lever that carries unconditionally is the causal skip** (1.95×, `causal_attention_skips_the_masked_half.md`).

## Gotchas

- **Attention split-K combine is `O = Σ acc_s / Σ l_s` only after rescaling each partial by `exp2(m_i)`** —
  a naive sum is `nan`/wrong.
- **Keep `V` at fp16.** Casting `V` to fp8 (to "save bytes") forces an fp8→fp16 upcast before the PV GEMM
  and erases the QK saving — this is exactly the mistake that produced the wrong "+1 %" number.
- **`P fp8 × V fp16` is rejected** (`Mixed-dtype GEMM only supports f8f6f4`); both PV operands must match,
  and matching them at fp8 is slower — so the PV GEMM stays fp16.
- **The causal grid check is `(SEQ/BM)·H`, not `SEQ/BM`** — the head axis multiplies it; compute the real
  block count before arguing for split-K.

## Sources

1. TileLang `0.1.15`: fp8 `T.gemm`, the mixed-dtype PV rejection, `T.atomic_add`/partial-combine for
   split-K, the SMEM-cap error for dual fp8/fp16 staging.
2. `causal_attention_skips_the_masked_half.md` (the fp16 causal baseline and the 1.95× skip),
   `flash_attention_is_compute_bound_too.md` (fp8 +14 % non-causal — consistent, this file's 1.2–1.4× is
   that same lever on a length sweep), `split_k_fills_a_starved_grid.md` (the `< 1 wave` test),
   `decode_is_the_one_bandwidth_bound_case.md` (the upcast tax — the error this file corrects).
3. This repo's probes on an RTX 6000D, clock-locked; grid waves and cos computed alongside each timing.
