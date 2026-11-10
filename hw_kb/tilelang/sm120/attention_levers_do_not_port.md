# What transfers to attention and what does not: split-K and fp8 both evaporate under causal skipping

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp16/fp8, `cos = 1.00000` (f16) and
> `0.99906` (fp8) against `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`.

## Purpose

The GEMM map's two winners — **split-K** (grid shaping) and **fp8** (halve the operand bytes) — are the
natural things to port to attention. `causal_attention_skips_the_masked_half.md` already showed the
hand-written FA reaches SDPA parity by skipping the masked tiles; the question here is whether stacking
**split-K** or **fp8** on top of that adds anything, and why.

Short answer: **neither does.** split-K buys ~2.6 % (a tie) because the causal grid is already
`6.6 waves`; fp8 buys ~1 % (`cos = 0.99906`) because the skipped tiles have *already* removed the KV
traffic fp8 would have halved — the two savings are the same savings, and the fp16 upcast of `V` eats
the remainder. Both results are the **grid-fill rule** and the **"fp8 pays only when it feeds the MMA"**
rule from the GEMM map, now shown to gate the ports too.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- Causal FA-2 forward, `H = 16`, `SEQ = 4096`, `D = 128`, `BM = 64`, `BN = 128`
- Baseline: fp16 causal FK from `causal_attention_skips_the_masked_half.md` (`448.4 µs`, idx `153.2` eff TF)
- split-K here is **split over the k-block range** of each query tile (partial `(acc, l)` combined with
  `O = Σ acc_s / Σ l_s` after rescaling each partial by `exp2(m_i)`); this is the attention analogue of
  the GEMM K-split, since a query tile's k-loop is its own reduction axis.

## Finding 1 — split-K is a tie because the causal grid is already full

| split | time (µs) | eff TF | cos |
|---|---|---|---|
| 1 | 448.4 | 153.2 | 1.00000 |
| 2 | **436.8** | **157.3** | 1.00000 |
| 4 | 455.5 | 150.9 | 1.00000 |

±2.6 % — a tie, not a win. The grid for causal FA at `SEQ 4096, BM 64, H 16` is `NQ·H = 64·16 = 1024`
blocks = **6.6 waves** on 156 SMs. That is the same test `split_k_fills_a_starved_grid.md` applies to a
GEMM: split-K only pays below **one** wave. Attention prefill tiles are `(SEQ/BM)·H`, which is hundreds
to thousands of blocks for any real `SEQ·H` — **prefill is never tile-starved**, so split-K has nothing
to fill. (Split-K *would* matter for MHA decode, where the grid is `H = 32`, but decode is
bandwidth-bound, so it still ties — see `decode_is_the_one_bandwidth_bound_case.md`, +3 %.)

## Finding 2 — fp8 evaporates under causal skipping (the skips already removed the bytes)

| variant | time (µs) | eff TF | cos |
|---|---|---|---|
| fp16 causal | 448.4 | 153.2 | 1.00000 |
| **fp8 causal (Q8·K8, V upcast)** | **444.1** | **154.7** | 0.99906 |
| fp8 causal `BN 64` | 524.7 | 131.0 | 0.99906 |

**+1 %**, versus fp8's **+14 %** on *non-causal* prefill (`flash_attention_is_compute_bound_too.md`).
The reason is compositional, not a measurement artifact:

- non-causal fp8 wins because reading `K, V` at 1 byte instead of 2 **halves the operand bytes** on a
  kernel whose bottleneck is feeding the tensor cores from SMEM/HBM;
- **causal tile-skip has already skipped half the k-blocks**, so the bytes that fp8 would halve are half
  gone before fp8 is applied. The two levers attack the *same* resource (KV bytes) and do not stack —
  crossing them yields roughly `1.95× (skip) × 1.01× (fp8) ≈ 1.97×`, not `1.95 × 1.14`.
- the fp8 `V` is also **upcast to fp16 before the PV GEMM** (the mixed-dtype limit), which costs a
  conversion pass that eats the small remainder — the exact "upcast tax" measured for fp8 KV cache in
  `decode_is_the_one_bandwidth_bound_case.md` (there it lost 26 %).

So the fp8 causal result is the **honest** number: on a kernel that already skips the masked half, fp8
QK^T is worth ~1 %, and the free lunch was the skip, not the dtype.

## Finding 3 — the portable principle

Both ports fail for the *same* meta-reason that has governed the whole map: **a lever pays only if it
attacks the resource that is actually binding, and only if nothing else already attacked it.**

| lever | binds on | pays in GEMM | pays in causal attention | why the difference |
|---|---|---|---|---|
| split-K | idle SMs (grid < 1 wave) | yes, when starved | **no** — grid is 6.6 waves | prefill grid is always full |
| fp8 | operand bytes (compute-bound) | yes (+2.1×) | **~no (+1 %)** | causal skip already removed those bytes |
| causal tile-skip | wasted FLOPs on masked tiles | n/a | **yes (+1.95×)** | attacks work the others do not |

The one lever that *is* new to attention — **causal tile-skipping** — is the one that changes the *work*
(the masked half is never computed), not the schedule or the dtype. This is the same lesson `MLA`
proves at the algorithm level (`mla_absorbed_decode_beats_mha.md`): the wins reduce work, never rearrange it.

## When needed (trigger)

- **Do not port split-K to attention prefill.** The grid is `(SEQ/BM)·H ≫ 156`; split-K is a tie, and in
  decode it is also a tie (bandwidth-bound). Only consider it for a deliberately tiny `SEQ·H` — and then
  confirm the grid is `< 156` tiles first.
- **Do not expect fp8's +14 % on causal attention.** Once tile-skipping is on, fp8 QK^T is ~1 % (and the
  `V` upcast can erase it). fp8 is worth applying to *non-causal* prefill and to GEMM, where the bytes it
  halves are still on the critical path.
- **The lever that carries to causal attention is the skip.** Bound the k-loop to the diagonal
  (`causal_attention_skips_the_masked_half.md`); everything else is a tie.

## Gotchas

- **Attention split-K combine is `O = Σ acc_s / Σ l_s` only after rescaling each partial by `exp2(m_i)`** —
  the per-split `m_i` differs, so a naive sum is wrong (`nan` or a wrong matrix). Same trap as decode.
- **Naively staging fp16 `Ks`/`Vs` alongside the fp8 buffers blows SMEM** (`132 096 B` → cap error);
  fp8 must feed the GEMM directly, and only `V` (for the PV GEMM) needs an fp16 stage. Do not keep both.
- **The causal grid check is `(SEQ/BM)·H`, not `SEQ/BM`** — the head axis multiplies it, so even short
  `SEQ` with many heads is full. Compute the real block count before arguing for split-K.

## Sources

1. TileLang `0.1.15`: `T.atomic_add` / partial-combine for split-K, fp8 `T.gemm`, the mixed-dtype PV
   limit that forces the `V` upcast, the SMEM-cap error for dual fp8/fp16 staging.
2. `causal_attention_skips_the_masked_half.md` (the fp16 causal baseline, and the skip that is the real
   win), `flash_attention_is_compute_bound_too.md` (fp8's +14 % on *non-causal*, the number that does not
   transfer), `split_k_fills_a_starved_grid.md` (the `< 1 wave` test re-applied to the attention grid),
   `decode_is_the_one_bandwidth_bound_case.md` (the upcast tax, and decode's split-K tie).
3. This repo's probes on an RTX 6000D, clock-locked; causal grid waves computed and compared to measured
   time so the tie is attributed to grid fill.
