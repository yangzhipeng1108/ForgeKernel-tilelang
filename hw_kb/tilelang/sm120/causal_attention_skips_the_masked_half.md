# Causal attention: skipping the masked half is a 1.95× win on hardware, and closes the SDPA gap

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp16, `cos = 1.00000` at every length.
> Correctness `cos = 1.00000` against `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`.

## Purpose

`flash_attention_is_compute_bound_too.md` closed the non-causal case: at large `SEQ` the hand-written
FA sits at ~0.82× SDPA and the gap is structural. Causal is the one variation that changes the *work*,
not just the schedule. A lower-triangular mask means the query tile at block `bq` only needs the key
blocks `k ≤ bq`; the upper tiles are entirely masked and need never be loaded or multiplied.

The question: does skipping them recover the ~2× the mask implies, and does it close the SDPA gap?

Short answer: **yes, both.** Tile-skipping is **1.95×** over the same kernel without skipping
(460 vs 895 µs at `SEQ 4096`), `cos = 1.00000` throughout, and the resulting kernel is at **parity with
SDPA** at causal `SEQ 2048 / 4096 / 8192` (1.05× / 0.95× / 1.03×) — the 0.82× non-causal gap is gone,
because SDPA also skips the same tiles and the debt cancels.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- `H = 16` heads, `SEQ ∈ {2048, 4096, 8192}`, `D = 128`, fp16, **causal = on**
- Baseline: `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`, same fp16 inputs
- TileLang: FA-2 forward (from `flash_attention_is_compute_bound_too.md`) + the causal k-range bound

## The change: bound the k-loop, mask what is left

One line bounds the loop; the mask handles the diagonal block that is only *partly* masked:

```python
nkb = T.ceildiv((bq + 1) * BM, BN)            # block bq needs key blocks 0 .. ceildiv((bq+1)*BM, BN) - 1
for k in T.Pipelined(nkb, num_stages=1):
    ...
    for i, j in T.Parallel(BM, BN):
        S[i, j] = T.if_then_else((k * BN + j) <= (bq * BM + i),
                                 S[i, j] * scale, -T.infinity("float32"))
```

Everything else is the FA-2 body from the parent recipe. The `-inf` (after `scale`) drives those
scores to zero in the `exp2`, so `reduce_max` / `reduce_sum` / the PV GEMM need no special case — the
masked entries contribute nothing and the online bookkeeping is unchanged.

## Finding 1 — tile-skipping is 1.95× and matches the exact skipped-block count

Same kernel, mask on, `skip` toggled by using `nkb` vs the full `SEQ/BN`:

| `BM` | `BN` | k-blocks skipped? | time (µs) | effective TFLOP/s (causal, ÷2) |
|---|---|---|---|---|
| 64 | 128 | **yes** | **460.1** | **149.4** |
| 64 | 128 | no | 895.2 | 76.8 |
| 64 | 64 | yes | 516.7 | 133.0 |
| 128 | 64 | yes | 649.1 | 105.9 |

**1.95×** (460 / 895). It is not a free lunch beyond the mask, and it should not be: the exact number
of k-block visits is

| SEQ | `BM` | `BN` | skipped visits / full visits |
|---|---|---|---|
| 4096 | 64 | 128 | 0.516 |
| 4096 | 128 | 64 | 0.516 |
| 4096 | 64 | 64 | 0.508 |
| 8192 | 64 | 128 | 0.508 |

i.e. **half the blocks**, exactly as the triangle predicts. The measured 1.95× is the block count
(0.516 → 1.94×) and nothing more — no hidden extra gain, and no slack left on the table.

## Finding 2 — `BM < BN` wins, and it is not the visit count

The skip *ratio* is the same for `BM 64 / BN 128` and `BM 128 / BN 64` (both 0.516), yet the times
differ 460 vs 649 µs. So the 1.4× between those two tilings is **per-block efficiency, not skipping**:
the wide `BN = 128` key tile makes a wider, more efficient `QK^T` and PV GEMM, while the narrow-`BN`
case pays for many more, smaller block visits per remaining k-block. Keep `BM = 64, BN = 128` for
causal — the same "wide `BN`" preference as the non-causal prefill recipe, now confirmed with skipping.

## Finding 3 — causal is at parity with SDPA (the 0.82× non-causal gap is gone)

| SEQ | TileLang (µs) | eff TFLOP/s | SDPA (µs) | SDPA TFLOP/s | TileLang / SDPA | cos |
|---|---|---|---|---|---|---|
| 2048 | **152.0** | 113.0 | 160.0 | 107.4 | **1.05×** | 1.00000 |
| 4096 | 460.8 | 149.1 | 439.7 | 156.3 | **0.95×** | 1.00000 |
| 8192 | **1919.8** | 143.2 | 1971.5 | 139.4 | **1.03×** | 1.00000 |

Compare the non-causal table in `flash_attention_is_compute_bound_too.md` (0.82× f16 at `SEQ 2048`).
With causal skipping the hand-written kernel is **at parity across the whole length sweep** (0.95–1.05×).
The mechanism is clean: SDPA's large-`SEQ` edge came from work/schedule efficiency the generic tail
lacks; once both kernels skip the same lower-triangular half, the masked work no longer dominates and the
residual difference is noise. **The 0.82× "structural" gap was a non-causal-specific deficit, not a
universal one** — causal is a case where the hand-written FA is the right tool.

## When needed (trigger)

- **For causal attention, always bound the k-loop to `ceildiv((bq+1)*BM, BN)`.** It is a one-line change
  and worth 1.95× — the single largest attention win measured on this arch, bigger than fp8's +14 %.
- **Use `BM = 64, BN = 128` for causal** (wide `BN`: 460 vs 649 µs over `BM 128 / BN 64`, a per-block
  efficiency effect, not a skip-count one).
- **Causal prefill is a case to prefer the hand-written FA**: it is at parity with SDPA (0.95–1.05×) and
  beats it at `SEQ 2048` and `SEQ 8192`. Do **not** carry over the non-causal conclusion ("0.82×,
  structural, use SDPA") — that was for the dense (fully-masked-out) case only.
- Decode is bandwidth-bound and separate: see `decode_is_the_one_bandwidth_bound_case.md`.

## Gotchas

- **Mask with `-inf` *after* `scale`, not `0`.** Setting a masked score to `0` makes it the max and
  poisons the row; `-inf` sends `exp2(-inf - m) = 0`. Use `T.if_then_else(cond, S*scale,
  -T.infinity("float32"))` in one pass so the scaling and the mask compose correctly.
- **The diagonal block is partly masked and must still be computed** — only *strictly* upper blocks
  (`k > bq` once tiles align) are skipped. `nkb = ceildiv((bq+1)*BM, BN)` includes the diagonal block;
  a plain `k <= bq` bound would drop the on-diagonal contribution.
- **`nkb` depends on `bq`, so the loop trip count varies per block.** That is fine with `T.Pipelined`
  (it is a runtime bound), but do not unroll it or assume a fixed trip count in any host-side sizing.
- **The skip ratio is `~0.5` only when `SEQ ≫ BM`.** At `SEQ 2048, BM 64` it is 0.531 (fewer blocks to
  amortize the diagonal), still ~1.9× — the win shrinks slowly, never disappears within prefill range.

## Sources

1. TileLang `0.1.15`: `T.Pipelined` accepting a runtime trip count, `T.if_then_else`, `T.infinity`;
   the FA-2 body from `flash_attention_is_compute_bound_too.md`.
2. `flash_attention_is_compute_bound_too.md` (the non-causal 0.82× baseline this file contrasts against),
   `sparse_gemm_two_mechanisms.md` (block-tile skipping — the same "launch only the needed tiles" idea,
   here driven by the causal mask rather than a sparsity map).
3. This repo's probes on an RTX 6000D, clock-locked; skipped-block counts computed exactly for the
   tiling and compared to measured time, so the 1.95× is attributed to the block count and nothing else.
