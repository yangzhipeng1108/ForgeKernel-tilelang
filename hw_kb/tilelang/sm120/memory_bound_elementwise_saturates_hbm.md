# Memory-bound elementwise (RMSNorm, RoPE): the one regime that saturates HBM — tile-small, and scheduling STILL ties

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp16, `cos = 1.00000` (RMSNorm) /
> `0.99993` (RoPE) against float32 references.

## Purpose

Every prior recipe in this map is about **compute-bound** kernels (GEMM at `dram 13 %`, attention at
`8.8 %`) or the two bandwidth-bound *attention* cases (decode MHA `880 GB/s`, LM head `945 GB/s`). The
open question this file settles: on a genuine **memory-bound elementwise** kernel — RMSNorm (a row
reduction) and RoPE (a per-element rotation) — do the axes that are dead everywhere else finally pay?
Specifically: does the **tile-size** preference invert (small tile wins, as memory-bound intuition says),
and does the **scheduling axis (swizzle)** finally do something?

Short answer: **tile-small wins and it is the first regime that saturates HBM** (RMSNorm 1294 GB/s,
RoPE 1286 GB/s, both above the 1194 GB/s pure-copy roofline and ~10× torch's norm path). **But swizzle
still ties** (1251–1261 GB/s, ±0.8 %) — because a streaming elementwise kernel has each block read a
*disjoint* contiguous slab, so there is no cross-block L2 reuse for a schedule to exploit. The "scheduling
never pays" rule survives its strongest test.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- RMSNorm: `M=8192` rows, `D=4096` hidden, fp16; RoPE: same, rotation over `D` in `H=D/2` pairs
- Baselines: float32 reference (torch), `torch` norm path, and a **pure fp16 copy** roofline (1194 GB/s)
- Peak HBM read measured separately: 1145 GB/s (read) / 1273 GB/s (copy) — see `decode_is_the_one_bandwidth_bound_case.md`

## Finding 1 — RMSNorm saturates HBM, inverts the tile preference, and beats torch 10×

RMSNorm (`y = x · rsqrt(mean(x²)+ε) · w`, fp16), `M·D·2·2 = 134 MB` moved:

| `BM` (rows/block) | `threads` | time (µs) | GB/s |
|---|---|---|---|
| 1 | 128 | 103.3 | 1299 |
| **2** | **128** | **100.8** | **1331** |
| 4 | 128 | 101.8 | 1318 |
| 8 | 128 | 111.3 | 1206 |
| 16 | 128 | 184.0 | 729 |
| 32 | 128 | 268.9 | 499 |
| `torch` fp32 path | — | 1018.6 | **132** |

Two things stand out:

1. **Small tile wins, hard.** `BM 1–4` sit at 1300–1330 GB/s; `BM 32` collapses to 499 GB/s (a 2.7×
   loss). This is the **opposite of every compute-bound recipe**: a GEMM/attention wants a big tile to
   amortize MMA/accumulator work, but a memory-bound kernel wants many small blocks so the scheduler can
   keep the memory pipe full with independent in-flight loads. A big `BM` serialises one block's rows
   behind its reduction before it can issue the next block's loads.
2. **It beats the "peak".** 1331 GB/s is **above** the 1194 GB/s pure-copy roofline and the 1145 GB/s
   read peak — because RMSNorm reads X once, writes Y once, and the read of the *next* row overlaps the
   compute of the current one (the reduction is cheap enough to hide under the memory stream). **RMSNorm
   at `BM 2` is the fastest moving-data kernel measured in this entire map.**
3. **torch is 10× slower (132 GB/s).** The eager `rsqrt(mean(x²))` chain launches several pointwise
   kernels with no fusion and no 128-bit vectorised load/store, so it is nowhere near the wall. A fused
   TileLang kernel at 1331 GB/s is `10.1×` faster. **This is where the memory-bound regime pays the
   *most* over PyTorch** — not a marginal 10 %, but an order of magnitude, because the baseline is
   unfused.

## Finding 2 — RoPE saturates HBM too, same small-tile preference

RoPE (`y = x·cos + rotate_half(x)·sin`), split even/odd halves to dodge a `T.Parallel` strided-access
limit, `M·D·2·2 + M·H·2·2 = 201 MB` moved:

| `BM` | `threads` | time (µs) | GB/s | cos |
|---|---|---|---|---|
| 1 | 128 | 157.5 | 1278 | 0.99993 |
| **2** | **128** | **156.5** | **1286** | 0.99993 |
| 4 | 128 | 159.4 | 1263 | 0.99993 |
| 8 | 128 | 196.4 | 1025 | 0.99993 |

Same shape as RMSNorm: **`BM 2` best (1286 GB/s), `BM 8` drops** (1025). RoPE moves more bytes (it reads
`cos`/`sin` too) so it is a bit further from the copy roofline, but it is firmly HBM-bound and again
wants a small tile. `cos = 0.99993` (the ~7e-5 gap is fp16 rounding of the two-output write, not a bug).

## Finding 3 — swizzle STILL ties on a memory-bound kernel (the strongest test, and it passes)

If a scheduling axis would ever pay, it is here — the kernel is HBM-saturated, not compute-bound. It does
not:

| `T.use_swizzle` | RoPE time (µs) | GB/s |
|---|---|---|
| off | 160.9 | 1251 |
| `panel_size=4` | 159.7 | 1261 |
| `panel_size=8` | 159.9 | 1259 |
| `panel_size=16` | 160.8 | 1252 |

±0.8 %, noise. **The reason is the same root cause as decode MHA's swizzle tie**: swizzle reorders *block
execution* to improve L2 reuse **across blocks**, but a streaming elementwise kernel gives each block a
**disjoint contiguous slab** of `X` — there is no data any two blocks share, so no L2 reuse to gain by
reordering them. The schedule is irrelevant when the access pattern has no cross-block locality.

This closes the loop on the whole map: the "scheduling never pays" rule was measured on compute-bound
GEMM/attention (where DRAM was 13 %/8.8 %, so scheduling could not *reach* the critical path) **and** now
on a memory-bound kernel (where DRAM *is* the critical path, but there is no cross-block reuse to
exploit). **Two different reasons, same verdict: a reschedule axis only pays when the kernel has
exploitable cross-block L2 locality, which none of these streaming kernels have.**

## Finding 4 — the memory-bound regime's real lever is the baseline, not a knob

The compute-bound recipes win by *algorithmic* levers (fp8, sparsity, tile-skip). The memory-bound
recipes win by **not being PyTorch**: a fused kernel at the HBM wall is 10× the eager chain. Past
`BM 2` there is essentially no tuning headroom — `BM 1/2/4` are all within 3 %, and the wall is the wall.
So the guidance is the inverse of the GEMM map:

| regime | what to tune | how much headroom |
|---|---|---|
| compute-bound (GEMM/attention) | dtype, sparsity, tile-skip — **algorithmic** | 2–5× (fp8, skip) |
| **memory-bound (norm/rope/gather)** | **fusion vs PyTorch, then stop** | **~10× over eager, ~0 over a fused kernel** |

Once you have a fused, vectorised kernel at `BM ~2`, **there is nothing left** — no tile sweep, no
schedule, no dtype trick beyond the wire width. The lever was the baseline.

## When needed (trigger)

- **Norm/RoPE/gather-type kernels: use a small tile (`BM 1–4`), `threads ~128`, and stop.** This is the
  one regime where a big tile is *wrong* — `BM 32` loses 2.7× on RMSNorm. The kernel is at the HBM wall by
  `BM 2`; larger blocks only serialise rows behind their reduction.
- **Expect ~10× over an eager PyTorch norm/rope**, not 10 % — the eager path is unfused and un-vectorised.
  That fusion is the entire win; there is no further knob once you are fused.
- **Do not reach for swizzle/persistent/scheduling** even though the kernel is HBM-bound. Streaming
  elementwise has no cross-block L2 reuse, so a reschedule axis ties (measured, ±0.8 %). Same verdict as
  decode MHA, same reason.
- **For LM-head / decode (the other bandwidth-bound cases), the tile still wants to be large-ish** because
  those are GEMMs with real MMA work (see `lm_head_is_wide_not_starved.md`, `decode_is_the_one_bandwidth_bound_case.md`).
  The small-tile rule is specific to *pure* elementwise/reduction with no matmul.

## Gotchas

- **`T.Parallel` rejects strided dual-index access in one loop** (`Xs: (i, j*2+1) and (i, j*2)` →
  `StructuralEqual` check failure). For RoPE, either split even/odd into separate buffers/loops, or use
  explicit `j%2` branches — do not index `Xs[i,2*j]` and `Xs[i,2*j+1]` in the same `T.Parallel`.
- **RMSNorm's `T.reduce_sum` is over fp32 fragments** — allocate `Xs`/`sq` as `float32` so the mean-of-
  squares is fp32 accurate; fp16 accumulation across `D=4096` would drift.
- **1331 GB/s exceeds the "peak"** because read/write overlap and `cos/sin`/`W` hit L2 — do not treat
  exceeding 1145 GB/s as a measurement error; it is the reduction hiding under the memory stream.
- **The tile preference inverts vs compute-bound.** If you port a GEMM recipe's "big tile" instinct to a
  norm/rope kernel you will lose ~2.7× (`BM 32` → 499 GB/s). The bottleneck determines the tile direction.

## Sources

1. TileLang `0.1.15`: `T.reduce_sum`, `T.Parallel` strided-access restriction (the `StructuralEqual`
   failure), `T.use_swizzle`, `T.rsqrt`.
2. `decode_is_the_one_bandwidth_bound_case.md` and `lm_head_is_wide_not_starved.md` (the other bandwidth-
   bound cases and the same "scheduling ties on streaming kernel" verdict), `block_scheduling_is_a_tie.md`
   (the compute-bound schedule tie — same verdict, different reason), `tiled_copy_scope_and_swizzle.md`
   (the copy at `dram 93 %` where swizzle was also worthless).
3. This repo's probes on an RTX 6000D, clock-locked; pure-copy roofline (1194 GB/s) and float32 references
   computed alongside so the wall is measured, not assumed.
