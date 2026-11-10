# MLA (absorbed) decode: 256× less KV traffic, 3.8× faster than MHA (1.16× over GQA/MQA), back on the tensor cores

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp16, `cos = 1.00000` on every cell.
> Correctness `cos = 1.00000` (`md = 5.9e-7`) against a float32 absorbed-attention reference.

## Purpose

`decode_is_the_one_bandwidth_bound_case.md` showed MHA decode is the one case that saturates HBM
(`880 GB/s`, `dram 90 %`) and that no scheduling axis helps it. MLA (Multi-head Latent Attention, the
DeepSeek-V2/V3 shape) is the architectural answer to exactly that problem: cache a **single shared
latent** `c_KV` of width `Dc` instead of per-head K and V, and *absorb* the up-projections into the
query and the output so the decode reads the latent once, for all heads.

The question: does absorption actually move the bottleneck, and what does it cost in return?

Short answer: **yes, and it costs matrix work — but the win is over un-shared MHA, not over GQA/MQA.**
The latent cache is **256× smaller** than the equivalent 128-head MHA cache (2.1 MB vs 537 MB), decode
bandwidth collapses from `880 GB/s` to `13–21 GB/s`, the kernel is squarely compute-bound again
(`intensity ≈ 3300` FLOP/byte), and at `HQ=128` it runs **3.8× faster than same-shape MHA** (465 → 123 µs)
and **1.16× faster than GQA/MQA** (143 → 123 µs). The trade is that the absorbed `P @ c_KV` GEMM and the
output up-projection add FLOPs the MHA decode never had. **At low `HQ` (32) the ordering reverses**: MQA
(94 µs) beats MLA (121 µs), because GQA/MQA already removed the KV traffic MLA would cut. See Finding 1.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- MLA absorbed decode: `HQ = 128` heads, `QLEN = 16`, `KVLEN = 8192`, latent `Dc = 128`, `DV = 128`, fp16
- Reference: float32 absorbed attention `softmax(Qa·c^T/√Dc) @ c @ W_uv` in torch
- Baseline contrast: MHA decode `HQ = 32, q = 16, KV = 8192, D = 128` from the decode recipe

## The kernel (absorbed: cache the latent, project at the ends)

```python
# cache holds only c_KV (KVLEN, Dc). Qa is already Q·W_UK^T (HQ, QLEN, Dc).
S = Qa @ c^T                      # scores in latent space, contract over Dc
P = softmax(S)
acc = P @ c                       # latent output (BM, Dc) -- accumulated over K blocks
out = acc @ W_uv                  # single up-projection at the end, (Dc -> DV)
```

The whole point is that `c` is read **once per k-block for every head** (`T.Kernel(HQ)` shares the
latent load), so the KV traffic is `KVLEN·Dc` bytes total, not `HQ·KVLEN·(2·D)`.

## Finding 1 — the latent cache removes the bandwidth wall; it beats MHA 3.8× but only edges GQA/MQA

The honest same-shape comparison (`HQ=128, QLEN=16, KVLEN=8192, D=128`, all clock-locked):

| method | cache bytes | time (µs) | vs MLA |
|---|---|---|---|
| MHA decode (`HKV 128`, no sharing) | 537 MB | 465.5 | 3.8× slower |
| GQA (`HKV 16`, G = 8) | 67 MB | 143.0 | 1.16× slower |
| MQA (`HKV 1`, G = 128) | 4 MB | 142.5 | 1.16× slower |
| **MLA absorbed** (`Dc 128`, single latent) | **2.1 MB** | **123.3** | — |

So MLA absorbed **is** the fastest decode at `HQ=128`: **3.8× over MHA** (537 → 2.1 MB cache) and a
smaller **1.16× over GQA/MQA**. The big win is specifically **against the un-shared MHA cache**, which at
`HQ=128` is 537 MB and hopelessly bandwidth-starved. But **GQA/MQA already recover most of that** by
sharing KV heads, so once a model already shares KV, MLA's extra absorption buys only ~16 %.

**The advantage is `HQ`-dependent, and it reverses at low `HQ`** — the crossover this file used to hide:

| `HQ` | GQA (best `HKV`) | GQA cache | MLA absorbed | winner |
|---|---|---|---|---|
| 32 | 94.3 µs (`HKV 1`, MQA) | 4 MB | 121.5 µs | **GQA/MQA** (MLA 1.29× slower) |
| 128 | 142.5 µs (`HKV 1`) | 67–143 MB* | 123.3 µs | **MLA** (1.16× faster) |

*at `HQ=128` GQA/MQA sit at 142–143 µs (a 4 MB MQA cache is not slower than a 67 MB GQA one — they are
compute-bound at that point), while at `HQ=32` MQA is 94 µs because there are only 32 head-blocks reading
the tiny cache.

The rule: **MLA absorption pays when the per-head traffic multiplies a lot (`HQ` high) and no KV sharing is
assumed; once GQA/MQA already shares KV, MLA's marginal gain shrinks to ~16 % at high `HQ` and goes
negative at low `HQ`** (where GQA/MQA is compute-bound at a lower per-block cost and MLA's fixed per-head
latent+up-projection work is pure overhead). This is the same "reduce the binding resource, and only if
nothing else already did" rule: GQA/MQA already removed the KV traffic, so MLA's byte cut has little left
to remove. **Choose MLA over GQA/MQA only for high-`HQ` MHA-style decode; if the model can share KV heads
(GQA/MQA), that is cheaper than absorbing.**

An earlier draft of this file reported "5× faster (70 vs 14 TF)" by comparing `HQ=128` MLA against `HQ=32`
MHA — a cross-shape TF ratio, not a wall-clock speedup. The fair same-shape number is **3.8×** (465 vs
123 µs at `HQ=128`), and against GQA/MQA it is **1.16×**, not 5×. The bandwidth story is real; the
multiplier was overstated by the mismatched baseline.

The latent is **256× smaller** than the 128-head MHA cache it replaces (2.1 vs 537 MB), so against MHA the
kernel is no longer bandwidth-bound at all — it is back to tensor-core territory. **The MLA speedup is a
bandwidth story, not a scheduling one**: it wins by not moving the bytes, which is the same rule the whole
map keeps landing on (fp8, split-K, 2:4 sparse, causal tile-skip all win by reducing work/traffic; no
re-schedule axis ever has).

## Finding 2 — the absorbed GEMMs flip the stage/tile preference back

The MHA decode recipe found `stages = 1` wins and `stages = 2` overflows SMEM. MLA is the opposite:

| `BM` | `BN` | `stages` | time (µs) | TFLOP/s |
|---|---|---|---|---|
| 16 | 64 | 1 | 204.3 | 42.4 |
| **16** | **64** | **2** | **123.5** | **70.1** |
| 16 | 128 | 1 | 139.0 | 62.3 |
| 32 | 64 | 2 | 146.1 | 59.3 |
| 32 | 128 | 1 | 174.5 | 49.6 |

`stages = 2` is a **1.65× win** (204 → 123.5 µs) and the best cell. The SMEM profile is different from
MHA decode: the latent `c` tile is `BN·Dc` (not `BN·2·D`) and there is no separate V, so a second
pipeline stage fits. **This is the point of measuring rather than porting a recipe**: the same
"attention decode" label hides two opposite stage verdicts, decided by whether the operand tile is a
`Dc`-wide latent or a `D`-wide K plus a `D`-wide V.

`BM = 16` (the minimum for `T.gemm`) beats `BM = 32` everywhere — the decode query tile is tiny by
definition, and a bigger query tile only costs accumulator registers (`16·Dc fp32`) without amortizing
anything, since the KV stream dominates.

## Finding 3 — the absorbed `acc (BM, Dc)` is the SMEM ceiling, not the KV tile — and `Dc=512` runs as two passes

The natural DeepSeek `Dc = 512` does **not fit** this kernel as written: `acc (16·512 fp32 = 32 KiB)`
plus `Qs (16·512·2)`, `Cs (BN·512·2)`, and `Ws (512·128·2 = 128 KiB)` blows the 99 KiB cap from the
output-projection weight alone. The measurements above use `Dc = 128` to keep the mechanism intact. The
fix is to **split the output projection out**: pass 1 computes the attention in latent space and writes
the normalized latent `L (HQ, QLEN, Dc)` to global; pass 2 up-projects `L @ W_uv` in a tiny GEMM.
Measured on the real `Dc = 512` shape (`HQ=128, QLEN=16, KVLEN=8192`):

| pass | config | time (µs) | TFLOP/s | notes |
|---|---|---|---|---|
| 1: latent attention | `BN 32, stages 1` | 552.2 | 62.2 | `cos = 1.00000` on `L` |
| **1: latent attention** | **`BN 32, stages 2`** | **356.7** | **96.3** | **best — `stages=2` fits at `Dc`-only footprint** |
| 1: latent attention | `BN 64, stages 1` | 459.8 | 74.7 | `BN 64` + `stages 2` overflows |
| 2: up-projection | `BM 16, BN 128, BK 64` | 12.7 | — | negligible |
| **total (two passes)** | | **369.4** | — | `cos = 1.00000`, `md = 1.0e-6` |

So `Dc = 512` **does** come real as two passes, and the split is almost free (12.7 µs for the up-projection).
Pass 1 dominates and is the same `stages=2` winner as the `Dc=128` single kernel — because once `Ws` is out
of the kernel, the latent attention fits a second stage. `eff 96.3 TF` at `Dc=512` is higher than the
`Dc=128` single-kernel `70.1 TF` in absolute terms (it is doing 4× the contraction work); the point is that
the production width is reachable, not that it is cheaper than the small toy.

A full tile sweep of pass 1 (`BN ∈ {16,32,64}`, `stages ∈ {1,2,3}`, `threads ∈ {128,256}`) confirms
**`BN 32, stages 2` is the optimum** — `stages 2` beats `stages 1` at every runnable `BN` (356.7 vs 552.2 µs
at `BN 32`; 459.0 at `BN 64`), `BN 16` fails the warp-partition check (`M=16, N=16` not coverable), and
`stages ≥ 3` does not fit. So the `Dc=512` operating point is settled, not left open.

## When needed (trigger)

- **MLA absorption beats un-shared MHA decode 3.8× at `HQ=128`** (465 → 123 µs, 537 → 2.1 MB cache), and
  edges GQA/MQA by 1.16×. Use it when the baseline is MHA (`HKV = HQ`) at high `HQ`.
- **Do not assume MLA beats GQA/MQA.** At `HQ=32` MQA (94 µs) is *faster* than MLA (121 µs); at `HQ=128`
  MLA wins by only 16 %. If the model can share KV heads (GQA/MQA), that is cheaper than absorbing —
  choose MLA only when sharing is not an option (true MHA) and `HQ` is high (the per-head traffic is large).
- **MLA decode wants `stages = 2` and `BM = 16`** — the opposite of MHA decode (`stages = 1`). Do not
  port the stage verdict between them; the operand footprint (`Dc` latent vs `D` K + `D` V) decides it.
- **For the real `Dc = 512`, split the output projection into a second pass** (write the latent, then
  `L @ W_uv`). Measured total 369 µs at `cos = 1.00000`; pass 1 runs `BN 32, stages 2` at 96.3 TF, pass 2 is
  12.7 µs. Do not try to keep `W_uv` in the attention kernel — it alone is 128 KiB.

## Gotchas

- **The final up-projection `acc(fp32) @ W_uv(fp16)` is a mixed-dtype GEMM and fails** (`Mixed-dtype
  GEMM only supports f8f6f4`). Write the normalized latent into an **fp16 shared** buffer first
  (`Ls`), then GEMM `Ls @ Ws`.
- **`T.gemm` needs both operands in shared**: `A: shared.dyn, B: global` is rejected. Copy `W_uv` into a
  shared tile (which is what makes the `Dc = 512` SMEM blow up — hence the two-pass split in Finding 3).
- **At `Dc = 512` the attention latent tile needs `BN = 32` to fit** (`BN 64` + `stages 2` overflows); `M=16,
  N=16` also fails the warp-partition check (`each warp must own a multiple of 16 rows and 8 cols`) — use
  `BN ≥ 32`.
- **The absorbed reference must be built in float32.** Construct `softmax(Qa·c^T/√Dc) @ c @ W_uv` in
  fp32; an fp16 reference chain hides the `md` and you cannot see the `5.9e-7` the kernel actually hits.
- **`cos` alone is not enough here** — with the `0.05`-scaled random operands the values are small;
  check `md` too (`5.9e-7`), since a `cos` near 1.0 can still hide a constant offset at this scale.

## Sources

1. TileLang `0.1.15`: `T.gemm` mixed-dtype restriction (the same one as `flash_attention_*`), shared-
   operand requirement, `T.Pipelined(num_stages=)`, the 99 KiB SMEM-cap error for `Dc = 512`.
2. `decode_is_the_one_bandwidth_bound_case.md` (the MHA decode this replaces, its `880 GB/s` wall and
   the opposite `stages = 1` verdict), `gemm_dtype_throughput.md` (fp8 pays only when it feeds the MMA —
   the same "reduce the work" principle MLA applies at the algorithm level).
3. This repo's probes on an RTX 6000D, clock-locked; the absorbed reference built and compared in fp32.
