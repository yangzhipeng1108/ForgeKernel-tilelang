# MLA (absorbed) decode: 200× less KV traffic, 5× faster than MHA decode, and back on the tensor cores

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp16, `cos = 1.00000` on every cell.
> Correctness `cos = 1.00000` (`md = 5.9e-7`) against a float32 absorbed-attention reference.

## Purpose

`decode_is_the_one_bandwidth_bound_case.md` showed MHA decode is the one case that saturates HBM
(`880 GB/s`, `dram 90 %`) and that no scheduling axis helps it. MLA (Multi-head Latent Attention, the
DeepSeek-V2/V3 shape) is the architectural answer to exactly that problem: cache a **single shared
latent** `c_KV` of width `Dc` instead of per-head K and V, and *absorb* the up-projections into the
query and the output so the decode reads the latent once, for all heads.

The question: does absorption actually move the bottleneck, and what does it cost in return?

Short answer: **yes, and it costs matrix work.** The latent cache is **200× smaller** than the
equivalent 128-head MHA cache (2.6 MB vs 537 MB), decode bandwidth collapses from `880 GB/s` to
`13–21 GB/s`, the kernel is squarely compute-bound again (`intensity ≈ 3300` FLOP/byte), and it runs
**5× faster** than the MHA decode of the same shape (70 vs 14 TF). The trade is that the absorbed
`P @ c_KV` GEMM and the output up-projection add FLOPs that the MHA decode never had.

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

## Finding 1 — the latent cache removes the bandwidth wall, and the kernel is 5× faster

| | latent/cache bytes | `dram` | time (µs) | TFLOP/s | intensity |
|---|---|---|---|---|---|
| MHA decode (`HQ 32`) | 134 MB | 90 % | 152.6 | 14.1 | 16 |
| **MLA absorbed (`HQ 128`)** | **2.6 MB** | ~0 % (`BW 21 GB/s`) | **123.5** | **70.1** | **3302** |

The latent is **200× smaller** than the 128-head MHA cache it replaces (2.6 vs 537 MB), so the kernel
is no longer bandwidth-bound at all — it is back to tensor-core territory, exactly where every other
kernel in this map lives. **The MLA speedup is a bandwidth story, not a scheduling one**: it wins by
not moving the bytes, which is the same rule the whole map keeps landing on (fp8, split-K, 2:4 sparse,
causal tile-skip all win by reducing work/traffic; no re-schedule axis ever has).

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

## Finding 3 — the absorbed `acc (BM, Dc)` is the SMEM ceiling, not the KV tile

The natural DeepSeek `Dc = 512` does **not fit** this kernel as written: `acc (16·512 fp32 = 32 KiB)`
plus `Qs (16·512·2)`, `Cs (BN·512·2)`, and `Ws (512·128·2 = 128 KiB)` blows the 99 KiB cap from the
output-projection weight alone. The measurements above use `Dc = 128` to keep the mechanism intact; a
production `Dc = 512` needs the output projection **split out of the attention kernel** (write the
latent, up-project in a second pass) or an `Ws` tile loop. This is a real constraint, not a
benchmarking choice — record it before anyone ports the `Dc = 512` shape and hits a 216 KiB request.

## When needed (trigger)

- **For decode with a bandwidth wall, absorb the KV into a latent (MLA).** It trades the `880 GB/s`
  HBM wall for matrix work and measured **5× faster** (70 vs 14 TF) on this shape, `cos = 1.0`.
  This is the structural fix the MHA decode recipe could not find through scheduling.
- **MLA decode wants `stages = 2` and `BM = 16`** — the opposite of MHA decode (`stages = 1`). Do not
  port the stage verdict between them; the operand footprint (`Dc` latent vs `D` K + `D` V) decides it.
- **`Dc = 512` does not fit in one kernel** with the up-projection weight in SMEM. Split the output
  projection into a second pass before porting the real DeepSeek shape.

## Gotchas

- **The final up-projection `acc(fp32) @ W_uv(fp16)` is a mixed-dtype GEMM and fails** (`Mixed-dtype
  GEMM only supports f8f6f4`). Write the normalized latent into an **fp16 shared** buffer first
  (`Ls`), then GEMM `Ls @ Ws`.
- **`T.gemm` needs both operands in shared**: `A: shared.dyn, B: global` is rejected. Copy `W_uv` into a
  shared tile (which is what makes the `Dc = 512` SMEM blow up — see Finding 3).
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
