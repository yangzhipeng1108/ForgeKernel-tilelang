# Cross-stack comparison: one box, one protocol, every stack

> State: WORKS — measured on hardware; use these numbers to compare stacks, not to quote a
> single stack's best case.

This entry lives outside the three language trees (`cutedsl/`, `cuda/`, `tilelang/`) on purpose.
Each tree measures the same hardware with its own spelling, and until now a student reading
`tilelang/sm120/` had no way to know what CuTeDSL or cuBLAS reached on the same box with the
same protocol. That comparison is the one thing a per-language tree structurally cannot hold.

Read the numbers as a **floor and a ceiling for the same problem**, not as a ranking: a stack
absent from a cell may simply not expose that mechanism, and a stack ahead in one cell may be
behind in another.

## Where the readings come from

- Hardware: NVIDIA RTX 6000D, `sm_120a`, 156 SMs, 600 W cap, free-running clocks.
- Shape: 4096³ matmul unless a row says otherwise.
- Protocol: wall-clock loop, warmup 5, 40 iterations, one `synchronize` at each end.
- Gate: cosine against an fp32 reference, so a fast-but-wrong kernel cannot pass.

The full per-implementation table, the nameplate derivation, and the four corrections that
produced it are in `../tilelang/sm120/precision_roofline_all_stacks.md`; that file remains the
authority and this one is the extract. If the two ever disagree, the detailed table wins.

## Overview: precision × stack (RTX 6000D, 4096³, cos-gated)

One row per precision, one column per stack, TF/s at 4096³. `—` = that stack has no usable
tensor-core path for the dtype (not "slow": absent). `—†` = cuBLAS exposes no public entry
(FP4 / sparse). `⚠` = exceeds the derived nameplate (the FP4 nameplate of 592 is too low).
`~` = approximate (same compute path as the row above; not separately timed).

| precision | nameplate | cuBLAS | CuTeDSL | TileLang |
|---|---|---|---|---|
| FP16 dense | 148 | 133.1 | **143.1** | 131.1 |
| BF16 dense | 148 | 130.7 | ~143 (same as fp16) | 132.2 |
| FP8 e4m3 | 296 | **285.3** | 281.1 | 272.9 |
| FP8 e5m2 | 296 | — (unsupported) | — | 271.6 |
| NVFP4 | 592 | —† | 504.9 | **835.9** ⚠ |
| MXFP4 | 592 | —† | **509.6** | not runnable (cos 0.59) |
| MXFP8 | 296 | —† | **256.5** | not implemented |
| FP16 2:4 sparse | 296 | —† | — | **262.9** (2.00× dense) |
| FP8 2:4 sparse | 593 | —† | — | **522.0** (1.91× dense) |
| TF32 | ~68 (≈148/2) | **65.1** | — | — |
| FP32 (true, no TF32) | 148 (CUDA core) | **44.93** | — | — |
| FP64 | 74 (misleading) | **1.06** | — | — |

## What the table says that no single tree can

Three readings here are only visible when the stacks sit side by side:

1. **Every stack lands in the same band per precision** — fp16 131–143, fp8 273–285 against a
   296 nameplate. A stack that reads far below its band has a bug, not a tuning opportunity.
2. **The TileLang NVFP4 column is the fastest FP4 result on the box** (835.9), above the
   CuTeDSL hand kernel (504.9 at the same shape). That is not a general claim about the two
   stacks — the CuTeDSL row is wave-quantized at this shape — but it does mean "switch to the
   other stack" is the wrong first move when TileLang is already ahead.
3. **MX is CuTeDSL-only.** The MXFP4/MXFP8 cells under TileLang are status, not performance
   (see `../tilelang/sm120/mxfp4_source_build_state.md`). A student looking for MX numbers
   must use the CuTeDSL column.

## Not in this table

- Anything measured on another card. The same precision on Hopper or B200 reads differently,
  and those readings belong in `../cutedsl/sm90/` or `../cutedsl/sm100/`.
- Non-GEMM regimes (RMSNorm, RoPE, attention, LM head). Those are in
  `../tilelang/sm120/nongemm_regimes_one_protocol.md` under the same protocol.
- cuBLAS FP4 and sparse: there is no public entry, so a blank is not a slow stack.
