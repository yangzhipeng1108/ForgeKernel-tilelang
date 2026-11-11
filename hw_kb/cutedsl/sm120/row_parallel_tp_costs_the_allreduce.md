# Row-parallel TP on sm120 loses to one card — the all-reduce is the whole cost

`two_card_parallelism.md` showed column-parallel scaling at 1.93x because it has no collective.
Row-parallel is the honest counter-case: the layer ends in an all-reduce of the output, and on
this box that all-reduce costs more than the compute it parallelizes. Measured on the real NVFP4
kernel with a real NCCL all-reduce, **two cards are 1.5–2.8x slower than one at every M tested.**

## The result

Down-projection layer, H = N = 2560, I (input dim, the split axis) = 8192, tile (128,128),
NCCL over the P2P-forced link:

| M | 1 GPU (full K) | TP=2 total | compute (half K) | all-reduce | AR share | speedup |
|---|---|---|---|---|---|---|
| 128 | 0.0380 ms | 0.0560 ms | 0.0223 ms | 0.0368 ms | 66% | **0.68x** |
| 512 | 0.0383 ms | 0.1072 ms | 0.0212 ms | 0.0872 ms | 81% | **0.36x** |
| 4096 | 0.3301 ms | 0.6883 ms | 0.1003 ms | 0.5895 ms | 86% | **0.48x** |

All TP results are correct against an fp32 reference (`cos 0.999997`), so this is a fair
comparison, not a broken split.

## Why the all-reduce dominates

The machine is extremely compute-rich and interconnect-poor:

- fp4 GEMM: **~850 TFLOP/s** on one card.
- cross-card all-reduce: **~34 GB/s** (and only that high with `NCCL_P2P_LEVEL=SYS`).

The arithmetic-intensity break-even is `850e12 / 34e9 ≈ 25000 FLOP per byte`. A row-parallel
layer moves `M·H·2` bytes through the collective while doing `2·M·H·(I/2)` FLOPs per card, so the
ratio is `I` FLOPs per collective byte — about 8192 here, a third of the break-even. The
collective is therefore structurally larger than the compute.

Two regimes, same verdict:

- **Small M (decode, M = 128):** the GEMM is launch-overhead bound (~0.02 ms flat), so halving
  the K barely helps; the all-reduce adds 0.037 ms and dominates (66%). Net 0.68x.
- **Large M (prefill, M = 4096):** compute finally scales (0.10 ms) but the all-reduce scales
  with M too and grows faster (0.59 ms, 86%). Net 0.48x. The M = 512 point is the worst (0.36x)
  because it sits where compute is still flat-overhead but the collective already tracks M.

## What this means for the deployment

**TP = 2 on this box is a capacity decision, not a speed decision.** The 130 GB NVFP4 model does
not fit in 85 GB, so the weights must be split. But row-parallel split *slows the math down*; it
is the price paid for fitting, not a win. That is consistent with the serving config choosing
`--disable-custom-all-reduce` and leaning on expert-parallel rather than a wide row-parallel
tensor split — EP moves tokens, not the full activation, and keeps most traffic off this link.

Practical rules:

- **Column-parallel** (output dim): near-linear 1.93x, use it freely.
- **Row-parallel** (input dim): always a net loss here on a standalone layer; only worth it if
  the collective overlaps real compute, and even then the overlap has to hide 66–86% of the
  layer. Do not assume TP=2 ≈ 2x.
- Prefer axis choices that shrink the cross-card payload (EP, or replication) over ones that
  add a per-layer all-reduce of the full `M·H` activation.

## A side finding: this kernel is wrong below M = 128

While building the layer, `DenseGemmKernel` (b12x) returned garbage for `M = 16/32/64`
(`max abs diff ≈ 1100` against a correct reference) and hit `CUDA_ERROR_ILLEGAL_INSTRUCTION` at
`M = 1`. It is correct from `M = 128` up — one full tile row. Real small-M decode needs the
`use_m1_non_tma_a/c/sfa` and `direct_one_m_tile_scheduler` constructor flags; the default
configuration assumes at least one 128-row tile. See
[`blockscaled_three_paths`](blockscaled_three_paths.md) for the kernel's small-M note.

## Measured on

- RTX 6000D ×2, compute capability **(12, 0)** = sm_120a, 85651 MiB per card, SYS link (no NVLink)
- NCCL **2.30.7+cuda13.3** with `NCCL_P2P_LEVEL=SYS`, torch 2.14.1+cu130, cutlass 4.8.0
- Probe: `_probes/probe_sm120_two_card_layer.py`
