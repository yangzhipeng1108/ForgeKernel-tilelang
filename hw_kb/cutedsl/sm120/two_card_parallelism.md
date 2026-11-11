# Two cards actually help: 1.93x column-parallel, and the NCCL trap that hides it

`clusters_and_two_gpus.md` answered "can the two cards talk" (P2P yes, 45 GB/s, no cross-card
multicast). This answers the question that follows: **is a two-card run actually faster, and by
how much?** Both numbers are now measured on real kernels, not inferred.

The short answer: a column-parallel split is near-linear (1.93x of 2x), because it needs no
per-layer collective. The danger is not the link — it is that NCCL, left to itself, does not use
the link.

## The trap: NCCL silently routes over Socket, not P2P

First run, no hints, `torchrun --nproc_per_node=2`, `dist.all_reduce`:

```
NCCL INFO NVLS multicast support is not available on dev 0
NCCL INFO NET/Socket : Using [0]enP2p97s0u3:192.168.8.127<0> ...
  all_reduce fp32 256 MiB: 21.5 ms
```

NCCL saw no NVLink, found no IB device (`NET/IB : No device found`), and fell back to the
container's **TCP socket** transport over `192.168.x`. That is a ~12 GB/s path sitting next to a
Gen5 x16 PCIe link that does ~45 GB/s. Roughly **3.7x** left on the table with no warning beyond
an INFO line.

Forcing the PCIe peer path fixes it:

```
NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 ...
```

and the same collectives become:

| collective | size | Socket (default) | P2P forced (`NCCL_P2P_LEVEL=SYS`) |
|---|---|---|---|
| all_reduce fp32 | 256 MiB | ~12 GB/s | **33.4 GB/s** |
| all_reduce bf16 | 256 MiB | — | **34.2 GB/s** |
| p2p send 0→1 fp32 | 256 MiB | — | **43.9 GB/s** |

The one-way `send` reaching 43.9 GB/s matches the independent `copy_` reading (45.5 GB/s) in
`clusters_and_two_gpus.md`, which confirms the P2P path is genuinely in use — not a cached or
loopback shortcut. **This is the same `NCCL_P2P_LEVEL=SYS` the serving config already sets; the
probe explains why it is not optional on this box.**

## A unit bug worth recording

The very first version of this probe printed `all_reduce 256 MiB: 12465 GB/s` (12 TB/s). Two
mistakes compounded: the bandwidth divided by `1e6` (giving MB/ms mislabelled as GB/s) *and* the
transport was Socket. If a collective bandwidth exceeds the physical link by more than ~2x, the
transport or the units are wrong — never the hardware.

## Column-parallel TP GEMM: 1.93x on the real NVFP4 kernel

The honest test of "does the second card help" is a split with no per-layer collective:
column-parallel, where each GPU owns half the output columns and computes independently.

`DenseGemmKernel` (NVFP4) at M=2048, K=4096, tile (128,128):

| run | work per GPU | wall time | throughput | correctness |
|---|---|---|---|---|
| 1 GPU | 2048 × 4096 × 4096 | 0.0806 ms | 852 TF/s | md=2, cos 0.999999 |
| 2 GPU (column split) | 2048 × **2048** × 4096 | 0.0418 ms | **1647 TF/s** aggregate | both halves md=2, cos 0.999999 |

**Speedup 1.93x of a possible 2x — 96.5% efficiency.** The missing 3.5% is launch/barrier
overhead; there is no cross-card data movement in this split, so the SYS link is not touched
during the GEMM. That is exactly why column-parallel scales here and row-parallel does not.

## What this implies for the two parallel axes

- **Column-parallel (output dim split): near-linear.** Each card does independent work; no
  collective in the mainloop. This is where the second card pays off, and 1.93x is what to
  expect for a balanced split.
- **Row-parallel (input dim split): pays the all-reduce.** Every layer needs an activation
  all-reduce at ~34 GB/s. For a decode step the tensor is small (`batch · hidden · 2 B`), so the
  cost is a few tens of µs and acceptable; for a long prefill it grows with sequence length and
  must be overlapped with compute or avoided. This is the real reason the serving config treats
  the pair carefully rather than as one wide GPU.
- **Expert-parallel**: each card owns experts; traffic is the token dispatch/combine, also on the
  ~34 GB/s path. Whether it wins depends on how much per-token routing traffic the split creates.

## Reproduce

- `_probes/probe_sm120_two_card_nccl.py` — collectives; run with and without `NCCL_P2P_LEVEL=SYS`
  to see the 12 → 34 GB/s jump.
- `_probes/probe_sm120_two_card_gemm.py` — 1-GPU vs 2-GPU column-parallel NVFP4 GEMM, with an
  fp32 reference so a wrong split cannot pass.

## Measured on

- RTX 6000D ×2, compute capability **(12, 0)** = sm_120a, 85651 MiB per card
- GPU0↔GPU1 = **SYS** (PCIe Gen5 x16), no NVLink; both cards 156 SMs
- NCCL **2.30.7+cuda13.3**, `lmsysorg/sglang:dev`, torch 2.14.1+cu130, cutlass 4.8.0
- Probes: `_probes/probe_sm120_two_card_nccl.py`, `_probes/probe_sm120_two_card_gemm.py`
