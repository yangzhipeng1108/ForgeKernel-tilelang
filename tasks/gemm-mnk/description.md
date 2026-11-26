# gemm-mnk

Half-precision GEMM (`C = A · B`, bf16 in, fp32 accumulate, fp32 out) over an explicit
**MNK configuration grid** — the CUDA-L2 style of naming each problem by its
`M_N_K` triple so one kernel is judged across the distinct shapes a real deployment
actually runs, not a single tile. A GEMM is the densest, most-optimised operator on a GPU;
matching the library here is the clearest proof a kernel-optimisation loop works.

## Signature

- inputs, in order:
  - `a`: `[m, k]`, bfloat16 — left operand, row-major
  - `b`: `[k, n]`, bfloat16 — right operand, row-major
- output:
  - `c`: `[m, n]`, float32 — the product `a · b`

## Semantics

`c[i, j] = sum_{p=0}^{k-1} a[i, p] * b[p, j]`, accumulated in **float32** (this task pins
`acc_precise = "fp32"`), written as float32. Inputs are bfloat16, each product widened to
fp32 before it joins the sum; the accumulation order is not part of the contract.

## The MNK grid

Each workload is one `M_N_K` configuration, chosen to span the regimes a 3090 meets:

| workload | M | N | K | regime |
|---|---|---|---|---|
| `tiny` | 256 | 256 | 256 | launch/occupancy bound — the kernel barely fills the SMs |
| `square` | 4096 | 4096 | 4096 | the classic balanced GEMM |
| `llm-prefill` | 8192 | 8192 | 8192 | large, compute/bandwidth balanced |
| `tall-skinny` | 16384 | 1024 | 1024 | tall M, small N — row-major-friendly, N-tile bound |
| `wide` | 1024 | 16384 | 1024 | wide N, small M — column-tile bound |
| `decode` | 64 | 4096 | 4096 | the GEMV-like decode shape — bandwidth bound, M ≪ TM |

One kernel must serve all six; a kernel that only wins the balanced `square` is not done.

## Legal inputs

Any finite bfloat16 values in `a` and `b`. The reduction axis `k` is shared. No overflow in
fp32 accumulation is expected at these sizes. `m`, `n`, `k` are each a positive multiple of
the micro-tile the kernel chooses.
