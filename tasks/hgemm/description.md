# hgemm

Half-precision general matrix multiply on Ampere (sm_86 / RTX 3090): `C = A · B`, both operands
bfloat16, fp32 accumulation, fp32 output. The shape of every transformer's dense layer and of
attention's QK^T / PV products — the operator a 3090 owner most wants faster than the library.

## Signature

- inputs, in order:
  - `a`: `[m, k]`, bfloat16 — left operand, row-major, one row per output row
  - `b`: `[k, n]`, bfloat16 — right operand, row-major, one column per output column
- output:
  - `c`: `[m, n]`, float32 — the product `a · b`

## Semantics

`c[i, j] = sum_{p=0}^{k-1} a[i, p] * b[p, j]`, with the sum accumulated in **float32** and the
result written as float32. Inputs are bfloat16, so each product is `bf16 * bf16` widened to fp32
before it joins the accumulation; the accumulation order is not part of the contract (any order
that sums in at least fp32 is legal). K is the reduction (contraction) axis and may be large.

## Legal inputs

Any finite bfloat16 values in `a` and `b`. The reduction axis `k` is the same on both inputs.
Real inputs are activations and weights; extreme magnitudes are not assumed away (no
renormalisation is part of the operator), but no overflow is expected in fp32 accumulation for
the shapes below. `m`, `n`, `k` are each a positive multiple of the micro-tile dimensions the
kernel chooses; the shapes file fixes the three tested workloads.
