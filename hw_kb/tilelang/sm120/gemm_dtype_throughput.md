# `T.gemm` input dtype on sm120: f16/bf16 are equal, fp8 is 2.1× — and fp8 wants a bigger BK

> Status: **measured on hardware.** Every number is a clock-locked RTX 6000D reading, with
> CUDA-event timing. The correctness column is a cosine similarity against an fp32 reference, so a
> fast-but-wrong configuration cannot pass silently.

## Purpose

Pick the operand datatype the matrix multiply runs in, and know what each one actually buys on
this arch — before pricing a milestone around "use fp8 for speed" or assuming bf16 is faster than
f16.

## Measured on

- TileLang version: `0.1.15`
- arch: `sm_120a`
- Hardware: NVIDIA RTX 6000D (driver 610.43.02, CUDA 13.3), SM clock pinned at 2000 MHz
  (`sudo nvidia-smi -i 0 -lgc 2000,2000`)

4096³ matmul, tile `(128,128,BK)`, `threads=128`, `num_stages=2`, fp32 accumulate, best of a block
of CUDA-event-timed iterations:

| dtype | BK | smem | TFLOP/s | vs f16 | cosine vs fp32 |
|---|---|---|---|---|---|
| **float16** | 32 | 32 KiB | **141** | 1.00× | 1.0 |
| bfloat16 | 32 | 32 KiB | 144 | 1.02× | 1.0 |
| float8_e4m3 | 32 | 16 KiB | 269 | 1.91× | 1.0 |
| float8_e4m3 | 64 | 32 KiB | 281 | 1.99× | 1.0 |
| float8_e4m3 | 96 | 48 KiB | 283 | 2.01× | 1.0 |
| **float8_e4m3** | **128** | 64 KiB | **297** | **2.11×** | 1.0 |
| float8_e5m2 | 64 | 32 KiB | 306 | 2.17× | 1.0 |
| float8_e4m3 | 256 | 128 KiB | — | launch fail (> 99 KiB opt-in) | — |

Stable values (3 repeats, ±2 %); the `BK=32` fp8 row is a single reading near 269 and is the
least-optimized point, not the best.

- Certification: ✅ **measured on hardware**, `CUDA_VISIBLE_DEVICES=0`, clock locked. The fp8
  correctness check is **cosine = 1.0** on all rows — fp8's low `max_diff` here is *not* a bug in
  the harness: with `torch.randn` inputs (|x| ≲ 4) and e4m3's 3 mantissa bits, the products
  round-trip close enough that the fp32-accumulated result matches to 1e-4. **Do not read this as
  "fp8 is always accurate"** — it is accurate for the *distribution* tested (unit-normal). Real
  activations with outliers need per-tensor scaling and re-measurement.

## f16 and bf16 are the same speed; the dtype that matters is fp8

f16 and bf16 are **within 2 % of each other** (141 vs 144) — both are 2-byte operands on the same
tensor-core path, so the choice between them is a *numerics* decision (bf16's 8 exponent bits vs
f16's 10 mantissa bits), never a *speed* decision on this arch. Do not switch to bf16 expecting a
speedup, and do not pay a task to "try both for throughput".

**fp8 is the real lever: ~2.1× f16.** It wins twice — 1 byte per element halves the shared-memory
footprint (so a bigger BK fits) *and* the tensor core does more per instruction. The 4 nm node's
FP8 path is what an RTX 6000D is sold for; a f16-only kernel leaves most of it idle.

## fp8 wants a bigger BK, and the smem budget allows it

Because an fp8 element is half an f16 element, the same shared-memory budget buys **twice the BK**:

```
smem = stages x (BM x BK + BK x BN) x bytes_per_element
```

At `(128,128)` and `stages=2`: f16 tops out around BK=48 before the 99 KiB ceiling; **fp8 reaches
BK=128 (64 KiB) comfortably**, and BK=128 is the measured best (297 vs 283 at BK=96). f16 over the
same BK range is flat (BK=32 → 48 is 141 → 142, i.e. nothing) — so this is a *dtype* effect, not a
tile effect: enlarging BK only helps once the operand is fp8.

At BK=256 fp8 hits `Failed to set the allowed dynamic shared memory size to 131072` — that is
128 KiB, over the 99 KiB `shared_memory_per_block_optin` ceiling. See
`tiled_copy_scope_and_swizzle.md` for the ceiling; the same opt-in rule applies here.

## When needed (trigger)

- A milestone is priced around "fp8" or "bf16" and you need the actual throughput delta on this
  card before committing to it.
- Achieved TFLOP/s is well under the card's fp8 capability and the kernel is still f16.
- Someone proposes a bf16-for-speed change; the reading says it is worth ≤2 %.

## Minimal runnable example

```python
import tilelang
import tilelang.language as T

M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 128, 128      # fp8: 64 KiB of smem at 2 stages
DT = "float8_e4m3"              # or "float8_e5m2"; f16/bf16 use BK=32 in practice


@T.prim_func
def matmul(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT), C: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        a_s = T.alloc_shared((BM, BK), DT)
        b_s = T.alloc_shared((BK, BN), DT)
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            T.copy(A[by * BM, k * BK], a_s)
            T.copy(B[k * BK, bx * BN], b_s)
            T.gemm(a_s, b_s, acc)
        T.copy(acc, C[by * BM, bx * BN])


kernel = tilelang.compile(matmul, out_idx=[2], target={"kind": "cuda", "arch": "sm_120a"})
```

fp8 operands in hand-written code are usually produced by a scaling pass; TileLang takes the
`float8_e4m3` / `float8_e5m2` tensors directly and the GEMM path selects the fp8 tensor-core
instruction on this arch.

## What TileLang decides for you here (and what it does not)

**Decided by the pass:** the MMA instruction and its shape for the operand dtype; how the fp32
accumulator tiles across warps.

**Decided by you:** the operand dtype; `BK` (which the dtype constrains through the smem budget);
whether an external scaling scheme is applied before the GEMM.

## Gotchas

- **f16 and bf16 are the same speed here.** Pick by numerics, not throughput.
- **fp8 `max_diff` looks small on `randn` inputs — that is the distribution, not a guarantee.**
  Check with the real activation range and a cosine/relative metric, not `max_diff` alone.
- **fp8 BK is bounded by the same 99 KiB opt-in ceiling**, at twice the BK an f16 kernel can use.
  BK=256 fails at launch.
- **`float8_e5m2` also works** (306 TFLOP/s at BK=64, e5m2's wider range, fewer mantissa bits) —
  choose e4m3 for accuracy, e5m2 for dynamic range.
- **A separate `nn.Module` dtype choice and the kernel operand dtype are not the same thing** —
  this entry measures the GEMM operand dtype the kernel is traced with.

## Sources

1. TileLang 0.1.15, installed package: `tilelang/language/gemm.py`, the fp8 dtype handling in the
   GEMM lowering (`tilelang/cuda/op/gemm/`).
2. TileLang docs in `tile-ai/tilelang`; <https://tilelang.com>.
3. NVIDIA Ada/Blackwell tuning guides on FP8 tensor-core throughput (the 2× figure is the FP8:FP16
   peak ratio the arch targets).
4. This repo's probes on an RTX 6000D, clock-locked, CUDA-event timed (`/tmp/dt2.py` in the probe
   session; method mirrored in `gpu_probe_ladder.md`).
