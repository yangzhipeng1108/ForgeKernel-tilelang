# `T.gemm` on sm120: `mma.sync`, and a warp policy that actually changes something

> Status: **measured on hardware.** Every number below is from an RTX 6000D, not a plan.
> The NCU axis is still open; see the last line of `## Measured on`.

## Purpose

Issue the matrix multiply of a tile without writing the MMA sequence, and — when it matters —
choose how the output tile is split across warps.

## Measured on

- TileLang version: `0.1.15`
- arch: `sm_120a`
- Hardware: NVIDIA RTX 6000D, driver 610.43.02, CUDA 13.3

4096³ f16 matmul, tile `(128,128,32)`, `threads=128`, `num_stages=3`, executed and timed:

| policy | `max_diff` vs fp32 ref | latency | throughput |
|---|---|---|---|
| `Square` | 2.66e-3 | 1.031 ms | 133.3 TFLOP/s |
| `FullRow` | 2.66e-3 | 1.069 ms | 128.6 TFLOP/s |
| `FullCol` | 2.66e-3 | 1.077 ms | 127.6 TFLOP/s |

All three are numerically identical (same fp16 accumulate noise) and **~4% apart in time**.
The emitted source differs between them (4865–4944 chars), so this is a real scheduling
difference, not noise in the measurement.

- Certification: executed on hardware; **the NCU axis (`tensor_pct`) is not yet taken**, so the
  4% spread is a wall-clock reading and not yet attributed to a counter.

## The MMA path on this arch is neither of the two you would guess

Counting tokens in the emitted CUDA source for the same kernel:

| token | sm120 (measured) |
|---|---|
| `wgmma` | 0 |
| `tcgen05` | 0 |
| `mma_sync` | present |

So sm120 is a **warp-level `mma.sync`** machine. A recipe, a counter threshold, or a heuristic
lifted from the Hopper (`wgmma`) or datacenter-Blackwell (`tcgen05`) trees does not transfer:
the instruction being scheduled is a different one.

Practical consequence for a probe: **do not use `wgmma`/`tcgen05` counts to decide whether the
matrix path is working here.** They are 0 on a kernel that runs correctly at 133 TFLOP/s.

## When needed (trigger)

- The GEMM's achieved throughput is well under the machine's and you want to know whether the
  warp split is the reason.
- A milestone prices "which warp policy". On this arch the knob is live, so the sweep means
  something — but read the next section before spending a job on it.
- An operand is refused with a layout complaint.

## Minimal runnable example

```python
import tilelang
import tilelang.language as T

M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 128, 32


@T.prim_func
def matmul(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
           C: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        a_s = T.alloc_shared((BM, BK), "float16")
        b_s = T.alloc_shared((BK, BN), "float16")
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, k * BK], a_s)
            T.copy(B[k * BK, bx * BN], b_s)
            T.gemm(a_s, b_s, acc, policy=T.GemmWarpPolicy.Square)
        T.copy(acc, C[by * BM, bx * BN])


kernel = tilelang.compile(matmul, out_idx=[2], target={"kind": "cuda", "arch": "sm_120a"})
```

**`transpose_A` / `transpose_B` describe the stored orientation of the operand**, not the
mathematical intent. A wrong flag produces a shape error at trace time — the good outcome.

## What TileLang decides for you here (and what it does not)

**Decided by the pass:**

- the MMA instruction (on this arch: `mma.sync`);
- the register layout of the accumulator;
- how the instruction is tiled inside a warp;
- **whether a given policy is distinguishable at all.** That is shape-dependent, and it differs
  between architectures — the sm90 tree measured several shapes where all three policies were
  byte-identical. Do not carry that conclusion here, and do not carry this one there: check the
  shape in front of you.

**Decided by you:**

- `policy` ∈ `Square` / `FullRow` / `FullCol`;
- the tile shape, and therefore the smem budget;
- `clear_accum`.

## Gotchas

- **A policy sweep is only meaningful if the emitted source differs.** Compare the sources
  first; if they are identical the sweep measures noise. On sm120 at `(128,128,32)` they differ,
  so the 4% is real.
- **Policy changes the warp split, not the instruction count.** Neither an unchanged nor a
  changed instruction count proves anything on its own.
- **`wgmma` = 0 and `tcgen05` = 0 are expected here.** They are the wrong arch's markers.
- **`clear_accum=False` accumulates into whatever was in `acc`.** In a pipelined loop the
  `T.clear(acc)` before the loop is what makes the first iteration correct.
- **A `T.gemm` outside the pipelined body leaves the feed unoverlapped**, and the stage sweep
  then measures nothing.
- **`sm_120` (no `a`) does not compile** when a producer warpgroup is present. See
  `software_pipeline_stage_budget.md` on this tree.

## Sources

1. TileLang 0.1.15, installed package: `tilelang/language/gemm.py` (`T.gemm` signature),
   the `GemmWarpPolicy` enum.
2. TileLang docs and `examples/` in `tile-ai/tilelang`; <https://tilelang.com>.
3. NVIDIA PTX ISA, `mma.sync` (the instruction this path emits on this arch):
   <https://docs.nvidia.com/cuda/parallel-thread-execution/>.
4. This repo's probes on an RTX 6000D, via `scripts/measure_kb_entries.py --arch sm120`.
