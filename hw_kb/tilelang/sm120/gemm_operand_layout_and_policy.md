# `T.gemm` on sm120: `mma.sync`, and a warp policy that actually changes something

> Status: **measured on hardware, NCU axis included.** Every number below is from an RTX 6000D,
> and the counter columns come from Nsight Compute rather than being inferred from wall clock.

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

Under `ncu`, the same three configurations, which attributes the spread to a counter:

| policy | `tensor_pct` | `smem_wavefronts_ld` | `stall_long_sb` | `dram_pct` | kernel time |
|---|---|---|---|---|---|
| `Square` | 71.10 % | **3.36e7** | 1.16 | 12.55 % | 538 µs |
| `FullRow` | 70.81 % | **4.19e7** | 1.13 | 12.48 % | 540 µs |
| `FullCol` | 70.99 % | **4.19e7** | 1.17 | 12.58 % | 536 µs |

- Certification: ✅ **NCU axis taken** (`scripts/measure_kb_entries.py --arch sm120 --ncu`).
  `emitted launch_bounds = (256, 1)` and profiling confirms a `block=(256,1,1)` launch — the
  producer warpgroup is real, not a source-level artifact. Note the `kernel time` column is
  from `gpu__time_active.sum` under the profiler and is not comparable to the wall-clock table
  above it: NCU serializes and replays, so use it for the *column-to-column* comparison inside
  the table and the wall-clock table for absolute latency.

### The 4 % is a shared-memory read-amplification difference, not a math difference

`tensor_pct` is effectively flat across the three policies (70.8–71.1 %) and `stall_long_sb` is
flat too (1.13–1.17). What moves is **`smem_wavefronts_ld`: 3.36e7 for `Square` versus 4.19e7
for both row and column splits — 25 % more shared-load wavefronts for the same math.** The MMA
issue rate is unchanged; `FullRow`/`FullCol` simply have to read the operands out of shared
memory more times to feed a differently-shaped warp ownership.

The wall-clock spread (~4 %) is smaller than the wavefront spread (25 %) because the kernel is
not shared-read bound — `stall_long_sb ≈ 1.2` says it is waiting on almost nothing. That is the
useful lesson: **on a well-pipelined kernel a 25 % increase in one counter class can cost only
4 % of time, so a counter delta is not a performance delta.** Attribute, then measure.

Practical: prefer `Square` here, but do not expect the policy knob to be worth more than a few
percent while the pipeline is doing its job. The table above is a `num_stages=3` table; with
the pipeline off the balance shifts, so re-measure rather than reusing this ranking.

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
