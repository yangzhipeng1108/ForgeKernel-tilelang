# `T.gemm`: the operand layouts it accepts and the warp policy that splits the work

> Status: **skeleton — the mechanism is documented, the readings are not taken yet.**
> Every `[to be measured]` below must be produced on a real box before this counts as
> evidence. The signature and the enum members are read off the installed 0.1.15 package.

## Purpose

Issue the matrix multiply of a tile without writing the MMA sequence: `T.gemm` takes two
smem/fragment operands and an accumulator fragment, chooses the instruction path for the arch,
and splits the output tile across warps according to a policy.

## Measured on

- TileLang version: `0.1.15`
- arch: `[to be measured]` (this file is under `sm90/`)
- Certification: `[to be measured]` — a `PROBE PASS` job id, or the NCU census axis that moved
  (`sm__inst_executed_pipe_tensor`, or the emitted MMA instruction count for the tile).

## When needed (trigger)

- The GEMM's achieved throughput is well under the machine's, and the first thing to rule out
  is whether the warps are splitting the tile the way you assumed.
- A milestone decision prices "which warp policy" — the answer is one of three enum values,
  and the difference shows up as a different N-split, not as a different instruction count.
- The kernel refuses an operand with a layout complaint.

## Minimal runnable example

```python
import tilelang
import tilelang.language as T


@T.prim_func
def gemm_tile(A: T.Tensor((4096, 4096), "float16"),
              B: T.Tensor((4096, 4096), "float16"),
              C: T.Tensor((4096, 4096), "float32")):
    BM, BN, BK = 128, 128, 32
    with T.Kernel(T.ceildiv(4096, BN), T.ceildiv(4096, BM), threads=128) as (bx, by):
        a_s = T.alloc_shared((BM, BK), "float16")
        b_s = T.alloc_shared((BK, BN), "float16")
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(4096, BK), num_stages=3):
            T.copy(A[by * BM, k * BK], a_s)
            T.copy(B[k * BK, bx * BN], b_s)
            T.gemm(a_s, b_s, acc, policy=T.GemmWarpPolicy.Square)
        T.copy(acc, C[by * BM, bx * BN])
```

The 0.1.15 signature is
`T.gemm(A, B, C, transpose_A=False, transpose_B=False, policy=GemmWarpPolicy.Square,
clear_accum=False, mbar=None, annotations=None)`.

**`transpose_A` / `transpose_B` describe the stored orientation of the operand**, not the
mathematical intent. `B` stored `(N, K)` and `B` stored `(K, N)` need different flags and
different smem shapes; getting this backwards produces a shape error at trace time, which is
the good outcome — see Gotchas for the bad one.

## What TileLang decides for you here (and what it does not)

**Decided by the pass:**

- which MMA instruction the arch gets (this is the "MMA path" `../README.md` warns about — a
  DSL reading of an equivalent hand-written kernel does **not** transfer);
- the register layout of the accumulator fragment;
- how the instruction is tiled inside the warp.

**Decided by you:**

- `policy`, one of `GemmWarpPolicy.Square` / `.FullRow` / `.FullCol`. This is the **only**
  warp-split knob, and it is a policy name, not a warp count: the actual split is derived from
  `threads=` and the tile shape.
- the tile shape (`BM`, `BN`, `BK`) and therefore the smem budget;
- `clear_accum`, when the accumulator must not be zeroed per call.

## Gotchas

- **`policy` does not change the instruction count; it changes which warps get which part of
  the output tile.** A row-split and a column-split can execute the same number of MMA
  instructions and differ in smem traffic and in the reduction that follows. Do not infer "the
  policy did nothing" from an unchanged instruction count.
- **`threads=` and the tile shape have to admit the split.** A policy whose split does not
  divide the tile evenly fails at compile time; `[to be measured: the exact refusal text for
  this card's skeleton]`.
- **`clear_accum=False` accumulates into whatever was in `acc`.** In a `T.Pipelined` loop the
  `T.clear(acc)` before the loop is the thing that makes the first iteration correct.
- **A `T.gemm` inside the pipelined loop is the consumer that makes the stage count meaningful.**
  Moving it out (or computing it from a non-staged buffer) leaves the copy unoverlapped, and
  the stage sweep then measures nothing.
- **`T.gemm` is TileLang's own primitive and is the intended matrix path in this mode.** Calling
  `torch.matmul` / cuBLAS is forbidden by the card, and it would also take the measurement out
  of the kernel being measured.

## Sources

1. TileLang 0.1.15, installed package: `tilelang/language/gemm.py` (`T.gemm` signature),
   the `GemmWarpPolicy` enum (`Square`, `FullRow`, `FullCol`).
2. TileLang docs: `examples/` (the matmul examples in the `tile-ai/tilelang` repository) and
   `docs/`; <https://tilelang.com>.
3. NVIDIA PTX ISA, for the MMA instruction the path lowers to:
   <https://docs.nvidia.com/cuda/parallel-thread-execution/>.
4. This repo's probe ladder: `hw_kb/tilelang/sm90/gpu_probe_ladder.md`.
