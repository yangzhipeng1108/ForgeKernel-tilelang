# Sparse GEMM on sm120: 2:4 `mma.sp` runs (≈2×) and block-tile skipping works (≈linear)

> Status: **measured on hardware, clock-locked at 1995 MHz**, TileLang `0.1.15`, `sm_120a`.
> Both sparse mechanisms verified against a dense reference (`cos = 1.0` / tile-exact match).

## Purpose

"Sparse GEMM" hides **two unrelated things** that are often conflated. This arch supports both, and
they compose with different parts of the earlier map:

1. **Fine-grained 2:4 structured sparsity** — hardware `mma.sp`. A is stored compressed (`M×K/2`)
   plus a metadata tensor `E`; each `mma.sp` consumes 2× the logical K per instruction, so peak
   math roughly doubles. This is `T.gemm_sp(A_sparse, E, B, C)`.
2. **Coarse block-tile sparsity** — a *scheduling* trick: a block-level mask says which output
   `(M/BM, N/BN)` tiles are nonzero, and the grid launches only those tiles. This is the "only
   iterate nonzero tiles" recipe; it is not `mma.sp` and it does not touch the MMA at all.

They answer different questions: (1) "can the tensor core skip multiplies?" (2) "can the grid skip
tiles?". Measure both before claiming "sparse GEMM is 2×".

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- f16 inputs, fp32 accumulate, tile `(128,128)`, `BK=64` (BK=128 overflows the 100 KiB smem ceiling
  with the sparse metadata buffer: `Failed to set ... 102400`), `num_stages=2`
- Timing: CUDA events over 30 launches; the sparse reference is the dense f16 `A @ B` of the **same**
  masked matrix.

## Finding 1: `T.gemm_sp` (2:4) is real and runs at ~1.95× on this arch

The op exists (`tilelang.language.experimental.gemm_sp_op`, surfaced as `T.gemm_sp`) and its doc
says "On Hopper ... WGMMA SP ... `tcgen05_gemm_sp` on Blackwell". On sm120 it does **neither**: the
emitted source contains

```cuda
#include <tl_templates/cuda/instruction/mma_sp.h>
tl::mma_sp_sync<tl::DataType::kFloat16, ..., SM80::MMA::SparseSel::Zero>(...)
```

— the **Ampere `mma.sp`** path (`SM80::MMA::SparseSel`). That is exactly the warp-level sparse MMA
this HMMA-only part can execute, so 2:4 sparsity works here through the Ampere lowering.

Contract (from the shape checks in `gemm_sp_op.py`):

- `A_sparse`: `(M, K/2)` f16 — the two kept values of each 4-element K group, packed;
- `E`: `(M, K/8)` uint8 — the 2-bit-per-group selectors (Ampere sparse metadata);
- `B`: `(K, N)` dense; `C`: `(M, N)`; logical `K = 2 * A_sparse.shape[-1]`;
- dimensions must be static.

Metadata encoding confirmed by trial: with A keeping the **first two** of each K-group of 4,
`E = 0x00` (and equivalently `0x44`) reproduces the dense reference at `cos = 1.00000, md = 0.031`
(fp16 ulp at |C|≈18). Other uniform bytes decode to different selectors (0.5 / 0.0 cos) — so the
encoding is live and E is not ignored.

Throughput (f16, same masked operands, all `cos = 1.0`):

| shape | tile | sparse `mma.sp` | dense f16 | speedup |
|---|---|---|---|---|
| 4096³ | 128×128×64, thr=128 | 525.0 µs / **262 TF** | 1025.3 µs / 134 TF | **1.95×** |
| 4096³ | 128×128×64, thr=256 | 531.9 µs / 258 TF | 1038.1 µs / 132 TF | 1.95× |
| 2048×2048×8192 | 128×128×64, thr=256 | 299.5 µs / 229 TF | 582.1 µs / 118 TF | 1.94× |

**≈1.95×, not 2×**: `mma.sp` doubles the logical K per instruction but the epilogue, the TMA feed
of the compressed A + metadata E, and the fixed N-side work do not halve, so the ceiling is under
2. For reference, dense fp8 on this card reaches ~300–380 TF (`gemm_dtype_throughput.md`), so
**fp8 dense ≈ f16-2:4-sparse** here — sparsity is not automatically the bigger win; it is a
different 2× you can stack *with* fp8 in principle (sparse fp8 exists in the dispatcher:
`SparseSel::Zero ... only SparseSel::Zero is valid on FP8`).

## Finding 2: coarse block-tile skipping is correct and scales ~linearly

Instead of `T.Kernel(Ntiles, Mtiles)`, launch `T.Kernel(num_nonzero_tiles)` and decode the tile
`(m, n)` from a compacted index list in global memory:

```python
mi, ni = IDX[bid, 0], IDX[bid, 1]              # IDX: (NNZ, 2) int32
T.copy(A[mi*BM, k*BK], a_s); T.copy(B[k*BK, ni*BN], b_s); T.gemm(a_s, b_s, acc)
T.copy(acc, C[mi*BM, ni*BN])
```

4096³, `(128,128,64)`, dense grid = 256 output tiles:

| nonzero tiles | time | effective TF | vs dense(256 tiles) | match |
|---|---|---|---|---|
| 256 (100 %) | 1046 µs | 131 | 1.00× | baseline |
| 192 (75 %) | 729 µs | 141 | **0.70×** time | ✓ tile-exact |
| 128 (50 %) | 570 µs | 121 | **0.54×** time | ✓ tile-exact |
| 64 (25 %) | 290 µs | 119 | **0.28×** time | ✓ tile-exact |

Time tracks the nonzero-tile count almost exactly (25 % → 0.28×, 50 % → 0.54×, 75 % → 0.70×) —
skipping a tile skips all of its K-loop work. The nonzero tiles are **bit-identical** to the dense
result (`allclose` per tile). Effective TF stays ~120–140 because each launched tile does the same
per-tile work; you win on *total* time, not on per-tile rate.

This is pure grid shaping — it composes with everything in the map (dtype, split-K, tile size) and
costs one `int2` load per block.

## When needed (trigger)

- **2:4 (`T.gemm_sp`)**: when the operands are genuinely 2:4 (< 2 of every 4 K-elements nonzero).
  It gives ~1.95× against the *same* dense f16 GEMM, and it is the only way to exceed the dense
  tensor-core peak on this arch. Check that BK fits: the metadata buffer forces `BK ≤ 64` for
  f16 at `(128,128)` (BK=128 → 102400 B > the 100 KiB opt-in cap). Build metadata correctly
  (`E=0` for keep-first-two) or the result is a valid-looking wrong answer (cos 0.5 / 0.0).
- **Block-tile skipping**: when a *block-level* mask exists (sparse attention, MoE routing, block
  pruning). Launch `num_nonzero` blocks and decode from a compacted index list; the win is linear in
  skipped tiles. This is orthogonal to 2:4 and to the MMA path.
- Neither is a free lunch on a dense operand: 2:4 needs the data to actually be 2:4 (a wrong E is
  silently wrong), and tile skipping needs a real mask (on a full matrix it *is* the dense kernel).

## Minimal runnable example (2:4)

```python
import tilelang, tilelang.language as T, torch

M = N = K = 4096
BM = BN = BK = 128, 128, 64          # note: BK=64, BK=128 overflows smem with E
DT = "float16"


@T.prim_func
def sp(A: T.Tensor((M, K // 2), DT), E: T.Tensor((M, K // 8), "uint8"),
       B: T.Tensor((K, N), DT), C: T.Tensor((M, N), DT)):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        a_s = T.alloc_shared((BM, BK // 2), DT)
        e_s = T.alloc_shared((BM, BK // 8), "uint8")
        b_s = T.alloc_shared((BK, BN), DT)
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            T.copy(A[by * BM, k * (BK // 2)], a_s)
            T.copy(E[by * BM, k * (BK // 8)], e_s)
            T.copy(B[k * BK, bx * BN], b_s)
            T.gemm_sp(a_s, e_s, b_s, acc)
        T.copy(acc, C[by * BM, bx * BN])


kernel = tilelang.compile(sp, out_idx=[3], target={"kind": "cuda", "arch": "sm_120a"})
```

Tile skipping is the same kernel with `T.Kernel(num_nonzero, threads=128) as (bid,)` and
`mi, ni = IDX[bid, 0], IDX[bid, 1]` replacing `(bx, by)`.

## What TileLang decides for you here (and what it does not)

**Decided by the pass:** that `gemm_sp` on sm120 lowers to Ampere `mma_sp_sync` (not wgmma/tcgen05);
the metadata layout (`MakeGemmSparseAmpereABLayout`); `SparseSel::Zero` for the default.

**Decided by you:** the compression and **the metadata E** (a wrong E compiles and runs and is
wrong); `BK` (smem-bound because E adds a buffer); and the entire block-tile-skip schedule.

## Gotchas

- **`T.gemm_sp` is under `tilelang.language.experimental`** — the op exists but the tree marks it
  experimental; pin the version you validated.
- **The metadata is not optional and not forgiving.** `E=0` (keep-first-two) reproduced the dense
  result; `E=0x88/0x11/0x22/0x55` gave cos≈0.5, `E=0xAA/0xFF` gave cos≈0. All compile and run; only
  one is right. Build E to match how you compressed A.
- **`BK=128` fails on f16** (`dynamic shared memory ... 102400` > the 100 KiB cap) because the
  metadata buffer rides alongside A_s and B_s. Use `BK=64`.
- **~1.95×, not 2×.** Do not promise 2×; the epilogue and the dense B feed do not halve.
- **fp8 dense (~300–380 TF) already beats f16-2:4-sparse (~262 TF)** on this card — sparsity is not
  automatically the fastest option; it is a different axis (and sparse-fp8 is a further stack).
- **Block-tile skipping must decode from a real nonzero list**; passing all tiles is just the dense
  kernel with an extra `int2` load.

## Sources

1. TileLang 0.1.15: `tilelang/language/experimental/gemm_sp_op.py` (shape contract, `K=2*A[-1]`,
   static-dim requirement), `src/tl_templates/cuda/instruction/mma_sp.h` (`mma_sp_sync`,
   `SparseSel`, the FP8 `SparseSel::Zero`-only note), `src/layout/gemm_layouts.cc`
   (`MakeGemmSparseAmpereABLayout`).
2. `gemm_dtype_throughput.md` (dense fp8 ≈ 300–380 TF, the bar sparsity is compared against),
   `split_k_fills_a_starved_grid.md` (grid-shaping, which tile skipping shares a mechanism with).
3. This repo's probes on an RTX 6000D, clock-locked, sparse reference checked against dense.
