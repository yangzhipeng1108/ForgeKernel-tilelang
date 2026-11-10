# Block scheduling on sm120: Raster / Swizzle / Persistent are all the same speed

> Status: **measured on hardware, clock-locked**, 24 cells (4 problem shapes × 6 scheduling
> variants), fp8 e4m3, tile `(128,128,128)`, `num_stages=2`, `do_bench(warmup=20, rep=60)`.

## Purpose

Decide whether *which block computes which tile* — the schedule — matters on this card, before
spending time on it. Two families are commonly cited (and often confused with each other):

- **block scheduling** (this file): `Raster` (default `bx,by` order), `Swizzle`
  (`T.use_swizzle`, CUTLASS-style L2 panel rasterization), `Persistent`
  (`T.PersistentTileScheduler`, a resident block loops over tiles);
- **shared-memory layout swizzle** (`make_swizzled_layout`): a bank-conflict fix, not a schedule.
  See `tiled_copy_scope_and_swizzle.md`; do not conflate the two words.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 2000 MHz
- 156 SMs (`tilelang.carver.arch.driver.get_num_sms()`), so persistent waves = `ceil(tiles/156)`
- Problem shapes: `4096³`, `1024×1024×8192`, `512×512×16384` (K-heavy, tile-starved),
  `8192×8192×1024` (wide, shallow K)

## The finding: the schedule is worth ±2 % — noise

All values are TFLOP/s.

| shape | raster | swizzle 4 | swizzle 8 | persistent s=1 col | persistent s=8 col | persistent s=8 row |
|---|---|---|---|---|---|---|
| 4096³ | 367.5 | 363.7 | 369.5 | 365.9 | 371.4 | 365.4 |
| 1024×1024×8192 | 167.4 | 168.1 | 166.8 | 168.3 | 166.7 | 166.2 |
| 512×512×16384 | 47.5 | 47.0 | 47.2 | 47.1 | 47.5 | 47.7 |
| 8192×8192×1024 | 335.8 | 334.1 | 335.3 | 334.4 | 335.0 | 334.0 |

Within every row the spread is **under 2 %** and does not track the schedule. The `512×512×16384`
row is the interesting one: it only has `4×4 = 16` tiles for 156 SMs, so it reads 47 TFLOP/s —
that is a **tile-supply** limit (16 blocks cannot fill the machine), not a scheduling effect, and
the schedule does not move it either way.

## Why: this kernel is compute-bound, so L2 order cannot matter

Block swizzle and persistent scheduling both buy **L2 reuse** — the next block to run reuses
operand panels the previous one left in L2. That is worth a lot only when the kernel is waiting
on DRAM. On this card the 4096³ GEMM is at `tensor_pct ≈ 72 %` and `dram_pct ≈ 13 %` (see
`software_pipeline_stage_budget.md`); DRAM is 7× under the tensor pipe, so nothing the schedule
does to L2 can reach the critical path. The same census that closed the copy-swizzle axis
(`tiled_copy_scope_and_swizzle.md`: "the copy is at 93 % DRAM and not waiting on shared") is the
reason this axis moves nothing: the operands already fit the reuse window the TMA feed gives them.

## Proof the schedule really did change (it is not silently ignored)

The emitted CUDA is not the same across variants — so "no effect" is a measurement, not a no-op:

| variant | grid | how the tile index is derived |
|---|---|---|
| raster | `32 × 32` (1024 blocks) | `blockIdx.x` / `blockIdx.y` used directly |
| swizzle 8, row | `32 × 32` (same 1024) | `const dim3 blockIdx = tl::rasterization2DRow<8>();` — index is remapped, block count unchanged |
| persistent, `swizzle_size=8` | `156 × 1` | `sched.coord(...)` decodes a linear tile id; the kernel carries a `for w in range(waves)` loop with the `(w*39 + blockIdx.x>>2)` arithmetic |

So raster→swizzle permutes 1024 blocks (L2 panel order), and raster→persistent collapses 1024
blocks onto 156 resident workers with an internal loop. **Both are real changes; both are free.**

## When needed (trigger)

- **Never as a speed move for a compute-bound GEMM on this arch.** A `use_swizzle` or
  `PersistentTileScheduler` sweep will read inside noise; do not add it "because FlashAttention
  did."
- Consider the schedule **only if a counter shows the kernel is DRAM/L2 bound** — `dram_pct`
  above roughly 60 %, or `stall_long_sb` stuck high while `tensor_pct` sits in the 30s. None of
  the GEMM shapes measured here come close (highest `dram_pct` is 13 %).
- The schedule is still worth reaching for on a **bandwidth-bound kernel** (an elementwise or a
  transposing copy); that is a different kernel and this file does not measure it.

## Minimal runnable example

```python
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver          # NOT `from tilelang import driver`

M = N = K = 4096
BM = BN = BK = 128
DT, NSM = "float8_e4m3", driver.get_num_sms()


@T.prim_func
def persistent(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT),
               C: T.Tensor((M, N), "float16")):
    mb, nb = T.ceildiv(M, BM), T.ceildiv(N, BN)
    total, waves = mb * nb, (mb * nb + NSM - 1) // NSM
    with T.Kernel(NSM, threads=256) as (bid,):
        sched = T.PersistentTileScheduler(mb, nb, swizzle_size=8, column_major=True,
                                          stateful=False)
        a_s = T.alloc_shared((BM, BK), DT)
        b_s = T.alloc_shared((BK, BN), DT)
        acc = T.alloc_fragment((BM, BN), "float32")
        for w in range(waves):
            t = NSM * w + bid
            if t < total:
                by, bx = sched.coord(t)
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                    T.copy(A[by * BM, k * BK], a_s)
                    T.copy(B[k * BK, bx * BN], b_s)
                    T.gemm(a_s, b_s, acc)
                T.copy(acc, C[by * BM, bx * BN])


kernel = tilelang.compile(persistent, out_idx=[2], target={"kind": "cuda", "arch": "sm_120a"})
# same speed as the default:  T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM)) with no scheduler
```

## What TileLang decides for you here (and what it does not)

**Decided by the pass:**

- that a `T.use_swizzle` annotation becomes `tl::rasterization2DRow<N>()` replacing `blockIdx`;
- that `stateful=False` exposes only `coord(tile_id)` (for `for w` loops) while `stateful=True`
  allocates `m_idx`/`n_idx` state and enables `while sched.valid()` / `next_tile()`;
- the default `num_workers` of a persistent scheduler is `get_num_sms() // cluster_size`.

**Decided by you:**

- the order / panel width (`swizzle_size`, `column_major`), all of which measured identical here;
- whether to use persistent at all — the only real reason is to bound the grid to resident
  workers, which buys nothing on a compute-bound GEMM.

## Gotchas

- **`from tilelang import driver` does not exist** (`ImportError`). Use
  `from tilelang.carver.arch import driver`.
- **`stateful=True` + an inner `T.Pipelined` fails to relate**: `ProducerConsumerWS: failed to
  replace pipeline loop`. The producer/consumer warp-specialization pass cannot rewrite a
  pipelined loop nested in a `while sched.valid()`. Fixes measured: set
  `pass_configs={"tl.disable_warp_specialized": True}` (runs, but 329 vs 380 TFLOP/s — losing
  the producer warpgroup costs more than persistent buys), or use the **`stateful=False`** `for w`
  form, which keeps WS and measures 377.7 — i.e. identical to raster.
- **`swizzle order="mlx"` is rejected at compile** (`rasterization2DMLX` is not a supported
  threadblock swizzle); only `row` / `column` work.
- **A 16-tile problem is tile-starved, not schedule-starved** (47 TFLOP/s at `512×512×16384`).
  Do not read that number as "this schedule is bad."

## Sources

1. TileLang 0.1.15: `tilelang/language/annotations.py` (`use_swizzle` →
   `rasterization2DRow`/`rasterization2DColumn`), `tilelang/language/tile_schedule.py`
   (`PersistentTileScheduler`, incl. the CUTLASS-style `swizzle_size` panel semantics),
   `tilelang/carver/arch/driver` (`get_num_sms`), `src/cuda/transform/lower_l2_persistent_annotation.cc`.
2. `software_pipeline_stage_budget.md` (`tensor_pct 72 %` / `dram_pct 13 %` — the compute-bound
   premise), `tiled_copy_scope_and_swizzle.md` (the shared-memory *layout* swizzle, a different
   axis, also closed).
3. This repo's probes on an RTX 6000D, clock-locked, emitted-source grid inspection included.
