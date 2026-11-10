# Stream-K on sm120: not in TileLang 0.1.15, and hand-writing it loses to split-K

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp8 e4m3, tile `(128,128,128)`,
> 156 SMs. Stream-K is hand-written (there is no API). Every row `cos = 1.00000`.

## Purpose

Stream-K is the scheduling family that flattens the whole `(M, N, K)` iteration space into a linear
index and gives each SM one contiguous slice, so a worker may cross output-tile boundaries and a
tile's K may be split across workers. The claim is that this load-balances shapes where persistent
scheduling (whole-tile per block) tilts.

Two facts to settle before believing the hype on this arch:

1. **Does TileLang ship it?** No — see below.
2. **Does a correct hand-written version beat what we already have?** Measured: no.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz via
  `nvidia-smi -i 0 -lgc 2000,2000`, 156 SMs
- fp8 e4m3, `BM=BN=BK=128`, `num_stages=2`, C in fp32
- Timing: CUDA events over 20 launches into pre-zeroed buffers (see the trap below)

## Fact 1: there is no Stream-K API in this version

- `streamk` / `stream_k` / `stream-k`: **zero matches** in `tilelang/*.py` and `tilelang/src/**`
  (C++ included).
- `T.Kernel`'s signature is `(*blocks, threads=, prelude=, cluster_dims=)` — **no
  `streamk_programs`**. Any recipe that writes `T.Kernel(streamk_programs=True)` does not run here;
  it is a CUTLASS / DeepGEMM spelling, not TileLang 0.1.15.
- The only scheduler primitive in the package is `PersistentTileScheduler` (+ its `BaseTileScheduler`
  base, `tilelang/language/tile_schedule.py`).

So Stream-K is measured here as a **hand-written kernel**, which is what it takes on this version.

## The hand-written Stream-K (and why it must flush per tile)

```python
total = (M/BM) * (N/BN) * (K/BK)          # flattened (tile, k) space
start, end = wid * total // NW, (wid + 1) * total // NW
T.clear(acc)
for it in T.serial(end - start):
    idx  = start + it
    tile, kk = idx // kt, idx % kt
    m, n = tile // nt, tile % nt
    if it > 0 and tile != (idx - 1) // kt:          # crossed into a new output tile
        T.atomic_add(C[(prev_tile // nt) * BM, (prev_tile % nt) * BN], acc)
        T.clear(acc)                                 # accumulate per-tile, not across tiles
    T.copy(A[m*BM, kk*BK], a_s); T.copy(B[kk*BK, n*BN], b_s); T.gemm(a_s, b_s, acc)
T.atomic_add(C[(last_tile // nt) * BM, (last_tile % nt) * BN], acc)
```

The per-tile flush is mandatory: a contiguous slice spans several output tiles, so the accumulator
must be drained at every boundary (that drain is exactly the atomic traffic split-K also pays).

## Fact 2: it is correct, but it is never the fastest option

TFLOP/s, clock-locked. `waves = tiles / 156`.

| shape | tiles | waves | plain | best split-K | Stream-K best |
|---|---|---|---|---|---|
| 512×512×16384 | 16 | 0.10 | 49.1 | **277.4** (SK=8) | 222.6 (NW=312) |
| 768×768×8192 | 36 | 0.23 | 106.6 | **293.4** (SK=4) | 231.7 (NW=312) |
| 1024×1024×8192 | 64 | 0.41 | 189.6 | **323.0** (SK=2) | 267.2 (NW=312) |
| 2048×2048×8192 | 256 | 1.64 | **364.5** | 335.8 (SK=1) | 329.1 (NW=312) |
| 2048×2048×4096 | 256 | 1.64 | **341.9** | 326.8 (SK=4) | 299.2 (NW=312) |
| 4096³ | 1024 | 6.56 | **280.5** | 279.8 (SK=2) | 281.2 (NW=312) |

Read it in two bands:

- **Starved (waves < 1):** Stream-K beats plain by 2–4.5× (49 → 223, 107 → 232, 190 → 267) — it does
  fill idle SMs. **But split-K beats it in every one of those rows** (277 vs 223, 293 vs 232, 323 vs
  267). The gap is 20–33 %.
- **Filled (waves ≥ 1.6):** Stream-K loses to plain outright (329 vs 364, 299 vs 342). Nothing to
  balance; the per-tile atomic flush is pure overhead.

Full sweep for the record (Stream-K `NW` is the worker count = grid size):

| shape | plain | Stream-K NW=156 | NW=312 | NW=624 |
|---|---|---|---|---|
| 512×512×16384 | 49.1 | 213.6 | **222.6** | 170.4 |
| 1024×1024×8192 | 189.6 | 247.0 | **267.2** | 241.9 |
| 4096³ | 280.5 | 274.8 | 281.2 | 280.4 |

`NW = 2×SMs` (312) is consistently the best Stream-K setting — one worker per SM underfills because
a worker's slice begins with a serial flush; two waves per SM hide it. Past 2×SMs it degrades
(624 collapses on the smallest shape).

## Why split-K wins on this arch

Both schemes fill idle SMs by multiplying the block count. The difference is the **granularity of
the split and the shape of the reduction**:

- **split-K** keeps one worker per *output tile* and splits only **K**. Every worker writes its own
  `(BM,BN)` partial exactly **once**, at the end, with no tile-boundary flushes. The reduction is
  `SK` atomic adds per output element.
- **Stream-K** lets a worker span tiles, so it must flush **once per tile it touches** — a worker
  covering `L` tiles pays `L` flushes, and every tile is touched by up to `ceil(slice/kt)+1`
  workers. On these small grids the number of atomic writes per output element is higher than
  split-K's, and the register/shared state is re-cleared more often.

On a compute-bound card (`dram_pct ≈ 13 %`) the extra atomics and clears are the whole story —
there is no L2 balance left to win back, so the lighter reduction (split-K) wins.

## When needed (trigger)

- **Do not hand-roll Stream-K on this TileLang.** For a starved grid, split-K gives 20–33 % more for
  strictly less code; for a filled grid, plain wins. There is no measured cell where Stream-K is the
  right pick.
- The one situation that could change this is **load imbalance that split-K cannot express** — a
  shape where tiles have genuinely unequal cost (e.g. a ragged/sparse mask) and a static `SK` cannot
  equalize them. None of the dense shapes measured here have that; a dense GEMM's tiles are equal
  cost, so split-K's uniform split is already optimal.
- If a future TileLang ships native Stream-K (a real `streamk` scheduler, not the hand-written form),
  re-measure — the compiler could fuse the per-tile flush away, which is exactly the cost that sinks
  the hand-written version.

## Minimal runnable example

See the snippet above — full form in this file's "hand-written Stream-K" section. It compiles with
`tilelang.compile(fn, target={"kind":"cuda","arch":"sm_120a"})` (no `out_idx`; pass a zeroed fp32 C).

## Gotchas

- **`T.Kernel(streamk_programs=True)` does not exist here** — `TypeError`. Do not copy CUTLASS-era
  spellings into a 0.1.15 script.
- **A Stream-K worker must flush at every output-tile boundary**, not just at the end; skipping it
  sums different tiles into one output block. This is the correctness bug that is easy to write and
  that a `4096³`-only test won't catch when the slice happens to align with tile boundaries.
- **`NW` (grid) should be ~2×SMs, not 1×SMs.** One worker per SM leaves the flush exposed.
- **The clock-lock is not optional.** An unlocked run of the same sweep read 4096³ plain at 288 vs
  372 TFLOP/s across two runs — a 23 % swing that would have inverted the split-K vs plain verdict.

## Sources

1. TileLang 0.1.15: absence of any `streamk` symbol in `tilelang/*.py` / `src/**`; `T.Kernel`
   signature in `tilelang/language/kernel.py`; `PersistentTileScheduler` in
   `tilelang/language/tile_schedule.py`.
2. `split_k_fills_a_starved_grid.md` (the winning scheme, and the shared `out_idx` trap),
   `block_scheduling_is_a_tie.md` (why scheduling structure alone is free here),
   `gemm_tuning_summary.md` (the decision tree this feeds).
3. This repo's probes on an RTX 6000D, clock-locked at 1995 MHz.
