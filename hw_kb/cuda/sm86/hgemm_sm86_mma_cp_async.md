# hgemm_sm86_mma_cp_async (a half-precision GEMM on Ampere sm_86: `mma.sync` HMMA + `cp.async` feed, the two engines RTX 3090 has, and the ones it does not)

> Skeleton, not a certified recipe. The mechanism is present and correct against the PTX ISA;
> no `PROBE PASS` reading is attached yet, so every number row below is `[to be measured]`.
> A row earns ✅ only when its `## Measured on` block carries a `PROBE PASS` job id on an
> `sm86` target. Until then this entry exists so a student on a 3090 has the right *shape*
> to fill in, and never reaches for an interface the chip does not have.
>
> One file per hardware interface. Operator-agnostic: mechanism only. This is the sm86
> counterpart to the Hopper/Blackwell GEMM entries: where those feed operands with TMA and
> accumulate in TMEM via `wgmma`/`tcgen05`, sm86 has neither — it feeds with `cp.async` and
> accumulates in registers with `mma.sync`. See "What sm86 does not have" in `index.md`.

## Purpose

Compute `C = A · B` (f16/bf16 inputs, fp32 accumulate) on Ampere `sm_86`, using the only two
tensor-path engines the chip has: **`mma.sync.aligned.m16n8k16`** (SASS `HMMA.16816.F32`, warp
MMA with register accumulators) for the math, and **`cp.async`** (16 B granularity, `commit_group`
/ `wait_group`) for the global→shared feed that overlaps memory with compute. There is no warpgroup
MMA, no tensor memory, no TMA descriptor copy, and no CTA cluster — so a pipeline that works here
is built from the same primitives a Volta/Turing kernel used, not the Hopper ones.

## When needed (trigger)

- The target arch token is `sm86` (RTX 3090 / A10-class GA10x) and the task is a GEMM or any
  matmul-shaped op (attention QK/PV, MLP up/down projection).
- You are about to write `wgmma`, `tcgen05`, `cp.async.bulk.tensor`, an `mbarrier`-tracked bulk
  copy, or `cluster.sync` — none exist on sm86; the compile error (or a silent fall to a slower
  path) is the trigger to rewrite onto `mma.sync` + `cp.async`.
- The stall ledger (`skills/stall`) is dominated by `long_scoreboard` on the operand loads: the
  feed is not overlapped, so the k-loop waits on global memory every step — the `cp.async` ring
  below is the mechanism that hides it.
- `LDL`/`STL` (local-memory spill) shows up in SASS: the accumulator or a register array fell out
  of registers — the fix is more independent accumulators spread across warps, not bigger tiles
  (sm86 has 255 regs/thread and 64 KB-96 KB smem/SM; a tile that needs more spills).

## Minimal runnable example

```cpp
// fk-cuda-cflags: -arch=sm_86 (or sm_86a) -std=c++17
// Half-precision GEMM C[M,N] = A[M,K] * B[K,N], A row-major, B row-major.
// Every block computes one 128x128 tile with an 8x8 warp grid; each warp owns a 32x64
// micro-tile composed of 16x8 HMMA fragments. K is walked in BK=32 chunks double-buffered
// through shared memory with cp.async. Nothing here is sm90-only.
#include <cuda_fp16.h>
#include <cuda_pipeline.h>   // __pipeline_memcpy_async / commit / wait

#define BM 128
#define BN 128
#define BK 32
#define WM 32
#define WN 64
#define NTHREADS 256

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t a[4],
                                             const uint32_t b[2], const float (&c)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Stage one BK-chunk of A and B into smem with cp.async (16 B per thread per copy).
__device__ __forceinline__ void stage_async(__half* sA, __half* sB,
                                            const __half* gA, const __half* gB,
                                            int block_row, int block_col, int k0, int M, int K, int N) {
  const int tid = threadIdx.x;
  // A: BM x BK, row-major in global; each thread copies 16 B (8 halfs).
  for (int idx = tid * 8; idx < BM * BK; idx += NTHREADS * 8) {
    int r = idx / BK, c = idx % BK;
    const __half* src = gA + (size_t)(block_row + r) * K + (k0 + c);
    __pipeline_memcpy_async(&sA[idx], src, 16);
  }
  // B: BK x BN, row-major; symmetric.
  for (int idx = tid * 8; idx < BK * BN; idx += NTHREADS * 8) {
    int r = idx / BN, c = idx % BN;
    const __half* src = gB + (size_t)(k0 + r) * N + (block_col + c);
    __pipeline_memcpy_async(&sB[idx], src, 16);
  }
  __pipeline_commit();
}

__global__ void __launch_bounds__(NTHREADS) hgemm(const __half* A, const __half* B, float* C,
                                                  int M, int N, int K) {
  __shared__ __half sA[2][BM * BK];   // double buffer
  __shared__ __half sB[2][BK * BN];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int wRow = warp / 4, wCol = warp % 4;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;

  // accumulate a WM x WN micro-tile as (WM/16) x (WN/8) fp32 HMMA fragments
  float acc[(WM / 16) * (WN / 8)][4] = {};

  stage_async(sA[0], sB[0], A, B, block_row, block_col, 0, M, K, N);  // prime buffer 0
  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    int next = buf ^ 1;
    if (k0 + BK < K) {   // prefetch the next chunk into the other buffer
      stage_async(sA[next], sB[next], A, B, block_row, block_col, k0 + BK, M, K, N);
    }
    __pipeline_wait_prior(1);   // buffer `buf` is ready; the in-flight one is `next`
    __syncthreads();

    // main loop over the BK chunk: load fragments from smem, run HMMAs.
    // (Fragment load from sA/sB into HMMA operand registers omitted for brevity: A's 4
    //  registers per lane are rows {lane/4, lane/4+8} of the 16-row fragment at cols 2*(lane%4)
    //  and +1, +8, +9; B's 2 are the 8-row fragment at the k and k+8 rows. ldmatrix.x4 loads
    //  A; ldmatrix.x2.trans / ld.shared loads B — see `mma_sync_hmma_run_register_feed.md`.)
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t a[4], b[2];
      load_fragments(sA[buf], sB[buf], kk, wRow, wCol, lane, a, b);
      mma_m16n8k16(acc[0], a, b, acc[0]);   // one fragment; the real kernel loops the 2x8 grid
    }
    __syncthreads();
    buf = next;
  }

  // epilogue: write acc to global (row = block_row + wRow*WM + ...; col = block_col + wCol*WN + ...).
  store_accumulators(C, acc, block_row, block_col, wRow, wCol, lane, N);
}
```

Acceptance (`cuobjdump -sass`):

| Count | How to count |
|---|---|
| `HMMA` (not `OMMA`/`QMMA`/fallback) | the math landed on tensor cores; a single-precision fallback or an `FFMA` loop means the fragment types were wrong |
| `LDGSTS` / `cp.async` groups | one `LDGSTS` per 16 B copy; `CPASYNC.WAITGROUP` placement matches the double buffer |
| `LDL` / `STL` | must be `0` (no spill); a non-zero count means the accumulator array exceeded the register budget |
| `BAR.SYNC` per k-chunk | exactly the `__syncthreads()` pair around the HMMA inner block |
| `TMA` / `UTMALDG` / `UMMA` | must be `0`; their presence means sm90+ code leaked onto an sm86 target |

## Gotchas

1. **One `mma.sync` is one warp's 16x8x16 tile.** To fill a 128x128 block you compose many
   fragments across the 8x8 warp grid and accumulate each into its own register group. A
   micro-tile too large for 255 regs/thread spills to local memory (`LDL`/`STL`) and the spill
   traffic erases the tensor-core gain — tile down, or raise occupancy, before anything else.
2. **`cp.async` is 16 B granularity and needs `commit_group`/`wait_group`.** A `cp.async.cg`
   copy of a misaligned address falls back to a slower path or is undefined; every address is
   a multiple of 16 B. The `wait_prior(N)` count is the number of *groups still allowed in
   flight*, so a double buffer waits with `wait_prior(1)` before consuming the just-committed
   group while the prefetch of the next stays in flight.
3. **`cp.async` does not imply a barrier.** It makes the data visible to the issuing thread's
   warp (or the CTA after `__syncthreads()`), but the consuming HMMAs of *other* warps need a
   `__syncthreads()` between `wait_prior` and the fragment loads — the example's barrier is that;
   dropping it gives a race that is usually numerically wrong and occasionally right.
4. **No `mbarrier`-tracked bulk copies / no TMA on sm86.** `cp.async.bulk.tensor`, tensor-map
   descriptors and async-arrive barriers are Hopper+; on sm86 the only async load is `cp.async`
   (or `__pipeline_memcpy_async`), and the only barrier is `__syncthreads` / `bar.sync`.
   Reaching for the Hopper spelling is a compile error at best.
5. **No `ldmatrix.trans` shortcut for every fragment mapping.** `ldmatrix.x4` (A) and
   `ldmatrix.trans` (B) load standard fragment mappings; a custom smem layout must still present
   the byte pattern those instructions expect (16 B rows, the 8x8-block interleave), or the
   numerics are wrong in a way that passes a spot check. See
   `mma_sync_hmma_run_register_feed.md` for the mapping and the bank-conflict faces.
6. **K must agree on both operand layouts.** A is row-major `.row` and B row-major `.col` in the
   `mma.sync` spelling above (the `.col` means B is consumed transposed per fragment); feeding B
   row-major without the transposed fragment load produces a plausible but wrong C. sm86 gives no
   free transpose — it is either an `ldmatrix.trans` on smem or a swapped index at the load.
7. **Bank conflicts are the smem tax, not the HMMA.** With no swizzle a `ldmatrix`/`ld.shared`
   pattern can serialize on shared-memory banks; `skills/stall`'s `short_scoreboard`/`mio_throttle`
   on the fragment-load rows is the reading. The fix is a swizzled layout (xor of the row index
   into the column unit), not more pipeline stages. See `index.md`'s smem-traffic row.
8. **`cp.async.ca` vs `.cg`**: `.cg` (cache-global, bypass L1) is the usual choice for streaming
   operands larger than L1 reuse; `.ca` (cache-all) helps when B is reused across blocks and fits
   L1. Getting this wrong is not a correctness bug, only a bandwidth one — measure before switching.

## Measured on

```text
[to be measured] — no PROBE PASS on sm86 yet.
A row is certified only when a job id from an arch = "sm86" target lands here with its
same-job ratio and the NCU/stall reading that moved.
```

## Sources

- PTX ISA: `mma.sync.aligned.m16n8k16` fragment layouts, `cp.async` (`cp.async.cg` / `.ca`),
  `ldmatrix` / `ldmatrix.trans`, `bar.sync`: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- CUDA C++ Programming Guide: asynchronous data copies (`__pipeline_memcpy_async`,
  `cuda::pipeline`), shared-memory bank conflicts, `__launch_bounds__`, register pressure:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- CUDA Binary Utilities (`cuobjdump -sass`, the `HMMA` / `LDGSTS` / `LDL` mnemonics):
  <https://docs.nvidia.com/cuda/cuda-binary-utilities/>
- Ampere tuning guide (smem/SM, registers/thread, bank-conflict guidance):
  <https://docs.nvidia.com/cuda/ampere-tuning-guide/>
- Neighboring entries: [`mma_sync_hmma_run_register_feed.md`](mma_sync_hmma_run_register_feed.md)
  (the HMMA fragment feed and dependency latency in detail), and the sm86 `index.md`
  ("What sm86 does not have").
