# Software pipeline on sm120: `num_stages`, the producer warpgroup, and the `a`-suffix gate

> Status: **measured on hardware, NCU axis included.** Every number below is a real reading
> from an RTX 6000D — lower-level counts from the emitted source, and the stall/pipe-utilization
> columns from Nsight Compute.

## Purpose

Overlap the global → shared feed of a loop body with the compute that consumes it, so the copy
latency of iteration `i+k` hides behind the math of iteration `i`.

## Measured on

- TileLang version: `0.1.15` (`python -c "import tilelang; print(tilelang.__version__)"`)
- arch: `sm_120a` — **not `sm_120`**, see the target table below
- Hardware: NVIDIA RTX 6000D (driver 610.43.02, CUDA 13.3), 2 devices per box

Lower-level facts, read out of the emitted source for a 4096³ f16 matmul, tile
`(128,128,32)`, traced with `threads=128`:

| `num_stages` | emitted `__launch_bounds__` | `tma` | `mbarrier` | `wgmma` |
|---|---|---|---|---|
| 0 | `(128, 1)` | 0 | 0 | 0 |
| 2 | `(256, 1)` | 4 | 14 | 0 |
| 3 | `(256, 1)` | 4 | 16 | 0 |
| 4 | `(256, 1)` | 4 | 18 | 0 |

Executed (L3), 4096³ f16, `num_stages=3`:

| | value |
|---|---|
| numerical check | `max_diff = 2.7e-3` vs an fp32 `a @ b` reference (fp16 accumulate noise) |
| latency | ≈ 1.03 ms |
| throughput | ≈ 133 TFLOP/s |

Executed (L3) under `ncu`, same 4096³ f16 problem, tile `(128,128,32)`, one row per stage count:

| `num_stages` | `smem_dynamic` | regs/thread | `stall_long_sb` | `dram_pct` | `tensor_pct` | kernel time |
|---|---|---|---|---|---|---|
| 0 | 16.4 KiB | 198 | **9.38** | 11.17 % | 62.76 % | 606 µs |
| 2 | 32.8 KiB | 240 | **1.47** | 12.74 % | **71.98 %** | 530 µs |
| 3 | 49.2 KiB | 240 | **1.16** | 12.55 % | 71.10 % | 538 µs |
| 4 | 65.5 KiB | 240 | 1.50 | 12.93 % | 72.78 % | 523 µs |

- Certification: ✅ **NCU axis taken** (`scripts/measure_kb_entries.py --arch sm120 --ncu`,
  Nsight Compute 2026.3.0, 6 configs, all parsed against the `k_kernel` launch). The pipeline
  does what it claims: `stall_long_sb` collapses from 9.38 to 1.16 and the tensor pipe goes
  from 62.8 % to 71.1 %. **But see the plateau below — reads of "more stages is better" on this
  arch are wrong.** The `kernel time` column is `gpu__time_active.sum` under the profiler;
  compare it across rows only, and use the wall-clock figure above for absolute latency.

### The stage count plateaus at 2, and the KB must say so

`num_stages=2` already recovers almost the whole gain. Going 2 → 4 costs **+32.8 KiB of shared
memory per block** and buys **+0.8 points of `tensor_pct`**, with `stall_long_sb` *rising* from
1.16 (at 3) to 1.50 (at 4). The win is between **0 and 2**, not between 2 and 4.

The mechanism is visible in the shared-memory store counters: at `num_stages=0`,
`smem_wavefronts_st = 1.83e7` and `bankconf_st = 1.53e6` — the copies are scalar-ish shared
stores with real bank conflicts. From stage 2 onward both collapse to **0**, because the copies
become `tma` (`tma=4` at every stage count ≥ 2). Once the feed is a TMA bulk transfer it never
touches the shared-memory store path again, so there is nothing left for a fourth buffer to
overlap.

**Practical reading: budget `num_stages=2` on sm120 and spend the saved shared memory on a
larger tile.** On a card whose `shared_memory_per_block_optin` is 99 KiB, an extra 33 KiB of
pipeline depth is the most expensive way to buy 0.8 %.

## The target is `sm_120a`, and getting this wrong is the first failure on this arch

| target | compile | executes | why |
|---|---|---|---|
| `sm_120a` | ✅ | ✅ | correct |
| `sm_120` | ❌ | — | `nvcc`: `static assertion failed: tl::warpgroup_reg_dealloc requires a target with CTA register reconfiguration, such as sm_90a` |
| `sm_100a` | ✅ | ❌ | `CUDA_ERROR_NO_BINARY_FOR_GPU` |

The trap is that `compile_kernel_source(...)` — the lower-only path, and the one that works
without a GPU — **succeeds for `sm_120`** and hands back source that `nvcc` then refuses. A
producer warpgroup needs `setmaxnreg`, which is an architecture-*accelerated* feature and is
only present on `a`-suffixed targets.

## When needed (trigger)

- The feed is serialized against the math: NCU shows the memory pipe idle while the math pipe
  runs, or the reverse. On this arch that reads as **`stall_long_sb` above ~5 with
  `tensor_pct` in the low 60s** — the `num_stages=0` row above is the canonical example.
- A stage-count sweep that shows no movement. Check `num_stages` is not 0 first, and check the
  target carries the `a` suffix second.
- Conversely, **`stall_long_sb` around 1 means the pipeline is done and adding stages is waste.**
  Do not sweep past 2 on this arch without first enlarging the tile.

## Minimal runnable example

```python
import tilelang
import tilelang.language as T

M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 128, 32
THREADS = 128
STAGES = 3


@T.prim_func
def matmul(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
           C: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=THREADS) as (bx, by):
        a_s = T.alloc_shared((BM, BK), "float16")
        b_s = T.alloc_shared((BK, BN), "float16")
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=STAGES):
            T.copy(A[by * BM, k * BK], a_s)
            T.copy(B[k * BK, bx * BN], b_s)
            T.gemm(a_s, b_s, acc)
        T.copy(acc, C[by * BM, bx * BN])


kernel = tilelang.compile(matmul, out_idx=[2], target={"kind": "cuda", "arch": "sm_120a"})
```

## What TileLang decides for you here (and what it does not)

**Decided by the pass:**

- how many of the `num_stages` buffers are actually materialized;
- where the wait is placed relative to the consumer;
- **how many threads the kernel launches.** With `num_stages=3`, a kernel traced with
  `threads=128` compiles to `__launch_bounds__(256, 1)`: TileLang appends a producer warpgroup.
  `threads=` is the consumer count, not the launch count, and any occupancy or register
  argument built on it is wrong whenever the pipeline is on.

**Decided by you:**

- `num_stages`. `0` means the pipeline is **not enabled at all**.
- which loop carries `T.Pipelined` — only a body holding both the copy and its consumer stages.
- the manual `order=` / `stage=` form, which the docstring recommends *instead of*
  `num_stages` when scheduling by hand.

## Gotchas

- **`sm_120` lowers and does not compile.** See the table above; the error names
  `warpgroup_reg_dealloc`, not the arch.
- **A lower-only tool cannot certify an arch decision.** `compile_kernel_source` never runs
  `nvcc`, so it cannot see arch-gated intrinsics. Any arch conclusion has to survive an actual
  `tilelang.compile(...)`.
- **`num_stages=0` is off, not "compiler chooses"** (0.1.15 docstring). The emitted source makes
  this visible without a GPU: `tma`/`mbarrier` are 0/0 at 0, 4/14 at 2.
- **`cp.async` never appears here.** It is the Ampere-era spelling; counting it is a false
  negative. Use `tma` / `mbarrier`.
- **Buffers must be declared outside the loop**, or each iteration gets a fresh buffer and
  there is nothing to overlap.
- **The stage count is a ceiling on shared memory**: `STAGES x (BM x BK + BK x BN) x
  sizeof(dtype)` has to fit alongside everything else the kernel allocates.

## Sources

1. TileLang 0.1.15, installed package: `tilelang/language/pipelined.py` (`T.Pipelined`
   docstring), `tilelang/transform/` (the pipeline pass),
   `src/tl_templates/cuda/intrin.h` (the `setmaxnreg` static assertion quoted above).
2. TileLang docs: `docs/programming_guides/software_pipeline.md` in `tile-ai/tilelang`;
   <https://tilelang.com>.
3. NVIDIA PTX ISA, `setmaxnreg` and `cp.async.bulk.tensor`:
   <https://docs.nvidia.com/cuda/parallel-thread-execution/>.
4. This repo's probes on an RTX 6000D, via `scripts/measure_kb_entries.py --arch sm120`.
