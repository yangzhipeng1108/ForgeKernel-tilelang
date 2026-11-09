# Software pipeline on sm120: `num_stages`, the producer warpgroup, and the `a`-suffix gate

> Status: **measured on hardware.** The `## Measured on` block below is a real reading from an
> RTX 6000D; the numbers are not placeholders. What is still open is the NCU axis (see the
> last line of that block): a lower-level count certifies that the pipeline *exists*, not that
> it *helped*.

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

- Certification: the readings above are from a real run, but the **NCU axis is not yet taken**
  (`smem_dynamic`, `stall_long_sb`, `dram_pct` before → after). Until it is, this entry
  establishes that the pipeline is emitted and executes; it does not establish that raising the
  stage count is a win.

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
  runs, or the reverse.
- A stage-count sweep that shows no movement. Check `num_stages` is not 0 first, and check the
  target carries the `a` suffix second.

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
