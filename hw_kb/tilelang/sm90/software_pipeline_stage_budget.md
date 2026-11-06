# Software pipeline: `T.Pipelined(num_stages=...)` and what it actually stages

> Status: **skeleton — the mechanism is documented, the readings are not taken yet.**
> Every `[to be measured]` below is a number this card must produce on a real box before the
> entry counts as evidence. Everything else is read off the installed 0.1.15 source and its
> docstrings, which is what a student needs to write the code at all.

## Purpose

Overlap the global → shared feed of a loop body with the compute that consumes it, so the
`cp.async` / bulk-copy latency of iteration `i+k` hides behind the math of iteration `i`
instead of being paid on the critical path.

## Measured on

- TileLang version: `0.1.15` (`python -c "import tilelang; print(tilelang.__version__)"`)
- arch: `[to be measured]` — this file lives under `sm90/`; the Hopper reading is the one that
  certifies it
- Certification: `[to be measured]` — a `PROBE PASS` job id, **or** the NCU census axis that
  moved. On this repo the axis is one of `inner/ncu.py`'s `KEY_METRICS` short keys:
  `stall_long_sb` (waiting on the feed) and `dram_pct` are the two that a working pipeline
  moves; `smem_dynamic` gives the stage count the compiler actually allocated. Report
  before → after on the same job.

Without that line this entry is a description, not evidence. TileLang chooses part of the
schedule itself, so the only honest reading is one taken of the code it emitted for this
source, on this version, on this arch.

## When needed (trigger)

- The kernel's `T.copy` from global to shared sits inside the main loop and the loop body has
  enough math to hide the copy, but NCU shows the memory pipe idle while the math pipe runs
  (or the reverse) — the two are serialized.
- A stage-count change the milestone calls "tunable" is about to be swept. See **Gotchas**: the
  sweep is meaningless if the loop you pipelined is not the one carrying the copy.

## Minimal runnable example

```python
import tilelang
import tilelang.language as T

M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 128, 32
THREADS = 128
STAGES = 3          # the knob: buffer count between the producer and the consumer


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
```

`num_stages` is the **maximum number of buffers** the pass may allocate between producer and
consumer. It is a budget, not a command: the pass decides how many of them it actually uses.

## What TileLang decides for you here (and what it does not)

**Decided by the pass** (do not try to control these from the source):

- how many of the `num_stages` buffers are actually materialized;
- where the wait is placed relative to the consumer;
- whether the copy becomes a bulk copy, a TMA descriptor feed, or a plain load for this shape
  and dtype;
- **how many threads the kernel actually launches.** This one surprises people: with
  `num_stages > 0` TileLang adds a producer warpgroup. A kernel traced with `threads=128`
  compiles to `__launch_bounds__(256, 1)`. Verified on 0.1.15:

  | traced `threads=` | `num_stages` | emitted `__launch_bounds__` |
  |---|---|---|
  | 128 | 0 | `(128, 1)` |
  | 128 | 3 | `(256, 1)` |
  | 256 | 3 | `(384, 1)` |

  So `threads=` is the **consumer** count, not the launch count. A grid/occupancy calculation
  built from `threads=` alone is wrong whenever the pipeline is on, and a smem or register
  budget derived from it is wrong in the same way. Read `__launch_bounds__` out of the emitted
  source (or `launch__registers_per_thread` from NCU) rather than trusting the source you wrote.

**Decided by you** (these are the knobs the milestone prices):

- `num_stages` itself. `0` means **the pipeline is not enabled at all** — the loop runs
  synchronously. This is the single most common reason a "tuned" stage count shows no win:
  it was never on.
- which loop carries `T.Pipelined`. Only the loop body that contains both the copy and the
  consumer can be staged; wrapping an outer loop stages nothing.
- the manual form `order=` / `stage=`, which the docstring recommends **instead of**
  `num_stages` when you schedule by hand. Mixing the two is not a supported combination.

## Gotchas

- **`threads=` is not the launched thread count when the pipeline is on.** See the table above:
  `threads=128, num_stages=3` launches 256 threads. Any occupancy or register argument that
  starts from `threads=` has to start from `__launch_bounds__` instead.
- **`num_stages` lives where the copy is.** A `T.copy` outside the `T.Pipelined` body is not
  staged, whatever the stage count says.
- **`num_stages=0` is off, not "let the compiler choose"** (0.1.15 docstring: "if `num_stages`
  is 0, pipeline will not be enabled"). If a stage sweep produces byte-identical output at
  every value, check this first. The emitted source makes it visible without a GPU: at
  `num_stages=0` a Hopper matmul lowers with **zero** `tma` and **zero** `mbarrier`
  occurrences; at `num_stages=2` and above it has 4 and 14+. (`cp.async` is the Ampere-era
  spelling and does **not** appear in Hopper output — counting it is a false negative.)
- **Buffers must be declared outside the loop.** Allocating `a_s` inside the `T.Pipelined` body
  defeats the buffering — each iteration gets a fresh buffer and there is nothing to overlap.
- **The stage count is a ceiling on shared memory.** `STAGES x (BM x BK + BK x BN) x sizeof(dtype)`
  has to fit the block's budget together with everything else the kernel allocates, or the
  kernel fails to compile. `[to be measured: the exact smem ceiling this card's skeleton hits]`
- **A symbolic extent here fails to lower** in this mode — the card's compile path pins static
  shapes. See `../README.md`.

## Sources

1. TileLang 0.1.15, installed package: `tilelang/language/pipelined.py` (`T.Pipelined`
   docstring — the `num_stages` semantics quoted above), `tilelang/transform/` (the pipeline
   pass).
2. TileLang docs: `docs/programming_guides/software_pipeline.md` in the `tile-ai/tilelang`
   repository; <https://tilelang.com>.
3. This repo's probe ladder: `hw_kb/tilelang/sm90/gpu_probe_ladder.md`.
