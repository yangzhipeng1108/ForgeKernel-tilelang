# Warp specialization on sm120: already on by default, and the explicit schedule makes it slower

> Status: **measured on hardware, clock-locked, CUDA-graph-free `do_bench` (events over many
> iterations).** Every number is a real reading from an RTX 6000D.

## Purpose

Keep the feed (global → shared, TMA) off the warps that issue the math (`HMMA`), so the copy
never stalls the tensor pipe. The instinct from Hopper is "turn on warp specialization"; on this
arch the question is whether turning it on *again* helps.

## Measured on

- TileLang `0.1.15`, target `sm_120a`, RTX 6000D (driver 610.43.02, CUDA 13.3)
- SM clock locked at 2000 MHz (`nvidia-smi -i 0 -lgc 2000,2000`)
- Problem: 4096³, tile `(128,128)`, `num_stages=2`; f16 `BK=32`, fp8 `BK=128`
- Timing: `do_bench(warmup=25, rep=100)`, TFLOP/s = `2*M*N*K / t`

## The finding: the pipeline is already warp-specialized

`T.Pipelined` **appends a producer warpgroup on its own.** A kernel traced with `threads=128`
compiles to `__launch_bounds__(256)`, and the emitted source already contains one
`warpgroup_reg_dealloc` (producer) and one `warpgroup_reg_alloc` (consumer). This is the same
mechanism `software_pipeline_stage_budget.md` records (`threads=` is the *consumer* count, not
the launch count). So the TMA feed is already on dedicated warps before you ask for anything.

`tl.enable_auto_warp_specialization: "role_based"` (the only scheduler registered in 0.1.15,
`src/cuda/transform/auto_schedule.cc`) **adds a second layer**: it re-derives an explicit
`WSSchedule` (roles / pipelines / scopes) and materials it on top of the pipeline that already
split producer from consumer. The result is a kernel with `reg_dealloc` sites doubled (2 vs 1)
and a consumer that is carved up further — measured below.

## Throughput: the explicit schedule is 2–8× slower, everywhere

| dtype | threads (`=`) | `__launch_bounds__` | default | `role_based` | ratio |
|---|---|---|---|---|---|
| fp8 e4m3, BK=128 | 128 | 256 | 350.5 | 182.7 | 0.52× |
| fp8 e4m3, BK=128 | 256 | 384 | **381.1** | 170.0 | 0.45× |
| fp8 e4m3, BK=128 | 512 | 640 | 247.8 | 31.0 | 0.13× |
| f16, BK=32 | 128 | 256 | 155.8 | 35.6 | 0.23× |
| f16, BK=32 | 256 | 384 | **156.7** | 20.3 | 0.13× |
| f16, BK=32 | 512 | 640 | 156.1 | 19.5 | 0.12× |

There is **no** (dtype, thread count) cell where the explicit schedule wins. The default is not
merely "already good enough" — it is the fastest thing measured at every size. The f16 collapse
is the worse of the two (down to 0.12×): at `BK=32` the loop runs 128 iterations, so the extra
synchronization points the explicit schedule inserts are paid 128 times.

`num_stages` and an explicit register hint do not rescue it:

| fp8, threads=256 | TFLOP/s |
|---|---|
| default, stages=2 | 381.1 |
| `role_based`, stages=2 | 170.0 |
| `role_based`, stages=3 | 348.1 |
| `role_based`, stages=2 + `T.annotate_producer_reg_dealloc(24)` + `consumer_reg_alloc(240)` | 171.0 |

Stages=3 makes the explicit schedule far less bad (it recovers to 348 because the pipeline now
has room to hide the extra sync points), but it still loses to the default — and stages=3 costs
16 KiB more shared memory to get there. The explicit register hint is a no-op (171 vs 170),
which is consistent with the default path already allocating registers correctly.

## Why this is the expected result on this arch

The explicit WS schedule is written for the Hopper regime, where the producer is a full
`cp.async.bulk`/wgmma producer warpgroup and separating it *unlocks* the consumer. On sm120 the
math is warp-level `HMMA` (see `gemm_operand_layout_and_policy.md`), and `T.Pipelined` already
puts the TMA feed on its own warpgroup. Adding a second role split does not free warps for math —
it removes warps from the `HMMA` issue path (the `T.gemm` warp partition shrinks; some thread
counts no longer even cover a 128×128 tile, e.g. 384 threads fails with
`No valid warp partition ... 12 warps ... policy=Square`). The explicit schedule is strictly a
subtraction of compute warps here.

## When needed (trigger)

- **Never on this arch, as a performance move.** A `role_based` sweep on sm120 loses in every
  cell; do not "try warp specialization" as an optimization axis here.
- The only reason to touch it is a **kernel the default pipeline does not specialize** — a body
  whose feed is not a plain `T.copy` the pipeline pass recognizes. Confirm with the emitted
  source first: if `__launch_bounds__` is already `threads + 128` and a `warpgroup_reg_dealloc`
  is present, the default path already did it.
- Do **not** read the doubled `reg_dealloc` sites as "more specialization is better"; that is
  the extra layer, and it costs throughput.

## Minimal runnable example

```python
import tilelang
import tilelang.language as T

M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 128, 128          # fp8
THREADS, STAGES = 256, 2


@T.prim_func
def matmul(A: T.Tensor((M, K), "float8_e4m3"), B: T.Tensor((K, N), "float8_e4m3"),
           C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=THREADS) as (bx, by):
        a_s = T.alloc_shared((BM, BK), "float8_e4m3")
        b_s = T.alloc_shared((BK, BN), "float8_e4m3")
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=STAGES):
            T.copy(A[by * BM, k * BK], a_s)
            T.copy(B[k * BK, bx * BN], b_s)
            T.gemm(a_s, b_s, acc)
        T.copy(acc, C[by * BM, bx * BN])


# fastest: leave the pipeline alone
fast = tilelang.compile(matmul, out_idx=[2], target={"kind": "cuda", "arch": "sm_120a"})

# slower: force the explicit role-based schedule (do not ship this)
slow = tilelang.compile(matmul, out_idx=[2], target={"kind": "cuda", "arch": "sm_120a"},
                        pass_configs={"tl.enable_auto_warp_specialization": "role_based"})
```

## What TileLang decides for you here (and what it does not)

**Decided by the pass:**

- whether to append a producer warpgroup at all (it does, for a `T.Pipelined` body with a
  `T.copy` feed — you cannot opt out and do not need to);
- the `setmaxnreg` split between producer and consumer;
- the consumer `T.gemm` warp partition, which is what shrinks when an explicit schedule is added.

**Decided by you:**

- `threads=` — **the consumer count**; the launch adds 128 for the producer (`128 -> 256`,
  `256 -> 384`, `512 -> 640`). Any occupancy argument built on `threads=` alone is wrong.
- whether to add `tl.enable_auto_warp_specialization`, which on this arch you should not.

## Gotchas

- **`threads=384` cannot cover a 128×128 tile** (`No valid warp partition ... 12 warps`). The
  consumer warp count must evenly divide the tile under `Square` policy.
- **`warpgroup_reg_alloc`/`dealloc` in the source is not evidence you enabled WS.** The default
  pipeline emits them; grep the source only after knowing the baseline already has one pair.
- **`setmaxnreg` needs the `sm_120a` target.** `sm_120` refuses at `nvcc` with a
  `warpgroup_reg_dealloc` assertion — see the target table in `software_pipeline_stage_budget.md`.
- **A doubled-duration `do_bench` under an unlocked clock will look like ±30 % noise**; lock the
  clock or the 0.12×–0.45× gaps above can be read as much smaller. They are not noise.

## Sources

1. TileLang 0.1.15: `tilelang/language/warp_specialize.py` (`WSSchedule`/`WSRole`/`WSPipeline`),
   `tilelang/language/annotations.py` (`annotate_ws_schedule`), `src/cuda/transform/auto_schedule.cc`
   (the `role_based` registry — the only scheduler), `tilelang/cuda/pipeline.py` (the
   `AutoWarpSpecialization` → `MaterializeWSSchedule` chain).
2. `software_pipeline_stage_budget.md` — `threads=` is the consumer count and the producer
   warpgroup is appended by the pipeline pass.
3. This repo's probes on an RTX 6000D, clock-locked.
