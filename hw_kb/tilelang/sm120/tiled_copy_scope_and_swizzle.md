# `T.copy` across scopes on sm120, and the 99 KiB shared-memory ceiling

> Status: **measured on hardware, NCU axis included.** Numbers from an RTX 6000D, with the
> counter columns read from Nsight Compute.

## Purpose

Move a tile between global, shared and `fragment` memory with one call the compiler maps onto
threads — and know, before writing the kernel, how big a tile this card's shared memory can hold.

## Measured on

- TileLang version: `0.1.15`
- arch: `sm_120a`
- Hardware: NVIDIA RTX 6000D, driver 610.43.02, CUDA 13.3, 156 SMs

**Device limits this arch imposes** (`torch.cuda.get_device_properties`):

| limit | value |
|---|---|
| `shared_memory_per_block` | 49 152 B (48 KiB) |
| `shared_memory_per_block_optin` | **101 376 B (99 KiB)** |
| `shared_memory_per_multiprocessor` | 102 400 B (100 KiB) |
| `max_threads_per_multi_processor` | 1536 |
| `regs_per_multiprocessor` | 65 536 |

Copy throughput, 4096² f32, executed and timed:

| kernel | smem | `max_diff` | latency | bandwidth |
|---|---|---|---|---|
| fragment staging, `BLK=64` | — | 0 | 0.1664 ms | 806.5 GB/s |
| fragment staging, `BLK=128` | — | 0 | 0.1895 ms | 708.1 GB/s |
| shared tile `(64,64)` | 16 KiB | 0 | 0.0775 ms | 1732.7 GB/s |
| shared tile `(64,128)` | 32 KiB | 0 | 0.0859 ms | 1562.3 GB/s |
| shared tile `(128,64)` | 32 KiB | 0 | 0.0793 ms | 1693.3 GB/s |
| shared tile `(128,128)` | 64 KiB | 0 | 0.0859 ms | 1562.5 GB/s |

- Certification: ✅ **NCU axis taken** (`scripts/measure_kb_entries.py --arch sm120 --ncu`).
  The bandwidth column is wall-clock; the layout comparison below is the counter reading.

### Swizzling a copy kernel on this arch buys nothing — it is DRAM bound

A 4096² f32 shared-tile copy (`BLK=64`, `threads=256`), default layout vs
`T.annotate_layout({s: make_swizzled_layout(s)})`:

| layout | `dram_pct` | `stall_long_sb` | `smem_wavefronts_st` | `bankconf_st` | time |
|---|---|---|---|---|---|
| default | 93.36 % | 177.8 | 5.384e5 | 1.410e4 | 62.7 µs |
| swizzled | 93.36 % | 178.8 | 5.383e5 | 1.404e4 | 62.9 µs |

The two rows are the same measurement. Bank conflicts are **2.6 % of shared-store wavefronts**
and the kernel spends its time with `stall_long_sb ≈ 178` — it is waiting on DRAM, not on the
shared-memory port. `dram_pct = 93.4 %` says the copy is already at the memory-system ceiling;
no layout can move a number that is set by the outer bandwidth, and both rows land within 1 %.

**Practical reading: do not spend a task on swizzling a pure copy on sm120.** Swizzle pays when
the shared read is on the critical path — which is the
[GEMM entry](gemm_operand_layout_and_policy.md), where `smem_wavefronts_ld` actually
distinguishes the policies — not here, where the kernel never waits on shared memory. This
entry's `Measured on` bandwidth table is therefore a *bandwidth* result; its layout axis is
closed, with "no effect" as the finding.

**A padded tile does not work here at all**, which is worth knowing before trying the obvious
alternative (`alloc_shared((BLK, BLK+4))`): the copy lowers as a TMA bulk transfer and fails with

```
tvm::tl::cuda::Copy::LowerBulk ... src/cuda/op/copy.cc
```

because `LowerBulk` requires the shared tile's row stride to equal its extent. On this arch the
layout knob is `annotate_layout`, not padding.

## The 99 KiB ceiling is the constraint that will bite first

A shared staging buffer of a whole row of a 4096-wide f32 tensor is `4096 x 4 = 16 KiB`, so
`BM=128` rows needs 2 MiB — far past the ceiling. What that looks like from the outside:

```
tvm.error.InternalError: Failed to set the allowed dynamic shared memory size to 2097152
```

The same error already fires at 131 072 (BM=8) and 262 144 (BM=16) — i.e. **above 99 KiB, with
no hint that the usable limit is 99 KiB and not the 48 KiB static figure.** A kernel sized from
`shared_memory_per_block` (48 KiB) is over-conservative; one sized from the datacenter-Blackwell
figure (227 KiB) fails at launch. Size from `shared_memory_per_block_optin`.

For a pipelined kernel this ceiling also bounds `num_stages`, because the stage count multiplies
the buffer:

```
STAGES x (BM x BK + BK x BN) x sizeof(dtype)  <=  99 KiB
```

### Bigger tiles on this arch are SLOWER — the constraint is shared-memory bank conflicts

The 66 KiB of unused shared memory at `(128,128,32)` looks like headroom for a bigger tile. It is
not: a full tile sweep on a clock-locked RTX 6000D (SM fixed at 2000 MHz, CUDA-event timing, 4096³
f16, `num_stages=2`) shows **`(128,128,32)` is already the optimum, and every larger BM/BN loses
badly**:

| tile `(BM,BN,BK)` | smem | TFLOP/s | shared bank conflicts |
|---|---|---|---|
| **(128,128,32)** | 32 KiB | **142** | **0** |
| (128,128,64) | 64 KiB | 142 | — |
| (128,128,96) | 96 KiB | 143 | — |
| (128,192,64) | 80 KiB | 140 | — |
| (128,256,32) | 48 KiB | **82** | 14 362 |
| (256,128,32) | 48 KiB | **79** | — |
| (128,256,64) | 96 KiB | **50** | — |
| (256,256,32) | 64 KiB | **27** | **246 814** |

The mechanism is not occupancy (every config is 240 regs/thread, block=256, `warps_active ≈ 16 %`)
and not grid fill (at 8192³ the `(256,256)` grid grows 4× to 1024 blocks and still runs at 28
TFLOP/s). It is **`l1tex__data_bank_conflicts_pipe_lsu_mem_shared`**: 0 at `(128,128,32)`, 1.4e4 at
`(128,256,32)`, **2.5e5 — 17× more — at `(256,256,32)`**. A `(BK, 256)` B-tile is 512 bytes wide,
exactly four 128-byte bank rows, so the shared loads collide. Wider BM/BN buys fewer global loads
and loses far more to the shared port.

**Manual swizzle does NOT rescue it.** Annotating `make_swizzled_layout` on both operands changes
nothing — `(256,256,32)` goes 27.1 → 27.1 TFLOP/s, `(128,256,32)` 81.8 → 81.7 — because `T.gemm`
**already applies its own MMA swizzle layout** to the operands it is handed; a user annotation is
redundant. So on this arch the tile-size axis is closed in the *opposite* direction to the Hopper
intuition: **do not enlarge the tile to use the spare shared memory; enlarge BK if anything** (BK
32→96 is worth ~+1 %, and costs the whole budget).

**The sizing recipe on this arch**: `(128,128,32)`, `num_stages=2`, stop. Bank conflicts are zero
here; the moment BM or BN exceeds 128 the conflict count explodes and throughput collapses.

> Method note: an unlocked GPU oscillates between 180 MHz and 2430 MHz, and per-launch
> `perf_counter` timing adds Python/launch jitter on top. The first, unclocked sweep of this same
> code reported `(128,128,64)` at 167 TFLOP/s and `(128,128,32)` at 118 — the opposite ranking and
> ±30 % run-to-run. Both artifacts disappear once SM clocks are pinned (`nvidia-smi -lgc`) and
> timing uses CUDA events over a block of iterations. **Lock the clock and use events before
> trusting any tile ranking on this card.**

## Fragment vs shared: the measurement does not say what the name suggests

The shared-tile kernels are **2–2.2x faster** than the fragment-staging ones (1733 vs 807 GB/s
at comparable shape). That is not a claim that "shared is better than fragment" in general — the
two kernels do different amounts of work (the fragment version stages through registers with two
extra copies). It is a warning: **do not read "fragment" as the fast path on this arch**; measure.

## When needed (trigger)

- **smem traffic over the line**: `smem_wavefronts_st` / `smem_wavefronts_ld` above what the tile
  volume implies.
- **A layout you annotated appears to be ignored.** `T.annotate_layout` is consulted only for the
  buffers it names.
- **The kernel fails to launch with a dynamic-shared-memory error** — read the ceiling above.

## Minimal runnable example

```python
import tilelang
import tilelang.language as T

M, N = 4096, 4096
BM, BN = 128, 128          # 64 KiB of smem -- under the 99 KiB opt-in ceiling


@T.prim_func
def copy_tile(x: T.Tensor((M, N), "float32"), y: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        s = T.alloc_shared((BM, BN), "float32")
        T.copy(x[by * BM, bx * BN], s)      # global -> shared
        T.copy(s, y[by * BM, bx * BN])      # shared -> global


kernel = tilelang.compile(copy_tile, out_idx=[1], target={"kind": "cuda", "arch": "sm_120a"})
```

`T.copy` also takes `coalesced_width`, `disable_tma`, `eviction_policy`, `prefer_instruction`,
`loop_layout` and `annotations` in 0.1.15.

## What TileLang decides for you here (and what it does not)

**Decided by the pass:** how the copy maps onto threads; which instruction it picks for a scope
pair and shape; the default smem layout and the index map for every buffer `T.annotate_layout`
did not name.

**Decided by you:** the buffers and their shapes; which buffers you name in
`T.annotate_layout`; `eviction_policy` when L2 residency is what you are pricing.

## Gotchas

- **99 KiB, not 48 KiB and not 227 KiB.** `shared_memory_per_block_optin = 101 376`. Exceeding
  it fails at launch with a dynamic-shared-memory error that does not state the limit.
- **`T.annotate_layout` only touches the buffers whose names are in the dict.**
- **Annotation is a request; index inference still runs.** If the emitted layout differs, that is
  a reading to take, not a bug to report.
- **`T.copy(x[...], x)` with the same buffer on both sides is a silent no-op** — this is what the
  starter kernel's `_tile_in` / `_tile_out` naming exists to avoid.
- **Global ↔ fragment directly is legal** but skips the smem stage, so a swizzle annotated on a
  shared buffer never applies.
- **The fragment staging path measured slower here.** See above; measure rather than assume.

## Sources

1. TileLang 0.1.15, installed package: `tilelang/language/copy.py` (`T.copy` signature),
   `tilelang/language/allocate.py`, `tilelang/language/annotations.py`.
2. TileLang docs in `tile-ai/tilelang`; <https://tilelang.com>.
3. NVIDIA CUDA C++ Programming Guide, "Shared Memory" — the opt-in dynamic smem limit:
   <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>.
4. This repo's probes on an RTX 6000D, via `scripts/measure_kb_entries.py --arch sm120` and the
   device-property dump in `torch.cuda.get_device_properties`.
