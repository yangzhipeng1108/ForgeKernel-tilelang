# SM120 (Blackwell workstation) interface index — TileLang mode

The sm120 tree of the TileLang mode (a Python `kernel_tl.py` student). Demand-driven: a recipe
is written when the student gets stuck on that interface, following `../TEMPLATE.md` (examples
are TileLang, not `cute.*`), and is marked ✅ on **only** one of the two kinds of evidence the
other trees accept:

- **a GPU probe returning `PROBE PASS`** — the default kind;
- **a shipped KEEP whose own NCU census shows the contract axis moved** — for knobs that are
  numerically invisible (cache-policy bits, descriptor modes, drain depth): both spellings
  compute the same bytes, so a `max_diff` probe cannot see the change.

This file is not a plan; it is the ledger of what has already been earned.

> **Start here for a GEMM:** `gemm_tuning_summary.md` is the one-page map — the 10 axes tried,
> have a verdict table, and a decision tree (only **dtype** and **split-K** pay; everything else is a
> tie or a loss). The rows below are the per-recipe detail behind it.

## The one thing this arch gets wrong if you copy sm90/sm100

**`sm_120` is not the same target as `sm_120a`, and only one of them runs.** Measured on this
tree's box (RTX 6000D, driver 610.43.02, CUDA 13.3, TileLang 0.1.15):

| target | `compile` | executes | why |
|---|---|---|---|
| `sm_120a` | ✅ | ✅ 4096³ f16 matmul, `max_diff ≈ 2e-3`, 135.8 TFLOP/s | the correct target |
| `sm_120` | ❌ | — | `nvcc` refuses: `static assertion failed: tl::warpgroup_reg_dealloc requires a target with CTA register reconfiguration, such as sm_90a` |
| `sm_100a` | ✅ | ❌ | `CUDA_ERROR_NO_BINARY_FOR_GPU` — compiles, cannot be loaded |

The trap is that `compile_kernel_source(...)` (the lower-only path) **succeeds for `sm_120`**
and produces source that then cannot be compiled. A pipeline whose producer warpgroup needs
`setmaxnreg` is only representable on an `a`-suffixed target. So:

- the card's `target` must be `{"kind": "cuda", "arch": "sm_120a"}`;
- `arch = "sm_120"` in a `task.toml` or `loop.toml` is a configuration error, not a fallback —
  it will compile-lower and then fail at `nvcc` with a message that does not name the fix.

## Recipes (measured on hardware, NCU census included)

These four carry real readings from an RTX 6000D (driver 610.43.02, CUDA 13.3, TileLang 0.1.15):
emitted-source facts, `max_diff` from execution, wall-clock timings, and an NCU census taken with
`scripts/measure_kb_entries.py --arch sm120 --ncu` (Nsight Compute 2026.3.0).

Three of the census results **contradict the advice a Hopper-trained reader would write**, and all
are the kind of thing only a counter can settle:

- the pipeline **plateaus at `num_stages=2`** — 2 → 4 costs 33 KiB of the 99 KiB budget for
  0.8 points of `tensor_pct` (`software_pipeline_stage_budget.md`);
- **swizzling a pure copy does nothing** — the copy is at `dram_pct = 93 %` and is not waiting on
  the shared-memory port; the layout axis is closed with "no effect"
  (`tiled_copy_scope_and_swizzle.md`);
- **a bigger tile is slower** — `(128,128,32)` is already optimal; enlarging BM/BN to 256 drives
  shared-bank conflicts from 0 to 2.5e5 and throughput from 142 to 27 TFLOP/s, and manual swizzle
  does not help (`tiled_copy_scope_and_swizzle.md`, "Bigger tiles are slower").

| Interface | Entry | Evidence |
|---|---|---|
| `T.Pipelined(num_stages=)`, the producer warpgroup, the `sm_120a` target gate | `software_pipeline_stage_budget.md` | ✅ NCU census, plateau at 2 |
| `T.gemm` → `mma.sync` on this arch; `GemmWarpPolicy` sweep | `gemm_operand_layout_and_policy.md` | ✅ NCU census, `Square` reads shared 25 % less |
| `T.gemm` operand dtype: f16 ≈ bf16, fp8 ≈ 2.1× (and fp8 wants BK=128) | `gemm_dtype_throughput.md` | ✅ clock-locked timing, cosine=1.0 on all rows |
| `T.copy` across scopes; the 99 KiB ceiling; tile size vs bank conflicts | `tiled_copy_scope_and_swizzle.md` | ✅ NCU census, bank conflicts 0→2.5e5 as tile grows |
| Warp specialization — `T.Pipelined`'s producer warpgroup vs `tl.enable_auto_warp_specialization` | `warp_specialization_is_already_on.md` | ✅ clock-locked sweep, explicit schedule 2–8× slower in every cell |
| Block scheduling — `T.use_swizzle` (L2 panel) and `T.PersistentTileScheduler` (resident blocks) vs raster | `block_scheduling_is_a_tie.md` | ✅ clock-locked, 24 cells, all within ±2 % (compute-bound) |
| Split-K — hand-written K-axis split + `T.atomic_add`, for a tile-starved grid | `split_k_fills_a_starved_grid.md` | ✅ clock-locked, up to 5.6× on 10 %-fill shape, a loss above one wave |
| Stream-K — **absent from 0.1.15**; hand-written, measured against split-K/plain | `stream_k_is_absent_and_loses.md` | ✅ clock-locked, 20–33 % below split-K on starved grids, loses to plain when filled |
| CLC (`clc_try_cancel`) and cooperative warp-group (`GemmWarpPolicy`) | `clc_compiles_to_nothing.md` | ✅ clock-locked + SASS/`nvcc -E` verified: CLC emits 0 instructions, policy default wins |
| Sparse GEMM — 2:4 `T.gemm_sp` (fine) and block-tile skipping (coarse) | `sparse_gemm_two_mechanisms.md` | ✅ clock-locked, sparse cos=1.0 / tile-exact: 1.95× (2:4), linear in skipped tiles |
| Thread-block clusters — `T.ClusterKernel` launch vs `T.copy_cluster` TMA multicast | `clusters_free_multicast_broken.md` | ✅ clock-locked, cluster ties plain, multicast ~6×10⁴ slower (correct) |
| Writing a GPU probe: one rung per process, verdicts to a file | `gpu_probe_ladder.md` | harness ran ✓ 4/4 PASS; SASS via `orch_census_run.py --kernel`; timing + precision + timeline all verified on hardware |

## Symptoms that will demand a recipe (no entry yet — probe it, then write it)

| What you see / what you are about to do | The reading that turns it into an entry |
|---|---|
| **`nvcc` fails with `warpgroup_reg_alloc` / `warpgroup_reg_dealloc` static assertion** | The target is not `a`-suffixed. Fix the target before touching the kernel — see the table above. This is the first thing to rule out on this arch, and it is not a kernel bug. |
| **`CUDA_ERROR_NO_BINARY_FOR_GPU` at load** | The binary was built for another arch (an `sm_100a` cubin will compile cleanly here and then not load). The build target and the device capability must agree exactly. |
| **Lower-only says the pipeline is on, but compilation fails** | `compile_kernel_source` does not invoke `nvcc`; it cannot see arch-gated intrinsics. Confirm any arch conclusion with an actual `tilelang.compile(...)`, not with the lower output alone. |
| **MMA path is neither `wgmma` nor `tcgen05`** | On sm120 both counters are 0 in the emitted source — the opcode is `HMMA` and the copy is `UTMALDG.2D`. Do not infer "the matrix path is broken" from that; read the compiled cubin (`orch_census_run.py --kernel <x>_tl.py`) and check `SASS-PRECISION-JSON` before writing an entry about it. |
| **`SASS-OPERANDS-FAIL ... no-mma-instructions-parsed` on a kernel that runs fine** | The operand census was arch-blind: it only knew the warpgroup families and missed sm120's `HMMA`, and it rejected the `R60.reuse` operand form ptxas emits. Both are fixed; after them the same cubins read `a_reg=64 a_other=0`. Measured — see `gpu_probe_ladder.md` ("two blind spots"). |
| **A milestone keeps a T15 `[VERIFY: sass-derived] commit` line on sm120** | There is no `commit`/DEPBAR group to read here: the warpgroup-MMA serialization it measures does not exist on this arch (`commit=[]`). The pipeline waits on an **mbarrier phase-check + trywait** instead (21 mbarrier sites, `SYNCS.PHASECHK.TRANS64.TRYWAIT`). Downgrade the line to a description or re-anchor it to the mbarrier count. See `gpu_probe_ladder.md`. |
| **Kernel hangs / never returns after launch** | One rung per process, each with its own budget in seconds; a bare timed run only burns the whole clock and tells you nothing. |
| **Writing a multi-contract GPU probe**; or a round comes back all FAIL with only one fact | One contract per rung, self-contained digest per rung, verdicts and facts on separate channels. |
| **`host_bound_ratio` over 0.5 on a TileLang kernel** | Almost always the kernel is small, not the harness: Python→FFI→launch costs ~6 µs, so a 6 µs kernel reads ~1.0 and a 1 ms kernel reads ~0.005. Check `graph_status` — a `1.0x` or higher graph reading means the graph cannot help either. See `gpu_probe_ladder.md`. |
| **smem traffic over the line** (`smem_wavefronts_st/ld` above the reference) | Wavefronts × bankconf four quadrants, to separate conflict-type from volume-type before touching the swizzle — **but check `dram_pct` first.** If it is above ~90 % the kernel is bandwidth bound and the smem counters are not on the critical path; `tiled_copy_scope_and_swizzle.md` measured a 2.6 % conflict rate against 93 % DRAM utilization and found swizzle worth nothing. |
| **"I have spare shared memory — I'll enlarge the tile"** | Opposite of the Hopper instinct: on sm120 `(128,128,32)` is already optimal and any bigger BM/BN collapses throughput, because a `(BK,256)` B-tile is 512 B wide and shared-bank conflicts jump from 0 to 2.5e5 (17×). Measured sweep in `tiled_copy_scope_and_swizzle.md` ("Bigger tiles are slower"). Do not enlarge BM/BN; BK 32→96 is worth ~+1 % at most. Manual swizzle does not help (`T.gemm` already swizzles). |
| **"Let's turn on warp specialization for the GEMM"** | It is already on: `T.Pipelined` appends a producer warpgroup (`threads=256` launches with `__launch_bounds__(384)` and emits `warpgroup_reg_dealloc`). Adding `tl.enable_auto_warp_specialization: "role_based"` stacks a second role split and is **2–8× slower in every measured cell** (fp8 381→170, f16 157→20 TFLOP/s). The extra split removes warps from the `HMMA` issue path. See `warp_specialization_is_already_on.md`; do not treat WS as an optimization axis here. |
| **"Add `T.use_swizzle` / a persistent schedule like FlashAttention does"** | Both are a **tie** on a compute-bound GEMM here: 24 clock-locked cells (raster vs L2 swizzle 4/8 vs persistent swizzle 1/8 col/row) land within ±2 % on every shape, because `dram_pct ≈ 13 %` — L2 order cannot reach the critical path. The schedules *are* different in the emitted code (swizzle remaps `blockIdx` via `rasterization2DRow<8>` with the grid unchanged; persistent collapses 1024 blocks onto 156 workers with a `for w` loop) — they just measure the same. Only try scheduling when a counter shows DRAM/L2 bound. See `block_scheduling_is_a_tie.md`. |
| **A skinny GEMM (small M/N, large K) leaves the machine half idle** | The grid `(N/BN)×(M/BM)` is smaller than 156 SMs — tile-starved, and **no block scheduler fixes that** (raster/swizzle/persistent are a tie, they only reorder the same blocks). Split the K axis: `SK ≈ ceil(156 / tiles)`, `T.atomic_add` the fp32 partials. Measured **46 → 183 TFLOP/s (4×)** on 512²×16384 at SK=8 (10 %→82 % fill); split-K is a *loss* once the grid is already full (4096³: 372→144). Watch the trap: `out_idx` zeroes C only once, so an `atomic_add` kernel is wrong after its first call — zero C yourself. See `split_k_fills_a_starved_grid.md`. |
| **"Use a thread-block cluster / TMA multicast to cut operand traffic"** | The cluster *launch* is free (`T.ClusterKernel` ties plain, 372–375 TFLOP/s) but **TMA multicast is unusable**: `T.copy_cluster(..., cluster_mask=3)` returns the correct matrix (`md` identical to plain) and takes **22 083 ms vs 0.348 ms** per launch — ~6×10⁴ slower. Not a deadlock, not WS (WS-off 22 173 ms, stages=0 24 895 ms). Cause is in TileLang's own source: multicast completes against a `shared::cluster` barrier, but the pipeline materializes a **CTA-local** mbarrier (`role_based_scheduler.cc`: "cluster-wide multicast backpressure cannot be expressed yet"). Re-probe after a version bump. See `clusters_free_multicast_broken.md`. |
| **"Let's use Stream-K like DeepGEMM"** or **"`T.Kernel(streamk_programs=True)`"** | Stream-K does **not exist** in TileLang 0.1.15 — `streamk`/`stream_k` have zero matches in the package (Python and C++), and `T.Kernel`'s signature is `(*blocks, threads=, prelude=, cluster_dims=)`, so that keyword is a `TypeError`. `streamk_programs` is a CUTLASS/DeepGEMM spelling. Hand-writing it (flatten `(M,N,K)`, contiguous slice per worker, flush at every output-tile boundary) is correct but lands **20–33 % below split-K** on starved grids (222 vs 277, 232 vs 293, 267 vs 323) and loses to plain when the grid is filled — the per-tile atomic flush is the cost. Use split-K instead. See `stream_k_is_absent_and_loses.md`. |
| **"Use CLC (cluster launch control) for dynamic scheduling"** or **"add warp-group cooperation"** | CLC **compiles but emits nothing** on this build: `clc_try_cancel` lowers into a `static_assert(dependent_false)` no-op (never fires), and the compiled SASS has **0** `clusterlaunchcontrol` — because `CUTLASS_ARCH_CLC_ENABLED` is off under plain `-arch=sm_120a` (`nvcc -E` probe `#errors` CLC_OFF). Worse, pairing the dead call with a live barrier **corrupts the output** (4096³ `md` 0.12 → 8498). Cooperation is already the default: `T.gemm` splits the tile across warps by `GemmWarpPolicy.Square`, which beats FullRow/FullCol by 25 % fewer smem wavefronts. Check SASS, not "it compiled". See `clc_compiles_to_nothing.md`. |
| **"Sparse GEMM gets 2× for free"** | There are **two** unrelated mechanisms. (a) **2:4 structured** via `T.gemm_sp(A_sparse, E, B, C)` — lowers to Ampere `mma_sp_sync` on this arch (not wgmma/tcgen05), measured **1.95×** vs the same dense f16 (4096³ 262 vs 134 TF, `cos=1.0`), caps at `BK=64` because the metadata buffer `E` overflows smem at BK=128, and a wrong `E` is silently wrong (0x00 right, 0x88 → cos 0.5). (b) **block-tile skipping** — launch only nonzero output tiles from a compacted index list; time drops ~linearly (25/50/75 % → 0.28/0.54/0.70×), tile-exact. Note dense fp8 (~300–380 TF) already beats f16-2:4-sparse (~262 TF). See `sparse_gemm_two_mechanisms.md`. |
| **A tile/stage sweep gives an unstable ranking** | Lock the SM clock (`sudo nvidia-smi -i 0 -lgc 2000,2000`) and time with CUDA events over a block of iterations. An unlocked RTX 6000D idles at 180–292 MHz and boosts to 2430; `perf_counter` per launch adds more. The same sweep ranked tiles oppositely and varied ±30 % before this was fixed. |
| **Three timing methods give three answers for the same kernel** | On this card the same 4096³ f16 GEMM reads 224/122 TFLOP/s (single-launch event min/median), 265 (NCU `gpu__time_duration`, L2 warm), and **137 (CUDA-graph replay, sustained)**. Tune on graph replay; use NCU only for *why*. See `gpu_probe_ladder.md` ("Timing a 4k GEMM"). |
| **"switch to bf16 for throughput" or "fp8 is 2× faster"** | bf16 = f16 within 2 % here (both 2-byte, same tensor path) — a numerics choice, not a speed one. fp8 is genuinely ~2.1× (141 → 297 TFLOP/s) and wants a bigger BK. Read `gemm_dtype_throughput.md`; and check fp8 accuracy with cosine on real activations, not `max_diff` on `randn`. |
| **A layout I annotated is ignored — or my padded tile will not lower** | Prove the emitted layout on a probe: `T.annotate_layout` applies only to the buffers it names. Padding a shared tile (`alloc_shared((BLK, BLK+4))`) is *not* an alternative here: the copy lowers as a TMA bulk transfer and `LowerBulk` rejects a stride that differs from the extent. Use `make_swizzled_layout`. |
