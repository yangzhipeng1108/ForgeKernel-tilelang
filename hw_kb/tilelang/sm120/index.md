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

Two of the census results **contradict the advice a Hopper-trained reader would write**, and both
are the kind of thing only a counter can settle:

- the pipeline **plateaus at `num_stages=2`** — 2 → 4 costs 33 KiB of the 99 KiB budget for
  0.8 points of `tensor_pct` (`software_pipeline_stage_budget.md`);
- **swizzling a pure copy does nothing** — the copy is at `dram_pct = 93 %` and is not waiting on
  the shared-memory port; the layout axis is closed with "no effect"
  (`tiled_copy_scope_and_swizzle.md`).

| Interface | Entry | Evidence |
|---|---|---|
| `T.Pipelined(num_stages=)`, the producer warpgroup, the `sm_120a` target gate | `software_pipeline_stage_budget.md` | ✅ NCU census, plateau at 2 |
| `T.gemm` → `mma.sync` on this arch; `GemmWarpPolicy` sweep | `gemm_operand_layout_and_policy.md` | ✅ NCU census, `Square` reads shared 25 % less |
| `T.copy` across scopes; the 99 KiB shared-memory ceiling | `tiled_copy_scope_and_swizzle.md` | ✅ NCU census, swizzle inert on a DRAM-bound copy |
| Writing a GPU probe: one rung per process, verdicts to a file | `gpu_probe_ladder.md` | harness ran ✓ 4/4 PASS |

## Symptoms that will demand a recipe (no entry yet — probe it, then write it)

| What you see / what you are about to do | The reading that turns it into an entry |
|---|---|
| **`nvcc` fails with `warpgroup_reg_alloc` / `warpgroup_reg_dealloc` static assertion** | The target is not `a`-suffixed. Fix the target before touching the kernel — see the table above. This is the first thing to rule out on this arch, and it is not a kernel bug. |
| **`CUDA_ERROR_NO_BINARY_FOR_GPU` at load** | The binary was built for another arch (an `sm_100a` cubin will compile cleanly here and then not load). The build target and the device capability must agree exactly. |
| **Lower-only says the pipeline is on, but compilation fails** | `compile_kernel_source` does not invoke `nvcc`; it cannot see arch-gated intrinsics. Confirm any arch conclusion with an actual `tilelang.compile(...)`, not with the lower output alone. |
| **MMA path is neither `wgmma` nor `tcgen05`** | On sm120 both counters are 0 in the emitted source. Do not infer "the matrix path is broken" from that — probe the instruction that *is* emitted before writing an entry about it. |
| **Kernel hangs / never returns after launch** | One rung per process, each with its own budget in seconds; a bare timed run only burns the whole clock and tells you nothing. |
| **Writing a multi-contract GPU probe**; or a round comes back all FAIL with only one fact | One contract per rung, self-contained digest per rung, verdicts and facts on separate channels. |
| **smem traffic over the line** (`smem_wavefronts_st/ld` above the reference) | Wavefronts × bankconf four quadrants, to separate conflict-type from volume-type before touching the swizzle — **but check `dram_pct` first.** If it is above ~90 % the kernel is bandwidth bound and the smem counters are not on the critical path; `tiled_copy_scope_and_swizzle.md` measured a 2.6 % conflict rate against 93 % DRAM utilization and found swizzle worth nothing. |
| **A layout I annotated is ignored — or my padded tile will not lower** | Prove the emitted layout on a probe: `T.annotate_layout` applies only to the buffers it names. Padding a shared tile (`alloc_shared((BLK, BLK+4))`) is *not* an alternative here: the copy lowers as a TMA bulk transfer and `LowerBulk` rejects a stride that differs from the extent. Use `make_swizzled_layout`. |
