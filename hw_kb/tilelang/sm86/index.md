# SM86 (Ampere / RTX 3090) interface index — TileLang mode

The sm86 tree of the TileLang mode (a Python `kernel_tl.py` student). Demand-driven: a recipe is written when the
student gets stuck on that interface, following `../TEMPLATE.md` (examples are TileLang, not `cute.*`), and is
marked ✅ on **only** one of the two kinds of evidence the other trees accept:

- **a GPU probe returning `PROBE PASS`** — the default kind;
- **a shipped KEEP whose own NCU census shows the contract axis moved** — for knobs that are numerically
  invisible (cache-policy bits, drain depth): both spellings compute the same bytes, so a `max_diff` probe
  cannot see the change.

This file is not a plan; it is the ledger of what has already been earned. **Every recipe row is empty on
purpose.** A TileLang entry may not be produced by translating a `cute.*` or an sm90/sm100 entry: TileLang
chooses `num_stages` and the MMA path and infers part of the layout itself, and the sm86 hardware is a
different engine from Hopper/Blackwell (see "What sm86 does not have").

**Not finding something here does not mean it cannot be done** — for interface details, TileLang's own
`examples/` and `docs/`, the PTX ISA, and your own probes are authoritative; what you win on a probe gets
written in here in the finish round.

## What sm86 is, and what it does not have

RTX 3090 is Ampere `sm_86`: 82 SMs (GA102), 24 GB GDDR6X, third-gen Tensor Cores. Relative to the Hopper
(sm90) and Blackwell (sm100) trees, three interface families **do not exist here** and a student that emits
them will get a compile error or silent lowering to something slower:

- **No `wgmma`** (warpgroup MMA is sm90+). Matrix multiplies use the `mma.sync` family (`HMMA`, `m16n8k16`
  f16/bf16, `m16n8k8` tf32). TileLang's `T.gemm` must be left to pick the `mma` path — do not force a
  warpgroup spelling.
- **No `tcgen05` / TMEM** (5th-gen Tensor Core with tensor memory is sm100+). There is no `tmem` allocator,
  no `tcgen05.mma`; the accumulator lives in registers.
- **No TMA descriptor / bulk `cp.async.bulk` with a tensor map** (Hopper+). Asynchronous global→shared copies
  use **`cp.async`** (`cp.async.cg`/`.ca`, 16 B granularity), not the TMA box engine. `T.copy` on this arch
  lowers to `cp.async`, not to a TMA.

What sm86 **does** have: `cp.async`, `mma.sync` HMMA/tf32, `ldmatrix`/`stmatrix`, `mbarrier` arrival/wait,
standard smem swizzling, `__pipeline` wrong-order-free double buffering. Cluster / distributed shared memory
(`sm90`+) is **absent** — no CTA clusters, no multicast.

## Recipes (skeletons — mechanism documented, readings not taken)

A row here is **not** a certified recipe. These files exist so that the code a student needs is written down
with the pitfalls already mapped; each carries `[to be measured]` where the evidence has to go. A row becomes
certified only when its `## Measured on` block is filled with a `PROBE PASS` job id or a moved NCU census
axis, and records the TileLang version and `arch = "sm86"`.

| Interface | Entry | Evidence |
|---|---|---|
| `T.Pipelined(num_stages=)` — stage budget over `cp.async` (no TMA to hide latency behind) | `software_pipeline_stage_budget.md` | `[to be measured]` |
| `T.copy` across scopes — lowers to `cp.async`, not TMA; `T.annotate_layout` only pins named buffers | `tiled_copy_scope_and_swizzle.md` | `[to be measured]` |
| `T.gemm` operand orientation and `GemmWarpPolicy` on the `mma` path | `gemm_operand_layout_and_policy.md` | `[to be measured]` |
| Writing a GPU probe: one rung per process, verdicts to a file | `gpu_probe_ladder.md` | `[to be measured]` |

## Symptoms that will demand a recipe (no entry yet — probe it, then write it)

| What you see / what you are about to do | The reading that turns it into an entry |
|---|---|
| **`T.gemm` fails to compile or emits a `wgmma`/`tcgen05` instruction** | sm86 has neither; confirm the lowering picked the `mma.sync` path (SASS shows `HMMA`/`OMMA`), and record the TileLang version |
| **Latency not hidden by `num_stages`** — synchronous-looking SASS, no win from more stages | On sm86 the feed is `cp.async`, not TMA; prove the `cp.async.commit_group`/`wait_group` placement and stage count before adding stages |
| **Kernel hangs / never returns after launch** — `synchronize` never returns, killed at the job clock | One rung per process, each with its own seconds budget; a bare timed run only burns the whole clock |
| **Is the lane↔element mapping / swizzle / matrix-atom membership right** (`ldmatrix`/`stmatrix` readback) | Double-edge ramp, a fine + a coarse pattern, sentinel prefill, scalar cross-check |
| **A layout I annotated is ignored**, or the emitted smem layout differs from what I asked | Prove the emitted layout on a probe: `T.annotate_layout` applies only to the buffers it names |
| **A TileLang API refuses my spelling** (`T.copy` src/dst shape, `T.gemm` operand layout, `num_warps`) | Read the refusal text against the TileLang version's own source, and record the version in the entry's `Measured on` block |
