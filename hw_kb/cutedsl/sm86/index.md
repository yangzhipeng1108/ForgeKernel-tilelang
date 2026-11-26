# SM86 (Ampere / RTX 3090) interface index — CuTe DSL mode

The sm86 tree of the CuTe DSL mode (a Python `cute.*` student). Demand-driven: a recipe is written when the
student gets stuck on that interface, following `../TEMPLATE.md` (examples are `cute.*`), and is marked ✅ on
**only** one of the two kinds of evidence below. This file is not a plan; it is the ledger of what has already
been earned. **Every recipe row is empty on purpose.**

CuTe DSL targets CUTLASS atoms; on sm86 the atom set is the Ampere one, not Hopper/Blackwell. A recipe certified
on the sm90/sm100 trees is **not** carried over here. See "What sm86 does not have".

## What sm86 is, and what it does not have

RTX 3090 is Ampere `sm_86`: 82 SMs (GA102), 24 GB GDDR6X, third-gen Tensor Cores. In CUTLASS/CuTe terms the
available atoms are the **mma.sync** family (`SM80_16x8x16_F16F16F16F16_TN` etc.); these are **absent**:

- **No warpgroup MMA** (`SM90_*` wgmma atoms). The Hopper `wgmma` atom set does not lower on sm86.
- **No `tcgen05` / TMEM atoms** (Blackwell `SM100_*`).
- **No TMA copy atoms** (`SM90_TMA_*`, tensor-map descriptors). Global→shared copies use the **`cp.async`**
  copy atoms (`SM80_CP_ASYNC_CACHEALWAYS` / `CACHEGLOBAL`), not `SM90_TMA_LOAD`.
- **No CTA clusters / multicast** (`SM90_CLUSTER_*`).

What sm86 **does** have: the `SM80_*` `mma.sync` atoms, `cp.async` copy atoms, `ldmatrix`/`stmatrix` smem
round-trips, mbarrier atomics, and the standard pipeline primitives.

## Recipes (skeletons — mechanism present, readings not taken)

A row here is **not** a certified recipe. A row becomes certified only when its `## Measured on` block is filled
with a `PROBE PASS` job id or a moved NCU census axis, and records the `arch = "sm86"` and the CuTe DSL version.

| Interface | Entry | Evidence |
|---|---|---|
| `SM80_*` mma.sync atom selection and operand orientation | `mma_sync_subtile_broadcast_splitk.md` | `[to be measured]` |
| `cp.async` copy atom ring + `commit_group`/`wait_group` | `cp_async_mbarrier_ring.md` | `[to be measured]` |
| `ldmatrix`/`stmatrix` smem exchange face | `ldmatrix_stmatrix.md` | `[to be measured]` |
| Writing a GPU probe: one rung per process, verdicts to a file | `gpu_probe_ladder.md` | `[to be measured]` |

## Symptoms that will demand a recipe (no entry yet — probe it, then write it)

| What you see / what you are about to do | The reading that turns it into an entry |
|---|---|
| **You reach for an `SM90_*` / `SM100_*` atom** | sm86 lowers only the `SM80_*` mma.sync / cp.async set; confirm the chosen atom's supported arch and record the CuTe DSL version |
| **Kernel hangs / never returns after launch** | One rung per process, each with its own seconds budget |
| **Is the lane↔element mapping / swizzle / matrix-atom membership right** | Double-edge ramp, sentinel prefill, scalar cross-check |
| **The kernel computes right but the pipeline does not overlap** | Read the emitted `cp.async` groups and prove the wait placement before adding stages |
| **A spelling is refused** | Read the refusal against the CuTe DSL version's own source, and record the version in `Measured on` |
