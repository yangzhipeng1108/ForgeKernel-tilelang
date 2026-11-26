# SM86 (Ampere / RTX 3090) interface index — CUDA mode

The sm86 tree of CUDA mode (single-file CUDA C++ student). Demand-driven: a recipe is written when the student
gets stuck on that interface, following `../TEMPLATE.md` (examples are CUDA C++ / CuTe C++ / inline PTX), and is
marked ✅ on **only** one of the two kinds of evidence below. This file is not a plan; it is the ledger of what
has already been earned. **Every recipe row is empty on purpose.**

The sm86 hardware is a different engine from Hopper (sm90) and Blackwell (sm100): a recipe certified there is
**not** carried over here. See "What sm86 does not have".

## What sm86 is, and what it does not have

RTX 3090 is Ampere `sm_86`: 82 SMs (GA102), 24 GB GDDR6X, third-gen Tensor Cores. These instruction families
**do not exist here**; emitting them is a compile error or a silent fall back to something slower:

- **No `wgmma`** (warpgroup MMA, sm90+). Use the `mma.sync` family: `m16n8k16` f16/bf16, `m16n8k8` tf32,
  `m8n8k4` fp64. Accumulators live in registers.
- **No `tcgen05` / tensor memory** (sm100+). There is no `tmem` allocator, no `tcgen05.mma`.
- **No TMA** (`cp.async.bulk.tensor`, tensor-map descriptors, mbarrier-tracked bulk copies — Hopper+). Async
  global→shared copies use **`cp.async`** (`cp.async.cg.shared.global` / `.ca`, 16 B granularity) with
  `cp.async.commit_group` / `cp.async.wait_group`, or `__pipeline_memcpy_async`.
- **No CTA clusters / distributed shared memory / multicast** (sm90+). No `cluster.sync`, no `mapa`, no
  cluster-scoped barriers.

What sm86 **does** have: `cp.async`, `mma.sync` HMMA/tf32, `ldmatrix`/`stmatrix`, `mbarrier` arrival/wait,
standard `__syncthreads`, smem bank-conflict swizzling, `__ldg`, `__shfl_*`, and the usual `ld.global`/
`st.global` cache-policy modifiers (`.ca`/`.cg`/`.cs`/`.lu`/`.cv`). L2 is 6 MB (GA102).

## Recipes (skeletons — mechanism present, readings not taken)

A row here is **not** a certified recipe. A row becomes certified only when its `## Measured on` block is
filled with a `PROBE PASS` job id or a moved NCU census axis, and records the `arch = "sm86"`.

| Interface | Entry | Evidence |
|---|---|---|
| `cp.async` 16 B copy ring + `commit_group`/`wait_group` double buffering | `cp_async_ring.md` | `[to be measured]` |
| `mma.sync` HMMA run + operand feed from `ldmatrix` | `mma_sync_hmma_run_register_feed.md` | `[to be measured]` |
| smem swizzle / bank-conflict face for `ldmatrix`↔`stmatrix` | `smem_traffic_triage.md` | `[to be measured]` |
| Writing a GPU probe: one rung per process, verdicts to a file | `gpu_probe_ladder.md` | `[to be measured]` |

## Symptoms that will demand a recipe (no entry yet — probe it, then write it)

| What you see / what you are about to do | The reading that turns it into an entry |
|---|---|
| **You reach for `wgmma` / `tcgen05` / TMA** | sm86 has none; rewrite to `mma.sync` + `cp.async` and prove the SASS shows `HMMA`/`OMMA`, not a fallback |
| **Kernel hangs / never returns after launch** | One rung per process, each with its own seconds budget |
| **Is the lane↔element mapping / swizzle / matrix-atom membership right** (`ldmatrix`/`stmatrix` readback) | Double-edge ramp, sentinel prefill, scalar cross-check |
| **smem traffic over the line** — unsure whether to tune swizzle or inspect the feed | Wavefronts × bankconf four quadrants, separating conflict-type from volume-type |
| **The kernel is numerically right but the pipeline does not overlap** | Read the emitted `cp.async` groups in SASS/PTX and prove the wait placement before adding stages |
| **A spelling is refused** | Read the refusal against the CUDA version's headers, and record the toolchain version in `Measured on` |
