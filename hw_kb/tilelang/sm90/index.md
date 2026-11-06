# SM90 (Hopper) interface index — TileLang mode

The sm90 tree of the TileLang mode (a Python `kernel_tl.py` student). Demand-driven: a recipe is written when the
student gets stuck on that interface, following `../TEMPLATE.md` (examples are TileLang, not `cute.*`), and is
marked ✅ on **only** one of the two kinds of evidence the other trees accept:

- **a GPU probe returning `PROBE PASS`** — the default kind;
- **a shipped KEEP whose own NCU census shows the contract axis moved** — for knobs that are numerically
  invisible (cache-policy bits, descriptor modes, drain depth): both spellings compute the same bytes, so a
  `max_diff` probe cannot see the change.

This file is not a plan; it is the ledger of what has already been earned. **Every recipe row is empty on
purpose.** A TileLang entry may not be produced by translating a `cute.*` entry: TileLang chooses `num_stages`
and the MMA path and infers part of the layout itself, so a DSL reading is a hypothesis about TileLang at best.

**Not finding something here does not mean it cannot be done** — for interface details, TileLang's own
`examples/` and `docs/`, the PTX ISA, and your own probes are authoritative; what you win on a probe gets written
in here in the finish round.

## Recipes (skeletons — mechanism documented, readings not taken)

A row here is **not** a certified recipe. These four files exist so that the code a student needs
is written down with the pitfalls already mapped; each carries `[to be measured]` where the
evidence has to go. A row becomes certified only when its `## Measured on` block is filled with a
`PROBE PASS` job id or a moved NCU census axis.

| Interface | Entry | Evidence |
|---|---|---|
| `T.Pipelined(num_stages=)` — stage budget, what it does and does not stage | `software_pipeline_stage_budget.md` | `[to be measured]` |
| `T.copy` across scopes; `T.annotate_layout` only pins the buffers it names | `tiled_copy_scope_and_swizzle.md` | `[to be measured]` |
| `T.gemm` operand orientation and `GemmWarpPolicy` | `gemm_operand_layout_and_policy.md` | `[to be measured]` |
| Writing a GPU probe: one rung per process, verdicts to a file | `gpu_probe_ladder.md` | `[to be measured]` |

## Symptoms that will demand a recipe (no entry yet — probe it, then write it)

| What you see / what you are about to do | The reading that turns it into an entry |
|---|---|
| **Kernel hangs / never returns after launch** — `synchronize` never returns, the probe runs out the job's clock and is killed | One rung per process, each process with its own budget in seconds; a bare timed run only burns the whole clock and tells you nothing |
| **Writing a multi-contract GPU probe**; or a round comes back all FAIL with only one fact; or "no verdict" with the reason already scrolled out of the log tail | One contract per rung, self-contained digest per rung, verdicts and facts on separate channels |
| **Is the lane↔element mapping / swizzle / matrix-atom membership right** (`ldmatrix` / `stmatrix` readback, TMA box and smem face) | Double-edge ramp, a fine + a coarse pattern, sentinel prefill, scalar cross-check |
| **smem traffic over the line** (`smem_wavefronts_st/ld` above the reference) — unsure whether to tune the swizzle or inspect the feed path | Wavefronts × bankconf four quadrants, to separate conflict-type from volume-type before touching the swizzle |
| **The kernel computes the right numbers but the pipeline does not overlap** (synchronous-looking SASS, no win from `num_stages`) | Read the emitted pipeline (SASS / MLIR) and prove the stage count and the wait placement, before assuming `T.Pipelined` did what you asked |
| **A layout I annotated is ignored**, or the emitted smem layout differs from the one I asked for | Prove the emitted layout on a probe: `T.annotate_layout` applies only to the buffers it names, and index inference may still pick another |
| **A TileLang API refuses my spelling** (`T.copy` src/dst shape, `T.gemm` operand layout, `num_warps`) | Read the refusal text against the TileLang version's own source, and record the version in the entry's `Measured on` block |
