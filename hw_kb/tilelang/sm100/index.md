# SM100 (Blackwell) interface index — TileLang mode

The sm100 tree of the TileLang mode (a Python `kernel_tl.py` student). Demand-driven: a recipe is written when the
student gets stuck on that interface, following `../TEMPLATE.md` (examples are TileLang, not `cute.*`), and is
marked ✅ on **only** one of the two kinds of evidence the other trees accept:

- **a GPU probe returning `PROBE PASS`** — the default kind;
- **a shipped KEEP whose own NCU census shows the contract axis moved** — for knobs that are numerically
  invisible (cache-policy bits, descriptor modes, DEPBAR depth): both spellings compute the same bytes, so a
  `max_diff` probe cannot see the change.

This file is not a plan; it is the ledger of what has already been earned. **Every recipe row is empty on
purpose.** A TileLang entry may not be produced by translating a `cute.*` entry: TileLang chooses `num_stages`,
the MMA path (`tcgen05` vs `wgmma`) and part of the layout itself, so a DSL reading is a hypothesis about TileLang
at best.

**Not finding something here does not mean it cannot be done** — for interface details, TileLang's own
`examples/` and `docs/`, the PTX ISA, and your own probes are authoritative; what you win on a probe gets written
in here in the finish round.

## Recipes (none yet)

| Interface | Entry | Evidence |
|---|---|---|
| — | — | — |

## Symptoms that will demand a recipe (no entry yet — probe it, then write it)

| What you see / what you are about to do | The reading that turns it into an entry |
|---|---|
| **Kernel hangs / never returns after launch** — `synchronize` never returns, the probe runs out the job's clock and is killed | One rung per process, each process with its own budget in seconds |
| **Cluster / CTA-pair pipeline hangs on the first slot** (job timeout, not wrong numbers) while `cluster = 1` runs | TileLang's cluster support is expressed on the kernel attributes; a mis-shaped one hangs rather than erroring, so turn it into a reading on a probe before writing the entry |
| **Is the lane↔element mapping / swizzle / matrix-atom membership right** (`tcgen05.ld` readback, TMA box and smem face) | Double-edge ramp, a fine + a coarse pattern, sentinel prefill, scalar cross-check |
| **smem traffic over the line** (`smem_wavefronts_st/ld` above the reference) — unsure whether to tune the swizzle or inspect the feed path | Wavefronts × bankconf four quadrants, to separate conflict-type from volume-type first |
| **Registers spill after I asked for a warp-specialized schedule**, or the `max_nreg` I set is not the one in SASS | `num_warps % 4 == 0` is a hard constraint, and the emitted register assignment must be read off SASS before the entry is written |
| **The kernel computes the right numbers but the pipeline does not overlap** | Read the emitted pipeline (SASS / MLIR) and prove the stage count and the wait placement |
| **`tcgen05` vs `wgmma`: which MMA did TileLang pick, and can I steer it** | Read the SASS opcode family off the emit, then record what the source spelling was that produced it |
| **A layout I annotated is ignored**, or the emitted smem layout differs from the one I asked for | Prove the emitted layout on a probe: `T.annotate_layout` applies only to the buffers it names |
