# <interface name> (e.g. T.Pipelined stage count / T.copy with swizzle / T.gemm policy)

> One file per hardware interface. Operator-agnostic: mechanism only. TileLang mode: examples are TileLang
> (`@T.prim_func`, `T.Kernel`, `T.copy`, `T.gemm`, ...), not `cute.*`.

> State: WORKS | ABSENT | BROKEN | PARTIAL — one sentence saying what a reader should do.

The state is the first thing a reader acts on, so it comes first and it is machine-read
(`src/forgekernel/loop/kb_state.py`). It is **not** a quality ranking; each value is a
different instruction:

- `WORKS` — measured on hardware, use it as written.
- `ABSENT` — the interface is not there on this target (missing, or compiles to nothing).
  Route around it; do not build a schedule on it.
- `BROKEN` — it exists and produces wrong results. Do not use it as shipped: record the fix
  in the body, or wait for a version bump.
- `PARTIAL` — part of the claimed surface works and part does not; the body must say which.
  This exists because a run can legitimately ship an entry mid-way, and filing that under
  `WORKS` is the one outcome a reader must never hit.

`BROKEN` and `PARTIAL` need a concrete explanation (what is wrong **and** what to do
instead); `WORKS` and `ABSENT` do not. The same mark then goes on the entry's row in
`index.md` (✅ / ⛔ / 🐛 / ⚠️); the lint checks the two agree, so an index can never tell a
reader to use something the entry calls broken.

## Purpose
What hardware-level problem this interface solves (one sentence).

## Measured on
- TileLang version: `<x.y.z>` (`python -c "import tilelang; print(tilelang.__version__)"`)
- arch: `<sm90 | sm100>`
- Certification: `PROBE PASS` (job id) **or** a NCU census axis that moved (before → after).

Without this block the entry is not evidence: TileLang chooses part of the schedule itself, so a reading is a
reading of the code TileLang emitted for this source on this version and this arch.

## When needed (trigger)
Which student need / symptom calls for it (profiler symptom / SASS footprint / dataflow shape / compiler error text).

## Contract
What this mechanism promises and what it demands, stated so another stage could be wired to it
without reading the example. Four lines, each optional but each one you omit is a seam a
reader has to rediscover by hand:

```
provides:  <what lands where when this is done>   e.g. accumulator f32[128,128] in registers, done-barrier
requires:  <what must already be true>            e.g. A,B tiles bf16[128,k] in smem, mma-barrier
resources: <what it occupies>                     e.g. 4 warps, 64 KiB smem, 240 regs/thread
sync:      <how it hands off>                     e.g. mbarrier arrive/wait, ring depth >= 2
```

This is a reading, not a design: write down what the measured build actually does, and mark a
field `unknown` rather than guessing. It is what makes an entry reusable as a *component*: a
later operator can ask "can I feed this?" instead of reading the source of the thing it is
joining. Keep it operator-agnostic like the rest of the entry — say `accumulator`, not
"the attention P tile".

## Minimal runnable example
```python
# Minimal, self-contained. From a verifiable source; nothing made up.
# Show only "how to use this interface"; do not embed any specific operator's algorithm.
```

## What TileLang decides for you here (and what it does not)
Which parts of this mechanism come from the source and which are chosen by the compiler (`num_stages`, MMA path,
layout inference). This is the section that keeps a DSL reading from being mistaken for a TileLang reading.

## Gotchas
- What error a misuse raises / where it goes silently wrong (`num_warps % 4 == 0` for warp specialization,
  `T.annotate_layout` only applying to named buffers, `prelude` replacing the body with raw C, ...).

## Sources

**The student reads this section; only three kinds of citation are allowed** (a tool's symbols may be cited, a
reference solution's may not):

1. Official documentation links (tilelang.com docs, the `tile-ai/tilelang` repo, NVIDIA docs for the PTX the kernel
   lowers to);
2. Public infrastructure API symbols (TileLang builtins, PTX ISA mnemonics);
3. This repo's own probes and `hw_kb/` entries.
