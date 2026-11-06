# <interface name> (e.g. T.Pipelined stage count / T.copy with swizzle / T.gemm policy)

> One file per hardware interface. Operator-agnostic: mechanism only. TileLang mode: examples are TileLang
> (`@T.prim_func`, `T.Kernel`, `T.copy`, `T.gemm`, ...), not `cute.*`.

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
