# TileLang hardware knowledge base (hw_kb/tilelang)

ForgeKernel's **TileLang mode** (loop language `tilelang`: the student writes a Python `kernel_tl.py`
whose body is a `@T.prim_func`-style TileLang kernel) has its own hardware knowledge base.
It is **fully separate** from `hw_kb/cutedsl/` and `hw_kb/cuda/`: a TileLang run reads and writes only this tree
(`src/forgekernel/loop/lang_mode.py`).

The mechanism knowledge is the same hardware on all three sides; the spelling differs. DSL entries teach `cute.*`
calls, CUDA entries teach CUDA C++ / inline PTX, and these teach the TileLang language surface (`T.Kernel`,
`T.copy`, `T.gemm`, `T.Pipelined`, `T.annotate_layout`, ...). Entries of the other two trees are **not** copied
over as they are: their readings were taken on a different spelling, and TileLang decides part of the layout and
pipeline itself, so a reading taken from a `cute.*` kernel says nothing about what TileLang actually emitted.

## Invariants (same as `hw_kb/cutedsl/` and `hw_kb/cuda/`)

- **Hardware mechanisms only, no operator algorithms.** "How to get TileLang to emit a swizzled smem layout" ✅;
  "how this operator's 5 GEMMs are wired" ❌.
- Every entry is self-contained, minimal and runnable, and carries its own sources. Example code comes from
  **verifiable sources** (the TileLang repo's `examples/`, its `docs/`, `tilelang.com`, or this repo's GPU probes).
- **The evidence chain is split in two layers by reader**: the student reads an entry's "Sources", which may cite
  only official docs / public API symbols / this repo's probes; another implementation's file names,
  function names, line numbers, paraphrases of its content and measured footprints all go to
  `docs/kb-provenance/tilelang/<entry name>.md` (for human review, not in the agent read set). The mechanical gate
  is the same as for the other trees: `src/forgekernel/loop/kb_lint.py`.
- **Only GPU-certified material enters**: a probe returns `PROBE PASS`, or (for numerically invisible knobs) a
  KEEP's own NCU census shows the contract axis moved.

## A limit this tree has to state on every entry

TileLang decides part of the schedule itself: `T.Pipelined(..., num_stages=)` selects a software pipeline,
`T.gemm` picks the MMA path, and `T.annotate_layout` is only consulted for the buffers it names. So a reading
in this tree is **a reading of the code TileLang emitted for that source**, not of a hand-written kernel. An entry
that cannot say which TileLang version and which `arch` it was measured on is not an entry.

## Layout

```
hw_kb/tilelang/
  TEMPLATE.md        # entry template (example code is TileLang, not cute.*)
  sm90/
    index.md         # SM90 (Hopper) list; the entry point is the "Look up by symptom first" table
    <interface>.md
  sm100/
    index.md         # SM100 (Blackwell) list
    <interface>.md
```

**One tree per architecture.** `kb_arch` comes from `[loop] kb_arch` in `operators/<op>/loop.toml` (or
`forge loop --kb-arch`); the TileLang mode resolves it under `hw_kb/tilelang/<kb_arch>/`, and the tree must exist
and have an `index.md`, otherwise the run is refused at start (exit 14). Opening a tree for a new architecture =
create the directory + write `index.md`.

## Initial contents

Both trees ship with an `index.md` and **nothing else**. That is deliberate: `hw_kb/cutedsl/README.md` says the
tree is demand-driven and must not be pre-filled, and a pre-filled TileLang entry would be a `cute.*` reading
wearing TileLang syntax -- exactly the failure this tree exists to prevent. Entries are written when a run gets
stuck on that interface, and each one records the TileLang version and arch it was measured on.
