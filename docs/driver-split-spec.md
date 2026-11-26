# Refactor spec: splitting `loop/driver.py`

`src/forgekernel/loop/driver.py` is 11,200 lines / 689 KB: 327 top-level
functions, 14 classes, 187 module-level constants, and a small public surface
(`loop/__init__.py` re-exports 8 names) over a very large private one —
`tests/test_loop_driver.py` (7,387 lines, 411 tests) imports **142 private names**
directly from `driver`. This is the single largest structural debt in the tree.

This document is the plan for paying it down **safely**, in test-gated steps. It is
deliberately not a one-shot rewrite: every move below must keep
`tests/test_loop_driver.py` green (plus the rest of `tests/`), because that suite is
the only thing standing between a split and a silent behaviour change.

## Why it cannot be span-cut

The file is a God module: sections call each other in both directions. Measured
cross-references (how many of a section's own definitions are referenced from outside
it) show almost no section is a leaf:

| Section (by header) | lines | defs | referenced outside |
|---|---|---|---|
| prompt rendering | 493 | 21 | 15 |
| placeholder ledger | 212 | 11 | 6 |
| machine-aware comparisons | 886 | 39 | 33 |
| card-owned harness | 442 | 2 | 1 |
| rows that buy no eval | 509 | 10 | 9 |
| meta loop | 321 | 14 | 12 |
| reopen budget | 557 | 17 | 6 |
| dossier contract signature | 364 | 12 | 8 |
| phase exhaustion | 735 | 21 | 17 |
| price staleness | 450 | 18 | 8 |
| timeline probe | 3,028 | 70 | 37 |
| out-of-run gap queue | 3,629 | 89 | 23 |

Copying a span into a new module and importing it back creates circular imports
unless the moved code's *dependencies* are also moved or injected. So the order
matters: extract leaves first, then the nodes once their leaves are out.

## The facade contract (what every step must preserve)

1. `forgekernel.loop.driver` must keep exposing **every name the tests import** —
   the 8 public ones and all 142 private ones — at the same dotted path. The
   extraction keeps that true by re-importing the moved names back:
   `from forgekernel.loop._xxx import name1, name2, _private` at the top of
   `driver.py`, or `from forgekernel.loop import _xxx` if a module-qualified call site
   is acceptable.
2. `loop/__init__.py` keeps its imports unchanged (it reads from `driver`).
3. No move changes behaviour: it is `git mv` of definitions + `from` re-export, with
   `tests/test_loop_driver.py` run after each step. A red suite means revert that step.
4. Module-level constants move with the functions that use them (187 of them); a
   constant used by two sections moves to the first-extracted one and the other
   imports it.

## Ordered steps (each a separate commit, each gated by the 411 tests)

The order is leaf-ish → central. Suggested target modules under `loop/`:

> **Progress:** steps 2, 3, 4 are DONE and gated (no new failures vs baseline):
> `loop/harness_section.py`, `loop/placeholder.py`, `loop/pricing.py` (dossier + price).
> `driver.py` is down from 327 to 287 top-level defs. Steps 1 and 5–9 remain.

1. **`loop/prompt.py`** ← "prompt rendering" (15/21 referenced outside; the render
   functions like `render_prompt`, `_student_prompt`, `_meta_prompt`). Largest
   inbound ref count, so do it once the two below prove the pattern — or do it first
   and accept a big re-export block. **Start here only if willing to carry ~15
   re-exports.**
2. ✅ **`loop/harness_section.py`** ← "card-owned harness" (`_harness_hygiene` +
   `_harness_edit_revert`). DONE. `run_inner_iteration` stays in `driver` (it is the
   orchestrator, not a leaf).
3. ✅ **`loop/placeholder.py`** ← "placeholder ledger" (8 defs + 6 regexes). DONE.
   Self-contained; no driver import.
4. ✅ **`loop/pricing.py`** ← "dossier contract signature" + "price staleness" (30
   defs + 30 constants). DONE. **Not** a leaf: it calls 11 driver helpers
   (`_read_milestone`, `_write_milestone`, `_on_cluster`, `_evidence_dir`,
   `_bank_evidence`, `_seed_pricing_toolkits`, `_fmt_pct`, `_lever_spec_refused`,
   `_persist_lever_table`, `_table_noise_pct`, `archive_is_whole`) — these go through
   thin lazy shims (imported at call time) to keep the graph acyclic; `BackendError`
   is imported at module load (driver defines it before it imports pricing). Also needs
   `JOB_END_KILLED`/`job_abend` (loop.parse) and `meta_shapes`.
5. **`loop/reopen.py`** ← "reopen budget" + "phase exhaustion" (6/17 and 17/21; phase
   exhaustion depends on reopen, extract together).
6. **`loop/comparisons.py`** ← "machine-aware comparisons" (33/39 — almost entirely
   inward-referenced, extract late).
7. **`loop/timeline.py`** ← "timeline probe" (37/70 — 3,028 lines; the biggest single
   win once its leaves are out).
8. **`loop/gap_queue.py`** ← "out-of-run gap queue" (23/89 — 3,629 lines; extract
   last, it sits on top of most of the above).
9. What remains in `driver.py` after 1–8 is the orchestration core
   (`LoopContext`, `run_inner_loop`, `run_meta_step`, `run_two_layer`,
   `run_kb_certify`, `MetaResult`, `KbCertifyResult`) — the genuinely driver-shaped
   code. Target: `driver.py` under ~2,000 lines.

## How to do one step mechanically (avoid transcription bugs)

Do not hand-retype. For a step extracting lines `[S..E]` whose definitions are
`defs`:

1. Cut `[S..E]` verbatim (bodies + the constants it uses) into `loop/<mod>.py`,
   with the same `from __future__ import annotations` header and the module-level
   imports the section needs (grep the section for names it references that are
   imported at `driver` top).
2. In `driver.py`, replace the cut block with
   `from forgekernel.loop.<mod> import <defs and constants>`.
3. `PYTHONPATH=src pytest -q tests/test_loop_driver.py` — 411 must pass. Then the full
   `tests/` run. Only then commit.

## Facade contract enforced by test

The facade contract above is now **enforced** by `tests/test_loop_facade.py`:

- it AST-scans `tests/` for every `from forgekernel.loop.driver import ...` and
  freezes the set (**172 names** across 285 import sites) in `FROZEN_DRIVER_IMPORTS`;
- `test_driver_facade_superset_of_imports` fails if any imported name vanishes from
  `driver` (the step-1..4 re-exports keep this true);
- `test_frozen_set_matches_live_imports` fails if the import surface drifts, so a
  split can never silently drop a caller — the change becomes an explicit diff;
- each extracted module must import on its own (no import-time cycle), and every
  `loop.__all__` name must resolve.

So a future step cannot regress the facade unnoticed. Update the frozen set as part
of any intentional rename/removal.

## Status: split stops after step 4 (no clean leaf remains)

Steps 2–4 are done and gated. Steps 5–9 were investigated and **stopped here
deliberately**, because the remaining sections are not leaves:

| section | lines | top-level defs | defs referenced outside |
|---|---|---|---|
| prompt rendering | 492 | 23 | 15 |
| machine-aware comparisons | 885 | 53 | 36 |
| noeval rows | 508 | 14 | 10 (incl. `run_inner_loop`) |
| meta loop | 320 | 24 | 13 |
| reopen budget + KB certify/harvest | ~560 | ~30 | dense |
| phase exhaustion + lever pricing | 734 | 17 | **`pricing.py` shims point back here** |
| timeline + meta engine (incl. `run_meta_step`) | 3028 | 70 | densest |
| out-of-run gap queue | 3629 | — | spine |

`placeholder`/`harness_section`/`pricing` were the real leaves (done). Everything
left is the load-bearing spine — `run_inner_loop`, `run_meta_step`, `run_kb_harvest`,
`_run_lever_pricing` — mutually referencing, and `phase_exhaustion` is exactly where
the `pricing.py` lazy shims (`_lever_spec_refused`, `_persist_lever_table`,
`_table_noise_pct`, `_fmt_pct`) point back to, so extracting it would **re-couple**
`pricing.py`. Forcing any of these would be the sprawling, risky change this spec
warns against. The God-module *risk* (silent breakage on refactor) is what the facade
test above bounds; the line count is left as one cohesive engine on purpose.

If more splitting is ever wanted, the safe next target is **`loop/prompt.py`** (step
1: 15/23 outward refs, a mechanical re-export block) — it does not touch the pricing
shims. The 4000–7600 "timeline" block is really the meta-phase engine and should be
split by concern (milestone/resume/reopen/pacing), not by line span.

## Acceptance

- Every step: `tests/test_loop_driver.py` green and full `tests/` no worse than the
  known baseline (see below).
- Facade: `tests/test_loop_facade.py` green — all imported names still resolve from
  `forgekernel.loop.driver`, frozen set matches live imports, extracted modules import
  cleanly, `loop.__all__` resolves.

## Known pre-existing test failures (not caused by this refactor)

The baseline on a CPU-only box already fails ~10–15 tests that need a GPU or network
(`test_cuda_mode.py`, `test_leaderboard.py` lock/torch cases, `test_kb_lint.py`,
`test_harness_guard.py`, and a few `test_loop_driver.py`/`test_lever_pricing_run.py`
cases). A split must not *increase* that count; compare against the pre-change run.
