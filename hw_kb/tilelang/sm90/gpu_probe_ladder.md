# Writing a TileLang GPU probe: one rung per process, verdicts off the log tail

> Status: **skeleton — the shape is documented, the readings are not taken yet.**
> `[to be measured]` marks what this card must produce on a real box. The mechanism below is
> the reason the main variant of this entry exists in the other two trees; here it is restated
> for a TileLang source, not copied from them.

## Purpose

Get a verdict out of a TileLang kernel on a real GPU without letting one hanging kernel burn the
whole job clock, and without losing the one fact that says *why* it failed when the log tail
scrolls it away.

## Measured on

- TileLang version: `0.1.15`
- arch: `[to be measured]` (this file is under `sm90/`)
- Certification: `[to be measured]` — a `PROBE PASS` job id.

This entry is about the probe **harness**, so its certification is "the harness itself ran and
returned a verdict", not a performance axis.

## When needed (trigger)

- **The kernel hangs**: `torch.cuda.synchronize()` never returns and the job is killed by its
  own clock. A probe that runs one rung in one process, each with its own budget, at least
  tells you *which* rung hangs.
- **A multi-part probe comes back all FAIL with a single fact**, or "no verdict" with the reason
  already scrolled out of the log tail.
- **Before writing any probe that has more than one thing to prove.**

## Minimal runnable example

```python
"""One rung, one process. `python probe.py <rung>`; the parent gives each child its own timeout."""
import json
import subprocess
import sys

import torch
import tilelang
import tilelang.language as T

M, N = 1024, 1024
BLK = 128


@T.prim_func
def copy_scale(x: T.Tensor((M, N), "float32"), y: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(M, BLK), threads=128) as bx:
        _tile_in = T.alloc_fragment((BLK, N), "float32")
        _tile_out = T.alloc_fragment((BLK, N), "float32")
        T.copy(x[bx * BLK, 0], _tile_in)
        T.copy(_tile_in, _tile_out)
        T.copy(_tile_out, y[bx * BLK, 0])


def rung_copy():
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    kernel = tilelang.compile(copy_scale, out_idx=[1],
                              target={"kind": "cuda", "arch": "sm_90a"})
    kernel(x, y)
    torch.cuda.synchronize()          # the call that hangs when the kernel does
    return {"verdict": "PASS", "facts": {"max_diff": float((x - y).abs().max())}}


RUNGS = {"copy": rung_copy}


def main():
    rung = sys.argv[1]
    try:
        out = RUNGS[rung]()
    except Exception as exc:                       # noqa: BLE001 — a rung must not take the run down
        out = {"verdict": "FAIL", "facts": {"error": f"{type(exc).__name__}: {exc}"}}
    # the verdict goes to a file the parent reads; the log can scroll, the file cannot
    with open(f"rung-{rung}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
```

The parent spawns `python probe.py <rung>` per rung with its own `subprocess.run(timeout=...)`,
and treats a timeout as the verdict `HANG` for that rung only.

## What TileLang decides for you here (and what it does not)

**Decided by TileLang (and therefore part of what a probe has to pin):**

- whether the source compiles at all for the target, and the exact refusal text when it does not;
- the emitted schedule — so a probe that measures performance is measuring the schedule TileLang
  chose for this source, which is why the `Measured on` block carries the version.

**Decided by you:**

- the target. It has to be an explicit dict: `{"kind": "cuda", "arch": "sm_90a"}`. A bare
  `target="cuda"` leaves the arch unpinned and the emitted code is not the one you are certifying.
- one claim per rung. Two claims in one rung means one hang hides both.
- the verdict/facts split, and writing it to a file rather than to stdout.

## Gotchas

- **`@tilelang.jit` probes the GPU at definition time.** A module that decorates its kernel with
  `@tilelang.jit` fails to import on a box with no GPU, before any probe logic runs. Use
  `@T.prim_func` plus an explicit `tilelang.compile(...)` inside the rung function.
- **Compiling without a GPU is possible, executing is not.** `tilelang.lower(...,
  enable_device_compile=False)` and `tilelang.tools.compile_only.compile_kernel_source(...)` both
  work on a CPU-only box — that is how a compile-only rung is certified. A rung that
  `synchronize()`s is a GPU rung and needs a GPU.
- **A timeout is a verdict, not an error.** If the parent raises on timeout, the whole probe
  reports nothing; record `HANG` for that rung and continue.
- **Facts computed after a hang do not exist.** Put each fact inside the rung that produces it;
  do not accumulate in module-level state a killed child will never flush.

## Sources

1. TileLang 0.1.15, installed package: `tilelang/__init__.py` (`compile`, `lower`),
   `tilelang/language/kernel.py` (`T.Kernel` — "operates at the block level").
2. TileLang docs: `docs/` in the `tile-ai/tilelang` repository; <https://tilelang.com>.
3. This repo's own probe entries:
   `hw_kb/tilelang/sm90/software_pipeline_stage_budget.md`,
   `hw_kb/tilelang/sm90/tiled_copy_scope_and_swizzle.md`.
