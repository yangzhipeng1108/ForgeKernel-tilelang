# Writing a TileLang GPU probe on sm120: one rung per process, verdicts off the log tail

> Status: **measured on hardware.** The harness described here ran on an RTX 6000D and returned
> four verdicts. The failures it produced along the way are recorded below because each one is
> a failure mode worth recognising.

## Purpose

Get a verdict out of a TileLang kernel on a real GPU without letting one hanging kernel burn the
whole job clock, and without losing the one fact that says *why* it failed when the log tail
scrolls it away.

## Measured on

- TileLang version: `0.1.15`
- arch: `sm_120a`
- Hardware: NVIDIA RTX 6000D, driver 610.43.02, CUDA 13.3

The four rungs of `scripts/measure_kb_entries.py --arch sm120` all returned `PASS` on this box
(pipeline, copy_swizzle, gemm_policy, probe_harness), exit 0.

- Certification: the harness itself ran and returned verdicts on hardware. Its certification is
  "the harness works", not a performance axis.

### Reading the compiled SASS off a TileLang kernel

`orch_census_run.py --kernel <name>_tl.py` works on a TileLang kernel file, and the mechanism is
worth knowing because it differs from the JIT-dump path the runner was written for:

- TileLang has **no keep-the-cubin switch**, and does not need one — every compile writes
  `<TILELANG_CACHE_DIR>/<version>/cuda-binaries/<sha>/kernel.cubin` as part of normal caching.
  Pointing `TILELANG_CACHE_DIR` at a temp dir (and forcing `TILELANG_DISABLE_CACHE=0`) is the
  entire mechanism. The env has to reach the **child** that compiles, not just the wrapper.
- That cache also holds every *other* kernel the process compiled, so **do not pick by size and
  do not pick by CSV row count.** A torch `randn`/reference launch emits one CSV row per metric
  and can outrow the kernel under test. The runner takes the CSV kernel names in order and
  requires one to exist as a symbol in the dumped cubins (`matched ... by symbol`), falling back
  to largest-by-instruction-count only when none does.

Verified on an RTX 6000D with a 2048³ f16 matmul where the CSV carried four rows for torch's
`distribution_elementwise_...` and two for `k_kernel`: the runner printed
`matched kernel.cubin by symbol k_kernel` and censused 616 instructions.

### The three instruments, and the one reading each one lies about

All three of ForgeKernel's measurement paths work on a TileLang kernel with **no change to the
kernel or to the card** — verified on the same box:

| instrument | how it reaches a TileLang kernel | the reading it can get wrong |
|---|---|---|
| `eval/timing.py` | `load_entry()` imports the `_tl.py` and times the callable it returns | `host_bound_ratio` on a **small** kernel |
| `eval/precision.py` | `upcast_call` wraps the callable in a `TorchFunctionMode` and re-runs it in fp64 | nothing found; it reported `pass` at `e_cand=6.5e-7` against a `limit` of `8.1e-6` |
| `orch_census_run.py` | reads the cubin the compile cache just wrote | which cubin (see above) |

Measured, 4096³ f16 GEMM, `num_stages=3`:

```
device_median_ms      1.0437     host_enqueue_ms   0.0057
host_bound_ratio      0.0054     (warn at 0.5 -> safe)
graph_device_median   1.0402     (0.997x the device reading: a real capture, not an empty graph)
TFLOP/s               131.6      max_diff          2.0e-3
```

Measured again on a 512³ GEMM — the same code, a 175x smaller kernel:

```
device_median_ms      0.00594    host_enqueue_ms   0.00590
host_bound_ratio      0.9938     <-- over the 0.5 warn line
graph_device_median   0.00623    (1.049x the device reading: SLOWER than not graphing)
```

**That is not a TileLang defect, it is a size threshold, and it is the thing to know before
pricing a small kernel.** The Python → FFI → launch path costs about 6 µs. When the kernel takes
1 ms that is 0.5 % and the device window is valid; when it takes 6 µs the host and the device are
the same size and `host_bound_ratio` ~ 1. CUDA-graph replay does **not** rescue it here: the
capture works and is honest, but at 512³ the replay costs slightly more than the launch it
replaces, so the graph reading comes back *above* the device reading.

Practical reading: `host_bound_ratio` above 0.5 on a TileLang kernel is a statement about the
kernel's duration, not about the harness. Read `graph_status`; if it says `1.0x` or higher, the
kernel is too small for the graph to help and the honest number is the device window plus a note.

### Stamping a TileLang kernel for the timeline

The timeline probe supports TileLang; the marker convention did not change (`def _seg(site,
row=0): return None` + `_seg("<role>.<seg>.<event>", i)`). Two findings make it work:

- `T.import_source(text)`, called **inside** the `T.Kernel` block, pastes `text` into the
  generated `.cu` **verbatim** — no `%`-format step, so a literal `%clock64` survives a template
  that also uses `%` for modulo;
- a **module-level** `@T.macro` named `_seg` is callable from a `prim_func` body, and its
  `T.call_extern` becomes a plain C call to whatever the imported source defined. A macro
  defined *inside* `@T.prim_func` is not visible at the call site (`NameError`).

The slab address cannot be baked into the source (it is allocated when the module is imported,
after stamping), so it is passed as the stamp call's first argument. A source that named it
textually fails in nvcc with `identifier "_TL_BASE" is undefined`.

Verified: `TIMELINE-OK mode=all`, `numeric=identical`, `TIMELINE-SITES 2 0:copy.in.begin
1:copy.in.end`.

## When needed (trigger)

- **The kernel hangs**: `torch.cuda.synchronize()` never returns and the job is killed by its own
  clock. One rung per process at least says *which* rung hangs.
- **A multi-part probe comes back all FAIL with a single fact**, or "no verdict" with the reason
  scrolled out of the tail.
- **Before writing any probe that has more than one thing to prove.**

## Minimal runnable example

```python
"""One rung, one process. `python probe.py <rung> <out.json>`; the parent gives each child its budget."""
import json
import sys

import torch
import tilelang
import tilelang.language as T

M, N, BLK, THREADS = 4096, 4096, 128, 128


@T.prim_func
def copy_scale(x: T.Tensor((M, N), "float32"), y: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(M, BLK), threads=THREADS) as bx:
        _tile_in = T.alloc_fragment((BLK, N), "float32")
        _tile_out = T.alloc_fragment((BLK, N), "float32")
        T.copy(x[bx * BLK, 0], _tile_in)
        T.copy(_tile_in, _tile_out)
        T.copy(_tile_out, y[bx * BLK, 0])


def rung_copy():
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    kernel = tilelang.compile(copy_scale, out_idx=[1],
                              target={"kind": "cuda", "arch": "sm_120a"})
    y = kernel(x)                    # out_idx=[1] -> out is allocated and RETURNED
    torch.cuda.synchronize()          # the call that hangs when the kernel does
    return {"verdict": "PASS" if float((x - y).abs().max()) == 0.0 else "FAIL",
            "facts": {"max_diff": float((x - y).abs().max())}}


if __name__ == "__main__":
    out_path = sys.argv[2]
    try:
        result = rung_copy()
    except Exception as exc:                       # noqa: BLE001
        result = {"verdict": "FAIL", "facts": {"error": f"{type(exc).__name__}: {exc}"}}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
```

The parent spawns one child per rung with its own `subprocess.run(timeout=...)`, and treats a
timeout as `HANG` for that rung only.

## Four ways this harness failed on its first real run

Every one of these cost a cycle, and every one is a shape a probe can take:

1. **The body was written to the caller's cwd.** `outdir = TMPDIR or TEMP or "."` — neither
   environment variable is set on this Linux box, so the file landed in the repo and the child,
   started elsewhere, could not find it. Use `tempfile.mkdtemp()`.
2. **A crashed child left nothing behind.** Without capturing stdout/stderr, "the probe crashed"
   and "the probe was never run" were indistinguishable. Write each rung's output to its own log
   and put the tail into the FAIL fact.
3. **`out_idx=[1]` was treated as "takes two arguments".** It means the kernel *allocates and
   returns* the second tensor, so it is called with the inputs only. Passing both is an arity
   error: `Expected 2 inputs, got 3 with 2 inputs and 1 outputs`.
4. **A wrong `arch` looked like a wrong kernel.** See the target table in
   `software_pipeline_stage_budget.md`: `sm_120` lowers fine and fails at `nvcc`, so a probe that
   only lowers reports "the pipeline works" for a kernel that cannot be built.

## What TileLang decides for you here (and what it does not)

**Decided by TileLang** (and therefore part of what a probe has to pin): whether the source
compiles for the target and the exact refusal text; the emitted schedule — so a performance
probe measures the schedule TileLang chose for this source, which is why `Measured on` carries
the version and the arch.

**Decided by you:** the target (an explicit dict, `{"kind": "cuda", "arch": "sm_120a"}`); one
claim per rung; the verdict/facts split, written to a file rather than stdout.

## Gotchas

- **`@tilelang.jit` probes the GPU at definition time.** A module that decorates with it fails to
  import on a box with no GPU, before any probe logic runs. Use `@T.prim_func` plus an explicit
  `tilelang.compile(...)` inside the rung.
- **`compile_kernel_source` never invokes `nvcc`.** It can succeed for a target that cannot be
  built — it is a lower-only reading, not a compile check.
- **`target` must be an explicit dict.** A bare `target="cuda"` leaves the arch unpinned and the
  emitted code is not the one you are certifying.
- **A timeout is a verdict, not an error.** Record `HANG` for that rung and continue.
- **Facts computed after a hang do not exist.** Put each fact inside the rung that produces it.

## Sources

1. TileLang 0.1.15, installed package: `tilelang/__init__.py` (`compile`, `lower`),
   `tilelang/tools/compile_only.py` (`compile_kernel_source`),
   `tilelang/language/kernel.py` (`T.Kernel`).
2. TileLang docs in `tile-ai/tilelang`; <https://tilelang.com>.
3. This repo's harness: `scripts/measure_kb_entries.py`, and the sm120 entries it produced.
