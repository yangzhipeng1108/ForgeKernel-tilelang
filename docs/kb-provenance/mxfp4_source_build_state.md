# Provenance: MXFP4 / MXFP8 source build on sm120 (TileLang 0.1.15)

> People-only archive. The exact upstream pins the KB entry
> `hw_kb/tilelang/sm120/mxfp4_source_build_state.md` refers to as "the source-build pins in
> docs/kb-provenance/" live here, with the run's commit narrative. This file is outside
> `hw_kb/`, so it is not in any agent's read set and not subject to the KB lint; the KB entry
> keeps only the mechanism and points here for the reproducibility detail.

## The exact TVM pair (TileLang 0.1.15 needs the *old* ffi API)

The 0.1.15 C++ uses `Enum::Get` / `Enum::_value`, which every published `tvm-ffi` **tag** has
since dropped. A source build therefore must pin the commit pair below, not a tag:

| Repo | Commit | Why this one |
|------|--------|--------------|
| `TileLang/tvm` | `4211874e9` | the TVM revision 0.1.15's C++ was written against; exposes the old `Enum::Get/_value` |
| `3rdparty/tvm-ffi` (inside it) | `3c35034f` | the ffi that still carries the old enum API; a later tag breaks the build |
| `3rdparty/tvm-ffi/3rdparty/dlpack` | `84d107bf` | the dlpack the pinned ffi expects |
| `3rdparty/tvm-ffi/3rdparty/libbacktrace` | `793921876` | the backtrace lib the pinned ffi expects |

Build: in the `TileLang/tvm` tree, check out `4211874e9`, then in its `3rdparty/tvm-ffi`
checkout `3c35034f` (with the `dlpack` / `libbacktrace` pins above), then
`pip install --no-build-isolation .` builds `tvm_runtime` + `tvm_compiler` and the
`tilelang 0.1.15` wheel cleanly on a 352-core host with `export PATH=/usr/local/cuda/bin:$PATH`
(nvcc 13.3).

## The multi-stage crash (fixed)

MXFP4 with `K > block_K` and `num_stages >= 2` faulted an invalid `__shared__` read:
compute-sanitizer reported a read at `0x11800`, past the whole `buf_dyn_shmem`. Root cause was
`word_k = k_block // (packed // micro)` folding the multiply-then-divide into a 2x stride, OOB
only at `W=2`. `num_stages=1` and single-block `K = block_K` both ran, which is how the
bisection isolated it.

## Remaining gap (why the entry is PARTIAL, not WORKS)

With all scales set to 1.0 the kernel is cos 1.0, but with real scales the per-block SF values
are still misplaced (cos 0.59; two byte-id/offset attempts gave 0.28 and 0.344, both reverted).
MXFP8 is wired only at config/PTX/enum level and needs a new 8-bit `m16n8k32` ldmatrix. For MX
throughput numbers use the CuTeDSL rows in `precision_roofline_all_stacks.md`, not TileLang.
