# MXFP4 / MXFP8 on sm120 in TileLang 0.1.15: source-build state

> Status: **PARTIAL.** Building TileLang 0.1.15 *from source* on this box works once the exact
> upstream TVM pair is pinned; MXFP4 then compiles and launches, and the multi-stage crash is
> fixed — but the per-block scale **values** are still misplaced (**cos ≈ 0.59**, not 1.0), so
> MXFP4 is not yet numerically correct in TileLang. MXFP8 is wired at the config/PTX/enum level
> but not implemented (needs a new 8-bit `m16n8k32` ldmatrix layout). Stopped here per user
> request; this note records what is proven, what is patched, and the exact remaining defect.
>
> The headline fact for sizing: **the CuTeDSL MX path is the correct, usable MXFP4/MXFP8 on this
> box** (cos 1.0, MXFP4 509.6 TF/s / MXFP8 256.5 TF/s at 4096³ — see
> `precision_roofline_all_stacks.md`). TileLang's MX is a *source-level extension in progress*,
> not a working path.

## Why the source build matters (and where it was blocked)

The shipped wheel plus the `nvfp4_blockscaled_is_broken_in_tilelang.md` 3-line Python patch gives
working NVFP4 without any compiler rebuild, because that fix is pure Python. MX is different:
the SM120 block-scale instruction selection is gated in **C++** —
`src/cuda/codegen/codegen_cuda.cc` hard-`ICHECK`s `supported_mxf4nvf4_4x_ue4m3` at the
`ptx_mma_block_scale` handler, and `src/tl_templates/cuda/instruction/mma_block_scale.h` has
`static_assert` pinning `kMxf4nvf4 / 4X / ue4m3`. Adding MX therefore **requires rebuilding
`libtilelang.so`**, which requires a working source build.

The source build was blocked for a long time on a TVM/tvm-ffi API mismatch. The root cause:

- TileLang v0.1.15's `3rdparty/tvm` submodule is pinned to commit
  **`4211874e9e4b8770fd20ad0d98ab80afd1df5028`** of the **`TileLang/tvm`** repo (not apache/tvm).
- That tvm pins its own `3rdparty/tvm-ffi` to
  **`3c35034fd1026011736e19a4e0e1ed0f22058c42`** (apache/tvm-ffi, the 0.1.12 line).
- The 0.1.15 C++ sources use the **old** tvm-ffi C++ API: `Enum::Get`, `EnumObj::_value`,
  `GetEnumEntriesColumn`. Error signatures when mismatched:
  `no matching function for call to tvm::ffi::Enum::Enum(...)`,
  `SwizzleModeObj has no member _value`,
  `'Get' is not a member of 'tvm::ffi::EnumObj'`.
- **Every** `apache/tvm-ffi` *tag* (v0.1.6 … v0.1.11) lacks those symbols — the old API exists
  only on that specific SHA. So you cannot resolve the pair by tag or branch name; you must
  check out the exact commit the submodule pointer names. `tile-ai/tvm@main` and
  `@tilelang_main` are both `0.25.dev0` and carry the *new* ffi → they do **not** work.

Fix: clone `TileLang/tvm`, `git checkout 4211874e9…`, then in its `3rdparty/tvm-ffi` checkout
`3c35034f…` (plus that ffi's `dlpack 84d107bf…` and `libbacktrace 793921876…`). With that tree,
`pip install --no-build-isolation .` builds `tvm_runtime` + `tvm_compiler` and the wheel
(`tilelang 0.1.15`) cleanly (352-core host, `export PATH=/usr/local/cuda/bin:$PATH` for nvcc 13.3).
An unmodified 0.1.15 then runs an fp16 `T.gemm` at cos ≈ 1.0 — the build chain is proven.

## What is patched (all edits rebuild clean; NVFP4 example still verifies)

Three layers, mirroring how NVFP4 is wired:

1. **Python config table** — `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py`,
   `_SUPPORTED_BLOCK_SCALE_MMA_CONFIGS`, add:
   - `("mxf4nvf4", 2, "ue8m0")`: `mma_prefix=m16n8k64`, `atom_k=64`, `sf_vec_size=32`, e2m1/e2m1
   - `("mxf8f6f4", 1, "ue8m0")`: `mma_prefix=m16n8k32`, `atom_k=32`, `sf_vec_size=32`, e4m3/e4m3
2. **Python inference** — `tilelang/cuda/op/gemm/gemm_mma_sm120.py`, `_make_mma_emitter` calls a new
   `_infer_block_scale_kind()` that maps `sf_a_granularity_k` + operand dtype to `(kind,
   scale_vec_size, scale_type)`:
   `gran 16 → NVFP4 (4X ue4m3)`; `gran 32 + e4m3 → MXFP8 (1X)`; `gran 32 + e2m1 → MXFP4 (2X)`.
   SF buffers are packed `uint32` words, so the dtype signal is the **operand** dtype, not the SF
   dtype.
3. **C++ / PTX / enum** — `mma_block_scale.h`: add `kMxf8f6f4`, `kUE8M0`, and the two PTX bodies
   `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0`
   (SFA/SFB regs are **16-bit** here) and
   `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`
   (**8-bit** SF). `static_assert` replaced with `kSupported` + `if constexpr` dispatch.
   `codegen_cuda.cc`: relax the `ICHECK` to the three configs and map `kind`/`scale_type` strings
   to the enum names.

CUTLASS is the authority for the op names: MXFP4 is `kind::mxf4nvf4 … scale_vec::2X … ue8m0`
(`mma_sm120.hpp` ~3136), MXFP8 is a **different kind** `kind::mxf8f6f4 … scale_vec::1X …
m16n8k32 … e4m3.e4m3 … ue8m0` (~2763). Both are auto-enabled under the `sm_120a` compile
(`cute/arch/config.hpp`: `CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED`,
`CUTE_ARCH_MXF8F6F4_MMA_ENABLED`).

## The multi-stage crash — fixed

Symptom: MXFP4 with `K > block_K` and `num_stages ≥ 2` faults an **invalid `__shared__` read**
(compute-sanitizer: line 176, address `0x11800`, past the whole `buf_dyn_shmem`). `num_stages=1`
and single-block `K = block_K` both run.

Root cause: in `_load_kblock` the scale K-word index was computed as
`word_k = k_block * micro_size_k // packed_word_k`. For MXFP4 (`micro=64`, `packed=128`) this is
`k_block/2`, but the *multiply-then-divide* form lets the TIR simplifier fold the subsequent
`word_k * tile_rows(128)` into a **2× stride** (effectively `k_block * 128`), so the refilled
K-blocks (k_block ≥ 2) indexed past the half-sized (W=2) scale tile. For NVFP4 (`micro=packed=64`)
`word_k = k_block` exactly and the tile is full-size, so it never showed.

Fix: compute the word index with exact integer division —
`word_k = k_block // (packed_word_k // micro_size_k)` (group = 1 for NVFP4 → unchanged;
group = 2 for MXFP4 → `k_block // 2`). With this, K=512 / stages=2 runs without faulting.

## The remaining defect — MXFP4 scale values (cos ≈ 0.59)

The decisive gate: **set every scale to 1.0** (`ue8m0 = 0x7F`) → **cos = 0.99999985**. That proves
the e2m1 operands, the `ldmatrix` layout, the `scale_vec::2X` PTX, and the accumulator are all
correct; only the **per-block scale values** land in the wrong place (cos 0.591 with real
scales). This is the MX analogue of the NVFP4 "scale operand" bug, but the 2X packing differs
from 4X and the half-selector is not yet right.

CUTLASS ground truth for the packing:

- `MMA_NSF = TiledMma::K / SFVecSize` = 64/32 = **2** SFs per k64 atom for 2X (4 for 4X at VS=16).
- One `uint32` scale word holds 4 `ue8m0` = two k64 atoms' worth: even atom → low 16 bits,
  odd atom → high 16 bits.
- `Sm1xxBlockScaledBasicChunk<32>::SfKMajorAtom =
  Layout<Shape<Shape<_32,_4>,Shape<Int<SFVecSize>,_4>>, Stride<Stride<_16,_4>,Stride<_0,_1>>>`
  (`cutlass/detail/sm100_blockscaled_layout.hpp`). The `_4` mode (stride 1) is four contiguous
  K-scales; the `SFVecSize` sub-block is broadcast (stride 0).
- The SM120 2X `fma` uses **constant** `bidA = tidA = bidB = tidB = 0` — the 16-bit SF register
  must already contain the atom's two `ue8m0` **before** the fma. Selection is at register load,
  not via the asm `byte_id/thread_id` immediates.

Two attempts, both recorded as falsified:

- asm `byte_id = (i%2)*2` on the 2X path → cos **0.28** (worse). Reverted.
- codegen reads a `uint16` at a byte offset + matching emitter selectors → cos **0.344** (worse).
  This is the state the machine was left in when it went offline; `revert_bugb.py` restores the
  0.591 baseline (then rebuild).

The 0.591 baseline (whole-`uint32` read, original 4X selectors) is *closer* than either offset
attempt, which points at the **host-side byte↔K mapping / SF copy-partition**, not the register
half-selector. Next step is to derive the lane→(M, K, SF-byte) map from the CUTLASS collective
(`sm120_blockscaled_mma_tma.hpp`, `deduce_smem_layoutSFA`, `Sm1xxBlockScaledConfig<32>`) and
gate on `test_mxfp4.py --m 256 --n 256 --k 256` (single block, stable), not guessing byte ids.

## MXFP8 is not implemented beyond the wiring

The config/PTX/enum are in place, but MXFP8 uses `m16n8k32` with **8-bit** operands. The existing
SM120 block-scale `ldmatrix` helpers (`_warp_ld_a_e2m1` / `_warp_ld_b_e2m1`) are the 4-bit
`k=64` layouts, so MXFP8 needs new `m16n8k32` 8-bit A/B fragment layouts (and, if feeding e2m1
into the f8f6f4 atom, the CUTLASS `fp4_shift` `<<2`; pure e4m3 does not). This is a larger lift
than MXFP4 and was not started.

## What to reuse

- Source-build recipe (the tvm pair) is the valuable, transferable result: pin
  `TileLang/tvm@4211874e9…` and its `tvm-ffi@3c35034f…`. Any 0.1.15 C++ edit needs exactly this.
- The `num_stages` / single-block bisection isolates SF-tile geometry bugs from MMA bugs fast.
- **Gate for scale packing:** set all scales to 1.0 to split "operand/MMA correct" from "scale
  placement correct" (cos jumps to 1.0 iff only placement is wrong). For finding placement bugs,
  use powers-of-two per scale column (an all-ones gate is blind, exactly as the NVFP4 note warns).
- NVFP4 remains the validated TileLang 4-bit path; do not regress it (`examples/gemm_sm120/
  sm120_nvfp4_blockscaled_gemm.py --verify` must keep passing after any of these edits).
