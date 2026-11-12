# NVFP4 blockscaled GEMM on sm120: the instruction exists, the TileLang path does not work

> Status: **hardware path confirmed, TileLang lowering broken**. The `mma.sync` instruction is
> real and selectable on this card; driving it through `T.gemm_blockscaled` produces a
> deterministic wrong answer. This entry records exactly what was tried and what the numbers
> were, so the next person does not repeat the sweep. **No correctness claim is made.**
>
> **Root cause is NOT located.** The first version of this note blamed `_scale_word_offset`;
> that function turned out to be dead code with no callers. The real failing site is the
> word-offset split at `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py:983-990`, narrowed to a single remaining
> hypothesis that was not tested. The trace, the two failed patches, and the compile-cache trap
> are all recorded below 閳?start there, not from the old paragraph.

## The instruction is real

`tilelang/src/tl_templates/cuda/instruction/mma_block_scale.h` guards and emits:

```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
```

with `static_assert(ScaleVecSize == 4)` and a `ue4m3 scale factors` requirement. The guard
is `sm_120a` + CUDA 12.8 or newer. So RTX 6000D is a supported target at the PTX level 閳?this is not "the arch cannot do NVFP4".

CUTLASS ships the collective for it too (`cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp`,
`builders/sm120_blockscaled_mma_builder.inl`).

## The dispatch reaches the sm120 path

`tilelang/src/cuda/op/gemm_blockscaled.cc` selects:

```
bool matches_sm120_mma =
    TargetIsSM120(target) &&
    (IsSharedBuffer(op->a_) || IsFragmentBuffer(op->a_)) &&
    IsSharedBuffer(op->b_) && IsFragmentBuffer(op->c_);
if (!requires_tcgen05 && matches_sm120_mma) return kCudaMMABlockScaled;
```

A kernel with shared A/B and a fragment C therefore compiles to `cuda.mma.blockscaled`, not
an error and not the Blackwell tcgen05 path. **Compilation succeeds.** This matters: a
compile failure would have been a clear "unsupported" signal. Instead the kernel launches.

## Two SF layouts, two different failure modes

`cuda/op/gemm/gemm_mma_sm120.py` lists `supported_sf_layouts = ("rowmajor", "blockscaled_chunk_kmajor")`.

| `sf_layout` | what happens |
|---|---|
| `rowmajor` (default) | **compiles, then `CUDA_ERROR_MISALIGNED_ADDRESS` (716) at launch** |
| `blockscaled_chunk_kmajor` | **compiles, runs, returns a wrong answer** |

Scales must also be a packed word covering 64 K elements, so `sf_granularity_k=32` is rejected
outright (`mxf4nvf4 expects packed scale words covering 64 K elements`). Only `16` is accepted.

> **Measurement hazard:** a `MISALIGNED_ADDRESS` makes the CUDA context sticky. Every later
> `torch.zeros` in the *same process* fails with the same error. A sweep that tries the
> failing variants first and the good one last will report the good one as failing too. Each
> variant has to run in its own process. This cost one full round of false negatives.

## The wrong answer is deterministic, not noise

With `sf_layout="blockscaled_chunk_kmajor"` and all scale factors set to 1.0, random fp4
`A (512,1024)` and `B (512,1024)`:

| tile (BM,BN,BK) | cos vs fp32 reference | output max | reference max |
|---|---|---|---|
| 128,128,128 | 0.4995 | 1.8e-4 | 94 |
| 128,128,64 | 0.5000 | 1.8e-4 | 94 |
| 64,64,64 | 0.5000 | 1.8e-4 | 94 |
| 128,128,256 | 0.5000 | 1.8e-4 | 94 |

The output magnitude is `1.8e-4` **independent of tile shape, K, and the data** 閳?the kernel
is not computing a small version of the right thing, it is reading the wrong memory.

A sharper probe settles it. With every A/B nibble set to the fp4 code for `1.0`, every scale
factor set to `2.0`, and `K = 256`, the correct output is `(1璺?)璺?1璺?)璺?56 = 1024` in every
element. The kernel returns

```
0.0009765625  ==  2**-10
```

uniformly. `2閳﹀鍏変紗` is not a plausible partial sum of this problem; it is the value obtained by
reinterpreting a scale-factor byte at the wrong offset. The scale-factor addressing is wrong.

## What the generated code does

Compiled source shows the SF load indexing rows with a stride of 8192 bytes for a
`(512, 64)` fp8 buffer (32768 B total), and hands the result to the MMA as a `uint32*`
built from an `(uint)` cast of a single fp8 byte.

> **Correction (later investigation).** An earlier version of this note blamed
> `_scale_word_offset` in `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py`. That function is **dead code** 閳?> it and its only would-be caller `compact_selector_scale_word_offsets` have **no callers
> anywhere in the tree**, which was confirmed both by a full-tree symbol search and by
> instrumenting it (it never fires). Blaming it was wrong. The real path is traced below.

## The real path, traced (supersedes the earlier root-cause guess)

Facts, each established by running the kernel rather than reading it:

| # | finding | how it was established |
|---|---|---|
| 1 | `_scale_word_offset`(156-167) and `compact_selector_scale_word_offsets`(190) in `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py` have **zero callers** | full-tree grep + an injected `print` that never fires |
| 2 | the executing path is `tilelang/cuda/op/gemm/gemm_mma_sm120.py:167` 閳?`mma_blockscaled_fulltile`(919) 閳?**`_tile_kmajor_scale_word`(563-566)** | `print` injected at `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py:569` fired once compiling the gate |
| 3 | `word_k` is **always 0** at runtime, so the `word_k * tile_rows` term is dead | same trace: `word_k= 0` printed |
| 4 | real geometry at K=4096/BK=128: `tile_m=tile_n=128`, **`k_blocks=2`**, `warp_row_tiles=64`, `warp_col_tiles=32`, `sfa_words=2`, `sfb_words=1` | trace at the call site |
| 5 | the SF **copy into shared memory is correct** | `smem == gmem for SFA tile 0 : True` from a debug buffer the kernel writes back |
| 6 | two candidate patches both fail to change the answer | (a) `// k_blocks` 閳?`// tile_m`/`// tile_n`: still `0.0039`; (b) `_sfa_row_in_atom`/`_sfb_col_in_atom` using `lane = tx % 32` instead of global `tx`: still `0.0039` |
| 7 | remaining suspect: `k_blocks=2` but each MMA block spans 4 scales (`micro_size_k/sf_vec_size = 64/16 = 4`), so the word offset is likely split by **4**, not by `k_blocks` | not yet tested |

The formula itself (`word_k * tile_rows + (idx % 32) * (tile_rows // 32) + idx // 32`) is
byte-identical to the dead `_scale_word_offset`, so "the formula is wrong" was never a useful
statement 閳?the same arithmetic is reached from a different caller. What actually differs is
**how the flattened word offset is split into `SFA_data[...]` coordinates** at
`tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py:983-990`:

```python
SFA_local_buf[g] = SFA_data[
    tuple(SFA_other) + (SFA_base_m + scale_a_word // k_blocks,
                        SFA_base_k + scale_a_word % k_blocks)
]
```

`_scale_word_offset` was deleted into `_tile_kmajor_scale_word`; the split still uses
`k_blocks`, while the offset advances by `tile_rows`. Finding 5 rules out the data being in the
wrong place, so the fault is in this address arithmetic, not in the copy and not in the packer.

Finding 4 also exposes an out-of-range read worth noting: with `tile_m=128`, `k_blocks=2` the
reachable word range is `[0, 256)`, yet enumerating the formula over `tx 閳?[0,256)` reaches
offset **256** for four lanes (tx 190/221/252), i.e. one word past the tile. Patching the
lane reduction (finding 6b) removes that overrun but does **not** fix the answer, so the
overrun is real but is not the primary fault.

**A trap worth recording:** TileLang's compile cache (`~/.tilelang/cache`, 207 MB) silently
serves the old kernel after a source edit, so an instrumented build looks like it never ran.
Set `TILELANG_CACHE_DIR=/tmp/fresh` (or `rm -rf ~/.tilelang/cache`) before believing any
patch result. This cost a round of false "the function is dead" readings.

## What this means for the deployment

The Qwen3.8 NVFP4 MoE does **not** run through this path, and should not be made to. On the
same box, `sglang` serves NVFP4 through CuTeDSL 閳?`kda_kernels/qwen3x_nvfp4_gemm_sm120.py`, ~1970 lines, built on
`CUTLASS blockscaled-layout helpers` and `sm120_utils`, with the scale-factor layout
(`tile_atom_to_shape_SF`) handled explicitly. That is the working route on sm120, and it is
CUTLASS DSL, not TileLang.

So the honest statement is: **sm120 has NVFP4 tensor-core instructions; TileLang exposes an
entry point for them that does not currently produce correct results; the production path on
this hardware is CuTeDSL.** Do not write a TileLang NVFP4 kernel for this card expecting it
to work, and do not claim the arch lacks the capability.

## If someone picks this up

1. Run each variant in its own process; a laid-over context hides real failures.
2. Use the all-ones / scale=2 / `K=256` probe as the first gate. It has a closed-form answer
   (`1024`) and does not depend on the fp4 packer being right. Add a K=4096 variant too: the
   small shape gives `k_blocks=1`, where `% k_blocks` is identically 0 and the signal degenerates
   to all-zeros, which hides everything.
3. Diff against `sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm_sm120.py`, which is known to
   work on this hardware 閳?specifically its `tile_atom_to_shape_SF` scale layout.
4. **Start from finding 7**: change the word split at
   `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py:983-990`
   from `// k_blocks` / `% k_blocks` to `// 4` / `% 4` (`micro_size_k / sf_vec_size`). That is
   the one hypothesis that fits findings 3, 4 and 6 and was never run.
5. Cross-check the unpacked-A path: try A as a fragment (`IsFragmentBuffer` is allowed for A
   in the sm120 clause) to see whether the shared-memory fp4 read is implicated.
6. Before believing any patch result, clear the TileLang compile cache. A stale
   `~/.tilelang/cache` will serve the old kernel and make a correct patch look like a no-op.
7. Do not trust a symbol-level argument alone. `_scale_word_offset` *looked* like the root cause
   for a whole round; it has no callers. Instrument the call site and read the printed values.

## Measured on

RTX 6000D (sm_120a), TileLang 0.1.15, torch 2.15.0+cu132, CUDA 13.3, driver 610.43.02.
All numbers above are from this machine. No result here was obtained on sm100 or sm90.
