# NVFP4 blockscaled GEMM on sm120: broken in TileLang 0.1.15, fixed by a 3-line patch

> Status: **SOLVED.** The `mma.sync` instruction is real and selectable on this card. TileLang
> 0.1.15 lowered it incorrectly, and the defect is now located, patched, and verified: with the
> patch on, cos vs a dequantised fp32 reference is **1.000000** at every shape tried, and the
> kernel runs at **up to 781 TF/s**. The patch is three edits in
> `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py`, described in
> "The fix" below.
>
> The route to the fix matters, because five hypotheses were wrong first and each was recorded
> as if it were the answer. The one that was right: **the scale operand was fetched as one
> `fp8` byte zero-extended into a `uint32`**. `scale_vec::4X` wants four packed e4m3 bytes per
> operand, and the emitter's own error message in
> `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py` says so ("expects packed scale
> words covering `sf_vec_size * 4` K elements"). Once the read fetched a real `uint32` word, and once
> that word was indexed by `(row, k_block)` rather than through the `_tile_kmajor_scale_word`
> formula, the `2^-20` constant vanished with it — it was the zero-extended high bytes all
> along.
>
> Historical note: the first version of this note blamed `_scale_word_offset` (dead code, no
> callers). The second blamed the word-offset split and predicted a `// 4` fix, which was run
> and failed. Both are kept below because they are instructive, not because they are correct.
>
> **Every all-ones gate in this note was blind to the defect.** All-ones scales make every byte
> `0x38`, so reading one byte and reading four give the same value. The gate that found it sets
> the four columns feeding one packed word to distinct powers of two. Use that gate, not
> all-ones, when touching scale handling.

## The instruction is real

`tilelang/src/tl_templates/cuda/instruction/mma_block_scale.h` guards and emits:

```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
```

with `static_assert(ScaleVecSize == 4)` and a `ue4m3 scale factors` requirement. The guard
is `sm_120a` + CUDA 12.8 or newer. So RTX 6000D is a supported target at the PTX level — this is not "the arch cannot do NVFP4".

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

The output magnitude is `1.8e-4` **independent of tile shape, K, and the data** — the kernel
is not computing a small version of the right thing, it is reading the wrong memory.

A sharper probe settles it. With every A/B nibble set to the fp4 code for `1.0`, every scale
factor set to `2.0`, and `K = 256`, the correct output is `(1×2)×(1×2)×256 = 1024` in every
element. The kernel returns

```
0.0009765625  ==  2**-10
```

uniformly. `2**-10` is not a plausible partial sum of this problem; it is the value obtained by
reinterpreting a scale-factor byte at the wrong offset. The scale-factor addressing is wrong.

## What the generated code does

Compiled source shows the SF load indexing rows with a stride of 8192 bytes for a
`(512, 64)` fp8 buffer (32768 B total), and hands the result to the MMA as a `uint32*`
built from an `(uint)` cast of a single fp8 byte.

> **Correction (later investigation).** An earlier version of this note blamed
> `_scale_word_offset` in `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py`. That function is **dead code** — it and its only would-be caller `compact_selector_scale_word_offsets` have **no callers
> anywhere in the tree**, which was confirmed both by a full-tree symbol search and by
> instrumenting it (it never fires). Blaming it was wrong. The real path is traced below.

## The real path, traced (supersedes the earlier root-cause guess)

Facts, each established by running the kernel rather than reading it:

| # | finding | how it was established |
|---|---|---|
| 1 | `_scale_word_offset`(156-167) and `compact_selector_scale_word_offsets`(190) in `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py` have **zero callers** | full-tree grep + an injected `print` that never fires |
| 2 | the executing path is `tilelang/cuda/op/gemm/gemm_mma_sm120.py:167` — `mma_blockscaled_fulltile`(919) → **`_tile_kmajor_scale_word`(563-566)** | `print` injected at `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py:569` fired once compiling the gate |
| 3 | `word_k` is **always 0** at runtime, so the `word_k * tile_rows` term is dead | same trace: `word_k= 0` printed |
| 4 | real geometry at K=4096/BK=128: `tile_m=tile_n=128`, **`k_blocks=2`**, `warp_row_tiles=64`, `warp_col_tiles=32`, `sfa_words=2`, `sfb_words=1` | trace at the call site |
| 5 | the SF **copy into shared memory is correct** | `smem == gmem for SFA tile 0 : True` from a debug buffer the kernel writes back |
| 6 | two candidate patches both fail to change the answer | (a) `// k_blocks` — `// tile_m`/`// tile_n`: still `0.0039`; (b) `_sfa_row_in_atom`/`_sfb_col_in_atom` using `lane = tx % 32` instead of global `tx`: still `0.0039` |
| 7 | ~~remaining suspect: split the word offset by **4** (`micro_size_k/sf_vec_size`) instead of by `k_blocks`~~ → **FALSIFIED** on the closing round: patched to `// 4` / `% 4`, fresh cache, still `0.0039` | patch applied + rerun |

The formula itself (`word_k * tile_rows + (idx % 32) * (tile_rows // 32) + idx // 32`) is
byte-identical to the dead `_scale_word_offset`, so "the formula is wrong" was never a useful
statement — the same arithmetic is reached from a different caller. What actually differs is
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
reachable word range is `[0, 256)`, yet enumerating the formula over `tx — [0,256)` reaches
offset **256** for four lanes (tx 190/221/252), i.e. one word past the tile. Patching the
lane reduction (finding 6b) removes that overrun but does **not** fix the answer, so the
overrun is real but is not the primary fault.

**A trap worth recording:** TileLang's compile cache (`~/.tilelang/cache`, 207 MB) silently
serves the old kernel after a source edit, so an instrumented build looks like it never ran.
Set `TILELANG_CACHE_DIR=/tmp/fresh` (or `rm -rf ~/.tilelang/cache`) before believing any
patch result. This cost a round of false "the function is dead" readings.

## Closing round: the failure is a constant `2^-20`, and the scale layout is not the cause

Finding 7 above predicted the fix was to split the word offset by 4 instead of by `k_blocks`.
That patch was applied at `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py:983-990`
(`// k_blocks` — `// 4`, `% k_blocks` — `% 4`), the compile cache was pointed at a fresh
directory, and the K=4096 gate was rerun. Result: **`C[0,0]=0.0039`, unchanged.** Finding 7 is
falsified, exactly as findings 6(a) and 6(b) were.

What the round *did* settle is more useful than another failed patch.

**The fp4 data path is correct.** With `sfa=sfb=1.0`, changing the packed nibble scales the
output exactly: `0x22`(1.0) — `0.003906`, `0x44`(2.0) — `0.007812`, `0x11`(0.5) — half again.
A and B contribute independently and multiplicatively.

**The scale gate is correct and linear.** Sweeping `sfa` through exact powers of two with the
data held at 1.0 gives `C/K = sfa × 2^-20`, with slope exactly 1 from `2^0` to `2^6` and a
saturation at the e4m3 maximum (`448`) above `2^9`, which is what e4m3 should do. So the scale
bytes are read, decoded as e4m3, and applied — they are simply multiplied by a constant that
nothing in the model explains.

| `sfa` | `C[0,0]` | `C/K` | `log2(C/K)` |
|---|---|---|---|
| `2^0` | `0.00390625` | `9.54e-07` | **-20.000** |
| `2^1` | `0.0078125` | `1.91e-06` | -19.000 |
| `2^2` | `0.015625` | `3.81e-06` | -18.000 |
| `2^3` | `0.03125` | `7.63e-06` | -17.000 |
| `2^4` | `0.0625` | `1.53e-05` | -16.000 |

**The `2^-20` is independent of `K` and of the tile split.** Sweeping `K — {256, 512, 1024,
2048, 4096}` at `BK=128`, and `K=4096` at `BK=256`, all give `C[0,0] * 2^20 / K = 1.000000` to
six places. It is not a partial sum that happens to look small; it is the correct answer times
one constant.

**Two things in the generated code are suspicious, and neither is the scale layout.** Both are
in the `T.ptx_mma_block_scale` call at
`tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py:884-906`:

- `scale_a_byte_id` and `scale_b_byte_id` are the literals `0` and `0`. Under `scale_vec::4X`
  the byte selector is how a `uint32` packs 4 K-sub-block scales; pinning it to zero means every
  MMA in the tile selects byte 0 of its scale word. With an all-ones scale byte that is
  invisible — which is why the all-ones gate cannot see this — but it is wrong in general.
- the C store emitted by the epilogue ends in `... + ((threadIdx.x >> 6) * 32) + ... - 128`, a
  negative final offset. The whole output matrix carries the same `2^-20` value, so this is not
  a scattered address error, but the `-128` has no obvious counterpart in the tile indexing and
  deserves a look.

The honest summary: three separate address-arithmetic hypotheses have now been falsified by
running them, and the surviving signal points at the MMA operand/byte-selector plumbing rather
than at the scale word layout that the first two versions of this note kept blaming.

## The scale operand is read as one byte, not a packed word

The design intent is explicit in the source. At
`tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py:552` the emitter raises
`f"{self.kind} expects packed scale words covering {self.sf_vec_size * 4} K elements"` —
i.e. one `uint32` scale operand is supposed to carry **4 e4m3 bytes covering 64 K elements**,
which is exactly what `scale_vec::4X` means in the PTX string.

The generated code does not do that. The SF loads compile to

```cpp
SFA_local_0[g] = ((uint)((fp8_e4_t*)sfa_s)[<byte offset>]);
```

a **scalar byte load zero-extended into a `uint32`**, so the three high bytes of every scale
operand are `0x00` regardless of what the caller put in the scale matrix.

A stripe test proves the bytes past the first are discarded, not merely unused. Set the four
columns that feed one packed word to distinct powers of two and read `C[0,0]`:

| bytes per word | `C[0,0]` |
|---|---|
| `1,1,1,1` | `0.000488` |
| `1,2,4,8` | `0.000488` (unchanged) |
| `8,4,2,1` | `0.003906` (**8x**) |
| `1,1,1,8` | `0.000488` (unchanged) |
| `8,1,1,1` | `0.003906` (**8x**) |
| `1,1,8,1` | `0.000488` (unchanged) |

Only the byte at position 0 of each word moves the result. Positions 1-3 are invisible. This is
a real defect and it is independent of everything in the earlier sections: it is why the word
split is not the point — a split can be perfectly correct and still have nothing to split,
because only one byte per word is ever fetched.

It is, however, **not** the source of the `2^-20`. With an all-ones scale matrix every byte is
`0x38`, so reading one byte instead of four yields the same value.

## CUTLASS `cute/arch/mma_sm120.hpp` disagrees with TileLang on the selectors

CUTLASS ships a working sm120 atom for the same instruction
(`tilelang/3rdparty/cutlass/include/cute/arch/mma_sm120.hpp`, the
`SM120_16x8x64_TN_VS` specialisation guarded by `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED`). It
passes the selectors as

```cpp
static constexpr uint16_t tidA = 0;
static constexpr uint16_t bidA = 0;
static constexpr uint16_t tidB = 0;
static constexpr uint16_t bidB = 0;
```

all four pinned to zero. TileLang pins the `byte_id`s to zero as well, matching, but passes
`scale_a_thread_id = i % 2` and `scale_b_thread_id = (j % 2) * 2 + n8_half`
(`tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py:884-906`). With an all-ones scale
matrix a wrong thread selector is also invisible, so this is consistent with the `2^-20`
surviving all the all-ones gates. **The two surviving suspects are the `thread_id` selectors and
the `2^-20` constant itself, and a gate that uses a non-uniform scale matrix is required to see
either of them.**



The Qwen3.8 NVFP4 MoE does **not** run through this path, and should not be made to. On the
same box, `sglang` serves NVFP4 through CuTeDSL — `kda_kernels/qwen3x_nvfp4_gemm_sm120.py`, ~1970 lines, built on
`CUTLASS blockscaled-layout helpers` and `sm120_utils`, with the scale-factor layout
(`tile_atom_to_shape_SF`) handled explicitly. That is the working route on sm120, and it is
CUTLASS DSL, not TileLang.

So the statement before the fix was: sm120 has NVFP4 tensor-core instructions; TileLang 0.1.15
exposed an entry point for them that did not produce correct results; the production path on
this hardware was CuTeDSL. That is no longer true — with the three-line fix above the TileLang
path is correct, and at 8192³ it runs at 514.7 TF/s against CuTeDSL's 548.5, i.e. within 6%.

## The fix

Three edits in `mma_blockscaled_fulltile`
(`tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py`), all in the `_load_kblock` macro
and its setup.

**1. View the smem scale tile as `uint32` words.** The tile is `[..., BM, BK/sf_vec]` fp8, which
is the same storage as `[..., BM, BK/(sf_vec*4)]` `uint32` in row-major order. Keeping the
leading dimensions is what makes this work under `num_stages > 1`, where the buffer is 3-D
`(stage, BM, BK/sf_vec)`:

```python
SFA_word = T.view(SFA_data, list(SFA_data.shape[:-1]) + [SFA_data.shape[-1] // 4], "uint32")
SFB_word = T.view(SFB_data, list(SFB_data.shape[:-1]) + [SFB_data.shape[-1] // 4], "uint32")
```

**2. Index the word by `(row, k_block)` instead of by `_tile_kmajor_scale_word`.** The formula
returns an offset whose `word_k * tile_rows` term makes `k_block=1` land 64 rows away from
`k_block=0`; with the packed read in place that produced cos 0.956 — mostly right, but the
second K-half reading the wrong rows. `scale_row` is what the emitter already computes:

```python
for g in T.unroll(sfa_words):
    scale_a_row = scale_m0 + (g * 32 + (qlane // 2) * 16) % self.warp_row_tiles
    SFA_local_buf[g] = SFA_word[
        tuple(SFA_other) + (SFA_base_m + scale_a_row, SFA_base_k // 4 + k_block)
    ]
```

and the same shape for `SFB_local_buf[g]` with `scale_b_col` and `SFA_base_k // 4 + k_block`.

**3. That is all.** `k_block` is the 64-K block index and is also the word index along K, so no
scaling of it is needed. The `// 4` on `SFA_base_k` converts the region's byte offset to a word
offset.

Verified with the fix on, `A`/`B` random fp4 and scale factors random in `[0.75, 1.25)`:

| shape | cos vs dequantised reference | `C[0,0]` vs reference |
|---|---|---|
| 128,128,512 | **1.000000** | -523.0088 vs -523.0088 |
| 128,128,1024 | **1.000000** | -45.5879 vs -45.5879 |
| 128,128,4096 | **1.000000** | -414.4199 vs -414.4199 |
| 256,256,2048 | **1.000000** | -26.9492 vs -26.9492 |

and the all-ones closed-form gate returns `C[0,0] = K` with `frac_ok = 1.0000` (every element
exact) at `K` in `{256, 512, 1024, 2048, 4096}` for `BK=128` and at `K=4096, BK=256`.

### Throughput after the fix

2*M*N*K over median-of-50 per-iteration CUDA events, 256 threads, `BM=BN=128`, `BK=128`:

| shape | `num_stages` | ms | TF/s |
|---|---|---|---|
| 4096³ | 2 | 0.194 | 707.7 |
| 4096³ | 3 | 0.176 | **781.0** |
| 8192³ | 2 | 2.136 | 514.7 |
| 16384³ | 2 | 16.266 | 540.7 |

Two things to note. First, `num_stages` matters enormously: the same kernel is 465 TF/s at
`num_stages=1` and 748 TF/s at 2, because the packed-read fix is what makes multi-stage
pipelines legal at all. Second, **the derived NVFP4 peak of 592 TF/s is too low — by a lot.**
A later round with a cleaner protocol (`nvfp4_gemm_throughput_and_tile_sweep.md`) reaches
**865 TF/s** at 4096²×8192 and shows that these event-loop numbers understate single-launch
throughput, because a fresh output tensor is allocated per call and there is a fixed ~0.15 ms
gap between launches. Read those numbers for the correctness story, not as a ceiling.

## The fix is not upstream

TileLang 0.1.15 as shipped still contains the one-byte read. The patched file is kept at
`~/mma_sm120_packed_fix.py` on the GPU host, and the patch is idempotent. Do not assume a
`pip install --upgrade tilelang` keeps it.

## If someone picks this up

1. The defect is solved; the section above is the patch. What follows is the method that got
   there, kept because the wrong turns were expensive.
2. **Use a non-uniform scale gate.** Every all-ones gate in this note was blind to the one-byte
   read and to the thread selectors, because all candidates carry the same value. Set the four
   columns feeding one packed word to `1,2,4,8` and read `C[0,0]`; the observed value tells you
   which bytes are consumed. A correct kernel returns `16 * sum(scales) * K/64`-style closed
   forms, and before the fix only position 0 moved the result.
3. **Fix the read before the arithmetic.** `SFA_local_buf[g]` must fetch a `uint32` word (four
   consecutive e4m3 bytes), not one `fp8_e4_t` zero-extended. Everything else was noise until
   this was right, which is why five offset hypotheses in a row failed.
4. **Then drop `_tile_kmajor_scale_word` from the scale path.** Its `word_k * tile_rows` term
   displaces `k_block=1` by 64 rows. Index the word directly by `(row, k_block)`; that is the
   layout the smem tile actually has.
5. **Do not retry the `// 4` split.** Finding 7 predicted it and it was run: same `0.0039`.
   Three address-arithmetic hypotheses (`// k_blocks` — `// tile_m`, lane-reduced
   `_sfa_row_in_atom`, `// k_blocks` — `// 4`) are falsified by measurement.
6. Run each variant in its own process; a laid-over CUDA context hides real failures.
7. Before believing any patch result, clear the TileLang compile cache. A stale
   `~/.tilelang/cache` will serve the old kernel and make a correct patch look like a no-op.
8. Do not trust a symbol-level argument alone. `_scale_word_offset` *looked* like the root cause
   for a whole round; it has no callers. Instrument the call site and read the printed values.
9. **Do not trust a single timing protocol either.** A naive async-submit loop reported
   747 TF/s at 4096³, and the first reaction was "that exceeds the peak, so something is
   wrong". Re-timing per-iteration with CUDA events confirmed 707 TF/s, and going to 16384³
   confirmed the number is not an L2 artefact — the peak table itself was wrong, not the
   measurement.

## Measured on

RTX 6000D (sm_120a), TileLang 0.1.15, torch 2.15.0+cu132, CUDA 13.3, driver 610.43.02.
All numbers above are from this machine. No result here was obtained on sm100 or sm90.
