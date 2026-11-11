# NVFP4 blockscaled GEMM on sm120: the instruction exists, the TileLang path does not work

> Status: **hardware path confirmed, TileLang lowering broken**. The `mma.sync` instruction is
> real and selectable on this card; driving it through `T.gemm_blockscaled` produces a
> deterministic wrong answer. This entry records exactly what was tried and what the numbers
> were, so the next person does not repeat the sweep. **No correctness claim is made.**

## The instruction is real

`tilelang/src/tl_templates/cuda/instruction/mma_block_scale.h` guards and emits:

```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
```

with `static_assert(ScaleVecSize == 4)` and a `ue4m3 scale factors` requirement. The guard
is `sm_120a` + CUDA 12.8 or newer. So RTX 6000D is a supported target at the PTX level —
this is not "the arch cannot do NVFP4".

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
factor set to `2.0`, and `K = 256`, the correct output is `(1·2)·(1·2)·256 = 1024` in every
element. The kernel returns

```
0.0009765625  ==  2**-10
```

uniformly. `2⁻¹⁰` is not a plausible partial sum of this problem; it is the value obtained by
reinterpreting a scale-factor byte at the wrong offset. The scale-factor addressing is wrong.

## What the generated code does

Compiled source shows the SF load indexing rows with a stride of 8192 bytes for a
`(512, 64)` fp8 buffer (32768 B total), and hands the result to the MMA as a `uint32*`
built from an `(uint)` cast of a single fp8 byte. The `blockscaled_chunk_kmajor` contract
(`_tile_kmajor_scale_word`: `word_k * 128 + (idx % 32) * 4 + idx // 32`) was implemented
against a plain row-major `(M, K/16)` discriminant and still missed, which points at the
lowering rather than at the caller.

## What this means for the deployment

The Qwen3.8 NVFP4 MoE does **not** run through this path, and should not be made to. On the
same box, `sglang` serves NVFP4 through CuTeDSL —
`kda_kernels/qwen3x_nvfp4_gemm_sm120.py`, ~1970 lines, built on
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
   (`1024`) and does not depend on the fp4 packer being right.
3. Diff against `sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm_sm120.py`, which is known to
   work on this hardware — specifically its `tile_atom_to_shape_SF` scale layout.
4. Cross-check the unpacked-A path: try A as a fragment (`IsFragmentBuffer` is allowed for A
   in the sm120 clause) to see whether the shared-memory fp4 read is implicated.

## Measured on

RTX 6000D (sm_120a), TileLang 0.1.15, torch 2.15.0+cu132, CUDA 13.3, driver 610.43.02.
All numbers above are from this machine. No result here was obtained on sm100 or sm90.
