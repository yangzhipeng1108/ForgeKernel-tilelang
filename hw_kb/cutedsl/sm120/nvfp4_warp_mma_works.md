# sm120 NVFP4: the warp-level blockscaled MMA that works (and the one TileLang breaks)

> Status: **mechanism verified in CuTeDSL on the RTX 6000D (sm_120a)**. The op constructs,
> the arch gate accepts this card, and the scale-factor ABI is identified. A full running
> GEMM is the next step and is called out as such below — do not read this as "we have a
> working NVFP4 kernel."

## Why this entry exists

`hw_kb/tilelang/sm120/nvfp4_blockscaled_is_broken_in_tilelang.md` records that TileLang's
`T.gemm_blockscaled` compiles on this card, dispatches to `cuda.mma.blockscaled`, and then
returns a **wrong answer** (2**-10 for a closed-form 1024 — a mis-addressed scale factor).

This entry records the other half: the same hardware instruction through CuTeDSL, where the
scale factors are bound explicitly and there is no offset arithmetic to get wrong.

## The instruction

Both paths target the same PTX:

```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
```

CuTeDSL names it `cute.nvgpu.warp.MmaMXF4NVF4Op`. Its constraints, read from the op's own
docstring and its `__post_init__`:

| property | value | enforced by |
|---|---|---|
| shape_mnk | `(16, 8, 64)` | `__post_init__` raises if not |
| A/B dtype | `Float4E2M1FN` (e2m1) | same |
| SF dtype | `Float8E4M3FN` (ue4m3) | same |
| SF vec size | `16` | same |
| accumulator | `Float32` | raises if not |
| operand layout | A row-major K-major, B col-major K-major, **no transpose** | docstring |

## The arch gate accepts sm120 — and names the trap for sm100 code

```
MmaSM120BlockScaledOp.admissible_archs
  = [arch.sm_120a, arch.sm_120f, arch.sm_121a, arch.sm_121f]
```

Constructing it on this card **succeeds** (`arch = sm_120a` was read back from the DSL).

The contrast is worth keeping because it is the single most useful diagnostic on this box:
the *datacenter* Blackwell path (`tcgen05.MmaMXF4NVF4Op`) **refuses** sm120 with a message
that names the fix:

```
expects arch to be one of [sm_100a, sm_100f, sm_103a, sm_107a, sm_110a], but got sm_120a
suggestion: tcgen05 MMA requires datacenter Blackwell (sm_100 / sm_103).
            On consumer Blackwell (sm_120 / sm_121) use the warp-level
            cute.nvgpu.warp.MmaMXF4Op / MmaMXF4NVF4Op / MmaMXF8Op;
            see the blackwell_geforce dense_gemm / blockscaled_gemm examples.
```

So "NVFP4 does not work on the 6000D" is wrong at every layer: the instruction exists, the
DSL exposes it, and the gate is a clean accept/reject rather than a silent wrong answer.
Only TileLang's lowerer is broken.

## The scale-factor ABI: SF is a pointer set on the trait

This is the mechanism that separates the working path from the broken one.

`cute.gemm(atom, d, a, b, c)` takes **no** scale-factor argument. The scales are attached to
the MMA atom's trait:

```python
op    = warp_mma.MmaMXF4NVF4Op(Float4E2M1FN, Float32, Float8E4M3FN)
atom  = cute.make_mma_atom(op)
# trait.set(Field.SFA, <pointer>)  /  trait.set(Field.SFB, <pointer>)
```

`MmaBlockScaledTrait.admissible_fields == [Field.SFA, Field.SFB]`, and `set()` rejects
anything that is not a `Pointer`:

```
expects value to be a pointer for SFA, but got <type>
```

`get()` is deliberately unsupported on this op (`SM120 warp-level block-scaled MMA does not
support get method for any field`) — the scales are write-only from the DSL's point of view.

That is the whole difference. CuTeDSL hands the instruction a **pointer**, so the address is
whatever the caller computed for a real buffer. TileLang computed a flat offset into the
scale tensor itself (`word_k * 128 + (idx % 32) * 4 + idx // 32`) and mis-addressed it,
which is why its output was a reinterpreted scale byte rather than a partial sum.

## Video memory layout helpers

`cutlass.utils.blockscaled_layout` ships the sm120-specific ones:

```
sm120_make_smem_layout_sfa(...)   -> smem layout for A's scale factors
sm120_make_smem_layout_sfb(...)   -> smem layout for B's scale factors
```

They are distinct from the `make_tmem_layout_sfa/sfb` used by the tcgen05 path. Use the
`sm120_*` pair here; the TMEM ones belong to sm100 and will not apply.

## The correctness gate to use before trusting any NVFP4 kernel

Do **not** start with random data: a broken scale path still returns plausible-looking
numbers. Use a closed-form case first.

```
A nibble = code for 1.0,  B nibble = code for 1.0,
SFA = SFB = 2.0 (ue4m3),  K = 256
=> C = (1*2) * (1*2) * 256 = 1024  in every element
```

TileLang returns `2**-10` here. A correct kernel returns `1024`. This probe does not depend
on the fp4 packer being right, which is why it isolates the scale path.

## What is verified vs. what is not

| claim | status |
|---|---|
| `MmaMXF4NVF4Op` constructs on sm_120a | **verified** (read `arch = sm_120a` back) |
| shape/dtype/sf constraints | **verified** (from the op's own checks) |
| SF is a pointer on the trait, not a gemm arg | **verified** (from `MmaBlockScaledTrait`) |
| `sm120_make_smem_layout_sfa/sfb` exist | **verified** |
| tcgen05 rejects sm120 and suggests the warp op | **verified** (ran it) |
| a full NVFP4 GEMM returning 1024 on the closed-form probe | **NOT yet run** |

The last row is the honest boundary. Constructing the atom and knowing the ABI is not the
same as a kernel that computes the right answer end to end; the K-loop, SMEM staging and
epilogue still have to be written and gated.

## Practical note: official examples split by arch

- `CuTeDSL/blackwell/tutorial_gemm/nvfp4_gemm_0.py` and `.../sm103_dense_blockscaled_gemm_persistent.py`
  are **tcgen05 / sm100** and refuse this card.
- `CuTeDSL/blackwell_geforce/dense_gemm.py` **runs on sm120** (fp16 verified PASS here) but
  supports only fp16/fp8 — its `_setup_attributes` hardcodes `MmaF16BF16Op` and rejects
  `a_dtype.width` outside {16, 8}. There is **no** fp4 path in the GeForce example.

So an NVFP4 kernel for this card has to be written, not copied. The two pieces above
(the op and the SF ABI) are what makes writing it tractable.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB
- `lmsysorg/sglang:dev` container: cutlass **4.8.0**, torch 2.14.1+cu130
- 2026-11, by running the probes inside that container on this machine
