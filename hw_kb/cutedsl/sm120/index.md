# SM120 (consumer Blackwell, RTX 6000D / RTX 50-series) interface index

Demand-driven, exactly like `sm90/` and `sm100/`: a recipe is written when a student is
blocked on that interface, follows `../TEMPLATE.md`, and is marked verified **only** on a
GPU probe that returned `PROBE PASS` on an sm120 part. This file is a ledger of what has been
earned on this architecture, not a plan.

## Why this tree exists at all, and what it is NOT

The RTX 6000D is **compute capability (12, 0) = sm_120a**. That is *consumer* Blackwell, and
most of `sm100/` does not apply to it. The single most useful fact in this tree:

> **sm120 has no `wgmma` and no `tcgen05`.** Its tensor-core path is warp-level `mma.sync`
> (HMMA) plus the warp-level block-scaled family (`mx*.block_scale`). Any `sm100/` entry
> whose mechanism is "issue the UMMA / allocate TMEM / drive the 2-CTA pair" is **not
> applicable**, no matter how good the entry is.

The DSL says so itself when you try: constructing `tcgen05.MmaMXF4NVF4Op` on this card raises

```
expects arch to be one of [sm_100a, sm_100f, sm_103a, sm_107a, sm_110a], but got sm_120a
suggestion: tcgen05 MMA requires datacenter Blackwell (sm_100 / sm_103).
            On consumer Blackwell (sm_120 / sm_121) use the warp-level
            cute.nvgpu.warp.MmaMXF4Op / MmaMXF4NVF4Op / MmaMXF8Op
```

So this tree is **not** a renamed copy of `sm100/`. Nothing is pre-filled from it. The entries
here name mechanisms that exist only, or differently, on sm120.

## Find by symptom first

| What you see / what you are about to do | Look first at |
|---|---|
| **About to run NVFP4 (`e2m1` x `ue4m3` scale) on the 6000D**; or a kernel that should be fp4 compiles but the numbers are garbage with no hang; or you want to know why TileLang's `T.gemm_blockscaled` gives a wrong answer here; or `tcgen05.MmaMXF4NVF4Op` refuses sm120 and you do not know what to use instead | [`nvfp4_warp_mma_works`](nvfp4_warp_mma_works.md): the instruction `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X`; `MmaMXF4NVF4Op` constructs on sm_120a with shape `(16,8,64)`, sf_vec 16; **the SF is a pointer set on the trait (`Field.SFA/SFB`), never a `cute.gemm` argument** — that is the exact difference from the broken TileLang path; the closed-form acceptance probe (A=B=1.0, SF=2.0, K=256 -> 1024) |

## Status of the known sm120 interfaces

| Interface | State on this card | Where |
|---|---|---|
| `cute.nvgpu.warp.MmaMXF4NVF4Op` (NVFP4) | **op verified**, full GEMM not yet gated | [`nvfp4_warp_mma_works`](nvfp4_warp_mma_works.md) |
| `cute.nvgpu.warp.MmaMXF8Op` (fp8, `(16,8,32)`, UE8M0 scale) | op constructs (verified alongside the fp4 one) | same entry, "also tried" |
| `cute.nvgpu.warp.MmaF16BF16Op` / `MmaFP8Op` / `MmaTF32Op` | used by the shipped GeForce example; fp16 run **PASS** on this card | not yet written up |
| `cutlass.utils.blockscaled_layout.sm120_make_smem_layout_sfa/sfb` | exist; the pair to use on sm120 (the `*_tmem_*` ones are sm100) | [`nvfp4_warp_mma_works`](nvfp4_warp_mma_works.md) |
| `tcgen05.*` (UMMA, TMEM, 2-CTA) | **not present on this card** — rejected at construction | see the quote above |
| `wgmma.*` (Hopper) | **not present on this card** | — |

## Official examples, by architecture (this saves the most time)

The CUTLASS DSL examples split by target and pick the wrong one silently looks like a bug:

| example | target | runs on 6000D? |
|---|---|---|
| `CuTeDSL/blackwell/tutorial_gemm/nvfp4_gemm_0.py` | tcgen05 / sm100 | **no** — refuses sm120 |
| `CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py` | sm103 | **no** |
| `CuTeDSL/blackwell_geforce/dense_gemm.py` | sm120 | **yes** — fp16 verified PASS here; but **fp16/fp8 only**, its `_setup_attributes` hardcodes `MmaF16BF16Op` and rejects any `a_dtype.width` outside {16, 8} |

So there is **no shipped fp4 example for this card**. An NVFP4 kernel has to be written.

## What is deliberately absent

- No cluster / multicast / DSMEM entries yet: the RTX 6000D is being used as a single GPU
  here, and its two cards are SYS-linked (no NVLink), so the cross-GPU multicast story is
  unreachable rather than unmeasured.
- No `sm100` content copied over. If a mechanism is arch-neutral (DSL algebra, probe
  methodology) it belongs in every tree, but it has to be **re-earned or explicitly
  re-certified** here with a "Certification source" line — location is not provenance.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB per card, no NVLink between them
- `lmsysorg/sglang:dev` container: cutlass **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- TileLang readings quoted in this tree come from `~/tlvenv` (tilelang 0.1.15)
