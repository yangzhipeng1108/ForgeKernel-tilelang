# From-scratch sm120 NVFP4 CuTe DSL: the load path is the wall

Status: **negative result, load path only**. The MMA half compiles clean; the
gmem→smem half is what blocks a hand-written kernel on this card.

Measured on RTX 6000D (sm_120a), cutlass-dsl 4.8.0, container `cutedsl_probe`.

## What did get built

A standalone single-CTA `128x128x128` NVFP4 GEMM that authors its own tiled MMA,
shared-memory layouts, SF fragment partition, ldmatrix reads, and epilogue —
none of `DenseGemmKernel` / `Sm120BlockScaledGemmKernel` involved:

```python
op = cute.nvgpu.warp.MmaMXF4NVF4Op(F4, F32, E4M3)
tiled_mma = cute.make_tiled_mma(
    op, cute.make_layout((2, 2, 1)),
    permutation_mnk=sm120_utils.get_permutation_mnk((128,128,128), 16, False))
```

`get_permutation_mnk` returns `(128, layout((8,2,2),(1,16,8)), 64)` for NVFP4
(`use_mxf8f6f4=False` → perm_k = 64). Everything downstream of the SF layout is
mechanical once these two lines are right.

MMA call form for block-scaled operands (list operands, per-atom slicing):

```python
cute.gemm(tiled_mma, acc[None, mm, mn],
          [tCrA[None, mm, kb], tCrSFA[None, mm, kb]],
          [tCrB[None, mn, kb], tCrSFB[None, mn, kb]],
          acc[None, mm, mn])
```

Note the **mma_m reversal when mma_n is odd** (`mma_m = size-1-step if mma_n % 2`),
copied from the reference's pingpong schedule; it is an accumulator-reuse order
trick, not a correctness requirement for a single-tile probe.

## Trap 1 — pipeline barrier storage is 2 stages wide

`PipelineTmaAsync.create(barrier_storage=ptr)` carves **two** mbarrier arrays out
of that pointer: `sync_object_full` at `ptr` and `sync_object_empty` at
`ptr + num_stages`. A `MemRange[Int64, 1]` (8 bytes) is silently too small; the
empty phase writes past it and the barrier never flips, so `consumer_wait` hangs
in `torch.cuda.synchronize()` with no error. Allocate `2 * num_stages` Int64.

## Trap 2 — fp4 objects cannot be copied one element at a time

`tensor[i] = other[i]` on `Float4E2M1FN` raises
`Sub-byte scalar dereference not supported for type Float4E2M1FN`. Every fp4
movement has to go through a vectorized copy atom (`num_bits_per_copy` a multiple
of 32 bits) or TMA. This kills the "just loop over the tile and assign" fallback.

## Trap 3 — a raw fp4 TMA atom rejects this smem layout

`cpasync.make_tiled_tma_atom(CopyBulkTensorTileG2SOp(), mA, a_smem, (128,128))`
on the swizzled fp4 smem layout fails at IR build:

```
invalid input types, got 'non_exec_tiled_tma_load<sm_90, f4E2M1FN,
copy_bits = 65536, tma_gbasis = "(128,128):(1@1,1@0)">'
```

`copy_bits = 65536` is the fp4 tile expressed in bits — over what this descriptor
accepts. The reference does **not** hit this because it routes NVFP4 through the
scale-factor path with `internal_type=cutlass.Int16` on the SF tensors and never
makes a plain fp4 TMA for a 128-deep K tile in one shot the way the naive call does.

## Trap 4 — swizzled smem coordinates are not gmem order (the actual wall)

`make_tiled_copy_tv(atom, layout((32,)), layout((1,)))` + `partition_S/partition_D`
produces tensors whose modes do not line up:

```
gmem side: ((1,1),4,128,1):((0,0),4096,1,0)
smem side: ((1,1),4,(128,1)):((0,0),4096,(1,0))
```

The smem view carries an extra nested `(128,1)` inside its third mode (swizzle
grouping); any `[None,None,i,0]` slice on one side is weakly incongruent with the
other. This is the same mismatch that produced NaNs in the earlier from-scratch
attempt: without TMA's hardware coordinate mapping you must reconstruct the
swizzle permutation by hand, and getting it wrong silently corrupts the operands.

## Conclusion

On sm120 NVFP4 the *compute* side is fully reachable from CuTe DSL primitives
(MmaMXF4NVF4Op + permutation + SF fragment partition all work). The **load** side
is where a from-scratch kernel stalls: fp4 forces vectorized/TMA-only movement,
and the swizzled smem layout only lines up with gmem through TMA's own descriptor
mapping. So the honest options on this card are:

1. wrap a verified kernel that already owns that mapping (what
   `kernels/gemm/nvfp4_gemm/sm120/kernel.py` does, 855 TFLOP/s);
2. or reproduce the reference's Int16-internal-type SF TMA plumbing exactly.

A hand-rolled cp.async path would have to re-derive the swizzle permutation — the
same reasoning that makes TileLang's `T.gemm_blockscaled` wrong here
(`hw_kb/tilelang/sm120/nvfp4_blockscaled_is_broken_in_tilelang.md`) applies.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB
- `lmsysorg/sglang:dev` container: cutlass-dsl **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- probe `_probes/probe_sm120_nvfp4_from_scratch.py` — compiles end to end, stalls in
  `torch.cuda.synchronize()` at the first TMA issue (trap 3/4)

