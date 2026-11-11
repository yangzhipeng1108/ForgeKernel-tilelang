# NVFP4 on sm120 computes correctly — via the shipped flashinfer kernel

The sm120 tree already records that the warp-level block-scaled instruction exists
(`nvfp4_warp_mma_works.md`). That entry stops at "the op constructs and the SF ABI is a
pointer". This one closes the loop: on this card the NVFP4 GEMM **returns the right numbers
end to end**, and the shortest route to that is not to hand-build the smem plumbing.

## The result

`Sm120BlockScaledGemmKernel` from `flashinfer.gemm.kernels.sm12x_raw` runs unmodified
and is exact:

| M | N | K | mode | got | expected | max abs diff | cos |
|---|---|---|---|---|---|---|-----|
| 128 | 128 | 128 | closed form | 512 | 4·K = 512 | **0** | 1.000000 |
| 128 | 128 | 256 | closed form | 1024 | 4·K = 1024 | **0** | 1.000000 |
| 256 | 128 | 512 | closed form | 2048 | 4·K = 2048 | **0** | 1.000000 |
| 128 | 256 | 1024 | closed form | 4096 | 4·K = 4096 | **0** | 1.000000 |
| 256 | 256 | 1024 | random fp4 | 411 | 411 | **0** | 1.000000 |

The closed-form operand is A = B = 1.0 on the e2m1 grid with every scale factor 2.0, so each
dot product is `K · (1·2) · (1·2) = 4K`. The random row matters more: it is not 1.0
propagation, and it is still bit-exact against an fp32 reference.

## Why this settles the TileLang question

TileLang's `T.gemm_blockscaled` on this same card returns `2**-10` for the same `4K` gate
(`../../tilelang/sm120/nvfp4_blockscaled_is_broken_in_tilelang.md`). Same instruction, same
GPU, same operands. The hardware and the PTX are fine. The defect is in one lowering.

## What the kernel expects from the caller

The signature is compact and the SF side is the only subtle part:

```
Sm120BlockScaledGemmKernel(acc_dtype, sf_vec_size, tile_shape_mnk=(128,128,128), epi_tile=(128,32))
kernel(a, b, sfa, sfb, c, alpha, max_active_clusters, stream)
```

- `tile_shape_mnk` M and N must be 128 or 256 — the block-scaled SF smem layout covers whole
  128-row/col atoms, and the fragment helpers only handle whole atoms.
- `sf_vec_size` 16 selects NVFP4 (`kind::mxf4nvf4`, `scale_vec::4X`, UE4M3 scales); 32 selects
  the MX formats with UE8M0.
- **A / B / C / SFA / SFB must all be rank-3** `(M, K, 1)` / `(N, K, 1)` / `(M, N, 1)`.
  The kernel re-views the SF tensors with `tile_atom_to_shape_SF(a.shape, sf_vec_size)`, which
  asserts that the target shape and the atom order have equal rank. A rank-2 `(M, K)` tensor
  fails with `expects target shape and order operands have same rank, but got (128,128) vs
  (2,1,3)`. This is the one error that costs time.
- SF storage is already in MMA atom order; the values are packed as
  `(l, ceil(mn/128), ceil(sf_k/4), 32, 4, 4)` then permuted by `(3, 4, 1, 5, 2, 0)`. The probe
  below shows the repack.

## The things that made a hand-built kernel fail

Written down because they are all silent-corruption shaped, and all cost real time:

1. **`make_tiled_copy_tv(atom, layout_threads, layout_values)` — the first layout is the
   thread count, not the element count.** Passing `size(gA)` (16384) as the thread layout
   makes the CuTeDSL compiler **abort with a core dump**, not raise. A compile-time abort in
   `compile_and_jit` with no Python traceback is almost always a malformed copy tiler.
2. **The block-scaled atom does not accept `Field.SFA` / `Field.SFB` the way sm100 does.**
   the block-scaled trait setter demands a `Pointer`, and a raw smem `.iterator` is rejected by
   the IR verifier (`Invalid operand type for field sf_a`). The working kernels never touch
   the field: they pass the scale factors **as extra list operands to the gemm primitive**:
   `gemm(tiled, acc, [tCrA, tCrSFA], [tCrB, tCrSFB], acc)`.
3. **All operands of that list form must share a rank.** A/B fragments from
   `make_fragment_A(partition_A(sA))` and SF fragments from `partition_fragment_SFA` are both
   `(V, MN, K)` — rank 3 — and slicing with `[None, None, 0]` on one but not the other gives
   `expects all operands to have the same rank`.
4. **fp4 cannot be stored element-by-element.** Sub-byte scalar dereference raises
   `Sub-byte scalar dereference not supported for type Float4E2M1FN`. Write the packed byte
   image (`0x22` = both nibbles are code 2 = 1.0) through a `Uint8` buffer and `recast_ptr` to
   the fp4 view.
5. **`make_smem_layout_a` returns a swizzled `ComposedLayout`**, which the universal copy atom
   refuses (`got composed_layout<...>`). Either use the matching ldmatrix copy atom, or for a
   correctness kernel use a plain layout.

## Two constraints that are not obvious

- `import os` / `return` / `print` inside a device kernel body are all illegal. Early exit needs
  `if const_expr(...)`, and any `if` on a runtime value becomes staged control flow.
- `make_mma_atom` and friends build kernel IR, so they are only legal inside a jit or
  a kernel region. Pointers (`make_ptr`) are legal at plain-Python level; tensors and layouts
  are not.

## A shippable module, and what it is not

`kernels/gemm/nvfp4_gemm/sm120/kernel.py` turns this into a `build()/gemm_nvfp4()/_bench()`
module in the repo's standard shape. It is **a wrapper around the verified container kernel**
(`DenseGemmKernel`), not a hand-written one — the from-scratch attempt is exactly the failure
recorded above. What the wrapper owns is the caller-side encoding (e2m1 snapping + packing, the
pre-permuted SF atom layout) and one cached `cute.compile` launch per shape. Measured:

| shape | time | throughput | correctness |
|---|---|---|---|
| 2048 × 4096 × 4096 | 0.0804 ms | 855 TFLOP/s | md=2, cos 0.999999 |
| 4096 × 4096 × 4096 | 0.2721 ms | 505 TFLOP/s | md=2, cos 0.999999 |

Both exceed the TileLang sm120 baselines (fp8 375 TF, f16 158.8 TF); TileLang has no working fp4
kernel at all. The 4096³ point is lower than 2048³ because a fixed 128×128 tile leaves the 4096
grid under-filled — the same "wide, not starved" effect the TileLang lm_head entry records.

The from-scratch path was pursued further after this entry was written: the MMA half (tiled MMA,
permutation, SF partition, list-operand `cute.gemm`) does compile from primitives, and the load
half is where it stops. That result is [`from_scratch_tma_load_path_works`](from_scratch_tma_load_path_works.md).

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB, no NVLink
- `lmsysorg/sglang:dev`: cutlass **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- Probe: `_probes/probe_sm120_nvfp4_gemm_gate.py`
