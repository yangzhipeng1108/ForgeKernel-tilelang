# From-scratch sm120 TMA load path: solved

Status: **the wall is down for the plain-dtype path.** A hand-written
`cpasync.make_tiled_tma_atom` + `PipelineTmaAsync` load lands bit-exact
(`maxdiff 0.0`) in 128 threads. The four things that made it hang or fault were
all API-shape mistakes, not hardware limits.

Measured on RTX 6000D (sm_120a), cutlass-dsl 4.8.0, container `cutedsl_probe`.

## What now works

`_probes/probe_sm120_tma_pipeline_smoke.py` — one CTA, `TILE=128`, one fp16 A
and one fp16 B buffer, `PipelineTmaAsync` with a dedicated producer warp. It
TMA-loads `a[:128,:128]` into smem, reads the diagonal back, and compares to
`torch.diagonal(a)`:

```
exp[:6] [-0.9248, -0.5713, -0.2498, -0.7642, -0.9731, -0.0876]
got[:6] [-0.9248, -0.5713, -0.2498, -0.7642, -0.9731, -0.0876]
maxdiff 0.0
PASS
```

Single stage only (see "Multi-stage is a budget problem" below), but the
producer/consumer handshake, the barrier phase, and the gmem→smem coordinate
mapping are all proven correct.

## The four fixes (each one was a silent hang or a 719)

### Fix 1 — the smem layout must match gmem's major order

This was the real corruption. `cute.make_layout((TILE, TILE), stride=(1, TILE))`
is **column-major**; the torch tensor is row-major. The first element coincidentally
lands at the same address in both, so `got[0]` matched and every other sample was
garbage (`maxdiff 3.97`) with no error at all. Correct:

```python
smem_a = cute.make_layout((TILE, TILE), stride=(TILE, 1))   # row-major, matches gmem
```

The tell: **one correct element followed by noise means a stride/major-order bug,
not a TMA bug.** A genuine TMA failure hangs or faults; it does not return mostly-wrong data.

### Fix 2 — go through `tma_partition`, never slice the TMA tensor by hand

`gA[(None, 0, 0)]` fails IR verification:

```
expects `coord` and shape of view are weakly congruent,
got layout "<(((128,128),1)):...>" coord "(_,0,0)"
```

The TMA tensor is not a plain tile tensor; it needs partitioning first:

```python
tAsA, tAgA = cpasync.tma_partition(
    atom_a, 0, cute.make_layout((1,)),
    cute.group_modes(sA, 0, 2), cute.group_modes(tma_a, 0, 2))
cute.copy(atom_a, tAgA, tAsA[(None, pp.index)], tma_bar_ptr=bp)
```

After partitioning, `tAgA` is a **single-mode** tensor — index it as `tAgA`,
not `tAgA[(None, 0)]`.

### Fix 3 — the second return of `make_tiled_tma_atom` is the tensor

```python
atom_a, tma_a = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(), a, smem_a, (TILE, TILE))
```

Discarding `tma_a` and passing the raw tensor (or a `local_tile` of it) to
`tma_partition` gives `expects static shape in the first mode ... gmemShape: (?)`.

### Fix 4 — `from_dlpack` the torch tensor before it enters `@cute.jit`

A raw `torch.Tensor` argument keeps a dynamic shape, so the TMA tile mode is `?`
and partition rejects it. `from_dlpack(a, assumed_align=32)` makes the shape
static. This is how the reference's host `__call__` receives its arguments too.

## API details that also matter

- `consumer_group=` is a `pipeline.CooperativeGroup(pipeline.Agent.Thread, n)`,
  not a bare int.
- `barrier_storage=` wants a `cute.Pointer`. Only `cute.struct.MemRange.data_ptr()`
  produces one — `allocate_array(...).raw_ptr()` gives
  `base_dsl.typing.Pointer` and raises `Expected barrier_storage to be a cute.Pointer`.
- `producer_tail(state)` is mandatory after the k loop, or the kernel hangs on exit.
- `cute.arch.mbarrier_init_fence()` after `PipelineTmaAsync.create`.
- `producer_state.reset_count()` before the loop.
- `@cute.kernel` functions must be **module-level** and take their tensors as
  parameters; a kernel defined inside `@cute.jit` returns `None` and cannot be
  launched. `@cute.jit` may not receive a `torch.cuda.Stream`.

## Multi-stage works with a small K tile

The single-stage pass above used a full 128×128 K tile. The pipelined version
also lands exact — 4 stages, 16 K-tiles — but only once the K tile shrinks to 32:

`_probes/probe_sm120_tma_pipeline_multistage.py` — `M=128, K=512, TK=32`, 4-stage
ring, each stage loads a different K block, all 16 samples exact (`maxdiff 0.0`).

Two things that trip the multi-K version specifically:

### The TMA tensor needs `local_tile` to gain its loop modes

A 2-D `(M, K)` tensor handed straight to `make_tiled_tma_atom` produces a TMA
tensor whose first mode is the *whole* K extent, so `tma_partition` sees
`gmemShape ((128,512))` against `smemShape ((128,32),4)` and refuses. Run the
partitioned tensor through `local_tile` first, exactly like the reference:

```python
gA_mkl = cute.local_tile(tma_a, (M, TK), (None, None))   # (bM,bK,loopM,loopK)
tAsA, tAgA = cpasync.tma_partition(atom_a, 0, cute.make_layout((1,)),
                                   cute.group_modes(sA, 0, 2),
                                   cute.group_modes(gA_mkl, 0, 2))
cute.copy(atom_a, tAgA[(None, 0, kt)], tAsA[(None, pp.index)], tma_bar_ptr=bp)
```

Note the index is `tAgA[(None, 0, kt)]` — loopM=0, loopK=kt.

### The stage count is a byte budget, not a correctness knob

Each fp16 128×TK stage is `128 * TK * 2` bytes per operand. The sm_120 cap is
101376 B, so with two operands:

| K tile | NSTAGE | smem (A+B) | result |
|---|---|---|---|
| 128 | 1 | 65792 B | single-stage PASS |
| 128 | 2 | 131200 B | `CUDA_LAUNCH_INVALID_CONFIG` |
| 128 | 4 | 262272 B | `CUDA_LAUNCH_INVALID_CONFIG` |
| **32** | **4** | **32768 B** | **multi-stage PASS, md 0.0** |

Swizzling does not shrink the byte count; a real pipelined mainloop must shrink K
per stage (or split M). This is why the reference's `ab_stage` tops out at 4 with
its `tile_k`-class tiles.

## The MMA half: the compute path is now verified

The register-side plumbing that returned inf/nan is fixed. A hand-written
single-CTA fp16 GEMM computes a full 128×128 over K=64 with **rel error 2.3e-7,
every element correct** (`_probes/probe_sm120_wmma_gemm_computes.py`). Three
things had to be right:

### Fix A — ldmatrix is mandatory, not optional

`make_fragment_A/B` only *allocates* the register tile. The data has to be copied
smem → register before `cute.gemm`:

```python
lldm = cute.make_tiled_copy_A(
    cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), DT), tiled_mma)
thr_c = ldm.get_slice(tidx)              # tidx, not tidx % 128
tCs_v = thr_c.partition_S(sA)
tCr_v = thr_c.retile(tCrA)
cute.copy(ldm, tCs_v[None, None, kb], tCr_v[None, None, kb])
cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
```

Skipping the `cute.copy` leaves `acc` at all-zeros and the kernel **runs without
error, returning a zero matrix** — the accumulator is allocated but never fed. The
earlier `inf` was a different symptom of the same missing step (uninitialised
registers read as garbage).

### Fix B — `acc_shape = tCc.shape[:3]`, not `tCc.shape`

The C partition carries a 4th pipelined-accumulator mode. `make_rmem_tensor` must
see only the first three, or the register tile is the wrong size.

### Fix C — `get_slice(tidx)` on the reg-copy atoms

The tiled-MMA slice wants `tidx`; the ldmatrix tiled-copy slice also wants `tidx`,
not `tidx % 128`. Using the modulo silently gives the wrong warp's data.

With `atom_layout=(2,2,1)` (4 MMA warps = 128 threads) and the GeForce example's
`MmaF16BF16Op(DT, ACC, (16,8,16))`, the whole chain is correct.

## Still open: joining the two halves — SOLVED

The join now works. `_probes/probe_sm120_wmma_gemm_tma_pipeline.py` — one CTA, 128
threads, full 128×128 over **K=1024**, 32 K-tiles through the 4-stage TMA ring —
computes with **rel error 1.6e-6, every element correct**.

The only thing that stood in the way was a hardcoded ldmatrix major flag, and it
presented as an alignment error:

```
error: 'cute.copy' op src ptr alignment (16 bits) does not meet requirement
       (128 bits) of atom 'ldsm<val_type = f16, num_matrices = 4, t>'
```

That error is **misleading**. The ldsm atom was built with
`LdMatrix8x8x16bOp(True, 4)` unconditionally, but a row-major `(M,K)` A is
**K-major**, so `is_m_major_a()` is `False`. With the wrong flag the atom expects
a transposed layout the swizzled smem does not provide, and the verifier reports it
as a 16-bit-vs-128-bit alignment failure rather than a transpose mismatch. The fix:

```python
la = utils.LayoutEnum.from_tensor(a)
lb = utils.LayoutEnum.from_tensor(b)
...
LdMatrix8x8x16bOp(la.is_m_major_a(), 4)   # False for a row-major A
LdMatrix8x8x16bOp(lb.is_n_major_b(), 4)   # False for a row-major B
```

Both flags must reach the `@cute.kernel` as `cutlass.Constexpr` (a runtime bool
raises `expects the 'transpose' Op parameter to be a bool instance`).

So the full hand-built chain on sm120 is now: **TMA (multi-stage, swizzled smem) →
ldmatrix → WMMA → register epilogue, end to end, exact.** Nothing here needs the
shipped kernels any more.

## NVFP4 on the same skeleton: what carries over and where it stops

Swapping the fp16 WMMA for the block-scaled FP4 op gets **further than the earlier
entry implied, but not to a running kernel**. Verified so far:

| piece | status |
|---|---|
| `make_sm120_blockscaled_mma_op(F4,F4,F32,E4M3,16)` | **builds** |
| `get_permutation_mnk((128,128,128), 16, False)` | **returns `(128, (8,2,2):(1,16,8), 64)`** — matches the earlier NVFP4 record |
| `make_tiled_mma(op, (2,2,1), permutation_mnk=perm)` | **builds** |
| `sm120_make_smem_layout_sfa/sfb(tiled_mma, tile, 16, 1)` | **builds** |
| `partition_fragment_SFA/SFB(sSFA, thr, tidx)` | **returns a 4-D fragment** `((16,4),4,4,1)` |
| list-operand `cute.gemm(..., [tCrA, tCrSFA], [tCrB, tCrSFB], acc)` | **fails rank check** unless A and SF are the same rank |

Two concrete traps hit on the way, both now recorded:

1. **fp4 cannot be assigned element-wise.** `sA[i] = F4(1.0)` raises `Sub-byte
   scalar dereference not supported for type Float4E2M1FN`. Write packed bytes
   (`Uint8(0x22)` is two 1.0 nibbles) or go through a vectorized copy atom / TMA.
   Likewise `tCrA[...] = tCsA[...]` raises `Expected TensorSSA` — the fp4 smem→reg
   move must also be a copy atom, not an assignment.

2. **The SF fragment rank is fixed at 4, and `cute.gemm` demands A match it.**
   The error is
   `'cute.gemm' op expects all operands to have the same rank, but got: A[1]: 4 != A[0]: 2`.
   The shipped `raw.py` satisfies this because its `tCrA` is `((8,2,2),4)` and it
   slices **both** A and SF with `[None, mma_m, k_block_idx]` — i.e. A carries an
   explicit K-block mode from a `tile_k`-tiled, staged smem layout. A flat `(M,K)`
   smem tensor gives a 2-D `tCrA` and no amount of index juggling fixes it.

So the missing piece is structural, not a parameter: **the FP4 path needs the same
`tile_k`-tiled staged-smem + K-block loop the shipped kernel uses** (`ab_stage`,
`tCsA[None,None,None,stage]`, `num_k_blocks = size(tCrA, mode=[2])`). That is why
`raw.py` is 1457 lines and why the honest recommendation is to wrap the verified
kernel (what `kernels/gemm/nvfp4_gemm/sm120/kernel.py` does at 853 TF/s) rather
than re-derive its mainloop. The fp16 skeleton above is the right scaffold; the
FP4 mainloop is a separate, larger port.

### Follow-up round: the port was attempted; everything but the pipeline now lines up

Acting on the recommendation above, the actual port was tried. The result narrows
the remaining gap to exactly one thing — **the mbarrier pipeline** — with every
type/layout/shape question in front of it now answered.

**The rank wall falls once the smem is genuinely staged.** Building `sA` from the
staged layout (`Storage` holding `MemRange[F4, M*TK]`, `get_tensor(outer, swizzle=inner)`)
makes `partition_A(sA)` return a 3-D `tCsA` whose layout is
`((8,2,2),4,2):((1,1024,32),4096,64)` — the trailing `2` **is** the K-block mode
(`tile_k / mma_k = 128/64`). `make_fragment_A(tCsA)` (no extra slicing) then yields a
`tCrA` that `cute.gemm` accepts. So the earlier advice to slice
`tCsA[None,None,None,0]` was the error; the staged tensor must be passed whole.

**The host-side fp4 recipe**, taken from the shipped sm120 blockscaled wrapper
(`dense_blockscaled_gemm_sm120_b12x.py`, its `recast_ptr(dtype=...)` block) — the
four details that each cost a round:

| detail | correct form | what breaks otherwise |
|---|---|---|
| recast to fp4 | `cute.recast_ptr(t.iterator, **dtype=F4**)` | positional arg raises `AttributeError: type object 'Float4E2M1FN' has no attribute 'type'` |
| A/B tensor rank | 3-D `make_ordered_layout((m,k,l), order=(1,0,2))` | `local_tile` then needs 3 coords, and a 2-D layout gives `failed to construct a valid coordinate` |
| SF tensor | `make_tensor(ptr, bs_utils.tile_atom_to_shape_SF((m,k,1), 16))` | feeding a plain `(m, sf_k)` tensor to the TMA atom gives `expected top-level shape equivalence between the SMEM layout and the CTA V-map` |
| SF TMA | `make_tiled_tma_atom(..., (m, tile_k), internal_type=cutlass.Int16)` — tile spans the **full** tile_k, not `(m,1)` | `(m,1)` is the wrong tile and the atom refuses to build |

**TMA partition indexing is one dimension deeper than the fp16 case.** With the
L-mode present, `tAgA` is 5-D `(tile,1,1,loopK,1)`; the copy index is
`tAgA[(None,0,0,kt,0)]`. The fp16 skeleton's `tAgA[(None,0,kt)]` raises
`expects coord and shape of view to be weakly congruent`.

**Where it still stops: the barrier.** `cute.copy(atom, g, s)` with no pipeline
raises `expects a pointer to an mbarrier to be provided via the tma_bar_ptr kw
argument`. Passing a plain `MemRange[Int64,2]` pointer with a hand-rolled
`mbarrier_init` / `mbarrier_arrive_and_expect_tx` / `mbarrier_wait` **compiles and
then hangs** — for the same reason the fp16 load path did before it was switched to
`PipelineTmaAsync`: the expect-tx byte count and the barrier phase both have to be
owned by the pipeline, and a bare barrier gets them wrong. `cute.arch` has no
`alloc_barrier` (the available names are `mbarrier_init`, `mbarrier_arrive`,
`mbarrier_arrive_and_expect_tx`, `mbarrier_expect_tx`, `mbarrier_wait`,
`mbarrier_try_wait`, `mbarrier_test_wait`, `mbarrier_conditional_try_wait`).

So the remaining work is exactly: **drive A, B, SFA and SFB through one
`PipelineTmaAsync`**, with `tx_count` summed over all four copies per stage and SF
counted in Int16 elements. That is a contained change to the working skeleton, not a
re-derivation — but it was not completed in this round, so the entry stays honest:
**fp16 hand-built is solved end to end; NVFP4 hand-built now builds every piece and
fails only at the un-pipelined TMA.**

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB
- `lmsysorg/sglang:dev` container: cutlass-dsl **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- probes `_probes/probe_sm120_tma_pipeline_smoke.py` (single stage load),
  `_probes/probe_sm120_tma_pipeline_multistage.py` (4 stages × 16 K-tiles),
  `_probes/probe_sm120_wmma_gemm_computes.py` (MMA only, rel 2.3e-7), and
  `_probes/probe_sm120_wmma_gemm_tma_pipeline.py` (full GEMM, K=1024, rel 1.6e-6)
  — all PASS
- official `dense_gemm.py --mnkl 1024,1024,1024,1 --tile_shape_mnk 128,128,64` —
  PASS, 125 µs
