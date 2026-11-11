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

## Still open: joining the two halves

The TMA load path needs a **swizzled** smem layout (`sm90_utils.make_smem_layout_a`
/ `_b`), because TMA descriptors expect the swizzle. The ldmatrix atom rejects that
layout at IR verify:

```
error: 'cute.copy' op src ptr alignment (16 bits) does not meet requirement
       (128 bits) of atom 'ldsm<val_type = f16, num_matrices = 4, t>'
```

The non-swizzled probe above ldmatrixes fine, so this is purely the swizzled-smem
hand-off — the `partition_S` of a composed (swizzled) tensor loses the 128-bit
alignment fact the ldsm atom needs. The official `dense_gemm.py` does both in one
kernel and PASSes, so the recipe exists; the missing piece is how it threads the
swizzled `tCsA_view[(None,None,None,stage)]` into `smem_tiled_copy_A` without the
alignment being lost. That is the next thing to copy verbatim rather than derive.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB
- `lmsysorg/sglang:dev` container: cutlass-dsl **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- probes `_probes/probe_sm120_tma_pipeline_smoke.py` (single stage load),
  `_probes/probe_sm120_tma_pipeline_multistage.py` (4 stages × 16 K-tiles), and
  `_probes/probe_sm120_wmma_gemm_computes.py` (fp16 GEMM rel 2.3e-7) — all PASS
- official `dense_gemm.py --mnkl 1024,1024,1024,1 --tile_shape_mnk 128,128,64` —
  PASS, 125 µs
