# TMA and the copy family on sm120: all present, and the 99 KB ceiling

This consumer part ships the entire TMA / bulk-copy family, which is not what the "GeForce is
a cut-down datacenter chip" intuition predicts. What it does **not** ship is the smem to feed
the shapes those copies were designed for: **101376 bytes** of dynamic smem per CTA, against
232448 on sm_100. Every stage count and tile size copied from an sm100 kernel is wrong here.

## The names are not the sm90 names

The single most costly detail: **there is no `cute.make_tma_copy`**. That symbol belongs to an
older API and does not exist in cutlass-dsl 4.8.0. The TMA builders live under `cpasync`:

| what you want | actual symbol here |
|---|---|
| build a TMA descriptor | `cpasync.make_tiled_tma_atom` |
| im2col TMA descriptor | `cpasync.make_im2col_tma_atom` |
| split a TMA atom across threads | `cpasync.tma_partition` |
| TMA load tile (gm → smem) | `cpasync.CopyBulkTensorTileG2SOp` |
| TMA store tile (smem → gm) | `cpasync.CopyBulkTensorTileS2GOp` |
| TMA multicast (needs a cluster) | `cpasync.CopyBulkTensorTileG2SMulticastOp` |
| TMA gather4 | `cpasync.CopyBulkTensor2DGather4G2SOp` |
| non-tensor bulk copy | `cpasync.CopyBulkG2SOp` / `CopyBulkS2GOp` / `CopyBulkS2SOp` |
| classic cp.async | `cpasync.CopyG2SOp` |
| prefetch a descriptor | `cpasync.prefetch_descriptor` |

All of them construct on this card. `CopyBulkS2SOp` (smem-to-smem bulk) is a sm90+ feature and
is present; it is the mechanism behind distributed-smem tricks, though the cluster story
separately limits what you can do with it here.

## ldmatrix / stmatrix

Present in three widths, and one trap:

```
warp.LdMatrix8x8x16bOp(transposed, num_matrices)   x1/x2/x4    the fp16/bf16 workhorse
warp.StMatrix8x8x16bOp(transposed, num_matrices)   x1/x2/x4
warp.LdMatrix8x16x8bOp()                                       the fp8-shaped variant
warp.LdMatrix16x8x8bOp()                                       constructs
warp.LdMatrix16x16x8bOp()                          *** constructs then fails IR verify ***
```

`LdMatrix16x16x8bOp` is the trap: the constructor succeeds and the failure only surfaces at
IR verification, far from the call site. If a copy does not lower and the atom "exists", check
whether it verified, not just whether it built.

For anything irregular, `nvgpu.CopyUniversalOp` (scalar) plus `CopyR2SOp` / `CopyS2ROp`
(register↔smem) are always available, at the obvious cost.

## The smem ceiling, concretely

`cutlass.memory.get_smem_capacity_in_bytes("sm_120")` → **101376**. Cell by cell:

| | sm_120 | sm_100 |
|---|---|---|
| dynamic smem / CTA | 101376 B (99 KB) | 232448 B (227 KB) |

What that costs in practice: a triple-buffered 128×128 fp16 tile stage is
`128*128*2 = 32768` B. Four stages of two operands is `4*2*32768 = 262144` B — impossible here
and comfortable on sm100. On sm120 the same kernel runs 1–2 stages, or drops tile_k, or uses a
narrower N tile. Any ported config must be re-derived from 99 KB, not adjusted by feel.

The block-scaled kernel already encodes this: its tactic search filters candidate stage counts
by `stages * smem_bytes <= sm_smem`, and on this card that filter is the binding constraint
more often than the grid shape.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB, no NVLink
- `lmsysorg/sglang:dev`: cutlass **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- Probe: `_probes/probe_sm120_capability_census.py` (the copy / smem rows)
