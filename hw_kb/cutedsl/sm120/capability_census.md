# What sm120 actually has: a 39-row capability census

Every entry in this tree so far was earned one hard-won kernel at a time. This one is the
map underneath them: a single probe that constructs **every CuTeDSL primitive worth knowing
about on this card** and records the result. Half the rows say OK; the other half say the
interface exists upstream but is *arch-gated off here*, which is just as actionable.

The census matters because the wrong mental model is expensive. `sm100/` and `sm90/` in this
KB contain 131 entries built on `tcgen05`/TMEM and `wgmma` respectively — **none of which
exist on sm_120a**. Without the census, "does sm120 support this?" gets answered by reading
an sm100 file and trying it, which is a lost afternoon each time.

## The five things sm120 does *not* have

| mechanism | status here | consequence |
|---|---|---|
| `tcgen05` (TMEM MMA) | construct refused; `admissible_archs = [sm_100a, sm_100f, sm_103a, sm_107a, sm_110a]` | no tensor-memory accumulator, no 2-CTA MMA, no `tcgen05.mma` |
| `wgmma` namespace | absent entirely | no warpgroup MMA; `warpgroup` module has no `wgmma` symbol |
| `MmaI8Op` | not in this cutlass 4.8.0 | no int8 tensor-core path exposed |
| `LdMatrix16x16x8b` | constructs then fails IR verification | only the 8x8x16b / 8x16x8b / 16x8x8b forms work |
| `smem` 232 KB | **101376 B** on sm_120 vs 232448 B on sm_100 | every stage count / tile size from sm100 is wrong by 2.3x |

The smem row is the one that silently breaks ported kernels: a config that "fit" on sm100
now fails a `requires N bytes of shared memory` check, or worse, gets `__launch_bounds__`
wrong.

## The tensor-core family it *does* have

All warp-level `mma.sync`, all `(16, 8, *)`:

| op | instruction shape | operands | arch list |
|---|---|---|---|
| `MmaF16BF16Op` | (16, 8, 16) | fp16 / bf16 | arch-neutral |
| `MmaFP8Op` | (16, 8, 32) | e4m3 | arch-neutral (SM89 lineage) |
| `MmaTF32Op` | (16, 8, 8) | tf32 | arch-neutral |
| `MmaMXF8Op` | (16, 8, 32) | e4m3 + UE8M0 scales, sf_vec 32 | sm_120a/f, sm_121a/f |
| `MmaMXF4Op` | (16, 8, 64) | e2m1 + UE8M0 scales, sf_vec 32 | sm_120a/f, sm_121a/f |
| `MmaMXF4NVF4Op` | (16, 8, 64) | e2m1 + UE4M3 scales, sf_vec 16 | sm_120a/f, sm_121a/f |
| `MmaF16BF16SparseOp` | (16, 8, 16) | 2:4 structured fp16/bf16 | arch-neutral |

Note the split: the three **block-scaled** ops carry an explicit `admissible_archs` list that
is *only* GeForce Blackwell (120/121), while plain fp16/fp8/tf32 do not carry the attribute at
all — they are the universal path. That is why an fp16 kernel written for sm90 often just
runs here, and an fp4 kernel almost never does.

## Copy, TMA, and cluster are fully present

The surprise: this consumer part has the whole TMA/bulk-copy family. Every one constructs:

```
cpasync.CopyG2SOp                          cp.async gm->sm
cpasync.CopyBulkG2SOp / S2GOp / S2SOp      bulk (non-tensor) g2s, s2g, smem<->smem
cpasync.CopyBulkTensorTileG2SOp / S2GOp    TMA tile copy both directions
cpasync.CopyBulkTensorTileG2SMulticastOp   TMA multicast (needs a cluster to target)
cpasync.CopyBulkTensor2DGather4G2SOp       TMA gather4
cpasync.make_tiled_tma_atom                builds a TMA descriptor
cpasync.make_im2col_tma_atom               im2col TMA descriptor
cpasync.tma_partition                      partitions a TMA atom across threads
warp.LdMatrix8x8x16bOp / StMatrix8x8x16bOp x4 ldmatrix / stmatrix
warp.LdMatrix8x16x8bOp                     the fp8-shaped ldmatrix
nvgpu.CopyUniversalOp / CopyR2SOp / CopyS2ROp
```

There is **no `cute.make_tma_copy`** — that name exists in sm90-era docs. The constructor is
`cpasync.make_tiled_tma_atom` (or `make_im2col_tma_atom`), and partitioning is
`cpasync.tma_partition`. Chasing `make_tma_copy` is a real time sink; it simply is not here.

Cluster primitives are all callable: `cluster_arrive`, `cluster_arrive_relaxed`,
`cluster_wait`, `cluster_size`, `block_idx_in_cluster`, `cluster_idx`. Being callable is not
the same as being *useful* on this box — see the cluster entry for why the two-card machine
cannot exploit multicast.

## Fabric

`peer access 0->1 = True` — P2P copies work — but the link is SYS (no NVLink). Everything in
`sm100/multimem_*` (NVLink multimem addressing) is therefore unreachable, not merely
unmeasured.

## How to re-run

`_probes/probe_sm120_capability_census.py` prints the 39-row table. A row flipping from
REFUSED to OK after an image bump is the signal that a port became possible; a row flipping
the other way is a regression worth a bug report.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB per card, 2 cards SYS-linked
- `lmsysorg/sglang:dev`: cutlass **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- Probe: `_probes/probe_sm120_capability_census.py` — 34 OK / 5 REFUSED
