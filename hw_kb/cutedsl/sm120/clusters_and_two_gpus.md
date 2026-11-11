# Clusters work here; the second GPU does not help them

A cluster of CTAs is a single-GPU concept: all CTAs in a cluster must land on the same GPC of
the same device. So on a two-card box the useful question is not "can I multicast across
cards" (you cannot — a cluster is device-local by definition) but "how big a cluster can this
part boot, and how fast can I move data between the two GPUs outside one".

Both answers are measured. The cluster story is healthier than expected; the inter-GPU story is
the bottleneck.

## Cluster launches, up to the limit tested

A minimal kernel that reads `cluster_size` / `cluster_idx` and touches the barrier at
`cluster_arrive` / `cluster_wait` boots at every size tried:

| cluster (M,N) | CTAs | result |
|---|---|---|
| (1,1) | 1 | booted, size 1 |
| (2,1) | 2 | booted, size 2 |
| (1,2) | 2 | booted, size 2 |
| (2,2) | 4 | booted, size 4 |
| (4,1) | 4 | booted, size 4 |
| (8,1) | 8 | booted, size 8 |
| (16,1) | 16 | booted, size 16 |

The card reports 156 SMs and `max_threads_per_block = 1024`. The launch form that actually
works is:

```python
kernel(...).launch(grid=[gm, gn, 1], block=[128, 1, 1],
                   cluster=[cm, cn, 1] if (cm * cn) > 1 else None,
                   smem=0, stream=stream)
```

The `kernel[grid, cluster, block](...)` bracket form from sm90-era docs does **not** exist in
this cutlass — it raises `TypeError: 'function' object is not subscriptable`.

## But every consumer feature built on clusters is single-device

What a cluster buys you is distributed shared memory (DSMEM) and TMA multicast **among its
CTAs**, which are all on one GPU. The `CopyBulkTensorTileG2SMulticastOp` the census found
multicasts one gmem load to several CTAs' smem — again, all within the cluster, all within one
device. None of this reaches GPU1.

So the sm100 `multimem_*` entries in this KB (NVLink multimem load-reduce-store) have **no
sm120 analogue that spans cards**. On a machine with NVLink you would use multimem; here the
cards are SYS-linked and there is nothing to translate to.

## The inter-GPU path: P2P yes, bandwidth no

- Peer access 0→1 and 1→0: **yes** (`can_device_access_peer` true both ways).
- A 256 MiB `copy_` across the pair: **45.5 GB/s** (5.90 ms). This matches the earlier 44.7
  GB/s direct peer-copy reading — the link is PCIe/SYS, not NVLink.

For scale: a single RTX 6000D does >1 TB/s to its own HBM. The cross-card link is ~4.5% of
that. Any split across the two cards — tensor parallel, expert parallel, pipeline — pays this
on every token boundary. That is why the serving config for this box prefers `--disable-custom-
all-reduce` and treats the pair as two independent replicas rather than one wide GPU.

## What to do instead of clusters across cards

The cluster primitives are worth using *within* one GPU: DSMEM lets a 2-CTA cluster share an
operand through smem instead of round-tripping gmem, and multicast halves the gmem read traffic
for a shared tile. On a 156-SM part that can matter for big GEMMs. Just keep the whole cluster
on one device and use P2P copies (or NCCL over PCIe) for the card-to-card step.

## Measured on

- RTX 6000D ×2, compute capability **(12, 0)** = sm_120a, 85651 MiB per card
- 156 SMs, 101376 B dynamic smem/CTA, cluster up to (16,1) booted
- inter-card: `can_device_access_peer` true both ways, 45.5 GB/s copy_, SYS link (no NVLink)
- `lmsysorg/sglang:dev`: cutlass **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- Probe: `_probes/probe_sm120_cluster_reality.py`
