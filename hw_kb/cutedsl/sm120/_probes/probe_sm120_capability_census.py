"""sm120 capability census: what exists, what refuses, and what actually runs.

Every row was produced by calling the API on this card, not by reading source. A
"REFUSED" row is as much a result as an "OK" one: it tells the next person not to spend a
day on a path that cannot exist.

Run in the sglang container (cutlass 4.8.0):
    python3 probe_sm120_capability_census.py
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp, tcgen05

F4, F8, F16, BF16, F32, TF32 = (
    cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, cutlass.Float16,
    cutlass.BFloat16, cutlass.Float32, cutlass.TFloat32)
U8 = cutlass.Uint8

rows = []


def probe(name, fn):
    try:
        detail = fn()
        rows.append((name, "OK", "" if detail is None else str(detail)))
    except Exception as e:
        rows.append((name, "REFUSED", f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"))


def must(x, what):
    if not x:
        raise AttributeError(what)
    return x


# ---------------------------------------------------------------- warp MMA family
probe("warp.MmaF16BF16Op(F16)", lambda: warp.MmaF16BF16Op(F16, F32, (16, 8, 16)) and "shape (16,8,16)")
probe("warp.MmaF16BF16Op(BF16)", lambda: warp.MmaF16BF16Op(BF16, F32, (16, 8, 16)) and "shape (16,8,16)")
probe("warp.MmaFP8Op", lambda: warp.MmaFP8Op(F8, F32, (16, 8, 32)) and "shape (16,8,32), SM89 lineage")
probe("warp.MmaMXF8Op",
      lambda: "archs=" + str([str(a) for a in warp.MmaMXF8Op.admissible_archs])
      if hasattr(warp.MmaMXF8Op, "admissible_archs")
      else warp.MmaMXF8Op(F8, F32, cutlass.Float8E8M0FNU) and "(16,8,32) sf_vec 32")
probe("warp.MmaMXF4Op",
      lambda: "archs=" + str([str(a) for a in warp.MmaMXF4Op.admissible_archs])
      if hasattr(warp.MmaMXF4Op, "admissible_archs")
      else warp.MmaMXF4Op(F4, F32, cutlass.Float8E8M0FNU) and "(16,8,64) UE8M0")
probe("warp.MmaMXF4NVF4Op",
      lambda: warp.MmaMXF4NVF4Op(F4, F32, F8) and "(16,8,64) UE4M3 sf_vec 16")
probe("warp.MmaTF32Op", lambda: warp.MmaTF32Op(TF32, (16, 8, 8)) and "shape (16,8,8)")
probe("warp.MmaI8Op", lambda: must(getattr(warp, "MmaI8Op", None), "no MmaI8Op in this cutlass"))
probe("warp.MmaF16BF16SparseOp",
      lambda: must(getattr(warp, "MmaF16BF16SparseOp", None), "no sparse op") and "present")

# ---------------------------------------------------------------- copy / ld / st
probe("cpasync.CopyG2SOp", lambda: cpasync.CopyG2SOp() and "cp.async g2s")
probe("cpasync.CopyBulkG2SOp", lambda: cpasync.CopyBulkG2SOp() and "bulk (TMA, non-tensor)")
probe("cpasync.CopyBulkS2GOp", lambda: cpasync.CopyBulkS2GOp() and "bulk s2g")
probe("cpasync.CopyBulkS2SOp", lambda: cpasync.CopyBulkS2SOp() and "bulk s2s (smem<->smem)")
probe("cpasync.CopyBulkTensorTileG2SOp",
      lambda: cpasync.CopyBulkTensorTileG2SOp() and "TMA tile g2s")
probe("cpasync.CopyBulkTensorTileS2GOp",
      lambda: cpasync.CopyBulkTensorTileS2GOp() and "TMA tile s2g")
probe("cpasync.CopyBulkTensorTileG2SMulticastOp",
      lambda: cpasync.CopyBulkTensorTileG2SMulticastOp() and "TMA multicast")
probe("cpasync.CopyBulkTensor2DGather4G2SOp",
      lambda: cpasync.CopyBulkTensor2DGather4G2SOp() and "TMA gather4")
probe("cpasync.make_tiled_tma_atom", lambda: must(cpasync.make_tiled_tma_atom, "absent") and "callable")
probe("cpasync.make_im2col_tma_atom", lambda: must(cpasync.make_im2col_tma_atom, "absent") and "callable")
probe("cpasync.tma_partition", lambda: must(cpasync.tma_partition, "absent") and "callable")
probe("warp.LdMatrix8x8x16bOp", lambda: warp.LdMatrix8x8x16bOp(True, 4) and "x4")
probe("warp.StMatrix8x8x16bOp", lambda: warp.StMatrix8x8x16bOp(True, 4) and "x4")
probe("warp.LdMatrix16x16x8bOp", lambda: warp.LdMatrix16x16x8bOp() and "16x16x8b")
probe("warp.LdMatrix8x16x8bOp", lambda: warp.LdMatrix8x16x8bOp() and "8x16x8b")
probe("nvgpu.CopyUniversalOp", lambda: cute.nvgpu.CopyUniversalOp() and "scalar fallback")
probe("nvgpu.CopyR2SOp", lambda: cute.nvgpu.CopyR2SOp() and "reg->smem")
probe("nvgpu.CopyS2ROp", lambda: cute.nvgpu.CopyS2ROp() and "smem->reg")

# ---------------------------------------------------------------- arch gates
probe("tcgen05.MmaMXF4NVF4Op.admissible_archs",
      lambda: str([str(a) for a in tcgen05.MmaMXF4NVF4Op.admissible_archs]))
probe("tcgen05.MmaMXF4NVF4Op on THIS card",
      lambda: tcgen05.MmaMXF4NVF4Op(F4, (128, 128, 64), tcgen05.CtaGroup.ONE,
                                    tcgen05.OperandSource.SMEM) and "constructed?!")
probe("wgmma namespace", lambda: must(getattr(cute.nvgpu, "wgmma", None)
                                      or getattr(warp, "wgmma", None), "no wgmma on sm120"))

# ---------------------------------------------------------------- cluster / sync
for nm in ("cluster_arrive", "cluster_arrive_relaxed", "cluster_wait", "cluster_size",
           "block_idx_in_cluster", "cluster_idx"):
    probe(f"cute.arch.{nm}", lambda nm=nm: must(getattr(cute.arch, nm, None), "absent") and "callable")

# ---------------------------------------------------------------- smem
probe("smem capacity sm_120",
      lambda: str(cutlass.memory.get_smem_capacity_in_bytes("sm_120")) + " B")
probe("smem capacity sm_100 (for contrast)",
      lambda: str(cutlass.memory.get_smem_capacity_in_bytes("sm_100")) + " B")

# ---------------------------------------------------------------- fabric
def _fabric():
    if torch.cuda.device_count() < 2:
        return f"only {torch.cuda.device_count()} device visible"
    p2p = torch.cuda.can_device_access_peer(0, 1)
    return f"devices={torch.cuda.device_count()} peer0->1={p2p}"


probe("peer access", _fabric)


print("=" * 104, flush=True)
print(f"{'capability':56s} {'state':9s} detail", flush=True)
print("-" * 104, flush=True)
for n, s, d in rows:
    print(f"{n:56s} {s:9s} {d}", flush=True)
ok = sum(1 for _, s, _ in rows if s == "OK")
ref = sum(1 for _, s, _ in rows if s == "REFUSED")
print(f"\nOK={ok} REFUSED={ref} total={len(rows)}", flush=True)
print("CENSUS DONE", flush=True)
