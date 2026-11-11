"""Minimal tcgen05 UMMA CTA-pair skeleton (Blackwell sm100a).

Source recipe: hw_kb/cutedsl/sm100/tcgen05_umma_issue_cta_pair.md
Mechanism: `tcgen05.mma` has a `cta_group=2` variant where a CTA pair completes
one MMA together. The accumulator lives in TMEM (addressed by 32-bit columns) and
must be allocated before use; completion is signalled by `tcgen05.commit` onto an
mbarrier, and a hand-written pair commit must reach BOTH the leader's and the
follower's barrier (mask_self | mask_peer).

!!! IMPORTANT HARDWARE CAVEAT !!!
This recipe is scoped to sm100 (Blackwell DATACENTER). The RTX 6000D in this
repo is sm120, which has NO tcgen05/wgmma -- its MMA path is plain HMMA
(see hw_kb/tilelang/sm120/index.md: "MMA path is neither wgmma nor tcgen05").
So this skeleton will NOT run on the 6000D. It is here for an sm100 part.

Run ONLY on sm100-class hardware inside an image that ships CuTeDSL:

    python3 tcgen05_2cta_mma.py
"""

from __future__ import annotations

import cutlass
import cutlass.utils as utils
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05

# ---------------------------------------------------------------------------
# geometry (single-CTA and pair forms differ only in CtaGroup + cluster)
# ---------------------------------------------------------------------------
IO_DTYPE = cutlass.Float16
ACC_DTYPE = cutlass.Float32
TILE_M, TILE_N, TILE_K = 128, 128, 64
TILER_MNK = (TILE_M, TILE_N, TILE_K)
STAGES = 2

# harbour whether a machine is even allowed to build this
REQUIRED_CAPABILITY = (10, 0)          # sm100. NOT sm120.


def assert_eligible(capability: tuple[int, int]) -> None:
    major, minor = capability
    if (major, minor) != REQUIRED_CAPABILITY:
        raise RuntimeError(
            f"tcgen05 UMMA needs sm{REQUIRED_CAPABILITY[0]}{REQUIRED_CAPABILITY[1]}; "
            f"this device reports sm{major}{minor}. On sm120 the MMA path is HMMA, "
            "so use a plain mma.sync GEMM (or TileLang) instead."
        )


def build_pair_mma(CtaGroup):
    """The only difference between the single-CTA and pair forms is CtaGroup.

    In the real kernel you pass this into the tiled_mma you construct for the
    mainloop; required shapes are:
      * smem operand  : allocate_tensor(layout=a_smem_layout.outer, swizzle=inner)
      * fragment A/B  : tiled_mma.make_fragment_A/B(sA) -> (MMA, MMA_MN, MMA_K, STAGE)
      * accumulator   : tiled_mma.make_fragment_C(partition_shape_C(tiler_mn))
      * TMEM          : TmemAllocator(..., two_cta_tmem_dealloc_mbar_ptr=...) then allocate
    """
    return CtaGroup


def main():
    print("cutlass", getattr(cutlass, "__version__", "?"))
    print("pair op :", cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO))
    print("mma group (pair):", build_pair_mma(tcgen05.CtaGroup.TWO))

    # eligibility gate first: do not burn a compile on the wrong arch
    try:
        assert_eligible((12, 0))       # this repo's RTX 6000D
    except RuntimeError as exc:
        print("gate:", exc)

    # What a full example still needs, in the recipe's order:
    #   1) smem: allocate_tensor(element_type=IO_DTYPE,
    #            layout=a_smem_layout.outer, byte_alignment=128,
    #            swizzle=a_smem_layout.inner)     <- NOT the composed layout
    #   2) atoms: CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)
    #            + PipelineTmaUmma.create(..., tx_count=bytes*size(vmnk[0]),
    #              cta_layout_vmnk=...)
    #   3) fragments: tCrA/tCrB = tiled_mma.make_fragment_A/B(sA/sB)
    #                 tCtAcc    = tiled_mma.make_fragment_C(partition_shape_C(...))
    #   4) TMEM: utils.TmemAllocator(storage.tmem_holding_buf.ptr,
    #              two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr);
    #              tmem.allocate(512); bind the ptr into tCtAcc
    #   5) mainloop: for k in range_constexpr(...): cute.gemm(tiled_mma, tCrA, tCrB, tCtAcc)
    #   6) commit: hand-written pair -> tcgen05.commit(mbar_ptr, mask_self|mask_peer, TWO)
    #   7) teardown: ab_producer.tail() / acc_producer.tail(); tmem.free(ptr)
    #      (missing teardown => the SECOND launch of the same entry never returns)
    print("todo: smem -> atoms -> fragments -> TMEM -> mainloop -> commit -> teardown")


if __name__ == "__main__":
    main()
