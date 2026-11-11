"""Minimal TMA-multicast cluster GEMM skeleton (Hopper sm90a / Blackwell sm100a).

Source recipe: hw_kb/cutedsl/sm90/tma_multicast_cluster.md
Mechanism: with cluster=[N,1,1], each CTA fetches only 1/N of a gmem tile and the
TMA unit multicasts it into every CTA's smem, so HBM read volume of that operand
drops to 1/N while every CTA still sees the whole tile.

This is a SKELETON: it builds the multicast atom and the pipeline counts, and it
runs the required cluster arrival handshake. The MMA body is left as the single
place to fill in (the recipe deliberately covers multicast only, not MMA).

Run inside an image that ships CuTeDSL (this repo's box: lmsysorg/sglang:dev,
cutlass 4.8.0):

    python3 -c "import cutlass; print(cutlass.__version__)"
    python3 tma_multicast_gemm.py

Two knobs must move TOGETHER or the multicast buys nothing (recipe's own warning):
  * host launch: the cluster topology
  * atom:         num_multicast == cluster extent along that axis

Leaving them out of step "raises no error and gives no wrong numbers; the barrier
just never fills" -- which shows up as a hang. Assert the pair below.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
TILE_M, TILE_N, TILE_K = 128, 128, 64
CLUSTER_M, CLUSTER_N = 1, 2          # cluster [2,1,1] laid out along N
NUM_MULTICAST = CLUSTER_N            # A multicasts along N
STAGES = 2


def build_a_multicast_atom(dtype_a, m, k, a_smem_layout_staged, cta_tiler):
    """The A-side load atom: one fetch per CTA, multicast to the whole cluster.

    `CopyBulkTensorTileG2SMulticastOp` is a DIFFERENT class from the plain
    `CopyBulkTensorTileG2SOp`; passing num_multicast != 1 to the plain one is
    rejected ("expects num_multicast to be 1 for non multicast G2S copies").
    """
    a = cute.runtime.make_fake_tensor(dtype_a, (m, k))

    # the per-stage smem face (the staged layout minus the stage mode)
    smem_one = cute.slice_(a_smem_layout_staged, (None, None, 0))

    tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SMulticastOp(),
        a,
        smem_one,
        cta_tiler,
        num_multicast=NUM_MULTICAST,
    )
    return tma_atom_a, tma_tensor_a


def build_b_plain_atom(dtype_b, k, n, b_smem_layout_staged, cta_tiler):
    """The B side does not multicast in this layout, so it uses the plain op."""
    b = cute.runtime.make_fake_tensor(dtype_b, (k, n))
    smem_one = cute.slice_(b_smem_layout_staged, (None, None, 0))
    tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        b,
        smem_one,
        cta_tiler,
        num_multicast=1,
    )
    return tma_atom_b, tma_tensor_b


def cluster_pipeline(consumer_warps: int):
    """PipelineTmaAsync with the cluster layout, so the empty barriers are
    accounted once per CTA in the cluster instead of once per CTA alone."""
    cta_layout_mnk = cute.make_layout((CLUSTER_M, CLUSTER_N, 1))
    cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))

    mcast_size = NUM_MULTICAST
    mainloop = pipeline.PipelineTmaAsync.create(
        num_stages=STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_warps * mcast_size
        ),
        tx_count=1,                       # placeholder; real count = bytes per stage
        cta_layout_vmnk=cta_layout_vmnk,
    )
    return mainloop, cta_layout_vmnk


def cluster_handshake():
    """Required before the first multicast load: publish the mbarrier inits to
    the whole cluster, then meet at the cluster barrier."""
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()


def check_knobs():
    assert NUM_MULTICAST == CLUSTER_N, (
        "num_multicast must equal the cluster extent along the multicast axis; "
        "mismatch is silent until the barrier hangs"
    )


def main():
    print("cute", cutlass.__version__ if hasattr(cutlass, "__version__") else "?")
    check_knobs()
    print(f"cluster=[{CLUSTER_M},{CLUSTER_N},1] num_multicast={NUM_MULTICAST}")
    print("skeleton only: fill the MMA body, see recipe 'tma_multicast_cluster'")

    # What a full example still needs, in the order the recipe lists them:
    #   1) allocate staged smem for A and B and build a_smem_layout_staged
    #   2) build_a_multicast_atom(...) / build_b_plain_atom(...)
    #   3) cluster_pipeline(...) with the real tx_count
    #   4) cluster_handshake() once, before the first load
    #   5) mainloop over K with cute.gemm inside
    print("todo: smem staging -> atoms -> pipeline -> handshake -> mainloop -> epilogue")


if __name__ == "__main__":
    main()
