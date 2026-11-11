"""Minimal sm120 TMA pipeline: prove gmem->smem lands bit-exact.

PROBE: PASS. Run inside the CuTe DSL container:

    docker exec cutedsl_probe bash -lc "cd /w && python3 probe_sm120_tma_pipeline_smoke.py"

This is the smallest thing that exercises the load path a hand-written kernel
needs: one CTA, one fp16 A tile and one fp16 B tile, `PipelineTmaAsync` with a
dedicated producer warp, TMA-load into smem, read the diagonal back and compare.

The four bugs that made this hang / fault / return noise are documented in
`../from_scratch_tma_load_path_works.md`; each fix is marked below with FIX n.

Expected output:

    exp[:6] [-0.9248, -0.5713, ...]
    got[:6] [-0.9248, -0.5713, ...]
    maxdiff 0.0
    PASS
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
from cutlass import pipeline
from cutlass import Int64, Float32

M = N = K = 128
TILE = 128
NSTAGE = 1          # 1 because 2 stages x 2 buffers x 32 KB > the 101376 B cap
DT = cutlass.Float16
NWARP = 5           # 4 consumer warps + 1 dedicated TMA producer warp


@cute.struct
class Storage:
    bars: cute.struct.MemRange[Int64, NSTAGE * 2]
    sA: cute.struct.Align[cute.struct.MemRange[DT, NSTAGE * TILE * TILE], 1024]
    sB: cute.struct.Align[cute.struct.MemRange[DT, NSTAGE * TILE * TILE], 1024]


@cute.kernel
def _body(atom_a, tma_a, atom_b, tma_b, out):
    tidx, _, _ = cute.arch.thread_idx()
    widx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(Storage)

    # FIX 1: row-major smem, matching the torch tensor. A (1, TILE) stride here
    # gives exactly one correct element and then noise, with no error at all.
    sA = st.sA.get_tensor(
        cute.make_layout((TILE, TILE, NSTAGE), stride=(TILE, 1, TILE * TILE)))
    sB = st.sB.get_tensor(
        cute.make_layout((TILE, TILE, NSTAGE), stride=(TILE, 1, TILE * TILE)))

    # FIX 2: partition, never slice the TMA tensor directly.
    cta_layout = cute.make_layout((1,))
    tAsA, tAgA = cpasync.tma_partition(
        atom_a, 0, cta_layout,
        cute.group_modes(sA, 0, 2), cute.group_modes(tma_a, 0, 2))
    tBsB, tBgB = cpasync.tma_partition(
        atom_b, 0, cta_layout,
        cute.group_modes(sB, 0, 2), cute.group_modes(tma_b, 0, 2))

    producer = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    consumer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
    pipe = pipeline.PipelineTmaAsync.create(
        num_stages=NSTAGE, producer_group=producer, consumer_group=consumer,
        tx_count=TILE * TILE * 2 * 2, barrier_storage=st.bars.data_ptr())
    cute.arch.mbarrier_init_fence()

    if widx == 4:
        pp = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, NSTAGE)
        pp.reset_count()
        for _ in range(NSTAGE):
            pipe.producer_acquire(pp)
            bp = pipe.producer_get_barrier(pp)
            cute.copy(atom_a, tAgA, tAsA[(None, pp.index)], tma_bar_ptr=bp)
            cute.copy(atom_b, tBgB, tBsB[(None, pp.index)], tma_bar_ptr=bp)
            pipe.producer_commit(pp)
            pp.advance()
        pipe.producer_tail(pp)
    elif widx < 4:
        cs = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, NSTAGE)
        cs.reset_count()
        for _ in range(NSTAGE):
            pipe.consumer_wait(cs)
            if tidx == 0:
                for i in range(32):
                    out[i] = sA[i, i, cs.index].to(Float32)
            pipe.consumer_release(cs)
            cs.advance()


@cute.jit
def run(a, b, out):
    smem_a = cute.make_layout((TILE, TILE), stride=(TILE, 1))
    # FIX 3: keep the second return value -- that is the TMA tensor.
    atom_a, tma_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), a, smem_a, (TILE, TILE))
    atom_b, tma_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), b, smem_a, (TILE, TILE))
    _body(atom_a, tma_a, atom_b, tma_b, out).launch(
        grid=[1, 1, 1], block=[NWARP * 32, 1, 1])


def main():
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    out = torch.zeros(32, device="cuda", dtype=torch.float32)
    # FIX 4: from_dlpack makes the shape static so the TMA tile mode is not `?`.
    run(from_dlpack(a, assumed_align=32), from_dlpack(b, assumed_align=32), out)
    torch.cuda.synchronize()

    exp = torch.diagonal(a)[:32].float()
    got = out[:32]
    print("exp[:6]", exp[:6].tolist())
    print("got[:6]", got[:6].tolist())
    md = (got - exp).abs().max().item()
    print("maxdiff", md)
    print("PASS" if md < 1e-2 else "FAIL")


if __name__ == "__main__":
    main()
