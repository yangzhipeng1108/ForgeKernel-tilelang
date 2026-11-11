"""Multi-stage TMA pipeline over K tiles: prove the handshake across many loads.

PROBE: PASS. Run inside the CuTe DSL container:

    docker exec cutedsl_probe bash -lc "cd /w && python3 probe_sm120_tma_pipeline_multistage.py"

Where probe_sm120_tma_pipeline_smoke.py proves a single load lands bit-exact,
this proves the pipelined version: 4 stages, 16 K-tiles, small K tile (32) so the
4-stage ring fits in the 101376 B smem cap. Each stage loads a different K block
and the per-tile sample is checked against torch -- all 16 exact (maxdiff 0.0).

Two things this pins down:
 - the TMA tensor needs local_tile(...) to become the 5-D (bM,bK,loopM,loopK,loopL)
   form the reference uses, or tma_partition sees the whole K extent (128,512)
   instead of one tile and refuses;
 - tAgA is indexed tAgA[(None, 0, kt)] once partitioned -- loopM=0, loopK=kt.

See ../from_scratch_tma_load_path_works.md for the four fixes this builds on.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
from cutlass import pipeline
from cutlass import Int64, Float32

M = 128
K = 512
TK = 32
NKT = K // TK
NSTAGE = 4
DT = cutlass.Float16
NWARP = 5
TILE_MNK = (M, M, TK)


@cute.struct
class Storage:
    bars: cute.struct.MemRange[Int64, NSTAGE * 2]
    sA: cute.struct.Align[cute.struct.MemRange[DT, NSTAGE * M * TK], 1024]


@cute.kernel
def _body(atom_a, tma_a, out):
    tidx, _, _ = cute.arch.thread_idx()
    widx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(Storage)
    sA = st.sA.get_tensor(
        cute.make_layout((M, TK, NSTAGE), stride=(TK, 1, M * TK)))

    # 5-D local tiles, exactly like the reference: (bM,bK,loopM,loopK,loopL)
    gA_mkl = cute.local_tile(tma_a, (M, TK), (None, None))

    cta_layout = cute.make_layout((1,))
    tAsA, tAgA = cpasync.tma_partition(
        atom_a, 0, cta_layout,
        cute.group_modes(sA, 0, 2), cute.group_modes(gA_mkl, 0, 2))

    producer = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    consumer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
    pipe = pipeline.PipelineTmaAsync.create(
        num_stages=NSTAGE, producer_group=producer, consumer_group=consumer,
        tx_count=M * TK * 2, barrier_storage=st.bars.data_ptr())
    cute.arch.mbarrier_init_fence()

    if widx == 4:
        pp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NSTAGE)
        pp.reset_count()
        for kt in range(NKT):
            pipe.producer_acquire(pp)
            bp = pipe.producer_get_barrier(pp)
            cute.copy(atom_a, tAgA[(None, 0, kt)], tAsA[(None, pp.index)], tma_bar_ptr=bp)
            pipe.producer_commit(pp)
            pp.advance()
        pipe.producer_tail(pp)
    elif widx < 4:
        cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NSTAGE)
        cs.reset_count()
        for kt in range(NKT):
            pipe.consumer_wait(cs)
            if tidx == 0:
                out[kt] = sA[7, 3, cs.index].to(Float32)
            pipe.consumer_release(cs)
            cs.advance()


@cute.jit
def run(a, out):
    smem_a = cute.make_layout((M, TK), stride=(TK, 1))
    atom_a, tma_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), a, smem_a, (M, TK))
    _body(atom_a, tma_a, out).launch(grid=[1, 1, 1], block=[NWARP * 32, 1, 1])


def main():
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    out = torch.zeros(NKT, device="cuda", dtype=torch.float32)
    run(from_dlpack(a, assumed_align=32), out)
    torch.cuda.synchronize()
    exp = a[7, 3::TK].float()
    print("exp", exp.tolist())
    print("got", out.tolist())
    md = (out - exp).abs().max().item()
    print("maxdiff", md)
    print("PASS" if md < 1e-2 else "FAIL")


if __name__ == "__main__":
    main()
