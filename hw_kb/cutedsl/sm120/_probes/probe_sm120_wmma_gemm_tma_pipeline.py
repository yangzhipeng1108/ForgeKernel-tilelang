"""Hand-written fp16 GEMM on sm120: TMA pipeline + ldmatrix + MMA, end to end.

PROBE: PASS. Run inside the CuTe DSL container:

    docker exec cutedsl_probe bash -lc "cd /w && python3 probe_sm120_wmma_gemm_tma_pipeline.py"

One CTA, 128 threads, computes a full 128x128 over K=1024, streaming 32 K-tiles
through a 4-stage TMA ring. rel error 1.6e-6, every element correct. No shipped
kernel involved.

This joins the two halves that were built separately:
  - the TMA load path (probe_sm120_tma_pipeline_multistage.py), and
  - the MMA compute path (probe_sm120_wmma_gemm_computes.py).

The one thing that made the join fail with "src ptr alignment (16 bits) does not
meet requirement (128 bits)": the ldmatrix atom's major flag was hardcoded True.
A row-major (K-major) A is is_m_major_a() == False, and B is is_n_major_b(). The
wrong flag makes the ldsm atom expect a layout the swizzled smem does not have,
which surfaces as an alignment error rather than a transpose error. Read both
flags from utils.LayoutEnum and pass them in as cutlass.Constexpr.

See ../from_scratch_tma_load_path_works.md for the full list of fixes.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
from cutlass import pipeline
from cutlass import Int64, utils
import cutlass.utils.hopper_helpers as sm90_utils

M = N = 128
K = 1024
TK = 32
NKT = K // TK
NSTAGE = 4
DT = cutlass.Float16
ACC = cutlass.Float32
NWARP = 5
NTHREADS = NWARP * 32
MMA_TILER = (2, 2, 1)
MMA_INST = (16, 8, 16)
TILE_MNK = (M, N, TK)


@cute.struct
class Storage:
    bars: cute.struct.MemRange[Int64, NSTAGE * 2]
    sA: cute.struct.Align[cute.struct.MemRange[DT, NSTAGE * M * TK], 1024]
    sB: cute.struct.Align[cute.struct.MemRange[DT, NSTAGE * N * TK], 1024]


@cute.kernel
def _body(atom_a, tma_a, atom_b, tma_b, tiled_mma, c,
          a_smem_staged, b_smem_staged,
          a_m_major: cutlass.Constexpr, b_n_major: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    widx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(Storage)
    sA = st.sA.get_tensor(
        a_smem_staged.outer, swizzle=a_smem_staged.inner)
    sB = st.sB.get_tensor(
        b_smem_staged.outer, swizzle=b_smem_staged.inner)

    gA_mkl = cute.local_tile(tma_a, (M, TK), (None, None))
    gB_nkl = cute.local_tile(tma_b, (N, TK), (None, None))
    cta_layout = cute.make_layout((1,))
    tAsA, tAgA = cpasync.tma_partition(
        atom_a, 0, cta_layout,
        cute.group_modes(sA, 0, 2), cute.group_modes(gA_mkl, 0, 2))
    tBsB, tBgB = cpasync.tma_partition(
        atom_b, 0, cta_layout,
        cute.group_modes(sB, 0, 2), cute.group_modes(gB_nkl, 0, 2))

    producer = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    consumer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
    pipe = pipeline.PipelineTmaAsync.create(
        num_stages=NSTAGE, producer_group=producer, consumer_group=consumer,
        tx_count=(M * TK + N * TK) * 2, barrier_storage=st.bars.data_ptr())
    cute.arch.mbarrier_init_fence()

    thr = tiled_mma.get_slice(tidx % 128)
    ldm_a = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(a_m_major, 4), DT)
    ldm_b = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(b_n_major, 4), DT)
    scA = cute.make_tiled_copy_A(ldm_a, tiled_mma)
    scB = cute.make_tiled_copy_B(ldm_b, tiled_mma)

    if widx == 4:
        pp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NSTAGE)
        pp.reset_count()
        for kt in range(NKT):
            pipe.producer_acquire(pp)
            bp = pipe.producer_get_barrier(pp)
            cute.copy(atom_a, tAgA[(None, 0, kt)], tAsA[(None, pp.index)], tma_bar_ptr=bp)
            cute.copy(atom_b, tBgB[(None, 0, kt)], tBsB[(None, pp.index)], tma_bar_ptr=bp)
            pipe.producer_commit(pp)
            pp.advance()
        pipe.producer_tail(pp)
    elif widx < 4:
        tCsA = thr.partition_A(sA[None, None, 0])
        tCsB = thr.partition_B(sB[None, None, 0])
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        thr_cA = scA.get_slice(tidx % 128)
        thr_cB = scB.get_slice(tidx % 128)
        tCsA_view = thr_cA.partition_S(sA)
        tCsB_view = thr_cB.partition_S(sB)
        tCrA_v = thr_cA.retile(tCrA)
        tCrB_v = thr_cB.retile(tCrB)

        tCc = thr.partition_C(c)
        acc = cute.make_rmem_tensor(tCc.shape[:3], ACC)
        acc.fill(0.0)

        nkb = cute.size(tCrA, mode=[2])
        cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NSTAGE)
        cs.reset_count()
        for kt in range(NKT):
            pipe.consumer_wait(cs)
            tCsA_v = tCsA_view[None, None, None, cs.index]
            tCsB_v = tCsB_view[None, None, None, cs.index]
            for kb in range(nkb):
                cute.copy(scA, tCsA_v[None, None, kb], tCrA_v[None, None, kb])
                cute.copy(scB, tCsB_v[None, None, kb], tCrB_v[None, None, kb])
                cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
            pipe.consumer_release(cs)
            cs.advance()

        for i in range(cute.size(acc)):
            tCc[i] = acc[i]


@cute.jit
def run(a, b, c):
    la = utils.LayoutEnum.from_tensor(a)
    lb = utils.LayoutEnum.from_tensor(b)
    a_smem_staged = sm90_utils.make_smem_layout_a(la, TILE_MNK, DT, NSTAGE)
    b_smem_staged = sm90_utils.make_smem_layout_b(lb, TILE_MNK, DT, NSTAGE)
    smem_a = cute.slice_(a_smem_staged, (None, None, 0))
    smem_b = cute.slice_(b_smem_staged, (None, None, 0))
    op = cute.nvgpu.warp.MmaF16BF16Op(DT, ACC, MMA_INST)
    tiled_mma = cute.make_tiled_mma(op, cute.make_layout(MMA_TILER))
    atom_a, tma_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), a, smem_a, (M, TK))
    atom_b, tma_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), b, smem_b, (N, TK))
    _body(atom_a, tma_a, atom_b, tma_b, tiled_mma, c,
          a_smem_staged, b_smem_staged,
          la.is_m_major_a(), lb.is_n_major_b()).launch(
        grid=[1, 1, 1], block=[NTHREADS, 1, 1])


def main():
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    run(from_dlpack(a), from_dlpack(b), from_dlpack(c))
    torch.cuda.synchronize()
    ref = a.float() @ b.float().T
    rel = (c - ref).abs().max().item() / ref.abs().max().item()
    ok = ((c - ref).abs() < 1e-2).float().mean().item()
    print("rel", rel, "frac correct", ok)
    print("PASS" if rel < 5e-3 else "FAIL")


if __name__ == "__main__":
    main()
