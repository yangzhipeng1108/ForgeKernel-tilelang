"""NVFP4 GEMM on sm120, hand-built: TMA pipeline + block-scaled MMA.

Closed-form gate: A=B=1.0 nibbles (byte 0x22), SF=2.0, K=256
-> every C element = (1*2)*(1*2)*256 = 1024.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
from cutlass import pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blockscaled_layout as bs_utils
import cutlass.utils.blackwell_helpers as sm120_utils
from flashinfer.gemm.kernels.sm12x_cute.blockscaled_gemm_dispatch import (
    make_sm120_blockscaled_mma_op,
)

M = N = 128
K = 256
TK = 256
SF_VEC = 16
SFK = K // SF_VEC
NSTAGE = 2
NKT = K // TK
F4 = cutlass.Float4E2M1FN
E4M3 = cutlass.Float8E4M3FN
F32 = cutlass.Float32
TILE = (M, N, TK)
ATOM = (2, 2, 1)


@cute.struct
class Storage:
    bars: cute.struct.MemRange[cutlass.Int64, 2 * NSTAGE]
    sA: cute.struct.Align[cute.struct.MemRange[F4, NSTAGE * M * TK], 1024]
    sB: cute.struct.Align[cute.struct.MemRange[F4, NSTAGE * N * TK], 1024]
    sSFA: cute.struct.Align[cute.struct.MemRange[E4M3, NSTAGE * M * SFK], 1024]
    sSFB: cute.struct.Align[cute.struct.MemRange[E4M3, NSTAGE * N * SFK], 1024]


@cute.kernel
def _body(tiled_mma, atom_a, tma_a, atom_b, tma_b, atom_sfa, tma_sfa,
          atom_sfb, tma_sfb, c, a_smem_staged, b_smem_staged,
          sfa_staged, sfb_staged):
    tidx, _, _ = cute.arch.thread_idx()
    widx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(Storage)
    sA = st.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
    sB = st.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
    sSFA = st.sSFA.get_tensor(sfa_staged)
    sSFB = st.sSFB.get_tensor(sfb_staged)

    gA = cute.local_tile(tma_a, (M, TK), (None, None, None))
    gB = cute.local_tile(tma_b, (N, TK), (None, None, None))
    gSFA = cute.local_tile(tma_sfa, (M, TK), (None, None, None))
    gSFB = cute.local_tile(tma_sfb, (N, TK), (None, None, None))
    cta = cute.make_layout((1,))
    tAsA, tAgA = cpasync.tma_partition(
        atom_a, 0, cta, cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
    tBsB, tBgB = cpasync.tma_partition(
        atom_b, 0, cta, cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))
    tSsSFA, tSgSFA = cpasync.tma_partition(
        atom_sfa, 0, cta, cute.group_modes(sSFA, 0, 2), cute.group_modes(gSFA, 0, 2))
    tSsSFB, tSgSFB = cpasync.tma_partition(
        atom_sfb, 0, cta, cute.group_modes(sSFB, 0, 2), cute.group_modes(gSFB, 0, 2))
    tSsSFA = cute.filter_zeros(tSsSFA)
    tSgSFA = cute.filter_zeros(tSgSFA)
    tSsSFB = cute.filter_zeros(tSsSFB)
    tSgSFB = cute.filter_zeros(tSgSFB)

    producer = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    consumer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
    tx = (M * TK // 2) + (N * TK // 2) + (M * TK // SF_VEC) + (N * TK // SF_VEC)
    pipe = pipeline.PipelineTmaAsync.create(
        num_stages=NSTAGE, producer_group=producer, consumer_group=consumer,
        tx_count=tx, barrier_storage=st.bars.data_ptr())
    cute.arch.mbarrier_init_fence()

    if widx == 4:
        pp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NSTAGE)
        pp.reset_count()
        for kt in range(NKT):
            pipe.producer_acquire(pp)
            bp = pipe.producer_get_barrier(pp)
            cute.copy(atom_a, tAgA[(None, 0, kt, 0)],
                      tAsA[(None, pp.index)], tma_bar_ptr=bp)
            cute.copy(atom_b, tBgB[(None, 0, kt, 0)],
                      tBsB[(None, pp.index)], tma_bar_ptr=bp)
            cute.copy(atom_sfa, tSgSFA[(None, 0, kt, 0)],
                      tSsSFA[(None, pp.index)], tma_bar_ptr=bp)
            cute.copy(atom_sfb, tSgSFB[(None, 0, kt, 0)],
                      tSsSFB[(None, pp.index)], tma_bar_ptr=bp)
            pipe.producer_commit(pp)
            pp.advance()
        pipe.producer_tail(pp)
    elif widx < 4:
        thr = tiled_mma.get_slice(tidx % 128)
        tCsA = thr.partition_A(sA[None, None, 0])
        tCsB = thr.partition_B(sB[None, None, 0])
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrSFA = sm120_utils.partition_fragment_SFA(sSFA[None, None, 0], thr, tidx)
        tCrSFB = sm120_utils.partition_fragment_SFB(sSFB[None, None, 0], thr, tidx)

        tCc = thr.partition_C(c)
        acc = cute.make_rmem_tensor(tCc.shape[:3], F32)
        acc.fill(0.0)

        nkb = cute.size(tCrA, mode=[2])
        cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NSTAGE)
        cs.reset_count()
        for kt in range(NKT):
            pipe.consumer_wait(cs)
            for kb in range(nkb):
                cute.gemm(tiled_mma, acc,
                          [tCrA[None, None, kb], tCrSFA[None, None, kb]],
                          [tCrB[None, None, kb], tCrSFB[None, None, kb]],
                          acc)
            pipe.consumer_release(cs)
            cs.advance()

        for i in range(cute.size(acc)):
            tCc[i] = acc[i]


@cute.jit
def run(a_u8, b_u8, sfa, sfb, c):
    a = cute.make_tensor(
        cute.recast_ptr(a_u8.iterator, dtype=F4),
        layout=cute.make_ordered_layout((M, K, 1), order=(1, 0, 2)))
    b = cute.make_tensor(
        cute.recast_ptr(b_u8.iterator, dtype=F4),
        layout=cute.make_ordered_layout((N, K, 1), order=(1, 0, 2)))
    mma_op, use_mxf8f6f4 = make_sm120_blockscaled_mma_op(F4, F4, F32, E4M3, SF_VEC)
    perm = sm120_utils.get_permutation_mnk(TILE, SF_VEC, use_mxf8f6f4)
    tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout(ATOM), permutation_mnk=perm)

    la = utils.LayoutEnum.from_tensor(a)
    lb = utils.LayoutEnum.from_tensor(b)
    a_smem = cute.tile_to_shape(
        cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(la, F4, TK), F4),
        (M, TK, NSTAGE), order=(0, 1, 2))
    b_smem = cute.tile_to_shape(
        cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(lb, F4, TK), F4),
        (N, TK, NSTAGE), order=(0, 1, 2))
    sfa_staged = bs_utils.sm120_make_smem_layout_sfa(tiled_mma, TILE, SF_VEC, NSTAGE)
    sfb_staged = bs_utils.sm120_make_smem_layout_sfb(tiled_mma, TILE, SF_VEC, NSTAGE)

    sfa_t = cute.make_tensor(sfa.iterator, bs_utils.tile_atom_to_shape_SF((M, K, 1), SF_VEC))
    sfb_t = cute.make_tensor(sfb.iterator, bs_utils.tile_atom_to_shape_SF((N, K, 1), SF_VEC))

    atom_a, tma_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), a,
        cute.slice_(a_smem, (None, None, 0)), (M, TK))
    atom_b, tma_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), b,
        cute.slice_(b_smem, (None, None, 0)), (N, TK))
    atom_sfa, tma_sfa = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), sfa_t,
        cute.slice_(sfa_staged, (None, None, 0)), (M, TK),
        internal_type=cutlass.Int16)
    atom_sfb, tma_sfb = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), sfb_t,
        cute.slice_(sfb_staged, (None, None, 0)), (N, TK),
        internal_type=cutlass.Int16)

    _body(tiled_mma, atom_a, tma_a, atom_b, tma_b, atom_sfa, tma_sfa,
          atom_sfb, tma_sfb, c, a_smem, b_smem, sfa_staged, sfb_staged).launch(
        grid=[1, 1, 1], block=[160, 1, 1])


if __name__ == "__main__":
    import numpy as np

    def sf_perm(mn):
        p = np.zeros(mn * SFK, dtype=np.int64)
        for m in range(mn):
            m1, m0 = divmod(m, 32)
            for k in range(SFK):
                k0, rem = divmod(k, 16)
                k1, k2 = divmod(rem, 4)
                p[16 * m0 + 4 * m1 + k1 + 512 * k2] = m * SFK + k
        return p

    def prep(v, mn):
        src = torch.full((mn * SFK,), v, device="cuda", dtype=torch.float8_e4m3fn)
        p = torch.from_numpy(sf_perm(mn)).to("cuda")
        return src[p]

    a = torch.zeros(M, K // 2, device="cuda", dtype=torch.uint8)
    a.fill_(0x22)
    b = torch.zeros(N, K // 2, device="cuda", dtype=torch.uint8)
    b.fill_(0x22)
    sfa = prep(1.0, M)
    sfb = prep(1.0, N)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    run(from_dlpack(a), from_dlpack(b), from_dlpack(sfa), from_dlpack(sfb), from_dlpack(c))
    torch.cuda.synchronize()
    print("c[0,0]", c[0, 0].item(), "c[7,3]", c[7, 3].item())
    print("expect 256.0 (SF=1)")