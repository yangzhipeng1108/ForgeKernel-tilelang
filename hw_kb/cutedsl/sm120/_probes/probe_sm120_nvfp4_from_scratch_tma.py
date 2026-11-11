"""NVFP4 GEMM on sm120, hand-built: how far the port gets.

Status: every type/layout/shape step is solved and the kernel compiles; it stops
at the TMA barrier (a bare mbarrier compiles then hangs -- A/B/SF must go through
one PipelineTmaAsync). This probe is kept as the record of the solved half, not as
a passing test. Expect it to hang; run it under a timeout.

Solved here, each of which cost a round to find:
  * staged fp4 smem -> partition_A gives tCsA a real K-block mode ((8,2,2),4,2)
    -> pass the staged tensor to make_fragment_A whole (not [None,None,None,0])
  * recast_ptr(.., dtype=F4) must use the keyword; positional raises AttributeError
  * A/B need a 3-D (m,k,l) layout, so local_tile takes 3 coords and tAgA is 5-D
    -> copy index is tAgA[(None,0,0,kt,0)]
  * SF tensor must be tile_atom_to_shape_SF((m,k,1), 16); its TMA tile spans the
    FULL tile_k and uses internal_type=cutlass.Int16
  * fp4 cannot be written element-wise; packed Uint8 (0x22 = two 1.0 nibbles) only
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blockscaled_layout as bs_utils
import cutlass.utils.blackwell_helpers as sm120_utils
from flashinfer.gemm.kernels.sm12x_cute.blockscaled_gemm_dispatch import (
    make_sm120_blockscaled_mma_op,
)

M = N = 128
K = 256
TK = 128
SF_VEC = 16
SFK = K // SF_VEC
F4 = cutlass.Float4E2M1FN
E4M3 = cutlass.Float8E4M3FN
F32 = cutlass.Float32
TILE = (M, N, TK)
ATOM = (2, 2, 1)


@cute.struct
class Storage:
    bar: cute.struct.MemRange[cutlass.Int64, 2]
    sA: cute.struct.Align[cute.struct.MemRange[F4, M * TK], 1024]
    sB: cute.struct.Align[cute.struct.MemRange[F4, N * TK], 1024]
    sSFA: cute.struct.Align[cute.struct.MemRange[E4M3, 8192], 1024]
    sSFB: cute.struct.Align[cute.struct.MemRange[E4M3, 8192], 1024]


@cute.kernel
def _body(tiled_mma, atom_a, tma_a, atom_b, tma_b, c,
          a_smem_staged, b_smem_staged, sfa_staged, sfb_staged):
    tidx, _, _ = cute.arch.thread_idx()
    thr_mma = tiled_mma.get_slice(tidx)

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(Storage)
    sA = st.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
    sB = st.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
    sSFA = st.sSFA.get_tensor(sfa_staged)
    sSFB = st.sSFB.get_tensor(sfb_staged)

    gA = cute.local_tile(tma_a, (M, TK, 1), (None, None, None))
    gB = cute.local_tile(tma_b, (N, TK, 1), (None, None, None))
    cta = cute.make_layout((1,))
    tAsA, tAgA = cpasync.tma_partition(
        atom_a, 0, cta, cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
    tBsB, tBgB = cpasync.tma_partition(
        atom_b, 0, cta, cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

    tCsA = thr_mma.partition_A(sA[None, None, 0])
    tCsB = thr_mma.partition_B(sB[None, None, 0])
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCrSFA = sm120_utils.partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
    tCrSFB = sm120_utils.partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)

    tCc = thr_mma.partition_C(c)
    acc = cute.make_rmem_tensor(tCc.shape[:3], F32)
    acc.fill(0.0)

    if tidx == 0:
        cute.arch.mbarrier_init(st.bar.data_ptr(), 1)
    cute.arch.sync_threads()

    ntile = K // TK
    for kt in range(ntile):
        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                st.bar.data_ptr(), (M * TK // 2 + N * TK // 2))
            cute.copy(atom_a, tAgA[(None, 0, 0, kt, 0)], tAsA[(None, 0)],
                      tma_bar_ptr=st.bar.data_ptr())
            cute.copy(atom_b, tBgB[(None, 0, 0, kt, 0)], tBsB[(None, 0)],
                      tma_bar_ptr=st.bar.data_ptr())
            cute.arch.mbarrier_wait(st.bar.data_ptr(), kt % 2)
        cute.arch.sync_threads()
        nkb = cute.size(tCrA, mode=[2])
        for kb in range(nkb):
            cute.gemm(tiled_mma, acc,
                      [tCrA[None, None, kb], tCrSFA[None, None, kb]],
                      [tCrB[None, None, kb], tCrSFB[None, None, kb]],
                      acc)
        cute.arch.sync_threads()

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
            sm90_utils.get_smem_layout_atom(la, F4, TK), F4), (M, TK, 1), order=(0, 1, 2))
    b_smem = cute.tile_to_shape(
        cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(lb, F4, TK), F4), (N, TK, 1), order=(0, 1, 2))
    sfa_staged = bs_utils.sm120_make_smem_layout_sfa(tiled_mma, TILE, SF_VEC, 1)
    sfb_staged = bs_utils.sm120_make_smem_layout_sfb(tiled_mma, TILE, SF_VEC, 1)

    sfa_t = cute.make_tensor(sfa.iterator, bs_utils.tile_atom_to_shape_SF((M, K, 1), SF_VEC))
    sfb_t = cute.make_tensor(sfb.iterator, bs_utils.tile_atom_to_shape_SF((N, K, 1), SF_VEC))

    smem_a = cute.slice_(a_smem, (None, None, 0))
    smem_b = cute.slice_(b_smem, (None, None, 0))
    atom_a, tma_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), a, smem_a, (M, TK))
    atom_b, tma_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), b, smem_b, (N, TK))
    atom_sfa, tma_sfa = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), sfa_t,
        cute.slice_(sfa_staged, (None, None, 0)), (M, TK),
        internal_type=cutlass.Int16)
    atom_sfb, tma_sfb = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), sfb_t,
        cute.slice_(sfb_staged, (None, None, 0)), (N, TK),
        internal_type=cutlass.Int16)

    _body(tiled_mma, atom_a, tma_a, atom_b, tma_b, c,
          a_smem, b_smem, sfa_staged, sfb_staged).launch(
        grid=[1, 1, 1], block=[128, 1, 1])


if __name__ == "__main__":
    a = torch.zeros(M, K // 2, device="cuda", dtype=torch.uint8)
    a.fill_(0x22)
    b = torch.zeros(N, K // 2, device="cuda", dtype=torch.uint8)
    b.fill_(0x22)
    sfa = torch.full((M, SFK), 2.0, device="cuda", dtype=torch.float8_e4m3fn)
    sfb = torch.full((N, SFK), 2.0, device="cuda", dtype=torch.float8_e4m3fn)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    run(from_dlpack(a), from_dlpack(b), from_dlpack(sfa), from_dlpack(sfb), from_dlpack(c))
    torch.cuda.synchronize()
    print("c[0,0]", c[0, 0].item(), "c[7,3]", c[7, 3].item())
    print("expect 1024.0")
