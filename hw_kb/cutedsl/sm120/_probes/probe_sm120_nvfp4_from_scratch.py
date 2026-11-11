"""sm120 NVFP4 GEMM from primitives -- the MMA half, and where the load half dies.

PROBE: NEGATIVE (MMA-side PASS, load-side BLOCKED). Run inside the CuTe DSL
container:

    docker exec cutedsl_probe bash -lc "cd /w && python3 probe_sm120_nvfp4_from_scratch.py"

This authors its own tiled MMA, smem layouts, SF fragments, ldmatrix reads and
epilogue for a single 128x128x128 tile -- no DenseGemmKernel, no
Sm120BlockScaledGemmKernel. It compiles end to end. It does not produce numbers,
because the gmem->smem half is blocked on this card; see
`../from_scratch_load_path_is_the_wall.md` for the four traps and why.

Kept as a probe because the working half is the useful artifact: the exact
`MmaMXF4NVF4Op` + `get_permutation_mnk` + SF-fragment + list-operand `cute.gemm`
recipe that any hand-written sm120 fp4 kernel has to reproduce.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from cutlass.cute.nvgpu import cpasync
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cuda.bindings.driver as cuda

F4 = cutlass.Float4E2M1FN
E4M3 = cutlass.Float8E4M3FN
F32 = cutlass.Float32

SF_VEC = 16
TILE = (128, 128, 128)
FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _to_fp4_grid(x):
    g = torch.tensor(FP4, device=x.device)
    s = x.abs().clamp(min=1e-8)
    return g[torch.bucketize(s, g).clamp(1, 7) - 1] * x.sign()


def _pack_fp4(q):
    m, k = q.shape
    a, sg = q.abs(), q.sign()
    nib = torch.zeros_like(a, dtype=torch.uint8)
    for c, v in enumerate(FP4):
        nib = nib | ((a == v).to(torch.uint8) * c).to(torch.uint8)
    nib = nib | ((sg < 0).to(torch.uint8) << 3).to(torch.uint8)
    n2 = nib.view(m, k // 2, 2)
    return (n2[:, :, 0] | (n2[:, :, 1] << 4)).to(torch.uint8)


def _sf_atom_layout(sf, mn, sf_k):
    atom_k = 4
    pad_m = ((mn + 127) // 128) * 128
    pad_k = ((sf_k + atom_k - 1) // atom_k) * atom_k
    full = torch.zeros(1, pad_m, pad_k, dtype=sf.dtype, device=sf.device)
    full[0, :mn, :sf_k] = sf
    shape = (1, pad_m // 128, pad_k // atom_k, 32, 4, atom_k)
    return full.reshape(shape).permute(3, 4, 1, 5, 2, 0).contiguous()


def _ldmatrix_atom(transpose, n=4):
    return cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=n), F4)


def _mma_op():
    return cute.nvgpu.warp.MmaMXF4NVF4Op(F4, F32, E4M3)


def _tma_g2s(tensor, smem_layout, smem_tile, internal_type=None):
    op = cpasync.CopyBulkTensorTileG2SOp()
    return cpasync.make_tiled_tma_atom(
        op, tensor, smem_layout, smem_tile,
        num_multicast=1, internal_type=internal_type)


def _build(m=128, n=128, k=128):
    dev = "cuda"
    aq = _to_fp4_grid(torch.randn(m, k, device=dev) * 2)
    bq = _to_fp4_grid(torch.randn(n, k, device=dev) * 2)
    a_u8 = _pack_fp4(aq)
    b_u8 = _pack_fp4(bq)
    sf_k = k // SF_VEC
    sfa = torch.ones(m, sf_k, device=dev).to(torch.float8_e4m3fn)
    sfb = torch.ones(n, sf_k, device=dev).to(torch.float8_e4m3fn)
    sfa_p = _sf_atom_layout(sfa, m, sf_k)
    sfb_p = _sf_atom_layout(sfb, n, sf_k)
    C = torch.zeros(m, n, device=dev, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    a_ptr = make_ptr(F4, a_u8.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    b_ptr = make_ptr(F4, b_u8.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfa_ptr = make_ptr(E4M3, sfa_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(E4M3, sfb_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    c_ptr = make_ptr(F32, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    @cute.jit
    def launch(ap, bp, sfap, sfbp, cp, s):
        mA = cute.make_tensor(ap, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
        mB = cute.make_tensor(bp, cute.make_layout((n, k, 1), stride=(k, 1, 0)))
        mC = cute.make_tensor(cp, cute.make_layout((m, n, 1), stride=(n, 1, 0)))
        sfa_lyt = blockscaled_utils.tile_atom_to_shape_SF((m, k, 1), SF_VEC)
        sfb_lyt = blockscaled_utils.tile_atom_to_shape_SF((n, k, 1), SF_VEC)
        mSFA = cute.make_tensor(sfap, sfa_lyt)
        mSFB = cute.make_tensor(sfbp, sfb_lyt)

        op = _mma_op()
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout((2, 2, 1)),
            permutation_mnk=sm120_utils.get_permutation_mnk(TILE, SF_VEC, False))

        a_layout = utils.LayoutEnum.from_tensor(mA)
        b_layout = utils.LayoutEnum.from_tensor(mB)
        a_smem = cute.tile_to_shape(
            cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(a_layout, F4, TILE[2]), F4),
            (TILE[0], TILE[2], 1), order=(0, 1, 2))
        b_smem = cute.tile_to_shape(
            cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(b_layout, F4, TILE[2]), F4),
            (TILE[1], TILE[2], 1), order=(0, 1, 2))
        sfa_smem = blockscaled_utils.sm120_make_smem_layout_sfa(tiled_mma, TILE, SF_VEC, 1)
        sfb_smem = blockscaled_utils.sm120_make_smem_layout_sfb(tiled_mma, TILE, SF_VEC, 1)

        tma_a, t_a = _tma_g2s(mA, cute.slice_(a_smem, (None, None, 0)), (TILE[0], TILE[2]))
        tma_b, t_b = _tma_g2s(mB, cute.slice_(b_smem, (None, None, 0)), (TILE[1], TILE[2]))
        tma_sfa, t_sfa = _tma_g2s(
            mSFA, sfa_smem, (TILE[0], TILE[2]), internal_type=cutlass.Int16)
        tma_sfb, t_sfb = _tma_g2s(
            mSFB, sfb_smem, (TILE[1], TILE[2]), internal_type=cutlass.Int16)

        @cute.struct
        class SS:
            # barrier storage must hold 2 * num_stages mbarriers (full at +0,
            # empty at +num_stages); a single Int64 silently hangs consumer_wait.
            bar: cute.struct.MemRange[cutlass.Int64, 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[F4, cute.cosize(a_smem)], 1024]
            sB: cute.struct.Align[
                cute.struct.MemRange[F4, cute.cosize(b_smem)], 1024]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[E4M3, cute.cosize(sfa_smem)], 1024]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[E4M3, cute.cosize(sfb_smem)], 1024]

        kern(tma_a, t_a, tma_b, t_b, tma_sfa, t_sfa, tma_sfb, t_sfb, mC,
             tiled_mma, a_smem, b_smem, sfa_smem, sfb_smem, SS).launch(
            grid=(1, 1, 1), block=[128, 1, 1], cluster=(1, 1, 1), stream=s)

    compiled = cute.compile(launch, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, stream)

    def call():
        compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, stream)

    call()
    torch.cuda.synchronize()
    ref = aq @ bq.T
    md = (C - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        C.flatten(), ref.flatten(), dim=0).item()
    print(f"md={md:.5g} cos={cos:.6f}  (expect md~0; see the KB entry for why it will not run)")


# The kernel below compiles but is unreachable: the fp4 gmem->smem path does not
# line up (trap 3/4 in the KB entry). Kept so the MMA half stays reproducible.
@cute.kernel
def kern(tma_a, tA_mkl, tma_b, tB_nkl, tma_sfa, tSFA_mkl, tma_sfb, tSFB_nkl,
         mC, tiled_mma, a_smem, b_smem, sfa_smem, sfb_smem, SS: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(SS)
    bar = st.bar.data_ptr()

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    sA = st.sA.get_tensor(a_smem.outer, swizzle=a_smem.inner)
    sB = st.sB.get_tensor(b_smem.outer, swizzle=b_smem.inner)
    sSFA = st.sSFA.get_tensor(sfa_smem)
    sSFB = st.sSFB.get_tensor(sfb_smem)

    gA = cute.local_tile(tA_mkl, (TILE[0], TILE[2], 1), (0, 0, 0))
    gB = cute.local_tile(tB_nkl, (TILE[1], TILE[2], 1), (0, 0, 0))
    gSFA = cute.local_tile(tSFA_mkl, (TILE[0], TILE[2], 1), (0, 0, 0))
    gSFB = cute.local_tile(tSFB_nkl, (TILE[1], TILE[2], 1), (0, 0, 0))
    gC = cute.local_tile(mC, (TILE[0], TILE[1], 1), (0, 0, 0))

    tAsA, tAgA = cpasync.tma_partition(
        tma_a, 0, cute.make_layout((1,)),
        cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
    tBsB, tBgB = cpasync.tma_partition(
        tma_b, 0, cute.make_layout((1,)),
        cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_sfa, 0, cute.make_layout((1,)),
        cute.group_modes(sSFA, 0, 2), cute.group_modes(gSFA, 0, 2))
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_sfb, 0, cute.make_layout((1,)),
        cute.group_modes(sSFB, 0, 2), cute.group_modes(gSFB, 0, 2))

    thr_mma = tiled_mma.get_slice(tidx % 128)
    tCsA = thr_mma.partition_A(cute.slice_(sA, (None, None, 0)))
    tCsB = thr_mma.partition_B(cute.slice_(sB, (None, None, 0)))
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCrSFA = sm120_utils.partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
    tCrSFB = sm120_utils.partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)

    tCgC = thr_mma.partition_C(gC)
    acc = cute.make_rmem_tensor(tCgC.shape[:3], F32)

    tx = (cute.size_in_bytes(F4, cute.slice_(a_smem, (None, None, 0)))
          + cute.size_in_bytes(F4, cute.slice_(b_smem, (None, None, 0)))
          + cute.size_in_bytes(E4M3, sfa_smem)
          + cute.size_in_bytes(E4M3, sfb_smem))

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, tx)
            cute.copy(tma_a, tAgA[None, 0], tAsA, tma_bar_ptr=bar)
            cute.copy(tma_b, tBgB[None, 0], tBsB, tma_bar_ptr=bar)
            cute.copy(tma_sfa, tAgSFA[None, 0], tAsSFA[None, 0], tma_bar_ptr=bar)
            cute.copy(tma_sfb, tBgSFB[None, 0], tBsSFB[None, 0], tma_bar_ptr=bar)
    cute.arch.mbarrier_wait(bar, 0)
    cute.arch.sync_threads()

    smem_tiled_copy_A = cute.make_tiled_copy_A(_ldmatrix_atom(False), tiled_mma)
    smem_tiled_copy_B = cute.make_tiled_copy_B(_ldmatrix_atom(False), tiled_mma)
    thrA = smem_tiled_copy_A.get_slice(tidx % 128)
    thrB = smem_tiled_copy_B.get_slice(tidx % 128)
    sA1 = cute.slice_(sA, (None, None, 0))
    sB1 = cute.slice_(sB, (None, None, 0))
    tCsA_cv = thrA.partition_S(sA1)
    tCrA_cv = thrA.retile(tCrA)
    tCsB_cv = thrB.partition_S(sB1)
    tCrB_cv = thrB.retile(tCrB)

    sf_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), E4M3)
    smem_copy_SFA = cute.make_tiled_copy(
        sf_copy_atom, sm120_utils.get_layoutSFA_TV(tiled_mma),
        (cute.size(tiled_mma.permutation_mnk[0]), cute.size(tiled_mma.permutation_mnk[2])))
    smem_copy_SFB = cute.make_tiled_copy(
        sf_copy_atom, sm120_utils.get_layoutSFB_TV(tiled_mma),
        (cute.size(tiled_mma.permutation_mnk[1]), cute.size(tiled_mma.permutation_mnk[2])))
    thrSFA = smem_copy_SFA.get_slice(tidx % 128)
    thrSFB = smem_copy_SFB.get_slice(tidx % 128)
    tCsSFA_cv = thrSFA.partition_S(sSFA[None, None, 0])
    tCrSFA_cv = thrSFA.retile(tCrSFA)
    tCsSFB_cv = thrSFB.partition_S(sSFB[None, None, 0])
    tCrSFB_cv = thrSFB.retile(tCrSFB)

    nkb = cute.size(tCrA, mode=[2])
    for kb in cutlass.range_constexpr(nkb):
        cute.copy(smem_tiled_copy_A, tCsA_cv[None, None, kb], tCrA_cv[None, None, kb])
        cute.copy(smem_tiled_copy_B, tCsB_cv[None, None, kb], tCrB_cv[None, None, kb])
        cute.copy(smem_copy_SFA, cute.filter_zeros(tCsSFA_cv)[None, None, kb],
                  cute.filter_zeros(tCrSFA_cv)[None, None, kb])
        cute.copy(smem_copy_SFB, cute.filter_zeros(tCsSFB_cv)[None, None, kb],
                  cute.filter_zeros(tCrSFB_cv)[None, None, kb])
        for mn in cutlass.range_constexpr(cute.size(acc, mode=[2])):
            for mm in cutlass.range_constexpr(cute.size(acc, mode=[1])):
                cute.gemm(
                    tiled_mma, acc[None, mm, mn],
                    [tCrA[None, mm, kb], tCrSFA[None, mm, kb]],
                    [tCrB[None, mn, kb], tCrSFB[None, mn, kb]],
                    acc[None, mm, mn])

    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), F32),
              acc, tCgC[None, None, None, 0])


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _build()
