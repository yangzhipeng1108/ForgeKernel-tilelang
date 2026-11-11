"""Hand-written fp16 GEMM on sm120, multi-CTA: TMA pipeline + ldmatrix + MMA.

PROBE: PASS. Run inside the CuTe DSL container:

    docker exec cutedsl_probe bash -lc "cd /w && python3 probe_sm120_wmma_gemm_multicta.py"

Same verified skeleton as probe_sm120_wmma_gemm_tma_pipeline.py, extended to a 2-D
grid so it can be timed on realistic shapes. Each CTA owns one 128x128 output tile:

    blockIdx.x = tile_n, blockIdx.y = tile_m
    A rows   [tile_m*128 : +128]   all of K
    B rows   [tile_n*128 : +128]   all of K
    C tile   [tile_m*128 : +128, tile_n*128 : +128]

Two things this probe had to get right beyond the single-CTA version:
  * a 2-D grid instead of an index divided in-kernel -- `bidx // ntile_n` lowers to
    arith.floordivsi, which fails IR verification ("using value defined outside the
    region"); `block_idx()` returning (x, y) sidesteps it entirely
  * **timing must go through `cute.compile`**. Calling the `@cute.jit` function in a
    loop recompiles every iteration and shows a flat ~22.8 ms regardless of shape,
    which reads as "0.1 TF/s" and is pure host compile time
  * C must be sliced with `cute.local_tile(c, (BM, BN), (tm, tn))`; partitioning the
    whole C would give every CTA the same (M*N/128)-shaped destination

Measured (correctness first on 512x384x1024, rel 1.7e-6, frac 1.0):
    1024^3  ->   30.2 us,  71.1 TF/s    (shipped dense_gemm.py: 125 us)
    2048^3  ->  100.1 us, 171.7 TF/s
    4096^3  ->  961.1 us, 143.0 TF/s

2048^3 is the peak because 256 tiles / 156 SMs = 1.64 waves (two full waves);
4096^3 is 1024 / 156 = 6.56 waves and loses the tail to wave quantization, the
same effect tuned in tuning_the_nvfp4_gemm. TK=64 with 3 stages instead of TK=32
with 4 changes nothing (73/173/143), so the limit is MMA issue + wave fill, not
TMA granularity.
"""
import sys
import time
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
from cutlass import pipeline
from cutlass import Int64, utils
import cutlass.utils.hopper_helpers as sm90_utils

BM = BN = 128
TK = 32
NSTAGE = 4
DT = cutlass.Float16
ACC = cutlass.Float32
NWARP = 5
NTHREADS = NWARP * 32
MMA_TILER = (2, 2, 1)
MMA_INST = (16, 8, 16)


@cute.struct
class Storage:
    bars: cute.struct.MemRange[Int64, NSTAGE * 2]
    sA: cute.struct.Align[cute.struct.MemRange[DT, NSTAGE * BM * TK], 1024]
    sB: cute.struct.Align[cute.struct.MemRange[DT, NSTAGE * BN * TK], 1024]


@cute.kernel
def _body(atom_a, tma_a, atom_b, tma_b, tiled_mma, c, nkt,
          a_smem_staged, b_smem_staged,
          a_m_major: cutlass.Constexpr, b_n_major: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    widx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tn, tm, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(Storage)
    sA = st.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
    sB = st.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)

    gA = cute.local_tile(tma_a, (BM, TK), (None, None))
    gB = cute.local_tile(tma_b, (BN, TK), (None, None))
    cta_layout = cute.make_layout((1,))
    tAsA, tAgA = cpasync.tma_partition(
        atom_a, 0, cta_layout,
        cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
    tBsB, tBgB = cpasync.tma_partition(
        atom_b, 0, cta_layout,
        cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

    producer = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    consumer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
    pipe = pipeline.PipelineTmaAsync.create(
        num_stages=NSTAGE, producer_group=producer, consumer_group=consumer,
        tx_count=(BM * TK + BN * TK) * 2, barrier_storage=st.bars.data_ptr())
    cute.arch.mbarrier_init_fence()

    thr = tiled_mma.get_slice(tidx % 128)
    ldm_a = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(a_m_major, 4), DT)
    ldm_b = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(b_n_major, 4), DT)
    scA = cute.make_tiled_copy_A(ldm_a, tiled_mma)
    scB = cute.make_tiled_copy_B(ldm_b, tiled_mma)

    if widx == 4:
        pp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NSTAGE)
        pp.reset_count()
        for kt in range(nkt):
            pipe.producer_acquire(pp)
            bp = pipe.producer_get_barrier(pp)
            cute.copy(atom_a, tAgA[(None, tm, kt)], tAsA[(None, pp.index)], tma_bar_ptr=bp)
            cute.copy(atom_b, tBgB[(None, tn, kt)], tBsB[(None, pp.index)], tma_bar_ptr=bp)
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

        c_tile = cute.local_tile(c, (BM, BN), (tm, tn))
        tCc = thr.partition_C(c_tile)
        acc = cute.make_rmem_tensor(tCc.shape[:3], ACC)
        acc.fill(0.0)

        nkb = cute.size(tCrA, mode=[2])
        cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NSTAGE)
        cs.reset_count()
        for kt in range(nkt):
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
def run(a, b, c, nkt, ntile_n, ntile_m):
    la = utils.LayoutEnum.from_tensor(a)
    lb = utils.LayoutEnum.from_tensor(b)
    tile = (BM, BN, TK)
    a_smem_staged = sm90_utils.make_smem_layout_a(la, tile, DT, NSTAGE)
    b_smem_staged = sm90_utils.make_smem_layout_b(lb, tile, DT, NSTAGE)
    smem_a = cute.slice_(a_smem_staged, (None, None, 0))
    smem_b = cute.slice_(b_smem_staged, (None, None, 0))
    op = cute.nvgpu.warp.MmaF16BF16Op(DT, ACC, MMA_INST)
    tiled_mma = cute.make_tiled_mma(op, cute.make_layout(MMA_TILER))
    atom_a, tma_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), a, smem_a, (BM, TK))
    atom_b, tma_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), b, smem_b, (BN, TK))
    _body(atom_a, tma_a, atom_b, tma_b, tiled_mma, c, nkt,
          a_smem_staged, b_smem_staged,
          la.is_m_major_a(), lb.is_n_major_b()).launch(
        grid=[ntile_n, ntile_m, 1], block=[NTHREADS, 1, 1])


def check():
    M, N, K = 512, 384, 1024
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    run(from_dlpack(a), from_dlpack(b), from_dlpack(c),
        K // TK, N // BN, M // BM)
    torch.cuda.synchronize()
    ref = a.float() @ b.float().T
    rel = (c - ref).abs().max().item() / ref.abs().max().item()
    frac = ((c - ref).abs() < 1e-2).float().mean().item()
    print(f"[check] {M}x{N}x{K} rel {rel:.3e} frac {frac:.4f}")
    return rel < 5e-3 and frac > 0.999


def bench(M, N, K, iters=20):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    da, db, dc = from_dlpack(a), from_dlpack(b), from_dlpack(c)
    nkt, ntile_n, ntile_m = K // TK, N // BN, M // BM
    compiled = cute.compile(run, da, db, dc, nkt, ntile_n, ntile_m)
    compiled(da, db, dc, nkt, ntile_n, ntile_m)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        compiled(da, db, dc, nkt, ntile_n, ntile_m)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"[bench] {M}x{N}x{K}  {ms*1e3:.1f} us  {tflops:.1f} TF/s")


if __name__ == "__main__":
    if not check():
        print("FAIL: correctness gate")
        sys.exit(1)
    print("PASS: correctness")
    for shape in [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]:
        bench(*shape)
