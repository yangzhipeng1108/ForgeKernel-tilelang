"""Hand-written fp16 GEMM on sm120, persistent CTAs: NEGATIVE RESULT.

PROBE: negative. It computes correctly (rel 1.7e-6) but it is NOT faster than the
plain multi-CTA launch, so do not use it -- the non-persistent
probe_sm120_wmma_gemm_multicta.py is the better kernel. Kept to document why the
obvious optimisation does not pay.

    docker exec cutedsl_probe bash -lc "cd /w && python3 probe_sm120_wmma_gemm_persistent.py"

The idea: 4096^3 is 1024 tiles over 156 SMs = 6.56 waves, so the last wave is ~44%
idle, and a persistent grid that loops over tiles should recover that tail. Measured
against the plain launch it does not:

    shape    plain multicta    persistent (GRID=312)
    1024^3   71.1 TF/s         69.1 TF/s
    2048^3  171.7 TF/s        166.2 TF/s
    4096^3  143.0 TF/s        137.4 TF/s

A smaller persistent grid is much worse still (GRID=156, i.e. one CTA per SM, drops
4096^3 to 115.6 TF/s) because 156 CTAs cannot fill the machine the way many short
CTAs can.

Why it loses: the hardware block scheduler already overlaps the partial wave with the
next one, and a persistent CTA pays for it -- the 4-stage pipeline must drain and
refill at every tile boundary, and consecutive tiles re-issue their A/B TMA from the
start of K instead of letting a fresh CTA start clean. So the nominal 44%-idle tail
is largely already hidden, and the added serialisation costs more than it saves.

What the probe does get right, and which is worth keeping for any future persistent
kernel on this card:
  * pipeline state objects must be created once outside the tile loop and keep
    advancing; re-creating them per tile desynchronises the barriers and hangs
  * only the accumulator is per-tile (zeroed at the start, stored at the end)
  * the tile loop must not divide in-kernel (`tile // ntile_n` lowers to
    arith.floordivsi and fails IR verification) -- stride along a 2-D grid axis
    instead, which is why this kernel walks the m-axis and keeps tile_n from
    blockIdx.x
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
def _body(atom_a, tma_a, atom_b, tma_b, tiled_mma, c, nkt, ntiles, grid_size,
          a_smem_staged, b_smem_staged,
          a_m_major: cutlass.Constexpr, b_n_major: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    widx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    ctid_n, ctid_m, _ = cute.arch.block_idx()

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
        tile = ctid_m
        while tile < ntiles:
            for kt in range(nkt):
                pipe.producer_acquire(pp)
                bp = pipe.producer_get_barrier(pp)
                cute.copy(atom_a, tAgA[(None, tile, kt)], tAsA[(None, pp.index)],
                          tma_bar_ptr=bp)
                cute.copy(atom_b, tBgB[(None, ctid_n, kt)], tBsB[(None, pp.index)],
                          tma_bar_ptr=bp)
                pipe.producer_commit(pp)
                pp.advance()
            tile += grid_size
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

        nkb = cute.size(tCrA, mode=[2])
        cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NSTAGE)
        cs.reset_count()
        tile = ctid_m
        while tile < ntiles:
            c_tile = cute.local_tile(c, (BM, BN), (tile, ctid_n))
            tCc = thr.partition_C(c_tile)
            acc = cute.make_rmem_tensor(tCc.shape[:3], ACC)
            acc.fill(0.0)
            for kt in range(nkt):
                pipe.consumer_wait(cs)
                tCsA_v = tCsA_view[None, None, None, cs.index]
                tCsB_v = tCsB_view[None, None, None, cs.index]
                for kb in range(nkb):
                    cute.copy(scA, tCsA_v[None, None, kb], tCrA_v[None, None, kb])
                    cute.copy(scB, tCsB_v[None, None, kb], tCrB_v[None, None, kb])
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb],
                              tCrB[None, None, kb], acc)
                pipe.consumer_release(cs)
                cs.advance()
            for i in range(cute.size(acc)):
                tCc[i] = acc[i]
            tile += grid_size


@cute.jit
def run(a, b, c, nkt, ntiles, grid_size, ntile_n):
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
    _body(atom_a, tma_a, atom_b, tma_b, tiled_mma, c, nkt, ntiles, grid_size,
          a_smem_staged, b_smem_staged,
          la.is_m_major_a(), lb.is_n_major_b()).launch(
        grid=[ntile_n, min(grid_size, ntiles), 1], block=[NTHREADS, 1, 1])


GRID = 312


def check():
    M, N, K = 512, 384, 1024
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    ntile_n, ntile_m = N // BN, M // BM
    gy = max(1, -(-GRID // ntile_n))
    run(from_dlpack(a), from_dlpack(b), from_dlpack(c),
        K // TK, ntile_m, gy, ntile_n)
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
    gy = max(1, -(-GRID // ntile_n))
    compiled = cute.compile(run, da, db, dc, nkt, ntile_m, gy, ntile_n)
    compiled(da, db, dc, nkt, ntile_m, gy, ntile_n)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        compiled(da, db, dc, nkt, ntile_m, gy, ntile_n)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"[bench] {M}x{N}x{K}  {ms*1e3:.1f} us  {tflops:.1f} TF/s  "
          f"(grid {ntile_n}x{gy} = {ntile_n*gy} CTAs, {ntile_m} m-tiles)")


if __name__ == "__main__":
    if not check():
        print("FAIL: correctness gate")
        sys.exit(1)
    print("PASS: correctness")
    for shape in [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]:
        bench(*shape)
