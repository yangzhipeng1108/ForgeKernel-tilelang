"""Hand-written fp16 GEMM on sm120: MMA path verified end to end.

PROBE: PASS. Run inside the CuTe DSL container:

    docker exec cutedsl_probe bash -lc "cd /w && python3 probe_sm120_wmma_gemm_computes.py"

One CTA, 128 threads, computes a full 128x128 output over K=64:
relative error 2.3e-7, every element correct. No shipped kernel involved.

This is the compute half that was blocked after the TMA load path was solved. The
three things that had to be right (all learned the hard way):

 1. ldmatrix is mandatory. make_fragment_A/B only *allocate* the register tile;
    the data has to be copied smem -> register with
    cute.copy(smem_tiled_copy_A, tCsA_v, tCrA_v) before cute.gemm. Skipping it
    leaves the accumulator at 0.0 and the kernel "runs" returning zeros.

 2. acc_shape is tCc.shape[:3], not tCc.shape. The C partition carries a 4th
    pipelined-accumulator mode that make_rmem_tensor must not see.

 3. get_slice(tidx), not get_slice(tidx % 128), on the ldmatrix copy atoms, and
    the atom_layout is (2,2,1) = 4 MMA warps -- the GeForce example's split.

The next step, joining this to the TMA load path, hits an ldmatrix alignment
error (see ../from_scratch_tma_load_path_works.md) because the TMA path needs a
swizzled smem layout that the ldsm atom will not accept at 16-bit alignment.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

M = N = 128
K = 64
DT = cutlass.Float16
ACC = cutlass.Float32
MMA_TILER = (2, 2, 1)
MMA_INST = (16, 8, 16)


@cute.struct
class Storage:
    sA: cute.struct.Align[cute.struct.MemRange[DT, M * K], 1024]
    sB: cute.struct.Align[cute.struct.MemRange[DT, N * K], 1024]


@cute.kernel
def _body(a, b, c):
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(Storage)
    sA = st.sA.get_tensor(cute.make_layout((M, K)))
    sB = st.sB.get_tensor(cute.make_layout((N, K)))

    for r in range(M * K // 128):
        p = r * 128 + tidx
        sA[p // K, p % K] = a[p // K, p % K]
    for r in range(N * K // 128):
        p = r * 128 + tidx
        sB[p // K, p % K] = b[p // K, p % K]
    cute.arch.sync_threads()

    op = cute.nvgpu.warp.MmaF16BF16Op(DT, ACC, MMA_INST)
    tiled_mma = cute.make_tiled_mma(op, cute.make_layout(MMA_TILER))
    thr = tiled_mma.get_slice(tidx)

    tCsA = thr.partition_A(sA)
    tCsB = thr.partition_B(sB)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)

    ldm_a = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), DT)
    ldm_b = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), DT)
    scA = cute.make_tiled_copy_A(ldm_a, tiled_mma)
    scB = cute.make_tiled_copy_B(ldm_b, tiled_mma)
    thr_cA = scA.get_slice(tidx)
    thr_cB = scB.get_slice(tidx)
    tCsA_v = thr_cA.partition_S(sA)
    tCrA_v = thr_cA.retile(tCrA)
    tCsB_v = thr_cB.partition_S(sB)
    tCrB_v = thr_cB.retile(tCrB)

    tCc = thr.partition_C(c)
    acc = cute.make_rmem_tensor(tCc.shape[:3], ACC)
    acc.fill(0.0)

    nk = cute.size(tCrA, mode=[2])
    for kb in range(nk):
        cute.copy(scA, tCsA_v[None, None, kb], tCrA_v[None, None, kb])
        cute.copy(scB, tCsB_v[None, None, kb], tCrB_v[None, None, kb])
        cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)

    for i in range(cute.size(acc)):
        tCc[i] = acc[i]


@cute.jit
def run(a, b, c):
    _body(a, b, c).launch(grid=[1, 1, 1], block=[128, 1, 1])


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
