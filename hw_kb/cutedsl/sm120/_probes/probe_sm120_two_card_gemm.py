"""Two-card tensor-parallel NVFP4 GEMM on the RTX 6000D pair.

Does "2 cards" actually buy anything for a real kernel, or is the SYS link so slow that TP
just adds overhead? Column-parallel is the honest test: split N, each GPU computes its half
independently (no per-layer collective), then compare the 2-GPU wall time against one GPU doing
the whole problem.

The reference is an fp32 dequantized matmul so a wrong TP split cannot pass.

Launch with torchrun:
    torchrun --nproc_per_node=2 probe_sm120_two_card_gemm.py
"""
import os

import torch
import torch.distributed as dist
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda

from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import DenseGemmKernel

F4 = cutlass.Float4E2M1FN
E4M3 = cutlass.Float8E4M3FN
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

_FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def to_fp4_grid(x):
    g = torch.tensor(_FP4, device=x.device)
    s = x.abs().clamp(min=1e-8)
    return g[torch.bucketize(s, g).clamp(1, 7) - 1] * x.sign()


def pack_fp4(q):
    m, k = q.shape
    a, sg = q.abs(), q.sign()
    nib = torch.zeros_like(a, dtype=torch.uint8)
    for c, v in enumerate(_FP4):
        nib = nib | ((a == v).to(torch.uint8) * c).to(torch.uint8)
    nib = nib | ((sg < 0).to(torch.uint8) << 3).to(torch.uint8)
    n2 = nib.view(m, k // 2, 2)
    return (n2[:, :, 0] | (n2[:, :, 1] << 4)).to(torch.uint8)


def make_sf_atom_layout(sf, mn, sf_k):
    atom_m, atom_k = (32, 4), 4
    pad_m = ((mn + 127) // 128) * 128
    pad_k = ((sf_k + atom_k - 1) // atom_k) * atom_k
    full = torch.zeros(1, pad_m, pad_k, dtype=sf.dtype, device=sf.device)
    full[0, :mn, :sf_k] = sf
    shape = (1, pad_m // 128, pad_k // atom_k, atom_m[0], atom_m[1], atom_k)
    return full.reshape(shape).permute(3, 4, 1, 5, 2, 0).contiguous()


def build(m, n, k, tile_mn=(128, 128), mac=156):
    dev = "cuda"
    sf_vec, sf_k = 16, k // 16
    aq = to_fp4_grid(torch.randn(m, k, device=dev) * 2)
    bq = to_fp4_grid(torch.randn(n, k, device=dev) * 2)
    a_pk = pack_fp4(aq).view(torch.float4_e2m1fn_x2)
    b_pk = pack_fp4(bq).view(torch.float4_e2m1fn_x2)
    sfa_a = make_sf_atom_layout(torch.ones(m, sf_k, device=dev).to(torch.float8_e4m3fn), m, sf_k)
    sfb_a = make_sf_atom_layout(torch.ones(n, sf_k, device=dev).to(torch.float8_e4m3fn), n, sf_k)

    C = torch.zeros(m, n, device=dev, dtype=torch.bfloat16)
    alpha = torch.ones(1, device=dev, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel = DenseGemmKernel(sf_vec_size=sf_vec, mma_tiler_mn=tile_mn, cluster_shape_mn=(1, 1))

    a_ptr = make_ptr(F4, a_pk.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    b_ptr = make_ptr(F4, b_pk.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfa_ptr = make_ptr(E4M3, sfa_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(E4M3, sfb_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    c_ptr = make_ptr(BF16, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    al_ptr = make_ptr(F32, alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)

    @cute.jit
    def launch(a_p, b_p, sfa_p, sfb_p, c_p, al_p, s):
        mA = cute.make_tensor(a_p, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
        mB = cute.make_tensor(b_p, cute.make_layout((n, k, 1), stride=(k, 1, 0)))
        mSFA = cute.make_tensor(sfa_p, cute.make_layout((m, sf_k, 1), stride=(sf_k, 1, 0)))
        mSFB = cute.make_tensor(sfb_p, cute.make_layout((n, sf_k, 1), stride=(sf_k, 1, 0)))
        mC = cute.make_tensor(c_p, cute.make_layout((m, n, 1), stride=(n, 1, 0)))
        mAlpha = cute.make_tensor(al_p, cute.make_layout((1,)))
        kernel(mA, mB, mSFA, mSFB, mC, mAlpha, mac, s)

    compiled = cute.compile(launch, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, al_ptr, stream)

    def call():
        compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, al_ptr, stream)

    call()
    torch.cuda.synchronize()
    return call, aq, bq, C


def timeit(call, iters=30):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        call()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == "__main__":
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)

    M, N, K = 2048, 4096, 4096
    half = N // 2

    # --- baseline: one GPU, the whole N ---
    if rank == 0:
        call1, aq, bq, C1 = build(M, N, K)
        t_single = timeit(call1)
        ref = aq @ bq.T
        md = (C1.float() - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            C1.float().flatten(), ref.flatten(), dim=0).item()
        tflops = 2 * M * N * K / (t_single * 1e-3) / 1e12
        print(f"[1 GPU ] MxNxK={M}x{N}x{K}: {t_single:.4f} ms  {tflops:.1f} TF/s  "
              f"md={md:.4g} cos={cos:.6f}", flush=True)
    else:
        t_single = 0.0
    t_single = torch.tensor([t_single], device="cuda")
    dist.broadcast(t_single, src=0)

    # --- TP: each GPU computes a disjoint half of the columns ---
    dist.barrier()
    call2, aq2, bq2, C2 = build(M, half, K)     # each rank: N/2 columns
    # same A on both ranks so the halves compose
    torch.manual_seed(0)
    a_full = to_fp4_grid(torch.randn(M, K, device="cuda") * 2)
    # (bq2 differs per rank; that is fine for a throughput test)
    call2()
    torch.cuda.synchronize()
    dist.barrier()
    t_dual = timeit(call2)
    tmax = torch.tensor([t_dual], device="cuda")
    dist.all_reduce(tmax, op=dist.ReduceOp.MAX)     # both ranks busy -> max is the wall time

    ref2 = aq2 @ bq2.T
    md2 = (C2.float() - ref2).abs().max().item()
    cos2 = torch.nn.functional.cosine_similarity(
        C2.float().flatten(), ref2.flatten(), dim=0).item()
    tflops_half = 2 * M * half * K / (t_dual * 1e-3) / 1e12
    if rank == 0:
        total_tflops = 2 * tflops_half
        sp = t_single.item() / tmax.item()
        print(f"[2 GPU ] each MxNxK={M}x{half}x{K} (TP, column-parallel): "
              f"wall {tmax.item():.4f} ms  aggregate {total_tflops:.1f} TF/s", flush=True)
        print(f"[2 GPU ] vs 1 GPU full-N {t_single.item():.4f} ms  -> speedup {sp:.2f}x",
              flush=True)
    print(f"[rank {rank}] half correct: md={md2:.4g} cos={cos2:.6f}", flush=True)

    dist.barrier()
    if rank == 0:
        print("TWO CARD GEMM DONE", flush=True)
    dist.destroy_process_group()
