"""Row-parallel (input-split) TP layer on the two cards: does the all-reduce eat the win?

Column-parallel scales near-linearly (two_card_parallelism.md) because it has no collective.
Row-parallel is the opposite: every layer ends in an all-reduce of the M x H activation. This
probe builds a real NVFP4 down-projection layer both ways and measures what the collective costs:

  single GPU : one GEMM  M x N x K         (the whole input dim K on one card)
  TP = 2     : each card does M x N x (K/2), then all_reduce the M x N activation

K is the input (intermediate) dimension, which is what row-parallel splits. The TP result is
summed across ranks, so it equals the single-GPU result if the split is correct -- checked
against an fp32 reference so a silent drop cannot pass.

Sweeping M shows the crossover: decode (small M) is collective-latency bound, prefill (large M)
is collective-bandwidth bound, and both are measured against the fp4 compute time.

Launch with torchrun:
    NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node=2 probe_sm120_two_card_layer.py
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


def build(m, n, k, aq, bq, tile_mn=(128, 128), mac=156):
    """Build an NVFP4 GEMM for pre-quantized aq (m,k) and bq (n,k). Returns (call, C)."""
    dev = "cuda"
    sf_vec, sf_k = 16, k // 16
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
    return call, C


def timeit(fn, iters=30):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == "__main__":
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2

    H, I = 2560, 8192                       # hidden, intermediate (down-proj input dim K = I)
    N = H                                   # output dim
    half_i = I // 2

    print(f"rank {rank} up ({torch.cuda.get_device_name(local)})", flush=True)
    dist.barrier()

    # M < 128 is broken in the default kernel: M=16/32/64 return garbage and M=1
    # hits an illegal instruction. The smallest correct M is one full 128-row tile;
    # real small-M decode needs the use_m1_non_tma_* / direct_one_m_tile_scheduler flags.
    for M in (128, 512, 4096):
        torch.manual_seed(0)
        aq = to_fp4_grid(torch.randn(M, I, device="cuda") * 2)     # M x I (input dim = I)
        bq = to_fp4_grid(torch.randn(N, I, device="cuda") * 2)     # N x I
        ref = aq @ bq.T                                            # M x N

        # ---- single GPU (rank 0): full K = I ----
        if rank == 0:
            call1, C1 = build(M, N, I, aq, bq)
            t_single = timeit(call1)
            t_single_t = torch.tensor([t_single], device="cuda")
        else:
            t_single_t = torch.tensor([0.0], device="cuda")
        dist.broadcast(t_single_t, src=0)
        t_single = t_single_t.item()

        # ---- TP=2 row-parallel: rank r owns the r-th half of the input dim ----
        s = rank * half_i
        aq_r = aq[:, s:s + half_i].contiguous()                    # M x I/2
        bq_r = bq[:, s:s + half_i].contiguous()                    # N x I/2
        call2, C2 = build(M, N, half_i, aq_r, bq_r)

        # correctness on a clean single pass, BEFORE any timing loop mutates C2
        C2.zero_()
        call2()
        dist.all_reduce(C2)                                        # rank halves sum == single-GPU
        torch.cuda.synchronize()
        md = (C2.float() - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            C2.float().flatten(), ref.flatten(), dim=0).item()

        # compute-only (both ranks busy), then all-reduce-only, then total
        dist.barrier()
        t_comp = timeit(call2)
        dist.barrier()
        act = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
        t_ar = timeit(lambda: dist.all_reduce(act))
        dist.barrier()

        def tp_total():
            call2()
            dist.all_reduce(C2)                                    # real collective on the output

        dist.barrier()
        t_total = timeit(tp_total)
        dist.barrier()

        if rank == 0:
            tcomp = torch.tensor([t_comp]); tar = torch.tensor([t_total])
            ar_share = t_ar / t_total * 100 if t_total else 0
            speedup = t_single / t_total
            print(f"M={M:5d}: 1GPU={t_single:.4f} ms | TP2 total={t_total:.4f} ms "
                  f"(compute={t_comp:.4f}, allreduce={t_ar:.4f} = {ar_share:.0f}%) "
                  f"-> speedup {speedup:.2f}x  md={md:.4g} cos={cos:.6f}", flush=True)

        dist.barrier()

    if rank == 0:
        print("LAYER TP DONE", flush=True)
    dist.destroy_process_group()
