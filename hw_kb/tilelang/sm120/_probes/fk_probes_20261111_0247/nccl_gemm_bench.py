"""Multi-process NCCL harness for distributed TileLang GEMM (TP / PP reference).

This is the production-shaped counterpart to ``dual_gpu_bench.py``: instead of
threads in one process, it launches one process per GPU via ``torchrun`` and
communicates through ``torch.distributed`` (NCCL). TileLang itself has no
multi-GPU API, so the kernel runs per-rank and the collective is done in torch.

Modes:
  tp   tensor parallel over K: each rank computes a partial GEMM on its K-slice,
       then the partials are all-reduced (sum) into the full result.
  pp   pipeline parallel reference: stage 0 computes layer A and sends the
       activation to stage 1, which computes layer B (no micro-batch yet).

CRITICAL for this box (2x RTX 6000D, no NVLink, GPU0<->GPU1 = SYS):
    export NCCL_P2P_LEVEL=SYS
Without it NCCL silently falls back to host SHM/direct and all-reduce runs at
~12.5 GB/s. With it, NCCL uses CUDA P2P and all-reduce reaches ~34.5 GB/s
(broadcast ~45 GB/s, near the 45 GB/s raw P2P ceiling).

Usage:
    NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node=2 nccl_gemm_bench.py \
        --mode tp --M 8192 --N 4096 --K 65536
    NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node=2 nccl_gemm_bench.py \
        --mode pp --M 4096 --N 4096 --K 4096
"""

import argparse
import os

import torch
import torch.distributed as dist
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def build_gemm(M, N, K, out_dtype="float32", BM=128, BN=128, BK=64,
               stages=2, threads=128):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            a_s = T.alloc_shared((BM, BK), "float16")
            b_s = T.alloc_shared((BK, BN), "float16")
            acc = T.alloc_fragment((BM, BN), "float32")
            T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                T.copy(A[by * BM, kk * BK], a_s)
                T.copy(B[kk * BK, bx * BN], b_s)
                T.gemm(a_s, b_s, acc)
            T.copy(acc, C[by * BM, bx * BN])

    return main


def _timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    en.synchronize()
    return st.elapsed_time(en) / iters


def run_tp(rank, world, M, N, K):
    ks = K // world
    kfull = tilelang.compile(build_gemm(M, N, K), out_idx=[2], target="cuda")
    kpart = tilelang.compile(build_gemm(M, N, ks), out_idx=[2], target="cuda")
    dev = torch.device("cuda", rank)
    torch.manual_seed(0)
    A = torch.randn(M, K, device=dev, dtype=torch.float16) * 0.1
    B = torch.randn(K, N, device=dev, dtype=torch.float16) * 0.1

    t1 = do_bench(lambda: kfull(A, B), warmup=15, rep=30) if rank == 0 else 0.0
    dist.barrier()

    Aslice = A[:, rank * ks:(rank + 1) * ks].contiguous()
    Bslice = B[rank * ks:(rank + 1) * ks].contiguous()

    def tp_step():
        c = kpart(Aslice, Bslice)
        dist.all_reduce(c)
        return c

    tp_ms = _timed(tp_step)
    c_tp = tp_step()
    dist.barrier()

    if rank == 0:
        c_full = kfull(A, B)
        cos = torch.nn.functional.cosine_similarity(
            c_tp.flatten(), c_full.flatten(), dim=0
        ).item()
        flop = 2 * M * N * K
        print(f"TP world={world} {M}x{N}x{K} | 1GPU {t1 * 1000:.1f}us "
              f"{flop / (t1 * 1e-3) / 1e12:.1f}TF | TP {tp_ms * 1000:.1f}us "
              f"speedup {t1 / tp_ms:.2f}x | cos={cos:.6f}")


def run_pp(rank, world, M, N, K, nlayer, mb):
    # PP: rank 0 owns the first nlayer/world layers, rank 1 the rest; activations
    # are sent layer-group to layer-group. The batch is split into `mb`
    # micro-batches, which lets the two stages work on different micro-batches at
    # once (pipeline fill) instead of strict lock-step.
    #
    # NOTE the measured ceiling on this box: even with micro-batching the speedup
    # tops out near 1.69x (8 layers / 8 micro-batches). Pure-Python isend/irecv
    # does NOT overlap the send with the next GPU compute (the kernel launch is
    # synchronous on the default stream), so the pipeline bubble never fully
    # closes. True overlap needs CUDA-stream + event orchestration or a framework.
    per_stage = nlayer // world
    mbm = M // mb
    kmb = tilelang.compile(build_gemm(mbm, N, N, out_dtype="float16"),
                           out_idx=[2], target="cuda")
    kfull = tilelang.compile(build_gemm(M, N, N, out_dtype="float16"),
                             out_idx=[2], target="cuda")
    dev = torch.device("cuda", rank)
    torch.manual_seed(0)
    Ws = [torch.randn(N, N, device=dev, dtype=torch.float16) * 0.1
          for _ in range(per_stage)]

    def pp_step():
        for _ in range(mb):
            x = torch.randn(mbm, N, device=dev, dtype=torch.float16) * 0.1
            if rank == 0:
                for w in Ws:
                    x = kmb(x, w)
                dist.send(x, dst=1)
            else:
                got = torch.empty(mbm, N, device=dev, dtype=torch.float16)
                dist.recv(got, src=0)
                x = got
                for w in Ws:
                    x = kmb(x, w)

    pp_ms = _timed(pp_step, iters=10, warmup=3)
    if rank == 0:
        allW = [torch.randn(N, N, device=dev, dtype=torch.float16) * 0.1
                for _ in range(nlayer)]
        A = torch.randn(M, N, device=dev, dtype=torch.float16) * 0.1

        def ref():
            x = A
            for w in allW:
                x = kfull(x, w)
            return x

        t_ref = _timed(ref, iters=10, warmup=3)
        print(f"PP world={world} {M}x{N} nlayer={nlayer} mb={mb} | "
              f"1GPU all {nlayer}L {t_ref * 1000:.1f}us | PP {pp_ms * 1000:.1f}us "
              f"speedup {t_ref / pp_ms:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--mode", choices=["tp", "pp"], default="tp")
    ap.add_argument("--nlayer", type=int, default=8,
                    help="PP: total layers (split evenly across ranks)")
    ap.add_argument("--microbatch", type=int, default=8,
                    help="PP: number of micro-batches to pipeline")
    a = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    if os.environ.get("NCCL_P2P_LEVEL") != "SYS":
        if rank == 0:
            print("[warn] NCCL_P2P_LEVEL != SYS; NCCL may fall back to SHM "
                  "and all-reduce will be ~3x slower on this box")

    if a.mode == "tp":
        run_tp(rank, world, a.M, a.N, a.K)
    else:
        run_pp(rank, world, a.M, a.N, a.K, a.nlayer, a.microbatch)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
