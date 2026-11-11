"""Dual-GPU data-parallel benchmark harness for TileLang kernels.

Measures how well two GPUs scale on independent TileLang work. TileLang has no
multi-GPU API; each device gets its own torch stream via ``torch.cuda.set_device``,
so data parallelism is "run the same kernel on two devices and synchronize".

Two modes:

  full  each GPU runs the FULL problem independently (replica test). Reports the
        aggregate TFLOPS and scaling vs a single GPU.
  dp    the problem's N dimension is split across GPUs, so the two devices
        solve one problem together. Reports wall time for the full problem and
        speedup vs a single GPU.

Timing uses a dedicated CUDA Event per device; each worker thread synchronizes
its own events, and the round time is the max across devices.

IMPORTANT: lock BOTH GPU clocks before trusting the numbers, e.g.
    sudo nvidia-smi -i 0 -lgc 2000,2000
    sudo nvidia-smi -i 1 -lgc 2000,2000
An unlocked GPU idles at a low clock and makes concurrent scaling look worse
than it is.

Usage:
    python dual_gpu_bench.py --M 8192 --N 8192 --K 8192 --mode full
    python dual_gpu_bench.py --M 8192 --N 8192 --K 8192 --mode dp --check
"""

import argparse
import threading

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def build_gemm(M, N, K, BM=128, BN=128, BK=64, stages=2, threads=128):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
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


def _make_data(dev, M, N, K):
    with torch.cuda.device(dev):
        A = torch.randn(M, K, device=f"cuda:{dev}", dtype=torch.float16) * 0.1
        B = torch.randn(K, N, device=f"cuda:{dev}", dtype=torch.float16) * 0.1
    return A, B


def _time_dev(dev, fn, reps, res):
    torch.cuda.set_device(dev)
    for _ in range(5):
        fn()
    torch.cuda.synchronize(dev)
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(reps):
        fn()
    en.record()
    en.synchronize()
    res[dev] = st.elapsed_time(en) / reps


def run_concurrent(devices, fns, reps):
    res = {}
    ths = [
        threading.Thread(target=_time_dev, args=(d, fns[d], reps, res))
        for d in devices
    ]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return res


def _verify(kernel, M, N, K):
    with torch.cuda.device(0):
        A = torch.randn(M, K, device="cuda:0", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda:0", dtype=torch.float16)
        got = kernel(A, B).float()
        ref = A.float() @ B.float()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten(), ref.flatten(), dim=0
        ).item()
    print(f"[verify] cosine vs fp32 matmul: {cos:.6f}")
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8192)
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--K", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--mode", choices=["full", "dp"], default="full")
    ap.add_argument("--check", action="store_true", help="verify correctness first")
    a = ap.parse_args()

    n_dev = torch.cuda.device_count()
    devs = list(range(n_dev))
    print(f"devices={n_dev}  problem={a.M}x{a.N}x{a.K}  mode={a.mode}")
    if n_dev < 2:
        print("need >= 2 GPUs for scaling measurement")

    full_flop = 2 * a.M * a.N * a.K

    if a.mode == "full":
        if a.check:
            _verify(tilelang.compile(build_gemm(a.M, a.N, a.K), out_idx=[2],
                                     target="cuda"), a.M, a.N, a.K)
        kernel = tilelang.compile(build_gemm(a.M, a.N, a.K), out_idx=[2],
                                  target="cuda")
        data = {d: _make_data(d, a.M, a.N, a.K) for d in devs}
        t1 = do_bench(lambda: kernel(*data[0]), warmup=20, rep=a.reps)
        print(f"[1 GPU]  {t1 * 1000:.1f}us  {full_flop / (t1 * 1e-3) / 1e12:.1f} TF")
        if n_dev > 1:
            fns = {d: (lambda d=d: kernel(*data[d])) for d in devs}
            res = run_concurrent(devs, fns, a.reps)
            ts = max(res.values())
            agg = n_dev * full_flop / (ts * 1e-3)
            base = full_flop / (t1 * 1e-3)
            print(f"[{n_dev} GPU full-replica]  {ts * 1000:.1f}us round  "
                  f"aggregate {agg / 1e12:.1f} TF  scaling {agg / base:.2f}x")
    else:
        if a.check:
            _verify(tilelang.compile(build_gemm(a.M, a.N, a.K), out_idx=[2],
                                     target="cuda"), a.M, a.N, a.K)
        n = max(1, a.N // n_dev)
        kfull = tilelang.compile(build_gemm(a.M, a.N, a.K), out_idx=[2],
                                 target="cuda")
        dfull = _make_data(0, a.M, a.N, a.K)
        t1 = do_bench(lambda: kfull(*dfull), warmup=20, rep=a.reps)
        print(f"[1 GPU full]  {t1 * 1000:.1f}us  {full_flop / (t1 * 1e-3) / 1e12:.1f} TF")
        kpart = tilelang.compile(build_gemm(a.M, n, a.K), out_idx=[2],
                                 target="cuda")
        data = {d: _make_data(d, a.M, n, a.K) for d in devs}
        fns = {d: (lambda d=d: kpart(*data[d])) for d in devs}
        res = run_concurrent(devs, fns, a.reps)
        ts = max(res.values())
        print(f"[{n_dev} GPU DP N-split]  {ts * 1000:.1f}us for FULL problem  "
              f"speedup {t1 / ts:.2f}x  aggregate {full_flop / (ts * 1e-3) / 1e12:.1f} TF")

    torch.cuda.synchronize()
    print("done (devices synchronized)")


if __name__ == "__main__":
    main()
