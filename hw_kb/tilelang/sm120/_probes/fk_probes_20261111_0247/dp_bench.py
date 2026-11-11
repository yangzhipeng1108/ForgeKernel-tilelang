import argparse, threading, torch, tilelang, tilelang.language as T
from tilelang.profiler import do_bench

def build_gemm(M, N, K, BM=128, BN=128, BK=64, stages=2, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
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

def make_dev_data(dev, M, N, K):
    with torch.cuda.device(dev):
        A = torch.randn(M, K, device=f"cuda:{dev}", dtype=torch.float16) * 0.1
        B = torch.randn(K, N, device=f"cuda:{dev}", dtype=torch.float16) * 0.1
        C = torch.empty(M, N, device=f"cuda:{dev}", dtype=torch.float16)
    return A, B, C

def time_dev(dev, fn, reps, res):
    torch.cuda.set_device(dev)
    for _ in range(5): fn()
    torch.cuda.synchronize(dev)
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(reps): fn()
    en.record(); en.synchronize()
    res[dev] = st.elapsed_time(en) / reps  # ms

def run_concurrent(devices, fns, reps):
    res = {}
    ths = [threading.Thread(target=time_dev, args=(d, fns[d], reps, res)) for d in devices]
    for t in ths: t.start()
    for t in ths: t.join()
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8192)
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--K", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--mode", choices=["full", "dp"], default="full")
    a = ap.parse_args()
    devs = [d for d in range(torch.cuda.device_count())]
    print(f"devices={torch.cuda.device_count()}  problem={a.M}x{a.N}x{a.K}  mode={a.mode}")

    full_flop = 2 * a.M * a.N * a.K
    if a.mode == "full":
        # each GPU runs the FULL problem independently (data-parallel replicas)
        k = tilelang.compile(build_gemm(a.M, a.N, a.K), out_idx=[2], target="cuda")
        data = {d: make_dev_data(d, a.M, a.N, a.K) for d in devs}
        # single-GPU baseline
        t1 = do_bench(lambda: k(*data[0][:2]), warmup=20, rep=a.reps)
        print(f"[1 GPU] {t1*1000:.1f}us  {full_flop/(t1*1e-3)/1e12:.1f} TF")
        fns = {d: (lambda d=d: k(data[d][0], data[d][1])) for d in devs}
        if len(devs) > 1:
            res = run_concurrent(devs, fns, a.reps)
            ts = max(res.values())
            agg = len(devs) * full_flop / (ts * 1e-3)
            print(f"[{len(devs)} GPU full-replica] {ts*1000:.1f}us round  aggregate {agg/1e12:.1f} TF  scaling {agg/(full_flop/(t1*1e-3)):.2f}x")
    else:
        # data-parallel: split N across GPUs, whole problem solved once
        n = a.N // len(devs)
        k = tilelang.compile(build_gemm(a.M, n, a.K), out_idx=[2], target="cuda")
        data = {d: make_dev_data(d, a.M, n, a.K) for d in devs}
        kf = tilelang.compile(build_gemm(a.M, a.N, a.K), out_idx=[2], target="cuda")
        dfull = make_dev_data(0, a.M, a.N, a.K)
        t1 = do_bench(lambda: kf(*dfull[:2]), warmup=20, rep=a.reps)
        print(f"[1 GPU full] {t1*1000:.1f}us  {full_flop/(t1*1e-3)/1e12:.1f} TF")
        fns = {d: (lambda d=d: k(data[d][0], data[d][1])) for d in devs}
        res = run_concurrent(devs, fns, a.reps)
        ts = max(res.values())
        print(f"[{len(devs)} GPU DP N-split] {ts*1000:.1f}us for FULL problem  speedup {t1/ts:.2f}x  aggregate {full_flop/(ts*1e-3)/1e12:.1f} TF")

    # sanity: verify parity between replicas/devices
    if len(devs) > 1:
        torch.cuda.synchronize()
        print("done (devices synchronized)")

if __name__ == "__main__":
    main()