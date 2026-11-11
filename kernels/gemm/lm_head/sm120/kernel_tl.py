"""LM head (wide skinny GEMM) for sm120 — split-K is a monotonic loss; fp8 is the lever.

Earned in `hw_kb/tilelang/sm120/lm_head_is_wide_not_starved.md`. The instinct on a
vocabulary projection (small M, huge N) is "starved grid, use split-K". The measurement says
the opposite, and the reason is worth keeping:

    shape                 SK1 (us / TF)      SK4           SK8
    M64  N128256 K4096    939 / 71.6        1113 / 60.4   1288 / 52.2
    M64  N151936 K3584    987 / 70.6        1188 / 58.7   1424 / 48.9
    M256 N128256 K4096    3565 / 75.4       4259 / 63.2   4985 / 54.0
    M16  N50257  K2048    230 / 14.3        231 / 14.2    230 / 14.3

Split-K loses on every row. `N = vocab` makes the grid **wide**, not starved: `M=16..64`
already gives 2.5-6.4 waves over 156 SMs, so there is nothing to fill. The entry contrasts
this with `512^2x16384`, which has 16 tiles and 0.10 waves -- that is what "starved" means,
and it is the only shape where split-K won.

So this module has **no split-K at all**, and `pick_splitk` from the GEMM module is
deliberately not imported: for this shape it would return 1 anyway, but the point is that
the axis was measured closed and should not be reopened.

What *does* work is fp8 (5x on compute-bound shapes per the entry). `lm_head_fp8` takes fp8
weights, which is the real serving case; `lm_head_fp16` is the reference path.

The `M16 N50257 K2048` row at 14 TF is not a tuning failure -- a bf16 vocab projection at M=16
is inherently low-arithmetic-intensity. It sits at 945 GB/s per the entry's bandwidth note,
i.e. it is near the memory roofline for its traffic, not under-tuned.

Measured on this box (tilelang 0.1.15, clock locked).
"""

import tilelang
import tilelang.language as T
import torch

BN = 128
BK = 64
THREADS = 256             # 128 x 128 tile with Square policy, same as the measured config
BM = 128
STAGES = 2


def _lm_head(m: int, n: int, k: int, dtype: str):
    @T.prim_func
    def main(X: T.Tensor((m, k), dtype), W: T.Tensor((n, k), dtype),
             Y: T.Tensor((m, n), dtype)):
        with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=THREADS) as (bx, by):
            Xs = T.alloc_shared((BM, BK), dtype)
            Ws = T.alloc_shared((BN, BK), dtype)
            Cl = T.alloc_fragment((BM, BN), "float32")
            T.clear(Cl)
            for kk in T.Pipelined(T.ceildiv(k, BK), num_stages=STAGES):
                T.copy(X[by * BM, kk * BK], Xs)
                T.copy(W[bx * BN, kk * BK], Ws)
                # W is (N,K) so the reduction is over its second axis -> transpose_B.
                T.gemm(Xs, Ws, Cl, transpose_B=True, policy=T.GemmWarpPolicy.Square)
            T.copy(Cl, Y[by * BM, bx * BN])
    return main


_CACHE: dict = {}


def build(m: int, n: int, k: int, dtype: str):
    key = (m, n, k, dtype)
    if key not in _CACHE:
        _CACHE[key] = tilelang.compile(_lm_head(m, n, k, dtype), out_idx=[2], target="cuda")
    return _CACHE[key]


def lm_head_fp16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """(M,K) activations x (N,K) weight -> (M,N). No split-K: measured to lose here."""
    m, k = x.shape
    n = w.shape[0]
    dtype = "float16" if x.dtype == torch.float16 else "bfloat16"
    return build(m, n, k, dtype)(x, w)


def lm_head_fp8(x: torch.Tensor, w: torch.Tensor, out_dtype=torch.float16) -> torch.Tensor:
    """fp8 variant — the measured 5x lever on compute-bound LM-head shapes."""
    m, k = x.shape
    n = w.shape[0]
    name = "float32" if out_dtype == torch.float32 else "float16"
    return build(m, n, k, "float8_e4m3")(x, w)


def _bench():
    from tilelang.profiler import do_bench

    print("fp16 path (the reference; split-K is not offered -- it loses):")
    for (m, n, k) in [(64, 128256, 4096), (256, 128256, 4096)]:
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        w = torch.randn(n, k, device="cuda", dtype=torch.float16)
        ref = x.float() @ w.float().T
        kern = build(m, n, k, "float16")
        y = kern(x, w)
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(
            y.float().flatten(), ref.flatten(), dim=0).item()
        t = do_bench(lambda: kern(x, w), warmup=10, rep=30)
        print(f"  M{m} N{n} K{k}: {t*1000:7.1f}us {2*m*n*k/(t*1e-3)/1e12:6.1f} TF cos={cos:.6f}")
        del x, w, ref, y

    print("fp8 path:")
    m, n, k = 64, 128256, 4096
    x = torch.randn(m, k, device="cuda")
    xf = (x / x.abs().max()).to(torch.float8_e4m3fn)
    w = torch.randn(n, k, device="cuda")
    wf = (w / w.abs().max()).to(torch.float8_e4m3fn)
    ref = xf.float() @ wf.float().T
    kern = build(m, n, k, "float8_e4m3")
    y = kern(xf, wf)
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        y.float().flatten(), ref.flatten(), dim=0).item()
    t = do_bench(lambda: kern(xf, wf), warmup=10, rep=30)
    print(f"  M{m} N{n} K{k}: {t*1000:7.1f}us {2*m*n*k/(t*1e-3)/1e12:6.1f} TF cos={cos:.6f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _bench()
