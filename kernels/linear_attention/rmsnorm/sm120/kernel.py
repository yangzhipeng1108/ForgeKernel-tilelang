"""RMSNorm for sm120 — the one regime that saturates HBM, so the tile must be SMALL.

Earned in `hw_kb/tilelang/sm120/memory_bound_elementwise_saturates_hbm.md`. This is the only
kernel class on this card where the roofline, not the tensor core, is the limit, and that
inverts the usual advice:

    BM (rows/block)   threads   time (µs)   GB/s
    1                 128       103.3       1299
    2                 128       100.8       1331   <- best
    4                 128       101.8       1318
    8                 128       111.3       1206
    16                128       184.0       729
    32                128       268.9       499
    torch fp32 path   --        1018.6      132

**`BM = 2` wins and BM = 32 is 2.7x slower.** A large tile is the right instinct for a
compute-bound kernel and exactly wrong here: fewer, fatter blocks leave the memory
subsystem idle between bursts. 1331 GB/s is above the 1194 GB/s pure-fp16-copy roofline
because the norm reads a row and writes it back with no extra traffic.

Swizzle still ties (1251-1261 GB/s, +-0.8 %) -- a streaming kernel has each block read a
disjoint slice, so L2 layout cannot matter. That axis is closed; do not reopen it.

The two-pass reduction below (sum of squares, then scale) is what the measurement used. It
fits in one block because a row of D elements is small; do not add a grid-wide barrier.

Measured on this box (tilelang 0.1.15, fp16, clock locked): 1331 GB/s, ~10x torch.
"""

import tilelang
import tilelang.language as T
import torch

BM = 2              # rows per block -- the measured winner; larger is much slower
THREADS = 128
EPS = 1e-6


def _rmsnorm(m: int, d: int, dtype: str):
    @T.prim_func
    def main(X: T.Tensor((m, d), dtype), W: T.Tensor((d,), dtype),
             Y: T.Tensor((m, d), dtype)):
        with T.Kernel(T.ceildiv(m, BM), threads=THREADS) as bx:
            Xs = T.alloc_shared((BM, d), dtype)
            Ws = T.alloc_shared((d,), dtype)
            xf = T.alloc_fragment((BM, d), "float32")
            sq2 = T.alloc_fragment((BM, d), "float32")
            sq = T.alloc_fragment((BM,), "float32")
            inv = T.alloc_fragment((BM,), "float32")

            T.copy(X[bx * BM, 0], Xs)
            T.copy(W, Ws)

            for i, j in T.Parallel(BM, d):
                xf[i, j] = T.cast(Xs[i, j], "float32")
            for i, j in T.Parallel(BM, d):
                sq2[i, j] = xf[i, j] * xf[i, j]
            # row reduction via reduce_sum, not per-element atomics on a fragment
            T.reduce_sum(sq2, sq, dim=1)
            for i in T.Parallel(BM):
                inv[i] = T.rsqrt(sq[i] / d + EPS)
            for i, j in T.Parallel(BM, d):
                xf[i, j] = xf[i, j] * inv[i] * T.cast(Ws[j], "float32")
            for i, j in T.Parallel(BM, d):
                Xs[i, j] = T.cast(xf[i, j], dtype)
            T.copy(Xs, Y[bx * BM, 0])
    return main


_CACHE: dict = {}


def build(m: int, d: int, dtype: str = "float16"):
    key = (m, d, dtype)
    if key not in _CACHE:
        _CACHE[key] = tilelang.compile(_rmsnorm(m, d, dtype), out_idx=[2], target="cuda")
    return _CACHE[key]


def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """(M,D) x (D,) -> (M,D). Row-wise RMS normalisation with a learned weight."""
    m, d = x.shape
    dtype = "float16" if x.dtype == torch.float16 else "bfloat16"
    return build(m, d, dtype)(x, w)


def _bench():
    from tilelang.profiler import do_bench

    m, d = 8192, 2560
    x = torch.randn(m, d, device="cuda", dtype=torch.float16)
    w = torch.randn(d, device="cuda", dtype=torch.float16)
    kern = build(m, d, "float16")
    y = kern(x, w)
    torch.cuda.synchronize()
    ref = (x.float() / x.float().pow(2).mean(-1, keepdim=True).add(EPS).sqrt()) * w.float()
    cos = torch.nn.functional.cosine_similarity(
        y.float().flatten(), ref.flatten(), dim=0).item()
    t = do_bench(lambda: kern(x, w), warmup=15, rep=50)
    gb = 2 * m * d * 2 / 1e9
    print(f"rmsnorm {m}x{d} BM{BM}: {t*1000:7.1f}us {gb/(t*1e-3):7.1f} GB/s cos={cos:.6f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _bench()
