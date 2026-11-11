"""fp8 e4m3 GEMM for sm120 — the 2.1x lever on this card.

Earned in `hw_kb/tilelang/sm120/gemm_dtype_throughput.md`. One number matters: **fp8 is
~2.1x f16** (297 vs 141 TFLOP/s). It wins twice — a 1-byte operand halves the shared-memory
footprint so a bigger BK fits, and the tensor core does more per instruction. A f16-only
kernel leaves most of a 6000D idle.

The measured BK sweep at (128,128), stages=2:

    BK   32   64   96  128  256
    TF  269  281  283  297  launch fail (128 KiB > 99 KiB opt-in)

BK=128 is the best and fits comfortably at 64 KiB *because* the operand is fp8. At 256 it
asks for 131072 B and dies on `Failed to set the allowed dynamic shared memory size`.
That is the same 99 KiB ceiling as everything else on this arch, not an fp8 quirk.

PRECISION, HONESTLY
-------------------
The KB reports `cosine = 1.0` against an fp32 reference on unit-normal inputs. That is real
but narrow: e4m3 has 3 mantissa bits, and `torch.randn` keeps |x| under ~4, so products
round-trip closely enough for fp32 accumulation to match to 1e-4. It does **not** mean fp8
is generally accurate. Real activations have outliers and need per-tensor scaling; measure
again before trusting this on a new distribution.

Measured on this box (tilelang 0.1.15, 128x128x128, stages=2, clock locked): 297 TFLOP/s,
f16/bf16 = 141/144 for comparison.
"""

import tilelang
import tilelang.language as T
import torch

BM = BN = BK = 128
THREADS = 256
STAGES = 2
DTYPE = "float8_e4m3"


def _gemm(m: int, n: int, k: int, out_dtype: str, acc_dtype: str):
    @T.prim_func
    def main(A: T.Tensor((m, k), DTYPE), B: T.Tensor((k, n), DTYPE),
             C: T.Tensor((m, n), out_dtype)):
        with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=THREADS) as (bx, by):
            As = T.alloc_shared((BM, BK), DTYPE)
            Bs = T.alloc_shared((BK, BN), DTYPE)
            Cl = T.alloc_fragment((BM, BN), acc_dtype)
            T.clear(Cl)
            for kk in T.Pipelined(T.ceildiv(k, BK), num_stages=STAGES):
                T.copy(A[by * BM, kk * BK], As)
                T.copy(B[kk * BK, bx * BN], Bs)
                T.gemm(As, Bs, Cl, policy=T.GemmWarpPolicy.Square)
            T.copy(Cl, C[by * BM, bx * BN])
    return main


_CACHE: dict = {}


def build(m: int, n: int, k: int, out_dtype: str = "float16"):
    key = (m, n, k, out_dtype)
    if key not in _CACHE:
        acc = "float32" if out_dtype == "float32" else "float32"
        _CACHE[key] = tilelang.compile(_gemm(m, n, k, out_dtype, acc),
                                       out_idx=[2], target="cuda")
    return _CACHE[key]


def fp8_gemm(a: torch.Tensor, b: torch.Tensor, out_dtype=torch.float16) -> torch.Tensor:
    """(M,K) x (K,N) fp8 e4m3 -> (M,N). FP32 accumulation; fp16 or fp32 output."""
    m, k = a.shape
    n = b.shape[1]
    name = "float32" if out_dtype == torch.float32 else "float16"
    return build(m, n, k, name)(a, b)


def _bench():
    from tilelang.profiler import do_bench

    m = n = k = 4096
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(k, n, device="cuda").to(torch.float8_e4m3fn)
    ref = a.float() @ b.float()
    kern = build(m, n, k, "float16")
    c = kern(a, b)
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        c.float().flatten(), ref.flatten(), dim=0).item()
    md = (c.float() - ref).abs().max().item()
    t = do_bench(lambda: kern(a, b), warmup=15, rep=50)
    print(f"fp8_e4m3 {m}x{n}x{k} BK{BK}: {2*m*n*k/(t*1e-3)/1e12:7.1f} TFLOP/s "
          f"cos={cos:.6f} md={md:.3g}")
    print("  (f16 baseline on this card is 141 TFLOP/s -> fp8 is the lever)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _bench()
