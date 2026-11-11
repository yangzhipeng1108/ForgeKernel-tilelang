"""f16 / bf16 GEMM for sm120 — the baseline the fp8 entry is measured against.

Earned in `hw_kb/tilelang/sm120/gemm_dtype_throughput.md` and
`gemm_operand_layout_and_policy.md`. Three readings shape this file:

  * **f16 and bf16 are within 2 % of each other** (141 vs 144 TFLOP/s at BK=32). The choice
    between them is numerics, never speed. `dtype` is therefore a parameter, not a constant.
  * **BK is flat for f16** — BK 32 -> 48 moves 141 -> 142, i.e. nothing. Enlarging BK only
    pays once the operand is fp8, so this kernel keeps BK at 32 with two stages (32 KiB).
  * **`Square` beats `FullRow`** — 133.3 vs 128.6 TFLOP/s, and NCU shows why: `Square` issues
    3.36e7 shared-load wavefronts against 4.19e7, 25 % fewer for the same math. The policy
    is fixed to `Square` rather than exposed, because exposing it invites the slower choice.

If you want speed on this card, do not "try bf16" or "try a bigger BK" — both are measured
ties. Use fp8 (`kernels/gemm/fp8_gemm/sm120/`), which is 2.1x.

Measured on this box (tilelang 0.1.15, 128x128 tile, clock locked):
    f16 BK32  141 TFLOP/s   bf16 BK32  144 TFLOP/s   cos = 1.0 both
"""

import tilelang
import tilelang.language as T
import torch

BM = BN = 128
BK = 32                  # flat for 2-byte operands; bigger BK only pays for fp8
THREADS = 256
STAGES = 2


def _gemm(m: int, n: int, k: int, dtype: str):
    @T.prim_func
    def main(A: T.Tensor((m, k), dtype), B: T.Tensor((k, n), dtype),
             C: T.Tensor((m, n), dtype)):
        with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=THREADS) as (bx, by):
            As = T.alloc_shared((BM, BK), dtype)
            Bs = T.alloc_shared((BK, BN), dtype)
            Cl = T.alloc_fragment((BM, BN), "float32")
            T.clear(Cl)
            for kk in T.Pipelined(T.ceildiv(k, BK), num_stages=STAGES):
                T.copy(A[by * BM, kk * BK], As)
                T.copy(B[kk * BK, bx * BN], Bs)
                # Square is the measured winner; do not expose the policy.
                T.gemm(As, Bs, Cl, policy=T.GemmWarpPolicy.Square)
            T.copy(Cl, C[by * BM, bx * BN])
    return main


_CACHE: dict = {}


def build(m: int, n: int, k: int, dtype: str = "float16"):
    key = (m, n, k, dtype)
    if key not in _CACHE:
        _CACHE[key] = tilelang.compile(_gemm(m, n, k, dtype), out_idx=[2], target="cuda")
    return _CACHE[key]


def gemm_fp16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(M,K) x (K,N) -> (M,N), same 2-byte dtype as the input. FP32 accumulation inside."""
    m, k = a.shape
    n = b.shape[1]
    dtype = "float16" if a.dtype == torch.float16 else "bfloat16"
    return build(m, n, k, dtype)(a, b)


def _bench():
    from tilelang.profiler import do_bench

    for dtype, tdt in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        for (m, n, k) in [(4096, 4096, 4096), (8192, 8192, 8192)]:
            a = torch.randn(m, k, device="cuda", dtype=tdt)
            b = torch.randn(k, n, device="cuda", dtype=tdt)
            ref = a.float() @ b.float()
            kern = build(m, n, k, dtype)
            c = kern(a, b)
            torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(
                c.float().flatten(), ref.flatten(), dim=0).item()
            t = do_bench(lambda: kern(a, b), warmup=15, rep=50)
            print(f"{dtype} {m}x{n}x{k}: {2*m*n*k/(t*1e-3)/1e12:7.1f} TFLOP/s cos={cos:.6f}")
            del a, b, ref, c


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _bench()
