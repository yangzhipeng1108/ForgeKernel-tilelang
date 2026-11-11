"""fp8 split-K GEMM for sm120 (RTX 6000D) — what the recipe measured, as a kernel.

Earned in `hw_kb/tilelang/sm120/split_k_fills_a_starved_grid.md`. The claim there is
narrow and this file keeps it narrow: **split-K pays only when the grid is starved**,
and it is a loss once the grid is full.

Measured on this box (tilelang 0.1.15, fp8 e4m3, 128x128x128, stages=2, threads=256):

    shape          plain     SK=2     SK=4     SK=8
    512^2x16384      --     84.4 TF  133.6 TF  187.5 TF   <- starved, split-K wins
    1024^2x8192   170.4 TF  247.3 TF  215.7 TF  204.7 TF
    4096^3        375.3 TF  310.6 TF  277.0 TF  208.4 TF   <- full, split-K loses

`pick_splitk()` below is that verdict as code, so the caller does not have to remember it.

WHY THE CALLING CONVENTION LOOKS LIKE THIS
------------------------------------------
A split-K kernel accumulates with `T.atomic_add`, so C must be zeroed **every call**.
Compiling with `out_idx=[2]` breaks that: the kernel returns a fresh buffer and never
writes the C the caller zeroed, so `C.zero_()` resets an object the kernel does not read
and the second launch accumulates onto the first. The archived probe did exactly this and
reported md=455 / md=1.49e5. So this kernel compiles WITHOUT out_idx and takes C as an
explicit in/out argument. Do not "simplify" it back.
"""

import tilelang
import tilelang.language as T
import torch

BM = BN = BK = 128
DTYPE = "float8_e4m3"
THREADS = 256
STAGES = 2

# grid fill below which split-K is the measured winner on this card
STARVED_FRACTION = 0.20


def _n_sms() -> int:
    try:
        from tilelang.carver.arch.driver import get_num_sms
        return int(get_num_sms())
    except Exception:                                            # noqa: BLE001
        return 156


def pick_splitk(m: int, n: int, k: int) -> int:
    """How many K slices to split into, from the measured verdict.

    Split-K exists to fill a starved grid. tiles = ceil(M/BM)*ceil(N/BN); if that is
    already a healthy multiple of the SM count, splitting only adds atomic traffic.
    """
    tiles = -(-m // BM) * (-(-n // BN))
    waves = tiles / _n_sms()
    if waves >= 1.0:
        return 1
    if waves >= STARVED_FRACTION:
        return 2
    return 8


def _plain(m: int, n: int, kk_dim: int):
    @T.prim_func
    def main(A: T.Tensor((m, kk_dim), DTYPE), B: T.Tensor((kk_dim, n), DTYPE),
             C: T.Tensor((m, n), "float32")):
        with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=THREADS) as (bx, by):
            As = T.alloc_shared((BM, BK), DTYPE)
            Bs = T.alloc_shared((BK, BN), DTYPE)
            Cl = T.alloc_fragment((BM, BN), "float32")
            T.clear(Cl)
            for kk in T.Pipelined(T.ceildiv(kk_dim, BK), num_stages=STAGES):
                T.copy(A[by * BM, kk * BK], As)
                T.copy(B[kk * BK, bx * BN], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * BM, bx * BN])
    return main


def _splitk(m: int, n: int, kk_dim: int, sk: int):
    ktiles = T.ceildiv(kk_dim, BK)
    per = T.ceildiv(ktiles, sk)

    @T.prim_func
    def main(A: T.Tensor((m, kk_dim), DTYPE), B: T.Tensor((kk_dim, n), DTYPE),
             C: T.Tensor((m, n), "float32")):
        with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), sk, threads=THREADS) as (bx, by, sz):
            As = T.alloc_shared((BM, BK), DTYPE)
            Bs = T.alloc_shared((BK, BN), DTYPE)
            Cl = T.alloc_fragment((BM, BN), "float32")
            T.clear(Cl)
            k0 = sz * per
            for kk in T.Pipelined(per, num_stages=STAGES):
                kk_abs = k0 + kk
                if kk_abs < ktiles:
                    T.copy(A[by * BM, kk_abs * BK], As)
                    T.copy(B[kk_abs * BK, bx * BN], Bs)
                    T.gemm(As, Bs, Cl)
            T.atomic_add(C[by * BM, bx * BN], Cl)
    return main


def gemm_fp8_splitk(a: torch.Tensor, b: torch.Tensor, splitk: int | None = None) -> torch.Tensor:
    """fp8 e4m3 (M,K) x (K,N) -> fp32 (M,N).

    `splitk=None` picks the count from the measured verdict. C is allocated and zeroed
    here and passed in explicitly, which is the only correct way to drive an atomic
    kernel (see the module docstring).
    """
    m, k = a.shape
    n = b.shape[1]
    sk = splitk if splitk is not None else pick_splitk(m, n, k)
    builder = _plain(m, n, k) if sk == 1 else _splitk(m, n, k, sk)
    kernel = tilelang.compile(builder, target="cuda")
    c = torch.zeros(m, n, device=a.device, dtype=torch.float32)
    kernel(a, b, c)
    return c


def _bench():
    import math
    from tilelang.profiler import do_bench

    for (m, n, k) in [(512, 512, 16384), (1024, 1024, 8192), (4096, 4096, 4096)]:
        a = (torch.randn(m, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        b = (torch.randn(k, n, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        ref = a.float() @ b.float()
        auto = pick_splitk(m, n, k)
        print(f"\n{m}x{n}x{k}  tiles={-(-m//BM)*(-(-n//BN))} waves="
              f"{(-(-m//BM)*(-(-n//BN)))/_n_sms():.2f}  auto SK={auto}")
        for sk in (1, 2, 4, 8):
            builder = _plain(m, n, k) if sk == 1 else _splitk(m, n, k, sk)
            try:
                kernel = tilelang.compile(builder, target="cuda")
                c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
                kernel(a, b, c)
                torch.cuda.synchronize()
                md = (c - ref).abs().max().item()

                def run():
                    if sk > 1:
                        c.zero_()
                    kernel(a, b, c)

                t = do_bench(run, warmup=20, rep=60)
                tf = 2 * m * n * k / (t * 1e-3) / 1e12
                mark = " <-auto" if sk == auto else ""
                print(f"  SK={sk}: {tf:7.1f} TFLOP/s  md={md:.3g}{mark}")
            except Exception as exc:                             # noqa: BLE001
                print(f"  SK={sk}: FAIL {str(exc)[:120]}")
        del a, b, ref


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _bench()
