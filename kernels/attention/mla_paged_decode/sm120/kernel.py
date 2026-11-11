"""MLA absorbed decode for sm120 — 3.8x over unshared MHA, as a kernel.

Earned in `hw_kb/tilelang/sm120/mla_absorbed_decode_beats_mha.md`. The mechanism is to
cache a **single latent** `c_KV (KVLEN, Dc)` once for all heads and project at the ends,
instead of caching K and V per head. The measured consequence is a 256x smaller cache and
a bandwidth collapse from 880 GB/s to 21 GB/s.

Measured on this box (tilelang 0.1.15, fp16, HQ=128, QLEN=16, KVLEN=8192, Dc=DV=128):

    BM16 BN64  st2:  123.4us  70.2 TF  cos=1.00000  BW=21 GB/s   <- best
    BM16 BN128 st1:  138.8us  62.4 TF  cos=1.00000  BW=19 GB/s
    BM32 BN64  st2:  145.9us  59.3 TF  cos=1.00000  BW=18 GB/s

Two traps the sweep found, both encoded below:

  * **BM must be 16 here, and stages=2** — the Dc latent tile fits a second stage, which is
    the opposite of MHA decode's stages=1. BM32/stages=2 wants ~115 KiB and fails to set
    the dynamic smem;
  * the real `Dc=512` does **not** fit one kernel. `Dc=128` is what this kernel takes; a
    production Dc=512 needs the output projection split into a second pass.
"""

import math

import tilelang
import tilelang.language as T
import torch

DV = 128
BN = 64
BM = 16                 # q rows per block; 32 overflows the dynamic-smem cap (see docstring)
THREADS = 128
STAGES = 2
LOG2E = 1.4426950408889634


def _mla(hq: int, qlen: int, kvlen: int, dc: int, dv: int, bn: int, stages: int):
    scale = (1.0 / math.sqrt(dc)) * LOG2E
    bm = qlen

    @T.prim_func
    def main(Qa: T.Tensor((hq, qlen, dc), "float16"),
             C: T.Tensor((kvlen, dc), "float16"),
             Wuv: T.Tensor((dc, dv), "float16"),
             O: T.Tensor((hq, qlen, dv), "float16")):
        with T.Kernel(hq, threads=THREADS) as bh:
            Qs = T.alloc_shared((bm, dc), "float16")
            Cs = T.alloc_shared((bn, dc), "float16")
            Ws = T.alloc_shared((dc, dv), "float16")
            Ps = T.alloc_shared((bm, bn), "float16")
            Ls = T.alloc_shared((bm, dc), "float16")

            S = T.alloc_fragment((bm, bn), "float32")
            acc = T.alloc_fragment((bm, dc), "float32")
            out = T.alloc_fragment((bm, dv), "float32")

            m_i = T.alloc_fragment((bm,), "float32")
            l_i = T.alloc_fragment((bm,), "float32")
            m_prev = T.alloc_fragment((bm,), "float32")
            l_prev = T.alloc_fragment((bm,), "float32")
            rmax = T.alloc_fragment((bm,), "float32")
            rsum = T.alloc_fragment((bm,), "float32")

            T.copy(Qa[bh, 0, 0], Qs)
            T.copy(Wuv, Ws)
            T.fill(acc, 0.0)
            T.fill(l_i, 0.0)
            T.fill(m_i, -T.infinity("float32"))

            for kblk in T.Pipelined(T.ceildiv(kvlen, bn), num_stages=stages):
                T.copy(C[kblk * bn, 0], Cs)
                T.gemm(Qs, Cs, S, transpose_B=True, clear_accum=True)

                T.copy(m_i, m_prev)
                T.copy(l_i, l_prev)

                for i, j in T.Parallel(bm, bn):
                    S[i, j] = S[i, j] * scale
                T.reduce_max(S, rmax, dim=1, clear=True)
                for i in T.Parallel(bm):
                    m_i[i] = T.max(m_prev[i], rmax[i])
                for i, j in T.Parallel(bm, bn):
                    S[i, j] = T.exp2(S[i, j] - m_i[i])
                    Ps[i, j] = S[i, j]
                T.reduce_sum(S, rsum, dim=1)
                for i in T.Parallel(bm):
                    l_i[i] = l_prev[i] * T.exp2(m_prev[i] - m_i[i]) + rsum[i]
                for i, j in T.Parallel(bm, dc):
                    acc[i, j] = acc[i, j] * T.exp2(m_prev[i] - m_i[i])
                T.gemm(Ps, Cs, acc)

            for i, j in T.Parallel(bm, dc):
                Ls[i, j] = acc[i, j] / l_i[i]
            T.clear(out)
            T.gemm(Ls, Ws, out)
            T.copy(out, O[bh, 0, 0])
    return main


_CACHE: dict = {}


def build(hq: int, qlen: int, kvlen: int, dc: int, dv: int):
    """Compile once per shape. tilelang.compile is seconds; it must not sit in a timing loop."""
    key = (hq, qlen, kvlen, dc, dv)
    if key not in _CACHE:
        _CACHE[key] = tilelang.compile(_mla(hq, qlen, kvlen, dc, dv, BN, STAGES),
                                       out_idx=[3], target="cuda")
    return _CACHE[key]


def mla_absorbed_decode(qa: torch.Tensor, c: torch.Tensor, wuv: torch.Tensor) -> torch.Tensor:
    """(HQ, QLEN, Dc) x latent (KVLEN, Dc) x (Dc, DV) -> (HQ, QLEN, DV) fp16.

    QLEN must be 16 for the measured-best smem footprint.
    """
    hq, qlen, dc = qa.shape
    kvlen = c.shape[0]
    dv = wuv.shape[1]
    return build(hq, qlen, kvlen, dc, dv)(qa, c, wuv)


def _bench():
    from tilelang.profiler import do_bench

    hq, qlen, kvlen, dc = 128, 16, 8192, 128
    qa = torch.randn(hq, qlen, dc, device="cuda", dtype=torch.float16) * 0.05
    c = torch.randn(kvlen, dc, device="cuda", dtype=torch.float16) * 0.05
    wuv = torch.randn(dc, DV, device="cuda", dtype=torch.float16) * 0.05

    s = torch.einsum("hqd,kd->hqk", qa.float(), c.float()) / math.sqrt(dc)
    p = torch.softmax(s, dim=-1)
    ref = torch.einsum("hqk,kd->hqd", p, c.float()) @ wuv.float()

    kern = build(hq, qlen, kvlen, dc, DV)
    out = kern(qa, c, wuv)
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0).item()
    t = do_bench(lambda: kern(qa, c, wuv), warmup=15, rep=50)
    cache_bytes = kvlen * dc * 2
    flop = 4 * hq * qlen * kvlen * dc
    print(f"MLA-absorbed BM{BM} BN{BN} st{STAGES}: {t*1000:7.1f}us "
          f"{flop/(t*1e-3)/1e12:6.1f}TF cos={cos:.5f} "
          f"cache={cache_bytes/1e6:.1f}MB BW={cache_bytes/(t*1e-3)/1e9:.0f}GB/s")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _bench()
