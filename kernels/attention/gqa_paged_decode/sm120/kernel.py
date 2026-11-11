"""GQA decode for sm120 — the one bandwidth-bound case, as a kernel.

Earned in `hw_kb/tilelang/sm120/decode_is_the_one_bandwidth_bound_case.md`. Two facts from
that entry shape this file:

  * decode with a small q (q=16) is **bandwidth-bound**, not compute-bound: it runs at
    dram ~90 %, and **none of the scheduling axes help** (swizzle / split-KV are ties)
    because the grid is tiny and each block reads a disjoint KV slice;
  * **fp8 KV is slower**, not faster: `T.gemm` forces an fp8->fp16 upcast in SMEM. So this
    kernel keeps fp16 KV on purpose.

Measured on this box (tilelang 0.1.15, fp16, QLEN=16, D=128, HKV=8, GQA group 4, BN=64):

    GQA decode: 110.8us  19.4 TF  cos=1.00000  BW=303 GB/s  cache=34 MB

BN=128 exceeds the default dynamic-smem limit on this build (it asks for 107520 B); BN=64
fits. That is why the default here is 64 and not the larger tile the prefill kernels use.
"""

import math

import tilelang
import tilelang.language as T
import torch

D = 128
BN = 64                 # 128 overflows the dynamic-smem cap on this build (see docstring)
THREADS = 128
LOG2E = 1.4426950408889634


def _gqa_decode(hq: int, hkv: int, qlen: int, kvlen: int, bn: int):
    group = hq // hkv
    scale = (1.0 / math.sqrt(D)) * LOG2E

    @T.prim_func
    def main(Q: T.Tensor((hq, qlen, D), "float16"),
             K: T.Tensor((hkv, kvlen, D), "float16"),
             V: T.Tensor((hkv, kvlen, D), "float16"),
             O: T.Tensor((hq, qlen, D), "float16")):
        with T.Kernel(hq, threads=THREADS) as bh:
            kvh = bh // group
            Qs = T.alloc_shared((qlen, D), "float16")
            Ks = T.alloc_shared((bn, D), "float16")
            Vs = T.alloc_shared((bn, D), "float16")
            Ps = T.alloc_shared((qlen, bn), "float16")
            S = T.alloc_fragment((qlen, bn), "float32")
            acc = T.alloc_fragment((qlen, D), "float32")

            m_i = T.alloc_fragment((qlen,), "float32")
            l_i = T.alloc_fragment((qlen,), "float32")
            m_prev = T.alloc_fragment((qlen,), "float32")
            l_prev = T.alloc_fragment((qlen,), "float32")
            rmax = T.alloc_fragment((qlen,), "float32")
            rsum = T.alloc_fragment((qlen,), "float32")

            T.fill(acc, 0.0)
            T.fill(l_i, 0.0)
            T.fill(m_i, -T.infinity("float32"))
            T.copy(Q[bh, 0, 0], Qs)

            for kblk in T.Pipelined(T.ceildiv(kvlen, bn), num_stages=1):
                T.copy(K[kvh, kblk * bn, 0], Ks)
                T.copy(V[kvh, kblk * bn, 0], Vs)
                T.gemm(Qs, Ks, S, transpose_B=True, clear_accum=True)

                T.copy(m_i, m_prev)
                T.copy(l_i, l_prev)

                for i, j in T.Parallel(qlen, bn):
                    S[i, j] = S[i, j] * scale
                T.reduce_max(S, rmax, dim=1, clear=True)
                for i in T.Parallel(qlen):
                    m_i[i] = T.max(m_prev[i], rmax[i])
                for i, j in T.Parallel(qlen, bn):
                    S[i, j] = T.exp2(S[i, j] - m_i[i])
                    Ps[i, j] = S[i, j]
                T.reduce_sum(S, rsum, dim=1)
                for i in T.Parallel(qlen):
                    l_i[i] = l_prev[i] * T.exp2(m_prev[i] - m_i[i]) + rsum[i]
                for i, j in T.Parallel(qlen, D):
                    acc[i, j] = acc[i, j] * T.exp2(m_prev[i] - m_i[i])
                T.gemm(Ps, Vs, acc)

            for i, j in T.Parallel(qlen, D):
                acc[i, j] = acc[i, j] / l_i[i]
            T.copy(acc, O[bh, 0, 0])
    return main


_CACHE: dict = {}


def build(hq: int, hkv: int, qlen: int, kvlen: int, bn: int = BN):
    """Compile once per shape. tilelang.compile is seconds; it must not sit in a timing loop."""
    key = (hq, hkv, qlen, kvlen, bn)
    if key not in _CACHE:
        _CACHE[key] = tilelang.compile(_gqa_decode(hq, hkv, qlen, kvlen, bn),
                                       out_idx=[3], target="cuda")
    return _CACHE[key]


def gqa_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               bn: int = BN) -> torch.Tensor:
    """(HQ, QLEN, D) q against (HKV, KVLEN, D) k/v, GQA-grouped -> (HQ, QLEN, D) fp16."""
    hq, qlen, _ = q.shape
    hkv, kvlen, _ = k.shape
    return build(hq, hkv, qlen, kvlen, bn)(q, k, v)


def _bench():
    from tilelang.profiler import do_bench
    import torch.nn.functional as F

    hq, hkv, qlen, kvlen = 32, 8, 16, 8192
    group = hq // hkv
    q = torch.randn(hq, qlen, D, device="cuda", dtype=torch.float16) * 0.05
    k = torch.randn(hkv, kvlen, D, device="cuda", dtype=torch.float16) * 0.05
    v = torch.randn_like(k) * 0.05

    kr = k.repeat_interleave(group, 0)
    vr = v.repeat_interleave(group, 0)
    ref = F.scaled_dot_product_attention(q.float(), kr.float(), vr.float())

    kern = build(hq, hkv, qlen, kvlen, BN)
    out = kern(q, k, v)
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0).item()
    t = do_bench(lambda: kern(q, k, v), warmup=15, rep=50)
    flop = 4 * hq * qlen * kvlen * D
    kv_bytes = hkv * kvlen * D * 2 * 2
    print(f"GQA decode BN{BN}: {t*1000:7.1f}us {flop/(t*1e-3)/1e12:6.1f}TF "
          f"cos={cos:.5f} BW={kv_bytes/(t*1e-3)/1e9:.0f}GB/s cache={kv_bytes/1e6:.0f}MB")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _bench()
