"""Causal FlashAttention forward for sm120 — the 1.95x win, as a kernel.

Earned in `hw_kb/tilelang/sm120/causal_attention_skips_the_masked_half.md`. The win is
one idea: **bound the k-loop to the diagonal** instead of computing the full score matrix
and masking it. The skipped blocks are ~0.51 of the full triangle, so ~1.95x is all
there is; there is no second lever hiding behind it.

Measured on this box (tilelang 0.1.15, fp16, BM=64 BN=128, threads=128):

    SEQ 2048  tilelang 152.3us  eff 112.8 TF  cos=1.00000 | SDPA 159.1us | 1.04x
    SEQ 4096  tilelang 463.9us  eff 148.1 TF  cos=1.00000 | SDPA 436.3us | 0.94x
    SEQ 8192  tilelang 1925.2us eff 142.8 TF  cos=1.00000 | SDPA 1958.9us| 1.02x

Two details that are easy to get wrong and are therefore spelled out in the code:

  * the k-loop bound is `ceildiv((bq+1)*BM, BN)`, not `SEQ/BN`;
  * the diagonal block is **partly masked and must still be computed** — only the blocks
    strictly above the diagonal are skipped. Mask AFTER scaling, with -inf.
"""

import math

import tilelang
import tilelang.language as T
import torch

BM, BN = 64, 128
THREADS = 128
STAGES = 1     # stages=2 overflows dynamic smem: Ks+Vs are 32 KiB each at BN=128
LOG2E = 1.4426950408889634


def _fa_causal(h: int, seq: int, head_dim: int):
    scale = (1.0 / math.sqrt(head_dim)) * LOG2E
    nkb_full = T.ceildiv(seq, BN)

    @T.prim_func
    def main(Q: T.Tensor((h, seq, head_dim), "float16"),
             K: T.Tensor((h, seq, head_dim), "float16"),
             V: T.Tensor((h, seq, head_dim), "float16"),
             O: T.Tensor((h, seq, head_dim), "float16")):
        with T.Kernel(T.ceildiv(seq, BM), h, threads=THREADS) as (bq, bh):
            Qs = T.alloc_shared((BM, head_dim), "float16")
            Ks = T.alloc_shared((BN, head_dim), "float16")
            Vs = T.alloc_shared((BN, head_dim), "float16")
            S = T.alloc_fragment((BM, BN), "float32")
            Ps = T.alloc_shared((BM, BN), "float16")
            acc = T.alloc_fragment((BM, head_dim), "float32")

            m_i = T.alloc_fragment((BM,), "float32")
            l_i = T.alloc_fragment((BM,), "float32")
            m_prev = T.alloc_fragment((BM,), "float32")
            l_prev = T.alloc_fragment((BM,), "float32")
            rmax = T.alloc_fragment((BM,), "float32")
            rsum = T.alloc_fragment((BM,), "float32")

            T.fill(acc, 0.0)
            T.fill(l_i, 0.0)
            T.fill(m_i, -T.infinity("float32"))
            T.copy(Q[bh, bq * BM, 0], Qs)

            # >>> the whole win: stop at the diagonal block instead of nkb_full <<<
            nkb = T.ceildiv((bq + 1) * BM, BN)
            for k in T.Pipelined(nkb, num_stages=STAGES):
                T.copy(K[bh, k * BN, 0], Ks)
                T.copy(V[bh, k * BN, 0], Vs)
                T.gemm(Qs, Ks, S, transpose_B=True, clear_accum=True)

                T.copy(m_i, m_prev)
                T.copy(l_i, l_prev)

                # scale first, then mask: the diagonal block is partly above the line
                for i, j in T.Parallel(BM, BN):
                    S[i, j] = S[i, j] * scale
                for i, j in T.Parallel(BM, BN):
                    if (k * BN + j) > (bq * BM + i):
                        S[i, j] = -T.infinity("float32")

                T.reduce_max(S, rmax, dim=1, clear=True)
                for i in T.Parallel(BM):
                    m_i[i] = T.max(m_prev[i], rmax[i])
                for i, j in T.Parallel(BM, BN):
                    S[i, j] = T.exp2(S[i, j] - m_i[i])
                    Ps[i, j] = S[i, j]
                T.reduce_sum(S, rsum, dim=1)
                for i in T.Parallel(BM):
                    l_i[i] = l_prev[i] * T.exp2(m_prev[i] - m_i[i]) + rsum[i]
                for i, j in T.Parallel(BM, head_dim):
                    acc[i, j] = acc[i, j] * T.exp2(m_prev[i] - m_i[i])
                T.gemm(Ps, Vs, acc)

            for i, j in T.Parallel(BM, head_dim):
                acc[i, j] = acc[i, j] / l_i[i]
            T.copy(acc, O[bh, bq * BM, 0])
    return main


_CACHE: dict = {}


def build(h: int, seq: int, d: int):
    """Compile once per shape. tilelang.compile is seconds; it must not sit in a timing loop."""
    key = (h, seq, d)
    if key not in _CACHE:
        _CACHE[key] = tilelang.compile(_fa_causal(h, seq, d), out_idx=[3], target="cuda")
    return _CACHE[key]


def attn_fwd_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(H, SEQ, D) fp16 causal attention -> (H, SEQ, D) fp16. SEQ must be a multiple of BM."""
    h, seq, d = q.shape
    return build(h, seq, d)(q, k, v)


def _bench():
    from tilelang.profiler import do_bench
    import torch.nn.functional as F

    for seq in (2048, 4096, 8192):
        h, d = 16, 128
        q = torch.randn(h, seq, d, device="cuda", dtype=torch.float16) * 0.05
        k = torch.randn_like(q) * 0.05
        v = torch.randn_like(q) * 0.05
        kern = build(h, seq, d)
        out = kern(q, k, v)
        torch.cuda.synchronize()
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                             is_causal=True)
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.flatten(), dim=0).item()
        t = do_bench(lambda: kern(q, k, v), warmup=15, rep=60)
        flop = 4 * h * seq * seq * d * 0.5          # half the square: causal
        print(f"SEQ{seq} causal: {t*1000:7.1f}us effTF={flop/(t*1e-3)/1e12:6.1f} cos={cos:.5f}")
        del q, k, v, out, ref


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _bench()
