"""MLA prefill (non-causal flash attention) for sm120 — fp16, BM=64 / BN=128.

Earned in `hw_kb/tilelang/sm120/flash_attention_is_compute_bound_too.md` and
`head_dim_decides_the_stage_budget.md`. Two findings decide the shape of this file:

  * **Prefill is compute-bound, not bandwidth-bound.** NCU on the best config: `dram_pct`
    8.77 %, tensor pipe 56.6 %, warps active only 15.8 %, and `launch__occupancy_limit_
    registers` = 1 block at 240 registers/thread. The kernel is nowhere near memory. So do
    not chase bandwidth here — the levers are tile shape and fp8.
  * **`BM=64, BN=128` is the measured best** (the `S = T.alloc_fragment((BM, BN))` shape in
    the entry). `BM=128` loses (82.2 TF at stages=1 vs 114.8 for BM=64) because of that
    one-block occupancy limit, not because of tile arithmetic.

`stages` depends on head dim, and the entry is explicit about it:

    D=64  BM64 BN128 stages=2  293.5us  eff 117.1 TF  cos 1.00000   <- stages=2 fits
    D=64  BM64 BN128 stages=1  299.3us  eff 114.8 TF  cos 1.00000
    D=128 BM64 BN128 stages=2  --       SMEM cap
    D=128 BM64 BN128 stages=1  460.4us  eff 149.2 TF                 <- what this file uses

So `stages` is derived from `D`, not hardcoded: 2 when the K/V tiles fit (D<=64), 1 at
D=128. Asking for stages=2 at D=128 gives `Failed to set the allowed dynamic shared memory
size` — this is the same 99 KiB ceiling as everywhere else on this arch.

This is the non-causal prefill. For causal use `attention/attn_fwd_causal/sm120/`, which
bounds the k-loop to the diagonal and wins 1.95x.

Measured on this box (tilelang 0.1.15, H=16, SEQ=2048, fp16, SDPA baseline 347 TF).
"""

import math

import tilelang
import tilelang.language as T
import torch

D_DEFAULT = 128
BM, BN = 64, 128
THREADS = 128
LOG2E = 1.4426950408889634
SMEM_STAGE2_MAX_D = 64          # stages=2 fits only when the K/V tiles are small enough


def _stages_for(d: int) -> int:
    return 2 if d <= SMEM_STAGE2_MAX_D else 1


def _mla_prefill(h: int, seq: int, d: int, stages: int):
    scale = (1.0 / math.sqrt(d)) * LOG2E

    @T.prim_func
    def main(Q: T.Tensor((h, seq, d), "float16"),
             K: T.Tensor((h, seq, d), "float16"),
             V: T.Tensor((h, seq, d), "float16"),
             O: T.Tensor((h, seq, d), "float16")):
        with T.Kernel(T.ceildiv(seq, BM), h, threads=THREADS) as (bq, bh):
            Qs = T.alloc_shared((BM, d), "float16")
            Ks = T.alloc_shared((BN, d), "float16")
            Vs = T.alloc_shared((BN, d), "float16")
            Ps = T.alloc_shared((BM, BN), "float16")
            S = T.alloc_fragment((BM, BN), "float32")
            acc = T.alloc_fragment((BM, d), "float32")

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

            for k in T.Pipelined(T.ceildiv(seq, BN), num_stages=stages):
                T.copy(K[bh, k * BN, 0], Ks)
                T.copy(V[bh, k * BN, 0], Vs)
                T.gemm(Qs, Ks, S, transpose_B=True, clear_accum=True)

                T.copy(m_i, m_prev)
                T.copy(l_i, l_prev)

                for i, j in T.Parallel(BM, BN):
                    S[i, j] = S[i, j] * scale
                T.reduce_max(S, rmax, dim=1, clear=True)
                for i in T.Parallel(BM):
                    m_i[i] = T.max(m_prev[i], rmax[i])
                for i, j in T.Parallel(BM, BN):
                    S[i, j] = T.exp2(S[i, j] - m_i[i])
                    Ps[i, j] = S[i, j]
                T.reduce_sum(S, rsum, dim=1)
                for i in T.Parallel(BM):
                    l_i[i] = l_prev[i] * T.exp2(m_prev[i] - m_i[i]) + rsum[i]
                for i, j in T.Parallel(BM, d):
                    acc[i, j] = acc[i, j] * T.exp2(m_prev[i] - m_i[i])
                T.gemm(Ps, Vs, acc)

            for i, j in T.Parallel(BM, d):
                acc[i, j] = acc[i, j] / l_i[i]
            T.copy(acc, O[bh, bq * BM, 0])
    return main


_CACHE: dict = {}


def build(h: int, seq: int, d: int = D_DEFAULT):
    key = (h, seq, d)
    if key not in _CACHE:
        _CACHE[key] = tilelang.compile(_mla_prefill(h, seq, d, _stages_for(d)),
                                       out_idx=[3], target="cuda")
    return _CACHE[key]


def mla_paged_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(H, SEQ, D) fp16 non-causal attention -> (H, SEQ, D). SEQ must be a multiple of BM."""
    h, seq, d = q.shape
    return build(h, seq, d)(q, k, v)


def _bench():
    from tilelang.profiler import do_bench
    import torch.nn.functional as F

    h, seq, d = 16, 2048, D_DEFAULT
    q = torch.randn(h, seq, d, device="cuda", dtype=torch.float16) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn_like(q) * 0.05
    kern = build(h, seq, d)
    out = kern(q, k, v)
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False)
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0).item()
    t = do_bench(lambda: kern(q, k, v), warmup=15, rep=50)
    ts = do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False),
                  warmup=15, rep=50)
    flop = 4 * h * seq * seq * d
    print(f"prefill D{d} BM{BM} BN{BN} st{_stages_for(d)}: {t*1000:7.1f}us "
          f"effTF={flop/(t*1e-3)/1e12:6.1f} cos={cos:.6f} | SDPA {ts*1000:.1f}us "
          f"{flop/(ts*1e-3)/1e12:.1f}TF | {ts/t:.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _bench()
