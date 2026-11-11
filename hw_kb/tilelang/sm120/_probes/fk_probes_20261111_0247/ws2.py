import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
import torch as th
BM=BN=128
def build(M,N,K,BK,d,fp8,ws,stages=2,threads=256):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=threads) as (bx,by):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=stages):
                T.copy(A[by*BM,k*BK],As)
                T.copy(B[k*BK,bx*BN],Bs)
                T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    cfg={"tl.enable_auto_warp_specialization":"role_based"} if ws else None
    return tilelang.compile(main,out_idx=[2],target="cuda",pass_configs=cfg)
M=N=K=4096
for fp8 in (False,True):
  for ws in (False,True):
    dt=th.float8_e4m3fn if fp8 else th.float16
    d=T.float8_e4m3 if fp8 else T.float16
    BK=128 if fp8 else 32
    A=th.randn(M,K,device="cuda",dtype=th.float16).to(dt)
    B=th.randn(K,N,device="cuda",dtype=th.float16).to(dt)
    try:
      k=build(M,N,K,BK,d,fp8,ws)
      t=do_bench(lambda: k(A,B),warmup=25,rep=100)
      tf=2*M*N*K/(t*1e-3)/1e12
      C=k(A,B)
      ref=(A.float()@B.float())
      md=(C.float()-ref).abs().max().item()
      cs=th.nn.functional.cosine_similarity(C.float().flatten(),ref.flatten(),dim=0).item()
      print(f"fp8={fp8} BK={BK} ws={ws}: {t:.4f} ms {tf:.1f} TFLOP/s maxdiff={md:.3g} cos={cs:.6f}")
    except Exception as e:
      print(f"fp8={fp8} BK={BK} ws={ws}: FAIL {type(e).__name__}: {str(e)[:250]}")