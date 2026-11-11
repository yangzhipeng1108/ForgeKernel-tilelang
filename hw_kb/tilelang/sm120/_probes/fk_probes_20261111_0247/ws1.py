import tilelang, tilelang.language as T, torch, time, os
from tilelang.profiler import do_bench
TILE=128
def gemm(M,N,K,dt,fp8=False,ws=None,threads=256,stages=2):
    d=tilelang.float8_e4m3 if fp8 else dt
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),dt)):
        with T.Kernel(T.ceildiv(N,TILE),T.ceildiv(M,TILE),threads=threads) as (bx,by):
            As=T.alloc_shared((TILE,TILE),d); Bs=T.alloc_shared((TILE,TILE),d)
            Cl=T.alloc_fragment((TILE,TILE),"float32")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,TILE),num_stages=stages):
                T.copy(A[by*TILE,bx*TILE*0+k*TILE],As)
                T.copy(B[bx*TILE*0+k*TILE,by*TILE],Bs)
                T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*TILE,bx*TILE])
    cfg={"tl.enable_auto_warp_specialization":"role_based"} if ws else None
    return tilelang.compile(main,out_idx=[2],target="cuda",pass_configs=cfg)
M=N=K=4096
for fp8 in (False,True):
  dt=torch.float16
  A=torch.randn(M,K,device="cuda",dtype=dt); B=torch.randn(K,N,device="cuda",dtype=dt)
  Cd=(A@B)
  for ws in (False,True):
    try:
      k=gemm(M,N,K,dt,fp8=fp8,ws=ws)
      Af=A.to(tilelang.float8_e4m3) if fp8 else A
      Bf=B.to(tilelang.float8_e4m3) if fp8 else B
      C=k(Af,Bf)
      import torch as th
      ref=(Af.float()@Bf.float())
      t=do_bench(lambda: k(Af,Bf),warmup=25,rep=100)
      tf=2*M*N*K/(t*1e-3)/1e12
      md=(C.float()-ref).abs().max().item()
      print(f"fp8={fp8} ws={ws} {t:.4f} ms {tf:.1f} TFLOP/s maxdiff={md:.4g}")
    except Exception as e:
      print(f"fp8={fp8} ws={ws} FAIL {type(e).__name__}: {str(e)[:300]}")