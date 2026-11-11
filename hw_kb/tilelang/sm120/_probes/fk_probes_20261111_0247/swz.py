import tilelang, tilelang.language as T, torch as th
from tilelang.profiler import do_bench
BM=BN=128
def build(M,N,K,BK,d,panel=None,order="row",stages=2,threads=256):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=threads) as (bx,by):
            if panel: T.use_swizzle(panel_size=panel,order=order)
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=stages):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return tilelang.compile(main,out_idx=[2],target="cuda")
M=N=K=4096
for fp8 in (False,True):
  dt=th.float8_e4m3fn if fp8 else th.float16
  d=T.float8_e4m3 if fp8 else T.float16
  BK=128 if fp8 else 32
  A=th.randn(M,K,device="cuda",dtype=th.float16).to(dt); B=th.randn(K,N,device="cuda",dtype=th.float16).to(dt)
  for panel,order in [(None,"row"),(1,"row"),(4,"row"),(8,"row"),(16,"row"),(4,"column"),(8,"column"),(4,"mlx")]:
    try:
      k=build(M,N,K,BK,d,panel,order)
      t=do_bench(lambda: k(A,B),warmup=25,rep=100)
      print(f"fp8={int(fp8)} swizzle={panel},{order}: {2*M*N*K/(t*1e-3)/1e12:6.1f} TFLOP/s")
    except Exception as e: print(f"fp8={int(fp8)} swizzle={panel},{order}: FAIL {str(e)[:150]}")