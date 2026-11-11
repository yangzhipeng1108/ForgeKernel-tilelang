import tilelang, tilelang.language as T, torch as th
from tilelang.profiler import do_bench
BM=BN=128
def build(M,N,K,BK,d,ws,pd=0,stages=2,threads=256):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=threads) as (bx,by):
            if ws and pd:
                T.annotate_producer_reg_dealloc(pd)
                T.annotate_consumer_reg_alloc(240)
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=stages):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    cfg={"tl.enable_auto_warp_specialization":"role_based"} if ws else None
    return tilelang.compile(main,out_idx=[2],target="cuda",pass_configs=cfg)
M=N=K=4096
A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
for pd in (0,24):
  for ws in (False,True):
    try:
      k=build(M,N,K,128,T.float8_e4m3,ws,pd=pd)
      t=do_bench(lambda: k(A,B),warmup=25,rep=100)
      print(f"fp8 ws={int(ws)} regdealloc={pd}: {2*M*N*K/(t*1e-3)/1e12:6.1f} TFLOP/s")
    except Exception as e: print(f"ws={ws} pd={pd} FAIL {str(e)[:200]}")