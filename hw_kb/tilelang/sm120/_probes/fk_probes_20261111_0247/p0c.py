import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
def gemm(M,N,K,BM,BN,BK,stages=2):
    @T.prim_func
    def main(X:T.Tensor((M,K),"float16"),W:T.Tensor((N,K),"float16"),O:T.Tensor((M,N),"float32")):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bn,bm):
            Xs=T.alloc_shared((BM,BK),"float16"); Ws=T.alloc_shared((BN,BK),"float16")
            C=T.alloc_fragment((BM,BN),"float32"); T.clear(C)
            for ko in T.Pipelined(T.ceildiv(K,BK),num_stages=stages):
                T.copy(X[bm*BM,ko*BK],Xs); T.copy(W[bn*BN,ko*BK],Ws)
                T.gemm(Xs,Ws,C,transpose_B=True)
            for i,j in T.Parallel(BM,BN): O[bm*BM+i,bn*BN+j]=C[i,j]
    return main
def lmhead_sparse(M,N,K,BM,BN,BK,NTILE):
    @T.prim_func
    def main(X:T.Tensor((M,K),"float16"),W:T.Tensor((N,K),"float16"),
             IDX:T.Tensor((NTILE,),"int32"),O:T.Tensor((M,NTILE*BN),"float32")):
        with T.Kernel(NTILE,T.ceildiv(M,BM),threads=128) as (nt,bm):
            Xs=T.alloc_shared((BM,BK),"float16"); Ws=T.alloc_shared((BN,BK),"float16")
            C=T.alloc_fragment((BM,BN),"float32"); T.clear(C)
            realbn=IDX[nt]
            for ko in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(X[bm*BM,ko*BK],Xs); T.copy(W[realbn*BN,ko*BK],Ws)
                T.gemm(Xs,Ws,C,transpose_B=True)
            for i,j in T.Parallel(BM,BN): O[bm*BM+i,nt*BN+j]=C[i,j]
    return main
torch.manual_seed(0)
M,N,K=16,50257,2048; BM,BN,BK=16,128,64
X=torch.randn(M,K,device="cuda",dtype=torch.float16)*0.1
W=torch.randn(N,K,device="cuda",dtype=torch.float16)*0.1
FULLT=N//BN+1
kd=tilelang.compile(gemm(M,N,K,BM,BN,BK),out_idx=[2],target="cuda")
kd(X,W); torch.cuda.synchronize()
td=do_bench(lambda: kd(X,W),warmup=15,rep=50)
for nc in (512,1024):
  G=torch.randn(nc,K,device="cuda",dtype=torch.float16)*0.1
  ks=tilelang.compile(gemm(M,nc,K,BM,BN,BK),out_idx=[2],target="cuda")
  ks(X,G); torch.cuda.synchronize()
  ts=do_bench(lambda: ks(X,G),warmup=15,rep=50)
  print(f"[nc={nc}] dense full={td*1000:.1f}us ; screen X@G^T={ts*1000:.1f}us (G={nc*K*2/1e6:.1f}MB, {nc*K*2/(N*K*2)*100:.1f}% of W)")
  for frac in (0.25,0.1,0.05):
    ntile=max(1,int(FULLT*frac))
    idx=torch.randperm(FULLT,device="cuda",dtype=torch.int32)[:ntile].contiguous()
    k=tilelang.compile(lmhead_sparse(M,N,K,BM,BN,BK,ntile),out_idx=[3],target="cuda")
    k(X,W,idx); torch.cuda.synchronize()
    t=do_bench(lambda: k(X,W,idx),warmup=15,rep=50)
    tot=ts+t
    print(f"  frac={frac:.2f}: tile={t*1000:.1f}+screen={ts*1000:.1f}={tot*1000:.1f}us -> {tot/td:.2f}x of dense")