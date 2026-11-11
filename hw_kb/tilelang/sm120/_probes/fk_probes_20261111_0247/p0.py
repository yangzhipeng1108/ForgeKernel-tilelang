import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
# P0: LM head block-sparse. Only a subset of vocab tiles needed (top-k candidates).
# Kernel launches on a compacted list of output-tile indices.
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
def lmhead_dense(M,N,K,BM,BN,BK):
    @T.prim_func
    def main(X:T.Tensor((M,K),"float16"),W:T.Tensor((N,K),"float16"),O:T.Tensor((M,N),"float32")):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bn,bm):
            Xs=T.alloc_shared((BM,BK),"float16"); Ws=T.alloc_shared((BN,BK),"float16")
            C=T.alloc_fragment((BM,BN),"float32"); T.clear(C)
            for ko in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(X[bm*BM,ko*BK],Xs); T.copy(W[bn*BN,ko*BK],Ws)
                T.gemm(Xs,Ws,C,transpose_B=True)
            for i,j in T.Parallel(BM,BN): O[bm*BM+i,bn*BN+j]=C[i,j]
    return main
torch.manual_seed(0)
M,N,K=16,50257,2048; BM,BN,BK=16,128,64
X=torch.randn(M,K,device="cuda",dtype=torch.float16)*0.1
W=torch.randn(N,K,device="cuda",dtype=torch.float16)*0.1
FULLT=N//BN+1
FLOP_full=2*M*N*K
# dense baseline
kd=tilelang.compile(lmhead_dense(M,N,K,BM,BN,BK),out_idx=[2],target="cuda")
kd(X,W); torch.cuda.synchronize()
td=do_bench(lambda: kd(X,W),warmup=15,rep=50)
print(f"dense: {td*1000:.1f}us {FLOP_full/(td*1e-3)/1e12:.1f}TF {N*K*2/(td*1e-3)/1e9:.0f}GB/s")
# block-sparse: skip fraction s of vocab tiles
for frac in (1.0,0.5,0.25,0.1):
    ntile=max(1,int(FULLT*frac))
    idx=torch.randperm(FULLT,device="cuda",dtype=torch.int32)[:ntile].contiguous()
    # selection overhead: pretend we did a cheap top-k screening = one GEMM against a coarse proj
    # measure the tile-skip kernel alone first
    try:
        k=tilelang.compile(lmhead_sparse(M,N,K,BM,BN,BK,ntile),out_idx=[3],target="cuda")
        o=k(X,W,idx); torch.cuda.synchronize()
        t=do_bench(lambda: k(X,W,idx),warmup=15,rep=50)
        FLOP=2*M*ntile*BN*K
        print(f"sparse frac={frac:.2f} ntile={ntile}/{FULLT}: {t*1000:.1f}us {t/td:.2f}x of dense  W-read={ntile*BN*K*2/1e6:.1f}MB BW={ntile*BN*K*2/(t*1e-3)/1e9:.0f}GB/s")
    except Exception as e: print(f"frac{frac}: FAIL {str(e)[:100]}")