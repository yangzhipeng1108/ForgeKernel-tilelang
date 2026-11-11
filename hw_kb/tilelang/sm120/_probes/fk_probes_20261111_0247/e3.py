import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
def gemm(M,N,K,BM,BN,BK,xdt,wdt,NTILE=None):
    if NTILE is None:
        @T.prim_func
        def main(X:T.Tensor((M,K),xdt),W:T.Tensor((N,K),wdt),O:T.Tensor((M,N),"float32")):
            with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bn,bm):
                Xs=T.alloc_shared((BM,BK),xdt); Ws=T.alloc_shared((BN,BK),wdt)
                C=T.alloc_fragment((BM,BN),"float32"); T.clear(C)
                for ko in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                    T.copy(X[bm*BM,ko*BK],Xs); T.copy(W[bn*BN,ko*BK],Ws)
                    T.gemm(Xs,Ws,C,transpose_B=True)
                for i,j in T.Parallel(BM,BN): O[bm*BM+i,bn*BN+j]=C[i,j]
        return main
    else:
        @T.prim_func
        def main(X:T.Tensor((M,K),xdt),W:T.Tensor((N,K),wdt),IDX:T.Tensor((NTILE,),"int32"),O:T.Tensor((M,NTILE*BN),"float32")):
            with T.Kernel(NTILE,T.ceildiv(M,BM),threads=128) as (nt,bm):
                Xs=T.alloc_shared((BM,BK),xdt); Ws=T.alloc_shared((BN,BK),wdt)
                C=T.alloc_fragment((BM,BN),"float32"); T.clear(C)
                rb=IDX[nt]
                for ko in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                    T.copy(X[bm*BM,ko*BK],Xs); T.copy(W[rb*BN,ko*BK],Ws)
                    T.gemm(Xs,Ws,C,transpose_B=True)
                for i,j in T.Parallel(BM,BN): O[bm*BM+i,nt*BN+j]=C[i,j]
        return main
torch.manual_seed(0)
M,N,K=16,50257,2048; BM,BN,BK=16,128,64; FULLT=N//BN+1
X=torch.randn(M,K,device="cuda",dtype=torch.float16)*0.1
W=torch.randn(N,K,device="cuda",dtype=torch.float16)*0.1
X8=X.to(torch.float8_e4m3fn); W8=W.to(torch.float8_e4m3fn)
kd=tilelang.compile(gemm(M,N,K,BM,BN,BK,"float16","float16"),out_idx=[2],target="cuda")
kd(X,W); torch.cuda.synchronize(); td=do_bench(lambda: kd(X,W),warmup=15,rep=50)
# fp8 dense
k8=tilelang.compile(gemm(M,N,K,BM,BN,BK,"float8_e4m3","float8_e4m3"),out_idx=[2],target="cuda")
k8(X8,W8); torch.cuda.synchronize(); t8=do_bench(lambda: k8(X8,W8),warmup=15,rep=50)
print(f"dense f16={td*1000:.1f}us  dense f8={t8*1000:.1f}us ({td/t8:.2f}x)")
# fp8 + sparse 10%
frac=0.1; ntile=int(FULLT*frac)
idx=torch.randperm(FULLT,device="cuda",dtype=torch.int32)[:ntile].contiguous()
ks=tilelang.compile(gemm(M,N,K,BM,BN,BK,"float8_e4m3","float8_e4m3",ntile),out_idx=[3],target="cuda")
ks(X8,W8,idx); torch.cuda.synchronize(); tsp=do_bench(lambda: ks(X8,W8,idx),warmup=15,rep=50)
# screen cost (fp8 coarse) nc=512
nc=512; G=torch.randn(nc,K,device="cuda",dtype=torch.float16)*0.1; G8=G.to(torch.float8_e4m3fn)
ksc=tilelang.compile(gemm(M,nc,K,BM,BN,BK,"float8_e4m3","float8_e4m3"),out_idx=[2],target="cuda")
ksc(X8,G8); torch.cuda.synchronize(); tsc=do_bench(lambda: ksc(X8,G8),warmup=15,rep=50)
print(f"fp8+sparse10%: tile={tsp*1000:.1f}us  screen={tsc*1000:.1f}us  total={(tsp+tsc)*1000:.1f}us -> {(tsp+tsc)/td:.2f}x of f16 dense")
print(f"  compare: fp8 dense alone={t8*1000:.1f}us; fp16+sparse10% was 70.5us")