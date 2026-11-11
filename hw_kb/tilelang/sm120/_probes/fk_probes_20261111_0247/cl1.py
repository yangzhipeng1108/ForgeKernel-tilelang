import tilelang, tilelang.language as T, torch as th
from tilelang.profiler import do_bench
BM=BN=128; BK=128; d=T.float8_e4m3; M=N=K=4096
def plain():
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=256) as (bx,by):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return main
def cluster(cd):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        with T.ClusterKernel(T.ceildiv(N,BN),T.ceildiv(M,BM),cluster_dims=cd,threads=256) as (bx,by):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return main
A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
ref=(A.float()@B.float())
def bench(builder,tag):
    try:
        k=tilelang.compile(builder,out_idx=[2],target="cuda")
        out=k(A,B); md=(out.float()-ref).abs().max().item()
        t=do_bench(lambda: k(A,B),warmup=20,rep=60)
        print(f"{tag}: {2*M*N*K/(t*1e-3)/1e12:6.1f} TFLOP/s md={md:.3g}")
    except Exception as e: print(f"{tag}: FAIL {type(e).__name__}: {str(e)[:220]}")
bench(plain(),"plain")
for cd in (2,(2,1,1),(4,1,1)):
    bench(cluster(cd),f"cluster={cd}")