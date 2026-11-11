import tilelang, tilelang.language as T, torch as th
from tilelang.profiler import do_bench
BM=BN=128; BK=128; d=T.float8_e4m3

def plain(M,N,K):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float32)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=256) as (bx,by):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return main

def splitk(M,N,K,SK):
    kt=T.ceildiv(K,BK); per=T.ceildiv(kt,SK)
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float32)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),SK,threads=256) as (bx,by,sz):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            k0=sz*per
            for kk in T.Pipelined(per,num_stages=2):
                k=k0+kk
                if k<kt:
                    T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.atomic_add(C[by*BM,bx*BN],Cl)
    return main

def bench(M,N,K,builder,tag):
    A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
    ref=(A.float()@B.float())
    try:
        k=tilelang.compile(builder,out_idx=[2],target="cuda")
        out=k(A,B)     # fresh tensor returned; C zeroed inside
        md=(out-ref).abs().max().item()
        cs=th.nn.functional.cosine_similarity(out.flatten(),ref.flatten(),dim=0).item()
        t=do_bench(lambda: k(A,B),warmup=20,rep=60)
        print(f"{tag}: {2*M*N*K/(t*1e-3)/1e12:6.1f} TFLOP/s md={md:.3g} cos={cs:.6f}")
    except Exception as e: print(f"{tag}: FAIL {str(e)[:180]}")

for (M,N,K) in [(512,512,16384),(1024,1024,8192),(4096,4096,4096)]:
    bench(M,N,K,plain(M,N,K),f"{M}x{N}x{K} plain")
    for SK in (2,4,8,16):
        bench(M,N,K,splitk(M,N,K,SK),f"{M}x{N}x{K} splitK={SK}")