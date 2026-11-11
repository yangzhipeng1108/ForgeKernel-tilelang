import tilelang, tilelang.language as T, torch as th, os, glob
BM=BN=128; BK=128; d=T.float8_e4m3; M=N=K=4096
@T.prim_func
def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
    with T.ClusterKernel(T.ceildiv(N,BN),T.ceildiv(M,BM),cluster_dims=1,threads=256) as (bx,by):
        As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
        Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
        res=T.alloc_shared((16,),"uint32"); bar=T.alloc_barrier(1)
        T.clc_try_cancel(res,bar)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
            T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
        T.copy(Cl,C[by*BM,bx*BN])
k=tilelang.compile(main,out_idx=[2],target="cuda")
src=k.get_kernel_source()
open("/tmp/clc_k.cu","w").write(src)
print("wrote cu", len(src))
print("has clusterlaunchcontrol in CUDA src:", "clusterlaunchcontrol" in src)