import tilelang, tilelang.language as T, torch as th
BM=BN=128; BK=128; d=T.float8_e4m3; M=N=K=4096
@T.prim_func
def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
    with T.ClusterKernel(T.ceildiv(N,BN),T.ceildiv(M,BM),cluster_dims=2,threads=256) as (bx,by):
        As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
        Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
            T.copy_cluster(A[by*BM,k*BK],As,cluster_mask=3)
            T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
        T.copy(Cl,C[by*BM,bx*BN])
src=tilelang.compile(main,out_idx=[2],target="cuda").get_kernel_source()
open("/tmp/mc.cu","w").write(src)