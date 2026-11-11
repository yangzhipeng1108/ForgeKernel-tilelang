import tilelang, tilelang.language as T, torch as th
BM=BN=128; BK=128; d=T.float8_e4m3; M=N=K=4096
def build(mc):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        with T.ClusterKernel(T.ceildiv(N,BN),T.ceildiv(M,BM),cluster_dims=2,threads=256) as (bx,by):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                if mc: T.copy_cluster(A[by*BM,k*BK],As,cluster_mask=3)
                else:  T.copy(A[by*BM,k*BK],As)
                T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return tilelang.compile(main,out_idx=[2],target="cuda")
A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
for mc in (False,True):
    k=build(mc)
    for _ in range(3): k(A,B)
    th.cuda.synchronize()
    e0=th.cuda.Event(True); e1=th.cuda.Event(True)
    e0.record()
    for _ in range(5): k(A,B)
    e1.record(); th.cuda.synchronize()
    ms=e0.elapsed_time(e1)/5
    print(f"multicast={mc}: {ms:.4f} ms -> {2*M*N*K/(ms*1e-3)/1e12:.1f} TFLOP/s")