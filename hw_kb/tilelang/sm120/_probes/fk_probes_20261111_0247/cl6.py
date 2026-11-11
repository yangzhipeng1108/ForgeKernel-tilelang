import tilelang, tilelang.language as T, torch as th
d=T.float8_e4m3; BM=BN=128; BK=128
def build(M,N,K,cd,mask):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        with T.ClusterKernel(T.ceildiv(N,BN),T.ceildiv(M,BM),cluster_dims=cd,threads=256) as (bx,by):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy_cluster(A[by*BM,k*BK],As,cluster_mask=mask)
                T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return tilelang.compile(main,out_idx=[2],target="cuda")
# tiny: 2 clusters only, K small
for (M,N,K,cd,mask) in [(256,256,256,2,3),(256,256,1280,2,3),(256,384,256,2,3)]:
    A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
    ref=(A.float()@B.float())
    try:
        k=build(M,N,K,cd,mask); out=k(A,B); th.cuda.synchronize(); md=(out.float()-ref).abs().max().item()
        e0=th.cuda.Event(True);e1=th.cuda.Event(True);e0.record()
        for _ in range(10): k(A,B)
        e1.record(); th.cuda.synchronize(); ms=e0.elapsed_time(e1)/10
        print(f"{M}x{N}x{K} cd={cd}: {ms:.4f} ms md={md:.3g}")
    except Exception as e: print(f"{M}x{N}x{K}: FAIL {str(e)[:150]}")