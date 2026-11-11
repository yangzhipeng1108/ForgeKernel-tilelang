import tilelang, tilelang.language as T, torch as th
BM=BN=BK=64; M=N=1024; K=1024; d="float16"
@T.prim_func
def main(A:T.Tensor((M,K//2),d),E:T.Tensor((M,K//8),"uint8"),B:T.Tensor((K,N),d),C:T.Tensor((M,N),d)):
    with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
        As=T.alloc_shared((BM,BK//2),d); Es=T.alloc_shared((BM,BK//8),"uint8")
        Bs=T.alloc_shared((BK,BN),d)
        Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
            T.copy(A[by*BM,k*(BK//2)],As); T.copy(E[by*BM,k*(BK//8)],Es); T.copy(B[k*BK,bx*BN],Bs)
            T.gemm_sp(As,Es,Bs,Cl)
        T.copy(Cl,C[by*BM,bx*BN])
try:
    k=tilelang.compile(main,out_idx=[3],target="cuda")
    print("COMPILE OK")
    src=k.get_kernel_source()
    import re
    print("mma.sp in src:", "mma.sp" in src, "| sparse refs:", len(re.findall(r"sparse|\.sp\.", src)))
    print("wgmma:", "wgmma" in src, "| tcgen05:", "tcgen05" in src)
except Exception as e:
    print("FAIL:", type(e).__name__); print(str(e)[:1500])