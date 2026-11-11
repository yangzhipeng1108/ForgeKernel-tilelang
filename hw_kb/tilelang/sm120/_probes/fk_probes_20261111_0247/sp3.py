import tilelang, tilelang.language as T, torch as th
BM=BN=BK=64; M=N=1024; K=1024; d="float16"
def sp():
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
    return main
def dense():
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),d)):
        with T.Kernel(T.ceildiv(N,128),T.ceildiv(M,128),threads=128) as (bx,by):
            As=T.alloc_shared((128,64),d); Bs=T.alloc_shared((64,128),d)
            Cl=T.alloc_fragment((128,128),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,64),num_stages=2):
                T.copy(A[by*128,k*64],As); T.copy(B[k*64,bx*128],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*128,bx*128])
    return main

# 2:4 structured sparsity: keep 2 of every 4 along K
th.manual_seed(0)
Ad=th.randn(M,K,device="cuda",dtype=th.float16)
A4=Ad.view(M,K//4,4); mask=th.zeros(K//4,4,device="cuda",dtype=th.bool); mask[:,:2]=True
A4=A4*mask
Ad=A4.view(M,K)
# compress: A_sparse (M,K/2) keeps the 2 nonzero per group; E metadata
Aval=A4.view(M,K//4,2)  # first two of each group
A_sparse=Aval.reshape(M,K//2).contiguous()
B=th.randn(K,N,device="cuda",dtype=th.float16)
# metadata: for simple "first two" pattern, E encodes indices 0,1 -> standard 2:4 code
# tilelang expects E (M, K/8) uint8. Build per 16-element? Simpler: try zeros and check magnitude
E=th.zeros(M,K//8,device="cuda",dtype=th.uint8)
ref=(Ad.float()@B.float())
for tag,builder,args in [("dense",dense(),(Ad,B)),("sparse",sp(),(A_sparse,E,B))]:
    try:
        k=tilelang.compile(builder,out_idx=[2 if tag=="dense" else 3],target="cuda")
        out=k(*args); th.cuda.synchronize()
        md=(out.float()-ref).abs().max().item()
        cos=th.nn.functional.cosine_similarity(out.float().flatten(),ref.flatten(),dim=0).item()
        e0=th.cuda.Event(True); e1=th.cuda.Event(True)
        for _ in range(3): k(*args)
        th.cuda.synchronize(); e0.record()
        for _ in range(20): k(*args)
        e1.record(); th.cuda.synchronize(); ms=e0.elapsed_time(e1)/20
        print(f"{tag}: {ms*1000:.1f} us md={md:.3g} cos={cos:.5f}")
    except Exception as e: print(f"{tag}: FAIL {str(e)[:300]}")