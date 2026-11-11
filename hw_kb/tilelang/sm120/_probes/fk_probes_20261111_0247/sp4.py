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
k=tilelang.compile(main,out_idx=[3],target="cuda")
th.manual_seed(0)
Ad=th.randn(M,K,device="cuda",dtype=th.float16)
# keep first 2 of every 4 along K
g=Ad.view(M,K//4,4); keep=g.clone(); keep[...,2:]=0; Ad=keep.view(M,K)
A_sparse=g[...,:2].reshape(M,K//2).contiguous()
B=th.randn(K,N,device="cuda",dtype=th.float16)
ref=(Ad.float()@B.float())
# metadata encoding trials: 2-bit selector per group-of-4, packed K/8 bytes/row.
# Try several uniform byte values; find which reproduces keep{0,1}.
for val in (0x00,0x44,0x88,0x11,0x22,0x55,0xAA,0xFF):
    E=th.full((M,K//8),val,device="cuda",dtype=th.uint8)
    try:
        out=k(A_sparse,E,B); th.cuda.synchronize()
        cos=th.nn.functional.cosine_similarity(out.float().flatten(),ref.flatten(),dim=0).item()
        md=(out.float()-ref).abs().max().item()
        print(f"E=0x{val:02x}: cos={cos:.5f} md={md:.3g} mean|out|={out.float().abs().mean():.3f}")
    except Exception as e: print(f"E=0x{val:02x}: FAIL {str(e)[:120]}")
print("ref mean|.|=", ref.abs().mean().item())