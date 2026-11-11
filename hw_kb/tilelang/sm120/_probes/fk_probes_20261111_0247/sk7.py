import tilelang, tilelang.language as T, torch as th
BM=BN=128; BK=128; d=T.float8_e4m3
M=N=512; K=16384; NW=32
mt=(M+BM-1)//BM; nt=(N+BN-1)//BN; kt=(K+BK-1)//BK
total=mt*nt*kt
# Use one program per output tile but with a strided/balanced K partition is split-K.
# True stream-K: flatten, each worker gets contiguous [start,end). Flush at tile change.
@T.prim_func
def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float32)):
    with T.Kernel(NW,threads=256) as (wid,):
        As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
        Cl=T.alloc_fragment((BM,BN),"float32")
        T.clear(Cl)
        for it in T.serial((wid+1)*total//NW - wid*total//NW):
            idx=wid*total//NW + it
            tile=idx//kt; kk=idx%kt
            m=tile//nt; n=tile%nt
            # accumulate; flush when moving to a new tile (before first iter of new tile)
            if (it>0) and (tile != (idx-1)//kt):
                ptile=(idx-1)//kt
                T.atomic_add(C[(ptile//nt)*BM,(ptile%nt)*BN],Cl)
                T.clear(Cl)
            T.copy(A[m*BM,kk*BK],As); T.copy(B[kk*BK,n*BN],Bs); T.gemm(As,Bs,Cl)
        ltile=((wid+1)*total//NW - 1)//kt
        T.atomic_add(C[(ltile//nt)*BM,(ltile%nt)*BN],Cl)
k=tilelang.compile(main,target="cuda")
A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
ref=(A.float()@B.float())
C=th.zeros(M,N,device="cuda",dtype=th.float32)
k(A,B,C); th.cuda.synchronize()
print("md",(C-ref).abs().max().item(),"cos",th.nn.functional.cosine_similarity(C.flatten(),ref.flatten(),dim=0).item())