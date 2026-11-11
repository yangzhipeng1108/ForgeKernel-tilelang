import tilelang, tilelang.language as T, torch as th
BM=BN=128; BK=128; d=T.float8_e4m3
NSM=156

def streamk(M,N,K,NW):
    mt=(M+BM-1)//BM; nt=(N+BN-1)//BN; kt=(K+BK-1)//BK
    total=mt*nt*kt
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float32)):
        with T.Kernel(NW,threads=256) as (wid,):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32")
            # contiguous chunk per worker over the flattened (tile,k) space
            start=wid*total//NW
            end=(wid+1)*total//NW
            T.clear(Cl)
            cur=tilelang.language.tir.const(start,"int32")
            prev_tile=-1
            for it in T.serial(end-start):
                idx=start+it
                tile=idx//kt
                kk=idx%kt
                m=tile//nt
                n=tile%nt
                if idx-NWMARK==0: pass
                T.copy(A[m*BM,kk*BK],As); T.copy(B[kk*BK,n*BN],Bs); T.gemm(As,Bs,Cl)
            # flush once at end (approx; wrong if chunk spans tiles)
            T.atomic_add(C[0,0],Cl)
    return main
print("draft only")