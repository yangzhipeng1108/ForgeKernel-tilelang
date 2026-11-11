import tilelang, tilelang.language as T, torch as th
BM=BN=128; BK=128; d=T.float8_e4m3
M=N=512; K=16384; SK=2
kt=T.ceildiv(K,BK); per=T.ceildiv(kt,SK)
print("kt",kt,"per",per,"SK*per",SK*per)
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
k=tilelang.compile(main,out_idx=[2],target="cuda")
A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
C=th.zeros(M,N,device="cuda",dtype=th.float32)
out=k(A,B)
print("T.atomic_add C[slice] where C shape", C.shape)
print("out shape", out.shape, "out[0,:4]", out[0,:4].tolist())
print("ref[0,:4]", (A.float()@B.float())[0,:4].tolist())
# manual: zero then call twice
C.zero_()