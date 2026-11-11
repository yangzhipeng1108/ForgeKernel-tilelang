import tilelang, tilelang.language as T, torch as th
BM=BN=128; BK=128; d=T.float8_e4m3
M=N=K=512; NW=32
K=16384
mt=(M+BM-1)//BM; nt=(N+BN-1)//BN; kt=(K+BK-1)//BK
total=mt*nt*kt
print("tiles",mt*nt,"kt",kt,"total",total)
@T.prim_func
def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float32)):
    with T.Kernel(NW,threads=256) as (wid,):
        As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
        Cl=T.alloc_fragment((BM,BN),"float32")
        acc_tile=T.alloc_var("int32",init=0)
        start=T.alloc_var("int32",init=0); end=T.alloc_var("int32",init=0)
        start[0]=wid*total//NW
        end[0]=(wid+1)*total//NW
        T.clear(Cl)
        idx=T.alloc_var("int32",init=0)
        for it in T.serial(end[0]-start[0]):
            idx[0]=start[0]+it
            tile=idx[0]//kt; kk=idx[0]%kt
            m=tile//nt; n=tile%nt
            if it>0:
                if tile!=acc_tile[0]:
                    T.atomic_add(C[(acc_tile[0]//nt)*BM,(acc_tile[0]%nt)*BN],Cl)
                    T.clear(Cl)
            acc_tile[0]=tile
            T.copy(A[m*BM,kk*BK],As); T.copy(B[kk*BK,n*BN],Bs); T.gemm(As,Bs,Cl)
        T.atomic_add(C[(acc_tile[0]//nt)*BM,(acc_tile[0]%nt)*BN],Cl)
k=tilelang.compile(main,target="cuda")
A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
ref=(A.float()@B.float())
C=th.zeros(M,N,device="cuda",dtype=th.float32)
k(A,B,C); th.cuda.synchronize()
print("md",(C-ref).abs().max().item(),"cos",th.nn.functional.cosine_similarity(C.flatten(),ref.flatten(),dim=0).item())