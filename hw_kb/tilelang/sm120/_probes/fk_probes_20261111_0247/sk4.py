import tilelang, tilelang.language as T, torch as th
BM=BN=128; BK=128; d=T.float8_e4m3
M=N=512; K=16384
def splitk(SK, outidx):
    kt=T.ceildiv(K,BK); per=T.ceildiv(kt,SK)
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
    return tilelang.compile(main,out_idx=outidx,target="cuda")
A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
ref=(A.float()@B.float())
for SK in (2,4,8):
  # (a) out_idx alloc
  k=splitk(SK,[2]); o1=k(A,B); o2=k(A,B); o3=k(A,B)
  print(f"SK={SK} outidx: run1 md={(o1-ref).abs().max():.2g} run2 md={(o2-ref).abs().max():.2g} run3 md={(o3-ref).abs().max():.2g} r1==r2 {th.allclose(o1,o2)}")
  # (b) explicit zeroed C, no out_idx
  k2=splitk(SK,None)
  C=th.zeros(M,N,device="cuda",dtype=th.float32); k2(A,B,C); th.cuda.synchronize()
  print(f"SK={SK} manualC: md={(C-ref).abs().max():.2g}")
  C.zero_(); k2(A,B,C); th.cuda.synchronize()
  print(f"SK={SK} manualC 2nd: md={(C-ref).abs().max():.2g}")