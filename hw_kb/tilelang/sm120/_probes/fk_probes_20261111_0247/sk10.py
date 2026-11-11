import tilelang, tilelang.language as T, torch as th
from tilelang.carver.arch import driver
NSM=driver.get_num_sms(); BM=BN=128; BK=128; d=T.float8_e4m3
def streamk(M,N,K,NW):
    mt=(M+BM-1)//BM; nt=(N+BN-1)//BN; kt=(K+BK-1)//BK; total=mt*nt*kt
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float32)):
        with T.Kernel(NW,threads=256) as (wid,):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for it in T.serial((wid+1)*total//NW - wid*total//NW):
                idx=wid*total//NW + it
                tile=idx//kt; kk=idx%kt; m=tile//nt; n=tile%nt
                if (it>0) and (tile != (idx-1)//kt):
                    ptile=(idx-1)//kt
                    T.atomic_add(C[(ptile//nt)*BM,(ptile%nt)*BN],Cl); T.clear(Cl)
                T.copy(A[m*BM,kk*BK],As); T.copy(B[kk*BK,n*BN],Bs); T.gemm(As,Bs,Cl)
            ltile=((wid+1)*total//NW - 1)//kt
            T.atomic_add(C[(ltile//nt)*BM,(ltile%nt)*BN],Cl)
    return main
def splitk(M,N,K,SK):
    mt=(M+BM-1)//BM; nt=(N+BN-1)//BN; kt=(K+BK-1)//BK; per=(kt+SK-1)//SK
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float32)):
        with T.Kernel(nt,mt,SK,threads=256) as (bx,by,sz):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for kk in T.Pipelined(per,num_stages=2):
                k=sz*per+kk
                if k<kt:
                    T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.atomic_add(C[by*BM,bx*BN],Cl)
    return main
def plain(M,N,K):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float32)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=256) as (bx,by):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return main
def bench(M,N,K,builder,tag,atomic):
    A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
    ref=(A.float()@B.float())
    try:
        if atomic:
            k=tilelang.compile(builder,target="cuda"); NI=20
            buf=th.zeros(NI,M,N,device="cuda",dtype=th.float32)
            k(A,B,buf[0]); th.cuda.synchronize(); out=buf[0].clone()
            th.cuda.synchronize(); e0=th.cuda.Event(True); e1=th.cuda.Event(True); e0.record()
            for i in range(NI): k(A,B,buf[i])
            e1.record(); th.cuda.synchronize(); ms=e0.elapsed_time(e1)/NI
        else:
            k=tilelang.compile(builder,out_idx=[2],target="cuda"); out=k(A,B)
            for _ in range(3): k(A,B)
            th.cuda.synchronize(); e0=th.cuda.Event(True); e1=th.cuda.Event(True); e0.record()
            for _ in range(20): k(A,B)
            e1.record(); th.cuda.synchronize(); ms=e0.elapsed_time(e1)/20
        md=(out.float()-ref).abs().max().item(); cos=th.nn.functional.cosine_similarity(out.flatten(),ref.flatten(),dim=0).item()
        print(f"{tag}: {2*M*N*K/(ms*1e-3)/1e12:6.1f} TFLOP/s md={md:.3g} cos={cos:.5f}")
    except Exception as e: print(f"{tag}: FAIL {str(e)[:150]}")
for (M,N,K) in [(2048,2048,8192),(2048,2048,4096),(768,768,8192)]:
    tiles=((M+127)//128)*((N+127)//128)
    print(f"--- {M}x{N}x{K} tiles={tiles} waves={tiles/NSM:.2f} ---")
    bench(M,N,K,plain(M,N,K),"plain",False)
    for SK in (1,2,4): bench(M,N,K,splitk(M,N,K,SK),f"splitK={SK}",True)
    for NW in (NSM,NSM*2): bench(M,N,K,streamk(M,N,K,NW),f"streamK NW={NW}",True)