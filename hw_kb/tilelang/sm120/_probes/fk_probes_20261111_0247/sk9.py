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

import torch
def bench(M,N,K,builder,tag,atomic):
    A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
    ref=(A.float()@B.float())
    try:
        if atomic:
            k=tilelang.compile(builder,target="cuda")
            C0=th.zeros(M,N,device="cuda",dtype=th.float32)
            k(A,B,C0); th.cuda.synchronize(); out=C0.clone()
            # graph replay: C is re-zeroed by a separate pre-node each iter isn't in graph;
            # instead use 2 buffers and alternate, C stays dirty -> use plain events w/o zero in timed region
            C1=th.zeros(M,N,device="cuda",dtype=th.float32)
            g=th.cuda.CUDAGraph()
            s=th.cuda.Stream(); s.wait_stream(th.cuda.current_stream()); th.cuda.synchronize()
            with th.cuda.stream(s):
                for _ in range(3): k(A,B,C0)
            th.cuda.current_stream().wait_stream(s); th.cuda.synchronize()
            # can't replay atomic kernel into same C (accumulates). time N launches of fresh-zeroed C2
            Cref=th.zeros(M,N,device="cuda",dtype=th.float32)
            NITER=20
            buf=th.zeros(NITER,M,N,device="cuda",dtype=th.float32)
            th.cuda.synchronize(); e0=th.cuda.Event(True); e1=th.cuda.Event(True)
            e0.record()
            for i in range(NITER): k(A,B,buf[i])
            e1.record(); th.cuda.synchronize(); ms=e0.elapsed_time(e1)/NITER
            out=buf[0]  # correct (was zero-initialized)
        else:
            k=tilelang.compile(builder,out_idx=[2],target="cuda"); out=k(A,B)
            for _ in range(3): k(A,B)
            th.cuda.synchronize(); e0=th.cuda.Event(True); e1=th.cuda.Event(True); e0.record()
            for _ in range(20): k(A,B)
            e1.record(); th.cuda.synchronize(); ms=e0.elapsed_time(e1)/20
        md=(out.float()-ref).abs().max().item(); cos=th.nn.functional.cosine_similarity(out.flatten(),ref.flatten(),dim=0).item()
        print(f"{tag}: {2*M*N*K/(ms*1e-3)/1e12:6.1f} TFLOP/s  md={md:.3g} cos={cos:.5f}")
    except Exception as e: print(f"{tag}: FAIL {str(e)[:160]}")

for (M,N,K) in [(512,512,16384),(1024,1024,8192),(4096,4096,4096)]:
    print(f"--- {M}x{N}x{K} (tiles={((M+127)//128)*((N+127)//128)}) ---")
    bench(M,N,K,plain(M,N,K),"plain",False)
    for SK in (2,4,8): bench(M,N,K,splitk(M,N,K,SK),f"splitK={SK}",True)
    for NW in (NSM,NSM*2,NSM*4): bench(M,N,K,streamk(M,N,K,NW),f"streamK NW={NW}",True)