import tilelang, tilelang.language as T, torch as th
from tilelang.profiler import do_bench
from tilelang.carver.arch import driver
NSM=driver.get_num_sms()
BM=BN=128; M=N=K=4096; BK=128; d=T.float8_e4m3

def persistent_stateful(swz,nsm,colmaj,disable_ws):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        with T.Kernel(nsm,threads=256) as (bid,):
            sched=T.PersistentTileScheduler(T.ceildiv(M,BM),T.ceildiv(N,BN),num_workers=nsm,swizzle_size=swz,column_major=colmaj)
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32")
            sched.init(bid)
            while sched.valid():
                by=sched.m_idx; bx=sched.n_idx
                T.clear(Cl)
                for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                    T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
                T.copy(Cl,C[by*BM,bx*BN])
                sched.next_tile()
    cfg={"tl.disable_warp_specialized":True} if disable_ws else None
    return main,cfg

def persistent_for(swz,nsm,colmaj):
    m_blocks=T.ceildiv(M,BM); n_blocks=T.ceildiv(N,BN); total=m_blocks*n_blocks
    import math
    waves=(total+nsm-1)//nsm
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        with T.Kernel(nsm,threads=256) as (bid,):
            sched=T.PersistentTileScheduler(m_blocks,n_blocks,swizzle_size=swz,column_major=colmaj,stateful=False)
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32")
            for w in range(waves):
                t=nsm*w+bid
                if t<total:
                    by, bx = sched.coord(t)
                    if by*BM<M and bx*BN<N:
                        T.clear(Cl)
                        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                            T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
                        T.copy(Cl,C[by*BM,bx*BN])
    return main,None

A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
ref=(A.float()@B.float())
def bench(fn,cfg,tag):
    try:
        k=tilelang.compile(fn,out_idx=[2],target="cuda",pass_configs=cfg)
        t=do_bench(lambda: k(A,B),warmup=25,rep=100)
        C=k(A,B); md=(C.float()-ref).abs().max().item()
        print(f"{tag}: {2*M*N*K/(t*1e-3)/1e12:6.1f} TFLOP/s md={md:.3g}")
    except Exception as e: print(f"{tag}: FAIL {str(e)[:160]}")
fn,cfg=persistent_stateful(8,NSM,True,True); bench(fn,cfg,"stateful swz=8 col+maj, ws-off")
fn,cfg=persistent_for(8,NSM,True); bench(fn,cfg,"for-loop swz=8 col-major")
fn,cfg=persistent_for(1,NSM,False); bench(fn,cfg,"for-loop swz=1 row-major")