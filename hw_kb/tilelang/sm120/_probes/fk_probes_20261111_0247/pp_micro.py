import os, torch, torch.distributed as dist, tilelang, tilelang.language as T
from tilelang.profiler import do_bench
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); r=dist.get_rank(); w=dist.get_world_size()
N=int(os.environ["N"]); M=int(os.environ["M"])
NLAYER=int(os.environ.get("NLAYER","4")); MB=int(os.environ.get("MB","2"))
def gemm(Mx,Nx,Kx,out="float16",BM=128,BN=128,BK=64,stages=2,threads=128):
    @T.prim_func
    def main(A:T.Tensor((Mx,Kx),"float16"),B:T.Tensor((Kx,Nx),"float16"),C:T.Tensor((Mx,Nx),out)):
        with T.Kernel(T.ceildiv(Nx,BN),T.ceildiv(Mx,BM),threads=threads) as (bx,by):
            a_s=T.alloc_shared((BM,BK),"float16"); b_s=T.alloc_shared((BK,BN),"float16")
            acc=T.alloc_fragment((BM,BN),"float32"); T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(Kx,BK),num_stages=stages):
                T.copy(A[by*BM,kk*BK],a_s); T.copy(B[kk*BK,bx*BN],b_s); T.gemm(a_s,b_s,acc)
            T.copy(acc,C[by*BM,bx*BN])
    return main
MBm=M//MB
k_mb=tilelang.compile(gemm(MBm,N,N,out="float16"),out_idx=[2],target="cuda")
k_full=tilelang.compile(gemm(M,N,N,out="float16"),out_idx=[2],target="cuda")
dev=torch.device("cuda",lr); torch.manual_seed(0)
layers_per=NLAYER//w
Ws=[torch.randn(N,N,device=dev,dtype=torch.float16)*0.1 for _ in range(layers_per)]
def run_pipeline_micro():
    for mb in range(MB):
        A=torch.randn(MBm,N,device=dev,dtype=torch.float16)*0.1
        if r==0:
            x=A
            for wl in Ws: x=k_mb(x,wl)
            dist.send(x,dst=1)
        else:
            x=torch.empty(MBm,N,device=dev,dtype=torch.float16)
            dist.recv(x,src=0)
            for wl in Ws: x=k_mb(x,wl)
def timeit(fn,it=10):
    for _ in range(2): fn()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); e.synchronize(); dist.barrier()
    return s.elapsed_time(e)/it
if r==0:
    A=torch.randn(M,N,device=dev,dtype=torch.float16)*0.1
    allW=[torch.randn(N,N,device=dev,dtype=torch.float16)*0.1 for _ in range(NLAYER)]
    def ref():
        x=A
        for wl in allW: x=k_full(x,wl)
        return x
    t_ref=do_bench(ref,warmup=10,rep=20)
else:
    t_ref=0
dist.barrier()
tp=timeit(run_pipeline_micro)
if r==0:
    print(f"PP-micro world={w} NLAYER={NLAYER} MB={MB} M={M} N={N} | 1GPU all {NLAYER}L {t_ref*1000:.1f}us | PP {tp*1000:.1f}us speedup {t_ref/tp:.2f}x")
dist.barrier(); dist.destroy_process_group()
