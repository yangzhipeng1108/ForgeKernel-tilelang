import os, torch, torch.distributed as dist, tilelang, tilelang.language as T
from tilelang.profiler import do_bench
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); r=dist.get_rank(); w=dist.get_world_size()
N=int(os.environ["N"]); M=int(os.environ["M"])
NLAYER=int(os.environ.get("NLAYER","8")); MB=int(os.environ.get("MB","8"))
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
# async pipeline: stage0 computes mb i then isend; overlaps with computing mb i+1.
# stage1 irecv mb i, computes, while stage0 works on i+1.
def run_async():
    reqs=[]
    bufs=[torch.randn(MBm,N,device=dev,dtype=torch.float16)*0.1 for _ in range(MB)]
    sends=[torch.empty(MBm,N,device=dev,dtype=torch.float16) for _ in range(MB)]
    for mb in range(MB):
        if r==0:
            x=bufs[mb]
            for wl in Ws: x=k_mb(x,wl)
            sends[mb].copy_(x)
            req=dist.isend(sends[mb],dst=1); reqs.append(req)
        else:
            req=dist.irecv(sends[mb],src=0); req.wait()
            x=sends[mb]
            for wl in Ws: x=k_mb(x,wl)
    for q in reqs: q.wait()
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
tp=timeit(run_async)
if r==0:
    print(f"PP-async world={w} NLAYER={NLAYER} MB={MB} M={M} N={N} | 1GPU {NLAYER}L {t_ref*1000:.1f}us | PP-async {tp*1000:.1f}us speedup {t_ref/tp:.2f}x")
dist.barrier(); dist.destroy_process_group()
