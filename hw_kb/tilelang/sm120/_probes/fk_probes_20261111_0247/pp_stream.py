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
kmb=tilelang.compile(gemm(MBm,N,N,out="float16"),out_idx=[2],target="cuda")
kfull=tilelang.compile(gemm(M,N,N,out="float16"),out_idx=[2],target="cuda")
dev=torch.device("cuda",lr); torch.manual_seed(0)
per=NLAYER//w
Ws=[torch.randn(N,N,device=dev,dtype=torch.float16)*0.1 for _ in range(per)]
# ---- KEY: use a dedicated comm stream + async send/recv with explicit events.
# TileLang launches on the CURRENT torch stream, so we drive compute on a compute
# stream and p2p on a comm stream, and use tensor.copy_ on comm stream to stage.
comp_s=torch.cuda.Stream(device=lr)
comm_s=torch.cuda.Stream(device=lr)
torch.cuda.synchronize()

def run_overlap():
    # Both ranks iterate micro-batches. For each mb:
    #   stage0: compute (comp_s) -> record ev -> comm_s copies+isend
    #   stage1: comm_s irecv -> record ev -> comp_s waits ev -> compute
    # Sends and the *next* mb compute proceed concurrently because they are on
    # different streams and we only synchronize via events.
    ev_done=[torch.cuda.Event() for _ in range(MB)]
    inbuf=[torch.randn(MBm,N,device=dev,dtype=torch.float16)*0.1 for _ in range(MB)]
    sendbuf=[torch.empty(MBm,N,device=dev,dtype=torch.float16) for _ in range(MB)]
    recvbuf=[torch.empty(MBm,N,device=dev,dtype=torch.float16) for _ in range(MB)]
    reqs=[]
    for mb in range(MB):
        if r==0:
            with torch.cuda.stream(comp_s):
                x=inbuf[mb]
                for wl in Ws: x=kmb(x,wl)
                sendbuf[mb].copy_(x)  # stage into pinned-ish buffer on comp stream
            ev_done[mb].record(comp_s)
        else:
            comm_s.wait_event(ev_done[mb]) if mb<0 else None
    # launch p2p separately
    if r==0:
        for mb in range(MB):
            comm_s.wait_event(ev_done[mb])
            with torch.cuda.stream(comm_s):
                req=dist.isend(sendbuf[mb],dst=1); reqs.append(req)
    else:
        for mb in range(MB):
            with torch.cuda.stream(comm_s):
                req=dist.irecv(recvbuf[mb],src=0)
            req.wait()
            e=torch.cuda.Event(); e.record(comm_s)
            comp_s.wait_event(e)
            with torch.cuda.stream(comp_s):
                x=recvbuf[mb]
                for wl in Ws: x=kmb(x,wl)
    for q in reqs: q.wait()
    comp_s.synchronize(); comm_s.synchronize()

def timeit(fn,it=8):
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
        for wl in allW: x=kfull(x,wl)
        return x
    t_ref=do_bench(ref,warmup=8,rep=15)
else: t_ref=0
dist.barrier()
tp=timeit(run_overlap)
if r==0:
    print(f"PP-stream world={w} NLAYER={NLAYER} MB={MB} M={M} N={N} | 1GPU {NLAYER}L {t_ref*1000:.1f}us | PP-stream {tp*1000:.1f}us speedup {t_ref/tp:.2f}x")
dist.barrier(); dist.destroy_process_group()
