import os, torch, torch.distributed as dist, tilelang, tilelang.language as T
from tilelang.profiler import do_bench
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); r=dist.get_rank(); w=dist.get_world_size()
M=int(os.environ["M"]);N=int(os.environ["N"]);K=int(os.environ["K"]); Ks=K//w
CH=int(os.environ.get("CHUNKS",str(w)))  # number of K-chunks to pipeline
def gemm(Min,Nin,Kin,BM=128,BN=128,BK=64,stages=2,threads=128,out="float32"):
    @T.prim_func
    def main(A:T.Tensor((Min,Kin),"float16"),B:T.Tensor((Kin,Nin),"float16"),C:T.Tensor((Min,Nin),out)):
        with T.Kernel(T.ceildiv(Nin,BN),T.ceildiv(Min,BM),threads=threads) as (bx,by):
            a_s=T.alloc_shared((BM,BK),"float16"); b_s=T.alloc_shared((BK,BN),"float16")
            acc=T.alloc_fragment((BM,BN),"float32"); T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(Kin,BK),num_stages=stages):
                T.copy(A[by*BM,kk*BK],a_s); T.copy(B[kk*BK,bx*BN],b_s); T.gemm(a_s,b_s,acc)
            T.copy(acc,C[by*BM,bx*BN])
    return main
kfull=tilelang.compile(gemm(M,N,K),out_idx=[2],target="cuda")
# For overlap TP: split K into CH chunks per rank? Simpler: each rank owns K/w,
# but process in CH micro-chunks; all-reduce each chunk's partial on a comm stream
# while computing next chunk -> overlap.
Ksl=K//w
Kchunk=Ksl//CH
kpart=tilelang.compile(gemm(M,N,Kchunk),out_idx=[2],target="cuda")
dev=torch.device("cuda",lr); torch.manual_seed(0)
A=torch.randn(M,K,device=dev,dtype=torch.float16)*0.1; B=torch.randn(K,N,device=dev,dtype=torch.float16)*0.1
t1=do_bench(lambda:kfull(A,B),warmup=15,rep=30)
dist.barrier()
# rank r owns K columns [r*Ksl:(r+1)*Ksl], sliced into CH chunks
Asl=A[:, r*Ksl:(r+1)*Ksl].contiguous()
Bsl=B[r*Ksl:(r+1)*Ksl].contiguous()
Ach=[Asl[:,i*Kchunk:(i+1)*Kchunk].contiguous() for i in range(CH)]
Bch=[Bsl[i*Kchunk:(i+1)*Kchunk].contiguous() for i in range(CH)]
comm_stream=torch.cuda.Stream(device=lr)
acc=torch.zeros(M,N,device=dev,dtype=torch.float32)
# non-overlap baseline: allreduce each chunk serially on default stream
def tp_serial():
    acc.zero_()
    for i in range(CH):
        c=kpart(Ach[i],Bch[i]); dist.all_reduce(c); acc.add_(c)
    return acc
# overlap: compute chunk i on main stream, allreduce on comm stream, wait only when needed
def tp_overlap():
    acc.zero_()
    evs=[torch.cuda.Event() for _ in range(CH)]
    for i in range(CH):
        c=kpart(Ach[i],Bch[i])        # default stream
        evs[i].record()               # mark compute done
    # launch all allreduces on comm stream (they can overlap with subsequent compute
    # only if we interleave; simplest: comm stream waits each event then AR)
    with torch.cuda.stream(comm_stream):
        for i in range(CH):
            comm_stream.wait_event(evs[i])
            dist.all_reduce(c)  # NOTE: needs the actual tensor; see below
    torch.cuda.current_stream().wait_stream(comm_stream)
    return acc
# The above is wrong ref; do a correct ping-pong overlap:
def tp_overlap2():
    acc.zero_()
    comp_stream=torch.cuda.current_stream()
    prev_ev=None
    bufs=[torch.empty(M,N,device=dev,dtype=torch.float32) for _ in range(CH)]
    for i in range(CH):
        bufs[i]=kpart(Ach[i],Bch[i])   # compute on comp_stream
    # now overlap: for each buffer, AR on comm stream concurrently with... nothing left.
    # True overlap requires interleaving compute & comm across chunks:
    comp_evs=[]
    acc.zero_()
    for i in range(CH):
        c=kpart(Ach[i],Bch[i]); e=torch.cuda.Event(); e.record(comp_stream); comp_evs.append((c,e))
    with torch.cuda.stream(comm_stream):
        for c,e in comp_evs:
            comm_stream.wait_event(e); dist.all_reduce(c)
    comp_stream.wait_stream(comm_stream)
    for c,_ in comp_evs: acc.add_(c)
    return acc
# proper interleaved overlap:
def tp_ovlp():
    acc.zero_()
    comp_stream=torch.cuda.current_stream()
    evs=[torch.cuda.Event() for _ in range(CH)]
    cs=[]
    # issue compute for chunk0
    c0=kpart(Ach[0],Bch[0]); evs[0].record(comp_stream); cs.append(c0)
    for i in range(CH):
        # start AR of chunk i on comm stream
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(evs[i])
            dist.all_reduce(cs[i])
        # meanwhile compute chunk i+1 on comp stream
        if i+1<CH:
            c=kpart(Ach[i+1],Bch[i+1]); evs[i+1].record(comp_stream); cs.append(c)
    comp_stream.wait_stream(comm_stream)
    for c in cs: acc.add_(c)
    return acc
def timeit(fn,it=15):
    for _ in range(3): fn()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); e.synchronize(); dist.barrier()
    return s.elapsed_time(e)/it
ts=timeit(tp_serial)
to=timeit(tp_ovlp)
Ctp=tp_ovlp(); dist.barrier()
flop=2*M*N*K
if r==0:
    Cf=kfull(A,B); cos=torch.nn.functional.cosine_similarity(Ctp.flatten(),Cf.flatten(),dim=0).item()
    print(f"world={w} {M}x{N}x{K} CHUNKS={CH} | 1GPU {t1*1000:.1f}us | TP-serial {ts*1000:.1f}us {t1/ts:.2f}x | TP-overlap {to*1000:.1f}us {t1/to:.2f}x | cos={cos:.6f}")
dist.barrier(); dist.destroy_process_group()
