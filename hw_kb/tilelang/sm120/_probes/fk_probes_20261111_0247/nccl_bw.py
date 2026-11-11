import os, time, torch, torch.distributed as dist
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); r=dist.get_rank(); w=dist.get_world_size()
dev=torch.device("cuda",lr)
def timeit(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); e.synchronize(); return s.elapsed_time(e)/iters
res=[]
for sz,label in [(1<<20,"1MB"),(8<<20,"8MB"),(32<<20,"32MB"),(128<<20,"128MB")]:
    n=sz//4
    t=torch.ones(n,device=dev,dtype=torch.float32)
    out=torch.empty(n*w,device=dev,dtype=torch.float32)
    for op in ["allreduce","allgather","bcast"]:
        if op=="allreduce": fn=lambda: dist.all_reduce(t,op=dist.ReduceOp.SUM)
        elif op=="allgather": fn=lambda: dist.all_gather_into_tensor(out,t)
        else: fn=lambda: dist.broadcast(t,src=0)
        ms=timeit(fn)
        # for allreduce ring: busbw = alg * 2(w-1)/w / t ; report algorithmic GB/s here
        bw=sz/(ms*1e-3)/1e9
        res.append(f"  {label} {op}: {ms*1000:.1f}us  {bw:.1f} GB/s algo")
if r==0:
    print("world",w,"nccl",torch.cuda.nccl.version())
    print("\n".join(res))
dist.barrier(); dist.destroy_process_group()
