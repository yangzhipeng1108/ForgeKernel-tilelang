import os, torch, torch.distributed as dist
os.environ.setdefault("NCCL_P2P_LEVEL","SYS")
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
dev=torch.device("cuda",lr)

sizes=[(512,512),(512,2048),(2048,512),(2048,512),(512,2048),(512,512)]
params=[torch.randn(*s,device=dev,dtype=torch.float32) for s in sizes]
grads=[torch.randn_like(p) for p in params]

def timeit(fn,it=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); e.synchronize(); dist.barrier()
    return s.elapsed_time(e)/it

def naive():
    for g in grads:
        dist.all_reduce(g, op=dist.ReduceOp.SUM); g/=world

def bucket_fn(bucket_bytes):
    per=bucket_bytes//4
    def f():
        flat=torch._utils._flatten_dense_tensors(grads)
        for i in range(0,flat.numel(),per):
            dist.all_reduce(flat[i:i+per],op=dist.ReduceOp.SUM)
        flat/=world
        outs=torch._utils._unflatten_dense_tensors(flat, grads)
        for g,o in zip(grads,outs): g.copy_(o)
    return f

tn=timeit(naive)
res=[f"rank{rank} naive {tn*1000:.1f}us({len(grads)}calls)"]
for bm in [1,2,4,8,16]:
    t=timeit(bucket_fn(bm*1024*1024))
    tot=sum(g.numel() for g in grads)*4
    ncalls=max(1,(tot+bm*1024*1024-1)//(bm*1024*1024))
    res.append(f"bucket{bm}MB {t*1000:.1f}us({ncalls}calls)")
if rank==0: print("\n".join(res))
dist.barrier(); dist.destroy_process_group()
