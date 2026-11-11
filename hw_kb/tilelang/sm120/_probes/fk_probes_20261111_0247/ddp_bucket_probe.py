import os, torch, torch.distributed as dist
os.environ.setdefault("NCCL_P2P_LEVEL","SYS")
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
dev=torch.device("cuda",lr)

# Compare three gradient-sync strategies on a chunk of realistic-sized params.
# We only measure the SYNC cost + a stand-in for backward compute (sleep-like GEMMs
# are overkill; use real all_reduce vs bucketed all_reduce, and measure whether
# bucketing reduces launch overhead and increases message size efficiency).

sizes=[(512,512),(512,2048),(2048,512),(2048,512),(512,2048),(512,512)]  # ~ mlp/gqa-ish
params=[torch.randn(*s,device=dev,dtype=torch.float32) for s in sizes]
grads=[torch.randn_like(p) for p in params]

def naive_sync():
    # one all_reduce per parameter (what train_dp does now)
    for g in grads:
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        g /= world
    return

# bucket: flatten into ~ fixed-byte buckets, one all_reduce per bucket
BUCKET_BYTES = int(os.environ.get("BUCKET_MB","4"))*1024*1024
def bucketed_sync():
    flat=torch._utils._flatten_dense_tensors(grads)
    n=flat.numel(); per=BUCKET_BYTES//4
    for i in range(0,n,per):
        chunk=flat[i:i+per]
        dist.all_reduce(chunk, op=dist.ReduceOp.SUM)
        chunk /= world
    flat2=flat/world  # also need scale on full (simplify: scale once)
    outs=torch._utils._unflatten_dense_tensors(flat,n_splits_grads)
    # (we skip copyback for timing purity; timing dominated by all_reduce)

def timeit(fn,name,it=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); e.synchronize(); dist.barrier()
    return s.elapsed_time(e)/it

n_splits_grads=[len(grads)]
t_naive=timeit(naive_sync,"naive")
print(f"[rank{rank}] naive per-param allreduce: {t_naive*1000:.1f}us  ({len(grads)} calls)")
# bucketed with proper copyback
def bucketed_sync2():
    flat=torch._utils._flatten_dense_tensors(grads)
    n=flat.numel(); per=BUCKET_BYTES//4
    for i in range(0,n,per):
        dist.all_reduce(flat[i:i+per], op=dist.ReduceOp.SUM)
    flat/=world
    outs=torch._utils._unflatten_dense_tensors(flat,[g.numel() for g in grads])
    for g,o in zip(grads,outs): g.copy_(o)
for bm in [1,4,16]:
    BUCKET_BYTES=bm*1024*1024
    def mk(bm=bm):
        def f():
            flat=torch._utils._flatten_dense_tensors(grads)
            per=bm*1024*1024//4
            for i in range(0,flat.numel(),per):
                dist.all_reduce(flat[i:i+per],op=dist.ReduceOp.SUM)
            flat/=world
            outs=torch._utils._unflatten_dense_tensors(flat,[g.numel() for g in grads])
            for g,o in zip(grads,outs): g.copy_(o)
        return f
    t=timeit(mk(),"bucket")
    print(f"[rank{rank}] bucketed {bm}MB: {t*1000:.1f}us")
dist.barrier(); dist.destroy_process_group()
