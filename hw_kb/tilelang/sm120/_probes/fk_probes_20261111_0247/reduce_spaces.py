import os, torch, torch.distributed as dist, time
os.environ.setdefault("NCCL_P2P_LEVEL","SYS")
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
dev=torch.device("cuda",lr)
def timeit(fn,it=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); e.synchronize(); dist.barrier()
    return s.elapsed_time(e)/it
# 1) does reducing in fp16 / bf16 beat fp32 for same elements? (halves bytes)
res=[]
for dt in [torch.float32, torch.float16, torch.bfloat16]:
    t=torch.ones(8*1024*1024,device=dev,dtype=dt)  # 8M elems
    ms=timeit(lambda: dist.all_reduce(t,op=dist.ReduceOp.SUM))
    nbytes=t.numel()*t.element_size()
    res.append(f"{str(dt).split('.')[-1]} 8M: {ms*1000:.0f}us {nbytes/(ms*1e-3)/1e9:.1f}GB/s")
# 2) reduce-scatter + all-gather vs all-reduce (same algo bytes) - torch supports via _reduce_scatter_base
n=8*1024*1024
inp=torch.ones(n*world,device=dev,dtype=torch.float32)
out=torch.empty(n,device=dev,dtype=torch.float32)
ms_ar=timeit(lambda: dist.all_reduce(inp,op=dist.ReduceOp.SUM))
res.append(f"all_reduce 8M*2={inp.numel()}: {ms_ar*1000:.0f}us")
# 3) can we overlap compute with comm? measure all_reduce time while a big gemm runs on same stream vs diff stream
# 4) NCCL algorithms: ring vs tree
os.environ["NCCL_ALGO"]="Tree"
if rank==0: print(" | ".join(res))
dist.barrier(); dist.destroy_process_group()
