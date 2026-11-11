import os,torch,torch.distributed as dist
os.environ.setdefault("NCCL_P2P_LEVEL","SYS")
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr); dist.init_process_group("nccl")
r=dist.get_rank(); dev=torch.device("cuda",lr)
t=torch.ones(8*1024*1024,device=dev,dtype=torch.float32)
def ti(fn,it=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();e.synchronize();dist.barrier();return s.elapsed_time(e)/it
algo=os.environ.get("NCCL_ALGO","default")
ms=ti(lambda: dist.all_reduce(t,op=dist.ReduceOp.SUM))
if r==0: print(f"ALGO={algo}: {ms*1000:.0f}us")
dist.barrier();dist.destroy_process_group()
