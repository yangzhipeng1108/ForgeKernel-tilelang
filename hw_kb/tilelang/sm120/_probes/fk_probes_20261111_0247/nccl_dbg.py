import os, torch, torch.distributed as dist
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl")
t=torch.ones(8*1024*1024//4,device=f"cuda:{lr}",dtype=torch.float32)
for _ in range(10): dist.all_reduce(t)
torch.cuda.synchronize(); dist.barrier(); dist.destroy_process_group()
