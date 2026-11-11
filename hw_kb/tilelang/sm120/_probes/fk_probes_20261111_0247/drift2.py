import os, math, torch, torch.distributed as dist, torch.nn as nn
os.environ.setdefault("NCCL_P2P_LEVEL","SYS")
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
dev=torch.device("cuda",lr)
src=open("/tmp/ddp_overlap.py").read().split("def run(")[0]
head=src.split('lr=int(os.environ["LOCAL_RANK"])')[1]
head='import os, math, torch, torch.distributed as dist, torch.nn as nn'+head
head=head.replace('dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()','rank=dist.get_rank(); world=dist.get_world_size()')
exec(head)
torch.manual_seed(1234)
m=MiniGPT(1024,512,2048,4,8,256).to(dev)   # SAME init on all ranks
opt=torch.optim.AdamW(m.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss()
ddp=BucketDDP(m,16); gen=torch.Generator(device=dev).manual_seed(1234+rank*7919)
for it in range(6):
    idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
    loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
    opt.zero_grad(set_to_none=True); ddp.reset(); loss.backward(); ddp.sync(); opt.step()
# compare weights across ranks via gather
flat=torch.cat([p.detach().flatten().float() for p in m.parameters()]).clone()
gathered=[torch.empty_like(flat) for _ in range(world)]
dist.all_gather(gathered,flat)
d01=(gathered[0]-gathered[1]).abs().max().item()
if rank==0: print(f"rank0 vs rank1 weight max diff after 6 steps = {d01:.2e}")
dist.barrier(); dist.destroy_process_group()
