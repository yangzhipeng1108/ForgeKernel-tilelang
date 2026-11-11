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
m=MiniGPT(1024,512,2048,4,8,256).to(dev)
lf=nn.CrossEntropyLoss()
ddp=BucketDDP(m,16); gen=torch.Generator(device=dev).manual_seed(1234+rank*7919)
idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
m.zero_grad(set_to_none=True); ddp.reset(); loss.backward(); ddp.sync()
# after sync, grads must be identical across ranks
flat=torch.cat([p.grad.flatten().float() for p in m.parameters()]).clone()
g=[torch.empty_like(flat) for _ in range(world)]; dist.all_gather(g,flat)
if rank==0: print(f"after 1 backward+sync: grad r0-r1 maxdiff={(g[0]-g[1]).abs().max().item():.2e}")
# apply optimizer step (AdamW) then compare weights
opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
opt.step()
w=torch.cat([p.detach().flatten().float() for p in m.parameters()]).clone()
gw=[torch.empty_like(w) for _ in range(world)]; dist.all_gather(gw,w)
if rank==0: print(f"after 1 opt.step: weight r0-r1 maxdiff={(gw[0]-gw[1]).abs().max().item():.2e}")
dist.barrier(); dist.destroy_process_group()
