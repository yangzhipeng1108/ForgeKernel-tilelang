import os, math, torch, torch.distributed as dist, torch.nn as nn
os.environ.setdefault("NCCL_P2P_LEVEL","SYS")
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
dev=torch.device("cuda",lr)
exec(open("/tmp/ddp_overlap.py").read().split("class BucketDDP")[0].split("dev=torch.device")[1].split("\n",1)[1]) if False else None
# import model/BucketDDP defs from the file
import importlib.util
spec=importlib.util.spec_from_file_location("ddpmod","/tmp/ddp_overlap.py")
# can't import (it runs main). Instead re-declare minimal by exec of class defs:
src=open("/tmp/ddp_overlap.py").read()
# grab everything up to the 'def run(' section
head=src.split("def run(")[0]
head=head.split('lr=int(os.environ["LOCAL_RANK"])')[1]
head='import os, math, torch, torch.distributed as dist, torch.nn as nn'+head
# but re-init dist? already inited; remove init lines
head=head.replace('dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()','rank=dist.get_rank(); world=dist.get_world_size()')
exec(head)
torch.manual_seed(1234)
m=MiniGPT(1024,512,2048,4,8,256).to(dev)
opt=torch.optim.AdamW(m.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss()
ddp=BucketDDP(m,16); gen=torch.Generator(device=dev).manual_seed(1234+rank*7919)
for _ in range(5):
    idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
    loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
    opt.zero_grad(set_to_none=True); ddp.reset(); loss.backward(); ddp.sync(); opt.step()
# check weight agreement across ranks
flat=torch.cat([p.detach().flatten().float() for p in m.parameters()])
mine=flat.clone(); dist.all_reduce(flat,op=dist.ReduceOp.SUM)
drift=(mine-flat/world).abs().max().item()
if rank==0: print(f"bucket-DDP weight drift max={drift:.2e} (0 => synced)")
dist.barrier(); dist.destroy_process_group()
