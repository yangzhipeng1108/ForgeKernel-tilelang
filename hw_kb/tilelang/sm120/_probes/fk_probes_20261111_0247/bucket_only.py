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
lf=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
gen=torch.Generator(device=dev).manual_seed(1234+rank*7919)
# deterministic bucketed sync (no overlap, default stream): flatten ALL grads, one all_reduce, copy back
params=[p for p in m.parameters() if p.requires_grad]
for it in range(6):
    idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
    loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward()
    flat=torch._utils._flatten_dense_tensors([p.grad for p in params])
    dist.all_reduce(flat,op=dist.ReduceOp.SUM); flat/=world
    grads=torch._utils._unflatten_dense_tensors(flat,[p.grad for p in params])
    for p,ng in zip(params,grads): p.grad.copy_(ng)
    opt.step()
w=torch.cat([p.detach().flatten().float() for p in m.parameters()]).clone()
gw=[torch.empty_like(w) for _ in range(world)]; dist.all_gather(gw,w)
if rank==0: print(f"deterministic bucket-only after 6 steps: weight r0-r1 maxdiff={(gw[0]-gw[1]).abs().max().item():.2e}")
dist.barrier(); dist.destroy_process_group()
