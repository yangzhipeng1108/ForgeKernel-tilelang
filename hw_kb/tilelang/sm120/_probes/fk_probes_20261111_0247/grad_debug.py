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
# same fixed batch on both ranks to compare gradients directly
torch.manual_seed(999)
idx=torch.randint(0,1024,(8,256),device=dev)
# bucket path
abs_grads={n:None for n,_ in m.named_parameters()}
for n,p in m.named_parameters():
    p.register_post_accumulate_grad_hook(lambda g,n=n: abs_grads.__setitem__(n,g.detach().clone()) if False else None)
ddp=BucketDDP(m,16)
opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
opt.zero_grad(set_to_none=True); ddp.reset(); loss.backward(); ddp.sync()
bucket_grad={n:p.grad.detach().clone() for n,p in m.named_parameters()}
# reference: dist all_reduce of un-sharded grad (same batch so equal)
ref_grad={}
for n,p in m.named_parameters():
    g=p.grad.detach().clone()
    dist.all_reduce(g,op=dist.ReduceOp.SUM); g/=world
    ref_grad[n]=g
maxdiff=max((bucket_grad[n]-ref_grad[n]).abs().max().item() for n in ref_grad)
if rank==0:
    for n in list(ref_grad)[:2]:
        print(f"{n}: bucket[:3]={bucket_grad[n].flatten()[:3].tolist()} ref[:3]={ref_grad[n].flatten()[:3].tolist()}")
    print(f"max grad diff bucket vs ref = {maxdiff:.2e}")
dist.barrier(); dist.destroy_process_group()
