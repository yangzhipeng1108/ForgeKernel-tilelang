import os, math, torch, torch.distributed as dist, torch.nn as nn
os.environ.setdefault("NCCL_P2P_LEVEL","SYS")
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
dev=torch.device("cuda",lr)
src=open("/tmp/ddp_overlap.py").read().split("class BucketDDP")[0]
head=src.split('lr=int(os.environ["LOCAL_RANK"])')[1]
head='import os, math, torch, torch.distributed as dist, torch.nn as nn'+head
head=head.replace('dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()','rank=dist.get_rank(); world=dist.get_world_size()')
exec(head)
class OneBucket:
    # single flattened bucket, all_reduce on comm stream after last grad ready (overlap tail of backward)
    def __init__(s,model):
        s.model=model; s.world=world
        s.params=[p for p in model.parameters() if p.requires_grad]
        s.flat=torch.zeros(sum(p.numel() for p in s.params),device=dev,dtype=torch.float32)
        s.n=len(s.params); s.cnt=0; s.ev=None; s.comm=torch.cuda.Stream(device=lr)
        s._done=False
        for p in s.params:
            p.register_post_accumulate_grad_hook(s._hook)
    def _hook(s,g):
        s.cnt+=1
        if s.cnt==s.n:
            off=0
            for p in s.params:
                n=p.numel(); s.flat[off:off+n].copy_(p.grad.flatten()); off+=n
            with torch.cuda.stream(s.comm):
                dist.all_reduce(s.flat,op=dist.ReduceOp.SUM)
            s.ev=torch.cuda.Event(); s.ev.record(s.comm)
    def sync(s):
        s.comm.synchronize()
        f=s.flat/s.world; off=0
        for p in s.params:
            n=p.numel(); p.grad.copy_(f[off:off+n].view_as(p.grad)); off+=n
        torch.cuda.current_stream().wait_stream(s.comm)
    def reset(s):
        s.cnt=0; s.ev=None
def run(steps=15):
    torch.manual_seed(1234); m=MiniGPT(1024,512,2048,4,8,256).to(dev)
    lf=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    d=OneBucket(m); gen=torch.Generator(device=dev).manual_seed(1234+rank*7919)
    def step():
        idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
        loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True); d.reset(); loss.backward(); d.sync(); opt.step()
        return loss.item()
    for _ in range(3): step()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(steps): step()
    e.record(); e.synchronize(); dist.barrier()
    ms=s.elapsed_time(e)/steps
    # drift
    w=torch.cat([p.detach().flatten().float() for p in m.parameters()]).clone()
    gw=[torch.empty_like(w) for _ in range(world)]; dist.all_gather(gw,w)
    dr=(gw[0]-gw[1]).abs().max().item()
    return ms,dr
ms,dr=run()
if rank==0: print(f"single-bucket stream overlap: {ms:.2f}ms/step, weight drift r0-r1={dr:.2e}")
dist.barrier(); dist.destroy_process_group()
