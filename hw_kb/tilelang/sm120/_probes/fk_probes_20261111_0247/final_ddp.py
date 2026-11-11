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

class MultiBucket:
    """True bucketed DDP: params grouped into buckets; each bucket's all_reduce
    fires on a comm stream as soon as its last grad is ready (post-accumulate
    grad hook), overlapping bucket k's comm with bucket k+1's backward."""
    def __init__(s,model,bucket_mb=8):
        s.model=model; s.world=world
        s.params=[p for p in model.parameters() if p.requires_grad]
        order=sorted(range(len(s.params)),key=lambda i:-s.params[i].numel())
        s.buckets=[];cur=[];cb=0
        for i in order:
            nb=s.params[i].numel()*s.params[i].element_size()
            if cur and cb+nb>bucket_mb*1024*1024: s.buckets.append(cur);cur=[];cb=0
            cur.append(i);cb+=nb
        if cur:s.buckets.append(cur)
        s.p2b={i:bi for bi,b in enumerate(s.buckets) for i in b}
        s.cnt=[0]*len(s.buckets); s.flat=[torch.zeros(sum(s.params[i].numel() for i in b),device=dev,dtype=torch.float32) for b in s.buckets]
        s.ev=[None]*len(s.buckets); s.comm=torch.cuda.Stream(device=lr)
        for i,p in enumerate(s.params): p.register_post_accumulate_grad_hook(lambda g,i=i: s._hook(i))
    def _hook(s,i):
        bi=s.p2b[i]; s.cnt[bi]+=1
        if s.cnt[bi]==len(s.buckets[bi]):
            off=0
            for j in s.buckets[bi]:
                n=s.params[j].numel(); s.flat[bi][off:off+n].copy_(s.params[j].grad.flatten()); off+=n
            s.comm.wait_stream(torch.cuda.current_stream())   # copy done before AR
            with torch.cuda.stream(s.comm):
                dist.all_reduce(s.flat[bi],op=dist.ReduceOp.SUM)
            e=torch.cuda.Event(); e.record(s.comm); s.ev[bi]=e
    def sync(s):
        s.comm.synchronize()
        for bi,b in enumerate(s.buckets):
            f=s.flat[bi]/s.world; off=0
            for j in b:
                n=s.params[j].numel(); s.params[j].grad.copy_(f[off:off+n].view_as(s.params[j].grad)); off+=n
        torch.cuda.current_stream().wait_stream(s.comm)
    def reset(s): s.cnt=[0]*len(s.buckets); s.ev=[None]*len(s.buckets)

def time_run(sync_mode,bucket_mb=8,steps=15):
    torch.manual_seed(1234); m=MiniGPT(1024,512,2048,4,8,256).to(dev)
    lf=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    gen=torch.Generator(device=dev).manual_seed(1234+rank*7919)
    params=[p for p in m.parameters() if p.requires_grad]
    ddp=MultiBucket(m,bucket_mb) if sync_mode!="naive" else None
    def step():
        idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
        loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        if ddp: ddp.reset()
        loss.backward()
        if sync_mode=="naive":
            for p in params:
                if p.grad is not None: dist.all_reduce(p.grad,op=dist.ReduceOp.SUM); p.grad/=world
        else: ddp.sync()
        opt.step(); return loss.item()
    for _ in range(3): step()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(steps): step()
    e.record(); e.synchronize(); dist.barrier()
    ms=s.elapsed_time(e)/steps
    w=torch.cat([p.detach().flatten().float() for p in m.parameters()]).clone()
    gw=[torch.empty_like(w) for _ in range(world)]; dist.all_gather(gw,w)
    return ms,(gw[0]-gw[1]).abs().max().item()

res=[]
tn,_=time_run("naive"); res.append(f"naive {tn:.2f}ms")
for bm in [4,8,16]:
    t,dr=time_run("bucket",bm); res.append(f"bucket{bm} {t:.2f}ms drift={dr:.1e}")
if rank==0: print(" | ".join(res))
dist.barrier(); dist.destroy_process_group()
