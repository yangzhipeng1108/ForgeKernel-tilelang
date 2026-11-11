import os, math, torch, torch.distributed as dist, torch.nn as nn
os.environ.setdefault("NCCL_P2P_LEVEL","SYS"); os.environ.setdefault("NCCL_ALGO","Tree")
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
dev=torch.device("cuda",lr)
src=open("/tmp/ddp_overlap.py").read().split("class BucketDDP")[0]
head=src.split('lr=int(os.environ["LOCAL_RANK"])')[1]
head='import os, math, torch, torch.distributed as dist, torch.nn as nn'+head
head=head.replace('dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()','rank=dist.get_rank(); world=dist.get_world_size()')
exec(head)

def build():
    torch.manual_seed(1234)
    return MiniGPT(1024,512,2048,4,8,256).to(dev)

# Baseline A: naive per-param all_reduce, single batch (no overlap)
def bench_naive(steps=12):
    m=build(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss()
    gen=torch.Generator(device=dev).manual_seed(1234+rank*7919)
    P=[p for p in m.parameters() if p.requires_grad]
    def step():
        idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
        loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        for p in P:
            if p.grad is not None: dist.all_reduce(p.grad,op=dist.ReduceOp.SUM); p.grad/=world
        opt.step()
    for _ in range(3): step()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(steps): step()
    e.record();e.synchronize(); dist.barrier()
    return s.elapsed_time(e)/steps

# Baseline B: bucketed overlap, single batch (from earlier)
class Bucket:
    def __init__(s,m,bm=8):
        s.p=[p for p in m.parameters() if p.requires_grad]
        order=sorted(range(len(s.p)),key=lambda i:-s.p[i].numel())
        s.bks=[];cur=[];cb=0
        for i in order:
            nb=s.p[i].numel()*s.p[i].element_size()
            if cur and cb+nb>bm*1024*1024: s.bks.append(cur);cur=[];cb=0
            cur.append(i);cb+=nb
        if cur:s.bks.append(cur)
        s.p2b={i:bi for bi,b in enumerate(s.bks) for i in b}
        s.cnt=[0]*len(s.bks)
        s.flat=[torch.zeros(sum(s.p[i].numel() for i in b),device=dev,dtype=torch.float32) for b in s.bks]
        s.comm=torch.cuda.Stream(device=lr)
        for i,p in enumerate(s.p): p.register_post_accumulate_grad_hook(lambda g,i=i:s._h(i))
    def _h(s,i):
        bi=s.p2b[i]; s.cnt[bi]+=1
        if s.cnt[bi]==len(s.bks[bi]):
            off=0
            for j in s.bks[bi]:
                n=s.p[j].numel(); s.flat[bi][off:off+n].copy_(s.p[j].grad.flatten()); off+=n
            s.comm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s.comm): dist.all_reduce(s.flat[bi],op=dist.ReduceOp.SUM)
            e=torch.cuda.Event(); e.record(s.comm); 
    def sync(s):
        s.comm.synchronize()
        for bi,b in enumerate(s.bks):
            f=s.flat[bi]/world; off=0
            for j in b:
                n=s.p[j].numel(); s.p[j].grad.copy_(f[off:off+n].view_as(s.p[j].grad)); off+=n
        torch.cuda.current_stream().wait_stream(s.comm)
    def reset(s): s.cnt=[0]*len(s.bks)

def bench_bucket(steps=12):
    m=build(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss()
    gen=torch.Generator(device=dev).manual_seed(1234+rank*7919); d=Bucket(m)
    def step():
        idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
        loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True); d.reset(); loss.backward(); d.sync(); opt.step()
    for _ in range(3): step()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(steps): step()
    e.record();e.synchronize(); dist.barrier()
    return s.elapsed_time(e)/steps

tn=bench_naive(); tb=bench_bucket()
# Correctness: naive and bucket must give same loss trajectory (same seed/data)
def loss_traj(kind, n=3):
    m=build(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss()
    gen=torch.Generator(device=dev).manual_seed(1234+rank*7919)
    P=[p for p in m.parameters() if p.requires_grad]; d=Bucket(m) if kind=="b" else None
    out=[]
    for _ in range(n):
        idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
        loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        if d: d.reset()
        loss.backward()
        if d: d.sync()
        else:
            for p in P:
                if p.grad is not None: dist.all_reduce(p.grad,op=dist.ReduceOp.SUM); p.grad/=world
        opt.step(); out.append(round(loss.item(),6))
    return out
ln=loss_traj("n"); lb=loss_traj("b")
if rank==0:
    print(f"naive {tn:.2f}ms | bucket {tb:.2f}ms | loss naive={ln} bucket={lb} match={ln==lb}")
dist.barrier(); dist.destroy_process_group()
