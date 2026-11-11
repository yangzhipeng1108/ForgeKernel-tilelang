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
class Bucket:
    def __init__(s,m,bm):
        s.p=[p for p in m.parameters() if p.requires_grad]
        # order by REVERSED parameter order (deepest layers first) so buckets match
        # the order grads become ready during backward -> earlier comm start
        order=list(range(len(s.p)-1,-1,-1))
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
            e=torch.cuda.Event(); e.record(s.comm)
    def sync(s):
        s.comm.synchronize()
        for bi,b in enumerate(s.bks):
            f=s.flat[bi]/world; off=0
            for j in b:
                n=s.p[j].numel(); s.p[j].grad.copy_(f[off:off+n].view_as(s.p[j].grad)); off+=n
        torch.cuda.current_stream().wait_stream(s.comm)
    def reset(s): s.cnt=[0]*len(s.bks)
def bench(bm,steps=12):
    torch.manual_seed(1234); m=MiniGPT(1024,512,2048,4,8,256).to(dev)
    lf=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    gen=torch.Generator(device=dev).manual_seed(1234+rank*7919); d=Bucket(m,bm)
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
r=[]
for bm in [1,2,4,8,16]:
    r.append(f"bm{bm}MB {bench(bm):.2f}ms")
if rank==0: print(" | ".join(r))
dist.barrier(); dist.destroy_process_group()
