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
# What if the real overlap target is: while the all-reduce of step t runs on the
# comm stream, start the FORWARD of step t+1 on the compute stream. But forward
# needs post-optimizer weights (depends on sync). So true cross-step overlap needs
# stale gradients (1-step-delayed update) -- a known technique (LocalSGD / stale).
# Implement 1-step-delayed optimizer: forward t+1 uses weights AFTER step t's
# update, but we run it while step t's allreduce is still finishing? Can't: update
# needs allreduce. So instead measure: forward of t+1 (with weights from t) can
# overlap the allreduce of t IF we update optimizer-only params on the fly...
# Bottom line to test: does issuing next forward while comm runs give anything?
# Simple test: pipeline is impossible without stale grads. Confirm by measuring
# theoretical: comm time / step time.
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
        s.comm=torch.cuda.Stream(device=lr); s.bytes=sum(f.numel()*f.element_size() for f in s.flat)
        for i,p in enumerate(s.p): p.register_post_accumulate_grad_hook(lambda g,i=i:s._h(i))
    def _h(s,i):
        bi=s.p2b[i]; s.cnt[bi]+=1
        if s.cnt[bi]==len(s.bks[bi]):
            off=0
            for j in s.bks[bi]:
                n=s.p[j].numel(); s.flat[bi][off:off+n].copy_(s.p[j].grad.flatten()); off+=n
            s.comm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s.comm): dist.all_reduce(s.flat[bi],op=dist.ReduceOp.SUM)
            torch.cuda.Event().record(s.comm)
    def sync(s):
        s.comm.synchronize()
        for bi,b in enumerate(s.bks):
            f=s.flat[bi]/world; off=0
            for j in b:
                n=s.p[j].numel(); s.p[j].grad.copy_(f[off:off+n].view_as(s.p[j].grad)); off+=n
        torch.cuda.current_stream().wait_stream(s.comm)
    def reset(s): s.cnt=[0]*len(s.bks)

torch.manual_seed(1234); m=MiniGPT(1024,512,2048,4,8,256).to(dev)
lf=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
gen=torch.Generator(device=dev).manual_seed(1234+rank*7919); d=Bucket(m)
P=[p for p in m.parameters() if p.requires_grad]
# measure fwd+bwd time alone (no comm) and comm time alone
def t_fwdbwd(it=20):
    for _ in range(3):
        idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
        loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward()
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(it):
        idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
        loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward()
    e.record();e.synchronize(); return s.elapsed_time(e)/it
# generic comm-only: 35MB allreduce-ish flat
def t_comm(it=20):
    tot=d.bytes; flat=torch.ones(tot//4,device=dev,dtype=torch.float32)
    for _ in range(3): dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(it): dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    e.record();e.synchronize(); return s.elapsed_time(e)/it
tf=t_fwdbwd(); tc=t_comm()
if rank==0:
    print(f"fwd+bwd {tf:.2f}ms | comm(allreduce {d.bytes/1e6:.1f}MB) {tc:.3f}ms | comm/fwd+bwd = {tc/tf*100:.1f}%  => perfect-overlap best-case saves {tc:.2f}ms")
dist.barrier(); dist.destroy_process_group()
