import os, math, torch, torch.distributed as dist, torch.nn as nn, time
os.environ.setdefault("NCCL_P2P_LEVEL","SYS")
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
dev=torch.device("cuda",lr)

class MiniMLP(nn.Module):
    def __init__(s,d,h):
        super().__init__(); s.fc1=nn.Linear(d,h); s.fc2=nn.Linear(h,d); s.ln=nn.LayerNorm(d)
    def forward(s,x): return x+s.fc2(torch.relu(s.fc1(s.ln(x))))

class MiniBlock(nn.Module):
    def __init__(s,d,h,nhead):
        super().__init__(); s.ln1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.proj=nn.Linear(d,d)
        s.nhead=nhead; s.ln2=nn.LayerNorm(d); s.mlp=MiniMLP(d,h)
    def forward(s,x):
        B,S,D=x.shape; h=s.ln1(x)
        qkv=s.qkv(h).view(B,S,3,s.nhead,D//s.nhead).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        a=(q@k.transpose(-2,-1))/math.sqrt(D//s.nhead); a=a.softmax(-1)
        y=(a@v).transpose(1,2).reshape(B,S,D)
        x=x+s.proj(y); return x+s.mlp(s.ln2(x))

class MiniGPT(nn.Module):
    def __init__(s,vocab,d,h,L,nhead,seq):
        super().__init__(); s.emb=nn.Embedding(vocab,d); s.pos=nn.Parameter(torch.zeros(1,seq,d))
        s.blocks=nn.ModuleList([MiniBlock(d,h,nhead) for _ in range(L)])
        s.ln=nn.LayerNorm(d); s.head=nn.Linear(d,vocab,bias=False)
    def forward(s,idx):
        x=s.emb(idx)+s.pos[:,:idx.shape[1]]
        for b in s.blocks: x=b(x)
        return s.head(s.ln(x))

class BucketDDP:
    """Bucketed gradient all-reduce overlapped with backward via
    post-accumulate-grad hooks. Each bucket launches its all_reduce on a comm
    stream the moment all its params have grads; optimizer waits on all events."""
    def __init__(self,model,bucket_mb=25):
        self.model=model; self.world=world; self.bucket_bytes=bucket_mb*1024*1024
        self.params=[p for p in model.parameters() if p.requires_grad]
        # group params into buckets by size (largest-first for better packing)
        order=sorted(range(len(self.params)), key=lambda i:-self.params[i].numel())
        self.buckets=[]; cur=[]; cur_bytes=0
        for i in order:
            nb=self.params[i].numel()*self.params[i].element_size()
            if cur and cur_bytes+nb>self.bucket_bytes:
                self.buckets.append(cur); cur=[]; cur_bytes=0
            cur.append(i); cur_bytes+=nb
        if cur: self.buckets.append(cur)
        self.ready=[0]*len(self.buckets)
        self.events=[None]*len(self.buckets)
        self.flatbufs=[torch.zeros(sum(self.params[i].numel() for i in b),device=dev,dtype=torch.float32) for b in self.buckets]
        self.comm_stream=torch.cuda.Stream(device=lr)
        self.pidx_to_bucket={}
        for bi,b in enumerate(self.buckets):
            for i in b: self.pidx_to_bucket[i]=bi
        self._register()
    def _register(self):
        for pi,p in enumerate(self.params):
            bi=self.pidx_to_bucket[pi]
            def hook(grad, pi=pi, bi=bi):
                self.ready[bi]+=1
                if self.ready[bi]==len(self.buckets[bi]):
                    # all grads of this bucket ready -> launch all_reduce on comm stream
                    flat=self.flatbufs[bi]
                    off=0
                    for i in self.buckets[bi]:
                        n=self.params[i].numel(); flat[off:off+n].copy_(self.params[i].grad.flatten()); off+=n
                    with torch.cuda.stream(self.comm_stream):
                        dist.all_reduce(flat,op=dist.ReduceOp.SUM)
                    ev=torch.cuda.Event(); ev.record(self.comm_stream); self.events[bi]=ev
            p.register_post_accumulate_grad_hook(hook)
    def sync(self):
        self.comm_stream.synchronize()
        for bi,b in enumerate(self.buckets):
            flat=self.flatbufs[bi]/self.world
            off=0
            for i in b:
                n=self.params[i].numel()
                self.params[i].grad.copy_(flat[off:off+n].view_as(self.params[i].grad))
                off+=n
        torch.cuda.current_stream().wait_stream(self.comm_stream)
    def reset(self):
        self.ready=[0]*len(self.buckets); self.events=[None]*len(self.buckets)

def run(bucket_mb, steps=15):
    torch.manual_seed(1234)
    m=MiniGPT(1024,512,2048,4,8,256).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss()
    ddp=BucketDDP(m,bucket_mb); gen=torch.Generator(device=dev).manual_seed(1234+rank*7919)
    def step():
        idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
        loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True); ddp.reset(); loss.backward(); ddp.sync(); opt.step()
        return loss.item()
    for _ in range(3): step()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(steps): step()
    e.record(); e.synchronize(); dist.barrier()
    return s.elapsed_time(e)/steps

def run_naive(steps=15):
    torch.manual_seed(1234)
    m=MiniGPT(1024,512,2048,4,8,256).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss()
    gen=torch.Generator(device=dev).manual_seed(1234+rank*7919)
    def step():
        idx=torch.randint(0,1024,(8,256),device=dev,generator=gen)
        loss=lf(m(idx)[:,:-1].reshape(-1,1024),idx[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        for p in m.parameters():
            if p.grad is not None: dist.all_reduce(p.grad,op=dist.ReduceOp.SUM); p.grad/=world
        opt.step(); return loss.item()
    for _ in range(3): step()
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for _ in range(steps): step()
    e.record(); e.synchronize(); dist.barrier()
    return s.elapsed_time(e)/steps

tn=run_naive()
res=[f"naive {tn:.2f}ms/step"]
for bm in [4,16,64]:
    t=run(bm); res.append(f"bucket+overlap {bm}MB {t:.2f}ms/step")
if rank==0: print(" | ".join(res))
dist.barrier(); dist.destroy_process_group()
