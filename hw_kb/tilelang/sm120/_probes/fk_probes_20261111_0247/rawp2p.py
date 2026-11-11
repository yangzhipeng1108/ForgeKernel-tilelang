import torch
torch.cuda.set_device(0)
n=256*1024*1024  # 256MB
a=torch.empty(n,device="cuda:0",dtype=torch.uint8); a.fill_(1)
b=torch.empty_like(a,device="cuda:1")
for _ in range(3): b.copy_(a)
torch.cuda.synchronize()
s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
for _ in range(20): b.copy_(a)
e.record(); e.synchronize()
ms=s.elapsed_time(e)/20
print(f"raw peer copy 256MB: {ms:.2f}ms  {n/(ms*1e-3)/1e9:.1f} GB/s one-way")
