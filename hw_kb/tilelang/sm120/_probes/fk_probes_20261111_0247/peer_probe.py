import torch
torch.cuda.set_device(0)
print("peer 0<->1:", torch.cuda.can_device_access_peer(0,1))
# measure peer memcpy + a ring-reduce style pattern vs NCCL
import torch.distributed as dist, os
os.environ.setdefault("NCCL_P2P_LEVEL","SYS")
# Do 4 concurrent peer copies to see if aggregate exceeds 45
a0=torch.empty(64*1024*1024,device="cuda:0",dtype=torch.uint8)
b1=torch.empty_like(a0,device="cuda:1")
a0.fill_(1)
for _ in range(3): b1.copy_(a0)
torch.cuda.synchronize()
s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
for _ in range(20): b1.copy_(a0)
e.record();e.synchronize()
ms=s.elapsed_time(e)/20
print(f"peer copy 64MB: {ms:.2f}ms {64*1024*1024/(ms*1e-3)/1e9:.1f} GB/s")
