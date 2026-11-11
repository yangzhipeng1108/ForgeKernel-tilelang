import os, torch, torch.distributed as dist, ctypes
os.environ.setdefault("NCCL_P2P_LEVEL","SYS")
import multiprocessing as mp
def worker(rank):
    torch.cuda.set_device(rank)
    # RAW peer copy via cudaMemcpyPeer (no NCCL, no host staging) - the true P2P ceiling
    a=torch.empty(64*1024*1024,device=f"cuda:{rank}",dtype=torch.uint8)  # 64MB
    b=torch.empty_like(a,device=f"cuda:{(rank+1)%2}")
    a.fill_(1)
    torch.cuda.synchronize()
    # use torch's copy_ with peer enabled; it may use cudaMemcpyPeer under the hood
    torch.cuda.synchronize()
    s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    # warmup
    for _ in range(3): b.copy_(a)
    torch.cuda.synchronize()
    s.record()
    N=20
    for _ in range(N): b.copy_(a)
    e.record(); e.synchronize()
    ms=s.elapsed_time(e)/N
    if rank==0:
        print(f"raw peer copy_ 64MB: {ms:.2f}ms  {64*1024*1024/(ms*1e-3)/1e9:.1f} GB/s one-way")
    torch.distributed.destroy_process_group() if dist.is_initialized() else None
if __name__=="__main__":
    os.environ["MASTER_ADDR"]="127.0.0.1"; os.environ["MASTER_PORT"]="29671"
    dist.init_process_group("nccl",rank=int(os.environ["RANK"]),world_size=2)
    worker(int(os.environ["RANK"]))
