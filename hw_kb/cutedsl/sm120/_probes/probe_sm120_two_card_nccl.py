"""Two-card collective bandwidth on the RTX 6000D pair (SYS link, no NVLink).

This is the number every tensor-parallel / expert-parallel design on this box pays on each
token boundary. It exists because the first version of this probe produced impossible numbers
(12 TB/s all-reduce) from two separate mistakes, both fixed here:

  1. **Unit bug.** bytes / (ms * 1e-3) / 1e6 is GB/s only if you want 1e6; it is actually
     1e-3 GB per MB, so dividing by 1e6 gave TB/s-scale garbage. Correct factor is 1e9.
  2. **NCCL silently chose Socket.** With no NVLink and NCCL_P2P left to its own devices the
     log says "NVLS multicast support is not available" and routes through NET/Socket over the
     container's 192.168.x interface. That is a ~12 GB/s TCP path, not the PCIe P2P link.

Here we force P2P and report both the raw link bandwidth and a corrected busBW.

Launch with torchrun, one process per GPU:
    torchrun --nproc_per_node=2 probe_sm120_two_card_nccl.py
"""
import os
import sys

import torch
import torch.distributed as dist

# Force the PCIe P2P path. NCCL defaults to SHM/Socket when it cannot see NVLink unless told
# the peer link is usable; P2P_LEVEL=SYS matches what nvidia-smi topo reports for this pair.
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "0")

GB = 1e9


def bus_bw(nbytes, ms, world):
    """NCCL convention: busBW = bytes / time, scaled by 2*(world-1)/world for allreduce."""
    return nbytes * 2 * (world - 1) / world / (ms * 1e-3) / GB


def bench_allreduce(dtype, nbytes, iters=20, warmup=5):
    rank, world = dist.get_rank(), dist.get_world_size()
    n = nbytes // torch.tensor([], dtype=dtype).element_size()
    x = torch.ones(n, dtype=dtype, device="cuda")
    for _ in range(warmup):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        dist.all_reduce(x)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    raw = nbytes / (ms * 1e-3) / GB
    if rank == 0:
        print(f"  all_reduce {str(dtype).split('.')[-1]:9s} {nbytes >> 20:4d} MiB: "
              f"{ms:8.4f} ms  raw={raw:7.2f} GB/s  busBW={bus_bw(nbytes, ms, world):7.2f} GB/s",
              flush=True)


def bench_p2p(dtype, nbytes, iters=20):
    """One-way send 0 -> 1 using NCCL point-to-point (batched via isend/irecv)."""
    rank, world = dist.get_rank(), dist.get_world_size()
    n = nbytes // torch.tensor([], dtype=dtype).element_size()
    buf = torch.ones(n, dtype=dtype, device="cuda")

    for _ in range(3):                                  # warm
        if rank == 0:
            dist.send(buf, dst=1)
        else:
            dist.recv(buf, src=0)
    torch.cuda.synchronize()

    if rank == 0:
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            dist.send(buf, dst=1)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        print(f"  send 0->1  {str(dtype).split('.')[-1]:9s} {nbytes >> 20:4d} MiB: "
              f"{ms:8.4f} ms  {nbytes / (ms * 1e-3) / GB:7.2f} GB/s", flush=True)
    else:
        for _ in range(iters):
            dist.recv(buf, src=0)


if __name__ == "__main__":
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()

    dist.barrier()
    print(f"rank {rank}: local {local} -> {torch.cuda.get_device_name(local)}  "
          f"world={world} nccl={torch.cuda.nccl.version()}", flush=True)
    p = torch.cuda.get_device_properties(local)
    if rank == 0:
        print(f"PCIe gen/width: {p.pci_bus_id if hasattr(p, 'pci_bus_id') else 'n/a'}", flush=True)
    dist.barrier()

    if rank == 0:
        print("=== all_reduce fp32 (P2P forced) ===", flush=True)
    for mb in (1, 4, 16, 64, 256):
        bench_allreduce(torch.float32, mb << 20)
    dist.barrier()

    if rank == 0:
        print("=== all_reduce bf16 ===", flush=True)
    for mb in (16, 64, 256):
        bench_allreduce(torch.bfloat16, mb << 20)
    dist.barrier()

    if rank == 0:
        print("=== p2p send fp32 ===", flush=True)
    for mb in (16, 64, 256):
        bench_p2p(torch.float32, mb << 20)

    dist.barrier()
    if rank == 0:
        print("NCCL TWO CARD DONE", flush=True)
    dist.destroy_process_group()
