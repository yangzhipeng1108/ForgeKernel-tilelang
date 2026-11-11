"""What a cluster can actually do on a 2x RTX 6000D (SYS link, no NVLink).

The census proves the cluster primitives *exist*. This checks whether a cluster launch works
at all, how big it can be, and whether multicast reaches the second GPU -- the three questions
that decide if any cross-card DSMEM strategy is even on the table.

Launch syntax is `.launch(grid=..., block=..., cluster=..., smem=..., stream=...)`; the
`k[grid, cluster, block](...)` form from sm90 docs does not exist in this cutlass.

Run in the sglang container (cutlass 4.8.0):
    python3 probe_sm120_cluster_reality.py
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda

F32 = cutlass.Float32
rows = []


def probe(name, fn):
    try:
        rows.append((name, "OK", str(fn())))
    except Exception as e:
        rows.append((name, "REFUSED", f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"))


def try_cluster(cm, cn):
    """Boot an (cm,cn) cluster; every CTA touches the barrier so a broken cluster faults here."""

    @cute.kernel
    def k(gOut):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            gOut[0] = cute.arch.cluster_size()
            gOut[1] = cute.arch.cluster_idx()[0]
            gOut[2] = cute.arch.cluster_idx()[1]
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def host(gOut, stream):
        k(gOut).launch(grid=[cm, cn, 1], block=[128, 1, 1],
                       cluster=[cm, cn, 1] if (cm * cn) > 1 else None,
                       smem=0, stream=stream)

    O = torch.full((8,), -1.0, device="cuda", dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    host(make_ptr(F32, O.data_ptr(), cute.AddressSpace.gmem, assumed_align=4), stream)
    torch.cuda.synchronize()
    return f"booted, cluster_size={int(O[0].item())} (want {cm*cn})"


for cm, cn in ((1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (8, 1), (16, 1)):
    probe(f"cluster ({cm},{cn})", lambda cm=cm, cn=cn: try_cluster(cm, cn))


def _attr():
    p = torch.cuda.get_device_properties(0)
    out = {}
    for nm in ("max_blocks_per_cluster", "max_threads_per_block",
               "shared_memory_per_block_optin", "multi_processor_count", "l2_cache_size"):
        if hasattr(p, nm):
            out[nm] = getattr(p, nm)
    return out


probe("device properties", _attr)


def _peer_matrix():
    n = torch.cuda.device_count()
    return [[("self" if i == j else ("P2P" if torch.cuda.can_device_access_peer(i, j) else "no"))
             for j in range(n)] for i in range(n)]


probe("peer matrix", _peer_matrix)


def _peer_copy_bw():
    if not (torch.cuda.device_count() > 1 and torch.cuda.can_device_access_peer(0, 1)):
        raise RuntimeError("no peer access")
    nbytes = 256 * 1024 * 1024
    src = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda:0")
    with torch.cuda.device(1):
        dst = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda:1")
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        dst.copy_(src, non_blocking=True)
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return f"256 MiB copy_ {best:.2f} ms -> {nbytes / best / 1e6:.1f} GB/s (SYS)"


probe("cross-card copy bandwidth", _peer_copy_bw)


print("=" * 100, flush=True)
print(f"{'probe':44s} {'state':9s} detail", flush=True)
print("-" * 100, flush=True)
for n, s, d in rows:
    print(f"{n:44s} {s:9s} {d}", flush=True)
print("\nCLUSTER REALITY DONE", flush=True)
