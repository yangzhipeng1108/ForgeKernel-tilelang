# TP break-even model using MEASURED numbers
# GEMM: MxNxK on one GPU = 2MNK flops / 155e12 flops/s
# TP: split K across 2 ranks -> each does MxNx(K/2); partial sums all-reduced (MxN fp32? use fp16)
# comm: allreduce of MxN fp16 @ 12.5 GB/s algo (use measured 32MB-class rate ~12.6)
single_tflops=155e12
allreduce_gbs=12.5e9
print(f"{'shape':>22} {'1-GPU ms':>10} {'TP compute ms':>14} {'TP comm ms':>12} {'TP total ms':>12} {'speedup':>8}")
for (M,N,K) in [(4096,4096,4096),(8192,8192,8192),(4096,11008,4096),(1,4096,4096),(8192,4096,4096*4),(16384,16384,16384)]:
    flop=2*M*N*K
    t1=flop/single_tflops*1e3
    # TP over K: each rank half the K work -> compute halves
    tp_comp = t1/2
    # comm: allreduce MxN fp16 (2 bytes) with ring busbw; algorithmic payload per rank = M*N*2
    bytes_=M*N*2
    tp_comm = bytes_/allreduce_gbs*1e3  # but allreduce ring time ~= 2*(w-1)/w * payload/busbw; at w=2 that's 1x payload
    tp_total = tp_comp + tp_comm
    sp = t1/tp_total
    print(f"{M}x{N}x{K:>6} {t1:10.3f} {tp_comp:14.3f} {tp_comm:12.3f} {tp_total:12.3f} {sp:8.2f}x")