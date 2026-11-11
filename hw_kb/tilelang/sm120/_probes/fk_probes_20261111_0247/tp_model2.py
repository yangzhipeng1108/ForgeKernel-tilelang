single_tflops=155e12
for ar_gbs in [12.5,34.5]:
  print(f"\n=== all-reduce busbw = {ar_gbs} GB/s ===")
  print(f"{'shape':>20} {'1GPU ms':>9} {'TP comp':>9} {'TP comm':>9} {'TP none-ovlp':>12} {'speedup':>8} {'ovlp(max)':>10}")
  for (M,N,K) in [(4096,4096,4096),(8192,8192,8192),(4096,11008,4096),(8192,4096,65536),(16384,16384,16384),(2,11008,4096)]:
    flop=2*M*N*K; t1=flop/single_tflops*1e3
    tp_comp=t1/2
    bytes_=M*N*2
    tp_comm=bytes_/(ar_gbs*1e9)*1e3
    tot=tp_comp+tp_comm; ovlp=max(tp_comp,tp_comm)
    print(f"{M}x{N}x{K:>6} {t1:9.3f} {tp_comp:9.3f} {tp_comm:9.3f} {tot:12.3f} {t1/tot:7.2f}x {t1/ovlp:9.2f}x")