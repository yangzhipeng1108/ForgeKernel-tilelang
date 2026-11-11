import torch, tilelang, tilelang.language as T
from tilelang.profiler import do_bench
# 1) peak HBM bandwidth via big copy
n=1<<28  # 256M elems fp16 = 512MB
a=torch.randn(n,device="cuda",dtype=torch.float16); b=torch.empty_like(a)
t=do_bench(lambda: b.copy_(a),warmup=20,rep=50)
print(f"copy 2x{n*2/1e6:.0f}MB: {t*1000:.1f}us -> {2*n*2/(t*1e-3)/1e9:.0f} GB/s (read+write)")
# 2) pure read (sum)
t2=do_bench(lambda: a.sum(),warmup=20,rep=50)
print(f"read {n*2/1e6:.0f}MB sum: {t2*1000:.1f}us -> {n*2/(t2*1e-3)/1e9:.0f} GB/s")
# 3) torch profiler dev props
p=torch.cuda.get_device_properties(0)
print(f"device={p.name} SMs={p.multi_processor_count} mem={p.total_memory/1e9:.1f}GB")