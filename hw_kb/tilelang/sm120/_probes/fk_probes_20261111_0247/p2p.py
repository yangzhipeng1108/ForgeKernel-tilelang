import torch
print("torch cuda available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} {p.total_memory/1e9:.1f}GB SMs={p.multi_processor_count}")
if torch.cuda.device_count()>=2:
    print("can peer 0->1:", torch.cuda.can_device_access_peer(0,1))
    print("can peer 1->0:", torch.cuda.can_device_access_peer(1,0))
    # measure P2P / inter-GPU copy bandwidth both ways
    from tilelang.profiler import do_bench
    for src,dst in [(0,1),(1,0)]:
        n=64*1024*1024//2  # 64MB fp16... actually elements
        n=32*1024*1024  # 32M fp32 elems = 128MB? use fp16
        n=32*1024*1024
        a=torch.empty(n,device=f"cuda:{src}",dtype=torch.float16)
        b=torch.empty(n,device=f"cuda:{dst}",dtype=torch.float16)
        def cp(): b.copy_(a)
        cp(); torch.cuda.synchronize()
        t=do_bench(cp,warmup=10,rep=30)
        bytes_=n*2
        print(f"copy GPU{src}->GPU{dst}: {t*1000:.1f}us {bytes_/(t*1e-3)/1e9:.1f}GB/s each-way ({2*bytes_/(t*1e-3)/1e9:.1f} aggregate)")
    # intra-GPU (baseline) for reference
    a=torch.empty(32*1024*1024,device="cuda:0",dtype=torch.float16)
    b=torch.empty_like(a,device="cuda:0")
    t=do_bench(lambda:b.copy_(a),warmup=10,rep=30)
    print(f"intra GPU0 copy: {32*1024*1024*2/(t*1e-3)/1e9:.0f}GB/s")