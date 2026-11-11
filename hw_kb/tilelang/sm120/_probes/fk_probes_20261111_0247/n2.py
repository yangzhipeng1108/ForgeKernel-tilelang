import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
# RMSNorm manual-copy variant to test vectorization (contiguous 128-bit loads via T.copy)
def rmsnorm_vec(M,D,BM,threads,dt="float16"):
    @T.prim_func
    def main(X:T.Tensor((M,D),dt),W:T.Tensor((D,),dt),Y:T.Tensor((M,D),dt)):
        with T.Kernel(T.ceildiv(M,BM),threads=threads) as bx:
            Xs=T.alloc_fragment((BM,D),"float32")
            sq=T.alloc_fragment((BM,D),"float32"); ss=T.alloc_fragment((BM,),"float32")
            T.copy(X[bx*BM,0],Xs)
            for i,j in T.Parallel(BM,D): sq[i,j]=Xs[i,j]*Xs[i,j]
            T.reduce_sum(sq,ss,dim=1)
            for i,j in T.Parallel(BM,D): Y[bx*BM+i,j]=T.cast(Xs[i,j]*T.rsqrt(ss[i]/D+1e-6)*T.cast(W[j],"float32"),dt)
    return main
torch.manual_seed(0)
M,D=8192,4096
x=torch.randn(M,D,device="cuda",dtype=torch.float16); w=torch.randn(D,device="cuda",dtype=torch.float16)
xv=x.float()
ref=(xv*torch.rsqrt(xv.pow(2).mean(-1,keepdim=True)+1e-6)*w.float()).to(torch.float16)
BYTES=M*D*2*2+D*2
# fine BM scan around the winner, more threads
print("=== fine sweep (vectorized copy) ===")
best=None
for BM in (1,2,4):
  for th in (64,128,256,512,1024):
    try:
        k=tilelang.compile(rmsnorm_vec(M,D,BM,th),out_idx=[2],target="cuda")
        y=k(x,w); torch.cuda.synchronize()
        cos=torch.nn.functional.cosine_similarity(y.float().flatten(),ref.float().flatten(),dim=0).item()
        t=do_bench(lambda: k(x,w),warmup=20,rep=50)
        bw=BYTES/(t*1e-3)/1e9
        mark=" *" if (best is None or t<best[0]) else ""
        if best is None or t<best[0]: best=(t,BM,th,bw)
        print(f"  BM{BM} th{th}: {t*1000:.1f}us {bw:.0f}GB/s cos={cos:.5f}{mark}")
    except Exception as e: print(f"  BM{BM} th{th}: FAIL {str(e)[:50]}")
print("BEST:",best)
# pure copy roofline for reference
def purecopy(M,D,BM,threads):
    @T.prim_func
    def main(X:T.Tensor((M,D),"float16"),Y:T.Tensor((M,D),"float16")):
        with T.Kernel(T.ceildiv(M,BM),threads=threads) as bx:
            Xs=T.alloc_fragment((BM,D),"float16")
            T.copy(X[bx*BM,0],Xs)
            T.copy(Xs,Y[bx*BM,0])
    return main
k=tilelang.compile(purecopy(M,D,1,128),out_idx=[1],target="cuda")
k(x); torch.cuda.synchronize()
t=do_bench(lambda: k(x),warmup=20,rep=50)
print(f"pure copy f16 (read+write {M*D*2*2/1e6:.0f}MB): {t*1000:.1f}us {M*D*2*2/(t*1e-3)/1e9:.0f}GB/s")