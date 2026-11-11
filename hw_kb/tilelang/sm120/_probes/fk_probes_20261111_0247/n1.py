import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
# RMSNorm: y = x * rsqrt(mean(x^2)+eps) * w, over last dim D
def rmsnorm(M,D,BM,threads,dt="float16"):
    @T.prim_func
    def main(X:T.Tensor((M,D),dt),W:T.Tensor((D,),dt),Y:T.Tensor((M,D),dt)):
        with T.Kernel(M//BM,threads=threads) as bx:
            Xs=T.alloc_fragment((BM,D),"float32")
            ss=T.alloc_fragment((BM,),"float32")
            T.copy(X[bx*BM,0],Xs)
            T.reduce_sum(Xs*Xs,ss,dim=1)  # placeholder; compute manually below
            for i,j in T.Parallel(BM,D):
                Y[bx*BM+i,j]=(Xs[i,j]*T.rsqrt(ss[i]/D+1e-6))*T.cast(W[j],dt)
    return main
# simpler explicit version
def rmsnorm2(M,D,BM,threads,dt="float16"):
    @T.prim_func
    def main(X:T.Tensor((M,D),dt),W:T.Tensor((D,),dt),Y:T.Tensor((M,D),dt)):
        with T.Kernel(T.ceildiv(M,BM),threads=threads) as bx:
            Xs=T.alloc_fragment((BM,D),"float32")
            sq=T.alloc_fragment((BM,D),"float32")
            ss=T.alloc_fragment((BM,),"float32")
            T.copy(X[bx*BM,0],Xs)
            for i,j in T.Parallel(BM,D): sq[i,j]=Xs[i,j]*Xs[i,j]
            T.reduce_sum(sq,ss,dim=1)
            for i,j in T.Parallel(BM,D): Y[bx*BM+i,j]=T.cast(Xs[i,j]*T.rsqrt(ss[i]/D+1e-6)*T.cast(W[j],"float32"),dt)
    return main
torch.manual_seed(0)
M,D=8192,4096  # typical: batch*seq rows, hidden 4096
x=torch.randn(M,D,device="cuda",dtype=torch.float16)
w=torch.randn(D,device="cuda",dtype=torch.float16)
xv=x.float()
ref=(xv*torch.rsqrt(xv.pow(2).mean(-1,keepdim=True)+1e-6)*w.float()).to(torch.float16)
BYTES=M*D*2*2 + D*2  # read X + write Y
print(f"RMSNorm M{M} D{D}: bytes={BYTES/1e6:.0f}MB")
# torch baseline
def torch_rms():
    xf=x.float(); return (xf*torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+1e-6)*w.float()).to(torch.float16)
tt=do_bench(torch_rms,warmup=20,rep=50)
print(f"torch: {tt*1000:.1f}us {BYTES/(tt*1e-3)/1e9:.0f}GB/s")
for BM in (1,2,4,8,16,32):
  for th in (128,256,512):
    try:
        k=tilelang.compile(rmsnorm2(M,D,BM,th),out_idx=[2],target="cuda")
        y=k(x,w); torch.cuda.synchronize()
        cos=torch.nn.functional.cosine_similarity(y.float().flatten(),ref.float().flatten(),dim=0).item()
        t=do_bench(lambda: k(x,w),warmup=20,rep=50)
        print(f"  BM{BM} th{th}: {t*1000:.1f}us {BYTES/(t*1e-3)/1e9:.0f}GB/s cos={cos:.5f} x{tt/t:.2f}vs_torch")
    except Exception as e: print(f"  BM{BM} th{th}: FAIL {str(e)[:60]}")