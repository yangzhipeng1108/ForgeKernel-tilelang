import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
def rope(M,D,BM,threads,use_swz=None):
    H=D//2
    @T.prim_func
    def main(X:T.Tensor((M,D),"float16"),COS:T.Tensor((M,H),"float16"),SIN:T.Tensor((M,H),"float16"),Y:T.Tensor((M,D),"float16")):
        with T.Kernel(T.ceildiv(M,BM),threads=threads) as bx:
            if use_swz: T.use_swizzle(panel_size=use_swz)
            Xs=T.alloc_fragment((BM,D),"float32"); Cs=T.alloc_fragment((BM,H),"float32"); Ss=T.alloc_fragment((BM,H),"float32")
            E=T.alloc_fragment((BM,D),"float32"); O=T.alloc_fragment((BM,D),"float32")
            T.copy(X[bx*BM,0],Xs); T.copy(COS[bx*BM,0],Cs); T.copy(SIN[bx*BM,0],Ss)
            for i,j in T.Parallel(BM,D):
                if j%2==0: E[i,j]=Xs[i,j]; O[i,j]=Xs[i,j+1]
                else: E[i,j]=Xs[i,j-1]; O[i,j]=Xs[i,j]
            for i,j in T.Parallel(BM,D):
                hj=j//2
                if j%2==0: Y[bx*BM+i,j]=T.cast(E[i,j]*Cs[i,hj]-O[i,j]*Ss[i,hj],"float16")
                else: Y[bx*BM+i,j]=T.cast(E[i,j]*Cs[i,hj]+O[i,j]*Ss[i,hj],"float16")
    return main
# simpler: separate cos/sin application via two passes avoiding e/o confusion
def rope2(M,D,BM,threads,use_swz=None):
    H=D//2
    @T.prim_func
    def main(X:T.Tensor((M,D),"float16"),COS:T.Tensor((M,H),"float16"),SIN:T.Tensor((M,H),"float16"),Y:T.Tensor((M,D),"float16")):
        with T.Kernel(T.ceildiv(M,BM),threads=threads) as bx:
            if use_swz: T.use_swizzle(panel_size=use_swz)
            Xs=T.alloc_fragment((BM,D),"float32"); Cs=T.alloc_fragment((BM,H),"float32"); Ss=T.alloc_fragment((BM,H),"float32")
            rev=T.alloc_fragment((BM,D),"float32")
            Xf=T.alloc_fragment((BM,D),"float32")
            T.copy(X[bx*BM,0],Xs); T.copy(COS[bx*BM,0],Cs); T.copy(SIN[bx*BM,0],Ss)
            # rotate_half: swap adjacent pairs with sign flip
            for i,j in T.Parallel(BM,D):
                if j%2==0: rev[i,j]=-Xs[i,j+1]
                else: rev[i,j]=Xs[i,j-1]
            # broadcast cos/sin: out = x*cos[j//2] + rev*sin[j//2]
            for i,j in T.Parallel(BM,D):
                hj=j//2
                Y[bx*BM+i,j]=T.cast(Xs[i,j]*Cs[i,hj]+rev[i,j]*Ss[i,hj],"float16")
    return main
torch.manual_seed(0)
M,D=8192,4096; H=D//2
x=torch.randn(M,D,device="cuda",dtype=torch.float16)
cos=torch.randn(M,H,device="cuda",dtype=torch.float16)*0.5; sin=torch.randn(M,H,device="cuda",dtype=torch.float16)*0.5
xv=x.float(); c=cos.float(); s=sin.float()
x0=xv[:,0::2]; x1=xv[:,1::2]
ref=torch.empty(M,D,device="cuda",dtype=torch.float16)
ref[:,0::2]=(x0*c-x1*s).to(torch.float16); ref[:,1::2]=(x1*c+x0*s).to(torch.float16)
BYTES=M*D*2*2 + M*H*2*2
print(f"RoPE M{M} D{D}: bytes={BYTES/1e6:.0f}MB")
best=None
for BM in (1,2,4,8):
  for th in (128,256):
    try:
      k=tilelang.compile(rope2(M,D,BM,th),out_idx=[3],target="cuda")
      y=k(x,cos,sin); torch.cuda.synchronize()
      cs=torch.nn.functional.cosine_similarity(y.float().flatten(),ref.float().flatten(),dim=0).item()
      t=do_bench(lambda: k(x,cos,sin),warmup=20,rep=50)
      bw=BYTES/(t*1e-3)/1e9
      if best is None or t<best[0]: best=(t,BM,th,bw,cs)
      print(f"  BM{BM} th{th}: {t*1000:.1f}us {bw:.0f}GB/s cos={cs:.5f}")
    except Exception as e: print(f"  BM{BM} th{th}: FAIL {str(e)[:60]}")
print("BEST rope:",best)
print("=== swizzle stress on memory-bound RoPE ===")
for swz in (None,4,8,16):
  try:
    k=tilelang.compile(rope2(M,D,4,128,swz),out_idx=[3],target="cuda")
    k(x,cos,sin); torch.cuda.synchronize()
    t=do_bench(lambda: k(x,cos,sin),warmup=20,rep=50)
    print(f"  swz={swz}: {t*1000:.1f}us {BYTES/(t*1e-3)/1e9:.0f}GB/s")
  except Exception as e: print(f"  swz={swz}: FAIL {str(e)[:50]}")