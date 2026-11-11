import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
# RoPE via two separate half-dim buffers: X0=(M,H) even, X1=(M,H) odd (pre-split on host)
def rope_split(M,H,BM,threads,use_swz=None):
    @T.prim_func
    def main(X0:T.Tensor((M,H),"float16"),X1:T.Tensor((M,H),"float16"),COS:T.Tensor((M,H),"float16"),SIN:T.Tensor((M,H),"float16"),Y0:T.Tensor((M,H),"float16"),Y1:T.Tensor((M,H),"float16")):
        with T.Kernel(T.ceildiv(M,BM),threads=threads) as bx:
            if use_swz: T.use_swizzle(panel_size=use_swz)
            a=T.alloc_fragment((BM,H),"float32"); b=T.alloc_fragment((BM,H),"float32")
            c=T.alloc_fragment((BM,H),"float32"); s=T.alloc_fragment((BM,H),"float32")
            T.copy(X0[bx*BM,0],a); T.copy(X1[bx*BM,0],b); T.copy(COS[bx*BM,0],c); T.copy(SIN[bx*BM,0],s)
            for i,j in T.Parallel(BM,H):
                Y0[bx*BM+i,j]=T.cast(a[i,j]*c[i,j]-b[i,j]*s[i,j],"float16")
            for i,j in T.Parallel(BM,H):
                Y1[bx*BM+i,j]=T.cast(b[i,j]*c[i,j]+a[i,j]*s[i,j],"float16")
    return main
torch.manual_seed(0)
M,D=8192,4096; H=D//2
x=torch.randn(M,D,device="cuda",dtype=torch.float16)
cos=torch.randn(M,H,device="cuda",dtype=torch.float16)*0.5; sin=torch.randn(M,H,device="cuda",dtype=torch.float16)*0.5
x0=x[:,0::2].contiguous(); x1=x[:,1::2].contiguous()
xv=x.float(); c=cos.float(); s=sin.float()
r0=(xv[:,0::2]*c-xv[:,1::2]*s).to(torch.float16); r1=(xv[:,1::2]*c+xv[:,0::2]*s).to(torch.float16)
BYTES=M*D*2*2 + M*H*2*2
print(f"RoPE(split) M{M} D{D}: bytes={BYTES/1e6:.0f}MB")
best=None
for BM in (1,2,4,8):
  for th in (128,256):
    try:
      k=tilelang.compile(rope_split(M,H,BM,th),out_idx=[4,5],target="cuda")
      y0,y1=k(x0,x1,cos,sin); torch.cuda.synchronize()
      cs=(torch.nn.functional.cosine_similarity(y0.float().flatten(),r0.flatten(),dim=0).item()+torch.nn.functional.cosine_similarity(y1.float().flatten(),r1.flatten(),dim=0).item())/2
      t=do_bench(lambda: k(x0,x1,cos,sin),warmup=20,rep=50)
      bw=BYTES/(t*1e-3)/1e9
      if best is None or t<best[0]: best=(t,BM,th,bw,cs)
      print(f"  BM{BM} th{th}: {t*1000:.1f}us {bw:.0f}GB/s cos={cs:.5f}")
    except Exception as e: print(f"  BM{BM} th{th}: FAIL {str(e)[:70]}")
print("BEST:",best)
print("=== swizzle stress ===")
for swz in (None,4,8,16):
  try:
    k=tilelang.compile(rope_split(M,H,4,128,swz),out_idx=[4,5],target="cuda")
    k(x0,x1,cos,sin); torch.cuda.synchronize()
    t=do_bench(lambda: k(x0,x1,cos,sin),warmup=20,rep=50)
    print(f"  swz={swz}: {t*1000:.1f}us {BYTES/(t*1e-3)/1e9:.0f}GB/s")
  except Exception as e: print(f"  swz={swz}: FAIL {str(e)[:50]}")