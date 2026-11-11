import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
def gemm(M,N,K,BM,BN,BK,xdt,wdt):
    @T.prim_func
    def main(X:T.Tensor((M,K),xdt),W:T.Tensor((N,K),wdt),O:T.Tensor((M,N),"float32")):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bn,bm):
            Xs=T.alloc_shared((BM,BK),xdt); Ws=T.alloc_shared((BN,BK),wdt)
            C=T.alloc_fragment((BM,BN),"float32"); T.clear(C)
            for ko in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(X[bm*BM,ko*BK],Xs); T.copy(W[bn*BN,ko*BK],Ws)
                T.gemm(Xs,Ws,C,transpose_B=True)
            for i,j in T.Parallel(BM,BN): O[bm*BM+i,bn*BN+j]=C[i,j]
    return main
torch.manual_seed(0)
M,N,K=16,50257,2048; BM,BN,BK=16,128,64
X=torch.randn(M,K,device="cuda",dtype=torch.float16)*0.1
W=torch.randn(N,K,device="cuda",dtype=torch.float16)*0.1
ref=(X.float()@W.float().T)
FLOP=2*M*N*K
X8=X.to(torch.float8_e4m3fn); W8=W.to(torch.float8_e4m3fn)
for tag,Xo,Wo,xd,wd,byts in [
   ("f16",X,W,"float16","float16",N*K*2),
   ("f8 both",X8,W8,"float8_e4m3","float8_e4m3",N*K*1)]:
  try:
    k=tilelang.compile(gemm(M,N,K,BM,BN,BK,xd,wd),out_idx=[2],target="cuda")
    o=k(Xo,Wo); torch.cuda.synchronize()
    cos=torch.nn.functional.cosine_similarity(o.flatten(),ref.flatten(),dim=0).item()
    t=do_bench(lambda: k(Xo,Wo),warmup=15,rep=50)
    print(f"{tag}: {t*1000:.1f}us {FLOP/(t*1e-3)/1e12:.1f}TF BW@W={byts/(t*1e-3)/1e9:.0f}GB/s cos={cos:.5f}")
  except Exception as e: print(f"{tag}: FAIL {str(e)[:110]}")
# bigger M where compute matters
M=256
X=torch.randn(M,K,device="cuda",dtype=torch.float16)*0.1
X8=X.to(torch.float8_e4m3fn)
ref=(X.float()@W.float().T); FLOP=2*M*N*K
for tag,Xo,xd,wd,byts in [("f16",X,"float16","float16",M*K*2+N*K*2),("f8",X8,"float8_e4m3","float8_e4m3",M*K*1+N*K*1)]:
  k=tilelang.compile(gemm(M,N,K,BM,BN,BK,xd,wd),out_idx=[2],target="cuda")
  o=k(Xo,W8 if tag=="f8" else W); torch.cuda.synchronize()
  cos=torch.nn.functional.cosine_similarity(o.flatten(),ref.flatten(),dim=0).item()
  t=do_bench(lambda: k(Xo,W8 if tag=="f8" else W),warmup=15,rep=50)
  print(f"M256 {tag}: {t*1000:.1f}us {FLOP/(t*1e-3)/1e12:.1f}TF cos={cos:.5f}")