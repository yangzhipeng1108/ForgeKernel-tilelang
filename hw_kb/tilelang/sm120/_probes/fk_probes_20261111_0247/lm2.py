import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
def lmhead(M,N,K,BM,BN,BK,stages=2):
    @T.prim_func
    def main(X:T.Tensor((M,K),"float16"),W:T.Tensor((N,K),"float16"),O:T.Tensor((M,N),"float32")):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bn,bm):
            Xs=T.alloc_shared((BM,BK),"float16"); Ws=T.alloc_shared((BN,BK),"float16")
            C=T.alloc_fragment((BM,BN),"float32"); T.clear(C)
            for ko in T.Pipelined(T.ceildiv(K,BK),num_stages=stages):
                T.copy(X[bm*BM,ko*BK],Xs); T.copy(W[bn*BN,ko*BK],Ws)
                T.gemm(Xs,Ws,C,transpose_B=True)
            for i,j in T.Parallel(BM,BN): O[bm*BM+i,bn*BN+j]=C[i,j]
    return main
torch.manual_seed(0)
M,N,K=16,50257,2048
X=torch.randn(M,K,device="cuda",dtype=torch.float16)*0.1
W=torch.randn(N,K,device="cuda",dtype=torch.float16)*0.1
ref=(X.float()@W.float().T)
FLOP=2*M*N*K; BYTES=M*K*2+N*K*2+M*N*4
print(f"M{M} N{N} K{K}: FLOP={FLOP/1e9:.1f}G bytes={BYTES/1e6:.1f}MB int={FLOP/BYTES:.1f}")
for BM,BN,BK in [(16,128,64),(16,256,64),(16,512,64),(16,256,128),(16,128,128),(32,256,64),(16,1024,32)]:
    try:
        k=tilelang.compile(lmhead(M,N,K,BM,BN,BK),out_idx=[2],target="cuda")
        o=k(X,W); torch.cuda.synchronize()
        cos=torch.nn.functional.cosine_similarity(o.flatten(),ref.flatten(),dim=0).item()
        t=do_bench(lambda: k(X,W),warmup=15,rep=50)
        print(f"  BM{BM} BN{BN} BK{BK}: {t*1000:8.1f}us {FLOP/(t*1e-3)/1e12:6.1f}TF BW={BYTES/(t*1e-3)/1e9:.0f}GB/s cos={cos:.5f}")
    except Exception as e: print(f"  BM{BM} BN{BN} BK{BK}: FAIL {str(e)[:90]}")
# also torch matmul baseline
tt=do_bench(lambda: X@t.W, warmup=15,rep=50) if False else None
tt=do_bench(lambda: torch.nn.functional.linear(X.float(),W.float()),warmup=15,rep=50)
print(f"  torch fp32 linear: {tt*1000:.1f}us {FLOP/(tt*1e-3)/1e12:.1f}TF")