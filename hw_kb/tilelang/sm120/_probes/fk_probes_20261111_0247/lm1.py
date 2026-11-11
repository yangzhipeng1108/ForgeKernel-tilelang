import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
# LM head: logits = X (B*S, H) @ W_vocab^T (V, H)^T -> (B*S, V). Skinny: M small, N=V huge, K=H
def lmhead(M,N,K,BM,BN,BK,SK=1,stages=2,acc_dtype="float32"):
    scale=1.0/ (K**0.5)
    @T.prim_func
    def main(X:T.Tensor((M,K),"float16"),W:T.Tensor((N,K),"float16"),O:T.Tensor((M,N),"float32")):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),SK,threads=128) as (bn,bm,sk):
            Xs=T.alloc_shared((BM,BK),"float16"); Ws=T.alloc_shared((BN,BK),"float16")
            C=T.alloc_fragment((BM,BN),"float32")
            T.clear(C)
            kper=T.ceildiv(K,BK)
            k0=sk*(kper//SK); k1=(sk+1)*(kper//SK) if sk<SK-1 else kper
            for ko in T.Pipelined(k1-k0,num_stages=stages):
                k=k0+ko
                T.copy(X[bm*BM,k*BK],Xs); T.copy(W[bn*BN,k*BK],Ws)
                T.gemm(Xs,Ws,C,transpose_B=True)
            if SK==1:
                for i,j in T.Parallel(BM,BN): O[bm*BM+i,bn*BN+j]=C[i,j]
            else:
                for i,j in T.Parallel(BM,BN): T.atomic_add(O[bm*BM+i,bn*BN+j],C[i,j])
    return main
torch.manual_seed(0)
# typical LLM LM head decode: batch*seq=64 (decode), H=4096, V=128256 (round 128256)
for (M,N,K) in [(64,128256,4096),(64,151936,3584),(256,128256,4096),(16,50257,2048)]:
    X=torch.randn(M,K,device="cuda",dtype=torch.float16)*0.1
    W=torch.randn(N,K,device="cuda",dtype=torch.float16)*0.1
    ref=(X.float()@W.float().T)
    FLOP=2*M*N*K; BYTES=M*K*2+N*K*2+M*N*4
    print(f"--- M{M} N{N} K{K}: grid_full={((N+127)//128)*((M+63)//64)} int={FLOP/BYTES:.1f}")
    for BM,BN,BK,SK in [(64,128,32,1),(64,256,32,1),(Mid:=64,128,64,1)]:
        pass
    for BM,BN,BK,SK in [(64,128,32,1),(64,128,32,4),(64,128,32,8),(64,256,64,1),(64,256,64,4),(64,64,64,1)]:
        try:
            O=torch.zeros(M,N,device="cuda",dtype=torch.float32)
            k=tilelang.compile(lmhead(M,N,K,BM,BN,BK,SK),out_idx=[2],target="cuda")
            o=k(X,W); torch.cuda.synchronize()
            cos=torch.nn.functional.cosine_similarity(o.flatten(),ref.flatten(),dim=0).item()
            t=do_bench(lambda: k(X,W),warmup=15,rep=50)
            print(f"  BM{BM} BN{BN} BK{BK} SK{SK}: {t*1000:8.1f}us {FLOP/(t*1e-3)/1e12:6.1f}TF cos={cos:.5f}")
        except Exception as e: print(f"  BM{BM} BN{BN} BK{BK} SK{SK}: FAIL {str(e)[:90]}")