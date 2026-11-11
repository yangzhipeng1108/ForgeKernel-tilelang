import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
def gemm(M,N,K,BM,BN,BK,NTILE=None):
    if NTILE is None:
        @T.prim_func
        def main(X:T.Tensor((M,K),"float16"),W:T.Tensor((N,K),"float16"),O:T.Tensor((M,N),"float32")):
            with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bn,bm):
                Xs=T.alloc_shared((BM,BK),"float16"); Ws=T.alloc_shared((BN,BK),"float16")
                C=T.alloc_fragment((BM,BN),"float32"); T.clear(C)
                for ko in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                    T.copy(X[bm*BM,ko*BK],Xs); T.copy(W[bn*BN,ko*BK],Ws)
                    T.gemm(Xs,Ws,C,transpose_B=True)
                for i,j in T.Parallel(BM,BN): O[bm*BM+i,bn*BN+j]=C[i,j]
        return main
    else:
        @T.prim_func
        def main(X:T.Tensor((M,K),"float16"),W:T.Tensor((N,K),"float16"),IDX:T.Tensor((NTILE,),"int32"),O:T.Tensor((M,NTILE*BN),"float32")):
            with T.Kernel(NTILE,T.ceildiv(M,BM),threads=128) as (nt,bm):
                Xs=T.alloc_shared((BM,BK),"float16"); Ws=T.alloc_shared((BN,BK),"float16")
                C=T.alloc_fragment((BM,BN),"float32"); T.clear(C)
                rb=IDX[nt]
                for ko in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                    T.copy(X[bm*BM,ko*BK],Xs); T.copy(W[rb*BN,ko*BK],Ws)
                    T.gemm(Xs,Ws,C,transpose_B=True)
                for i,j in T.Parallel(BM,BN): O[bm*BM+i,nt*BN+j]=C[i,j]
        return main
torch.manual_seed(0)
M,N,K=16,50257,2048; BM,BN,BK=16,128,64; NT=N//BN
def score_recall(W,struct):
    X=torch.randn(M,K,device="cuda",dtype=torch.float16)*0.3
    logits=(X.float()@W.float().T)  # (M,N) true
    # coarse: G[tile] = mean of W rows in tile
    Wt=W.float().view(NT,BN,K)
    G=Wt.mean(1)  # (NT,K)
    coarse=X.float()@G.T  # (M,NT)
    for kept in (0.25,0.10):
        ntile=int(NT*kept)
        top=coarse.topk(ntile,dim=1).indices  # (M,ntile)
        # true top-1 and top-10 tokens per row
        true1=logits.argmax(1)
        true10=logits.topk(10,dim=1).indices
        # convert token -> tile
        tile_of=true1//BN
        # is true top-1's tile in selected tiles?
        hit1=sum((top[b]==tile_of[b]).any().item() for b in range(M))
        # recall@10: fraction of true top-10 tokens whose tile is selected
        hit10=0
        for b in range(M):
            sel=set(top[b].tolist())
            hit10+=sum((true10[b]//BN).tolist().count(t) for t in sel if True for _ in [0]) # placeholder
        # simpler: fraction of true top10 tokens in selected tiles
        cnt=0
        for b in range(M):
            sel=set(top[b].tolist())
            for t in true10[b].tolist():
                if (t//BN) in sel: cnt+=1
        print(f"  {struct} kept={kept:.2f}: top1 tile-recall={hit1}/{M}  top10 token-recall={cnt/(M*10):.2f}")
# random W (no structure)
torch.manual_seed(1)
W_rand=torch.randn(N,K,device="cuda",dtype=torch.float16)*0.1
print("RANDOM W (no cluster structure):")
score_recall(W_rand,"rand")
# structured W: rows in a tile share a common component -> tile-mean screening has signal
torch.manual_seed(2)
common=torch.randn(NT,K,device="cuda",dtype=torch.float16)*0.1
noise=torch.randn(NT,BN,K,device="cuda",dtype=torch.float16)*0.03
W_struct=(common.unsqueeze(1)+noise).reshape(N,K).contiguous()
print("STRUCTURED W (rows within a tile share a common component):")
score_recall(W_struct,"struct")