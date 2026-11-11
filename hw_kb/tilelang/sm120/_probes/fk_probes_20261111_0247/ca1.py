import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
D=128; L2E=math.log2(math.e)
# causal FA-2: q-tile at index bq covers rows [bq*BM,(bq+1)*BM); only k-blocks <= bq needed
def fa_causal(H,SEQ,BM,BN,skip=True):
    scale=1.0/math.sqrt(D)*L2E
    NQ=SEQ//BM; NK=SEQ//BN
    @T.prim_func
    def main(Q:T.Tensor((H,SEQ,D),"float16"),K:T.Tensor((H,SEQ,D),"float16"),V:T.Tensor((H,SEQ,D),"float16"),O:T.Tensor((H,SEQ,D),"float16")):
        with T.Kernel(NQ,H,threads=128) as (bq,bh):
            Qs=T.alloc_shared((BM,D),"float16"); Ks=T.alloc_shared((BN,D),"float16"); Vs=T.alloc_shared((BN,D),"float16")
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,D),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rowmax=T.alloc_fragment((BM,),"float32"); rowsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Q[bh,bq*BM,0],Qs)
            # number of k-blocks needed for this q-tile
            nkb = T.ceildiv((bq+1)*BM, BN) if skip else NK
            for k in T.Pipelined(nkb,num_stages=1):
                T.copy(K[bh,k*BN,0],Ks); T.copy(V[bh,k*BN,0],Vs)
                T.gemm(Qs,Ks,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                for i,j in T.Parallel(BM,BN):
                    # causal mask: global row = bq*BM+i, global col = k*BN+j
                    S[i,j]=T.if_then_else((k*BN+j)<=(bq*BM+i), S[i,j]*scale, -T.infinity("float32"))
                T.reduce_max(S,rowmax,dim=1,clear=True)
                for i in T.Parallel(BM): m_i[i]=T.max(m_prev[i],rowmax[i])
                for i,j in T.Parallel(BM,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
                for i,j in T.Parallel(BM,BN): S[i,j]=T.exp2(S[i,j]-m_i[i])
                T.reduce_sum(S,rowsum,dim=1)
                for i in T.Parallel(BM): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rowsum[i]
                for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
                T.gemm(Ps,Vs,acc)
            for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]/l_i[i]
            T.copy(acc,O[bh,bq*BM,0])
    return main
torch.manual_seed(0)
H=16; SEQ=4096
Q=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
K=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16); V=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),K.float(),V.float(),is_causal=True)
FULL=4*H*SEQ*SEQ*D
for BM,BN,skip in [(64,128,True),(64,128,False),(128,64,True),(64,64,True)]:
    try:
        k=tilelang.compile(fa_causal(H,SEQ,BM,BN,skip),out_idx=[3],target="cuda")
        out=k(Q,K,V); torch.cuda.synchronize()
        cos=torch.nn.functional.cosine_similarity(out.float().flatten(),ref.flatten(),dim=0).item()
        md=(out.float()-ref).abs().max().item()
        t=do_bench(lambda: k(Q,K,V),warmup=20,rep=100)
        # actual flops done ~ half of full for causal
        tag="skip" if skip else "noskip"
        print(f"BM{BM} BN{BN} {tag}: {t*1000:7.1f}us cos={cos:.5f} md={md:.2e} fullTF={FULL/(t*1e-3)/1e12:.1f} effTF(causal*0.5)={FULL*0.5/(t*1e-3)/1e12:.1f}")
    except Exception as e: print(f"BM{BM} BN{BN} skip{skip}: FAIL {str(e)[:110]}")