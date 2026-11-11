import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
D=128; L2E=math.log2(math.e)
def mha_splitKV(HQ,KVLEN,BM,BN,SPLIT):
    scale=1.0/math.sqrt(D)*L2E
    NBLK=KVLEN//BN
    per=(NBLK+SPLIT-1)//SPLIT
    @T.prim_func
    def main(Q:T.Tensor((HQ,BM,D),"float16"),K:T.Tensor((HQ,KVLEN,D),"float16"),V:T.Tensor((HQ,KVLEN,D),"float16"),
             O:T.Tensor((HQ,SPLIT,BM,D),"float32"),L:T.Tensor((HQ,SPLIT,BM),"float32")):
        with T.Kernel(HQ,SPLIT,threads=128) as (bh,sp):
            Qs=T.alloc_shared((BM,D),"float16"); Ks=T.alloc_shared((BN,D),"float16"); Vs=T.alloc_shared((BN,D),"float16")
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,D),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rowmax=T.alloc_fragment((BM,),"float32"); rowsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Q[bh,0,0],Qs)
            for kk in T.Pipelined(per,num_stages=1):
                k=sp*per+kk
                if k<NBLK:
                    T.copy(K[bh,k*BN,0],Ks); T.copy(V[bh,k*BN,0],Vs)
                    T.gemm(Qs,Ks,S,transpose_B=True,clear_accum=True)
                    T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                    for i,j in T.Parallel(BM,BN): S[i,j]=S[i,j]*scale
                    T.reduce_max(S,rowmax,dim=1,clear=True)
                    for i in T.Parallel(BM): m_i[i]=T.max(m_prev[i],rowmax[i])
                    for i,j in T.Parallel(BM,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
                    for i,j in T.Parallel(BM,BN): S[i,j]=T.exp2(S[i,j]-m_i[i])
                    T.reduce_sum(S,rowsum,dim=1)
                    for i in T.Parallel(BM): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rowsum[i]
                    for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
                    T.gemm(Ps,Vs,acc)
            # rescale acc to absolute (multiply by exp(m_i)) so global sum works
            for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]*T.exp2(m_i[i])
            for i in T.Parallel(BM): l_i[i]=l_i[i]*T.exp2(m_i[i])
            T.copy(acc,O[bh,sp,0,0]); T.copy(l_i,L[bh,sp,0])
    return main
torch.manual_seed(0)
HQ=32; QLEN=16; KVLEN=8192; BM=16; BN=128
Q=torch.randn(HQ,QLEN,D,device="cuda",dtype=torch.float16)
K=torch.randn(HQ,KVLEN,D,device="cuda",dtype=torch.float16); V=torch.randn(HQ,KVLEN,D,device="cuda",dtype=torch.float16)
ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),K.float(),V.float())
FLOP=4*HQ*QLEN*KVLEN*D
for SPLIT in (1,4,8,16):
    try:
        k=tilelang.compile(mha_splitKV(HQ,KVLEN,BM,BN,SPLIT),out_idx=[3,4],target="cuda")
        O,L=k(Q,K,V); torch.cuda.synchronize()
        Oc=O.float().sum(1)/(L.float().sum(1).unsqueeze(-1)+1e-9)
        cos=torch.nn.functional.cosine_similarity(Oc.flatten(),ref.flatten(),dim=0).item()
        t=do_bench(lambda: k(Q,K,V),warmup=20,rep=100)
        print(f"SPLIT={SPLIT} grid={HQ*SPLIT}: {FLOP/(t*1e-3)/1e12:6.1f}TF {t*1000:7.1f}us cos={cos:.5f}")
    except Exception as e: print(f"SPLIT={SPLIT}: FAIL {str(e)[:120]}")