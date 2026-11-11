import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
D=128; L2E=math.log2(math.e)
def fa8(H,SEQ,BM,BN):
    scale=1.0/math.sqrt(D)*L2E
    NQ=SEQ//BM
    @T.prim_func
    def main(Q:T.Tensor((H,SEQ,D),"float16"),K:T.Tensor((H,SEQ,D),"float8_e4m3"),V:T.Tensor((H,SEQ,D),"float8_e4m3"),O:T.Tensor((H,SEQ,D),"float16")):
        with T.Kernel(NQ,H,threads=128) as (bq,bh):
            Qs=T.alloc_shared((BM,D),"float16"); Q8=T.alloc_shared((BM,D),"float8_e4m3")
            Ks=T.alloc_shared((BN,D),"float8_e4m3"); Vs=T.alloc_shared((BN,D),"float8_e4m3")
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,D),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rmax=T.alloc_fragment((BM,),"float32"); rsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Q[bh,bq*BM,0],Qs)
            for i,j in T.Parallel(BM,D): Q8[i,j]=T.cast(Qs[i,j],"float8_e4m3")
            for ko in T.Pipelined(T.ceildiv((bq+1)*BM,BN),num_stages=1):
                k=ko
                T.copy(K[bh,k*BN,0],Ks); T.copy(V[bh,k*BN,0],Vs)
                T.gemm(Q8,Ks,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                for i,j in T.Parallel(BM,BN): S[i,j]=T.if_then_else((k*BN+j)<=(bq*BM+i),S[i,j]*scale,-T.infinity("float32"))
                T.reduce_max(S,rmax,dim=1,clear=True)
                for i in T.Parallel(BM): m_i[i]=T.max(m_prev[i],rmax[i])
                for i,j in T.Parallel(BM,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
                for i,j in T.Parallel(BM,BN): S[i,j]=T.exp2(S[i,j]-m_i[i])
                T.reduce_sum(S,rsum,dim=1)
                for i in T.Parallel(BM): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rsum[i]
                for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
                # PV in fp8: need V cast to fp16 for acc(fp32)? try direct fp8 V gemm
                T.gemm(Ps,T.alloc_shared((BN,D),"float16") if False else Vs, acc) if False else None
                Vs16=T.alloc_shared((BN,D),"float16")
                for i,j in T.Parallel(BN,D): Vs16[i,j]=T.cast(Vs[i,j],"float16")
                T.gemm(Ps,Vs16,acc)
            for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]/l_i[i]
            T.copy(acc,O[bh,bq*BM,0])
    return main
torch.manual_seed(0)
H=16; SEQ=4096
Q=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
K=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16); V=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),K.float(),V.float(),is_causal=True)
K8=K.to(torch.float8_e4m3fn); V8=V.to(torch.float8_e4m3fn)
FULL=4*H*SEQ*SEQ*D
for tag,BM,BN in [("fp8 causal 64/128",64,128),("fp8 causal 64/64",64,64)]:
  try:
    k=tilelang.compile(fa8(H,SEQ,BM,BN),out_idx=[3],target="cuda")
    o=k(Q,K8,V8); torch.cuda.synchronize()
    cos=torch.nn.functional.cosine_similarity(o.float().flatten(),ref.flatten(),dim=0).item()
    t=do_bench(lambda: k(Q,K8,V8),warmup=15,rep=50)
    print(f"{tag}: {t*1000:7.1f}us effTF={FULL*0.5/(t*1e-3)/1e12:.1f} cos={cos:.5f}")
  except Exception as e: print(f"{tag}: FAIL {str(e)[:130]}")
print("(fp16 causal baseline: 448.4us effTF=153.2)")