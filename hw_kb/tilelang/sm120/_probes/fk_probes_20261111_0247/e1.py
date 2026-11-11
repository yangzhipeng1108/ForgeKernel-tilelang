import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
D=128; L2E=math.log2(math.e)
def fa(H,SEQ,BM,BN,q_dt,k_dt,skip=True):
    scale=1.0/math.sqrt(D)*L2E
    NQ=SEQ//BM; NK=SEQ//BN
    @T.prim_func
    def main(Q:T.Tensor((H,SEQ,D),q_dt),K:T.Tensor((H,SEQ,D),k_dt),V:T.Tensor((H,SEQ,D),"float16"),O:T.Tensor((H,SEQ,D),"float16")):
        with T.Kernel(NQ,H,threads=128) as (bq,bh):
            Qs=T.alloc_shared((BM,D),q_dt); Ks=T.alloc_shared((BN,D),k_dt); Vs=T.alloc_shared((BN,D),"float16")
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,D),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rmax=T.alloc_fragment((BM,),"float32"); rsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Q[bh,bq*BM,0],Qs)
            nkb=T.ceildiv((bq+1)*BM,BN) if skip else NK
            for ko in T.Pipelined(nkb,num_stages=1):
                k=ko
                T.copy(K[bh,k*BN,0],Ks); T.copy(V[bh,k*BN,0],Vs)
                T.gemm(Qs,Ks,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                if skip:
                    for i,j in T.Parallel(BM,BN): S[i,j]=T.if_then_else((k*BN+j)<=(bq*BM+i),S[i,j]*scale,-T.infinity("float32"))
                else:
                    for i,j in T.Parallel(BM,BN): S[i,j]=S[i,j]*scale
                T.reduce_max(S,rmax,dim=1,clear=True)
                for i in T.Parallel(BM): m_i[i]=T.max(m_prev[i],rmax[i])
                for i,j in T.Parallel(BM,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
                for i,j in T.Parallel(BM,BN): S[i,j]=T.exp2(S[i,j]-m_i[i])
                T.reduce_sum(S,rsum,dim=1)
                for i in T.Parallel(BM): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rsum[i]
                for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
                T.gemm(Ps,Vs,acc)
            for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]/l_i[i]
            T.copy(acc,O[bh,bq*BM,0])
    return main
torch.manual_seed(0)
H=16; SEQ=16384
Q=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
K=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16); V=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
Q8=Q.to(torch.float8_e4m3fn); K8=K.to(torch.float8_e4m3fn)
ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),K.float(),V.float(),is_causal=True)
FULL=4*H*SEQ*SEQ*D
for tag,Qo,Ko,qd,kd in [("f16",Q,K,"float16","float16"),("f8QK",Q8,K8,"float8_e4m3","float8_e4m3")]:
  try:
    k=tilelang.compile(fa(H,SEQ,64,128,qd,kd,True),out_idx=[3],target="cuda")
    o=k(Qo,Ko,V); torch.cuda.synchronize()
    cos=torch.nn.functional.cosine_similarity(o.float().flatten(),ref.flatten(),dim=0).item()
    t=do_bench(lambda: k(Qo,Ko,V),warmup=10,rep=30)
    print(f"SEQ{SEQ} causal {tag}: {t*1000:.1f}us effTF={FULL*0.5/(t*1e-3)/1e12:.1f} cos={cos:.5f}")
  except Exception as e: print(f"SEQ{SEQ} {tag}: FAIL {str(e)[:100]}")
# also seq 32768 f16 vs f8
SEQ=32768
Q=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16);K=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16);V=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
Q8=Q.to(torch.float8_e4m3fn);K8=K.to(torch.float8_e4m3fn)
ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),K.float(),V.float(),is_causal=True)
FULL=4*H*SEQ*SEQ*D
for tag,Qo,Ko,qd,kd in [("f16",Q,K,"float16","float16"),("f8QK",Q8,K8,"float8_e4m3","float8_e4m3")]:
  k=tilelang.compile(fa(H,SEQ,64,128,qd,kd,True),out_idx=[3],target="cuda")
  o=k(Qo,Ko,V); torch.cuda.synchronize()
  cos=torch.nn.functional.cosine_similarity(o.float().flatten(),ref.flatten(),dim=0).item()
  t=do_bench(lambda: k(Qo,Ko,V),warmup=10,rep=30)
  print(f"SEQ{SEQ} causal {tag}: {t*1000:.1f}us effTF={FULL*0.5/(t*1e-3)/1e12:.1f} cos={cos:.5f}")