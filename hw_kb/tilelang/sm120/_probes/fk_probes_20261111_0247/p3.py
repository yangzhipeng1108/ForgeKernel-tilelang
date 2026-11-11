import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
L2E=math.log2(math.e)
def fa(H,SEQ,D,BM,BN,stages):
    scale=1.0/math.sqrt(D)*L2E
    NQ=SEQ//BM
    @T.prim_func
    def main(Q:T.Tensor((H,SEQ,D),"float16"),K:T.Tensor((H,SEQ,D),"float16"),V:T.Tensor((H,SEQ,D),"float16"),O:T.Tensor((H,SEQ,D),"float16")):
        with T.Kernel(NQ,H,threads=128) as (bq,bh):
            Qs=T.alloc_shared((BM,D),"float16"); Ks=T.alloc_shared((BN,D),"float16"); Vs=T.alloc_shared((BN,D),"float16")
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,D),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rmax=T.alloc_fragment((BM,),"float32"); rsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Q[bh,bq*BM,0],Qs)
            for ko in T.Pipelined(T.ceildiv((bq+1)*BM,BN),num_stages=stages):
                k=ko
                T.copy(K[bh,k*BN,0],Ks); T.copy(V[bh,k*BN,0],Vs)
                T.gemm(Qs,Ks,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                for i,j in T.Parallel(BM,BN): S[i,j]=T.if_then_else((k*BN+j)<=(bq*BM+i),S[i,j]*scale,-T.infinity("float32"))
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
H=16; SEQ=4096
FULLF=4*H*SEQ*SEQ  # times D below
for D in (64,96,128):
  Q=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
  K=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16); V=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
  ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),K.float(),V.float(),is_causal=True)
  FULL=FULLF*D
  print(f"--- D={D}: smem per stage Qs+K s+V s = (BM+2*BN)*D*2")
  for BM,BN,st in [(64,128,1),(64,128,2),(64,64,2),(128,128,1),(128,64,2),(128,128,2)]:
    try:
      k=tilelang.compile(fa(H,SEQ,D,BM,BN,st),out_idx=[3],target="cuda")
      o=k(Q,K,V); torch.cuda.synchronize()
      cos=torch.nn.functional.cosine_similarity(o.float().flatten(),ref.flatten(),dim=0).item()
      t=do_bench(lambda: k(Q,K,V),warmup=10,rep=30)
      print(f"  D{D} BM{BM} BN{BN} st{st}: {t*1000:7.1f}us effTF={FULL*0.5/(t*1e-3)/1e12:6.1f} cos={cos:.5f}")
    except Exception as e: print(f"  D{D} BM{BM} BN{BN} st{st}: FAIL {str(e)[:60]}")