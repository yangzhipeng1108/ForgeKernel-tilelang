import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
D=128; L2E=math.log2(math.e)
def gqa(HQ,HKV,QLEN,KVLEN,BN):
    G=HQ//HKV; scale=1.0/math.sqrt(D)*L2E
    @T.prim_func
    def main(Q:T.Tensor((HQ,QLEN,D),"float16"),K:T.Tensor((HKV,KVLEN,D),"float16"),V:T.Tensor((HKV,KVLEN,D),"float16"),O:T.Tensor((HQ,QLEN,D),"float16")):
        with T.Kernel(HQ,threads=128) as (bh):
            kvh=bh//G
            Qs=T.alloc_shared((QLEN,D),"float16"); Ks=T.alloc_shared((BN,D),"float16"); Vs=T.alloc_shared((BN,D),"float16")
            Ps=T.alloc_shared((QLEN,BN),"float16"); acc=T.alloc_fragment((QLEN,D),"float32")
            m_i=T.alloc_fragment((QLEN,),"float32"); l_i=T.alloc_fragment((QLEN,),"float32")
            m_prev=T.alloc_fragment((QLEN,),"float32"); l_prev=T.alloc_fragment((QLEN,),"float32")
            S=T.alloc_fragment((QLEN,BN),"float32"); rmax=T.alloc_fragment((QLEN,),"float32"); rsum=T.alloc_fragment((QLEN,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Q[bh,0,0],Qs)
            for k in T.Pipelined(T.ceildiv(KVLEN,BN),num_stages=1):
                T.copy(K[kvh,k*BN,0],Ks); T.copy(V[kvh,k*BN,0],Vs)
                T.gemm(Qs,Ks,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                for i,j in T.Parallel(QLEN,BN): S[i,j]=S[i,j]*scale
                T.reduce_max(S,rmax,dim=1,clear=True)
                for i in T.Parallel(QLEN): m_i[i]=T.max(m_prev[i],rmax[i])
                for i,j in T.Parallel(QLEN,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
                for i,j in T.Parallel(QLEN,BN): S[i,j]=T.exp2(S[i,j]-m_i[i])
                T.reduce_sum(S,rsum,dim=1)
                for i in T.Parallel(QLEN): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rsum[i]
                for i,j in T.Parallel(QLEN,D): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
                T.gemm(Ps,Vs,acc)
            for i,j in T.Parallel(QLEN,D): acc[i,j]=acc[i,j]/l_i[i]
            T.copy(acc,O[bh,0,0])
    return main
def mla(HQ,QLEN,KVLEN,BN,Dc,DV):
    scale=1.0/math.sqrt(Dc)*L2E
    @T.prim_func
    def main(Qa:T.Tensor((HQ,QLEN,Dc),"float16"),C:T.Tensor((KVLEN,Dc),"float16"),Wuv:T.Tensor((Dc,DV),"float16"),O:T.Tensor((HQ,QLEN,DV),"float16")):
        with T.Kernel(HQ,threads=128) as (bh):
            Qs=T.alloc_shared((QLEN,Dc),"float16"); Cs=T.alloc_shared((BN,Dc),"float16")
            Ws=T.alloc_shared((Dc,DV),"float16"); Ps=T.alloc_shared((QLEN,BN),"float16"); Ls=T.alloc_shared((QLEN,Dc),"float16")
            acc=T.alloc_fragment((QLEN,Dc),"float32"); out=T.alloc_fragment((QLEN,DV),"float32")
            m_i=T.alloc_fragment((QLEN,),"float32"); l_i=T.alloc_fragment((QLEN,),"float32")
            m_prev=T.alloc_fragment((QLEN,),"float32"); l_prev=T.alloc_fragment((QLEN,),"float32")
            S=T.alloc_fragment((QLEN,BN),"float32"); rmax=T.alloc_fragment((QLEN,),"float32"); rsum=T.alloc_fragment((QLEN,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Qa[bh,0,0],Qs); T.copy(Wuv,Ws)
            for k in T.Pipelined(T.ceildiv(KVLEN,BN),num_stages=2):
                T.copy(C[k*BN,0],Cs)
                T.gemm(Qs,Cs,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                for i,j in T.Parallel(QLEN,BN): S[i,j]=S[i,j]*scale
                T.reduce_max(S,rmax,dim=1,clear=True)
                for i in T.Parallel(QLEN): m_i[i]=T.max(m_prev[i],rmax[i])
                for i,j in T.Parallel(QLEN,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
                for i,j in T.Parallel(QLEN,BN): S[i,j]=T.exp2(S[i,j]-m_i[i])
                T.reduce_sum(S,rsum,dim=1)
                for i in T.Parallel(QLEN): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rsum[i]
                for i,j in T.Parallel(QLEN,Dc): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
                T.gemm(Ps,Cs,acc)
            for i,j in T.Parallel(QLEN,Dc): Ls[i,j]=acc[i,j]/l_i[i]
            T.clear(out); T.gemm(Ls,Ws,out)
            T.copy(out,O[bh,0,0])
    return main
torch.manual_seed(0)
QLEN=16; KVLEN=8192; Dc=128
for HKV in (8,4,1):  # G=4,8,32
  HQ=32; G=HQ//HKV
  Q=torch.randn(HQ,QLEN,D,device="cuda",dtype=torch.float16)*0.05
  K=torch.randn(HKV,KVLEN,D,device="cuda",dtype=torch.float16)*0.05; V=torch.randn(HKV,KVLEN,D,device="cuda",dtype=torch.float16)*0.05
  kg=tilelang.compile(gqa(HQ,HKV,QLEN,KVLEN,128),out_idx=[3],target="cuda")
  kg(Q,K,V); torch.cuda.synchronize(); tg=do_bench(lambda: kg(Q,K,V),warmup=15,rep=50)
  Qa=torch.randn(HQ,QLEN,Dc,device="cuda",dtype=torch.float16)*0.05
  C=torch.randn(KVLEN,Dc,device="cuda",dtype=torch.float16)*0.05
  Wuv=torch.randn(Dc,D,device="cuda",dtype=torch.float16)*0.05
  km=tilelang.compile(mla(HQ,QLEN,KVLEN,64,Dc,D),out_idx=[3],target="cuda")
  km(Qa,C,Wuv); torch.cuda.synchronize(); tm=do_bench(lambda: km(Qa,C,Wuv),warmup=15,rep=50)
  gqa_b=HKV*KVLEN*D*2*2; mla_b=KVLEN*Dc*2
  print(f"HKV{HKV}(G{G}): GQA {tg*1000:.1f}us cache={gqa_b/1e6:.0f}MB | MLA {tm*1000:.1f}us cache={mla_b/1e6:.1f}MB | MLA/GQA={tm/tg:.2f}x")