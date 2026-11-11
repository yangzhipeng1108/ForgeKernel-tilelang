import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
Dc=512; DV=128; HQ=128; QLEN=16; KVLEN=8192
def attn_latent(HQ,QLEN,KVLEN,BM,BN,Dc,stages):
    scale=1.0/math.sqrt(Dc)*math.log2(math.e)
    @T.prim_func
    def main(Qa:T.Tensor((HQ,QLEN,Dc),"float16"),C:T.Tensor((KVLEN,Dc),"float16"),L:T.Tensor((HQ,QLEN,Dc),"float16")):
        with T.Kernel(HQ,threads=128) as (bh):
            Qs=T.alloc_shared((BM,Dc),"float16"); Cs=T.alloc_shared((BN,Dc),"float16")
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,Dc),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rmax=T.alloc_fragment((BM,),"float32"); rsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Qa[bh,0,0],Qs)
            for k in T.Pipelined(T.ceildiv(KVLEN,BN),num_stages=stages):
                T.copy(C[k*BN,0],Cs)
                T.gemm(Qs,Cs,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                for i,j in T.Parallel(BM,BN): S[i,j]=S[i,j]*scale
                T.reduce_max(S,rmax,dim=1,clear=True)
                for i in T.Parallel(BM): m_i[i]=T.max(m_prev[i],rmax[i])
                for i,j in T.Parallel(BM,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
                for i,j in T.Parallel(BM,BN): S[i,j]=T.exp2(S[i,j]-m_i[i])
                T.reduce_sum(S,rsum,dim=1)
                for i in T.Parallel(BM): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rsum[i]
                for i,j in T.Parallel(BM,Dc): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
                T.gemm(Ps,Cs,acc)
            for i,j in T.Parallel(BM,Dc): L[bh,i,j]=acc[i,j]/l_i[i]
    return main
torch.manual_seed(0)
Qa=torch.randn(HQ,QLEN,Dc,device="cuda",dtype=torch.float16)*0.05
C=torch.randn(KVLEN,Dc,device="cuda",dtype=torch.float16)*0.05
Lref=torch.einsum('hqk,kd->hqd',torch.softmax(torch.einsum('hqd,kd->hqk',Qa.float(),C.float())/math.sqrt(Dc),dim=-1),C.float())
FLOP=2*HQ*QLEN*KVLEN*Dc*2
best=None
for thr in (128,256):
 for BN in (16,32,64):
  for st in (1,2,3):
    try:
      k=tilelang.compile(attn_latent(HQ,QLEN,KVLEN,QLEN,BN,Dc,st),out_idx=[2],target="cuda")
      L=k(Qa,C); torch.cuda.synchronize()
      cos=torch.nn.functional.cosine_similarity(L.float().flatten(),Lref.flatten(),dim=0).item()
      t=do_bench(lambda: k(Qa,C),warmup=10,rep=30)
      tf=FLOP/(t*1e-3)/1e12
      mark=""
      if cos>0.999 and (best is None or t<best[0]): best=(t,thr,BN,st,tf); mark=" *"
      print(f"BN{BN} st{st}: {t*1000:.1f}us {tf:.1f}TF cos={cos:.5f}{mark}")
    except Exception as e: pass
print("BEST:",best)