import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
# Small-Dc MLA to fit smem while keeping the mechanism: latent Dc, decode q=16, wide or out
Dc=128; DV=128; HQ=128; QLEN=16; KVLEN=8192
def build(HQ,QLEN,KVLEN,BM,BN,Dc,DV,stages=1):
    scale=1.0/math.sqrt(Dc)*math.log2(math.e)
    @T.prim_func
    def main(Qa:T.Tensor((HQ,QLEN,Dc),"float16"),C:T.Tensor((KVLEN,Dc),"float16"),Wuv:T.Tensor((Dc,DV),"float16"),O:T.Tensor((HQ,QLEN,DV),"float16")):
        with T.Kernel(HQ,threads=128) as (bh):
            Qs=T.alloc_shared((BM,Dc),"float16"); Cs=T.alloc_shared((BN,Dc),"float16")
            Ws=T.alloc_shared((Dc,DV),"float16"); Ps=T.alloc_shared((BM,BN),"float16")
            Ls=T.alloc_shared((BM,Dc),"float16")
            acc=T.alloc_fragment((BM,Dc),"float32"); out=T.alloc_fragment((BM,DV),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rowmax=T.alloc_fragment((BM,),"float32"); rowsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Qa[bh,0,0],Qs); T.copy(Wuv,Ws)
            for k in T.Pipelined(T.ceildiv(KVLEN,BN),num_stages=stages):
                T.copy(C[k*BN,0],Cs)
                T.gemm(Qs,Cs,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                for i,j in T.Parallel(BM,BN): S[i,j]=S[i,j]*scale
                T.reduce_max(S,rowmax,dim=1,clear=True)
                for i in T.Parallel(BM): m_i[i]=T.max(m_prev[i],rowmax[i])
                for i,j in T.Parallel(BM,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
                for i,j in T.Parallel(BM,BN): S[i,j]=T.exp2(S[i,j]-m_i[i])
                T.reduce_sum(S,rowsum,dim=1)
                for i in T.Parallel(BM): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rowsum[i]
                for i,j in T.Parallel(BM,Dc): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
                T.gemm(Ps,Cs,acc)
            for i,j in T.Parallel(BM,Dc): Ls[i,j]=acc[i,j]/l_i[i]
            T.clear(out); T.gemm(Ls,Ws,out)
            T.copy(out,O[bh,0,0])
    return main
torch.manual_seed(0)
Qa=torch.randn(HQ,QLEN,Dc,device="cuda",dtype=torch.float16)*0.05
C=torch.randn(KVLEN,Dc,device="cuda",dtype=torch.float16)*0.05
Wuv=torch.randn(Dc,DV,device="cuda",dtype=torch.float16)*0.05
S=torch.einsum('hqd,kd->hqk',Qa.float(),C.float())/math.sqrt(Dc)
P=torch.softmax(S,dim=-1)
Oref=(torch.einsum('hqk,kd->hqd',P,C.float()))@Wuv.float()
FLOP=2*HQ*QLEN*KVLEN*Dc*2 + HQ*QLEN*Dc*DV*2
BYTES=HQ*QLEN*Dc*2 + KVLEN*Dc*2
mha_b=HQ*QLEN*128*2 + HQ*KVLEN*128*2*2
print(f"MLA: latent={BYTES/1e6:.2f}MB vs MHA128h={mha_b/1e6:.1f}MB ratio={BYTES/mha_b:.4f} int={FLOP/BYTES:.0f}")
for BN in (64,128,256):
  try:
    k=tilelang.compile(build(HQ,QLEN,KVLEN,QLEN,BN,Dc,DV),out_idx=[3],target="cuda")
    o=k(Qa,C,Wuv); torch.cuda.synchronize()
    cos=torch.nn.functional.cosine_similarity(o.float().flatten(),Oref.flatten(),dim=0).item()
    md=(o.float()-Oref).abs().max().item()
    t=do_bench(lambda: k(Qa,C,Wuv),warmup=20,rep=100)
    print(f"BN{BN}: {t*1000:7.1f}us {FLOP/(t*1e-3)/1e12:6.1f}TF cos={cos:.5f} md={md:.2e} BW={BYTES/(t*1e-3)/1e9:.0f}GB/s")
  except Exception as e: print(f"BN{BN}: FAIL {str(e)[:110]}")