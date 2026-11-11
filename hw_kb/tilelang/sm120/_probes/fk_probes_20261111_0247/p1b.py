import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
Dc=512; DV=128; HQ=128; QLEN=16; KVLEN=8192
def attn_latent(HQ,QLEN,KVLEN,BM,BN,Dc,stages=1):
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
def upproj(HQ,QLEN,Dc,DV,BM,BN,BK):
    @T.prim_func
    def main(L:T.Tensor((HQ,QLEN,Dc),"float16"),Wuv:T.Tensor((DV,Dc),"float16"),O:T.Tensor((HQ,QLEN,DV),"float16")):
        with T.Kernel(HQ,threads=128) as (bh):
            Ls=T.alloc_shared((BM,BK),"float16"); Ws=T.alloc_shared((BN,BK),"float16")
            C=T.alloc_fragment((BM,BN),"float32"); T.clear(C)
            for ko in T.Pipelined(T.ceildiv(Dc,BK),num_stages=2):
                T.copy(L[bh,0,ko*BK],Ls)
                T.copy(Wuv[0,ko*BK],Ws)   # Wuv stored as (DV,Dc), tile rows 0..DV, cols ko*BK
                T.gemm(Ls,Ws,C,transpose_B=True)
            for i,j in T.Parallel(BM,BN): O[bh,i,j]=C[i,j]
    return main
torch.manual_seed(0)
Qa=torch.randn(HQ,QLEN,Dc,device="cuda",dtype=torch.float16)*0.05
C=torch.randn(KVLEN,Dc,device="cuda",dtype=torch.float16)*0.05
Wuv=torch.randn(DV,Dc,device="cuda",dtype=torch.float16)*0.05   # (DV,Dc)
S=torch.einsum('hqd,kd->hqk',Qa.float(),C.float())/math.sqrt(Dc)
P=torch.softmax(S,dim=-1)
Oref=(torch.einsum('hqk,kd->hqd',P,C.float()))@Wuv.float().T
FLOP=2*HQ*QLEN*KVLEN*Dc*2 + HQ*QLEN*Dc*DV*2
k1=tilelang.compile(attn_latent(HQ,QLEN,KVLEN,QLEN,32,Dc),out_idx=[2],target="cuda")
k2=tilelang.compile(upproj(HQ,QLEN,Dc,DV,QLEN,128,64),out_idx=[2],target="cuda")
L=k1(Qa,C); O=k2(L,Wuv); torch.cuda.synchronize()
cos=torch.nn.functional.cosine_similarity(O.float().flatten(),Oref.flatten(),dim=0).item()
md=(O.float()-Oref).abs().max().item()
t1=do_bench(lambda: k1(Qa,C),warmup=15,rep=50)
t2=do_bench(lambda: k2(L,Wuv),warmup=15,rep=50)
print(f"pass1(Dc512 attn)={t1*1000:.1f}us  pass2(upproj)={t2*1000:.1f}us  total={(t1+t2)*1000:.1f}us")
print(f"cos={cos:.5f} md={md:.2e}  effTF={FLOP/((t1+t2)*1e-3)/1e12:.1f}")
print(f"(single-kernel Dc128 for ref: 123.5us 70.1TF)")