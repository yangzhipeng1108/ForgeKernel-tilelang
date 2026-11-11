import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
D=128; L2E=math.log2(math.e)
# FA-2 causal, fp8 QK vs fp16, with optional split-K (split kv range across blocks)
def fa(H,SEQ,BM,BN,qdt,kvdt,SPLIT=1):
    scale=1.0/math.sqrt(D)*L2E
    NQ=SEQ//BM; NK=SEQ//BN
    @T.prim_func
    def main(Q:T.Tensor((H,SEQ,D),qdt),K:T.Tensor((H,SEQ,D),kvdt),V:T.Tensor((H,SEQ,D),kvdt),
             O:T.Tensor((H,SPLIT,SEQ,D),"float32"),Ls:T.Tensor((H,SPLIT,SEQ),"float32")):
        with T.Kernel(NQ,H,SPLIT,threads=128) as (bq,bh,sp):
            Qs=T.alloc_shared((BM,D),"float16")
            Ks8=T.alloc_shared((BN,D),kvdt); Vs8=T.alloc_shared((BN,D),kvdt)
            Ks=T.alloc_shared((BN,D),"float16"); Vs=T.alloc_shared((BN,D),"float16")
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,D),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rowmax=T.alloc_fragment((BM,),"float32"); rowsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            for i,j in T.Parallel(BM,D): Qs[i,j]=T.cast(Q[bh,bq*BM+i,j],"float16")
            ktot=T.ceildiv((bq+1)*BM,BN)
            kper=T.ceildiv(ktot,SPLIT); k0=sp*kper; k1=T.min(ktot,(sp+1)*kper)
            for ko in T.Pipelined(k1-k0,num_stages=1):
                k=k0+ko
                for i,j in T.Parallel(BN,D): Ks8[i,j]=K[bh,k*BN+i,j]
                for i,j in T.Parallel(BN,D): Ks[i,j]=T.cast(K[bh,k*BN+i,j],"float16")
                for i,j in T.Parallel(BN,D): Vs[i,j]=T.cast(V[bh,k*BN+i,j],"float16")
                T.gemm(Qs,Ks,T.alloc_fragment((BM,BN),"float32"),transpose_B=True,clear_accum=True) if False else None
                T.gemm(Qs,Ks,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                for i,j in T.Parallel(BM,BN): S[i,j]=T.if_then_else((k*BN+j)<=(bq*BM+i),S[i,j]*scale,-T.infinity("float32"))
                T.reduce_max(S,rowmax,dim=1,clear=True)
                for i in T.Parallel(BM): m_i[i]=T.max(m_prev[i],rowmax[i])
                for i,j in T.Parallel(BM,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
                for i,j in T.Parallel(BM,BN): S[i,j]=T.exp2(S[i,j]-m_i[i])
                T.reduce_sum(S,rowsum,dim=1)
                for i in T.Parallel(BM): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rowsum[i]
                for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
                T.gemm(Ps,Vs,acc)
            for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]*T.exp2(m_i[i])
            for i in T.Parallel(BM): l_i[i]=l_i[i]*T.exp2(m_i[i])
            for i,j in T.Parallel(BM,D): O[bh,sp,bq*BM+i,j]=acc[i,j]
            for i in T.Parallel(BM): Ls[bh,sp,bq*BM+i]=l_i[i]
    return main
torch.manual_seed(0)
H=16; SEQ=4096
Q16=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
K16=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16); V16=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
ref=torch.nn.functional.scaled_dot_product_attention(Q16.float(),K16.float(),V16.float(),is_causal=True)
FULL=4*H*SEQ*SEQ*D
K8=K16.to(torch.float8_e4m3fn); V8=V16.to(torch.float8_e4m3fn)
for qdt,kvdt,Q,K,V,tag,SPLIT in [
    ("float16","float16",Q16,K16,V16,"fp16 SK1",1),
    ("float16","float16",Q16,K16,V16,"fp16 SK4",4),
    ("float8_e4m3","float8_e4m3",Q16.to(torch.float8_e4m3fn),K8,V8,"fp8 SK1",1),
    ("float8_e4m3","float8_e4m3",Q16.to(torch.float8_e4m3fn),K8,V8,"fp8 SK4",4)]:
  try:
    k=tilelang.compile(fa(H,SEQ,64,128,qdt,kvdt,SPLIT),out_idx=[3,4],target="cuda")
    O,L=k(Q,K,V); torch.cuda.synchronize()
    if SPLIT==1: Oc=O.float().sum(1)/L.float().sum(1).unsqueeze(-1)
    else: Oc=O.float().sum(2)/L.float().sum(2).unsqueeze(-1)
    cos=torch.nn.functional.cosine_similarity(Oc.flatten(),ref.flatten(),dim=0).item()
    t=do_bench(lambda: k(Q,K,V),warmup=15,rep=50)
    print(f"{tag}: {t*1000:7.1f}us effTF={FULL*0.5/(t*1e-3)/1e12:.1f} cos={cos:.5f}")
  except Exception as e: print(f"{tag}: FAIL {str(e)[:130]}")