import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
D=128; L2E=math.log2(math.e)
def mha_fp8kv(HQ,KVLEN,BM,BN):
    scale=1.0/math.sqrt(D)*L2E
    @T.prim_func
    def main(Q:T.Tensor((HQ,BM,D),"float16"),K:T.Tensor((HQ,KVLEN,D),"float8_e4m3"),
             V:T.Tensor((HQ,KVLEN,D),"float8_e4m3"),O:T.Tensor((HQ,BM,D),"float16")):
        with T.Kernel(HQ,threads=128) as (bh):
            Qs=T.alloc_shared((BM,D),"float16")
            K8=T.alloc_shared((BN,D),"float8_e4m3"); V8=T.alloc_shared((BN,D),"float8_e4m3")
            Ks=T.alloc_shared((BN,D),"float16"); Vs=T.alloc_shared((BN,D),"float16")
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,D),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rowmax=T.alloc_fragment((BM,),"float32"); rowsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Q[bh,0,0],Qs)
            for k in T.Pipelined(T.ceildiv(KVLEN,BN),num_stages=2):
                T.copy(K[bh,k*BN,0],K8); T.copy(V[bh,k*BN,0],V8)
                T.copy(K8,Ks); T.copy(V8,Vs)
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
            for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]/l_i[i]
            T.copy(acc,O[bh,0,0])
    return main
torch.manual_seed(0)
HQ=32; QLEN=16; KVLEN=8192; BM=16
Q=torch.randn(HQ,QLEN,D,device="cuda",dtype=torch.float16)
Kh=torch.randn(HQ,KVLEN,D,device="cuda",dtype=torch.float16); Vh=torch.randn(HQ,KVLEN,D,device="cuda",dtype=torch.float16)
ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),Kh.float(),Vh.float())
FLOP=4*HQ*QLEN*KVLEN*D
K8=Kh.to(torch.float8_e4m3fn); V8=Vh.to(torch.float8_e4m3fn)
for BN in (64,128):
  try:
    k=tilelang.compile(mha_fp8kv(HQ,KVLEN,BM,BN),out_idx=[3],target="cuda")
    out=k(Q,K8,V8); torch.cuda.synchronize()
    cos=torch.nn.functional.cosine_similarity(out.float().flatten(),ref.flatten(),dim=0).item()
    t=do_bench(lambda: k(Q,K8,V8),warmup=20,rep=100)
    KVb=HQ*KVLEN*D*1*2
    print(f"fp8kv BN{BN}: {t*1000:7.1f}us {FLOP/(t*1e-3)/1e12:6.1f}TF cos={cos:.5f} KVbytes={KVb/1e6:.1f}MB BW={KVb/(t*1e-3)/1e9:.0f}GB/s")
  except Exception as e: print(f"fp8kv BN{BN}: FAIL {str(e)[:110]}")
print(f"(fp16 baseline: 152.6us 14.1TF 134.2MB 880GB/s)")