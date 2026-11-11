import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
D=128; L2E=math.log2(math.e)
def gqa(HQ,HKV,QLEN,KVLEN,BM,BN,q_dt,k_dt,v_dt,stages):
    G=HQ//HKV; scale=1.0/math.sqrt(D)*L2E
    @T.prim_func
    def main(Q:T.Tensor((HQ,QLEN,D),q_dt),K:T.Tensor((HKV,KVLEN,D),k_dt),V:T.Tensor((HKV,KVLEN,D),v_dt),O:T.Tensor((HQ,QLEN,D),"float16")):
        with T.Kernel(HQ,threads=128) as (bh):
            kvh=bh//G
            Qs=T.alloc_shared((BM,D),q_dt); Ks=T.alloc_shared((BN,D),k_dt); Vs=T.alloc_shared((BN,D),v_dt)
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,D),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rmax=T.alloc_fragment((BM,),"float32"); rsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Q[bh,0,0],Qs)
            for k in T.Pipelined(T.ceildiv(KVLEN,BN),num_stages=stages):
                T.copy(K[kvh,k*BN,0],Ks); T.copy(V[kvh,k*BN,0],Vs)
                T.gemm(Qs,Ks,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
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
            T.copy(acc,O[bh,0,0])
    return main
torch.manual_seed(0)
HQ,HKV,QLEN,KVLEN=32,8,16,8192; G=HQ//HKV
Q=torch.randn(HQ,QLEN,D,device="cuda",dtype=torch.float16)
K=torch.randn(HKV,KVLEN,D,device="cuda",dtype=torch.float16); V=torch.randn(HKV,KVLEN,D,device="cuda",dtype=torch.float16)
Kr=K.repeat_interleave(G,0); Vr=V.repeat_interleave(G,0)
ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),Kr.float(),Vr.float())
FLOP=4*HQ*QLEN*KVLEN*D
Q8=Q.to(torch.float8_e4m3fn);K8=K.to(torch.float8_e4m3fn);V8=V.to(torch.float8_e4m3fn)
for tag,Qo,Ko,Vo,qd,kd,vd in [
   ("f16",Q,K,V,"float16","float16","float16"),
   ("f8 QK, f16 V",Q8,K8,V,"float8_e4m3","float8_e4m3","float16"),
   ("f8 QK, f8 V(upcast?)",Q8,K8,V8,"float8_e4m3","float8_e4m3","float8_e4m3")]:
  try:
    k=tilelang.compile(gqa(HQ,HKV,QLEN,KVLEN,16,128,qd,kd,vd,1),out_idx=[3],target="cuda")
    o=k(Qo,Ko,Vo); torch.cuda.synchronize()
    cos=torch.nn.functional.cosine_similarity(o.float().flatten(),ref.flatten(),dim=0).item()
    t=do_bench(lambda: k(Qo,Ko,Vo),warmup=15,rep=50)
    ts=do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(Q,Kr,Vr),warmup=15,rep=50)
    print(f"GQA {tag}: {t*1000:.1f}us {FLOP/(t*1e-3)/1e12:.1f}TF cos={cos:.5f} | SDPA {ts*1000:.1f} | {ts/t:.2f}x")
  except Exception as e: print(f"GQA {tag}: FAIL {str(e)[:110]}")