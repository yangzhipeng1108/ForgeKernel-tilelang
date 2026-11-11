import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
D=128; L2E=math.log2(math.e)
# GQA decode: q_len=1 (or small), kv_len large, H_q heads share H_kv kv heads
def gqa_decode(HQ,HKV,KVLEN,BM,BN,DV=128,stages=1):
    G=HQ//HKV
    scale=1.0/math.sqrt(DV)*L2E
    @T.prim_func
    def main(Q:T.Tensor((HQ,BM,DV),"float16"),K:T.Tensor((HKV,KVLEN,DV),"float16"),V:T.Tensor((HKV,KVLEN,DV),"float16"),O:T.Tensor((HQ,BM,DV),"float16")):
        with T.Kernel(HQ,T.ceildiv(BM,BM),threads=128) as (bh,bx):
            kvh=bh//G
            Qs=T.alloc_shared((BM,DV),"float16"); Ks=T.alloc_shared((BN,DV),"float16"); Vs=T.alloc_shared((BN,DV),"float16")
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,DV),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rowmax=T.alloc_fragment((BM,),"float32"); rowsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Q[bh,bx*BM,0],Qs)
            for k in T.Pipelined(T.ceildiv(KVLEN,BN),num_stages=stages):
                T.copy(K[kvh,k*BN,0],Ks); T.copy(V[kvh,k*BN,0],Vs)
                T.gemm(Qs,Ks,S,transpose_B=True,clear_accum=True)
                T.copy(m_i,m_prev); T.copy(l_i,l_prev)
                for i,j in T.Parallel(BM,BN): S[i,j]=S[i,j]*scale
                T.reduce_max(S,rowmax,dim=1,clear=True)
                for i in T.Parallel(BM): m_i[i]=T.max(m_prev[i],rowmax[i])
                for i,j in T.Parallel(BM,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
                for i,j in T.Parallel(BM,BN): S[i,j]=T.exp2(S[i,j]-m_i[i])
                T.reduce_sum(S,rowsum,dim=1)
                for i in T.Parallel(BM): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rowsum[i]
                for i,j in T.Parallel(BM,DV): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
                T.gemm(Ps,Vs,acc)
            for i,j in T.Parallel(BM,DV): acc[i,j]=acc[i,j]/l_i[i]
            T.copy(acc,O[bh,bx*BM,0])
    return main
torch.manual_seed(0)
for (HQ,HKV,KVLEN,qlen) in [(32,8,8192,1),(32,8,8192,16),(32,8,32768,1)]:
    Q=torch.randn(HQ,qlen,D,device="cuda",dtype=torch.float16)
    K=torch.randn(HKV,KVLEN,D,device="cuda",dtype=torch.float16); V=torch.randn(HKV,KVLEN,D,device="cuda",dtype=torch.float16)
    # ref via repeat_kv
    G=HQ//HKV
    Kr=K.repeat_interleave(G,0); Vr=V.repeat_interleave(G,0)
    ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),Kr.float(),Vr.float())
    FLOP=4*HQ*qlen*KVLEN*D
    BYTES=HQ*qlen*D*2 + HKV*KVLEN*D*2*2 + HQ*qlen*D*2  # Q+K+V+O (KV shared!)
    try:
        k=tilelang.compile(gqa_decode(HQ,HKV,KVLEN,qlen,128),out_idx=[3],target="cuda")
        out=k(Q,K,V); torch.cuda.synchronize()
        cos=torch.nn.functional.cosine_similarity(out.float().flatten(),ref.flatten(),dim=0).item()
        t=do_bench(lambda: k(Q,K,V),warmup=20,rep=100)
        ts=do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(Q,Kr,Vr),warmup=20,rep=100)
        print(f"HQ{HQ}/HKV{HKV} KVLEN{KVLEN} q{qlen}: FA {FLOP/(t*1e-3)/1e12:.1f}TF {t*1000:.1f}us cos={cos:.4f} | SDPA {FLOP/(ts*1e-3)/1e12:.1f}TF | {ts/t:.2f}x | BW={BYTES/(t*1e-3)/1e9:.0f}GB/s arith_int={FLOP/BYTES:.1f}")
    except Exception as e: print(f"HQ{HQ} KVLEN{KVLEN} q{qlen}: FAIL {str(e)[:150]}")