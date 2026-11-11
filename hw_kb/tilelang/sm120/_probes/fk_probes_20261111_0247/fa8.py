import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
H=16; SEQ=2048; D=128; L2E=math.log2(math.e)
torch.manual_seed(0)
Q=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
K=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16); V=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
# also test short seq (decode-like) and long seq
for SEQ2 in (512,8192):
    H2=16
    def fa(BM,BN,threads,stages):
        scale=1.0/math.sqrt(D)*L2E
        @T.prim_func
        def main(Q:T.Tensor((H2,SEQ2,D),"float16"),K:T.Tensor((H2,SEQ2,D),"float16"),V:T.Tensor((H2,SEQ2,D),"float16"),O:T.Tensor((H2,SEQ2,D),"float16")):
            with T.Kernel(T.ceildiv(SEQ2,BM),H2,threads=threads) as (bx,bh):
                Qs=T.alloc_shared((BM,D),"float16"); Ks=T.alloc_shared((BN,D),"float16"); Vs=T.alloc_shared((BN,D),"float16")
                Ps=T.alloc_shared((BM,BN),"float16")
                acc=T.alloc_fragment((BM,D),"float32")
                m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
                m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
                S=T.alloc_fragment((BM,BN),"float32")
                rowmax=T.alloc_fragment((BM,),"float32"); rowsum=T.alloc_fragment((BM,),"float32")
                T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
                T.copy(Q[bh,bx*BM,0],Qs)
                for k in T.Pipelined(T.ceildiv(SEQ2,BN),num_stages=stages):
                    T.copy(K[bh,k*BN,0],Ks); T.copy(V[bh,k*BN,0],Vs)
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
                T.copy(acc,O[bh,bx*BM,0])
        return main
    q=Q[:, :SEQ2].contiguous(); kk=K[:, :SEQ2].contiguous(); vv=V[:, :SEQ2].contiguous()
    FLOP=2*H2*SEQ2*SEQ2*D*2*2
    ts=do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q,kk,vv),warmup=25,rep=100)
    try:
        k=tilelang.compile(fa(64,128,128,1),out_idx=[3],target="cuda")
        for _ in range(3): k(q,kk,vv)
        t=do_bench(lambda: k(q,kk,vv),warmup=25,rep=100)
        print(f"SEQ={SEQ2}: SDPA {FLOP/(ts*1e-3)/1e12:.0f}TF | FA {FLOP/(t*1e-3)/1e12:.0f}TF | {ts/t:.2f}x")
    except Exception as e: print(f"SEQ={SEQ2}: FAIL {str(e)[:100]}")