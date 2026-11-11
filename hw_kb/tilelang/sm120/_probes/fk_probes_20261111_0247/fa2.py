import tilelang, tilelang.language as T, torch, math
BM=64; BN=64; dt="float16"
H=4; SEQ=512; D=64
scale=1.0/math.sqrt(D)
@T.prim_func
def main(Q:T.Tensor((H,SEQ,D),dt),K:T.Tensor((H,SEQ,D),dt),V:T.Tensor((H,SEQ,D),dt),O:T.Tensor((H,SEQ,D),dt)):
    with T.Kernel(T.ceildiv(SEQ,BM),H,threads=128) as (bx,bh):
        Qs=T.alloc_shared((BM,D),dt); Ks=T.alloc_shared((BN,D),dt); Vs=T.alloc_shared((BN,D),dt)
        Ps=T.alloc_shared((BM,BN),dt)
        acc=T.alloc_fragment((BM,D),"float32")
        m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
        m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
        S=T.alloc_fragment((BM,BN),"float32")
        rowmax=T.alloc_fragment((BM,),"float32"); rowsum=T.alloc_fragment((BM,),"float32")
        T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
        T.copy(Q[bh,bx*BM,0],Qs)
        for k in T.Pipelined(T.ceildiv(SEQ,BN),num_stages=2):
            T.copy(K[bh,k*BN,0],Ks); T.copy(V[bh,k*BN,0],Vs)
            T.gemm(Qs,Ks,S,transpose_B=True,clear_accum=True)
            T.copy(m_i,m_prev); T.copy(l_i,l_prev)
            for i,j in T.Parallel(BM,BN): S[i,j]=S[i,j]*scale
            T.reduce_max(S,rowmax,dim=1,clear=True)
            for i in T.Parallel(BM): m_i[i]=T.max(m_prev[i],rowmax[i])
            for i,j in T.Parallel(BM,BN): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
            T.reduce_sum(S,rowsum,dim=1)
            for i in T.Parallel(BM): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rowsum[i]
            for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
            T.gemm(Ps,Vs,acc)
        for i,j in T.Parallel(BM,D): acc[i,j]=acc[i,j]/l_i[i]
        T.copy(acc,O[bh,bx*BM,0])
try:
    k=tilelang.compile(main,out_idx=[3],target="cuda")
    print("COMPILE OK")
    torch.manual_seed(0)
    Q=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
    K=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16); V=torch.randn(H,SEQ,D,device="cuda",dtype=torch.float16)
    out=k(Q,K,V); torch.cuda.synchronize()
    ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),K.float(),V.float())
    md=(out.float()-ref).abs().max().item()
    cos=torch.nn.functional.cosine_similarity(out.float().flatten(),ref.flatten(),dim=0).item()
    print(f"md={md:.4g} cos={cos:.6f} mean|out|={out.float().abs().mean():.4f} mean|ref|={ref.abs().mean():.4f}")
except Exception as e:
    print("FAIL:",type(e).__name__); print(str(e)[:1200])