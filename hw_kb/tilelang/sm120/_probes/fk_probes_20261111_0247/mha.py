import tilelang, tilelang.language as T, torch, math
D=128; L2E=math.log2(math.e)
HQ=32; HKV=32; QLEN=16; KVLEN=8192
scale=1.0/math.sqrt(D)*L2E
@T.prim_func
def main(Q:T.Tensor((HQ,QLEN,D),"float16"),K:T.Tensor((HKV,KVLEN,D),"float16"),V:T.Tensor((HKV,KVLEN,D),"float16"),O:T.Tensor((HQ,QLEN,D),"float16")):
    with T.Kernel(HQ,T.ceildiv(QLEN,16),threads=128) as (bh,bx):
        kvh=bh
        Qs=T.alloc_shared((16,D),"float16"); Ks=T.alloc_shared((128,D),"float16"); Vs=T.alloc_shared((128,D),"float16")
        Ps=T.alloc_shared((16,128),"float16")
        acc=T.alloc_fragment((16,D),"float32")
        m_i=T.alloc_fragment((16,),"float32"); l_i=T.alloc_fragment((16,),"float32")
        m_prev=T.alloc_fragment((16,),"float32"); l_prev=T.alloc_fragment((16,),"float32")
        S=T.alloc_fragment((16,128),"float32"); rowmax=T.alloc_fragment((16,),"float32"); rowsum=T.alloc_fragment((16,),"float32")
        T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
        T.copy(Q[bh,bx*16,0],Qs)
        for k in T.Pipelined(T.ceildiv(KVLEN,128),num_stages=1):
            T.copy(K[kvh,k*128,0],Ks); T.copy(V[kvh,k*128,0],Vs)
            T.gemm(Qs,Ks,S,transpose_B=True,clear_accum=True)
            T.copy(m_i,m_prev); T.copy(l_i,l_prev)
            for i,j in T.Parallel(16,128): S[i,j]=S[i,j]*scale
            T.reduce_max(S,rowmax,dim=1,clear=True)
            for i in T.Parallel(16): m_i[i]=T.max(m_prev[i],rowmax[i])
            for i,j in T.Parallel(16,128): Ps[i,j]=T.exp2(S[i,j]-m_i[i])
            for i,j in T.Parallel(16,128): S[i,j]=T.exp2(S[i,j]-m_i[i])
            T.reduce_sum(S,rowsum,dim=1)
            for i in T.Parallel(16): l_i[i]=l_prev[i]*T.exp2(m_prev[i]-m_i[i])+rowsum[i]
            for i,j in T.Parallel(16,D): acc[i,j]=acc[i,j]*T.exp2(m_prev[i]-m_i[i])
            T.gemm(Ps,Vs,acc)
        for i,j in T.Parallel(16,D): acc[i,j]=acc[i,j]/l_i[i]
        T.copy(acc,O[bh,bx*16,0])
k=tilelang.compile(main,out_idx=[3],target="cuda")
torch.manual_seed(0)
Q=torch.randn(HQ,QLEN,D,device="cuda",dtype=torch.float16)
K=torch.randn(HKV,KVLEN,D,device="cuda",dtype=torch.float16); V=torch.randn(HKV,KVLEN,D,device="cuda",dtype=torch.float16)
for _ in range(3): k(Q,K,V)
torch.cuda.synchronize()
print("READY")