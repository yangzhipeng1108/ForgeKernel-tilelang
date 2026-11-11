import tilelang, tilelang.language as T, torch, math
from tilelang.profiler import do_bench
D=128; L2E=math.log2(math.e)
def mha(HQ,KVLEN,BM,BN,swz=None,order="row",stages=1,splitk=1):
    scale=1.0/math.sqrt(D)*L2E
    @T.prim_func
    def main(Q:T.Tensor((HQ,BM,D),"float16"),K:T.Tensor((HQ,KVLEN,D),"float16"),V:T.Tensor((HQ,KVLEN,D),"float16"),O:T.Tensor((HQ,BM,D),"float16")):
        with T.Kernel(HQ,T.ceildiv(BM,BM),threads=128) as (bh,bx):
            if swz: T.use_swizzle(panel_size=swz,order=order)
            Qs=T.alloc_shared((BM,D),"float16"); Ks=T.alloc_shared((BN,D),"float16"); Vs=T.alloc_shared((BN,D),"float16")
            Ps=T.alloc_shared((BM,BN),"float16")
            acc=T.alloc_fragment((BM,D),"float32")
            m_i=T.alloc_fragment((BM,),"float32"); l_i=T.alloc_fragment((BM,),"float32")
            m_prev=T.alloc_fragment((BM,),"float32"); l_prev=T.alloc_fragment((BM,),"float32")
            S=T.alloc_fragment((BM,BN),"float32"); rowmax=T.alloc_fragment((BM,),"float32"); rowsum=T.alloc_fragment((BM,),"float32")
            T.fill(acc,0); T.fill(l_i,0); T.fill(m_i,-T.infinity("float32"))
            T.copy(Q[bh,bx*BM,0],Qs)
            for k in T.Pipelined(T.ceildiv(KVLEN,BN),num_stages=stages):
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
torch.manual_seed(0)
HQ=32; QLEN=16; KVLEN=8192
Q=torch.randn(HQ,QLEN,D,device="cuda",dtype=torch.float16)
K=torch.randn(HQ,KVLEN,D,device="cuda",dtype=torch.float16); V=torch.randn(HQ,KVLEN,D,device="cuda",dtype=torch.float16)
ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),K.float(),V.float())
FLOP=4*HQ*QLEN*KVLEN*D
for tag,args in [("base BN128",(16,128,None)),("BN64",(16,64,None)),("BN256",(16,256,None)),
                 ("BN128 swz4",(16,128,4)),("BN128 swz8",(16,128,8)),("BN128 col",(16,128,4,"column"))]:
    BM=16;BN=args[1];swz=args[2] if len(args)>2 else None;order=args[3] if len(args)>3 else "row"
    try:
        k=tilelang.compile(mha(HQ,KVLEN,BM,BN,swz,order),out_idx=[3],target="cuda")
        out=k(Q,K,V); torch.cuda.synchronize(); cos=torch.nn.functional.cosine_similarity(out.float().flatten(),ref.flatten(),dim=0).item()
        t=do_bench(lambda: k(Q,K,V),warmup=20,rep=100)
        print(f"{tag}: {FLOP/(t*1e-3)/1e12:6.1f}TF {t*1000:7.1f}us cos={cos:.4f}")
    except Exception as e: print(f"{tag}: FAIL {str(e)[:100]}")