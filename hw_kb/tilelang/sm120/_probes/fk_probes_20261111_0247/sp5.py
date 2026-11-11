import tilelang, tilelang.language as T, torch as th
d="float16"
def sp(M,N,K,BM,BN,BK,st=2):
    @T.prim_func
    def main(A:T.Tensor((M,K//2),d),E:T.Tensor((M,K//8),"uint8"),B:T.Tensor((K,N),d),C:T.Tensor((M,N),d)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
            As=T.alloc_shared((BM,BK//2),d); Es=T.alloc_shared((BM,BK//8),"uint8")
            Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=st):
                T.copy(A[by*BM,k*(BK//2)],As); T.copy(E[by*BM,k*(BK//8)],Es); T.copy(B[k*BK,bx*BN],Bs)
                T.gemm_sp(As,Es,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return main
def dns(M,N,K,BM,BN,BK,st=2):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),d)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=st):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return main
th.manual_seed(0)
def bench(M,N,K,BM,BN,BK,tag):
    Ad=th.randn(M,K,device="cuda",dtype=th.float16)
    g=Ad.view(M,K//4,4); keep=g.clone(); keep[...,2:]=0; Adk=keep.view(M,K)
    Asp=g[...,:2].reshape(M,K//2).contiguous()
    E=th.zeros(M,K//8,device="cuda",dtype=th.uint8)
    B=th.randn(K,N,device="cuda",dtype=th.float16)
    ref=(Adk.float()@B.float())
    try:
        ks=tilelang.compile(sp(M,N,K,BM,BN,BK),out_idx=[3],target="cuda")
        os_=ks(Asp,E,B); th.cuda.synchronize()
        coss=th.nn.functional.cosine_similarity(os_.float().flatten(),ref.flatten(),dim=0).item()
        for _ in range(3): ks(Asp,E,B)
        th.cuda.synchronize(); e0=th.cuda.Event(True); e1=th.cuda.Event(True); e0.record()
        for _ in range(30): ks(Asp,E,B)
        e1.record(); th.cuda.synchronize(); mss=e0.elapsed_time(e1)/30
        ts=2*M*N*K/(mss*1e-3)/1e12
    except Exception as e:
        print(f"  {tag} sparse FAIL {str(e)[:120]}"); return
    try:
        kd=tilelang.compile(dns(M,N,K,BM,BN,BK),out_idx=[2],target="cuda")
        for _ in range(3): kd(Adk,B)
        th.cuda.synchronize(); e0=th.cuda.Event(True); e1=th.cuda.Event(True); e0.record()
        for _ in range(30): kd(Adk,B)
        e1.record(); th.cuda.synchronize(); msd=e0.elapsed_time(e1)/30
        td=2*M*N*K/(msd*1e-3)/1e12
    except Exception as e:
        print(f"  {tag} dense FAIL {str(e)[:120]}"); return
    # note: sparse does half the MACs on A but mma.sp logical K==K, so compare wall time.
    # effective "dense-equivalent" throughput = same FLOPs / sparse time
    print(f"  {tag} BM{BM}x{BN}x{BK}: sparse {mss*1000:6.1f}us ({ts:6.1f} TF) cos={coss:.5f} | dense {msd*1000:6.1f}us ({td:6.1f} TF) | speedup {msd/mss:.2f}x")
print("=== 2:4 sparse (mma.sp) vs dense, sm120 f16, threads=128 ===")
bench(512,512,8192,64,64,64,"A")
bench(1024,1024,8192,64,64,64,"B")
bench(4096,4096,4096,128,128,64,"C")