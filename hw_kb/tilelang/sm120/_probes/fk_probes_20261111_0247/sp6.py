import tilelang, tilelang.language as T, torch as th
d="float16"
def sp(M,N,K,BM,BN,BK,thr,st=2):
    @T.prim_func
    def main(A:T.Tensor((M,K//2),d),E:T.Tensor((M,K//8),"uint8"),B:T.Tensor((K,N),d),C:T.Tensor((M,N),d)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=thr) as (bx,by):
            As=T.alloc_shared((BM,BK//2),d); Es=T.alloc_shared((BM,BK//8),"uint8")
            Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=st):
                T.copy(A[by*BM,k*(BK//2)],As); T.copy(E[by*BM,k*(BK//8)],Es); T.copy(B[k*BK,bx*BN],Bs)
                T.gemm_sp(As,Es,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return main
def dns(M,N,K,BM,BN,BK,thr,st=2,dt="float8_e4m3"):
    @T.prim_func
    def main(A:T.Tensor((M,K),dt),B:T.Tensor((K,N),dt),C:T.Tensor((M,N),d)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=thr) as (bx,by):
            As=T.alloc_shared((BM,BK),dt); Bs=T.alloc_shared((BK,BN),dt)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=st):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return main
th.manual_seed(0)
def bench(M,N,K,BM,BN,BK,thr):
    Ad=th.randn(M,K,device="cuda",dtype=th.float16)
    g=Ad.view(M,K//4,4); keep=g.clone(); keep[...,2:]=0; Adk=keep.view(M,K)
    Asp=g[...,:2].reshape(M,K//2).contiguous()
    E=th.zeros(M,K//8,device="cuda",dtype=th.uint8)
    B=th.randn(K,N,device="cuda",dtype=th.float16)
    ref=(Adk.float()@B.float()); res={}
    for nm,mk in [("sp16",lambda: sp(M,N,K,BM,BN,BK,thr)),
                  ("dense16",lambda: dns(M,N,K,BM,BN,BK,thr,dt="float16"))]:
        try:
            k=tilelang.compile(mk(),out_idx=[3] if "sp" in nm else [2],target="cuda")
            a=(Asp,E,B) if "sp" in nm else (Adk,B)
            o=k(*a); th.cuda.synchronize(); res[nm]=th.nn.functional.cosine_similarity(o.float().flatten(),ref.flatten(),dim=0).item()
            for _ in range(3): k(*a)
            th.cuda.synchronize(); e0=th.cuda.Event(True); e1=th.cuda.Event(True); e0.record()
            for _ in range(30): k(*a)
            e1.record(); th.cuda.synchronize(); ms=e0.elapsed_time(e1)/30
            res[nm+"_t"]=ms; res[nm+"_tf"]=2*M*N*K/(ms*1e-3)/1e12
        except Exception as e: res[nm]="FAIL:"+str(e)[:80]
    print(f"  {M}x{N}x{K} BM{BM}BN{BN}BK{BK} thr{thr}: "
          f"sp {res.get('sp16_t',0)*1000:.1f}us {res.get('sp16_tf',0):.0f}TF cos{res.get('sp16','-')} | "
          f"dense {res.get('dense16_t',0)*1000:.1f}us {res.get('dense16_tf',0):.0f}TF | "
          f"xp {res.get('dense16_t',1)/max(res.get('sp16_t',1),1e-9):.2f}")
for thr in (128,256):
    bench(4096,4096,4096,128,128,64,thr)
bench(4096,4096,4096,128,128,128,256)
bench(2048,2048,8192,128,128,64,256)