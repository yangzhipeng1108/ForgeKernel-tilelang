import tilelang, tilelang.language as T, torch as th, torch
d="float16"
def blocksparse(M,N,K,BM,BN,BK,NT,dt="float16"):
    @T.prim_func
    def main(A:T.Tensor((M,K),dt),B:T.Tensor((K,N),dt),C:T.Tensor((M,N),dt),IDX:T.Tensor((NT,2),"int32")):
        with T.Kernel(NT,threads=128) as (bid,):
            As=T.alloc_shared((BM,BK),dt); Bs=T.alloc_shared((BK,BN),dt)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            mi=IDX[bid,0]; ni=IDX[bid,1]
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(A[mi*BM,k*BK],As); T.copy(B[k*BK,ni*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[mi*BM,ni*BN])
    return main
def dense(M,N,K,BM,BN,BK,dt="float16"):
    @T.prim_func
    def main(A:T.Tensor((M,K),dt),B:T.Tensor((K,N),dt),C:T.Tensor((M,N),dt)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
            As=T.alloc_shared((BM,BK),dt); Bs=T.alloc_shared((BK,BN),dt)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return main
th.manual_seed(0); M=N=K=4096; BM=BN=128; BK=64; tm=tn=M//BM
def timeit(k,args,n=30):
    for _ in range(3): k(*args)
    th.cuda.synchronize(); e0=th.cuda.Event(True); e1=th.cuda.Event(True); e0.record()
    for _ in range(n): k(*args)
    e1.record(); th.cuda.synchronize(); return e0.elapsed_time(e1)/n
A=th.randn(M,K,device="cuda",dtype=th.float16); B=th.randn(K,N,device="cuda",dtype=th.float16)
kd=tilelang.compile(dense(M,N,K,BM,BN,BK),out_idx=[2],target="cuda")
Cd=kd(A,B); th.cuda.synchronize()
md=timeit(kd,(A,B))
print(f"dense 256 tiles: {md*1000:.0f}us {2*M*N*K/(md*1e-3)/1e12:.0f}TF")
for frac in (0.5,0.75,0.25):
    mask=th.zeros(tm,tn,dtype=torch.bool,device="cuda")
    cnt=int((tm*tn)*frac); flat=mask.view(-1)
    flat[:cnt]=True
    mask=mask.view(tm,tn)
    idx=torch.nonzero(mask).to(torch.int32).contiguous(); NT=idx.shape[0]
    kb=tilelang.compile(blocksparse(M,N,K,BM,BN,BK,NT),out_idx=[2],target="cuda")
    Cb=kb(A,B,idx); th.cuda.synchronize()
    mb=timeit(kb,(A,B,idx))
    work=NT*(2*BM*BN*K); eff=work/(mb*1e-3)/1e12
    ok=all(th.allclose(Cd[i*BM:(i+1)*BM,j*BN:(j+1)*BN],Cb[i*BM:(i+1)*BM,j*BN:(j+1)*BN],atol=1e-1) for i in range(tm) for j in range(tn) if mask[i,j])
    print(f"blocksparse {int(frac*100)}% ({NT} tiles): {mb*1000:.0f}us eff={eff:.0f}TF match={ok}")
print("dense baseline 100%: %.0f us %.0fTF" % (md*1000, 2*M*N*K/(md*1e-3)/1e12))