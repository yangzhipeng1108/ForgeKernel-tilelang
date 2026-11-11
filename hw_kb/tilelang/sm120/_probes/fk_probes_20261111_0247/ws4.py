import tilelang, tilelang.language as T, torch as th, re
BM=BN=128
def build(M,N,K,BK,d,ws,stages=2,threads=256):
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=threads) as (bx,by):
            As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
            Cl=T.alloc_fragment((BM,BN),"float32")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=stages):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    cfg={"tl.enable_auto_warp_specialization":"role_based"} if ws else None
    return tilelang.compile(main,out_idx=[2],target="cuda",pass_configs=cfg)
M=N=K=4096
for ws in (False,True):
    k=build(M,N,K,128,T.float8_e4m3,ws)
    src=k.get_kernel_source()
    smem=[x for x in re.findall(r"(?:extern __shared__|dynamic shared memory)[^\n]*",src)][:2]
    setmax=re.findall(r"setmaxnreg[^;]*;",src)
    barriers=len(re.findall(r"mbarrier|barrier",src))
    blocks=re.search(r"__launch_bounds__\((\d+)\)",src)
    print(f"--- ws={ws} launch_bounds={blocks.group(1) if blocks else None} barriers={barriers}")
    print("  setmaxnreg:", setmax[:4])
    print("  smemlines:", smem)