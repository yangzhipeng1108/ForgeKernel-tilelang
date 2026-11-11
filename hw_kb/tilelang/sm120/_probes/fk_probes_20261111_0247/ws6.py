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
A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
for threads in (128,256):
  for ws in (False,True):
    k=build(M,N,K,128,T.float8_e4m3,ws,threads=threads)
    src=k.get_kernel_source()
    lb=re.findall(r"__launch_bounds__\s*\(\s*(\d+)",src)
    regd=len(re.findall(r"warpgroup_reg_dealloc|setmaxnreg|reg_dealloc",src))
    rega=len(re.findall(r"warpgroup_reg_alloc|reg_alloc",src))
    print(f"threads={threads} ws={int(ws)}: launch_bounds={lb} reg_dealloc_sites={regd} reg_alloc_sites={rega} smem_bytes={len(re.findall(r'buf_dyn_shmem',src))}")