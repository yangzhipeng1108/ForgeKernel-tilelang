import tilelang, tilelang.language as T, torch, subprocess, os, glob
M=N=K=1024; BM=BN=BK=128
@T.prim_func
def with_clc(A:T.Tensor((M,K),"float16"),B:T.Tensor((K,N),"float16"),C:T.Tensor((M,N),"float16")):
    with T.ClusterKernel(T.ceildiv(N,BN),T.ceildiv(M,BM),cluster_dims=1,threads=256) as (bx,by):
        a_s=T.alloc_shared((BM,BK),"float16"); b_s=T.alloc_shared((BK,BN),"float16")
        acc=T.alloc_fragment((BM,BN),"float32"); T.clear(acc)
        res=T.alloc_shared((16,),"uint32"); bar=T.alloc_barrier(1)
        T.clc_try_cancel(res,bar)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
            T.copy(A[by*BM,k*BK],a_s); T.copy(B[k*BK,bx*BN],b_s)
            T.gemm(a_s,b_s,acc)
        T.copy(acc,C[by*BM,bx*BN])
k=tilelang.compile(with_clc,out_idx=[2],target={"kind":"cuda","arch":"sm_120a"})
# find the cubin/source cache
src = k.get_kernel_source() if hasattr(k,"get_kernel_source") else None
if src:
    on = "CUTLASS_ARCH_CLC_ENABLED" in src
    has_asm = "clusterlaunchcontrol" in src
    print("emitted source has CLC macro ref:", on, "| clusterlaunchcontrol asm in src:", has_asm)
# dump to file and compile SASS via the kernel's own nvcc pipeline: easier -> use k.adapter
# tilelang stores lib path; find .cubin or use get_kernel_source and recompile
torch.manual_seed(0)
A=torch.randn(M,K,device="cuda",dtype=torch.float16); B=torch.randn(K,N,device="cuda",dtype=torch.float16)
C=k(A,B); torch.cuda.synchronize()
ref=(A@B)
md=(C.float()-ref).abs().max().item()
print(f"CLC kernel md={md:.4f}  (fp16 exact ~0.x)")
# locate compiled artifact for SASS
base=os.path.expanduser("~/.tilelang")
cands=glob.glob(base+"/**/*.cubin",recursive=True)+glob.glob(base+"/**/*.so",recursive=True)
print("artifacts found:", len(cands))