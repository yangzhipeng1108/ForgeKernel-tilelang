import tilelang, tilelang.language as T, torch as th, re, time
BM=BN=128; BK=128; d=T.float8_e4m3; M=N=K=4096
@T.prim_func
def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
    with T.ClusterKernel(T.ceildiv(N,BN),T.ceildiv(M,BM),cluster_dims=2,threads=256) as (bx,by):
        As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
        Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
            T.copy_cluster(A[by*BM,k*BK],As,cluster_mask=3)
            T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
        T.copy(Cl,C[by*BM,bx*BN])
k=tilelang.compile(main,out_idx=[2],target="cuda")
src=k.get_kernel_source()
print("multicast refs:", len(re.findall(r"multicast|cluster_mask|mapa|cluster", src)))
for line in src.splitlines():
    if "multicast" in line or "tma_load" in line or "cluster" in line.lower():
        print("  ",line.strip()[:180])
A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
out=k(A,B); th.cuda.synchronize()
import torch
# raw timing
for _ in range(3): k(A,B)
th.cuda.synchronize(); s=time.perf_counter()
for _ in range(20): k(A,B)
th.cuda.synchronize(); dt=(time.perf_counter()-s)/20
print(f"raw: {dt*1000:.4f} ms -> {2*M*N*K/(dt*1e-3)/1e12:.1f} TFLOP/s")