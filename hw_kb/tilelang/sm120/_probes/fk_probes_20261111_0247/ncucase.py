import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch,tilelang,tilelang.language as T
M=N=K=4096; BM,BN,BK=128,128,32; ST=2
@T.prim_func
def mm(A:T.Tensor((M,K),"float16"),B:T.Tensor((K,N),"float16"),C:T.Tensor((M,N),"float32")):
    with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
        a=T.alloc_shared((BM,BK),"float16"); b=T.alloc_shared((BK,BN),"float16")
        acc=T.alloc_fragment((BM,BN),"float32"); T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=ST):
            T.copy(A[by*BM,k*BK],a); T.copy(B[k*BK,bx*BN],b); T.gemm(a,b,acc)
        T.copy(acc,C[by*BM,bx*BN])
ker=tilelang.compile(mm,out_idx=[2],target={"kind":"cuda","arch":"sm_120a"})
a=torch.randn((M,K),device="cuda",dtype=torch.float16);b=torch.randn((K,N),device="cuda",dtype=torch.float16)
for _ in range(5): ker(a,b)
torch.cuda.synchronize()
for _ in range(6): ker(a,b)
torch.cuda.synchronize()