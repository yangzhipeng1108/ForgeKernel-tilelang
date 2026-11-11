import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch,tilelang,tilelang.language as T
M=N=K=4096
BM,BN,BK=map(int,sys.argv[1:4]); ST=int(sys.argv[4])
@T.prim_func
def mm(A:T.Tensor((M,K),"float16"),B:T.Tensor((K,N),"float16"),C:T.Tensor((M,N),"float32")):
    with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
        a=T.alloc_shared((BM,BK),"float16");b=T.alloc_shared((BK,BN),"float16")
        acc=T.alloc_fragment((BM,BN),"float32");T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=ST):
            T.copy(A[by*BM,k*BK],a);T.copy(B[k*BK,bx*BN],b);T.gemm(a,b,acc)
        T.copy(acc,C[by*BM,bx*BN])
ker=tilelang.compile(mm,out_idx=[2],target={"kind":"cuda","arch":"sm_120a"})
torch.manual_seed(0)
a=torch.randn((M,K),device="cuda",dtype=torch.float16);b=torch.randn((K,N),device="cuda",dtype=torch.float16)
for _ in range(3): ker(a,b)
torch.cuda.synchronize()
ker(a,b)
torch.cuda.synchronize()