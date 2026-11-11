import sys,os,json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch,tilelang,tilelang.language as T
BM,BN,BK,ST=map(int,sys.argv[1:5])
M=int(sys.argv[5]); N=K=M
@T.prim_func
def mm(A:T.Tensor((M,K),"float16"),B:T.Tensor((K,N),"float16"),C:T.Tensor((M,N),"float32")):
    with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
        a=T.alloc_shared((BM,BK),"float16");b=T.alloc_shared((BK,BN),"float16")
        acc=T.alloc_fragment((BM,BN),"float32");T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=ST):
            T.copy(A[by*BM,k*BK],a);T.copy(B[k*BK,bx*BN],b);T.gemm(a,b,acc)
        T.copy(acc,C[by*BM,bx*BN])
ker=tilelang.compile(mm,out_idx=[2],target={"kind":"cuda","arch":"sm_120a"})
a=torch.randn((M,K),device="cuda",dtype=torch.float16);b=torch.randn((K,N),device="cuda",dtype=torch.float16)
for _ in range(20): ker(a,b)
torch.cuda.synchronize()
best=9e9
for _ in range(10):
    e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(15): ker(a,b)
    e1.record();torch.cuda.synchronize();best=min(best,e0.elapsed_time(e1)/15)
grid=(M//BM)*(N//BN)
print("RESULT",json.dumps({"tile":[BM,BN,BK],"M":M,"grid":grid,"best_ms":round(best,3),"tflops":round(2*M*M*M/(best*1e-3)/1e12,1)}))