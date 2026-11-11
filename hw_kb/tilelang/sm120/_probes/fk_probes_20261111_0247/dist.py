import sys,os,json,statistics
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
torch.manual_seed(0)
a=torch.randn((M,K),device="cuda",dtype=torch.float16);b=torch.randn((K,N),device="cuda",dtype=torch.float16)
for _ in range(50): ker(a,b)
torch.cuda.synchronize()
# 200 individual event-timed launches, look at distribution
ts=[]
for _ in range(200):
    e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record();ker(a,b);e1.record();torch.cuda.synchronize()
    ts.append(e0.elapsed_time(e1))
ts.sort()
flops=2*M*M*M
def tf(ms): return round(flops/(ms*1e-3)/1e12,1)
print("RESULT",json.dumps({"n":len(ts),"min_ms":round(ts[0],4),"p10":round(ts[20],4),
  "median_ms":round(ts[100],4),"p90":round(ts[180],4),"max_ms":round(ts[-1],4),
  "tf_best":tf(ts[0]),"tf_median":tf(ts[100]),"tf_p10":tf(ts[20])}))