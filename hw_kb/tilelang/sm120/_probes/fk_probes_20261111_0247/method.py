import os,json
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
for _ in range(30): ker(a,b)
torch.cuda.synchronize()
flops=2*M*M*M
# method A: one big event over N back-to-back launches (amortize host)
def amethod(NIT):
    best=9e9
    for _ in range(8):
        e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
        for _ in range(NIT): ker(a,b)
        e1.record();torch.cuda.synchronize()
        best=min(best,e0.elapsed_time(e1)/NIT)
    return best
for nit in (10,50,200):
    ms=amethod(nit); print("RESULT batch",nit,"ms",round(ms,4),"tflops",round(flops/(ms*1e-3)/1e12,1))
# method B: cuda graph replay
g=torch.cuda.CUDAGraph()
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): ker(a,b)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    for _ in range(20): ker(a,b)
torch.cuda.synchronize()
best=9e9
for _ in range(8):
    e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record();g.replay();e1.record();torch.cuda.synchronize()
    best=min(best,e0.elapsed_time(e1)/20)
print("RESULT graph ms",round(best,4),"tflops",round(flops/(best*1e-3)/1e12,1))