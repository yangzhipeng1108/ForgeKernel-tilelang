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
flush=torch.empty(256*1024*1024//4,device="cuda",dtype=torch.float32)  # 256MB > L2
def timeit(clear):
    for _ in range(30): ker(a,b)
    torch.cuda.synchronize()
    ts=[]
    for _ in range(40):
        if clear: flush.zero_()  # evict L2
        torch.cuda.synchronize()
        e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record();ker(a,b);e1.record();torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort(); return ts
warm=timeit(False); cold=timeit(True)
def s(ts): return {"min":round(ts[0],3),"med":round(ts[len(ts)//2],3),"max":round(ts[-1],3)}
print("RESULT warm_noflush",json.dumps(s(warm)))
print("RESULT cold_flush",json.dumps(s(cold)))
flops=2*M*M*M
def tf(ms): return round(flops/(ms*1e-3)/1e12,1)
print("RESULT warm tf min/med",tf(warm[0]),tf(warm[len(warm)//2]))
print("RESULT cold tf min/med",tf(cold[0]),tf(cold[len(cold)//2]))