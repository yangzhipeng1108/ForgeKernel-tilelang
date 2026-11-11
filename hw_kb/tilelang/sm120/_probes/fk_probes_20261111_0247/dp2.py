import tilelang, tilelang.language as T, torch, threading, time
from tilelang.profiler import do_bench
def gemm(M,N,K,BM=128,BN=128,BK=64):
    @T.prim_func
    def main(A:T.Tensor((M,K),"float16"),B:T.Tensor((K,N),"float16"),C:T.Tensor((M,N),"float16")):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
            a_s=T.alloc_shared((BM,BK),"float16"); b_s=T.alloc_shared((BK,BN),"float16")
            acc=T.alloc_fragment((BM,BN),"float32"); T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(A[by*BM,kk*BK],a_s); T.copy(B[kk*BK,bx*BN],b_s)
                T.gemm(a_s,b_s,acc)
            T.copy(acc,C[by*BM,bx*BN])
    return main
M=N=K=4096; FLOP=2*M*N*K
k=tilelang.compile(gemm(M,N,K),out_idx=[2],target="cuda")
data={}
for d in (0,1):
    A=torch.randn(M,K,device=f"cuda:{d}",dtype=torch.float16)*0.1
    B=torch.randn(K,N,device=f"cuda:{d}",dtype=torch.float16)*0.1
    C=torch.empty(M,N,device=f"cuda:{d}",dtype=torch.float16)
    data[d]=(A,B,C)
A,B,C=data[0]; k(A,B); torch.cuda.synchronize(0)
t1=do_bench(lambda:k(A,B),warmup=20,rep=50)
print(f"1 GPU: {t1*1000:.1f}us {FLOP/(t1*1e-3)/1e12:.1f}TF")
for d in (0,1):
    A,B,C=data[d]; torch.cuda.set_device(d); k(A,B)
torch.cuda.synchronize()
REPS=50
def timed(d,res):
    torch.cuda.set_device(d); A,B,C=data[d]
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(REPS): k(A,B)
    en.record(); en.synchronize()
    res[d]=(st.elapsed_time(en))/REPS
res={}
ths=[threading.Thread(target=timed,args=(d,res)) for d in (0,1)]
for t in ths:t.start()
for t in ths:t.join()
ts=max(res.values())
print(f"2 GPU concurrent (full 4096^3 each): {ts*1000:.1f}us -> aggregate {2*FLOP/(ts*1e-3)/1e12:.1f}TF  scaling={2*FLOP/(ts*1e-3)/(FLOP/(t1*1e-3)):.2f}x (ideal 2.0)")