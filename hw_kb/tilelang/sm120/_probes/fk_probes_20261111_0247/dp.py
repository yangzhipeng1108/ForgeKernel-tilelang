import tilelang, tilelang.language as T, torch, time
from tilelang.profiler import do_bench
def gemm(M,N,K,BM=128,BN=128,BK=64):
    @T.prim_func
    def main(A:T.Tensor((M,K),"float16"),B:T.Tensor((K,N),"float16"),C:T.Tensor((M,N),"float16")):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
            a_s=T.alloc_shared((BM,BK),"float16"); b_s=T.alloc_shared((BK,BN),"float16")
            acc=T.alloc_fragment((BM,BN),"float32"); T.clear(acc)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(A[by*BM,k*BK],a_s); T.copy(B[k*BK,bx*BN],b_s)
                T.gemm(a_s,b_s,acc)
            T.copy(acc,C[by*BM,bx*BN])
    return main
M=N=K=4096
k=tilelang.compile(gemm(M,N,K),out_idx=[2],target="cuda")
# per-device data
data={}
for d in (0,1):
    A=torch.randn(M,K,device=f"cuda:{d}",dtype=torch.float16)*0.1
    B=torch.randn(K,N,device=f"cuda:{d}",dtype=torch.float16)*0.1
    C=torch.empty(M,N,device=f"cuda:{d}",dtype=torch.float16)
    data[d]=(A,B,C)
FLOP=2*M*N*K
# single-GPU (device 0)
A,B,C=data[0]
k(A,B); torch.cuda.synchronize(0)
t1=do_bench(lambda:k(A,B),warmup=20,rep=50)
print(f"1 GPU: {t1*1000:.1f}us {FLOP/(t1*1e-3)/1e12:.1f}TF")
# dual concurrent (2 threads, each pinned to a device)
import threading
def run(d,reps,barrier):
    torch.cuda.set_device(d)
    A,B,C=data[d]
    barrier.wait()
    for _ in range(reps): k(A,B)
    torch.cuda.synchronize(d)
# warm
for d in (0,1):
    A,B,C=data[d]; torch.cuda.set_device(d); k(A,B); torch.cuda.synchronize(d)
# timed concurrent: use events per device
evs={d:(torch.cuda.Event(True),torch.cuda.Event(True)) for d in (0,1)}
barrier=threading.Barrier(2)
def timed(d):
    torch.cuda.set_device(d); A,B,C=data[d]
    st,en=evs[d]; st.record()
    for _ in range(50): k(A,B)
    en.record()
barrier=threading.Barrier(2)
ths=[threading.Thread(target=timed,args=(d,)) for d in (0,1)]
for t in ths:t.start()
for t in ths:t.join()
torch.cuda.synchronize()
ts=max(evs[d][0].elapsed_time(evs[d][1]) for d in (0,1))/50  # ms
print(f"2 GPU concurrent (each full 4096^3): {ts*1000:.1f}us -> aggregate {2*FLOP/(ts*1e-3)/1e12:.1f}TF  scaling={2*FLOP/(ts*1e-3)/(FLOP/(t1*1e-3)):.2f}x")