import tilelang, tilelang.language as T, torch, threading
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
data={}; graphs={}; streams={}
for d in (0,1):
    with torch.cuda.device(d):
        A=torch.randn(M,K,device=f"cuda:{d}",dtype=torch.float16)*0.1
        B=torch.randn(K,N,device=f"cuda:{d}",dtype=torch.float16)*0.1
        C=torch.empty(M,N,device=f"cuda:{d}",dtype=torch.float16)
        data[d]=(A,B,C)
        k(A,B)  # warm
        torch.cuda.synchronize(d)
        # capture graph
        s=torch.cuda.Stream(device=d)
        with torch.cuda.stream(s):
            for _ in range(3): k(A,B)
        torch.cuda.synchronize(d)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            k(A,B)
        graphs[d]=g; streams[d]=s
torch.cuda.synchronize()
A,B,C=data[0]; k(A,B); torch.cuda.synchronize(0)
t1=do_bench(lambda:k(A,B),warmup=20,rep=50)
print(f"1 GPU (jit call): {t1*1000:.1f}us {FLOP/(t1*1e-3)/1e12:.1f}TF")
REPS=50
def timed_run(d,use_graph,res):
    torch.cuda.set_device(d); A,B,C=data[d]
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    if use_graph:
        for _ in range(REPS): graphs[d].replay()
    else:
        for _ in range(REPS): k(A,B)
    en.record(); en.synchronize()
    res[d]=st.elapsed_time(en)/REPS
# threaded concurrent, jit calls
res={}
ths=[threading.Thread(target=timed_run,args=(d,False,res)) for d in (0,1)]
for t in ths:t.start()
for t in ths:t.join()
ts=max(res.values())
print(f"2 GPU THREADED jit-call: {ts*1000:.1f}us aggregate {2*FLOP/(ts*1e-3)/1e12:.1f}TF scaling {2*FLOP/(ts*1e-3)/(FLOP/(t1*1e-3)):.2f}x")
# threaded concurrent, graph replay
res={}
ths=[threading.Thread(target=timed_run,args=(d,True,res)) for d in (0,1)]
for t in ths:t.start()
for t in ths:t.join()
ts=max(res.values())
print(f"2 GPU THREADED graph-replay: {ts*1000:.1f}us aggregate {2*FLOP/(ts*1e-3)/1e12:.1f}TF scaling {2*FLOP/(ts*1e-3)/(FLOP/(t1*1e-3)):.2f}x")