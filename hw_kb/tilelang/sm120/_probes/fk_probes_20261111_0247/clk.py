import os,json,threading,time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch,tilelang,tilelang.language as T,subprocess
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
for _ in range(50): ker(a,b)
torch.cuda.synchronize()
clocks=[]
stop=False
def poll():
    while not stop:
        o=subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,clocks.mem,power.draw","--format=csv,noheader","-i","0"],capture_output=True,text=True).stdout.strip()
        clocks.append(o); time.sleep(0.05)
t=threading.Thread(target=poll); t.start()
ts=[]
for _ in range(150):
    e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record();ker(a,b);e1.record();torch.cuda.synchronize();ts.append(e0.elapsed_time(e1))
stop=True;t.join()
ts_s=sorted(ts)
print("RESULT times",json.dumps({"min":round(ts_s[0],3),"med":round(ts_s[len(ts_s)//2],3),"max":round(ts_s[-1],3)}))
from collections import Counter
cl=Counter(c.split(",")[0] for c in clocks)
print("RESULT clocks",json.dumps(dict(cl)))
print("RESULT power",json.dumps(sorted(set(c.split(",")[2] for c in clocks))[:8]))