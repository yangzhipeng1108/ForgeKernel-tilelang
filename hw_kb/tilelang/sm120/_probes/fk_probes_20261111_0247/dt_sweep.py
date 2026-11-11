import sys,os,json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch,tilelang,tilelang.language as T
M=N=K=4096
BM,BN,BK=128,128,32
ST=2
DT=sys.argv[1]           # float16 / bfloat16 / float8_e4m3
ACC=sys.argv[2] if len(sys.argv)>2 else "float32"

@T.prim_func
def mm(A:T.Tensor((M,K),DT),B:T.Tensor((K,N),DT),C:T.Tensor((M,N),ACC)):
    with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
        a=T.alloc_shared((BM,BK),DT); b=T.alloc_shared((BK,BN),DT)
        acc=T.alloc_fragment((BM,BN),ACC); T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=ST):
            T.copy(A[by*BM,k*BK],a); T.copy(B[k*BK,bx*BN],b); T.gemm(a,b,acc)
        T.copy(acc,C[by*BM,bx*BN])

try:
    ker=tilelang.compile(mm,out_idx=[2],target={"kind":"cuda","arch":"sm_120a"})
except Exception as e:
    print("RESULT",json.dumps({"dt":DT,"acc":ACC,"status":"COMPILE_FAIL","err":str(e).strip().splitlines()[-1][:200]}));sys.exit(0)

tdt={"float16":torch.float16,"bfloat16":torch.bfloat16,"float8_e4m3":torch.float8_e4m3fn}[DT]
tacc={"float32":torch.float32,"float16":torch.float16}[ACC]
torch.manual_seed(0)
a=torch.randn((M,K),device="cuda",dtype=torch.float32).to(tdt)
b=torch.randn((K,N),device="cuda",dtype=torch.float32).to(tdt)
try:
    c=ker(a,b)
except Exception as e:
    print("RESULT",json.dumps({"dt":DT,"acc":ACC,"status":"RUN_FAIL","err":str(e).strip().splitlines()[-1][:200]}));sys.exit(0)
ref=a.float()@b.float()
md=(c.float()-ref).abs().max().item(); rel=((c.float()-ref).abs()/(ref.abs()+1e-3)).median().item()
for _ in range(20): ker(a,b)
torch.cuda.synchronize();best=9e9
for _ in range(10):
    e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(15): ker(a,b)
    e1.record();torch.cuda.synchronize();best=min(best,e0.elapsed_time(e1)/15)
print("RESULT",json.dumps({"dt":DT,"acc":ACC,"status":"OK","best_ms":round(best,4),
    "tflops":round(2*M*M*M/(best*1e-3)/1e12,1),"max_diff":md,"median_rel":round(rel,5),
    "bytes_per_el":{"float16":2,"bfloat16":2,"float8_e4m3":1}[DT]}))