import sys,os,json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch,tilelang,tilelang.language as T
M=N=K=4096
BM,BN,BK=map(int,sys.argv[1:4]); ST=int(sys.argv[4]); DT=sys.argv[5]
@T.prim_func
def mm(A:T.Tensor((M,K),DT),B:T.Tensor((K,N),DT),C:T.Tensor((M,N),"float32")):
    with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
        a=T.alloc_shared((BM,BK),DT); b=T.alloc_shared((BK,BN),DT)
        acc=T.alloc_fragment((BM,BN),"float32"); T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=ST):
            T.copy(A[by*BM,k*BK],a); T.copy(B[k*BK,bx*BN],b); T.gemm(a,b,acc)
        T.copy(acc,C[by*BM,bx*BN])
tag={"float16":"f16","float8_e4m3":"fp8"}[DT]
bin_=1 if "e4m3" in DT else 2
try:
    ker=tilelang.compile(mm,out_idx=[2],target={"kind":"cuda","arch":"sm_120a"})
except Exception as e:
    print("RESULT",json.dumps({"dt":tag,"tile":[BM,BN,BK],"ST":ST,"status":"COMPILE_FAIL","err":str(e).strip().splitlines()[-1][:140]}));sys.exit(0)
tdt=torch.float16 if DT=="float16" else torch.float8_e4m3fn
torch.manual_seed(0)
a=torch.randn((M,K),device="cuda",dtype=torch.float32).to(tdt)
b=torch.randn((K,N),device="cuda",dtype=torch.float32).to(tdt)
try: c=ker(a,b)
except Exception as e:
    print("RESULT",json.dumps({"dt":tag,"tile":[BM,BN,BK],"ST":ST,"status":"RUN_FAIL","err":str(e).strip().splitlines()[-1][:140]}));sys.exit(0)
ref=a.float()@b.float()
cos=torch.nn.functional.cosine_similarity(c.float().flatten(),ref.flatten(),dim=0).item()
for _ in range(20): ker(a,b)
torch.cuda.synchronize();best=9e9
for _ in range(12):
    e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(15): ker(a,b)
    e1.record();torch.cuda.synchronize();best=min(best,e0.elapsed_time(e1)/15)
smem=ST*(BM*BK+BK*BN)*bin_/1024
print("RESULT",json.dumps({"dt":tag,"tile":[BM,BN,BK],"ST":ST,"status":"OK","best_ms":round(best,4),
    "tflops":round(2*M*M*M/(best*1e-3)/1e12,1),"cos":round(cos,5),"smem_kib":round(smem,1)}))