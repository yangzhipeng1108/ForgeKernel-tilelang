import sys,os,json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch,tilelang,tilelang.language as T
M=N=K=4096; BM,BN,BK=128,128,32; ST=2
EA=sys.argv[1]; EB=sys.argv[2]; DIS=sys.argv[3]=="1"
@T.prim_func
def mm(A:T.Tensor((M,K),"float16"),B:T.Tensor((K,N),"float16"),C:T.Tensor((M,N),"float32")):
    with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
        a=T.alloc_shared((BM,BK),"float16"); b=T.alloc_shared((BK,BN),"float16")
        acc=T.alloc_fragment((BM,BN),"float32"); T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=ST):
            T.copy(A[by*BM,k*BK],a,eviction_policy=(None if EA=="none" else EA),disable_tma=DIS)
            T.copy(B[k*BK,bx*BN],b,eviction_policy=(None if EB=="none" else EB),disable_tma=DIS)
            T.gemm(a,b,acc)
        T.copy(acc,C[by*BM,bx*BN])
try:
    ker=tilelang.compile(mm,out_idx=[2],target={"kind":"cuda","arch":"sm_120a"})
except Exception as e:
    print("RESULT",json.dumps({"EA":EA,"EB":EB,"disable_tma":DIS,"status":"FAIL","err":str(e).strip().splitlines()[-1][:120]}));sys.exit(0)
torch.manual_seed(0)
a=torch.randn((M,K),device="cuda",dtype=torch.float16);b=torch.randn((K,N),device="cuda",dtype=torch.float16)
c=ker(a,b);cos=torch.nn.functional.cosine_similarity(c.float().flatten(),(a.float()@b.float()).flatten(),dim=0).item()
for _ in range(20): ker(a,b)
torch.cuda.synchronize();best=9e9
for _ in range(12):
    e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(15): ker(a,b)
    e1.record();torch.cuda.synchronize();best=min(best,e0.elapsed_time(e1)/15)
print("RESULT",json.dumps({"EA":EA,"EB":EB,"disable_tma":DIS,"status":"OK","best_ms":round(best,4),"tflops":round(2*M*M*M/(best*1e-3)/1e12,1),"cos":round(cos,5)}))