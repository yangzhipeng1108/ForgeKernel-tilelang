import sys, json, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch, tilelang
import tilelang.language as T

M = N = K = 4096
BM, BN, BK = (int(x) for x in sys.argv[1:4])
STAGES = int(sys.argv[4])
POLICY = sys.argv[5] if len(sys.argv) > 5 else "Square"

@T.prim_func
def matmul(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
           C: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        a_s = T.alloc_shared((BM, BK), "float16"); b_s = T.alloc_shared((BK, BN), "float16")
        acc = T.alloc_fragment((BM, BN), "float32"); T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=STAGES):
            T.copy(A[by*BM, k*BK], a_s); T.copy(B[k*BK, bx*BN], b_s)
            T.gemm(a_s, b_s, acc, policy=getattr(T.GemmWarpPolicy, POLICY))
        T.copy(acc, C[by*BM, bx*BN])

try:
    ker = tilelang.compile(matmul, out_idx=[2], target={"kind":"cuda","arch":"sm_120a"})
except Exception as e:
    print("RESULT", json.dumps({"cfg":[BM,BN,BK,STAGES,POLICY],"status":"COMPILE_FAIL","err":str(e).strip().splitlines()[-1][:180]})); sys.exit(0)

torch.manual_seed(0)
a=torch.randn((M,K),device="cuda",dtype=torch.float16); b=torch.randn((K,N),device="cuda",dtype=torch.float16)
call=(lambda: ker(a,b)) if not isinstance(ker,tuple) else (lambda: ker[0](a,b))
try: c=call()
except Exception as e:
    print("RESULT", json.dumps({"cfg":[BM,BN,BK,STAGES,POLICY],"status":"RUN_FAIL","err":str(e).strip().splitlines()[-1][:180]})); sys.exit(0)
maxdiff=(c.float()-(a.float()@b.float())).abs().max().item()

for _ in range(30): call()
torch.cuda.synchronize()
# CUDA-event timing, one sync per block of iters
best=None
for trial in range(15):
    ev0=torch.cuda.Event(enable_timing=True); ev1=torch.cuda.Event(enable_timing=True)
    NITER=20
    ev0.record()
    for _ in range(NITER): call()
    ev1.record(); torch.cuda.synchronize()
    ms=ev0.elapsed_time(ev1)/NITER
    best=ms if best is None else min(best,ms)
flops=2*M*N*K
print("RESULT", json.dumps({"cfg":[BM,BN,BK,STAGES,POLICY],"status":"OK",
    "best_ms":round(best,4),"tflops":round(flops/(best*1e-3)/1e12,2),
    "max_diff":maxdiff,"smem_kib":round(STAGES*(BM*BK+BK*BN)*2/1024,1)}))