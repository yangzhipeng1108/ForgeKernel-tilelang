import os, torch, torch.distributed as dist, tilelang, tilelang.language as T
from tilelang.profiler import do_bench
lr=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl"); r=dist.get_rank(); w=dist.get_world_size()
M=int(os.environ["M"]);N=int(os.environ["N"]);K=int(os.environ["K"]); Ks=K//w
def gemm(Min,Nin,Kin,BM=128,BN=128,BK=64,stages=2,threads=128):
    @T.prim_func
    def main(A:T.Tensor((Min,Kin),"float16"),B:T.Tensor((Kin,Nin),"float16"),C:T.Tensor((Min,Nin),"float32")):
        with T.Kernel(T.ceildiv(Nin,BN),T.ceildiv(Min,BM),threads=threads) as (bx,by):
            a_s=T.alloc_shared((BM,BK),"float16"); b_s=T.alloc_shared((BK,BN),"float16")
            acc=T.alloc_fragment((BM,BN),"float32"); T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(Kin,BK),num_stages=stages):
                T.copy(A[by*BM,kk*BK],a_s); T.copy(B[kk*BK,bx*BN],b_s); T.gemm(a_s,b_s,acc)
            T.copy(acc,C[by*BM,bx*BN])
    return main
kfull=tilelang.compile(gemm(M,N,K),out_idx=[2],target="cuda")
kpart=tilelang.compile(gemm(M,N,Ks),out_idx=[2],target="cuda")
dev=torch.device("cuda",lr)
torch.manual_seed(0)
A=torch.randn(M,K,device=dev,dtype=torch.float16)*0.1; B=torch.randn(K,N,device=dev,dtype=torch.float16)*0.1
# single gpu ref
t1=do_bench(lambda:kfull(A,B),warmup=15,rep=30)
dist.barrier()
# TP over K: rank r owns A[:, r*Ks:(r+1)*Ks] and B[r*Ks:(r+1)*Ks, :]
Aslice=A[:, r*Ks:(r+1)*Ks].contiguous(); Bslice=B[r*Ks:(r+1)*Ks].contiguous()
def tp_step():
    C=kpart(Aslice,Bslice)
    dist.all_reduce(C)
    return C
for _ in range(5): tp_step()
torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True)
s.record()
for _ in range(20): tp_step()
e.record(); e.synchronize(); tpms=s.elapsed_time(e)/20
# correctness vs full on rank0(compare C of tp vs kfull)
Ctp=tp_step(); dist.barrier()
flop=2*M*N*K
if r==0:
    Cfull=kfull(A,B)
    cos=torch.nn.functional.cosine_similarity(Ctp.flatten(),Cfull.flatten(),dim=0).item()
    print(f"world={w} {M}x{N}x{K} | 1GPU {t1*1000:.1f}us {flop/(t1*1e-3)/1e12:.1f}TF | TP {tpms*1000:.1f}us speedup {t1/tpms:.2f}x | cos={cos:.6f}")
dist.barrier(); dist.destroy_process_group()
