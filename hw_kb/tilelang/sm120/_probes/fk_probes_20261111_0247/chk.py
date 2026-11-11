import torch, tilelang, tilelang.language as T
def build_gemm(M, N, K, BM=128, BN=128, BK=64, stages=2, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            a_s = T.alloc_shared((BM, BK), "float16"); b_s = T.alloc_shared((BK, BN), "float16")
            acc = T.alloc_fragment((BM, BN), "float32"); T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                T.copy(A[by*BM, kk*BK], a_s); T.copy(B[kk*BK, bx*BN], b_s); T.gemm(a_s, b_s, acc)
            T.copy(acc, C[by*BM, bx*BN])
    return main
M=N=K=4096
k=tilelang.compile(build_gemm(M,N,K),out_idx=[2],target="cuda")
ref=None
dev0=0
with torch.cuda.device(0):
    A=torch.randn(M,K,device="cuda:0",dtype=torch.float16)
    B=torch.randn(K,N,device="cuda:0",dtype=torch.float16)
    C0=k(A,B)
    ref=(A.float()@B.float()).half()
    print("dev0 vs torch cos:", torch.nn.functional.cosine_similarity(C0.flatten().float(),ref.flatten().float(),dim=0).item())
# device1 stacked copy parity check (independent kernel on GPU1)
with torch.cuda.device(1):
    A1=A.cuda(1); B1=B.cuda(1); C1=k(A1,B1)
torch.cuda.synchronize()
print("dev1 vs dev0 max abs diff:", (C1.cpu().float()-C0.cpu().float()).abs().max().item())