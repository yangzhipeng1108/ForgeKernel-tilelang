import tilelang, tilelang.language as T, torch as th
d="float16"
# Coarse block-sparse OUTPUT scheduling: only nonzero output tiles are launched.
# Build with a grid = number of nonzero tiles, decode (m,n) from a compacted index list in global mem.
def blocksparse(M,N,K,BM,BN,BK,nnz_tiles_map,tile_m,tile_n,dt="float16"):
    NT=nnz_tiles_map.numel()
    @T.prim_func
    def main(A:T.Tensor((M,K),dt),B:T.Tensor((K,N),dt),C:T.Tensor((M,N),dt),IDX:T.Tensor((NT,2),"int32")):
        with T.Kernel(NT,threads=128) as (bid,):
            As=T.alloc_shared((BM,BK),dt); Bs=T.alloc_shared((BK,BN),dt)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            mi=IDX[bid,0]; ni=IDX[bid,1]
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(A[mi*BM,k*BK],As); T.copy(B[k*BK,ni*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[mi*BM,ni*BN])
    return main
def dense(M,N,K,BM,BN,BK,dt="float16"):
    @T.prim_func
    def main(A:T.Tensor((M,K),dt),B:T.Tensor((K,N),dt),C:T.Tensor((M,N),dt)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
            As=T.alloc_shared((BM,BK),dt); Bs=T.alloc_shared((BK,BN),dt)
            Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*BM,bx*BN])
    return main
th.manual_seed(0)
M=N=4096; K=4096; BM=BN=128; BK=64
tm,tn=M//BM,N//BN
# nonzero output-tile mask: keep 50% of tiles (checkerboard) -> but output tiles nonzero depends on data; use arbitrary mask to measure scheduling
mask=th.zeros(tm,tn,dtype=th.bool,device="cuda"); 
for i in range(tm):
  for j in range(tn):
    if (i+j)%2==0: mask[i,j]=True
idx=torch.nonzero(mask).to(torch.int32).contiguous() if False else None
import torch
idx=torch.nonzero(mask).to(torch.int32).contiguous()
NT=idx.shape[0]
print(f"tiles total {tm*tn} nnz {NT} ({NT/(tm*tn)*100:.0f}%)")
A=th.randn(M,K,device="cuda",dtype=th.float16); B=th.randn(K,N,device="cuda",dtype=th.float16)
def timeit(k,args,n=30):
    for _ in range(3): k(*args)
    th.cuda.synchronize(); e0=th.cuda.Event(True); e1=th.cuda.Event(True); e0.record()
    for _ in range(n): k(*args)
    e1.record(); th.cuda.synchronize(); return e0.elapsed_time(e1)/n
kd=tilelang.compile(dense(M,N,K,BM,BN,BK),out_idx=[2],target="cuda")
md=timeit(kd,(A,B)); print(f"dense all 256 tiles: {md*1000:.1f}us -> {2*M*N*K/(md*1e-3)/1e12:.0f} TF (equiv all work)")
kb=tilelang.compile(blocksparse(M,N,K,BM,BN,BK,idx,tm,tn),out_idx=[2],target="cuda")
mb=timeit(kb,(A,B,idx)); work=NT*(2*BM*BN*K)
print(f"blocksparse {NT} tiles: {mb*1000:.1f}us -> {work/(mb*1e-3)/1e12:.0f} TF on done work | {work/(mb*1e-3)/1e12:.0f} effective")
# verify: compare only the nonzero tiles
Cd=kd(A,B); Cb=kb(A,B,idx); th.cuda.synchronize()
ok=True
for i in range(tm):
  for j in range(tn):
    if mask[i,j]:
      if not th.allclose(Cd[i*BM:(i+1)*BM, j*BN:(j+1)*BN], Cb[i*BM:(i+1)*BM,j*BN:(j+1)*BN], atol=1e-2): ok=False;break
print("nonzero-tiles match dense:", ok)