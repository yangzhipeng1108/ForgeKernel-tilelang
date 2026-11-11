import tilelang, tilelang.language as T, torch
from tilelang.profiler import do_bench
def streamread(NBLK,CHUNK,thr,BN):
    N=CHUNK//BN
    @T.prim_func
    def main(X:T.Tensor((NBLK,CHUNK),"float16"),O:T.Tensor((NBLK,),"float32")):
        with T.Kernel(NBLK,threads=thr) as (bx):
            s=T.alloc_shared((BN,),"float16"); acc=T.alloc_fragment((BN,),"float32")
            tot=T.alloc_fragment((1,),"float32")
            T.fill(tot,0); T.fill(acc,0)
            for k in T.Pipelined(N,num_stages=4):
                T.copy(X[bx,k*BN],s)
                for i in T.Parallel(BN): acc[i]+=T.cast(s[i],"float32")
            for i in T.Parallel(BN): tot[0]+=acc[i]
            T.copy(tot,O[bx])
    return main
CHUNK=1<<20
for NBLK in (32,64,128,156,312,624):
    try:
        X=torch.randn(NBLK,CHUNK,device="cuda",dtype=torch.float16)
        k=tilelang.compile(streamread(NBLK,CHUNK,256,4096),out_idx=[1],target="cuda")
        k(X); torch.cuda.synchronize()
        t=do_bench(lambda: k(X),warmup=10,rep=50)
        print(f"NBLK={NBLK:4d} read {NBLK*CHUNK*2/1e6:6.0f}MB: {t*1000:7.1f}us -> {NBLK*CHUNK*2/(t*1e-3)/1e9:5.0f} GB/s")
    except Exception as e: print(f"NBLK={NBLK}: FAIL {str(e)[:70]}")