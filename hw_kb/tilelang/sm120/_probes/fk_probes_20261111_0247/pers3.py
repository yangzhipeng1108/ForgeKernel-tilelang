import tilelang, tilelang.language as T, torch as th
from tilelang.profiler import do_bench
from tilelang.carver.arch import driver
NSM=driver.get_num_sms(); BM=BN=128; d=T.float8_e4m3

def build(M,N,K,BK,mode,swz=1,colmaj=True,panel=None,order="row"):
    m_blocks=T.ceildiv(M,BM); n_blocks=T.ceildiv(N,BN); total=m_blocks*n_blocks
    waves=(total+NSM-1)//NSM
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        if mode=="raster":
            with T.Kernel(n_blocks,m_blocks,threads=256) as (bx,by):
                if panel: T.use_swizzle(panel_size=panel,order=order)
                As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
                Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
                for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                    T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
                T.copy(Cl,C[by*BM,bx*BN])
        else:
            with T.Kernel(NSM,threads=256) as (bid,):
                sched=T.PersistentTileScheduler(m_blocks,n_blocks,swizzle_size=swz,column_major=colmaj,stateful=False)
                As=T.alloc_shared((BM,BK),d); Bs=T.alloc_shared((BK,BN),d)
                Cl=T.alloc_fragment((BM,BN),"float32")
                for w in range(waves):
                    t=NSM*w+bid
                    if t<total:
                        by,bx=sched.coord(t)
                        T.clear(Cl)
                        for k in T.Pipelined(T.ceildiv(K,BK),num_stages=2):
                            T.copy(A[by*BM,k*BK],As); T.copy(B[k*BK,bx*BN],Bs); T.gemm(As,Bs,Cl)
                        T.copy(Cl,C[by*BM,bx*BN])
    return main

def bench(M,N,K,tag):
    A=th.randn(M,K,device="cuda").to(th.float8_e4m3fn); B=th.randn(K,N,device="cuda").to(th.float8_e4m3fn)
    ref=(A.float()@B.float())
    rows=[]
    for t in tag:
        try:
            k=tilelang.compile(build(M,N,K,128,*t[0]),out_idx=[2],target="cuda")
            tm=do_bench(lambda: k(A,B),warmup=20,rep=60)
            C=k(A,B); md=(C.float()-ref).abs().max().item()
            rows.append(f"{t[1]}={2*M*N*K/(tm*1e-3)/1e12:.1f}(md{md:.2g})")
        except Exception as e: rows.append(f"{t[1]}=FAIL:{str(e)[:60]}")
    print(f"{M}x{N}x{K}: "+" | ".join(rows))

cases=[
 ("raster",("raster",)),
 ("swz4row",("raster",None,True,4,"row")),
 ("swz8row",("raster",None,True,8,"row")),
]
def args(mode,swz=1,cm=True,panel=None,order="row"): return ((mode,BK:=128,swz,cm,panel,order),)
# rebuild tuples cleanly
def tup(mode,swz=1,cm=True,panel=None,order="row"): return ((mode,swz,cm,panel,order),)
for (M,N,K) in [(4096,4096,4096),(1024,1024,8192),(512,512,16384),(8192,8192,1024)]:
    sig=[(tup("raster"),"raster"),(tup("raster",panel=4),"swz4r"),(tup("raster",panel=8),"swz8r"),
         (tup("pers",1,True),"p_s1cm"),(tup("pers",8,True),"p_s8cm"),(tup("pers",8,False),"p_s8rm")]
    bench(M,N,K,sig)