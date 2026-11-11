import tilelang, tilelang.language as T, torch as th
from tilelang.carver.arch import driver
NSM=driver.get_num_sms(); BM=BN=128; d=T.float8_e4m3
M=N=K=4096
def build(mode,swz=1,cm=True,panel=None):
    mb=T.ceildiv(M,BM); nb=T.ceildiv(N,BN); tot=mb*nb; waves=(tot+NSM-1)//NSM
    @T.prim_func
    def main(A:T.Tensor((M,K),d),B:T.Tensor((K,N),d),C:T.Tensor((M,N),th.float16)):
        if mode=="raster":
            with T.Kernel(nb,mb,threads=256) as (bx,by):
                if panel: T.use_swizzle(panel_size=panel,order="row")
                As=T.alloc_shared((BM,128),d); Bs=T.alloc_shared((128,BN),d)
                Cl=T.alloc_fragment((BM,BN),"float32"); T.clear(Cl)
                for k in T.Pipelined(T.ceildiv(K,128),num_stages=2):
                    T.copy(A[by*BM,k*128],As); T.copy(B[k*128,bx*BN],Bs); T.gemm(As,Bs,Cl)
                T.copy(Cl,C[by*BM,bx*BN])
        else:
            with T.Kernel(NSM,threads=256) as (bid,):
                s=T.PersistentTileScheduler(mb,nb,swizzle_size=swz,column_major=cm,stateful=False)
                As=T.alloc_shared((BM,128),d); Bs=T.alloc_shared((128,BN),d)
                Cl=T.alloc_fragment((BM,BN),"float32")
                for w in range(waves):
                    t=NSM*w+bid
                    if t<tot:
                        by,bx=s.coord(t); T.clear(Cl)
                        for k in T.Pipelined(T.ceildiv(K,128),num_stages=2):
                            T.copy(A[by*BM,k*128],As); T.copy(B[k*128,bx*BN],Bs); T.gemm(As,Bs,Cl)
                        T.copy(Cl,C[by*BM,bx*BN])
    return main
for tag,args in [("raster",("raster",)),("swz8",("raster",1,True,8)),("pers8",("pers",8,True))]:
    src=tilelang.compile(build(*args),out_idx=[2],target="cuda").get_kernel_source()
    open(f"/tmp/k_{tag}.cu","w").write(
      '#include <torch/extension.h>\n'+src if False else src)
    print("wrote",tag,len(src))