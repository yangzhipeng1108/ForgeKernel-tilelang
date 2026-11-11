import tilelang, tilelang.language as T, torch, os
@T.prim_func
def with_clc(A:T.Tensor((4096,4096),"float16"),B:T.Tensor((4096,4096),"float16"),C:T.Tensor((4096,4096),"float16")):
    with T.ClusterKernel(32,32,cluster_dims=1,threads=256) as (bx,by):
        a_s=T.alloc_shared((128,128),"float16"); b_s=T.alloc_shared((128,128),"float16")
        acc=T.alloc_fragment((128,128),"float32"); T.clear(acc)
        res=T.alloc_shared((16,),"uint32"); bar=T.alloc_barrier(1)
        T.clc_try_cancel(res,bar)
        for k in T.Pipelined(32,num_stages=2):
            T.copy(A[by*128,k*128],a_s); T.copy(B[k*128,bx*128],b_s)
            T.gemm(a_s,b_s,acc)
        T.copy(acc,C[by*128,bx*128])
k=tilelang.compile(with_clc,out_idx=[2],target={"kind":"cuda","arch":"sm_120a"})
src=k.get_kernel_source()
open("/tmp/clc_kernel.cu","w").write(src)
print("src saved; has clc_try_cancel:", "clc_try_cancel" in src)
print("line with clc:", [l.strip() for l in src.splitlines() if "clc_try_cancel" in l][:2])