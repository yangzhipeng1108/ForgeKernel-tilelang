import torch, torch.nn as nn, tilelang, tilelang.language as T
# does a tilelang-compiled gemm propagate autograd?
@T.prim_func
def main(A:T.Tensor((64,64),"float16"),B:T.Tensor((64,64),"float16"),C:T.Tensor((64,64),"float16")):
    with T.Kernel(1,1,threads=128) as (bx,by):
        a_s=T.alloc_shared((64,64),"float16"); b_s=T.alloc_shared((64,64),"float16")
        acc=T.alloc_fragment((64,64),"float32"); T.clear(acc)
        T.copy(A,a_s); T.copy(B,b_s); T.gemm(a_s,b_s,acc); T.copy(acc,C)
k=tilelang.compile(main,out_idx=[2],target="cuda")
x=torch.randn(64,64,device="cuda",requires_grad=True)
w=torch.randn(64,64,device="cuda",requires_grad=True)
out=k(x.half(),w.half())
print("out.requires_grad =", out.requires_grad)
print("out.grad_fn =", out.grad_fn)
try:
    out.float().sum().backward()
    print("x.grad is None:", x.grad is None, "| w.grad is None:", w.grad is None)
except Exception as e:
    print("backward failed:", repr(e))
