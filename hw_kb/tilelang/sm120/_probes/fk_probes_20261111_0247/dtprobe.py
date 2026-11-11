import torch
print("torch fp8:", hasattr(torch,"float8_e4m3fn"), hasattr(torch,"float8_e5m2"))
print("bf16:", hasattr(torch,"bfloat16"))
import sys; sys.path.insert(0,"src")
import tilelang, tilelang.language as T
# probe accepted dtype strings by trying allocations in a tiny kernel
for dt in ["float8_e4m3","float8_e5m2","float8_e4m3fn"]:
    try:
        @T.prim_func
        def dummy(A: T.Tensor((16,16), dt)):
            with T.Kernel(1, threads=32) as bx:
                s=T.alloc_shared((16,16), dt)
                T.copy(A, s)
        tilelang.compile(dummy, target={"kind":"cuda","arch":"sm_120a"})
        print("OK", dt)
    except Exception as e:
        print("NO", dt, type(e).__name__, str(e).strip().splitlines()[-1][:80])