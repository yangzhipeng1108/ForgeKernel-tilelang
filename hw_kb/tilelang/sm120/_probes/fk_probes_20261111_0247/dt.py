import sys
sys.path.insert(0,"src")
try:
    from tilelang.carver.arch.cuda import is_tensorcore_supported_precision, CUDA
    for arch in ["sm_120a"]:
        a=CUDA(arch)
        for dt in ["float16","bfloat16","float32","float8_e4m3","float8_e5m2","int8"]:
            for acc in ["float32","float16"]:
                try:
                    if is_tensorcore_supported_precision(dt,acc,a): print("SUP",dt,"->",acc)
                except Exception as e:
                    print("ERR",dt,acc,type(e).__name__,str(e)[:60])
except Exception as e:
    print("import fail", e)
# also show what dtype strings tilelang gemm accepts
import tilelang.language as T, inspect
src=inspect.getsource(T.gemm)
import re
for m in set(re.findall(r'"(float8[_a-z0-9]*|float16|bfloat16|float32)"', src)): print("gemm-mentions", m)