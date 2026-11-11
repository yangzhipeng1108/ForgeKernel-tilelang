import tilelang, tilelang.language as T
for n in ["streamk_programs","StreamK"]:
    try:
        k=tilelang.compile(lambda:None,target="cuda")
    except: pass
print("T.Kernel accepts streamk_programs:", end=" ")
import inspect
sig=inspect.signature(T.Kernel)
print("streamk" in str(sig))
print("signature:",sig)