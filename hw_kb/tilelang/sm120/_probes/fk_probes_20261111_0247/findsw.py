import tilelang.language as T, pkgutil, tilelang
print("T swizzle attrs:", [n for n in dir(T) if "swizz" in n.lower()])
import importlib
for mod in ["tilelang.intrinsics","tilelang.layout","tilelang.layout.swizzle"]:
    try:
        m=importlib.import_module(mod); print("OK",mod, [n for n in dir(m) if "swizz" in n.lower()])
    except Exception as e: print("NO",mod,e)