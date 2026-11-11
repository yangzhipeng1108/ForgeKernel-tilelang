import tilelang.language as T, inspect
import tilelang.language.experimental.gemm_sp_op as g
print("T.gemm_sp:", hasattr(T,"gemm_sp"))
o=getattr(T,"gemm_sp",None)
if o: print("sig:", inspect.signature(o))
print("--- doc ---")
print((g.gemm_sp.__doc__ or "")[:1500])
print("--- module names ---")
print([n for n in dir(g) if not n.startswith("_")])