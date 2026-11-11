import tilelang, tilelang.language as T, inspect
for n in ["use_swizzle","Persistent","persistent","annotate_l2_hit_ratio","clusters","alloc_barrier"]:
    o=getattr(T,n,None)
    print(f"T.{n}:", "MISSING" if o is None else type(o).__name__)
    if o is not None:
        try: print("   sig:", str(inspect.signature(o)))
        except Exception: pass
import tilelang.language as TL
print("--- use_swizzle doc ---")
print((T.use_swizzle.__doc__ or "")[:800])
print("--- Persistent doc ---")
p=getattr(T,"Persistent",None)
print(((p.__doc__ or "")[:1200]) if p else "none")