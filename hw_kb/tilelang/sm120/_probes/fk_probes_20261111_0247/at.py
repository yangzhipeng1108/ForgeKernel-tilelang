import tilelang.language as T, inspect
for n in ["atomic_add","atomic_addx2","atomic_addx4","atomic_max","atomic_min"]:
    o=getattr(T,n,None)
    print(n, "->", "MISSING" if o is None else "ok")
    if o is not None:
        try: print("   ", str(inspect.signature(o)))
        except: pass
print((T.atomic_add.__doc__ or "")[:600] if hasattr(T,"atomic_add") else "")