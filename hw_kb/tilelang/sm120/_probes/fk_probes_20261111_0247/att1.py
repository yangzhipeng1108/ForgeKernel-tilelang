import tilelang, tilelang.language as T, inspect
names=[n for n in dir(T) if not n.startswith("_")]
for key in ["softmax","reduce_max","reduce_sum","exp","exp2","logsumexp","fill","reduce"]:
    hit=[n for n in names if key in n.lower()]
    if hit: print(key,"->",hit)
print("---")
for n in ["reduce_max","reduce_sum","reduce_abssum","cumsum"]:
    o=getattr(T,n,None)
    if o is not None:
        try: print(n, str(inspect.signature(o)))
        except: print(n,"ok")