import tilelang.language as T, inspect
for n in ["ClusterKernel","Kernel","alloc_barrier","block_rank_in_cluster","cluster_rank","map_shared_rank"]:
    o=getattr(T,n,None)
    print(n,"->","MISSING" if o is None else "ok")
    if o is not None:
        try: print("   ",str(inspect.signature(o)))
        except Exception: pass
print("--- ClusterKernel doc ---")
d=getattr(T,"ClusterKernel",None)
print(((d.__doc__ or "")[:900]) if d else "none")