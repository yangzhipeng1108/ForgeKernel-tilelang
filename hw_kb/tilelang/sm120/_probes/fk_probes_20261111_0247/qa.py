import tilelang.language as T, tilelang.language.annotations as A
for n in ["annotate_producer_reg_dealloc","annotate_consumer_reg_alloc","annotate_min_blocks_per_sm","annotate_l2_hit_ratio","annotate_ws_pipeline_depth"]:
    print(n, "->", hasattr(T,n), hasattr(A,n))
    o=getattr(T,n,None) or getattr(A,n,None)
    if o:
        import inspect
        try: print("   sig:", str(inspect.signature(o)))
        except: pass