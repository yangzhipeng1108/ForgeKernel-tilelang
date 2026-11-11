import sys; sys.path.insert(0,"src")
import tilelang.language as T, inspect
import tilelang.language.annotations as A
names=[n for n in dir(A) if not n.startswith("_")]
print("annotations:", names)
for n in ["WsSchedule","WSSchedule","annotate_ws_schedule","WSRole","WSPipeline","WSScope","WSSync","ws_schedule"]:
    if hasattr(A,n):
        obj=getattr(A,n)
        try: print("===",n,"==="); print(inspect.getsource(obj)[:1200])
        except Exception as e: print(n, type(obj))
for n in ["annotate_ws_schedule","ws"]:
    if hasattr(T,n): print("T has", n)