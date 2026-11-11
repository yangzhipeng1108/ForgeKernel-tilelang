import tilelang, tilelang.language as T, torch as th
from tilelang.tileop.base import GemmWarpPolicy
print("policies:", [x for x in dir(GemmWarpPolicy) if not x.startswith("_")])
for p in ["Square","FullRow","FullCol"]:
    print(p, getattr(GemmWarpPolicy,p,None))