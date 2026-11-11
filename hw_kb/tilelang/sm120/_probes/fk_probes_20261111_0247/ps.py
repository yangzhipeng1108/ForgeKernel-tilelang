import tilelang, tilelang.language as T, torch as th, inspect
src=inspect.getsource(T.Persistent)
print(src[:2500])