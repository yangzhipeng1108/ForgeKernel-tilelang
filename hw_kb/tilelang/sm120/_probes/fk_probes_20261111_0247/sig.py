import sys; sys.path.insert(0,"src")
import tilelang.language as T, inspect
sig=inspect.signature(T.copy)
print("T.copy params:", list(sig.parameters))
# eviction policy accepted values
src=inspect.getsource(T.copy)
import re
for m in re.findall(r'eviction[^\n]*', src)[:8]: print("  ",m.strip()[:100])