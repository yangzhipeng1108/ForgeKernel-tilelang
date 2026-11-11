import tilelang, tilelang.language as T
print("tl:", [n for n in dir(tilelang) if "float8" in n.lower() or "e4m3" in n.lower() or "e5m2" in n.lower()])
print("T:", [n for n in dir(T) if "float8" in n.lower() or "e4m3" in n.lower() or "e5m2" in n.lower()])
from tvm import DataType
for s in ["float8_e4m3","float8_e4m3fn","e4m3_float8","float8_e5m2"]:
    try: print(s, DataType(s))
    except Exception as e: print(s,"ERR")