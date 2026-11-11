import tilelang, tilelang.language as T, torch as th
from tilelang.carver.arch import driver
NSM=driver.get_num_sms()
print("NSM",NSM)
for (M,N,K) in [(512,512,16384),(1024,1024,8192),(4096,4096,4096)]:
    tiles=(M//128)*(N//128)
    print(f"{M}x{N}x{K}: tiles={tiles} fill={tiles/NSM*100:.0f}% of {NSM} SMs", end="  ")
    for SK in (2,4,8,16):
        print(f"SK{SK}={tiles*SK/NSM*100:.0f}%", end=" ")
    print()