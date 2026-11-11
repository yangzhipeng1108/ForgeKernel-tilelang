#include <cstdio>
__global__ void probe() {
#ifdef CUTLASS_ARCH_CLC_ENABLED
  printf("CLC_ON\n");
#else
  printf("CLC_OFF\n");
#endif
}
int main(){ probe<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
