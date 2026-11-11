#include <cstdio>
#include <cutlass/arch/config.h>
__global__ void probe() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
  #ifdef CUTLASS_ARCH_MMA_SM120A_ENABLED
  printf("DEV: SM120A_ENABLED=1 ");
  #else
  printf("DEV: SM120A_ENABLED=0 ");
  #endif
  #ifdef CUTLASS_ARCH_CLC_ENABLED
  printf("CLC_ENABLED=1\n");
  #else
  printf("CLC_ENABLED=0\n");
  #endif
#endif
}
int main(){ probe<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
