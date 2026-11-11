#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* As = ((void*)((char*)buf_dyn_shmem + 0));
  void* Bs = ((void*)((char*)buf_dyn_shmem + 32768));
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  int producer_phase_cnt[1];
  int consumer_phase_cnt[1];
  float Cl[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(256);
    mbarrier[3].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  producer_phase_cnt[0] = 0;
  consumer_phase_cnt[0] = 0;
  for (int w = 0; w < 7; ++w) {
    __syncthreads();
    if (((w * 39) + (((int)blockIdx.x) >> 2)) < 256) {
      if (((int)threadIdx.x) < 128) {
        for (int k = 0; k < 32; ++k) {
          mbarrier[((producer_phase_cnt[0] & 1) + 2)].wait((((producer_phase_cnt[0] & 3) >> 1) ^ 1));
          if (tl::tl_shuffle_elect<128>()) {
            mbarrier[(producer_phase_cnt[0] & 1)].expect_transaction(16384);
            tl::tma_load(A_desc, mbarrier[(producer_phase_cnt[0] & 1)], (&(((fp8_e4_t*)As)[((producer_phase_cnt[0] & 1) * 16384)])), (k * 128), (((((w * 39) + (((int)blockIdx.x) >> 2)) >> 1) * 128) - ((((w * 39) + (((int)blockIdx.x) >> 2)) >> 6) * 4096)));
            mbarrier[(producer_phase_cnt[0] & 1)].arrive_and_expect_tx(16384);
            tl::tma_load(B_desc, mbarrier[(producer_phase_cnt[0] & 1)], (&(((fp8_e4_t*)Bs)[((producer_phase_cnt[0] & 1) * 16384)])), (((((w * 39) + (((int)blockIdx.x) >> 2)) >> 6) * 1024) + ((((w * 156) + ((int)blockIdx.x)) & 7) * 128)), (k * 128));
          }
          producer_phase_cnt[0] = (producer_phase_cnt[0] + 1);
        }
      } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
          float broadcast_var = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(Cl + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
        }
        for (int k_1 = 0; k_1 < 32; ++k_1) {
          mbarrier[(consumer_phase_cnt[0] & 1)].wait(((consumer_phase_cnt[0] & 3) >> 1));
          {
            fp8_e4_t A_local[64];
            fp8_e4_t B_local[32];
            for (int ki = 0; ki < 4; ++ki) {
              #pragma unroll
              for (int i_1 = 0; i_1 < 4; ++i_1) {
                tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)As)[((((((((consumer_phase_cnt[0] & 1) * 16384) + (((((int)threadIdx.x) & 63) >> 5) * 8192)) + (i_1 * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A_local[(i_1 * 16)])));
              }
              #pragma unroll
              for (int i_2 = 0; i_2 < 2; ++i_2) {
                for (int j = 0; j < 16; ++j) {
                  B_local[((i_2 * 16) + j)] = ((fp8_e4_t*)Bs)[(((((((((((consumer_phase_cnt[0] & 1) * 16384) + (ki * 4096)) + (((j & 7) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 512)) + ((j & 3) * 128)) + (((((((((((((int)threadIdx.x) >> 6) + 2) & 3) * 32) + (i_2 * 16)) + ((j >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((j & 3) >> 1)) & 1) * 32)) + (((i_2 + (j & 1)) & 1) * 16)) + ((j >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
                }
              }
              for (int i_3 = 0; i_3 < 4; ++i_3) {
                for (int j_1 = 0; j_1 < 2; ++j_1) {
                  tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(Cl + ((i_3 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_3 * 16)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 16)));
                  tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(Cl + (((i_3 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 16)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 16) + 8)));
                }
              }
            }
          }
          mbarrier[((consumer_phase_cnt[0] & 1) + 2)].arrive();
          consumer_phase_cnt[0] = (consumer_phase_cnt[0] + 1);
        }
        #pragma unroll
        for (int i_4 = 0; i_4 < 32; ++i_4) {
          half_t C_local_cast[2];
          uint1 __1;
          float2 v_ = *(float2*)(Cl + (i_4 * 2));
          ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
          *(uint1*)(C_local_cast + 0) = __1;
          *(uint1*)(C + ((((((((((((((w * 39) + (((int)blockIdx.x) >> 2)) >> 1) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_4 >> 3) * 65536)) + ((i_4 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + ((((w * 156) + ((int)blockIdx.x)) & 7) * 128)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_4 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - ((((w * 39) + (((int)blockIdx.x) >> 2)) >> 6) * 16776192)) - 64)) = *(uint1*)(C_local_cast + 0);
        }
      }
    }
  }
}

