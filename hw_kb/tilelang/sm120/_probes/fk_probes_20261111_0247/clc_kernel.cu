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
#include <tl_templates/cuda/cluster.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void with_clc_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(384, 1) with_clc_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* res = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_s = ((void*)((char*)buf_dyn_shmem + 65536));
  __shared__ __align__(16) uint64_t bar_mem[2];
  auto bar = reinterpret_cast<Barrier*>(bar_mem);
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float acc[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    bar[0].init(1);
    bar[1].init(1);
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(256);
    mbarrier[3].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  tl::clc_try_cancel((&(((uint*)res)[0])), (&(bar[0])));
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    tl::fence_proxy_async();
    for (int k = 0; k < 32; ++k) {
      mbarrier[((k & 1) + 2)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].expect_transaction(32768);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          tl::tma_load(A_desc, mbarrier[(k & 1)], (&(((half_t*)a_s)[(((k & 1) * 16384) + (i * 8192))])), ((k * 128) + (i * 64)), (((int)blockIdx.y) * 128));
        }
        mbarrier[(k & 1)].arrive_and_expect_tx(32768);
        #pragma unroll
        for (int i_1 = 0; i_1 < 2; ++i_1) {
          tl::tma_load(B_desc, mbarrier[(k & 1)], (&(((half_t*)b_s)[(((k & 1) * 16384) + (i_1 * 8192))])), ((((int)blockIdx.x) * 128) + (i_1 * 64)), (k * 128));
        }
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i_2 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      {
        half_t A_local[32];
        half_t B_local[16];
        for (int ki = 0; ki < 8; ++ki) {
          #pragma unroll
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4((&(((half_t*)a_s)[(((((((((k_1 & 1) * 16384) + ((ki >> 2) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (i_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_3 * 8)])));
          }
          #pragma unroll
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)b_s)[((((((((k_1 & 1) * 16384) + (((((((int)threadIdx.x) >> 6) + 2) & 3) >> 1) * 8192)) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
          }
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            for (int j = 0; j < 2; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((i_5 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (((i_5 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      mbarrier[((k_1 & 1) + 2)].arrive();
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 32; ++i_6) {
      half_t C_local_cast[2];
      uint1 __1;
      float2 v_ = *(float2*)(acc + (i_6 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C_local_cast + 0) = __1;
      *(uint1*)(C + ((((((((((((int)blockIdx.y) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_6 >> 3) * 65536)) + ((i_6 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_6 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64)) = *(uint1*)(C_local_cast + 0);
    }
  }
}

