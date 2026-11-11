#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma_sp.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C, const uchar* __restrict__ E);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C, const uchar* __restrict__ E) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Bs = ((void*)((char*)buf_dyn_shmem + 0));
  void* As = ((void*)((char*)buf_dyn_shmem + 16384));
  void* Es = ((void*)((char*)buf_dyn_shmem + 24576));
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float Cl[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::__sync_thread_partial(3, 128);
    for (int k = 0; k < 16; ++k) {
      mbarrier[((k & 1) + 2)].wait((((k & 3) >> 1) ^ 1));
      *(uint*)(((uchar*)Es) + (((k & 1) * 512) + (((int)threadIdx.x) * 4))) = *(uint*)(E + ((((((int)blockIdx.y) * 8192) + ((((int)threadIdx.x) >> 1) * 128)) + (k * 8)) + ((((int)threadIdx.x) & 1) * 4)));
      tl::__sync_thread_partial(3, 128);
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].expect_transaction(4096);
        tl::fence_proxy_async();
        tl::tma_load(A_desc, mbarrier[(k & 1)], (&(((half_t*)As)[((k & 1) * 2048)])), (k * 32), (((int)blockIdx.y) * 64));
        mbarrier[(k & 1)].arrive_and_expect_tx(8192);
        tl::tma_load(B_desc, mbarrier[(k & 1)], (&(((half_t*)Bs)[((k & 1) * 4096)])), (((int)blockIdx.x) * 64), (k * 64));
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(Cl + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      {
        half_t A_local[16];
        uchar E_local[8];
        half_t B_local[32];
        for (int ki = 0; ki < 2; ++ki) {
          for (int i_1 = 0; i_1 < 2; ++i_1) {
            tl::ptx_ldmatrix_x4((&(((half_t*)As)[(((((((k_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_1 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_1 * 8)])));
          }
          for (int i_2 = 0; i_2 < 2; ++i_2) {
            for (int j = 0; j < 4; ++j) {
              E_local[((i_2 * 4) + j)] = ((uchar*)Es)[(((((((((k_1 & 1) * 512) + (((((int)threadIdx.x) & 63) >> 5) * 256)) + (i_2 * 128)) + ((j >> 1) * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 8)) + (ki * 4)) + ((((int)threadIdx.x) & 1) * 2)) + (j & 1))];
            }
          }
          for (int i_3 = 0; i_3 < 2; ++i_3) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[(((((((k_1 & 1) * 4096) + (ki * 2048)) + ((((int)threadIdx.x) & 31) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_3) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8))])), (&(B_local[(i_3 * 16)])));
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[(((((((k_1 & 1) * 4096) + (ki * 2048)) + ((((int)threadIdx.x) & 31) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_3) & 1) * 16)) + ((((((int)threadIdx.x) & 1) + 1) & 1) * 8))])), (&(B_local[((i_3 * 16) + 8)])));
          }
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              tl::mma_sp_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 32, false, true, SM80::MMA::SparseSel::Zero>(reinterpret_cast<float*>(Cl + ((i_4 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_4 * 8)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 16)), reinterpret_cast<const uint32_t*>(E_local + (i_4 * 4)));
              tl::mma_sp_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 32, false, true, SM80::MMA::SparseSel::Zero>(reinterpret_cast<float*>(Cl + (((i_4 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_4 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 16) + 8)), reinterpret_cast<const uint32_t*>(E_local + (i_4 * 4)));
            }
          }
        }
      }
      mbarrier[((k_1 & 1) + 2)].arrive();
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      half_t C_local_cast[2];
      uint1 __1;
      float2 v_ = *(float2*)(Cl + (i_5 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C_local_cast + 0) = __1;
      *(uint1*)(C + ((((((((((((int)blockIdx.y) * 65536) + (((((int)threadIdx.x) & 63) >> 5) * 32768)) + ((i_5 >> 3) * 16384)) + ((i_5 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_5 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64)) = *(uint1*)(C_local_cast + 0);
    }
  }
}

