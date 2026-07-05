/*
 * W4A8 (int4 weights, dynamically quantized int8 activations) decode GEMM
 * for Ampere-class GPUs (sm_80..sm_89), built on dp4a.
 *
 * Weights are s4, packed 8 per int32 in "byte-nibble" order: byte b of a
 * word holds w[8u+b] in its low nibble and w[8u+4+b] in its high nibble.
 * The kernel shifts nibbles to the byte's high half, so dp4a operands are
 * exactly 16*s4; the /16 is folded into the weight scales at repack time.
 * Output: y[t][n] = a_scale[t] * sum_g b_scale[n][g] * sum_{k in g} w*q_x.
 *
 * Group size is fixed at 128 (gated in the Python kernel selector).
 *
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef USE_ROCM

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "libtorch_stable/torch_utils.h"

namespace w4a8_dp4a {

constexpr int kGroup = 128;
constexpr int kTile = 2048;  // x-tile elements staged in smem when K is large

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <>
__device__ __forceinline__ float to_float<half>(half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ half from_float<half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// Bank swizzles for the staged x tiles. NC==1 uses Swizzle<1,4,3> on byte
// offsets; NC>1 interleaves columns per 32-bit word and swizzles the
// NC*4-byte blocks (lanes stride 8 blocks, which all alias bank 0 unswizzled).
// Alignment of each vectorizable block is preserved.
template <int NC>
__device__ __forceinline__ unsigned swizzle(unsigned off) {
  if (NC == 1) return off ^ ((off >> 3) & 0x10u);
  if (NC == 2) return off ^ ((off >> 4) & 0x78u);
  if (NC == 4) return off ^ ((off >> 4) & 0x70u);
  if (NC == 8) return off ^ ((off >> 4) & 0x60u);
  return off;
}

template <typename T, int NC, int R>
__global__ void gemm_kernel(const uint4* __restrict__ b_q,
                            const int8_t* __restrict__ a_q,
                            const float* __restrict__ a_scales,
                            const T* __restrict__ b_scales,
                            T* __restrict__ out, int N, int K, int tk) {
  extern __shared__ char smem[];  // NC interleaved int8 x tiles

  const int warps = blockDim.x / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * warps + warp) * R;
  const int chunks_row = K / 32;

  float acc[R][NC];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int nc = 0; nc < NC; ++nc) acc[r][nc] = 0.f;

  for (int kt = 0; kt < K; kt += tk) {
    const int tlen = min(tk, K - kt);
    // cooperative staging of the x tiles
    for (int i = threadIdx.x; i < NC * tlen / 4; i += blockDim.x) {
      int nc, ii, off;
      if (NC == 1) {
        nc = 0;
        ii = i;
        off = (int)swizzle<1>((unsigned)(4 * ii));
      } else {
        ii = i / NC;
        nc = i % NC;
        off = (int)swizzle<NC>((unsigned)(ii * NC * 4)) + nc * 4;
      }
      *(int*)(smem + off) = ((const int*)(a_q + (size_t)nc * K + kt))[ii];
    }
    __syncthreads();

    if (n0 < N) {
      const int c0 = kt / 32, c1 = (kt + tlen) / 32;
      // Per chunk (c += 32) every derived address advances by a constant:
      // precompute swizzled bases per lane and march pointers.
      const int j00 = (c0 + lane) * 32 - kt;  // byte offset in tile
      const char* xp0;
      const char* xp1;
      if (NC == 1) {
        // the swizzle swaps a chunk's two int4 halves for lanes with the
        // flip bit set; the partner half is always at (offset ^ 16)
        const unsigned off = swizzle<1>((unsigned)j00);
        xp0 = smem + off;
        xp1 = smem + (off ^ 16u);
      } else {
        xp0 = smem;
        xp1 = smem;
      }
      const T* scp[R];
      const uint4* wp[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        scp[r] = b_scales + (size_t)(n0 + r) * (K / kGroup) +
                 (c0 + lane) * 32 / kGroup;
        wp[r] = b_q + (size_t)(n0 + r) * chunks_row + c0 + lane;
      }
      const int nchunks = (c1 - c0 - lane + 31) / 32;
      for (int it = 0; it < nchunks; ++it) {
        uint4 q[R];
#pragma unroll
        for (int r = 0; r < R; ++r)
          if (n0 + r < N) q[r] = __ldcs(wp[r]);
        int xw[8][NC];
        if (NC == 1) {
          const int4 xa = *(const int4*)xp0;
          const int4 xb = *(const int4*)xp1;
          xw[0][0] = xa.x, xw[1][0] = xa.y, xw[2][0] = xa.z, xw[3][0] = xa.w;
          xw[4][0] = xb.x, xw[5][0] = xb.y, xw[6][0] = xb.z, xw[7][0] = xb.w;
        } else {
          const int jb = (j00 + it * 1024) * NC;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const char* base = smem + swizzle<NC>((unsigned)(jb + j * NC * 4));
#pragma unroll
            for (int nc = 0; nc < NC; ++nc)
              xw[j][nc] = *(const int*)(base + nc * 4);
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          if (n0 + r >= N) break;
          const float sw = to_float<T>(scp[r][0]);
          const unsigned qw[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
#pragma unroll
          for (int nc = 0; nc < NC; ++nc) {
            int isum = 0;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              // nibbles to the byte's high half: bytes are 16*s4, exact
              const unsigned lo = (qw[u] << 4) & 0xF0F0F0F0u;
              const unsigned hi = qw[u] & 0xF0F0F0F0u;
              isum = __dp4a((int)lo, xw[2 * u][nc], isum);
              isum = __dp4a((int)hi, xw[2 * u + 1][nc], isum);
            }
            acc[r][nc] += sw * (float)isum;  // sw carries the /16
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          wp[r] += 32;
          scp[r] += 32 * 32 / kGroup;
        }
        if (NC == 1) {
          xp0 += 1024;
          xp1 += 1024;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (n0 + r >= N) break;
#pragma unroll
    for (int nc = 0; nc < NC; ++nc) {
      float a = acc[r][nc];
#pragma unroll
      for (int off = 16; off; off >>= 1)
        a += __shfl_down_sync(0xffffffff, a, off);
      if (lane == 0)
        out[(size_t)nc * N + n0 + r] = from_float<T>(a * a_scales[nc]);
    }
  }
}

// fp16/bf16 dequantization of the packed weights (large-batch fallback path:
// dequantize once, then a regular GEMM).
template <typename T>
__global__ void dequant_kernel(const unsigned* __restrict__ b_q,
                               const T* __restrict__ b_scales,
                               T* __restrict__ out, int N, int K) {
  const int n = blockIdx.y;
  const int u = blockIdx.x * blockDim.x + threadIdx.x;  // word index
  if (u >= K / 8) return;
  const unsigned q = b_q[(size_t)n * (K / 8) + u];
  const float s16 =
      to_float<T>(b_scales[(size_t)n * (K / kGroup) + u * 8 / kGroup]);
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    // low nibble -> k = 8u + b, high nibble -> k = 8u + 4 + b (both s4)
    const int lo = (int)(q << (28 - 8 * b)) >> 28;
    const int hi = (int)(q << (24 - 8 * b)) >> 28;
    out[(size_t)n * K + 8 * u + b] = from_float<T>(lo * s16 * 16.f);
    out[(size_t)n * K + 8 * u + 4 + b] = from_float<T>(hi * s16 * 16.f);
  }
}

template <typename T>
void launch(const torch::stable::Tensor& b_q,
            const torch::stable::Tensor& a_q,
            const torch::stable::Tensor& a_scales,
            const torch::stable::Tensor& b_scales, torch::stable::Tensor& out,
            int bs, int N, int K, cudaStream_t stream) {
  int dev;
  cudaGetDevice(&dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  const bool sm80 = prop.major == 8 && prop.minor == 0;
  const int block = sm80 ? 256 : 512;
  constexpr int kMaxSmem = 96 * 1024;

  auto run = [&](auto nc_tag, auto r_tag) {
    constexpr int NC = decltype(nc_tag)::value;
    constexpr int R = decltype(r_tag)::value;
    const int rows_per_block = (block / 32) * R;
    const int tk = ((size_t)NC * K <= kMaxSmem) ? K : kTile;
    const size_t smem = (size_t)NC * tk;
    auto* kern = gemm_kernel<T, NC, R>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kMaxSmem);
    kern<<<(N + rows_per_block - 1) / rows_per_block, block, smem, stream>>>(
        (const uint4*)b_q.const_data_ptr(),
        (const int8_t*)a_q.const_data_ptr(),
        (const float*)a_scales.const_data_ptr(),
        (const T*)b_scales.const_data_ptr(), (T*)out.mutable_data_ptr(), N, K,
        tk);
    STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "w4a8_dp4a_gemm launch failed");
  };
  using one = std::integral_constant<int, 1>;
  using two = std::integral_constant<int, 2>;
  if (sm80) {  // measured: 1 row/warp on sm_80, 2 elsewhere
    if (bs == 1)
      run(one{}, one{});
    else if (bs == 2)
      run(two{}, one{});
    else if (bs == 4)
      run(std::integral_constant<int, 4>{}, one{});
    else
      run(std::integral_constant<int, 8>{}, one{});
  } else {
    if (bs == 1)
      run(one{}, two{});
    else if (bs == 2)
      run(two{}, two{});
    else if (bs == 4)
      run(std::integral_constant<int, 4>{}, two{});
    else
      run(std::integral_constant<int, 8>{}, one{});
  }
}

}  // namespace w4a8_dp4a

void w4a8_dp4a_gemm(torch::stable::Tensor& out,
                    const torch::stable::Tensor& a_q,
                    const torch::stable::Tensor& a_scales,
                    const torch::stable::Tensor& b_q,
                    const torch::stable::Tensor& b_scales) {
  const int bs = a_q.size(0);
  const int K = a_q.size(1);
  const int N = b_q.size(0);
  STD_TORCH_CHECK(K % w4a8_dp4a::kGroup == 0, "K must be a multiple of 128");
  STD_TORCH_CHECK(bs == 1 || bs == 2 || bs == 4 || bs == 8,
                  "batch must be padded to 1/2/4/8, got ", bs);
  STD_TORCH_CHECK(a_q.scalar_type() == torch::headeronly::ScalarType::Char,
                  "a_q must be int8");
  STD_TORCH_CHECK(b_q.scalar_type() == torch::headeronly::ScalarType::Int,
                  "b_q must be int32");
  STD_TORCH_CHECK(b_q.size(1) == K / 8, "b_q must be [N, K/8] int32");
  STD_TORCH_CHECK(
      a_scales.scalar_type() == torch::headeronly::ScalarType::Float,
      "a_scales must be float32");
  STD_TORCH_CHECK(a_scales.numel() >= bs, "a_scales must have >= bs entries");
  STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half ||
                      out.scalar_type() ==
                          torch::headeronly::ScalarType::BFloat16,
                  "out must be fp16 or bf16");
  STD_TORCH_CHECK(out.scalar_type() == b_scales.scalar_type(),
                  "out and b_scales dtypes must match");
  STD_TORCH_CHECK(out.size(0) == bs && out.size(1) == N,
                  "out must be [bs, N]");
  STD_TORCH_CHECK(b_scales.size(0) == N &&
                      b_scales.size(1) == K / w4a8_dp4a::kGroup,
                  "b_scales must be [N, K/128]");
  STD_TORCH_CHECK(a_q.is_contiguous() && b_q.is_contiguous() &&
                      b_scales.is_contiguous() && out.is_contiguous(),
                  "all tensors must be contiguous");

  const auto device_index = a_q.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);
  if (out.scalar_type() == torch::headeronly::ScalarType::Half) {
    w4a8_dp4a::launch<half>(b_q, a_q, a_scales, b_scales, out, bs, N, K,
                            stream);
  } else {
    w4a8_dp4a::launch<nv_bfloat16>(b_q, a_q, a_scales, b_scales, out, bs, N, K,
                                   stream);
  }
}

void w4a8_dp4a_dequant(torch::stable::Tensor& out,
                       const torch::stable::Tensor& b_q,
                       const torch::stable::Tensor& b_scales) {
  const int N = b_q.size(0);
  const int K = b_q.size(1) * 8;
  STD_TORCH_CHECK(b_q.scalar_type() == torch::headeronly::ScalarType::Int,
                  "b_q must be int32");
  STD_TORCH_CHECK(out.scalar_type() == b_scales.scalar_type(),
                  "out and b_scales dtypes must match");
  STD_TORCH_CHECK(out.size(0) == N && out.size(1) == K,
                  "out must be [N, K]");
  const auto device_index = b_q.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);
  dim3 grid((K / 8 + 255) / 256, N);
  if (out.scalar_type() == torch::headeronly::ScalarType::Half) {
    w4a8_dp4a::dequant_kernel<half><<<grid, 256, 0, stream>>>(
        (const unsigned*)b_q.const_data_ptr(),
        (const half*)b_scales.const_data_ptr(),
        (half*)out.mutable_data_ptr(), N, K);
  } else {
    w4a8_dp4a::dequant_kernel<nv_bfloat16><<<grid, 256, 0, stream>>>(
        (const unsigned*)b_q.const_data_ptr(),
        (const nv_bfloat16*)b_scales.const_data_ptr(),
        (nv_bfloat16*)out.mutable_data_ptr(), N, K);
  }
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                  "w4a8_dp4a_dequant launch failed");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("w4a8_dp4a_gemm", TORCH_BOX(&w4a8_dp4a_gemm));
  m.impl("w4a8_dp4a_dequant", TORCH_BOX(&w4a8_dp4a_dequant));
}

#endif  // !USE_ROCM
