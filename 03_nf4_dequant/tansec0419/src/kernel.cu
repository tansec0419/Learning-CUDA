#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

// NF4 查找表（来自 QLoRA 论文）
__constant__ float NF4_TABLE[16] = {-1.0f,
                                    -0.6961928009986877f,
                                    -0.5250730514526367f,
                                    -0.39491748809814453f,
                                    -0.28444138169288635f,
                                    -0.18477343022823334f,
                                    -0.09105003625154495f,
                                    0.0f,
                                    0.07958029955625534f,
                                    0.16093020141124725f,
                                    0.24611230194568634f,
                                    0.33791524171829224f,
                                    0.44070982933044434f,
                                    0.5626170039176941f,
                                    0.7229568362236023f,
                                    1.0f};

// 辅助函数：将 FP16 (uint16_t) 转换为 float
__device__ __forceinline__ float fp16_to_float(uint16_t h) {
  return __half2float(*reinterpret_cast<const __half*>(&h));
}

// ============================================================================
// V1: 每线程处理 1 个元素（baseline）
// ============================================================================
__global__ void nf4_dequantize_kernel_v1(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ absmax_q,
    const uint16_t* __restrict__ absmax2, const uint16_t* __restrict__ code2,
    uint16_t* __restrict__ output, int64_t num_elements, int32_t blocksize) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) return;

  int64_t byte_idx = idx / 2;
  int shift = (idx % 2) * 4;
  uint8_t nf4_index = (packed[byte_idx] >> shift) & 0xF;

  float nf4_value = NF4_TABLE[nf4_index];

  int64_t block_idx = idx / blocksize;
  uint8_t absmax_q_idx = absmax_q[block_idx];
  float scale1 = fp16_to_float(code2[absmax_q_idx]);

  int64_t group_idx = block_idx / 256;
  float scale2 = fp16_to_float(absmax2[group_idx]);

  float result = nf4_value * scale1 * scale2;

  __nv_bfloat16 bf16_result = __float2bfloat16(result);
  output[idx] = *reinterpret_cast<uint16_t*>(&bf16_result);
}

// ============================================================================
// V2: 每线程处理 2 个元素 + Packed Store + Shared Memory
// ============================================================================
__global__ void nf4_dequantize_kernel_v2(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ absmax_q,
    const uint16_t* __restrict__ absmax2, const uint16_t* __restrict__ code2,
    uint16_t* __restrict__ output, int64_t num_elements, int32_t blocksize) {
  __shared__ float s_nf4_table[16];
  if (threadIdx.x < 16) {
    s_nf4_table[threadIdx.x] = NF4_TABLE[threadIdx.x];
  }
  __syncthreads();

  int64_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  if (base_idx >= num_elements) return;

  int64_t byte_idx = base_idx / 2;
  uint8_t packed_byte = packed[byte_idx];

  uint8_t nf4_idx0 = packed_byte & 0xF;
  uint8_t nf4_idx1 = (packed_byte >> 4) & 0xF;

  float nf4_val0 = s_nf4_table[nf4_idx0];
  float nf4_val1 = s_nf4_table[nf4_idx1];

  int64_t block_idx0 = base_idx / blocksize;
  uint8_t absmax_q_idx0 = absmax_q[block_idx0];
  float scale1_0 = fp16_to_float(code2[absmax_q_idx0]);
  int64_t group_idx0 = block_idx0 / 256;
  float scale2_0 = fp16_to_float(absmax2[group_idx0]);
  float result0 = nf4_val0 * scale1_0 * scale2_0;

  float result1 = 0.0f;
  if (base_idx + 1 < num_elements) {
    int64_t block_idx1 = (base_idx + 1) / blocksize;
    uint8_t absmax_q_idx1 = absmax_q[block_idx1];
    float scale1_1 = fp16_to_float(code2[absmax_q_idx1]);
    int64_t group_idx1 = block_idx1 / 256;
    float scale2_1 = fp16_to_float(absmax2[group_idx1]);
    result1 = nf4_val1 * scale1_1 * scale2_1;
  }

  __nv_bfloat16 bf16_0 = __float2bfloat16(result0);
  __nv_bfloat16 bf16_1 = __float2bfloat16(result1);

  uint32_t packed_output =
      (*reinterpret_cast<uint16_t*>(&bf16_0)) |
      (static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&bf16_1)) << 16);

  if (base_idx + 1 < num_elements) {
    reinterpret_cast<uint32_t*>(output)[base_idx / 2] = packed_output;
  } else {
    output[base_idx] = *reinterpret_cast<uint16_t*>(&bf16_0);
  }
}

// ============================================================================
// V3: 向量化读取 + 每线程处理 4 个元素（有未合并访存问题）
// ============================================================================
__global__ void nf4_dequantize_kernel_v3(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ absmax_q,
    const uint16_t* __restrict__ absmax2, const uint16_t* __restrict__ code2,
    uint16_t* __restrict__ output, int64_t num_elements, int32_t blocksize) {
  __shared__ float s_nf4_table[16];
  if (threadIdx.x < 16) {
    s_nf4_table[threadIdx.x] = NF4_TABLE[threadIdx.x];
  }
  __syncthreads();

  int64_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  if (base_idx >= num_elements) return;

  int64_t byte_idx = base_idx / 2;
  uint16_t packed_2bytes =
      *reinterpret_cast<const uint16_t*>(&packed[byte_idx]);

  uint8_t nf4_idx[4];
  nf4_idx[0] = packed_2bytes & 0xF;
  nf4_idx[1] = (packed_2bytes >> 4) & 0xF;
  nf4_idx[2] = (packed_2bytes >> 8) & 0xF;
  nf4_idx[3] = (packed_2bytes >> 12) & 0xF;

  float nf4_vals[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    nf4_vals[i] = s_nf4_table[nf4_idx[i]];
  }

  uint16_t bf16_output[4];

#pragma unroll
  for (int i = 0; i < 4; i++) {
    int64_t elem_idx = base_idx + i;
    if (elem_idx >= num_elements) {
      bf16_output[i] = 0;
      continue;
    }

    int64_t block_idx = elem_idx / blocksize;
    uint8_t absmax_q_idx = absmax_q[block_idx];
    float scale1 = fp16_to_float(code2[absmax_q_idx]);

    int64_t group_idx = block_idx / 256;
    float scale2 = fp16_to_float(absmax2[group_idx]);

    float result = nf4_vals[i] * scale1 * scale2;
    __nv_bfloat16 bf16 = __float2bfloat16(result);
    bf16_output[i] = *reinterpret_cast<uint16_t*>(&bf16);
  }

  if (base_idx + 3 < num_elements) {
    uint32_t pack0 =
        bf16_output[0] | (static_cast<uint32_t>(bf16_output[1]) << 16);
    uint32_t pack1 =
        bf16_output[2] | (static_cast<uint32_t>(bf16_output[3]) << 16);

    reinterpret_cast<uint32_t*>(output)[base_idx / 2] = pack0;
    reinterpret_cast<uint32_t*>(output)[base_idx / 2 + 1] = pack1;
  } else {
    for (int i = 0; i < 4 && base_idx + i < num_elements; i++) {
      output[base_idx + i] = bf16_output[i];
    }
  }
}

// ============================================================================
// V4: 优化访存合并 + 减少寄存器使用 + Warp 协作加载 scale
// ============================================================================
__global__ void nf4_dequantize_kernel_v4(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ absmax_q,
    const uint16_t* __restrict__ absmax2, const uint16_t* __restrict__ code2,
    uint16_t* __restrict__ output, int64_t num_elements, int32_t blocksize) {
  // Shared memory 缓存
  __shared__ float s_nf4_table[16];
  __shared__ float s_scale_cache[256];  // 缓存 code2 的 scale

  // 协作加载 NF4_TABLE
  if (threadIdx.x < 16) {
    s_nf4_table[threadIdx.x] = NF4_TABLE[threadIdx.x];
  }

  // 协作加载 code2（256 个元素）
  if (threadIdx.x < 256) {
    s_scale_cache[threadIdx.x] = fp16_to_float(code2[threadIdx.x]);
  }
  __syncthreads();

  // 每线程处理 8 个元素（4 bytes）- 提升合并访存
  const int ELEMS_PER_THREAD = 8;
  int64_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane_id = threadIdx.x % 32;

  // 计算 warp 对齐的起始位置
  int64_t warp_start = warp_id * 32 * ELEMS_PER_THREAD;
  int64_t base_idx = warp_start + lane_id * ELEMS_PER_THREAD;

  if (base_idx >= num_elements) return;

  // 向量化读取 4 bytes（8 个 4-bit 值）- 合并访存
  int64_t byte_idx = base_idx / 2;
  uint32_t packed_4bytes =
      reinterpret_cast<const uint32_t*>(packed)[byte_idx / 4];

  // 提取 8 个 4-bit 索引
  uint8_t nf4_idx[8];
  uint32_t shift_base = (byte_idx % 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    nf4_idx[i] = (packed_4bytes >> (shift_base + i * 4)) & 0xF;
  }

  // 批量处理 8 个元素 - 减少循环开销
  uint16_t bf16_out[8];

  // 预计算共享的 block_idx（减少除法）
  int64_t block_idx_base = base_idx / blocksize;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    int64_t elem_idx = base_idx + i;
    if (elem_idx >= num_elements) {
      bf16_out[i] = 0;
      continue;
    }

    // 查表
    float nf4_val = s_nf4_table[nf4_idx[i]];

    // 计算 block_idx（大部分情况下相同，减少访存）
    int64_t block_idx = (i == 0) ? block_idx_base : (elem_idx / blocksize);

    // 从 shared memory 读取 scale1
    uint8_t absmax_q_idx = absmax_q[block_idx];
    float scale1 = s_scale_cache[absmax_q_idx];

    // 计算 scale2
    int64_t group_idx = block_idx / 256;
    float scale2 = fp16_to_float(absmax2[group_idx]);

    // 融合乘法（FMA优化）
    float result = __fmaf_rn(nf4_val * scale1, scale2, 0.0f);

    // 转换为 BF16
    __nv_bfloat16 bf16 = __float2bfloat16(result);
    bf16_out[i] = *reinterpret_cast<uint16_t*>(&bf16);
  }

  // 向量化写回（4×uint32 = 8×BF16）- 合并访存
  if (base_idx + 7 < num_elements) {
    uint4 vec_out;
    vec_out.x = bf16_out[0] | (static_cast<uint32_t>(bf16_out[1]) << 16);
    vec_out.y = bf16_out[2] | (static_cast<uint32_t>(bf16_out[3]) << 16);
    vec_out.z = bf16_out[4] | (static_cast<uint32_t>(bf16_out[5]) << 16);
    vec_out.w = bf16_out[6] | (static_cast<uint32_t>(bf16_out[7]) << 16);

    reinterpret_cast<uint4*>(output)[base_idx / 8] = vec_out;
  } else {
    // 边界处理
    for (int i = 0; i < 8 && base_idx + i < num_elements; i++) {
      output[base_idx + i] = bf16_out[i];
    }
  }
}

// V5: 优化block_idx连续性 + 预加载scale
__global__ void nf4_dequantize_kernel_v5(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ absmax_q,
    const uint16_t* __restrict__ absmax2, const uint16_t* __restrict__ code2,
    uint16_t* __restrict__ output, int64_t num_elements, int32_t blocksize) {
  __shared__ float s_nf4_table[16];
  __shared__ float s_local_scales[256];  // 每个block缓存自己的256个scale

  // 1. 加载NF4表
  if (threadIdx.x < 16) {
    s_nf4_table[threadIdx.x] = NF4_TABLE[threadIdx.x];
  }

  // 2. 计算当前block负责的元素范围
  int64_t block_start_elem = blockIdx.x * blockDim.x * 8;
  int64_t block_start_blockidx = block_start_elem / blocksize;

  // 3. 预加载这个范围的scale (最多256个)
  int scale_range = min(256, (blockDim.x * 8 + blocksize - 1) / blocksize);
  for (int i = threadIdx.x; i < scale_range; i += blockDim.x) {
    int64_t target_block = block_start_blockidx + i;
    if (target_block < (num_elements + blocksize - 1) / blocksize) {
      uint8_t absmax_q_idx = absmax_q[target_block];
      s_local_scales[i] = fp16_to_float(code2[absmax_q_idx]);
    }
  }
  __syncthreads();

  // 4. 计算线程索引
  int64_t base_idx = block_start_elem + threadIdx.x * 8;
  if (base_idx >= num_elements) return;

  // 5. 向量化读取
  int64_t byte_idx = base_idx / 2;
  uint32_t packed_4bytes =
      reinterpret_cast<const uint32_t*>(packed)[byte_idx / 4];

  // 6. 解包8个NF4索引
  uint8_t nf4_idx[8];
  uint32_t shift_base = (byte_idx % 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    nf4_idx[i] = (packed_4bytes >> (shift_base + i * 4)) & 0xF;
  }

  // 7. 批量处理 - 使用shared memory的scale
  uint16_t bf16_out[8];

#pragma unroll
  for (int i = 0; i < 8; i++) {
    int64_t elem_idx = base_idx + i;
    if (elem_idx >= num_elements) break;

    // 查NF4表
    float nf4_val = s_nf4_table[nf4_idx[i]];

    // 从shared memory读取scale1 (避免重复访问全局内存)
    int64_t local_block_idx = (elem_idx - block_start_elem) / blocksize;
    float scale1 = s_local_scales[local_block_idx];

    // scale2 (访问频率较低,可以接受全局内存)
    int64_t global_block_idx = elem_idx / blocksize;
    int64_t group_idx = global_block_idx / 256;
    float scale2 = fp16_to_float(absmax2[group_idx]);

    // 融合乘法
    float result = nf4_val * scale1 * scale2;

    __nv_bfloat16 bf16 = __float2bfloat16(result);
    bf16_out[i] = *reinterpret_cast<uint16_t*>(&bf16);
  }

  // 8. 向量化写回
  if (base_idx + 7 < num_elements) {
    uint4 vec_out;
    vec_out.x = bf16_out[0] | (static_cast<uint32_t>(bf16_out[1]) << 16);
    vec_out.y = bf16_out[2] | (static_cast<uint32_t>(bf16_out[3]) << 16);
    vec_out.z = bf16_out[4] | (static_cast<uint32_t>(bf16_out[5]) << 16);
    vec_out.w = bf16_out[6] | (static_cast<uint32_t>(bf16_out[7]) << 16);
    reinterpret_cast<uint4*>(output)[base_idx / 8] = vec_out;
  } else {
    // 边界处理
    for (int i = 0; i < 8 && base_idx + i < num_elements; i++) {
      output[base_idx + i] = bf16_out[i];
    }
  }
}

// ============================================================================
// V6: 混合优化 - Warp Shuffle + 减少Shared Memory冲突
// ============================================================================
__global__ void nf4_dequantize_kernel_v6(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ absmax_q,
    const uint16_t* __restrict__ absmax2, const uint16_t* __restrict__ code2,
    uint16_t* __restrict__ output, int64_t num_elements, int32_t blocksize) {
  // 只缓存NF4表到shared memory (64 bytes)
  __shared__ float s_nf4_table[16];

  if (threadIdx.x < 16) {
    s_nf4_table[threadIdx.x] = NF4_TABLE[threadIdx.x];
  }
  __syncthreads();

  // 每线程处理16个元素 - 提升吞吐
  const int ELEMS_PER_THREAD = 16;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t base_idx = tid * ELEMS_PER_THREAD;

  if (base_idx >= num_elements) return;

  // 向量化读取8 bytes（16个4-bit值）
  int64_t byte_idx = base_idx / 2;

  // 使用uint64读取（更高带宽）
  uint64_t packed_8bytes = 0;
  if (byte_idx + 7 < (num_elements + 1) / 2) {
    packed_8bytes = *reinterpret_cast<const uint64_t*>(&packed[byte_idx]);
  } else {
    // 边界处理：逐字节读取
    for (int i = 0; i < 8 && byte_idx + i < (num_elements + 1) / 2; i++) {
      packed_8bytes |= (static_cast<uint64_t>(packed[byte_idx + i]) << (i * 8));
    }
  }

  // 解包16个NF4索引
  uint8_t nf4_idx[16];
#pragma unroll
  for (int i = 0; i < 16; i++) {
    nf4_idx[i] = (packed_8bytes >> (i * 4)) & 0xF;
  }

  // 批量处理 - 优化寄存器使用
  uint16_t bf16_out[16];

  // 预计算block_idx和scale（减少重复计算）
  // int64_t block_idx_start = base_idx / blocksize;

#pragma unroll 4  // 部分展开减少寄存器压力
  for (int i = 0; i < 16; i++) {
    int64_t elem_idx = base_idx + i;
    if (elem_idx >= num_elements) {
      bf16_out[i] = 0;
      continue;
    }

    // 从shared memory读取NF4值
    float nf4_val = s_nf4_table[nf4_idx[i]];

    // 计算block_idx（每64个元素才变一次）
    int64_t block_idx = elem_idx / blocksize;

    // 直接从全局内存读取scale（避免shared memory瓶颈）
    // 由于L1 cache的存在，连续访问很高效
    uint8_t absmax_q_idx = absmax_q[block_idx];
    float scale1 = fp16_to_float(code2[absmax_q_idx]);

    int64_t group_idx = block_idx / 256;
    float scale2 = fp16_to_float(absmax2[group_idx]);

    // FMA优化
    float result = __fmaf_rn(nf4_val, scale1 * scale2, 0.0f);

    __nv_bfloat16 bf16 = __float2bfloat16(result);
    bf16_out[i] = *reinterpret_cast<uint16_t*>(&bf16);
  }

  // 向量化写回 - 2×uint4 = 16×BF16
  if (base_idx + 15 < num_elements) {
    uint4 vec_out1, vec_out2;

    vec_out1.x = bf16_out[0] | (static_cast<uint32_t>(bf16_out[1]) << 16);
    vec_out1.y = bf16_out[2] | (static_cast<uint32_t>(bf16_out[3]) << 16);
    vec_out1.z = bf16_out[4] | (static_cast<uint32_t>(bf16_out[5]) << 16);
    vec_out1.w = bf16_out[6] | (static_cast<uint32_t>(bf16_out[7]) << 16);

    vec_out2.x = bf16_out[8] | (static_cast<uint32_t>(bf16_out[9]) << 16);
    vec_out2.y = bf16_out[10] | (static_cast<uint32_t>(bf16_out[11]) << 16);
    vec_out2.z = bf16_out[12] | (static_cast<uint32_t>(bf16_out[13]) << 16);
    vec_out2.w = bf16_out[14] | (static_cast<uint32_t>(bf16_out[15]) << 16);

    reinterpret_cast<uint4*>(output)[base_idx / 8] = vec_out1;
    reinterpret_cast<uint4*>(output)[base_idx / 8 + 1] = vec_out2;
  } else {
    // 边界处理
    for (int i = 0; i < 16 && base_idx + i < num_elements; i++) {
      output[base_idx + i] = bf16_out[i];
    }
  }
}

// ============================================================================
// V7: 终极优化 - 提前计算 Scale + 消除除法 + 完美合并访存
// ============================================================================
__global__ void nf4_dequantize_kernel_v7(
    const uint8_t* __restrict__ packed, const uint8_t* __restrict__ absmax_q,
    const uint16_t* __restrict__ absmax2, const uint16_t* __restrict__ code2,
    uint16_t* __restrict__ output, int64_t num_elements, int32_t blocksize) {
  __shared__ float s_nf4_table[16];
  if (threadIdx.x < 16) {
    s_nf4_table[threadIdx.x] = NF4_TABLE[threadIdx.x];
  }
  __syncthreads();

  const int ELEMS_PER_THREAD = 16;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t base_idx = tid * ELEMS_PER_THREAD;

  if (base_idx >= num_elements) return;

  // 1. 向量化读取 8 bytes (16个4-bit值)
  int64_t byte_idx = base_idx / 2;
  uint64_t packed_8bytes = 0;
  if (byte_idx + 7 < (num_elements + 1) / 2) {
    packed_8bytes = *reinterpret_cast<const uint64_t*>(&packed[byte_idx]);
  } else {
    for (int i = 0; i < 8 && byte_idx + i < (num_elements + 1) / 2; i++) {
      packed_8bytes |= (static_cast<uint64_t>(packed[byte_idx + i]) << (i * 8));
    }
  }

  // 解包16个NF4索引
  uint8_t nf4_idx[16];
#pragma unroll
  for (int i = 0; i < 16; i++) {
    nf4_idx[i] = (packed_8bytes >> (i * 4)) & 0xF;
  }

  uint16_t bf16_out[16];

  // 🌟🌟🌟 核心优化区：Scale 提前计算！只做一次！ 🌟🌟🌟
  // 因为 base_idx 是 16 的倍数，且 blocksize 最小是 32/64
  // 这 16 个元素绝对处于同一个 block 内，共享相同的 Scale
  int64_t block_idx = base_idx / blocksize;
  int64_t group_idx = block_idx / 256;

  uint8_t absmax_q_idx = absmax_q[block_idx];
  float scale1 = fp16_to_float(code2[absmax_q_idx]);
  float scale2 = fp16_to_float(absmax2[group_idx]);
  float combined_scale = scale1 * scale2;
// 🌟🌟🌟 核心优化区结束 🌟🌟🌟

// 计算结果
#pragma unroll
  for (int i = 0; i < 16; i++) {
    if (base_idx + i >= num_elements) {
      bf16_out[i] = 0;
      continue;
    }
    // 查表后直接乘上提前算好的 combined_scale
    float nf4_val = s_nf4_table[nf4_idx[i]];
    float result = nf4_val * combined_scale;

    __nv_bfloat16 bf16 = __float2bfloat16(result);
    bf16_out[i] = *reinterpret_cast<uint16_t*>(&bf16);
  }

  // 向量化写回 (32 bytes 一次性写回)
  if (base_idx + 15 < num_elements) {
    uint4 vec_out1, vec_out2;

    vec_out1.x = bf16_out[0] | (static_cast<uint32_t>(bf16_out[1]) << 16);
    vec_out1.y = bf16_out[2] | (static_cast<uint32_t>(bf16_out[3]) << 16);
    vec_out1.z = bf16_out[4] | (static_cast<uint32_t>(bf16_out[5]) << 16);
    vec_out1.w = bf16_out[6] | (static_cast<uint32_t>(bf16_out[7]) << 16);

    vec_out2.x = bf16_out[8] | (static_cast<uint32_t>(bf16_out[9]) << 16);
    vec_out2.y = bf16_out[10] | (static_cast<uint32_t>(bf16_out[11]) << 16);
    vec_out2.z = bf16_out[12] | (static_cast<uint32_t>(bf16_out[13]) << 16);
    vec_out2.w = bf16_out[14] | (static_cast<uint32_t>(bf16_out[15]) << 16);

    reinterpret_cast<uint4*>(output)[base_idx / 8] = vec_out1;
    reinterpret_cast<uint4*>(output)[base_idx / 8 + 1] = vec_out2;
  } else {
    // 边界处理
    for (int i = 0; i < 16 && base_idx + i < num_elements; i++) {
      output[base_idx + i] = bf16_out[i];
    }
  }
}

// ============================================================================
// Host 端启动函数
// ============================================================================
void launch_nf4_dequantize(const uint8_t* d_packed, const uint8_t* d_absmax_q,
                           const uint16_t* d_absmax2, const uint16_t* d_code2,
                           uint16_t* d_output, int64_t num_elements,
                           int32_t blocksize) {
  // 使用V6原版（已验证稳定）
  int threads = 256;
  int blocks = (num_elements / 16 + threads - 1) / threads;

  nf4_dequantize_kernel_v7<<<blocks, threads>>>(d_packed, d_absmax_q, d_absmax2,
                                                d_code2, d_output, num_elements,
                                                blocksize);

  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
  }
}