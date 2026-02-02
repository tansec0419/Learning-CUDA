#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "../tester/utils.h"

__device__ __forceinline__ void kahan_accumulate(double& sum,
                                                 double& compensation,
                                                 double input) {
  double y = input - compensation;
  double t = sum + y;
  compensation = (t - sum) - y;
  sum = t;
}

// 8 字节对齐
__host__ __device__ __forceinline__ size_t align_to_double(size_t bytes) {
  return (bytes + sizeof(double) - 1) / sizeof(double) * sizeof(double);
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

// Warp 内归约求和
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__global__ void trace_kernel_v3(const T* __restrict__ input,
                                T* __restrict__ block_sums, size_t rows,
                                size_t cols) {
  size_t diag_len = (rows < cols) ? rows : cols;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // 线程内累加
  double thread_sum = 0.0;
  for (size_t i = idx; i < diag_len; i += stride) {
    thread_sum += static_cast<double>(input[i * cols + i]);
  }

  // Warp 内归约
  double warp_sum = thread_sum;
  for (int offset = 16; offset > 0; offset /= 2) {
    warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
  }

  // 每个 warp 的结果写入共享内存
  static __shared__ double smem_warp_sums[32];
  int lane_id = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  if (lane_id == 0) {
    smem_warp_sums[warp_id] = warp_sum;
  }
  __syncthreads();

  // 第一个 warp 对所有 warp 结果再次归约
  if (warp_id == 0) {
    double block_sum =
        (lane_id < (blockDim.x / 32)) ? smem_warp_sums[lane_id] : 0.0;
    block_sum = warp_reduce_sum(block_sum);
    if (lane_id == 0) {
      block_sums[blockIdx.x] = static_cast<T>(block_sum);
    }
  }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  if (h_input.empty() || rows == 0 || cols == 0) return T(0);

  const size_t diag_len = std::min(rows, cols);
  const size_t bytes = h_input.size() * sizeof(T);

  const int block_size = 256;
  // 限制 grid 大小避免过多空闲线程
  const int grid_size =
      std::min((size_t)256, (diag_len + block_size - 1) / block_size);

  T* d_in = nullptr;
  T* d_block_sums = nullptr;
  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_block_sums, grid_size * sizeof(T));

  cudaMemcpy(d_in, h_input.data(), bytes, cudaMemcpyHostToDevice);

  trace_kernel_v3<T><<<grid_size, block_size>>>(d_in, d_block_sums, rows, cols);

  std::vector<T> h_sums(grid_size);
  cudaMemcpy(h_sums.data(), d_block_sums, grid_size * sizeof(T),
             cudaMemcpyDeviceToHost);

  double final_sum = 0.0;
  for (T val : h_sums) final_sum += static_cast<double>(val);

  cudaFree(d_in);
  cudaFree(d_block_sums);

  return static_cast<T>(final_sum);
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads,
 * head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len,
 * query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query
 * attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

template <typename T>
__global__ void flash_attn_kernel_optimized(
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
    T* __restrict__ O, int batch, int seq_q, int seq_k, int num_heads_q,
    int num_heads_kv, int head_dim, int TileQ, int TileKV, bool is_causal,
    float scale_val) {
  // 动态共享内存指针分配
  extern __shared__ char smem_buffer[];
  size_t offset = 0;

  T* smem_q = reinterpret_cast<T*>(smem_buffer + offset);
  offset += align_to_double(TileQ * head_dim * sizeof(T));

  T* smem_k = reinterpret_cast<T*>(smem_buffer + offset);
  offset += align_to_double(TileKV * head_dim * sizeof(T));

  T* smem_v = reinterpret_cast<T*>(smem_buffer + offset);
  offset += align_to_double(TileKV * head_dim * sizeof(T));

  // 使用 double 存储 Score 和 Output Accumulator 以通过高精度测试点
  double* smem_scores = reinterpret_cast<double*>(smem_buffer + offset);
  offset += align_to_double(TileQ * TileKV * sizeof(double));

  double* smem_o_acc = reinterpret_cast<double*>(smem_buffer + offset);
  offset += align_to_double(TileQ * head_dim * sizeof(double));

  double* smem_m = reinterpret_cast<double*>(smem_buffer + offset);
  offset += align_to_double(TileQ * sizeof(double));

  double* smem_l = reinterpret_cast<double*>(smem_buffer + offset);

  // 计算当前 block 负责的 query 范围
  int bx = blockIdx.x;  // Q tile 索引
  int by = blockIdx.y;  // query head 索引
  int bz = blockIdx.z;  // batch 索引

  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  int q_start = bx * TileQ;
  int q_valid_len = min(TileQ, seq_q - q_start);
  int kv_head_idx = (by * num_heads_kv) / num_heads_q;  // GQA 处理

  // 计算全局偏移
  size_t batch_offset_q = (size_t)bz * seq_q * num_heads_q * head_dim;
  size_t batch_offset_kv = (size_t)bz * seq_k * num_heads_kv * head_dim;
  size_t head_offset_q = (size_t)by * head_dim;
  size_t head_offset_kv = (size_t)kv_head_idx * head_dim;

  // 初始化 online softmax 状态：m = -inf, l = 0, O = 0
  for (int i = tid; i < TileQ; i += nthreads) {
    smem_m[i] = -INFINITY;
    smem_l[i] = 0.0;
  }
  for (int i = tid; i < TileQ * head_dim; i += nthreads) {
    smem_o_acc[i] = 0.0;
  }
  __syncthreads();

  // 加载 Q tile 到共享内存
  for (int i = tid; i < q_valid_len * head_dim; i += nthreads) {
    int r = i / head_dim;
    int c = i % head_dim;
    size_t idx = batch_offset_q + (q_start + r) * num_heads_q * head_dim +
                 head_offset_q + c;
    smem_q[i] = Q[idx];
  }
  // Padding无效位置
  for (int i = q_valid_len * head_dim + tid; i < TileQ * head_dim;
       i += nthreads) {
    smem_q[i] = static_cast<T>(0.0f);
  }
  __syncthreads();

  // 循环 KV Tiles
  int num_kv_tiles = (seq_k + TileKV - 1) / TileKV;
  double scale = static_cast<double>(scale_val);

  for (int j = 0; j < num_kv_tiles; ++j) {
    int kv_start = j * TileKV;
    int kv_valid_len = min(TileKV, seq_k - kv_start);

    // 加载 K, V
    for (int i = tid; i < kv_valid_len * head_dim; i += nthreads) {
      int r = i / head_dim;
      int c = i % head_dim;
      size_t idx_base = batch_offset_kv +
                        (kv_start + r) * num_heads_kv * head_dim +
                        head_offset_kv + c;
      smem_k[i] = K[idx_base];
      smem_v[i] = V[idx_base];
    }
    __syncthreads();

    // 计算 Score (S = Q * K^T)
    // 使用 double 避免精度损失
    for (int i = tid; i < q_valid_len * kv_valid_len; i += nthreads) {
      int row = i / kv_valid_len;
      int col = i % kv_valid_len;

      int global_q = q_start + row;
      int global_k = kv_start + col;

      if (is_causal && global_k > global_q) {
        smem_scores[row * TileKV + col] = -INFINITY;
        continue;
      }

      double sum = 0.0;
      if constexpr (std::is_same_v<T, float>) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot =
              fmaf(smem_q[row * head_dim + d], smem_k[col * head_dim + d], dot);
        }
        sum = static_cast<double>(dot);
      } else {
        for (int d = 0; d < head_dim; ++d) {
          sum += static_cast<double>(smem_q[row * head_dim + d]) *
                 static_cast<double>(smem_k[col * head_dim + d]);
        }
      }
      smem_scores[row * TileKV + col] = sum * scale;
    }
    __syncthreads();

    // Online Softmax 更新
    // 对每行 Q：更新 m, l, O
    for (int r = tid; r < q_valid_len; r += nthreads) {
      double m_prev = smem_m[r];
      double l_prev = smem_l[r];

      // 当前tile的行最大值
      double m_curr = -INFINITY;
      for (int k = 0; k < kv_valid_len; ++k) {
        double s = smem_scores[r * TileKV + k];
        if (s > m_curr) m_curr = s;
      }

      if (m_curr == -INFINITY) continue;

      // 新的全局最大值
      double m_new = fmax(m_prev, m_curr);
      double alpha_prev = exp(m_prev - m_new);  // 旧输出的缩放因子
      double alpha_curr = exp(m_curr - m_new);

      // 更新分母l
      double p_sum = 0.0, p_err = 0.0;
      for (int k = 0; k < kv_valid_len; ++k) {
        double s = smem_scores[r * TileKV + k];
        if (s == -INFINITY) continue;
        double p = exp(s - m_new);
        kahan_accumulate(p_sum, p_err, p);
      }
      double l_new = l_prev * alpha_prev + p_sum;

      smem_m[r] = m_new;
      smem_l[r] = l_new;

      // 更新O
      for (int d = 0; d < head_dim; ++d) {
        double o_val = smem_o_acc[r * head_dim + d];
        o_val *= alpha_prev;

        // 累加P*V
        double pv_err = 0.0;
        for (int k = 0; k < kv_valid_len; ++k) {
          double s = smem_scores[r * TileKV + k];
          if (s == -INFINITY) continue;

          double p = exp(s - m_new);
          double v = static_cast<double>(smem_v[k * head_dim + d]);
          kahan_accumulate(o_val, pv_err, p * v);
        }
        smem_o_acc[r * head_dim + d] = o_val;
      }
    }
    __syncthreads();
  }

  // 归一化并写回全局内存
  for (int i = tid; i < q_valid_len * head_dim; i += nthreads) {
    int r = i / head_dim;
    int d = i % head_dim;

    double l_val = smem_l[r];
    size_t idx = batch_offset_q + (q_start + r) * num_heads_q * head_dim +
                 head_offset_q + d;

    if (l_val > 1e-9) {
      O[idx] = static_cast<T>(smem_o_acc[i] / l_val);
    } else {
      O[idx] = static_cast<T>(0.0f);
    }
  }
}

// 计算共享内存需求
template <typename T>
size_t calculate_smem_usage(int Bq, int Bkv, int head_dim) {
  size_t total = 0;
  total += align_to_double(Bq * head_dim * sizeof(T));       // Q
  total += align_to_double(Bkv * head_dim * sizeof(T));      // K
  total += align_to_double(Bkv * head_dim * sizeof(T));      // V
  total += align_to_double(Bq * Bkv * sizeof(double));       // Scores
  total += align_to_double(Bq * head_dim * sizeof(double));  // O accumulator
  total += align_to_double(Bq * sizeof(double));             // m
  total += align_to_double(Bq * sizeof(double));             // l
  return total;
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int tgt_len, int src_len, int query_heads,
                    int kv_heads, int head_dim, bool is_causal) {
  // TODO: Implement the flash attention function
  if (h_q.empty()) return;
  size_t size_q = h_q.size() * sizeof(T);
  size_t size_k = h_k.size() * sizeof(T);
  size_t size_v = h_v.size() * sizeof(T);
  size_t size_o = batch_size * tgt_len * query_heads * head_dim * sizeof(T);

  h_o.resize(size_o / sizeof(T));

  // 分配设备内存
  T *d_q, *d_k, *d_v, *d_o;
  RUNTIME_CHECK(cudaMalloc(&d_q, size_q));
  RUNTIME_CHECK(cudaMalloc(&d_k, size_k));
  RUNTIME_CHECK(cudaMalloc(&d_v, size_v));
  RUNTIME_CHECK(cudaMalloc(&d_o, size_o));

  // 拷贝输入数据
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), size_q, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), size_k, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), size_v, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_o, 0, size_o));

  // 动态 Tile 选择策略：根据 head_dim 选择合适的 block size
  int TileQ = 16;
  int TileKV = 32;

  if (head_dim <= 64) {
    TileQ = 32;
    TileKV = 64;
  } else if (head_dim >= 128) {
    TileQ = 16;
    TileKV = 32;
  }

  const size_t MAX_SMEM = 48 * 1024;
  size_t needed = calculate_smem_usage<T>(TileQ, TileKV, head_dim);

  // 如果超限则降级 tile 大小
  if (needed > MAX_SMEM) {
    TileQ = 8;
    TileKV = 16;
    needed = calculate_smem_usage<T>(TileQ, TileKV, head_dim);

    if (needed > MAX_SMEM) {
      TileQ = 4;
      TileKV = 8;
      needed = calculate_smem_usage<T>(TileQ, TileKV, head_dim);
    }
  }

  float scale = 1.0f / sqrtf((float)head_dim);

  dim3 grid((tgt_len + TileQ - 1) / TileQ, query_heads, batch_size);
  dim3 block(256);

  flash_attn_kernel_optimized<<<grid, block, needed>>>(
      d_q, d_k, d_v, d_o, batch_size, tgt_len, src_len, query_heads, kv_heads,
      head_dim, TileQ, TileKV, is_causal, scale);

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, size_o, cudaMemcpyDeviceToHost));

  // 拷回结果
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&,
                                    const std::vector<float>&,
                                    const std::vector<float>&,
                                    std::vector<float>&, int, int, int, int,
                                    int, int, bool);
template void flashAttention<half>(const std::vector<half>&,
                                   const std::vector<half>&,
                                   const std::vector<half>&, std::vector<half>&,
                                   int, int, int, int, int, int, bool);