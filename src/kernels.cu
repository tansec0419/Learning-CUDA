#include <cuda_fp16.h>

#include <cmath>
#include <vector>

#include "../tester/utils.h"

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
template <typename T>
__global__ void trace_kernel(const T* d_input, T* d_result, int cols,
                             int limit) {
  // 1. 计算当前线程的全局 ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 2. 边界检查：防止线程 ID 超过了我们要加的元素个数
  if (tid < limit) {
    // 3. 计算要读取的数据在 d_input 中的索引
    int idx = tid * (cols + 1);

    // 4. 原子加法：把 d_input[idx] 安全地加到 d_result 指向的内存里
    atomicAdd(d_result, d_input[idx]);
  }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function

  // 1.准备输入数据
  size_t bytes = rows * cols * sizeof(T);  // 整个矩阵的字节数
  T* d_input;                              // 指向GPU内存
  cudaMalloc((void**)&d_input, bytes);
  cudaMemcpy((void*)d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

  // 2.准备输出数据
  T* d_result;
  cudaMalloc((void**)&d_result, sizeof(T));
  cudaMemset(d_result, 0, sizeof(T));  // 初始化为0

  // 3.计算grid和block的大小
  size_t limit = min(rows, cols);  // 对角线元素个数
  int blockSize = 256;
  int gridSize = (limit + blockSize - 1) / blockSize;
  trace_kernel<<<gridSize, blockSize>>>(d_input, d_result, cols, limit);

  // 4.从GPU拷贝结果到CPU
  T h_result;
  cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

  // 5.释放GPU内存
  cudaFree(d_input);
  cudaFree(d_result);

  return h_result;
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

// 辅助函数：二维跨步读取将全局内存的数据加载到共享内存（添加边界检查）
template <typename T>
__device__ void load_2d(T* dst, const T* src, int rows, int cols,
                        int src_stride, int max_rows) {
  int tid = threadIdx.x;
  int total_elements = rows * cols;

  for (int i = tid; i < total_elements; i += blockDim.x) {
    int row = i / cols;
    int col = i % cols;
    // 边界检查：确保不读取超出范围的数据
    if (row < max_rows) {
      dst[i] = src[row * src_stride + col];
    } else {
      dst[i] = (T)0.0f;  // 填充0
    }
  }
}

template <typename T>
__global__ void flash_attention_kernel(
    const int q_tile_size, const int kv_tile_size, const int batch_size,
    const int head_dim, const int target_seq_len, const int src_seq_len,
    const int query_heads, const int kv_heads, const T* dev_q, const T* dev_k,
    const T* dev_v, T* dev_o, const bool is_causal) {
  int tid = threadIdx.x;
  extern __shared__ char smem_byte[];
  T* smem = reinterpret_cast<T*>(smem_byte);

  T* s_q = smem;
  T* s_k = s_q + q_tile_size * head_dim;
  T* s_v = s_k + kv_tile_size * head_dim;

  int batch_id = blockIdx.x;
  int head_id = blockIdx.y;
  int q_block_id = blockIdx.z;
  int tgt_seq_id = q_block_id * q_tile_size;

  int stride_head_q = head_dim;
  int stride_seq_q = query_heads * stride_head_q;
  int stride_batch_q = target_seq_len * stride_seq_q;

  long long q_offset = (long long)batch_id * stride_batch_q +
                       (long long)head_id * stride_head_q +
                       (long long)tgt_seq_id * stride_seq_q;

  int stride_head_kv = head_dim;
  int stride_seq_kv = kv_heads * stride_head_kv;
  int stride_batch_kv = src_seq_len * stride_seq_kv;

  int kv_head_id = head_id / (query_heads / kv_heads);

  const T* q_ptr = dev_q + q_offset;
  int actual_q_rows = min(q_tile_size, target_seq_len - tgt_seq_id);
  load_2d(s_q, q_ptr, q_tile_size, head_dim, stride_seq_q, actual_q_rows);

  __syncthreads();

  int num_kv_blocks = (src_seq_len + kv_tile_size - 1) / kv_tile_size;

  if (tid < q_tile_size && (tgt_seq_id + tid) < target_seq_len) {
    float acc[128] = {0.0f};
    float l = 0.0f;
    float m = -INFINITY;
    int global_q_idx = tgt_seq_id + tid;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
      int src_seq_id = kv_block_idx * kv_tile_size;
      int actual_kv_rows = min(kv_tile_size, src_seq_len - src_seq_id);

      long long k_offset = (long long)batch_id * stride_batch_kv +
                           (long long)kv_head_id * stride_head_kv +
                           (long long)src_seq_id * stride_seq_kv;
      long long v_offset = k_offset;

      load_2d(s_k, dev_k + k_offset, kv_tile_size, head_dim, stride_seq_kv,
              actual_kv_rows);
      load_2d(s_v, dev_v + v_offset, kv_tile_size, head_dim, stride_seq_kv,
              actual_kv_rows);

      __syncthreads();

      // 处理当前 KV block
      for (int j = 0; j < actual_kv_rows; ++j) {
        int global_k_idx = src_seq_id + j;

        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          score +=
              (float)s_q[tid * head_dim + d] * (float)s_k[j * head_dim + d];
        }
        score *= scale;

        // 因果掩码
        if (is_causal && global_k_idx > global_q_idx) {
          continue;  // 跳过而不是设为 -INF
        }

        // Online softmax 更新
        float m_prev = m;
        m = fmaxf(m_prev, score);

        // 计算缩放因子
        float exp_m_diff = expf(m_prev - m);
        float exp_score = expf(score - m);

        // 更新累加器：重新缩放旧值 + 新值
        for (int d = 0; d < head_dim; ++d) {
          acc[d] =
              acc[d] * exp_m_diff + exp_score * (float)s_v[j * head_dim + d];
        }

        // 更新归一化因子
        l = l * exp_m_diff + exp_score;
      }

      __syncthreads();
    }

    // 写回结果
    if (l > 0.0f) {  // 防止除0
      for (int d = 0; d < head_dim; ++d) {
        long long o_offset = (long long)batch_id * stride_batch_q +
                             (long long)head_id * stride_head_q +
                             (long long)global_q_idx * stride_seq_q + d;
        dev_o[o_offset] = (T)(acc[d] / l);
      }
    }
  }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  // TODO: Implement the flash attention function

  // 1.计算字节数
  size_t q_bytes =
      batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
  size_t k_bytes = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
  size_t v_bytes = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
  size_t o_bytes =
      batch_size * target_seq_len * query_heads * head_dim * sizeof(T);

  // 2.在GPU上分配内存
  T* d_q;
  T* d_k;
  T* d_v;
  T* d_o;
  cudaMalloc((void**)&d_q, q_bytes);
  cudaMalloc((void**)&d_k, k_bytes);
  cudaMalloc((void**)&d_v, v_bytes);
  cudaMalloc((void**)&d_o, o_bytes);

  // 3.将数据从CPU复制到GPU
  cudaMemcpy((void*)d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_k, h_k.data(), k_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_v, h_v.data(), v_bytes, cudaMemcpyHostToDevice);
  // cudaMemcpy((void*)d_o, h_o.data(), o_bytes, cudaMemcpyHostToDevice);

  // 1.查询设备属性，获取最大共享内存
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  size_t max_smem = prop.sharedMemPerBlock;

  // 2.根据 head_dim 动态调整 tile size
  int q_tile_size, kv_tile_size, blockSize;

  if (head_dim <= 64) {
    q_tile_size = 64;
    kv_tile_size = 64;
    blockSize = 128;
  } else if (head_dim <= 128) {
    q_tile_size = 32;
    kv_tile_size = 32;
    blockSize = 64;
  } else {
    q_tile_size = 16;
    kv_tile_size = 16;
    blockSize = 32;
  }

  // 3.计算所需共享内存并检查是否超限
  size_t smem_size = (q_tile_size + 2 * kv_tile_size) * head_dim * sizeof(T);

  // 如果超限，继续缩小 tile size
  while (smem_size > max_smem && q_tile_size > 8) {
    q_tile_size /= 2;
    kv_tile_size /= 2;
    blockSize /= 2;
    smem_size = (q_tile_size + 2 * kv_tile_size) * head_dim * sizeof(T);
  }

  // 4.定义grid和block的维度
  int num_query_blocks = (target_seq_len + q_tile_size - 1) / q_tile_size;
  dim3 grid_Dim(batch_size, query_heads, num_query_blocks);
  dim3 block_Dim(blockSize);

  // 5.启动kernel
  flash_attention_kernel<<<grid_Dim, block_Dim, smem_size>>>(
      q_tile_size, kv_tile_size, batch_size, head_dim, target_seq_len,
      src_seq_len, query_heads, kv_heads, d_q, d_k, d_v, d_o, is_causal);

  // 6.检查 kernel 启动错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // 如果仍然失败，可以在这里打印错误信息
    // printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  // 7.等待 kernel 完成
  cudaDeviceSynchronize();

  // 8.将结果从GPU复制回CPU
  cudaMemcpy(h_o.data(), d_o, o_bytes, cudaMemcpyDeviceToHost);

  // 9.释放GPU内存
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
