#include <cuda_fp16.h>

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

// 辅助函数：将全局内存的数据加载到共享内存
template <typename T>
__device__ void load_to_shared(T* dst, const T* src, int num_elements) {
  int tid = threadIdx.x;    // 当前线程在Block内的 ID
  int stride = blockDim.x;  // 每次跨越的步长等于线程总数

  // 让线程从 tid 开始，每次跳跃 stride，直到搬完 num_elements
  for (int i = tid; i < num_elements; i += blockDim.x) {
    dst[i] = src[i];
  }
}

__global__ void flash_attention_kernel(/* Add necessary parameters */) {
  // 1.获取当前block的身份信息
  int batch_id = blockIdx.x;
  int head_id = blockIdx.y;
  int q_block_id = blockIdx.z;
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  // TODO: Implement the flash attention function

  // 1.设定超参数
  const int blockSize = 128;
  const int q_tile_size = 64;
  const int kv_tile_size = 64;

  // 2.定义grid和block的维度
  int num_query_blocks = (target_seq_len + q_tile_size - 1) / q_tile_size;
  // x-batch数量；y-head数量；z-query分块数量
  dim3 grid_Dim(batch_size, query_heads, num_query_blocks);
  dim3 block_Dim(blockSize);

  // 3.计算动态shared memory大小
  size_t smem_size = 3 * q_tile_size * head_dim * sizeof(T);

  // 4.启动kernel
  flash_attention_kernel<<<grid_Dim, block_Dim, smem_size>>>(
      /* Pass necessary arguments */);
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
