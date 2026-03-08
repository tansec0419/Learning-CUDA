#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "nf4_dequant.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "❌ CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main(int argc, char** argv) {
  std::cout << "🚀 NF4 Dequantization Program" << std::endl;
  std::cout << "================================\n" << std::endl;

  // 1. 解析命令行参数
  std::string data_file = "data/input.bin";
  std::string config_file = "data/config.txt";

  if (argc >= 2) data_file = argv[1];
  if (argc >= 3) config_file = argv[2];

  // 2. 加载数据
  NF4Data data;
  Config config;

  if (!load_nf4_data(data_file, data)) {
    return 1;
  }

  load_config(config_file, config);
  print_nf4_info(data);

  std::cout << "⚙️  Config: blocksize=" << config.blocksize
            << ", type=" << config.compute_type << ", GPU=" << config.target_gpu
            << "\n"
            << std::endl;

  // 3. 分配 GPU 内存
  std::cout << "📦 Allocating GPU memory..." << std::endl;

  uint8_t *d_packed, *d_absmax_q;
  uint16_t *d_absmax2, *d_code2, *d_output;

  size_t packed_size = data.packed.size();
  size_t absmax_q_size = data.absmax_q.size();
  size_t absmax2_size = data.absmax2.size() * sizeof(uint16_t);
  size_t code2_size = 256 * sizeof(uint16_t);
  size_t output_size = data.total_elements() * sizeof(uint16_t);

  CUDA_CHECK(cudaMalloc(&d_packed, packed_size));
  CUDA_CHECK(cudaMalloc(&d_absmax_q, absmax_q_size));
  CUDA_CHECK(cudaMalloc(&d_absmax2, absmax2_size));
  CUDA_CHECK(cudaMalloc(&d_code2, code2_size));
  CUDA_CHECK(cudaMalloc(&d_output, output_size));

  // 4. 拷贝数据到 GPU
  std::cout << "📤 Copying data to GPU..." << std::endl;

  CUDA_CHECK(cudaMemcpy(d_packed, data.packed.data(), packed_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_absmax_q, data.absmax_q.data(), absmax_q_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_absmax2, data.absmax2.data(), absmax2_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_code2, data.code2.data(), code2_size,
                        cudaMemcpyHostToDevice));

  // 5. 运行 kernel（计时）
  std::cout << "⚡ Running dequantization kernel..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  launch_nf4_dequantize(d_packed, d_absmax_q, d_absmax2, d_code2, d_output,
                        data.total_elements(), data.blocksize);

  auto end = std::chrono::high_resolution_clock::now();
  float time_ms = std::chrono::duration<float, std::milli>(end - start).count();

  std::cout << "✅ Kernel completed in " << time_ms << " ms" << std::endl;

  // 6. 计算带宽
  float bytes_read = packed_size + absmax_q_size + absmax2_size + code2_size;
  float bytes_write = output_size;
  float total_gb = (bytes_read + bytes_write) / 1e9;
  float bandwidth_gb_s = total_gb / (time_ms / 1000.0);

  std::cout << "📊 Effective bandwidth: " << bandwidth_gb_s << " GB/s"
            << std::endl;

  // 7. 拷贝结果回 CPU
  std::cout << "📥 Copying results back..." << std::endl;

  std::vector<uint16_t> output(data.total_elements());
  CUDA_CHECK(
      cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

  // 8. 保存结果
  save_output("output.bin", output.data(), output_size);

  // 9. 清理
  cudaFree(d_packed);
  cudaFree(d_absmax_q);
  cudaFree(d_absmax2);
  cudaFree(d_code2);
  cudaFree(d_output);

  std::cout << "\n🎉 Done!" << std::endl;
  return 0;
}