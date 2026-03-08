#pragma once

#include <cstdint>
#include <string>
#include <vector>

// NF4 权重数据结构
struct NF4Data {
  int64_t rows;       // 矩阵行数
  int64_t cols;       // 矩阵列数
  int32_t blocksize;  // 块大小（通常是64）

  // 实际数据
  std::vector<uint8_t> packed;    // 压缩的权重（每字节2个4-bit值）
  std::vector<uint8_t> absmax_q;  // 一级缩放因子（量化的）
  std::vector<uint16_t> absmax2;  // 二级缩放因子（FP16格式）
  std::vector<uint16_t> code2;    // 码表（用于解码absmax_q）
  float offset;                   // 偏移量

  // 辅助函数：计算总元素数
  int64_t total_elements() const { return rows * cols; }

  // 辅助函数：计算块的数量
  int64_t num_blocks() const {
    return (total_elements() + blocksize - 1) / blocksize;
  }
};

// 配置参数
struct Config {
  int32_t blocksize;         // 块大小
  std::string compute_type;  // "bf16" 或 "fp16"
  std::string target_gpu;    // "A100", "T4" 等
};

// 函数声明

// 加载 NF4 数据文件
bool load_nf4_data(const std::string& filename, NF4Data& data);

// 加载配置文件
bool load_config(const std::string& filename, Config& config);

// 保存输出
bool save_output(const std::string& filename, const void* data, size_t size);

// 打印数据信息
void print_nf4_info(const NF4Data& data);

// CUDA kernel 启动函数（在 .cu 文件中实现）
void launch_nf4_dequantize(
    const uint8_t* d_packed,    // 输入：压缩的权重
    const uint8_t* d_absmax_q,  // 输入：一级缩放因子
    const uint16_t* d_absmax2,  // 输入：二级缩放因子
    const uint16_t* d_code2,    // 输入：码表
    uint16_t* d_output,         // 输出：解量化后的权重（BF16/FP16）
    int64_t num_elements,       // 总元素数
    int32_t blocksize           // 块大小
);
