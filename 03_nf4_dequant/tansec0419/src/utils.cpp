#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "nf4_dequant.h"

bool load_nf4_data(const std::string& filename, NF4Data& data) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "❌ Failed to open file: " << filename << std::endl;
    return false;
  }

  // 读取头部
  file.read(reinterpret_cast<char*>(&data.rows), sizeof(int64_t));
  file.read(reinterpret_cast<char*>(&data.cols), sizeof(int64_t));
  file.read(reinterpret_cast<char*>(&data.blocksize), sizeof(int32_t));

  // 计算各部分的大小
  int64_t num_weights = data.rows * data.cols;
  int64_t packed_size = num_weights / 2;
  int64_t num_blocks = num_weights / data.blocksize;
  int64_t num_groups = num_blocks / 256 + 1;

  // 调整容器大小
  data.packed.resize(packed_size);
  data.absmax_q.resize(num_blocks);
  data.absmax2.resize(num_groups);
  data.code2.resize(256);

  // 读取数据
  file.read(reinterpret_cast<char*>(data.packed.data()), packed_size);
  file.read(reinterpret_cast<char*>(data.absmax_q.data()), num_blocks);
  file.read(reinterpret_cast<char*>(data.absmax2.data()),
            num_groups * sizeof(uint16_t));
  file.read(reinterpret_cast<char*>(data.code2.data()), 256 * sizeof(uint16_t));
  file.read(reinterpret_cast<char*>(&data.offset), sizeof(float));

  if (!file) {
    std::cerr << "❌ Error reading data from file" << std::endl;
    return false;
  }

  std::cout << "✅ Successfully loaded NF4 data from " << filename << std::endl;
  return true;
}

bool load_config(const std::string& filename, Config& config) {
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "⚠️  Config file not found, using defaults" << std::endl;
    config.blocksize = 64;
    config.compute_type = "bf16";
    config.target_gpu = "A100";
    return true;  // 使用默认值也算成功
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string key, eq, value;
    if (iss >> key >> eq >> value) {
      if (key == "blocksize") {
        config.blocksize = std::stoi(value);
      } else if (key == "compute_type") {
        // 去掉引号
        value.erase(remove(value.begin(), value.end(), '"'), value.end());
        config.compute_type = value;
      } else if (key == "target_gpu") {
        value.erase(remove(value.begin(), value.end(), '"'), value.end());
        config.target_gpu = value;
      }
    }
  }

  std::cout << "✅ Loaded config from " << filename << std::endl;
  return true;
}

bool save_output(const std::string& filename, const void* data, size_t size) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "❌ Failed to create output file: " << filename << std::endl;
    return false;
  }

  file.write(reinterpret_cast<const char*>(data), size);
  std::cout << "✅ Saved output to " << filename << " (" << size << " bytes)"
            << std::endl;
  return true;
}

void print_nf4_info(const NF4Data& data) {
  std::cout << "\n📊 NF4 Data Information:" << std::endl;
  std::cout << "   Rows: " << data.rows << std::endl;
  std::cout << "   Cols: " << data.cols << std::endl;
  std::cout << "   Blocksize: " << data.blocksize << std::endl;
  std::cout << "   Total elements: " << data.total_elements() << std::endl;
  std::cout << "   Total blocks: " << data.num_blocks() << std::endl;
  std::cout << "   Packed size: " << data.packed.size() << " bytes"
            << std::endl;
  std::cout << "   Offset: " << data.offset << std::endl;
  std::cout << std::endl;
}