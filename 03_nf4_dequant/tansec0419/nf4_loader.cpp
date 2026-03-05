#include <stdint.h>

#include <fstream>
#include <iostream>
#include <vector>

struct NF4Data {
  int64_t rows;
  int64_t cols;
  int32_t blocksize;

  std::vector<uint8_t> packed;
};

NF4Data load_nf4(std::string filename) {
  std::ifstream f(filename, std::ios::binary);

  NF4Data data;

  f.read((char*)&data.rows, sizeof(int64_t));
  f.read((char*)&data.cols, sizeof(int64_t));
  f.read((char*)&data.blocksize, sizeof(int32_t));

  size_t packed_size = (data.rows * data.cols + 1) / 2;

  data.packed.resize(packed_size);

  f.read((char*)data.packed.data(), packed_size);

  return data;
}

int main() {
  NF4Data d = load_nf4("input.bin");

  std::cout << d.rows << std::endl;
  std::cout << d.cols << std::endl;
  std::cout << d.blocksize << std::endl;

  std::cout << d.packed[0] << std::endl;
}