#include <stdint.h>

#include <fstream>
#include <iostream>
#include <vector>

struct NF4Data {
  int64_t rows;
  int64_t cols;
  int32_t blocksize;

  std::vector<uint8_t> packed;
  std::vector<uint8_t> absmax_q;
  std::vector<uint16_t> absmax2;
  std::vector<uint16_t> code2;

  float offset;
};

NF4Data load_nf4(std::string filename) {
  std::ifstream f(filename, std::ios::binary);

  NF4Data d;

  f.read((char*)&d.rows, sizeof(int64_t));
  f.read((char*)&d.cols, sizeof(int64_t));
  f.read((char*)&d.blocksize, sizeof(int32_t));

  int num_weights = d.rows * d.cols;

  int packed_size = num_weights / 2;

  int num_blocks = num_weights / d.blocksize;

  int num_groups = num_blocks / 256 + 1;

  d.packed.resize(packed_size);
  d.absmax_q.resize(num_blocks);
  d.absmax2.resize(num_groups);
  d.code2.resize(256);

  f.read((char*)d.packed.data(), packed_size);
  f.read((char*)d.absmax_q.data(), num_blocks);
  f.read((char*)d.absmax2.data(), num_groups * 2);
  f.read((char*)d.code2.data(), 256 * 2);

  f.read((char*)&d.offset, sizeof(float));

  return d;
}

int main() {
  NF4Data d = load_nf4("input.bin");

  std::cout << "rows: " << d.rows << std::endl;
  std::cout << "cols: " << d.cols << std::endl;
  std::cout << "blocksize: " << d.blocksize << std::endl;

  std::cout << "packed size: " << d.packed.size() << std::endl;
}