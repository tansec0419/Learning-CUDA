#include <cstdint>
#include <iostream>
#include <vector>

float nf4_table[16] = {-1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848,
                       -0.0911, 0.0000,  0.0796,  0.1609,  0.2461,  0.3379,
                       0.4407,  0.5626,  0.7230,  1.0000};

int main() {
  std::vector<uint8_t> packed = {0x12, 0x34, 0xAB};

  for (int i = 0; i < packed.size(); i++) {
    uint8_t byte = packed[i];

    uint8_t idx0 = byte & 0xF;
    uint8_t idx1 = byte >> 4;

    float w0 = nf4_table[idx0];
    float w1 = nf4_table[idx1];

    std::cout << w0 << " " << w1 << std::endl;
  }
}