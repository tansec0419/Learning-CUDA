#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__constant__ float nf4_table[16] = {
    -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0000,
    0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000};

__global__ void nf4_kernel(const uint8_t* packed, float* output, int n_bytes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= n_bytes) return;

  uint8_t byte = packed[idx];

  uint8_t i0 = byte & 0xF;
  uint8_t i1 = byte >> 4;

  float nf0 = nf4_table[i0];
  float nf1 = nf4_table[i1];

  output[idx * 2] = nf0;
  output[idx * 2 + 1] = nf1;
}

int main() {
  uint8_t h_packed[3] = {0x12, 0x34, 0xAB};

  float h_output[6];

  uint8_t* d_packed;
  float* d_output;

  cudaMalloc(&d_packed, 3);
  cudaMalloc(&d_output, 6 * sizeof(float));

  cudaMemcpy(d_packed, h_packed, 3, cudaMemcpyHostToDevice);

  nf4_kernel<<<1, 32>>>(d_packed, d_output, 3);

  cudaMemcpy(h_output, d_output, 6 * sizeof(float), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for (int i = 0; i < 6; i++) printf("%f\n", h_output[i]);

  cudaFree(d_packed);
  cudaFree(d_output);
}