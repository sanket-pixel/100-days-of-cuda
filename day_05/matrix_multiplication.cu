#include <cuda.h>
#include <iostream>
#include <random>

#define BLOCKSIZE 1024
#define N 3

__global__ void matrix_square_gpu(int *da, int *squared_da, int matrix_size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N * N) {
    int col_id = id % matrix_size;
    int row_id = id / matrix_size;
    int sum = 0;
    for (int i = 0; i < matrix_size; i++) {
      sum += da[row_id * matrix_size + i] * da[i * matrix_size + col_id];
    }
    squared_da[row_id * matrix_size + col_id] = sum;
  }
}

int main() {
  int *ha, *da;
  int *squared_ha, *squared_da;
  ha = static_cast<int *>(malloc(sizeof(int) * N * N));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      ha[i * N + j] = i * N + j;
    }
  }
  cudaMalloc(&da, sizeof(int) * N * N);
  cudaMemcpy(da, ha, sizeof(int) * N * N, cudaMemcpyHostToDevice);
  cudaMalloc(&squared_da, sizeof(int) * N * N);
  int number_of_thread_blocks = (N * N) / BLOCKSIZE + 1;
  matrix_square_gpu<<<number_of_thread_blocks, BLOCKSIZE>>>(da, squared_da, N);
  squared_ha = static_cast<int *>(malloc(sizeof(int) * N * N));
  cudaMemcpy(squared_ha, squared_da, sizeof(int) * N * N,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << squared_ha[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}