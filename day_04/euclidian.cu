#include <cuda.h>
#include <iostream>
#include <random>

#define BLOCKSIZE 1024
#define N 40

struct Pair {
  int x{0};
  int y{0};
};

__global__ void pairwise_distance(Pair *dpairs,
                                  float *pairwise_distance_matrix) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N * N) {
    int row = id / N;
    int col = id % N;
    Pair row_element = dpairs[row];
    Pair col_element = dpairs[col];
    float distance = sqrtf(powf(row_element.x - col_element.x, 2.0f) +
                           powf(row_element.y - col_element.y, 2.0f));
    pairwise_distance_matrix[row * N + col] = distance;
    printf("%d : { %d,%d} -> {(%d,%d),(%d,%d)} : {%f}\n", id, row, col,
           row_element.x, row_element.y, col_element.x, col_element.y,
           distance);
  }
}

std::vector<Pair> generate_deterministic_pairs(const int pair_count) {
  std::vector<Pair> pairs{};
  for (int i = 0; i < pair_count; i++) {
    pairs.emplace_back(Pair{i, i});
  }
  return pairs;
}

int main() {
  // generate pairs on host
  auto hpairs = generate_deterministic_pairs(N);
  // copy pairs to device
  Pair *dpairs;
  cudaMalloc(&dpairs, N * sizeof(Pair));
  cudaMemcpy(dpairs, hpairs.data(), N * sizeof(Pair), cudaMemcpyHostToDevice);
  // allocate matrix for storing pairswise distance on host and device
  float *hdistance, *ddistance;
  hdistance = static_cast<float *>(malloc(N * N * sizeof(float)));
  cudaMalloc(&ddistance, N * N * sizeof(float));
  // compute pairwise distance
  // The output matrix will have N*N elements and each thread will compute one
  // element of this matrix So the kernel should be launched with N*N threads
  int number_of_thread_blocks = (N * N) / BLOCKSIZE + 1;
  pairwise_distance<<<number_of_thread_blocks, BLOCKSIZE>>>(dpairs, ddistance);
  // copy distance from device to host
  cudaMemcpy(hdistance, ddistance, N * N * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << hdistance[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}