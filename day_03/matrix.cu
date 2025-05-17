#include <cuda.h>
#include <filesystem>
#include <iostream>
# define N 5
# define M 6

__global__ void matrix_initialize(int* dmatrix) {
    unsigned int id = blockDim.x * threadIdx.y + threadIdx.x;
    dmatrix[id] = id;
}
int main() {
  int *hmatrix, *dmatrix;
  dim3 blocks(N,M,1);
  // allocate memory for matrix on host
  hmatrix = static_cast<int *>(malloc(N * M * sizeof(int)));
  // assign memory for matrix on device
  cudaMalloc(&dmatrix, N*M*sizeof(int));
  // call the kernel to initialize unique IDs
  matrix_initialize<<<1,blocks>>>(dmatrix);
  // copy matrix from device to host
  cudaMemcpy(hmatrix, dmatrix, N*M*sizeof(int), cudaMemcpyDeviceToHost);
  for (int r = 0; r < M; r++) {
    for ( int c = 0; c < N; c++) {
      std::cout << hmatrix[r*N + c] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}