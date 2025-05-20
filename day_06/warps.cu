#include <cuda.h>
#include <iostream>

#define N 64          // Total number of threads
#define BLOCK_SIZE 64 // One block for simplicity
#define WARP_SIZE 32

__global__ void log_warp_id() {
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  printf("Thread %d is in warp %d\n", tid, warp_id);
}

int main() {
  log_warp_id<<<1, BLOCK_SIZE>>>();
  cudaDeviceSynchronize();
  return 0;
}