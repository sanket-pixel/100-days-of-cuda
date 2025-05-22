#include <cuda.h>
#include <stdio.h>
#include <vector>
#define BLOCKSIZE 1024

__global__ void thread_divergence() {
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  if (tid % 4 == 0) {
    printf("Thread %d in warp %d: Path 0\n", tid, warp_id);
  } else if (tid % 4 == 1) {
    printf("Thread %d in warp %d: Path 1\n", tid, warp_id);
  } else if (tid % 4 == 2) {
    printf("Thread %d in warp %d: Path 2\n", tid, warp_id);
  } else {
    printf("Thread %d in warp %d: Path 3\n", tid, warp_id);
  }
}

int main() {
  thread_divergence<<<1, 8>>>();
  cudaDeviceSynchronize();
}