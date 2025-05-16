#include <stdio.h>
#include <iostream>
#include <cuda.h>
#define N 100
__global__ void dkernel(int *da) {
  da[threadIdx.x] = threadIdx.x * threadIdx.x;
}

int main() {
  int a[N], *da;
  cudaMalloc(&da, sizeof(int)*N);
  dkernel<<<1,N>>>(da);
  cudaMemcpy(a,da,sizeof(int)*N,cudaMemcpyDeviceToHost);
  for (int i = 0; i< N; i++) {
    std::cout << a[i] << std::endl;
  }
  return 0;
}

