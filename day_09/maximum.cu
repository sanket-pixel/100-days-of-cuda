#include <cuda.h>
#include <iostream>
#include <vector>
using namespace std;
#define K 3
#define BLOCKSIZE 1024

__global__ void max(int *dnums, int k, int N, int *dmax_nums) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int start = id * k;
  if (id < (N / k + 1)) {
    int start = id * k;
    int max = dnums[start];
    for (int i = 0; i < k; i++) {
      if (dnums[start + i] > max) {
        max = dnums[start + i];
      }
    }
    dmax_nums[id] = max;
  }
}
int main() {
  vector<int> hnums{1, 2, 4, 10, 3, 8, 9, 1, 16, 12, 3, 5};
  int N = hnums.size();
  int data_size_in_bytes = N * sizeof(int);
  int *dnums;
  cudaMalloc(&dnums, data_size_in_bytes);
  cudaMemcpy(dnums, hnums.data(), data_size_in_bytes, cudaMemcpyHostToDevice);
  int number_of_threads = ceil(N / K);
  int *dmax_nums;
  cudaMalloc(&dmax_nums, number_of_threads * sizeof(int));
  int number_of_thread_blocks = number_of_threads / BLOCKSIZE + 1;
  max<<<number_of_thread_blocks, BLOCKSIZE>>>(dnums, K, N, dmax_nums);
  vector<int> hmax_nums(number_of_threads, 0);
  cudaMemcpy(hmax_nums.data(), dmax_nums, number_of_threads * sizeof(int),
             cudaMemcpyDeviceToHost);
  for (int num : hmax_nums) {
    cout << num << " ";
  }
}