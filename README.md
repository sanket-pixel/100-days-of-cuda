# 100 Days of CUDA Challenge

Welcome to my 100-day CUDA challenge — a daily push to sharpen my parallel programming skills, one kernel at a time.
From simple computations to advanced GPU algorithms, this is my hands-on journey to mastering CUDA and high-performance
computing.

### Day 1 : Squaring 100 numbers

```15.05.2025```
Kickstarting with the basics — squaring 100 numbers in parallel. Clean, efficient, and GPU-accelerated.

- Learned how to launch a CUDA kernel with proper grid and block dimensions.
- Understood thread indexing to map computations to data.
- Practiced basic parallel output using printf from device.

### Day 2 : Store squares on number in an array

```16.05.2025```
Built a simple CUDA kernel to store squares of numbers in an array
on the GPU and copy it from GPU to CPU.

- Allocated memory on the GPU using `cudaMalloc`.
- Transferred data from device to host using `cudaMemcpy`.
- Reinforced understanding of device-side execution and host-device interaction.

### Day 3 : Initialize 2D matrix with unique IDs

```17.05.2025```
Today’s CUDA kernel assigns unique thread IDs to a 2D matrix. Each element gets initialized to the flattened thread ID (
threadIdx.y * blockDim.x + threadIdx.x).
A simple but powerful way to build intuition around thread layout in 2D grids.

- Learned how to launch a 2D CUDA thread block using dim3.
- Practiced flattening 2D thread indices into a 1D memory layout.
- Understand use of `threadIdx.x`, `threadIdx.y`, `blockDim.x`, `blockDim.y`.

### Day 4 : Compute pairwise distance

```18.05.2025```
Today’s CUDA kernel computes the full pairwise distance matrix between N (x, y) pairs using Euclidean distance. Each
thread is responsible for calculating one matrix element by flattening the 2D row–column index from a 1D thread ID.

- Learned to launch 1D thread blocks with N×N threads to compute a 2D matrix.
- Practiced mapping a 1D thread index to (row, col) in a 2D grid using integer division and modulo.
- Used sqrtf() and powf() inside device code to compute Euclidean distances.

### Day 5: Square a Matrix with CUDA

```19.05.2025```
Today’s kernel squares a 3×3 matrix on the GPU by treating it as a flat array. Each thread computes a single element of
the output matrix using classic matrix multiplication logic.

- Learned how to flatten 2D matrix multiplication into a 1D thread space.
- Practiced accessing rows and columns from a flat index using % and /.
- Used shared indexing patterns to simulate 2D memory layout on a 1D thread grid.

### Day 6: Understanding Warps in CUDA

```20.05.2025```
Today’s CUDA kernel highlights the concept of warps—groups of 32 threads executed together by the GPU. Each thread
calculates and prints its warp ID using warp_id = threadIdx.x / 32.

- Learned that threads within a warp execute instructions in lockstep.
- Saw how to group threads by their warp ID.
- Reinforced why branching within a warp (warp divergence) is costly in performance-sensitive code.

### Day 7: Image RGB to Grayscale Conversion

```21.05.2025```
Today’s kernel converts a float32 RGB image to grayscale on the GPU by computing a weighted sum of color channels in
parallel.

- Learned to process interleaved RGB float data with one thread per pixel.
- Used the luminance formula: 0.299*R + 0.587*G + 0.144*B.
- Practiced indexing a flat array for 2D image data in CUDA.
- Handled OpenCV’s CV_32F to CV_8U conversion for image saving.

### Day 8: Demonstrating Thread Divergence in a Warp

```22.05.2025```
Today’s CUDA kernel showcases thread divergence within a warp by assigning different execution paths to threads based on
`tid % 4`.

- Used 8 threads in a single block to simulate divergence across 4 branches.
- Printed each thread’s ID and its warp to show which path it follows.
- Reinforced how branching inside a warp can cause serialization and reduce performance.
- Observed how threads in the same warp follow different code paths based on their tid.

### Day 9: Find maxiumum of a large array

```24.05.2025```
Today’s CUDA kernel performs a parallel reduction to find the maximum value from an array in chunks of size K.

- Divided input array into N/K buckets, each handled by one thread.
- Each thread computes the maximum of its K elements.
- Laid groundwork for full parallel reduction (multi-step) by producing intermediate results.
- Practiced device memory allocation, kernel launch config, and host-device transfers.

### Day 7: Image RGB to Grayscale Conversion with 2D grid and block

```21.05.2025```
Today’s kernel converts a float32 RGB image to grayscale on the GPU by computing a weighted sum of color channels in
parallel but using 2D grid and blocks.

- Learned to process interleaved RGB float data with one thread per pixel.
- Used the luminance formula: 0.299*R + 0.587*G + 0.144*B.
- Practiced indexing a 2D grid and block 2D image data in CUDA.