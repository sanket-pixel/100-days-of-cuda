# 100 Days of CUDA Challenge
Welcome to my 100-day CUDA challenge — a daily push to sharpen my parallel programming skills, one kernel at a time. 
From simple computations to advanced GPU algorithms, this is my hands-on journey to mastering CUDA and high-performance computing.

### Day 1
Kickstarting with the basics — squaring 100 numbers in parallel. Clean, efficient, and GPU-accelerated. 
- Learned how to launch a CUDA kernel with proper grid and block dimensions.
- Understood thread indexing to map computations to data.
- Practiced basic parallel output using printf from device.

### Day 2
Diving deeper — explored GPU memory allocation and host-device data transfer. Built a simple CUDA kernel to store squares of numbers in an array
on the GPU and copy it from GPU to CPU.
- Allocated memory on the GPU using `cudaMalloc`.
- Transferred data from device to host using `cudaMemcpy`.
- Reinforced understanding of device-side execution and host-device interaction.
