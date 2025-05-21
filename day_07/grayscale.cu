#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <vector>

#define BLOCKSIZE 1024

__global__ void convert_to_grayscale(float *dimage_color,
                                     float *dimage_grayscale, int height,
                                     int width) {
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < height * width) {
    dimage_grayscale[id] = 0.144 * dimage_color[id * 3] +
                           0.587 * dimage_color[id * 3 + 1] +
                           0.299 * dimage_color[id * 3 + 2];
  }
}
int main() {
  cv::Mat himage_color = cv::imread("../sample.png");
  auto width = himage_color.cols;
  auto height = himage_color.rows;
  auto channels = himage_color.channels();
  himage_color.convertTo(himage_color, CV_32F, 1.0 / 255.0);
  auto color_image_size_in_bytes = height * width * channels * sizeof(float);
  float *dimage_color;
  cudaMalloc(&dimage_color, color_image_size_in_bytes);
  cudaMemcpy(dimage_color, himage_color.data, color_image_size_in_bytes,
             cudaMemcpyHostToDevice);

  auto grayscale_image_size_in_bytes = height * width * sizeof(float);
  float *d_image_grayscale;
  cudaMalloc(&d_image_grayscale, grayscale_image_size_in_bytes);
  int number_of_blocks = height * width / BLOCKSIZE + 1;
  convert_to_grayscale<<<number_of_blocks, BLOCKSIZE>>>(
      dimage_color, d_image_grayscale, height, width);
  cv::Mat himage_grayscale(height, width, CV_32FC1);
  float *himage_grayscale_data =
      reinterpret_cast<float *>(himage_grayscale.data);
  cudaMemcpy(himage_grayscale_data, d_image_grayscale,
             grayscale_image_size_in_bytes, cudaMemcpyDeviceToHost);
  himage_grayscale.convertTo(himage_grayscale, CV_8U, 255.0);
  cv::imwrite("../grayscale_sample.png", himage_grayscale);
  return 0;
}