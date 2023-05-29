#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

void load_image(unsigned char** image, const char* filename, int* w, int* h, int* channels);
void save_image(unsigned char* image, const char* filename, int w, int h);
void allocate_memory(unsigned char** d_input, unsigned char** d_output, unsigned char** h_output, unsigned char** h_output_cpu, int w, int h, int new_w, int new_h);
void copy_host_to_device(unsigned char* d_input, unsigned char* h_input, int w, int h);
void copy_device_to_host(unsigned char* d_output, unsigned char* h_output, int new_w, int new_h);
void deallocate_memory(unsigned char* d_input, unsigned char* d_output, unsigned char* h_input, unsigned char* h_output, unsigned char* h_output_cpu);
void resize_h(unsigned char* input, unsigned char* output, int w, int h, int new_w, int new_h);
void resize_parallel(unsigned char* input, unsigned char* output, int w, int h, int new_w, int new_h);

__global__ void resize(unsigned char* input, unsigned char* output, int w, int h, int new_w, int new_h);

int main(int argc, char** argv) {
    printf("Begin\n");

    // Load image using STB
    int w, h, channels;
    unsigned char* h_input;
    load_image(&h_input, argv[1], &w, &h, &channels);

    // New dimensions
    int new_w = w / 2;  // example for downsizing
    int new_h = h / 2;

    // Allocate memory
    unsigned char* h_output, *h_output_cpu, *d_input, *d_output;
    allocate_memory(&d_input, &d_output, &h_output, &h_output_cpu, w, h, new_w, new_h);

    // Copy data from host to device
    copy_host_to_device(d_input, h_input, w, h);

    // Measure sequential CPU execution time
    double start_seq = omp_get_wtime();
    resize_h(h_input, h_output_cpu, w, h, new_w, new_h);
    double end_seq = omp_get_wtime();
    double time_seq = end_seq - start_seq;

    // Measure 16-thread parallel CPU execution time
    double start_parallel = omp_get_wtime();
    resize_parallel(h_input, h_output, w, h, new_w, new_h);
    double end_parallel = omp_get_wtime();
    double time_parallel = end_parallel - start_parallel;

    // Calculate grid and block dimensions
    int block_size = 16;
    int block_no_x = (new_w + block_size - 1) / block_size;
    int block_no_y = (new_h + block_size - 1) / block_size;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(block_no_x, block_no_y);

    // Measure GPU execution time
    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu, 0);
    resize<<<dimGrid, dimBlock>>>(d_input, d_output, w, h, new_w, new_h);
    cudaEventRecord(end_gpu, 0);
    cudaEventSynchronize(end_gpu);
    float time_gpu;
    cudaEventElapsedTime(&time_gpu, start_gpu, end_gpu);

    // Copy data from device to host
    copy_device_to_host(d_output, h_output, new_w, new_h);

    // Save the result to an image file using STB
    save_image(h_output, argv[2], new_w, new_h);

    // Free the memory
    deallocate_memory(d_input, d_output, h_input, h_output, h_output_cpu);

    // Calculate speed-up
    double speedup_seq = time_seq / (time_gpu / 1000.0);  // Convert GPU time to seconds
    double speedup_parallel = time_parallel / (time_gpu / 1000.0);

    // Print execution times and speed-up
    printf("Sequential CPU time: %f seconds\n", time_seq);
    printf("16-thread Parallel CPU time: %f seconds\n", time_parallel);
    printf("GPU time: %f seconds\n", time_gpu / 1000.0);
    printf("Speed-up (Sequential): %.2f\n", speedup_seq);
    printf("Speed-up (16-thread Parallel): %.2f\n", speedup_parallel);

    return 0;
}

void load_image(unsigned char** image, const char* filename, int* w, int* h, int* channels) {
    *image = stbi_load(filename, w, h, channels, 1);
    if (!*image) {
        printf("Error in loading image\n");
        exit(-1);
    }
}

void save_image(unsigned char* image, const char* filename, int w, int h) {
    if (!stbi_write_jpg(filename, w, h, 1, image, 100)) {
        printf("Error in saving image\n");
        exit(-1);
    }
}

void allocate_memory(unsigned char** d_input, unsigned char** d_output, unsigned char** h_output, unsigned char** h_output_cpu, int w, int h, int new_w, int new_h) {
    *h_output = (unsigned char*)malloc(sizeof(unsigned char) * new_w * new_h);
    *h_output_cpu = (unsigned char*)malloc(sizeof(unsigned char) * new_w * new_h);
    cudaMalloc((void**)d_input, sizeof(unsigned char) * w * h);
    cudaMalloc((void**)d_output, sizeof(unsigned char) * new_w * new_h);
}

void copy_host_to_device(unsigned char* d_input, unsigned char* h_input, int w, int h) {
    cudaMemcpy(d_input, h_input, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
}

void copy_device_to_host(unsigned char* d_output, unsigned char* h_output, int new_w, int new_h) {
    cudaMemcpy(h_output, d_output, sizeof(unsigned char) * new_w * new_h, cudaMemcpyDeviceToHost);
}

void deallocate_memory(unsigned char* d_input, unsigned char* d_output, unsigned char* h_input, unsigned char* h_output, unsigned char* h_output_cpu) {
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(h_output_cpu);
}

__global__ void resize(unsigned char* input, unsigned char* output, int w, int h, int new_w, int new_h) {
    // Resize computation using bilinear interpolation
    // ...
}

void resize_h(unsigned char* input, unsigned char* output, int w, int h, int new_w, int new_h) {
    float x_ratio = float(w - 1) / new_w;
    float y_ratio = float(h - 1) / new_h;
    for (int i = 0; i < new_h; i++) {
        for (int j = 0; j < new_w; j++) {
            int x = int(x_ratio * j);
            int y = int(y_ratio * i);
            float x_diff = (x_ratio * j) - x;
            float y_diff = (y_ratio * i) - y;
            int index = y * w + x;

            // calculate interpolated value
            output[i * new_w + j] =
                input[index] * (1 - x_diff) * (1 - y_diff) +
                input[index + 1] * x_diff * (1 - y_diff) +
                input[index + w] * y_diff * (1 - x_diff) +
                input[index + w + 1] * x_diff * y_diff;
        }
    }
}

void resize_parallel(unsigned char* input, unsigned char* output, int w, int h, int new_w, int new_h) {
    float x_ratio = float(w - 1) / new_w;
    float y_ratio = float(h - 1) / new_h;
#pragma omp parallel for num_threads(16)
    for (int i = 0; i < new_h; i++) {
        for (int j = 0; j < new_w; j++) {
            int x = int(x_ratio * j);
            int y = int(y_ratio * i);
            float x_diff = (x_ratio * j) - x;
            float y_diff = (y_ratio * i) - y;
            int index = y * w + x;

            // calculate interpolated value
            output[i * new_w + j] =
                input[index] * (1 - x_diff) * (1 - y_diff) +
                input[index + 1] * x_diff * (1 - y_diff) +
                input[index + w] * y_diff * (1 - x_diff) +
                input[index + w + 1] * x_diff * y_diff;
        }
    }
}

