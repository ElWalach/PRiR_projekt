#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define N 10000000   
#define THREADS_PER_BLOCK 256

// Kernel CUDA
__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_out = new float[N];

    
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    
    float *d_a, *d_b, *d_out;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    
    int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::cout << "Blocks per grid: " << blocks_per_grid << std::endl;
    std::cout << "Threads per block: " << THREADS_PER_BLOCK << std::endl;

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    
    vector_add<<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_a, d_b, d_out, N);
    
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "Czas obliczeń na GPU: " << gpu_time / 1000.0f << " s" << std::endl;

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (abs(h_out[i] - (h_a[i] + h_b[i])) > 1e-5) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Wynik poprawny " << std::endl;
    } else {
        std::cout << "BŁĄD " << std::endl;
    }


    delete[] h_a;
    delete[] h_b;
    delete[] h_out;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}