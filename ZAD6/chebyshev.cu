#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

const int MAX_ITERATIONS = 5000;
const double TOLERANCE = 1e-6;
const int THREADS_PER_BLOCK = 256;

// Makro sprawdzania błędów CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
             << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
}



__global__ void vectorNormKernelAtomic(const double* v, double* result, int n) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double val = (i < n) ? v[i] * v[i] : 0.0;
    sdata[tid] = val;
    __syncthreads();
    
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}



__global__ void reduceBlocksKernel(const double* v, double* block_results, int n) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? v[i] * v[i] : 0.0;
    __syncthreads();
    
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        block_results[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceFinalKernel(const double* block_results, double* result, int num_blocks) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? block_results[tid] : 0.0;
    __syncthreads();
    
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s && tid + s < num_blocks) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        *result = sdata[0];
    }
}



__global__ void vectorAddScaledKernel(double* x, const double* p, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        x[i] += alpha * p[i];
    }
}

__global__ void vectorSubtractKernel(double* r, const double* b, const double* Ax, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        r[i] = b[i] - Ax[i];
    }
}

__global__ void updateDirectionKernel(double* p, const double* r, double beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        p[i] = r[i] + beta * p[i];
    }
}

__global__ void matrixVectorMultiplyKernel(const double* A, const double* x, double* result, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < n) {
        double sum = 0.0;
        for(int col = 0; col < n; col++) {
            sum += A[row * n + col] * x[col];
        }
        result[row] = sum;
    }
}

// SOLVER CUDA


class ChebyshevSolverCUDA {
private:
    int n;
    double *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ax;
    double *d_norm_result, *d_block_results;
    
    cudaStream_t stream_compute;
    cudaStream_t stream_transfer;
    cudaStream_t stream_reduction;
    
    int num_blocks;
    int threads_per_block;

public:
    ChebyshevSolverCUDA(int size) : n(size) {
        threads_per_block = THREADS_PER_BLOCK;
        num_blocks = (n + threads_per_block - 1) / threads_per_block;
        
        if (num_blocks == 0) num_blocks = 1;
        
        size_t matrix_size = (size_t)n * n * sizeof(double);
        size_t vector_size = (size_t)n * sizeof(double);
        
        CHECK_CUDA(cudaMalloc(&d_A, matrix_size));
        CHECK_CUDA(cudaMalloc(&d_b, vector_size));
        CHECK_CUDA(cudaMalloc(&d_x, vector_size));
        CHECK_CUDA(cudaMalloc(&d_r, vector_size));
        CHECK_CUDA(cudaMalloc(&d_p, vector_size));
        CHECK_CUDA(cudaMalloc(&d_Ax, vector_size));
        CHECK_CUDA(cudaMalloc(&d_norm_result, sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_block_results, (size_t)num_blocks * sizeof(double)));
        
        CHECK_CUDA(cudaStreamCreate(&stream_compute));
        CHECK_CUDA(cudaStreamCreate(&stream_transfer));
        CHECK_CUDA(cudaStreamCreate(&stream_reduction));
    }
    
    ~ChebyshevSolverCUDA() {
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_r);
        cudaFree(d_p);
        cudaFree(d_Ax);
        cudaFree(d_norm_result);
        cudaFree(d_block_results);
        
        cudaStreamDestroy(stream_compute);
        cudaStreamDestroy(stream_transfer);
        cudaStreamDestroy(stream_reduction);
    }
    
    void setMatrix(const double* A) {
        CHECK_CUDA(cudaMemcpyAsync(d_A, A, (size_t)n * n * sizeof(double), 
                                   cudaMemcpyHostToDevice, stream_transfer));
    }
    
    void setRHS(const double* b) {
        CHECK_CUDA(cudaMemcpyAsync(d_b, b, n * sizeof(double), 
                                   cudaMemcpyHostToDevice, stream_transfer));
        CHECK_CUDA(cudaStreamSynchronize(stream_transfer));
    }
    
    double computeNormAtomic(const double* d_vec) {
        CHECK_CUDA(cudaMemsetAsync(d_norm_result, 0, sizeof(double), stream_reduction));
        
        int smem_size = threads_per_block * sizeof(double);
        vectorNormKernelAtomic<<<num_blocks, threads_per_block, smem_size, stream_reduction>>>
            (d_vec, d_norm_result, n);
        
        double h_result;
        CHECK_CUDA(cudaMemcpyAsync(&h_result, d_norm_result, sizeof(double), 
                                   cudaMemcpyDeviceToHost, stream_reduction));
        CHECK_CUDA(cudaStreamSynchronize(stream_reduction));
        
        return sqrt(h_result);
    }
    
    int solve(double l_min, double l_max, vector<double>& x_result) {
        double d = (l_max + l_min) * 0.5;
        double c = (l_max - l_min) * 0.5;
        
        CHECK_CUDA(cudaMemsetAsync(d_x, 0, n * sizeof(double), stream_compute));
        CHECK_CUDA(cudaMemcpyAsync(d_r, d_b, n * sizeof(double), 
                                   cudaMemcpyDeviceToDevice, stream_compute));
        CHECK_CUDA(cudaMemcpyAsync(d_p, d_b, n * sizeof(double), 
                                   cudaMemcpyDeviceToDevice, stream_compute));
        
        double alpha = 1.0 / d;
        double beta = 0.0;
        int iter;
        
        for(iter = 0; iter < MAX_ITERATIONS; iter++) {
            vectorAddScaledKernel<<<num_blocks, threads_per_block, 0, stream_compute>>>
                (d_x, d_p, alpha, n);
            
            matrixVectorMultiplyKernel<<<num_blocks, threads_per_block, 0, stream_compute>>>
                (d_A, d_x, d_Ax, n);
            
            vectorSubtractKernel<<<num_blocks, threads_per_block, 0, stream_compute>>>
                (d_r, d_b, d_Ax, n);
            
            CHECK_CUDA(cudaStreamSynchronize(stream_compute));
            
            double norm_r = computeNormAtomic(d_r);
            if(norm_r < TOLERANCE) break;
            
            double beta_new = pow(c / (2.0 * d), 2) * (1.0 - beta);
            double alpha_new = 1.0 / (d - beta_new * d);
            
            updateDirectionKernel<<<num_blocks, threads_per_block, 0, stream_compute>>>
                (d_p, d_r, beta_new, n);
            
            alpha = alpha_new;
            beta = beta_new;
        }
        
        x_result.resize(n);
        CHECK_CUDA(cudaMemcpyAsync(x_result.data(), d_x, n * sizeof(double), 
                                   cudaMemcpyDeviceToHost, stream_transfer));
        CHECK_CUDA(cudaStreamSynchronize(stream_transfer));
        
        return iter;
    }
};


// FUNKCJE POMOCNICZE

vector<double> matrixVectorMultiplySeq(const vector<double>& A, const vector<double>& x, int n) {
    vector<double> result(n, 0.0);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            result[i] += A[i * n + j] * x[j];
        }
    }
    return result;
}

// ============================================================================
// TESTY
// ============================================================================

void runAutotest() {
    cout << "=== AUTOTEST CUDA (2x2) - RTX 5050 Optimized ===\n";
    
    int n = 2;
    double A[4] = {4.0, 1.0, 1.0, 3.0};
    double b[2] = {6.0, 7.0};
    
    ChebyshevSolverCUDA solver(n);
    solver.setMatrix(A);
    solver.setRHS(b);
    
    vector<double> x;
    int iter = solver.solve(2.0, 5.0, x);
    
    cout << "Wynik: x1=" << x[0] << ", x2=" << x[1] << " (Oczekiwane: 1, 2)\n";
    cout << "Liczba iteracji: " << iter << "\n";
    cout << "============================\n\n";
}

void runBenchmark() {
    // STAŁY SEED dla powtarzalności
    srand(42);
    
    int sizes[] = {10, 100, 1000, 10000};
    int num_tests = sizeof(sizes) / sizeof(sizes[0]);
    
    cout << fixed << setprecision(2);
    cout << setw(6) << "N" << " | " 
         << setw(11) << "T_sekw [ms]" << " | "
         << setw(8) << "It_sekw" << " | "
         << setw(11) << "T_CUDA [ms]" << " | " 
         << setw(8) << "It_CUDA" << " | "
         << "Przysp." << endl;
    cout << string(78, '-') << endl;
    
    for(int t = 0; t < num_tests; t++) {
        int n = sizes[t];
        
        // Generuj macierz
        vector<double> A((size_t)n * n);
        vector<double> b(n);
        
        for(int i = 0; i < n; i++) {
            double row_sum = 0;
            for(int j = 0; j < n; j++) {
                A[i * n + j] = (double)(rand() % 10) / 100.0;
                row_sum += A[i * n + j];
            }
            A[i * n + i] = row_sum + 10.0;
            b[i] = (double)(rand() % 100);
        }
        
        double l_min = 5.0, l_max = n * 1.5;
        double d = (l_max + l_min) * 0.5;
        double c = (l_max - l_min) * 0.5;
        
        // ========================================================================
        // WERSJA SEKWENCYJNA
        // ========================================================================
        auto t_seq_start = chrono::high_resolution_clock::now();
        
        vector<double> x_seq(n, 0.0);
        vector<double> r_seq = b;
        vector<double> p_seq = r_seq;
        double alpha_seq = 1.0 / d;
        double beta_seq = 0.0;
        int iter_seq;
        
        for(iter_seq = 0; iter_seq < MAX_ITERATIONS; iter_seq++) {
            for(int i = 0; i < n; i++) {
                x_seq[i] += alpha_seq * p_seq[i];
            }
            
            vector<double> Ax_seq = matrixVectorMultiplySeq(A, x_seq, n);
            
            for(int i = 0; i < n; i++) {
                r_seq[i] = b[i] - Ax_seq[i];
            }
            
            double norm_r = 0.0;
            for(int i = 0; i < n; i++) {
                norm_r += r_seq[i] * r_seq[i];
            }
            norm_r = sqrt(norm_r);
            
            if(norm_r < TOLERANCE) break;
            
            double beta_new = pow(c / (2.0 * d), 2) * (1.0 - beta_seq);
            double alpha_new = 1.0 / (d - beta_new * d);
            
            for(int i = 0; i < n; i++) {
                p_seq[i] = r_seq[i] + beta_new * p_seq[i];
            }
            
            alpha_seq = alpha_new;
            beta_seq = beta_new;
        }
        
        auto t_seq_end = chrono::high_resolution_clock::now();
        float time_seq_ms = chrono::duration<float, milli>(t_seq_end - t_seq_start).count();
        
        // ========================================================================
        // WERSJA CUDA
        // ========================================================================
        ChebyshevSolverCUDA solver(n);
        solver.setMatrix(A.data());
        solver.setRHS(b.data());
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        vector<double> x_cuda;
        int iter_cuda = solver.solve(l_min, l_max, x_cuda);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float time_cuda_ms = 0;
        cudaEventElapsedTime(&time_cuda_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        // Wyniki
        double speedup = (time_cuda_ms > 0) ? (time_seq_ms / time_cuda_ms) : 0;
        
        cout << setw(6) << n << " | ";
        cout << setw(11) << time_seq_ms << " | ";
        cout << setw(8) << iter_seq << " | ";
        cout << setw(11) << time_cuda_ms << " | ";
        cout << setw(8) << iter_cuda << " | ";
        cout << speedup << "x" << endl;
    }
}

int main() {
    cout << "=== Chebyshev Solver CUDA - Optimized for RTX 5050 ===\n";
    cout << "Using NATIVE atomicAdd for double precision\n\n";
    
    runAutotest();
    runBenchmark();
    return 0;
}