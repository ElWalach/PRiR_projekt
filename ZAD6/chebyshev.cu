#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
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

// =============================================================================
// PROBLEM 1: OPERACJE ATOMOWE - Redukcja z wykorzystaniem atomicAdd
// =============================================================================

// Implementacja atomicAdd dla double (dla starszych GPU)
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

// Kernel redukcji z użyciem shared memory i operacji atomowych
__global__ void vectorNormKernel(const double* v, double* partial_sums, int n) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Każdy wątek oblicza swój kwadrat
    double val = (i < n) ? v[i] * v[i] : 0.0;
    sdata[tid] = val;
    __syncthreads();
    
    // Redukcja w shared memory (logarytmiczna)
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // OPERACJA ATOMOWA: pierwszy wątek w bloku dodaje wynik do globalnej pamięci
    if(tid == 0) {
        atomicAddDouble(partial_sums, sdata[0]);
    }
}

// =============================================================================
// PROBLEM 2: OPERACJE REDUKCJI - Dwuetapowa redukcja dla dużych danych
// =============================================================================

// Etap 1: Redukcja w blokach
__global__ void reduceBlocksKernel(const double* v, double* block_results, int n) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? v[i] * v[i] : 0.0;
    __syncthreads();
    
    // Redukcja w bloku
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Zapisz wynik bloku (bez atomicAdd - każdy blok ma swoje miejsce)
    if(tid == 0) {
        block_results[blockIdx.x] = sdata[0];
    }
}

// Etap 2: Finalna redukcja (dla małej liczby bloków)
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

// =============================================================================
// Kernele operacji wektorowych
// =============================================================================

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

// =============================================================================
// PROBLEM 3: STRUMIENIE - Asynchroniczne wykonywanie operacji
// =============================================================================

class ChebyshevSolverCUDA {
private:
    int n;
    double *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ax;
    double *d_norm_result, *d_block_results;
    
    // Strumienie dla różnych operacji
    cudaStream_t stream_compute;
    cudaStream_t stream_transfer;
    cudaStream_t stream_reduction;
    
    int num_blocks;
    int threads_per_block;

public:
    ChebyshevSolverCUDA(int size) : n(size) {
        threads_per_block = THREADS_PER_BLOCK;
        num_blocks = (n + threads_per_block - 1) / threads_per_block;
        
        // Alokacja pamięci na GPU
        CHECK_CUDA(cudaMalloc(&d_A, (size_t)n * n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_Ax, n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_norm_result, sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_block_results, num_blocks * sizeof(double)));
        
        // STRUMIENIE: Tworzenie trzech niezależnych strumieni
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
        // Transfer asynchroniczny w osobnym strumieniu
        CHECK_CUDA(cudaMemcpyAsync(d_A, A, (size_t)n * n * sizeof(double), 
                                   cudaMemcpyHostToDevice, stream_transfer));
    }
    
    void setRHS(const double* b) {
        CHECK_CUDA(cudaMemcpyAsync(d_b, b, n * sizeof(double), 
                                   cudaMemcpyHostToDevice, stream_transfer));
        CHECK_CUDA(cudaStreamSynchronize(stream_transfer));
    }
    
    // Obliczanie normy z użyciem redukcji dwuetapowej
    double computeNorm(const double* d_vec) {
        // Resetuj wynik
        CHECK_CUDA(cudaMemsetAsync(d_norm_result, 0, sizeof(double), stream_reduction));
        
        // Etap 1: Redukcja w blokach (w strumieniu redukcji)
        int smem_size = threads_per_block * sizeof(double);
        reduceBlocksKernel<<<num_blocks, threads_per_block, smem_size, stream_reduction>>>
            (d_vec, d_block_results, n);
        
        // Etap 2: Finalna redukcja
        reduceFinalKernel<<<1, threads_per_block, smem_size, stream_reduction>>>
            (d_block_results, d_norm_result, num_blocks);
        
        // Kopiuj wynik
        double h_result;
        CHECK_CUDA(cudaMemcpyAsync(&h_result, d_norm_result, sizeof(double), 
                                   cudaMemcpyDeviceToHost, stream_reduction));
        CHECK_CUDA(cudaStreamSynchronize(stream_reduction));
        
        return sqrt(h_result);
    }
    
    int solve(double l_min, double l_max, vector<double>& x_result) {
        double d = (l_max + l_min) * 0.5;
        double c = (l_max - l_min) * 0.5;
        
        // Inicjalizacja x = 0, r = b, p = r
        CHECK_CUDA(cudaMemsetAsync(d_x, 0, n * sizeof(double), stream_compute));
        CHECK_CUDA(cudaMemcpyAsync(d_r, d_b, n * sizeof(double), 
                                   cudaMemcpyDeviceToDevice, stream_compute));
        CHECK_CUDA(cudaMemcpyAsync(d_p, d_b, n * sizeof(double), 
                                   cudaMemcpyDeviceToDevice, stream_compute));
        
        double alpha = 1.0 / d;
        double beta = 0.0;
        int iter;
        
        for(iter = 0; iter < MAX_ITERATIONS; iter++) {
            // STRUMIENIE: Operacje wykonywane w strumieniu obliczeniowym
            
            // x = x + alpha * p
            vectorAddScaledKernel<<<num_blocks, threads_per_block, 0, stream_compute>>>
                (d_x, d_p, alpha, n);
            
            // Ax = A * x
            matrixVectorMultiplyKernel<<<num_blocks, threads_per_block, 0, stream_compute>>>
                (d_A, d_x, d_Ax, n);
            
            // r = b - Ax
            vectorSubtractKernel<<<num_blocks, threads_per_block, 0, stream_compute>>>
                (d_r, d_b, d_Ax, n);
            
            // Synchronizuj przed obliczaniem normy
            CHECK_CUDA(cudaStreamSynchronize(stream_compute));
            
            // Sprawdź zbieżność (używa stream_reduction)
            double norm_r = computeNorm(d_r);
            if(norm_r < TOLERANCE) break;
            
            // Aktualizacja parametrów
            double beta_new = pow(c / (2.0 * d), 2) * (1.0 - beta);
            double alpha_new = 1.0 / (d - beta_new * d);
            
            // p = r + beta_new * p
            updateDirectionKernel<<<num_blocks, threads_per_block, 0, stream_compute>>>
                (d_p, d_r, beta_new, n);
            
            alpha = alpha_new;
            beta = beta_new;
        }
        
        // Kopiuj wynik z powrotem
        x_result.resize(n);
        CHECK_CUDA(cudaMemcpy(x_result.data(), d_x, n * sizeof(double), 
                              cudaMemcpyDeviceToHost));
        
        return iter;
    }
};

// =============================================================================
// Funkcje testowe
// =============================================================================

void runAutotest() {
    cout << "=== AUTOTEST CUDA (2x2) ===\n";
    
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
    int sizes[] = {100, 500, 1000, 10000};
    int num_tests = sizeof(sizes) / sizeof(sizes[0]);
    
    cout << fixed << setprecision(2);
    cout << setw(6) << "N" << " | " 
         << setw(11) << "T_CUDA [ms]" << " | " 
         << setw(8) << "Iteracje" << endl;
    cout << string(40, '-') << endl;
    
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
        
        ChebyshevSolverCUDA solver(n);
        solver.setMatrix(A.data());
        solver.setRHS(b.data());
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        vector<double> x;
        int iter = solver.solve(5.0, n * 1.5, x);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cout << setw(6) << n << " | " 
             << setw(11) << milliseconds << " | " 
             << setw(8) << iter << endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main() {
    runAutotest();
    runBenchmark();
    return 0;
}