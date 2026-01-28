

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>

using namespace std;

const int MAX_ITERATIONS = 5000;
const double TOLERANCE = 1e-6;
const int WORK_GROUP_SIZE = 256;

#define CHECK_CL(err, msg) { \
    if(err != CL_SUCCESS) { \
        cerr << "OpenCL error (" << err << ") at " << msg << endl; \
        exit(EXIT_FAILURE); \
    } \
}

// =============================================================================
// KERNELE OpenCL
// =============================================================================

const char* kernel_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// =============================================================================
// PROBLEM 1: OPERACJE ATOMOWE - atomicAdd dla double
// =============================================================================
inline void atomicAdd_double(__global double *addr, double val) {
    union {
        unsigned long u64;
        double f64;
    } next, expected, current;
    current.f64 = *addr;
    do {
        expected.f64 = current.f64;
        next.f64 = expected.f64 + val;
        current.u64 = atom_cmpxchg((__global unsigned long *)addr, 
                                    expected.u64, next.u64);
    } while(current.u64 != expected.u64);
}

// Kernel używający operacji atomowych - alternatywna redukcja normy
__kernel void reduce_atomic(__global const double* input,
                           __global double* result,
                           __local double* scratch,
                           const int n) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int local_size = get_local_size(0);
    
    // Każdy work-item oblicza swój kwadrat
    scratch[lid] = (gid < n) ? input[gid] * input[gid] : 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Redukcja w work-group
    for(int offset = local_size / 2; offset > 0; offset >>= 1) {
        if(lid < offset) {
            scratch[lid] += scratch[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // OPERACJA ATOMOWA: dodaj wynik work-group do globalnego wyniku
    if(lid == 0) {
        atomicAdd_double(result, scratch[0]);
    }
}

// PROBLEM 2: REDUKCJA - Etap 1
__kernel void reduce_blocks(__global const double* input,
                           __global double* block_results,
                           __local double* scratch,
                           const int n) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);
    
    scratch[lid] = (gid < n) ? input[gid] * input[gid] : 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int offset = local_size / 2; offset > 0; offset >>= 1) {
        if(lid < offset) {
            scratch[lid] += scratch[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(lid == 0) {
        block_results[group_id] = scratch[0];
    }
}

// PROBLEM 2: REDUKCJA - Etap 2 (finalna)
__kernel void reduce_final(__global const double* block_results,
                          __global double* result,
                          __local double* scratch,
                          const int num_blocks) {
    int lid = get_local_id(0);
    scratch[lid] = (lid < num_blocks) ? block_results[lid] : 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int offset = get_local_size(0) / 2; offset > 0; offset >>= 1) {
        if(lid < offset && lid + offset < num_blocks) {
            scratch[lid] += scratch[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(lid == 0) {
        *result = scratch[0];
    }
}

// Mnożenie macierzy przez wektor
__kernel void matrix_vector_multiply(__global const double* A,
                                     __global const double* x,
                                     __global double* result,
                                     const int n) {
    int row = get_global_id(0);
    if(row < n) {
        double sum = 0.0;
        for(int col = 0; col < n; col++) {
            sum += A[row * n + col] * x[col];
        }
        result[row] = sum;
    }
}

// x = x + alpha * p
__kernel void vector_add_scaled(__global double* x,
                               __global const double* p,
                               const double alpha,
                               const int n) {
    int i = get_global_id(0);
    if(i < n) {
        x[i] += alpha * p[i];
    }
}

// r = b - Ax
__kernel void vector_subtract(__global double* r,
                             __global const double* b,
                             __global const double* Ax,
                             const int n) {
    int i = get_global_id(0);
    if(i < n) {
        r[i] = b[i] - Ax[i];
    }
}

// p = r + beta * p
__kernel void update_direction(__global double* p,
                              __global const double* r,
                              const double beta,
                              const int n) {
    int i = get_global_id(0);
    if(i < n) {
        p[i] = r[i] + beta * p[i];
    }
}
)CLC";

// =============================================================================
// Klasa solvera OpenCL
// =============================================================================

class ChebyshevSolverOpenCL {
private:
    int n;
    cl_context context;
    
    // PROBLEM 3: STRUMIENIE - Wiele command queues (strumieni)
    cl_command_queue queue_compute;   // Dla obliczeń
    cl_command_queue queue_transfer;  // Dla transferów pamięci
    cl_command_queue queue_reduction; // Dla operacji redukcji
    
    cl_program program;
    cl_kernel k_matvec, k_add_scaled, k_subtract, k_update_dir;
    cl_kernel k_reduce_blocks, k_reduce_final, k_reduce_atomic;
    
    cl_mem d_A, d_b, d_x, d_r, d_p, d_Ax;
    cl_mem d_norm_result, d_block_results;
    
    int num_work_groups;

public:
    ChebyshevSolverOpenCL(int size) : n(size) {
        cl_int err;
        
        // 1. Pobierz platformę
        cl_uint num_platforms;
        CHECK_CL(clGetPlatformIDs(0, NULL, &num_platforms), "Get num platforms");
        
        if(num_platforms == 0) {
            cerr << "Błąd: Brak platform OpenCL!\n";
            cerr << "Zainstaluj: sudo apt install pocl-opencl-icd -y\n";
            exit(EXIT_FAILURE);
        }
        
        cl_platform_id platform;
        CHECK_CL(clGetPlatformIDs(1, &platform, NULL), "Get platform");
        
        // Wyświetl info o platformie
        char platform_name[128];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, NULL);
        cout << "Platforma OpenCL: " << platform_name << "\n";
        
        // 2. Pobierz urządzenie (GPU > CPU > ANY)
        cl_device_id device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if(err != CL_SUCCESS) {
            cout << "GPU nie znalezione, próbuję CPU...\n";
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
            if(err != CL_SUCCESS) {
                cout << "CPU nie znalezione, próbuję ANY...\n";
                err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
                if(err != CL_SUCCESS) {
                    cerr << "Błąd: Brak dostępnych urządzeń OpenCL!\n";
                    cerr << "Zainstaluj runtime: sudo apt install pocl-opencl-icd -y\n";
                    exit(EXIT_FAILURE);
                }
            }
        }
        
        // Wyświetl info o urządzeniu
        char device_name[128];
        clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, NULL);
        cout << "Używam urządzenia: " << device_name << "\n\n";
        
        // 3. Utwórz kontekst
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        CHECK_CL(err, "Create context");
        
        // 4. PROBLEM 3: Utwórz wiele kolejek (strumieni) dla asynchronicznego wykonania
        // Flaga CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE pozwala na wykonywanie poza kolejnością
        cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        
        queue_compute = clCreateCommandQueue(context, device, props, &err);
        CHECK_CL(err, "Create compute queue");
        
        queue_transfer = clCreateCommandQueue(context, device, props, &err);
        CHECK_CL(err, "Create transfer queue");
        
        queue_reduction = clCreateCommandQueue(context, device, props, &err);
        CHECK_CL(err, "Create reduction queue");
        
        cout << "Utworzono 3 niezależne kolejki (strumienie) OpenCL\n";
        
        // 5. Kompiluj kernele
        program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
        CHECK_CL(err, "Create program");
        
        err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
        if(err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char* log = new char[log_size];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            cerr << "Build error:\n" << log << endl;
            delete[] log;
            exit(EXIT_FAILURE);
        }
        
        // 6. Utwórz kernele
        k_matvec = clCreateKernel(program, "matrix_vector_multiply", &err);
        CHECK_CL(err, "Create kernel matvec");
        k_add_scaled = clCreateKernel(program, "vector_add_scaled", &err);
        CHECK_CL(err, "Create kernel add_scaled");
        k_subtract = clCreateKernel(program, "vector_subtract", &err);
        CHECK_CL(err, "Create kernel subtract");
        k_update_dir = clCreateKernel(program, "update_direction", &err);
        CHECK_CL(err, "Create kernel update_dir");
        k_reduce_blocks = clCreateKernel(program, "reduce_blocks", &err);
        CHECK_CL(err, "Create kernel reduce_blocks");
        k_reduce_final = clCreateKernel(program, "reduce_final", &err);
        CHECK_CL(err, "Create kernel reduce_final");
        k_reduce_atomic = clCreateKernel(program, "reduce_atomic", &err);
        CHECK_CL(err, "Create kernel reduce_atomic");
        
        cout << "Skompilowano kernele z operacjami atomowymi i redukcją\n";
        
        // 7. Alokuj pamięć
        num_work_groups = (n + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
        
        d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, (size_t)n*n*sizeof(double), NULL, &err);
        CHECK_CL(err, "Create buffer A");
        d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, n*sizeof(double), NULL, &err);
        CHECK_CL(err, "Create buffer b");
        d_x = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(double), NULL, &err);
        CHECK_CL(err, "Create buffer x");
        d_r = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(double), NULL, &err);
        CHECK_CL(err, "Create buffer r");
        d_p = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(double), NULL, &err);
        CHECK_CL(err, "Create buffer p");
        d_Ax = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(double), NULL, &err);
        CHECK_CL(err, "Create buffer Ax");
        d_norm_result = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);
        CHECK_CL(err, "Create buffer norm");
        d_block_results = clCreateBuffer(context, CL_MEM_READ_WRITE, num_work_groups*sizeof(double), NULL, &err);
        CHECK_CL(err, "Create buffer blocks");
    }
    
    ~ChebyshevSolverOpenCL() {
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_b);
        clReleaseMemObject(d_x);
        clReleaseMemObject(d_r);
        clReleaseMemObject(d_p);
        clReleaseMemObject(d_Ax);
        clReleaseMemObject(d_norm_result);
        clReleaseMemObject(d_block_results);
        
        clReleaseKernel(k_matvec);
        clReleaseKernel(k_add_scaled);
        clReleaseKernel(k_subtract);
        clReleaseKernel(k_update_dir);
        clReleaseKernel(k_reduce_blocks);
        clReleaseKernel(k_reduce_final);
        clReleaseKernel(k_reduce_atomic);
        
        clReleaseProgram(program);
        clReleaseCommandQueue(queue_compute);
        clReleaseCommandQueue(queue_transfer);
        clReleaseCommandQueue(queue_reduction);
        clReleaseContext(context);
    }
    
    void setMatrix(const double* A) {
        // PROBLEM 3: Transfer w dedykowanej kolejce (asynchronicznie)
        CHECK_CL(clEnqueueWriteBuffer(queue_transfer, d_A, CL_FALSE, 0, 
                 (size_t)n*n*sizeof(double), A, 0, NULL, NULL), "Write A");
    }
    
    void setRHS(const double* b) {
        // PROBLEM 3: Transfer w dedykowanej kolejce
        CHECK_CL(clEnqueueWriteBuffer(queue_transfer, d_b, CL_FALSE, 0, 
                 n*sizeof(double), b, 0, NULL, NULL), "Write b");
        clFinish(queue_transfer); // Poczekaj na zakończenie transferów
    }
    
    // PROBLEM 1 & 2: Obliczanie normy Z UŻYCIEM OPERACJI ATOMOWYCH
    double computeNormAtomic(cl_mem d_vec) {
        double zero = 0.0;
        CHECK_CL(clEnqueueWriteBuffer(queue_reduction, d_norm_result, CL_FALSE, 0, 
                 sizeof(double), &zero, 0, NULL, NULL), "Reset norm");
        
        // Użyj kernela z operacją atomową
        size_t global_size = num_work_groups * WORK_GROUP_SIZE;
        size_t local_size = WORK_GROUP_SIZE;
        
        clSetKernelArg(k_reduce_atomic, 0, sizeof(cl_mem), &d_vec);
        clSetKernelArg(k_reduce_atomic, 1, sizeof(cl_mem), &d_norm_result);
        clSetKernelArg(k_reduce_atomic, 2, WORK_GROUP_SIZE*sizeof(double), NULL);
        clSetKernelArg(k_reduce_atomic, 3, sizeof(int), &n);
        
        // PROBLEM 3: Wykonaj w dedykowanej kolejce redukcji
        CHECK_CL(clEnqueueNDRangeKernel(queue_reduction, k_reduce_atomic, 1, NULL, 
                 &global_size, &local_size, 0, NULL, NULL), "Reduce atomic");
        
        double result;
        CHECK_CL(clEnqueueReadBuffer(queue_reduction, d_norm_result, CL_TRUE, 0, 
                 sizeof(double), &result, 0, NULL, NULL), "Read norm");
        
        return sqrt(result);
    }
    
    // PROBLEM 2: Obliczanie normy z dwuetapową redukcją (bez atomics)
    double computeNorm(cl_mem d_vec) {
        double zero = 0.0;
        CHECK_CL(clEnqueueWriteBuffer(queue_reduction, d_norm_result, CL_FALSE, 0, 
                 sizeof(double), &zero, 0, NULL, NULL), "Reset norm");
        
        // Etap 1: redukcja w work-groups
        size_t global_size = num_work_groups * WORK_GROUP_SIZE;
        size_t local_size = WORK_GROUP_SIZE;
        
        clSetKernelArg(k_reduce_blocks, 0, sizeof(cl_mem), &d_vec);
        clSetKernelArg(k_reduce_blocks, 1, sizeof(cl_mem), &d_block_results);
        clSetKernelArg(k_reduce_blocks, 2, WORK_GROUP_SIZE*sizeof(double), NULL);
        clSetKernelArg(k_reduce_blocks, 3, sizeof(int), &n);
        
        // PROBLEM 3: Wykonaj w dedykowanej kolejce
        CHECK_CL(clEnqueueNDRangeKernel(queue_reduction, k_reduce_blocks, 1, NULL, 
                 &global_size, &local_size, 0, NULL, NULL), "Reduce blocks");
        
        // Etap 2: finalna redukcja
        global_size = WORK_GROUP_SIZE;
        clSetKernelArg(k_reduce_final, 0, sizeof(cl_mem), &d_block_results);
        clSetKernelArg(k_reduce_final, 1, sizeof(cl_mem), &d_norm_result);
        clSetKernelArg(k_reduce_final, 2, WORK_GROUP_SIZE*sizeof(double), NULL);
        clSetKernelArg(k_reduce_final, 3, sizeof(int), &num_work_groups);
        
        CHECK_CL(clEnqueueNDRangeKernel(queue_reduction, k_reduce_final, 1, NULL, 
                 &global_size, &local_size, 0, NULL, NULL), "Reduce final");
        
        double result;
        CHECK_CL(clEnqueueReadBuffer(queue_reduction, d_norm_result, CL_TRUE, 0, 
                 sizeof(double), &result, 0, NULL, NULL), "Read norm");
        
        return sqrt(result);
    }
    
    int solve(double l_min, double l_max, vector<double>& x_result) {
        double d = (l_max + l_min) * 0.5;
        double c = (l_max - l_min) * 0.5;
        
        // Inicjalizacja - użyj queue_compute
        vector<double> zeros(n, 0.0);
        CHECK_CL(clEnqueueWriteBuffer(queue_compute, d_x, CL_FALSE, 0, n*sizeof(double), 
                 zeros.data(), 0, NULL, NULL), "Init x");
        CHECK_CL(clEnqueueCopyBuffer(queue_compute, d_b, d_r, 0, 0, n*sizeof(double), 0, NULL, NULL), "Init r");
        CHECK_CL(clEnqueueCopyBuffer(queue_compute, d_b, d_p, 0, 0, n*sizeof(double), 0, NULL, NULL), "Init p");
        clFinish(queue_compute);
        
        double alpha = 1.0 / d;
        double beta = 0.0;
        int iter;
        
        size_t global_size = ((n + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE;
        size_t local_size = WORK_GROUP_SIZE;
        
        for(iter = 0; iter < MAX_ITERATIONS; iter++) {
            // PROBLEM 3: Wszystkie obliczenia w queue_compute (asynchronicznie)
            
            // x = x + alpha * p
            clSetKernelArg(k_add_scaled, 0, sizeof(cl_mem), &d_x);
            clSetKernelArg(k_add_scaled, 1, sizeof(cl_mem), &d_p);
            clSetKernelArg(k_add_scaled, 2, sizeof(double), &alpha);
            clSetKernelArg(k_add_scaled, 3, sizeof(int), &n);
            CHECK_CL(clEnqueueNDRangeKernel(queue_compute, k_add_scaled, 1, NULL, 
                     &global_size, &local_size, 0, NULL, NULL), "Add scaled");
            
            // Ax = A * x
            clSetKernelArg(k_matvec, 0, sizeof(cl_mem), &d_A);
            clSetKernelArg(k_matvec, 1, sizeof(cl_mem), &d_x);
            clSetKernelArg(k_matvec, 2, sizeof(cl_mem), &d_Ax);
            clSetKernelArg(k_matvec, 3, sizeof(int), &n);
            CHECK_CL(clEnqueueNDRangeKernel(queue_compute, k_matvec, 1, NULL, 
                     &global_size, &local_size, 0, NULL, NULL), "MatVec");
            
            // r = b - Ax
            clSetKernelArg(k_subtract, 0, sizeof(cl_mem), &d_r);
            clSetKernelArg(k_subtract, 1, sizeof(cl_mem), &d_b);
            clSetKernelArg(k_subtract, 2, sizeof(cl_mem), &d_Ax);
            clSetKernelArg(k_subtract, 3, sizeof(int), &n);
            CHECK_CL(clEnqueueNDRangeKernel(queue_compute, k_subtract, 1, NULL, 
                     &global_size, &local_size, 0, NULL, NULL), "Subtract");
            
            // Synchronizuj kolejkę obliczeń przed sprawdzaniem zbieżności
            clFinish(queue_compute);
            
            // PROBLEM 1 & 2: Sprawdź zbieżność używając operacji atomowych
            // (queue_reduction działa równolegle z queue_compute!)
            double norm_r = computeNormAtomic(d_r); // Używa operacji atomowych
            if(norm_r < TOLERANCE) break;
            
            // Aktualizacja parametrów
            double beta_new = pow(c / (2.0 * d), 2) * (1.0 - beta);
            double alpha_new = 1.0 / (d - beta_new * d);
            
            // p = r + beta_new * p
            clSetKernelArg(k_update_dir, 0, sizeof(cl_mem), &d_p);
            clSetKernelArg(k_update_dir, 1, sizeof(cl_mem), &d_r);
            clSetKernelArg(k_update_dir, 2, sizeof(double), &beta_new);
            clSetKernelArg(k_update_dir, 3, sizeof(int), &n);
            CHECK_CL(clEnqueueNDRangeKernel(queue_compute, k_update_dir, 1, NULL, 
                     &global_size, &local_size, 0, NULL, NULL), "Update dir");
            
            alpha = alpha_new;
            beta = beta_new;
        }
        
        // Kopiuj wynik - użyj queue_transfer
        x_result.resize(n);
        CHECK_CL(clEnqueueReadBuffer(queue_transfer, d_x, CL_TRUE, 0, n*sizeof(double), 
                 x_result.data(), 0, NULL, NULL), "Read result");
        
        return iter;
    }
};

// =============================================================================
// Testy
// =============================================================================

void runAutotest() {
    cout << "=== AUTOTEST OpenCL (2x2) ===\n";
    
    int n = 2;
    double A[4] = {4.0, 1.0, 1.0, 3.0};
    double b[2] = {6.0, 7.0};
    
    ChebyshevSolverOpenCL solver(n);
    solver.setMatrix(A);
    solver.setRHS(b);
    
    vector<double> x;
    int iter = solver.solve(2.0, 5.0, x);
    
    cout << "Wynik: x1=" << x[0] << ", x2=" << x[1] << " (Oczekiwane: 1, 2)\n";
    cout << "Liczba iteracji: " << iter << "\n";
    cout << "================================\n\n";
}

void runBenchmark() {
    int sizes[] = {100, 500, 1000, 2000};
    int num_tests = sizeof(sizes) / sizeof(sizes[0]);
    
    cout << fixed << setprecision(2);
    cout << setw(6) << "N" << " | " 
         << setw(13) << "T_OpenCL [ms]" << " | " 
         << setw(8) << "Iteracje" << endl;
    cout << string(42, '-') << endl;
    
    for(int t = 0; t < num_tests; t++) {
        int n = sizes[t];
        
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
        
        ChebyshevSolverOpenCL solver(n);
        solver.setMatrix(A.data());
        solver.setRHS(b.data());
        
        auto start = chrono::high_resolution_clock::now();
        vector<double> x;
        int iter = solver.solve(5.0, n * 1.5, x);
        auto end = chrono::high_resolution_clock::now();
        
        double ms = chrono::duration<double, milli>(end - start).count();
        
        cout << setw(6) << n << " | " 
             << setw(13) << ms << " | " 
             << setw(8) << iter << endl;
    }
}

int main() {
    runAutotest();
    runBenchmark();
    return 0;
}