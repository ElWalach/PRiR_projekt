#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <mpi.h>

using namespace std;
using namespace chrono;

const int MAX_ITERATIONS = 10000;
const double TOLERANCE = 1e-6;

// Mnożenie macierzy przez wektor - wersja sekwencyjna (macierz 1D)
vector<double> matrixVectorMultiply(const vector<double>& A, const vector<double>& x, int n) {
    vector<double> r(n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            r[i] += A[i * n + j] * x[j];
    return r;
}

// Mnożenie macierzy przez wektor - wersja równoległa (Struktura Gwiazdy)
vector<double> matrixVectorMultiplyMPI(const vector<double>& local_A, 
                                       const vector<double>& x,
                                       int n, int rank, int size) {
    int rows_per_proc = n / size;
    vector<double> local_res(rows_per_proc, 0.0);
    
    // 1. Obliczenia lokalne na procesie (każdy liczy swoje wiersze)
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < n; j++) {
            local_res[i] += local_A[i * n + j] * x[j];
        }
    }
    
    vector<double> global_res(n);
    
    // 2. GATHER: Każdy worker wysyła swój fragment wyniku DO PROCESU 0 (Centrum gwiazdy)
    MPI_Gather(local_res.data(), rows_per_proc, MPI_DOUBLE, 
               global_res.data(), rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // 3. BCAST: Proces 0 rozsyła scalony, pełny wektor DO WSZYSTKICH (Gwiazda)
    MPI_Bcast(global_res.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return global_res;
}

double dotProduct(const vector<double>& a, const vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}

double vectorNorm(const vector<double>& v) {
    return sqrt(dotProduct(v, v));
}

// Solver metodą Czebyszewa
pair<double, int> solveChebyshev(const vector<double>& A, const vector<double>& b,
                                 bool parallel, vector<double>& x_out,
                                 int rank, int size, int n) {
    int rows_per_proc = n / size;
    vector<double> local_A(rows_per_proc * n);
    vector<double> x(n, 0.0);

    // SCATTER: Macierz A płynie z Procesu 0 do wszystkich (Gwiazda)
    if (parallel) {
        MPI_Scatter(A.data(), rows_per_proc * n, MPI_DOUBLE, 
                    local_A.data(), rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    auto start = high_resolution_clock::now();
    
    // Parametry szacowane (uproszczone dla stabilności testu)
    double lambda_max = n + 20.0;
    double lambda_min = 1.0;
    
    double d = (lambda_max + lambda_min) * 0.5;
    double c = (lambda_max - lambda_min) * 0.5;
    
    vector<double> r(n), p(n);
    vector<double> Ax = parallel ? matrixVectorMultiplyMPI(local_A, x, n, rank, size) : matrixVectorMultiply(A, x, n);
    
    for (int i = 0; i < n; i++) r[i] = b[i] - Ax[i];
    
    double alpha = 1.0 / d;
    double beta = 0.0;
    p = r;
    
    int iter;
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        if (vectorNorm(r) < TOLERANCE) break;
        
        for (int i = 0; i < n; i++) x[i] += alpha * p[i];
        
        vector<double> Ap = parallel ? matrixVectorMultiplyMPI(local_A, p, n, rank, size) : matrixVectorMultiply(A, p, n);
        
        for (int i = 0; i < n; i++) r[i] -= alpha * Ap[i];
        
        double beta_new = pow(c / (2.0 * d), 2) * (1.0 - beta);
        double alpha_new = 1.0 / (d - beta_new * d);
        
        for (int i = 0; i < n; i++) p[i] = r[i] + beta_new * p[i];
        
        alpha = alpha_new;
        beta = beta_new;
    }
    
    auto end = high_resolution_clock::now();
    x_out = x;
    return {(double)duration_cast<milliseconds>(end - start).count(), iter};
}

void generateMatrix1D(vector<double>& A, vector<double>& b, int n) {
    mt19937 gen(42);
    uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) A[i * n + j] = n + 30.0;
            else A[i * n + j] = A[j * n + i] = dis(gen);
        }
        b[i] = dis(gen) * 10.0;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> sizes = {100, 500, 1000, 2000};

    if (rank == 0) {
        cout << "Testy metody Czebyszewa (MPI Scatter/Gather/Bcast - Topologia Gwiazdy)\n";
        cout << "Liczba procesów: " << size << "\n\n";
        cout << setw(6) << "n" << " | " << setw(12) << "Czas Sekw" << " | " << setw(12) << "Czas MPI" << " | " << "Speedup\n";
        cout << string(60, '-') << "\n";
    }

    for (int n_val : sizes) {
        // KAŻDY proces musi obliczyć to samo n przed alokacją i komunikacją
        int n = n_val;
        if (n % size != 0) n = (n / size + 1) * size;

        // KAŻDY proces musi mieć wektor b o rozmiarze n
        vector<double> A_global, b(n), x_final(n);

        if (rank == 0) {
            A_global.resize(n * n);
            generateMatrix1D(A_global, b, n);
        }

        // BCAST wektora b - teraz rozmiary na wszystkich procesach się zgadzają
        MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double t_seq = 0;
        if (rank == 0) {
            auto res_seq = solveChebyshev(A_global, b, false, x_final, rank, 1, n);
            t_seq = res_seq.first;
        }

        // Synchronizacja przed testem równoległym
        MPI_Barrier(MPI_COMM_WORLD);

        // Test równoległy
        auto res_par = solveChebyshev(A_global, b, true, x_final, rank, size, n);
        
        if (rank == 0) {
            double speedup = (res_par.first > 0) ? t_seq / res_par.first : 0;
            cout << setw(6) << n << " | " 
                 << setw(10) << fixed << setprecision(2) << t_seq << " ms | " 
                 << setw(10) << res_par.first << " ms | " 
                 << speedup << "x\n";
        }
    }

    MPI_Finalize();
    return 0;
}