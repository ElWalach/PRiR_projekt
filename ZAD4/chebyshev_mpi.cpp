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

// Mnożenie macierzy przez wektor - wersja sekwencyjna
vector<double> matrixVectorMultiply(const vector<vector<double>>& A, const vector<double>& x) {
    int n = A.size();
    vector<double> r(n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            r[i] += A[i][j] * x[j];
    return r;
}

// Mnożenie macierzy przez wektor - wersja równoległa z topologią bez sąsiedztwa
// Tylko proces 0 komunikuje się bezpośrednio z innymi (topologia gwiazdy)
vector<double> matrixVectorMultiplyMPI(const vector<vector<double>>& A, 
                                       const vector<double>& x,
                                       int rank, int size) {
    int n = A.size();
    vector<double> result(n, 0.0);
    
    // Obliczamy zakres wierszy dla tego procesu
    int rows_per_proc = n / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? n : (rank + 1) * rows_per_proc;
    
    // Każdy proces oblicza swoją część
    vector<double> local_result(n, 0.0);
    for (int i = start_row; i < end_row; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += A[i][j] * x[j];
        local_result[i] = sum;
    }
    
    
    if (rank == 0) {
        // Proces 0 odbiera wyniki od wszystkich workerów
        result = local_result;  // Najpierw własne wyniki
        
        for (int worker = 1; worker < size; worker++) {
            vector<double> worker_result(n, 0.0);
            MPI_Recv(worker_result.data(), n, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Dodaj wyniki workera do globalnego wyniku
            for (int i = 0; i < n; i++) {
                result[i] += worker_result[i];
            }
        }
        
        // Proces 0 rozsyła wynik końcowy do wszystkich workerów
        for (int worker = 1; worker < size; worker++) {
            MPI_Send(result.data(), n, MPI_DOUBLE, worker, 1, MPI_COMM_WORLD);
        }
    } else {
        // Procesy worker wysyłają swoje wyniki TYLKO do procesu 0
        MPI_Send(local_result.data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        
        // Odbierają wynik końcowy z powrotem od procesu 0
        MPI_Recv(result.data(), n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    return result;
}

// Iloczyn skalarny
double dotProduct(const vector<double>& a, const vector<double>& b) {
    double s = 0.0;
    for (int i = 0; i < a.size(); i++) 
        s += a[i] * b[i];
    return s;
}

// Norma euklidesowa
double vectorNorm(const vector<double>& v) {
    return sqrt(dotProduct(v, v));
}

// Szacowanie wartości własnych
pair<double, double> estimateEigenvalues(const vector<vector<double>>& A, 
                                         bool parallel, int rank, int size) {
    int n = A.size();
    vector<double> v(n, 1.0 / sqrt(n));
    
    double lambda_max = 0.0;
    for (int iter = 0; iter < 50; iter++) {
        vector<double> Av = parallel ? 
            matrixVectorMultiplyMPI(A, v, rank, size) :
            matrixVectorMultiply(A, v);
        
        double norm = vectorNorm(Av);
        if (norm < 1e-12) break;
        
        for (int i = 0; i < n; i++)
            v[i] = Av[i] / norm;
        
        lambda_max = norm;
    }
    
    // Metoda Gerszgorina dla lambda_min
    double lambda_min = 1e10;
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++)
            if (i != j) row_sum += abs(A[i][j]);
        lambda_min = min(lambda_min, abs(A[i][i]) - row_sum);
    }
    
    if (lambda_min <= 0)
        lambda_min = lambda_max * 0.01;
    
    return {lambda_min, lambda_max};
}

// Solver metodą Czebyszewa
pair<double, int> solveChebyshev(const vector<vector<double>>& A,
                                 const vector<double>& b,
                                 bool parallel,
                                 vector<double>& x_out,
                                 int rank, int size) {
    int n = A.size();
    vector<double> x(n, 0.0);
    
    auto start = high_resolution_clock::now();
    
    // Szacowanie wartości własnych
    auto [lambda_min, lambda_max] = estimateEigenvalues(A, parallel, rank, size);
    
    // Parametry Czebyszewa
    double d = (lambda_max + lambda_min) * 0.5;
    double c = (lambda_max - lambda_min) * 0.5;
    
    // Początkowe residuum
    vector<double> Ax = parallel ?
        matrixVectorMultiplyMPI(A, x, rank, size) :
        matrixVectorMultiply(A, x);
    
    vector<double> r(n), p(n);
    for (int i = 0; i < n; i++)
        r[i] = b[i] - Ax[i];
    
    double alpha = 1.0 / d;
    double beta = 0.0;
    p = r;
    
    // Główna pętla iteracji
    int iter;
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        if (vectorNorm(r) < TOLERANCE) break;
        
        // x = x + alpha * p
        for (int i = 0; i < n; i++)
            x[i] += alpha * p[i];
        
        // r = r - alpha * A*p
        vector<double> Ap = parallel ?
            matrixVectorMultiplyMPI(A, p, rank, size) :
            matrixVectorMultiply(A, p);
        
        for (int i = 0; i < n; i++)
            r[i] -= alpha * Ap[i];
        
        // Aktualizacja parametrów
        double beta_new = pow(c / (2.0 * d), 2) * (1.0 - beta);
        double alpha_new = 1.0 / (d - beta_new * d);
        
        // p = r + beta_new * p
        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta_new * p[i];
        
        alpha = alpha_new;
        beta = beta_new;
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<milliseconds>(end - start).count();
    
    x_out = x;
    return {time_ms, iter};
}

// Generowanie macierzy symetrycznej dodatnio określonej
void generateMatrix(vector<vector<double>>& A, 
                   vector<double>& b, 
                   vector<double>& x_true, 
                   int n) {
    mt19937 gen(42);
    uniform_real_distribution<> dis(-1.0, 1.0);
    
    // Macierz symetryczna
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            if (i == j) {
                A[i][j] = n + abs(dis(gen)) * 10.0;
            } else {
                A[i][j] = dis(gen);
                A[j][i] = A[i][j];
            }
        }
    }
    
    // Prawdziwe rozwiązanie
    for (int i = 0; i < n; i++)
        x_true[i] = dis(gen) * 10.0;
    
    // Wektor b = A * x_true
    for (int i = 0; i < n; i++) {
        b[i] = 0.0;
        for (int j = 0; j < n; j++)
            b[i] += A[i][j] * x_true[j];
    }
}

// Autotest
void runAutotest(int rank) {
    if (rank == 0) {
        vector<vector<double>> A = {
            {4, 1},
            {1, 3}
        };
        vector<double> b = {6, 7};
        vector<double> x;
        
        auto [time, iter] = solveChebyshev(A, b, false, x, 0, 1);
        
        cout << "=== AUTOTEST 2x2 ===\n";
        cout << "Rozwiązywanie układu:\n";
        cout << "[4 1] [x1] = [6]\n";
        cout << "[1 3] [x2] = [7]\n\n";
        cout << "Otrzymany wynik: x = ("
             << x[0] << ", " << x[1] << ")\n";
        cout << "Czas: " << time << " ms, Iteracje: " << iter << "\n";
        cout << "=====================\n\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Autotest tylko na proces 0
    runAutotest(rank);
    
    vector<int> sizes = {10, 100, 1000, 10000};
    
    if (rank == 0) {
        
        cout << "Liczba procesów: " << size << "\n";
        
        cout << setw(6) << "n" << " | "
             << setw(12) << "czas_sekw" << " | "
             << setw(10) << "iter_sekw" << " | "
             << setw(12) << "czas_MPI" << " | "
             << setw(10) << "iter_MPI" << " | "
             << setw(10) << "przyspieszenie\n";
        cout << string(80, '-') << "\n";
    }
    
    for (int n : sizes) {
        vector<vector<double>> A(n, vector<double>(n));
        vector<double> b(n), x_true(n);
        
        // Proces 0 generuje dane i rozsyła je PUNKT-PUNKT do innych
        if (rank == 0) {
            generateMatrix(A, b, x_true, n);
            
            // Rozsyłanie punkt-punkt (bez broadcast - brak relacji sąsiedztwa)
            for (int dest = 1; dest < size; dest++) {
                for (int i = 0; i < n; i++) {
                    MPI_Send(A[i].data(), n, MPI_DOUBLE, dest, i, MPI_COMM_WORLD);
                }
                MPI_Send(b.data(), n, MPI_DOUBLE, dest, n, MPI_COMM_WORLD);
            }
        } else {
            // Każdy proces worker odbiera dane TYLKO od procesu 0
            for (int i = 0; i < n; i++) {
                MPI_Recv(A[i].data(), n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            MPI_Recv(b.data(), n, MPI_DOUBLE, 0, n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Test sekwencyjny (tylko proces 0)
        vector<double> x_seq;
        double time_seq = 0;
        int iter_seq = 0;
        
        if (rank == 0) {
            auto result = solveChebyshev(A, b, false, x_seq, rank, 1);
            time_seq = result.first;
            iter_seq = result.second;
        }
        
        // Synchronizacja poprzez komunikację punkt-punkt z procesem 0
        if (rank == 0) {
            int dummy;
            for (int worker = 1; worker < size; worker++) {
                MPI_Recv(&dummy, 1, MPI_INT, worker, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int worker = 1; worker < size; worker++) {
                MPI_Send(&dummy, 1, MPI_INT, worker, 1000, MPI_COMM_WORLD);
            }
        } else {
            int dummy = 0;
            MPI_Send(&dummy, 1, MPI_INT, 0, 999, MPI_COMM_WORLD);
            MPI_Recv(&dummy, 1, MPI_INT, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Test równoległy (wszystkie procesy)
        vector<double> x_par;
        auto result = solveChebyshev(A, b, true, x_par, rank, size);
        double time_par = result.first;
        int iter_par = result.second;
        
        // Wyniki wypisuje tylko proces 0
        if (rank == 0) {
            double speedup = time_seq / time_par;
            cout << setw(6) << n << " | "
                 << setw(12) << fixed << setprecision(2) << time_seq << " | "
                 << setw(10) << iter_seq << " | "
                 << setw(12) << time_par << " | "
                 << setw(10) << iter_par << " | "
                 << setw(10) << setprecision(2) << speedup << "x\n";
        }
    }
    
    if (rank == 0) {
        cout << "\n=== KONIEC TESTÓW ===\n";
    }
    
    MPI_Finalize();
    return 0;
}