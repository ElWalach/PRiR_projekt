#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <mpi.h>
#include <cstring>

using namespace std;

const int MAX_ITERATIONS = 5000;
const double TOLERANCE = 1e-6;

// Funkcja pomocnicza do obliczania normy wektora
double vectorNorm(const vector<double>& v) {
    double sum = 0.0;
    for (double val : v) sum += val * val;
    return sqrt(sum);
}


vector<double> matrixVectorMultiplyParallel(const double* local_A, const vector<double>& x, int n, int local_rows, int rank, int size, int* row_counts, int* row_start) {
    vector<double> local_res(local_rows, 0.0);
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            local_res[i] += local_A[i * n + j] * x[j];
        }
    }

    vector<double> full_res(n);
    
    // Zbieramy wyniki cząstkowe do procesu 0
    MPI_Gatherv(local_res.data(), local_rows, MPI_DOUBLE,
                full_res.data(), row_counts, row_start, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Rozsyłamy gotowy wektor do wszystkich procesów
    MPI_Bcast(full_res.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return full_res;
}

// Wersja sekwencyjna mnożenia
vector<double> matrixVectorMultiplySeq(const vector<double>& A, const vector<double>& x, int n) {
    vector<double> r(n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            r[i] += A[i * n + j] * x[j];
    return r;
}

void runAutotest(int rank, int size) {
    if (rank == 0) cout << "=== URUCHAMIANIE AUTOTESTU (2x2) ===\n";
    
    int n = 2;
    double A_raw[4] = {4.0, 1.0, 1.0, 3.0};
    double b_raw[2] = {6.0, 7.0};
    
    double l_min = 2.0, l_max = 5.0;
    double d = (l_max + l_min) * 0.5;
    double c = (l_max - l_min) * 0.5;

    int row_counts[size], row_start[size];
    for (int i = 0; i < size; i++) {
        row_counts[i] = (n / size) + (i < (n % size) ? 1 : 0);
        row_start[i] = (i == 0) ? 0 : row_start[i-1] + row_counts[i-1];
    }

    int local_rows = row_counts[rank];
    double* local_A = new double[local_rows * n];
    int send_counts[size], displs[size];
    for(int i=0; i<size; i++) { send_counts[i] = row_counts[i]*n; displs[i] = row_start[i]*n; }

    MPI_Scatterv(A_raw, send_counts, displs, MPI_DOUBLE, local_A, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    vector<double> b = {b_raw[0], b_raw[1]};
    MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> x(n, 0.0);
    vector<double> r = b;
    vector<double> p = r;
    double alpha = 1.0 / d, beta = 0.0;
    int iter;

    for (iter = 0; iter < 100; iter++) {
        for (int i = 0; i < n; i++) x[i] += alpha * p[i];
        vector<double> Ap = matrixVectorMultiplyParallel(local_A, x, n, local_rows, rank, size, row_counts, row_start);
        for (int i = 0; i < n; i++) r[i] = b[i] - Ap[i];
        if (vectorNorm(r) < 1e-5) break;
        double beta_new = pow(c / (2.0 * d), 2) * (1.0 - beta);
        double alpha_new = 1.0 / (d - beta_new * d);
        for (int i = 0; i < n; i++) p[i] = r[i] + beta_new * p[i];
        alpha = alpha_new; beta = beta_new;
    }

    if (rank == 0) {
        cout << "Wynik: x1=" << x[0] << ", x2=" << x[1] << " (Oczekiwane: 1, 2)\n";
        cout << "Liczba iteracji: " << iter << "\n";
        cout << "====================================\n\n";
    }
    delete[] local_A;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    runAutotest(rank, size);

    int sizes[] = {10, 100, 1000, 10000};
    int num_tests = sizeof(sizes) / sizeof(sizes[0]);

    if (rank == 0) {
        cout << fixed << setprecision(2);
        cout << setw(6) << "N" << " | " 
             << setw(11) << "T_sekw [ms]" << " | " 
             << setw(8) << "It_sekw" << " | "
             << setw(11) << "T_MPI [ms]" << " | " 
             << setw(8) << "It_MPI" << " | "
             << "Przysp." << endl;
        cout << string(78, '-') << endl;
    }


//podzial wierszy dla procesów

    for (int t = 0; t < num_tests; t++) {
        int n = sizes[t];
        int* row_counts = new int[size];
        int* row_start = new int[size];
        int rows_per_proc = n / size;
        int remainder = n % size;

        for (int i = 0; i < size; i++) {
            row_counts[i] = rows_per_proc + (i < remainder ? 1 : 0);
            row_start[i] = (i == 0) ? 0 : row_start[i - 1] + row_counts[i - 1];
        }

        int local_rows = row_counts[rank];
        double* local_A = new double[(size_t)local_rows * n];
        vector<double> b(n);
        double* full_A = nullptr;

// generowanie macierzy (tylko proces macierzysty)

        if (rank == 0) {
            full_A = new double[(size_t)n * n];
            for (int i = 0; i < n; i++) {
                double row_sum = 0;
                for (int j = 0; j < n; j++) {
                    full_A[i * n + j] = (double)(rand() % 10) / 100.0;
                    row_sum += full_A[i * n + j];
                }
                full_A[i * n + i] = row_sum + 10.0;
                b[i] = (double)(rand() % 100);
            }
        }

        int* send_counts_A = new int[size];
        int* displs_A = new int[size]; 

        //przeliczenie wierszy na liczbe elementow do wyslania
        for (int i = 0; i < size; i++) {
            send_counts_A[i] = row_counts[i] * n;
            displs_A[i] = row_start[i] * n;
        }
// rozeslanie podzielonej macierzy i calego wektora do procesow
        MPI_Scatterv(full_A, send_counts_A, displs_A, MPI_DOUBLE, local_A, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double l_min = 5.0, l_max = n * 1.5; 
        double d = (l_max + l_min) * 0.5;
        double c = (l_max - l_min) * 0.5;

        // --- TEST RÓWNOLEGŁY ---

        //synchronizacja przed pomiarem czasu
        MPI_Barrier(MPI_COMM_WORLD);
        double t_par_start = MPI_Wtime();
        vector<double> x(n, 0.0), r = b, p = r;
        double alpha = 1.0 / d, beta = 0.0;
        int iter_par;

        for (iter_par = 0; iter_par < MAX_ITERATIONS; iter_par++) {
            for (int i = 0; i < n; i++) x[i] += alpha * p[i];

            vector<double> Ap = matrixVectorMultiplyParallel(local_A, x, n, local_rows, rank, size, row_counts, row_start);
            
            
            for (int i = 0; i < n; i++) 
            r[i] = b[i] - Ap[i];

            if (vectorNorm(r) < TOLERANCE) break;

            double beta_new = pow(c / (2.0 * d), 2) * (1.0 - beta);
            double alpha_new = 1.0 / (d - beta_new * d);

            for (int i = 0; i < n; i++) p[i] = r[i] + beta_new * p[i];
            alpha = alpha_new; beta = beta_new;
        }
        double time_parallel_ms = (MPI_Wtime() - t_par_start) * 1000.0;

        // --- TEST SEKWENCYJNY ---
        if (rank == 0) {
            double t_seq_start = MPI_Wtime();
            vector<double> x_s(n, 0.0), r_s = b, p_s = r_s;
            vector<double> A_vec(full_A, full_A + (size_t)n * n);
            double sa = 1.0 / d, sb = 0.0;
            int iter_seq;

            for (iter_seq = 0; iter_seq < MAX_ITERATIONS; iter_seq++) {
               
                for (int i = 0; i < n; i++) x_s[i] += sa * p_s[i];
               
                vector<double> Ap_s = matrixVectorMultiplySeq(A_vec, x_s, n);
               
                for (int i = 0; i < n; i++) r_s[i] = b[i] - Ap_s[i];
               
                if (vectorNorm(r_s) < TOLERANCE) break;
               
                double bn = pow(c / (2.0 * d), 2) * (1.0 - sb);
                double an = 1.0 / (d - bn * d);
               
                for (int i = 0; i < n; i++) p_s[i] = r_s[i] + bn * p_s[i];
                sa = an; sb = bn;
            }
            double time_seq_ms = (MPI_Wtime() - t_seq_start) * 1000.0;
            
            double speedup = (time_parallel_ms > 0) ? (time_seq_ms / time_parallel_ms) : 0;
            
            cout << setw(6) << n << " | " 
                 << setw(11) << time_seq_ms << " | " 
                 << setw(8) << iter_seq << " | "
                 << setw(11) << time_parallel_ms << " | " 
                 << setw(8) << iter_par << " | "
                 << speedup << "x" << endl;
        }

        delete[] row_counts; delete[] row_start;
        delete[] send_counts_A; delete[] displs_A;
        delete[] local_A;
        if (rank == 0) delete[] full_A;
    }

    MPI_Finalize();
    return 0;
}