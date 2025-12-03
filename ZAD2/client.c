#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "chebyshev.h"

// Generator liczb pseudolosowych
double randomDouble(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

void generateSymmetricPositiveDefiniteMatrix(double** A, double* b, 
                                            double* x_true, int n) {
    srand(42);
    
    // Generuj macierz symetryczną
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            if (i == j) {
                A[i][j] = n + fabs(randomDouble(-1.0, 1.0)) * 10.0;
            } else {
                A[i][j] = randomDouble(-1.0, 1.0);
                A[j][i] = A[i][j];
            }
        }
    }

    // Generuj prawdziwe rozwiązanie
    for (int i = 0; i < n; i++)
        x_true[i] = randomDouble(-10.0, 10.0);

    // Oblicz b = A * x_true
    for (int i = 0; i < n; i++) {
        b[i] = 0.0;
        for (int j = 0; j < n; j++)
            b[i] += A[i][j] * x_true[j];
    }
}

double** allocMatrix(int n) {
    double** m = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
        m[i] = (double*)malloc(n * sizeof(double));
    return m;
}

void freeMatrix(double** m, int n) {
    for (int i = 0; i < n; i++)
        free(m[i]);
    free(m);
}

SolverResult* callRPCServer(CLIENT* clnt, double** A, double* b, int n, int mode) {
    MatrixData data;
    data.n = n;
    data.mode = mode;
    
    int total = n * n + n;
    data.matrix.matrix_len = total;
    data.matrix.matrix_val = (double*)malloc(total * sizeof(double));
    
    // Spłaszcz macierz A i wektor b
    int idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            data.matrix.matrix_val[idx++] = A[i][j];
        }
    }
    
    for (int i = 0; i < n; i++) {
        data.matrix.matrix_val[idx++] = b[i];
    }
    
    SolverResult* result = solve_linear_system_1(&data, clnt);
    
    free(data.matrix.matrix_val);
    
    return result;
}

void runTest(CLIENT* clnt, int n, double* t_seq, int* it_seq,
             double* t_par, int* it_par) {
    double** A = allocMatrix(n);
    double* b = (double*)malloc(n * sizeof(double));
    double* x_true = (double*)malloc(n * sizeof(double));

    generateSymmetricPositiveDefiniteMatrix(A, b, x_true, n);

    // Test sekwencyjny
    SolverResult* rs = callRPCServer(clnt, A, b, n, 0);
    if (rs == NULL) {
        fprintf(stderr, "Błąd RPC dla n=%d (sekwencyjny)\n", n);
        freeMatrix(A, n);
        free(b);
        free(x_true);
        return;
    }
    
    *t_seq = rs->time_ms;
    *it_seq = rs->iterations;

    // Test równoległy
    SolverResult* rp = callRPCServer(clnt, A, b, n, 1);
    if (rp == NULL) {
        fprintf(stderr, "Błąd RPC dla n=%d (równoległy)\n", n);
        freeMatrix(A, n);
        free(b);
        free(x_true);
        return;
    }
    
    *t_par = rp->time_ms;
    *it_par = rp->iterations;

    freeMatrix(A, n);
    free(b);
    free(x_true);
}

int main(int argc, char* argv[]) {
    char* host;

    if (argc < 2) {
        printf("Użycie: %s <host>\n", argv[0]);
        exit(1);
    }

    host = argv[1];

    CLIENT* clnt = clnt_create(host, CHEBYSHEV_SOLVER, CHEBYSHEV_VERS, "tcp");
    if (clnt == NULL) {
        clnt_pcreateerror(host);
        exit(1);
    }

    printf("Połączono z serwerem RPC na %s\n", host);

    int sizes[] = {10, 100, 1000, 10000};
    int num_sizes = 4;

    printf("\n=== TESTY WYDAJNOŚCIOWE ===\n");
    printf("\n    n | czas_sekw | iter_sekw | czas_row | iter_row | przyspieszenie\n");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        double ts, tp;
        int is, ip;
        
        runTest(clnt, n, &ts, &is, &tp, &ip);

        double speedup = (tp > 0) ? ts / tp : 0.0;

        printf("%5d | %10.2f | %10d | %9.2f | %8d | %14.3fx\n",
               n, ts, is, tp, ip, speedup);
    }

    printf("\n");

    clnt_destroy(clnt);
    return 0;
}