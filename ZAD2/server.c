#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include "chebyshev.h"

// Dodatkowe deklaracje dla starszych systemów
extern int mkstemp(char *template);
extern int ftruncate(int fd, off_t length);

#define MAX_ITERATIONS 10000
#define TOLERANCE 1e-6
#define NUM_PROCESSES 4

// Pomocnicze funkcje alokacji
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

double* allocVector(int n) {
    return (double*)malloc(n * sizeof(double));
}

// Mnożenie macierz-wektor
void matrixVectorMultiply(double** A, double* x, double* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = 0.0;
        for (int j = 0; j < n; j++)
            result[i] += A[i][j] * x[j];
    }
}

// Równoległe mnożenie macierz-wektor
void matrixVectorMultiplyParallel(double** A, double* x, double* result, int n, int num_procs) {
    if (n < 200 || num_procs <= 1) {
        matrixVectorMultiply(A, x, result, n);
        return;
    }

    char tmpfile[] = "/tmp/matrix_result_XXXXXX";
    int fd = mkstemp(tmpfile);
    if (fd == -1) {
        matrixVectorMultiply(A, x, result, n);
        return;
    }

    ftruncate(fd, n * sizeof(double));
    double* shared = (double*)mmap(NULL, n * sizeof(double),
                                   PROT_READ | PROT_WRITE,
                                   MAP_SHARED, fd, 0);

    int rows_per = n / num_procs;

    for (int p = 0; p < num_procs; p++) {
        pid_t pid = fork();
        if (pid == 0) {
            int start = p * rows_per;
            int end = (p == num_procs - 1) ? n : (p + 1) * rows_per;

            for (int i = start; i < end; i++) {
                double s = 0.0;
                for (int j = 0; j < n; j++)
                    s += A[i][j] * x[j];
                shared[i] = s;
            }
            exit(0);
        }
    }

    for (int p = 0; p < num_procs; p++)
        wait(NULL);

    for (int i = 0; i < n; i++)
        result[i] = shared[i];

    munmap(shared, n * sizeof(double));
    close(fd);
    unlink(tmpfile);
}

double dotProduct(double* a, double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

double vectorNorm(double* v, int n) {
    return sqrt(dotProduct(v, v, n));
}

void estimateEigenvalues(double** A, int n, int parallel, 
                        double* lambda_min, double* lambda_max) {
    double* v = allocVector(n);
    double* Av = allocVector(n);
    
    // Inicjalizacja
    for (int i = 0; i < n; i++)
        v[i] = 1.0 / sqrt(n);

    *lambda_max = 0.0;
    
    // Metoda potęgowa
    for (int iter = 0; iter < 50; iter++) {
        if (parallel)
            matrixVectorMultiplyParallel(A, v, Av, n, NUM_PROCESSES);
        else
            matrixVectorMultiply(A, v, Av, n);

        double norm = vectorNorm(Av, n);
        if (norm < 1e-12) break;

        for (int i = 0; i < n; i++)
            v[i] = Av[i] / norm;

        *lambda_max = norm;
    }

    // Oszacowanie lambda_min (kryterium Gershgorina)
    *lambda_min = 1e10;
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++)
            if (i != j)
                row_sum += fabs(A[i][j]);

        double estimate = fabs(A[i][i]) - row_sum;
        if (estimate < *lambda_min)
            *lambda_min = estimate;
    }

    if (*lambda_min <= 0)
        *lambda_min = *lambda_max * 0.01;

    free(v);
    free(Av);
}

void solveChebyshev(double** A, double* b, int n, int parallel,
                   double* x_out, double* time_ms, int* iterations) {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Estymacja wartości własnych
    double lambda_min, lambda_max;
    estimateEigenvalues(A, n, parallel, &lambda_min, &lambda_max);

    double d = (lambda_max + lambda_min) * 0.5;
    double c = (lambda_max - lambda_min) * 0.5;

    // Inicjalizacja
    double* x = allocVector(n);
    double* r = allocVector(n);
    double* p = allocVector(n);
    double* Ax = allocVector(n);
    double* Ap = allocVector(n);

    for (int i = 0; i < n; i++)
        x[i] = 0.0;

    // Oblicz początkowe residuum
    if (parallel)
        matrixVectorMultiplyParallel(A, x, Ax, n, NUM_PROCESSES);
    else
        matrixVectorMultiply(A, x, Ax, n);

    for (int i = 0; i < n; i++)
        r[i] = b[i] - Ax[i];

    double alpha = 1.0 / d;
    double beta = 0.0;

    for (int i = 0; i < n; i++)
        p[i] = r[i];

    int iter;
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        if (vectorNorm(r, n) < TOLERANCE)
            break;

        // x = x + alpha * p
        for (int i = 0; i < n; i++)
            x[i] += alpha * p[i];

        // Ap = A * p
        if (parallel)
            matrixVectorMultiplyParallel(A, p, Ap, n, NUM_PROCESSES);
        else
            matrixVectorMultiply(A, p, Ap, n);

        // r = r - alpha * Ap
        for (int i = 0; i < n; i++)
            r[i] -= alpha * Ap[i];

        // Aktualizacja współczynników Czebyszewa
        double beta_new = pow(c / (2.0 * d), 2) * (1.0 - beta);
        double alpha_new = 1.0 / (d - beta_new * d);

        // p = r + beta_new * p
        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta_new * p[i];

        alpha = alpha_new;
        beta = beta_new;
    }

    gettimeofday(&end, NULL);
    *time_ms = (end.tv_sec - start.tv_sec) * 1000.0 + 
               (end.tv_usec - start.tv_usec) / 1000.0;
    *iterations = iter;

    for (int i = 0; i < n; i++)
        x_out[i] = x[i];

    free(x);
    free(r);
    free(p);
    free(Ax);
    free(Ap);
}

void runAutotest() {
    printf("=== AUTOTEST 2x2 ===\n");
    printf("Rozwiązywanie układu:\n");
    printf("[4 1] [x1] = [6]\n");
    printf("[1 3] [x2] = [7]\n\n");

    double** A = allocMatrix(2);
    A[0][0] = 4; A[0][1] = 1;
    A[1][0] = 1; A[1][1] = 3;

    double b[2] = {6, 7};
    double x[2];
    double time_ms;
    int iterations;

    solveChebyshev(A, b, 2, 0, x, &time_ms, &iterations);

    printf("Otrzymany wynik: x = (%f, %f)\n", x[0], x[1]);
    printf("=====================\n\n");

    freeMatrix(A, 2);
}

SolverResult* solve_linear_system_1_svc(MatrixData* arg, struct svc_req* rqstp) {
    static SolverResult result;
    
    int n = arg->n;
    int mode = arg->mode;
    
    printf("Otrzymano zadanie: n=%d, mode=%s\n", 
           n, mode ? "równoległy" : "sekwencyjny");
    
    // Rekonstrukcja macierzy A i wektora b
    double** A = allocMatrix(n);
    double* b = allocVector(n);
    
    int idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = arg->matrix.matrix_val[idx++];
        }
    }
    
    for (int i = 0; i < n; i++) {
        b[i] = arg->matrix.matrix_val[idx++];
    }
    
    // Rozwiązanie układu
    double* solution = allocVector(n);
    double time_ms;
    int iterations;
    
    solveChebyshev(A, b, n, mode, solution, &time_ms, &iterations);
    
    // Przygotowanie wyniku
    result.n = n;
    result.time_ms = time_ms;
    result.iterations = iterations;
    
    result.solution.solution_len = n;
    result.solution.solution_val = (double*)malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        result.solution.solution_val[i] = solution[i];
    }
    
    printf("Rozwiązano w %.2f ms, iteracji: %d\n", time_ms, iterations);
    
    freeMatrix(A, n);
    free(b);
    free(solution);
    
    return &result;
}

// Ta funkcja zostanie wywołana przed startem serwera RPC
// (rpcgen generuje main() w chebyshev_svc.c który wywołuje nasze procedury)
void __attribute__((constructor)) init_server(void) {
    runAutotest();
    printf("Serwer RPC uruchomiony...\n");
}