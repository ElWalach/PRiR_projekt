

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <algorithm>

using namespace std;
using namespace chrono;

const int PORT = 8080;
const int MAX_ITERATIONS = 10000;
const double TOLERANCE = 1e-6;
const int NUM_PROCESSES = 4;

struct MatrixHeader {
    int n;
    int mode;
};

struct Result {
    int n;
    double time_ms;
    int iterations;
};

//  MULTIPLIKACJE
vector<double> matrixVectorMultiply(const vector<vector<double>>& A, const vector<double>& x) {
    int n = A.size();
    vector<double> r(n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            r[i] += A[i][j] * x[j];
    return r;
}

vector<double> matrixVectorMultiplyParallel(const vector<vector<double>>& A,
                                           const vector<double>& x,
                                           int num_processes) {
    int n = A.size();
    if (n < 200 || num_processes <= 1)
        return matrixVectorMultiply(A, x);

    char tmpfile[] = "/tmp/matrix_result_XXXXXX";
    int fd = mkstemp(tmpfile);
    if (fd == -1)
        return matrixVectorMultiply(A, x);

    ftruncate(fd, n * sizeof(double));
    double* shared = (double*)mmap(NULL, n * sizeof(double),
                                   PROT_READ | PROT_WRITE,
                                   MAP_SHARED, fd, 0);

    int rows_per = n / num_processes;
    vector<pid_t> pids;

    for (int p = 0; p < num_processes; p++) {
        pid_t pid = fork();
        if (pid == 0) {
            int start = p * rows_per;
            int end = (p == num_processes - 1 ? n : (p + 1) * rows_per);

            for (int i = start; i < end; i++) {
                double s = 0.0;
                for (int j = 0; j < n; j++)
                    s += A[i][j] * x[j];
                shared[i] = s;
            }
            exit(0);
        } else {
            pids.push_back(pid);
        }
    }

    for (pid_t id : pids) waitpid(id, nullptr, 0);

    vector<double> r(n);
    for (int i = 0; i < n; i++) r[i] = shared[i];

    munmap(shared, n * sizeof(double));
    close(fd);
    unlink(tmpfile);

    return r;
}

double dotProduct(const vector<double>& a, const vector<double>& b) {
    double s = 0.0;
    for (int i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}

double vectorNorm(const vector<double>& v) {
    return sqrt(dotProduct(v, v));
}

pair<double, double> estimateEigenvalues(const vector<vector<double>>& A, bool parallel) {
    int n = A.size();
    vector<double> v(n, 1.0 / sqrt(n));

    double lambda_max = 0.0;
    for (int iter = 0; iter < 50; iter++) {
        vector<double> Av = parallel ?
            matrixVectorMultiplyParallel(A, v, NUM_PROCESSES) :
            matrixVectorMultiply(A, v);

        double norm = vectorNorm(Av);
        if (norm < 1e-12) break;

        for (int i = 0; i < n; i++)
            v[i] = Av[i] / norm;

        lambda_max = norm;
    }

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

Result solveChebyshev(const vector<vector<double>>& A,
                      const vector<double>& b,
                      bool parallel,
                      vector<double>& x_out)
{
    Result res{};
    int n = A.size();
    vector<double> x(n, 0.0);

    auto start = high_resolution_clock::now();

    // --- Estimate eigenvalues ---
    auto [lambda_min, lambda_max] = estimateEigenvalues(A, parallel);

    double d = (lambda_max + lambda_min) * 0.5;
    double c = (lambda_max - lambda_min) * 0.5;

    // --- Initial residual ---
    vector<double> Ax = parallel ?
        matrixVectorMultiplyParallel(A, x, NUM_PROCESSES) :
        matrixVectorMultiply(A, x);

    vector<double> r(n), p(n);
    for (int i = 0; i < n; i++)
        r[i] = b[i] - Ax[i];

    // --- Chebyshev iteration params ---
    double alpha = 1.0 / d;
    double beta = 0.0;

    p = r; // initial direction

    int iter;
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {

        if (vectorNorm(r) < TOLERANCE) break;

        // x = x + alpha * p
        for (int i = 0; i < n; i++)
            x[i] += alpha * p[i];

        // r = r - alpha * A*p
        vector<double> Ap = parallel ?
            matrixVectorMultiplyParallel(A, p, NUM_PROCESSES) :
            matrixVectorMultiply(A, p);

        for (int i = 0; i < n; i++)
            r[i] -= alpha * Ap[i];

        // Update Chebyshev coefficients
        double beta_new = pow(c / (2.0 * d), 2) * (1.0 - beta);
        double alpha_new = 1.0 / (d - beta_new * d);

        // p = r + beta_new * p
        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta_new * p[i];

        alpha = alpha_new;
        beta = beta_new;
    }

    auto end = high_resolution_clock::now();
    res.time_ms = duration_cast<milliseconds>(end - start).count();
    res.iterations = iter;
    res.n = n;

    x_out = x;
    return res;
}


void runAutotest() {
    vector<vector<double>> A = {
        {4, 1},
        {1, 3}
    };
    vector<double> b = {6, 7};
    vector<double> x;

    solveChebyshev(A, b, false, x);

    cout << "=== AUTOTEST 2x2 ===\n";
    cout << "Rozwiazywanie układu:\n";
    cout << "[4 1] [x1] = [6]\n";
    cout << "[1 3] [x2] = [7]\n\n";

    cout << "Otrzymany wynik: x = ("
         << x[0] << ", " << x[1] << ")\n";

    cout << "=====================\n\n";
}

void handleClient(int client_socket) {
    MatrixHeader header;
    if (recv(client_socket, &header, sizeof(header), 0) != sizeof(header)) {
        close(client_socket);
        return;
    }

    int n = header.n;
    int total = (n*n + n) * sizeof(double);

    double* buffer = new double[n*n + n];
    int received = 0;

    while (received < total) {
        int chunk = recv(client_socket, ((char*)buffer) + received, total - received, 0);
        if (chunk <= 0) { delete[] buffer; close(client_socket); return; }
        received += chunk;
    }

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    int idx = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = buffer[idx++];

    for (int i = 0; i < n; i++) b[i] = buffer[idx++];
    delete[] buffer;

    vector<double> sol;
    Result r = solveChebyshev(A, b, header.mode == 1, sol);

    send(client_socket, &r, sizeof(r), 0);
    send(client_socket, sol.data(), n * sizeof(double), 0);

    close(client_socket);
}

int main() {
    runAutotest();

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    bind(server_fd, (sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 5);

    cout << "Serwer nasłuchuje na porcie " << PORT << "...\n\n";

    while (true) {
        sockaddr_in caddr;
        socklen_t clen = sizeof(caddr);

        int client = accept(server_fd, (sockaddr*)&caddr, &clen);

        pid_t pid = fork();
        if (pid == 0) {
            close(server_fd);
            handleClient(client);
            exit(0);
        } else {
            close(client);
        }
    }

    return 0;
}
