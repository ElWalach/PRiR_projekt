#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <cmath>
#include <random>
#include <chrono>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "ws2_32.lib")

using namespace std;

// ===== Pomocnicze funkcje =====
vector<vector<double>> genTridiagonalSPD(int N) {
    vector<vector<double>> A(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++) {
        A[i][i] = 2.0;
        if (i > 0) A[i][i - 1] = -1.0;
        if (i < N - 1) A[i][i + 1] = -1.0;
    }
    return A;
}

vector<double> genRHS(int N) {
    vector<double> b(N);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N; i++) b[i] = dis(gen);
    return b;
}

pair<double, double> estimateEigenBounds(const vector<vector<double>>& A) {
    int N = A.size();
    double minv = numeric_limits<double>::infinity();
    double maxv = -numeric_limits<double>::infinity();
    for (int i = 0; i < N; i++) {
        double d = A[i][i];
        double off = 0;
        for (int j = 0; j < N; j++) if (i != j) off += abs(A[i][j]);
        minv = min(minv, d - off);
        maxv = max(maxv, d + off);
    }
    if (minv <= 0) minv = 1e-6;
    return {minv, maxv};
}

// ===== Sekwencyjna metoda Czebyszewa =====
vector<double> solveChebyshevSeq(const vector<vector<double>>& A, const vector<double>& b, int maxIter, double tol) {
    int N = b.size();
    vector<double> x(N, 0.0), r = b, p = r, Ap(N, 0.0);
    auto [lambdaMin, lambdaMax] = estimateEigenBounds(A);
    double theta = (lambdaMax + lambdaMin) / 2.0;
    double delta = (lambdaMax - lambdaMin) / 2.0;
    double alphaPrev = 1.0 / theta;

    for (int iter = 0; iter < maxIter; iter++) {
        for (int i = 0; i < N; i++) {
            double sum = 0;
            for (int j = 0; j < N; j++) sum += A[i][j] * p[j];
            Ap[i] = sum;
        }

        double alpha = (iter == 0)
            ? 1.0 / theta
            : 1.0 / (theta - (alphaPrev * alphaPrev * delta * delta) / 4.0);

        for (int i = 0; i < N; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double norm = 0;
        for (auto v : r) norm += v * v;
        norm = sqrt(norm);
        if (norm < tol) break;

        double beta = (alphaPrev * alphaPrev * delta * delta) / 4.0;
        for (int i = 0; i < N; i++) p[i] = r[i] + beta * p[i];
        alphaPrev = alpha;
    }
    return x;
}

// ===== Równoległa metoda Czebyszewa =====
vector<double> solveChebyshevParallel(const vector<vector<double>>& A, const vector<double>& b,
                                       int nThreads, int maxIter, double tol) {
    int N = b.size();
    vector<double> x(N, 0.0), r = b, p = r, Ap(N, 0.0);
    auto [lambdaMin, lambdaMax] = estimateEigenBounds(A);
    double theta = (lambdaMax + lambdaMin) / 2.0;
    double delta = (lambdaMax - lambdaMin) / 2.0;
    double alphaPrev = 1.0 / theta;

    int chunk = (N + nThreads - 1) / nThreads;

    for (int iter = 0; iter < maxIter; iter++) {
        vector<future<void>> tasks;
        for (int t = 0; t < nThreads; t++) {
            int start = t * chunk;
            int end = min(N, (t + 1) * chunk);
            if (start >= end) break;
            tasks.push_back(async(launch::async, [&, start, end]() {
                for (int i = start; i < end; i++) {
                    double sum = 0;
                    for (int j = 0; j < N; j++) sum += A[i][j] * p[j];
                    Ap[i] = sum;
                }
            }));
        }
        for (auto& t : tasks) t.get();

        double alpha = (iter == 0)
            ? 1.0 / theta
            : 1.0 / (theta - (alphaPrev * alphaPrev * delta * delta) / 4.0);

        for (int i = 0; i < N; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double norm = 0;
        for (auto v : r) norm += v * v;
        norm = sqrt(norm);
        if (norm < tol) break;

        double beta = (alphaPrev * alphaPrev * delta * delta) / 4.0;
        for (int i = 0; i < N; i++) p[i] = r[i] + beta * p[i];
        alphaPrev = alpha;
    }

    return x;
}

// ======= Serwer główny =======
int main() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    int port = 12345;
    int nThreads = 8;
    int maxIter = 10000;
    double tol = 1e-6;

    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port);

    bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr));
    listen(serverSocket, 5);

    cout << "Serwer uruchomiony. Oczekiwanie na polaczenia..." << endl;

    while (true) {
        sockaddr_in clientAddr{};
        int clientSize = sizeof(clientAddr);
        SOCKET clientSocket = accept(serverSocket, (sockaddr*)&clientAddr, &clientSize);

        cout << "Polaczono z klientem." << endl;

        thread([clientSocket, nThreads, maxIter, tol]() {
            int N;
            recv(clientSocket, (char*)&N, sizeof(int), 0);
            cout << "Odebrano ządanie N=" << N << endl;

            auto A = genTridiagonalSPD(N);
            auto b = genRHS(N);

            auto start = chrono::high_resolution_clock::now();
            auto xPar = solveChebyshevParallel(A, b, nThreads, maxIter, tol);
            auto end = chrono::high_resolution_clock::now();
            double czasPar = chrono::duration<double>(end - start).count();

            start = chrono::high_resolution_clock::now();
            auto xSeq = solveChebyshevSeq(A, b, maxIter, tol);
            end = chrono::high_resolution_clock::now();
            double czasSeq = chrono::duration<double>(end - start).count();

            send(clientSocket, (char*)&czasPar, sizeof(double), 0);
            send(clientSocket, (char*)&czasSeq, sizeof(double), 0);

            int len = xPar.size();
            send(clientSocket, (char*)&len, sizeof(int), 0);
            send(clientSocket, (char*)xPar.data(), len * sizeof(double), 0);

            closesocket(clientSocket);
            cout << "Wyniki wyslane do klienta." << endl;
        }).detach();
    }

    closesocket(serverSocket);
    WSACleanup();
    return 0;
}
