#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <chrono>

#pragma comment(lib, "ws2_32.lib")

using namespace std;

const int PORT = 8080;
const double EPSILON = 1e-6;
const int MAX_ITER = 1000;


vector<double> solveChebyshev(int N, vector<vector<double>>& A, vector<double>& b) {
    vector<double> x(N, 0.0);
    vector<double> x_prev(N, 0.0);
    vector<double> x_temp(N, 0.0);
    double lambda_min = 0.1, lambda_max = 2.0;
    double d = (lambda_max + lambda_min) / 2.0;
    double c = (lambda_max - lambda_min) / 2.0;

    
    double omega = 2.0 / (lambda_max + lambda_min);
    for (int i = 0; i < N; i++) {
        double sum = b[i];
        for (int j = 0; j < N; j++) { if (i != j) sum -= A[i][j] * x[j]; }
        double residual = sum / A[i][i] - x[i];
        x_temp[i] = x[i] + omega * residual;
    }
    x_prev = x;
    x = x_temp;

    
    for (int k = 1; k < MAX_ITER; k++) {
        double mu_k = (k == 1) ? (2.0 * d / c) : (4.0 * d / c - 2.0);
        double omega_k = 4.0 / (c * mu_k);
        double rho_k = 1.0 - omega_k * c / 2.0;

        for (int i = 0; i < N; i++) {
            double sum = b[i];
            for (int j = 0; j < N; j++) { if (i != j) sum -= A[i][j] * x[j]; }
            double x_richardson = sum / A[i][i];
            x_temp[i] = omega_k * x_richardson + rho_k * x[i] + (1.0 - omega_k - rho_k) * x_prev[i];
        }

        double error = 0.0;
        for (int i = 0; i < N; i++) { error += abs(x_temp[i] - x[i]); }
        
        x_prev = x;
        x = x_temp;

        if (error < EPSILON) break;
    }
    return x;
}


void generateSystem(int N, vector<vector<double>>& A, vector<double>& b) {
    A.assign(N, vector<double>(N));
    b.resize(N);
    srand(time(NULL) + N + GetCurrentProcessId());
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                A[i][j] = (rand() % 10) + 1;
                sum += abs(A[i][j]);
            }
        }
        A[i][i] = sum + (rand() % 10) + 10;
        b[i] = (rand() % 100) + 1;
    }
}

// Funkcja obsługi klienta
DWORD WINAPI handleClient(LPVOID lpParam) {
    SOCKET clientSocket = (SOCKET)lpParam;
    char buffer[1024];
    int bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0);

    if (bytesReceived > 0) {
        buffer[bytesReceived] = '\0';
        int N = atoi(buffer);

        

        auto start = chrono::high_resolution_clock::now();
        vector<vector<double>> A;
        vector<double> b;
        generateSystem(N, A, b);
        vector<double> solution = solveChebyshev(N, A, b);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        
        stringstream ss;
        ss << "N=" << N << " Czas=" << duration.count() << "ms Rozwiazanie: ";
        for (int i = 0; i < min(5, N); i++) {
            ss << "x[" << i << "]=" << solution[i] << " ";
        }
        
        string result = ss.str();
        send(clientSocket, result.c_str(), result.length(), 0);

        
    }

    closesocket(clientSocket);
    return 0;
}

int main() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr));
    listen(serverSocket, 10);

    cout << "=== SERWER URUCHOMIONY (tryb cichy) ===" << endl;
    cout << "Nasluchuje na porcie " << PORT << "..." << endl;

    while (true) {
        SOCKET clientSocket = accept(serverSocket, NULL, NULL);
        if (clientSocket != INVALID_SOCKET) {
            CreateThread(NULL, 0, handleClient, (LPVOID)clientSocket, 0, NULL);
        }
    }

    closesocket(serverSocket);
    WSACleanup();
    return 0;
}