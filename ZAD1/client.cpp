#include <iostream>
#include <vector>
#include <cmath>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace chrono;

const int PORT = 8080;
const char* SERVER_IP = "127.0.0.1";

struct MatrixHeader {
    int n;
    int mode;
};

struct Result {
    int n;
    double time_ms;
    int iterations;
};


// generowanie macierzy symetrycznej dodatnio określonej i wektora prawej strony b
void generateSymmetricPositiveDefiniteMatrix(vector<vector<double>>& A, 
                                                 vector<double>& b, 
                                                 vector<double>& x_true, 
                                                 int n) {
    mt19937 gen(42);    //generator losowych liczb z seedem 42
    uniform_real_distribution<> dis(-1.0, 1.0);     //generuje liczby losowe w przedziale [-1, 1]
    

//tworzymy macierz symetryczną

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

    for (int i = 0; i < n; i++)
        x_true[i] = dis(gen) * 10.0;


    //generujemy prawdziwe rozwiązanie - istnieje i wynosi x_true

    for (int i = 0; i < n; i++) {
        b[i] = 0.0;
        for (int j = 0; j < n; j++)
            b[i] += A[i][j] * x_true[j];
    }
}


//wyslanie danych na serwer

Result sendToServer(const vector<vector<double>>& A, const vector<double>& b, 
                    int mode, vector<double>& solution) {

    int sock = socket(AF_INET, SOCK_STREAM, 0); //tworzymy socket
    if (sock < 0) return {};

    //adres i port serwera
    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
        close(sock);
        return {};
    }


    //laczymy
    if (connect(sock, (sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        close(sock);
        return {};
    }

    //wysylamy naglowek do serwera

    MatrixHeader header{(int)A.size(), mode};
    send(sock, &header, sizeof(header), 0);


    //tworzemy bufor do wysylki macierzy, splaszczamy macierz
    int n = A.size();
    int total = n * n + n;
    vector<double> buffer(total);

    int idx = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            buffer[idx++] = A[i][j];

    for (int i = 0; i < n; i++)
        buffer[idx++] = b[i];

    send(sock, buffer.data(), total * sizeof(double), 0);       //wysylamy dane na serwer

    Result r{};
    recv(sock, &r, sizeof(r), 0);       //odbieramy wyniki

    solution.resize(n);
    recv(sock, solution.data(), n * sizeof(double), 0); //odbieramy rozwiazanie ukladu

    close(sock);    //zamykamy socket, zwracamy wynik
    return r;   
}

void runTest(int n, double& t_seq, int& it_seq,
                       double& t_par, int& it_par) {
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);
    vector<double> x_true(n);

    generateSymmetricPositiveDefiniteMatrix(A, b, x_true, n);

    vector<double> xs, xp;

    Result rs = sendToServer(A, b, 0, xs);
    Result rp = sendToServer(A, b, 1, xp);

    t_seq = rs.time_ms;
    it_seq = rs.iterations;

    t_par = rp.time_ms;
    it_par = rp.iterations;
}

int main() {
    vector<int> sizes = {10, 100, 1000, 10000};

  

    cout << "\n n | czas_sekw | iter_sekw | czas_row | iter_row\n";
    cout << "----------------------------------------------------------\n";

    for (int n : sizes) {
        double ts, tp;
        int is, ip;
        runTest(n, ts, is, tp, ip);

        cout << setw(4) << n << " | "
             << setw(10) << ts << " | "
             << setw(10) << is << " | "
             << setw(10) << tp << " | "
             << setw(8)  << ip << "\n";
    }

    return 0;
}
