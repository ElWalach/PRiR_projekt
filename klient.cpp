#include <iostream>
#include <vector>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "ws2_32.lib")

using namespace std;

int main() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    string host = "127.0.0.1";
    int port = 12345;

    cout << "Podaj rozmiar macierzy N: ";
    int N;
    cin >> N;

    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in servAddr{};
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &servAddr.sin_addr);

    if (connect(sock, (sockaddr*)&servAddr, sizeof(servAddr)) == SOCKET_ERROR) {
        cerr << "Blad polaczenia z serwerem.\n";
        WSACleanup();
        return 1;
    }

    send(sock, (char*)&N, sizeof(int), 0);

    double czasPar, czasSeq;
    recv(sock, (char*)&czasPar, sizeof(double), 0);
    recv(sock, (char*)&czasSeq, sizeof(double), 0);

    int len;
    recv(sock, (char*)&len, sizeof(int), 0);
    vector<double> xPar(len);
    recv(sock, (char*)xPar.data(), len * sizeof(double), 0);

    cout << "Czas rownoległy: " << czasPar << " s\n";
    cout << "Czas sekwencyjny: " << czasSeq << " s\n";
    cout << "Przykladowe rozwiazanie (rownolegle): ";
    for (int i = 0; i < min(10, len); i++) cout << xPar[i] << " ";
    cout << endl;

    closesocket(sock);
    WSACleanup();
    return 0;
}
