#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <vector>
#include <string>
#include <iomanip> // Do formatowania wyjścia (setw)
#include <chrono>  // Do mierzenia czasu

#pragma comment(lib, "ws2_32.lib")

using namespace std;

const int PORT = 8080;
const char* SERVER_IP = "127.0.0.1";

/**
 * @brief Wysyła zadanie, odbiera odpowiedź i PARSUJE z niej czas wykonania.
 * @param N Rozmiar problemu.
 * @return Czas wykonania zadania w [ms] lub -1 w przypadku błędu.
 */
long long sendRequestAndGetTime(int N) {
    long long executionTime = -1;

    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return -1;

    SOCKET clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == INVALID_SOCKET) { WSACleanup(); return -1; }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, SERVER_IP, &serverAddr.sin_addr);

    if (connect(clientSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == 0) {
        string msg = to_string(N);
        send(clientSocket, msg.c_str(), (int)msg.length(), 0);

        char buffer[4096];
        int bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0);
        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            string response(buffer);
            try {
                size_t czas_pos = response.find("Czas=") + 5;
                size_t ms_pos = response.find("ms");
                if (czas_pos != string::npos && ms_pos != string::npos) {
                    executionTime = stoll(response.substr(czas_pos, ms_pos - czas_pos));
                }
            } catch (...) { /* Błąd parsowania, executionTime pozostanie -1 */ }
        }
    }
    closesocket(clientSocket);
    WSACleanup();
    return executionTime;
}

// Funkcja dla procesów potomnych - tabelka
void childProcessTask(int N) {
    long long time = sendRequestAndGetTime(N);
    if (time >= 0) {
        cout << left << setw(12) << N << time << endl;
    }
}

// Uruchamia test w trybie sekwencyjnym
void runSequential(const vector<int>& sizes) {
    cout << "=== Test w trybie SEKWENCYJNYM ===" << endl;
    cout << left << setw(12) << "Rozmiar N" << "Czas [ms]" << endl;
    cout << string(30, '-') << endl;

    auto totalStart = chrono::high_resolution_clock::now();
    long long sumOfTimes = 0;

    for (int n_size : sizes) {
        long long time = sendRequestAndGetTime(n_size);
        if (time >= 0) {
            cout << left << setw(12) << n_size << time << endl;
            sumOfTimes += time;
        }
    }

    auto totalEnd = chrono::high_resolution_clock::now();
    auto totalDuration = chrono::duration_cast<chrono::milliseconds>(totalEnd - totalStart).count();

    cout << string(30, '-') << endl;
    cout << left << setw(25) << "Suma czasow obliczen:" << sumOfTimes << " ms" << endl;
    cout << left << setw(25) << "Calkowity czas wykonania:" << totalDuration << " ms" << endl;
    cout << string(40, '=') << endl;
}

// Uruchamia test w trybie współbieżnym
void runConcurrent(const vector<int>& sizes) {
    cout << "\n=== Test w trybie WSPOLBIEZNYM ===" << endl;
    cout << left << setw(12) << "Rozmiar N" << "Czas [ms]" << endl;
    cout << string(30, '-') << endl;

    auto totalStart = chrono::high_resolution_clock::now();
    vector<HANDLE> runningProcesses;

    for (int n_size : sizes) {
        STARTUPINFOA si = {sizeof(si)};
        PROCESS_INFORMATION pi;
        string commandLine = "klient.exe " + to_string(n_size);

        if (CreateProcessA(NULL, &commandLine[0], NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
            runningProcesses.push_back(pi.hProcess);
            CloseHandle(pi.hThread);
        }
    }

    for (HANDLE procHandle : runningProcesses) {
        WaitForSingleObject(procHandle, INFINITE);
        CloseHandle(procHandle);
    }

    auto totalEnd = chrono::high_resolution_clock::now();
    auto totalDuration = chrono::duration_cast<chrono::milliseconds>(totalEnd - totalStart).count();

    cout << string(30, '-') << endl;
    cout << left << setw(25) << "Calkowity czas wykonania:" << totalDuration << " ms" << endl;
    cout << string(40, '=') << endl;
}

int main(int argc, char* argv[]) {
   
    if (argc > 1) {
        childProcessTask(atoi(argv[1]));
        return 0;
    }

    
    vector<int> sizes = {10, 100, 1000, 10000};
    
    
    runSequential(sizes);
    runConcurrent(sizes);



    cout << "\nNacisnij Enter, aby zakonczyc...";
    cin.get();
    return 0;
}