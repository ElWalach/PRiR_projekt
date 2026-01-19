#include <mpi.h>
#include <iostream>
#include <cstring>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    const int message_size = 100;
    char message[message_size];
    
    if (world_rank == 0) {
        // Tylko proces 0 zna wiadomość
        strcpy(message, "Hello from root process!");
        std::cout << world_rank << ": Broadcasting message: " << message << std::endl;
    } else {
        std::cout << world_rank << ": Waiting for broadcast..." << std::endl;
    }
    
    // Wszystkie procesy wywołują Bcast
    MPI_Bcast(message, message_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if (world_rank != 0) {
        std::cout << world_rank << ": Received message: " << message << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}