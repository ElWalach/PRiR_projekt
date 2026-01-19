#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int number;
    
    if (world_rank == 0) {
        
        number = 42;
        cout << world_rank << ": Starting with number " << number << endl;
        
        
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        cout << world_rank << ": Sent " << number << " to process 1" << endl;
        
        
        MPI_Recv(&number, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << world_rank << ": Received " << number << " from process " 
                 << world_size - 1 << " (ring completed)" << endl;
    } 
    else {
        
        MPI_Recv(&number, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << world_rank << ": Received " << number << " from process " 
                  << world_rank - 1 << endl;
        
        
        number += world_rank;
        
        
        int next_process = (world_rank + 1) % world_size;
        MPI_Send(&number, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD);
        std::cout << world_rank << ": Sent " << number << " to process " 
                  << next_process << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}