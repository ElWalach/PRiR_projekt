#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    const int total_size = 10000;
    const int chunk_size = total_size / world_size;
    
    std::vector<int> array;
    int expected_sum = 0;
    
    if (world_rank == 0) {
        array.resize(total_size);
        for (int i = 0; i < total_size; ++i) {
            array[i] = i + 1;  
        }
        
        expected_sum = std::accumulate(array.begin(), array.end(), 0);
        std::cout << world_rank << ": Created array of size " << total_size 
                  << ", expected sum: " << expected_sum << std::endl;
    }
    
    std::vector<int> local_chunk(chunk_size);
    
    MPI_Scatter(array.data(), chunk_size, MPI_INT,
                local_chunk.data(), chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);
    
    std::cout << world_rank << ": Received " << chunk_size << " elements" << std::endl;
    
    int local_sum = std::accumulate(local_chunk.begin(), local_chunk.end(), 0);
    std::cout << world_rank << ": Local sum = " << local_sum << std::endl;
    
    std::vector<int> all_sums;
    if (world_rank == 0) {
        all_sums.resize(world_size);
    }
    
    MPI_Gather(&local_sum, 1, MPI_INT,
               all_sums.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        int total_sum = std::accumulate(all_sums.begin(), all_sums.end(), 0);
        std::cout << world_rank << ": Received sums from all processes" << std::endl;
        std::cout << world_rank << ": Total sum = " << total_sum << std::endl;
        std::cout << world_rank << ": Expected sum = " << expected_sum << std::endl;
        
        if (total_sum == expected_sum) {
            std::cout << world_rank << ": SUCCESS! Sums match!" << std::endl;
        } else {
            std::cout << world_rank << ": ERROR! Sums don't match!" << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}