#include <stdio.h>
#include <cuda_runtime.h>
#define CHECK(call) \
{ \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA error at %s:%d: %s\n", \
__FILE__, __LINE__, cudaGetErrorString(err)); \
return 1; \
} \
}
int main() {
int device = 0;
CHECK(cudaGetDevice(&device));
cudaDeviceProp devProp;
CHECK(cudaGetDeviceProperties(&devProp, device));
printf("Device Name: %s\n", devProp.name);
printf("Total Global Memory: %lu bytes\n", devProp.totalGlobalMem);
}

