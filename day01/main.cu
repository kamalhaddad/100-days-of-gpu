#include <stdio.h>

// CUDA kernel function
__global__ void hello_cuda() {
    printf("Hello from CUDA Kernel!\n");
}

int main() {
    hello_cuda<<<1, 1>>>(); // Launch CUDA kernel
    cudaDeviceSynchronize(); // Wait for CUDA to finish
    return 0;
}