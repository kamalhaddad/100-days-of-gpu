#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] + B[idx];
        printf("Index %d: A = %.2f, B = %.2f, C = %.2f\n", idx, A[idx], B[idx], C[idx]);
    }
    
}

int main() {
    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Start recording

    // Contain in anonymous scope
    {
        const int N = 1024;
        float A[N], B[N], C[N];
        for(int i = 0; i < N; ++i) {
            A[i] = rand() % 100;
            B[i] = rand() % 100;
            C[i] = 0;
            // printf("Constructing vectors: A[%d]=%.2f, B[%d]=%.2f\n", i, A[i], i, B[i]);
        }

        float* d_A, *d_B, *d_C;

        cudaMalloc(&d_A, sizeof(float) * N);
        cudaMalloc(&d_B, sizeof(float) * N);
        cudaMalloc(&d_C, sizeof(float) * N);
        cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);

        const int numberOfThreadsPerBlock = N/4;
        const int numberOfBlocks = ceil(N/numberOfThreadsPerBlock);
        
        vecAdd<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        
        cudaMemcpy(C,d_C, N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    cudaEventRecord(stop); // Stop recording
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}