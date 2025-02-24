#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

__global__ void matAdd(const float* A, const float* B, float* C, const int N, const int M) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    // printf("(%d,%d)\n", x, y);
    
    if(x < N && y < M) {
        C[x * M + y] = A[x * M + y] + B[x * M + y];
        // printf("Index [%d][%d]: A = %.2f, B = %.2f, C = %.2f\n", x, y,A[x * M + y], B[x * M + y], C[x * M + y]);
    }
    
}

void matAddCpu(const float *A, const float *B, float *C, const int N, const int M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            C[i * M + j] = A[i * M + j] + B[i * M + j];
        }
    }
}


int main() {

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int N = 1024, M = 768;
        
    float* A, *B, *C;

    A = new float[N * M];
    B = new float[N * M];
    C = new float[N * M];

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j) {
            A[i * M + j] = rand() % 100;
            B[i * M + j] = rand() % 100;
            C[i * M + j] = 0.0f;
            // printf("A[%d][%d] = %.2f, B[%d][%d] = %.2f\n", i, j, A[i * M + j], i, j, B[i * M + j]); 
        }
    }
    float* d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * M * sizeof(float));
    cudaMalloc(&d_B, N * M * sizeof(float));
    cudaMalloc(&d_C, N * M * sizeof(float));

    cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * M *sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(32, 32);
    dim3 dimBlock(N/32, M/32);

    cudaEventRecord(start); // Start recording

    // Contain in anonymous scope
    {

        matAdd<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, M);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            // Handle the error (e.g., exit)
            // Added this because kernel was silently not running and failing because of invald configuration argument meaning that I was giving block threads > max block threads for my GPU
        }
        
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop); // Stop recording
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Kernel execution time: %f ms\n", milliseconds);

    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    auto startCPU = std::chrono::high_resolution_clock::now();
    matAddCpu(A, B, C, N, M);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuDuration = endCPU - startCPU;

    std::cout << "CPU Execution Time: " << cpuDuration.count() * 1000. << " ms\n";

    delete[] A;
    A = nullptr;
    delete[] B;
    B = nullptr;
    delete[] C;
    C = nullptr;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}