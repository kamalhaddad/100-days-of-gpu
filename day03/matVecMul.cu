#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

__global__ void matVecMul(const float* A, const float* B, float* C, const int N, const int M) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < N) {
        float sum = 0.0f;
        
        for(int j = 0; j < M; ++j) {
            sum += A[i * N + j] * B[j];
        }
        C[i] = sum;
        // printf("%.2f\n", sum);
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
    B = new float[M];
    C = new float[N];

    for(int i = 0; i < N; ++i) {
        C[i] = 0.0f;
        for(int j = 0; j < M; ++j) {
            A[i * M + j] = rand() % 100;
        }
    }

    for(int i = 0; i < M; ++i) {
        B[i] = rand() % 100;
    }

    float* d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * M * sizeof(float));
    cudaMalloc(&d_B, M * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M *sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;    
    
    cudaEventRecord(start); // Start recording

    // Contain in anonymous scope
    {

        matVecMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M);

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

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

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