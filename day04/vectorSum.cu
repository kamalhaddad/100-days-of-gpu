#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

__global__ void vectorSum(const float* A, float* C, const int N) {
    // Shared memory 
    extern __shared__ int sharedMemory[];
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if(idx < N) {
        sharedMemory[tid] = A[idx];
    } else {
        sharedMemory[tid] = 0;
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if(tid  < stride) {
            sharedMemory[tid] += sharedMemory[tid + stride];
            __syncthreads();    
        }
    }

    if(tid == 0) {
        C[blockIdx.x] = sharedMemory[tid];
    }

}

int main() {
    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int N = 1024;
        
    float* A, *C;

    A = new float[N ];
    C = new float[N];

    for(int i = 0; i < N; ++i) {
        C[i] = 0.0f;
        A[i] = rand() % 100;
        C[i] = 0.0f;
    }

    float* d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N  * sizeof(float));
    cudaMalloc(&d_C, N  * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;    
    
    cudaEventRecord(start); // Start recording
    
    {
        vectorSum<<<gridSize, blockSize, blockSize>>>(d_A, d_C, N);

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
    
    float sum = 0.0f;
    for(int i = 0; i < N; ++i) {
        sum += C[i];
    }
    printf("GPU Total sum: %.2f\n", sum);

    cudaFree(d_A);
    cudaFree(d_C);


    auto startCPU = std::chrono::high_resolution_clock::now();
    sum = 0.0f;
    for(int i = 0; i < N; ++i) {
        sum += A[i]; 
    }
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuDuration = endCPU - startCPU;

    std::cout << "CPU Execution Time: " << cpuDuration.count() * 1000. << " ms\n";
    std::cout << "CPU Total Sum: " << sum << "\n";

    delete[] A;
    A = nullptr;
    delete[] C;
    C = nullptr;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}