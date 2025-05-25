#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>

#include "matmul.cuh"
#include "inout.cuh"
#include "utils.cuh"

// Main function
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "用法: ./main <mode>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    if (mode != "random_test") {
        std::cerr << "无效模式，目前仅支持 'random_test'" << std::endl;
        return 1;
    }


    // Parameters for BERT-base
    int num_heads = 12;  // Number of attention heads
    int N = 128;         // Rows of Q and QK
    int M = 128;         // Columns of K^T and QK
    int d_k = 64;        // Columns of Q and rows of K^T
    int block_size = 64; // Block size for matrix multiplication

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory for Q, K, K^T, and QK for all heads
    std::vector<float*> d_Q(num_heads);
    std::vector<float*> d_K(num_heads);
    std::vector<float*> d_KT(num_heads);
    std::vector<float*> d_QK(num_heads);
    for (int i = 0; i < num_heads; i++) {
        cudaMalloc(&d_Q[i], N * d_k * sizeof(float));    // Q: 128x64
        cudaMalloc(&d_K[i], M * d_k * sizeof(float));    // K: 128x64
        cudaMalloc(&d_KT[i], d_k * M * sizeof(float));   // K^T: 64x128
        cudaMalloc(&d_QK[i], N * M * sizeof(float));     // QK: 128x128
    }

    // Load Q and K from existing files
    for (int i = 0; i < num_heads; i++) {
        float* h_Q = new float[N * d_k];
        float* h_K = new float[M * d_k];
        
        std::string Q_file = "../data/random_test/Q_head_" + std::to_string(i) + ".txt";
        std::string K_file = "../data/random_test/K_head_" + std::to_string(i) + ".txt";
        
        loadMatrix(Q_file, h_Q, N, d_k);
        loadMatrix(K_file, h_K, M, d_k);

        // Save Q and K to random_verify
        std::string Q_save_file = "../data/random_verify/Q_head_" + std::to_string(i) + ".txt";
        std::string K_save_file = "../data/random_verify/K_head_" + std::to_string(i) + ".txt";
        saveMatrix(Q_save_file, h_Q, N, d_k);
        saveMatrix(K_save_file, h_K, M, d_k);

        std::cout << "Q[head_" << i << "]: " << N << "x" << d_k << std::endl;
        std::cout << "K[head_" << i << "]: " << M << "x" << d_k << std::endl;
        
        cudaMemcpy(d_Q[i], h_Q, N * d_k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K[i], h_K, M * d_k * sizeof(float), cudaMemcpyHostToDevice);
        
        delete[] h_Q;
        delete[] h_K;
    }

    // Transpose K to K^T for all heads and save K^T
    batchedTranspose(d_K, d_KT, num_heads, M, d_k, handle);
    for (int i = 0; i < num_heads; i++) {
        float* h_KT = new float[d_k * M];
        cudaMemcpy(h_KT, d_KT[i], d_k * M * sizeof(float), cudaMemcpyDeviceToHost);
        std::string KT_save_file = "../data/random_verify/KT_head_" + std::to_string(i) + ".txt";
        saveMatrix(KT_save_file, h_KT, d_k, M);
        std::cout << "K^T[head_" << i << "]: " << d_k << "x" << M << std::endl;
        delete[] h_KT;
    }

    // Compute QK^T for all heads and save unscaled QK^T
    batchedMatmul(d_Q, d_KT, d_QK, num_heads, N, M, d_k, block_size, handle);
    for (int i = 0; i < num_heads; i++) {
        float* h_QK = new float[N * M];
        cudaMemcpy(h_QK, d_QK[i], N * M * sizeof(float), cudaMemcpyDeviceToHost);
        std::string QK_save_file = "../data/random_verify/QK_head_" + std::to_string(i) + ".txt";
        saveMatrix(QK_save_file, h_QK, N, M);
        delete[] h_QK;
    }

    // Scale QK^T by 1/sqrt(d_k) for each head
    float scale = sqrtf(static_cast<float>(d_k));  // Scaling factor

    // Define thread block and grid dimensions
    const int THREADS_PER_BLOCK = 16;
    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 gridDim((M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 
                 (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Apply scaling to each head's QK^T matrix
    for (int i = 0; i < num_heads; i++) {
        scaleKernel<<<gridDim, blockDim>>>(d_QK[i], N, M, scale);
        cudaDeviceSynchronize();  // Ensure kernel completion
    }

    // Save the result
    for (int i = 0; i < num_heads; i++) {
        float* h_QK = new float[N * M];
        cudaMemcpy(h_QK, d_QK[i], N * M * sizeof(float), cudaMemcpyDeviceToHost);
        std::string result_file = "../data/random_verify/QK_scaled_head_" + std::to_string(i) + ".txt";
        saveMatrix(result_file, h_QK, N, M);
        std::cout << "QK_scaled[head_" << i << "]: " << N << "x" << M << std::endl;
        delete[] h_QK;
    }

    // Cleanup
    for (int i = 0; i < num_heads; i++) {
        cudaFree(d_Q[i]);
        cudaFree(d_K[i]);
        cudaFree(d_KT[i]);
        cudaFree(d_QK[i]);
    }
    cublasDestroy(handle);

    return 0;
}