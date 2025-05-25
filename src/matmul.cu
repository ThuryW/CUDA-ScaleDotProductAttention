#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <iostream>

#include "matmul.cuh"

// Encapsulated matrix multiplication function for multiple attention heads
void batchedMatmul(const std::vector<float*>& d_Q, const std::vector<float*>& d_KT, 
                   std::vector<float*>& d_QK, int num_heads, int N, int M, int d_k, 
                   int block_size, cublasHandle_t handle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < num_heads; i++) {
        // Q: N x d_k, K^T: d_k x M, QK: N x M
        cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                            M, N, d_k, 
                                            &alpha, d_KT[i], M, 
                                            d_Q[i], d_k, 
                                            &beta, d_QK[i], M);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS error in batchedMatmul for head " << i << std::endl;
            exit(1);
        }
    }
}

// Scaling kernel function
__global__ void scaleKernel(float *matrix, int N, int M, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        int index = row * M + col;
        matrix[index] /= scale;
    }
}