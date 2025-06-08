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

/**
 * @brief 使用 cuBLAS 执行批量的 Attention * V 矩阵乘法.
 * * @param d_attention_weights 设备端存储的注意力权重矩阵指针向量 (N, M)
 * @param d_V 设备端存储的 V 矩阵指针向量 (M, d_v)
 * @param d_O 设备端存储的输出矩阵指针向量 (N, d_v)
 * @param num_heads 注意力头的数量
 * @param N Q 矩阵的行数 (序列长度)
 * @param M K/V 矩阵的行数 (序列长度)
 * @param d_v V 矩阵的列数 (值的维度)
 * @param handle cuBLAS 句柄
 */
void batchedAttentionVMatmul(
    const std::vector<float*>& d_attention_weights,
    const std::vector<float*>& d_V,
    std::vector<float*>& d_O,
    int num_heads,
    int N,
    int M,
    int d_v,
    cublasHandle_t handle) 
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 对每个头执行矩阵乘法: O(N, d_v) = Attention(N, M) * V(M, d_v)
    for (int i = 0; i < num_heads; i++) {
        // cuBLAS 默认使用列主序存储，而C/C++使用行主序。
        // 为了在行主序数据上执行 C = A * B，我们可以利用 (A*B)^T = B^T * A^T 的思想，
        // 将其视为列主序中的 C_col(d_v, N) = V_col(d_v, M) * Attention_col(M, N)。
        // 因此，传递给 cublasSgemm 的矩阵顺序是 V, Attention, O。
        cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_N, // V 矩阵不转置
            CUBLAS_OP_N, // Attention 矩阵不转置
            d_v,         // m: O 和 V 的行数 (在列主序视角下)
            N,           // n: O 和 Attention 的列数 (在列主序视角下)
            M,           // k: V 的列数和 Attention 的行数 (在列主序视角下)
            &alpha,
            d_V[i], d_v, // 矩阵 V (B_col) 及其 leading dimension
            d_attention_weights[i], M, // 矩阵 Attention (A_col) 及其 leading dimension
            &beta,
            d_O[i], d_v  // 矩阵 O (C_col) 及其 leading dimension
        );
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS sgemm for Attention*V failed for head " << i << std::endl;
        }
    }
}