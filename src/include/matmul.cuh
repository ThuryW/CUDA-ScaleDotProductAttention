// matmul.cuh
#ifndef MATMUL_CUH
#define MATMUL_CUH

#include <vector>
#include <cublas_v2.h>

// Encapsulated matrix multiplication function for multiple attention heads
void batchedMatmul(const std::vector<float*>& d_Q, const std::vector<float*>& d_KT, 
                   std::vector<float*>& d_QK, int num_heads, int N, int M, int d_k, 
                   int block_size, cublasHandle_t handle);

// Scaling kernel function
__global__ void scaleKernel(float *matrix, int N, int M, float scale);

void batchedAttentionVMatmul(
    const std::vector<float*>& d_attention_weights,
    const std::vector<float*>& d_V,
    std::vector<float*>& d_O,
    int num_heads,
    int N,
    int M,
    int d_v,
    cublasHandle_t handle);

#endif // MATMUL_CUH