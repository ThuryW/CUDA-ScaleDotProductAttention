#include "utils.cuh"

void batchedTranspose(const std::vector<float*>& d_K, std::vector<float*>& d_KT, 
                     int num_heads, int M, int d_k, cublasHandle_t handle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < num_heads; i++) {
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, d_k, 
                    &alpha, d_K[i], d_k, &beta, nullptr, M, d_KT[i], M);
    }
}