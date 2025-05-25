#ifndef UTILS_CUH
#define UTILS_CUH

#include <vector>
#include <cublas_v2.h>

void batchedTranspose(const std::vector<float*>& d_K, std::vector<float*>& d_KT, 
                     int num_heads, int M, int d_k, cublasHandle_t handle);

#endif // UTILS_CUH