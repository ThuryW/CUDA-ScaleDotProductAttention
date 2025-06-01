#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#include <cuda_runtime.h>

/**
 * @brief Computes the softmax activation row-wise for an input matrix.
 * * Each row of the input matrix is processed independently.
 * The kernel should be launched with:
 * - Grid dimension: (num_rows, 1, 1) or simply num_rows
 * - Block dimension: (num_cols, 1, 1) or simply num_cols, where num_cols should be <= 1024 (max threads per block)
 * and ideally a power of 2 for efficiency (e.g., 64, 128, 256).
 * - Shared memory: num_cols * sizeof(float)
 * * @param input Pointer to the input matrix on the device. (e.g., scaled QK^T)
 * @param output Pointer to the output matrix on the device. (e.g., attention weights)
 * @param rows Number of rows in the input/output matrix (e.g., N).
 * @param cols Number of columns in the input/output matrix (e.g., M).
 */
__global__ void newSoftmaxKernel(const float* __restrict__ input, 
                                 float* __restrict__ output, 
                                 int rows, 
                                 int cols);

#endif // SOFTMAX_CUH