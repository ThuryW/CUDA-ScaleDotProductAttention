#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
    // Define matrix dimensions
    const int M = 2; // Rows of A
    const int N = 2; // Columns of B
    const int K = 2; // Columns of A / Rows of B

    // Host matrices
    float h_A[M * K] = {1.0, 2.0, 3.0, 4.0}; // Matrix A: [[1, 2], [3, 4]]
    float h_B[K * N] = {5.0, 6.0, 7.0, 8.0}; // Matrix B: [[5, 6], [7, 8]]
    float h_C[M * N] = {0.0};                 // Matrix C: Result

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set matrix multiplication parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Matrix C (Result of A * B):" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}