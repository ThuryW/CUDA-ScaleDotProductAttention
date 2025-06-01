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
#include "softmax.cuh"

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
    int N = 128;         // Rows of Q and QK (sequence length)
    int M = 128;         // Columns of K^T and QK (sequence length)
    int d_k = 64;        // Columns of Q and rows of K^T (dimension of key/query)
    int block_size = 64; // Block size for matrix multiplication (can be different from softmax block)

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory for Q, K, K^T, and QK for all heads
    std::vector<float*> d_Q(num_heads);
    std::vector<float*> d_K(num_heads);
    std::vector<float*> d_KT(num_heads);
    std::vector<float*> d_QK(num_heads);
    std::vector<float*> d_attention_weights(num_heads);
    for (int i = 0; i < num_heads; i++) {
        cudaMalloc(&d_Q[i], N * d_k * sizeof(float));    // Q: N x d_k
        cudaMalloc(&d_K[i], M * d_k * sizeof(float));    // K: M x d_k (Note: M is seq_len_K, N is seq_len_Q. Here they are same)
        cudaMalloc(&d_KT[i], d_k * M * sizeof(float));   // K^T: d_k x M
        cudaMalloc(&d_QK[i], N * M * sizeof(float));     // QK^T: N x M
        cudaMalloc(&d_attention_weights[i], N * M * sizeof(float)); // attention: N x M
    }

    // Load Q and K from existing files
    for (int i = 0; i < num_heads; i++) {
        float* h_Q = new float[N * d_k];
        float* h_K = new float[M * d_k]; // M is correct for K's original number of rows
        
        std::string Q_file = "../data/random_test/Q_head_" + std::to_string(i) + ".txt";
        std::string K_file = "../data/random_test/K_head_" + std::to_string(i) + ".txt";
        
        loadMatrix(Q_file, h_Q, N, d_k);
        loadMatrix(K_file, h_K, M, d_k); // K is M rows, d_k columns

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
    // K is (M, d_k), K^T is (d_k, M)
    batchedTranspose(d_K, d_KT, num_heads, M, d_k, handle); // M is rows of K, d_k is cols of K
    for (int i = 0; i < num_heads; i++) {
        float* h_KT = new float[d_k * M];
        cudaMemcpy(h_KT, d_KT[i], d_k * M * sizeof(float), cudaMemcpyDeviceToHost);
        std::string KT_save_file = "../data/random_verify/KT_head_" + std::to_string(i) + ".txt";
        saveMatrix(KT_save_file, h_KT, d_k, M);
        std::cout << "K^T[head_" << i << "]: " << d_k << "x" << M << std::endl;
        delete[] h_KT;
    }

    // Compute QK^T for all heads and save unscaled QK^T
    // Q (N, d_k) * K^T (d_k, M) -> QK^T (N, M)
    batchedMatmul(d_Q, d_KT, d_QK, num_heads, N, M, d_k, block_size, handle);
    for (int i = 0; i < num_heads; i++) {
        float* h_QK = new float[N * M];
        cudaMemcpy(h_QK, d_QK[i], N * M * sizeof(float), cudaMemcpyDeviceToHost);
        std::string QK_save_file = "../data/random_verify/QK_head_" + std::to_string(i) + ".txt";
        saveMatrix(QK_save_file, h_QK, N, M);
        delete[] h_QK;
    }

    // Scale QK^T by 1/sqrt(d_k) for each head
    float scale_factor = 1.0f / sqrtf(static_cast<float>(d_k));  // Correct scaling factor

    // Define thread block and grid dimensions for scaling kernel
    const int THREADS_PER_BLOCK_SCALE = 16; // Example, can be tuned
    dim3 blockDimScale(THREADS_PER_BLOCK_SCALE, THREADS_PER_BLOCK_SCALE);
    // For scaleKernel, it seems to operate on the N*M matrix.
    // The original scaleKernel probably expects gridDim to cover all elements.
    // If scaleKernel is element-wise:
    dim3 gridDimScale((M + THREADS_PER_BLOCK_SCALE - 1) / THREADS_PER_BLOCK_SCALE, 
                      (N + THREADS_PER_BLOCK_SCALE - 1) / THREADS_PER_BLOCK_SCALE);

    // Apply scaling to each head's QK^T matrix
    for (int i = 0; i < num_heads; i++) {
        // Assuming scaleKernel applies `value * scale_factor` (or `value / scale` if `scale` is sqrt(d_k))
        // If scaleKernel divides, then pass sqrtf(d_k). If it multiplies, pass 1.0f/sqrtf(d_k).
        // Your original code had `scale = sqrtf(d_k)` and presumably scaleKernel did division.
        // Let's rename `scale` in main to avoid confusion if `scaleKernel` takes a divisor.
        float divisor = sqrtf(static_cast<float>(d_k));
        scaleKernel<<<gridDimScale, blockDimScale>>>(d_QK[i], N, M, divisor); // Pass N, M to define matrix bounds
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after scaleKernel launch for head " << i << ": " << cudaGetErrorString(err) << std::endl;
            return 1; // Or handle error appropriately
        }
        cudaDeviceSynchronize();  // Ensure kernel completion
    }

    // Save the scaled QK^T result
    for (int i = 0; i < num_heads; i++) {
        float* h_QK_scaled = new float[N * M]; // Renamed for clarity
        cudaMemcpy(h_QK_scaled, d_QK[i], N * M * sizeof(float), cudaMemcpyDeviceToHost);
        std::string result_file = "../data/random_verify/QK_scaled_head_" + std::to_string(i) + ".txt";
        saveMatrix(result_file, h_QK_scaled, N, M);
        std::cout << "QK_scaled[head_" << i << "]: " << N << "x" << M << std::endl;
        delete[] h_QK_scaled;
    }

    // --- Call the new Softmax Kernel ---
    // For newSoftmaxKernel:
    // Grid dimension should be N (number of rows/sequences).
    // Block dimension should be M (number of columns/sequences to attend to).
    // Shared memory size should be M * sizeof(float).
    // M (128) is a good size for threads per block. It's a power of 2 if M is chosen so.
    // Here M=128, which is fine for threads per block.
    
    int threads_per_block_softmax = M; // M is the number of columns, cols
    if (threads_per_block_softmax > 1024) {
        std::cerr << "Error: Softmax threads per block (" << threads_per_block_softmax 
                  << ") exceeds maximum of 1024." << std::endl;
        // Handle error: This indicates M is too large for this simple kernel design.
        // A more complex kernel would be needed (e.g., multiple reads per thread or grid-stride loop).
        return 1; 
    }
    size_t shared_mem_size_softmax = static_cast<size_t>(threads_per_block_softmax) * sizeof(float);

    std::cout << "Applying newSoftmaxKernel..." << std::endl;
    for (int i = 0; i < num_heads; i++) {
        dim3 grid_dim_softmax(N); // N blocks, one for each row
        dim3 block_dim_softmax(threads_per_block_softmax); // M threads per block, processing M columns

        newSoftmaxKernel<<<grid_dim_softmax, block_dim_softmax, shared_mem_size_softmax>>>(
            d_QK[i],                // input: scaled QK^T matrix
            d_attention_weights[i], // output: attention weights matrix
            N,                      // number of rows
            M                       // number of columns
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after newSoftmaxKernel launch for head " << i << ": " 
                      << cudaGetErrorString(err) << std::endl;
            // Free allocated memory before exiting
            for (int j = 0; j <= i; ++j) { // Free up to the current head
                 cudaFree(d_Q[j]); cudaFree(d_K[j]); cudaFree(d_KT[j]);
                 cudaFree(d_QK[j]); cudaFree(d_attention_weights[j]);
            }
            // Free remaining if any (though loop structure might make this tricky)
            cublasDestroy(handle);
            return 1;
        }
        cudaDeviceSynchronize(); // Ensure kernel execution is complete before proceeding
    }
    std::cout << "newSoftmaxKernel applied successfully." << std::endl;


    // 保存注意力权重到文件
    for (int i = 0; i < num_heads; i++) {
        float* h_attention_weights = new float[N * M];
        cudaMemcpy(h_attention_weights, d_attention_weights[i], N * M * sizeof(float), cudaMemcpyDeviceToHost);
        std::string attention_weights_file = "../data/random_verify/attention_weights_head_" + std::to_string(i) + ".txt";
        saveMatrix(attention_weights_file, h_attention_weights, N, M);
        std::cout << "Attention Weights[head_" << i << "]: " << N << "x" << M << std::endl;
        delete[] h_attention_weights;
    }

    // Cleanup
    for (int i = 0; i < num_heads; i++) {
        cudaFree(d_Q[i]);
        cudaFree(d_K[i]);
        cudaFree(d_KT[i]);
        cudaFree(d_QK[i]);
        cudaFree(d_attention_weights[i]);
    }
    cublasDestroy(handle);

    std::cout << "Execution finished successfully." << std::endl;
    return 0;
}