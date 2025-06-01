# Progress of Week 2

## 概述
今天的工作聚焦于调试和优化一个基于 CUDA 的多头注意力机制实现，具体涉及矩阵 \( Q \)、\( K \)、\( K^T \)、未缩放 \( QK^T \) 和缩放 \( QK^T \) 的计算。目标是确保 CUDA 代码的计算结果与 Python 的 NumPy 实现一致，并生成中间结果以便于调试。通过多次修改代码和验证脚本，最终解决了矩阵乘法中的错误，达到了预期效果。

以下介绍具体工作内容，简要功能更新可见[README.md](https://github.com/ThuryW/CUDA-ScaleDotProductAttention/blob/main/README.md)。

## 工作内容

### 1. CUDA 代码调试与优化
- **初始问题**：
  - 运行 `verify_qk.py` 脚本后发现，CUDA 计算的未缩放 \( QK^T \) 和缩放 \( QK^T \) 与 Python 的 NumPy 实现不一致，最大差异在未缩放 \( QK^T \) 为 8.8–10.4，缩放 \( QK^T \) 为 1.1–1.3。
  - 验证输出表明 \( Q \)、\( K \) 和 \( K^T \) 正确，问题出在矩阵乘法步骤 (`batchedMatmul`)。

- **问题定位**：
  - 分析 `batchedMatmul` 函数，发现其使用 `cublasSgemm` 的参数错误：
    - 维度参数错误：`block_N` 和 `block_M` 的使用导致矩阵乘法维度颠倒。
    - 前导维度错误：\( Q \)、\( K^T \) 和 \( QK^T \) 的前导维度配置不正确（例如，错误地将 \( Q \) 的前导维度设为 \( N \)，应为 \( d_k \））。
    - 分块逻辑不必要且引入错误，增加了内存访问复杂性。
  - 缩放步骤 (`scaleKernel`) 正确，差异值比例 (\( \frac{1}{\sqrt{d_k}} = \frac{1}{8} \)) 确认了缩放逻辑无误。

- **解决方案**：
  - 修改 `batchedMatmul.cu`，移除不必要的分块逻辑，使用正确的 `cublasSgemm` 参数：
    - 维度：`m = M`, `n = N`, `k = d_k`。
    - 前导维度：\( Q \) 为 `d_k`，\( K^T \) 为 `M`，\( QK^T \) 为 `M`。
    - 添加 cuBLAS 错误检查以确保运行时无误。
  - 更新后的 `batchedMatmul` 实现：
    ```cpp
    void batchedMatmul(const std::vector<float*>& d_Q, const std::vector<float*>& d_KT, 
                       std::vector<float*>& d_QK, int num_heads, int N, int M, int d_k, 
                       int block_size, cublasHandle_t handle) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        for (int i = 0; i < num_heads; i++) {
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
    ```

- **结果**：
  - 修改后的代码重新编译并运行，验证脚本显示所有步骤（\( Q \)、\( K \)、\( K^T \)、未缩放 \( QK^T \)、缩放 \( QK^T \)）均与 Python 计算结果一致，最大差异在 \( 10^{-5} \) 量级，符合单精度浮点误差范围。

### 2. 验证脚本改进
- **初始问题**：
  - 早期的 `verify_qk.py` 脚本在比较 \( Q \) 矩阵时因输入文件形状错误（128x128 而非 128x64）导致 `ValueError`。
  - 验证脚本仅比较缩放 \( QK^T \)，无法定位中间步骤的错误。

- **解决方案**：
  - 修正 `generate_qk.py` 脚本，确保生成正确的 \( Q \) 和 \( K \) 矩阵（128x64）：
    ```python
    import numpy as np
    import os

    num_heads = 12
    N = 128
    d_k = 64
    os.makedirs('../data/random_test', exist_ok=True)
    for i in range(num_heads):
        Q = np.random.rand(N, d_k).astype(np.float32)
        K = np.random.rand(N, d_k).astype(np.float32)
        np.savetxt(f'../data/random_test/Q_head_{i}.txt', Q, fmt='%.6f')
        np.savetxt(f'../data/random_test/K_head_{i}.txt', K, fmt='%.6f')
    print("随机 Q 和 K 矩阵已生成并保存到 ../data/random_test 目录下。")
    ```
  - 更新 `verify_qk.py`，增加对 \( Q \)、\( K \)、\( K^T \)、未缩放 \( QK^T \) 和缩放 \( QK^T \) 的逐步骤验证，并添加形状检查：
    ```python
    import subprocess
    import numpy as np

    num_heads = 12
    N = 128
    M = 128
    d_k = 64

    subprocess.run(["./main", "random_test"], check=True, cwd="./build")
    for i in range(num_heads):
        print(f"\nVerifying Head {i}...")
        Q_cuda = np.loadtxt(f'../data/random_verify/Q_head_{i}.txt', dtype=np.float32)
        K_cuda = np.loadtxt(f'../data/random_verify/K_head_{i}.txt', dtype=np.float32)
        KT_cuda = np.loadtxt(f'../data/random_verify/KT_head_{i}.txt', dtype=np.float32)
        QK_cuda = np.loadtxt(f'../data/random_verify/QK_head_{i}.txt', dtype=np.float32)
        QK_scaled_cuda = np.loadtxt(f'../data/random_verify/QK_scaled_head_{i}.txt', dtype=np.float32)
        Q_test = np.loadtxt(f'../data/random_test/Q_head_{i}.txt', dtype=np.float32)
        K_test = np.loadtxt(f'../data/random_test/K_head_{i}.txt', dtype=np.float32)
        
        assert Q_cuda.shape == (N, d_k), f"Q_cuda shape mismatch: {Q_cuda.shape}"
        assert K_cuda.shape == (M, d_k), f"K_cuda shape mismatch: {K_cuda.shape}"
        assert KT_cuda.shape == (d_k, M), f"KT_cuda shape mismatch: {KT_cuda.shape}"
        assert QK_cuda.shape == (N, M), f"QK_cuda shape mismatch: {QK_cuda.shape}"
        assert QK_scaled_cuda.shape == (N, M), f"QK_scaled_cuda shape mismatch: {QK_scaled_cuda.shape}"
        assert Q_test.shape == (N, d_k), f"Q_test shape mismatch: {Q_test.shape}"
        assert K_test.shape == (M, d_k), f"K_test shape mismatch: {K_test.shape}"
        
        if np.allclose(Q_cuda, Q_test, atol=1e-5):
            print("Q matrix: Correct")
        else:
            print("Q matrix: Mismatch")
            print(f"Maximum difference: {np.max(np.abs(Q_cuda - Q_test))}")
        # 类似检查 K、K^T、QK^T、缩放 QK^T
    ```

- **结果**：
  - 修正输入文件后，消除了形状错误。
  - 逐步骤验证帮助定位矩阵乘法错误，最终确认所有步骤正确。

### 3. CUDA 代码功能增强
- **矩阵转置封装**：
  - 将 \( K \) 到 \( K^T \) 的转置操作封装为 `batchedTranspose` 函数，定义在 `transpose.cu` 和 `transpose.cuh` 中：
    ```cpp
    void batchedTranspose(const std::vector<float*>& d_K, std::vector<float*>& d_KT, 
                         int num_heads, int M, int d_k, cublasHandle_t handle) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        for (int i = 0; i < num_heads; i++) {
            cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, d_k, 
                        &alpha, d_K[i], d_k, &beta, nullptr, M, d_KT[i], M);
        }
    }
    ```
  - 在 `main.cu` 中调用此函数，替换原始转置循环，提高代码模块化。

- **保存中间结果**：
  - 修改 `main.cu`，保存所有中间矩阵（\( Q \)、\( K \)、\( K^T \)、未缩放 \( QK^T \)、缩放 \( QK^T \)）到 `../data/random_verify/` 目录，文件名为：
    - `Q_head_<i>.txt`
    - `K_head_<i>.txt`
    - `KT_head_<i>.txt`
    - `QK_head_<i>.txt`
    - `QK_scaled_head_<i>.txt`
  - 添加目录创建逻辑：`system("mkdir -p ../data/random_verify");`。

- **打印矩阵形状**：
  - 在 `main.cu` 中添加代码，打印每个矩阵的形状（例如，`Q[head_0]: 128x64`），便于调试和验证。

### 4. 最终验证
- 运行更新后的 `main` 和 `verify_qk.py`，所有注意力头的 \( Q \)、\( K \)、\( K^T \)、未缩放 \( QK^T \)、缩放 \( QK^T \) 均通过验证，最大差异在 \( 10^{-5} \) 量级，表明 CUDA 和 Python 计算结果一致。

## 遇到的问题与解决
1. **输入矩阵形状错误**：
   - 问题：`verify_qk.py` 报错，`Q_test` 形状为 128x128，而应为 128x64。
   - 解决：修正 `generate_qk.py`，确保生成正确的 \( Q \) 和 \( K \) 矩阵（128x64）。
   - 结果：消除了 `ValueError`，验证脚本正常运行。

2. **矩阵乘法错误**：
   - 问题：未缩放 \( QK^T \) 差异为 8.8–10.4，缩放 \( QK^T \) 差异为 1.1–1.3，定位到 `batchedMatmul` 的 `cublasSgemm` 参数错误。
   - 解决：修正维度和前导维度，移除分块逻辑，使用标准 `cublasSgemm` 调用。
   - 结果：未缩放和缩放 \( QK^T \) 结果与 Python 一致。

## 总结
- **成果**：成功调试并优化了 CUDA 实现的注意力机制代码，确保矩阵乘法和缩放与 Python 一致。
- **关键改进**：
  - 修正 `batchedMatmul` 的错误参数。
  - 增强验证脚本，逐步骤比较中间结果。
  - 封装矩阵转置功能，提高代码模块化。
  - 保存所有中间矩阵，便于调试。
- **后续工作**：
  - 考虑使用 `cublasSgemmBatched` 优化 `batchedMatmul` 的性能。
  - 添加更全面的 CUDA 错误检查（例如，`cudaGetLastError`）。
  - 测试更大规模的矩阵或更多注意力头，验证代码的鲁棒性。

## 附录
- **运行环境**：
  - 系统：Linux (Pollux)
  - CUDA：cuBLAS 库
  - Python：3.11（Conda 环境 `llm`）
- **编译命令**：
  ```bash
  cmake .. # 在build目录下执行
  make
  ```
- **运行验证**：
  ```bash
  ./main random_test
  ~/anaconda3/envs/llm/bin/python ../scripts/verify_qk.py
  ```

**日期**：2025年5月25日  
**作者**：王田宇