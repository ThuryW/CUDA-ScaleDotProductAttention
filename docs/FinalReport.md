# 使用 CUDA 实现缩放点积注意力机制

## 团队成员
- 王田宇 12431095

## 项目文件结构
本项目除提交的压缩包外，还已同步上传至 <https://github.com/ThuryW/CUDA-ScaleDotProductAttention>。文件结构如下：
- `build`：编译目录，需手动创建。
- `data`：
  - `random_test`：存储脚本生成的随机 Q、K、V 矩阵测试数据。
  - `random_verify`：存储 CUDA 计算结果，用于验证。
- `docs`：文档目录。
- `scripts`：
  - `generate_random.py`：生成随机 Q、K、V 矩阵的脚本。
  - `verify.py`：验证 CUDA 计算结果的脚本，通过与 NumPy 的矩阵计算结果对比验证正确性。
- `src`：
  - `include`：头文件目录。
    - `inout.cuh`, `matmul.cuh`, `softmax.cuh`, `utils.cuh`
  - `inout.cu`：包含文件数据读取和保存函数。
  - `matmul.cu`：实现矩阵乘法函数。
  - `softmax.cu`：实现 softmax 函数。
  - `utils.cu`：包含矩阵转置函数。
  - `main.cu`：主函数，执行完整的缩放点积注意力计算流程。
- `test`：
  - `test_matmul.cu`：测试文件，用于第一周验证矩阵乘法和 CUDA 代码执行。
- `CMakeLists.txt`：CMake 项目构建配置文件。

## 总体设计方法
本项目旨在利用 CUDA 实现 GPU 加速的缩放点积注意力（Scaled Dot-Product Attention）机制，这是 Transformer 模型的核心组件。该机制通过计算查询（Query）与键（Key）之间的相似度生成注意力分数，并将其应用于值（Value）向量。其数学公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q$：查询矩阵 (N × d_k)
- $K$：键矩阵 (M × d_k)
- $V$：值矩阵 (M × d_v)
- $d_k$：键/查询的维度
- $d_v$：值的维度（在本实现中，$d_v = d_k$）

任务被分解为矩阵乘法 ($QK^T$)、缩放、softmax 计算和最终与 $V$ 的矩阵乘法。

### 项目实现策略
实现利用 NVIDIA 的 cuBLAS 库进行高效矩阵乘法，并通过自定义 CUDA 内核实现缩放和 softmax 操作。主要步骤包括：
1. **数据准备**：从文件中读取输入矩阵 $Q$、$K$ 和 $V$，使用 `cudaMalloc` 和 `cudaMemcpy` 传输至 GPU 内存。
2. **矩阵乘法 ($QK^T$)**：使用 cuBLAS 的 `cublasSgemm` 函数进行多头批处理矩阵乘法。
3. **缩放**：通过自定义 CUDA 内核将 $QK^T$ 矩阵除以 $\sqrt{d_k}$，以稳定注意力分数。
4. **Softmax**：开发高效的 CUDA 内核，按行应用 softmax 函数，优化并行性和数值稳定性。
5. **加权和**：再次调用 `cublasSgemm`，将注意力权重与 $V$ 矩阵相乘，得到最终输出。
6. **结果检索**：使用 `cudaMemcpy` 将结果从 GPU 内存复制回主机内存并保存。

## 实现细节
### 矩阵乘法
`matmul.cu` 文件实现了两个关键函数：
- **batchedMatmul**：为多个注意力头计算 $QK^T$，使用 cuBLAS 处理矩阵 $Q$ (N × d_k) 和 $K^T$ (d_k × M)，生成 $QK^T$ (N × M)。该函数逐头迭代，确保高效并行计算。
- **batchedAttentionVMatmul**：计算最终加权和，将注意力权重 (N × M) 与 $V$ (M × d_v) 相乘，生成输出 (N × d_v)。通过调整矩阵顺序适配 cuBLAS 的列主序存储。

### 缩放
`matmul.cu` 中的 `scaleKernel` 是一个 CUDA 内核，将 $QK^T$ 的每个元素除以 $\sqrt{d_k}$。它使用二维线程网格覆盖矩阵，每个线程处理一个元素。内核为每个注意力头单独启动，支持多头并行。

### Softmax
`softmax.cu` 中的 `newSoftmaxKernel` 实现了按行的 softmax 计算，注重数值稳定性：
- **共享内存**：每个线程块处理一行数据，将数据存储在共享内存中，减少全局内存访问。
- **并行归约**：通过并行归约查找每行的最大值，避免指数运算溢出。
- **指数和求和**：计算 $\exp(x - \max(x))$ 并通过并行归约求和。
- **归一化**：将每个指数值除以总和，生成 softmax 概率。
内核假设每个线程块的线程数等于列数 (M)，并动态分配共享内存。

### 主流程
`main.cu` 协调整个计算流程：
- **初始化**：为 12 个注意力头（基于 BERT-base 配置：N=128, M=128, d_k=64, d_v=64）分配 GPU 内存，包括 $Q$、$K$、$K^T$、$QK^T$、注意力权重、$V$ 和输出矩阵。
- **数据加载**：从 `random_test` 目录读取 $Q$、$K$ 和 $V$，并将副本保存至 `random_verify` 用于验证。
- **计算步骤**：
  - 使用 `batchedTranspose` 将 $K$ 转置为 $K^T$。
  - 使用 `batchedMatmul` 计算 $QK^T$。
  - 使用 `scaleKernel` 缩放 $QK^T$。
  - 使用 `newSoftmaxKernel` 应用 softmax。
  - 使用 `batchedAttentionVMatmul` 计算最终输出。
- **结果保存**：保存中间和最终结果（例如 $K^T$、$QK^T$、缩放后的 $QK^T$、注意力权重、输出）至 `random_verify`。
- **清理**：释放 GPU 内存并销毁 cuBLAS 句柄。

### 验证
`verify.py` 脚本验证每个计算步骤：
- **输入验证**：比较加载的 $Q$、$K$ 和 $V$ 矩阵与原始测试数据。
- **转置验证**：验证 $K^T$ 与 NumPy 的 $K$ 转置结果。
- **矩阵乘法**：检查未缩放的 $QK^T$ 与 NumPy 的 `np.dot(Q, K.T)` 结果。
- **缩放**：验证缩放后的 $QK^T$ 与 $QK^T / \sqrt{d_k}$ 结果。
- **Softmax**：使用 NumPy 计算数值稳定的 softmax 并与 CUDA 结果比较。
- **加权和**：验证最终输出与 `np.dot(attention_weights, V)` 结果。
容差（例如，大多数步骤为 atol=1e-5，矩阵乘法为 atol=1e-4）考虑了 CUDA 和 NumPy 之间的浮点数差异。

## 挑战与解决方案
- **Softmax 数值稳定性**：softmax 内核通过减去每行最大值避免溢出，并使用并行归约提高效率。
- **内存管理**：多头的大矩阵需要谨慎分配和释放内存，以避免内存泄漏。
- **cuBLAS 列主序**：通过调整矩阵乘法参数适配 cuBLAS 的列主序存储，确保行主序 C++ 数据的正确性。
- **线程块配置**：调整线程块大小（例如，缩放为 16×16，softmax 为 M 个线程）以平衡占用率和性能。

## 结果
实现成功为 12 个注意力头计算了缩放点积注意力，所有步骤均通过 `verify.py` 验证。CUDA 实现与 NumPy 参考计算结果在可接受容差内一致，确认了正确性。由于时间限制，未进行定量性能分析，但使用 cuBLAS 和优化内核确保了高效的 GPU 利用率。

## 未来改进
- **性能优化**：分析并调整内核配置（例如，块大小、共享内存使用）以提升性能。
- **cuDNN 集成**：探索 cuDNN 的 softmax 实现以获得潜在性能提升。
- **可扩展性**：支持可变的序列长度和头维度。
- **错误处理**：增强健壮性，加入更全面的错误检查和恢复机制。

## 参考资料
1. NVIDIA Corporation. "CUDA C 编程指南." https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
2. Vaswani, Ashish, et al. "Attention is All You Need." Advances in Neural Information Processing Systems 30 (2017). https://arxiv.org/abs/1706.03762
3. NVIDIA Corporation. "cuBLAS 库用户指南." https://docs.nvidia.com/cuda/cublas/index.html
4. Sanders, Jason, and Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming. Addison-Wesley Professional, 2010.