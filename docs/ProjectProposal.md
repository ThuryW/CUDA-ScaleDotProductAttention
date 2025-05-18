# Implementing Scaled Dot-Product Attention Using CUDA

## 团队成员


## 整体设计方法、算法选择和实现思路

### 整体设计方法
本项目旨在利用 CUDA 实现 GPU 加速的 Scaled Dot-Product Attention 机制，这是 Transformer 模型中的核心组件。该机制通过计算查询（Query）与键（Key）之间的相似度来生成注意力分数，并将其应用于值（Value）向量。其数学公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：$Q$为查询矩阵, $K$为键矩阵, $V$为值矩阵, $d_k$为键的维度。

### 算法选择
- **矩阵乘法 (Matmul)**: 利用NVIDIA的cuBLAS库实现$QK^T$的高效矩阵乘法，充分发挥 GPU 的并行计算能力。
  - 具体而言，考虑以下两种思路：
    1. 基于内积的矩阵乘法+分块矩阵乘法: 即左矩阵按行分割，右矩阵按列分割，每次乘法计算会直接得出最终的输出值。分块算法可以减小计算规模和充分利用并行性。
    2. 基于外积的矩阵乘法+分块矩阵乘法：即左矩阵按列分割，右矩阵按行分割。此时每次乘法得出的结果仍然是一个矩阵，最终需要将这些矩阵相加得到最终输出结果。同样支持分块操作。
  - 采用哪一种策略还在考虑，会根据具体实现难度、并行效果进行选择。
- **Softmax**: 编写自定义 CUDA 内核，针对缩放后的矩阵按行计算 Softmax，确保数值稳定性。
  - 由于softmax是按行进行的，且最初的exp函数计算过程相互独立，因此可以充分利用GPU的并行性进行分块计算，以适应其内存和计算资源的限制。每个小块可以在不同的线程块中并行处理，最终合并结果。
  - 考虑到数值稳定性，将采用“减去最大值”的优化技巧。
  - Nvidia提供的cuDNN库中似乎包含了针对Softmax的专门实现，可以参考或直接使用。

### 实现思路
实现过程分为以下步骤：
1. **数据准备**: 使用 CUDA 提供的内存管理函数（例如 `cudaMalloc` 和 `cudaMemcpy`），将输入矩阵$Q$、$K$和$V$从主机内存传输到设备内存。
2. **矩阵乘法**: 调用 cuBLAS 库中的 `cublasSgemm` 函数（适用于单精度浮点运算），计算$QK^T$。
3. **缩放**: 编写一个 CUDA 内核，将$QK^T$的每个元素除以$d_k$，完成缩放操作。
1. **Softmax**: 开发一个高效的 CUDA 内核，对缩放后的矩阵按行应用Softmax函数，优化并行性和数值稳定性。
2. **加权和**: 再次调用 `cublasSgemm`，将注意力权重与$V$矩阵相乘，得到最终输出。
1. **结果检索**: 使用 `cudaMemcpy` 将计算结果从设备内存复制回主机内存。

## 实现计划
项目计划为期四周，从 2025 年 5 月 11 日至 6 月 8 日，具体安排如下：
- **第 1 周（5 月 11 日 - 5 月 17 日）**:
  - 学习 CUDA 编程基础，包括内存管理、内核启动和线程组织。
  - 研究 cuBLAS 库，熟悉 `cublasSgemm` 的用法。
  - 搭建并测试一个简单的 cuBLAS 矩阵乘法程序。
- **第 2 周（5 月 18 日 - 5 月 24 日）**:
  - 使用 cuBLAS 实现$QK^T$的矩阵乘法。 
  - 编写并测试一个 CUDA 内核，用于缩放矩阵（除以$d_k$）。
- **第 3 周（5 月 25 日 - 5 月 31 日）**:
  - 开发并优化 Softmax 的 CUDA 内核，确保数值稳定性。
  - 集成矩阵乘法、缩放和 Softmax，计算注意力权重。
- **第 4 周（6 月 1 日 - 6 月 8 日）**:
  - 使用 cuBLAS 完成加权和计算（注意力权重与$V$的乘法）。
  - 对完整的 Scaled Dot-Product Attention 实现进行测试和调试。
  - 完成文档，记录代码细节和性能分析结果。

## 参考资料
1. NVIDIA Corporation. "CUDA C Programming Guide." https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
2. Vaswani, Ashish, et al. "Attention is All You Need." Advances in Neural Information Processing Systems 30 (2017). https://arxiv.org/abs/1706.03762
3. NVIDIA Corporation. "cuBLAS Library User Guide." https://docs.nvidia.com/cuda/cublas/index.html
4. Sanders, Jason, and Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming. Addison-Wesley Professional, 2010.