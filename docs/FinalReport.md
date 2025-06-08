# Implementing Scaled Dot-Product Attention Using CUDA

## 团队成员
- 王田宇 12431095

## 项目文件结构
具体项目除了提交的zip附件外，也已同步上传到<https://github.com/ThuryW/CUDA-ScaleDotProductAttention>。
- `build` 编译目录，需要手动生成
- `data`
  - `random_test` 用于存放脚本生成的随机QKV矩阵测试数据
  - `random_verify` 用于存放CUDA计算结果
- `docs`
- `scripts`
  - `generate_random.py` 生成随机QKV矩阵
  - `verify.py` 用于验证CUDA计算是否正确。会用python的numpy库进行矩阵计算，并对比二者的计算结果是否一致
- `src`
  - `include` 头文件目录
    - `inout.cuh`, `matmul.cuh`, `softmax.cuh`, `utils.cuh`
  - `inout.cu` 包含文件数据读取和保存函数
  - `matmul.cu` 包含矩阵乘法函数
  - `softmax.cu` 包含softmax函数
  - `utils.cu` 包含矩阵转置函数
  - `main.cu` 主函数，执行整个Scaled Dot-Product Attention计算过程
- `test`
  - `test_matmul.cu` 测试文件，在第一周任务中使用，用于验证矩阵乘法和cuda代码能否成功执行
- `CMakeLists.txt` CMake项目构建文件
  
## 整体设计方法
本项目旨在利用 CUDA 实现 GPU 加速的 Scaled Dot-Product Attention 机制，这是 Transformer 模型中的核心组件。该机制通过计算查询（Query）与键（Key）之间的相似度来生成注意力分数，并将其应用于值（Value）向量。其数学公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：$Q$为查询矩阵, $K$为键矩阵, $V$为值矩阵, $d_k$为键的维度。

根据以上公式，可以将任务大体分解为矩阵乘法和softmax函数计算两部分

### 项目实现策略