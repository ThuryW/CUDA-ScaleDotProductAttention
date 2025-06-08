import subprocess
import numpy as np

# Parameters
num_heads = 12
N = 128
M = 128
d_k = 64
d_v = 64 # Dimension of Value matrix

# 可选：在验证前自动运行CUDA代码
# print("Running CUDA implementation...")
# subprocess.run(["./main", "random_test"], check=True, cwd="./build")
# print("CUDA implementation finished.")

# Verify each step for all heads
for i in range(num_heads):
    print(f"\nVerifying Head {i}...")
    
    # --- Data Loading ---
    # 从 random_verify 目录加载 CUDA 计算的所有中间和最终结果
    Q_cuda = np.loadtxt(f'../data/random_verify/Q_head_{i}.txt', dtype=np.float32)
    K_cuda = np.loadtxt(f'../data/random_verify/K_head_{i}.txt', dtype=np.float32)
    V_cuda = np.loadtxt(f'../data/random_verify/V_head_{i}.txt', dtype=np.float32)
    KT_cuda = np.loadtxt(f'../data/random_verify/KT_head_{i}.txt', dtype=np.float32)
    QK_cuda = np.loadtxt(f'../data/random_verify/QK_head_{i}.txt', dtype=np.float32)
    QK_scaled_cuda = np.loadtxt(f'../data/random_verify/QK_scaled_head_{i}.txt', dtype=np.float32)
    attention_weights_cuda = np.loadtxt(f'../data/random_verify/attention_weights_head_{i}.txt', dtype=np.float32)
    output_cuda = np.loadtxt(f'../data/random_verify/output_head_{i}.txt', dtype=np.float32)
    
    # 从 random_test 目录加载原始输入，用于 Python 参考计算
    Q_test = np.loadtxt(f'../data/random_test/Q_head_{i}.txt', dtype=np.float32)
    K_test = np.loadtxt(f'../data/random_test/K_head_{i}.txt', dtype=np.float32)
    V_test = np.loadtxt(f'../data/random_test/V_head_{i}.txt', dtype=np.float32)
    
    # --- Verification Steps ---

    # Step 1: 验证 Q, K, V 矩阵加载
    if np.allclose(Q_cuda, Q_test, atol=1e-5):
        print("Q matrix: Correct")
    else:
        print("Q matrix: Mismatch")
        print(f"  - Maximum difference: {np.max(np.abs(Q_cuda - Q_test))}")
    
    if np.allclose(K_cuda, K_test, atol=1e-5):
        print("K matrix: Correct")
    else:
        print("K matrix: Mismatch")
        print(f"  - Maximum difference: {np.max(np.abs(K_cuda - K_test))}")
        
    if np.allclose(V_cuda, V_test, atol=1e-5):
        print("V matrix: Correct")
    else:
        print("V matrix: Mismatch")
        print(f"  - Maximum difference: {np.max(np.abs(V_cuda - V_test))}")
    
    # Step 2: 验证 K^T (K的转置)
    KT_python = K_test.T
    if np.allclose(KT_cuda, KT_python, atol=1e-5):
        print("K^T matrix: Correct")
    else:
        print("K^T matrix: Mismatch")
        print(f"  - Maximum difference: {np.max(np.abs(KT_cuda - KT_python))}")
    
    # Step 3: 验证未缩放的 QK^T
    QK_python = np.dot(Q_test, K_test.T)
    # 增加对矩阵乘法结果的容忍度
    if np.allclose(QK_cuda, QK_python, atol=1e-4):
        print("Unscaled QK^T: Correct")
    else:
        print("Unscaled QK^T: Mismatch")
        print(f"  - Maximum difference: {np.max(np.abs(QK_cuda - QK_python))}")
    
    # Step 4: 验证缩放后的 QK^T
    QK_scaled_python = QK_python / np.sqrt(d_k)
    if np.allclose(QK_scaled_cuda, QK_scaled_python, atol=1e-5):
        print("Scaled QK^T: Correct")
    else:
        print("Scaled QK^T: Mismatch")
        print(f"  - Maximum difference: {np.max(np.abs(QK_scaled_cuda - QK_scaled_python))}")

    # Step 5: 验证 Softmax (注意力权重)
    # 使用数值稳定的方法在 Python 中计算 Softmax
    max_per_row = np.max(QK_scaled_python, axis=1, keepdims=True)
    exp_values = np.exp(QK_scaled_python - max_per_row)
    sum_exp = np.sum(exp_values, axis=1, keepdims=True)
    attention_weights_python = exp_values / sum_exp

    # 由于浮点数计算的差异，softmax的容忍度可能需要稍微放宽
    if np.allclose(attention_weights_cuda, attention_weights_python, atol=1e-5):
        print("Softmax (Attention Weights): Correct")
    else:
        print("Softmax (Attention Weights): Mismatch")
        print(f"  - Maximum difference: {np.max(np.abs(attention_weights_cuda - attention_weights_python))}")
        
    # Step 6: 验证加权和 (Attention * V) -- 新增部分
    output_python = np.dot(attention_weights_python, V_test)
    if np.allclose(output_cuda, output_python, atol=1e-4):
        print("Weighted Sum (Output): Correct")
    else:
        print("Weighted Sum (Output): Mismatch")
        print(f"  - Maximum difference: {np.max(np.abs(output_cuda - output_python))}")