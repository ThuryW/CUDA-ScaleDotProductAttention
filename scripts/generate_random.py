import numpy as np
import os

# 参数
num_heads = 12
N = 128  # Q 和 K 的行数
d_k = 64  # Q 和 K 的列数

# 创建保存目录
os.makedirs('./data/random_test', exist_ok=True)

# 生成并保存 Q 和 K 矩阵
for i in range(num_heads):
    Q = np.random.rand(N, d_k).astype(np.float32)
    K = np.random.rand(N, d_k).astype(np.float32)
    V = np.random.rand(N, d_k).astype(np.float32)
    
    # 保存为 txt 格式
    np.savetxt(f'../data/random_test/Q_head_{i}.txt', Q, fmt='%.6f')
    np.savetxt(f'../data/random_test/K_head_{i}.txt', K, fmt='%.6f')
    np.savetxt(f'../data/random_test/V_head_{i}.txt', V, fmt='%.6f')

print("随机 Q, K, V 矩阵已生成并保存到 ../data/random_test 目录下。")