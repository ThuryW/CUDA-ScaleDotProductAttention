import subprocess
import numpy as np

# Parameters
num_heads = 12
N = 128
M = 128
d_k = 64

# Call the CUDA main function
# subprocess.run(["./main", "random_test"], check=True, cwd="./build")

# Verify each step for all heads
for i in range(num_heads):
    print(f"\nVerifying Head {i}...")
    
    # Load matrices from random_verify
    Q_cuda = np.loadtxt(f'../data/random_verify/Q_head_{i}.txt', dtype=np.float32)
    K_cuda = np.loadtxt(f'../data/random_verify/K_head_{i}.txt', dtype=np.float32)
    KT_cuda = np.loadtxt(f'../data/random_verify/KT_head_{i}.txt', dtype=np.float32)
    QK_cuda = np.loadtxt(f'../data/random_verify/QK_head_{i}.txt', dtype=np.float32)
    QK_scaled_cuda = np.loadtxt(f'../data/random_verify/QK_scaled_head_{i}.txt', dtype=np.float32)
    
    # Load original Q and K from random_test for consistency check
    Q_test = np.loadtxt(f'../data/random_test/Q_head_{i}.txt', dtype=np.float32)
    K_test = np.loadtxt(f'../data/random_test/K_head_{i}.txt', dtype=np.float32)
    
    # Step 1: Verify Q and K loading
    if np.allclose(Q_cuda, Q_test, atol=1e-5):
        print("Q matrix: Correct")
    else:
        print("Q matrix: Mismatch")
        print(f"Maximum difference: {np.max(np.abs(Q_cuda - Q_test))}")
    
    if np.allclose(K_cuda, K_test, atol=1e-5):
        print("K matrix: Correct")
    else:
        print("K matrix: Mismatch")
        print(f"Maximum difference: {np.max(np.abs(K_cuda - K_test))}")
    
    # Step 2: Verify K^T (transpose of K)
    KT_python = K_test.T
    if np.allclose(KT_cuda, KT_python, atol=1e-5):
        print("K^T matrix: Correct")
    else:
        print("K^T matrix: Mismatch")
        print(f"Maximum difference: {np.max(np.abs(KT_cuda - KT_python))}")
    
    # Step 3: Verify unscaled QK^T
    QK_python = np.dot(Q_test, K_test.T)
    if np.allclose(QK_cuda, QK_python, atol=1e-5):
        print("Unscaled QK^T: Correct")
    else:
        print("Unscaled QK^T: Mismatch")
        print(f"Maximum difference: {np.max(np.abs(QK_cuda - QK_python))}")
    
    # Step 4: Verify scaled QK^T
    QK_scaled_python = QK_python / np.sqrt(d_k)
    if np.allclose(QK_scaled_cuda, QK_scaled_python, atol=1e-5):
        print("Scaled QK^T: Correct")
    else:
        print("Scaled QK^T: Mismatch")
        print(f"Maximum difference: {np.max(np.abs(QK_scaled_cuda - QK_scaled_python))}")