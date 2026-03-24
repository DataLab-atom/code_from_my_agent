"""
验证双稳态预测：K2=1.0, K3=3.0, N=50
精确 Gaussian 预测：低解 r≈0.50，高解 r≈0.96
从随机初始条件出发应该看到两种结局
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from kuramoto import KuramotoHigherOrder

K2 = 1.0
K3 = 3.0
N = 50
sigma = 1.0

print(f"Bistability test: K2={K2}, K3={K3}, N={N}, sigma={sigma}")
print(f"Exact Gaussian prediction: r_low≈0.50, r_high≈0.96")
print()

# 从多个随机种子出发
results = []
for seed in range(20):
    model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=seed)
    r, tc = model.simulate(T=200.0, dt=0.01)
    results.append(r)
    print(f"  seed={seed:2d}: r={r:.4f}")

results = np.array(results)
print(f"\nMean: {results.mean():.4f}, Std: {results.std():.4f}")
print(f"Min: {results.min():.4f}, Max: {results.max():.4f}")

# 检查是否有双峰分布
low_count = np.sum(results < 0.7)
high_count = np.sum(results >= 0.7)
print(f"Low (<0.7): {low_count}, High (>=0.7): {high_count}")
if low_count > 0 and high_count > 0:
    print("BISTABLE! Both attractors visited.")
else:
    print("Single attractor only.")
