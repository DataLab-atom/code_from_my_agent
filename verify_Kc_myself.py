"""
KuramotoThinker 独立验证 Kc
正确公式: Kc = 2*sigma*sqrt(2*pi)/pi
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from kuramoto import KuramotoHigherOrder, order_parameter

sigma = 1.0
Kc_correct = 2.0 * sigma * np.sqrt(2 * np.pi) / np.pi
Kc_wrong = 2.0 * np.sqrt(2 * np.pi) * sigma
print(f"Kc (correct) = 2*sigma*sqrt(2pi)/pi = {Kc_correct:.4f}")
print(f"Kc (wrong, in code) = 2*sqrt(2pi)*sigma = {Kc_wrong:.4f}")
print()

# 扫描 K2 在正确 Kc 附近
N = 200
K2_list = np.linspace(0.5, 3.0, 20)
print(f"N={N}, sigma={sigma}, K3=0")
print(f"{'K2':>8s} {'K2/Kc':>8s} {'r_mean':>8s} {'r_std':>8s}")
print("-" * 40)

for K2 in K2_list:
    rs = []
    for seed in range(5):
        model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=0.0, seed=seed)
        r, _ = model.simulate(T=200.0, dt=0.01)
        rs.append(r)
    r_mean = np.mean(rs)
    r_std = np.std(rs)
    print(f"{K2:8.3f} {K2/Kc_correct:8.3f} {r_mean:8.4f} {r_std:8.4f}")
