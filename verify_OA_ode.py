"""
KuramotoThinker: 直接积分 OA ODE 验证理论预测
dr/dt = -Delta*r + r*(1-r^2)/2 * (K2 + K3*r^2)
这只是一维ODE，瞬间完成
"""
import numpy as np

Delta = np.sqrt(2*np.pi) / np.pi  # ≈ 0.7979
Kc = 2 * Delta  # ≈ 1.5958
print(f"Delta = {Delta:.4f}")
print(f"Kc = 2*Delta = {Kc:.4f}")

def f(r, K2, K3):
    return -Delta*r + r*(1-r**2)/2 * (K2 + K3*r**2)

def steady_state(K2, K3, r0=0.01, dt=0.001, T=500):
    r = r0
    for _ in range(int(T/dt)):
        r = r + dt * f(r, K2, K3)
        r = max(r, 0.0)
        r = min(r, 1.0)
    return r

# 1. 经典验证：K3=0
print("\n=== 经典 K3=0 ===")
for K2 in np.linspace(0.5, 3.0, 15):
    r = steady_state(K2, 0.0)
    print(f"  K2={K2:.3f} ({K2/Kc:.2f}*Kc)  r*={r:.4f}")

# 2. 重入同步测试：固定 K2=1.1*Kc，扫描 K3
print(f"\n=== 重入同步测试: K2 = 1.1*Kc = {1.1*Kc:.4f} ===")
for K3 in np.linspace(0, 5, 25):
    r = steady_state(1.1*Kc, K3)
    print(f"  K3={K3:.2f}  r*={r:.4f}")

# 3. 重入同步测试：多个 K2
print("\n=== 重入同步: 多个 K2 ===")
for K2_mult in [1.05, 1.1, 1.2, 1.5, 2.0]:
    K2 = K2_mult * Kc
    print(f"\nK2 = {K2_mult:.2f}*Kc = {K2:.3f}:")
    for K3 in [0, 1, 2, 3, 5, 10]:
        r = steady_state(K2, K3)
        print(f"  K3={K3:5.1f}  r*={r:.4f}")

# 4. 鞍结分岔点验证
print("\n=== 鞍结分岔点 K2_sn(K3) ===")
print(f"{'K3':>6s} {'K2_sn_theory':>14s} {'K2_sn_numeric':>14s} {'diff':>8s}")
for K3 in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
    # 理论
    K2_sn_th = np.sqrt(8*Delta*K3) - K3
    # 数值：从同步态（r=0.99）出发，降低 K2 直到同步消失
    K2_sn_num = None
    for K2 in np.linspace(2.0, 0.0, 200):
        r = steady_state(K2, K3, r0=0.99, T=1000)
        if r < 0.05:
            K2_sn_num = K2
            break
    K2_sn_num = K2_sn_num if K2_sn_num else float('nan')
    diff = abs(K2_sn_th - K2_sn_num) if K2_sn_num else float('nan')
    print(f"{K3:6.2f} {K2_sn_th:14.4f} {K2_sn_num:14.4f} {diff:8.4f}")
