"""
KuramotoThinker 快速验证 Kc（跳过三体项，K3=0）
"""
import numpy as np
from numba import njit

@njit
def dtheta_paironly(theta, omega, K2, N):
    """仅两两耦合，O(N²)"""
    dtheta_dt = np.empty(N)
    for i in range(N):
        pair = 0.0
        for j in range(N):
            pair += np.sin(theta[j] - theta[i])
        dtheta_dt[i] = omega[i] + (K2 / N) * pair
    return dtheta_dt

@njit
def rk4_pair(theta, omega, K2, N, dt):
    k1 = dtheta_paironly(theta, omega, K2, N)
    k2 = dtheta_paironly(theta + 0.5*dt*k1, omega, K2, N)
    k3 = dtheta_paironly(theta + 0.5*dt*k2, omega, K2, N)
    k4 = dtheta_paironly(theta + dt*k3, omega, K2, N)
    return theta + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate_classic(N, sigma, K2, seed, T=200.0, dt=0.01):
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, sigma, N)
    theta = rng.uniform(0, 2*np.pi, N)
    steps = int(T / dt)
    burn = int(0.5 * steps)
    rs = []
    for s in range(steps):
        theta = rk4_pair(theta, omega, K2, N, dt)
        if s >= burn:
            z = np.mean(np.exp(1j * theta))
            rs.append(np.abs(z))
    return np.mean(rs)

sigma = 1.0
Kc_correct = 2.0 * sigma * np.sqrt(2 * np.pi) / np.pi
print(f"Kc (theory) = {Kc_correct:.4f}")
print()

# 预热 numba
_ = simulate_classic(50, 1.0, 1.0, 0, T=1.0)

for N in [100, 200, 500]:
    print(f"\n=== N={N} ===")
    K2_list = np.linspace(0.8, 2.5, 15)
    for K2 in K2_list:
        rs = []
        for seed in range(3):
            r = simulate_classic(N, sigma, K2, seed, T=200.0)
            rs.append(r)
        r_mean = np.mean(rs)
        marker = " <-- Kc" if abs(K2 - Kc_correct) < 0.06 else ""
        print(f"  K2={K2:.3f} ({K2/Kc_correct:.2f}*Kc)  r={r_mean:.4f}{marker}")
