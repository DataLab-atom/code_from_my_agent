"""
GPU 加速的高阶 Kuramoto 模型 (CuPy)
均场分解后 O(N) per step，整个 RK4 在 GPU 上运行
支持批量并行模拟多个参数点
"""

import cupy as cp
import numpy as np
import json
import time


def dtheta_gpu(theta, omega, K2, K3, N):
    """GPU 版相位导数 (向量化)"""
    S1 = cp.sum(cp.sin(theta))
    C1 = cp.sum(cp.cos(theta))
    S2 = cp.sum(cp.sin(2.0 * theta))
    C2 = cp.sum(cp.cos(2.0 * theta))

    si = cp.sin(theta)
    ci = cp.cos(theta)

    pair = (K2 / N) * (S1 * ci - C1 * si)

    sum_cos_ki = C1 * ci - S1 * si
    sum_sin_ki = S1 * ci + C1 * si
    triplet = (K3 / (N * N)) * (S2 * sum_cos_ki - C2 * sum_sin_ki)

    return omega + pair + triplet


def rk4_step_gpu(theta, omega, K2, K3, N, dt):
    """GPU RK4"""
    k1 = dtheta_gpu(theta, omega, K2, K3, N)
    k2 = dtheta_gpu(theta + 0.5 * dt * k1, omega, K2, K3, N)
    k3 = dtheta_gpu(theta + 0.5 * dt * k2, omega, K2, K3, N)
    k4 = dtheta_gpu(theta + dt * k3, omega, K2, K3, N)
    return theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def order_param_gpu(theta):
    """GPU 序参量"""
    N = len(theta)
    z = cp.sum(cp.exp(1j * theta)) / N
    return float(cp.abs(z).get())


def simulate_gpu(N, sigma, K2, K3, T=200.0, dt=0.01, seed=0, burn_frac=0.1, device=0):
    """单次 GPU 模拟"""
    with cp.cuda.Device(device):
        rng = np.random.default_rng(seed)
        omega = cp.array(rng.normal(0, sigma, N), dtype=cp.float64)
        theta = cp.array(rng.uniform(0, 2 * np.pi, N), dtype=cp.float64)

        steps = int(T / dt)
        burn = int(steps * burn_frac)
        r_sum = 0.0
        count = 0

        for s in range(steps):
            theta = rk4_step_gpu(theta, omega, K2, K3, N, dt)
            if s >= burn:
                r_sum += order_param_gpu(theta)
                count += 1

        r_avg = r_sum / count if count > 0 else 0.0
        return r_avg


def batch_scan_gpu(param_list, N=200, sigma=1.0, T=200.0, dt=0.01,
                   n_seeds=10, device=0):
    """
    批量扫描参数点
    param_list: [(K2, K3), ...]
    """
    results = []
    with cp.cuda.Device(device):
        for K2, K3 in param_list:
            r_trials = []
            for seed in range(n_seeds):
                r = simulate_gpu(N, sigma, K2, K3, T=T, dt=dt, seed=seed, device=device)
                r_trials.append(r)
            results.append({
                'K2': float(K2), 'K3': float(K3),
                'r_mean': float(np.mean(r_trials)),
                'r_std': float(np.std(r_trials)),
            })
    return results


if __name__ == '__main__':
    # 基准测试: GPU vs CPU
    print("=== GPU Kuramoto Benchmark ===")

    # Warmup
    simulate_gpu(200, 1.0, 2.0, 0.0, T=10.0, device=0)

    for N in [200, 500, 1000]:
        t0 = time.time()
        r = simulate_gpu(N, 1.0, 3.0, 0.5, T=200.0, device=0)
        elapsed = time.time() - t0
        print(f"N={N:5d}: r={r:.4f}, time={elapsed:.2f}s (GPU)")

    # 批量扫描测试
    print("\n=== Batch scan test ===")
    params = [(K2, K3) for K2 in np.linspace(0, 4, 10) for K3 in np.linspace(-2, 2, 10)]
    t0 = time.time()
    results = batch_scan_gpu(params, N=200, T=100.0, n_seeds=5, device=0)
    elapsed = time.time() - t0
    print(f"{len(params)} parameter points, 5 seeds each: {elapsed:.1f}s ({elapsed/len(params):.2f}s/point)")
