"""
任务 K.1: 频率-相位关联：哪些振荡器先锁定？
测量锁定时间 t_lock(i) vs 自然频率 ωᵢ
"""

import numpy as np
import json
import time
from kuramoto import rk4_step, order_parameter


def measure_locking_order(N=200, sigma=1.0, K2=3.0, K3=0.0, T=500.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, sigma, N)
    theta = rng.uniform(0, 2 * np.pi, N)

    steps = int(T / dt)
    lock_threshold = 0.1
    lock_duration = 100  # 需持续 100 步
    lock_counter = np.zeros(N, dtype=int)
    t_lock = np.full(N, np.inf)  # 锁定时间

    for s in range(steps):
        theta = rk4_step(theta, omega, K2, K3, N, dt)

        # 计算群体平均相位速度
        r, psi = order_parameter(theta)

        # 估计每个振荡器的瞬时频率（简单差分）
        # 用 dtheta/dt 的均场近似
        S1 = np.sum(np.sin(theta))
        C1 = np.sum(np.cos(theta))
        # 群体平均相位速度 ~ d(psi)/dt，近似为 0（旋转参照系）
        # 判断锁定：|ωᵢ_eff - ω_mean| < threshold
        # 简化：看 |θᵢ(t) - ψ(t)| mod 2π 是否稳定
        phase_diff = np.abs(np.angle(np.exp(1j * (theta - psi))))

        for i in range(N):
            if t_lock[i] < np.inf:
                continue  # 已锁定
            if phase_diff[i] < 0.5:  # 相位差小于 0.5 rad
                lock_counter[i] += 1
                if lock_counter[i] >= lock_duration:
                    t_lock[i] = (s - lock_duration) * dt
            else:
                lock_counter[i] = 0

    return omega, t_lock


def scan_locking():
    N = 200
    sigma = 1.0
    K2 = 3.0
    K3_values = [-1.0, 0.0, 1.0]

    results = {}
    t0 = time.time()

    for K3 in K3_values:
        omega, t_lock = measure_locking_order(N=N, sigma=sigma, K2=K2, K3=K3)
        key = f"K3_{K3:+.1f}"
        # 将 inf 替换为 -1 表示未锁定
        t_lock_clean = [float(t) if t < np.inf else -1.0 for t in t_lock]
        results[key] = t_lock_clean

        locked = np.sum(t_lock < np.inf)
        mean_t = np.mean(t_lock[t_lock < np.inf]) if locked > 0 else float('inf')
        print(f"K₃={K3:+.1f}: locked={locked}/{N}, mean_t_lock={mean_t:.1f}")

    # omega 对所有 K₃ 相同（seed=0）
    rng = np.random.default_rng(0)
    omega = rng.normal(0, sigma, N)

    output = {
        'omega_i': omega.tolist(),
        'K2': K2,
        'sigma': sigma,
        'N': N,
        't_lock_K3neg': results['K3_-1.0'],
        't_lock_K3zero': results['K3_+0.0'],
        't_lock_K3pos': results['K3_+1.0'],
    }
    with open('locking_order.json', 'w') as f:
        json.dump(output, f)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved locking_order.json")


if __name__ == '__main__':
    scan_locking()
