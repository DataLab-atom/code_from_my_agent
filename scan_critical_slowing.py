"""
任务 N.1: 临界慢化检测
测量弛豫时间 τ_relax 在相变点附近的发散行为
"""

import numpy as np
import json
import time
from kuramoto import rk4_step, order_parameter


def measure_relaxation_time(N, sigma, K2, K3, dt=0.01, seed=0):
    """
    从 r≈0.5 初始条件出发，测量弛豫到稳态的时间常数
    """
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, sigma, N)

    # 构造 r≈0.5 的初始条件：一半集中，一半随机
    theta = np.zeros(N)
    n_sync = N // 2
    theta[:n_sync] = rng.normal(0, 0.3, n_sync)  # 紧密聚集 → r≈0.5
    theta[n_sync:] = rng.uniform(0, 2 * np.pi, N - n_sync)

    # 模拟并记录 r(t)
    T = 500.0
    steps = int(T / dt)
    r_ts = []
    for s in range(steps):
        theta = rk4_step(theta, omega, K2, K3, N, dt)
        if s % 50 == 0:
            r, _ = order_parameter(theta)
            r_ts.append(float(r))

    r_ts = np.array(r_ts)
    r_steady = np.mean(r_ts[-len(r_ts)//5:])

    # 拟合指数衰减: |r(t) - r_steady| ~ exp(-t/tau)
    deviation = np.abs(r_ts - r_steady)
    deviation = np.maximum(deviation, 1e-10)

    # 简单估计：找 deviation 衰减到初始值 1/e 的时间
    if deviation[0] < 1e-5:
        return 0.0, r_steady

    target = deviation[0] / np.e
    tau = T  # default
    for i in range(len(deviation)):
        if deviation[i] < target:
            tau = i * 50 * dt  # 每 50 步记录一次
            break

    return float(tau), float(r_steady)


def scan_critical_slowing():
    N = 200
    sigma = 1.0
    dt = 0.01

    # 估计 Kc ≈ 1.6 (from our benchmarks)
    Kc_est = 1.6
    K3_list = [0.0, 0.5, 1.0, 1.5]
    K2_list = np.linspace(Kc_est * 0.8, Kc_est * 1.2, 30)

    all_tau = []
    all_r_steady = []
    t0 = time.time()

    for K3 in K3_list:
        tau_arr = []
        r_arr = []
        for K2 in K2_list:
            tau, r_s = measure_relaxation_time(N, sigma, K2, K3)
            tau_arr.append(tau)
            r_arr.append(r_s)
        all_tau.append(tau_arr)
        all_r_steady.append(r_arr)

        # 找最大 tau
        max_tau = max(tau_arr)
        max_idx = tau_arr.index(max_tau)
        print(f"K₃={K3:.1f}: max_τ={max_tau:.1f} at K₂={K2_list[max_idx]:.3f}")

    output = {
        'K3_list': K3_list,
        'K2_list': K2_list.tolist(),
        'tau_relax': all_tau,
        'r_steady': all_r_steady,
        'Kc_est': Kc_est,
        'N': N, 'sigma': sigma,
    }
    with open('critical_slowing.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved critical_slowing.json")


if __name__ == '__main__':
    scan_critical_slowing()
