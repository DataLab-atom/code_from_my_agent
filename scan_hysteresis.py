"""
任务 G.1: 滞后环与一阶相变检测
正向扫描(K₂↑) vs 反向扫描(K₂↓)，检测爆炸式同步
"""

import numpy as np
import json
import time
from kuramoto import KuramotoHigherOrder, order_parameter, rk4_step


def hysteresis_scan():
    sigma = 1.0
    N = 200
    T_equil = 300.0  # 每个 K₂ 值平衡时间
    dt = 0.01
    K3_list = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    K2_list = np.linspace(0, 6, 40)

    r_forward = []   # [len(K3_list) x len(K2_list)]
    r_backward = []

    t0 = time.time()
    for ki, K3 in enumerate(K3_list):
        # === 正向扫描：K₂ 从 0 到 6 ===
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, 2 * np.pi, N)
        omega = rng.normal(0, sigma, N)
        r_fwd = []
        for K2 in K2_list:
            steps = int(T_equil / dt)
            for _ in range(steps):
                theta = rk4_step(theta, omega, K2, K3, N, dt)
            r, _ = order_parameter(theta)
            r_fwd.append(float(r))

        # === 反向扫描：K₂ 从 6 到 0 ===
        # 从正向最后状态开始
        r_bwd = []
        for K2 in reversed(K2_list):
            steps = int(T_equil / dt)
            for _ in range(steps):
                theta = rk4_step(theta, omega, K2, K3, N, dt)
            r, _ = order_parameter(theta)
            r_bwd.append(float(r))
        r_bwd.reverse()  # 重新排列为 K₂ 升序

        r_forward.append(r_fwd)
        r_backward.append(r_bwd)

        # 检测滞后
        hysteresis_area = np.trapz(
            np.abs(np.array(r_fwd) - np.array(r_bwd)),
            K2_list
        )
        elapsed = time.time() - t0
        eta = elapsed / (ki + 1) * (len(K3_list) - ki - 1)
        print(f"K₃={K3:+.1f}: hysteresis_area={hysteresis_area:.4f}  "
              f"[{ki+1}/{len(K3_list)}] ETA {eta/60:.1f}min")

    result = {
        'K3_list': K3_list,
        'K2_list': K2_list.tolist(),
        'sigma': sigma,
        'N': N,
        'T_equil': T_equil,
        'r_forward': r_forward,
        'r_backward': r_backward,
    }
    with open('hysteresis.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved hysteresis.json")


if __name__ == '__main__':
    hysteresis_scan()
