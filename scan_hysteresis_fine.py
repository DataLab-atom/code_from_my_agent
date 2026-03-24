"""
任务 X.1: 滞后面积非单调性精细扫描
K₃ ∈ [0, 3], 30 个点
"""

import numpy as np
import json
import time
from kuramoto import rk4_step, order_parameter


def hysteresis_fine():
    sigma = 1.0
    N = 200
    T_equil = 300.0
    dt = 0.01
    K3_list = np.linspace(0, 3, 30)
    K2_list = np.linspace(0, 4, 40)

    areas = []
    r_fwd_all = []
    r_bwd_all = []
    t0 = time.time()

    for ki, K3 in enumerate(K3_list):
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, 2 * np.pi, N)
        omega = rng.normal(0, sigma, N)

        # Forward
        r_fwd = []
        for K2 in K2_list:
            for _ in range(int(T_equil / dt)):
                theta = rk4_step(theta, omega, K2, K3, N, dt)
            r, _ = order_parameter(theta)
            r_fwd.append(float(r))

        # Backward
        r_bwd = []
        for K2 in reversed(K2_list):
            for _ in range(int(T_equil / dt)):
                theta = rk4_step(theta, omega, K2, K3, N, dt)
            r, _ = order_parameter(theta)
            r_bwd.append(float(r))
        r_bwd.reverse()

        area = float(np.trapezoid(np.abs(np.array(r_fwd) - np.array(r_bwd)), K2_list))
        areas.append(area)
        r_fwd_all.append(r_fwd)
        r_bwd_all.append(r_bwd)

        elapsed = time.time() - t0
        eta = elapsed / (ki + 1) * (len(K3_list) - ki - 1)
        print(f"[{ki+1}/{len(K3_list)}] K₃={K3:.3f}: area={area:.4f}  ETA {eta/60:.1f}min")

    output = {
        'K3_list': K3_list.tolist(),
        'K2_list': K2_list.tolist(),
        'hysteresis_area': areas,
        'r_forward': r_fwd_all,
        'r_backward': r_bwd_all,
        'sigma': sigma, 'N': N, 'T_equil': T_equil,
    }
    with open('hysteresis_fine.json', 'w') as f:
        json.dump(output, f)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved hysteresis_fine.json")


if __name__ == '__main__':
    hysteresis_fine()
