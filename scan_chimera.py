"""
任务 P.1: Chimera 态检测
高阶耦合是否在全连接网络中诱导空间异质同步？
"""

import numpy as np
import json
import time
from kuramoto import rk4_step, order_parameter


def local_order_parameter(theta, M=50):
    """局部序参量：滑窗 M 个邻居"""
    N = len(theta)
    r_local = np.zeros(N)
    for i in range(N):
        indices = [(i + j - M // 2) % N for j in range(M)]
        z = np.mean(np.exp(1j * theta[indices]))
        r_local[i] = np.abs(z)
    return r_local


def detect_chimera(r_local, threshold_high=0.8, threshold_low=0.3):
    """检测 Chimera：同时存在高同步和低同步区域"""
    high = np.mean(r_local > threshold_high)
    low = np.mean(r_local < threshold_low)
    return bool(high > 0.1 and low > 0.1)


def scan_chimera():
    N = 500
    sigma = 1.0
    T = 500.0
    dt = 0.01
    K2_list = [2.0, 3.0, 4.0]
    K3_list = [-2.0, -1.0, 0.0, 1.0, 2.0]

    rng = np.random.default_rng(0)
    omega = rng.normal(0, sigma, N)

    results = []
    t0 = time.time()
    total = len(K2_list) * len(K3_list)
    done = 0

    for K2 in K2_list:
        for K3 in K3_list:
            # 非随机初始条件：前半同步，后半随机
            theta = np.zeros(N)
            theta[:N//2] = rng.normal(0, 0.1, N//2)
            theta[N//2:] = rng.uniform(0, 2 * np.pi, N - N//2)

            steps = int(T / dt)
            for _ in range(steps):
                theta = rk4_step(theta, omega, K2, K3, N, dt)

            r_global, _ = order_parameter(theta)
            r_local = local_order_parameter(theta, M=50)
            chimera = detect_chimera(r_local)

            results.append({
                'K2': K2, 'K3': K3,
                'theta_final': theta.tolist(),
                'r_local': r_local.tolist(),
                'r_global': float(r_global),
                'chimera_detected': chimera,
                'r_local_mean': float(np.mean(r_local)),
                'r_local_std': float(np.std(r_local)),
            })
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            tag = " *** CHIMERA ***" if chimera else ""
            print(f"K₂={K2}, K₃={K3:+.1f}: r_global={r_global:.4f}, "
                  f"r_local μ={np.mean(r_local):.4f} σ={np.std(r_local):.4f}{tag}  "
                  f"[{done}/{total}] ETA {eta/60:.1f}min")

    output = {
        'N': N, 'sigma': sigma, 'T': T, 'M_local': 50,
        'results': results,
    }
    with open('chimera.json', 'w') as f:
        json.dump(output, f)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved chimera.json")


if __name__ == '__main__':
    scan_chimera()
