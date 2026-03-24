"""
任务 O.1: 频率分布形状效应：Gaussian vs Lorentzian vs Uniform
"""

import numpy as np
import json
import time
from kuramoto import rk4_step, order_parameter, order_parameter_k


def generate_omega(N, sigma, dist_type, seed=0):
    rng = np.random.default_rng(seed)
    if dist_type == 'gaussian':
        return rng.normal(0, sigma, N)
    elif dist_type == 'lorentzian':
        # Lorentzian with half-width gamma such that IQR matches Gaussian
        # For Gaussian, IQR ≈ 1.349σ. For Cauchy, IQR = 2γ. So γ ≈ 0.6745σ
        gamma = 0.6745 * sigma
        return rng.standard_cauchy(N) * gamma
    elif dist_type == 'uniform':
        # U[-a, a] with std = σ → a = σ√3
        a = sigma * np.sqrt(3)
        return rng.uniform(-a, a, N)
    raise ValueError(f"Unknown dist_type: {dist_type}")


def simulate(omega, N, K2, K3, T=200.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed + 999)
    theta = rng.uniform(0, 2 * np.pi, N)
    steps = int(T / dt)
    burn = int(T * 0.1 / dt)
    r_hist = []
    for s in range(steps):
        theta = rk4_step(theta, omega, K2, K3, N, dt)
        if s >= burn:
            r, _ = order_parameter(theta)
            r_hist.append(r)
    r1 = float(np.mean(r_hist))
    r2, _ = order_parameter_k(theta, k=2)
    return r1, float(r2), theta


def scan_freq_dist():
    N = 200
    sigma = 1.0
    T = 200.0
    K2_list = np.linspace(0, 6, 25)
    K3_values = [-1.0, 0.0, 1.0]
    dist_types = ['gaussian', 'lorentzian', 'uniform']

    all_results = []
    t0 = time.time()

    for dist in dist_types:
        omega = generate_omega(N, sigma, dist, seed=0)
        # Clip Lorentzian outliers for stability
        if dist == 'lorentzian':
            omega = np.clip(omega, -10 * sigma, 10 * sigma)

        for K3 in K3_values:
            r1_arr = []
            r2_arr = []
            basin_arr = []

            for K2 in K2_list:
                r1, r2, _ = simulate(omega, N, K2, K3, T=T)

                # basin prob: 5 trials
                succ = 0
                for trial in range(5):
                    r_t, _, _ = simulate(omega, N, K2, K3, T=T, seed=trial*42+100)
                    if r_t > 0.5:
                        succ += 1
                basin = succ / 5.0

                r1_arr.append(r1)
                r2_arr.append(r2)
                basin_arr.append(basin)

            # Find Kc
            Kc = None
            for i in range(len(K2_list) - 1):
                if r1_arr[i] < 0.15 and r1_arr[i+1] > 0.15:
                    frac = (0.15 - r1_arr[i]) / (r1_arr[i+1] - r1_arr[i])
                    Kc = float(K2_list[i] + frac * (K2_list[i+1] - K2_list[i]))
                    break

            all_results.append({
                'dist_type': dist,
                'K3': K3,
                'K2_list': K2_list.tolist(),
                'r1': r1_arr,
                'r2': r2_arr,
                'basin_prob': basin_arr,
                'Kc_measured': Kc,
            })
            print(f"{dist}, K₃={K3:+.1f}: Kc={Kc if Kc else 'N/A'}")

    elapsed = time.time() - t0
    output = {
        'N': N, 'sigma': sigma, 'T': T,
        'results': all_results,
    }
    with open('freq_distribution.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDone in {elapsed/60:.1f}min. Saved freq_distribution.json")


if __name__ == '__main__':
    scan_freq_dist()
