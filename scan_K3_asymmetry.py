"""
任务 C.1: 正负 K₃ 不对称性深度扫描
固定 σ=1, N=200, K₂=2.0, 扫描 K₃ ∈ [-3, 3], 60 个点
记录 r₁, r₂, r₁(t) 时间序列, basin_prob
"""

import numpy as np
import json
import time
from kuramoto import KuramotoHigherOrder, order_parameter_k


def scan_K3_asymmetry():
    sigma = 1.0
    N = 200
    K2 = 2.0
    T = 500.0
    dt = 0.01
    K3_list = np.linspace(-3, 3, 60)
    record_every = 100  # 每 100 步记录一次 r₁

    r1_steady = []
    r2_steady = []
    r1_timeseries = []
    basin_probs = []

    t0 = time.time()
    for idx, K3 in enumerate(K3_list):
        model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=0)
        steps = int(T / dt)
        burn_steps = int(50.0 / dt)  # 50s burn-in

        r1_ts = []
        for s in range(steps):
            model.theta = __import__('kuramoto').rk4_step(
                model.theta, model.omega, model.K2, model.K3, model.N, dt)
            if s % record_every == 0:
                r1, _ = __import__('kuramoto').order_parameter(model.theta)
                r1_ts.append(float(r1))

        # 稳态 r₁, r₂ (最后 20% 平均)
        n_last = max(1, len(r1_ts) // 5)
        r1_final = float(np.mean(r1_ts[-n_last:]))
        r2_final, _ = order_parameter_k(model.theta, k=2)
        r2_final = float(r2_final)

        # basin probability: 10 trials
        successes = 0
        for trial in range(10):
            model.reset_theta(seed=trial * 42 + 12345)
            r, _ = model.simulate(T=200.0, dt=dt)
            if r > 0.5:
                successes += 1
        basin = successes / 10.0

        r1_steady.append(r1_final)
        r2_steady.append(r2_final)
        r1_timeseries.append(r1_ts)
        basin_probs.append(basin)

        elapsed = time.time() - t0
        eta = elapsed / (idx + 1) * (len(K3_list) - idx - 1)
        print(f"[{idx+1}/{len(K3_list)}] K₃={K3:+.3f} → r₁={r1_final:.4f}, r₂={r2_final:.4f}, basin={basin:.2f}  ETA {eta/60:.1f}min")

    result = {
        'K3_list': K3_list.tolist(),
        'K2': K2,
        'sigma': sigma,
        'N': N,
        'T': T,
        'r1_steady': r1_steady,
        'r2_steady': r2_steady,
        'r1_timeseries': r1_timeseries,
        'basin_prob': basin_probs,
        'record_every_steps': record_every,
    }
    with open('K3_asymmetry.json', 'w') as f:
        json.dump(result, f)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved K3_asymmetry.json")


if __name__ == '__main__':
    scan_K3_asymmetry()
