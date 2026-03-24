"""
扩展版 Re-entrant 同步验证
K₃ ∈ [0, 8], K₂ ∈ {1.8, 2.0, 2.5, 3.0, 4.0}
MathAgent 预测: K2=2→K3_re≈5.4, K2=3→K3_re≈4.4, K2=4→K3_re≈3.3
"""

import numpy as np
import json
import time
from kuramoto import KuramotoHigherOrder, order_parameter_k


def scan_reentrant_extended():
    N = 200
    sigma = 1.0
    T = 300.0
    dt = 0.01

    K2_list = [1.8, 2.0, 2.5, 3.0, 4.0]
    K3_list = np.linspace(0, 8, 40)

    all_r1 = []
    all_basin = []
    t0 = time.time()
    total = len(K2_list) * len(K3_list)
    done = 0

    for K2 in K2_list:
        r1_arr = []
        basin_arr = []
        for K3 in K3_list:
            r1_trials = []
            for trial in range(10):
                model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=trial)
                r, _ = model.simulate(T=T, dt=dt)
                r1_trials.append(r)
            r1_mean = float(np.mean(r1_trials))
            basin = sum(1 for r in r1_trials if r > 0.5) / 10.0
            r1_arr.append(r1_mean)
            basin_arr.append(basin)
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] K₂={K2:.1f}, K₃={K3:.2f}: r₁={r1_mean:.4f}, basin={basin:.1f}  ETA {eta/60:.1f}min")

        all_r1.append(r1_arr)
        all_basin.append(basin_arr)

        # Find peak and reentrant
        r_np = np.array(r1_arr)
        peak_idx = np.argmax(r_np)
        K3_re = None
        for i in range(peak_idx, len(K3_list)):
            if r_np[i] < 0.1:
                K3_re = float(K3_list[i])
                break
        print(f"  K₂={K2:.1f}: r_max={r_np[peak_idx]:.4f} at K₃={K3_list[peak_idx]:.2f}, reentrant at K₃={K3_re}")

    output = {
        'K2_list': K2_list,
        'K3_list': K3_list.tolist(),
        'r1': all_r1,
        'basin_prob': all_basin,
        'N': N, 'sigma': sigma, 'T': T,
    }
    with open('reentrant_extended.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved reentrant_extended.json")


if __name__ == '__main__':
    scan_reentrant_extended()
