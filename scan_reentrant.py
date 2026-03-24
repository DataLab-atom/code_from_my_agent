"""
任务 R.1: 验证重入同步 (reentrant synchronization)
OA 预测: 固定 K₂ 略高于 Kc，增大 K₃ 时 r 先升后降
这是论文最强原创预测！
"""

import numpy as np
import json
import time
from kuramoto import KuramotoHigherOrder, order_parameter_k


def scan_reentrant():
    N = 200
    sigma = 1.0
    T = 300.0
    dt = 0.01
    Kc = 2.0 * sigma * np.sqrt(2 * np.pi) / np.pi  # ≈1.596

    K2_factors = [1.05, 1.1, 1.2, 1.5, 2.0]
    K2_list = [f * Kc for f in K2_factors]
    K3_list = np.linspace(0, 5, 50)

    all_r1 = []
    all_r2 = []
    all_basin = []
    K3_peaks = []
    K3_reentrant = []

    t0 = time.time()
    total = len(K2_list) * len(K3_list)
    done = 0

    for ki, K2 in enumerate(K2_list):
        r1_arr = []
        r2_arr = []
        basin_arr = []

        for K3 in K3_list:
            # 10 个随机初始条件取平均
            r1_trials = []
            for trial in range(10):
                model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=trial)
                r, _ = model.simulate(T=T, dt=dt)
                r1_trials.append(r)
            r1_mean = float(np.mean(r1_trials))

            model_last = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=0)
            model_last.simulate(T=T, dt=dt)
            r2, _ = order_parameter_k(model_last.theta, k=2)

            # basin prob
            succ = sum(1 for r in r1_trials if r > 0.5)
            basin = succ / 10.0

            r1_arr.append(r1_mean)
            r2_arr.append(float(r2))
            basin_arr.append(basin)
            done += 1

        all_r1.append(r1_arr)
        all_r2.append(r2_arr)
        all_basin.append(basin_arr)

        # 找 peak 和 reentrant point
        r1_np = np.array(r1_arr)
        peak_idx = np.argmax(r1_np)
        K3_peak = float(K3_list[peak_idx])

        # reentrant: r₁ 从 peak 之后降到 < 0.1
        K3_re = None
        for i in range(peak_idx, len(K3_list)):
            if r1_np[i] < 0.1:
                K3_re = float(K3_list[i])
                break

        K3_peaks.append(K3_peak)
        K3_reentrant.append(K3_re)

        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(f"K₂={K2:.3f} ({K2/Kc:.2f}×Kc): r₁_max={r1_np[peak_idx]:.4f} at K₃={K3_peak:.2f}, "
              f"reentrant K₃={K3_re}  [{done}/{total}] ETA {eta/60:.1f}min")

    output = {
        'Kc': float(Kc),
        'K2_list': [float(k) for k in K2_list],
        'K2_factors': K2_factors,
        'K3_list': K3_list.tolist(),
        'r1': all_r1,
        'r2': all_r2,
        'basin_prob': all_basin,
        'K3_peak': K3_peaks,
        'K3_reentrant': K3_reentrant,
        'N': N, 'sigma': sigma, 'T': T,
    }
    with open('reentrant_sync.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved reentrant_sync.json")


if __name__ == '__main__':
    scan_reentrant()
