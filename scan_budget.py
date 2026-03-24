"""
任务 E.1: 耦合预算约束下的最优分配
K₂ + K₃ = C, 固定 N=200
扫描 C ∈ {1,2,3,4,5}, σ ∈ {0.5, 1.0, 1.5}
"""

import numpy as np
import json
import time
from kuramoto import KuramotoHigherOrder, order_parameter_k


def scan_budget():
    N = 200
    T = 200.0
    dt = 0.01
    C_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    sigma_list = [0.5, 1.0, 1.5]
    n_points = 50

    all_results = []
    t0 = time.time()
    total = len(C_list) * len(sigma_list) * n_points
    done = 0

    for C in C_list:
        for sigma in sigma_list:
            K2_line = np.linspace(0, C, n_points)
            r1_arr = []
            r2_arr = []
            basin_arr = []

            for K2 in K2_line:
                K3 = C - K2
                model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=0)
                r1, _ = model.simulate(T=T, dt=dt)
                r2, _ = order_parameter_k(model.theta, k=2)

                successes = 0
                for trial in range(10):
                    model.reset_theta(seed=trial * 42 + 12345)
                    r_t, _ = model.simulate(T=T, dt=dt)
                    if r_t > 0.5:
                        successes += 1
                basin = successes / 10.0

                r1_arr.append(float(r1))
                r2_arr.append(float(r2))
                basin_arr.append(float(basin))
                done += 1

            # 找最优 K₂*
            best_idx = int(np.argmax(r1_arr))
            K2_optimal = float(K2_line[best_idx])

            all_results.append({
                'C': C,
                'sigma': sigma,
                'K2_list': K2_line.tolist(),
                'r1': r1_arr,
                'r2': r2_arr,
                'basin_prob': basin_arr,
                'K2_optimal': K2_optimal,
                'K3_optimal': C - K2_optimal,
                'r1_at_optimal': r1_arr[best_idx],
            })

            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"C={C}, σ={sigma}: K₂*={K2_optimal:.2f}, K₃*={C-K2_optimal:.2f}, r₁*={r1_arr[best_idx]:.4f}  [{done}/{total}] ETA {eta/60:.1f}min")

    output = {
        'C_list': C_list,
        'sigma_list': sigma_list,
        'N': N,
        'T': T,
        'n_points': n_points,
        'results': all_results,
    }
    with open('budget_constraint.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved budget_constraint.json")


if __name__ == '__main__':
    scan_budget()
