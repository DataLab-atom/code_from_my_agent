"""
生成小规模演示数据，供绘图开发使用
N=50, 粗网格，快速运行
"""
import numpy as np
import json
from kuramoto import KuramotoHigherOrder

def generate_demo():
    sigma_list = [0.5, 1.0, 1.5]
    K2_list = np.linspace(0, 5, 12).tolist()
    K3_list = np.linspace(-2, 2, 12).tolist()
    N = 50
    T = 50.0
    n_basin_trials = 10

    nS = len(sigma_list)
    nK2 = len(K2_list)
    nK3 = len(K3_list)

    r_mat = np.zeros((nS, nK2, nK3))
    tc_mat = np.zeros((nS, nK2, nK3))
    basin_mat = np.zeros((nS, nK2, nK3))

    total = nS * nK2 * nK3
    done = 0

    for i, sigma in enumerate(sigma_list):
        for j, K2 in enumerate(K2_list):
            for k, K3 in enumerate(K3_list):
                model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=42)
                r, tc = model.simulate(T=T, dt=0.02)
                basin = model.basin_probability(n_trials=n_basin_trials, T=T, dt=0.02)
                r_mat[i, j, k] = r
                tc_mat[i, j, k] = tc
                basin_mat[i, j, k] = basin
                done += 1
                if done % 20 == 0:
                    print(f"[{done}/{total}] σ={sigma:.1f} K₂={K2:.2f} K₃={K3:+.2f} → r={r:.3f}")

    data = dict(
        sigma_list=sigma_list,
        K2_list=K2_list,
        K3_list=K3_list,
        N=N, T=T,
        n_basin_trials=n_basin_trials,
        r=r_mat.tolist(),
        tc=tc_mat.tolist(),
        basin=basin_mat.tolist(),
    )
    with open("demo_scan_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nDone. Saved to demo_scan_data.json ({total} points)")

if __name__ == "__main__":
    generate_demo()
