"""
K₂ × K₃ 参数扫描
目标：复现 Muolo 2025 "弱K₃帮助，强K₃阻碍" 结论
测量：序参量 r、收敛时间、吸引域概率
"""

import numpy as np
import json
from kuramoto import KuramotoHigherOrder


def scan_2d(K2_range, K3_range, N=200, sigma=1.0, T=100.0, seed_base=0):
    """
    二维参数扫描
    返回: dict with r_matrix, tc_matrix, basin_matrix
    """
    nK2 = len(K2_range)
    nK3 = len(K3_range)

    r_mat = np.zeros((nK2, nK3))
    tc_mat = np.zeros((nK2, nK3))
    basin_mat = np.zeros((nK2, nK3))

    for i, K2 in enumerate(K2_range):
        for j, K3 in enumerate(K3_range):
            model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=seed_base)
            r, tc = model.simulate(T=T, dt=0.01)
            basin = model.basin_probability(n_trials=30, T=T, dt=0.01)

            r_mat[i, j] = r
            tc_mat[i, j] = tc
            basin_mat[i, j] = basin

            print(f"K₂={K2:.2f}, K₃={K3:+.2f} → r={r:.4f}, tc={tc:.1f}, basin={basin:.2f}")

    return dict(
        K2_list=list(K2_range),
        K3_list=list(K3_range),
        r=r_mat.tolist(),
        tc=tc_mat.tolist(),
        basin=basin_mat.tolist()
    )


def find_phase_boundary(K2_range, K3_range, N=200, sigma=1.0, r_thresh=0.5):
    """
    找相边界: r = r_thresh 的等值线
    用于标定同步/非同步区域
    """
    boundary_points = []
    for i, K2 in enumerate(K2_range):
        for j, K3 in enumerate(K3_range):
            model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=seed_base)
            r, _ = model.simulate(T=100.0, dt=0.01)
            if abs(r - r_thresh) < 0.05:
                boundary_points.append((K2, K3, r))
    return boundary_points


if __name__ == '__main__':
    print("=== K₂ × K₃ 参数扫描 (N=200, σ=1) ===")
    K2_range = np.linspace(0, 4, 9)    # 0 to 4
    K3_range = np.linspace(-2, 2, 9)  # -2 to 2

    print(f"K₂ ∈ [{K2_range[0]}, {K2_range[-1]}], {len(K2_range)} points")
    print(f"K₃ ∈ [{K3_range[0]}, {K3_range[-1]}], {len(K3_range)} points")
    print()

    results = scan_2d(K2_range, K3_range, N=200, sigma=1.0, T=100.0)

    # 保存结果
    with open('scan_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n结果已保存至 scan_results.json")

    # 找相边界
    print("\n=== 相边界搜索 (r=0.5 等值线) ===")
    boundary = find_phase_boundary(K2_range, K3_range, r_thresh=0.5)
    for K2, K3, r in boundary[:10]:
        print(f"  K₂={K2:.2f}, K₃={K3:+.2f} → r={r:.4f}")
