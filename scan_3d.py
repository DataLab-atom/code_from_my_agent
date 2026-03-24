"""
σ × K₂ × K₃ 三维参数扫描
目标：研究频率分布宽度 σ 如何影响高阶 Kuramoto 同步行为
输出：scan_sigma_K2_K3.json
"""

import numpy as np
import json
import time
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from kuramoto import KuramotoHigherOrder


def run_one(args):
    """单个参数点的模拟（供并行调用）"""
    sigma, K2, K3, N, T, n_basin_trials, seed = args
    model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=seed)
    r, tc = model.simulate(T=T, dt=0.01)
    basin = model.basin_probability(n_trials=n_basin_trials, T=T, dt=0.01)
    return sigma, K2, K3, r, tc, basin


def scan_3d(
    sigma_list,
    K2_list,
    K3_list,
    N=200,
    T=100.0,
    n_basin_trials=20,
    seed=0,
    n_workers=4,
):
    """
    三维参数扫描：σ × K₂ × K₃

    返回 dict，结构与 scan_2d 兼容，增加 sigma_list 和三维矩阵
    """
    nS = len(sigma_list)
    nK2 = len(K2_list)
    nK3 = len(K3_list)

    # 预分配结果矩阵 [sigma, K2, K3]
    r_mat = np.zeros((nS, nK2, nK3))
    tc_mat = np.zeros((nS, nK2, nK3))
    basin_mat = np.zeros((nS, nK2, nK3))

    # 构建任务列表
    tasks = []
    for i, sigma in enumerate(sigma_list):
        for j, K2 in enumerate(K2_list):
            for k, K3 in enumerate(K3_list):
                tasks.append((sigma, K2, K3, N, T, n_basin_trials, seed + i * 1000 + j * 100 + k))

    total = len(tasks)
    print(f"总计 {total} 个参数点，N={N}, T={T}, basin_trials={n_basin_trials}")
    print(f"并行 workers={n_workers}")
    t0 = time.time()

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_one, t): t for t in tasks}
        for future in as_completed(futures):
            sigma, K2, K3, r, tc, basin = future.result()
            i = list(sigma_list).index(sigma)
            j = list(K2_list).index(K2)
            k = list(K3_list).index(K3)
            r_mat[i, j, k] = r
            tc_mat[i, j, k] = tc
            basin_mat[i, j, k] = basin
            completed += 1
            elapsed = time.time() - t0
            eta = elapsed / completed * (total - completed)
            print(
                f"[{completed}/{total}] σ={sigma:.2f}, K₂={K2:.2f}, K₃={K3:+.2f} "
                f"→ r={r:.4f}, tc={tc:.1f}, basin={basin:.2f}  "
                f"ETA {eta/60:.1f}min"
            )

    return dict(
        sigma_list=[float(s) for s in sigma_list],
        K2_list=[float(k) for k in K2_list],
        K3_list=[float(k) for k in K3_list],
        N=N,
        T=T,
        n_basin_trials=n_basin_trials,
        r=r_mat.tolist(),
        tc=tc_mat.tolist(),
        basin=basin_mat.tolist(),
    )


def analyze_results(data):
    """分析扫描结果，提取关键发现"""
    sigma_list = np.array(data["sigma_list"])
    K2_list = np.array(data["K2_list"])
    K3_list = np.array(data["K3_list"])
    r = np.array(data["r"])
    basin = np.array(data["basin"])

    print("\n" + "=" * 60)
    print("分析结果")
    print("=" * 60)

    # 1. 对每个 σ，找临界 K₂（K₃=0 切片）
    k3_zero_idx = np.argmin(np.abs(K3_list))
    print(f"\n[1] K₃≈0 切片下各 σ 的临界 K₂（r 从 <0.1 跨越到 >0.5）：")
    Kc_theory = lambda s: 2.0 * np.sqrt(2 * np.pi) * s
    for i, sigma in enumerate(sigma_list):
        r_slice = r[i, :, k3_zero_idx]
        # 找 r=0.3 的临界点
        cross = None
        for j in range(len(K2_list) - 1):
            if r_slice[j] < 0.3 <= r_slice[j + 1]:
                cross = K2_list[j]
                break
        Kc_th = Kc_theory(sigma)
        print(f"  σ={sigma:.2f}: Kc_numerical≈{cross if cross else 'N/A':.3f}, Kc_theory={Kc_th:.3f}")

    # 2. K₃ 对相边界的影响（各 σ 下）
    print(f"\n[2] K₃ 对同步区域的影响（r>0.5 面积占比，各 σ）：")
    for i, sigma in enumerate(sigma_list):
        sync_area_by_K3 = (r[i] > 0.5).mean(axis=0)  # shape: [nK3]
        best_k3_idx = np.argmax(sync_area_by_K3)
        worst_k3_idx = np.argmin(sync_area_by_K3)
        print(
            f"  σ={sigma:.2f}: 最优K₃={K3_list[best_k3_idx]:+.2f}(覆盖率{sync_area_by_K3[best_k3_idx]:.2f}), "
            f"最差K₃={K3_list[worst_k3_idx]:+.2f}(覆盖率{sync_area_by_K3[worst_k3_idx]:.2f})"
        )

    # 3. 是否存在临界 σ 区间（K₃ 效果方向改变）
    print(f"\n[3] K₃ 效果随 σ 的方向变化：")
    for i, sigma in enumerate(sigma_list):
        # 弱K₃(正) vs 无K₃ vs 强K₃(正) 的 r 差异
        k3_weak_idx = np.argmin(np.abs(K3_list - 0.5))
        k3_strong_idx = np.argmin(np.abs(K3_list - 1.5))
        k3_zero_i = k3_zero_idx
        r_mid = r[i, len(K2_list) // 2, :]  # 中等 K₂ 切片
        delta_weak = r_mid[k3_weak_idx] - r_mid[k3_zero_i]
        delta_strong = r_mid[k3_strong_idx] - r_mid[k3_zero_i]
        print(
            f"  σ={sigma:.2f}: 弱K₃(+0.5) Δr={delta_weak:+.4f}, 强K₃(+1.5) Δr={delta_strong:+.4f}"
        )


if __name__ == "__main__":
    # 参数设置
    sigma_list = np.array([0.3, 0.5, 0.8, 1.0, 1.2, 1.5])
    K2_list = np.linspace(0, 4, 16)
    K3_list = np.linspace(-2, 2, 16)

    print("=" * 60)
    print("σ × K₂ × K₃ 三维 Kuramoto 相图扫描")
    print("=" * 60)
    print(f"σ:  {sigma_list}")
    print(f"K₂: {len(K2_list)} points in [0, 4]")
    print(f"K₃: {len(K3_list)} points in [-2, 2]")
    print(f"总计: {len(sigma_list) * len(K2_list) * len(K3_list)} 参数点")
    print()

    results = scan_3d(
        sigma_list=sigma_list,
        K2_list=K2_list,
        K3_list=K3_list,
        N=200,
        T=100.0,
        n_basin_trials=20,
        n_workers=4,
    )

    # 保存 JSON
    out_path = "scan_sigma_K2_K3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n数据已保存至 {out_path}")

    # 分析
    analyze_results(results)
