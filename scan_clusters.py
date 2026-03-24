"""
任务 M.1: 同步簇结构检测
用层次聚类分析稳态相位，检测多簇结构
"""

import numpy as np
import json
import time
from kuramoto import KuramotoHigherOrder, order_parameter, order_parameter_k


def detect_clusters(theta, threshold=0.3):
    """简单层次聚类：相位差 < threshold 归为一簇"""
    N = len(theta)
    # 相位归一化到 [0, 2π)
    theta_mod = theta % (2 * np.pi)
    # 排序
    idx = np.argsort(theta_mod)
    sorted_theta = theta_mod[idx]

    clusters = [[idx[0]]]
    for i in range(1, N):
        diff = sorted_theta[i] - sorted_theta[i-1]
        if diff < threshold:
            clusters[-1].append(idx[i])
        else:
            clusters.append([idx[i]])

    # 检查首尾是否应合并（环形）
    if len(clusters) > 1:
        wrap_diff = (2 * np.pi - sorted_theta[-1] + sorted_theta[0])
        if wrap_diff < threshold:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()

    return clusters


def scan_clusters():
    N = 200
    sigma = 1.0
    T = 1000.0
    dt = 0.01
    K2_list = [1.0, 2.0, 3.0, 4.0]
    K3_list = [-2.0, -1.0, 0.0, 1.0, 2.0]

    results = []
    t0 = time.time()

    for K2 in K2_list:
        for K3 in K3_list:
            model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=0)
            r1, _ = model.simulate(T=T, dt=dt)
            r2, _ = order_parameter_k(model.theta, k=2)

            clusters = detect_clusters(model.theta, threshold=0.3)
            n_clusters = len(clusters)
            cluster_sizes = sorted([len(c) for c in clusters], reverse=True)

            results.append({
                'K2': K2, 'K3': K3,
                'n_clusters': n_clusters,
                'cluster_sizes': cluster_sizes[:5],  # top 5
                'r1': float(r1),
                'r2': float(r2),
            })
            print(f"K₂={K2:.1f}, K₃={K3:+.1f}: {n_clusters} clusters, sizes={cluster_sizes[:3]}, r₁={r1:.4f}, r₂={r2:.4f}")

    output = {
        'N': N, 'sigma': sigma, 'T': T,
        'results': results,
    }
    with open('cluster_structure.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved cluster_structure.json")


if __name__ == '__main__':
    scan_clusters()
