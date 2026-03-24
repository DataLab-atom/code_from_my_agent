"""
拓扑验证: 稀疏 Erdos-Renyi 网络上 K₃ 效应
假说: K₃>0 在全连接网络帮助同步，在稀疏网络可能阻碍
"""

import numpy as np
import json
import time
from numba import njit


@njit
def dtheta_sparse(theta, omega, adj, tri_list, K2, K3, N, n_edges, n_tri):
    """稀疏网络上的相位导数"""
    dtheta_dt = np.zeros(N)
    for i in range(N):
        dtheta_dt[i] = omega[i]

    # 两两耦合 (只在边上)
    for e in range(n_edges):
        i, j = adj[e, 0], adj[e, 1]
        dtheta_dt[i] += (K2 / N) * np.sin(theta[j] - theta[i])
        dtheta_dt[j] += (K2 / N) * np.sin(theta[i] - theta[j])

    # 三体耦合 (只在三角形上)
    for t in range(n_tri):
        i, j, k = tri_list[t, 0], tri_list[t, 1], tri_list[t, 2]
        # 每个三角形贡献给 3 个顶点
        dtheta_dt[i] += (K3 / (N * N)) * np.sin(2 * theta[j] - theta[k] - theta[i])
        dtheta_dt[i] += (K3 / (N * N)) * np.sin(2 * theta[k] - theta[j] - theta[i])
        dtheta_dt[j] += (K3 / (N * N)) * np.sin(2 * theta[i] - theta[k] - theta[j])
        dtheta_dt[j] += (K3 / (N * N)) * np.sin(2 * theta[k] - theta[i] - theta[j])
        dtheta_dt[k] += (K3 / (N * N)) * np.sin(2 * theta[i] - theta[j] - theta[k])
        dtheta_dt[k] += (K3 / (N * N)) * np.sin(2 * theta[j] - theta[i] - theta[k])
    return dtheta_dt


def generate_er_graph(N, p, seed=0):
    """生成 Erdos-Renyi 随机图，返回边列表和三角形列表"""
    rng = np.random.default_rng(seed)
    adj_matrix = rng.random((N, N)) < p
    adj_matrix = np.triu(adj_matrix, k=1)

    edges = np.argwhere(adj_matrix).astype(np.int32)
    adj_set = set(map(tuple, edges))

    # 找三角形
    triangles = []
    for e in range(len(edges)):
        i, j = edges[e]
        for k in range(j + 1, N):
            if (i, k) in adj_set and (j, k) in adj_set:
                triangles.append([i, j, k])

    tri_array = np.array(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)
    return edges, tri_array


def simulate_sparse(N, sigma, K2, K3, p, T=100.0, dt=0.01, seed=0):
    edges, triangles = generate_er_graph(N, p, seed=seed + 10000)
    n_edges = len(edges)
    n_tri = len(triangles)

    rng = np.random.default_rng(seed)
    omega = rng.normal(0, sigma, N)
    theta = rng.uniform(0, 2 * np.pi, N)

    steps = int(T / dt)
    burn = int(steps * 0.1)
    r_sum = 0.0
    cnt = 0

    for s in range(steps):
        f = dtheta_sparse(theta, omega, edges, triangles, K2, K3, N, n_edges, n_tri)
        theta = theta + dt * f  # Euler
        if s >= burn:
            z = np.sum(np.exp(1j * theta)) / N
            r_sum += np.abs(z)
            cnt += 1

    return r_sum / cnt if cnt else 0, n_edges, n_tri


def scan_sparse():
    N = 200
    sigma = 1.0
    K2 = 2.0
    T = 100.0
    p_list = [0.1, 0.3, 0.5, 1.0]
    K3_list = [-1.0, 0.0, 0.5, 1.0, 1.5]
    n_seeds = 5

    results = []
    t0 = time.time()

    for p in p_list:
        for K3 in K3_list:
            rs = []
            for seed in range(n_seeds):
                r, ne, nt = simulate_sparse(N, sigma, K2, K3, p, T=T, seed=seed)
                rs.append(r)
            r_mean = float(np.mean(rs))
            results.append({'p': p, 'K3': K3, 'r_mean': r_mean, 'r_std': float(np.std(rs))})
            print(f"p={p:.1f}, K₃={K3:+.1f}: r={r_mean:.4f}±{np.std(rs):.4f}")

    output = {'N': N, 'sigma': sigma, 'K2': K2, 'T': T, 'results': results}
    with open('sparse_network_K3.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min")


if __name__ == '__main__':
    scan_sparse()
