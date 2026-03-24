"""
解析相图计算
基于 Ott-Antonsen 降维结果: dr/dt = -Delta*r + r(1-r^2)/2 * (K2 + K3*r^2)

输出解析曲线的JSON数据，供 HaibiPlotAnalyst 叠加在数值热力图上。
"""

import numpy as np
import json


def delta_from_sigma(sigma):
    """Gaussian -> Lorentzian 匹配: Delta = sigma*sqrt(2pi)/pi"""
    return sigma * np.sqrt(2 * np.pi) / np.pi


def rdot(r, K2, K3, Delta):
    """径向动力学 ODE 右端"""
    return -Delta * r + r * (1 - r**2) / 2 * (K2 + K3 * r**2)


def find_fixed_points(K2, K3, Delta, n_grid=1000):
    """
    在 r ∈ (0, 1) 上搜索 rdot=0 的不动点
    返回所有不动点的列表
    """
    r_grid = np.linspace(1e-6, 1 - 1e-6, n_grid)
    # 对 r>0, rdot(r)/r = -Delta + (1-r^2)/2*(K2+K3*r^2)
    # 不动点条件: Delta = (1-r^2)/2*(K2+K3*r^2)
    lhs = np.full_like(r_grid, Delta)
    rhs = (1 - r_grid**2) / 2 * (K2 + K3 * r_grid**2)
    diff = rhs - lhs

    # 找零点（符号变化处）
    fps = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            # 线性插值
            r_fp = r_grid[i] - diff[i] * (r_grid[i + 1] - r_grid[i]) / (diff[i + 1] - diff[i])
            fps.append(float(r_fp))
    return fps


def onset_boundary(K3_range, Delta):
    """
    同步 onset 边界: K2c = 2*Delta (与 K3 无关)
    """
    return np.full_like(K3_range, 2 * Delta)


def explosive_boundary(K2_range):
    """
    爆炸式同步边界: K3 = K2
    """
    return K2_range.copy()


def compute_order_parameter_analytical(K2, K3, Delta):
    """
    计算解析稳态序参量 r*
    取最大的稳定不动点
    """
    fps = find_fixed_points(K2, K3, Delta)
    if not fps:
        return 0.0

    # 检查每个不动点的稳定性
    stable_fps = []
    for r_fp in fps:
        dr = 1e-6
        slope = (rdot(r_fp + dr, K2, K3, Delta) - rdot(r_fp - dr, K2, K3, Delta)) / (2 * dr)
        if slope < 0:  # 稳定
            stable_fps.append(r_fp)

    if stable_fps:
        return max(stable_fps)
    return 0.0


def compute_analytical_phase_diagram(K2_range, K3_range, sigma):
    """
    计算完整的解析相图
    """
    Delta = delta_from_sigma(sigma)
    nK2 = len(K2_range)
    nK3 = len(K3_range)

    r_mat = np.zeros((nK2, nK3))
    n_fps_mat = np.zeros((nK2, nK3), dtype=int)

    for i, K2 in enumerate(K2_range):
        for j, K3 in enumerate(K3_range):
            fps = find_fixed_points(K2, K3, Delta)
            n_fps_mat[i, j] = len(fps)
            r_mat[i, j] = compute_order_parameter_analytical(K2, K3, Delta)

    return r_mat, n_fps_mat


def time_delay_reachable(K2_bare, tau_range):
    """
    时延可达区域: K2_eff = K2 - K2^2*tau/2, K3_eff = K2^2*tau/4
    """
    K2_eff = K2_bare - K2_bare**2 * tau_range / 2
    K3_eff = K2_bare**2 * tau_range / 4
    return K2_eff, K3_eff


def potential(r, K2, K3, Delta):
    """
    径向势能 V(r) = -∫₀ʳ rdot(r') dr'
    """
    return (Delta / 2 * r**2
            - K2 / 2 * (r**2 / 2 - r**4 / 4)
            - K3 / 2 * (r**4 / 4 - r**6 / 6))


if __name__ == '__main__':
    sigma = 1.0
    Delta = delta_from_sigma(sigma)
    print(f"sigma = {sigma}, Delta = {Delta:.4f}")
    print(f"Kc (onset) = {2 * Delta:.4f}")

    K2_range = np.linspace(0, 8, 80)
    K3_range = np.linspace(-4, 4, 80)

    print(f"\n计算解析相图: K2 x K3 = {len(K2_range)} x {len(K3_range)}")
    r_mat, n_fps = compute_analytical_phase_diagram(K2_range, K3_range, sigma)

    # 各 sigma 值的相图
    sigmas = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    all_results = {}
    for s in sigmas:
        D = delta_from_sigma(s)
        r_s, _ = compute_analytical_phase_diagram(K2_range, K3_range, s)
        all_results[f'sigma={s}'] = {
            'Delta': float(D),
            'Kc': float(2 * D),
            'r': r_s.tolist()
        }
        print(f"  sigma={s}: Delta={D:.4f}, Kc={2*D:.4f}")

    # 时延可达区域（几个 K2_bare 值）
    delay_curves = {}
    for K2b in [2.0, 3.0, 4.0, 6.0]:
        tau_r = np.linspace(0, 0.5, 100)
        K2e, K3e = time_delay_reachable(K2b, tau_r)
        mask = K2e > 0
        delay_curves[f'K2_bare={K2b}'] = {
            'K2_eff': K2e[mask].tolist(),
            'K3_eff': K3e[mask].tolist()
        }

    output = {
        'K2_range': K2_range.tolist(),
        'K3_range': K3_range.tolist(),
        'sigma_results': all_results,
        'onset_K2c': {f'sigma={s}': float(2 * delta_from_sigma(s)) for s in sigmas},
        'explosive_line': 'K3 = K2',
        'delay_reachable_curves': delay_curves,
        'n_fixed_points_sigma1': n_fps.tolist()
    }

    with open('analytical_phase_diagram.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n数据已保存至 analytical_phase_diagram.json")

    # 势能景观示例
    print("\n=== 势能景观示例 ===")
    r_plot = np.linspace(0, 0.99, 100)
    cases = [
        ('经典 (K3=0)', 3.0, 0.0),
        ('弱K3帮助', 3.0, 0.5),
        ('强K3 basin缩小', 3.0, 2.0),
        ('爆炸式 (K3>K2)', 3.0, 4.0),
    ]
    for label, K2, K3 in cases:
        V = potential(r_plot, K2, K3, Delta)
        fps = find_fixed_points(K2, K3, Delta)
        r_star = compute_order_parameter_analytical(K2, K3, Delta)
        print(f"  {label}: K2={K2}, K3={K3}, r*={r_star:.4f}, #fps={len(fps)}")
