"""
N=3 高阶 Kuramoto 模型精确分岔分析
验证 Dai & Kori 2025: 随耦合强度变化可出现多达9次同步转变

对于 N=3，方程可以精确写出，不需要均场近似。
利用相位差 φ₁=θ₂-θ₁, φ₂=θ₃-θ₁ 将3维系统降为2维。
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import json


def three_body_rhs(t, phi, omega_diff, K2, K3):
    """
    N=3 高阶 Kuramoto 的相位差方程
    phi = [φ₁, φ₂] = [θ₂-θ₁, θ₃-θ₁]
    omega_diff = [ω₂-ω₁, ω₃-ω₁]
    """
    p1, p2 = phi
    dw1, dw2 = omega_diff

    # 两两耦合项对 φ₁ 的贡献
    pair_1 = (K2 / 3) * (
        -2 * np.sin(p1) + np.sin(p2 - p1) - np.sin(p2) + np.sin(p1)
    )
    # 简化: (K2/3)*(-sin(p1) + sin(p2-p1) - sin(p2))
    pair_1 = (K2 / 3) * (-np.sin(p1) + np.sin(p2 - p1) - np.sin(p2))

    # 两两耦合项对 φ₂ 的贡献
    pair_2 = (K2 / 3) * (-np.sin(p2) + np.sin(p1 - p2) - np.sin(p1))

    # 三体耦合项对 φ₁: d(θ₂-θ₁)/dt 中的三体部分
    # 需要精确展开 Σ_{j,k} sin(2θⱼ-θₖ-θᵢ) 对 i=1,2
    # θ₁=0 (参考), θ₂=φ₁, θ₃=φ₂
    tri_1 = (K3 / 9) * _triplet_diff(p1, p2, idx=1)
    tri_2 = (K3 / 9) * _triplet_diff(p1, p2, idx=2)

    dp1 = dw1 + pair_1 + tri_1
    dp2 = dw2 + pair_2 + tri_2

    return [dp1, dp2]


def _triplet_sum(p1, p2, i):
    """
    计算 Σ_{j,k} sin(2θⱼ - θₖ - θᵢ) for oscillator i
    θ₁=0, θ₂=p1, θ₃=p2
    """
    thetas = [0.0, p1, p2]
    ti = thetas[i]
    s = 0.0
    for j in range(3):
        for k in range(3):
            s += np.sin(2 * thetas[j] - thetas[k] - ti)
    return s


def _triplet_diff(p1, p2, idx):
    """
    三体项对 φ_{idx} = θ_{idx+1} - θ₁ 的贡献
    = triplet_sum(i=idx) - triplet_sum(i=0)
    """
    return _triplet_sum(p1, p2, idx) - _triplet_sum(p1, p2, 0)


def order_parameter_3(phi):
    """N=3 序参量"""
    thetas = np.array([0.0, phi[0], phi[1]])
    z = np.mean(np.exp(1j * thetas))
    return np.abs(z)


def scan_bifurcation(K2_range, K3_values, omega_diff, T=500, dt=0.01,
                     n_initial=20):
    """
    对每个 (K2, K3)，从多个初始条件模拟，记录最终 r
    """
    results = {}
    for K3 in K3_values:
        r_list = []
        for K2 in K2_range:
            rs = []
            for trial in range(n_initial):
                rng = np.random.default_rng(trial * 100 + 7)
                phi0 = rng.uniform(0, 2 * np.pi, 2)
                sol = solve_ivp(
                    three_body_rhs,
                    [0, T],
                    phi0,
                    args=(omega_diff, K2, K3),
                    method='RK45',
                    max_step=dt,
                    t_eval=[T]
                )
                r = order_parameter_3(sol.y[:, -1])
                rs.append(float(r))
            r_list.append({
                'K2': float(K2),
                'r_mean': float(np.mean(rs)),
                'r_max': float(np.max(rs)),
                'r_min': float(np.min(rs)),
                'r_all': rs
            })
        results[f'K3={K3:.2f}'] = r_list
    return results


def count_transitions(r_values, threshold=0.5):
    """
    统计 r 跨越 threshold 的次数（同步转变次数）
    """
    above = np.array(r_values) > threshold
    transitions = np.sum(np.diff(above.astype(int)) != 0)
    return int(transitions)


if __name__ == '__main__':
    print("=== N=3 高阶 Kuramoto 分岔分析 ===")
    print("验证 Dai & Kori 2025: 最多 9 次同步转变\n")

    # 频率差: 不对称配置
    omega_diff = [0.5, -0.3]

    K2_range = np.linspace(0, 10, 200)
    K3_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    print(f"K₂ ∈ [0, 10], {len(K2_range)} points")
    print(f"K₃ values: {K3_values}")
    print(f"ω_diff = {omega_diff}")
    print()

    results = scan_bifurcation(K2_range, K3_values, omega_diff,
                               T=500, n_initial=30)

    # 统计转变次数
    print("\n=== 同步转变次数统计 ===")
    max_transitions = 0
    for k3_label, data in results.items():
        r_means = [d['r_mean'] for d in data]
        n_trans = count_transitions(r_means, threshold=0.5)
        max_transitions = max(max_transitions, n_trans)
        print(f"  {k3_label}: {n_trans} 次转变")

    print(f"\n最大转变次数: {max_transitions}")
    if max_transitions >= 9:
        print("✓ 验证了 Dai 2025 的 9 次转变预测！")
    else:
        print(f"当前配置最多 {max_transitions} 次，需调整 ω_diff 或 K₃ 范围")

    # 保存结果
    out = {
        'omega_diff': omega_diff,
        'K2_range': K2_range.tolist(),
        'K3_values': K3_values,
        'results': results,
        'max_transitions': max_transitions
    }
    with open('three_oscillator_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\n结果已保存至 three_oscillator_results.json")
