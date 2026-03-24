"""
Re-entrant 同步边界的精确计算

核心方程: dr/dt = -Delta*r + r(1-r^2)/2 * (K2 + K3*r^2)

Re-entrant 条件: 对固定 K2, 增加 K3 时 r* 先增后减
=> dr*/dK3 = 0 的临界点

同时计算:
1. 鞍结分岔线 (saddle-node): 同步态消失的边界
2. 滞后区域边界 (hysteresis): 正向/反向扫描 r 不同
3. 完整的分岔图
"""

import numpy as np
from scipy.optimize import brentq, fsolve
import json


def delta_from_sigma(sigma):
    return sigma * np.sqrt(2 * np.pi) / np.pi


def fp_equation(r, K2, K3, Delta):
    """不动点条件: F(r) = Delta - (1-r^2)/2*(K2+K3*r^2) = 0"""
    return Delta - (1 - r**2) / 2 * (K2 + K3 * r**2)


def fp_equation_dr(r, K2, K3, Delta):
    """dF/dr"""
    return r * (K2 + K3 * r**2) + (1 - r**2) * K3 * r - r * (K2 + K3 * r**2)
    # 更准确地:
    # F(r) = Delta - (1-r^2)/2*(K2+K3*r^2)
    # dF/dr = r*(K2+K3*r^2) - (1-r^2)*K3*r
    #       = r*[K2 + K3*r^2 - K3 + K3*r^2]
    #       = r*[K2 + 2*K3*r^2 - K3]


def dF_dr(r, K2, K3, Delta):
    """正确的 dF/dr"""
    return r * (K2 + 2 * K3 * r**2 - K3)


def saddle_node_boundary(K2_range, sigma):
    """
    鞍结分岔线: F(r)=0 且 dF/dr=0 同时成立
    即同步不动点刚好消失的边界

    联立:
      Delta = (1-r^2)/2*(K2+K3*r^2)
      0 = r*(K2 + 2*K3*r^2 - K3)  => K2 = K3 - 2*K3*r^2 = K3(1-2r^2)

    从第二个方程: K2 = K3*(1-2*r^2), 即 r^2 = (K3-K2)/(2*K3)
    代入第一个方程可得 K3 关于 K2 的参数曲线
    """
    Delta = delta_from_sigma(sigma)
    boundary_K3 = []

    for K2 in K2_range:
        if K2 <= 2 * Delta:
            boundary_K3.append(None)
            continue

        # 参数扫描 r ∈ (0, 1) 找满足两个方程的 K3
        best_K3 = None
        for r_try in np.linspace(0.01, 0.99, 500):
            # 从 dF/dr=0: K2 + 2*K3*r^2 - K3 = 0 => K3 = K2/(1-2*r^2)
            denom = 1 - 2 * r_try**2
            if abs(denom) < 1e-10 or denom <= 0:
                continue
            K3_cand = K2 / denom
            # 检查 F(r)=0
            F_val = fp_equation(r_try, K2, K3_cand, Delta)
            if abs(F_val) < 0.01:
                if best_K3 is None or K3_cand < best_K3:
                    best_K3 = K3_cand
        boundary_K3.append(best_K3)

    return boundary_K3


def reentrant_locus(K2_range, sigma):
    """
    Re-entrant 边界: dr*/dK3 变号的临界 K3 值

    dr*/dK3 = r*^2(1-r*^2) / [2*D]
    其中 D = (K2+3*K3*r*^2)*(1-r*^2) - 2*r*^2*(K2+K3*r*^2)
    D = 0 时 dr*/dK3 发散 => re-entrant 点
    """
    Delta = delta_from_sigma(sigma)
    results = []

    for K2 in K2_range:
        if K2 <= 2 * Delta:
            results.append({'K2': float(K2), 'K3_reentrant': None, 'r_at_reentrant': None})
            continue

        # 扫描 K3, 找 D 变号的点
        K3_range = np.linspace(0, K2 * 2, 200)
        prev_D = None
        found = None

        for K3 in K3_range:
            # 先找稳定不动点 r*
            r_star = find_stable_fp(K2, K3, Delta)
            if r_star is None or r_star < 0.01:
                continue

            r2 = r_star**2
            D = (K2 + 3 * K3 * r2) * (1 - r2) - 2 * r2 * (K2 + K3 * r2)

            if prev_D is not None and prev_D * D < 0:
                found = float(K3)
                found_r = float(r_star)
                break
            prev_D = D

        results.append({
            'K2': float(K2),
            'K3_reentrant': found,
            'r_at_reentrant': found_r if found else None
        })

    return results


def find_stable_fp(K2, K3, Delta):
    """找最大的稳定不动点"""
    r_grid = np.linspace(0.01, 0.99, 500)
    F_vals = [fp_equation(r, K2, K3, Delta) for r in r_grid]

    fps = []
    for i in range(len(F_vals) - 1):
        if F_vals[i] * F_vals[i + 1] < 0:
            try:
                r_fp = brentq(fp_equation, r_grid[i], r_grid[i + 1],
                              args=(K2, K3, Delta))
                fps.append(r_fp)
            except ValueError:
                pass

    if not fps:
        return None

    # 检查稳定性, 返回最大的稳定不动点
    stable = []
    for r_fp in fps:
        dr = 1e-6
        from analytical_phase_diagram import rdot
        slope = (rdot(r_fp + dr, K2, K3, Delta) - rdot(r_fp - dr, K2, K3, Delta)) / (2 * dr)
        if slope < 0:
            stable.append(r_fp)

    return max(stable) if stable else None


def bifurcation_diagram_K3(K2_fixed, K3_range, sigma):
    """
    固定 K2, 扫描 K3 的分岔图
    记录所有不动点（稳定和不稳定）
    """
    Delta = delta_from_sigma(sigma)
    all_fps = []

    for K3 in K3_range:
        r_grid = np.linspace(0.01, 0.99, 1000)
        F_vals = [fp_equation(r, K2_fixed, K3, Delta) for r in r_grid]

        fps = []
        for i in range(len(F_vals) - 1):
            if F_vals[i] * F_vals[i + 1] < 0:
                try:
                    r_fp = brentq(fp_equation, r_grid[i], r_grid[i + 1],
                                  args=(K2_fixed, K3, Delta))
                    from analytical_phase_diagram import rdot
                    dr = 1e-6
                    slope = (rdot(r_fp + dr, K2_fixed, K3, Delta) -
                             rdot(r_fp - dr, K2_fixed, K3, Delta)) / (2 * dr)
                    fps.append({
                        'r': float(r_fp),
                        'stable': bool(slope < 0)
                    })
                except ValueError:
                    pass

        all_fps.append({
            'K3': float(K3),
            'fixed_points': fps
        })

    return all_fps


if __name__ == '__main__':
    sigma = 1.0
    Delta = delta_from_sigma(sigma)
    print(f"sigma={sigma}, Delta={Delta:.4f}, Kc={2*Delta:.4f}")

    # 1. Re-entrant 边界
    K2_range = np.linspace(2 * Delta + 0.1, 8, 40)
    print("\n=== Re-entrant 边界 ===")
    reentrant = reentrant_locus(K2_range, sigma)
    for item in reentrant:
        if item['K3_reentrant'] is not None:
            print(f"  K2={item['K2']:.2f}: K3_re={item['K3_reentrant']:.3f}, "
                  f"r*={item['r_at_reentrant']:.4f}")

    # 2. 鞍结分岔线
    print("\n=== 鞍结分岔线 ===")
    sn_boundary = saddle_node_boundary(K2_range, sigma)
    for K2, K3 in zip(K2_range, sn_boundary):
        if K3 is not None:
            print(f"  K2={K2:.2f}: K3_sn={K3:.3f}")

    # 3. 分岔图示例: K2=3.0, K3 变化
    print("\n=== 分岔图 (K2=3.0) ===")
    K3_scan = np.linspace(-2, 6, 100)
    bif = bifurcation_diagram_K3(3.0, K3_scan, sigma)

    for entry in bif:
        if entry['fixed_points']:
            fps_str = ', '.join(
                f"r={fp['r']:.3f}({'S' if fp['stable'] else 'U'})"
                for fp in entry['fixed_points']
            )
            print(f"  K3={entry['K3']:+.2f}: {fps_str}")

    # 保存结果
    output = {
        'sigma': sigma,
        'Delta': Delta,
        'reentrant_locus': reentrant,
        'saddle_node_boundary': [
            {'K2': float(K2), 'K3': float(K3) if K3 is not None else None}
            for K2, K3 in zip(K2_range, sn_boundary)
        ],
        'bifurcation_K2_3': bif,
        'sigmas_reentrant': {}
    }

    # 多个 sigma 的 re-entrant 边界
    for s in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        D = delta_from_sigma(s)
        K2r = np.linspace(2 * D + 0.1, 8, 30)
        re = reentrant_locus(K2r, s)
        output['sigmas_reentrant'][f'sigma={s}'] = re

    with open('reentrant_boundary.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n结果已保存至 reentrant_boundary.json")
