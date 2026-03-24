"""
OA 伪势 V(r) 分析
解释为什么 K₃>0 增强临界慢化

V(r) = ∫₀ʳ [Δs - s(1-s²)/2·(K₂+K₃s²)] ds
     = Δr²/2 - K₂r²/4 + (K₂+K₃)r⁴/8 - K₃r⁶/12

关键：V''(0) = Δ - K₂/2，与 K₃ 无关
但 K₃ 改变高阶项，使势阱形状变平 → 临界慢化增强
"""

import numpy as np
import json


def compute_pseudopotential(r_arr, K2, K3, Delta):
    """计算 OA 伪势 V(r)"""
    return (Delta * r_arr**2 / 2 - K2 * r_arr**2 / 4
            + (K2 + K3) * r_arr**4 / 8 - K3 * r_arr**6 / 12)


def find_fixed_points(K2, K3, Delta):
    """精确求解 OA 不动点: 0 = -Δ + (1-r²)/2·(K₂+K₃r²)"""
    # 令 u = r², 解: K₃u² - (K₂+K₃)u + (K₂-2Δ) = 0
    a = K3
    b = -(K2 + K3)
    c = K2 - 2 * Delta

    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        u = -c / b
        return [np.sqrt(u)] if 0 < u < 1 else []

    disc = b**2 - 4 * a * c
    if disc < 0:
        return []

    u1 = (-b + np.sqrt(disc)) / (2 * a)
    u2 = (-b - np.sqrt(disc)) / (2 * a)

    fps = []
    for u in sorted([u1, u2]):
        if 0 < u < 1:
            fps.append(np.sqrt(u))
    return fps


def analyze_potential_landscape():
    Delta = 0.798  # σ=1
    r_arr = np.linspace(0, 0.99, 1000)

    results = []
    print("K₂     K₃     V''(0)  Fixed points         V_barrier")
    print("-" * 65)

    for K2 in [1.4, 1.5, 1.596, 1.7, 2.0, 2.5, 3.0]:
        for K3 in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            V = compute_pseudopotential(r_arr, K2, K3, Delta)
            fps = find_fixed_points(K2, K3, Delta)
            curvature_0 = Delta - K2 / 2

            # 势垒: V(鞍点) - V(最低极小值)
            barrier = None
            if len(fps) >= 2:
                V_fps = [compute_pseudopotential(np.array([fp]), K2, K3, Delta)[0] for fp in fps]
                barrier = max(V_fps) - min(V_fps)

            fp_str = ", ".join([f"{fp:.4f}" for fp in fps])
            print(f"{K2:6.3f} {K3:6.1f} {curvature_0:+8.4f}  [{fp_str:20s}] {barrier if barrier else 'N/A':>10}")

            results.append({
                'K2': K2, 'K3': K3,
                'curvature_0': float(curvature_0),
                'fixed_points': [float(fp) for fp in fps],
                'V_barrier': float(barrier) if barrier else None,
            })

    with open('pseudopotential_analysis.json', 'w') as f:
        json.dump({'Delta': Delta, 'sigma': 1.0, 'results': results}, f, indent=2)
    print("\nSaved pseudopotential_analysis.json")


if __name__ == '__main__':
    analyze_potential_landscape()
