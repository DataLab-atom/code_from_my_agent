"""
验证 Ott-Antonsen 降维结果 (HaibiMathAgent 推导)

OA 径向 ODE: dr/dt = -Δ·r + r(1-r²)/2 · (K₂ + K₃·r²)
稳态: 0 = -Δ + (1-r²)/2 · (K₂ + K₃·r²)

临界耦合 (onset): K₂c = 2Δ (独立于 K₃，领头阶)
爆炸式同步边界: K₃ ≈ K₂

Δ = γ (Lorentzian), Δ ≈ σ√(π/2) (Gaussian 近似)
"""

import numpy as np
import json
from kuramoto import KuramotoHigherOrder


def oa_steady_r(K2, K3, Delta):
    """
    求 OA 径向方程的稳态解
    0 = -Δ + (1-r²)/2 · (K₂ + K₃·r²)
    => K₃·r⁴ - (K₂+K₃)·r² + (K₂ - 2Δ) = 0
    令 u = r², 解二次方程
    """
    a = K3
    b = -(K2 + K3)
    c = K2 - 2 * Delta

    if abs(a) < 1e-12:
        # K₃ ≈ 0, 退化为线性
        if abs(b) < 1e-12:
            return 0.0
        u = -c / b
        if u > 0 and u < 1:
            return np.sqrt(u)
        return 0.0

    disc = b**2 - 4 * a * c
    if disc < 0:
        return 0.0

    u1 = (-b + np.sqrt(disc)) / (2 * a)
    u2 = (-b - np.sqrt(disc)) / (2 * a)

    # 取物理解: 0 < u < 1
    solutions = []
    for u in [u1, u2]:
        if 0 < u < 1:
            solutions.append(np.sqrt(u))

    if not solutions:
        return 0.0
    return max(solutions)  # 取最大稳定解


def oa_Kc(Delta):
    """临界耦合强度"""
    return 2 * Delta


def gaussian_Delta(sigma):
    """Gaussian 频率分布的等效 Δ = σ√(2π)/π (matching Kc = 2/(πg(0)))"""
    return sigma * np.sqrt(2 * np.pi) / np.pi


def scan_oa_vs_numerical(K2_range, K3_range, sigma=1.0, N=200, T=100.0):
    """对比 OA 预测和数值模拟"""
    Delta = gaussian_Delta(sigma)
    Kc = oa_Kc(Delta)
    print(f"σ={sigma}, Δ={Delta:.4f}, Kc_OA={Kc:.4f}")

    results = []
    for K2 in K2_range:
        for K3 in K3_range:
            r_oa = oa_steady_r(K2, K3, Delta)
            model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K2, K3=K3, seed=0)
            r_num, _ = model.simulate(T=T, dt=0.01)
            results.append({
                'K2': float(K2), 'K3': float(K3),
                'r_oa': float(r_oa), 'r_num': float(r_num),
                'error': float(abs(r_oa - r_num))
            })
            print(f"  K₂={K2:.2f}, K₃={K3:+.2f}: r_OA={r_oa:.4f}, r_num={r_num:.4f}, err={abs(r_oa-r_num):.4f}")

    return results


if __name__ == '__main__':
    sigma = 1.0
    Delta = gaussian_Delta(sigma)
    Kc = oa_Kc(Delta)

    print("=" * 60)
    print("OA 降维验证: 解析 vs 数值模拟")
    print("=" * 60)
    print(f"σ = {sigma}")
    print(f"Δ (Gaussian approx) = {Delta:.4f}")
    print(f"Kc (OA onset) = {Kc:.4f}")
    print(f"爆炸式同步边界: K₃ ≈ K₂")
    print()

    # 测试1: K₃=0, 扫描 K₂ 验证经典 Kc
    print("--- Test 1: K₃=0, scan K₂ ---")
    K2_test = np.linspace(0.5, 4.0, 15)
    for K2 in K2_test:
        r_oa = oa_steady_r(K2, 0.0, Delta)
        model = KuramotoHigherOrder(N=500, sigma=sigma, K2=K2, K3=0.0, seed=0)
        r_num, _ = model.simulate(T=200.0, dt=0.01)
        print(f"  K₂={K2:.2f}: r_OA={r_oa:.4f}, r_num={r_num:.4f}")

    # 测试2: 固定 K₂=3.0, 扫描 K₃
    print("\n--- Test 2: K₂=3.0, scan K₃ ---")
    K3_test = np.linspace(-2, 2, 15)
    for K3 in K3_test:
        r_oa = oa_steady_r(3.0, K3, Delta)
        model = KuramotoHigherOrder(N=500, sigma=sigma, K2=3.0, K3=K3, seed=0)
        r_num, _ = model.simulate(T=200.0, dt=0.01)
        print(f"  K₃={K3:+.2f}: r_OA={r_oa:.4f}, r_num={r_num:.4f}")

    # 测试3: 多个 σ 的 Kc 对比
    print("\n--- Test 3: Kc vs σ ---")
    sigmas = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    kc_results = []
    for s in sigmas:
        D = gaussian_Delta(s)
        Kc_oa = oa_Kc(D)
        # 数值搜索 Kc
        Kc_num = None
        for K2 in np.linspace(0.1, 8.0, 40):
            model = KuramotoHigherOrder(N=500, sigma=s, K2=K2, K3=0.0, seed=0)
            r, _ = model.simulate(T=200.0, dt=0.01)
            if r > 0.15:
                Kc_num = K2
                break
        kc_results.append({'sigma': float(s), 'Kc_OA': float(Kc_oa),
                           'Kc_num': float(Kc_num) if Kc_num else None})
        print(f"  σ={s:.1f}: Kc_OA={Kc_oa:.4f}, Kc_num={Kc_num if Kc_num else 'N/A'}")

    # 保存
    with open('verify_oa_results.json', 'w') as f:
        json.dump({'sigma': sigma, 'Delta': Delta, 'Kc_OA': Kc,
                   'Kc_vs_sigma': kc_results}, f, indent=2)
    print("\nSaved verify_oa_results.json")
