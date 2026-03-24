"""
任务 J.1: 四体耦合 K₄ 的边际效应
dθᵢ/dt = ωᵢ + (K₂/N)Σⱼsin(θⱼ−θᵢ) + (K₃/N²)ΣⱼΣₖsin(2θⱼ−θₖ−θᵢ)
       + (K₄/N³)ΣⱼΣₖΣₗsin(3θⱼ−θₖ−θₗ−θᵢ)

四体项均场分解:
sin(3θⱼ−θₖ−θₗ−θᵢ) = sin(3θⱼ)cos(θₖ+θₗ+θᵢ) − cos(3θⱼ)sin(θₖ+θₗ+θᵢ)

Σₖ Σₗ cos(θₖ+θₗ+θᵢ) = cos(θᵢ)·(C₁²−S₁²) − sin(θᵢ)·2S₁C₁
  上式利用 Σₖcos(θₖ) = C₁, Σₖsin(θₖ) = S₁
  cos(θₖ+θₗ) = cos(θₖ)cos(θₗ)−sin(θₖ)sin(θₗ)
  Σₖ Σₗ cos(θₖ+θₗ) = C₁²−S₁²

实际实现用 N=50 直接 O(N⁴) 也可以，但均场分解更快
"""

import numpy as np
import json
import time
from numba import njit
from kuramoto import order_parameter, order_parameter_k


@njit
def dtheta_K4(theta, omega, K2, K3, K4, N):
    """含四体项的相位导数，均场分解 O(N)"""
    S1 = 0.0; C1 = 0.0
    S2 = 0.0; C2 = 0.0
    S3 = 0.0; C3 = 0.0
    for j in range(N):
        S1 += np.sin(theta[j])
        C1 += np.cos(theta[j])
        S2 += np.sin(2.0 * theta[j])
        C2 += np.cos(2.0 * theta[j])
        S3 += np.sin(3.0 * theta[j])
        C3 += np.cos(3.0 * theta[j])

    dtheta_dt = np.empty(N)
    for i in range(N):
        si = np.sin(theta[i])
        ci = np.cos(theta[i])

        # 两两耦合
        pair = (K2 / N) * (S1 * ci - C1 * si)

        # 三体耦合
        sum_cos_ki = C1 * ci - S1 * si
        sum_sin_ki = S1 * ci + C1 * si
        triplet = (K3 / (N * N)) * (S2 * sum_cos_ki - C2 * sum_sin_ki)

        # 四体耦合: Σⱼ Σₖ Σₗ sin(3θⱼ − θₖ − θₗ − θᵢ)
        # = S₃ · Σₖ Σₗ cos(θₖ+θₗ+θᵢ) − C₃ · Σₖ Σₗ sin(θₖ+θₗ+θᵢ)
        # Σₖ Σₗ cos(θₖ+θₗ) = C₁²−S₁²
        # Σₖ Σₗ sin(θₖ+θₗ) = 2·S₁·C₁
        cos_sum2 = C1 * C1 - S1 * S1  # Σₖₗ cos(θₖ+θₗ)
        sin_sum2 = 2.0 * S1 * C1      # Σₖₗ sin(θₖ+θₗ)
        # cos(θₖ+θₗ+θᵢ) = cos_sum2·ci − sin_sum2·si
        # sin(θₖ+θₗ+θᵢ) = sin_sum2·ci + cos_sum2·si
        cos_kli = cos_sum2 * ci - sin_sum2 * si
        sin_kli = sin_sum2 * ci + cos_sum2 * si
        quad = (K4 / (N * N * N)) * (S3 * cos_kli - C3 * sin_kli)

        dtheta_dt[i] = omega[i] + pair + triplet + quad
    return dtheta_dt


@njit
def rk4_step_K4(theta, omega, K2, K3, K4, N, dt):
    k1 = dtheta_K4(theta, omega, K2, K3, K4, N)
    k2 = dtheta_K4(theta + 0.5*dt*k1, omega, K2, K3, K4, N)
    k3 = dtheta_K4(theta + 0.5*dt*k2, omega, K2, K3, K4, N)
    k4 = dtheta_K4(theta + dt*k3, omega, K2, K3, K4, N)
    return theta + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def simulate_K4(N, sigma, K2, K3, K4, T=200.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, sigma, N)
    theta = rng.uniform(0, 2*np.pi, N)
    steps = int(T / dt)
    burn = int(T * 0.1 / dt)
    r_history = []
    for s in range(steps):
        theta = rk4_step_K4(theta, omega, K2, K3, K4, N, dt)
        if s >= burn:
            r, _ = order_parameter(theta)
            r_history.append(r)
    r1 = float(np.mean(r_history))
    r2, _ = order_parameter_k(theta, k=2)
    r3, _ = order_parameter_k(theta, k=3)
    return r1, float(r2), float(r3)


def scan_K4():
    N = 50
    sigma = 1.0
    K2 = 2.0
    K3 = 0.5
    K4_list = np.linspace(-1, 1, 20)

    # 基准: K₃ 对 r₁ 的影响
    print("=== K₃ baseline (K₄=0) ===")
    r1_K3_0, _, _ = simulate_K4(N, sigma, K2, 0.0, 0.0)
    r1_K3_05, _, _ = simulate_K4(N, sigma, K2, 0.5, 0.0)
    r1_K3_10, _, _ = simulate_K4(N, sigma, K2, 1.0, 0.0)
    print(f"  K₃=0.0 → r₁={r1_K3_0:.4f}")
    print(f"  K₃=0.5 → r₁={r1_K3_05:.4f}")
    print(f"  K₃=1.0 → r₁={r1_K3_10:.4f}")
    K3_effect = abs(r1_K3_10 - r1_K3_0)
    print(f"  K₃ effect (0→1): Δr₁={K3_effect:.4f}")

    print("\n=== K₄ scan (K₂=2, K₃=0.5) ===")
    r1_list, r2_list, r3_list = [], [], []
    t0 = time.time()
    for i, K4 in enumerate(K4_list):
        r1, r2, r3 = simulate_K4(N, sigma, K2, K3, K4)
        r1_list.append(r1)
        r2_list.append(r2)
        r3_list.append(r3)
        print(f"  K₄={K4:+.3f} → r₁={r1:.4f}, r₂={r2:.4f}, r₃={r3:.4f}")

    K4_effect = max(r1_list) - min(r1_list)
    print(f"\nK₄ effect (full range): Δr₁={K4_effect:.4f}")
    print(f"K₃ effect (0→1):       Δr₁={K3_effect:.4f}")
    print(f"Ratio K₄/K₃ effect: {K4_effect/K3_effect:.4f}" if K3_effect > 0 else "")
    negligible = K4_effect < 0.1 * K3_effect
    print(f"Conclusion: K₄ is {'NEGLIGIBLE' if negligible else 'NOT negligible'} compared to K₃")

    result = {
        'K4_list': K4_list.tolist(),
        'K2': K2, 'K3': K3, 'sigma': sigma, 'N': N,
        'r1': r1_list, 'r2': r2_list, 'r3': r3_list,
        'K3_effect': K3_effect, 'K4_effect': K4_effect,
        'negligible': negligible,
    }
    with open('K4_test.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Saved K4_test.json")


if __name__ == '__main__':
    scan_K4()
