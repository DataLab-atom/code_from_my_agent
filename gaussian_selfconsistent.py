"""
Gaussian frequency distribution: exact self-consistent equation for r*.

For Lorentzian g(omega), OA gives exact results.
For Gaussian, the self-consistent equation is:

  r = integral from -pi/2 to pi/2 of cos^2(theta) * g(K_eff * r * sin(theta)) dtheta

where K_eff depends on K2, K3, and r itself.

This script computes the exact phase diagram for Gaussian g(omega)
and compares with the OA (Lorentzian) approximation.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import json


def gaussian_g(omega, sigma):
    return np.exp(-omega**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def self_consistent_classical(r, K, sigma):
    """
    Classical Kuramoto self-consistent equation (Gaussian):
    r = K*r * integral_{-pi/2}^{pi/2} cos^2(theta) * g(K*r*sin(theta)) dtheta

    Returns F(r) = RHS - r = 0 for non-trivial solutions.
    """
    if r < 1e-10:
        # Linear stability: RHS ~ K*pi*g(0)/2 * r
        return K * np.pi * gaussian_g(0, sigma) / 2 - 1

    def integrand(theta):
        return np.cos(theta)**2 * gaussian_g(K * r * np.sin(theta), sigma)

    result, _ = quad(integrand, -np.pi / 2, np.pi / 2)
    rhs = K * r * result
    return rhs - r


def self_consistent_higher_order(r, K2, K3, sigma):
    """
    Higher-order Kuramoto self-consistent equation.

    In the rotating frame, the effective coupling for an oscillator
    with natural frequency omega is:
      K_eff(r) = K2 + K3 * r^2

    The self-consistent equation becomes:
      r = K_eff(r) * r * integral cos^2(theta) * g(K_eff(r)*r*sin(theta)) dtheta

    This is the Gaussian generalization of the OA result.
    """
    K_eff = K2 + K3 * r**2

    if K_eff <= 0:
        return -r  # No synchronization possible

    if r < 1e-10:
        # Linear stability (same as classical, K3 doesn't enter)
        return K2 * np.pi * gaussian_g(0, sigma) / 2 - 1

    def integrand(theta):
        return np.cos(theta)**2 * gaussian_g(K_eff * r * np.sin(theta), sigma)

    result, _ = quad(integrand, -np.pi / 2, np.pi / 2)
    rhs = K_eff * r * result
    return rhs - r


def find_r_star_gaussian(K2, K3, sigma, r_max=0.99):
    """Find non-trivial r* from Gaussian self-consistent equation"""
    # Check if incoherent state is unstable
    linear = self_consistent_higher_order(1e-8, K2, K3, sigma)
    if linear < 0:
        return 0.0  # Below threshold

    # Search for r* in (0, 1)
    try:
        # F(r) = RHS - r, we want F(r)=0 with r>0
        # At r=small, F>0 (unstable incoherent); at r~1, F<0
        r_star = brentq(
            lambda r: self_consistent_higher_order(r, K2, K3, sigma),
            0.01, r_max, xtol=1e-6
        )
        return float(r_star)
    except ValueError:
        # No crossing found, try larger range
        return 0.0


def compare_gaussian_vs_lorentzian(K2_range, K3, sigma):
    """
    Compare exact Gaussian self-consistent r* with OA (Lorentzian) prediction.
    """
    from analytical_phase_diagram import (
        delta_from_sigma, compute_order_parameter_analytical
    )
    Delta = delta_from_sigma(sigma)

    results = []
    for K2 in K2_range:
        r_gauss = find_r_star_gaussian(K2, K3, sigma)
        r_oa = compute_order_parameter_analytical(K2, K3, Delta)
        results.append({
            'K2': float(K2),
            'r_gaussian': r_gauss,
            'r_OA': r_oa,
            'diff': abs(r_gauss - r_oa)
        })
    return results


if __name__ == '__main__':
    sigma = 1.0
    print(f"=== Gaussian vs OA comparison (sigma={sigma}) ===")
    print(f"Kc_exact = {2 / (np.pi * gaussian_g(0, sigma)):.4f}")

    # Classical case (K3=0)
    K2_range = np.linspace(0, 6, 30)
    print("\n--- K3 = 0 (classical) ---")
    results_0 = compare_gaussian_vs_lorentzian(K2_range, K3=0.0, sigma=sigma)
    for r in results_0:
        if r['r_gaussian'] > 0 or r['r_OA'] > 0:
            print(f"  K2={r['K2']:.2f}: r_Gauss={r['r_gaussian']:.4f}, "
                  f"r_OA={r['r_OA']:.4f}, diff={r['diff']:.4f}")

    # Higher-order case (K3=1.0)
    print("\n--- K3 = 1.0 ---")
    results_1 = compare_gaussian_vs_lorentzian(K2_range, K3=1.0, sigma=sigma)
    for r in results_1:
        if r['r_gaussian'] > 0 or r['r_OA'] > 0:
            print(f"  K2={r['K2']:.2f}: r_Gauss={r['r_gaussian']:.4f}, "
                  f"r_OA={r['r_OA']:.4f}, diff={r['diff']:.4f}")

    # Higher-order case (K3=2.0)
    print("\n--- K3 = 2.0 ---")
    results_2 = compare_gaussian_vs_lorentzian(K2_range, K3=2.0, sigma=sigma)
    for r in results_2:
        if r['r_gaussian'] > 0 or r['r_OA'] > 0:
            print(f"  K2={r['K2']:.2f}: r_Gauss={r['r_gaussian']:.4f}, "
                  f"r_OA={r['r_OA']:.4f}, diff={r['diff']:.4f}")

    # Compute full 2D phase diagram (Gaussian exact)
    print("\n=== Computing Gaussian exact phase diagram ===")
    K2_full = np.linspace(0, 6, 30)
    K3_full = np.linspace(-2, 4, 30)
    r_gauss_2d = np.zeros((len(K2_full), len(K3_full)))
    for i, K2 in enumerate(K2_full):
        for j, K3 in enumerate(K3_full):
            r_gauss_2d[i, j] = find_r_star_gaussian(K2, K3, sigma)
        print(f"  Row {i+1}/{len(K2_full)}: K2={K2:.2f}")

    output = {
        'sigma': sigma,
        'K2_range': K2_full.tolist(),
        'K3_range': K3_full.tolist(),
        'r_gaussian_2d': r_gauss_2d.tolist(),
        'comparison_K3_0': results_0,
        'comparison_K3_1': results_1,
        'comparison_K3_2': results_2
    }
    with open('gaussian_exact_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to gaussian_exact_results.json")
