"""
Corrected self-consistent equation for higher-order Kuramoto with Gaussian g(omega).

The WRONG approach (gaussian_selfconsistent.py): assumes Z2 = Z1^2 = r^2
The CORRECT approach (this file): computes Z1(h) and Z2(h) from the full
stationary distribution, then solves h = (K2 + K3*Z2(h)) * Z1(h).

Key insight: Z2/Z1^2 = 2/pi for Gaussian (not 1), changing the explosive boundary.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def gaussian_g(omega, sigma=1.0):
    return np.exp(-omega**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))


def compute_Z1_Z2_from_h(h, g_func, omega_max=20.0):
    """Given mean-field amplitude h, compute Z1 and Z2 from stationary dist."""
    if h < 1e-15:
        return 0.0, 0.0

    # Locked oscillators: |omega| < h
    def z1_locked(omega):
        return np.sqrt(1 - (omega/h)**2) * g_func(omega)

    def z2_locked(omega):
        return (1 - 2*(omega/h)**2) * g_func(omega)

    Z1_L, _ = quad(z1_locked, -h, h, limit=200)
    Z2_L, _ = quad(z2_locked, -h, h, limit=200)

    # Drifting oscillators: |omega| > h
    # Z1_drift = 0 (standard result)
    # Z2_drift: <cos2theta> = -(omega - sqrt(omega^2-h^2))^2 / h^2
    def z2_drift(omega):
        lam = np.sqrt(omega**2 - h**2)
        return -(omega - lam)**2 / h**2 * g_func(omega)

    Z2_D, _ = quad(z2_drift, h*(1+1e-10), omega_max,
                   limit=200, points=[h*1.001, h*1.01, h*1.1])
    Z2_D *= 2  # symmetry

    return Z1_L, Z2_L + Z2_D


def solve_corrected(K2, K3, sigma, h_max=20.0):
    """
    Solve the CORRECT self-consistent equation:
      h = (K2 + K3*Z2(h)) * Z1(h)
    where Z1(h), Z2(h) come from the full stationary distribution.

    Returns (r_star, Z2_star, c_star) where c_star = Z2/Z1^2.
    """
    g_func = lambda omega: gaussian_g(omega, sigma)

    def equation(h):
        Z1, Z2 = compute_Z1_Z2_from_h(h, g_func)
        rhs = (K2 + K3 * Z2) * Z1
        return rhs - h

    # Check onset: at h=0+, equation ~ (K2*pi*g(0)/2 - 1)*h
    g0 = gaussian_g(0, sigma)
    onset_val = K2 * np.pi * g0 / 2 - 1

    if onset_val < 0 and K3 <= 0:
        return 0.0, 0.0, float('nan')

    # Search for h* > 0
    # Try a range of h values to find sign change
    h_test = np.logspace(-3, np.log10(h_max), 50)
    vals = []
    for h in h_test:
        try:
            v = equation(h)
            vals.append((h, v))
        except:
            pass

    # Find sign changes (roots)
    roots = []
    for i in range(len(vals)-1):
        h1, v1 = vals[i]
        h2, v2 = vals[i+1]
        if v1 * v2 < 0:
            try:
                h_star = brentq(equation, h1, h2, xtol=1e-8)
                roots.append(h_star)
            except:
                pass

    if not roots:
        return 0.0, 0.0, float('nan')

    # Take the largest stable root
    h_star = max(roots)
    Z1, Z2 = compute_Z1_Z2_from_h(h_star, g_func)
    r_star = Z1
    c_star = Z2 / Z1**2 if Z1 > 1e-10 else float('nan')

    return r_star, Z2, c_star


def solve_approximate(K2, K3, sigma, r_max=0.9999):
    """
    Solve the APPROXIMATE equation (Z2 = r^2, i.e., c=1).
    This is what gaussian_selfconsistent.py does.
    """
    g_func = lambda omega: gaussian_g(omega, sigma)
    g0 = gaussian_g(0, sigma)

    def equation(r):
        K_eff = K2 + K3 * r**2
        if K_eff <= 0:
            return -r
        def integrand(theta):
            return np.cos(theta)**2 * g_func(K_eff * r * np.sin(theta))
        result, _ = quad(integrand, -np.pi/2, np.pi/2)
        return K_eff * r * result - r

    # Check onset
    if K2 * np.pi * g0 / 2 - 1 < 0 and K3 <= 0:
        return 0.0

    try:
        r_star = brentq(equation, 0.001, r_max, xtol=1e-6)
        return float(r_star)
    except ValueError:
        return 0.0


if __name__ == '__main__':
    sigma = 1.0
    g0 = gaussian_g(0, sigma)
    Kc = 2 / (np.pi * g0)

    print(f"Gaussian sigma={sigma}")
    print(f"Kc = {Kc:.4f}")
    print(f"g(0) = {g0:.6f}")
    print()

    # Compare corrected vs approximate for K3=0 (classical Kuramoto)
    print("=" * 80)
    print("K3 = 0 (classical Kuramoto) - should agree")
    print("=" * 80)
    print(f"{'K2':>8s} {'r*(correct)':>12s} {'r*(approx)':>12s} {'diff':>10s} {'c=Z2/Z1^2':>10s}")
    print("-" * 55)
    for K2 in [1.8, 2.0, 2.5, 3.0, 4.0, 5.0]:
        r_corr, Z2, c = solve_corrected(K2, 0.0, sigma)
        r_approx = solve_approximate(K2, 0.0, sigma)
        diff = abs(r_corr - r_approx)
        print(f"{K2:8.2f} {r_corr:12.6f} {r_approx:12.6f} {diff:10.6f} {c:10.4f}")

    # Compare for K3 > 0 (where the difference matters!)
    print()
    print("=" * 80)
    print("K3 = 1.0 (higher-order) - expect significant differences")
    print("=" * 80)
    print(f"{'K2':>8s} {'r*(correct)':>12s} {'r*(approx)':>12s} {'diff':>10s} {'c=Z2/Z1^2':>10s}")
    print("-" * 55)
    for K2 in np.arange(0.5, 4.1, 0.5):
        r_corr, Z2, c = solve_corrected(K2, 1.0, sigma)
        r_approx = solve_approximate(K2, 1.0, sigma)
        diff = abs(r_corr - r_approx)
        print(f"{K2:8.2f} {r_corr:12.6f} {r_approx:12.6f} {diff:10.6f} {c:10.4f}")

    # Compare for K3 = 2.0
    print()
    print("=" * 80)
    print("K3 = 2.0")
    print("=" * 80)
    print(f"{'K2':>8s} {'r*(correct)':>12s} {'r*(approx)':>12s} {'diff':>10s} {'c=Z2/Z1^2':>10s}")
    print("-" * 55)
    for K2 in np.arange(0.5, 4.1, 0.5):
        r_corr, Z2, c = solve_corrected(K2, 2.0, sigma)
        r_approx = solve_approximate(K2, 2.0, sigma)
        diff = abs(r_corr - r_approx)
        print(f"{K2:8.2f} {r_corr:12.6f} {r_approx:12.6f} {diff:10.6f} {c:10.4f}")

    # Check explosive boundary
    print()
    print("=" * 80)
    print("Explosive boundary check near Kc")
    print("=" * 80)

    K3_test = np.arange(0.2, 2.0, 0.2)
    print(f"{'K3':>8s} {'r_jump(corr)':>14s} {'r_jump(approx)':>14s}")
    print("-" * 40)
    for K3 in K3_test:
        # Sweep K2 upward from 0 to find the jump
        r_prev_corr = 0
        r_prev_approx = 0
        jump_corr = 0
        jump_approx = 0
        for K2 in np.arange(0.5, Kc+0.5, 0.02):
            r_c, _, _ = solve_corrected(K2, K3, sigma)
            r_a = solve_approximate(K2, K3, sigma)
            if r_c - r_prev_corr > jump_corr:
                jump_corr = r_c - r_prev_corr
            if r_a - r_prev_approx > jump_approx:
                jump_approx = r_a - r_prev_approx
            r_prev_corr = r_c
            r_prev_approx = r_a
        print(f"{K3:8.2f} {jump_corr:14.4f} {jump_approx:14.4f}")
