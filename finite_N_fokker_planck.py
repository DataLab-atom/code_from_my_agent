"""
Finite-N Fokker-Planck analysis for the higher-order Kuramoto model.

Task AH.1: Why does K3>0 DECREASE <r> in the subcritical region?

The OA effective potential V(r) = alpha*r^2/2 + |beta|*r^4/8 has:
- alpha = Delta - K2/2 > 0 (subcritical)
- beta = K3 - K2 < 0 (for K3 < K2)

K3>0 makes |beta| SMALLER, making the potential LESS confining.
Naively this predicts LARGER <r> for K3>0. But data shows smaller.

Resolution: near the critical point (alpha ~ 0), the perturbative
expansion in beta*sigma^2/alpha^2 breaks down. Need non-perturbative
Fokker-Planck with the full quartic potential.
"""
import numpy as np
from scipy.integrate import quad


def compute_r_stats(alpha, beta, sigma_sq):
    """
    Compute <r> and <r^2> from Fokker-Planck steady state
    P(r) ~ r * exp(-2V(r)/sigma^2)
    V(r) = alpha*r^2/2 - beta*r^4/8
    (beta = K3-K2, so -beta = K2-K3 > 0 for K3<K2)
    """
    def log_P(r):
        V = alpha*r**2/2 - beta*r**4/8
        return np.log(r + 1e-300) - 2*V/sigma_sq

    # Find mode for numerical stability
    r_mode = np.sqrt(max(0, (sigma_sq - alpha) / (- beta / 2))) if beta < 0 else 0.1
    r_mode = max(r_mode, 0.01)

    log_P_mode = log_P(r_mode)

    def P(r):
        return np.exp(log_P(r) - log_P_mode)

    # Normalize and compute moments
    Z, _ = quad(P, 0, 5, limit=200)
    r1, _ = quad(lambda r: r * P(r), 0, 5, limit=200)
    r2, _ = quad(lambda r: r**2 * P(r), 0, 5, limit=200)

    return r1/Z, r2/Z


if __name__ == '__main__':
    # Parameters from PlotAnalyst observation
    sigma_freq = 1.2
    Delta = sigma_freq * np.sqrt(2*np.pi) / np.pi  # = 0.958
    Kc = 2 * Delta  # = 1.916
    K2 = 1.87  # subcritical (K2 < Kc)
    N = 200
    sigma_sq = 1.0 / (2*N)  # noise variance

    alpha = Delta - K2/2  # = 0.023

    print(f"sigma_freq={sigma_freq}, Delta={Delta:.4f}, Kc={Kc:.4f}")
    print(f"K2={K2}, alpha={alpha:.4f}, sigma^2={sigma_sq:.6f}")
    print(f"Near-critical: alpha*N = {alpha*2*N:.2f}")
    print()

    print(f"{'K3':>6s} {'beta':>8s} {'<r>':>8s} {'<r^2>':>10s} {'ratio':>8s}")
    print("-" * 45)

    r_ref = None
    for K3 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]:
        beta = K3 - K2  # negative for K3 < K2
        mean_r, mean_r2 = compute_r_stats(alpha, beta, sigma_sq)
        if r_ref is None:
            r_ref = mean_r
        print(f"{K3:6.1f} {beta:8.3f} {mean_r:8.4f} {mean_r2:10.6f} {mean_r/r_ref:8.4f}")

    # Also check with corrected beta (c[g] = 2/pi)
    print("\n--- With c[g] = 2/pi correction ---")
    c = 2/np.pi
    r_ref = None
    for K3 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]:
        beta = c*K3 - K2
        mean_r, mean_r2 = compute_r_stats(alpha, beta, sigma_sq)
        if r_ref is None:
            r_ref = mean_r
        print(f"{K3:6.1f} {beta:8.3f} {mean_r:8.4f} {mean_r2:10.6f} {mean_r/r_ref:8.4f}")

    # The REAL test: does the full potential (not just quartic) change things?
    print("\n--- Full OA potential (not just quartic approx) ---")
    def compute_r_stats_full(K2, K3, Delta, sigma_sq):
        def f(r):
            return -Delta*r + r*(1-r**2)/2*(K2+K3*r**2)
        def V(r):
            # V = -integral f(r) dr
            return Delta*r**2/2 - K2*r**2/4 - (K3-K2)*r**4/8 + K3*r**6/12
        def log_P(r):
            return np.log(r+1e-300) - 2*V(r)/sigma_sq
        r_test = np.linspace(0.01, 0.99, 100)
        lp_max = max(log_P(r) for r in r_test)
        def P(r):
            return np.exp(log_P(r) - lp_max)
        Z, _ = quad(P, 0.001, 2.0, limit=200)
        r1, _ = quad(lambda r: r*P(r), 0.001, 2.0, limit=200)
        return r1/Z

    r_ref = None
    print(f"{'K3':>6s} {'<r>_full':>10s} {'ratio':>8s}")
    print("-" * 28)
    for K3 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]:
        mean_r = compute_r_stats_full(K2, K3, Delta, sigma_sq)
        if r_ref is None:
            r_ref = mean_r
        print(f"{K3:6.1f} {mean_r:10.4f} {mean_r/r_ref:8.4f}")

    print(f"\nData: K3=0 -> r=0.34, K3=0.4 -> r=0.21 (ratio={0.21/0.34:.3f})")
