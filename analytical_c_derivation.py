"""
Attempt to derive c[g] = 2/pi for Gaussian analytically.

Strategy: Compute Z2 as a function of h for Gaussian, using the exact
integral representation, and extract the O(h^2) coefficient.

Z2 = h*g0 * integral_0^infty Q(u) * exp(-epsilon^2 * u^2) du
where epsilon = h/(sigma*sqrt(2)), Q(u) = kernel from locked+drift.

The key integral is I(eps) = integral_0^infty Q(u) exp(-eps^2 u^2) du
and c = I''(0) / (pi/2)^2... or something like this.

Let me compute I(eps) numerically for many eps values and extract the
asymptotic.
"""
import numpy as np
from scipy.integrate import quad


def Q_kernel(u):
    """Combined locked+drifting kernel for Z2."""
    if u < 1:
        return 2*(1 - 2*u**2)
    else:
        return -2*(u - np.sqrt(u**2 - 1))**2


def P_kernel(u):
    """Kernel for Z1."""
    if u < 1:
        return 2*np.sqrt(1 - u**2)
    return 0.0


def I_Q(eps):
    """Compute integral_0^infty Q(u) exp(-eps^2 u^2) du"""
    def integrand(u):
        return Q_kernel(u) * np.exp(-eps**2 * u**2)

    # Split at u=1 to handle the discontinuity in Q
    I1, _ = quad(lambda u: 2*(1-2*u**2)*np.exp(-eps**2*u**2), 0, 1, limit=200)
    I2, _ = quad(lambda u: -2*(u-np.sqrt(u**2-1))**2*np.exp(-eps**2*u**2),
                 1.0001, 200/max(eps, 0.01), limit=200)
    return I1 + I2


def I_P(eps):
    """Compute integral_0^infty P(u) exp(-eps^2 u^2) du"""
    return quad(lambda u: 2*np.sqrt(1-u**2)*np.exp(-eps**2*u**2), 0, 1, limit=200)[0]


if __name__ == '__main__':
    print("Asymptotic analysis of Z2 for Gaussian")
    print("=" * 60)

    # Z1 = h*g0 * I_P(eps) where eps = h/(sigma*sqrt(2))
    # Z2 = h*g0 * I_Q(eps)
    # Z1^2 = h^2*g0^2 * I_P(eps)^2
    # c = Z2/Z1^2 = I_Q(eps) / (h*g0*I_P(eps)^2)
    #             = I_Q(eps) / (eps*sigma*sqrt(2)*g0*I_P(eps)^2)
    # For Gaussian g0 = 1/(sigma*sqrt(2*pi)):
    # c = I_Q(eps) / (eps*sqrt(2)/(sqrt(2*pi)) * I_P(eps)^2)
    #   = I_Q(eps) * sqrt(pi) / (eps * I_P(eps)^2)

    print(f"\n{'eps':>10s} {'I_Q(eps)':>14s} {'I_P(eps)':>14s} {'I_Q/(eps*I_P^2)':>16s} {'*sqrt(pi)':>12s}")
    print("-" * 70)

    for eps in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        iq = I_Q(eps)
        ip = I_P(eps)
        ratio = iq / (eps * ip**2) if eps*ip**2 > 1e-30 else float('nan')
        c_val = ratio * np.sqrt(np.pi)

        print(f"{eps:10.4f} {iq:14.8f} {ip:14.8f} {ratio:16.8f} {c_val:12.6f}")

    print(f"\n2/pi = {2/np.pi:.6f}")
    print(f"sqrt(pi)*2/pi = {np.sqrt(np.pi)*2/np.pi:.6f}")

    # So: c = sqrt(pi) * lim_{eps->0} I_Q(eps) / (eps * I_P(eps)^2)
    # I_P(eps) -> pi/2 as eps->0
    # So c = sqrt(pi) * lim_{eps->0} I_Q(eps) / (eps * pi^2/4)
    #       = 4*sqrt(pi)/(pi^2) * lim_{eps->0} I_Q(eps)/eps

    print("\nChecking I_Q(eps)/eps:")
    for eps in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        iq = I_Q(eps)
        print(f"  eps={eps:.4f}: I_Q/eps = {iq/eps:.8f}")

    # If I_Q(eps) ~ A*eps as eps->0, then c = sqrt(pi)*A/(pi/2)^2 = 4*sqrt(pi)*A/pi^2
    # Need A such that c = 2/pi
    # A = c*pi^2/(4*sqrt(pi)) = (2/pi)*pi^2/(4*sqrt(pi)) = 2*pi/(4*sqrt(pi)) = pi/(2*sqrt(pi)) = sqrt(pi)/2
    # So I_Q(eps) ~ sqrt(pi)/2 * eps as eps->0?

    print(f"\nsqrt(pi)/2 = {np.sqrt(np.pi)/2:.8f}")
    print("Compare with I_Q(eps)/eps values above")
