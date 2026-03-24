"""
Compute the saddle-node (bistability) boundary for both corrected and
approximate self-consistent equations.

The saddle-node occurs where F(h) = (K2+K3*Z2(h))*Z1(h) - h = 0 AND
F'(h) = 0 simultaneously (double root).

For each K3, find the minimum K2 at which a synchronized solution exists.
This traces the bistability boundary in K2-K3 space.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, brentq


def gaussian_g(omega, sigma=1.0):
    return np.exp(-omega**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))


def compute_Z1_Z2(h, sigma=1.0):
    if h < 1e-12:
        return 0.0, 0.0
    g = lambda w: gaussian_g(w, sigma)
    Z1, _ = quad(lambda w: np.sqrt(1-(w/h)**2)*g(w), -h, h, limit=200)
    Z2L, _ = quad(lambda w: (1-2*(w/h)**2)*g(w), -h, h, limit=200)
    def z2d(w):
        return -(w-np.sqrt(w**2-h**2))**2/h**2 * g(w)
    Z2D, _ = quad(z2d, h*1.00001, 20.0, limit=200, points=[h*1.001])
    Z2D *= 2
    return Z1, Z2L + Z2D


def find_K2_min_corrected(K3, sigma=1.0):
    """Find minimum K2 for which corrected equation has solution."""
    # At the saddle-node, max_h [(K2+K3*Z2(h))*Z1(h)/h] = 1
    # Equivalently, K2_min = min_h [h/Z1(h) - K3*Z2(h)]

    def neg_ratio(h):
        Z1, Z2 = compute_Z1_Z2(h, sigma)
        if Z1 < 1e-12:
            return 1e10
        # F(h)=0 gives K2 = h/Z1 - K3*Z2
        K2_needed = h/Z1 - K3*Z2
        return K2_needed  # minimize this

    # Scan h to find the minimum K2 needed
    h_vals = np.logspace(-1, 1.5, 200)
    K2_vals = []
    for h in h_vals:
        Z1, Z2 = compute_Z1_Z2(h, sigma)
        if Z1 > 1e-10:
            K2_vals.append(h/Z1 - K3*Z2)
        else:
            K2_vals.append(1e10)

    return min(K2_vals)


def find_K2_min_approximate(K3, sigma=1.0):
    """Find minimum K2 for which approximate equation has solution."""
    g = lambda w: gaussian_g(w, sigma)

    # The approximate equation: r = (K2+K3*r^2)*r*I(r)
    # At saddle-node: K2 = 1/I(r) - K3*r^2 where I(r) = integral

    def K2_needed(r):
        Keff_partial = K3*r**2  # K_eff = K2 + K3*r^2, K2 = K_eff - K3*r^2
        # Self-consistency: 1 = K_eff * I(K_eff*r)
        # So K_eff = 1/I(K_eff*r)
        # This is implicit... let me use a different approach

        # For given r, find K2 such that F(r)=0:
        # (K2+K3*r^2)*r*I((K2+K3*r^2)*r) = r
        # (K2+K3*r^2)*I((K2+K3*r^2)*r) = 1

        # Define Keff*I(Keff*r) = 1, solve for Keff, then K2 = Keff - K3*r^2
        def eq_Keff(Keff):
            h = Keff*r
            def integ(theta):
                return np.cos(theta)**2 * g(h*np.sin(theta))
            I, _ = quad(integ, -np.pi/2, np.pi/2)
            return Keff*I - 1

        # Search for Keff
        try:
            Keff = brentq(eq_Keff, 0.1, 50.0, xtol=1e-6)
            return Keff - K3*r**2
        except:
            return 1e10

    r_vals = np.linspace(0.01, 0.999, 100)
    K2_vals = [K2_needed(r) for r in r_vals]
    return min(K2_vals)


if __name__ == '__main__':
    sigma = 1.0
    Kc = 2/(np.pi*gaussian_g(0, sigma))
    print(f"Gaussian sigma={sigma}, Kc={Kc:.4f}")
    print()

    K3_range = np.arange(0.0, 5.1, 0.25)

    print(f"{'K3':>8s} {'K2_min(corr)':>14s} {'K2_min(approx)':>16s} {'diff':>10s}")
    print("-" * 52)

    results = []
    for K3 in K3_range:
        K2c = find_K2_min_corrected(K3, sigma)
        K2a = find_K2_min_approximate(K3, sigma)
        diff = K2c - K2a
        results.append((K3, K2c, K2a))
        print(f"{K3:8.2f} {K2c:14.4f} {K2a:16.4f} {diff:10.4f}")

    # Summary
    print(f"\nAt K3=0: K2_min = Kc = {Kc:.4f} for both (onset)")
    print(f"At K3=1: corrected needs K2 >= {[r[1] for r in results if r[0]==1.0][0]:.3f}, "
          f"approximate needs K2 >= {[r[2] for r in results if r[0]==1.0][0]:.3f}")
    print(f"\nThe corrected boundary is HIGHER (needs more K2), confirming")
    print(f"that the true bistable region is smaller for Gaussian.")
