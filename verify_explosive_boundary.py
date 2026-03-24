"""
Verify the corrected explosive formula K3_exp = Kc^3|g''(0)|/(8*c*g(0))
against numerical self-consistent solutions.

For Gaussian: K3_exp_original = 0.508, K3_exp_corrected = 0.798
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def gaussian_g(omega, sigma=1.0):
    return np.exp(-omega**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))


def compute_Z1_Z2(h, sigma=1.0):
    if h < 1e-15:
        return 0.0, 0.0
    g = lambda w: gaussian_g(w, sigma)

    Z1, _ = quad(lambda w: np.sqrt(1-(w/h)**2)*g(w), -h, h, limit=200)

    Z2_L, _ = quad(lambda w: (1-2*(w/h)**2)*g(w), -h, h, limit=200)

    def z2d(w):
        return -(w - np.sqrt(w**2-h**2))**2/h**2 * g(w)
    Z2_D, _ = quad(z2d, h*1.00001, 20.0, limit=200, points=[h*1.001])
    Z2_D *= 2

    return Z1, Z2_L + Z2_D


def find_r_corrected(K2, K3, sigma):
    """Solve h = (K2 + K3*Z2(h)) * Z1(h) for h, return r=Z1(h)."""
    def eq(h):
        Z1, Z2 = compute_Z1_Z2(h, sigma)
        return (K2 + K3*Z2) * Z1 - h

    # Scan for roots
    h_vals = np.logspace(-3, 1.2, 80)
    roots = []
    prev_v = None
    for h in h_vals:
        try:
            v = eq(h)
            if prev_v is not None and v * prev_v < 0:
                h_star = brentq(eq, h_prev, h, xtol=1e-8)
                Z1, Z2 = compute_Z1_Z2(h_star, sigma)
                roots.append((h_star, Z1, Z2))
            prev_v = v
            h_prev = h
        except:
            prev_v = None

    if not roots:
        return 0.0, []

    # Return info about all roots (for detecting hysteresis)
    return roots[-1][1], roots  # largest r


def find_r_approx(K2, K3, sigma):
    """Solve approximate equation (Z2=r^2)."""
    g = lambda w: gaussian_g(w, sigma)
    g0 = gaussian_g(0, sigma)

    def eq(r):
        Keff = K2 + K3*r**2
        if Keff <= 0:
            return -r

        def integ(theta):
            return np.cos(theta)**2 * g(Keff*r*np.sin(theta))
        I, _ = quad(integ, -np.pi/2, np.pi/2)
        return Keff*r*I - r

    # Scan for roots in r space
    r_vals = np.linspace(0.01, 0.999, 100)
    roots = []
    prev_v = eq(r_vals[0])
    for i in range(1, len(r_vals)):
        v = eq(r_vals[i])
        if v * prev_v < 0:
            try:
                r_star = brentq(eq, r_vals[i-1], r_vals[i], xtol=1e-6)
                roots.append(r_star)
            except:
                pass
        prev_v = v

    return max(roots) if roots else 0.0


if __name__ == '__main__':
    sigma = 1.0
    g0 = gaussian_g(0, sigma)
    Kc = 2 / (np.pi * g0)
    c = 2/np.pi

    K3_orig = Kc**3 * abs(-g0/sigma**2) / (8*g0)
    K3_corr = Kc**3 * abs(-g0/sigma**2) / (8*c*g0)

    print(f"Gaussian sigma={sigma}, Kc={Kc:.4f}")
    print(f"K3_exp (original, c=1):   {K3_orig:.4f}")
    print(f"K3_exp (corrected, c=2/pi): {K3_corr:.4f}")
    print()

    # Fine K3 scan: sweep K2 upward for each K3, detect jump
    K3_range = np.arange(0.3, 1.3, 0.05)
    K2_range = np.arange(1.0, Kc + 0.3, 0.01)

    print(f"{'K3':>8s} {'max_jump_corr':>14s} {'max_jump_approx':>16s} {'explosive?':>12s}")
    print("-" * 55)

    for K3 in K3_range:
        # Corrected
        r_prev = 0
        max_jump_corr = 0
        for K2 in K2_range:
            r, _ = find_r_corrected(K2, K3, sigma)
            jump = r - r_prev
            if jump > max_jump_corr:
                max_jump_corr = jump
            r_prev = r

        # Approximate
        r_prev = 0
        max_jump_approx = 0
        for K2 in K2_range:
            r = find_r_approx(K2, K3, sigma)
            jump = r - r_prev
            if jump > max_jump_approx:
                max_jump_approx = jump
            r_prev = r

        explosive_corr = "YES" if max_jump_corr > 0.3 else "no"
        marker = ""
        if abs(K3 - K3_orig) < 0.03:
            marker = " <-- original K3_exp"
        if abs(K3 - K3_corr) < 0.03:
            marker = " <-- corrected K3_exp"

        print(f"{K3:8.3f} {max_jump_corr:14.4f} {max_jump_approx:16.4f} {explosive_corr:>12s}{marker}")
