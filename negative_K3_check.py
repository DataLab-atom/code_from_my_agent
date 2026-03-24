"""
Quick check: does the c[g] correction affect K3<0 (democratic sync)?

OA prediction: K_eff(r*) -> Kc as K3 -> -inf, r* -> 0.
Corrected: K_eff = K2 + K3*Z2, where Z2 < r^2. Since K3<0,
the corrected K_eff is LESS negative (closer to K2), so:
- r* should be LARGER in corrected than approximate
- Self-organized criticality should still hold but at different rates
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


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

def solve_corrected(K2, K3, sigma=1.0):
    def eq(h):
        Z1, Z2 = compute_Z1_Z2(h, sigma)
        return (K2 + K3*Z2)*Z1 - h
    h_test = np.logspace(-2, 1.2, 80)
    roots = []
    prev_v = None
    for h in h_test:
        try:
            v = eq(h)
            if prev_v is not None and v*prev_v < 0:
                h_star = brentq(eq, h_prev, h, xtol=1e-8)
                Z1, Z2 = compute_Z1_Z2(h_star, sigma)
                roots.append((Z1, Z2, K2+K3*Z2))
            prev_v = v
            h_prev = h
        except:
            prev_v = None
    return roots[-1] if roots else (0, 0, K2)

def solve_approx(K2, K3, sigma=1.0):
    g = lambda w: gaussian_g(w, sigma)
    def eq(r):
        Keff = K2 + K3*r**2
        if Keff <= 0: return -r
        def integ(theta):
            return np.cos(theta)**2 * g(Keff*r*np.sin(theta))
        I, _ = quad(integ, -np.pi/2, np.pi/2)
        return Keff*r*I - r
    r_test = np.linspace(0.01, 0.999, 80)
    roots = []
    prev = eq(r_test[0])
    for i in range(1, len(r_test)):
        v = eq(r_test[i])
        if v*prev < 0:
            try: roots.append(brentq(eq, r_test[i-1], r_test[i], xtol=1e-6))
            except: pass
        prev = v
    r = max(roots) if roots else 0
    Keff = K2 + K3*r**2
    return r, Keff

sigma = 1.0
Kc = 2/(np.pi*gaussian_g(0, sigma))
K2 = 3.0

print(f"K2={K2}, Kc={Kc:.4f}")
print(f"{'K3':>8s} {'r*(corr)':>10s} {'r*(approx)':>10s} {'Keff(corr)':>12s} {'Keff(appr)':>12s}")
print("-" * 58)

for K3 in [0, -1, -2, -3, -5, -8, -12, -20]:
    r_c, Z2_c, Keff_c = solve_corrected(K2, K3, sigma)
    r_a, Keff_a = solve_approx(K2, K3, sigma)
    print(f"{K3:8.0f} {r_c:10.4f} {r_a:10.4f} {Keff_c:12.4f} {Keff_a:12.4f}")

print(f"\nAs K3 -> -inf, both K_eff should approach Kc = {Kc:.4f}")
print("Corrected r* should be LARGER (K3 effect is weaker since Z2 < r^2)")
