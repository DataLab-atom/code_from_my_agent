"""
Z2/Z1^2 ratio computation for multiple distributions.
Key finding: c = Z2/Z1^2 is distribution-dependent!
  Lorentzian: c = 1
  Gaussian:   c = 2/pi
  Uniform:    c = ?
"""
import numpy as np
from scipy.integrate import quad


def gaussian_g(omega, sigma=1.0):
    return np.exp(-omega**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def lorentzian_g(omega, Delta=1.0):
    return (Delta / np.pi) / (omega**2 + Delta**2)

def uniform_g(omega, a=1.0):
    return 1.0/(2*a) if abs(omega) <= a else 0.0

def triangular_g(omega, a=1.0):
    if abs(omega) <= a:
        return (a - abs(omega)) / a**2
    return 0.0


def compute_Z1_Z2(h, g_func, omega_max=20.0):
    if h < 1e-15:
        return 0.0, 0.0

    def z1_locked(omega):
        return np.sqrt(1 - (omega/h)**2) * g_func(omega)

    def z2_locked(omega):
        return (1 - 2*(omega/h)**2) * g_func(omega)

    Z1_L, _ = quad(z1_locked, -h, h, limit=200)
    Z2_L, _ = quad(z2_locked, -h, h, limit=200)

    def z2_drift(omega):
        lam = np.sqrt(omega**2 - h**2)
        return -(omega - lam)**2 / h**2 * g_func(omega)

    Z2_D, _ = quad(z2_drift, h*(1+1e-10), omega_max,
                   limit=200, points=[h*1.001, h*1.01, h*1.1])
    Z2_D *= 2

    return Z1_L, Z2_L + Z2_D


def compute_c(g_func, name, h_test=0.002, omega_max=20.0):
    Z1, Z2 = compute_Z1_Z2(h_test, g_func, omega_max)
    c = Z2 / Z1**2 if Z1**2 > 1e-30 else float('nan')
    print(f"  {name:20s}: c = Z2/Z1^2 = {c:.6f}  (h={h_test})")
    return c


if __name__ == '__main__':
    print("=" * 60)
    print("Z2/Z1^2 ratio (c) for different distributions")
    print("=" * 60)

    # Scan h for Gaussian to confirm convergence
    print("\nGaussian convergence check:")
    for h in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        Z1, Z2 = compute_Z1_Z2(h, lambda w: gaussian_g(w, 1.0))
        c = Z2/Z1**2 if Z1**2 > 1e-30 else float('nan')
        print(f"  h={h:.4f}: c = {c:.6f}")

    print(f"\n  2/pi = {2/np.pi:.6f}")

    # Different distributions
    print("\n--- c for different distributions ---")
    c_lor = compute_c(lambda w: lorentzian_g(w, 1.0), "Lorentzian(D=1)")
    c_gau = compute_c(lambda w: gaussian_g(w, 1.0), "Gaussian(s=1)")
    c_gau2 = compute_c(lambda w: gaussian_g(w, 0.5), "Gaussian(s=0.5)", h_test=0.001)
    c_gau3 = compute_c(lambda w: gaussian_g(w, 2.0), "Gaussian(s=2)", h_test=0.004)
    c_uni = compute_c(lambda w: uniform_g(w, 1.0), "Uniform[-1,1]", omega_max=1.0)
    c_uni2 = compute_c(lambda w: uniform_g(w, 2.0), "Uniform[-2,2]", omega_max=2.0)
    c_tri = compute_c(lambda w: triangular_g(w, 1.0), "Triangular[-1,1]", omega_max=1.0)

    # Implications for explosive formula
    print("\n" + "=" * 60)
    print("Implications for explosive synchronization formula")
    print("=" * 60)
    print("\nOriginal formula:  K3_exp = Kc^3 |g''(0)| / (8 g(0))")
    print("Corrected formula: K3_exp = Kc^3 |g''(0)| / (8 c g(0))")
    print("where c = Z2/Z1^2 at onset\n")

    for name, c_val in [("Lorentzian", c_lor), ("Gaussian", c_gau)]:
        if name == "Lorentzian":
            g0 = lorentzian_g(0, 1.0)
            gpp0 = -2/(np.pi*1.0**3)  # g''(0) for Lorentzian
        else:
            g0 = gaussian_g(0, 1.0)
            gpp0 = -g0/1.0**2

        Kc = 2/(np.pi*g0)
        K3_orig = Kc**3 * abs(gpp0) / (8*g0)
        K3_corr = Kc**3 * abs(gpp0) / (8*c_val*g0)

        print(f"{name}:")
        print(f"  Kc = {Kc:.4f}, g(0) = {g0:.4f}, g''(0) = {gpp0:.4f}")
        print(f"  c = {c_val:.4f}")
        print(f"  K3_exp (original, c=1):      {K3_orig:.4f}  (K3_exp/Kc = {K3_orig/Kc:.4f})")
        print(f"  K3_exp (corrected, c={c_val:.4f}): {K3_corr:.4f}  (K3_exp/Kc = {K3_corr/Kc:.4f})")
        print()

    print("Key insight: c = 1 for Lorentzian (OA exact)")
    print(f"             c = 2/pi = {2/np.pi:.4f} for Gaussian")
    print(f"             Correction factor: pi/2 = {np.pi/2:.4f}")
