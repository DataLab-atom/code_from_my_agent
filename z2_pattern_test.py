"""
Test the c = (2/pi)^n pattern with more distributions.

Known: Lorentzian c=1, Gaussian c=2/pi, Uniform c=4/pi^2
Conjecture: c depends on tail behavior / smoothness of g
"""
import numpy as np
from scipy.integrate import quad


def compute_c(g_func, h=0.002, omega_max=20.0):
    """Compute c = Z2/Z1^2 at given h."""
    def z1_int(omega):
        return np.sqrt(1 - (omega/h)**2) * g_func(omega)
    def z2_locked_int(omega):
        return (1 - 2*(omega/h)**2) * g_func(omega)
    def z2_drift_int(omega):
        lam = np.sqrt(omega**2 - h**2)
        return -(omega - lam)**2 / h**2 * g_func(omega)

    Z1, _ = quad(z1_int, -h, h, limit=200)
    Z2_L, _ = quad(z2_locked_int, -h, h, limit=200)
    Z2_D, _ = quad(z2_drift_int, h*(1+1e-10), omega_max,
                   limit=200, points=[h*1.001, h*1.01])
    Z2_D *= 2
    Z2 = Z2_L + Z2_D
    return Z2 / Z1**2 if Z1**2 > 1e-30 else float('nan')


# Standard distributions
def lorentzian(omega, Delta=1.0):
    return (Delta/np.pi) / (omega**2 + Delta**2)

def gaussian(omega, sigma=1.0):
    return np.exp(-omega**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))

def uniform(omega, a=1.0):
    return 1.0/(2*a) if abs(omega) <= a else 0.0

# Additional test distributions
def super_gaussian(omega, sigma=1.0, p=2):
    """g(omega) ~ exp(-|omega|^(2p)/(2*sigma^(2p)))"""
    C, _ = quad(lambda w: np.exp(-abs(w)**(2*p)/(2*sigma**(2*p))), -20, 20)
    return np.exp(-abs(omega)**(2*p)/(2*sigma**(2*p))) / C

def laplace(omega, b=1.0):
    """g(omega) = exp(-|omega|/b)/(2b) - lighter than Lorentzian, heavier than Gaussian"""
    return np.exp(-abs(omega)/b) / (2*b)

def student_t(omega, nu=3.0):
    """Student's t-distribution with nu degrees of freedom"""
    from scipy.special import gamma
    C = gamma((nu+1)/2) / (np.sqrt(nu*np.pi) * gamma(nu/2))
    return C * (1 + omega**2/nu)**(-(nu+1)/2)

def raised_cosine(omega, a=1.0):
    """g(omega) = (1+cos(pi*omega/a))/(2a) for |omega|<=a, 0 otherwise"""
    if abs(omega) <= a:
        return (1 + np.cos(np.pi*omega/a)) / (2*a)
    return 0.0

def parabolic(omega, a=1.0):
    """g(omega) = (3/(4a))(1-omega^2/a^2) for |omega|<=a"""
    if abs(omega) <= a:
        return 3.0/(4*a) * (1 - omega**2/a**2)
    return 0.0


if __name__ == '__main__':
    two_over_pi = 2/np.pi
    print(f"2/pi = {two_over_pi:.6f}")
    print(f"(2/pi)^2 = {two_over_pi**2:.6f}")
    print(f"(2/pi)^3 = {two_over_pi**3:.6f}")
    print()

    results = []

    # Lorentzian (heavy algebraic tails)
    c = compute_c(lambda w: lorentzian(w, 1.0))
    results.append(("Lorentzian (nu=1)", c))

    # Student-t nu=3 (algebraic tails, ~1/omega^4)
    c = compute_c(lambda w: student_t(w, 3.0))
    results.append(("Student-t (nu=3)", c))

    # Student-t nu=5
    c = compute_c(lambda w: student_t(w, 5.0))
    results.append(("Student-t (nu=5)", c))

    # Student-t nu=10
    c = compute_c(lambda w: student_t(w, 10.0))
    results.append(("Student-t (nu=10)", c))

    # Laplace (exponential tails)
    c = compute_c(lambda w: laplace(w, 1.0))
    results.append(("Laplace", c))

    # Gaussian (sub-exponential tails)
    c = compute_c(lambda w: gaussian(w, 1.0))
    results.append(("Gaussian", c))

    # Super-Gaussian p=2 (super-exponential tails)
    # Precompute normalization
    C4, _ = quad(lambda w: np.exp(-w**4/2), -20, 20)
    c = compute_c(lambda w: np.exp(-w**4/2)/C4)
    results.append(("Super-Gaussian p=2", c))

    # Raised cosine (compact support, smooth)
    c = compute_c(lambda w: raised_cosine(w, 1.0), omega_max=1.0)
    results.append(("Raised cosine", c))

    # Parabolic (compact support, C^0 at boundary)
    c = compute_c(lambda w: parabolic(w, 1.0), omega_max=1.0)
    results.append(("Parabolic", c))

    # Uniform (compact support, discontinuous)
    c = compute_c(lambda w: uniform(w, 1.0), omega_max=1.0)
    results.append(("Uniform", c))

    print(f"{'Distribution':30s} {'c = Z2/Z1^2':>12s} {'log(c)/log(2/pi)':>18s}")
    print("-" * 65)
    for name, c_val in results:
        if c_val > 0:
            n = np.log(c_val) / np.log(two_over_pi)
        else:
            n = float('nan')
        print(f"{name:30s} {c_val:12.6f} {n:18.4f}")

    print()
    print("If n is integer -> c = (2/pi)^n pattern holds")
    print("If n is non-integer -> pattern is a coincidence for Lor/Gau/Uni")
