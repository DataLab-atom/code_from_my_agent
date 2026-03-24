"""
Derive closed-form c[g] for Student-t(nu) family.

Known values: c(1)=1, c(3)=3/4, c(5)=45/64
Conjecture: c(nu) involves Gamma functions.

Using the general formula:
  c = (2/(pi^2*g0^2)) * integral_0^inf (g0-g(omega))/omega^2 dw

For Student-t(nu):
  g(omega) = C_nu * (1+omega^2/nu)^{-(nu+1)/2}
  g(0) = C_nu where C_nu = Gamma((nu+1)/2) / (sqrt(nu*pi)*Gamma(nu/2))

  1 - g(omega)/g(0) = 1 - (1+omega^2/nu)^{-(nu+1)/2}

  J = integral_0^inf (1-(1+omega^2/nu)^{-(nu+1)/2})/omega^2 dw

This integral can be computed via differentiation:
  f(a) = integral_0^inf (1-(1+a*omega^2)^{-p})/omega^2 dw, a=1/nu, p=(nu+1)/2
  f'(a) = p * integral_0^inf (1+a*omega^2)^{-(p+1)} dw
         = p * sqrt(pi)*Gamma(p+1/2) / (2*sqrt(a)*Gamma(p+1))
  f(0) = 0
  f(a) = p*sqrt(pi)*Gamma(p+1/2)/Gamma(p+1) * sqrt(a)

So J = f(1/nu) = p*sqrt(pi)*Gamma(p+1/2)/Gamma(p+1) * 1/sqrt(nu)
with p = (nu+1)/2:
  J = ((nu+1)/2)*sqrt(pi)*Gamma((nu+2)/2) / (Gamma((nu+3)/2)*sqrt(nu))

And c = 2*J/(pi^2*g0) = 2*J*sqrt(nu*pi)*Gamma(nu/2) / (pi^2*Gamma((nu+1)/2))
"""
import numpy as np
from scipy.special import gamma
from scipy.integrate import quad


def c_student_t_analytical(nu):
    """Closed-form c for Student-t(nu) using Gamma functions."""
    p = (nu+1)/2
    # J = p*sqrt(pi)*Gamma(p+1/2) / (Gamma(p+1)*sqrt(nu))
    J = p * np.sqrt(np.pi) * gamma(p+0.5) / (gamma(p+1) * np.sqrt(nu))

    # g0 = Gamma((nu+1)/2) / (sqrt(nu*pi)*Gamma(nu/2))
    g0 = gamma((nu+1)/2) / (np.sqrt(nu*np.pi) * gamma(nu/2))

    # c = 2*J / (pi^2 * g0)
    c = 2 * J / (np.pi**2 * g0)
    return c


def c_student_t_numerical(nu, h=0.002):
    """Numerical c for Student-t(nu)."""
    g0 = gamma((nu+1)/2) / (np.sqrt(nu*np.pi) * gamma(nu/2))
    g = lambda w: g0 * (1+w**2/nu)**(-(nu+1)/2)

    def z1_int(w):
        return np.sqrt(1-(w/h)**2) * g(w)
    def z2_L(w):
        return (1-2*(w/h)**2) * g(w)
    def z2_D(w):
        return -(w-np.sqrt(w**2-h**2))**2/h**2 * g(w)

    Z1, _ = quad(z1_int, -h, h, limit=200)
    Z2L, _ = quad(z2_L, -h, h, limit=200)
    Z2D, _ = quad(z2_D, h*1.00001, 20, limit=200, points=[h*1.001])
    Z2D *= 2
    return (Z2L+Z2D)/Z1**2


def simplify_c(nu):
    """Try to express c(nu) as a simple fraction."""
    c = c_student_t_analytical(nu)
    # Check against simple fractions with small denominators
    for denom in range(1, 256):
        numer = round(c * denom)
        if abs(numer/denom - c) < 1e-8:
            return numer, denom, c
    return None, None, c


if __name__ == '__main__':
    print("Student-t(nu) closure ratio c(nu)")
    print("=" * 70)
    print(f"{'nu':>5s} {'c(analytical)':>14s} {'c(numerical)':>14s} {'fraction':>15s} {'match':>8s}")
    print("-" * 60)

    for nu in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100]:
        c_a = c_student_t_analytical(nu)
        c_n = c_student_t_numerical(nu)
        n, d, _ = simplify_c(nu)
        frac_str = f"{n}/{d}" if n is not None else "?"
        match = "OK" if abs(c_a-c_n)/c_a < 0.02 else "FAIL"
        print(f"{nu:5d} {c_a:14.8f} {c_n:14.8f} {frac_str:>15s} {match:>8s}")

    print(f"\n  nu->inf limit (Gaussian): c = 2/pi = {2/np.pi:.8f}")

    # Try to find pattern in the fractions
    print("\nPattern analysis:")
    for nu in [1, 3, 5, 7, 9, 11]:
        n, d, c = simplify_c(nu)
        if n is not None:
            print(f"  nu={nu}: c = {n}/{d} = {c:.8f}")

    # Check if c(nu) = Gamma(...)/Gamma(...)
    print("\nGamma function formula:")
    print("  c(nu) = 2*J/(pi^2*g0)")
    print("  where J = ((nu+1)/2)*sqrt(pi)*Gamma((nu+2)/2)/(Gamma((nu+3)/2)*sqrt(nu))")
    print("  and g0 = Gamma((nu+1)/2)/(sqrt(nu*pi)*Gamma(nu/2))")
    print("\n  Simplified:")
    for nu in [1, 3, 5, 7]:
        p = (nu+1)/2
        # c = 2/pi^2 * (nu+1)/2 * sqrt(pi) * Gamma(p+1/2) / (Gamma(p+1)*sqrt(nu))
        #     * sqrt(nu*pi)*Gamma(nu/2) / Gamma((nu+1)/2)
        # = 2/pi^2 * (nu+1)/2 * pi * Gamma(p+1/2)*Gamma(nu/2) / (Gamma(p+1)*Gamma(p))
        # = (nu+1)/(pi) * Gamma((nu+2)/2)*Gamma(nu/2) / (Gamma((nu+3)/2)*Gamma((nu+1)/2))
        c_check = (nu+1)/np.pi * gamma((nu+2)/2)*gamma(nu/2) / (gamma((nu+3)/2)*gamma((nu+1)/2))
        c_exact = c_student_t_analytical(nu)
        print(f"  nu={nu}: simplified = {c_check:.8f}, exact = {c_exact:.8f}, match = {abs(c_check-c_exact)<1e-8}")
