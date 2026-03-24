"""
GENERAL FORMULA for the OA closure ratio c[g]:

    c[g] = (2 / (pi^2 * g(0)^2)) * integral_0^inf (g(0) - g(omega)) / omega^2 d_omega

Derivation:
  Near onset (h->0), the drift oscillator correction to Z2 is:
    Delta_drift ~ h/2 * integral_0^inf (1 - g(omega)/g(0)) / omega^2 d_omega
  Combined with Z1 ~ h*g(0)*pi/2, this gives c = Z2/Z1^2 = 2*J/(pi^2*g(0)).

Verification against numerical c values for multiple distributions.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def c_general(g_func, g0, omega_max=200):
    """Compute c[g] using the general formula.
    c = 2*J_code / (pi^2 * g0^2)
    where J_code = integral (g(0)-g(omega))/omega^2 dw = g(0) * J
    """
    def integrand(omega):
        return (g0 - g_func(omega)) / omega**2

    J_code, _ = quad(integrand, 1e-10, omega_max, limit=300)
    return 2 * J_code / (np.pi**2 * g0**2)


def c_numerical(g_func, h=0.002, omega_max=20):
    """Compute c by direct Z1, Z2 calculation."""
    def z1_int(omega):
        return np.sqrt(1-(omega/h)**2) * g_func(omega)
    def z2_L(omega):
        return (1-2*(omega/h)**2) * g_func(omega)
    def z2_D(omega):
        return -(omega-np.sqrt(omega**2-h**2))**2/h**2 * g_func(omega)

    Z1, _ = quad(z1_int, -h, h, limit=200)
    Z2L, _ = quad(z2_L, -h, h, limit=200)
    Z2D, _ = quad(z2_D, h*1.00001, omega_max, limit=200, points=[h*1.001])
    Z2D *= 2
    Z2 = Z2L + Z2D
    return Z2/Z1**2 if Z1**2 > 1e-30 else float('nan')


if __name__ == '__main__':
    print("GENERAL FORMULA: c[g] = 2*J/(pi^2*g(0))")
    print("where J = integral_0^inf (g(0)-g(omega))/omega^2 d_omega")
    print("=" * 70)

    results = []

    # 1. Lorentzian
    Delta = 1.0
    g0 = 1/(np.pi*Delta)
    g = lambda w: (Delta/np.pi)/(w**2+Delta**2)
    # J = pi/(2*Delta) analytically
    J_exact = np.pi/(2*Delta)
    c_exact = 1.0
    c_form = c_general(g, g0, omega_max=5000)
    c_num = c_numerical(g, h=0.002)
    results.append(("Lorentzian", c_exact, c_form, c_num, J_exact))

    # 2. Gaussian
    sigma = 1.0
    g0 = 1/(sigma*np.sqrt(2*np.pi))
    g = lambda w: np.exp(-w**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    # J = sqrt(pi/(2*sigma^2)) = sqrt(pi)/(sigma*sqrt(2)) analytically
    J_exact = np.sqrt(np.pi)/(sigma*np.sqrt(2))
    c_exact = 2/np.pi
    c_form = c_general(g, g0, omega_max=200)
    c_num = c_numerical(g, h=0.002)
    results.append(("Gaussian", c_exact, c_form, c_num, J_exact))

    # 3. Uniform
    a = 1.0
    g0 = 1/(2*a)
    g = lambda w: 1/(2*a) if abs(w) <= a else 0.0
    # J = 1/a analytically
    J_exact = 1/a
    c_exact = 4/np.pi**2
    c_form = c_general(g, g0, omega_max=200)  # must integrate beyond support!
    c_num = c_numerical(g, h=0.002, omega_max=a)
    results.append(("Uniform", c_exact, c_form, c_num, J_exact))

    # 4. Student-t (nu=3)
    nu = 3
    g0_t = gamma((nu+1)/2)/(np.sqrt(nu*np.pi)*gamma(nu/2))
    g_t = lambda w: g0_t*(1+w**2/nu)**(-(nu+1)/2)
    # J = sqrt(3)*pi/4 analytically (from manual calculation)
    J_exact_t3 = np.sqrt(3)*np.pi/4
    c_exact_t3 = 3/4
    c_form = c_general(g_t, g0_t, omega_max=5000)
    c_num = c_numerical(g_t, h=0.002)
    results.append(("Student-t(3)", c_exact_t3, c_form, c_num, J_exact_t3))

    # 5. Student-t (nu=5)
    nu = 5
    g0_t5 = gamma(3)/(np.sqrt(5*np.pi)*gamma(5/2))
    g_t5 = lambda w: g0_t5*(1+w**2/5)**(-3)
    c_form = c_general(g_t5, g0_t5, omega_max=5000)
    c_num = c_numerical(g_t5, h=0.002)
    results.append(("Student-t(5)", 45/64, c_form, c_num, None))

    # Print results
    print(f"\n{'Distribution':>20s} {'c(exact)':>10s} {'c(formula)':>12s} {'c(numerical)':>12s} {'match?':>8s}")
    print("-" * 65)
    for name, c_ex, c_fo, c_nu, J in results:
        match_f = abs(c_ex - c_fo)/c_ex < 0.02
        match_n = abs(c_ex - c_nu)/c_ex < 0.02
        print(f"{name:>20s} {c_ex:10.6f} {c_fo:12.6f} {c_nu:12.6f} {'OK' if match_f and match_n else 'FAIL':>8s}")

    # The corrected explosive formula
    print("\n" + "=" * 70)
    print("CORRECTED EXPLOSIVE FORMULA:")
    print("  K3_exp = Kc^3 |g''(0)| / (8 c[g] g(0))")
    print("  c[g] = (2/(pi^2 g(0)^2)) * integral_0^inf (g(0)-g(omega))/omega^2 d_omega")
    print("=" * 70)

    print("\nThis makes the explosive threshold fully explicit:")
    print("  Given ONLY g(omega), compute three numbers:")
    print("  1. g(0)  -- height at center")
    print("  2. g''(0) -- curvature at center")
    print("  3. J[g] = integral (g(0)-g(omega))/omega^2 dw -- tail decay integral")
    print("  Then: Kc = 2/(pi*g(0)), c = 2*J/(pi^2*g(0)), K3_exp = Kc^3|g''(0)|/(8*c*g(0))")
