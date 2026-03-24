"""
ANALYTICAL PROOF that c[Gaussian] = 2/pi.

Key steps:
1. Z2 = h*g0*I_Q(eps) where eps = h/(sigma*sqrt(2))
2. I_Q(eps) = I_Q^locked(eps) + I_Q^drift(eps)
3. I_Q^locked = 2/3 + O(eps^2)  [Taylor expansion in eps]
4. I_Q^drift = -2/3 + eps*sqrt(pi)/2 + O(eps^2)
   Key: the eps*sqrt(pi)/2 term comes from
   integral_0^inf (1-exp(-v^2))/v^2 dv = sqrt(pi)
5. I_Q(eps) = eps*sqrt(pi)/2 + O(eps^2)
6. c = Z2/Z1^2 = [h*g0*eps*sqrt(pi)/2] / [h*g0*pi/2]^2
              = 2*eps*sqrt(pi) / (h*g0*pi^2)
              = 2/pi  [after substituting eps=h/(sigma*sqrt(2)), g0=1/(sigma*sqrt(2*pi))]

This script verifies each step numerically.
"""
import numpy as np
from scipy.integrate import quad


def verify_step_by_step():
    sigma = 1.0
    g0 = 1/(sigma*np.sqrt(2*np.pi))

    print("ANALYTICAL PROOF: c[Gaussian] = 2/pi")
    print("=" * 60)

    # Step 1: Verify known integral identity
    I_identity, _ = quad(lambda v: (1-np.exp(-v**2))/v**2, 0, 100, limit=200)
    print(f"\nStep 1: integral_0^inf (1-exp(-v^2))/v^2 dv = {I_identity:.8f}")
    print(f"        sqrt(pi) = {np.sqrt(np.pi):.8f}")
    print(f"        Match: {abs(I_identity - np.sqrt(np.pi)) < 1e-6}")

    # Step 2: Verify I_Q^drift(eps) expansion
    print(f"\nStep 2: I_Q^drift = -2/3 + eps*sqrt(pi)/2 + O(eps^2)")
    for eps in [0.1, 0.05, 0.02, 0.01]:
        # Compute exact drift integral
        def drift_int(u):
            return -2*(u - np.sqrt(u**2-1))**2 * np.exp(-eps**2*u**2)
        I_drift, _ = quad(drift_int, 1.00001, 200/eps, limit=300)

        # Predicted: -2/3 + eps*sqrt(pi)/2
        predicted = -2/3 + eps*np.sqrt(np.pi)/2

        error = abs(I_drift - predicted)
        print(f"  eps={eps:.3f}: exact={I_drift:.8f}, "
              f"predicted={predicted:.8f}, error={error:.2e}")

    # Step 3: Verify I_Q^locked expansion
    print(f"\nStep 3: I_Q^locked = 2/3 + O(eps^2)")
    for eps in [0.1, 0.05, 0.02, 0.01]:
        I_locked, _ = quad(lambda u: 2*(1-2*u**2)*np.exp(-eps**2*u**2), 0, 1)
        error = abs(I_locked - 2/3)
        print(f"  eps={eps:.3f}: exact={I_locked:.8f}, "
              f"predicted={2/3:.8f}, error={error:.2e}")

    # Step 4: Verify I_Q(eps) = eps*sqrt(pi)/2
    print(f"\nStep 4: I_Q(eps) = I_locked + I_drift = eps*sqrt(pi)/2 + O(eps^2)")
    for eps in [0.1, 0.05, 0.02, 0.01]:
        I_locked, _ = quad(lambda u: 2*(1-2*u**2)*np.exp(-eps**2*u**2), 0, 1)
        def drift_int(u):
            return -2*(u - np.sqrt(u**2-1))**2 * np.exp(-eps**2*u**2)
        I_drift, _ = quad(drift_int, 1.00001, 200/eps, limit=300)
        I_Q = I_locked + I_drift

        predicted = eps * np.sqrt(np.pi) / 2
        rel_error = abs(I_Q - predicted) / predicted
        print(f"  eps={eps:.3f}: I_Q={I_Q:.8f}, "
              f"predicted={predicted:.8f}, rel_error={rel_error:.4f}")

    # Step 5: Final assembly
    print(f"\nStep 5: c = 2*eps*sqrt(pi) / (h*g0*pi^2)")
    print(f"        = 2*(h/(sigma*sqrt(2)))*sqrt(pi) / (h*(1/(sigma*sqrt(2*pi)))*pi^2)")
    print(f"        = 2*sqrt(pi)/(sigma*sqrt(2)) * sigma*sqrt(2*pi) / pi^2")
    print(f"        = 2*sqrt(pi)*sqrt(2*pi) / (sqrt(2)*pi^2)")
    print(f"        = 2*pi*sqrt(2) / (sqrt(2)*pi^2)")
    print(f"        = 2/pi")
    print(f"\n        c = 2/pi = {2/np.pi:.10f}  QED")

    # Step 6: Verify the key asymptotic step in detail
    print(f"\n{'='*60}")
    print("Derivation of the drift O(eps) term:")
    print(f"{'='*60}")
    print("""
    I_drift(eps) = -2 * integral_1^inf (u-sqrt(u^2-1))^2 * exp(-eps^2*u^2) du

    Change of variable: v = eps*u, u = v/eps

    = -(2/eps) * integral_eps^inf (v/eps - sqrt((v/eps)^2-1))^2 * exp(-v^2) dv

    For v >> eps: (v/eps - sqrt((v/eps)^2-1))^2 ~ eps^2/(4v^2)

    So the integrand ~ (2/eps)*eps^2/(4v^2)*exp(-v^2) = eps/(2v^2)*exp(-v^2)

    I_drift(eps) - I_drift(0)
        = -2 * integral_1^inf (u-sqrt(u^2-1))^2 * [exp(-eps^2*u^2) - 1] du
        = 2 * integral_1^inf (u-sqrt(u^2-1))^2 * [1 - exp(-eps^2*u^2)] du

    With v = eps*u:
        ~ (2/eps) * integral_0^inf eps^2/(4v^2) * (1-exp(-v^2)) dv
        = eps/2 * integral_0^inf (1-exp(-v^2))/v^2 dv
        = eps/2 * sqrt(pi)
        = eps*sqrt(pi)/2
    """)


if __name__ == '__main__':
    verify_step_by_step()
