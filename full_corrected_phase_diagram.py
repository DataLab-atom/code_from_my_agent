"""
The REAL question: what does the corrected phase diagram look like?

The onset expansion (c = 2/pi at r->0) only tells us about the bifurcation type.
But the actual synchronization CONDITION for the higher-order model requires
solving the full self-consistent equation:

  h = (K2 + K3*Z2(h)) * Z1(h)

where Z1(h) and Z2(h) are computed from the full stationary distribution.

This script:
1. Computes c(r) = Z2/Z1^2 as a function of r for Gaussian
2. Solves the FULL corrected self-consistent equation across K2-K3 plane
3. Compares with the approximate (Z2=r^2) phase diagram
4. Identifies where the two qualitatively differ
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import json


def gaussian_g(omega, sigma=1.0):
    return np.exp(-omega**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))


def compute_Z1_Z2(h, sigma=1.0):
    """Full Z1, Z2 from stationary distribution at force amplitude h."""
    if h < 1e-15:
        return 0.0, 0.0
    g = lambda w: gaussian_g(w, sigma)

    Z1, _ = quad(lambda w: np.sqrt(1-(w/h)**2)*g(w), -h, h, limit=200)
    Z2L, _ = quad(lambda w: (1-2*(w/h)**2)*g(w), -h, h, limit=200)

    def z2d(w):
        return -(w-np.sqrt(w**2-h**2))**2/h**2 * g(w)
    Z2D, _ = quad(z2d, h*1.00001, 20.0, limit=200, points=[h*1.001])
    Z2D *= 2

    return Z1, Z2L + Z2D


# Part 1: c(r) as a function of r
print("Part 1: c(r) = Z2/Z1^2 as function of r")
print("=" * 50)

sigma = 1.0
g0 = gaussian_g(0, sigma)
Kc = 2/(np.pi*g0)

h_values = np.logspace(-2, 1.2, 40)
print(f"{'h':>10s} {'Z1=r':>10s} {'Z2':>12s} {'c=Z2/r^2':>10s}")
print("-" * 45)
for h in h_values:
    Z1, Z2 = compute_Z1_Z2(h, sigma)
    c = Z2/Z1**2 if Z1 > 1e-10 else float('nan')
    if Z1 > 0.01:
        print(f"{h:10.4f} {Z1:10.4f} {Z2:12.6f} {c:10.4f}")


# Part 2: Phase diagram comparison
print("\n\nPart 2: Phase diagram (corrected vs approximate)")
print("=" * 50)

K2_range = np.arange(0.5, 3.5, 0.1)
K3_range = np.arange(-1.0, 4.1, 0.25)


def solve_corrected(K2, K3, sigma):
    """Solve h = (K2+K3*Z2(h))*Z1(h)"""
    def eq(h):
        Z1, Z2 = compute_Z1_Z2(h, sigma)
        return (K2 + K3*Z2)*Z1 - h

    h_test = np.logspace(-2, 1.2, 60)
    roots = []
    prev_v = None
    for h in h_test:
        try:
            v = eq(h)
            if prev_v is not None and v*prev_v < 0:
                h_star = brentq(eq, h_prev, h, xtol=1e-8)
                Z1, Z2 = compute_Z1_Z2(h_star, sigma)
                roots.append(Z1)
            prev_v = v
            h_prev = h
        except:
            prev_v = None

    return max(roots) if roots else 0.0


def solve_approx(K2, K3, sigma):
    """Solve with Z2=r^2 assumption"""
    g = lambda w: gaussian_g(w, sigma)

    def eq(r):
        Keff = K2 + K3*r**2
        if Keff <= 0:
            return -r
        def integ(theta):
            return np.cos(theta)**2 * g(Keff*r*np.sin(theta))
        I, _ = quad(integ, -np.pi/2, np.pi/2)
        return Keff*r*I - r

    r_test = np.linspace(0.01, 0.999, 80)
    roots = []
    prev_v = eq(r_test[0])
    for i in range(1, len(r_test)):
        v = eq(r_test[i])
        if v*prev_v < 0:
            try:
                roots.append(brentq(eq, r_test[i-1], r_test[i], xtol=1e-6))
            except:
                pass
        prev_v = v
    return max(roots) if roots else 0.0


# Compute both phase diagrams
print(f"\nComputing phase diagrams ({len(K2_range)}x{len(K3_range)} grid)...")
r_corr = np.zeros((len(K2_range), len(K3_range)))
r_approx = np.zeros((len(K2_range), len(K3_range)))

for i, K2 in enumerate(K2_range):
    for j, K3 in enumerate(K3_range):
        r_corr[i,j] = solve_corrected(K2, K3, sigma)
        r_approx[i,j] = solve_approx(K2, K3, sigma)
    print(f"  K2={K2:.1f}: max |r_corr-r_approx| = {np.max(np.abs(r_corr[i]-r_approx[i])):.4f}")


# Part 3: Where do they qualitatively differ?
print("\n\nPart 3: Qualitative differences")
print("=" * 50)

diff = np.abs(r_corr - r_approx)
# Find where corrected has solution but approximate doesn't (or vice versa)
corr_sync = r_corr > 0.01
approx_sync = r_approx > 0.01
disagree = corr_sync != approx_sync

print(f"Grid points where corrected and approximate DISAGREE on sync/no-sync:")
for i in range(len(K2_range)):
    for j in range(len(K3_range)):
        if disagree[i,j]:
            print(f"  K2={K2_range[i]:.1f}, K3={K3_range[j]:.2f}: "
                  f"r_corr={r_corr[i,j]:.3f}, r_approx={r_approx[i,j]:.3f}")

# Save results
output = {
    'sigma': sigma, 'Kc': Kc,
    'K2_range': K2_range.tolist(), 'K3_range': K3_range.tolist(),
    'r_corrected': r_corr.tolist(), 'r_approximate': r_approx.tolist()
}
with open('corrected_phase_diagram.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved to corrected_phase_diagram.json")
