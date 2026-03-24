"""
Debug the disagreement point K2=1.0, K3=2.0, sigma=1.0.

Approximate says r=0.875, corrected says r=0.
Is the corrected equation solver missing a root?

Strategy: plot both equations' F(r) = RHS - r to see all roots.
"""
import numpy as np
from scipy.integrate import quad
import json


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


K2, K3, sigma = 1.0, 2.0, 1.0
g0 = gaussian_g(0, sigma)

print(f"K2={K2}, K3={K3}, sigma={sigma}")
print(f"Kc = {2/(np.pi*g0):.4f}")
print()

# Scan h space: equation is h = (K2 + K3*Z2(h)) * Z1(h)
# Equivalently: F(h) = (K2+K3*Z2)*Z1 - h = 0

print("Corrected equation: F(h) = (K2+K3*Z2(h))*Z1(h) - h")
print(f"{'h':>10s} {'Z1':>10s} {'Z2':>12s} {'K_eff':>10s} {'RHS':>12s} {'F(h)':>12s}")
print("-" * 70)

h_vals = np.logspace(-2, 1.5, 100)
F_vals = []
for h in h_vals:
    Z1, Z2 = compute_Z1_Z2(h, sigma)
    Keff = K2 + K3*Z2
    RHS = Keff * Z1
    F = RHS - h
    F_vals.append(F)
    if h < 0.02 or h > 10 or abs(F) < 0.3 or Z1 > 0.8:
        print(f"{h:10.4f} {Z1:10.4f} {Z2:12.6f} {Keff:10.4f} {RHS:12.6f} {F:12.6f}")

# Check for sign changes
sign_changes = []
for i in range(len(F_vals)-1):
    if F_vals[i]*F_vals[i+1] < 0:
        sign_changes.append((h_vals[i], h_vals[i+1], F_vals[i], F_vals[i+1]))

print(f"\nSign changes in F(h): {len(sign_changes)}")
for h1, h2, f1, f2 in sign_changes:
    print(f"  h in ({h1:.4f}, {h2:.4f}): F goes from {f1:.6f} to {f2:.6f}")

# Also check the approximate equation
print("\n\nApproximate equation: F(r) = (K2+K3*r^2)*r*I(r) - r")
print(f"{'r':>10s} {'K_eff':>10s} {'integral':>12s} {'RHS':>12s} {'F(r)':>12s}")
print("-" * 60)

g = lambda w: gaussian_g(w, sigma)
r_vals = np.linspace(0.01, 0.99, 100)
F_approx = []
for r in r_vals:
    Keff = K2 + K3*r**2
    def integ(theta):
        return np.cos(theta)**2 * g(Keff*r*np.sin(theta))
    I, _ = quad(integ, -np.pi/2, np.pi/2)
    RHS = Keff*r*I
    F = RHS - r
    F_approx.append(F)
    if r < 0.05 or r > 0.95 or abs(F) < 0.05:
        print(f"{r:10.4f} {Keff:10.4f} {I:12.6f} {RHS:12.6f} {F:12.6f}")

sign_changes_a = []
for i in range(len(F_approx)-1):
    if F_approx[i]*F_approx[i+1] < 0:
        sign_changes_a.append((r_vals[i], r_vals[i+1]))

print(f"\nSign changes in approximate F(r): {len(sign_changes_a)}")
for r1, r2 in sign_changes_a:
    print(f"  r in ({r1:.4f}, {r2:.4f})")

# Key diagnostic: compare K_eff * Z1 at the approximate root
if sign_changes_a:
    r_root = (sign_changes_a[0][0] + sign_changes_a[0][1])/2
    Keff_approx = K2 + K3*r_root**2
    h_approx = Keff_approx * r_root
    Z1_at_h, Z2_at_h = compute_Z1_Z2(h_approx, sigma)
    Keff_corr = K2 + K3*Z2_at_h
    print(f"\nAt approximate root r={r_root:.4f}:")
    print(f"  h_approx = {h_approx:.4f}")
    print(f"  Z1(h) = {Z1_at_h:.4f}, Z2(h) = {Z2_at_h:.6f}")
    print(f"  Z2/Z1^2 = {Z2_at_h/Z1_at_h**2:.4f}")
    print(f"  K_eff (approx, Z2=r^2) = {Keff_approx:.4f}")
    print(f"  K_eff (corrected, Z2 from dist) = {Keff_corr:.4f}")
    print(f"  (K_eff_corr * Z1) = {Keff_corr*Z1_at_h:.4f} vs h = {h_approx:.4f}")
    print(f"  Deficit: {Keff_corr*Z1_at_h - h_approx:.4f}")
