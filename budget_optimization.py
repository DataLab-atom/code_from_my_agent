"""
Budget-constrained optimal allocation: K2 + K3 = C

Under OA: dr/dt = -Delta*r + r(1-r^2)/2 * (K2 + K3*r^2)
With constraint K2 + K3 = C, substitute K2 = C - K3:

dr/dt = -Delta*r + r(1-r^2)/2 * (C - K3 + K3*r^2)
      = -Delta*r + r(1-r^2)/2 * (C - K3(1-r^2))
      = -Delta*r + r(1-r^2)/2 * (C - K3*(1-r^2))

Fixed point r* satisfies:
  Delta = (1-r^2)/2 * (C - K3*(1-r^2))

Let u = 1-r^2 (so r^2 = 1-u, u in (0,1)):
  Delta = u/2 * (C - K3*u)
  2*Delta = C*u - K3*u^2
  K3*u^2 - C*u + 2*Delta = 0
  u = (C +/- sqrt(C^2 - 8*Delta*K3)) / (2*K3)

Optimal K3* maximizes r* (minimizes u*):
  du*/dK3 = 0

This is an analytically solvable optimization problem.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from analytical_phase_diagram import delta_from_sigma, compute_order_parameter_analytical
import json


def r_star_budget(K3, C, Delta):
    """
    Analytical r* under budget constraint K2 + K3 = C.
    """
    K2 = C - K3
    if K2 < 0:
        return 0.0
    return compute_order_parameter_analytical(K2, K3, Delta)


def optimal_K3_analytical(C, Delta):
    """
    Find K3* that maximizes r* under K2 + K3 = C.

    From the fixed-point equation with u = 1-r^2:
      K3*u^2 - C*u + 2*Delta = 0

    Taking du/dK3 = 0 and using implicit differentiation:
      du/dK3 = u^2 / (2*K3*u - C)

    This is zero only at u=0 (full sync, trivial).
    At the boundary of validity:

    The optimal is found by maximizing r* numerically.
    """
    if C <= 2 * Delta:
        return None, 0.0  # Below threshold for any allocation

    result = minimize_scalar(
        lambda K3: -r_star_budget(K3, C, Delta),
        bounds=(0, C),
        method='bounded'
    )
    K3_opt = result.x
    r_opt = -result.fun
    return float(K3_opt), float(r_opt)


def optimal_K3_basin(K3, C, Delta):
    """
    Basin probability proxy: how far is r_sep from 0?
    Lower r_sep = larger basin.
    Under OA, r_sep is the unstable fixed point.
    """
    K2 = C - K3
    if K2 < 0 or K2 <= 2 * Delta:
        return 1.0  # No sync possible, worst basin

    # Find unstable fixed point
    from analytical_phase_diagram import find_fixed_points, rdot
    fps = find_fixed_points(K2, K3, Delta)
    if len(fps) < 2:
        return 0.0  # Only one stable FP, full basin

    # Check stability of each
    unstable = []
    for r_fp in fps:
        dr = 1e-6
        slope = (rdot(r_fp + dr, K2, K3, Delta) - rdot(r_fp - dr, K2, K3, Delta)) / (2 * dr)
        if slope > 0:
            unstable.append(r_fp)

    if unstable:
        return min(unstable)  # Lower separatrix = larger basin
    return 0.0


def pareto_front(C, Delta, n_points=100):
    """
    Compute Pareto front: r* vs basin size as K3 varies along K2+K3=C.
    """
    K3_range = np.linspace(0, C * 0.95, n_points)
    points = []
    for K3 in K3_range:
        r = r_star_budget(K3, C, Delta)
        basin_sep = optimal_K3_basin(K3, C, Delta)
        points.append({
            'K3': float(K3),
            'K2': float(C - K3),
            'r_star': float(r),
            'r_sep': float(basin_sep),
            'basin_proxy': float(1 - basin_sep)  # Larger = better
        })
    return points


if __name__ == '__main__':
    sigmas = [0.5, 1.0, 1.5]
    budgets = [1.0, 2.0, 3.0, 4.0, 5.0]

    all_results = {}

    print("=== Budget-Constrained Optimal Allocation ===\n")

    for sigma in sigmas:
        Delta = delta_from_sigma(sigma)
        print(f"--- sigma={sigma}, Delta={Delta:.4f}, Kc={2*Delta:.4f} ---")

        sigma_results = {}
        for C in budgets:
            K3_opt, r_opt = optimal_K3_analytical(C, Delta)
            if K3_opt is not None:
                K2_opt = C - K3_opt
                ratio = K3_opt / C if C > 0 else 0

                # Compare with pure pairwise (K3=0)
                r_pure = r_star_budget(0, C, Delta)

                # Pareto front
                pareto = pareto_front(C, Delta)

                print(f"  C={C:.1f}: K3*={K3_opt:.3f} (K2*={K2_opt:.3f}), "
                      f"ratio={ratio:.2f}, r*={r_opt:.4f}, r_pure={r_pure:.4f}, "
                      f"gain={r_opt-r_pure:+.4f}")

                sigma_results[f'C={C}'] = {
                    'K3_optimal': K3_opt,
                    'K2_optimal': K2_opt,
                    'ratio_K3_C': ratio,
                    'r_optimal': r_opt,
                    'r_pure_pairwise': r_pure,
                    'gain': r_opt - r_pure,
                    'pareto': pareto
                }
            else:
                print(f"  C={C:.1f}: below threshold")
                sigma_results[f'C={C}'] = {'below_threshold': True}

        all_results[f'sigma={sigma}'] = sigma_results
        print()

    output = {
        'sigmas': sigmas,
        'budgets': budgets,
        'results': all_results
    }
    with open('budget_optimization.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Results saved to budget_optimization.json")
