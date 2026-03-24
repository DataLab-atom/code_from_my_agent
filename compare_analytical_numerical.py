"""
Compare Ott-Antonsen analytical predictions with numerical simulation data.
Run this after GPU-Claude-Opus provides simulation results.

Usage:
  python compare_analytical_numerical.py <numerical_data.json>
"""

import numpy as np
import json
import sys
from analytical_phase_diagram import (
    delta_from_sigma, compute_order_parameter_analytical,
    onset_boundary, find_fixed_points
)


def load_numerical_data(filepath):
    """Load numerical simulation results (JSON from GPU-Claude-Opus)"""
    with open(filepath) as f:
        return json.load(f)


def compare_Kc(numerical_data, sigma=1.0):
    """
    Compare numerical vs analytical critical coupling.
    Analytical: Kc = 2*Delta = 2*sigma*sqrt(2pi)/pi
    """
    Delta = delta_from_sigma(sigma)
    Kc_analytical = 2 * Delta

    # Find numerical Kc: r crosses 0.1 threshold
    if 'K2_list' in numerical_data and 'r' in numerical_data:
        K2_list = np.array(numerical_data['K2_list'])
        # If 2D data, take K3=0 slice
        r_data = np.array(numerical_data['r'])
        if r_data.ndim == 2:
            K3_list = np.array(numerical_data['K3_list'])
            k3_zero_idx = np.argmin(np.abs(K3_list))
            r_slice = r_data[:, k3_zero_idx]
        else:
            r_slice = r_data

        # Find crossing
        Kc_numerical = None
        for i in range(len(r_slice) - 1):
            if r_slice[i] < 0.1 <= r_slice[i + 1]:
                # Linear interpolation
                Kc_numerical = K2_list[i] + (0.1 - r_slice[i]) / (
                    r_slice[i + 1] - r_slice[i]) * (K2_list[i + 1] - K2_list[i])
                break

        if Kc_numerical:
            error = abs(Kc_numerical - Kc_analytical) / Kc_analytical * 100
            return {
                'Kc_analytical': float(Kc_analytical),
                'Kc_numerical': float(Kc_numerical),
                'error_percent': float(error),
                'sigma': sigma
            }

    return {'Kc_analytical': float(Kc_analytical), 'Kc_numerical': None}


def compare_phase_diagram(numerical_data, sigma=1.0):
    """
    Compare r*(K2, K3) between analytical OA and numerical.
    Returns MAE, max error, and location of max error.
    """
    Delta = delta_from_sigma(sigma)
    K2_list = np.array(numerical_data['K2_list'])
    K3_list = np.array(numerical_data['K3_list'])
    r_numerical = np.array(numerical_data['r'])

    nK2, nK3 = r_numerical.shape
    r_analytical = np.zeros_like(r_numerical)

    for i, K2 in enumerate(K2_list):
        for j, K3 in enumerate(K3_list):
            r_analytical[i, j] = compute_order_parameter_analytical(K2, K3, Delta)

    diff = np.abs(r_analytical - r_numerical)
    mae = float(np.mean(diff))
    max_err = float(np.max(diff))
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    max_K2 = float(K2_list[max_idx[0]])
    max_K3 = float(K3_list[max_idx[1]])

    return {
        'MAE': mae,
        'max_error': max_err,
        'max_error_at': {'K2': max_K2, 'K3': max_K3},
        'r_analytical': r_analytical.tolist(),
        'r_numerical': r_numerical.tolist(),
        'diff': diff.tolist()
    }


def check_explosive_boundary(numerical_data, sigma=1.0):
    """
    Check if the explosive (discontinuous) transition boundary K3=K2
    is visible in numerical data (hysteresis or jump).
    """
    K2_list = np.array(numerical_data['K2_list'])
    K3_list = np.array(numerical_data['K3_list'])
    r_data = np.array(numerical_data['r'])

    # Along the diagonal K3=K2, check for large r gradients
    boundary_points = []
    for i, K2 in enumerate(K2_list):
        j_diag = np.argmin(np.abs(K3_list - K2))
        if j_diag > 0 and j_diag < len(K3_list) - 1:
            r_below = r_data[i, j_diag - 1]
            r_at = r_data[i, j_diag]
            r_above = r_data[i, j_diag + 1]
            gradient = abs(r_above - r_below)
            if gradient > 0.3:  # Large jump indicates explosive transition
                boundary_points.append({
                    'K2': float(K2),
                    'K3': float(K3_list[j_diag]),
                    'r_gradient': float(gradient)
                })

    return {
        'explosive_boundary_detected': len(boundary_points) > 0,
        'boundary_points': boundary_points,
        'analytical_prediction': 'K3 = K2 (diagonal)'
    }


def sigma_scaling_test(numerical_3d_data):
    """
    Test sigma-scaling proposition: phase boundaries should collapse
    when plotted in (K2/sigma, K3/sigma) coordinates.
    """
    sigma_list = numerical_3d_data['sigma_list']
    K2_list = np.array(numerical_3d_data['K2_list'])
    K3_list = np.array(numerical_3d_data['K3_list'])
    r_3d = np.array(numerical_3d_data['r'])  # [n_sigma, nK2, nK3]

    # For each sigma, find r=0.5 contour in rescaled coordinates
    contours = {}
    for s_idx, sigma in enumerate(sigma_list):
        K2_rescaled = K2_list / sigma
        K3_rescaled = K3_list / sigma
        r_slice = r_3d[s_idx]

        # Find boundary: r=0.5 iso-contour
        boundary = []
        for i in range(len(K2_list)):
            for j in range(len(K3_list) - 1):
                if (r_slice[i, j] - 0.5) * (r_slice[i, j + 1] - 0.5) < 0:
                    # Interpolate
                    K3_cross = K3_rescaled[j] + (0.5 - r_slice[i, j]) / (
                        r_slice[i, j + 1] - r_slice[i, j]) * (
                        K3_rescaled[j + 1] - K3_rescaled[j])
                    boundary.append((float(K2_rescaled[i]), float(K3_cross)))

        contours[f'sigma={sigma}'] = boundary

    # Measure collapse quality: variance of contour positions across sigmas
    return contours


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compare_analytical_numerical.py <data.json>")
        print("\nRunning self-test with analytical data only...")

        # Self-test: compare analytical with itself (should give zero error)
        sigma = 1.0
        Delta = delta_from_sigma(sigma)
        K2_range = np.linspace(0, 8, 40)
        K3_range = np.linspace(-4, 4, 40)

        print(f"Kc analytical = {2*Delta:.4f}")

        # Generate fake "numerical" data from analytical (for testing)
        r_mat = np.zeros((40, 40))
        for i, K2 in enumerate(K2_range):
            for j, K3 in enumerate(K3_range):
                r_mat[i, j] = compute_order_parameter_analytical(K2, K3, Delta)

        fake_data = {
            'K2_list': K2_range.tolist(),
            'K3_list': K3_range.tolist(),
            'r': r_mat.tolist()
        }

        result = compare_Kc(fake_data, sigma)
        print(f"Kc test: {result}")

        result2 = compare_phase_diagram(fake_data, sigma)
        print(f"Phase diagram MAE (self-test): {result2['MAE']:.6f}")
        print(f"Max error (should be ~0): {result2['max_error']:.6f}")

    else:
        filepath = sys.argv[1]
        print(f"Loading {filepath}...")
        data = load_numerical_data(filepath)

        sigma = data.get('sigma', 1.0)
        print(f"\n=== Comparison Report (sigma={sigma}) ===\n")

        # Kc comparison
        kc_result = compare_Kc(data, sigma)
        print(f"Critical coupling:")
        print(f"  Analytical: Kc = {kc_result['Kc_analytical']:.4f}")
        if kc_result.get('Kc_numerical'):
            print(f"  Numerical:  Kc = {kc_result['Kc_numerical']:.4f}")
            print(f"  Error: {kc_result['error_percent']:.2f}%")

        # Phase diagram
        if 'K3_list' in data:
            pd_result = compare_phase_diagram(data, sigma)
            print(f"\nPhase diagram comparison:")
            print(f"  MAE: {pd_result['MAE']:.4f}")
            print(f"  Max error: {pd_result['max_error']:.4f} at "
                  f"K2={pd_result['max_error_at']['K2']:.2f}, "
                  f"K3={pd_result['max_error_at']['K3']:.2f}")

            # Explosive boundary
            exp_result = check_explosive_boundary(data, sigma)
            print(f"\nExplosive boundary (K3=K2):")
            print(f"  Detected: {exp_result['explosive_boundary_detected']}")
            if exp_result['boundary_points']:
                for bp in exp_result['boundary_points'][:5]:
                    print(f"    K2={bp['K2']:.2f}, K3={bp['K3']:.2f}, "
                          f"gradient={bp['r_gradient']:.3f}")

        # Save report
        report = {
            'Kc': kc_result,
            'phase_diagram': pd_result if 'K3_list' in data else None,
            'explosive': exp_result if 'K3_list' in data else None
        }
        with open('comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("\nReport saved to comparison_report.json")
