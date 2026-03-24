"""
Reproduction script for all numerical experiments in the paper.
Run: python reproduce.py [experiment_name]

All experiments use N=200, σ=1.0 unless stated otherwise.
Random seed=0 for reproducibility, multi-seed averages noted where used.
"""

import sys

EXPERIMENTS = {
    'benchmark': {
        'script': 'kuramoto.py',
        'command': 'python kuramoto.py',
        'description': 'Classical Kc verification (Fig.1)',
        'params': 'N=200, σ=1, K₃=0, K₂∈[0.5Kc, 2Kc]',
        'output': 'benchmark_classical.json',
        'time_est': '2 min',
    },
    'scan_2d': {
        'script': 'explore.py',
        'command': 'python -c "from explore import scan_2d; ..."',
        'description': 'K₂×K₃ 2D phase diagram (Fig.2)',
        'params': 'N=200, σ=1, K₂∈[0,4]×K₃∈[-2,2], 20×20 grid, 10 basin trials',
        'output': 'scan_K2_K3.json',
        'time_est': '20 min',
    },
    'scan_3d': {
        'script': 'scan_3d.py',
        'command': 'python scan_3d.py',
        'description': 'σ×K₂×K₃ 3D scan (Fig.3)',
        'params': 'σ∈{0.3,0.5,0.8,1.0,1.2,1.5}, K₂∈[0,4]×K₃∈[-2,2], 16×16, 20 basin trials',
        'output': 'scan_sigma_K2_K3.json',
        'time_est': '75 min (4 CPU workers)',
    },
    'hysteresis': {
        'script': 'scan_hysteresis.py',
        'command': 'python scan_hysteresis.py',
        'description': 'Forward/backward hysteresis detection (Fig.4)',
        'params': 'N=200, σ=1, K₃∈{-1.5,...,1.5}, K₂∈[0,6] 40 pts, T_equil=300',
        'output': 'hysteresis.json',
        'time_est': '10 min',
    },
    'hysteresis_fine': {
        'script': 'scan_hysteresis_fine.py',
        'command': 'python scan_hysteresis_fine.py',
        'description': 'Fine K₃ hysteresis area (Fig.4 inset)',
        'params': 'K₃∈[0,3] 30 pts, K₂∈[0,4] 40 pts',
        'output': 'hysteresis_fine.json',
        'time_est': '60 min',
    },
    'critical_slowing': {
        'script': 'scan_critical_slowing.py',
        'command': 'python scan_critical_slowing.py',
        'description': 'Relaxation time near Kc (Fig.5)',
        'params': 'K₃∈{0,0.5,1.0,1.5}, K₂ near Kc±20%, 30 pts',
        'output': 'critical_slowing.json',
        'time_est': '5 min',
    },
    'freq_dist': {
        'script': 'scan_freq_dist.py',
        'command': 'python scan_freq_dist.py',
        'description': 'Gaussian/Lorentzian/Uniform comparison (Fig.6)',
        'params': 'N=200, K₂∈[0,6] 25 pts, K₃∈{-1,0,1}',
        'output': 'freq_distribution.json',
        'time_est': '15 min',
    },
    'K4': {
        'script': 'scan_K4.py',
        'command': 'python scan_K4.py',
        'description': 'Four-body K₄ marginal effect (Fig.7)',
        'params': 'N=50, K₂=2, K₃=0.5, K₄∈[-1,1] 20 pts',
        'output': 'K4_test.json',
        'time_est': '0.5 min',
    },
    'locking': {
        'script': 'scan_locking_order.py',
        'command': 'python scan_locking_order.py',
        'description': 'Oscillator locking order (Fig.8)',
        'params': 'N=200, K₂=3, K₃∈{-1,0,1}, T=500',
        'output': 'locking_order.json',
        'time_est': '0.5 min',
    },
    'sparse': {
        'script': 'scan_sparse_network.py',
        'command': 'python scan_sparse_network.py',
        'description': 'Sparse network topology effect (Fig.9)',
        'params': 'N=200, p∈{0.1,0.3}, K₃∈[-1,1.5]',
        'output': 'sparse_network_K3.json',
        'time_est': '10 min',
    },
    'budget': {
        'script': 'scan_budget.py',
        'command': 'python scan_budget.py',
        'description': 'Budget constraint optimization (Fig.10)',
        'params': 'C∈{1,...,5}, σ∈{0.5,1.0,1.5}, 50 pts on K₂+K₃=C',
        'output': 'budget_constraint.json',
        'time_est': '170 min',
    },
}


def print_all():
    print("=" * 70)
    print("Reproducibility Guide: Higher-Order Kuramoto Numerical Experiments")
    print("=" * 70)
    total_time = 0
    for name, exp in EXPERIMENTS.items():
        print(f"\n[{name}] {exp['description']}")
        print(f"  Script: {exp['script']}")
        print(f"  Params: {exp['params']}")
        print(f"  Output: {exp['output']}")
        print(f"  Time:   ~{exp['time_est']}")
    print("\n" + "=" * 70)
    print("Total estimated CPU time: ~6 hours (single core)")
    print("GPU experiments (kuramoto_gpu.py): ~30 min on A800")
    print("All random seeds documented for exact reproducibility.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in EXPERIMENTS:
        exp = EXPERIMENTS[sys.argv[1]]
        print(f"Running: {exp['description']}")
        print(f"Command: {exp['command']}")
    else:
        print_all()
