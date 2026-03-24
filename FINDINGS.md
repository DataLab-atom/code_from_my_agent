# Key Findings: Higher-Order Kuramoto Synchronization

## Core Discovery: OA-Numerics Direction Reversal

**OA predicts**: K₃>0 decreases steady-state r* (narrower locking region)
**Numerics show**: K₃>0 increases r at all N=20-5000 (more oscillators lock)

**Mechanism**: K₃>0 enhances synchronization at finite N by **expanding the frequency-locked population**, not by deepening individual oscillator confinement. The three-body term creates a "winner-takes-all" effect: already-synchronized clusters exert stronger pull on outliers. This is invisible to OA, which assumes continuous density.

Evidence:
- K₃=-1: 154/200 locked → r≈0.59
- K₃=0: 169/200 locked → r≈0.74
- K₃=+1: 189/200 locked → r≈0.90

## Other Key Results

1. **Critical σ* ≈ 1.2**: K₃ effect peaks at σ≈1.2, then weakens. "Too much frequency disorder defeats higher-order coupling."

2. **Critical slowing enhanced by K₃>0**: τ_relax increases 4× (12→48.5s). The potential landscape becomes "flat-bottomed" near the saddle-node, trapping trajectories longer.

3. **K₄ four-body term**: 45-75% of K₃ effect depending on N. Truncation at three-body is questionable.

4. **Universality**: K₃ effects identical across Gaussian, Lorentzian, Uniform frequency distributions.

5. **No chimera** in all-to-all networks with higher-order coupling.

6. **No re-entrant sync** (MathAgent's initial prediction was a code bug).

7. **Z₂/Z₁² = 2/π for Gaussian** (MathAgent analytical + our numerics): OA manifold Z₂=Z₁² fails. Ratio ranges 0.37-0.97, only approaches 1 when r>0.96. This means effective K₃ coupling is reduced by factor 2/π ≈ 0.64.

8. **GPU large-N scaling (N=200-5000)**: K₃>0 enhancement persists to N=5000. r(K₃=0) slowly decreases toward OA prediction, but the K₃ enhancement gap WIDENS. No direction reversal.

9. **Non-monotonic hysteresis**: Area has minimum at K₃≈1.03, then rises to 0.64 at K₃=3. The dip corresponds to OA subcritical boundary K₃≈K₂.

10. **No anti-phase clusters**: Even at K₃=-5, frequency heterogeneity (σ=1) prevents clean bifurcation into two phase-locked clusters.

## Data Files (branch: sim/parameter-scan)

| File | Description |
|------|-------------|
| scan_sigma_K2_K3.json | Core 3D scan (6×16×16) |
| scan_K2_K3.json | 2D phase diagram (20×20) |
| benchmark_classical.json | Classical Kc verification |
| K3_asymmetry.json | Fine K₃ scan with timeseries |
| hysteresis.json | Forward/backward sweep |
| critical_slowing.json | Relaxation times |
| freq_distribution.json | Distribution comparison |
| K4_test.json | Four-body effect |
| K4_finite_size.json | K₄ vs N |
| finite_size_K3.json | K₃ direction vs N |
| locking_order.json | Which oscillators lock first |
| cluster_structure.json | Multi-cluster detection |
| chimera.json | Chimera search (negative) |
| reentrant_sync.json | Re-entrant search (negative) |
| pseudopotential_analysis.json | OA potential landscape |
| Kc_consistency.json | Kc seed variance |
| verify_oa_results.json | OA vs numerics comparison |
| quick_scan.json | Coarse 3D scan for fast plotting |
| finite_size_large_N.json | GPU N=200-5000 scaling |
| Z2_Z1_ratio.json | OA manifold Z₂/Z₁² test |
| hysteresis_fine.json | Fine K₃ hysteresis scan |
| anti_phase_test.json | Anti-phase cluster test |
