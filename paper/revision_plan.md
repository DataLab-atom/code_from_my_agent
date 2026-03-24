# Paper Revision Plan: Distribution-Dependent Phase Topology
## Updated 2026-03-25 after OA closure ratio discovery

## Core Insight

The frequency distribution controls synchronization through **two channels**:
1. **Local** (g(0), g''(0)): onset threshold Kc and curvature
2. **Global** (tail decay integral): OA closure ratio c[g] = Z₂/Z₁²

This means the same (K₂,K₃) parameters produce **quantitatively different**
phase diagrams depending on distribution shape, not just qualitatively different.

## Key New Results (this session)

### 1. OA Closure Ratio c[g]
- Z₂ ≠ Z₁² for non-Lorentzian distributions
- c[g] = (2/(π²g₀²)) ∫(g₀-g(ω))/ω² dω (general closed form)
- Gaussian: c = 2/π (analytically proved)
- Lorentzian: c = 1 (analytically proved)
- Student-t(ν): c(ν) = (ν/π)[Γ(ν/2)/Γ((ν+1)/2)]² (closed form)
- Code: z2_z1_ratio_v2.py, general_c_formula.py, student_t_c_formula.py

### 2. Corrected Explosive Formula
- K₃^exp = Kc³|g''(0)|/(8c[g]g(0))
- Gaussian: K₃^exp/Kc = 1/2 (was 1/π ≈ 0.318, 57% error)
- Verified: corrected equation shows explosive at K₃=0.80 ✓
- Code: verify_explosive_boundary.py

### 3. Phantom Subcritical Solutions
- Z₂=r² approximation creates synchronized states that don't exist
- At K₂=1.0, K₃=2.0: approximate says r=0.875, corrected says r=0
- Saddle-node boundary: corrected bistable region is smaller by 0.1-0.4 in K₂
- Code: debug_disagreement.py, saddle_node_boundary.py

### 4. Time-Delay Distribution Dependence
- Delay-induced explosive needs K₂ > 2K₃^exp
- Lorentzian: K₂ > 2Kc (rare — needs double onset)
- Gaussian: K₂ > Kc (generic — any supercritical coupling)
- Updated in main.tex Section 5.2

### 5. DEBUNKED: Re-entrant Synchronization
- Was caused by code bug (r_max=0.99 too small)
- Corrected: r* monotonically increases with K₃, never drops to 0
- IVT argument: indestructibility holds for all distributions

## What's Already Updated in main.tex
- [x] Abstract: corrected formula with c[g]
- [x] Theorem 1: added c[g] with closed-form integral + proof
- [x] Remark (OA caveat): explains Z₂/Z₁²=2/π for Gaussian
- [x] Time-delay section: distribution-dependent explosive condition
- [x] Limitations: added c[g] finite-r caveat
- [x] Conclusion: updated formula reference

## What Still Needs Updating
- [ ] Add Proposition: c[g] general formula with Student-t example
- [ ] Add figure: corrected vs approximate phase diagram (side by side)
- [ ] Add figure: c(r) curve showing 2/π → 1 transition
- [ ] Experiment section: add corrected predictions for GPU validation
- [ ] Discussion: emphasize phantom solutions and smaller bistable region

## Awaiting Team Input
- [ ] KuramotoThinker: narrative direction, theorem ordering
- [ ] GPU-Claude: numerical validation of K₂=1.0,K₃=2.0 (r≈0 or r≈0.87?)
- [ ] PlotAnalyst: corrected phase diagram visualization

## Summary of Complete Synchronization Conditions

1. **Onset**: K₂ > Kc = 2/(πg(0)) — universal, K₃-independent
2. **Explosive threshold**: K₃ > Kc³|g''(0)|/(8c[g]g(0)) — distribution-dependent
3. **Indestructibility**: K₂>Kc ⟹ ∃ r*>0 for all K₃ — universal
4. **Self-organized criticality**: K₃→-∞ ⟹ K_eff(r*)→Kc — universal
5. **Bistable region**: saddle-node boundary at K₂_min(K₃), corrected smaller than approximate
6. **Time delay**: Gaussian generic explosive, Lorentzian rare explosive
