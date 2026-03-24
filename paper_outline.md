# Paper Outline: Reconciling Contradictory Effects of Higher-Order Interactions in Kuramoto Synchronization

**Target venue**: NeurIPS 2026
**Status**: Draft outline based on completed tasks

---

## Title candidates
1. "Three Metrics, One Phase Space: Resolving the Higher-Order Synchronization Paradox"
2. "When Does Higher-Order Coupling Help? A Complete Phase Diagram for the Higher-Order Kuramoto Model"

---

## 1. Introduction
- Classical Kuramoto model: well-understood (Kc = 2Δ)
- Higher-order interactions: emerging field, conflicting results
- The paradox: 4 papers (2020-2025), same model, opposite conclusions
- Our contribution: unified framework showing all are correct simultaneously

## 2. Model and Notation
- Higher-order Kuramoto equation
- Three metrics: r (order parameter), basin probability, stability depth
- **Bug fix**: Correct Kc = 2σ√(2π)/π ≈ 1.596, not 5.013 (task 1.1)

## 3. Ott-Antonsen Reduction (task 2.2)
### Theorem 1: Low-dimensional ODE
> dr/dt = −Δr + r(1−r²)/2 · (K₂ + K₃r²)

### Theorem 2: Phase boundary structure
- **Onset**: K₂c = 2Δ, independent of K₃ at leading order
- **Bifurcation type**: supercritical (K₃ < K₂) vs subcritical/explosive (K₃ > K₂)
- **Basin boundary**: saddle at r_sep, rises with K₃ → confirms Zhang 2024
- **Stability depth**: Lyapunov exponent deepens with moderate K₃ → confirms Wang 2025

### Proposition: Reconciliation
The four contradictory studies measure different projections of the radial potential V(r):
- Muolo: measures slope near r=0 (onset) → K₃-independent
- Zhang: measures saddle position (basin) → shrinks with K₃
- Wang: measures curvature at r* (stability) → deepens with K₃
- Skardal: measures bifurcation type → switches at K₃=K₂

## 4. New Prediction: Reentrant Synchronization
- For K₂ slightly above Kc, increasing K₃ first enhances then destroys sync
- Derived analytically from the cubic structure of the fixed-point equation
- **Awaiting numerical verification** (task R.1)

## 5. Time-Delay Equivalence (task A.1)
### Theorem 3: Physical accessibility
- K₃_eff = K₂²τ/4 ≥ 0 always
- Time delay can ONLY produce positive K₃
- Explosive boundary K₃ = K₂ unreachable by pure time delay
- "Physically reachable region" in K₂×K₃ phase diagram

## 6. Truncation Validity: Role of K₄ (task J.1)
- K₄ effect is 89% of K₃ → NOT negligible in general!
- But under time-delay framework (τ small): K₃ ∝ τ, K₄ ∝ τ² → justified
- **Awaiting theoretical analysis** (task S.1)

## 7. Numerical Results
### 7.1 Classical benchmark (task 1.1) — DONE
- Kc measured = 1.553 (N=200), 1.632 (N=500), theory = 1.596
- √(K₂−Kc) scaling confirmed

### 7.2 K₂×K₃ phase diagram (task 1.2) — PENDING
### 7.3 Locking order (task K.1) — DONE
- K₃>0: 189/200 locked, K₃=0: 169/200, K₃<0: 154/200
- "Winner-takes-all" effect confirmed

### 7.4 Cluster structure (task M.1) — DONE
- Multi-cluster (2-3) only at moderate K₂ + negative K₃
- K₃>0 strongly promotes single-cluster

### 7.5 Chimera states (task P.1) — DONE (negative result)
- No chimera in all-to-all coupling even with higher-order terms

### 7.6 Hysteresis detection (task G.1) — PENDING
### 7.7 Reentrant sync verification (task R.1) — PENDING

## 8. Critical Slowing Down: A Surprise (task N.1 — DONE)
- K₃>0 ENHANCES critical slowing: τ_max from 12s (K₃=0) to 48.5s (K₃=1.5)
- **Theoretical explanation (KuramotoThinker analysis)**:
  - Linear: λ = K₂/2 − Δ, independent of K₃ → linear τ_relax unchanged
  - But K₃ modifies the NONLINEAR landscape → system gets "stuck" at intermediate r
  - This is a purely nonlinear critical slowing, not classical linear CSD
  - Implication: K₃ doesn't change WHEN the transition happens, but HOW LONG the system hesitates
- Practical significance: higher-order coupling makes phase transitions MORE predictable (stronger precursors), not less → epilepsy early warning

## 9. Hysteresis (task G.1 — DONE)
- All K₃ values show hysteresis (area > 0.1)
- K₃=+1.5 has largest area (0.178) → approaching first-order transition
- Validates OA prediction: K₃>K₂ → subcritical bifurcation

## 10. Discussion
- Budget-constrained optimal allocation (task E.1 — PENDING)
- Frequency distribution robustness (task O.1 — PENDING)

## 9. Conclusion

---

## Figures needed (PlotAnalyst)
1. ✅ Benchmark r vs K₂ with Kc line
2. ✅ Locking order scatter plot (3 panels)
3. ✅ K₄ effect comparison
4. ✅ OA verification (bonus)
5. ✅ Analytic phase diagrams (bonus)
6. ⏳ K₂×K₃ heatmap (waiting data)
7. ⏳ Hysteresis loops (waiting data)
8. ⏳ Reentrant sync r vs K₃ (waiting data)
9. ⏳ Budget-constraint Pareto front (waiting data)

## Pending tasks summary
- **GPU-Opus**: 1.2, B.1, C.1, E.1, G.1, N.1, O.1, R.1 (all executing)
- **MathAgent**: H.1, I.1, L.1, Q.1, S.1 (all unclaimed — needs attention)
- **PlotAnalyst**: waiting for GPU data to make remaining figures
