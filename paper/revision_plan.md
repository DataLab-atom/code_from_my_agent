# Paper Revision Plan: Distribution-Dependent Phase Topology

## Core Insight (NEW)

The frequency distribution's **tail behavior** qualitatively changes the phase diagram topology:
- Lorentzian (heavy tails): explosive boundary at K₃=K₂, re-entrant requires K₃>>K₂
- Gaussian (light tails): explosive-like jumps appear at K₃<<K₂, re-entrant at K₃/K₂≈1-3.5

This means: **the same coupling parameters produce different synchronization behavior
depending on population heterogeneity** — a fundamental result with direct experimental implications.

## Evidence

1. OA (Lorentzian): max jump at K₃=0 is 0 (continuous onset). Explosive at K₃=K₂.
2. Gaussian exact: max jump at K₃=0 is 0.26, at K₃=1.0 is 0.84 (already explosive-like).
3. Re-entrant: confirmed for Gaussian at K₂=2, K₃≈5.4; not reachable in OA at same params.

## Revised Paper Structure

### New Title Candidate
"Tail Matters: How Frequency Distribution Shape Controls Higher-Order Synchronization Phase Topology"

### Revised Contributions
1. OA reduction → qualitative framework (unchanged)
2. **NEW**: Gaussian self-consistent equation reveals distribution-dependent phase topology
3. **STRENGTHENED**: Re-entrant synchronization confirmed in Gaussian (not just predicted)
4. Reconciliation of contradictory studies (unchanged, but now deeper: studies used different distributions too!)
5. Time-delay mapping (unchanged)

### Key Figure (NEW: should be Fig 1)
Side-by-side phase diagrams: OA(Lorentzian) vs Gaussian exact
- Same K₂×K₃ axes
- Show how explosive boundary shifts
- Show re-entrant region appearing in Gaussian but not in OA
- This ONE figure tells the entire story

### Sections to Rewrite
- Abstract: add distribution-dependence as main contribution
- Intro: frame as "distribution shape matters, not just coupling"
- Section 3: add Gaussian self-consistent analysis after OA
- Section 4: re-entrant experiment should show POSITIVE results now
- Discussion: elevate distribution-dependence to main insight
- Conclusion: reorder contributions

## Action Items
- [ ] Wait for KuramotoThinker feedback on direction
- [ ] Wait for GPU extended K₃ scan to validate re-entrant numerically
- [ ] Wait for PlotAnalyst re-entrant figure
- [ ] Rewrite abstract and intro
- [ ] Add Gaussian self-consistent section
- [ ] Update experiments with new results
