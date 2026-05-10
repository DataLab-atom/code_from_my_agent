# Study on Synchronization Conditions in Higher-Order Kuramoto Models

## Model

For N oscillators with natural frequencies ωᵢ ~ N(0, σ²) and phases θᵢ:

```text
dθᵢ/dt = ωᵢ + (K₂/N) Σⱼ sin(θⱼ − θᵢ) + (K₃/N²) Σⱼ Σₖ sin(2θⱼ − θₖ − θᵢ)
```

## Files

- `kuramoto.py` - Classical Kuramoto model with higher-order extensions, RK4 integration, and order-parameter calculation
- `explore.py` - K₂ x K₃ parameter sweep, reproducing Muolo 2025's conclusion that weak higher-order coupling helps while strong higher-order coupling hinders synchronization

## References

- Kuramoto 1984 - Classical critical threshold Kc = 2/πg(0)
- Muolo 2025 - Weak K₃ helps, while strong K₃ hinders synchronization
- Zhang 2024 - Higher-order interactions shrink the basin of attraction
- Wang 2025 - Moderate K₃ improves stability
