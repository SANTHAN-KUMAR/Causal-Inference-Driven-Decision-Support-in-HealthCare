# Layer 2 Testing: Implementation Notes & Tweaks

## Overview

This document details the implementation choices made during Layer 2 test development.

---

## Test L2-1/2/7: UDE Grey-Box Integration

### Paper's Proposed Method
> dx/dt = f_mech(x,u;θ_fixed) + f_NN(x,u;θ_learned)

### Implementation
- **Mechanistic component**: Bergman Minimal Model (glucose-insulin dynamics)
- **Neural component**: 2-layer MLP with tanh activation
- **Integration**: RK4 numerical integration
- **Training**: Finite-difference gradient descent

### Tweaks Made

| Tweak | Reason | Validity |
|-------|--------|----------|
| Input normalization | Prevent gradient explosion | Standard practice |
| Output scaling (0.1x) | Residuals should be small | Matches physical expectations |
| Small weight init | Stability | Standard practice |
| Gradient clipping | Prevent NaN | Standard practice |

**Note**: For production, replace with PyTorch/JAX with proper autodiff.

---

## Test L2-3/4: AC-UKF

### Paper's Proposed Method
> Q_{k+1} = Q_k + α * K * (ε_k * ε_k' - S_k) * K'

### Implementation
Faithful to paper. No major tweaks needed.

---

## Summary

| Test | Tweaked? | Core Method Preserved? |
|------|----------|------------------------|
| L2-1/2/7 | Yes (stability) | Yes (UDE structure) |
| L2-3/4 | No | Yes |

The core claims are validated. Tweaks were for numerical stability only.
