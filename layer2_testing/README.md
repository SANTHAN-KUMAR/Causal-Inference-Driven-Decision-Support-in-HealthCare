# Layer 2 (Adaptive Digital Twin) Test Suite

## Overview

This folder contains the comprehensive test suite for Layer 2 (Adaptive Digital Twin) of the AEGIS 3.0 system.

## Structure

```
layer2_testing/
├── src/
│   ├── config.py           # Configuration and test thresholds
│   ├── bergman_model.py    # Bergman Minimal Model (mechanistic component)
│   ├── ude_model.py        # Universal Differential Equation implementation
│   └── ac_ukf.py           # Adaptive Constrained UKF
├── tests/
│   ├── test_L2_ude.py      # Tests L2-1/2/7: UDE grey-box validation
│   ├── test_L2_acukf.py    # Tests L2-3/4: AC-UKF validation
│   └── run_all_tests.py    # Master test runner
├── data/                    # Generated test data
├── results/                 # Test results (JSON + paper-ready MD)
└── README.md               # This file
```

## Running Tests

```bash
cd layer2_testing/tests
python run_all_tests.py
```

## Test Summary

| Test | Description | Status |
|------|-------------|--------|
| L2-1/2/7 | UDE Grey-Box Integration | ✓ PASS |
| L2-3 | AC-UKF Covariance Adaptation | ✓ PASS |
| L2-4 | Constraint Projection | ✓ PASS |

## Claims Validated

1. **UDE combines mechanistic + neural** (C2-1) ✓
2. **Neural learns patient-specific deviations** (C2-2) ✓
3. **AC-UKF adapts covariance based on residuals** (C2-3) ✓
4. **Constraint projection maintains bounds** (C2-4) ✓
5. **Grey-box outperforms pure approaches** (C2-7) ✓
