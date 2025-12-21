# AEGIS 3.0 Layer 1 Testing

## Overview

This module contains the complete test suite for validating Layer 1 (Semantic Sensorium) of the AEGIS 3.0 architecture.

## Structure

```
layer1_testing/
├── data/                          # Test data files
│   ├── semantic_entropy_test_data.json   # 50 labeled snippets for entropy testing
│   ├── synthetic_causal_data.json        # Generated synthetic data w/ known causal structure
│   └── integration_test_data.json        # Larger dataset for integration testing
│
├── src/                           # Source modules
│   ├── config.py                  # Configuration and thresholds
│   ├── semantic_entropy.py        # Semantic entropy calculator
│   ├── proxy_classifier.py        # Z_t/W_t proxy classification
│   └── synthetic_data_generator.py # Generates data with known causal structure
│
├── tests/                         # Test implementations
│   ├── test_L1_1_semantic_entropy.py     # Entropy calibration
│   ├── test_L1_2_hitl_trigger.py         # HITL trigger accuracy
│   ├── test_L1_3_4_proxy_classification.py # Proxy classification
│   ├── test_L1_5_integration.py          # Integration with causal inference
│   └── run_all_tests.py                  # Master test runner
│
└── results/                       # Test output files
```

## Running Tests

```bash
# Run all tests
cd layer1_testing/tests
python run_all_tests.py

# Run individual tests
python test_L1_1_semantic_entropy.py
python test_L1_2_hitl_trigger.py
python test_L1_3_4_proxy_classification.py
python test_L1_5_integration.py
```

## Test Summary

| Test | Claim Validated | Success Criteria |
|------|-----------------|------------------|
| L1-1 | Semantic entropy reflects ambiguity | Spearman ρ > 0.6, AUC > 0.75 |
| L1-2 | HITL threshold catches errors | Error capture > 80%, FAR < 40% |
| L1-3 | Treatment proxy (Z_t) classification | Precision > 0.7, Recall > 0.6 |
| L1-4 | Outcome proxy (W_t) classification | Precision > 0.7, Recall > 0.6 |
| L1-5 | Proxies improve causal inference | Bias reduction > 30% |

## Dependencies

- numpy
- scipy  
- scikit-learn

Install with: `pip install numpy scipy scikit-learn`
