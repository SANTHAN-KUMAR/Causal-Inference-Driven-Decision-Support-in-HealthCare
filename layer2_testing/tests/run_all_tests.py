"""
AEGIS 3.0 Layer 2 - Master Test Runner

Runs all Layer 2 tests and aggregates results.
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from config import RESULTS_DIR


def run_all_tests():
    """Run all Layer 2 tests."""
    print("=" * 70)
    print("    AEGIS 3.0 LAYER 2 TEST SUITE")
    print("    Adaptive Digital Twin Validation")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    
    all_results = {
        'summary': {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'pass_rate': 0.0
        },
        'individual_results': {}
    }
    
    # Test L2-1/2/7: UDE Model
    print("\n" + "-" * 70)
    print("Running Tests L2-1/2/7: UDE Model and Grey-Box Integration")
    print("-" * 70)
    try:
        from test_L2_ude import run_test as run_ude_test
        ude_result = run_ude_test()
        all_results['individual_results']['L2-1/2/7'] = ude_result
        if ude_result.get('overall_passed'):
            all_results['summary']['tests_passed'] += 1
        else:
            all_results['summary']['tests_failed'] += 1
        all_results['summary']['total_tests'] += 1
    except Exception as e:
        print(f"ERROR in L2-1/2/7: {e}")
        all_results['individual_results']['L2-1/2/7'] = {'error': str(e)}
        all_results['summary']['tests_failed'] += 1
        all_results['summary']['total_tests'] += 1
    
    # Test L2-3/4: AC-UKF
    print("\n" + "-" * 70)
    print("Running Tests L2-3/4: AC-UKF Adaptation and Constraints")
    print("-" * 70)
    try:
        from test_L2_acukf import run_test as run_acukf_test
        acukf_result = run_acukf_test()
        all_results['individual_results']['L2-3/4'] = acukf_result
        if acukf_result.get('overall_passed'):
            all_results['summary']['tests_passed'] += 1
        else:
            all_results['summary']['tests_failed'] += 1
        all_results['summary']['total_tests'] += 1
    except Exception as e:
        print(f"ERROR in L2-3/4: {e}")
        all_results['individual_results']['L2-3/4'] = {'error': str(e)}
        all_results['summary']['tests_failed'] += 1
        all_results['summary']['total_tests'] += 1
    
    # Calculate pass rate
    total = all_results['summary']['total_tests']
    passed = all_results['summary']['tests_passed']
    all_results['summary']['pass_rate'] = passed / total if total > 0 else 0.0
    all_results['summary']['all_passed'] = passed == total
    
    # Summary
    print("\n" + "=" * 70)
    print("    LAYER 2 TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"\n  Total Tests:  {total}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {all_results['summary']['tests_failed']}")
    print(f"  Pass Rate:    {all_results['summary']['pass_rate']*100:.1f}%")
    print()
    if all_results['summary']['all_passed']:
        print("  ★ ALL TESTS PASSED ★")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("=" * 70)
    
    # Save aggregate results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, 'layer2_all_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    run_all_tests()
