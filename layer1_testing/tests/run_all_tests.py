"""
AEGIS 3.0 Layer 1 - Test Suite Runner

Runs all Layer 1 tests and generates a comprehensive report.
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import RESULTS_DIR


def run_all_layer1_tests() -> Dict:
    """
    Run all Layer 1 tests and aggregate results.
    
    Returns:
        Dictionary with all test results and summary
    """
    print("="*70)
    print("    AEGIS 3.0 LAYER 1 TEST SUITE")
    print("    Semantic Sensorium Validation")
    print("="*70)
    print(f"\nStarted at: {datetime.now().isoformat()}\n")
    
    all_results = {}
    tests_passed = 0
    tests_failed = 0
    
    # Test L1-1: Semantic Entropy Calibration
    print("\n" + "-"*70)
    print("Running Test L1-1: Semantic Entropy Calibration")
    print("-"*70)
    try:
        from test_L1_1_semantic_entropy import TestL1_1_SemanticEntropyCalibration
        test1 = TestL1_1_SemanticEntropyCalibration()
        result1 = test1.run_all_tests()
        test1.save_results()
        all_results['L1-1'] = result1
        if result1['overall_passed']:
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        print(f"ERROR in L1-1: {str(e)}")
        all_results['L1-1'] = {'error': str(e), 'overall_passed': False}
        tests_failed += 1
    
    # Test L1-2: HITL Trigger Calibration
    print("\n" + "-"*70)
    print("Running Test L1-2: HITL Trigger Calibration")
    print("-"*70)
    try:
        from test_L1_2_hitl_trigger import TestL1_2_HITLTrigger
        test2 = TestL1_2_HITLTrigger()
        result2 = test2.run_all_tests()
        test2.save_results()
        all_results['L1-2'] = result2
        if result2['overall_passed']:
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        print(f"ERROR in L1-2: {str(e)}")
        all_results['L1-2'] = {'error': str(e), 'overall_passed': False}
        tests_failed += 1
    
    # Tests L1-3 & L1-4: Proxy Classification
    print("\n" + "-"*70)
    print("Running Tests L1-3 & L1-4: Proxy Classification")
    print("-"*70)
    try:
        from test_L1_3_4_proxy_classification import TestL1_3_4_ProxyClassification
        test34 = TestL1_3_4_ProxyClassification()
        result34 = test34.run_all_tests()
        test34.save_results()
        all_results['L1-3/4'] = result34
        
        # Count L1-3 and L1-4 separately
        l1_3_passed = result34['individual_tests']['L1_3_treatment_proxy']['passed']
        l1_4_passed = result34['individual_tests']['L1_4_outcome_proxy']['passed']
        
        if l1_3_passed:
            tests_passed += 1
        else:
            tests_failed += 1
        if l1_4_passed:
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        print(f"ERROR in L1-3/4: {str(e)}")
        all_results['L1-3/4'] = {'error': str(e), 'overall_passed': False}
        tests_failed += 2
    
    # Test L1-5: Integration Test
    print("\n" + "-"*70)
    print("Running Test L1-5: Integration Test")
    print("-"*70)
    try:
        from test_L1_5_integration import TestL1_5_Integration
        test5 = TestL1_5_Integration()
        result5 = test5.run_all_tests()
        test5.save_results()
        all_results['L1-5'] = result5
        if result5['overall_passed']:
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        print(f"ERROR in L1-5: {str(e)}")
        all_results['L1-5'] = {'error': str(e), 'overall_passed': False}
        tests_failed += 1
    
    # Summary
    total_tests = tests_passed + tests_failed
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_tests,
        'tests_passed': tests_passed,
        'tests_failed': tests_failed,
        'pass_rate': tests_passed / total_tests if total_tests > 0 else 0,
        'all_passed': tests_failed == 0
    }
    
    # Print summary
    print("\n" + "="*70)
    print("    LAYER 1 TEST SUITE SUMMARY")
    print("="*70)
    print(f"\n  Total Tests:  {total_tests}")
    print(f"  Passed:       {tests_passed}")
    print(f"  Failed:       {tests_failed}")
    print(f"  Pass Rate:    {summary['pass_rate']*100:.1f}%")
    print(f"\n  {'★ ALL TESTS PASSED ★' if summary['all_passed'] else '✗ SOME TESTS FAILED'}")
    print("="*70)
    
    # Save aggregate results
    aggregate_results = {
        'summary': summary,
        'individual_results': all_results
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, 'layer1_all_results.json')
    with open(output_path, 'w') as f:
        json.dump(aggregate_results, f, indent=2, default=str)
    
    print(f"\nAll results saved to: {output_path}")
    
    return aggregate_results


if __name__ == "__main__":
    results = run_all_layer1_tests()
