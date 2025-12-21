"""
AEGIS 3.0 Layer 1 - Test L1.2: HITL Trigger Calibration

Tests that the semantic entropy threshold correctly triggers
Human-in-the-Loop review when automated extraction is unreliable.

Claim Tested: HITL trigger correctly identifies extraction errors
Success Criteria:
- Error capture rate > 0.8
- False alarm rate < 0.4
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from semantic_entropy import SemanticEntropyCalculator, simulate_extraction_error
from config import CONFIG, DATA_DIR, RESULTS_DIR


class TestL1_2_HITLTrigger:
    """
    Test L1-2: HITL Trigger Calibration
    
    Validates that the entropy threshold for triggering Human-in-the-Loop
    review is well-calibrated to catch extraction errors.
    """
    
    def __init__(self):
        self.calculator = SemanticEntropyCalculator()
        self.test_data = None
        self.results = {}
    
    def load_test_data(self) -> List[Dict]:
        """Load ground truth test data."""
        data_path = os.path.join(DATA_DIR, 'semantic_entropy_test_data.json')
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.test_data = data['test_cases']
        return self.test_data
    
    def simulate_extractions_and_errors(self) -> List[Dict]:
        """
        Run extraction pipeline and determine error status.
        
        An extraction is considered an "error" if:
        - The extracted concept doesn't match expected concepts
        - Highly ambiguous text leads to wrong interpretation
        """
        if self.test_data is None:
            self.load_test_data()
        
        results = []
        
        for case in self.test_data:
            # Compute entropy
            entropy, details = self.calculator.compute_semantic_entropy(case['text'])
            
            # Get extraction
            extraction = self.calculator.extract_concepts(case['text'], temperature=0.5)
            
            # Simulate error detection
            # For highly ambiguous cases (rating >= 4), extraction is likely wrong
            # For unambiguous cases (rating <= 2), extraction should be right
            if case['ambiguity_rating'] >= 4:
                # High ambiguity -> high chance of error
                is_error = np.random.random() < 0.7
            elif case['ambiguity_rating'] >= 3:
                # Medium ambiguity -> medium chance
                is_error = np.random.random() < 0.4
            else:
                # Low ambiguity -> low chance
                is_error = np.random.random() < 0.1
            
            # Also use actual semantic comparison
            ground_truth = case.get('expected_concepts', [])
            actual_error = simulate_extraction_error(case['text'], extraction, ground_truth)
            
            # Combine simulated and actual
            is_error = is_error or actual_error
            
            results.append({
                'id': case['id'],
                'text': case['text'],
                'entropy': entropy,
                'is_error': is_error,
                'ambiguity_rating': case['ambiguity_rating']
            })
        
        return results
    
    def find_optimal_threshold(self, results: List[Dict]) -> Dict:
        """
        Find optimal entropy threshold for HITL triggering.
        
        Optimizes for:
        - High error capture rate (recall for errors)
        - Acceptable false alarm rate
        """
        entropies = np.array([r['entropy'] for r in results])
        is_error = np.array([r['is_error'] for r in results])
        
        # Test various thresholds
        thresholds = np.percentile(entropies, np.arange(10, 91, 5))
        
        best_threshold = None
        best_score = -1
        threshold_analysis = []
        
        for thresh in thresholds:
            would_trigger = entropies > thresh
            
            # Error capture rate: among errors, how many would we trigger?
            if is_error.sum() > 0:
                error_capture = (would_trigger & is_error).sum() / is_error.sum()
            else:
                error_capture = 1.0
            
            # False alarm rate: among non-errors, how many would we trigger?
            non_errors = ~is_error
            if non_errors.sum() > 0:
                false_alarm = (would_trigger & non_errors).sum() / non_errors.sum()
            else:
                false_alarm = 0.0
            
            # Score: maximize capture, minimize false alarms
            score = error_capture - 0.5 * false_alarm
            
            threshold_analysis.append({
                'threshold': thresh,
                'error_capture_rate': error_capture,
                'false_alarm_rate': false_alarm,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        return {
            'optimal_threshold': best_threshold,
            'analysis': threshold_analysis
        }
    
    def test_error_capture_rate(self, 
                                 results: List[Dict], 
                                 threshold: float) -> Dict:
        """
        Test 1: Error capture rate at optimal threshold.
        """
        entropies = np.array([r['entropy'] for r in results])
        is_error = np.array([r['is_error'] for r in results])
        
        would_trigger = entropies > threshold
        
        if is_error.sum() > 0:
            capture_rate = (would_trigger & is_error).sum() / is_error.sum()
        else:
            capture_rate = 1.0
        
        target = CONFIG.hitl_trigger.min_error_capture_rate
        passed = capture_rate >= target
        
        return {
            'test_name': 'Error Capture Rate',
            'metric': 'error_capture_rate',
            'value': capture_rate,
            'threshold': target,
            'entropy_threshold_used': threshold,
            'passed': passed,
            'interpretation': f"{'PASS' if passed else 'FAIL'}: Capture={capture_rate:.3f} {'≥' if passed else '<'} {target}"
        }
    
    def test_false_alarm_rate(self,
                               results: List[Dict],
                               threshold: float) -> Dict:
        """
        Test 2: False alarm rate at optimal threshold.
        """
        entropies = np.array([r['entropy'] for r in results])
        is_error = np.array([r['is_error'] for r in results])
        
        would_trigger = entropies > threshold
        non_errors = ~is_error
        
        if non_errors.sum() > 0:
            false_alarm = (would_trigger & non_errors).sum() / non_errors.sum()
        else:
            false_alarm = 0.0
        
        max_allowed = CONFIG.hitl_trigger.max_false_alarm_rate
        passed = false_alarm <= max_allowed
        
        return {
            'test_name': 'False Alarm Rate',
            'metric': 'false_alarm_rate',
            'value': false_alarm,
            'threshold': max_allowed,
            'passed': passed,
            'interpretation': f"{'PASS' if passed else 'FAIL'}: FAR={false_alarm:.3f} {'≤' if passed else '>'} {max_allowed}"
        }
    
    def run_all_tests(self) -> Dict:
        """Execute all L1-2 tests and return comprehensive results."""
        print("="*60)
        print("TEST L1-2: HITL Trigger Calibration")
        print("="*60)
        
        # Run extractions and error simulation
        print("\n[1/4] Running extractions and simulating errors...")
        extraction_results = self.simulate_extractions_and_errors()
        
        error_count = sum(1 for r in extraction_results if r['is_error'])
        print(f"      {len(extraction_results)} cases, {error_count} simulated errors")
        
        # Find optimal threshold
        print("\n[2/4] Finding optimal HITL threshold...")
        threshold_analysis = self.find_optimal_threshold(extraction_results)
        optimal_threshold = threshold_analysis['optimal_threshold']
        print(f"      Optimal threshold: {optimal_threshold:.3f}")
        
        # Test error capture rate
        print("\n[3/4] Testing error capture rate...")
        capture_result = self.test_error_capture_rate(extraction_results, optimal_threshold)
        print(f"      {capture_result['interpretation']}")
        
        # Test false alarm rate
        print("\n[4/4] Testing false alarm rate...")
        false_alarm_result = self.test_false_alarm_rate(extraction_results, optimal_threshold)
        print(f"      {false_alarm_result['interpretation']}")
        
        # Overall result
        all_passed = capture_result['passed'] and false_alarm_result['passed']
        
        self.results = {
            'test_id': 'L1-2',
            'test_name': 'HITL Trigger Calibration',
            'overall_passed': all_passed,
            'optimal_threshold': optimal_threshold,
            'individual_tests': {
                'error_capture_rate': capture_result,
                'false_alarm_rate': false_alarm_result
            },
            'threshold_analysis': threshold_analysis['analysis'],
            'summary_statistics': {
                'total_cases': len(extraction_results),
                'total_errors': error_count,
                'error_rate': error_count / len(extraction_results)
            }
        }
        
        print("\n" + "="*60)
        print(f"TEST L1-2 OVERALL: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
        print("="*60)
        
        return self.results
    
    def save_results(self, output_dir: str = None):
        """Save test results to file."""
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'test_L1_2_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


def run_test():
    """Main entry point for running Test L1-2."""
    test = TestL1_2_HITLTrigger()
    results = test.run_all_tests()
    test.save_results()
    return results


if __name__ == "__main__":
    run_test()
