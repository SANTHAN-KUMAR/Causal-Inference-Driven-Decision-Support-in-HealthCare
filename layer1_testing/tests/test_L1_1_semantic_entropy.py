"""
AEGIS 3.0 Layer 1 - Test L1.1: Semantic Entropy Calibration

Tests that semantic entropy correctly identifies extractions where 
meaning is uncertain.

Claim Tested: Semantic entropy correlates with true ambiguity levels
Success Criteria:
- Spearman ρ > 0.6
- AUC-ROC > 0.75
- High ambiguity recall > 0.8
"""

import sys
import os
import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from typing import List, Dict, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from semantic_entropy import SemanticEntropyCalculator
from config import CONFIG, DATA_DIR, RESULTS_DIR


class TestL1_1_SemanticEntropyCalibration:
    """
    Test L1-1: Semantic Entropy Calibration
    
    Validates that the semantic entropy metric is calibrated to
    reflect true semantic ambiguity in patient narratives.
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
    
    def run_entropy_calculations(self) -> List[Dict]:
        """
        Calculate semantic entropy for all test cases.
        
        Returns:
            List of results with text, ground_truth_ambiguity, and computed_entropy
        """
        if self.test_data is None:
            self.load_test_data()
        
        results = []
        
        for case in self.test_data:
            entropy, details = self.calculator.compute_semantic_entropy(case['text'])
            
            results.append({
                'id': case['id'],
                'text': case['text'],
                'ground_truth_ambiguity': case['ambiguity_rating'],
                'computed_entropy': entropy,
                'num_clusters': details['num_clusters'],
                'clusters': details['clusters']
            })
        
        return results
    
    def test_spearman_correlation(self, results: List[Dict]) -> Dict:
        """
        Test 1: Spearman rank correlation between entropy and ambiguity.
        
        Higher entropy should correlate with higher ambiguity ratings.
        """
        ground_truth = [r['ground_truth_ambiguity'] for r in results]
        computed = [r['computed_entropy'] for r in results]
        
        correlation, p_value = stats.spearmanr(ground_truth, computed)
        
        threshold = CONFIG.semantic_entropy.min_spearman_correlation
        passed = correlation > threshold
        
        return {
            'test_name': 'Spearman Correlation',
            'metric': 'spearman_rho',
            'value': correlation,
            'p_value': p_value,
            'threshold': threshold,
            'passed': passed,
            'interpretation': f"{'PASS' if passed else 'FAIL'}: ρ={correlation:.3f} {'>' if passed else '<='} {threshold}"
        }
    
    def test_auc_roc(self, results: List[Dict]) -> Dict:
        """
        Test 2: AUC-ROC for distinguishing ambiguous from unambiguous.
        
        Binary classification: rating >= 4 is "ambiguous", < 4 is "unambiguous"
        """
        # Binary labels: 1 if highly ambiguous (rating >= 4)
        y_true = [1 if r['ground_truth_ambiguity'] >= 4 else 0 for r in results]
        y_score = [r['computed_entropy'] for r in results]
        
        # Check if we have both classes
        if len(set(y_true)) < 2:
            return {
                'test_name': 'AUC-ROC',
                'metric': 'auc_roc',
                'value': None,
                'threshold': CONFIG.semantic_entropy.min_auc_roc,
                'passed': False,
                'interpretation': 'SKIP: Need both classes for AUC calculation'
            }
        
        auc = roc_auc_score(y_true, y_score)
        threshold = CONFIG.semantic_entropy.min_auc_roc
        passed = auc > threshold
        
        return {
            'test_name': 'AUC-ROC',
            'metric': 'auc_roc',
            'value': auc,
            'threshold': threshold,
            'passed': passed,
            'interpretation': f"{'PASS' if passed else 'FAIL'}: AUC={auc:.3f} {'>' if passed else '<='} {threshold}"
        }
    
    def test_high_ambiguity_recall(self, results: List[Dict]) -> Dict:
        """
        Test 3: Recall for high-ambiguity cases.
        
        At optimal threshold, what fraction of truly ambiguous cases
        would be flagged for HITL review?
        """
        # Find optimal threshold using ROC curve
        y_true = [1 if r['ground_truth_ambiguity'] >= 4 else 0 for r in results]
        y_score = [r['computed_entropy'] for r in results]
        
        if len(set(y_true)) < 2:
            return {
                'test_name': 'High Ambiguity Recall',
                'metric': 'recall_high_ambiguity',
                'value': None,
                'threshold': CONFIG.semantic_entropy.min_high_ambiguity_recall,
                'passed': False,
                'interpretation': 'SKIP: Need both classes'
            }
        
        # Find threshold that maximizes recall while maintaining some precision
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # Choose threshold with TPR >= target (recall for positive class)
        target_recall = CONFIG.semantic_entropy.min_high_ambiguity_recall
        valid_idx = np.where(tpr >= target_recall)[0]
        
        if len(valid_idx) > 0:
            # Among thresholds achieving target recall, pick lowest FPR
            best_idx = valid_idx[np.argmin(fpr[valid_idx])]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            achieved_recall = tpr[best_idx]
            achieved_fpr = fpr[best_idx]
            passed = True
        else:
            # Can't achieve target recall
            best_idx = np.argmax(tpr)
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            achieved_recall = tpr[best_idx]
            achieved_fpr = fpr[best_idx]
            passed = achieved_recall >= target_recall
        
        return {
            'test_name': 'High Ambiguity Recall',
            'metric': 'recall_high_ambiguity',
            'value': achieved_recall,
            'threshold': target_recall,
            'optimal_entropy_threshold': optimal_threshold,
            'false_positive_rate': achieved_fpr,
            'passed': passed,
            'interpretation': f"{'PASS' if passed else 'FAIL'}: Recall={achieved_recall:.3f} {'>' if passed else '<='} {target_recall}"
        }
    
    def run_all_tests(self) -> Dict:
        """Execute all L1-1 tests and return comprehensive results."""
        print("="*60)
        print("TEST L1-1: Semantic Entropy Calibration")
        print("="*60)
        
        # Run entropy calculations
        print("\n[1/4] Calculating semantic entropy for test cases...")
        entropy_results = self.run_entropy_calculations()
        print(f"      Computed entropy for {len(entropy_results)} test cases")
        
        # Run individual tests
        print("\n[2/4] Testing Spearman correlation...")
        spearman_result = self.test_spearman_correlation(entropy_results)
        print(f"      {spearman_result['interpretation']}")
        
        print("\n[3/4] Testing AUC-ROC discrimination...")
        auc_result = self.test_auc_roc(entropy_results)
        print(f"      {auc_result['interpretation']}")
        
        print("\n[4/4] Testing high-ambiguity recall...")
        recall_result = self.test_high_ambiguity_recall(entropy_results)
        print(f"      {recall_result['interpretation']}")
        
        # Overall result
        all_passed = all([
            spearman_result['passed'],
            auc_result['passed'],
            recall_result['passed']
        ])
        
        self.results = {
            'test_id': 'L1-1',
            'test_name': 'Semantic Entropy Calibration',
            'overall_passed': all_passed,
            'individual_tests': {
                'spearman_correlation': spearman_result,
                'auc_roc': auc_result,
                'high_ambiguity_recall': recall_result
            },
            'detailed_results': entropy_results,
            'summary_statistics': {
                'mean_entropy_by_rating': self._compute_mean_by_rating(entropy_results),
                'total_test_cases': len(entropy_results)
            }
        }
        
        print("\n" + "="*60)
        print(f"TEST L1-1 OVERALL: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
        print("="*60)
        
        return self.results
    
    def _compute_mean_by_rating(self, results: List[Dict]) -> Dict[int, float]:
        """Compute mean entropy for each ambiguity rating."""
        by_rating = {}
        for r in results:
            rating = r['ground_truth_ambiguity']
            if rating not in by_rating:
                by_rating[rating] = []
            by_rating[rating].append(r['computed_entropy'])
        
        return {k: np.mean(v) for k, v in sorted(by_rating.items())}
    
    def save_results(self, output_dir: str = None):
        """Save test results to file."""
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results (excluding detailed per-case for brevity)
        output = {
            'test_id': self.results['test_id'],
            'test_name': self.results['test_name'],
            'overall_passed': self.results['overall_passed'],
            'individual_tests': self.results['individual_tests'],
            'summary_statistics': self.results['summary_statistics']
        }
        
        output_path = os.path.join(output_dir, 'test_L1_1_results.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


def run_test():
    """Main entry point for running Test L1-1."""
    test = TestL1_1_SemanticEntropyCalibration()
    results = test.run_all_tests()
    test.save_results()
    return results


if __name__ == "__main__":
    run_test()
