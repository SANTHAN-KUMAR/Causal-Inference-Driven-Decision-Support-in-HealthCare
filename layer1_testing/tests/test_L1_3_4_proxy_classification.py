"""
AEGIS 3.0 Layer 1 - Tests L1.3 & L1.4: Proxy Classification

Tests that the CausalProxyClassifier correctly identifies:
- Treatment-confounder proxies (Z_t) per Definition 5.1
- Outcome-confounder proxies (W_t) per Definition 5.2

Uses synthetic data with KNOWN causal structure for rigorous validation.

Success Criteria:
- Z Precision > 0.7, Z Recall > 0.6
- W Precision > 0.7, W Recall > 0.6
"""

import sys
import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from proxy_classifier import CausalProxyClassifier, ExtractedConcept, ProxyClassification
from synthetic_data_generator import SyntheticDataGenerator, SyntheticDataset
from config import CONFIG, DATA_DIR, RESULTS_DIR


class TestL1_3_4_ProxyClassification:
    """
    Tests L1-3 and L1-4: Treatment and Outcome Proxy Classification
    
    Uses synthetic data where Z_true_role and W_true_role are KNOWN,
    allowing rigorous validation of the classifier.
    """
    
    def __init__(self):
        self.classifier = CausalProxyClassifier()
        self.synthetic_data = None
        self.results = {}
    
    def generate_or_load_synthetic_data(self, regenerate: bool = False) -> SyntheticDataset:
        """
        Generate or load synthetic test data.
        
        Args:
            regenerate: If True, regenerate data even if file exists
        """
        data_path = os.path.join(DATA_DIR, 'synthetic_causal_data.json')
        
        if os.path.exists(data_path) and not regenerate:
            self.synthetic_data = SyntheticDataset.load(data_path)
        else:
            generator = SyntheticDataGenerator(seed=CONFIG.random_seed)
            self.synthetic_data = generator.generate_dataset(
                CONFIG.proxy_classification.num_synthetic_patient_days
            )
            os.makedirs(DATA_DIR, exist_ok=True)
            self.synthetic_data.save(data_path)
        
        return self.synthetic_data
    
    def classify_all_proxies(self) -> List[Dict]:
        """
        Run proxy classification on all synthetic data.
        
        Returns:
            List of classification results with ground truth
        """
        if self.synthetic_data is None:
            self.generate_or_load_synthetic_data()
        
        results = []
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        for patient_day in self.synthetic_data.patient_days:
            day_offset = timedelta(days=patient_day.day_id)
            
            # Create treatment time
            treatment_time = base_time + day_offset + timedelta(hours=patient_day.treatment_hour)
            treatment_times = [treatment_time]
            
            # Create Z concept (treatment proxy candidate)
            z_time = base_time + day_offset + timedelta(hours=patient_day.Z_hour)
            z_concept = ExtractedConcept(
                text=patient_day.Z_text,
                concept_id='stress_related',
                timestamp=z_time,
                confidence=0.9
            )
            
            # Classify Z
            z_classification = self.classifier.classify_proxy_role(z_concept, treatment_times)
            
            results.append({
                'day_id': patient_day.day_id,
                'type': 'Z',
                'text': patient_day.Z_text,
                'predicted_role': z_classification.role,
                'predicted_confidence': z_classification.confidence,
                'true_role': patient_day.Z_true_role,
                'classification_reason': z_classification.reason
            })
            
            # Create W concept (outcome proxy candidate)
            w_time = base_time + day_offset + timedelta(hours=patient_day.W_hour)
            w_concept = ExtractedConcept(
                text=patient_day.W_text,
                concept_id='symptom_related',
                timestamp=w_time,
                confidence=0.9
            )
            
            # Classify W
            w_classification = self.classifier.classify_proxy_role(w_concept, treatment_times)
            
            results.append({
                'day_id': patient_day.day_id,
                'type': 'W',
                'text': patient_day.W_text,
                'predicted_role': w_classification.role,
                'predicted_confidence': w_classification.confidence,
                'true_role': patient_day.W_true_role,
                'classification_reason': w_classification.reason
            })
        
        return results
    
    def compute_classification_metrics(self, 
                                         results: List[Dict],
                                         proxy_type: str) -> Dict:
        """
        Compute precision and recall for a specific proxy type.
        
        Args:
            results: Classification results
            proxy_type: 'treatment_proxy' or 'outcome_proxy'
        """
        # Filter to relevant results
        relevant = [r for r in results if (
            (proxy_type == 'treatment_proxy' and r['type'] == 'Z') or
            (proxy_type == 'outcome_proxy' and r['type'] == 'W')
        )]
        
        # True positives: predicted as proxy AND truly is proxy
        true_positives = sum(1 for r in relevant 
                            if r['predicted_role'] == proxy_type 
                            and r['true_role'] == proxy_type)
        
        # False positives: predicted as proxy BUT not truly
        false_positives = sum(1 for r in relevant
                             if r['predicted_role'] == proxy_type
                             and r['true_role'] != proxy_type)
        
        # False negatives: not predicted as proxy BUT truly is
        false_negatives = sum(1 for r in relevant
                             if r['predicted_role'] != proxy_type
                             and r['true_role'] == proxy_type)
        
        # True negatives
        true_negatives = sum(1 for r in relevant
                            if r['predicted_role'] != proxy_type
                            and r['true_role'] != proxy_type)
        
        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'proxy_type': proxy_type,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_predictions_as_proxy': true_positives + false_positives,
            'total_actual_proxies': true_positives + false_negatives
        }
    
    def test_treatment_proxy_classification(self, results: List[Dict]) -> Dict:
        """
        Test L1-3: Treatment-confounder proxy (Z_t) classification.
        """
        metrics = self.compute_classification_metrics(results, 'treatment_proxy')
        
        precision_thresh = CONFIG.proxy_classification.min_precision
        recall_thresh = CONFIG.proxy_classification.min_recall
        
        precision_passed = metrics['precision'] >= precision_thresh
        recall_passed = metrics['recall'] >= recall_thresh
        passed = precision_passed and recall_passed
        
        return {
            'test_name': 'Treatment Proxy Classification (Z_t)',
            'test_id': 'L1-3',
            'metrics': metrics,
            'thresholds': {
                'min_precision': precision_thresh,
                'min_recall': recall_thresh
            },
            'passed': passed,
            'interpretation': (
                f"Precision: {'PASS' if precision_passed else 'FAIL'} ({metrics['precision']:.3f} vs {precision_thresh}); "
                f"Recall: {'PASS' if recall_passed else 'FAIL'} ({metrics['recall']:.3f} vs {recall_thresh})"
            )
        }
    
    def test_outcome_proxy_classification(self, results: List[Dict]) -> Dict:
        """
        Test L1-4: Outcome-confounder proxy (W_t) classification.
        """
        metrics = self.compute_classification_metrics(results, 'outcome_proxy')
        
        precision_thresh = CONFIG.proxy_classification.min_precision
        recall_thresh = CONFIG.proxy_classification.min_recall
        
        precision_passed = metrics['precision'] >= precision_thresh
        recall_passed = metrics['recall'] >= recall_thresh
        passed = precision_passed and recall_passed
        
        return {
            'test_name': 'Outcome Proxy Classification (W_t)',
            'test_id': 'L1-4',
            'metrics': metrics,
            'thresholds': {
                'min_precision': precision_thresh,
                'min_recall': recall_thresh
            },
            'passed': passed,
            'interpretation': (
                f"Precision: {'PASS' if precision_passed else 'FAIL'} ({metrics['precision']:.3f} vs {precision_thresh}); "
                f"Recall: {'PASS' if recall_passed else 'FAIL'} ({metrics['recall']:.3f} vs {recall_thresh})"
            )
        }
    
    def test_temporal_logic(self, results: List[Dict]) -> Dict:
        """
        Additional test: Verify temporal logic is applied correctly.
        
        Z mentions (before treatment) should tend to be classified as treatment_proxy
        W mentions (after treatment) should tend to be classified as outcome_proxy
        """
        z_results = [r for r in results if r['type'] == 'Z']
        w_results = [r for r in results if r['type'] == 'W']
        
        # Z should more often be treatment_proxy than outcome_proxy
        z_as_treatment = sum(1 for r in z_results if r['predicted_role'] == 'treatment_proxy')
        z_as_outcome = sum(1 for r in z_results if r['predicted_role'] == 'outcome_proxy')
        
        # W should more often be outcome_proxy than treatment_proxy  
        w_as_outcome = sum(1 for r in w_results if r['predicted_role'] == 'outcome_proxy')
        w_as_treatment = sum(1 for r in w_results if r['predicted_role'] == 'treatment_proxy')
        
        z_temporal_correct = z_as_treatment > z_as_outcome
        w_temporal_correct = w_as_outcome > w_as_treatment
        passed = z_temporal_correct and w_temporal_correct
        
        return {
            'test_name': 'Temporal Logic Verification',
            'z_classified_as_treatment': z_as_treatment,
            'z_classified_as_outcome': z_as_outcome,
            'w_classified_as_outcome': w_as_outcome,
            'w_classified_as_treatment': w_as_treatment,
            'z_temporal_correct': z_temporal_correct,
            'w_temporal_correct': w_temporal_correct,
            'passed': passed,
            'interpretation': (
                f"Z: {'✓' if z_temporal_correct else '✗'} (treatment:{z_as_treatment} > outcome:{z_as_outcome}); "
                f"W: {'✓' if w_temporal_correct else '✗'} (outcome:{w_as_outcome} > treatment:{w_as_treatment})"
            )
        }
    
    def run_all_tests(self) -> Dict:
        """Execute all L1-3 and L1-4 tests."""
        print("="*60)
        print("TESTS L1-3 & L1-4: Proxy Classification")
        print("="*60)
        
        # Generate or load data
        print("\n[1/5] Loading synthetic causal data...")
        self.generate_or_load_synthetic_data()
        print(f"      Loaded {len(self.synthetic_data.patient_days)} patient-days")
        print(f"      True causal effect: {self.synthetic_data.true_causal_effect}")
        print(f"      Confounding strength: {self.synthetic_data.confounding_strength}")
        
        # Run classifications
        print("\n[2/5] Classifying all proxy candidates...")
        classification_results = self.classify_all_proxies()
        print(f"      Classified {len(classification_results)} concepts")
        
        # Test treatment proxy
        print("\n[3/5] Testing treatment proxy (Z_t) classification...")
        z_test = self.test_treatment_proxy_classification(classification_results)
        print(f"      {z_test['interpretation']}")
        
        # Test outcome proxy
        print("\n[4/5] Testing outcome proxy (W_t) classification...")
        w_test = self.test_outcome_proxy_classification(classification_results)
        print(f"      {w_test['interpretation']}")
        
        # Test temporal logic
        print("\n[5/5] Verifying temporal logic...")
        temporal_test = self.test_temporal_logic(classification_results)
        print(f"      {temporal_test['interpretation']}")
        
        # Overall results
        all_passed = z_test['passed'] and w_test['passed'] and temporal_test['passed']
        
        self.results = {
            'test_ids': ['L1-3', 'L1-4'],
            'test_name': 'Proxy Classification',
            'overall_passed': all_passed,
            'individual_tests': {
                'L1_3_treatment_proxy': z_test,
                'L1_4_outcome_proxy': w_test,
                'temporal_logic': temporal_test
            },
            'data_summary': {
                'num_patient_days': len(self.synthetic_data.patient_days),
                'num_classifications': len(classification_results),
                'true_causal_effect': self.synthetic_data.true_causal_effect,
                'confounding_strength': self.synthetic_data.confounding_strength
            }
        }
        
        print("\n" + "="*60)
        print(f"TEST L1-3 (Z_t): {'PASSED ✓' if z_test['passed'] else 'FAILED ✗'}")
        print(f"TEST L1-4 (W_t): {'PASSED ✓' if w_test['passed'] else 'FAILED ✗'}")
        print(f"OVERALL: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
        print("="*60)
        
        return self.results
    
    def save_results(self, output_dir: str = None):
        """Save test results to file."""
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'test_L1_3_4_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


def run_test():
    """Main entry point for running Tests L1-3 and L1-4."""
    test = TestL1_3_4_ProxyClassification()
    results = test.run_all_tests()
    test.save_results()
    return results


if __name__ == "__main__":
    run_test()
