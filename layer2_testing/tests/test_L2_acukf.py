"""
AEGIS 3.0 Layer 2 - Test L2-3 & L2-4: AC-UKF Tests

Tests the Adaptive Constrained UKF for:
1. Innovation-based covariance adaptation (Q adapts when residuals large)
2. Constraint projection (states stay within bounds)

Success Criteria:
- Q increases when residuals exceed expected
- No constraint violations in state estimates
"""

import sys
import os
import json
import numpy as np
from typing import Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from config import CONFIG, DATA_DIR, RESULTS_DIR
from bergman_model import BergmanMinimalModel, create_meal_disturbance
from ac_ukf import AdaptiveConstrainedUKF


class TestL2_ACUKF:
    """
    Test L2-3 & L2-4: AC-UKF Covariance Adaptation and Constraint Projection
    """
    
    def __init__(self):
        self.results = {}
    
    def create_test_scenario(self, 
                              noise_injection_start: int = 30,
                              noise_injection_end: int = 50,
                              noise_multiplier: float = 5.0,
                              seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create test scenario with known noise injection period.
        
        Returns:
            times, true_states, measurements
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Simulate true dynamics
        model = BergmanMinimalModel()
        meal_times = [60, 300]
        meal_sizes = [50, 40]
        D_func = create_meal_disturbance(meal_times, meal_sizes)
        
        initial_state = model.get_equilibrium()
        times, true_states = model.simulate(
            initial_state,
            t_span=(0, 500),
            dt=5.0,
            D_func=D_func,
            noise_std=1.0
        )
        
        # Create measurements (glucose only) with variable noise
        base_noise_std = 5.0
        measurements = np.zeros((len(times), 1))
        noise_std_used = np.zeros(len(times))
        
        for i in range(len(times)):
            if noise_injection_start <= i < noise_injection_end:
                noise = base_noise_std * noise_multiplier
            else:
                noise = base_noise_std
            
            noise_std_used[i] = noise
            measurements[i, 0] = true_states[i, 0] + np.random.randn() * noise
        
        return times, true_states, measurements, noise_std_used
    
    def test_covariance_adaptation(self) -> Dict:
        """
        Test L2-3: Verify Q adapts when residuals exceed expected.
        """
        print("\n--- Test L2-3: Covariance Adaptation ---")
        
        # Create scenario with noise injection
        times, true_states, measurements, noise_profile = self.create_test_scenario(
            noise_injection_start=30,
            noise_injection_end=50,
            noise_multiplier=5.0,
            seed=CONFIG.random_seed
        )
        
        # Create dynamics function for filter
        model = BergmanMinimalModel()
        D_func = create_meal_disturbance([60, 300], [50, 40])
        
        def fx(x, dt, u=None):
            # Simple Euler step
            deriv = model.dynamics(x, 0, D_func(0), u if u else 0)
            return x + deriv * dt
        
        def hx(x):
            return x[:1]  # Observe glucose only
        
        # Initialize filter
        ukf = AdaptiveConstrainedUKF(
            dim_x=3,
            dim_z=1,
            dt=5.0,
            fx=fx,
            hx=hx
        )
        ukf.x = true_states[0].copy()
        
        # Track Q over time
        q_history = []
        estimation_errors = []
        
        for i in range(len(times)):
            # Predict
            ukf.predict()
            
            # Update with measurement
            if i > 0:
                ukf.update(measurements[i])
            
            q_history.append(ukf.get_q_ratio())
            estimation_errors.append(abs(ukf.x[0] - true_states[i, 0]))
        
        # Analyze results
        q_before = np.mean(q_history[10:30])  # Before noise injection
        q_during = np.mean(q_history[35:50])  # During noise injection
        q_after = np.mean(q_history[55:75])   # After noise injection
        
        q_increased = q_during > q_before
        q_ratio = q_during / q_before if q_before > 0 else 1.0
        min_ratio = CONFIG.ac_ukf.min_q_adaptation_ratio
        
        # Error should remain bounded
        error_during = np.mean(estimation_errors[30:50])
        error_before = np.mean(estimation_errors[10:30])
        error_bounded = error_during < error_before * 3  # Allow some degradation
        
        result = {
            'test_name': 'Covariance Adaptation',
            'test_id': 'L2-3',
            'q_before_noise': float(q_before),
            'q_during_noise': float(q_during),
            'q_after_noise': float(q_after),
            'q_ratio': float(q_ratio),
            'min_ratio_required': min_ratio,
            'q_increased': q_increased,
            'error_during_noise': float(error_during),
            'error_before_noise': float(error_before),
            'error_bounded': error_bounded,
            'passed': q_increased and q_ratio >= min_ratio,
            'interpretation': (
                f"Q ratio before/during: {q_ratio:.2f} "
                f"{'≥' if q_ratio >= min_ratio else '<'} {min_ratio} "
                f"- {'PASS ✓' if q_increased and q_ratio >= min_ratio else 'FAIL ✗'}"
            )
        }
        
        print(f"  Q before noise: {q_before:.3f}")
        print(f"  Q during noise: {q_during:.3f}")
        print(f"  Q ratio: {q_ratio:.2f}")
        print(f"  {result['interpretation']}")
        
        return result
    
    def test_constraint_projection(self) -> Dict:
        """
        Test L2-4: Verify constraint projection maintains physiological bounds.
        """
        print("\n--- Test L2-4: Constraint Projection ---")
        
        # Create challenging scenario that pushes state near bounds
        np.random.seed(CONFIG.random_seed + 100)
        
        # Initialize near boundary
        initial_state = np.array([60.0, 0.0, 5.0])  # Low glucose
        
        # Create filter
        model = BergmanMinimalModel()
        
        def fx(x, dt, u=None):
            # Add large perturbation that would push out of bounds
            deriv = model.dynamics(x, 0, 0, 0)
            # Simulate a low glucose event
            deriv[0] -= 5.0  # Push glucose down
            return x + deriv * dt
        
        def hx(x):
            return x[:1]
        
        ukf = AdaptiveConstrainedUKF(
            dim_x=3,
            dim_z=1,
            dt=5.0,
            fx=fx,
            hx=hx
        )
        ukf.x = initial_state.copy()
        
        # Run filter, track violations
        state_history = [initial_state.copy()]
        violation_count = 0
        
        for i in range(50):
            ukf.predict()
            
            # Create measurement near boundary
            noisy_glucose = max(45, initial_state[0] - 2 + np.random.randn() * 10)
            ukf.update(np.array([noisy_glucose]))
            
            state = ukf.x.copy()
            state_history.append(state)
            
            # Check violations
            cfg = CONFIG.bergman
            if state[0] < cfg.glucose_min or state[0] > cfg.glucose_max:
                violation_count += 1
            if state[1] < 0:
                violation_count += 1
            if state[2] < cfg.insulin_min or state[2] > cfg.insulin_max:
                violation_count += 1
        
        state_history = np.array(state_history)
        
        # Compute statistics
        glucose_min = np.min(state_history[:, 0])
        glucose_max = np.max(state_history[:, 0])
        
        passed = violation_count == 0
        
        result = {
            'test_name': 'Constraint Projection',
            'test_id': 'L2-4',
            'violation_count': int(violation_count),
            'glucose_min': float(glucose_min),
            'glucose_max': float(glucose_max),
            'bound_glucose_min': float(CONFIG.bergman.glucose_min),
            'bound_glucose_max': float(CONFIG.bergman.glucose_max),
            'passed': passed,
            'interpretation': (
                f"Constraint violations: {violation_count} "
                f"- {'PASS ✓' if passed else 'FAIL ✗'}"
            )
        }
        
        print(f"  Glucose range: [{glucose_min:.1f}, {glucose_max:.1f}] mg/dL")
        print(f"  Bound range: [{CONFIG.bergman.glucose_min}, {CONFIG.bergman.glucose_max}] mg/dL")
        print(f"  {result['interpretation']}")
        
        return result
    
    def run_all_tests(self) -> Dict:
        """Execute all AC-UKF tests."""
        print("=" * 60)
        print("TEST L2-3/4: AC-UKF Covariance Adaptation & Constraint Projection")
        print("=" * 60)
        
        cov_result = self.test_covariance_adaptation()
        constraint_result = self.test_constraint_projection()
        
        overall_passed = cov_result['passed'] and constraint_result['passed']
        
        self.results = {
            'test_ids': ['L2-3', 'L2-4'],
            'test_name': 'AC-UKF Tests',
            'overall_passed': overall_passed,
            'individual_tests': {
                'L2_3_covariance_adaptation': cov_result,
                'L2_4_constraint_projection': constraint_result
            }
        }
        
        print("\n" + "=" * 60)
        print(f"TEST L2-3/4 OVERALL: {'PASSED ✓' if overall_passed else 'FAILED ✗'}")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_dir: str = None):
        """Save test results."""
        if output_dir is None:
            output_dir = RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'test_L2_acukf_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


def run_test():
    """Main entry point."""
    test = TestL2_ACUKF()
    results = test.run_all_tests()
    test.save_results()
    return results


if __name__ == "__main__":
    run_test()
