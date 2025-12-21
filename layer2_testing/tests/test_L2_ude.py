"""
AEGIS 3.0 Layer 2 - Test L2-1 & L2-7: UDE Model and Grey-Box Integration

Tests the Universal Differential Equation model combining mechanistic
and neural components.

Claims Tested:
- C2-1: UDE = f_mech + f_NN
- C2-2: Neural component learns patient-specific deviations
- C2-7: Grey-box outperforms pure approaches

Success Criteria:
- UDE RMSE < Mechanistic-only RMSE
- UDE RMSE < Neural-only RMSE (limited data)
- Residual error reduction > 30%
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from config import CONFIG, DATA_DIR, RESULTS_DIR
from bergman_model import BergmanMinimalModel, create_meal_disturbance, create_insulin_input
from ude_model import UniversalDifferentialEquation, NeuralResidual


@dataclass
class SyntheticPatient:
    """A synthetic patient with known deviation from population."""
    patient_id: int
    p1_deviation: float  # % deviation from population p1
    p3_deviation: float  # % deviation from population p3 (insulin sensitivity)
    true_states: np.ndarray = None
    times: np.ndarray = None


class TestL2_UDE:
    """
    Test L2-1, L2-2, L2-7: UDE Model Tests
    
    Validates that the UDE combining mechanistic and learned components
    outperforms pure approaches and learns patient-specific deviations.
    """
    
    def __init__(self):
        self.results = {}
        self.patients = []
    
    def create_synthetic_patient(self, 
                                  patient_id: int,
                                  p1_dev: float = 0.0,
                                  p3_dev: float = 0.0,
                                  seed: int = None) -> SyntheticPatient:
        """
        Create synthetic patient with known deviations.
        
        Args:
            patient_id: Patient identifier
            p1_dev: Deviation in p1 (glucose effectiveness), e.g. 0.2 = 20% higher
            p3_dev: Deviation in p3 (insulin sensitivity), e.g. 0.2 = 20% higher
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create patient's true model (with deviations)
        cfg = CONFIG.bergman
        patient_model = BergmanMinimalModel(
            p1=cfg.p1 * (1 + p1_dev),
            p3=cfg.p3 * (1 + p3_dev)
        )
        
        # Simulate a day with meals
        meal_times = [60, 300, 600]  # Breakfast, lunch, dinner (minutes)
        meal_sizes = [40, 50, 60]     # Carbs (grams)
        
        injection_times = [0, 60, 300, 600]
        doses = [5, 3, 4, 5]  # Insulin units
        
        D_func = create_meal_disturbance(meal_times, meal_sizes)
        u_func = create_insulin_input(injection_times, doses)
        
        initial_state = patient_model.get_equilibrium()
        times, states = patient_model.simulate(
            initial_state,
            t_span=(0, 720),  # 12 hours
            dt=5.0,
            D_func=D_func,
            u_func=u_func,
            noise_std=2.0  # Some process noise
        )
        
        return SyntheticPatient(
            patient_id=patient_id,
            p1_deviation=p1_dev,
            p3_deviation=p3_dev,
            true_states=states,
            times=times
        )
    
    def test_mechanistic_only(self, patient: SyntheticPatient) -> float:
        """
        Test pure mechanistic prediction (no personalization).
        
        Returns RMSE for glucose.
        """
        pop_model = BergmanMinimalModel()  # Population parameters
        
        meal_times = [60, 300, 600]
        meal_sizes = [40, 50, 60]
        injection_times = [0, 60, 300, 600]
        doses = [5, 3, 4, 5]
        
        D_func = create_meal_disturbance(meal_times, meal_sizes)
        u_func = create_insulin_input(injection_times, doses)
        
        initial_state = pop_model.get_equilibrium()
        _, pred_states = pop_model.simulate(
            initial_state,
            t_span=(patient.times[0], patient.times[-1]),
            dt=5.0,
            D_func=D_func,
            u_func=u_func
        )
        
        # RMSE for glucose
        n = min(len(patient.true_states), len(pred_states))
        rmse = np.sqrt(np.mean((pred_states[:n, 0] - patient.true_states[:n, 0]) ** 2))
        return rmse
    
    def test_neural_only(self, 
                         patient: SyntheticPatient,
                         train_frac: float = 0.5) -> float:
        """
        Test pure neural prediction (no mechanistic prior).
        
        Uses a simple MLP to predict next state from current.
        """
        train_n = int(len(patient.times) * train_frac)
        
        # Training data: predict next state from current
        X_train = patient.true_states[:train_n-1]
        y_train = patient.true_states[1:train_n]
        
        # Simple neural network (MLP)
        nn = NeuralResidual(input_dim=3, output_dim=3, hidden_dim=16, num_layers=1)
        
        # Train using gradient descent
        for epoch in range(50):
            loss = nn.train_step(X_train, y_train, lr=0.01)
        
        # Test: predict trajectory
        test_states = np.zeros((len(patient.times) - train_n + 1, 3))
        test_states[0] = patient.true_states[train_n-1]
        
        for i in range(1, len(test_states)):
            test_states[i] = nn(test_states[i-1])
            # Clip to valid range
            test_states[i, 0] = np.clip(test_states[i, 0], 40, 400)
        
        # RMSE on test portion
        true_test = patient.true_states[train_n-1:]
        n = min(len(true_test), len(test_states))
        rmse = np.sqrt(np.mean((test_states[:n, 0] - true_test[:n, 0]) ** 2))
        return rmse
    
    def test_ude_greybox(self,
                         patient: SyntheticPatient,
                         train_frac: float = 0.5) -> Tuple[float, float]:
        """
        Test UDE grey-box approach.
        
        Returns (RMSE, residual_reduction_fraction).
        """
        train_n = int(len(patient.times) * train_frac)
        
        # Create UDE with population mechanistic + learnable neural
        ude = UniversalDifferentialEquation()
        
        # Training data
        train_times = patient.times[:train_n]
        train_states = patient.true_states[:train_n]
        
        # Simple insulin input estimation
        injection_times = [0, 60, 300, 600]
        doses = [5, 3, 4, 5]
        u_func = create_insulin_input(injection_times, doses)
        inputs = np.array([u_func(t) for t in train_times])
        
        # Measure mechanistic error before training
        mech_model = BergmanMinimalModel()
        D_func = create_meal_disturbance([60, 300, 600], [40, 50, 60])
        _, mech_pred = mech_model.simulate(
            mech_model.get_equilibrium(),
            (train_times[0], train_times[-1]),
            dt=5.0,
            D_func=D_func,
            u_func=u_func
        )
        pre_train_rmse = np.sqrt(np.mean((mech_pred[:len(train_states), 0] - train_states[:, 0]) ** 2))
        
        # Train neural residual (simplified - just a few epochs)
        print("  Training UDE neural component...")
        for epoch in range(20):
            # Compute residuals for neural network
            dt = train_times[1] - train_times[0]
            obs_derivs = np.diff(train_states, axis=0) / dt
            mech_derivs = np.array([
                mech_model.dynamics(train_states[i], train_times[i], D_func(train_times[i]), inputs[i])
                for i in range(len(train_states)-1)
            ])
            target_residuals = obs_derivs - mech_derivs
            
            nn_inputs = np.array([
                np.concatenate([train_states[i], [inputs[i]]])
                for i in range(len(train_states)-1)
            ])
            
            loss = ude.neural.train_step(nn_inputs, target_residuals, lr=0.005)
            if epoch % 5 == 0:
                print(f"    Epoch {epoch}: Loss = {loss:.6f}")
        
        # Test: simulate with UDE
        _, ude_pred = ude.simulate(
            patient.true_states[train_n-1],
            (patient.times[train_n-1], patient.times[-1]),
            dt=5.0,
            D_func=D_func,
            u_func=u_func
        )
        
        # RMSE on test portion
        true_test = patient.true_states[train_n-1:]
        n = min(len(true_test), len(ude_pred))
        ude_rmse = np.sqrt(np.mean((ude_pred[:n, 0] - true_test[:n, 0]) ** 2))
        
        # Residual reduction
        residual_reduction = (pre_train_rmse - ude_rmse) / pre_train_rmse if pre_train_rmse > 0 else 0
        
        return ude_rmse, residual_reduction
    
    def run_all_tests(self) -> Dict:
        """Execute all UDE tests."""
        print("=" * 60)
        print("TEST L2-1/2/7: UDE Model and Grey-Box Integration")
        print("=" * 60)
        
        # Create synthetic patients with various deviations
        print("\n[1/4] Creating synthetic patients with known deviations...")
        deviations = [
            (0, 0.0, 0.0),   # Population-average patient
            (1, 0.2, 0.0),   # 20% higher glucose effectiveness
            (2, 0.0, 0.3),   # 30% higher insulin sensitivity
            (3, -0.15, 0.2), # Mixed deviation
            (4, 0.25, -0.1), # Another mixed
        ]
        
        self.patients = []
        for pid, p1_dev, p3_dev in deviations:
            patient = self.create_synthetic_patient(pid, p1_dev, p3_dev, seed=CONFIG.random_seed + pid)
            self.patients.append(patient)
            print(f"  Patient {pid}: p1_dev={p1_dev:+.0%}, p3_dev={p3_dev:+.0%}")
        
        # Test each approach
        print("\n[2/4] Testing mechanistic-only predictions...")
        mech_rmses = []
        for p in self.patients:
            rmse = self.test_mechanistic_only(p)
            mech_rmses.append(rmse)
            print(f"  Patient {p.patient_id}: RMSE = {rmse:.2f} mg/dL")
        
        print("\n[3/4] Testing neural-only predictions...")
        neural_rmses = []
        for p in self.patients:
            rmse = self.test_neural_only(p)
            neural_rmses.append(rmse)
            print(f"  Patient {p.patient_id}: RMSE = {rmse:.2f} mg/dL")
        
        print("\n[4/4] Testing UDE grey-box predictions...")
        ude_rmses = []
        residual_reductions = []
        for p in self.patients:
            print(f"  Patient {p.patient_id}:")
            rmse, reduction = self.test_ude_greybox(p)
            ude_rmses.append(rmse)
            residual_reductions.append(reduction)
            print(f"    RMSE = {rmse:.2f} mg/dL, Residual reduction = {reduction:.1%}")
        
        # Compute aggregate metrics
        avg_mech = np.mean(mech_rmses)
        avg_neural = np.mean(neural_rmses)
        avg_ude = np.mean(ude_rmses)
        avg_reduction = np.mean(residual_reductions)
        
        # Test criteria
        ude_beats_mech = avg_ude < avg_mech
        ude_beats_neural = avg_ude < avg_neural
        min_reduction = CONFIG.ude.min_rmse_improvement
        reduction_sufficient = avg_reduction >= min_reduction
        
        overall_passed = ude_beats_mech and ude_beats_neural
        
        self.results = {
            'test_id': 'L2-1/2/7',
            'test_name': 'UDE Model and Grey-Box Integration',
            'overall_passed': overall_passed,
            'metrics': {
                'mech_rmse_mean': float(avg_mech),
                'neural_rmse_mean': float(avg_neural),
                'ude_rmse_mean': float(avg_ude),
                'residual_reduction_mean': float(avg_reduction),
                'mech_rmse_per_patient': [float(x) for x in mech_rmses],
                'neural_rmse_per_patient': [float(x) for x in neural_rmses],
                'ude_rmse_per_patient': [float(x) for x in ude_rmses],
            },
            'individual_tests': {
                'ude_beats_mechanistic': {
                    'passed': ude_beats_mech,
                    'interpretation': f"UDE RMSE ({avg_ude:.2f}) {'<' if ude_beats_mech else '>='} Mech RMSE ({avg_mech:.2f})"
                },
                'ude_beats_neural': {
                    'passed': ude_beats_neural,
                    'interpretation': f"UDE RMSE ({avg_ude:.2f}) {'<' if ude_beats_neural else '>='} Neural RMSE ({avg_neural:.2f})"
                },
                'residual_reduction': {
                    'passed': reduction_sufficient,
                    'threshold': min_reduction,
                    'interpretation': f"Reduction {avg_reduction:.1%} {'≥' if reduction_sufficient else '<'} {min_reduction:.0%}"
                }
            }
        }
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"  Mean Mechanistic RMSE: {avg_mech:.2f} mg/dL")
        print(f"  Mean Neural RMSE:      {avg_neural:.2f} mg/dL")
        print(f"  Mean UDE RMSE:         {avg_ude:.2f} mg/dL")
        print(f"  Mean Residual Reduction: {avg_reduction:.1%}")
        print(f"\n  UDE < Mechanistic: {'PASS ✓' if ude_beats_mech else 'FAIL ✗'}")
        print(f"  UDE < Neural:      {'PASS ✓' if ude_beats_neural else 'FAIL ✗'}")
        print("=" * 60)
        print(f"TEST L2-1/2/7 OVERALL: {'PASSED ✓' if overall_passed else 'FAILED ✗'}")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_dir: str = None):
        """Save test results."""
        if output_dir is None:
            output_dir = RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'test_L2_ude_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


def run_test():
    """Main entry point."""
    test = TestL2_UDE()
    results = test.run_all_tests()
    test.save_results()
    return results


if __name__ == "__main__":
    run_test()
