"""
AEGIS 3.0 Layer 2 Testing - Configuration

Configuration parameters for Adaptive Digital Twin tests.
Thresholds based on the test specification document.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


@dataclass
class BergmanModelConfig:
    """Bergman Minimal Model parameters (population averages)."""
    
    # Glucose subsystem
    p1: float = 0.028  # Glucose effectiveness (min^-1)
    p2: float = 0.025  # Rate of insulin action decline (min^-1)
    p3: float = 0.000013  # Insulin sensitivity (min^-2 per μU/mL)
    
    # Insulin subsystem
    n: float = 0.23  # Insulin clearance rate (min^-1)
    gamma: float = 0.004  # Pancreatic response (min^-1 per mg/dL)
    
    # Baseline values
    Gb: float = 100.0  # Basal glucose (mg/dL)
    Ib: float = 10.0   # Basal insulin (μU/mL)
    
    # Physiological bounds
    glucose_min: float = 40.0   # mg/dL
    glucose_max: float = 400.0  # mg/dL
    insulin_min: float = 0.0    # μU/mL
    insulin_max: float = 100.0  # μU/mL


@dataclass
class UDEConfig:
    """Universal Differential Equation configuration."""
    
    # Neural network architecture
    hidden_dim: int = 32
    num_layers: int = 2
    activation: str = "tanh"
    
    # Training
    learning_rate: float = 0.01
    max_epochs: int = 100
    batch_size: int = 32
    
    # Success criteria
    min_rmse_improvement: float = 0.30  # 30% improvement over mechanistic
    min_residual_correlation: float = 0.5


@dataclass
class ACUKFConfig:
    """Adaptive Constrained UKF configuration."""
    
    # UKF parameters
    alpha: float = 0.001  # Spread of sigma points
    beta: float = 2.0     # Prior knowledge (Gaussian: 2)
    kappa: float = 0.0    # Secondary scaling
    
    # Adaptation
    adaptation_rate: float = 0.1  # Rate of Q adaptation
    residual_window: int = 10     # Window for residual statistics
    
    # Success criteria
    min_q_adaptation_ratio: float = 1.5  # Q should increase by 50% during noise


@dataclass
class RBPFConfig:
    """Rao-Blackwellized Particle Filter configuration."""
    
    # Particle filter settings
    num_particles: int = 100
    resampling_threshold: float = 0.5  # ESS/N threshold for resampling
    
    # Success criteria
    min_bimodal_coverage: float = 0.90  # 90% coverage when bimodal
    max_unimodal_coverage: float = 0.80  # UKF should fail below this when bimodal


@dataclass
class FilterSwitchingConfig:
    """Automatic filter switching configuration."""
    
    # Detection thresholds (from paper)
    shapiro_wilk_threshold: float = 0.05  # p-value threshold
    bimodality_coefficient_threshold: float = 0.555
    
    # Success criteria
    min_detection_accuracy: float = 0.90  # 90% correct switching


@dataclass
class IntegrationConfig:
    """Grey-box integration test configuration."""
    
    # Data generation
    num_training_obs: int = 100
    num_test_obs: int = 200
    num_synthetic_patients: int = 10
    
    # Simulation
    dt: float = 5.0  # minutes per timestep


@dataclass
class Layer2TestConfig:
    """Master configuration for Layer 2 tests."""
    
    bergman: BergmanModelConfig = None
    ude: UDEConfig = None
    ac_ukf: ACUKFConfig = None
    rbpf: RBPFConfig = None
    filter_switching: FilterSwitchingConfig = None
    integration: IntegrationConfig = None
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        if self.bergman is None:
            self.bergman = BergmanModelConfig()
        if self.ude is None:
            self.ude = UDEConfig()
        if self.ac_ukf is None:
            self.ac_ukf = ACUKFConfig()
        if self.rbpf is None:
            self.rbpf = RBPFConfig()
        if self.filter_switching is None:
            self.filter_switching = FilterSwitchingConfig()
        if self.integration is None:
            self.integration = IntegrationConfig()


# Global configuration instance
CONFIG = Layer2TestConfig()
