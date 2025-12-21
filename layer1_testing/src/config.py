"""
AEGIS 3.0 Layer 1 Testing - Configuration

This module defines all configuration parameters for Layer 1 tests.
Thresholds are set based on the test specification document.
"""

from dataclasses import dataclass
from typing import List, Dict
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


@dataclass
class SemanticEntropyConfig:
    """Configuration for semantic entropy testing."""
    
    # Number of candidate extractions to generate
    num_candidates: int = 10
    
    # Sampling temperatures for generating candidates
    temperatures: List[float] = None
    
    # HITL trigger threshold (tunable)
    entropy_threshold: float = 1.0
    
    # Success criteria from test specification
    min_spearman_correlation: float = 0.6
    min_auc_roc: float = 0.75
    min_high_ambiguity_recall: float = 0.8
    
    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = [0.3, 0.5, 0.7, 0.9, 1.1]


@dataclass
class HITLTriggerConfig:
    """Configuration for HITL trigger calibration."""
    
    # Success criteria
    min_error_capture_rate: float = 0.8
    max_false_alarm_rate: float = 0.4


@dataclass
class ProxyClassificationConfig:
    """Configuration for proxy classification testing."""
    
    # Success criteria
    min_precision: float = 0.7
    min_recall: float = 0.6
    
    # Synthetic data generation parameters
    num_synthetic_patient_days: int = 500
    confounding_strength: float = 1.0  # Reduced from 2.0 for realistic testing
    proxy_noise_std: float = 0.3  # Reduced from 0.5 for better signal
    
    # Temporal window for proxy classification (hours)
    treatment_proxy_window_before: float = 6.0  # Z mentioned within 6h before treatment
    outcome_proxy_window_after: float = 12.0     # W mentioned within 12h after treatment


@dataclass 
class IntegrationTestConfig:
    """Configuration for the integration test (L1-5)."""
    
    # Data generation
    num_observations: int = 1000
    true_causal_effect: float = 0.5
    
    # Success criteria
    min_bias_reduction: float = 0.3  # 30% improvement over naive


@dataclass
class Layer1TestConfig:
    """Master configuration for all Layer 1 tests."""
    
    semantic_entropy: SemanticEntropyConfig = None
    hitl_trigger: HITLTriggerConfig = None
    proxy_classification: ProxyClassificationConfig = None
    integration: IntegrationTestConfig = None
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        if self.semantic_entropy is None:
            self.semantic_entropy = SemanticEntropyConfig()
        if self.hitl_trigger is None:
            self.hitl_trigger = HITLTriggerConfig()
        if self.proxy_classification is None:
            self.proxy_classification = ProxyClassificationConfig()
        if self.integration is None:
            self.integration = IntegrationTestConfig()


# Global configuration instance
CONFIG = Layer1TestConfig()


# SNOMED-CT Concept Mapping (Simplified for testing)
# In production, this would use actual SNOMED-CT API
CONCEPT_MAPPING = {
    # Symptoms
    "headache": "25064002",
    "nausea": "422587007", 
    "fatigue": "84229001",
    "weakness": "13791008",
    "dizziness": "404640003",
    "drowsiness": "271782001",
    "tired": "84229001",  # Maps to fatigue
    "sleepy": "271782001",  # Maps to drowsiness
    "exhausted": "84229001",  # Maps to fatigue
    
    # Diabetes-specific
    "hypoglycemia": "302866003",
    "hyperglycemia": "80394007",
    "blood glucose": "33747003",
    "insulin": "412222008",
    "glucose": "33747003",
    
    # Measurements
    "blood pressure": "75367002", 
    "blood sugar": "33747003",
    
    # Activities
    "exercise": "256235009",
    "physical activity": "256235009",
    "sleep": "258158006",
    
    # Diet
    "meal": "226379006",
    "eating": "226379006",
    "carbohydrate": "2331003",
    
    # Psychological
    "stress": "73595000",
    "anxiety": "48694002",
    "work deadline": "73595000",  # Maps to stress (contextual)
    "work stress": "73595000",
    
    # Sleep issues
    "poor sleep": "193462001",
    "insomnia": "193462001",
    "couldnt sleep": "193462001",
    
    # Vague/ambiguous - multiple possible mappings
    "malaise": "367391008",
    "unwell": "367391008",
    "off": "367391008",  # Likely malaise
    "different": "367391008",
    
    # Non-specific
    "unknown": "261665006",
    "unclear": "261665006",
}


# Treatment-confounder proxy patterns (for classification)
# These are concepts that PRECEDE treatment decisions and relate to confounders
TREATMENT_PROXY_PATTERNS = [
    "work deadline",
    "work stress", 
    "meeting",
    "travel",
    "busy day",
    "schedule change",
    "stress",
    "anxiety",
    "rushing",
    "forgot",
    "running late",
]

# Outcome-confounder proxy patterns (for classification)
# These are concepts that FOLLOW treatment and relate to outcomes via confounders
OUTCOME_PROXY_PATTERNS = [
    "couldnt sleep",
    "poor sleep",
    "tired",
    "fatigue",
    "exhausted",
    "headache",
    "nausea",
    "dizziness",
    "irritable",
    "mood",
]
