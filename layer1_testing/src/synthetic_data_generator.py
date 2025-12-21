"""
AEGIS 3.0 Layer 1 - Synthetic Data Generator

Generates synthetic patient data with KNOWN causal structure for
testing proxy classification and integration with causal inference.

The key insight: we can only validate proxy classification rigorously
when we KNOW the true causal structure. This module creates such data.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os

try:
    from .config import CONFIG, TREATMENT_PROXY_PATTERNS, OUTCOME_PROXY_PATTERNS
except ImportError:
    from config import CONFIG, TREATMENT_PROXY_PATTERNS, OUTCOME_PROXY_PATTERNS


@dataclass
class SyntheticPatientDay:
    """Represents one day of synthetic patient data with known causal structure."""
    
    # Required fields (no defaults) must come first
    day_id: int
    U: float  # Latent confounder (stress level, not directly observed)
    A: int  # Treatment (insulin adjustment decision): 0 = no change, 1 = adjusted
    Y: float  # Outcome (glucose excursion count)
    Z_text: str  # Treatment-confounder proxy text
    W_text: str  # Outcome-confounder proxy text
    
    # Optional fields with defaults
    Z_true_role: str = "treatment_proxy"
    W_true_role: str = "outcome_proxy"
    treatment_hour: float = 8.0  # 8am treatment decision
    Z_hour: float = 6.0  # 6am diary entry (before treatment)
    W_hour: float = 20.0  # 8pm diary entry (after treatment/outcome)


@dataclass
class SyntheticDataset:
    """Collection of synthetic patient days for testing."""
    
    patient_days: List[SyntheticPatientDay] = field(default_factory=list)
    
    # True causal parameters
    true_causal_effect: float = 0.5
    confounding_strength: float = 2.0
    
    # Dataset metadata
    seed: int = 42
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'metadata': {
                'true_causal_effect': self.true_causal_effect,
                'confounding_strength': self.confounding_strength,
                'seed': self.seed,
                'num_days': len(self.patient_days)
            },
            'patient_days': [
                {
                    'day_id': pd.day_id,
                    'U': pd.U,
                    'A': pd.A,
                    'Y': pd.Y,
                    'Z_text': pd.Z_text,
                    'Z_true_role': pd.Z_true_role,
                    'W_text': pd.W_text,
                    'W_true_role': pd.W_true_role,
                    'treatment_hour': pd.treatment_hour,
                    'Z_hour': pd.Z_hour,
                    'W_hour': pd.W_hour
                }
                for pd in self.patient_days
            ]
        }
    
    def save(self, filepath: str):
        """Save dataset to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SyntheticDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        patient_days = [
            SyntheticPatientDay(**pd_data)
            for pd_data in data['patient_days']
        ]
        
        return cls(
            patient_days=patient_days,
            true_causal_effect=data['metadata']['true_causal_effect'],
            confounding_strength=data['metadata']['confounding_strength'],
            seed=data['metadata']['seed']
        )


class SyntheticDataGenerator:
    """
    Generates synthetic data with known causal structure:
    
    Causal Graph:
    
        U (stress)
         │
         ├──────► A (treatment decision)
         │
         ├──────► Y (outcome)
         │
         ├──────► Z (treatment proxy text)
         │
         └──────► W (outcome proxy text)
    
    Key properties:
    - Z is caused by U, precedes A, doesn't directly affect Y
    - W is caused by U, follows A, predicts Y (but via U, not treatment)
    - True causal effect A→Y is known
    """
    
    # Text templates for treatment-proxy mentions (stress-related, before treatment)
    Z_TEMPLATES = [
        "have a big work deadline today",
        "feeling stressed about meeting",
        "busy schedule this morning", 
        "rushing to get ready",
        "work stress building up",
        "lots of anxiety about presentation",
        "hectic morning ahead",
        "travel day stress",
        "running late this morning",
        "stressful day coming up",
    ]
    
    # Text templates for outcome-proxy mentions (symptom-related, after treatment)
    W_TEMPLATES = [
        "couldnt sleep well last night",
        "feeling tired this evening",
        "had a headache today",
        "felt fatigued all day",
        "poor sleep again",
        "exhausted by evening",
        "drowsy this afternoon",
        "low energy today",
        "nausea in the evening",
        "feeling wiped out",
    ]
    
    # Neutral templates (neither proxy)
    NEUTRAL_TEMPLATES = [
        "regular day today",
        "nothing special to report",
        "usual routine",
        "all normal",
        "typical activities",
    ]
    
    def __init__(self,
                 true_causal_effect: float = None,
                 confounding_strength: float = None,
                 seed: int = None):
        """
        Initialize the generator.
        
        Args:
            true_causal_effect: True treatment effect on outcome (τ)
            confounding_strength: How strongly U affects Y (γ)
            seed: Random seed for reproducibility
        """
        config = CONFIG.proxy_classification
        int_config = CONFIG.integration
        
        self.true_causal_effect = true_causal_effect or int_config.true_causal_effect
        self.confounding_strength = confounding_strength or config.confounding_strength
        self.seed = seed or CONFIG.random_seed
        
        np.random.seed(self.seed)
    
    def generate_patient_day(self, day_id: int) -> SyntheticPatientDay:
        """
        Generate one synthetic patient day.
        
        Causal mechanism:
        1. U ~ N(0, 1) : Latent stress confounder
        2. P(A=1 | U) = sigmoid(0.5 * U) : Stress affects treatment
        3. Y = baseline + τ*A + γ*U + ε : Outcome affected by treatment and confounder
        4. Z ~ templates[high stress indicator] if U > 0 : Stress causes Z mention
        5. W ~ templates[symptoms] if U > 0 : Stress causes W symptoms
        """
        # Step 1: Generate latent confounder
        U = np.random.randn()
        
        # Step 2: Generate treatment (confounded by U)
        treatment_prob = 1 / (1 + np.exp(-0.5 * U))  # sigmoid
        A = int(np.random.binomial(1, treatment_prob))
        
        # Step 3: Generate outcome
        baseline = 100.0
        noise = np.random.randn() * 2
        Y = baseline + self.true_causal_effect * A + self.confounding_strength * U + noise
        
        # Step 4: Generate treatment proxy text (caused by U)
        if U > 0.3:  # High stress → likely to mention stress-related text
            Z_text = np.random.choice(self.Z_TEMPLATES)
        elif U > -0.3:  # Medium stress → might mention
            if np.random.random() < 0.3:
                Z_text = np.random.choice(self.Z_TEMPLATES)
            else:
                Z_text = np.random.choice(self.NEUTRAL_TEMPLATES)
        else:  # Low stress → neutral
            Z_text = np.random.choice(self.NEUTRAL_TEMPLATES)
        
        # Step 5: Generate outcome proxy text (caused by U)
        if U > 0.3:  # High stress → likely to report symptoms
            W_text = np.random.choice(self.W_TEMPLATES)
        elif U > -0.3:
            if np.random.random() < 0.3:
                W_text = np.random.choice(self.W_TEMPLATES)
            else:
                W_text = np.random.choice(self.NEUTRAL_TEMPLATES)
        else:
            W_text = np.random.choice(self.NEUTRAL_TEMPLATES)
        
        # Determine true roles based on whether text matches proxy patterns
        Z_is_proxy = any(p.lower() in Z_text.lower() for p in TREATMENT_PROXY_PATTERNS)
        W_is_proxy = any(p.lower() in W_text.lower() for p in OUTCOME_PROXY_PATTERNS)
        
        return SyntheticPatientDay(
            day_id=day_id,
            U=U,
            A=A,
            Y=Y,
            Z_text=Z_text,
            Z_true_role='treatment_proxy' if Z_is_proxy else 'neither',
            W_text=W_text,
            W_true_role='outcome_proxy' if W_is_proxy else 'neither',
            treatment_hour=8.0 + np.random.randn() * 0.5,  # Around 8am
            Z_hour=6.0 + np.random.random() * 1.5,  # 6-7:30am
            W_hour=19.0 + np.random.random() * 2.0,  # 7-9pm
        )
    
    def generate_dataset(self, num_days: int = None) -> SyntheticDataset:
        """
        Generate complete synthetic dataset.
        
        Args:
            num_days: Number of patient-days to generate
            
        Returns:
            SyntheticDataset with known causal structure
        """
        num_days = num_days or CONFIG.proxy_classification.num_synthetic_patient_days
        
        patient_days = [
            self.generate_patient_day(i)
            for i in range(num_days)
        ]
        
        return SyntheticDataset(
            patient_days=patient_days,
            true_causal_effect=self.true_causal_effect,
            confounding_strength=self.confounding_strength,
            seed=self.seed
        )


def generate_and_save_test_data(output_dir: str = None):
    """
    Generate and save all synthetic test data.
    
    Args:
        output_dir: Directory to save data files
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate main synthetic dataset (500 days)
    generator = SyntheticDataGenerator(seed=CONFIG.random_seed)
    dataset = generator.generate_dataset(500)
    dataset.save(os.path.join(output_dir, 'synthetic_causal_data.json'))
    
    # Generate larger integration test dataset (1000 days)
    generator_large = SyntheticDataGenerator(seed=CONFIG.random_seed + 1)
    large_dataset = generator_large.generate_dataset(1000)
    large_dataset.save(os.path.join(output_dir, 'integration_test_data.json'))
    
    print(f"Generated synthetic data:")
    print(f"  - synthetic_causal_data.json: {len(dataset.patient_days)} days")
    print(f"  - integration_test_data.json: {len(large_dataset.patient_days)} days")
    print(f"  - True causal effect: {dataset.true_causal_effect}")
    print(f"  - Confounding strength: {dataset.confounding_strength}")


if __name__ == "__main__":
    generate_and_save_test_data()
