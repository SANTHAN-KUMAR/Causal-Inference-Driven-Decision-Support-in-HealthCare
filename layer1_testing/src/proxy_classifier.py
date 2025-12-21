"""
AEGIS 3.0 Layer 1 - Causal Proxy Classifier

Implements classification of extracted semantic features into:
- Treatment-confounder proxies (Z_t): Per Definition 5.1
- Outcome-confounder proxies (W_t): Per Definition 5.2

This classification enables proximal causal inference for unmeasured
confounding adjustment in Layer 3.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

try:
    from .config import (
        TREATMENT_PROXY_PATTERNS, 
        OUTCOME_PROXY_PATTERNS,
        CONFIG
    )
except ImportError:
    from config import (
        TREATMENT_PROXY_PATTERNS, 
        OUTCOME_PROXY_PATTERNS,
        CONFIG
    )


@dataclass
class ExtractedConcept:
    """Represents an extracted medical concept with temporal information."""
    text: str
    concept_id: str
    timestamp: datetime
    confidence: float
    source: str = "diary"  # diary, survey, sensor


@dataclass
class ProxyClassification:
    """Result of proxy classification for a concept."""
    concept: ExtractedConcept
    role: str  # 'treatment_proxy', 'outcome_proxy', 'neither'
    confidence: float
    reason: str


class CausalProxyClassifier:
    """
    Classifies extracted semantic features by their causal role
    for proximal causal inference.
    
    Based on paper §5.1.4:
    - Treatment-confounder proxy (Z_t): Affects treatment decision via 
      unmeasured confounder, but doesn't directly affect outcome
    - Outcome-confounder proxy (W_t): Predicts outcome via unmeasured
      confounder, but isn't affected by treatment
    
    Classification uses:
    1. Pattern matching with domain knowledge
    2. Temporal precedence analysis relative to treatment times
    """
    
    def __init__(self,
                 treatment_proxy_patterns: List[str] = None,
                 outcome_proxy_patterns: List[str] = None,
                 treatment_window_hours: float = None,
                 outcome_window_hours: float = None):
        """
        Initialize the proxy classifier.
        
        Args:
            treatment_proxy_patterns: Patterns indicating treatment proxies
            outcome_proxy_patterns: Patterns indicating outcome proxies  
            treatment_window_hours: Hours before treatment to consider for Z_t
            outcome_window_hours: Hours after treatment to consider for W_t
        """
        self.treatment_patterns = treatment_proxy_patterns or TREATMENT_PROXY_PATTERNS
        self.outcome_patterns = outcome_proxy_patterns or OUTCOME_PROXY_PATTERNS
        
        config = CONFIG.proxy_classification
        self.treatment_window = treatment_window_hours or config.treatment_proxy_window_before
        self.outcome_window = outcome_window_hours or config.outcome_proxy_window_after
        
        # Compile patterns for efficient matching
        self._treatment_regex = self._compile_patterns(self.treatment_patterns)
        self._outcome_regex = self._compile_patterns(self.outcome_patterns)
    
    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        """Compile pattern strings into regex objects."""
        compiled = []
        for pattern in patterns:
            # Handle special characters and create word boundary match
            escaped = re.escape(pattern)
            regex = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
            compiled.append(regex)
        return compiled
    
    def _matches_pattern(self, text: str, patterns: List[re.Pattern]) -> Tuple[bool, Optional[str]]:
        """Check if text matches any pattern, return match if found."""
        text_lower = text.lower()
        for pattern in patterns:
            match = pattern.search(text_lower)
            if match:
                return True, match.group()
        return False, None
    
    def classify_proxy_role(self,
                           concept: ExtractedConcept,
                           treatment_times: List[datetime],
                           current_time: datetime = None) -> ProxyClassification:
        """
        Classify whether a concept serves as treatment proxy, outcome proxy, or neither.
        
        Classification Logic:
        1. Check pattern match (domain knowledge)
        2. Check temporal precedence:
           - Z_t: Mentioned BEFORE nearest treatment decision
           - W_t: Mentioned AFTER most recent treatment
        3. Combine evidence for final classification
        
        Args:
            concept: Extracted concept with timestamp
            treatment_times: List of treatment decision timestamps
            current_time: Reference time (defaults to concept timestamp)
            
        Returns:
            ProxyClassification with role, confidence, and reasoning
        """
        current_time = current_time or concept.timestamp
        
        # Step 1: Pattern matching
        is_treatment_pattern, treatment_match = self._matches_pattern(
            concept.text, self._treatment_regex
        )
        is_outcome_pattern, outcome_match = self._matches_pattern(
            concept.text, self._outcome_regex
        )
        
        # Step 2: Temporal analysis
        temporal_context = self._analyze_temporal_context(
            concept.timestamp, treatment_times
        )
        
        # Step 3: Decision logic
        treatment_score = 0.0
        outcome_score = 0.0
        reasons = []
        
        # Pattern-based evidence
        if is_treatment_pattern:
            treatment_score += 0.5
            reasons.append(f"matches treatment pattern '{treatment_match}'")
        if is_outcome_pattern:
            outcome_score += 0.5
            reasons.append(f"matches outcome pattern '{outcome_match}'")
        
        # Temporal evidence
        if temporal_context['precedes_treatment']:
            treatment_score += 0.3
            hours_before = temporal_context['hours_to_next_treatment']
            reasons.append(f"precedes treatment by {hours_before:.1f}h")
        
        if temporal_context['follows_treatment']:
            outcome_score += 0.3
            hours_after = temporal_context['hours_since_last_treatment']
            reasons.append(f"follows treatment by {hours_after:.1f}h")
        
        # Determine role
        if treatment_score > outcome_score and treatment_score >= 0.5:
            role = 'treatment_proxy'
            confidence = min(treatment_score, 1.0)
        elif outcome_score > treatment_score and outcome_score >= 0.5:
            role = 'outcome_proxy'
            confidence = min(outcome_score, 1.0)
        else:
            role = 'neither'
            confidence = max(1.0 - treatment_score - outcome_score, 0.3)
            reasons.append("insufficient evidence for proxy role")
        
        return ProxyClassification(
            concept=concept,
            role=role,
            confidence=confidence * concept.confidence,  # Factor in extraction confidence
            reason="; ".join(reasons)
        )
    
    def _analyze_temporal_context(self,
                                   concept_time: datetime,
                                   treatment_times: List[datetime]) -> Dict:
        """
        Analyze temporal relationship between concept and treatments.
        
        Args:
            concept_time: When the concept was mentioned
            treatment_times: List of treatment timestamps
            
        Returns:
            Dictionary with temporal analysis results
        """
        if not treatment_times:
            return {
                'precedes_treatment': False,
                'follows_treatment': False,
                'hours_to_next_treatment': float('inf'),
                'hours_since_last_treatment': float('inf'),
            }
        
        # Find nearest treatment times
        hours_to_next = float('inf')
        hours_since_last = float('inf')
        
        for t_time in treatment_times:
            delta_hours = (t_time - concept_time).total_seconds() / 3600
            
            if delta_hours > 0:  # Treatment is in future
                hours_to_next = min(hours_to_next, delta_hours)
            else:  # Treatment is in past
                hours_since_last = min(hours_since_last, -delta_hours)
        
        return {
            'precedes_treatment': hours_to_next <= self.treatment_window,
            'follows_treatment': hours_since_last <= self.outcome_window,
            'hours_to_next_treatment': hours_to_next,
            'hours_since_last_treatment': hours_since_last,
        }
    
    def extract_proxies(self,
                        concepts: List[ExtractedConcept],
                        treatment_times: List[datetime]) -> Tuple[List[ExtractedConcept], 
                                                                    List[ExtractedConcept]]:
        """
        Extract Z_t (treatment proxies) and W_t (outcome proxies) from concepts.
        
        Args:
            concepts: List of extracted concepts
            treatment_times: List of treatment timestamps
            
        Returns:
            Tuple of (treatment_proxies, outcome_proxies)
        """
        Z_t = []  # Treatment proxies
        W_t = []  # Outcome proxies
        
        for concept in concepts:
            classification = self.classify_proxy_role(concept, treatment_times)
            
            if classification.role == 'treatment_proxy':
                Z_t.append(concept)
            elif classification.role == 'outcome_proxy':
                W_t.append(concept)
        
        return Z_t, W_t
    
    def classify_batch(self,
                       concepts: List[ExtractedConcept],
                       treatment_times: List[datetime]) -> List[ProxyClassification]:
        """
        Classify a batch of concepts.
        
        Args:
            concepts: List of extracted concepts
            treatment_times: List of treatment timestamps
            
        Returns:
            List of ProxyClassification results
        """
        return [
            self.classify_proxy_role(concept, treatment_times)
            for concept in concepts
        ]


def create_test_concepts_from_synthetic_data(
    synthetic_data: Dict,
    base_time: datetime = None
) -> Tuple[List[ExtractedConcept], List[datetime]]:
    """
    Create test concepts from synthetic causal data.
    
    Args:
        synthetic_data: Dictionary with 'concepts' and 'treatment_times' keys
        base_time: Base timestamp for temporal grounding
        
    Returns:
        Tuple of (concepts, treatment_times)
    """
    base_time = base_time or datetime(2025, 1, 1, 8, 0, 0)
    
    concepts = []
    for c_data in synthetic_data.get('concepts', []):
        timestamp = base_time + timedelta(hours=c_data.get('hour_offset', 0))
        concept = ExtractedConcept(
            text=c_data['text'],
            concept_id=c_data.get('concept_id', 'unknown'),
            timestamp=timestamp,
            confidence=c_data.get('confidence', 0.9),
            source=c_data.get('source', 'diary')
        )
        concepts.append(concept)
    
    treatment_times = [
        base_time + timedelta(hours=h)
        for h in synthetic_data.get('treatment_hours', [])
    ]
    
    return concepts, treatment_times
