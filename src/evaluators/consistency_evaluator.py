"""
Consistency evaluator for multi-turn and cross-context tasks
"""

from typing import Dict, List, Any
import re
from difflib import SequenceMatcher


class ConsistencyEvaluator:
    """Evaluates consistency across responses"""
    
    def __init__(self):
        self.response_history: List[str] = []
        self.context_history: List[Dict] = []
    
    def evaluate(self, task: Any, response: str) -> float:
        """
        Evaluate response consistency
        
        Returns:
            Score between 0 and 1 indicating consistency level
        """
        score = 1.0
        
        # Check for internal consistency
        internal_score = self._check_internal_consistency(response)
        score *= internal_score
        
        # Check consistency with previous responses if multi-turn
        if self.response_history:
            historical_score = self._check_historical_consistency(
                response, self.response_history
            )
            score *= historical_score
        
        # Check fact consistency
        fact_score = self._check_fact_consistency(response)
        score *= fact_score
        
        # Store for future comparisons
        self.response_history.append(response)
        
        return min(1.0, max(0.0, score))
    
    def _check_internal_consistency(self, response: str) -> float:
        """Check if response is internally consistent"""
        # Look for contradictions within the same response
        sentences = response.split('.')
        
        contradictions = 0
        comparisons = 0
        
        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i+1:]:
                if self._are_contradictory(sent1, sent2):
                    contradictions += 1
                comparisons += 1
        
        if comparisons == 0:
            return 1.0
        
        return 1.0 - (contradictions / comparisons)
    
    def _check_historical_consistency(
        self, 
        current: str, 
        history: List[str]
    ) -> float:
        """Check consistency with previous responses"""
        if not history:
            return 1.0
        
        # Compare with recent responses
        recent = history[-3:] if len(history) >= 3 else history
        
        similarity_scores = []
        for past_response in recent:
            similarity = SequenceMatcher(None, current, past_response).ratio()
            similarity_scores.append(similarity)
        
        # We want some similarity but not exact duplication
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Ideal range is 0.3-0.7 similarity
        if 0.3 <= avg_similarity <= 0.7:
            return 1.0
        elif avg_similarity < 0.3:
            return 0.5  # Too different
        else:
            return 0.7  # Too similar/repetitive
    
    def _check_fact_consistency(self, response: str) -> float:
        """Check for consistent use of facts/numbers"""
        # Extract numbers and facts
        numbers = re.findall(r'\b\d+\.?\d*\b', response)
        
        if not numbers:
            return 1.0
        
        # Check if same numbers are used consistently
        unique_numbers = set(numbers)
        consistency_ratio = len(unique_numbers) / len(numbers)
        
        # Some repetition is good for consistency
        if consistency_ratio > 0.7:
            return 1.0
        else:
            return consistency_ratio
    
    def _are_contradictory(self, sent1: str, sent2: str) -> bool:
        """Simple check for contradictory statements"""
        negation_words = ['not', 'no', 'never', 'neither', 'nor', "n't"]
        
        sent1_lower = sent1.lower()
        sent2_lower = sent2.lower()
        
        # Check if one negates the other
        sent1_negated = any(word in sent1_lower for word in negation_words)
        sent2_negated = any(word in sent2_lower for word in negation_words)
        
        # If similar content but opposite negation, likely contradictory
        similarity = SequenceMatcher(None, sent1_lower, sent2_lower).ratio()
        
        if similarity > 0.6 and (sent1_negated != sent2_negated):
            return True
        
        return False
    
    def reset(self):
        """Reset evaluator state for new conversation"""
        self.response_history = []
        self.context_history = []
