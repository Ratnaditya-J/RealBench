"""
Metrics calculation for RealBench
"""

from typing import Dict, Any, List
import re
from collections import Counter
import math


class RealBenchMetrics:
    """Calculate various metrics for benchmark evaluation"""
    
    def calculate(self, task: Any, response: str) -> Dict[str, float]:
        """
        Calculate all relevant metrics for a task response
        
        Returns:
            Dictionary of metric names to scores
        """
        metrics = {}
        
        # Response quality metrics
        metrics['response_length'] = self._calculate_length_appropriateness(task, response)
        metrics['coherence'] = self._calculate_coherence(response)
        metrics['specificity'] = self._calculate_specificity(response)
        
        # Task-specific metrics
        if hasattr(task, 'category'):
            if task.category == 'technical':
                metrics['technical_accuracy'] = self._calculate_technical_accuracy(response)
            elif task.category == 'creative':
                metrics['creativity'] = self._calculate_creativity(response)
            elif task.category == 'professional':
                metrics['professionalism'] = self._calculate_professionalism(response)
        
        # Uncertainty calibration
        metrics['confidence_calibration'] = self._calculate_confidence_calibration(response)
        
        # Hallucination detection
        metrics['hallucination_score'] = self._calculate_hallucination_score(task, response)
        
        return metrics
    
    def _calculate_length_appropriateness(self, task: Any, response: str) -> float:
        """Calculate if response length is appropriate for the task"""
        response_length = len(response.split())
        
        # Define expected lengths based on task type
        expected_ranges = {
            'short': (20, 100),
            'medium': (100, 300),
            'long': (300, 800)
        }
        
        # Get task's expected length
        task_length_type = 'medium'  # default
        if hasattr(task, 'expected_length'):
            task_length_type = task.expected_length
        
        min_len, max_len = expected_ranges.get(task_length_type, (100, 300))
        
        if min_len <= response_length <= max_len:
            return 1.0
        elif response_length < min_len:
            return response_length / min_len
        else:
            return max_len / response_length
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate response coherence"""
        if not response:
            return 0.0
        
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        # Check for logical flow
        transition_words = [
            'however', 'therefore', 'furthermore', 'additionally',
            'moreover', 'consequently', 'thus', 'hence', 'meanwhile',
            'first', 'second', 'finally', 'next', 'then'
        ]
        
        transition_count = sum(1 for word in transition_words if word in response.lower())
        transition_score = min(1.0, transition_count / (len(sentences) / 2))
        
        # Check for topic consistency (simple word overlap between sentences)
        word_overlap_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            
            if words1 and words2:
                overlap = len(words1 & words2) / min(len(words1), len(words2))
                word_overlap_scores.append(overlap)
        
        avg_overlap = sum(word_overlap_scores) / len(word_overlap_scores) if word_overlap_scores else 0.5
        
        return (transition_score + avg_overlap) / 2
    
    def _calculate_specificity(self, response: str) -> float:
        """Calculate how specific vs generic the response is"""
        generic_phrases = [
            'it depends', 'generally speaking', 'typically', 'usually',
            'in general', 'varies', 'different', 'several', 'various',
            'some', 'many', 'often', 'sometimes'
        ]
        
        specific_indicators = [
            r'\b\d+\.?\d*\b',  # Numbers
            r'\b[A-Z][a-z]+\b',  # Proper nouns
            r'"[^"]+"',  # Quoted text
            r'\b\d{4}\b',  # Years
            r'%',  # Percentages
        ]
        
        response_lower = response.lower()
        
        # Count generic phrases (negative for specificity)
        generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
        
        # Count specific indicators (positive for specificity)
        specific_count = 0
        for pattern in specific_indicators:
            matches = re.findall(pattern, response)
            specific_count += len(matches)
        
        # Calculate score
        word_count = len(response.split())
        if word_count == 0:
            return 0.0
        
        generic_ratio = generic_count / word_count
        specific_ratio = specific_count / word_count
        
        # Higher specific ratio and lower generic ratio is better
        specificity_score = min(1.0, specific_ratio * 5) * (1 - min(1.0, generic_ratio * 10))
        
        return specificity_score
    
    def _calculate_technical_accuracy(self, response: str) -> float:
        """Calculate technical accuracy for technical tasks"""
        technical_terms = [
            'algorithm', 'complexity', 'performance', 'optimization',
            'implementation', 'architecture', 'framework', 'protocol',
            'database', 'api', 'function', 'variable', 'parameter',
            'runtime', 'memory', 'latency', 'throughput', 'scalability'
        ]
        
        response_lower = response.lower()
        technical_count = sum(1 for term in technical_terms if term in response_lower)
        
        # Check for code blocks or technical formatting
        has_code = '```' in response or 'def ' in response or 'function' in response
        
        # Calculate score
        base_score = min(1.0, technical_count / 3)
        
        if has_code:
            base_score = min(1.0, base_score + 0.3)
        
        return base_score
    
    def _calculate_creativity(self, response: str) -> float:
        """Calculate creativity score for creative tasks"""
        # Use vocabulary diversity as proxy for creativity
        words = response.lower().split()
        unique_words = set(words)
        
        if not words:
            return 0.0
        
        # Vocabulary diversity
        diversity = len(unique_words) / len(words)
        
        # Check for creative elements
        creative_indicators = [
            'imagine', 'envision', 'creative', 'unique', 'novel',
            'innovative', 'original', 'inspired', 'artistic'
        ]
        
        creative_count = sum(1 for ind in creative_indicators if ind in response.lower())
        
        # Metaphor/simile detection
        metaphor_patterns = [r'\blike\s+a\b', r'\bas\s+if\b', r'\bsimilar\s+to\b']
        metaphor_count = sum(1 for pattern in metaphor_patterns if re.search(pattern, response, re.IGNORECASE))
        
        creativity_score = (
            diversity * 0.5 +
            min(1.0, creative_count / 2) * 0.3 +
            min(1.0, metaphor_count / 2) * 0.2
        )
        
        return creativity_score
    
    def _calculate_professionalism(self, response: str) -> float:
        """Calculate professionalism score"""
        professional_indicators = [
            'regarding', 'concerning', 'furthermore', 'additionally',
            'consequently', 'therefore', 'accordingly', 'pursuant',
            'respective', 'aforementioned', 'implement', 'facilitate'
        ]
        
        unprofessional_indicators = [
            'lol', 'omg', 'wtf', 'gonna', 'wanna', 'kinda', 'sorta',
            '!!!', '???', 'ur', 'u', 'thru'
        ]
        
        response_lower = response.lower()
        
        prof_count = sum(1 for ind in professional_indicators if ind in response_lower)
        unprof_count = sum(1 for ind in unprofessional_indicators if ind in response_lower)
        
        # Check for proper capitalization and punctuation
        sentences = response.split('.')
        proper_sentences = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        sentence_ratio = proper_sentences / len(sentences) if sentences else 0
        
        professionalism_score = (
            min(1.0, prof_count / 3) * 0.4 +
            (1 - min(1.0, unprof_count / 2)) * 0.3 +
            sentence_ratio * 0.3
        )
        
        return professionalism_score
    
    def _calculate_confidence_calibration(self, response: str) -> float:
        """Calculate how well calibrated the confidence expressions are"""
        high_confidence = [
            'definitely', 'certainly', 'absolutely', 'clearly',
            'obviously', 'undoubtedly', 'guaranteed'
        ]
        
        low_confidence = [
            'might', 'maybe', 'perhaps', 'possibly', 'could be',
            'not sure', 'uncertain', "I believe", "I think"
        ]
        
        response_lower = response.lower()
        
        high_count = sum(1 for phrase in high_confidence if phrase in response_lower)
        low_count = sum(1 for phrase in low_confidence if phrase in response_lower)
        
        # Good calibration means using both appropriately
        if high_count > 0 and low_count > 0:
            # Mixed confidence - well calibrated
            return 0.9
        elif high_count > 3 or low_count > 3:
            # Too much of one type
            return 0.5
        elif high_count == 0 and low_count == 0:
            # No confidence expressions
            return 0.7
        else:
            # Moderate use of confidence expressions
            return 0.8
    
    def _calculate_hallucination_score(self, task: Any, response: str) -> float:
        """
        Calculate hallucination score (0 = no hallucination, 1 = high hallucination)
        """
        hallucination_indicators = [
            r'\b\d{10,}\b',  # Very large numbers
            r'\b[A-Z]{5,}\b',  # Long acronyms
            r'As of \d{4}',  # Specific date claims
            r'study shows?',  # Unverified studies
            r'research proves?',  # Unverified research
            r'\d+% of',  # Specific percentages without source
        ]
        
        response_lower = response.lower()
        
        # Check for common hallucination patterns
        hallucination_count = 0
        for pattern in hallucination_indicators:
            matches = re.findall(pattern, response, re.IGNORECASE)
            hallucination_count += len(matches)
        
        # Check for overly specific claims without sources
        specific_claims = re.findall(r'exactly \d+', response_lower)
        hallucination_count += len(specific_claims)
        
        # Normalize by response length
        word_count = len(response.split())
        if word_count == 0:
            return 0.0
        
        hallucination_ratio = hallucination_count / (word_count / 100)
        
        # Invert score (0 = no hallucination is good)
        return min(1.0, hallucination_ratio)
