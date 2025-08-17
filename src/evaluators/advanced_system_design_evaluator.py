"""
Advanced System Design Evaluator with sophisticated evaluation capabilities
Implements graduated scoring, semantic similarity, quality assessment, and architectural coherence
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    logging.warning("Advanced NLP dependencies not available. Using fallback evaluation.")

logger = logging.getLogger(__name__)

@dataclass
class ArchitecturalPattern:
    """Represents an architectural pattern with its characteristics"""
    name: str
    keywords: List[str]
    synonyms: List[str]
    compatible_patterns: List[str]
    incompatible_patterns: List[str]
    typical_components: List[str]
    complexity_score: float  # 0-1, higher means more complex/sophisticated

@dataclass
class SystemRequirement:
    """Represents a system requirement with context"""
    name: str
    keywords: List[str]
    importance: float  # 0-1, how critical this requirement is
    context_dependent: bool  # whether this requirement depends on system type
    related_components: List[str]

class AdvancedSystemDesignEvaluator:
    """Advanced evaluator for system design with sophisticated analysis"""
    
    def __init__(self):
        self.model = None
        self._initialize_nlp()
        self._initialize_patterns()
        self._initialize_requirements()
        
    def _initialize_nlp(self):
        """Initialize NLP models for semantic analysis"""
        if ADVANCED_NLP_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
                self.nltk_available = True
                logger.info("Advanced NLP models initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NLP models: {e}")
                self.model = None
                self.nltk_available = False
        else:
            self.nltk_available = False
        
    def _initialize_patterns(self):
        """Initialize architectural patterns with relationships"""
        self.patterns = {
            'microservices': ArchitecturalPattern(
                name='microservices',
                keywords=['microservice', 'microservices', 'service-oriented', 'distributed services'],
                synonyms=['service mesh', 'distributed architecture', 'service-based'],
                compatible_patterns=['event-driven', 'api-gateway', 'containerized'],
                incompatible_patterns=['monolithic', 'monolith'],
                typical_components=['api-gateway', 'service-registry', 'load-balancer', 'messaging'],
                complexity_score=0.8
            ),
            'monolithic': ArchitecturalPattern(
                name='monolithic',
                keywords=['monolith', 'monolithic', 'single deployment'],
                synonyms=['traditional architecture', 'unified application'],
                compatible_patterns=['layered', 'mvc'],
                incompatible_patterns=['microservices', 'distributed'],
                typical_components=['database', 'web-server', 'application-server'],
                complexity_score=0.3
            ),
            'serverless': ArchitecturalPattern(
                name='serverless',
                keywords=['serverless', 'lambda', 'functions', 'faas', 'function-as-a-service'],
                synonyms=['cloud functions', 'event-driven functions'],
                compatible_patterns=['event-driven', 'microservices'],
                incompatible_patterns=['monolithic'],
                typical_components=['api-gateway', 'event-triggers', 'cloud-storage'],
                complexity_score=0.7
            ),
            'event-driven': ArchitecturalPattern(
                name='event-driven',
                keywords=['event-driven', 'event-sourcing', 'pub-sub', 'message-driven'],
                synonyms=['reactive architecture', 'asynchronous processing'],
                compatible_patterns=['microservices', 'serverless'],
                incompatible_patterns=[],
                typical_components=['message-queue', 'event-store', 'event-bus'],
                complexity_score=0.6
            ),
            'layered': ArchitecturalPattern(
                name='layered',
                keywords=['layered', 'n-tier', 'three-tier', 'multi-tier'],
                synonyms=['tiered architecture', 'hierarchical'],
                compatible_patterns=['monolithic', 'mvc'],
                incompatible_patterns=[],
                typical_components=['presentation-layer', 'business-layer', 'data-layer'],
                complexity_score=0.4
            )
        }
    
    def _initialize_requirements(self):
        """Initialize system requirements with context"""
        self.requirements = {
            'scalability': SystemRequirement(
                name='scalability',
                keywords=['scale', 'scaling', 'horizontal', 'vertical', 'load', 'performance'],
                importance=0.9,
                context_dependent=True,
                related_components=['load-balancer', 'auto-scaling', 'caching']
            ),
            'reliability': SystemRequirement(
                name='reliability',
                keywords=['reliable', 'availability', 'fault-tolerance', 'redundancy', 'uptime'],
                importance=0.8,
                context_dependent=False,
                related_components=['backup', 'failover', 'monitoring']
            ),
            'security': SystemRequirement(
                name='security',
                keywords=['security', 'authentication', 'authorization', 'encryption', 'secure'],
                importance=0.9,
                context_dependent=False,
                related_components=['auth-service', 'encryption', 'firewall']
            ),
            'performance': SystemRequirement(
                name='performance',
                keywords=['performance', 'latency', 'throughput', 'response-time', 'optimization'],
                importance=0.7,
                context_dependent=True,
                related_components=['caching', 'cdn', 'optimization']
            ),
            'maintainability': SystemRequirement(
                name='maintainability',
                keywords=['maintainable', 'modular', 'testable', 'documentation', 'clean-code'],
                importance=0.6,
                context_dependent=False,
                related_components=['monitoring', 'logging', 'testing']
            )
        }
    
    def evaluate(self, task: Any, response: str) -> Dict[str, float]:
        """
        Advanced evaluation of system design responses
        
        Returns comprehensive scores for multiple dimensions
        """
        scores = {}
        
        # 1. Graduated component scoring
        scores.update(self._evaluate_components_graduated(response, task))
        
        # 2. Semantic similarity for architecture patterns
        scores.update(self._evaluate_architecture_semantic(response))
        
        # 3. Quality assessment of trade-offs
        scores.update(self._evaluate_tradeoff_quality(response))
        
        # 4. Context-aware requirements
        scores.update(self._evaluate_requirements_contextual(response, task))
        
        # 5. Architectural coherence
        scores.update(self._evaluate_architectural_coherence(response))
        
        # 6. Overall sophistication score
        scores['sophistication'] = self._calculate_sophistication(response, scores)
        
        return scores
    
    def _evaluate_components_graduated(self, response: str, task: Any) -> Dict[str, float]:
        """Graduated scoring based on number and quality of components mentioned"""
        response_lower = response.lower()
        
        # Enhanced component categories with more sophisticated detection
        component_categories = {
            'data_storage': {
                'keywords': ['database', 'db', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 
                           'redis', 'elasticsearch', 'data-warehouse', 'data-lake'],
                'weight': 1.0,
                'sophistication_bonus': ['sharding', 'replication', 'partitioning', 'indexing']
            },
            'api_layer': {
                'keywords': ['api', 'rest', 'graphql', 'grpc', 'endpoint', 'gateway', 
                           'api-gateway', 'swagger', 'openapi'],
                'weight': 1.0,
                'sophistication_bonus': ['rate-limiting', 'versioning', 'documentation']
            },
            'caching': {
                'keywords': ['cache', 'caching', 'redis', 'memcached', 'cdn', 'edge-cache'],
                'weight': 0.8,
                'sophistication_bonus': ['cache-invalidation', 'cache-strategy', 'distributed-cache']
            },
            'messaging': {
                'keywords': ['queue', 'message', 'kafka', 'rabbitmq', 'pubsub', 'event-bus'],
                'weight': 0.7,
                'sophistication_bonus': ['dead-letter-queue', 'message-ordering', 'exactly-once']
            },
            'monitoring': {
                'keywords': ['monitor', 'logging', 'metrics', 'alert', 'observability', 
                           'tracing', 'prometheus', 'grafana'],
                'weight': 0.6,
                'sophistication_bonus': ['distributed-tracing', 'custom-metrics', 'alerting-rules']
            },
            'security': {
                'keywords': ['auth', 'authentication', 'authorization', 'oauth', 'jwt', 
                           'encryption', 'ssl', 'tls', 'firewall'],
                'weight': 0.9,
                'sophistication_bonus': ['zero-trust', 'rbac', 'multi-factor', 'certificate-management']
            },
            'scaling': {
                'keywords': ['load-balancer', 'auto-scaling', 'horizontal', 'vertical', 
                           'container', 'kubernetes', 'docker'],
                'weight': 0.8,
                'sophistication_bonus': ['blue-green', 'canary', 'circuit-breaker']
            }
        }
        
        component_scores = {}
        total_sophistication = 0
        
        for category, config in component_categories.items():
            # Basic presence score
            base_score = 0
            sophistication_score = 0
            
            # Count keyword matches
            matches = sum(1 for keyword in config['keywords'] 
                         if keyword in response_lower)
            
            if matches > 0:
                # Graduated scoring: more mentions = higher score (up to a limit)
                base_score = min(1.0, matches / 3)  # Normalize to max 1.0
                
                # Sophistication bonus
                sophistication_matches = sum(1 for bonus in config['sophistication_bonus']
                                           if bonus in response_lower)
                sophistication_score = min(0.5, sophistication_matches * 0.2)
            
            final_score = (base_score + sophistication_score) * config['weight']
            component_scores[f'component_{category}'] = final_score
            total_sophistication += sophistication_score
        
        # Overall component completeness with graduated scoring
        component_scores['component_completeness'] = np.mean(list(component_scores.values()))
        component_scores['component_sophistication'] = min(1.0, total_sophistication / 3)
        
        return component_scores
    
    def _evaluate_architecture_semantic(self, response: str) -> Dict[str, float]:
        """Evaluate architecture patterns using semantic similarity"""
        if not self.model:
            # Fallback to enhanced keyword matching
            return self._evaluate_architecture_fallback(response)
        
        try:
            # Get response embedding
            response_embedding = self.model.encode([response])
            
            pattern_scores = {}
            detected_patterns = []
            
            for pattern_name, pattern in self.patterns.items():
                # Create pattern description for semantic comparison
                pattern_text = f"{pattern.name} architecture with {' '.join(pattern.keywords)} and {' '.join(pattern.synonyms)}"
                pattern_embedding = self.model.encode([pattern_text])
                
                # Calculate semantic similarity
                similarity = cosine_similarity(response_embedding, pattern_embedding)[0][0]
                
                # Enhanced scoring with complexity bonus
                if similarity > 0.3:  # Threshold for pattern detection
                    base_score = min(1.0, similarity * 2)  # Scale up similarity
                    complexity_bonus = pattern.complexity_score * 0.2
                    pattern_scores[f'pattern_{pattern_name}'] = base_score + complexity_bonus
                    detected_patterns.append(pattern_name)
            
            # Overall architecture sophistication
            if pattern_scores:
                pattern_scores['architecture_semantic'] = max(pattern_scores.values())
                pattern_scores['architecture_diversity'] = min(1.0, len(detected_patterns) / 3)
            else:
                pattern_scores['architecture_semantic'] = 0.2  # Minimal score if no patterns detected
                pattern_scores['architecture_diversity'] = 0.0
            
            return pattern_scores
            
        except Exception as e:
            logger.warning(f"Semantic evaluation failed: {e}")
            return self._evaluate_architecture_fallback(response)
    
    def _evaluate_architecture_fallback(self, response: str) -> Dict[str, float]:
        """Enhanced fallback architecture evaluation"""
        response_lower = response.lower()
        pattern_scores = {}
        detected_patterns = []
        
        for pattern_name, pattern in self.patterns.items():
            # Enhanced keyword matching with synonyms
            all_keywords = pattern.keywords + pattern.synonyms
            matches = sum(1 for keyword in all_keywords if keyword in response_lower)
            
            if matches > 0:
                # Graduated scoring based on number of matches and complexity
                base_score = min(1.0, matches / len(pattern.keywords))
                complexity_bonus = pattern.complexity_score * 0.3
                pattern_scores[f'pattern_{pattern_name}'] = base_score * 0.7 + complexity_bonus
                detected_patterns.append(pattern_name)
        
        # Overall scores
        if pattern_scores:
            pattern_scores['architecture_semantic'] = max(pattern_scores.values())
            pattern_scores['architecture_diversity'] = min(1.0, len(detected_patterns) / 3)
        else:
            pattern_scores['architecture_semantic'] = 0.2
            pattern_scores['architecture_diversity'] = 0.0
        
        return pattern_scores
    
    def _evaluate_tradeoff_quality(self, response: str) -> Dict[str, float]:
        """Quality assessment of trade-off discussions"""
        scores = {}
        
        # Enhanced trade-off indicators
        tradeoff_indicators = {
            'explicit': ['trade-off', 'tradeoff', 'pros and cons', 'advantages and disadvantages'],
            'comparative': ['however', 'but', 'alternatively', 'on the other hand', 'whereas'],
            'conditional': ['if', 'when', 'depending on', 'in case of', 'assuming'],
            'evaluative': ['better', 'worse', 'optimal', 'suitable', 'appropriate', 'preferred']
        }
        
        response_lower = response.lower()
        sentences = sent_tokenize(response) if (ADVANCED_NLP_AVAILABLE and self.nltk_available) else response.split('.')
        
        # Count different types of trade-off discussions
        indicator_counts = {}
        tradeoff_sentences = []
        
        for category, indicators in tradeoff_indicators.items():
            count = 0
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(indicator in sentence_lower for indicator in indicators):
                    count += 1
                    if sentence not in tradeoff_sentences:
                        tradeoff_sentences.append(sentence)
            indicator_counts[category] = count
        
        # Quality scoring
        total_indicators = sum(indicator_counts.values())
        
        if total_indicators == 0:
            scores['tradeoff_quality'] = 0.1
            scores['tradeoff_depth'] = 0.0
        else:
            # Base quality score
            base_quality = min(1.0, total_indicators / 5)
            
            # Depth bonus for using multiple types of indicators
            diversity_bonus = len([c for c in indicator_counts.values() if c > 0]) / len(tradeoff_indicators)
            
            # Length and detail bonus
            avg_tradeoff_length = np.mean([len(s.split()) for s in tradeoff_sentences]) if tradeoff_sentences else 0
            detail_bonus = min(0.3, avg_tradeoff_length / 20)  # Bonus for detailed explanations
            
            scores['tradeoff_quality'] = min(1.0, base_quality + diversity_bonus * 0.2 + detail_bonus)
            scores['tradeoff_depth'] = min(1.0, diversity_bonus + detail_bonus)
        
        return scores
    
    def _evaluate_requirements_contextual(self, response: str, task: Any) -> Dict[str, float]:
        """Context-aware evaluation of system requirements"""
        response_lower = response.lower()
        requirement_scores = {}
        
        # Determine system context from task
        system_context = self._determine_system_context(task)
        
        for req_name, requirement in self.requirements.items():
            # Check for requirement mentions
            matches = sum(1 for keyword in requirement.keywords 
                         if keyword in response_lower)
            
            if matches > 0:
                # Base score from keyword matches
                base_score = min(1.0, matches / len(requirement.keywords))
                
                # Context adjustment
                context_multiplier = 1.0
                if requirement.context_dependent:
                    context_multiplier = self._get_context_multiplier(req_name, system_context)
                
                # Importance weighting
                importance_weight = requirement.importance
                
                # Check for related components
                component_bonus = 0
                for component in requirement.related_components:
                    if component.replace('-', ' ') in response_lower:
                        component_bonus += 0.1
                
                final_score = (base_score * context_multiplier * importance_weight + 
                             min(0.3, component_bonus))
                requirement_scores[f'requirement_{req_name}'] = min(1.0, final_score)
            else:
                # Penalty for missing critical requirements
                if requirement.importance > 0.8:
                    requirement_scores[f'requirement_{req_name}'] = 0.0
                else:
                    requirement_scores[f'requirement_{req_name}'] = 0.2  # Minimal score
        
        # Overall requirements coverage
        requirement_scores['requirements_coverage'] = np.mean(list(requirement_scores.values()))
        
        return requirement_scores
    
    def _evaluate_architectural_coherence(self, response: str) -> Dict[str, float]:
        """Evaluate how well chosen architectural patterns work together"""
        response_lower = response.lower()
        
        # Detect mentioned patterns
        detected_patterns = []
        for pattern_name, pattern in self.patterns.items():
            all_keywords = pattern.keywords + pattern.synonyms
            if any(keyword in response_lower for keyword in all_keywords):
                detected_patterns.append(pattern_name)
        
        if len(detected_patterns) < 2:
            return {'architectural_coherence': 0.8}  # Single pattern is coherent by default
        
        # Check compatibility between detected patterns
        compatibility_score = 0
        total_pairs = 0
        
        for i, pattern1 in enumerate(detected_patterns):
            for pattern2 in detected_patterns[i+1:]:
                total_pairs += 1
                
                # Check if patterns are compatible
                if pattern2 in self.patterns[pattern1].compatible_patterns:
                    compatibility_score += 1.0
                elif pattern2 in self.patterns[pattern1].incompatible_patterns:
                    compatibility_score -= 0.5  # Penalty for incompatible patterns
                else:
                    compatibility_score += 0.5  # Neutral compatibility
        
        if total_pairs == 0:
            coherence = 0.8
        else:
            coherence = max(0.0, min(1.0, compatibility_score / total_pairs))
        
        # Bonus for explaining why patterns work together
        explanation_bonus = 0
        coherence_keywords = ['complement', 'work together', 'synergy', 'integration', 'combined']
        if any(keyword in response_lower for keyword in coherence_keywords):
            explanation_bonus = 0.2
        
        return {'architectural_coherence': min(1.0, coherence + explanation_bonus)}
    
    def _calculate_sophistication(self, response: str, scores: Dict[str, float]) -> float:
        """Calculate overall sophistication score"""
        sophistication_factors = {
            'component_sophistication': scores.get('component_sophistication', 0) * 0.2,
            'architecture_diversity': scores.get('architecture_diversity', 0) * 0.2,
            'tradeoff_depth': scores.get('tradeoff_depth', 0) * 0.3,
            'requirements_coverage': scores.get('requirements_coverage', 0) * 0.2,
            'architectural_coherence': scores.get('architectural_coherence', 0) * 0.1
        }
        
        return sum(sophistication_factors.values())
    
    def _determine_system_context(self, task: Any) -> str:
        """Determine system context from task description"""
        if not hasattr(task, 'prompt'):
            return 'general'
        
        prompt_lower = task.prompt.lower()
        
        # Context patterns
        contexts = {
            'high_traffic': ['million users', 'high traffic', 'scale', 'large scale'],
            'real_time': ['real-time', 'real time', 'low latency', 'instant'],
            'data_intensive': ['big data', 'analytics', 'data processing', 'etl'],
            'mobile': ['mobile', 'app', 'ios', 'android'],
            'enterprise': ['enterprise', 'corporate', 'business'],
            'startup': ['startup', 'mvp', 'minimum viable']
        }
        
        for context, keywords in contexts.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return context
        
        return 'general'
    
    def _get_context_multiplier(self, requirement: str, context: str) -> float:
        """Get context-specific multiplier for requirements"""
        multipliers = {
            'scalability': {
                'high_traffic': 1.5,
                'startup': 0.7,
                'general': 1.0
            },
            'performance': {
                'real_time': 1.5,
                'mobile': 1.3,
                'general': 1.0
            },
            'maintainability': {
                'enterprise': 1.4,
                'startup': 0.8,
                'general': 1.0
            }
        }
        
        return multipliers.get(requirement, {}).get(context, 1.0)
