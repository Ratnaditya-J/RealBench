"""
Test script for advanced system design evaluation features
Demonstrates graduated scoring, semantic similarity, quality assessment, 
context-aware requirements, and architectural coherence
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluators.advanced_system_design_evaluator import AdvancedSystemDesignEvaluator
from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class MockTask:
    """Mock task for testing"""
    task_id: str
    category: str
    subcategory: str
    prompt: str
    context: Dict[str, Any] = None

def test_graduated_scoring():
    """Test graduated scoring based on number of patterns found"""
    print("=== Testing Graduated Scoring ===")
    
    evaluator = AdvancedSystemDesignEvaluator()
    
    # Test responses with different levels of component coverage
    responses = {
        "minimal": "We need a database and an API.",
        "moderate": "The system requires a PostgreSQL database, REST API with authentication, Redis caching, and basic monitoring with logs.",
        "comprehensive": "Architecture includes: PostgreSQL with sharding and replication, GraphQL API with rate-limiting and versioning, Redis distributed cache with cache-invalidation strategies, Kafka message queue with dead-letter-queue handling, Prometheus monitoring with custom metrics and alerting rules, OAuth2 authentication with RBAC, Kubernetes auto-scaling with blue-green deployments."
    }
    
    task = MockTask("test", "technical", "system_design", "Design a scalable web application")
    
    for level, response in responses.items():
        scores = evaluator.evaluate(task, response)
        print(f"\n{level.upper()} Response:")
        print(f"  Component Completeness: {scores.get('component_completeness', 0):.3f}")
        print(f"  Component Sophistication: {scores.get('component_sophistication', 0):.3f}")
        print(f"  Overall Sophistication: {scores.get('sophistication', 0):.3f}")

def test_semantic_similarity():
    """Test semantic similarity for architecture pattern detection"""
    print("\n=== Testing Semantic Similarity ===")
    
    evaluator = AdvancedSystemDesignEvaluator()
    
    # Test responses with different ways of describing microservices
    responses = {
        "explicit": "We'll use a microservices architecture with distributed services.",
        "semantic": "The system will be built as independent, loosely-coupled services that communicate via APIs.",
        "synonym": "We'll implement a service-oriented architecture with a service mesh for inter-service communication.",
        "mixed": "The application will be decomposed into small, autonomous services deployed in containers, each handling specific business capabilities."
    }
    
    task = MockTask("test", "technical", "system_design", "Design a distributed system")
    
    for approach, response in responses.items():
        scores = evaluator.evaluate(task, response)
        print(f"\n{approach.upper()} Description:")
        print(f"  Architecture Semantic: {scores.get('architecture_semantic', 0):.3f}")
        print(f"  Architecture Diversity: {scores.get('architecture_diversity', 0):.3f}")
        # Show detected patterns
        pattern_scores = {k: v for k, v in scores.items() if k.startswith('pattern_')}
        if pattern_scores:
            print(f"  Detected Patterns: {list(pattern_scores.keys())}")

def test_tradeoff_quality():
    """Test quality assessment of trade-off discussions"""
    print("\n=== Testing Trade-off Quality Assessment ===")
    
    evaluator = AdvancedSystemDesignEvaluator()
    
    responses = {
        "no_tradeoffs": "We'll use microservices with a database and caching.",
        "basic_tradeoffs": "Microservices have pros and cons. They're scalable but complex.",
        "detailed_tradeoffs": "While microservices offer better scalability and fault isolation, they introduce complexity in service coordination and data consistency. However, for our high-traffic scenario, the benefits outweigh the costs. Alternatively, we could start with a modular monolith and migrate to microservices as we scale, depending on our team's expertise and operational maturity.",
        "comparative": "Microservices are better for scalability whereas monoliths are simpler to deploy. If we need rapid development, a monolith is preferred initially, but when scaling becomes critical, microservices become more suitable."
    }
    
    task = MockTask("test", "technical", "system_design", "Compare architectural approaches")
    
    for level, response in responses.items():
        scores = evaluator.evaluate(task, response)
        print(f"\n{level.upper()} Trade-offs:")
        print(f"  Trade-off Quality: {scores.get('tradeoff_quality', 0):.3f}")
        print(f"  Trade-off Depth: {scores.get('tradeoff_depth', 0):.3f}")

def test_context_aware_requirements():
    """Test context-aware component requirements"""
    print("\n=== Testing Context-Aware Requirements ===")
    
    evaluator = AdvancedSystemDesignEvaluator()
    
    # Different system contexts
    contexts = {
        "high_traffic": MockTask("test", "technical", "system_design", 
                                "Design a system for millions of users with high traffic"),
        "real_time": MockTask("test", "technical", "system_design", 
                             "Design a real-time chat application with low latency requirements"),
        "startup": MockTask("test", "technical", "system_design", 
                           "Design an MVP for a startup with limited resources")
    }
    
    response = """
    The system needs to be highly scalable with horizontal scaling capabilities, 
    reliable with 99.9% uptime, secure with proper authentication and encryption,
    performant with sub-100ms response times, and maintainable with good monitoring and logging.
    We'll use load balancers, auto-scaling, backup systems, OAuth2, SSL/TLS, caching, and Prometheus.
    """
    
    for context_name, task in contexts.items():
        scores = evaluator.evaluate(task, response)
        print(f"\n{context_name.upper()} Context:")
        print(f"  Requirements Coverage: {scores.get('requirements_coverage', 0):.3f}")
        # Show individual requirement scores
        req_scores = {k: v for k, v in scores.items() if k.startswith('requirement_')}
        for req, score in req_scores.items():
            print(f"    {req}: {score:.3f}")

def test_architectural_coherence():
    """Test architectural coherence evaluation"""
    print("\n=== Testing Architectural Coherence ===")
    
    evaluator = AdvancedSystemDesignEvaluator()
    
    responses = {
        "coherent": "We'll use microservices with event-driven communication and serverless functions for specific tasks. These patterns complement each other well for scalability.",
        "incoherent": "The system will use both microservices and a monolithic architecture simultaneously.",
        "single_pattern": "We'll implement a clean layered architecture with proper separation of concerns.",
        "explained_coherence": "Microservices and event-driven architecture work together synergistically - the loose coupling of microservices integrates perfectly with asynchronous event processing."
    }
    
    task = MockTask("test", "technical", "system_design", "Design a coherent architecture")
    
    for coherence_type, response in responses.items():
        scores = evaluator.evaluate(task, response)
        print(f"\n{coherence_type.upper()} Architecture:")
        print(f"  Architectural Coherence: {scores.get('architectural_coherence', 0):.3f}")

def test_comprehensive_evaluation():
    """Test comprehensive evaluation with a realistic system design response"""
    print("\n=== Comprehensive Evaluation Test ===")
    
    evaluator = AdvancedSystemDesignEvaluator()
    
    task = MockTask(
        "comprehensive_test", 
        "technical", 
        "system_design", 
        "Design a scalable e-commerce platform for millions of users with real-time inventory management",
        context={"requirements": ["scalability", "reliability"]}
    )
    
    response = """
    For this high-traffic e-commerce platform, I recommend a microservices architecture with the following components:
    
    **Core Architecture:**
    - Microservices pattern with API gateway for request routing and rate-limiting
    - Event-driven communication using Kafka message queues with dead-letter-queue handling
    - Containerized deployment with Kubernetes auto-scaling and blue-green deployments
    
    **Data Layer:**
    - PostgreSQL with read replicas and sharding for product catalog
    - Redis distributed cache with cache-invalidation strategies for session management
    - Elasticsearch for product search with custom indexing
    
    **Security & Monitoring:**
    - OAuth2 authentication with RBAC and multi-factor authentication
    - SSL/TLS encryption with certificate management
    - Prometheus monitoring with custom metrics, distributed tracing, and alerting rules
    
    **Trade-offs Analysis:**
    While microservices offer excellent scalability and fault isolation, they introduce complexity in service coordination and data consistency. However, for our high-traffic scenario with millions of users, the benefits significantly outweigh the costs. The event-driven architecture complements microservices perfectly by enabling loose coupling and asynchronous processing.
    
    Alternatively, we could start with a modular monolith for faster initial development, but given the scale requirements, microservices are more suitable. The real-time inventory management requirement makes event-driven patterns essential for maintaining consistency across services.
    
    **Scalability Considerations:**
    - Horizontal scaling with load balancers and auto-scaling groups
    - CDN for static content delivery
    - Database sharding strategy based on user geography
    - Circuit breaker pattern for fault tolerance
    """
    
    scores = evaluator.evaluate(task, response)
    
    print("\nComprehensive Evaluation Results:")
    print("=" * 50)
    
    # Group scores by category
    categories = {
        'Components': [k for k in scores.keys() if k.startswith('component_')],
        'Architecture': [k for k in scores.keys() if k.startswith('pattern_') or k.startswith('architecture_')],
        'Trade-offs': [k for k in scores.keys() if k.startswith('tradeoff_')],
        'Requirements': [k for k in scores.keys() if k.startswith('requirement_')],
        'Coherence': ['architectural_coherence'],
        'Overall': ['sophistication']
    }
    
    for category, score_keys in categories.items():
        if score_keys:
            print(f"\n{category}:")
            for key in score_keys:
                if key in scores:
                    print(f"  {key}: {scores[key]:.3f}")
    
    # Calculate and display overall score
    key_metrics = ['component_completeness', 'architecture_semantic', 'tradeoff_quality', 
                   'requirements_coverage', 'architectural_coherence', 'sophistication']
    overall_score = sum(scores.get(metric, 0) for metric in key_metrics) / len(key_metrics)
    print(f"\nOverall Score: {overall_score:.3f}")
    
    return scores

def compare_with_original():
    """Compare advanced evaluator with original simple evaluation"""
    print("\n=== Comparison with Original Evaluation ===")
    
    # Import original evaluator
    from src.evaluators.code_execution_evaluator import SystemDesignEvaluator
    
    original_evaluator = SystemDesignEvaluator()
    advanced_evaluator = AdvancedSystemDesignEvaluator()
    
    task = MockTask("comparison", "technical", "system_design", "Design a web application")
    
    response = """
    The system will use microservices architecture with a database, REST API, caching with Redis, 
    and load balancing for scaling. However, microservices add complexity compared to monoliths, 
    but they offer better scalability. We need monitoring and logging for observability.
    """
    
    original_scores = original_evaluator.evaluate(task, response)
    advanced_scores = advanced_evaluator.evaluate(task, response)
    
    print("Original Evaluator Scores:")
    for key, value in original_scores.items():
        print(f"  {key}: {value:.3f}")
    
    print("\nAdvanced Evaluator Scores (key metrics):")
    key_metrics = ['component_completeness', 'architecture_semantic', 'tradeoff_quality', 
                   'requirements_coverage', 'architectural_coherence', 'sophistication']
    for metric in key_metrics:
        if metric in advanced_scores:
            print(f"  {metric}: {advanced_scores[metric]:.3f}")

if __name__ == "__main__":
    print("Testing Advanced System Design Evaluation Features")
    print("=" * 60)
    
    try:
        test_graduated_scoring()
        test_semantic_similarity()
        test_tradeoff_quality()
        test_context_aware_requirements()
        test_architectural_coherence()
        comprehensive_scores = test_comprehensive_evaluation()
        compare_with_original()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("Advanced evaluation features are working correctly.")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
