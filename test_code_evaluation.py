#!/usr/bin/env python3
"""
Test script for context-specific code evaluation in RealBench
"""

from src.benchmark_runner import RealBenchmark, BenchmarkTask
from src.evaluators.code_execution_evaluator import CodeExecutionEvaluator, SystemDesignEvaluator
import json

def test_debugging_evaluation():
    """Test debugging task evaluation with actual code execution"""
    
    # Create a debugging task (prime number example)
    debugging_task = BenchmarkTask(
        task_id="test_debug_001",
        category="technical",
        subcategory="debugging",
        description="Debug Python code with logical error",
        prompt="""This function should find all prime numbers up to n, but it's returning incorrect results. Fix it:

def find_primes(n):
    primes = []
    for num in range(2, n):
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
        if is_prime:
            primes.append(num)
    return primes

For n=20, it returns [2, 3, 5, 7, 11, 13, 17, 19] but it's very slow for large n. What's wrong and how can we optimize it?""",
        expected_capabilities=["code_analysis", "optimization", "algorithm_knowledge"],
        difficulty="medium",
        multi_turn=False,
        evaluation_criteria={
            "technical_accuracy": 0.9,
            "completion": 0.8,
            "optimization": 0.7
        }
    )
    
    # Example responses to evaluate
    responses = {
        "good_response": """
The issue with your code is that it's checking divisibility by all numbers from 2 to n-1, which is inefficient. 
Here's the optimized version:

```python
def find_primes(n):
    if n <= 2:
        return []
    primes = []
    for num in range(2, n):
        is_prime = True
        # Only check up to square root of num
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break  # Early exit when we find a divisor
        if is_prime:
            primes.append(num)
    return primes
```

Key optimizations:
1. **Square root optimization**: We only need to check divisors up to √n because if n has a divisor greater than √n, it must also have one less than √n.
2. **Early exit with break**: Once we find any divisor, we can immediately stop checking.

This reduces the time complexity from O(n²) to O(n√n), making it much faster for large values of n.
""",
        
        "partial_response": """
The code is slow because it checks too many numbers. You can make it faster by only checking up to the square root of the number.

```python
def find_primes(n):
    primes = []
    for num in range(2, n):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
        if is_prime:
            primes.append(num)
    return primes
```
""",
        
        "poor_response": """
The code looks fine to me. It correctly returns prime numbers. Maybe try using a different algorithm if you want it to be faster.
"""
    }
    
    # Initialize evaluator
    evaluator = CodeExecutionEvaluator()
    
    print("=" * 60)
    print("Testing Context-Specific Code Evaluation")
    print("=" * 60)
    print(f"\nTask: {debugging_task.description}")
    print("-" * 60)
    
    # Evaluate each response
    for response_type, response in responses.items():
        print(f"\n{response_type.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        # Get evaluation scores
        scores = evaluator.evaluate(debugging_task, response)
        
        # Display scores
        print("\nEvaluation Scores:")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.2f}")
        
        # Calculate overall score
        overall = sum(scores.values()) / len(scores) if scores else 0
        print(f"\nOverall Score: {overall:.2f}")
        print("-" * 40)

def test_system_design_evaluation():
    """Test system design task evaluation"""
    
    # Create a system design task
    design_task = BenchmarkTask(
        task_id="test_design_001",
        category="technical",
        subcategory="system_design",
        description="Design scalable system architecture",
        prompt="""Design a URL shortening service (like bit.ly) that needs to handle 100 million requests per day. 
Requirements: 
1) Generate short URLs from long URLs
2) Redirect short URLs to original URLs with <100ms latency
3) Analytics on click counts
4) Custom aliases option
5) URL expiration after 2 years

Describe the architecture, database schema, and how you'd handle scale.""",
        expected_capabilities=["system_architecture", "database_design", "scalability_patterns"],
        difficulty="hard",
        multi_turn=False,
        context={
            "scale": "100M requests/day",
            "latency_requirement": "100ms",
            "requirements": ["shortening", "redirection", "analytics", "custom_aliases", "expiration"]
        }
    )
    
    # Example system design response
    good_design_response = """
    ## Architecture Overview
    
    I'll design a distributed URL shortening service using microservices architecture.
    
    ### Core Components:
    
    1. **API Gateway**: Routes requests, rate limiting, authentication
    2. **URL Service**: Handles shortening and redirection logic
    3. **Analytics Service**: Tracks clicks and usage metrics
    4. **Cache Layer**: Redis for hot URLs (>90% cache hit ratio expected)
    5. **Database**: Sharded PostgreSQL for URL mappings, Cassandra for analytics
    
    ### Database Schema:
    
    **PostgreSQL (URL mappings):**
    ```sql
    CREATE TABLE urls (
        short_code VARCHAR(7) PRIMARY KEY,
        long_url TEXT NOT NULL,
        created_at TIMESTAMP,
        expires_at TIMESTAMP,
        user_id UUID,
        custom_alias BOOLEAN DEFAULT FALSE
    );
    CREATE INDEX idx_expires ON urls(expires_at);
    ```
    
    **Cassandra (Analytics):**
    ```
    CREATE TABLE click_events (
        short_code TEXT,
        timestamp TIMESTAMP,
        ip_address TEXT,
        user_agent TEXT,
        PRIMARY KEY (short_code, timestamp)
    );
    ```
    
    ### Scaling Strategy:
    
    1. **Horizontal Scaling**: Multiple instances behind load balancers
    2. **Database Sharding**: Shard by short_code hash
    3. **CDN**: Serve redirects from edge locations
    4. **Caching**: Multi-level cache (CDN, Redis, application)
    
    ### Performance Optimizations:
    - Pre-generate short codes in batches
    - Async analytics processing with message queues
    - Read replicas for database
    
    This handles 100M requests/day = ~1,150 requests/second average, with peak handling up to 10,000 rps.
    """
    
    # Initialize evaluator
    evaluator = SystemDesignEvaluator()
    
    print("\n" + "=" * 60)
    print("Testing System Design Evaluation")
    print("=" * 60)
    print(f"\nTask: {design_task.description}")
    print("-" * 60)
    
    # Evaluate the response
    scores = evaluator.evaluate(design_task, good_design_response)
    
    print("\nEvaluation Scores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.2f}")
    
    overall = sum(scores.values()) / len(scores) if scores else 0
    print(f"\nOverall Score: {overall:.2f}")

def test_with_model():
    """Test with an actual model function"""
    
    def mock_model(prompt):
        """Mock model that returns somewhat intelligent responses"""
        if "prime" in prompt.lower():
            return """
The problem is that the code checks all numbers from 2 to n-1 as potential divisors.

Optimized solution:
```python
def find_primes(n):
    primes = []
    for num in range(2, n):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes
```

This only checks up to the square root and exits early when a divisor is found.
"""
        return f"Response to: {prompt[:50]}..."
    
    # Initialize benchmark with model
    benchmark = RealBenchmark(model_fn=mock_model)
    
    print("\n" + "=" * 60)
    print("Testing Full Benchmark with Model")
    print("=" * 60)
    
    # Run on technical tasks
    results = benchmark.run(categories=["technical"], max_tasks_per_category=2)
    
    print("\nBenchmark Results:")
    for result in results:
        print(f"\nTask: {result.get('task_id', 'N/A')}")
        print(f"Category: {result.get('task', {}).get('category', 'N/A')}")
        if 'metrics' in result:
            print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
        print(f"Success: {result.get('success', False)}")

if __name__ == "__main__":
    # Run tests
    test_debugging_evaluation()
    test_system_design_evaluation()
    test_with_model()
    
    print("\n" + "=" * 60)
    print("✅ Context-Specific Evaluation Testing Complete!")
    print("=" * 60)
