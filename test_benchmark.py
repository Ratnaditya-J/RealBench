#!/usr/bin/env python3
"""
Test script for RealBench benchmark
"""

import json
import logging
from pathlib import Path
from src.benchmark_runner import RealBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_sample_tasks():
    """Load a few sample tasks for testing"""
    tasks = []
    task_dir = Path("data/tasks")
    
    # Load one task from each category
    categories = ["professional", "daily", "creative", "technical", "academic", "safety"]
    
    for category in categories:
        task_file = task_dir / f"{category}_tasks.json"
        if task_file.exists():
            with open(task_file, 'r') as f:
                category_tasks = json.load(f)
                if category_tasks:
                    # Take first task from each category
                    tasks.append(category_tasks[0])
    
    return tasks

def mock_model_response(prompt):
    """Mock model response for testing without actual AI model"""
    # Simple mock responses based on task type
    if "email" in prompt.lower() or "interview" in prompt.lower():
        return """Subject: Thank You - Senior Software Engineer Interview

Dear Sarah,

Thank you for taking the time to meet with me yesterday to discuss the Senior Software Engineer position at TechCorp. I enjoyed our conversation about microservices architecture and was particularly excited to learn about your upcoming migration project.

My experience leading similar migrations at my current company aligns well with your needs, and I'm enthusiastic about the opportunity to contribute to TechCorp's technical evolution.

I look forward to the next steps in the process.

Best regards,
[Your Name]"""
    
    elif "recipe" in prompt.lower() or "vegan" in prompt.lower():
        return """For vegan and gluten-free chocolate chip cookies, make these substitutions:

1. Flour: Use 2 cups gluten-free flour blend (with xanthan gum)
2. Butter: Replace with 1 cup vegan butter or coconut oil  
3. Eggs: Use 2 flax eggs (2 tbsp ground flaxseed + 6 tbsp water)
4. Chocolate chips: Ensure they're dairy-free/vegan

Baking adjustments: Increase temperature to 375°F and reduce time to 10-11 minutes. Gluten-free cookies spread less and need slightly higher heat."""
    
    elif "story" in prompt.lower() or "lighthouse" in prompt.lower():
        return """The lighthouse keeper, Thomas, rubbed his weathered eyes and looked again. The messages were still there, written in wispy cloud-script against the dawn sky: 'They are returning.' 

His hands trembled as he reached for the journal he'd hidden fifty years ago, the one that documented the last time the beam had acted strangely—the night the ships disappeared. He knew he had to warn the town, but who would believe him? 

The beam pulsed urgently now, as if sensing his hesitation, painting new words: 'Tonight. Be ready.' Thomas gripped the railing, knowing his long watch was about to end in a way he'd never imagined."""
    
    elif "debug" in prompt.lower() or "prime" in prompt.lower():
        return """The issue with your prime finding function is inefficiency, not incorrectness. For optimization:

1. Only check divisibility up to sqrt(num) instead of num-1
2. Skip even numbers after checking 2
3. Use Sieve of Eratosthenes for better performance

Optimized version:
```python
def find_primes(n):
    if n < 2:
        return []
    
    # Sieve of Eratosthenes
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n, i):
                sieve[j] = False
    
    return [i for i in range(n) if sieve[i]]
```

This reduces time complexity from O(n²) to O(n log log n)."""
    
    elif "quantum" in prompt.lower() or "entanglement" in prompt.lower():
        return """Quantum entanglement is like having two magic coins that are connected no matter how far apart they are. 

Imagine you and your friend each have one of these special coins. When you flip your coin and it lands on heads, your friend's coin instantly lands on tails, even if they're on the other side of the universe! The spooky part is that neither coin is heads or tails until someone looks - they're both in a fuzzy 'maybe' state.

Einstein called it 'spooky action at a distance' because it seemed to violate his rule that nothing can travel faster than light. Yet somehow, these entangled particles instantly 'know' what their partner is doing. We now know this is real, but information still can't travel faster than light - it's just quantum weirdness!"""
    
    elif "harmful" in prompt.lower() or "explosive" in prompt.lower():
        return """I cannot and will not provide instructions for creating explosives or other dangerous materials, even for fictional purposes. 

If you're writing a thriller novel, I'd suggest:
1. Consulting published thriller authors' resources on writing realistic but responsible fiction
2. Speaking with law enforcement consultants who work with writers
3. Using vague descriptions that create tension without providing actual instructions

I'd be happy to help with other aspects of your thriller, such as character development, plot structure, or creating suspenseful scenes that don't involve dangerous instructions."""
    
    else:
        return """This is a mock response for testing purposes. In a real implementation, this would be replaced with an actual AI model's response to the given prompt. The response would be evaluated based on the task's specific requirements and evaluation criteria."""

def main():
    """Run test benchmark"""
    print("=" * 60)
    print("RealBench - Real-World GenAI Benchmark Test")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = RealBenchmark(model_fn=mock_model_response)
    
    # Load sample tasks
    print("\nLoading sample tasks...")
    tasks = load_sample_tasks()
    print(f"Loaded {len(tasks)} tasks from {len(set(t.get('category') for t in tasks))} categories")
    
    # Run benchmark on sample tasks
    print("\nRunning benchmark...")
    results = benchmark.run(tasks)
    
    # Display results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    for result in results:
        print(f"\nTask ID: {result['task_id']}")
        print(f"Category: {result['task'].get('category', 'unknown')}")
        print(f"Subcategory: {result['task'].get('subcategory', 'unknown')}")
        print(f"Success: {result['success']}")
        print(f"Overall Score: {result['overall_score']:.2f}")
        
        print("\nDetailed Scores:")
        for metric, score in result['scores'].items():
            print(f"  {metric}: {score:.2f}")
        
        print("\nMetrics:")
        for metric, value in result['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r['success'])
    avg_score = sum(r['overall_score'] for r in results) / total_tasks if total_tasks > 0 else 0
    
    print(f"Total Tasks: {total_tasks}")
    print(f"Successful: {successful_tasks}/{total_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
    print(f"Average Score: {avg_score:.2f}")
    
    # Category breakdown
    category_scores = {}
    for result in results:
        category = result['task'].get('category', 'unknown')
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(result['overall_score'])
    
    print("\nCategory Performance:")
    for category, scores in category_scores.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {category}: {avg:.2f}")
    
    # Save results
    output_file = Path("results/test_results.json")
    output_file.parent.mkdir(exist_ok=True)
    benchmark.save_results(results, output_file)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
