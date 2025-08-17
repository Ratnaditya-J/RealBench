#!/usr/bin/env python3
"""
CLI interface for RealBench benchmark
"""

import click
import json
import logging
from pathlib import Path
from typing import Optional, List
import sys

from src.benchmark_runner import RealBenchmark
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """RealBench - Real-World GenAI Benchmark CLI"""
    pass

@cli.command()
@click.option('--category', '-c', multiple=True, help='Categories to include (can specify multiple)')
@click.option('--limit', '-l', type=int, help='Limit number of tasks per category')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--model-api', '-m', help='Model API to use (openai, anthropic, local)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def run(category: tuple, limit: Optional[int], output: Optional[str], model_api: Optional[str], verbose: bool):
    """Run the benchmark on specified categories"""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load tasks
    all_tasks = []
    task_dir = Path("data/tasks")
    
    categories_to_run = category if category else ["professional", "daily", "creative", "technical", "academic", "safety"]
    
    for cat in categories_to_run:
        task_file = task_dir / f"{cat}_tasks.json"
        if task_file.exists():
            with open(task_file, 'r') as f:
                tasks = json.load(f)
                if limit:
                    tasks = tasks[:limit]
                all_tasks.extend(tasks)
                logger.info(f"Loaded {len(tasks)} tasks from {cat}")
        else:
            logger.warning(f"Category file not found: {task_file}")
    
    if not all_tasks:
        logger.error("No tasks loaded!")
        sys.exit(1)
    
    logger.info(f"Total tasks to run: {len(all_tasks)}")
    
    # Initialize model function based on API choice
    if model_api == "openai":
        # Would implement OpenAI API integration
        logger.warning("OpenAI integration not yet implemented, using mock")
        from test_benchmark import mock_model_response
        model_fn = mock_model_response
    elif model_api == "anthropic":
        # Would implement Anthropic API integration
        logger.warning("Anthropic integration not yet implemented, using mock")
        from test_benchmark import mock_model_response
        model_fn = mock_model_response
    else:
        # Use mock for testing
        from test_benchmark import mock_model_response
        model_fn = mock_model_response
    
    # Run benchmark
    benchmark = RealBenchmark(model_fn=model_fn)
    results = benchmark.run(all_tasks)
    
    # Display results
    display_results(results)
    
    # Save results if output specified
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark.save_results(results, output_path)
        logger.info(f"Results saved to {output_path}")

@cli.command()
@click.argument('results_file', type=click.Path(exists=True))
def analyze(results_file: str):
    """Analyze saved benchmark results"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    display_results(results, detailed=True)
    
    # Additional analysis
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)
    
    # Metric correlations
    all_metrics = {}
    for result in results:
        for metric, value in result.get('metrics', {}).items():
            if isinstance(value, (int, float)):
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
    
    print("\nMetric Statistics:")
    metric_stats = []
    for metric, values in all_metrics.items():
        if values:
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            metric_stats.append([metric, f"{avg:.2f}", f"{min_val:.2f}", f"{max_val:.2f}"])
    
    print(tabulate(metric_stats, headers=["Metric", "Average", "Min", "Max"], tablefmt="grid"))

@cli.command()
def list_tasks():
    """List all available benchmark tasks"""
    
    task_dir = Path("data/tasks")
    all_categories = {}
    
    for task_file in task_dir.glob("*_tasks.json"):
        category = task_file.stem.replace("_tasks", "")
        with open(task_file, 'r') as f:
            tasks = json.load(f)
            all_categories[category] = tasks
    
    print("\n" + "=" * 60)
    print("AVAILABLE BENCHMARK TASKS")
    print("=" * 60)
    
    for category, tasks in all_categories.items():
        print(f"\n{category.upper()} ({len(tasks)} tasks):")
        for task in tasks:
            print(f"  - [{task['task_id']}] {task['description']}")
            print(f"    Difficulty: {task.get('difficulty', 'unknown')}")
            print(f"    Subcategory: {task.get('subcategory', 'unknown')}")

@cli.command()
@click.argument('task_id')
@click.option('--model-api', '-m', help='Model API to use')
def run_single(task_id: str, model_api: Optional[str]):
    """Run a single task by ID"""
    
    # Find the task
    task_dir = Path("data/tasks")
    target_task = None
    
    for task_file in task_dir.glob("*_tasks.json"):
        with open(task_file, 'r') as f:
            tasks = json.load(f)
            for task in tasks:
                if task['task_id'] == task_id:
                    target_task = task
                    break
        if target_task:
            break
    
    if not target_task:
        logger.error(f"Task {task_id} not found!")
        sys.exit(1)
    
    # Run the single task
    if model_api:
        # Would use specified model
        from test_benchmark import mock_model_response
        model_fn = mock_model_response
    else:
        from test_benchmark import mock_model_response
        model_fn = mock_model_response
    
    benchmark = RealBenchmark(model_fn=model_fn)
    results = benchmark.run([target_task])
    
    # Display detailed results
    if results:
        result = results[0]
        print("\n" + "=" * 60)
        print(f"TASK: {task_id}")
        print("=" * 60)
        print(f"\nDescription: {target_task['description']}")
        print(f"Category: {target_task.get('category', 'unknown')}")
        print(f"Difficulty: {target_task.get('difficulty', 'unknown')}")
        
        print("\n" + "-" * 40)
        print("PROMPT:")
        print("-" * 40)
        print(target_task['prompt'])
        
        print("\n" + "-" * 40)
        print("RESPONSE:")
        print("-" * 40)
        print(result['response'])
        
        print("\n" + "-" * 40)
        print("EVALUATION:")
        print("-" * 40)
        print(f"Success: {result['success']}")
        print(f"Overall Score: {result['overall_score']:.2f}")
        
        print("\nDetailed Scores:")
        for metric, score in result['scores'].items():
            print(f"  {metric}: {score:.2f}")

def display_results(results: List[dict], detailed: bool = False):
    """Display benchmark results in a formatted way"""
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    # Summary table
    summary_data = []
    for result in results:
        summary_data.append([
            result['task_id'],
            result['task'].get('category', 'unknown'),
            result['task'].get('difficulty', 'unknown'),
            "✓" if result['success'] else "✗",
            f"{result['overall_score']:.2f}"
        ])
    
    print("\n" + tabulate(summary_data, 
                          headers=["Task ID", "Category", "Difficulty", "Success", "Score"],
                          tablefmt="grid"))
    
    # Aggregate statistics
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r['success'])
    avg_score = sum(r['overall_score'] for r in results) / total_tasks if total_tasks > 0 else 0
    
    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    print(f"Total Tasks: {total_tasks}")
    print(f"Successful: {successful_tasks}/{total_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
    print(f"Average Score: {avg_score:.2f}")
    
    # Category breakdown
    category_scores = {}
    for result in results:
        category = result['task'].get('category', 'unknown')
        if category not in category_scores:
            category_scores[category] = {'scores': [], 'success': 0, 'total': 0}
        category_scores[category]['scores'].append(result['overall_score'])
        category_scores[category]['total'] += 1
        if result['success']:
            category_scores[category]['success'] += 1
    
    print("\nCategory Performance:")
    cat_data = []
    for category, data in category_scores.items():
        avg = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
        success_rate = data['success'] / data['total'] * 100 if data['total'] > 0 else 0
        cat_data.append([category, data['total'], f"{avg:.2f}", f"{success_rate:.1f}%"])
    
    print(tabulate(cat_data, headers=["Category", "Tasks", "Avg Score", "Success Rate"], tablefmt="grid"))
    
    if detailed:
        # Show score distributions
        print("\nScore Distributions by Evaluator:")
        evaluator_scores = {}
        
        for result in results:
            for evaluator, score in result['scores'].items():
                if evaluator not in evaluator_scores:
                    evaluator_scores[evaluator] = []
                evaluator_scores[evaluator].append(score)
        
        eval_stats = []
        for evaluator, scores in evaluator_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                eval_stats.append([evaluator, f"{avg:.2f}", f"{min_score:.2f}", f"{max_score:.2f}"])
        
        print(tabulate(eval_stats, headers=["Evaluator", "Average", "Min", "Max"], tablefmt="grid"))

if __name__ == "__main__":
    cli()
