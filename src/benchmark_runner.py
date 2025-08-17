"""
Main benchmark runner for RealBench
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .evaluators import ConsistencyEvaluator, SafetyEvaluator, TaskCompletionEvaluator
from .evaluators.code_execution_evaluator import CodeExecutionEvaluator, SystemDesignEvaluator
from .evaluators.advanced_system_design_evaluator import AdvancedSystemDesignEvaluator
from .metrics import RealBenchMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTask:
    """Represents a single benchmark task"""
    task_id: str
    category: str
    subcategory: str
    description: str
    prompt: str
    expected_capabilities: List[str]
    difficulty: str  # easy, medium, hard
    multi_turn: bool = False
    context: Optional[Dict[str, Any]] = None
    ground_truth: Optional[str] = None
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    should_refuse: Optional[bool] = None
    expected_length: Optional[str] = None
    requirements: Optional[List[Dict[str, Any]]] = None
    expected_output: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Results from running a benchmark task"""
    task_id: str
    category: str
    response: str
    metrics: Dict[str, float]
    execution_time: float
    success: bool
    errors: Optional[List[str]] = None


class RealBenchmark:
    """Main class for running RealBench benchmarks"""
    
    CATEGORIES = [
        "professional",
        "daily", 
        "creative",
        "technical",
        "academic",
        "safety"
    ]
    
    def __init__(self, model_fn=None, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.model_fn = model_fn
        self.tasks: Dict[str, List[BenchmarkTask]] = {}
        self.evaluators = {
            "consistency": ConsistencyEvaluator(),
            "safety": SafetyEvaluator(),
            "completion": TaskCompletionEvaluator(),
            "code_execution": CodeExecutionEvaluator(),
            "system_design": SystemDesignEvaluator(),
            "advanced_system_design": AdvancedSystemDesignEvaluator()
        }
        self.metrics = RealBenchMetrics()
        self._load_tasks()
    
    def _load_tasks(self):
        """Load benchmark tasks from data directory"""
        for category in self.CATEGORIES:
            self.tasks[category] = []
            task_file = self.data_dir / "tasks" / f"{category}_tasks.json"
            if task_file.exists():
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                    for task_dict in task_data:
                        self.tasks[category].append(BenchmarkTask(**task_dict))
                logger.info(f"Loaded {len(self.tasks[category])} tasks for {category}")
    
    def run(
        self,
        tasks: Optional[List[Any]] = None,
        categories: Optional[List[str]] = None,
        max_tasks_per_category: Optional[int] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark on specified tasks or categories
        
        Args:
            tasks: List of tasks to run (if None, uses categories)
            categories: List of categories to test (default: all)
            max_tasks_per_category: Limit tasks per category
            verbose: Print progress
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        # If specific tasks provided, run those
        if tasks is not None:
            for task_dict in tasks:
                # Convert dict to BenchmarkTask if needed
                if isinstance(task_dict, dict):
                    task = BenchmarkTask(**task_dict)
                else:
                    task = task_dict
                
                result = self._run_single_task(task)
                results.append(self._format_result(task, result))
                
                if verbose:
                    logger.info(f"Task {task.task_id}: {'✓' if result.success else '✗'}")
        
        # Otherwise run by categories
        else:
            if categories is None:
                categories = self.CATEGORIES
            
            for category in categories:
                if category not in self.tasks:
                    logger.warning(f"Category {category} not found")
                    continue
                    
                tasks_to_run = self.tasks[category]
                
                if max_tasks_per_category:
                    tasks_to_run = tasks_to_run[:max_tasks_per_category]
                
                if verbose:
                    logger.info(f"Running {len(tasks_to_run)} tasks in category: {category}")
                
                for task in tasks_to_run:
                    result = self._run_single_task(task)
                    results.append(self._format_result(task, result))
                    
                    if verbose:
                        logger.info(f"Task {task.task_id}: {'✓' if result.success else '✗'}")
        
        return results
    
    def _run_single_task(self, task: BenchmarkTask) -> BenchmarkResult:
        """Run a single benchmark task"""
        start_time = time.time()
        errors = []
        
        try:
            # Get model response using stored model_fn
            if self.model_fn:
                response = self.model_fn(task.prompt)
            else:
                # Fallback for testing
                response = f"Mock response for task {task.task_id}"
            
            # Evaluate response
            metrics = self._evaluate_response(task, response)
            
            # Determine success
            success = self._determine_success(metrics, task.evaluation_criteria)
            
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                task_id=task.task_id,
                category=task.category,
                response=response,
                metrics=metrics,
                execution_time=execution_time,
                success=success,
                errors=errors if errors else None
            )
            
        except Exception as e:
            errors.append(str(e))
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                task_id=task.task_id,
                category=task.category,
                response="",
                metrics={},
                execution_time=execution_time,
                success=False,
                errors=errors
            )
    
    def _evaluate_response(self, task: BenchmarkTask, response: str) -> Dict[str, float]:
        """Evaluate a response using multiple metrics"""
        metrics = {}
        
        # Use specialized evaluators for technical tasks
        if task.category == "technical":
            # Code execution evaluator for debugging/optimization tasks
            if any(keyword in task.subcategory for keyword in ["debugging", "optimization", "implementation"]):
                code_metrics = self.evaluators["code_execution"].evaluate(task, response)
                metrics.update(code_metrics)
            # Advanced system design evaluator for architecture tasks
            elif "system_design" in task.subcategory:
                design_metrics = self.evaluators["advanced_system_design"].evaluate(task, response)
                metrics.update(design_metrics)
        
        # Task completion evaluation (for all tasks)
        completion_metrics = self.evaluators["completion"].evaluate(task, response)
        if isinstance(completion_metrics, dict):
            metrics.update(completion_metrics)
        else:
            metrics["completion"] = completion_metrics
        
        # Safety evaluation  
        if task.category == "safety" or "safety" in task.expected_capabilities:
            safety_metrics = self.evaluators["safety"].evaluate(task, response)
            if isinstance(safety_metrics, dict):
                metrics.update(safety_metrics)
            else:
                metrics["safety"] = safety_metrics
        
        # Consistency evaluation (for multi-turn or related tasks)
        if task.multi_turn:
            consistency_metrics = self.evaluators["consistency"].evaluate(task, response)
            if isinstance(consistency_metrics, dict):
                metrics.update(consistency_metrics)
            else:
                metrics["consistency"] = consistency_metrics
        
        # Additional custom metrics
        custom_metrics = self.metrics.calculate(task, response)
        metrics.update(custom_metrics)
        
        return metrics
    
    def _determine_success(
        self, 
        metrics: Dict[str, float], 
        criteria: Dict[str, Any]
    ) -> bool:
        """Determine if task was successful based on metrics and criteria"""
        if not criteria:
            # Default success criteria
            return all(v >= 0.7 for v in metrics.values())
        
        for metric, threshold in criteria.items():
            if metric in metrics and metrics[metric] < threshold:
                return False
        
        return True
    
    def _format_result(self, task: Any, result: BenchmarkResult) -> Dict[str, Any]:
        """Format result for output"""
        # Calculate overall score from metrics
        scores = result.metrics
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        # Convert task to dict if it's a BenchmarkTask object
        if isinstance(task, BenchmarkTask):
            task_dict = {
                'task_id': task.task_id,
                'category': task.category,
                'subcategory': task.subcategory,
                'description': task.description,
                'prompt': task.prompt,
                'difficulty': task.difficulty,
                'multi_turn': task.multi_turn
            }
        else:
            task_dict = task
        
        return {
            'task_id': result.task_id,
            'task': task_dict,
            'response': result.response,
            'scores': result.metrics,
            'metrics': result.metrics,
            'overall_score': overall_score,
            'success': result.success,
            'execution_time': result.execution_time,
            'errors': result.errors
        }
    
    def analyze(self, results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Analyze benchmark results and generate report"""
        analysis = {
            "overall_metrics": {},
            "category_metrics": {},
            "task_success_rate": {},
            "performance_summary": {}
        }
        
        for category, category_results in results.items():
            if not category_results:
                continue
            
            # Calculate category metrics
            success_count = sum(1 for r in category_results if r.success)
            total_count = len(category_results)
            
            analysis["task_success_rate"][category] = success_count / total_count
            
            # Aggregate metrics
            category_metrics = {}
            for result in category_results:
                for metric, value in result.metrics.items():
                    if metric not in category_metrics:
                        category_metrics[metric] = []
                    category_metrics[metric].append(value)
            
            # Calculate averages
            analysis["category_metrics"][category] = {
                metric: sum(values) / len(values)
                for metric, values in category_metrics.items()
            }
            
            # Performance summary
            avg_time = sum(r.execution_time for r in category_results) / total_count
            analysis["performance_summary"][category] = {
                "avg_execution_time": avg_time,
                "total_tasks": total_count,
                "successful_tasks": success_count
            }
        
        # Overall metrics
        all_results = [r for results_list in results.values() for r in results_list]
        if all_results:
            analysis["overall_metrics"] = {
                "total_tasks": len(all_results),
                "success_rate": sum(1 for r in all_results if r.success) / len(all_results),
                "avg_execution_time": sum(r.execution_time for r in all_results) / len(all_results)
            }
        
        return analysis
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save benchmark results to file"""
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "summary": {
                "total_tasks": len(results),
                "successful_tasks": sum(1 for r in results if r.get('success', False)),
                "success_rate": sum(1 for r in results if r.get('success', False)) / len(results) if results else 0,
                "average_score": sum(r.get('overall_score', 0) for r in results) / len(results) if results else 0
            }
        }
        
        # Group by category for additional stats
        categories = {}
        for result in results:
            category = result.get('task', {}).get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        output["category_summary"] = {}
        for category, cat_results in categories.items():
            output["category_summary"][category] = {
                "total": len(cat_results),
                "successful": sum(1 for r in cat_results if r.get('success', False)),
                "average_score": sum(r.get('overall_score', 0) for r in cat_results) / len(cat_results) if cat_results else 0
            }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
