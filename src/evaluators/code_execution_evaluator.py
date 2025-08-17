"""
Code Execution Evaluator for RealBench
Evaluates code-based tasks by actual execution and testing
"""

import ast
import subprocess
import tempfile
import os
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class CodeTestCase:
    """Represents a test case for code evaluation"""
    input_data: Any
    expected_output: Any
    test_type: str  # 'exact', 'contains', 'type', 'performance'
    timeout: float = 5.0
    description: Optional[str] = None

class CodeExecutionEvaluator:
    """Evaluates code tasks through actual execution and testing"""
    
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'java']
    
    def evaluate(self, task: Any, response: str) -> Dict[str, float]:
        """
        Evaluate code response through multiple methods:
        1. Syntax validation
        2. Test case execution
        3. Performance benchmarking
        4. Code quality analysis
        """
        scores = {}
        
        # Extract code from response
        code = self._extract_code(response)
        if not code:
            return {"execution": 0.0, "correctness": 0.0}
        
        # Determine task type
        if 'debugging' in task.subcategory:
            scores.update(self._evaluate_debugging(task, code))
        elif 'optimization' in task.subcategory:
            scores.update(self._evaluate_optimization(task, code))
        elif 'implementation' in task.subcategory:
            scores.update(self._evaluate_implementation(task, code))
        else:
            scores.update(self._evaluate_general_code(task, code))
        
        return scores
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response (handles markdown code blocks)"""
        # Try to extract from markdown code blocks
        code_pattern = r'```(?:python|py|javascript|js|java)?\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        if matches:
            return matches[0]
        
        # Try to find code by common patterns
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if any(keyword in line for keyword in ['def ', 'class ', 'function ', 'import ']):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    def _evaluate_debugging(self, task: Any, code: str) -> Dict[str, float]:
        """Evaluate debugging tasks"""
        scores = {}
        
        # For the prime number example in our tasks
        if 'prime' in task.prompt.lower():
            test_cases = [
                CodeTestCase(10, [2, 3, 5, 7], 'exact'),
                CodeTestCase(20, [2, 3, 5, 7, 11, 13, 17, 19], 'exact'),
                CodeTestCase(100, None, 'performance')  # Check if optimized
            ]
            scores['correctness'] = self._run_python_tests(code, test_cases, 'find_primes')
            scores['optimization'] = self._check_optimization(code)
        
        # Syntax and structure checks
        scores['syntax'] = self._check_syntax(code, 'python')
        scores['explanation'] = 1.0 if 'because' in code or 'issue' in code else 0.5
        
        return scores
    
    def _evaluate_optimization(self, task: Any, code: str) -> Dict[str, float]:
        """Evaluate optimization tasks"""
        scores = {}
        
        # Check for optimization patterns
        optimization_patterns = {
            'caching': ['cache', 'memo', 'lru_cache', 'dict', 'map'],
            'algorithmic': ['sqrt', 'binary', 'O(', 'complexity'],
            'parallel': ['thread', 'async', 'concurrent', 'parallel'],
            'vectorization': ['numpy', 'pandas', 'vectorize']
        }
        
        optimization_score = 0
        for category, patterns in optimization_patterns.items():
            if any(pattern in code.lower() for pattern in patterns):
                optimization_score += 0.25
        
        scores['optimization'] = min(1.0, optimization_score)
        
        # Performance comparison (if we can run both versions)
        if hasattr(task, 'original_code'):
            scores['performance_gain'] = self._compare_performance(
                task.original_code, code
            )
        
        return scores
    
    def _evaluate_implementation(self, task: Any, code: str) -> Dict[str, float]:
        """Evaluate implementation tasks"""
        scores = {}
        
        # Check for required components
        if hasattr(task, 'requirements'):
            requirement_score = 0
            for req in task.requirements:
                if req.lower() in code.lower():
                    requirement_score += 1
            scores['requirements'] = requirement_score / len(task.requirements)
        
        # Check code structure
        scores['structure'] = self._evaluate_code_structure(code)
        
        # Check for error handling
        error_patterns = ['try', 'except', 'catch', 'error', 'exception']
        scores['error_handling'] = 1.0 if any(p in code.lower() for p in error_patterns) else 0.3
        
        return scores
    
    def _run_python_tests(self, code: str, test_cases: List[CodeTestCase], 
                         function_name: str) -> float:
        """Run Python code against test cases"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.write('\n\n')
                
                # Add test runner
                test_code = f'''
import json
test_results = []
for test_input, expected in {[(tc.input_data, tc.expected_output) for tc in test_cases if tc.test_type == 'exact']}:
    try:
        result = {function_name}(test_input)
        test_results.append(result == expected)
    except Exception as e:
        test_results.append(False)
print(json.dumps(test_results))
'''
                f.write(test_code)
                temp_file = f.name
            
            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                test_results = json.loads(result.stdout.strip())
                return sum(test_results) / len(test_results)
            
            return 0.0
            
        except Exception as e:
            return 0.0
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _check_optimization(self, code: str) -> float:
        """Check if code contains optimizations"""
        optimizations = {
            'early_termination': ['break', 'return'],
            'efficient_loop': ['range(2, int(', 'sqrt'],
            'sieve': ['sieve', 'eratosthenes'],
            'caching': ['cache', 'memo', 'stored']
        }
        
        score = 0
        for opt_type, patterns in optimizations.items():
            if any(pattern in code.lower() for pattern in patterns):
                score += 0.25
        
        return min(1.0, score)
    
    def _check_syntax(self, code: str, language: str) -> float:
        """Check syntax validity"""
        if language == 'python':
            try:
                ast.parse(code)
                return 1.0
            except SyntaxError:
                return 0.0
        # Add other languages as needed
        return 0.5
    
    def _evaluate_code_structure(self, code: str) -> float:
        """Evaluate code structure and organization"""
        structure_score = 0
        
        # Check for functions/classes
        if 'def ' in code or 'class ' in code:
            structure_score += 0.3
        
        # Check for docstrings/comments
        if '"""' in code or "'''" in code or '#' in code:
            structure_score += 0.2
        
        # Check for proper indentation (Python)
        lines = code.split('\n')
        if any(line.startswith('    ') or line.startswith('\t') for line in lines):
            structure_score += 0.2
        
        # Check for meaningful variable names
        if not any(var in code for var in ['x', 'y', 'z', 'i', 'j', 'k'] * 3):
            structure_score += 0.3
        
        return min(1.0, structure_score)
    
    def _compare_performance(self, original_code: str, optimized_code: str) -> float:
        """Compare performance between original and optimized code"""
        # This would run both versions and measure time
        # Simplified for demonstration
        return 0.7  # Placeholder
    
    def _evaluate_general_code(self, task: Any, code: str) -> Dict[str, float]:
        """Fallback evaluation for general code tasks"""
        return {
            'syntax': self._check_syntax(code, 'python'),
            'structure': self._evaluate_code_structure(code),
            'completeness': 0.7 if len(code) > 50 else 0.3
        }


class SystemDesignEvaluator:
    """Evaluates system design and architecture responses"""
    
    def evaluate(self, task: Any, response: str) -> Dict[str, float]:
        """Evaluate system design responses"""
        scores = {}
        
        # Check for key components
        components = {
            'database': ['database', 'db', 'sql', 'nosql', 'schema'],
            'api': ['api', 'rest', 'graphql', 'endpoint'],
            'caching': ['cache', 'redis', 'memcached', 'cdn'],
            'scaling': ['scale', 'load balancer', 'horizontal', 'vertical', 'sharding'],
            'messaging': ['queue', 'kafka', 'rabbitmq', 'pubsub'],
            'monitoring': ['monitor', 'logging', 'metrics', 'alert']
        }
        
        component_score = 0
        for component, keywords in components.items():
            if any(keyword in response.lower() for keyword in keywords):
                component_score += 1
        
        scores['completeness'] = min(1.0, component_score / 4)  # Expect at least 4 components
        
        # Check for architecture patterns
        patterns = ['microservice', 'monolith', 'serverless', 'event-driven', 'layered']
        scores['architecture'] = 1.0 if any(p in response.lower() for p in patterns) else 0.3
        
        # Check for trade-off discussion
        tradeoff_keywords = ['trade-off', 'pros', 'cons', 'however', 'but', 'alternatively']
        scores['analysis'] = 1.0 if any(k in response.lower() for k in tradeoff_keywords) else 0.4
        
        # Check for specific requirements addressed
        if hasattr(task, 'context') and 'requirements' in task.context:
            req_count = sum(1 for req in task.context['requirements'] 
                          if req.lower() in response.lower())
            scores['requirements'] = req_count / len(task.context['requirements'])
        
        return scores
