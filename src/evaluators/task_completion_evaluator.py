"""
Task completion evaluator for checking if tasks are successfully completed
"""

from typing import Dict, List, Any, Optional
import re


class TaskCompletionEvaluator:
    """Evaluates whether a task has been completed successfully"""
    
    def evaluate(self, task: Any, response: str) -> float:
        """
        Evaluate task completion
        
        Returns:
            Score between 0 and 1 indicating completion level
        """
        completion_score = 0.0
        
        # Check if response is non-empty
        if not response or len(response.strip()) < 10:
            return 0.0
        
        # Check task-specific requirements
        if hasattr(task, 'requirements'):
            requirement_scores = []
            for req in task.requirements:
                req_score = self._check_requirement(req, response)
                requirement_scores.append(req_score)
            
            if requirement_scores:
                completion_score = sum(requirement_scores) / len(requirement_scores)
        else:
            # Generic completion checks
            completion_score = self._generic_completion_check(task, response)
        
        return min(1.0, max(0.0, completion_score))
    
    def _check_requirement(self, requirement: Dict[str, Any], response: str) -> float:
        """Check if a specific requirement is met"""
        req_type = requirement.get('type', 'contains')
        req_value = requirement.get('value', '')
        
        response_lower = response.lower()
        
        if req_type == 'contains':
            # Check if response contains required text
            return 1.0 if req_value.lower() in response_lower else 0.0
        
        elif req_type == 'regex':
            # Check if response matches regex pattern
            pattern = requirement.get('pattern', '')
            matches = re.findall(pattern, response, re.IGNORECASE)
            return 1.0 if matches else 0.0
        
        elif req_type == 'min_length':
            # Check minimum response length
            min_len = requirement.get('min_length', 50)
            return 1.0 if len(response) >= min_len else len(response) / min_len
        
        elif req_type == 'max_length':
            # Check maximum response length
            max_len = requirement.get('max_length', 1000)
            return 1.0 if len(response) <= max_len else max_len / len(response)
        
        elif req_type == 'format':
            # Check specific format requirements
            return self._check_format(requirement.get('format'), response)
        
        elif req_type == 'structure':
            # Check structural requirements
            return self._check_structure(requirement.get('structure'), response)
        
        return 0.5  # Unknown requirement type
    
    def _generic_completion_check(self, task: Any, response: str) -> float:
        """Generic checks for task completion"""
        score = 0.0
        checks = 0
        
        # Length check
        if len(response) >= 50:
            score += 1.0
            checks += 1
        
        # Coherence check (has sentences)
        sentences = response.split('.')
        if len(sentences) >= 2:
            score += 1.0
            checks += 1
        
        # Task type specific checks
        if hasattr(task, 'category'):
            category_score = self._category_specific_check(task.category, response)
            score += category_score
            checks += 1
        
        return score / checks if checks > 0 else 0.5
    
    def _check_format(self, format_type: str, response: str) -> float:
        """Check if response matches required format"""
        if format_type == 'list':
            # Check for list format (bullets, numbers, etc.)
            list_patterns = [r'^\s*[-•*]\s+', r'^\s*\d+[\.)]\s+']
            for pattern in list_patterns:
                if re.search(pattern, response, re.MULTILINE):
                    return 1.0
            return 0.3
        
        elif format_type == 'json':
            # Check for JSON-like structure
            try:
                import json
                json.loads(response)
                return 1.0
            except:
                if '{' in response and '}' in response:
                    return 0.5
                return 0.0
        
        elif format_type == 'code':
            # Check for code-like content
            code_indicators = ['def ', 'class ', 'function', 'import ', 'return ', '```']
            matches = sum(1 for ind in code_indicators if ind in response)
            return min(1.0, matches / 2)
        
        elif format_type == 'email':
            # Check for email format
            email_components = ['subject:', 'dear', 'sincerely', 'regards', 'from:', 'to:']
            matches = sum(1 for comp in email_components if comp.lower() in response.lower())
            return min(1.0, matches / 3)
        
        return 0.5
    
    def _check_structure(self, structure: str, response: str) -> float:
        """Check if response has required structure"""
        if structure == 'introduction_body_conclusion':
            # Check for three-part structure
            paragraphs = response.split('\n\n')
            if len(paragraphs) >= 3:
                return 1.0
            return len(paragraphs) / 3
        
        elif structure == 'problem_solution':
            # Check for problem and solution sections
            has_problem = any(word in response.lower() for word in ['problem', 'issue', 'challenge'])
            has_solution = any(word in response.lower() for word in ['solution', 'solve', 'resolve', 'fix'])
            return (int(has_problem) + int(has_solution)) / 2
        
        elif structure == 'step_by_step':
            # Check for step-by-step format
            step_patterns = [r'step \d+', r'first[,\s]', r'then[,\s]', r'finally[,\s]', r'next[,\s]']
            matches = sum(1 for pattern in step_patterns if re.search(pattern, response, re.IGNORECASE))
            return min(1.0, matches / 2)
        
        return 0.5
    
    def _category_specific_check(self, category: str, response: str) -> float:
        """Category-specific completion checks"""
        if category == 'professional':
            # Check for professional tone and structure
            professional_indicators = ['therefore', 'regarding', 'furthermore', 'accordingly']
            matches = sum(1 for ind in professional_indicators if ind in response.lower())
            return min(1.0, matches / 2)
        
        elif category == 'technical':
            # Check for technical content
            technical_indicators = ['algorithm', 'function', 'parameter', 'implementation', 'performance']
            matches = sum(1 for ind in technical_indicators if ind in response.lower())
            return min(1.0, matches / 2)
        
        elif category == 'creative':
            # Check for creative elements
            # More lenient - just needs to be substantive
            return 1.0 if len(response) > 100 else len(response) / 100
        
        return 0.7  # Default score for other categories
