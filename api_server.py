from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import uuid
import asyncio
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.benchmark_runner import RealBenchmark, BenchmarkTask

app = FastAPI(
    title="RealBench API",
    description="REST API for RealBench - Real-World AI Benchmark",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo (use Redis/DB in production)
benchmark_jobs = {}
results_store = {}

class ModelRequest(BaseModel):
    prompt: str
    model_config: Optional[Dict[str, Any]] = None

class BenchmarkRunRequest(BaseModel):
    tasks: Optional[List[Dict[str, Any]]] = None
    categories: Optional[List[str]] = None
    max_tasks_per_category: Optional[int] = None
    model_endpoint: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None

class TaskSubmission(BaseModel):
    task_id: str
    category: str
    subcategory: str
    description: str
    prompt: str
    expected_capabilities: List[str]
    difficulty: str
    multi_turn: Optional[bool] = False
    context: Optional[Dict[str, Any]] = None
    ground_truth: Optional[str] = None
    evaluation_criteria: Optional[Dict[str, Any]] = None

def mock_model_response(prompt: str) -> str:
    """Mock model for demo purposes"""
    if "email" in prompt.lower():
        return "Professional email response..."
    elif "code" in prompt.lower() or "debug" in prompt.lower():
        return "```python\n# Optimized code solution\npass\n```"
    elif "hack" in prompt.lower() or "illegal" in prompt.lower():
        return "I cannot assist with illegal activities. Please ask for help with legitimate tasks."
    else:
        return "This is a mock response for demonstration. Connect your AI model here."

@app.get("/")
async def root():
    return {
        "message": "Welcome to RealBench API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "run_benchmark": "/benchmark/run",
            "get_results": "/benchmark/results/{job_id}",
            "list_tasks": "/tasks",
            "submit_task": "/tasks/submit"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RealBench API"}

@app.get("/tasks")
async def list_available_tasks():
    """Get all available benchmark tasks"""
    try:
        tasks_by_category = {}
        task_dir = Path("data/tasks")
        
        if task_dir.exists():
            for task_file in task_dir.glob("*_tasks.json"):
                category = task_file.stem.replace("_tasks", "")
                with open(task_file, 'r') as f:
                    tasks = json.load(f)
                    tasks_by_category[category] = tasks
        
        # Add sample tasks if no data directory
        if not tasks_by_category:
            tasks_by_category = {
                "professional": [
                    {
                        "task_id": "prof_001",
                        "category": "professional",
                        "subcategory": "email_composition",
                        "description": "Compose a professional follow-up email",
                        "prompt": "Write a follow-up email to a client who hasn't responded to your proposal in 2 weeks.",
                        "difficulty": "medium",
                        "expected_capabilities": ["communication", "professionalism"]
                    }
                ],
                "technical": [
                    {
                        "task_id": "tech_001",
                        "category": "technical", 
                        "subcategory": "debugging",
                        "description": "Debug a Python function",
                        "prompt": "Fix this slow prime number function: def find_primes(n): ...",
                        "difficulty": "medium",
                        "expected_capabilities": ["debugging", "optimization"]
                    }
                ]
            }
        
        total_tasks = sum(len(tasks) for tasks in tasks_by_category.values())
        
        return {
            "total_tasks": total_tasks,
            "categories": list(tasks_by_category.keys()),
            "tasks_by_category": tasks_by_category
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading tasks: {str(e)}")

@app.post("/tasks/submit")
async def submit_custom_task(task: TaskSubmission):
    """Submit a custom task for evaluation"""
    try:
        # Convert to BenchmarkTask format
        task_dict = {
            "task_id": task.task_id,
            "category": task.category,
            "subcategory": task.subcategory,
            "description": task.description,
            "prompt": task.prompt,
            "expected_capabilities": task.expected_capabilities,
            "difficulty": task.difficulty,
            "multi_turn": task.multi_turn or False,
            "context": task.context,
            "ground_truth": task.ground_truth,
            "evaluation_criteria": task.evaluation_criteria or {}
        }
        
        return {
            "message": "Task submitted successfully",
            "task_id": task.task_id,
            "task": task_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error submitting task: {str(e)}")

@app.post("/benchmark/run")
async def run_benchmark(request: BenchmarkRunRequest, background_tasks: BackgroundTasks):
    """Start a benchmark run"""
    try:
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        benchmark_jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "start_time": None,
            "end_time": None,
            "error": None
        }
        
        # Start benchmark in background
        background_tasks.add_task(
            run_benchmark_job,
            job_id,
            request.tasks,
            request.categories,
            request.max_tasks_per_category,
            request.model_endpoint
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Benchmark started. Use /benchmark/results/{job_id} to check progress."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting benchmark: {str(e)}")

async def run_benchmark_job(
    job_id: str,
    tasks: Optional[List[Dict[str, Any]]],
    categories: Optional[List[str]],
    max_tasks_per_category: Optional[int],
    model_endpoint: Optional[str]
):
    """Background task to run benchmark"""
    try:
        import time
        benchmark_jobs[job_id]["start_time"] = time.time()
        
        # Use mock model for demo
        model_fn = mock_model_response
        
        # Initialize benchmark
        benchmark = RealBenchmark(model_fn=model_fn)
        
        # Determine tasks to run
        if tasks:
            total_tasks = len(tasks)
            benchmark_jobs[job_id]["total_tasks"] = total_tasks
            results = benchmark.run(tasks=tasks, verbose=False)
        else:
            # Load from categories
            if not categories:
                categories = ["professional", "technical", "safety"]
            
            # Estimate total tasks (simplified)
            benchmark_jobs[job_id]["total_tasks"] = len(categories) * (max_tasks_per_category or 5)
            results = benchmark.run(
                categories=categories,
                max_tasks_per_category=max_tasks_per_category,
                verbose=False
            )
        
        # Store results
        results_store[job_id] = results
        
        # Update job status
        benchmark_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "completed_tasks": len(results),
            "end_time": time.time()
        })
        
    except Exception as e:
        benchmark_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": time.time()
        })

@app.get("/benchmark/results/{job_id}")
async def get_benchmark_results(job_id: str):
    """Get benchmark results by job ID"""
    if job_id not in benchmark_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = benchmark_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job_info["status"],
        "progress": job_info["progress"],
        "total_tasks": job_info["total_tasks"],
        "completed_tasks": job_info["completed_tasks"],
        "start_time": job_info["start_time"],
        "end_time": job_info["end_time"]
    }
    
    if job_info["status"] == "failed":
        response["error"] = job_info["error"]
    
    if job_info["status"] == "completed" and job_id in results_store:
        results = results_store[job_id]
        
        # Calculate summary statistics
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r['success'])
        avg_score = sum(r['overall_score'] for r in results) / total_tasks if total_tasks > 0 else 0
        
        response.update({
            "results": results,
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
                "average_score": avg_score
            }
        })
    
    return response

@app.get("/benchmark/jobs")
async def list_benchmark_jobs():
    """List all benchmark jobs"""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": info["status"],
                "total_tasks": info["total_tasks"],
                "completed_tasks": info["completed_tasks"],
                "start_time": info["start_time"]
            }
            for job_id, info in benchmark_jobs.items()
        ]
    }

@app.delete("/benchmark/results/{job_id}")
async def delete_benchmark_results(job_id: str):
    """Delete benchmark results"""
    if job_id not in benchmark_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up job and results
    del benchmark_jobs[job_id]
    if job_id in results_store:
        del results_store[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

@app.post("/model/test")
async def test_model_endpoint(request: ModelRequest):
    """Test model endpoint with a sample prompt"""
    try:
        response = mock_model_response(request.prompt)
        return {
            "prompt": request.prompt,
            "response": response,
            "model_config": request.model_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
